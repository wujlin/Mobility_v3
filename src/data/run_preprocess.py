"""
run_preprocess.py - 数据预处理主脚本

用法:
    python -m src.data.run_preprocess [--max-files N] [--output-dir DIR]
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from src.data.raw_io import (
    read_all_vehicle_files, 
    check_gps_data_quality, 
    print_data_quality_report
)
from src.data.preprocess import extract_all_trips, Normalizer
from src.data.trajectories import TrajectoryStorage
from src.config.settings import PATHS, GRID


def main():
    parser = argparse.ArgumentParser(description='Preprocess GPS data')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process (for debugging)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for processed data')
    parser.add_argument('--min-trip-length', type=int, default=10,
                        help='Minimum trip length (points)')
    parser.add_argument('--min-trip-distance', type=float, default=500,
                        help='Minimum trip distance (meters)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else PATHS.processed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GPS Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Grid size: {GRID.H} x {GRID.W}")
    print(f"Geographic bounds: lat [{GRID.min_lat}, {GRID.max_lat}], lon [{GRID.min_lon}, {GRID.max_lon}]")
    print()
    
    # Step 1: 读取原始数据
    print("[Step 1] Loading raw GPS data...")
    df = read_all_vehicle_files(max_files=args.max_files)
    
    # Step 2: 数据质量检查
    print("\n[Step 2] Checking data quality...")
    stats = check_gps_data_quality(df)
    print_data_quality_report(stats)
    
    # Step 3: 提取 trips
    print("\n[Step 3] Extracting trips...")
    from src.config.settings import TRIP_CONFIG
    TRIP_CONFIG.min_trip_length = args.min_trip_length
    TRIP_CONFIG.min_trip_distance = args.min_trip_distance
    
    trips = extract_all_trips(df, use_status=True)
    
    if not trips:
        print("ERROR: No valid trips extracted!")
        return
    
    # Step 4: 计算归一化统计量
    print("\n[Step 4] Computing normalization statistics...")
    normalizer = Normalizer.from_data(trips)
    
    # Step 5: 保存处理结果
    print("\n[Step 5] Saving processed data...")
    
    # 5.1 保存轨迹到 HDF5
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_file = traj_dir / "shenzhen_trajectories.h5"
    
    if traj_file.exists():
        traj_file.unlink()
    
    TrajectoryStorage.create(traj_file)
    with TrajectoryStorage(traj_file, mode='r+') as storage:
        # 分批写入
        batch_size = 1000
        for i in range(0, len(trips), batch_size):
            batch = trips[i:i+batch_size]
            storage.append(batch)
    print(f"  Saved {len(trips)} trajectories to {traj_file}")
    
    # 5.2 保存统计信息
    stats_file = output_dir / "data_stats.json"
    save_stats = {
        'created_at': datetime.now().isoformat(),
        'num_trajectories': len(trips),
        'num_points': sum(len(t['positions']) for t in trips),
        'grid_config': {
            'H': GRID.H,
            'W': GRID.W,
            'min_lat': GRID.min_lat,
            'max_lat': GRID.max_lat,
            'min_lon': GRID.min_lon,
            'max_lon': GRID.max_lon,
        },
        'normalization': {
            'pos_min': [float(x) for x in normalizer.config.pos_min],
            'pos_max': [float(x) for x in normalizer.config.pos_max],
            'vel_mean': [float(x) for x in normalizer.config.vel_mean],
            'vel_std': [float(x) for x in normalizer.config.vel_std],
        },
        'trip_length_stats': {
            'min': int(min(len(t['positions']) for t in trips)),
            'max': int(max(len(t['positions']) for t in trips)),
            'mean': float(np.mean([len(t['positions']) for t in trips])),
        },
        'data_quality': {
            'total_raw_points': stats['total_points'],
            'unique_vehicles': stats['unique_vehicles'],
            'date_range': [str(stats['date_range'][0]), str(stats['date_range'][1])],
        }
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(save_stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved statistics to {stats_file}")
    
    # 5.3 生成数据切分（按时间）
    print("\n[Step 6] Generating train/val/test splits...")
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # 按轨迹起始时间排序
    trip_times = [t['timestamp'][0] for t in trips]
    sorted_indices = np.argsort(trip_times)
    
    n = len(trips)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_ids = sorted_indices[:train_end]
    val_ids = sorted_indices[train_end:val_end]
    test_ids = sorted_indices[val_end:]
    
    np.save(splits_dir / "train_ids.npy", train_ids)
    np.save(splits_dir / "val_ids.npy", val_ids)
    np.save(splits_dir / "test_ids.npy", test_ids)
    
    print(f"  Train: {len(train_ids)} trips")
    print(f"  Val: {len(val_ids)} trips")
    print(f"  Test: {len(test_ids)} trips")
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
