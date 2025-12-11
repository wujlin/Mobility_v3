"""
raw_io.py - 原始 GPS 数据读取模块

支持深圳出租车 GPS 数据格式：
- 每辆车一个文件（以车牌号命名）
- CSV 格式，字段：name, time, jd, wd, status, v, angle
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Generator
from datetime import datetime
from tqdm import tqdm

from src.config.settings import GRID, PATHS


# ============================================================
# 深圳出租车 GPS 数据列名映射
# ============================================================
SHENZHEN_COLUMNS = {
    'name': 'vehicle_id',
    'time': 'time_str',
    'jd': 'lon',        # 经度
    'wd': 'lat',        # 纬度
    'status': 'status', # 0=空载, 1=重载
    'v': 'speed_kmh',   # 速度 km/h
    'angle': 'angle'    # 方向 0-7
}

# 方向角映射：0=东, 1=东南, 2=南, 3=西南, 4=西, 5=西北, 6=北, 7=东北
ANGLE_TO_DEGREES = {
    0: 0,    # 东
    1: 45,   # 东南
    2: 90,   # 南
    3: 135,  # 西南
    4: 180,  # 西
    5: 225,  # 西北
    6: 270,  # 北
    7: 315   # 东北
}


def read_single_vehicle_file(file_path: Union[str, Path], 
                              encoding: str = 'gbk') -> Optional[pd.DataFrame]:
    """
    读取单个车辆的 GPS 数据文件。
    
    Args:
        file_path: 文件路径
        encoding: 文件编码（深圳数据通常是 GBK）
    
    Returns:
        DataFrame with standardized columns, or None if file is empty/invalid
    """
    file_path = Path(file_path)
    
    try:
        # 读取 CSV，处理可能的尾部逗号
        df = pd.read_csv(file_path, encoding=encoding)
        
        # 移除可能的空列（由尾部逗号导致）
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        if df.empty:
            return None
        
        # 重命名列
        df = df.rename(columns=SHENZHEN_COLUMNS)
        
        # 解析时间
        df['timestamp'] = pd.to_datetime(df['time_str'], format='%Y/%m/%d %H:%M:%S')
        df['timestamp_unix'] = df['timestamp'].astype('int64') // 10**9  # 转为秒
        
        # 速度转换：km/h -> m/s
        df['speed_ms'] = df['speed_kmh'] / 3.6
        
        # 方向角转换为度数
        df['heading_deg'] = df['angle'].map(ANGLE_TO_DEGREES)
        
        # 按时间排序
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def read_all_vehicle_files(gps_dir: Union[str, Path] = None,
                            pattern: str = "*.txt",
                            max_files: Optional[int] = None,
                            show_progress: bool = True) -> pd.DataFrame:
    """
    读取目录下所有车辆 GPS 文件并合并。
    
    Args:
        gps_dir: GPS 数据目录，默认使用 PATHS.raw_gps
        pattern: 文件匹配模式
        max_files: 最大读取文件数（用于调试）
        show_progress: 是否显示进度条
    
    Returns:
        合并后的 DataFrame
    """
    if gps_dir is None:
        gps_dir = PATHS.raw_gps
    gps_dir = Path(gps_dir)
    
    files = list(gps_dir.glob(pattern))
    if max_files:
        files = files[:max_files]
    
    dfs = []
    iterator = tqdm(files, desc="Reading GPS files") if show_progress else files
    
    for f in iterator:
        df = read_single_vehicle_file(f)
        if df is not None and len(df) > 0:
            dfs.append(df)
    
    if not dfs:
        raise ValueError(f"No valid GPS files found in {gps_dir}")
    
    merged = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(merged):,} GPS points from {len(dfs)} vehicles")
    
    return merged


def iter_vehicle_files(gps_dir: Union[str, Path] = None,
                       pattern: str = "*.txt") -> Generator[pd.DataFrame, None, None]:
    """
    逐个迭代读取车辆文件（内存友好）。
    
    Yields:
        每辆车的 DataFrame
    """
    if gps_dir is None:
        gps_dir = PATHS.raw_gps
    gps_dir = Path(gps_dir)
    
    for f in gps_dir.glob(pattern):
        df = read_single_vehicle_file(f)
        if df is not None and len(df) > 0:
            yield df


# ============================================================
# 坐标转换函数
# ============================================================

def latlon_to_grid(lat: np.ndarray, lon: np.ndarray, 
                   grid_config=None) -> tuple:
    """
    经纬度转换为栅格坐标 (y, x)。
    
    注意：
    - y 对应纬度方向，向南增加（图像坐标系）
    - x 对应经度方向，向东增加
    
    Args:
        lat: 纬度数组
        lon: 经度数组
        grid_config: GridConfig 实例
    
    Returns:
        (y, x) 栅格坐标元组
    """
    if grid_config is None:
        grid_config = GRID
    
    # 归一化到 [0, 1]
    lat_norm = (lat - grid_config.min_lat) / grid_config.lat_range
    lon_norm = (lon - grid_config.min_lon) / grid_config.lon_range
    
    # 转换到栅格坐标
    # y: 纬度越大，y 越小（北边在上）
    y = (1 - lat_norm) * grid_config.H
    x = lon_norm * grid_config.W
    
    return y, x


def grid_to_latlon(y: np.ndarray, x: np.ndarray,
                   grid_config=None) -> tuple:
    """
    栅格坐标 (y, x) 转换为经纬度。
    
    Args:
        y: y 坐标数组
        x: x 坐标数组
        grid_config: GridConfig 实例
    
    Returns:
        (lat, lon) 经纬度元组
    """
    if grid_config is None:
        grid_config = GRID
    
    # 反向转换
    lat_norm = 1 - y / grid_config.H
    lon_norm = x / grid_config.W
    
    lat = lat_norm * grid_config.lat_range + grid_config.min_lat
    lon = lon_norm * grid_config.lon_range + grid_config.min_lon
    
    return lat, lon


def filter_by_bounds(df: pd.DataFrame, grid_config=None) -> pd.DataFrame:
    """
    过滤掉超出地理边界的点。
    
    Args:
        df: 包含 lat, lon 列的 DataFrame
        grid_config: GridConfig 实例
    
    Returns:
        过滤后的 DataFrame
    """
    if grid_config is None:
        grid_config = GRID
    
    mask = (
        (df['lat'] >= grid_config.min_lat) & 
        (df['lat'] <= grid_config.max_lat) &
        (df['lon'] >= grid_config.min_lon) & 
        (df['lon'] <= grid_config.max_lon)
    )
    
    n_before = len(df)
    df_filtered = df[mask].copy()
    n_after = len(df_filtered)
    
    if n_before > n_after:
        print(f"Filtered out {n_before - n_after} points outside bounds "
              f"({(n_before - n_after) / n_before * 100:.2f}%)")
    
    return df_filtered


# ============================================================
# 数据质量检查
# ============================================================

def check_gps_data_quality(df: pd.DataFrame) -> dict:
    """
    检查 GPS 数据质量，返回统计信息。
    
    Args:
        df: GPS DataFrame
    
    Returns:
        质量统计字典
    """
    stats = {
        'total_points': len(df),
        'unique_vehicles': df['vehicle_id'].nunique(),
        'date_range': (df['timestamp'].min(), df['timestamp'].max()),
        'lat_range': (df['lat'].min(), df['lat'].max()),
        'lon_range': (df['lon'].min(), df['lon'].max()),
        'speed_range': (df['speed_kmh'].min(), df['speed_kmh'].max()),
        'status_distribution': df['status'].value_counts().to_dict(),
    }
    
    # 检查异常值
    stats['abnormal_speed'] = (df['speed_kmh'] > 150).sum()  # > 150 km/h
    stats['abnormal_lat'] = ((df['lat'] < 22) | (df['lat'] > 23.5)).sum()
    stats['abnormal_lon'] = ((df['lon'] < 113) | (df['lon'] > 115)).sum()
    
    return stats


def print_data_quality_report(stats: dict):
    """打印数据质量报告"""
    print("\n" + "=" * 50)
    print("GPS Data Quality Report")
    print("=" * 50)
    print(f"Total points: {stats['total_points']:,}")
    print(f"Unique vehicles: {stats['unique_vehicles']:,}")
    print(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    print(f"Lat range: {stats['lat_range'][0]:.6f} to {stats['lat_range'][1]:.6f}")
    print(f"Lon range: {stats['lon_range'][0]:.6f} to {stats['lon_range'][1]:.6f}")
    print(f"Speed range: {stats['speed_range'][0]:.1f} to {stats['speed_range'][1]:.1f} km/h")
    print(f"Status distribution: {stats['status_distribution']}")
    print(f"\nAbnormal values:")
    print(f"  - Speed > 150 km/h: {stats['abnormal_speed']:,}")
    print(f"  - Latitude out of range: {stats['abnormal_lat']:,}")
    print(f"  - Longitude out of range: {stats['abnormal_lon']:,}")
    print("=" * 50 + "\n")
