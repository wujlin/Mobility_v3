#!/usr/bin/env python3
"""
Audit POI raster data distribution for Detroit core grid.
检查 POI 数据的实际分布情况，找出是否存在数据稀疏问题。
"""
import json
from pathlib import Path
import numpy as np

# 你需要把这个路径改成实际的 semantic_dir
SEMANTIC_DIR = Path("data/worldtrace/detroit_core_v1")  # 本地
# 或者在 wsA 上运行时：
# SEMANTIC_DIR = Path("/home/jinlin/data/geoexplicit_data/worldtrace/detroit_core_v1")

def main():
    d = SEMANTIC_DIR
    if not d.exists():
        print(f"ERROR: semantic_dir not found: {d}")
        print("请修改脚本中的 SEMANTIC_DIR 路径")
        return
    
    # 加载 POI 栅格
    poi_files = sorted(d.glob("poi_density_*.npy"))
    if not poi_files:
        print(f"ERROR: No poi_density_*.npy files found in {d}")
        return
    
    print(f"=== POI Raster Audit for {d} ===\n")
    
    categories = []
    poi_stack = []
    for p in poi_files:
        cat = p.stem.replace("poi_density_", "")
        arr = np.load(p)
        categories.append(cat)
        poi_stack.append(arr)
        
        nonzero = np.sum(arr > 0)
        total = np.sum(arr)
        max_val = np.max(arr)
        mean_nonzero = np.mean(arr[arr > 0]) if nonzero > 0 else 0
        
        print(f"Category: {cat:12s}")
        print(f"  Shape: {arr.shape}")
        print(f"  Non-zero cells: {nonzero:,} / {arr.size:,} ({100*nonzero/arr.size:.2f}%)")
        print(f"  Total POI count: {total:,.0f}")
        print(f"  Max in single cell: {max_val:.0f}")
        print(f"  Mean (non-zero cells): {mean_nonzero:.2f}")
        print()
    
    # 汇总
    poi_total = np.sum(poi_stack, axis=0)
    print(f"=== TOTAL (all categories) ===")
    print(f"  Non-zero cells: {np.sum(poi_total > 0):,} / {poi_total.size:,} ({100*np.sum(poi_total > 0)/poi_total.size:.2f}%)")
    print(f"  Total POI count: {np.sum(poi_total):,.0f}")
    print(f"  Max in single cell: {np.max(poi_total):.0f}")
    print(f"  Mean (non-zero cells): {np.mean(poi_total[poi_total > 0]):.2f}")
    print()
    
    # 分布分析
    print(f"=== Distribution Analysis ===")
    thresholds = [0, 1, 5, 10, 50, 100]
    for t in thresholds:
        count = np.sum(poi_total > t)
        print(f"  Cells with > {t:3d} POIs: {count:,} ({100*count/poi_total.size:.2f}%)")
    print()
    
    # 空间覆盖分析
    # 假设典型轨迹长度 ~500 pixels，看 500x500 区域的 POI 分布
    H, W = poi_total.shape
    region_size = 500
    regions_with_poi = 0
    total_regions = 0
    for y in range(0, H - region_size, region_size // 2):
        for x in range(0, W - region_size, region_size // 2):
            region = poi_total[y:y+region_size, x:x+region_size]
            total_regions += 1
            if np.sum(region) > 0:
                regions_with_poi += 1
    
    print(f"=== Spatial Coverage (500x500 regions) ===")
    print(f"  Regions with POI: {regions_with_poi}/{total_regions} ({100*regions_with_poi/total_regions:.1f}%)")
    print()
    
    # Entropy 分析
    entropy_path = d / "landuse_entropy.npy"
    if entropy_path.exists():
        entropy = np.load(entropy_path)
        print(f"=== Landuse Entropy ===")
        print(f"  Non-zero cells: {np.sum(entropy > 0):,} ({100*np.sum(entropy > 0)/entropy.size:.2f}%)")
        print(f"  Mean (non-zero): {np.mean(entropy[entropy > 0]):.3f}")
        print(f"  Max: {np.max(entropy):.3f}")
        print(f"  Std (non-zero): {np.std(entropy[entropy > 0]):.3f}")
    
    # 检查 meta 文件
    meta_path = d / "poi_raster_meta.json"
    if meta_path.exists():
        print(f"\n=== Meta Info (from poi_raster_meta.json) ===")
        with open(meta_path) as f:
            meta = json.load(f)
        print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
