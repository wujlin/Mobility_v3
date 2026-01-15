#!/usr/bin/env python3
"""
数据验证：分析native OSM边的长度分布
核心问题：如果不做Bresenham rasterization，直接用OSM way作为边，
segment的平均长度是多少？route的序列长度能降到多少？
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator

import numpy as np

# ============================================================
# 配置 - 需要根据实际workstation路径调整
# ============================================================
RAW_ROOT = Path("/data/WorldTrace")
CITY = "detroit"

OSM_PBF = RAW_ROOT / "osm" / "michigan-latest.osm.pbf"
SEMANTIC_DIR = RAW_ROOT / "semantic" / CITY
PATHS_NPZ = RAW_ROOT / f"experiments/icml2026_routegen/T3_casd_v0_seed0/T1_prep_segment_graph/paths_graph.npz"

# Road types to include
ROAD_TYPES = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "residential"
}


def _normalize_highway_tag(tag: object) -> str:
    if isinstance(tag, (list, tuple, set)):
        if not tag:
            return ""
        tag = next(iter(tag))
    s = str(tag or "").strip()
    if not s:
        return ""
    if s.endswith("_link"):
        s = s[: -len("_link")]
    return s


def _iter_geom_coords_lonlat(geom) -> Iterator[np.ndarray]:
    gtype = getattr(geom, "geom_type", None)
    if gtype == "LineString":
        yield np.asarray(geom.coords, dtype=np.float64)
    elif gtype == "MultiLineString":
        for part in geom.geoms:
            coords = np.asarray(part.coords, dtype=np.float64)
            if coords.shape[0] >= 2:
                yield coords


def haversine_m(lon1, lat1, lon2, lat2):
    """计算两点间的haversine距离（米）"""
    R = 6371000  # 地球半径（米）
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def analyze_native_osm_edges():
    """分析native OSM边的长度分布"""
    try:
        from pyrosm import OSM
    except ImportError:
        print("需要安装pyrosm: conda install -c conda-forge pyrosm")
        return
    
    # 加载bbox
    meta_path = SEMANTIC_DIR / "osm_road_prob_meta.json"
    meta = json.loads(meta_path.read_text())
    bbox_info = meta.get("grid", {}).get("bbox", {})
    bbox = [bbox_info["min_lon"], bbox_info["min_lat"], 
            bbox_info["max_lon"], bbox_info["max_lat"]]
    
    print(f"Loading OSM from {OSM_PBF}")
    print(f"BBox: {bbox}")
    
    osm = OSM(str(OSM_PBF), bounding_box=bbox)
    roads = osm.get_network(network_type="driving")
    
    hw = roads["highway"].apply(_normalize_highway_tag)
    roads = roads[hw.isin(ROAD_TYPES)]
    
    print(f"Loaded {len(roads)} road segments (ways)")
    
    # 分析每条way的长度
    way_lengths_m = []
    way_n_coords = []
    
    for _, row in roads.iterrows():
        geom = row.get("geometry")
        highway = _normalize_highway_tag(row.get("highway"))
        if not highway:
            continue
            
        for coords in _iter_geom_coords_lonlat(geom):
            if coords.shape[0] < 2:
                continue
            
            # 计算这条way的总长度
            total_len = 0.0
            for i in range(len(coords) - 1):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[i + 1]
                total_len += haversine_m(lon1, lat1, lon2, lat2)
            
            way_lengths_m.append(total_len)
            way_n_coords.append(len(coords))
    
    way_lengths_m = np.array(way_lengths_m)
    way_n_coords = np.array(way_n_coords)
    
    print("\n" + "="*60)
    print("Native OSM Way 长度分布 (米)")
    print("="*60)
    print(f"  总way数: {len(way_lengths_m)}")
    print(f"  min:  {np.min(way_lengths_m):.1f}")
    print(f"  p10:  {np.percentile(way_lengths_m, 10):.1f}")
    print(f"  p25:  {np.percentile(way_lengths_m, 25):.1f}")
    print(f"  p50:  {np.percentile(way_lengths_m, 50):.1f}")
    print(f"  p75:  {np.percentile(way_lengths_m, 75):.1f}")
    print(f"  p90:  {np.percentile(way_lengths_m, 90):.1f}")
    print(f"  max:  {np.max(way_lengths_m):.1f}")
    print(f"  mean: {np.mean(way_lengths_m):.1f}")
    
    print("\n" + "="*60)
    print("Native OSM Way 坐标点数分布")
    print("="*60)
    print(f"  p50: {np.percentile(way_n_coords, 50):.0f}")
    print(f"  p90: {np.percentile(way_n_coords, 90):.0f}")
    print(f"  max: {np.max(way_n_coords):.0f}")
    
    return way_lengths_m, way_n_coords


def estimate_route_segment_count():
    """估算：如果用native OSM边，route的平均segment数是多少？"""
    # 加载现有的paths数据
    if not PATHS_NPZ.exists():
        print(f"Paths file not found: {PATHS_NPZ}")
        return
    
    data = np.load(PATHS_NPZ, allow_pickle=True)
    node_seq_len = data["node_seq_len"]
    
    # 现有的node-level序列长度
    print("\n" + "="*60)
    print("现有 Node-level 序列长度分布")
    print("="*60)
    print(f"  p50: {np.percentile(node_seq_len, 50):.0f}")
    print(f"  p75: {np.percentile(node_seq_len, 75):.0f}")
    print(f"  p90: {np.percentile(node_seq_len, 90):.0f}")
    
    # 假设native OSM way平均长度是X米
    # 假设raster分辨率是27米
    # 那么一条OSM way大约对应 X/27 个raster节点
    # 如果用native OSM way，序列长度应该降低 X/27 倍
    
    print("\n" + "="*60)
    print("估算：如果用Native OSM Way作为边")
    print("="*60)
    
    # 需要先运行analyze_native_osm_edges获取way_len_p50
    # 这里先用占位值
    raster_res = 27  # 米
    for way_len_p50 in [50, 100, 150, 200, 300]:
        reduction_factor = way_len_p50 / raster_res
        estimated_seg_len_p50 = np.percentile(node_seq_len, 50) / reduction_factor
        print(f"  假设way_len_p50={way_len_p50}m -> 序列长度p50 ≈ {estimated_seg_len_p50:.0f}")


if __name__ == "__main__":
    print("="*60)
    print("数据验证：Native OSM Edge vs Rasterized Edge")
    print("="*60)
    
    # Part 1: 分析native OSM way的长度
    result = analyze_native_osm_edges()
    
    # Part 2: 估算route的segment数
    estimate_route_segment_count()
