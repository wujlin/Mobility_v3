"""
Porto Taxi CSV → segments_with_wayid.parquet

将 Porto Taxi train.csv 转换为与 WorldTrace 完全一致的 segments parquet 格式，
使后续 pipeline (run_way_casd_prep.sh) 无需任何修改即可复用。

核心逻辑：
  1. 流式读取 train.csv，解析 POLYLINE JSON
  2. 过滤 MISSING_DATA / 太短 / 太长 的行程
  3. 调 Valhalla trace_attributes API 获取每个 GPS 点的 osm_way_id
  4. 输出 parquet，列名与 build_detroit_segments.py 完全对齐

输出 parquet schema (与 WorldTrace 一致):
  traj_csv:            str       (行程 ID，用 TRIP_ID 填充)
  n_points:            int32
  unmatched_ratio:     float32
  way_id_missing_ratio:float32
  t:                   list<int64>
  lat:                 list<float32>
  lon:                 list<float32>
  y:                   list<int32>
  x:                   list<int32>
  is_matched:          list<int8>
  matched_distance:    list<float32>
  osm_way_id:          list<int64>

用法:
  # 调试（先跑 100 条）
  python -m tools.porto.porto_csv_to_segments_parquet \
      --csv $RAW_ROOT/porto_taxi/raw/train.csv \
      --out_parquet $RAW_ROOT/porto_taxi/segments_with_wayid.parquet \
      --bbox_meta tools/porto/porto_bbox_meta.json \
      --valhalla_url http://localhost:8002 \
      --workers 8 \
      --limit 100

  # 全量
  python -m tools.porto.porto_csv_to_segments_parquet \
      --csv $RAW_ROOT/porto_taxi/raw/train.csv \
      --out_parquet $RAW_ROOT/porto_taxi/segments_with_wayid.parquet \
      --bbox_meta tools/porto/porto_bbox_meta.json \
      --valhalla_url http://localhost:8002 \
      --workers 8

依赖: requests, pandas, pyarrow, tqdm (均在 requirements.txt 或 conda 环境中)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None

# ── 常量 ──────────────────────────────────────────────────
MIN_POINTS = 10   # < 150 秒的行程丢弃
MAX_POINTS = 300  # > 75 分钟的行程丢弃
CHECKPOINT_EVERY = 10_000  # 每 N 条刷一次 parquet


def _parse_polyline(s: str) -> List[List[float]]:
    """解析 POLYLINE JSON → [[lon, lat], ...]"""
    try:
        if not isinstance(s, str) or not s.strip():
            return []
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return []


def _load_bbox_meta(path: Path) -> Dict[str, Any]:
    """加载 bbox+grid 元数据 (与 build_way_features_from_osm_pbf 兼容)"""
    meta = json.loads(path.read_text(encoding="utf-8"))
    g = meta.get("grid", {})
    bbox = g.get("bbox", {})
    return {
        "H": int(g["H"]),
        "W": int(g["W"]),
        "min_lon": float(bbox["min_lon"]),
        "min_lat": float(bbox["min_lat"]),
        "max_lon": float(bbox["max_lon"]),
        "max_lat": float(bbox["max_lat"]),
    }


def _latlon_to_yx(lat: float, lon: float, meta: Dict) -> Tuple[int, int]:
    """WGS84 → grid (y, x)"""
    H, W = meta["H"], meta["W"]
    x01 = (lon - meta["min_lon"]) / max(meta["max_lon"] - meta["min_lon"], 1e-12)
    y01 = (meta["max_lat"] - lat) / max(meta["max_lat"] - meta["min_lat"], 1e-12)
    x = int(x01 * W)
    y = int(y01 * H)
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    return y, x


# ─── Valhalla trace_attributes 调用 ──────────────────────

def _match_one(args: Tuple[str, List[List[float]], int, str, Dict]) -> Optional[Dict[str, Any]]:
    """
    对单条轨迹调 Valhalla trace_attributes，返回 parquet 行数据。
    每个 GPS 点对应一个 osm_way_id（与 WorldTrace parquet schema 一致）。
    """
    import requests

    trip_id, polyline, timestamp, valhalla_url, bbox_meta = args

    n_pts = len(polyline)

    # 构造 shape（带 time 辅助匹配精度）
    shape = []
    for i, (lon, lat) in enumerate(polyline):
        pt: Dict[str, Any] = {"lon": lon, "lat": lat}
        if timestamp > 0:
            pt["time"] = timestamp + i * 15
        shape.append(pt)

    req = {
        "shape": shape,
        "costing": "auto",
        "shape_match": "map_snap",
        "filters": {
            "attributes": [
                "edge.way_id",
                "edge.length",
                "edge.road_class",
                "edge.begin_shape_index",
                "edge.end_shape_index",
                "matched.point",
                "matched.type",
                "matched.edge_index",
                "matched.distance_from_trace_point",
            ],
            "action": "include",
        },
        "trace_options": {
            "gps_accuracy": 30,
            "search_radius": 50,
        },
    }

    try:
        r = requests.post(f"{valhalla_url}/trace_attributes", json=req, timeout=120)
        if r.status_code != 200:
            return None
        resp = r.json()
    except Exception:
        return None

    edges = resp.get("edges", [])
    matched_points = resp.get("matched_points", [])

    if not edges or not matched_points:
        return None

    # 构建 edge_index → way_id 映射
    edge_way_ids = {}
    for ei, edge in enumerate(edges):
        wid = edge.get("way_id", -1)
        edge_way_ids[ei] = int(wid) if wid is not None else -1

    # 每个 GPS 点映射到 osm_way_id
    # matched_points 与 input shape 1:1 对应
    per_point_way_id = []
    per_point_is_matched = []
    per_point_match_dist = []
    per_point_lat = []
    per_point_lon = []

    for i, mp in enumerate(matched_points):
        mtype = mp.get("type", "unmatched")
        edge_idx = mp.get("edge_index")
        dist = mp.get("distance_from_trace_point", float("nan"))

        if mtype == "matched" and edge_idx is not None and edge_idx in edge_way_ids:
            wid = edge_way_ids[edge_idx]
            per_point_way_id.append(wid)
            per_point_is_matched.append(1)
        elif mtype == "interpolated" and edge_idx is not None and edge_idx in edge_way_ids:
            wid = edge_way_ids[edge_idx]
            per_point_way_id.append(wid)
            per_point_is_matched.append(1)
        else:
            per_point_way_id.append(-1)
            per_point_is_matched.append(0)

        per_point_match_dist.append(float(dist) if dist is not None and np.isfinite(dist) else 0.0)

        # 原始 GPS 坐标
        if i < n_pts:
            per_point_lat.append(float(polyline[i][1]))
            per_point_lon.append(float(polyline[i][0]))

    # 截断到原始点数（matched_points 可能与 input 不完全对齐）
    actual_n = min(len(per_point_way_id), n_pts)
    per_point_way_id = per_point_way_id[:actual_n]
    per_point_is_matched = per_point_is_matched[:actual_n]
    per_point_match_dist = per_point_match_dist[:actual_n]
    per_point_lat = per_point_lat[:actual_n]
    per_point_lon = per_point_lon[:actual_n]

    if actual_n < MIN_POINTS:
        return None

    # 计算 grid y/x
    per_point_y = []
    per_point_x = []
    for lat, lon in zip(per_point_lat, per_point_lon):
        yy, xx = _latlon_to_yx(lat, lon, bbox_meta)
        per_point_y.append(yy)
        per_point_x.append(xx)

    # 计算时间戳序列
    per_point_t = [timestamp + i * 15 for i in range(actual_n)]

    # 统计
    n_matched = sum(per_point_is_matched)
    n_way_missing = sum(1 for w in per_point_way_id if w <= 0)
    unmatched_ratio = 1.0 - n_matched / max(actual_n, 1)
    way_missing_ratio = n_way_missing / max(actual_n, 1)

    return {
        "traj_csv": trip_id,
        "n_points": actual_n,
        "unmatched_ratio": float(unmatched_ratio),
        "way_id_missing_ratio": float(way_missing_ratio),
        "t": per_point_t,
        "lat": per_point_lat,
        "lon": per_point_lon,
        "y": per_point_y,
        "x": per_point_x,
        "is_matched": per_point_is_matched,
        "matched_distance": per_point_match_dist,
        "osm_way_id": per_point_way_id,
    }


def run(
    csv_path: Path,
    out_parquet: Path,
    bbox_meta_path: Path,
    valhalla_url: str,
    n_workers: int,
    limit: int,
) -> None:
    import pandas as pd
    from tqdm import tqdm

    if pa is None or pq is None:
        raise SystemExit("需要 pyarrow: pip install pyarrow")

    bbox_meta = _load_bbox_meta(bbox_meta_path)
    print(f"bbox: lon=[{bbox_meta['min_lon']}, {bbox_meta['max_lon']}], "
          f"lat=[{bbox_meta['min_lat']}, {bbox_meta['max_lat']}], "
          f"grid={bbox_meta['H']}x{bbox_meta['W']}")

    # ── 阶段 1: 读取 CSV + 清洗 ──
    print(f"\n[1/2] 读取 {csv_path} ...")
    df = pd.read_csv(csv_path)
    n_raw = len(df)
    print(f"  原始: {n_raw}")

    # 过滤
    df = df[df["MISSING_DATA"] != True]  # noqa: E712
    # 有些版本 MISSING_DATA 是字符串
    df = df[~df["MISSING_DATA"].astype(str).str.lower().isin(["true"])]
    print(f"  过滤 MISSING_DATA 后: {len(df)}")

    # 解析 POLYLINE
    tasks = []
    n_short = 0
    n_long = 0
    n_empty = 0

    for _, row in df.iterrows():
        poly = _parse_polyline(str(row.get("POLYLINE", "")))
        n_pts = len(poly)
        if n_pts == 0:
            n_empty += 1
            continue
        if n_pts < MIN_POINTS:
            n_short += 1
            continue
        if n_pts > MAX_POINTS:
            n_long += 1
            continue
        tasks.append((
            str(row["TRIP_ID"]),
            poly,
            int(row.get("TIMESTAMP", 0)),
            valhalla_url,
            bbox_meta,
        ))
        if limit > 0 and len(tasks) >= limit:
            break

    n_clean = len(tasks)
    print(f"  空轨迹: {n_empty}, 太短(<{MIN_POINTS}): {n_short}, "
          f"太长(>{MAX_POINTS}): {n_long}")
    print(f"  清洗后: {n_clean} 条行程待 map matching")

    # ── 阶段 2: 多进程 map matching → 流式写 parquet ──
    print(f"\n[2/2] Map matching (workers={n_workers}) → {out_parquet}")

    schema = pa.schema([
        ("traj_csv", pa.string()),
        ("n_points", pa.int32()),
        ("unmatched_ratio", pa.float32()),
        ("way_id_missing_ratio", pa.float32()),
        ("t", pa.list_(pa.int64())),
        ("lat", pa.list_(pa.float32())),
        ("lon", pa.list_(pa.float32())),
        ("y", pa.list_(pa.int32())),
        ("x", pa.list_(pa.int32())),
        ("is_matched", pa.list_(pa.int8())),
        ("matched_distance", pa.list_(pa.float32())),
        ("osm_way_id", pa.list_(pa.int64())),
    ])

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(out_parquet), schema=schema, compression="zstd")

    n_ok = 0
    n_fail = 0
    buf: List[Dict] = []

    def _flush_buf():
        nonlocal buf
        if not buf:
            return
        table = pa.Table.from_pydict(
            {col: [r[col] for r in buf] for col in schema.names},
            schema=schema,
        )
        writer.write_table(table)
        buf = []

    t0 = time.time()

    with Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_match_one, tasks, chunksize=20),
            total=n_clean,
            desc="Matching",
        ):
            if result is not None:
                buf.append(result)
                n_ok += 1
            else:
                n_fail += 1

            if len(buf) >= CHECKPOINT_EVERY:
                _flush_buf()
                elapsed = time.time() - t0
                print(f"  checkpoint: ok={n_ok}, fail={n_fail}, "
                      f"rate={n_ok / elapsed:.1f}/s", file=sys.stderr)

    _flush_buf()
    writer.close()

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"完成! {out_parquet}")
    print(f"  成功: {n_ok}/{n_clean} ({n_ok / max(n_clean, 1):.1%})")
    print(f"  失败: {n_fail}")
    print(f"  耗时: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"\n下一步: 用 run_way_casd_prep.sh 走标准 pipeline")
    print(f"  export SEGMENTS_PARQUET={out_parquet}")


def main():
    parser = argparse.ArgumentParser(
        description="Porto Taxi CSV → segments_with_wayid.parquet (Valhalla map matching)"
    )
    parser.add_argument("--csv", type=Path, required=True,
                        help="Porto train.csv 路径")
    parser.add_argument("--out_parquet", type=Path, required=True,
                        help="输出 parquet 路径")
    parser.add_argument("--bbox_meta", type=Path, required=True,
                        help="Porto bbox 元数据 JSON (与 osm_road_prob_meta.json 格式一致)")
    parser.add_argument("--valhalla_url", type=str, default="http://localhost:8002")
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()))
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 条（调试用，0=全量）")
    args = parser.parse_args()

    run(
        csv_path=args.csv,
        out_parquet=args.out_parquet,
        bbox_meta_path=args.bbox_meta,
        valhalla_url=args.valhalla_url,
        n_workers=args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
