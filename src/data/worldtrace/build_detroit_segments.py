from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # optional
    pa = None
    pq = None

from src.utils.geo_grid import BBox, GridSpec


@dataclass(frozen=True)
class SegmentConfig:
    dt_gap_s: int = 5
    min_segment_points: int = 120
    matched_distance_max_m: float = 30.0
    max_unmatched_ratio: float = 0.20


def _truthy(v: str) -> bool:
    v = (v or "").strip().lower()
    return v in {"1", "true", "t", "yes", "y"}


def _safe_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: str) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _pick_coord(row: Dict[str, str], cfg: SegmentConfig) -> Tuple[Optional[float], Optional[float], int, Optional[float]]:
    """
    Return (lat, lon, is_matched, matched_distance).
    Uses matched_* if matched_type is truthy and matched_distance<=threshold; otherwise fallback to raw lat/lon.
    """
    md = _safe_float(row.get("matched_distance", ""))
    mt = _truthy(row.get("matched_type", ""))
    if mt and md is not None and md <= cfg.matched_distance_max_m:
        lat = _safe_float(row.get("matched_latitude", ""))
        lon = _safe_float(row.get("matched_longitude", ""))
        if lat is not None and lon is not None:
            return lat, lon, 1, md

    lat = _safe_float(row.get("latitude", ""))
    lon = _safe_float(row.get("longitude", ""))
    return lat, lon, 0, md


def _iter_csv_rows_from_zip(zf: zipfile.ZipFile, member: str) -> Iterator[Dict[str, str]]:
    with zf.open(member, "r") as f:
        text = (line.decode("utf-8", errors="ignore") for line in f)
        reader = csv.DictReader(text)
        for row in reader:
            yield row


def _split_bbox_segments(
    rows: Iterable[Dict[str, str]],
    *,
    bbox: BBox,
    grid: GridSpec,
    cfg: SegmentConfig,
) -> List[Dict[str, List]]:
    """
    Collect bbox-inside points, split by dt gap, return list of segments (as dict of lists).
    """
    segments: List[Dict[str, List]] = []
    cur: Optional[Dict[str, List]] = None
    last_t: Optional[int] = None

    def _flush():
        nonlocal cur, last_t
        if cur is not None and cur["n"] >= cfg.min_segment_points:
            segments.append(cur)
        cur = None
        last_t = None

    for row in rows:
        t = _safe_int(row.get("time", ""))
        if t is None:
            continue

        lat, lon, is_matched, md = _pick_coord(row, cfg)
        if lat is None or lon is None:
            continue

        if not bbox.contains(np.array([lat]), np.array([lon]))[0]:
            _flush()
            continue

        if cur is None:
            cur = {
                "t": [],
                "lat": [],
                "lon": [],
                "y": [],
                "x": [],
                "is_matched": [],
                "matched_distance": [],
                "n": 0,
                "unmatched_n": 0,
            }
            last_t = None

        if last_t is not None and (t - last_t) > cfg.dt_gap_s:
            _flush()
            cur = {
                "t": [],
                "lat": [],
                "lon": [],
                "y": [],
                "x": [],
                "is_matched": [],
                "matched_distance": [],
                "n": 0,
                "unmatched_n": 0,
            }

        last_t = t
        y, x = grid.latlon_to_yx(np.array([lat]), np.array([lon]))
        y0, x0 = int(y[0]), int(x[0])
        if not grid.in_bounds(np.array([y0]), np.array([x0]))[0]:
            _flush()
            continue

        cur["t"].append(t)
        cur["lat"].append(float(lat))
        cur["lon"].append(float(lon))
        cur["y"].append(y0)
        cur["x"].append(x0)
        cur["is_matched"].append(int(is_matched))
        cur["matched_distance"].append(float(md) if md is not None else float("nan"))
        cur["n"] += 1
        if not is_matched:
            cur["unmatched_n"] += 1

    _flush()
    return segments


def _select_longest_segment(segments: List[Dict[str, List]]) -> Optional[Dict[str, List]]:
    if not segments:
        return None
    return max(segments, key=lambda s: int(s.get("n", 0)))


def _segment_ok(seg: Dict[str, List], cfg: SegmentConfig) -> bool:
    n = int(seg.get("n", 0))
    if n < cfg.min_segment_points:
        return False
    unr = float(seg.get("unmatched_n", 0)) / float(n)
    return unr <= cfg.max_unmatched_ratio


def _default_detroit_core_grid() -> GridSpec:
    bbox = BBox(min_lon=-83.25, max_lon=-82.95, min_lat=42.25, max_lat=42.50)
    return GridSpec(H=1024, W=1024, bbox=bbox)


def _open_parquet_writer(path: Path, schema: pa.Schema) -> pq.ParquetWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(str(path), schema=schema, compression="zstd")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract Detroit core bbox segments from WorldTrace Trajectory.zip.")
    ap.add_argument("--trajectory_zip", type=Path, required=True, help="Path to Trajectory.zip")
    ap.add_argument("--out_parquet", type=Path, required=True, help="Output parquet (one row per segment)")
    ap.add_argument("--limit_files", type=int, default=0, help="Debug limit on number of csv files (0=no limit)")
    ap.add_argument("--dt_gap_s", type=int, default=5)
    ap.add_argument("--min_segment_points", type=int, default=120)
    ap.add_argument("--matched_distance_max_m", type=float, default=30.0)
    ap.add_argument("--max_unmatched_ratio", type=float, default=0.20)
    args = ap.parse_args()

    if pa is None or pq is None:
        raise SystemExit("pyarrow is required for --out_parquet. Install: pip/conda install pyarrow")

    grid = _default_detroit_core_grid()
    cfg = SegmentConfig(
        dt_gap_s=int(args.dt_gap_s),
        min_segment_points=int(args.min_segment_points),
        matched_distance_max_m=float(args.matched_distance_max_m),
        max_unmatched_ratio=float(args.max_unmatched_ratio),
    )

    schema = pa.schema(
        [
            ("traj_csv", pa.string()),
            ("n_points", pa.int32()),
            ("unmatched_ratio", pa.float32()),
            ("t", pa.list_(pa.int64())),
            ("lat", pa.list_(pa.float32())),
            ("lon", pa.list_(pa.float32())),
            ("y", pa.list_(pa.int32())),
            ("x", pa.list_(pa.int32())),
            ("is_matched", pa.list_(pa.int8())),
            ("matched_distance", pa.list_(pa.float32())),
        ]
    )

    out_writer = _open_parquet_writer(args.out_parquet, schema)
    try:
        with zipfile.ZipFile(args.trajectory_zip, "r") as zf:
            members = [m for m in zf.namelist() if m.endswith(".csv")]
            if args.limit_files:
                members = members[: int(args.limit_files)]

            wrote = 0
            scanned = 0
            for member in members:
                scanned += 1
                rows = _iter_csv_rows_from_zip(zf, member)
                segs = _split_bbox_segments(rows, bbox=grid.bbox, grid=grid, cfg=cfg)
                seg = _select_longest_segment(segs)
                if seg is None or not _segment_ok(seg, cfg):
                    if scanned % 50000 == 0:
                        print(f"[INFO] scanned={scanned} wrote={wrote}", file=sys.stderr)
                    continue

                n = int(seg["n"])
                unr = float(seg["unmatched_n"]) / float(max(n, 1))
                batch = pa.Table.from_pydict(
                    {
                        "traj_csv": [member],
                        "n_points": [n],
                        "unmatched_ratio": [np.float32(unr)],
                        "t": [seg["t"]],
                        "lat": [np.asarray(seg["lat"], np.float32).tolist()],
                        "lon": [np.asarray(seg["lon"], np.float32).tolist()],
                        "y": [np.asarray(seg["y"], np.int32).tolist()],
                        "x": [np.asarray(seg["x"], np.int32).tolist()],
                        "is_matched": [np.asarray(seg["is_matched"], np.int8).tolist()],
                        "matched_distance": [np.asarray(seg["matched_distance"], np.float32).tolist()],
                    },
                    schema=schema,
                )
                out_writer.write_table(batch)
                wrote += 1
                if wrote % 10000 == 0:
                    print(f"[INFO] scanned={scanned} wrote={wrote}", file=sys.stderr)

    finally:
        out_writer.close()

    report = {
        "trajectory_zip": str(args.trajectory_zip),
        "out_parquet": str(args.out_parquet),
        "grid": {"H": grid.H, "W": grid.W, "bbox": grid.bbox.__dict__},
        "cfg": cfg.__dict__,
        "scanned_files": scanned,
        "kept_segments": wrote,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

