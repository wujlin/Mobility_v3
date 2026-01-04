from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import datetime as dt
import io
import json
import math
import os
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


def _matched_type_available(v: str) -> bool:
    """
    WorldTrace 的 matched_type 在不同导出版本里可能不是布尔值（可能是字符串/类别/数字）。
    这里做“可用性”判定：只要不是空/0/false/none/unmatched/nan，就认为有 matched 信息可参考。
    """
    v = (v or "").strip().lower()
    if not v:
        return False
    return v not in {"0", "false", "f", "no", "n", "none", "null", "unmatched", "nan"}


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


def _parse_time_s(v: str) -> Optional[int]:
    """
    WorldTrace 的 time 字段在不同导出版本里可能是：
    - 数字（epoch seconds / ms）
    - 字符串（"YYYY-MM-DD HH:MM:SS" / ISO8601）

    这里统一转成 int 秒（UTC 假设仅用于绝对值；我们只关心相邻点的 Δt）。
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None

    # 1) numeric epoch (s / ms)
    try:
        fv = float(s)
        if fv > 1e12:  # likely ms
            fv = fv / 1000.0
        return int(fv)
    except (TypeError, ValueError):
        pass

    # 2) common datetime formats
    s19 = s[:19]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            t = dt.datetime.strptime(s19, fmt).replace(tzinfo=dt.timezone.utc)
            return int(t.timestamp())
        except Exception:
            continue

    # 3) ISO fallback (handles fractional seconds / timezone offsets)
    try:
        s2 = s.replace("Z", "+00:00")
        t = dt.datetime.fromisoformat(s2)
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return int(t.timestamp())
    except Exception:
        return None


def _pick_coord(row: Dict[str, str], cfg: SegmentConfig) -> Tuple[Optional[float], Optional[float], int, Optional[float]]:
    """
    Return (lat, lon, is_matched, matched_distance).
    Uses matched_* if it is available and passes a conservative quality gate; otherwise fallback to raw lat/lon.
    """
    md = _safe_float(row.get("matched_distance", ""))
    mt_avail = _matched_type_available(row.get("matched_type", ""))
    lat_m = _safe_float(row.get("matched_latitude", ""))
    lon_m = _safe_float(row.get("matched_longitude", ""))

    # Prefer matched coords if:
    # - matched coords exist, AND
    # - (matched_distance exists and <= threshold) OR (matched_distance missing but matched_type says available)
    if lat_m is not None and lon_m is not None:
        if (md is not None and md <= cfg.matched_distance_max_m) or (md is None and mt_avail):
            return lat_m, lon_m, 1, md

    lat = _safe_float(row.get("latitude", ""))
    lon = _safe_float(row.get("longitude", ""))
    return lat, lon, 0, md


def _iter_csv_rows_from_zip(zf: zipfile.ZipFile, member: str) -> Iterator[Dict[str, str]]:
    with zf.open(member, "r") as f:
        # TextIOWrapper is faster than per-line decode generator for large CSV streams.
        text = io.TextIOWrapper(f, encoding="utf-8", errors="ignore", newline="")
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

    min_lon, max_lon = float(bbox.min_lon), float(bbox.max_lon)
    min_lat, max_lat = float(bbox.min_lat), float(bbox.max_lat)
    lon_span = max(max_lon - min_lon, 1e-12)
    lat_span = max(max_lat - min_lat, 1e-12)

    H, W = int(grid.H), int(grid.W)
    inv_lon = float(W) / lon_span  # multiply instead of divide per-row
    inv_lat = float(H) / lat_span

    def _new_seg() -> Dict[str, List]:
        return {
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

    def _flush():
        nonlocal cur, last_t
        if cur is not None and cur["n"] >= cfg.min_segment_points:
            segments.append(cur)
        cur = None
        last_t = None

    for row in rows:
        t = _parse_time_s(row.get("time", ""))
        if t is None:
            continue

        lat, lon, is_matched, md = _pick_coord(row, cfg)
        if lat is None or lon is None:
            continue

        # Scalar bbox check (avoid per-row numpy allocations)
        if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
            _flush()
            continue

        if cur is None:
            cur = _new_seg()
            last_t = None

        if last_t is not None and (t - last_t) > cfg.dt_gap_s:
            _flush()
            cur = _new_seg()

        last_t = t
        # Scalar lat/lon -> y/x (avoid per-row numpy allocations)
        x0 = int((lon - min_lon) * inv_lon)
        y0 = int((max_lat - lat) * inv_lat)
        if not (0 <= x0 < W and 0 <= y0 < H):
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


_WORKER_ZIP_PATH: Optional[str] = None
_WORKER_ZF: Optional[zipfile.ZipFile] = None


def _get_worker_zip(zip_path: str) -> zipfile.ZipFile:
    global _WORKER_ZIP_PATH, _WORKER_ZF
    if _WORKER_ZF is None or _WORKER_ZIP_PATH != zip_path:
        _WORKER_ZIP_PATH = zip_path
        _WORKER_ZF = zipfile.ZipFile(zip_path, "r")
    return _WORKER_ZF


def _process_member_chunk(
    zip_path: str,
    members: List[str],
    *,
    bbox_dict: Dict[str, float],
    grid_hw: Tuple[int, int],
    cfg_dict: Dict[str, object],
) -> Dict[str, object]:
    """
    Worker: process a chunk of trajectory CSV members and return kept segments only.
    This keeps IPC small because only rare positive segments are returned.
    """
    bbox = BBox(**bbox_dict)
    H, W = grid_hw
    grid = GridSpec(H=int(H), W=int(W), bbox=bbox)
    cfg = SegmentConfig(
        dt_gap_s=int(cfg_dict["dt_gap_s"]),
        min_segment_points=int(cfg_dict["min_segment_points"]),
        matched_distance_max_m=float(cfg_dict["matched_distance_max_m"]),
        max_unmatched_ratio=float(cfg_dict["max_unmatched_ratio"]),
    )

    zf = _get_worker_zip(zip_path)
    kept: List[Tuple[str, Dict[str, List]]] = []
    for member in members:
        try:
            rows = _iter_csv_rows_from_zip(zf, member)
            segs = _split_bbox_segments(rows, bbox=bbox, grid=grid, cfg=cfg)
            seg = _select_longest_segment(segs)
            if seg is None or not _segment_ok(seg, cfg):
                continue
            kept.append((member, seg))
        except Exception:
            continue

    return {"scanned": len(members), "kept": kept}


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract bbox segments from WorldTrace Trajectory.zip (one row per segment).")
    ap.add_argument("--trajectory_zip", type=Path, required=True, help="Path to Trajectory.zip")
    ap.add_argument("--out_parquet", type=Path, required=True, help="Output parquet (one row per segment)")
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Override bbox in EPSG:4326. Default: Detroit core bbox.",
    )
    ap.add_argument("--grid_h", type=int, default=1024, help="Grid height H (default: 1024)")
    ap.add_argument("--grid_w", type=int, default=1024, help="Grid width  W (default: 1024)")
    ap.add_argument("--limit_files", type=int, default=0, help="Debug limit on number of csv files (0=no limit)")
    ap.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Parallel workers for scanning CSV members (0=auto, 1=disable).",
    )
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=5000,
        help="Chunk size for parallel scanning (only used when num_workers>1).",
    )
    ap.add_argument("--dt_gap_s", type=int, default=5)
    ap.add_argument("--min_segment_points", type=int, default=120)
    ap.add_argument("--matched_distance_max_m", type=float, default=30.0)
    ap.add_argument("--max_unmatched_ratio", type=float, default=0.20)
    args = ap.parse_args()

    if pa is None or pq is None:
        raise SystemExit("pyarrow is required for --out_parquet. Install: pip/conda install pyarrow")

    if args.bbox is None:
        grid = _default_detroit_core_grid()
        if int(args.grid_h) != int(grid.H) or int(args.grid_w) != int(grid.W):
            grid = GridSpec(H=int(args.grid_h), W=int(args.grid_w), bbox=grid.bbox)
    else:
        min_lon, min_lat, max_lon, max_lat = map(float, args.bbox)
        bbox = BBox(min_lon=min_lon, max_lon=max_lon, min_lat=min_lat, max_lat=max_lat)
        grid = GridSpec(H=int(args.grid_h), W=int(args.grid_w), bbox=bbox)
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
    scanned = 0
    wrote = 0
    try:
        with zipfile.ZipFile(args.trajectory_zip, "r") as zf:
            members = [m for m in zf.namelist() if m.endswith(".csv")]
        if args.limit_files:
            members = members[: int(args.limit_files)]

        num_workers = int(args.num_workers)
        if num_workers <= 0:
            num_workers = os.cpu_count() or 1
        chunk_size = max(1, int(args.chunk_size))

        def _write_one(member: str, seg: Dict[str, List]) -> None:
            nonlocal wrote
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

        if num_workers <= 1:
            with zipfile.ZipFile(args.trajectory_zip, "r") as zf:
                for member in members:
                    scanned += 1
                    rows = _iter_csv_rows_from_zip(zf, member)
                    segs = _split_bbox_segments(rows, bbox=grid.bbox, grid=grid, cfg=cfg)
                    seg = _select_longest_segment(segs)
                    if seg is None or not _segment_ok(seg, cfg):
                        if scanned % 50000 == 0:
                            print(f"[INFO] scanned={scanned} wrote={wrote}", file=sys.stderr)
                        continue
                    _write_one(member, seg)
                    if wrote % 10000 == 0:
                        print(f"[INFO] scanned={scanned} wrote={wrote}", file=sys.stderr)
        else:
            zip_path = str(args.trajectory_zip)
            chunks = [members[i : i + chunk_size] for i in range(0, len(members), chunk_size)]
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                futs = [
                    ex.submit(
                        _process_member_chunk,
                        zip_path,
                        chunk,
                        bbox_dict=grid.bbox.__dict__,
                        grid_hw=(grid.H, grid.W),
                        cfg_dict=cfg.__dict__,
                    )
                    for chunk in chunks
                ]
                for fut in as_completed(futs):
                    r = fut.result()
                    scanned += int(r.get("scanned", 0))
                    for member, seg in (r.get("kept") or []):  # type: ignore[assignment]
                        _write_one(member, seg)
                    if scanned % 50000 == 0:
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
