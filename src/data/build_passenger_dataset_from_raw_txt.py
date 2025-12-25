"""
build_passenger_dataset_from_raw_txt.py

从深圳出租车原始 txt（GBK CSV）构建 *Passenger Trip*（status==1）数据集。

核心合同（你已拍板）：
- 只保留 status == 1（Passenger Trip）：这是 Navigation Policy，不混入 status==0 的巡游/搜索策略
- max_gap_s = 300：gap 视为因果断裂，不跨 gap 插值（切碎/切段）
- max_speed_kmh = 120：城市出租车物理/法律上限；超速边界视为 GPS 漂移或时间戳错误（切段）

输出（与现有训练/评估管线兼容）：
- <output_dir>/trajectories/shenzhen_trajectories.h5（TrajectoryStorage 统一格式）
- <output_dir>/splits/{train_ids,val_ids,test_ids}.npy（按轨迹起始时间排序分割）
- <output_dir>/vehicle_id_map.json（车牌字符串 -> int64）
- <output_dir>/preprocess_meta.json（可复现合同 + 统计）

注意：
- 本脚本不依赖 pandas；仅用标准库 + numpy + h5py（通过 TrajectoryStorage）。
- 生成完成后建议再跑 strict(train-only) 产物：
    python -m src.data.build_strict_products --processed_dir <output_dir> --backup
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.config.settings import GRID
from src.data.trajectories import TrajectoryStorage


TZ_SHANGHAI = timezone(timedelta(hours=8))
TZ_UTC = timezone.utc


def _parse_time_to_unix_seconds(s: str, tz_mode: str) -> int:
    dt = datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S")
    if tz_mode == "utc":
        return int(dt.replace(tzinfo=TZ_UTC).timestamp())
    if tz_mode == "shanghai":
        return int(dt.replace(tzinfo=TZ_SHANGHAI).timestamp())
    raise ValueError(f"Unknown --time_zone: {tz_mode} (expected: utc|shanghai)")


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Great-circle distance (meters).
    R = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 1e-12)))
    return R * c


@dataclass(frozen=True)
class BBox:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def contains(self, lat: float, lon: float) -> bool:
        return (self.min_lat <= lat <= self.max_lat) and (self.min_lon <= lon <= self.max_lon)


def _latlon_to_grid_yx(lat: np.ndarray, lon: np.ndarray, H: int, W: int, bbox: BBox, flip_y: bool) -> np.ndarray:
    # Linear bbox mapping; consistent with visualization defaults:
    # y=0 -> min_lat, y=H-1 -> max_lat; x=0 -> min_lon, x=W-1 -> max_lon.
    denom_y = float(max(H - 1, 1))
    denom_x = float(max(W - 1, 1))
    y01 = (lat - float(bbox.min_lat)) / max(float(bbox.max_lat - bbox.min_lat), 1e-12)
    x01 = (lon - float(bbox.min_lon)) / max(float(bbox.max_lon - bbox.min_lon), 1e-12)
    y01 = np.clip(y01, 0.0, 1.0)
    x01 = np.clip(x01, 0.0, 1.0)
    if bool(flip_y):
        y01 = 1.0 - y01
    y = (y01 * denom_y).astype(np.float32)
    x = (x01 * denom_x).astype(np.float32)
    return np.stack([y, x], axis=1).astype(np.float32)


def _read_vehicle_txt(path: Path, tz_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Returns (ts, lat, lon, status) as numpy arrays.
    ts_list: List[int] = []
    lat_list: List[float] = []
    lon_list: List[float] = []
    status_list: List[int] = []

    with path.open("r", encoding="gbk", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.int8)

        # Expected: name,time,jd,wd,status,v,angle,  (last col empty)
        for row in reader:
            if not row or len(row) < 7:
                continue
            try:
                ts = _parse_time_to_unix_seconds(row[1], tz_mode=tz_mode)
                lon = float(row[2])
                lat = float(row[3])
                status = int(float(row[4]))
            except Exception:
                continue
            ts_list.append(ts)
            lat_list.append(lat)
            lon_list.append(lon)
            status_list.append(status)

    ts = np.array(ts_list, dtype=np.int64)
    lat = np.array(lat_list, dtype=np.float64)
    lon = np.array(lon_list, dtype=np.float64)
    status = np.array(status_list, dtype=np.int8)
    return ts, lat, lon, status


def _sort_and_dedup_last(ts: np.ndarray, lat: np.ndarray, lon: np.ndarray, status: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if ts.size == 0:
        return ts, lat, lon, status

    order = np.argsort(ts, kind="mergesort")  # stable
    ts = ts[order]
    lat = lat[order]
    lon = lon[order]
    status = status[order]

    # Dedup by timestamp: keep last occurrence.
    out_ts: List[int] = []
    out_lat: List[float] = []
    out_lon: List[float] = []
    out_status: List[int] = []

    last_ts: Optional[int] = None
    for i in range(ts.shape[0]):
        t = int(ts[i])
        if last_ts is not None and t == last_ts:
            out_lat[-1] = float(lat[i])
            out_lon[-1] = float(lon[i])
            out_status[-1] = int(status[i])
            continue
        out_ts.append(t)
        out_lat.append(float(lat[i]))
        out_lon.append(float(lon[i]))
        out_status.append(int(status[i]))
        last_ts = t

    return (
        np.array(out_ts, dtype=np.int64),
        np.array(out_lat, dtype=np.float64),
        np.array(out_lon, dtype=np.float64),
        np.array(out_status, dtype=np.int8),
    )


def _split_passenger_segments(
    ts: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    status: np.ndarray,
    *,
    keep_status: int,
    max_gap_s: int,
    max_speed_kmh: float,
    bbox: BBox,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], Counter]:
    """
    Returns:
      segments: list of (ts, lat, lon) arrays
      counters: break/drop counters for reporting
    """
    counters: Counter = Counter()
    segments: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    seg_ts: List[int] = []
    seg_lat: List[float] = []
    seg_lon: List[float] = []

    prev_t: Optional[int] = None
    prev_lat: Optional[float] = None
    prev_lon: Optional[float] = None

    def flush(reason: str) -> None:
        nonlocal seg_ts, seg_lat, seg_lon, prev_t, prev_lat, prev_lon
        if seg_ts:
            segments.append(
                (
                    np.array(seg_ts, dtype=np.int64),
                    np.array(seg_lat, dtype=np.float64),
                    np.array(seg_lon, dtype=np.float64),
                )
            )
        seg_ts = []
        seg_lat = []
        seg_lon = []
        prev_t = None
        prev_lat = None
        prev_lon = None
        counters[f"flush_{reason}"] += 1

    for i in range(ts.shape[0]):
        st = int(status[i])
        if st != int(keep_status):
            if seg_ts:
                flush("status_end")
            continue

        t = int(ts[i])
        la = float(lat[i])
        lo = float(lon[i])

        if not bbox.contains(la, lo):
            counters["break_oob"] += 1
            if seg_ts:
                flush("oob")
            continue

        if prev_t is None:
            seg_ts.append(t)
            seg_lat.append(la)
            seg_lon.append(lo)
            prev_t, prev_lat, prev_lon = t, la, lo
            continue

        dt = int(t - prev_t)
        if dt <= 0:
            counters["skip_dt_nonpos"] += 1
            continue

        if int(max_gap_s) > 0 and dt > int(max_gap_s):
            counters["break_gap"] += 1
            flush("gap")
            seg_ts.append(t)
            seg_lat.append(la)
            seg_lon.append(lo)
            prev_t, prev_lat, prev_lon = t, la, lo
            continue

        d_m = _haversine_m(float(prev_lat), float(prev_lon), la, lo)
        sp_kmh = (d_m / float(dt)) * 3.6
        if float(max_speed_kmh) > 0 and sp_kmh > float(max_speed_kmh):
            counters["break_overspeed"] += 1
            flush("overspeed")
            seg_ts.append(t)
            seg_lat.append(la)
            seg_lon.append(lo)
            prev_t, prev_lat, prev_lon = t, la, lo
            continue

        seg_ts.append(t)
        seg_lat.append(la)
        seg_lon.append(lo)
        prev_t, prev_lat, prev_lon = t, la, lo

    if seg_ts:
        flush("eof")

    return segments, counters


def _segment_pass_filter(ts: np.ndarray, lat: np.ndarray, lon: np.ndarray, min_points: int, min_od_m: float) -> bool:
    if ts.shape[0] < int(min_points):
        return False
    if float(min_od_m) <= 0:
        return True
    d = _haversine_m(float(lat[0]), float(lon[0]), float(lat[-1]), float(lon[-1]))
    return d >= float(min_od_m)


def _vehicle_key_from_path(path: Path) -> str:
    # Use filename stem (e.g., "粤BA0P65") as the vehicle key.
    return path.stem


def _safe_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build passenger-only dataset from raw taxi txt files")
    parser.add_argument("--raw_gps_dir", type=str, default="data/raw/gps", help="raw txt directory")
    parser.add_argument("--output_dir", type=str, default="data/processed_passenger", help="output processed directory")
    parser.add_argument("--max_files", type=int, default=None, help="for debug: only process first N files")

    parser.add_argument("--keep_status", type=int, default=1, help="only keep this status (default: 1)")
    parser.add_argument("--max_gap_s", type=int, default=300)
    parser.add_argument("--max_speed_kmh", type=float, default=120.0)
    parser.add_argument("--min_points", type=int, default=10)
    parser.add_argument("--min_od_m", type=float, default=500.0, help="min origin-destination displacement (meters)")

    parser.add_argument("--time_zone", type=str, default="shanghai", choices=["utc", "shanghai"], help="interpret raw time strings")
    parser.add_argument("--flip_y", action="store_true", help="if your grid y-axis is flipped (rare); default assumes y=0->min_lat")

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_trips", type=int, default=1000, help="HDF5 append batch size (trips)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_gps_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)

    traj_path = out_dir / "trajectories" / "shenzhen_trajectories.h5"
    vehicle_map_path = out_dir / "vehicle_id_map.json"
    meta_path = out_dir / "preprocess_meta.json"

    if traj_path.exists():
        if bool(args.overwrite):
            traj_path.unlink()
        elif bool(args.backup):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            traj_path.rename(traj_path.with_suffix(traj_path.suffix + f".legacy.{ts}"))
        else:
            raise FileExistsError(f"{traj_path} already exists (use --backup or --overwrite)")

    files = sorted(raw_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No *.txt found in {raw_dir}")
    if args.max_files is not None:
        files = files[: int(args.max_files)]

    bbox = BBox(
        min_lat=float(GRID.min_lat),
        max_lat=float(GRID.max_lat),
        min_lon=float(GRID.min_lon),
        max_lon=float(GRID.max_lon),
    )
    H, W = int(GRID.H), int(GRID.W)

    # Vehicle id mapping (string -> int64)
    vehicle_to_id: Dict[str, int] = {}
    next_vid = 0

    # Global stats
    counters: Counter = Counter()
    trip_start_times: List[int] = []
    trip_vehicle_ids: List[int] = []

    TrajectoryStorage.create(traj_path, overwrite=True)
    storage = TrajectoryStorage(traj_path, mode="r+")

    batch: List[dict] = []

    for fi, fp in enumerate(files, start=1):
        vkey = _vehicle_key_from_path(fp)
        if vkey not in vehicle_to_id:
            vehicle_to_id[vkey] = int(next_vid)
            next_vid += 1
        vid = int(vehicle_to_id[vkey])

        ts, lat, lon, status = _read_vehicle_txt(fp, tz_mode=str(args.time_zone))
        counters["raw_points"] += int(ts.size)
        if ts.size == 0:
            counters["empty_files"] += 1
            continue

        ts, lat, lon, status = _sort_and_dedup_last(ts, lat, lon, status)

        segs, c = _split_passenger_segments(
            ts,
            lat,
            lon,
            status,
            keep_status=int(args.keep_status),
            max_gap_s=int(args.max_gap_s),
            max_speed_kmh=float(args.max_speed_kmh),
            bbox=bbox,
        )
        counters.update(c)

        for ts_seg, lat_seg, lon_seg in segs:
            counters["segments_all"] += 1
            if not _segment_pass_filter(
                ts_seg,
                lat_seg,
                lon_seg,
                min_points=int(args.min_points),
                min_od_m=float(args.min_od_m),
            ):
                if ts_seg.shape[0] < int(args.min_points):
                    counters["drop_too_short_points"] += 1
                else:
                    counters["drop_too_short_od"] += 1
                continue

            yx = _latlon_to_grid_yx(
                lat_seg,
                lon_seg,
                H=H,
                W=W,
                bbox=bbox,
                flip_y=bool(args.flip_y),
            )
            batch.append({"positions": yx, "timestamp": ts_seg.astype(np.int64), "vehicle_id": int(vid)})
            trip_start_times.append(int(ts_seg[0]))
            trip_vehicle_ids.append(int(vid))
            counters["trips_out"] += 1

            if len(batch) >= int(args.batch_trips):
                storage.append(batch)
                batch = []

        if fi % 50 == 0 or fi == len(files):
            print(f"[{fi}/{len(files)}] processed files | trips_out={int(counters['trips_out'])}")

    if batch:
        storage.append(batch)
        batch = []

    storage.close()

    n_trips = int(counters["trips_out"])
    if n_trips <= 0:
        raise RuntimeError("No trips generated. Check filters/bbox/status assumptions.")

    # Splits: sort by trip start time (time-based split)
    idx = np.arange(n_trips, dtype=np.int64)
    start_times = np.array(trip_start_times, dtype=np.int64)
    order = np.argsort(start_times, kind="mergesort")
    idx_sorted = idx[order]

    train_ratio = float(args.train_ratio)
    val_ratio = float(args.val_ratio)
    if not (0.0 < train_ratio < 1.0) or not (0.0 <= val_ratio < 1.0) or (train_ratio + val_ratio >= 1.0):
        raise ValueError("Invalid split ratios: require 0<train<1, 0<=val<1, train+val<1")

    n_train = int(round(n_trips * train_ratio))
    n_val = int(round(n_trips * val_ratio))
    n_train = max(1, min(n_train, n_trips - 2))
    n_val = max(1, min(n_val, n_trips - n_train - 1))
    n_test = n_trips - n_train - n_val
    if n_test <= 0:
        raise ValueError("Split results in empty test set; adjust ratios.")

    train_ids = idx_sorted[:n_train]
    val_ids = idx_sorted[n_train : n_train + n_val]
    test_ids = idx_sorted[n_train + n_val :]

    np.save(out_dir / "splits" / "train_ids.npy", train_ids)
    np.save(out_dir / "splits" / "val_ids.npy", val_ids)
    np.save(out_dir / "splits" / "test_ids.npy", test_ids)

    _safe_write_json(vehicle_map_path, {k: int(v) for k, v in vehicle_to_id.items()})

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "raw_gps_dir": str(raw_dir),
        "output_dir": str(out_dir),
        "config": {
            "keep_status": int(args.keep_status),
            "max_gap_s": int(args.max_gap_s),
            "max_speed_kmh": float(args.max_speed_kmh),
            "min_points": int(args.min_points),
            "min_od_m": float(args.min_od_m),
            "time_zone": str(args.time_zone),
            "flip_y": bool(args.flip_y),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
        },
        "grid_config": {"H": H, "W": W, **asdict(bbox)},
        "counts": {k: int(v) for k, v in counters.items()},
        "splits": {"train": int(train_ids.size), "val": int(val_ids.size), "test": int(test_ids.size)},
        "time_range": {
            "start_time_min": int(start_times.min()),
            "start_time_max": int(start_times.max()),
            "start_time_min_iso": datetime.fromtimestamp(int(start_times.min()), tz=TZ_SHANGHAI).isoformat(),
            "start_time_max_iso": datetime.fromtimestamp(int(start_times.max()), tz=TZ_SHANGHAI).isoformat(),
        },
    }
    _safe_write_json(meta_path, meta)

    print("[OK] wrote:")
    print(f"  - {traj_path}")
    print(f"  - {out_dir / 'splits'}/*.npy")
    print(f"  - {vehicle_map_path}")
    print(f"  - {meta_path}")
    print("\nNext (recommended):")
    print(f"  python -m src.data.build_strict_products --processed_dir {out_dir} --backup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
