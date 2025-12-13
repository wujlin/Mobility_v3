"""
build_strict_products.py

基于已有的 processed 产物（HDF5 trajectories + splits），重新生成严格（train-only）
的数据产物，避免信息泄漏，并补齐 source/metadata。

默认策略：
- normalization 统计量仅使用 train split
- nav_field 仅使用 train split
- 输出覆盖到 output_dir（可选择备份旧文件）

用法：
    python -m src.data.build_strict_products --processed_dir data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np


TZ_SHANGHAI = timezone(timedelta(hours=8))


def sha256_file(path: Path, prefix: int = 16) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    digest = sha.hexdigest()
    return digest[:prefix] if prefix else digest


def iso_ts(ts: int, tz: timezone = TZ_SHANGHAI) -> str:
    return datetime.fromtimestamp(int(ts), tz=tz).isoformat()


def backup_if_exists(path: Path) -> None:
    if not path.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".legacy.{ts}")
    path.rename(backup)


def resolve_grid_shape(processed_dir: Path, positions: np.ndarray) -> Tuple[int, int]:
    stats_file = processed_dir / "data_stats.json"
    if stats_file.exists():
        try:
            stats = json.loads(stats_file.read_text(encoding="utf-8"))
            grid = stats.get("grid_config") or {}
            if "H" in grid and "W" in grid:
                return int(grid["H"]), int(grid["W"])
        except Exception:
            pass

    nav_file = processed_dir / "nav_field.npz"
    if nav_file.exists():
        nav = np.load(nav_file)
        if "direction" in nav:
            d = nav["direction"]
            if d.ndim == 3 and d.shape[0] == 2:
                return int(d.shape[1]), int(d.shape[2])
            if d.ndim == 3 and d.shape[-1] == 2:
                return int(d.shape[0]), int(d.shape[1])

    # fallback: derive from data
    y_max = int(np.ceil(float(np.max(positions[:, 0])))) + 1
    x_max = int(np.ceil(float(np.max(positions[:, 1])))) + 1
    return y_max, x_max


def build_train_point_mask(traj_ptr: np.ndarray, train_ids: np.ndarray, n_points: int) -> np.ndarray:
    train_ids = train_ids.astype(np.int64)
    starts = traj_ptr[train_ids]
    ends = traj_ptr[train_ids + 1]

    diff = np.zeros(n_points + 1, dtype=np.int32)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    mask = np.cumsum(diff[:-1]) > 0
    return mask


def build_same_traj_diff_mask(traj_ptr: np.ndarray, n_points: int) -> np.ndarray:
    mask = np.ones(n_points - 1, dtype=bool)
    # 禁止跨轨迹边界的 diff：在每条轨迹的 last_index 位置会跨到下一条轨迹
    last_indices = traj_ptr[1:-1] - 1  # exclude the last trajectory
    last_indices = last_indices[(last_indices >= 0) & (last_indices < n_points - 1)]
    mask[last_indices] = False
    return mask


def compute_dt_stats(timestamps: np.ndarray, traj_ptr: np.ndarray, train_ids: np.ndarray, sample_traj: int = 500) -> Dict[str, Any]:
    rng = np.random.default_rng(0)
    n_traj = len(traj_ptr) - 1
    if n_traj == 0:
        return {}

    train_ids = train_ids.astype(np.int64)
    sample_traj = int(min(sample_traj, len(train_ids)))
    sample_ids = rng.choice(train_ids, size=sample_traj, replace=False) if sample_traj > 0 else np.array([], dtype=np.int64)

    dts = []
    for tid in sample_ids:
        start = int(traj_ptr[tid])
        end = int(traj_ptr[tid + 1])
        if end - start < 2:
            continue
        dt = np.diff(timestamps[start:end].astype(np.int64))
        dts.append(dt)

    if not dts:
        return {"count": 0}

    dts = np.concatenate(dts)
    dts = dts[dts >= 0]
    if dts.size == 0:
        return {"count": 0}

    vals, counts = np.unique(dts, return_counts=True)
    top_idx = np.argsort(counts)[::-1][:10]
    top = [(int(vals[i]), int(counts[i])) for i in top_idx]

    return {
        "count": int(dts.size),
        "min": int(dts.min()),
        "max": int(dts.max()),
        "p50": float(np.percentile(dts, 50)),
        "p95": float(np.percentile(dts, 95)),
        "p99": float(np.percentile(dts, 99)),
        "top_dt": top,
    }


def compute_train_stats(
    processed_dir: Path,
    h5_path: Path,
    train_ids: np.ndarray,
) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        positions = f["positions"][:].astype(np.float32)
        timestamps = f["timestamps"][:].astype(np.int64)
        traj_ptr = f["traj_ptr"][:].astype(np.int64)
        start_time = f["meta/start_time"][:].astype(np.int64)
        end_time = f["meta/end_time"][:].astype(np.int64)

    n_points = positions.shape[0]
    grid_h, grid_w = resolve_grid_shape(processed_dir, positions)

    point_mask = build_train_point_mask(traj_ptr, train_ids, n_points)
    pos_train = positions[point_mask]

    pos_min = pos_train.min(axis=0)
    pos_max = pos_train.max(axis=0)

    same_traj = build_same_traj_diff_mask(traj_ptr, n_points)
    vel = positions[1:] - positions[:-1]  # step displacement
    vel_mask = same_traj & point_mask[:-1] & point_mask[1:]

    vel_train = vel[vel_mask]
    vel_mean = vel_train.mean(axis=0) if vel_train.size else np.array([0.0, 0.0], dtype=np.float32)
    vel_std = vel_train.std(axis=0) if vel_train.size else np.array([1.0, 1.0], dtype=np.float32)

    lengths = traj_ptr[1:] - traj_ptr[:-1]
    train_lengths = lengths[train_ids]

    dt_stats = compute_dt_stats(timestamps, traj_ptr, train_ids)

    st_min = int(start_time[train_ids].min())
    et_max = int(end_time[train_ids].max())

    return {
        "grid_config": {"H": grid_h, "W": grid_w},
        "normalization": {
            "pos_min": [float(pos_min[0]), float(pos_min[1])],
            "pos_max": [float(pos_max[0]), float(pos_max[1])],
            "vel_mean": [float(vel_mean[0]), float(vel_mean[1])],
            "vel_std": [float(vel_std[0]), float(vel_std[1])],
            "nav_scale": 1.0,
        },
        "trip_length_stats": {
            "min": int(train_lengths.min()),
            "max": int(train_lengths.max()),
            "mean": float(train_lengths.mean()),
        },
        "time_stats": {
            "train_start_time_min": st_min,
            "train_end_time_max": et_max,
            "train_date_range": [iso_ts(st_min), iso_ts(et_max)],
            "dt_stats_sample": dt_stats,
        },
    }


def estimate_nav_field(
    positions: np.ndarray,
    traj_ptr: np.ndarray,
    train_ids: np.ndarray,
    grid_shape: Tuple[int, int],
    min_speed: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_points = positions.shape[0]
    H, W = grid_shape

    point_mask = build_train_point_mask(traj_ptr, train_ids, n_points)
    same_traj = build_same_traj_diff_mask(traj_ptr, n_points)
    step_mask = same_traj & point_mask[:-1] & point_mask[1:]

    p0 = positions[:-1][step_mask]
    dv = (positions[1:] - positions[:-1])[step_mask]

    speed = np.linalg.norm(dv, axis=1)
    valid = speed > float(min_speed)
    if not np.any(valid):
        direction = np.zeros((2, H, W), dtype=np.float32)
        speed_mean = np.zeros((H, W), dtype=np.float32)
        count = np.zeros((H, W), dtype=np.float32)
        return direction, speed_mean, count

    p0 = p0[valid]
    dv = dv[valid]
    speed = speed[valid]

    yi = np.clip(p0[:, 0].astype(np.int64), 0, H - 1)
    xi = np.clip(p0[:, 1].astype(np.int64), 0, W - 1)
    flat = yi * W + xi
    flat_size = H * W

    count = np.bincount(flat, minlength=flat_size).astype(np.float32)
    sum_dy = np.bincount(flat, weights=dv[:, 0], minlength=flat_size).astype(np.float32)
    sum_dx = np.bincount(flat, weights=dv[:, 1], minlength=flat_size).astype(np.float32)
    sum_speed = np.bincount(flat, weights=speed, minlength=flat_size).astype(np.float32)

    count_hw = count.reshape(H, W)
    dy_hw = sum_dy.reshape(H, W)
    dx_hw = sum_dx.reshape(H, W)
    speed_hw = sum_speed.reshape(H, W)

    mean_dy = np.divide(dy_hw, count_hw, out=np.zeros_like(dy_hw), where=count_hw > 0)
    mean_dx = np.divide(dx_hw, count_hw, out=np.zeros_like(dx_hw), where=count_hw > 0)
    mean_speed = np.divide(speed_hw, count_hw, out=np.zeros_like(speed_hw), where=count_hw > 0)

    norm = np.sqrt(mean_dy**2 + mean_dx**2)
    dir_y = np.divide(mean_dy, norm, out=np.zeros_like(mean_dy), where=norm > 1e-8)
    dir_x = np.divide(mean_dx, norm, out=np.zeros_like(mean_dx), where=norm > 1e-8)

    direction = np.stack([dir_y, dir_x], axis=0).astype(np.float32)  # (2, H, W)
    return direction, mean_speed.astype(np.float32), count_hw.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build strict(train-only) data products")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="processed data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="output directory (default: overwrite in processed_dir)")
    parser.add_argument("--h5_file", type=str, default=None, help="trajectory h5 file (default: {processed_dir}/trajectories/shenzhen_trajectories.h5)")
    parser.add_argument("--backup", action="store_true", help="backup existing output files before overwrite")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir) if args.output_dir else processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = Path(args.h5_file) if args.h5_file else (processed_dir / "trajectories" / "shenzhen_trajectories.h5")
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    train_ids_path = processed_dir / "splits" / "train_ids.npy"
    if not train_ids_path.exists():
        raise FileNotFoundError(train_ids_path)
    train_ids = np.load(train_ids_path).astype(np.int64)

    resample_meta_path = processed_dir / "resample_meta.json"
    resample_meta = None
    if resample_meta_path.exists():
        try:
            resample_meta = json.loads(resample_meta_path.read_text(encoding="utf-8"))
        except Exception:
            resample_meta = None

    # 1) data_stats.json (train-only)
    print("=" * 60)
    print("BUILD STRICT PRODUCTS")
    print("=" * 60)
    print(f"processed_dir: {processed_dir}")
    print(f"output_dir:    {output_dir}")
    print(f"h5_file:       {h5_path}")
    print(f"train_ids:     {train_ids_path} (n={len(train_ids)})")

    stats_out = output_dir / "data_stats.json"
    nav_out = output_dir / "nav_field.npz"

    if args.backup:
        backup_if_exists(stats_out)
        backup_if_exists(nav_out)

    base_stats = compute_train_stats(processed_dir, h5_path, train_ids)

    # 补齐 legacy grid_config 的地理边界（如果存在）
    legacy_stats_file = processed_dir / "data_stats.json"
    if legacy_stats_file.exists():
        try:
            legacy = json.loads(legacy_stats_file.read_text(encoding="utf-8"))
            legacy_grid = legacy.get("grid_config") or {}
            for k in ["min_lat", "max_lat", "min_lon", "max_lon"]:
                if k in legacy_grid:
                    base_stats["grid_config"][k] = float(legacy_grid[k])
        except Exception:
            pass

    created_at = datetime.now(tz=TZ_SHANGHAI).isoformat()
    source = {
        "split": "train",
        "trajectory_ids_file": str(train_ids_path.relative_to(processed_dir)),
        "trajectory_ids_sha256": sha256_file(train_ids_path),
        "trajectories_h5_file": str(h5_path.relative_to(processed_dir.parent)),
        "trajectories_h5_sha256": sha256_file(h5_path),
        "date_range": base_stats["time_stats"]["train_date_range"],
    }

    # Phase B: dt-fixed resampling contract (optional)
    if resample_meta is not None:
        source["resample_meta_file"] = str(resample_meta_path.relative_to(processed_dir))
        source["resample_meta_sha256"] = sha256_file(resample_meta_path)
        cfg = (resample_meta.get("config") or {}) if isinstance(resample_meta, dict) else {}
        if isinstance(cfg, dict) and "dt_fixed" in cfg:
            try:
                base_stats["time_stats"]["dt_fixed"] = int(cfg["dt_fixed"])
                base_stats["time_stats"]["resample_config"] = cfg
            except Exception:
                pass

    out_stats = {
        "created_at": created_at,
        "source": source,
        "grid_config": base_stats["grid_config"],
        "normalization": base_stats["normalization"],
        "trip_length_stats": base_stats["trip_length_stats"],
        "time_stats": base_stats["time_stats"],
    }
    stats_out.write_text(json.dumps(out_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {stats_out}")

    # 2) nav_field (train-only)
    with h5py.File(h5_path, "r") as f:
        positions = f["positions"][:].astype(np.float32)
        traj_ptr = f["traj_ptr"][:].astype(np.int64)

    grid_shape = (int(out_stats["grid_config"]["H"]), int(out_stats["grid_config"]["W"]))
    direction, speed, count = estimate_nav_field(positions, traj_ptr, train_ids, grid_shape)

    nav_meta = {
        "created_at": created_at,
        "source_split": "train",
        "trajectory_ids_sha256": source["trajectory_ids_sha256"],
        "trajectories_h5_sha256": source["trajectories_h5_sha256"],
        "grid_shape": list(grid_shape),
        "direction_shape": list(direction.shape),
        "note": "direction/speed/count are estimated from train split only (step displacement).",
    }

    if "dt_fixed" in base_stats.get("time_stats", {}):
        nav_meta["dt_fixed_seconds"] = base_stats["time_stats"]["dt_fixed"]
    if "resample_config" in base_stats.get("time_stats", {}):
        nav_meta["resample_config"] = base_stats["time_stats"]["resample_config"]
    if "resample_meta_sha256" in source:
        nav_meta["resample_meta_sha256"] = source["resample_meta_sha256"]

    # metadata 使用 object 存储，需要 allow_pickle=True 加载（NavField 已支持）
    np.savez(
        nav_out,
        direction=direction.astype(np.float32),
        speed=speed.astype(np.float32),
        count=count.astype(np.float32),
        metadata=np.array(nav_meta, dtype=object),
    )
    print(f"[OK] wrote {nav_out}")

    print("=" * 60)
    print("DONE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

