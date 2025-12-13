"""
build_dt_fixed_dataset.py

Phase B（paper strict）数据集生成脚本：
将 Phase A 的不规则采样 HDF5 轨迹重采样到固定 dt（默认 30s），输出到新的 processed 目录。

核心约定（与 docs/TASK_DEFINITION.md 对齐）：
- 任务仍是 KnownDestination，vel 语义仍是 step displacement（步位移）
- 重采样只改变 timestamps/positions 的采样间隔语义，使每一步对应 dt_fixed 秒
- 处理规则必须可复现：去重/乱序/gap/最小长度阈值写入 resample_meta.json

用法：
    python -m src.data.build_dt_fixed_dataset \
      --input_processed_dir data/processed \
      --output_processed_dir data/processed_dt30 \
      --dt_fixed 30 \
      --max_gap 300 \
      --min_length 10
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional, Tuple

import h5py
import numpy as np

from src.data.trajectories import TrajectoryStorage


TZ_SHANGHAI = timezone(timedelta(hours=8))


def backup_if_exists(path: Path) -> None:
    if not path.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".legacy.{ts}")
    path.rename(backup)


def _dedup_timestamps(
    timestamps: np.ndarray,
    positions: np.ndarray,
    method: Literal["mean", "first", "last"],
) -> Tuple[np.ndarray, np.ndarray]:
    if timestamps.size == 0:
        return timestamps, positions

    if method == "first":
        uniq_ts, first_idx = np.unique(timestamps, return_index=True)
        order = np.argsort(first_idx)
        uniq_ts = uniq_ts[order]
        first_idx = first_idx[order]
        return uniq_ts.astype(np.int64), positions[first_idx].astype(np.float32)

    if method == "last":
        uniq_ts, last_idx = np.unique(timestamps[::-1], return_index=True)
        last_idx = (len(timestamps) - 1) - last_idx
        order = np.argsort(last_idx)
        uniq_ts = uniq_ts[order]
        last_idx = last_idx[order]
        return uniq_ts.astype(np.int64), positions[last_idx].astype(np.float32)

    # method == "mean"
    uniq_ts, inv = np.unique(timestamps, return_inverse=True)
    counts = np.bincount(inv).astype(np.float32)
    sums = np.zeros((uniq_ts.shape[0], 2), dtype=np.float64)
    np.add.at(sums, inv, positions.astype(np.float64))
    mean_pos = (sums / counts[:, None]).astype(np.float32)
    return uniq_ts.astype(np.int64), mean_pos


def _resample_one(
    positions: np.ndarray,
    timestamps: np.ndarray,
    dt_fixed: int,
    min_length: int,
    max_gap: int,
    dedup: Literal["mean", "first", "last"],
    require_monotonic: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    if positions.shape[0] < 2:
        return None, None, "too_short_raw"

    ts = timestamps.astype(np.int64)
    pos = positions.astype(np.float32)

    dt = np.diff(ts)
    if require_monotonic and np.any(dt < 0):
        return None, None, "non_monotonic"

    ts, pos = _dedup_timestamps(ts, pos, method=dedup)
    if ts.shape[0] < 2:
        return None, None, "too_short_dedup"

    dt2 = np.diff(ts)
    if np.any(dt2 <= 0):
        return None, None, "invalid_dt_after_dedup"

    if max_gap > 0 and int(dt2.max()) > int(max_gap):
        return None, None, "gap_too_large"

    duration = int(ts[-1] - ts[0])
    min_needed = int(dt_fixed) * int(max(min_length - 1, 1))
    if duration < min_needed:
        return None, None, "too_short_duration"

    t_new = np.arange(int(ts[0]), int(ts[-1]) + 1, int(dt_fixed), dtype=np.int64)
    if t_new.shape[0] < int(min_length):
        return None, None, "too_short_resampled"

    xp = ts.astype(np.float64)
    y = pos[:, 0].astype(np.float64)
    x = pos[:, 1].astype(np.float64)
    t_new_f = t_new.astype(np.float64)

    y_new = np.interp(t_new_f, xp, y).astype(np.float32)
    x_new = np.interp(t_new_f, xp, x).astype(np.float32)
    pos_new = np.stack([y_new, x_new], axis=1).astype(np.float32)
    return pos_new, t_new.astype(np.int64), None


@dataclass(frozen=True)
class ResampleConfig:
    dt_fixed: int
    min_length: int
    max_gap: int
    dedup: str
    require_monotonic: bool


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dt-fixed processed dataset (Phase B)")
    parser.add_argument("--input_processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_processed_dir", type=str, default="data/processed_dt30")
    parser.add_argument("--input_h5", type=str, default=None)
    parser.add_argument("--output_h5", type=str, default=None)

    parser.add_argument("--dt_fixed", type=int, default=30)
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--max_gap", type=int, default=300, help="max allowed raw gap (seconds); <=0 disables")
    parser.add_argument("--dedup", type=str, choices=["mean", "first", "last"], default="mean")
    parser.add_argument("--allow_non_monotonic", action="store_true", help="debug only: do not drop non-monotonic trajectories")

    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--max_traj", type=int, default=None, help="for quick debug; process only first N trajectories")
    parser.add_argument("--backup", action="store_true", help="backup existing outputs before overwrite")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing outputs (dangerous)")
    args = parser.parse_args()

    in_dir = Path(args.input_processed_dir)
    out_dir = Path(args.output_processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)

    input_h5 = Path(args.input_h5) if args.input_h5 else (in_dir / "trajectories" / "shenzhen_trajectories.h5")
    output_h5 = Path(args.output_h5) if args.output_h5 else (out_dir / "trajectories" / "shenzhen_trajectories.h5")
    if not input_h5.exists():
        raise FileNotFoundError(input_h5)

    if output_h5.exists():
        if args.overwrite:
            output_h5.unlink()
        elif args.backup:
            backup_if_exists(output_h5)
        else:
            raise FileExistsError(f"{output_h5} already exists (use --backup or --overwrite)")

    resample_meta_path = out_dir / "resample_meta.json"
    if resample_meta_path.exists():
        if args.overwrite:
            resample_meta_path.unlink()
        elif args.backup:
            backup_if_exists(resample_meta_path)
        else:
            raise FileExistsError(f"{resample_meta_path} already exists (use --backup or --overwrite)")

    old_to_new_path = out_dir / "resample_old_to_new.npy"
    new_to_old_path = out_dir / "resample_new_to_old.npy"
    for p in [old_to_new_path, new_to_old_path]:
        if p.exists():
            if args.overwrite:
                p.unlink()
            elif args.backup:
                backup_if_exists(p)
            else:
                raise FileExistsError(f"{p} already exists (use --backup or --overwrite)")

    cfg = ResampleConfig(
        dt_fixed=int(args.dt_fixed),
        min_length=int(args.min_length),
        max_gap=int(args.max_gap),
        dedup=str(args.dedup),
        require_monotonic=not bool(args.allow_non_monotonic),
    )

    print("=" * 60)
    print("BUILD DT-FIXED DATASET")
    print("=" * 60)
    print(f"input_processed_dir:  {in_dir}")
    print(f"output_processed_dir: {out_dir}")
    print(f"input_h5:             {input_h5}")
    print(f"output_h5:            {output_h5}")
    print(f"dt_fixed={cfg.dt_fixed}s, min_length={cfg.min_length}, max_gap={cfg.max_gap}, dedup={cfg.dedup}")

    # Load input splits (old ids)
    splits_in = in_dir / "splits"
    train_old = np.load(splits_in / "train_ids.npy").astype(np.int64)
    val_old = np.load(splits_in / "val_ids.npy").astype(np.int64)
    test_old = np.load(splits_in / "test_ids.npy").astype(np.int64)

    # Load HDF5 (single read; faster than per-traj random IO on /mnt/* mounts)
    print("Loading input HDF5 into memory (positions/timestamps/ptr/meta)...")
    with h5py.File(input_h5, "r") as f:
        positions = f["positions"][:].astype(np.float32)
        timestamps = f["timestamps"][:].astype(np.int64)
        traj_ptr = f["traj_ptr"][:].astype(np.int64)
        vehicle_id = f["meta/vehicle_id"][:].astype(np.int64) if "meta/vehicle_id" in f else None

    n_traj = int(len(traj_ptr) - 1)
    if args.max_traj is not None:
        n_traj = int(min(n_traj, int(args.max_traj)))
    print(f"Input trajectories: {len(traj_ptr) - 1} (processing={n_traj})")

    old_to_new = np.full((len(traj_ptr) - 1,), -1, dtype=np.int64)
    new_to_old = []
    dropped = Counter()

    TrajectoryStorage.create(output_h5, overwrite=True)
    batch = []
    with TrajectoryStorage(output_h5, mode="r+") as out_store:
        for tid in range(n_traj):
            start = int(traj_ptr[tid])
            end = int(traj_ptr[tid + 1])
            pos = positions[start:end]
            ts = timestamps[start:end]

            pos_new, ts_new, reason = _resample_one(
                positions=pos,
                timestamps=ts,
                dt_fixed=cfg.dt_fixed,
                min_length=cfg.min_length,
                max_gap=cfg.max_gap,
                dedup=cfg.dedup,  # type: ignore[arg-type]
                require_monotonic=cfg.require_monotonic,
            )
            if reason is not None or pos_new is None or ts_new is None:
                dropped[reason or "unknown"] += 1
                continue

            new_id = len(new_to_old)
            old_to_new[tid] = new_id
            new_to_old.append(tid)

            batch.append(
                {
                    "positions": pos_new.astype(np.float32),
                    "timestamp": ts_new.astype(np.int64),
                    "vehicle_id": int(vehicle_id[tid]) if vehicle_id is not None else -1,
                }
            )

            if len(batch) >= int(args.batch_size):
                out_store.append(batch)
                batch = []

        if batch:
            out_store.append(batch)

    n_out = int(len(new_to_old))
    print(f"Output trajectories: {n_out} (dropped={n_traj - n_out})")
    if dropped:
        print("Drop reasons:", dict(dropped))

    # Write mapping
    np.save(old_to_new_path, old_to_new)
    np.save(new_to_old_path, np.array(new_to_old, dtype=np.int64))

    # Build new splits by mapping old ids -> new ids, dropping invalid (-1)
    def _map_split(old_ids: np.ndarray) -> np.ndarray:
        mapped = old_to_new[old_ids]
        return mapped[mapped >= 0].astype(np.int64)

    train_new = _map_split(train_old)
    val_new = _map_split(val_old)
    test_new = _map_split(test_old)

    splits_out = out_dir / "splits"
    np.save(splits_out / "train_ids.npy", train_new)
    np.save(splits_out / "val_ids.npy", val_new)
    np.save(splits_out / "test_ids.npy", test_new)

    # Write resample meta (contract input for build_strict_products)
    created_at = datetime.now(tz=TZ_SHANGHAI).isoformat()
    meta = {
        "created_at": created_at,
        "input_processed_dir": str(in_dir),
        "output_processed_dir": str(out_dir),
        "input_h5": str(input_h5),
        "output_h5": str(output_h5),
        "config": asdict(cfg),
        "num_traj_in": int(len(traj_ptr) - 1),
        "num_traj_processed": int(n_traj),
        "num_traj_out": int(n_out),
        "dropped": dict(dropped),
        "splits": {
            "train_in": int(train_old.shape[0]),
            "val_in": int(val_old.shape[0]),
            "test_in": int(test_old.shape[0]),
            "train_out": int(train_new.shape[0]),
            "val_out": int(val_new.shape[0]),
            "test_out": int(test_new.shape[0]),
        },
    }
    resample_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Persist dt_fixed into HDF5 attrs for redundancy
    with h5py.File(output_h5, "a") as f:
        f.attrs["dt_fixed_seconds"] = int(cfg.dt_fixed)
        f.attrs["resample_meta_path"] = str(resample_meta_path)

    print(f"[OK] wrote {resample_meta_path}")
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
