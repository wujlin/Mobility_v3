from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class DumpConfig:
    obs_len: int
    pred_len: int
    num_samples: int
    seed: int
    mode: str


def _load_split_ids(processed_dir: Path, split: str, splits_dir: Optional[Path]) -> Optional[np.ndarray]:
    if split == "all":
        return None
    sd = splits_dir if splits_dir is not None else (processed_dir / "splits")
    path = sd / f"{split}_ids.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.int64)


def _compute_window_counts(ptr: np.ndarray, traj_ids: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = ptr[traj_ids].astype(np.int64)
    end = ptr[traj_ids + 1].astype(np.int64)
    length = (end - start).astype(np.int64)
    win = np.maximum(length - int(window) + 1, 0).astype(np.int64)
    return start, length, win


def _sample_windows_random(
    *,
    positions: np.ndarray,
    ptr: np.ndarray,
    traj_ids: np.ndarray,
    cfg: DumpConfig,
) -> Dict[str, np.ndarray]:
    window = int(cfg.obs_len) + int(cfg.pred_len)
    start, _, win = _compute_window_counts(ptr, traj_ids, window)
    mask = win > 0
    if not np.any(mask):
        raise RuntimeError("No valid windows (traj too short).")
    traj_ids = traj_ids[mask]
    start = start[mask]
    win = win[mask]

    w = win.astype(np.float64)
    w = w / max(float(w.sum()), 1.0)

    rng = np.random.default_rng(int(cfg.seed))
    pick = rng.choice(traj_ids.shape[0], size=int(cfg.num_samples), replace=True, p=w)
    tid = traj_ids[pick]
    base = start[pick]
    max_t = win[pick]
    t0 = rng.integers(0, max_t, size=pick.shape[0], endpoint=False, dtype=np.int64)

    # Trip-level origin/destination (same definition as training/evaluation condition).
    trip_start = ptr[tid].astype(np.int64, copy=False)
    trip_end = (ptr[tid + 1] - 1).astype(np.int64, copy=False)
    origin_pos = positions[trip_start].astype(np.float32, copy=False)
    dest_pos = positions[trip_end].astype(np.float32, copy=False)

    # start_pos is the last observed position: pos[t0 + obs_len - 1]
    idx_start = base + t0 + int(cfg.obs_len) - 1
    start_pos = positions[idx_start].astype(np.float32, copy=False)

    targets = np.zeros((int(cfg.num_samples), int(cfg.pred_len), 2), dtype=np.float32)
    for i in range(int(cfg.num_samples)):
        s = int(base[i] + t0[i] + int(cfg.obs_len))
        e = int(s + int(cfg.pred_len))
        targets[i] = positions[s:e].astype(np.float32, copy=False)

    return {
        "origin_pos": origin_pos,
        "dest_pos": dest_pos,
        "start_pos": start_pos,
        "targets": targets,
        "traj_idx": tid.astype(np.int64, copy=False),
        "start_t": t0.astype(np.int64, copy=False),
    }


def _dump_gt_windows(
    *,
    processed_dir: Path,
    split: str,
    splits_dir: Optional[Path],
    out_npz: Path,
    cfg: DumpConfig,
) -> Dict[str, object]:
    h5_path = processed_dir / "trajectories" / "shenzhen_trajectories.h5"
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    traj_ids = _load_split_ids(processed_dir, split, splits_dir)
    with h5py.File(h5_path, "r") as f:
        ptr = f["traj_ptr"][:].astype(np.int64)
        positions = f["positions"][:].astype(np.float32)
        n_traj = int(len(ptr) - 1)

    if traj_ids is None:
        traj_ids = np.arange(n_traj, dtype=np.int64)

    if cfg.mode != "random":
        raise ValueError(f"Unknown --mode {cfg.mode} (currently only supports: random)")

    data = _sample_windows_random(positions=positions, ptr=ptr, traj_ids=traj_ids, cfg=cfg)

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "processed_dir": str(processed_dir),
        "split": str(split),
        "h5_path": str(h5_path),
        "config": {
            "obs_len": int(cfg.obs_len),
            "pred_len": int(cfg.pred_len),
            "num_samples": int(cfg.num_samples),
            "seed": int(cfg.seed),
            "mode": str(cfg.mode),
        },
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **data, meta=meta)
    return {"out_npz": str(out_npz), "N": int(data["targets"].shape[0]), "F": int(data["targets"].shape[1]), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump GT windows into a lightweight samples.npz (CPU-only).")
    p.add_argument("--processed_dir", type=str, required=True, help="processed dir containing trajectories/ and splits/")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--splits_dir", type=str, default=None, help="override splits dir (default: <processed_dir>/splits)")
    p.add_argument("--out_npz", type=str, required=True, help="output samples.npz path (contains targets/start_pos)")

    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12)
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", type=str, default="random", choices=["random"])
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = DumpConfig(
        obs_len=int(args.obs_len),
        pred_len=int(args.pred_len),
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        mode=str(args.mode),
    )
    out = _dump_gt_windows(
        processed_dir=Path(args.processed_dir),
        split=str(args.split),
        splits_dir=(Path(args.splits_dir) if args.splits_dir else None),
        out_npz=Path(args.out_npz),
        cfg=cfg,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
