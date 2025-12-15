"""
Compute GT macroscopic metrics (MSD curve / Rog) for a window subset.

用途：
- 在没有 torch/matplotlib 的环境里（或只想做 GT 对照）生成一份可复现的 GT 宏观指标 json；
- 供可视化脚本 `plot_phase_a_report.py/plot_phase_b_report.py` 通过 `--gt_macro_json` 使用。

重要说明：
- 本脚本按与 DiffusionDataset 相同的 window 扫描顺序（traj_ids 顺序 + t 从 0 递增）生成前 N 条 window；
- 只使用 positions（未来 F 步的绝对位置），MSD/Rog 对整体平移不敏感，能与 evaluate.py 的 GT 口径对齐。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _accumulate_msd(pos: np.ndarray, msd_sum: np.ndarray, msd_count: np.ndarray) -> None:
    # pos: (F,2) or (B,F,2)
    if pos.ndim == 2:
        pos = pos[None, ...]
    B, T, _ = pos.shape
    for lag in range(1, T):
        diff = pos[:, lag:] - pos[:, :-lag]
        sq = np.sum(diff * diff, axis=-1)
        msd_sum[lag - 1] += float(np.sum(sq))
        msd_count[lag - 1] += int(sq.size)


def _rog(pos: np.ndarray) -> float:
    # pos: (F,2)
    mean_pos = pos.mean(axis=0, keepdims=True)
    diff = pos - mean_pos
    sq = np.sum(diff * diff, axis=-1).mean()
    return float(np.sqrt(sq))


def _read_dt_fixed_seconds(processed_dir: Path) -> Optional[int]:
    stats = processed_dir / "data_stats.json"
    if not stats.exists():
        return None
    try:
        d = _load_json(stats)
        dt = d.get("time_stats", {}).get("dt_fixed")
        return int(dt) if dt is not None else None
    except Exception:
        return None


def compute_gt_macro(
    processed_dir: Path,
    split: str,
    obs_len: int,
    pred_len: int,
    batch_size: int,
    max_batches: int,
    splits_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    h5_path = processed_dir / "trajectories" / "shenzhen_trajectories.h5"
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    sd = splits_dir if splits_dir is not None else (processed_dir / "splits")
    if split != "all":
        ids_path = sd / f"{split}_ids.npy"
        if not ids_path.exists():
            raise FileNotFoundError(ids_path)
        traj_ids = np.load(ids_path).astype(np.int64)
    else:
        traj_ids = None

    max_conditions = int(batch_size) * int(max_batches) if max_batches is not None else None
    window_size = int(obs_len) + int(pred_len)
    dt_fixed = _read_dt_fixed_seconds(processed_dir)

    msd_sum = np.zeros((pred_len - 1,), dtype=np.float64)
    msd_count = np.zeros((pred_len - 1,), dtype=np.int64)
    rog_sum = 0.0
    rog_count = 0
    total = 0

    with h5py.File(h5_path, "r") as f:
        positions = f["positions"]
        ptr = f["traj_ptr"][:].astype(np.int64)
        n_traj = int(len(ptr) - 1)

        if traj_ids is None:
            traj_iter = range(n_traj)
        else:
            traj_iter = traj_ids.tolist()

        for traj_idx in traj_iter:
            i = int(traj_idx)
            start = int(ptr[i])
            end = int(ptr[i + 1])
            length = end - start
            if length < window_size:
                continue

            pos_traj = positions[start:end].astype(np.float64, copy=False)
            for t in range(0, length - window_size + 1):
                future = pos_traj[t + obs_len : t + obs_len + pred_len]
                _accumulate_msd(future, msd_sum, msd_count)
                rog_sum += _rog(future)
                rog_count += 1
                total += 1
                if max_conditions is not None and total >= int(max_conditions):
                    break
            if max_conditions is not None and total >= int(max_conditions):
                break

    msd_curve = (msd_sum / np.maximum(msd_count, 1)).astype(np.float64)

    out: Dict[str, Any] = {
        "processed_dir": str(processed_dir),
        "split": split,
        "obs_len": int(obs_len),
        "pred_len": int(pred_len),
        "num_conditions": int(total),
        "msd_curve": msd_curve.tolist(),
        "Rog": float(rog_sum / max(rog_count, 1)),
    }
    if dt_fixed is not None:
        out["dt_fixed_seconds"] = int(dt_fixed)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--splits_dir", type=str, default=None)
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    splits_dir = Path(args.splits_dir) if args.splits_dir else None
    out = compute_gt_macro(
        processed_dir=processed_dir,
        split=str(args.split),
        obs_len=int(args.obs_len),
        pred_len=int(args.pred_len),
        batch_size=int(args.batch_size),
        max_batches=int(args.max_batches),
        splits_dir=splits_dir,
    )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

