from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.features.temporal import encode_route_temporal_2d
from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.training.route_npz_utils import (
    RouteNorm,
    denormalize_vel,
    load_route_windows_npz,
    make_default_pos_bounds,
    normalize_pos,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):  # type: ignore[no-redef]
        return x


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_norm_from_ckpt(cfg: dict, *, pos_max_default: int = 1023) -> RouteNorm:
    norm_cfg = cfg.get("norm") if isinstance(cfg, dict) else None
    if not isinstance(norm_cfg, dict):
        # Backward-safe fallback: assume 1024 grid and zero-mean unit-std velocities.
        pos_min, pos_max = make_default_pos_bounds(pos_max=int(pos_max_default))
        pos_range = (pos_max - pos_min + 1e-6).astype(np.float32)
        return RouteNorm(
            pos_min=pos_min,
            pos_max=pos_max,
            pos_range=pos_range,
            vel_mean=np.zeros((2,), dtype=np.float32),
            vel_std=np.ones((2,), dtype=np.float32),
        )

    pos_min = np.asarray(norm_cfg.get("pos_min", [0.0, 0.0]), dtype=np.float32).reshape(2)
    pos_max = np.asarray(norm_cfg.get("pos_max", [float(pos_max_default), float(pos_max_default)]), dtype=np.float32).reshape(2)
    pos_range = (pos_max - pos_min + 1e-6).astype(np.float32)
    vel_mean = np.asarray(norm_cfg.get("vel_mean", [0.0, 0.0]), dtype=np.float32).reshape(2)
    vel_std = np.asarray(norm_cfg.get("vel_std", [1.0, 1.0]), dtype=np.float32).reshape(2)
    vel_std = np.maximum(vel_std, 1e-3).astype(np.float32, copy=False)
    return RouteNorm(pos_min=pos_min, pos_max=pos_max, pos_range=pos_range, vel_mean=vel_mean, vel_std=vel_std)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample an end-to-end diffusion baseline on a fixed GT case npz (for Fig.1 collapse).")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--case_npz", type=str, required=True, help="case_XX/gt_case.npz from route_gt_baseline.py")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_samples_per_condition", type=int, default=20, help="K")
    p.add_argument("--cfg_scale", type=float, default=0.0, help="CFG inference scale (0 disables).")
    p.add_argument("--cfg_uncond_dest_mode", type=str, choices=["origin", "zeros"], default="origin")
    p.add_argument("--max_n", type=int, default=None, help="Optional: limit number of windows in the case")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    f = int(cfg.get("F", 0) if isinstance(cfg, dict) else 0)
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    diff_steps = int(model_cfg.get("diff_steps", 100))
    pred_type = str(model_cfg.get("pred_type", "eps"))

    norm = _load_norm_from_ckpt(cfg if isinstance(cfg, dict) else {})

    case = load_route_windows_npz(str(args.case_npz), max_n=(int(args.max_n) if args.max_n is not None else None), seed=int(args.seed))
    start_pos = np.asarray(case["start_pos"], dtype=np.float32)
    dest_pos = np.asarray(case["dest_pos"], dtype=np.float32)
    traj_idx = np.asarray(case["traj_idx"], dtype=np.int64)
    start_t = np.asarray(case["start_t"], dtype=np.int64)
    targets = np.asarray(case["targets"], dtype=np.float32)

    n = int(start_pos.shape[0])
    f_case = int(targets.shape[1])
    if f > 0 and f_case != f:
        raise ValueError(f"F mismatch: checkpoint F={f} vs case targets F={f_case}")
    f = int(f_case)

    # Build obs/cond (obs_len=1).
    start_pos_norm = normalize_pos(start_pos, norm)  # (N,2)
    dest_pos_norm = normalize_pos(dest_pos, norm)  # (N,2)
    obs = np.concatenate([start_pos_norm, np.zeros((n, 2), dtype=np.float32)], axis=1)[:, None, :]  # (N,1,4)
    temporal_meta = cfg.get("temporal", {}) if isinstance(cfg, dict) else {}
    temporal_mode = "zeros"
    temporal_tz = -5.0
    if isinstance(temporal_meta, dict):
        temporal_mode = str(temporal_meta.get("effective") or temporal_meta.get("mode") or "zeros")
        temporal_tz = float(temporal_meta.get("tz_offset_hours", -5.0))
    temporal, _temporal_eff = encode_route_temporal_2d(start_t, tz_offset_hours=float(temporal_tz), mode=str(temporal_mode))
    cond = np.concatenate([temporal.astype(np.float32, copy=False), start_pos_norm, dest_pos_norm], axis=1).astype(np.float32, copy=False)  # (N,6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionTrajectoryModel(
        obs_dim=4,
        act_dim=2,
        cond_dim=6,
        obs_len=1,
        pred_len=int(f),
        hidden_dim=int(hidden_dim),
        diffusion_steps=int(diff_steps),
        prediction_type=str(pred_type),
    ).to(device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
    cond_t = torch.from_numpy(cond).to(device=device, dtype=torch.float32)

    k = int(args.num_samples_per_condition)
    preds_k = np.zeros((n, k, f, 2), dtype=np.float32)
    cfg_scale = float(args.cfg_scale)
    cond_uncond_t = None
    if cfg_scale != 0.0:
        cond_uncond = cond.copy()
        mode = str(args.cfg_uncond_dest_mode)
        if mode == "origin":
            cond_uncond[:, 4:6] = cond_uncond[:, 2:4]
        elif mode == "zeros":
            cond_uncond[:, 4:6] = 0.0
        else:  # pragma: no cover
            raise ValueError(f"Unknown cfg_uncond_dest_mode: {mode}")
        cond_uncond_t = torch.from_numpy(cond_uncond).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for kk in tqdm(range(k), desc="sample", dynamic_ncols=True):
            # Make per-k randomness stable but distinct.
            torch.manual_seed(int(args.seed) + 1000 + int(kk))
            vel_norm = model.sample_trajectory(
                obs_t,
                cond_t,
                horizon=int(f),
                cond_uncond=cond_uncond_t,
                cfg_scale=cfg_scale,
            )  # (N,F,2) normalized
            vel = denormalize_vel(vel_norm.detach().cpu().numpy(), norm)  # (N,F,2) grid disp
            # Integrate: pos[t] = start + sum_{i<=t} vel[i]
            pos = start_pos[:, None, :] + np.cumsum(vel, axis=1)
            preds_k[:, kk, :, :] = pos.astype(np.float32, copy=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "samples.npz"
    out_json = out_dir / "sample_summary.json"

    np.savez_compressed(
        out_npz,
        start_pos=start_pos.astype(np.float32, copy=False),
        dest_pos=dest_pos.astype(np.float32, copy=False),
        traj_idx=traj_idx.astype(np.int64, copy=False),
        start_t=start_t.astype(np.int64, copy=False),
        preds_k=preds_k.astype(np.float32, copy=False),
    )

    result = {
        "inputs": {"checkpoint": str(ckpt_path.resolve()), "case_npz": str(Path(args.case_npz).resolve())},
        "config": {"K": int(k), "seed": int(args.seed), "cfg_scale": float(cfg_scale), "cfg_uncond_dest_mode": str(args.cfg_uncond_dest_mode)},
        "stats": {"N": int(n), "F": int(f)},
        "outputs": {"samples_npz": str(out_npz.resolve()), "summary_json": str(out_json.resolve())},
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
