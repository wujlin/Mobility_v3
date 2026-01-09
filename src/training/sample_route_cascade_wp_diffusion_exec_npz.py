from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from src.features.skeleton_prior import build_skeleton_prior_vel_norm_k2
from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.training.route_npz_utils import RouteNorm, denormalize_vel, load_route_windows_npz, make_default_pos_bounds, normalize_pos


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_pos_norm_from_decision_ckpt(cfg: dict, *, pos_max_default: int = 1023) -> RouteNorm:
    pos_norm = cfg.get("pos_norm") if isinstance(cfg, dict) else None
    if not isinstance(pos_norm, dict):
        pos_min, pos_max = make_default_pos_bounds(pos_max=int(pos_max_default))
        pos_range = (pos_max - pos_min + 1e-6).astype(np.float32)
        return RouteNorm(
            pos_min=pos_min,
            pos_max=pos_max,
            pos_range=pos_range,
            vel_mean=np.zeros((2,), dtype=np.float32),
            vel_std=np.ones((2,), dtype=np.float32),
        )
    pos_min = np.asarray(pos_norm.get("pos_min", [0.0, 0.0]), dtype=np.float32).reshape(2)
    pos_max = np.asarray(pos_norm.get("pos_max", [float(pos_max_default), float(pos_max_default)]), dtype=np.float32).reshape(2)
    pos_range = (pos_max - pos_min + 1e-6).astype(np.float32)
    return RouteNorm(
        pos_min=pos_min,
        pos_max=pos_max,
        pos_range=pos_range,
        vel_mean=np.zeros((2,), dtype=np.float32),
        vel_std=np.ones((2,), dtype=np.float32),
    )


def _load_norm_from_exec_ckpt(cfg: dict, *, pos_max_default: int = 1023) -> RouteNorm:
    norm_cfg = cfg.get("norm") if isinstance(cfg, dict) else None
    if not isinstance(norm_cfg, dict):
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


def _od_bin_center(pos: np.ndarray, *, bin_size: float) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    b = float(bin_size)
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("--od_bin must be > 0")
    return (np.floor(pos / b) + 0.5) * b


def _waypoints_from_rel_so(
    *,
    start: np.ndarray,  # (N,2)
    dest: np.ndarray,  # (N,2)
    rel: np.ndarray,  # (N,K,2) (s,o)
    eps: float = 1e-6,
) -> np.ndarray:
    start = np.asarray(start, dtype=np.float32)
    dest = np.asarray(dest, dtype=np.float32)
    rel = np.asarray(rel, dtype=np.float32)
    v = dest - start
    L = np.linalg.norm(v, axis=1).astype(np.float32)
    L = np.maximum(L, float(eps))
    e_par = v / L[:, None]
    e_perp = np.stack([-e_par[:, 1], e_par[:, 0]], axis=1)
    s = rel[:, :, 0]
    o = rel[:, :, 1]
    wp = (
        start[:, None, :]
        + (s[:, :, None] * L[:, None, None]) * e_par[:, None, :]
        + (o[:, :, None] * L[:, None, None]) * e_perp[:, None, :]
    )
    return wp.astype(np.float32, copy=False)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gate4: sample CascadeTraj by chaining WP-diffusion (decision) + residual diffusion (execution) on a fixed case npz.")
    p.add_argument("--decision_checkpoint", type=str, required=True, help="Gate3 decision model checkpoint (route_wp_diffusion_rel_npz).")
    p.add_argument("--exec_checkpoint", type=str, required=True, help="Execution residual diffusion checkpoint (route_exec_residual_diffusion_wp_npz).")
    p.add_argument("--case_npz", type=str, required=True, help="case_XX/gt_case.npz from route_gt_baseline.py")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_samples_per_condition", type=int, default=20, help="K")
    p.add_argument("--res_scale", type=float, default=1.0, help="Scale residual velocity from execution model (0 => skeleton-only).")
    p.add_argument("--seed", type=int, default=0)
    return p


def _poly_arclength(points: np.ndarray) -> Tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points (M,2), got {points.shape}")
    if points.shape[0] < 2:
        return np.zeros((points.shape[0],), dtype=np.float32), 0.0
    seg = points[1:] - points[:-1]
    seg_len = np.linalg.norm(seg, axis=1).astype(np.float32)
    s = np.concatenate([[0.0], np.cumsum(seg_len)], axis=0).astype(np.float32)
    return s, float(s[-1])


def _resample_by_arclength(points: np.ndarray, *, num: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    s_nodes, total = _poly_arclength(points)
    if int(num) <= 1:
        return points[:1]
    if not np.isfinite(total) or total <= 1e-6:
        return np.repeat(points[:1], repeats=int(num), axis=0)
    s_query = np.linspace(0.0, total, num=int(num), dtype=np.float32)
    y = np.interp(s_query, s_nodes, points[:, 0]).astype(np.float32)
    x = np.interp(s_query, s_nodes, points[:, 1]).astype(np.float32)
    return np.stack([y, x], axis=1).astype(np.float32)


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))

    case = load_route_windows_npz(str(args.case_npz), max_n=None, seed=int(args.seed))
    start_pos = np.asarray(case["start_pos"], dtype=np.float32)
    dest_pos = np.asarray(case["dest_pos"], dtype=np.float32)
    traj_idx = np.asarray(case["traj_idx"], dtype=np.int64)
    start_t = np.asarray(case["start_t"], dtype=np.int64)
    targets = np.asarray(case["targets"], dtype=np.float32)
    n = int(start_pos.shape[0])
    f = int(targets.shape[1])

    # ---- Load decision model ----
    dec_ckpt_path = Path(args.decision_checkpoint)
    dec_ckpt = torch.load(str(dec_ckpt_path), map_location="cpu")
    if not isinstance(dec_ckpt, dict) or "model_state_dict" not in dec_ckpt:
        raise TypeError(f"Unsupported decision checkpoint format: {type(dec_ckpt)}")
    dec_cfg = dec_ckpt.get("config", {})
    dec_model_cfg = dec_cfg.get("model", {}) if isinstance(dec_cfg, dict) else {}

    k_wp = int(dec_cfg.get("K_waypoints", 0) if isinstance(dec_cfg, dict) else 0)
    if k_wp != 2:
        raise ValueError(f"Gate4 script assumes K_waypoints=2, got {k_wp}")
    od_bin = float(dec_cfg.get("od_bin", 128.0))
    o_clip = float(dec_cfg.get("o_clip", 2.0))
    rel_norm = dec_cfg.get("rel_norm", {}) if isinstance(dec_cfg, dict) else {}
    rel_mean = np.asarray(rel_norm.get("mean", [0.0, 0.0]), dtype=np.float32).reshape(2)
    rel_std = np.asarray(rel_norm.get("std", [1.0, 1.0]), dtype=np.float32).reshape(2)
    rel_std = np.maximum(rel_std, 1e-3).astype(np.float32, copy=False)

    dec_hidden = int(dec_model_cfg.get("hidden_dim", 128))
    dec_steps = int(dec_model_cfg.get("diff_steps", 50))
    dec_pred_type = str(dec_model_cfg.get("pred_type", "eps"))
    dec_pos_norm = _load_pos_norm_from_decision_ckpt(dec_cfg if isinstance(dec_cfg, dict) else {})

    start_ctr = _od_bin_center(start_pos, bin_size=float(od_bin))
    dest_ctr = _od_bin_center(dest_pos, bin_size=float(od_bin))
    start_ctr_norm = normalize_pos(start_ctr, dec_pos_norm)
    dest_ctr_norm = normalize_pos(dest_ctr, dec_pos_norm)
    obs_dec = np.concatenate([start_ctr_norm, np.zeros((n, 2), dtype=np.float32)], axis=1)[:, None, :]  # (N,1,4)
    cond_dec = np.concatenate([np.zeros((n, 2), dtype=np.float32), start_ctr_norm, dest_ctr_norm], axis=1).astype(np.float32, copy=False)  # (N,6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dec_model = DiffusionTrajectoryModel(
        obs_dim=4,
        act_dim=2,
        cond_dim=6,
        obs_len=1,
        pred_len=int(k_wp),
        hidden_dim=int(dec_hidden),
        diffusion_steps=int(dec_steps),
        prediction_type=str(dec_pred_type),
    ).to(device=device)
    dec_model.load_state_dict(dec_ckpt["model_state_dict"])
    dec_model.eval()
    obs_dec_t = torch.from_numpy(obs_dec).to(device=device, dtype=torch.float32)
    cond_dec_t = torch.from_numpy(cond_dec).to(device=device, dtype=torch.float32)

    # ---- Load execution model ----
    exec_ckpt_path = Path(args.exec_checkpoint)
    exec_ckpt = torch.load(str(exec_ckpt_path), map_location="cpu")
    if not isinstance(exec_ckpt, dict) or "model_state_dict" not in exec_ckpt:
        raise TypeError(f"Unsupported exec checkpoint format: {type(exec_ckpt)}")
    exec_cfg = exec_ckpt.get("config", {})
    exec_model_cfg = exec_cfg.get("model", {}) if isinstance(exec_cfg, dict) else {}

    exec_f = int(exec_cfg.get("F", 0) if isinstance(exec_cfg, dict) else 0)
    if exec_f > 0 and int(exec_f) != int(f):
        raise ValueError(f"F mismatch: exec checkpoint F={exec_f} vs case targets F={f}")
    exec_hidden = int(exec_model_cfg.get("hidden_dim", 128))
    exec_steps = int(exec_model_cfg.get("diff_steps", 100))
    exec_pred_type = str(exec_model_cfg.get("pred_type", "eps"))
    exec_norm = _load_norm_from_exec_ckpt(exec_cfg if isinstance(exec_cfg, dict) else {})

    start_pos_norm = normalize_pos(start_pos, exec_norm)
    dest_pos_norm = normalize_pos(dest_pos, exec_norm)
    obs_exec = np.concatenate([start_pos_norm, np.zeros((n, 2), dtype=np.float32)], axis=1)[:, None, :]  # (N,1,4)
    obs_exec_t = torch.from_numpy(obs_exec).to(device=device, dtype=torch.float32)

    exec_model = DiffusionTrajectoryModel(
        obs_dim=4,
        act_dim=2,
        cond_dim=8,
        obs_len=1,
        pred_len=int(f),
        hidden_dim=int(exec_hidden),
        diffusion_steps=int(exec_steps),
        prediction_type=str(exec_pred_type),
    ).to(device=device)
    exec_model.load_state_dict(exec_ckpt["model_state_dict"])
    exec_model.eval()

    pos_min_t = torch.tensor(exec_norm.pos_min, device=device, dtype=torch.float32)
    pos_range_t = torch.tensor(exec_norm.pos_range, device=device, dtype=torch.float32)
    vel_mean_t = torch.tensor(exec_norm.vel_mean, device=device, dtype=torch.float32)
    vel_std_t = torch.tensor(exec_norm.vel_std, device=device, dtype=torch.float32)

    k_samples = int(args.num_samples_per_condition)
    preds_k = np.zeros((n, k_samples, f, 2), dtype=np.float32)
    wp_abs_k = np.zeros((n, k_samples, 2, 2), dtype=np.float32)
    wp_rel_k = np.zeros((n, k_samples, 2, 2), dtype=np.float32)

    with torch.no_grad():
        for kk in range(k_samples):
            torch.manual_seed(int(args.seed) + 1000 + int(kk))
            rel_norm_t = dec_model.sample_trajectory(obs_dec_t, cond_dec_t, horizon=int(k_wp))  # (N,2,2)
            rel = rel_norm_t.detach().cpu().numpy().astype(np.float32, copy=False)
            rel = rel * rel_std[None, None, :] + rel_mean[None, None, :]

            rel[:, :, 0] = np.clip(rel[:, :, 0], 0.0, 1.0)
            if float(o_clip) > 0.0:
                rel[:, :, 1] = np.clip(rel[:, :, 1], -float(o_clip), float(o_clip))
            order = np.argsort(rel[:, :, 0], axis=1)
            rel = np.take_along_axis(rel, order[:, :, None], axis=1)

            wp_abs = _waypoints_from_rel_so(start=start_pos, dest=dest_pos, rel=rel)  # (N,2,2)
            wp_abs_k[:, kk] = wp_abs
            wp_rel_k[:, kk] = rel

            wp_norm = normalize_pos(wp_abs.reshape(-1, 2), exec_norm).reshape(n, 2, 2)
            cond_exec = np.concatenate(
                [
                    np.zeros((n, 2), dtype=np.float32),  # hour/day placeholders
                    wp_norm.reshape(n, -1),
                    dest_pos_norm,
                ],
                axis=1,
            ).astype(np.float32, copy=False)  # (N,8)
            cond_exec_t = torch.from_numpy(cond_exec).to(device=device, dtype=torch.float32)

            prior_vel_norm = build_skeleton_prior_vel_norm_k2(
                obs=obs_exec_t,
                cond=cond_exec_t,
                pred_len=int(f),
                num_waypoints=2,
                pos_min=pos_min_t,
                pos_range=pos_range_t,
                vel_mean=vel_mean_t,
                vel_std=vel_std_t,
            )
            torch.manual_seed(int(args.seed) + 2000 + int(kk))
            res_vel_norm = exec_model.sample_trajectory(obs_exec_t, cond_exec_t, horizon=int(f))
            vel_norm = prior_vel_norm + res_vel_norm * float(args.res_scale)
            vel = denormalize_vel(vel_norm.detach().cpu().numpy(), exec_norm)  # (N,F,2)

            disp = np.sum(vel, axis=1)
            desired = dest_pos - start_pos
            delta = desired - disp
            vel = vel + delta[:, None, :] / float(f)
            pos = start_pos[:, None, :] + np.cumsum(vel, axis=1)
            preds_k[:, kk] = pos.astype(np.float32, copy=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "samples.npz"
    out_json = out_dir / "sample_summary.json"

    meta = {
        "decision_checkpoint": str(dec_ckpt_path.resolve()),
        "exec_checkpoint": str(exec_ckpt_path.resolve()),
        "case_npz": str(Path(args.case_npz).resolve()),
        "seed": int(args.seed),
        "k_samples": int(k_samples),
        "res_scale": float(args.res_scale),
        "od_bin": float(od_bin),
        "o_clip": float(o_clip),
        "decision_model": {"hidden_dim": int(dec_hidden), "diff_steps": int(dec_steps), "pred_type": str(dec_pred_type)},
        "exec_model": {"hidden_dim": int(exec_hidden), "diff_steps": int(exec_steps), "pred_type": str(exec_pred_type)},
    }

    np.savez_compressed(
        out_npz,
        start_pos=start_pos.astype(np.float32, copy=False),
        dest_pos=dest_pos.astype(np.float32, copy=False),
        traj_idx=traj_idx.astype(np.int64, copy=False),
        start_t=start_t.astype(np.int64, copy=False),
        preds_k=preds_k.astype(np.float32, copy=False),
        wp_abs_k=wp_abs_k.astype(np.float32, copy=False),
        wp_rel_k=wp_rel_k.astype(np.float32, copy=False),
        meta=meta,
    )
    result = {"inputs": meta, "stats": {"N": int(n), "F": int(f)}, "outputs": {"samples_npz": str(out_npz.resolve())}}
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
