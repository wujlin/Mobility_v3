from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.features.skeleton_prior import build_skeleton_prior_vel_norm_k2
from src.features.waypoints import WaypointConfig, extract_oracle_waypoints_from_future
from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.training.route_npz_utils import (
    RouteNorm,
    denormalize_vel,
    load_route_windows_npz,
    make_default_pos_bounds,
    normalize_pos,
)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_norm_from_ckpt(cfg: dict, *, pos_max_default: int = 1023) -> RouteNorm:
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


def _polyline_features_to_dest_single(start_pos: np.ndarray, path: np.ndarray, dest_pos: np.ndarray) -> np.ndarray:
    start_pos = np.asarray(start_pos, dtype=np.float64).reshape(2)
    path = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    dest_pos = np.asarray(dest_pos, dtype=np.float64).reshape(2)
    poly = np.concatenate([start_pos[None, :], path], axis=0)
    a = start_pos
    b = dest_pos
    ab = b - a
    chord = float(np.linalg.norm(ab)) + 1e-12

    ap = poly - a[None, :]
    cross = ab[0] * ap[:, 1] - ab[1] * ap[:, 0]
    dist_signed = cross / chord
    dist_signed[0] = 0.0
    idx = int(np.argmax(np.abs(dist_signed)))
    dev_signed = float(dist_signed[idx])
    signed_dev_ratio = float(dev_signed / chord)

    end_seg = poly[-1]
    proj = float(np.sum((end_seg - a) * ab) / (chord * chord))

    seg = poly[1:] - poly[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    path_len = float(np.sum(seg_len))
    len_ratio = float(path_len / chord)
    return np.asarray([signed_dev_ratio, proj, len_ratio], dtype=np.float64)


def _kmeans2(x: np.ndarray, *, seed: int, iters: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n < 2:
        return np.zeros((n,), dtype=np.int64), np.zeros((2, d), dtype=np.float64)

    i0 = int(np.argmin(x[:, 0]))
    i1 = int(np.argmax(x[:, 0]))
    if i0 == i1:
        rng = np.random.default_rng(int(seed))
        i1 = int(rng.integers(0, n))
    c = np.stack([x[i0], x[i1]], axis=0)

    labels = np.zeros((n,), dtype=np.int64)
    for _ in range(int(iters)):
        d0 = np.sum((x - c[0]) ** 2, axis=1)
        d1 = np.sum((x - c[1]) ** 2, axis=1)
        new_labels = (d1 < d0).astype(np.int64)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for k in (0, 1):
            mask = labels == k
            if not np.any(mask):
                continue
            c[k] = np.mean(x[mask], axis=0)
    return labels.astype(np.int64, copy=False), c


def _cluster_two_corridors(
    *,
    start_pos: np.ndarray,  # (N,2)
    targets: np.ndarray,  # (N,F,2)
    dest_pos: np.ndarray,  # (N,2)
    seed: int,
) -> np.ndarray:
    n = int(start_pos.shape[0])
    feats = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        feats[i] = _polyline_features_to_dest_single(start_pos[i], targets[i], dest_pos[i])
    mu = np.mean(feats, axis=0)
    sig = np.std(feats, axis=0) + 1e-6
    x = (feats - mu) / sig
    labels, _ = _kmeans2(x, seed=int(seed))
    return labels.astype(np.int64, copy=False)


def _waypoints_rel_so(
    *,
    start: np.ndarray,  # (N,2)
    dest: np.ndarray,  # (N,2)
    waypoints: np.ndarray,  # (N,K,2)
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Represent waypoints in chord-normalized coordinates:
      s: fraction along chord
      o: signed perpendicular offset (normalized by chord length)
    """
    start = np.asarray(start, dtype=np.float32)
    dest = np.asarray(dest, dtype=np.float32)
    wp = np.asarray(waypoints, dtype=np.float32)
    v = dest - start  # (N,2)
    L = np.linalg.norm(v, axis=1).astype(np.float32)
    L = np.maximum(L, float(eps))
    e_par = v / L[:, None]  # (N,2)
    e_perp = np.stack([-e_par[:, 1], e_par[:, 0]], axis=1)  # (N,2) 90-deg rot

    d = wp - start[:, None, :]  # (N,K,2)
    s = np.sum(d * e_par[:, None, :], axis=2) / L[:, None]
    o = np.sum(d * e_perp[:, None, :], axis=2) / L[:, None]
    return np.stack([s, o], axis=2).astype(np.float32, copy=False)  # (N,K,2)


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
    p = argparse.ArgumentParser(
        description="Sample a waypoint-conditioned residual diffusion execution model on a fixed GT case npz (for Fig.1)."
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--case_npz", type=str, required=True, help="case_XX/gt_case.npz from route_gt_baseline.py")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_samples_per_condition", type=int, default=20, help="K")
    p.add_argument("--max_n", type=int, default=None)
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
    if f > 0 and int(f_case) != int(f):
        raise ValueError(f"F mismatch: checkpoint F={f} vs case targets F={f_case}")
    f = int(f_case)

    # Build waypoint bank from GT case trajectories.
    wp_cfg = WaypointConfig(mode="rdp_dev", num_waypoints=2)
    wp_bank = np.zeros((n, 2, 2), dtype=np.float32)
    for i in range(n):
        _, wp = extract_oracle_waypoints_from_future(start_pos=start_pos[i], future_pos=targets[i], cfg=wp_cfg)
        wp_bank[i] = wp
    wp_bank_rel = _waypoints_rel_so(start=start_pos, dest=dest_pos, waypoints=wp_bank)  # (N,2,2)

    labels = _cluster_two_corridors(start_pos=start_pos, targets=targets, dest_pos=dest_pos, seed=int(args.seed))
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    if idx0.size == 0 or idx1.size == 0:
        # Fallback: no clustering separation; sample from all.
        idx0 = np.arange(n, dtype=np.int64)
        idx1 = np.arange(n, dtype=np.int64)

    # obs is fixed per condition (start only).
    start_pos_norm = normalize_pos(start_pos, norm)  # (N,2)
    dest_pos_norm = normalize_pos(dest_pos, norm)  # (N,2)
    obs = np.concatenate([start_pos_norm, np.zeros((n, 2), dtype=np.float32)], axis=1)[:, None, :]  # (N,1,4)
    obs_t = torch.from_numpy(obs).to(dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionTrajectoryModel(
        obs_dim=4,
        act_dim=2,
        cond_dim=8,
        obs_len=1,
        pred_len=int(f),
        hidden_dim=int(hidden_dim),
        diffusion_steps=int(diff_steps),
        prediction_type=str(pred_type),
    ).to(device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    obs_t = obs_t.to(device=device)
    pos_min_t = torch.tensor(norm.pos_min, device=device, dtype=torch.float32)
    pos_range_t = torch.tensor(norm.pos_range, device=device, dtype=torch.float32)
    vel_mean_t = torch.tensor(norm.vel_mean, device=device, dtype=torch.float32)
    vel_std_t = torch.tensor(norm.vel_std, device=device, dtype=torch.float32)

    k = int(args.num_samples_per_condition)
    half = int(k) // 2
    rng = np.random.default_rng(int(args.seed))

    preds_k = np.zeros((n, k, f, 2), dtype=np.float32)
    pick_bank = np.zeros((n, k), dtype=np.int64)

    with torch.no_grad():
        for kk in range(k):
            # Ensure both corridor options appear for every condition by construction:
            # first half draws from corridor-0 bank, second half from corridor-1 bank.
            pool = idx0 if kk < half else idx1
            bank_idx = rng.choice(pool, size=n, replace=True).astype(np.int64)
            pick_bank[:, kk] = bank_idx
            rel = wp_bank_rel[bank_idx]  # (N,2,2)
            wp_abs = _waypoints_from_rel_so(start=start_pos, dest=dest_pos, rel=rel)  # (N,2,2)
            wp_norm = normalize_pos(wp_abs.reshape(-1, 2), norm).reshape(n, 2, 2)

            cond = np.concatenate(
                [
                    np.zeros((n, 2), dtype=np.float32),  # [hour, day] placeholders
                    wp_norm.reshape(n, -1),
                    dest_pos_norm,
                ],
                axis=1,
            ).astype(np.float32, copy=False)  # (N,8)
            cond_t = torch.from_numpy(cond).to(device=device, dtype=torch.float32)

            prior_vel_norm = build_skeleton_prior_vel_norm_k2(
                obs=obs_t,
                cond=cond_t,
                pred_len=int(f),
                num_waypoints=2,
                pos_min=pos_min_t,
                pos_range=pos_range_t,
                vel_mean=vel_mean_t,
                vel_std=vel_std_t,
            )  # (N,F,2) normalized

            torch.manual_seed(int(args.seed) + 1000 + int(kk))
            res_vel_norm = model.sample_trajectory(obs_t, cond_t, horizon=int(f))  # (N,F,2) normalized residual
            vel_norm = res_vel_norm + prior_vel_norm
            vel = denormalize_vel(vel_norm.detach().cpu().numpy(), norm)  # (N,F,2) grid disp

            # Endpoint correction: keep final endpoint fixed to dest (distribute residual error uniformly).
            disp = np.sum(vel, axis=1)  # (N,2)
            desired = dest_pos - start_pos  # (N,2)
            delta = desired - disp
            vel = vel + delta[:, None, :] / float(f)
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
        bank_idx=pick_bank.astype(np.int64, copy=False),
        bank_labels=labels.astype(np.int64, copy=False),
    )

    result = {
        "inputs": {"checkpoint": str(ckpt_path.resolve()), "case_npz": str(Path(args.case_npz).resolve())},
        "config": {"K": int(k), "seed": int(args.seed), "bank_sampling": {"half0": int(half), "half1": int(k - half)}},
        "stats": {"N": int(n), "F": int(f), "bank_cluster_counts": {"c0": int(idx0.size), "c1": int(idx1.size)}},
        "outputs": {"samples_npz": str(out_npz.resolve()), "summary_json": str(out_json.resolve())},
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

