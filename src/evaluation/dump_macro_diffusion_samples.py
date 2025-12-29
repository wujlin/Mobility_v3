from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:  # Optional dependency (used in waypoint_gate.py); required for nav_query_field=dist.
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from src.data.datasets_diffusion import DiffusionDataset
from src.features.skeleton_prior import CondSpec, build_skeleton_prior_vel_norm_k2
from src.models.physics.physics_condition_diffusion import PhysicsConditionDiffusion
from src.models.seq.seq_baseline import SeqBaseline


def _load_nav_count(nav_file: str) -> np.ndarray:
    data = np.load(str(nav_file), allow_pickle=True)
    if "count" not in data.files:
        raise ValueError(f"nav_file must contain 'count' for feasibility gate, got {data.files}")
    count = np.asarray(data["count"], dtype=np.float32)
    if count.ndim != 2:
        raise ValueError(f"Expected nav count (H,W), got {count.shape}")
    return count


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_checkpoint(path: str, device: torch.device) -> Tuple[dict, Dict[str, object]]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")
    state = ckpt.get("model_state_dict")
    if state is None or not isinstance(state, dict):
        raise KeyError(f"Checkpoint missing model_state_dict: {path}")
    cfg = ckpt.get("config", {})
    return state, (cfg if isinstance(cfg, dict) else {})


def _infer_hidden_dim(state_dict: dict) -> int:
    w = state_dict.get("diffusion.unet.init_conv.weight")
    if hasattr(w, "shape") and len(w.shape) == 3:
        return int(w.shape[0])
    raise ValueError("Cannot infer hidden_dim from checkpoint (missing diffusion.unet.init_conv.weight).")


def _infer_nav_emb_dim(state_dict: dict) -> int:
    w = state_dict.get("nav_encoder.net.12.weight")
    if hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[0])
    # Fallback: try any linear with in_features=1024
    for k, v in state_dict.items():
        if str(k).startswith("nav_encoder.") and str(k).endswith(".weight") and hasattr(v, "shape") and len(v.shape) == 2:
            if int(v.shape[1]) == 64 * 4 * 4:
                return int(v.shape[0])
    return 32


def _infer_has_nav_gate(state_dict: dict) -> bool:
    return any(str(k).startswith("nav_gate.") for k in state_dict.keys())


def _load_macro_model(
    macro_checkpoint: str,
    *,
    device: torch.device,
    obs_len: int,
    patch_size: int,
    diff_steps: int,
    pred_type: str,
) -> PhysicsConditionDiffusion:
    state_dict, cfg = _load_checkpoint(macro_checkpoint, device=device)
    hidden_dim = int(cfg.get("hidden_dim")) if cfg.get("hidden_dim") is not None else _infer_hidden_dim(state_dict)
    nav_emb_dim = _infer_nav_emb_dim(state_dict)

    nav_gate = str(cfg.get("nav_gate", "none"))
    if nav_gate not in ("none", "obscond"):
        nav_gate = "obscond" if _infer_has_nav_gate(state_dict) else "none"
    nav_gate_hidden = int(cfg.get("nav_gate_hidden", 32))
    if nav_gate == "obscond":
        w = state_dict.get("nav_gate.0.weight")
        if hasattr(w, "shape") and len(w.shape) == 2:
            nav_gate_hidden = int(w.shape[0])

    nav_query = str(cfg.get("nav_query", "none"))
    nav_query_field = str(cfg.get("nav_query_field", "dist"))
    nav_query_dist_sigma = float(cfg.get("nav_query_dist_sigma", 3.0))
    nav_control = str(cfg.get("nav_control", "none"))
    nav_control_scale = float(cfg.get("nav_control_scale", 1.0))
    pos_min = cfg.get("pos_min", None)
    pos_range = cfg.get("pos_range", None)
    pos_min_t = None
    pos_range_t = None
    if isinstance(pos_min, (list, tuple)) and len(pos_min) == 2 and isinstance(pos_range, (list, tuple)) and len(pos_range) == 2:
        pos_min_t = (float(pos_min[0]), float(pos_min[1]))
        pos_range_t = (float(pos_range[0]), float(pos_range[1]))

    model = PhysicsConditionDiffusion(
        obs_dim=4,
        act_dim=2,
        cond_dim=6,
        nav_patch_size=int(patch_size),
        nav_emb_dim=int(nav_emb_dim),
        nav_emb_scale=float(cfg.get("nav_emb_scale", 1.0)),
        nav_emb_dropout=float(cfg.get("nav_emb_dropout", 0.0)),
        nav_gate=str(nav_gate),
        nav_gate_hidden=int(nav_gate_hidden),
        nav_gate_dropout=float(cfg.get("nav_gate_dropout", 0.0)),
        nav_query=str(nav_query),
        nav_query_field=str(nav_query_field),
        nav_query_dist_sigma=float(nav_query_dist_sigma),
        nav_control=str(nav_control),
        nav_control_scale=float(nav_control_scale),
        pos_min=pos_min_t,
        pos_range=pos_range_t,
        obs_len=int(obs_len),
        pred_len=3,
        hidden_dim=int(hidden_dim),
        diffusion_steps=int(diff_steps),
        prediction_type=str(pred_type),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_micro_model(micro_checkpoint: str, device: torch.device) -> SeqBaseline:
    ckpt = torch.load(micro_checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        cfg = ckpt.get("config", {})
        hidden_dim = cfg.get("hidden_dim") if isinstance(cfg, dict) else None
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        hidden_dim = None
    else:
        raise TypeError(f"Unsupported micro checkpoint format: {type(ckpt)}")

    if hidden_dim is None:
        w = state_dict.get("head.weight")
        if hasattr(w, "shape") and len(w.shape) == 2:
            hidden_dim = int(w.shape[1])
    if hidden_dim is None:
        raise ValueError(f"Cannot infer micro hidden_dim from checkpoint: {micro_checkpoint}")

    cond_dim = None
    w_enc = state_dict.get("encoder.weight_ih_l0")
    if hasattr(w_enc, "shape") and len(w_enc.shape) == 2:
        cond_dim = int(w_enc.shape[1]) - 4
    if cond_dim is None or cond_dim <= 0:
        cond_dim = 8

    model = SeqBaseline(obs_dim=4, act_dim=2, cond_dim=int(cond_dim), hidden_dim=int(hidden_dim)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _integrate_positions(start_pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    return start_pos[:, None, :] + np.cumsum(vel, axis=1)


def _keys_from_ids(traj_idx: np.ndarray, start_t: np.ndarray) -> np.ndarray:
    traj_idx = traj_idx.astype(np.int64, copy=False)
    start_t = start_t.astype(np.int64, copy=False)
    return (traj_idx << np.int64(32)) + start_t


def _index_round(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    yy = np.rint(pos[..., 0]).astype(np.int64)
    xx = np.rint(pos[..., 1]).astype(np.int64)
    return yy, xx


def _sample_segments(
    a: np.ndarray,  # (S,2)
    b: np.ndarray,  # (S,2)
    *,
    step: float,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    d = b - a
    seg_len = np.linalg.norm(d, axis=-1)  # (S,)
    n = np.ceil(seg_len / max(float(step), 1e-6)).astype(np.int64) + 1
    n = np.clip(n, 2, int(max_samples))
    m = int(np.max(n)) if int(n.size) else 0
    if m <= 0:
        return np.zeros((0, 0, 2), dtype=np.float32), np.zeros((0, 0), dtype=bool)
    t = np.linspace(0.0, 1.0, num=int(m), dtype=np.float32)[None, :, None]  # (1,m,1)
    pts = a[:, None, :] + t * d[:, None, :]  # (S,m,2)
    valid = (np.arange(int(m), dtype=np.int64)[None, :] < n[:, None])  # (S,m)
    return pts.astype(np.float32, copy=False), valid


def _collision_any_mask(
    *,
    start_pos: np.ndarray,  # (B,2) grid
    z_grid: np.ndarray,  # (B,K,3,2) grid
    drivable: np.ndarray,  # (H,W) bool
    sample_step: float,
    max_samples_per_segment: int,
) -> np.ndarray:
    """
    Collision check for skeleton polyline: start -> wp1 -> wp2 -> end.

    Returns:
      collided_any: (B,K) bool
    """
    start_pos = np.asarray(start_pos, dtype=np.float32)
    z_grid = np.asarray(z_grid, dtype=np.float32)
    if start_pos.ndim != 2 or start_pos.shape[-1] != 2:
        raise ValueError(f"Expected start_pos (B,2), got {start_pos.shape}")
    if z_grid.ndim != 4 or z_grid.shape[-2:] != (3, 2):
        raise ValueError(f"Expected z_grid (B,K,3,2), got {z_grid.shape}")
    if int(start_pos.shape[0]) != int(z_grid.shape[0]):
        raise ValueError("B mismatch between start_pos and z_grid")

    H, W = int(drivable.shape[0]), int(drivable.shape[1])
    B, K = int(z_grid.shape[0]), int(z_grid.shape[1])

    start_k = np.repeat(start_pos[:, None, :], repeats=int(K), axis=1)  # (B,K,2)
    vertices = np.concatenate([start_k[:, :, None, :], z_grid], axis=2)  # (B,K,4,2)
    S = int(B * K)
    v = vertices.reshape(S, 4, 2)

    a = np.concatenate([v[:, 0], v[:, 1], v[:, 2]], axis=0)  # (S*3,2)
    b = np.concatenate([v[:, 1], v[:, 2], v[:, 3]], axis=0)

    pts, valid = _sample_segments(a, b, step=float(sample_step), max_samples=int(max_samples_per_segment))
    yy, xx = _index_round(pts)
    inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    yy_c = np.clip(yy, 0, H - 1)
    xx_c = np.clip(xx, 0, W - 1)
    drv = drivable[yy_c, xx_c]
    bad = valid & (~inb | ~drv)
    seg_bad = np.any(bad, axis=1).reshape(3, S).T  # (S,3)
    collided_any = np.any(seg_bad, axis=1).reshape(B, K)
    return collided_any


def _select_feasible_indices(
    collided_any: np.ndarray,  # (B,K_raw)
    *,
    k_target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Select K indices per sample preferring feasible (non-colliding) candidates.
    If feasible candidates are insufficient, repeats feasible ones to fill K.
    """
    collided_any = np.asarray(collided_any, dtype=bool)
    if collided_any.ndim != 2:
        raise ValueError(f"Expected collided_any (B,K_raw), got {collided_any.shape}")
    B, K_raw = int(collided_any.shape[0]), int(collided_any.shape[1])
    if int(k_target) <= 0:
        raise ValueError("k_target must be > 0")

    out = np.zeros((B, int(k_target)), dtype=np.int64)
    all_idx = np.arange(int(K_raw), dtype=np.int64)
    for i in range(int(B)):
        feasible = all_idx[~collided_any[i]]
        if int(feasible.size) >= int(k_target):
            pick = rng.choice(feasible, size=int(k_target), replace=False)
        elif int(feasible.size) > 0:
            extra = rng.choice(feasible, size=int(k_target) - int(feasible.size), replace=True)
            pick = np.concatenate([feasible, extra], axis=0)
        else:
            pick = rng.choice(all_idx, size=int(k_target), replace=(int(K_raw) < int(k_target)))
        out[i] = pick.astype(np.int64, copy=False)
    return out


def _subset_indices_from_windows_npz(
    *,
    dataset: DiffusionDataset,
    windows_npz: str,
    save_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Map (traj_idx,start_t) in windows_npz to DiffusionDataset sample indices, then select up to save_samples.

    This avoids scanning the full dataloader (which is prohibitively slow when windows are sparse).
    """
    win = np.load(str(windows_npz), allow_pickle=True)
    if "traj_idx" not in win.files or "start_t" not in win.files:
        raise ValueError(f"--windows_npz must contain traj_idx/start_t, got {win.files}")
    traj_idx = np.asarray(win["traj_idx"], dtype=np.int64).reshape(-1)
    start_t = np.asarray(win["start_t"], dtype=np.int64).reshape(-1)
    if traj_idx.shape != start_t.shape:
        raise ValueError("windows_npz: traj_idx/start_t shape mismatch")

    if dataset.traj_ids is None:
        raise ValueError("--windows_npz requires dataset.traj_ids (use split ids, not split=all).")
    if int(dataset.step) != 1:
        raise ValueError(f"--windows_npz mapping currently assumes step=1, got step={dataset.step}")

    traj_ids = np.asarray(dataset.traj_ids, dtype=np.int64).reshape(-1)
    ptr = np.asarray(dataset.storage._ptr, dtype=np.int64)
    window_size = int(dataset.window_size)
    length = (ptr[traj_ids + 1] - ptr[traj_ids]).astype(np.int64)
    win_count = np.maximum(length - window_size + 1, 0).astype(np.int64)
    offsets = np.concatenate([np.zeros((1,), dtype=np.int64), np.cumsum(win_count[:-1], dtype=np.int64)], axis=0)

    pos_map = {int(tid): int(i) for i, tid in enumerate(traj_ids.tolist())}

    idx_list: list[int] = []
    for tid, t0 in zip(traj_idx.tolist(), start_t.tolist()):
        p = pos_map.get(int(tid))
        if p is None:
            continue
        t0_i = int(t0)
        if t0_i < 0 or t0_i >= int(win_count[p]):
            continue
        idx_list.append(int(offsets[p] + t0_i))

    if not idx_list:
        raise RuntimeError("No windows from --windows_npz found in the specified split/dataset ordering.")

    idx_arr = np.asarray(idx_list, dtype=np.int64)
    idx_arr = np.unique(idx_arr)  # deduplicate repeated sampled windows

    n_need = int(save_samples)
    if n_need <= 0:
        raise ValueError("--save_samples must be > 0")
    if idx_arr.size <= n_need:
        return idx_arr

    rng = np.random.default_rng(int(seed))
    pick = rng.choice(idx_arr, size=int(n_need), replace=False)
    pick = np.sort(pick)
    return pick.astype(np.int64, copy=False)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump macro-diffusion samples (z -> skeleton -> optional DetRes micro) into samples.npz.")
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--macro_checkpoint", type=str, required=True)
    p.add_argument("--micro_checkpoint", type=str, default=None, help="optional: deterministic residual executor (DetRes)")

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--nav_file", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--splits_dir", type=str, default=None)

    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12)
    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--nav_patch_channel2", type=str, choices=["count", "speed", "zeros"], default="count")
    p.add_argument(
        "--nav_patch_override",
        type=str,
        choices=["none", "zeros", "shuffle", "dir_zero", "ch2_zero"],
        default="none",
        help="Ablation for 'map usage' diagnosis. "
             "none: use dataset nav_patch; "
             "zeros: zero all 3 channels; "
             "shuffle: shuffle nav_patch across the batch (preserve marginal distribution, break alignment); "
             "dir_zero: zero direction channels [0,1]; "
             "ch2_zero: zero channel-2 only.",
    )

    p.add_argument("--k_samples", type=int, default=20)
    p.add_argument("--save_samples", type=int, default=400)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--windows_npz", type=str, default=None, help="Optional GT windows npz; restrict sampling to these (traj_idx,start_t).")

    # Feasibility-aware macro sampling (G1 hard gate)
    p.add_argument("--feasible_gate", action="store_true", help="Enable accept/reject to keep only drivable skeletons (count>=thr) in the output K set.")
    p.add_argument("--gate_count_thr", type=float, default=1.0, help="Drivable mask threshold: count >= thr.")
    p.add_argument("--gate_sample_step", type=float, default=0.5, help="Sampling step along segments (grid units).")
    p.add_argument("--gate_max_samples_per_segment", type=int, default=256)
    p.add_argument("--gate_oversample", type=int, default=3, help="Sample K_raw = K*oversample candidates then filter to K.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using seed: {int(args.seed)}")

    traj_ids = None
    if str(args.split) != "all":
        processed_dir = Path(args.data_path).resolve().parents[1]
        splits_dir = Path(args.splits_dir) if args.splits_dir else (processed_dir / "splits")
        split_file = splits_dir / f"{args.split}_ids.npy"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        traj_ids = np.load(split_file).astype(np.int64)
        print(f"Using split={args.split}: {len(traj_ids)} trajectories ({split_file})")

    base_dataset = DiffusionDataset(
        args.data_path,
        obs_len=int(args.obs_len),
        pred_len=int(args.pred_len),
        nav_field_file=str(args.nav_file),
        nav_patch_size=int(args.patch_size),
        nav_patch_channel2=str(args.nav_patch_channel2),
        traj_ids=traj_ids,
        cond_mode="trip_od",
        waypoint_mode="rdp_dev",
        num_waypoints=2,
    )
    norm = base_dataset.normalizer

    dataset_for_loader = base_dataset
    if args.windows_npz:
        subset_idx = _subset_indices_from_windows_npz(
            dataset=base_dataset,
            windows_npz=str(args.windows_npz),
            save_samples=int(args.save_samples),
            seed=int(args.seed),
        )
        dataset_for_loader = Subset(base_dataset, subset_idx.tolist())

    loader = DataLoader(dataset_for_loader, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    # Macro model config comes from checkpoint (to avoid mismatch)
    _state, cfg = _load_checkpoint(str(args.macro_checkpoint), device=device)
    diff_steps = int(cfg.get("diff_steps", 20))
    pred_type = str(cfg.get("pred_type", "eps"))

    macro = _load_macro_model(
        str(args.macro_checkpoint),
        device=device,
        obs_len=int(args.obs_len),
        patch_size=int(args.patch_size),
        diff_steps=int(diff_steps),
        pred_type=str(pred_type),
    )
    micro = _load_micro_model(str(args.micro_checkpoint), device=device) if args.micro_checkpoint else None

    pos_min_t = torch.tensor(norm.pos_min, dtype=torch.float32, device=device)
    pos_range_t = torch.tensor(norm.pos_range, dtype=torch.float32, device=device)
    vel_mean_t = torch.tensor(norm.vel_mean, dtype=torch.float32, device=device)
    vel_std_t = torch.tensor(norm.vel_std, dtype=torch.float32, device=device)

    cond_spec = CondSpec(cond_mode="oracle_wp_end", num_waypoints=2)
    K = int(args.k_samples)
    need = int(args.save_samples)
    rng = np.random.default_rng(int(args.seed))
    rng_patch = np.random.default_rng(int(args.seed) + 12345)

    drivable = None
    if bool(args.feasible_gate):
        count = _load_nav_count(str(args.nav_file))
        drivable = np.asarray(count >= float(args.gate_count_thr), dtype=bool)
        print(
            f"[FEASIBLE_GATE] enabled: K={int(K)}, oversample={int(args.gate_oversample)}, "
            f"count_thr={float(args.gate_count_thr)}, step={float(args.gate_sample_step)}"
        )

    # Scheme-2: global nav query field (for dynamic conditioning inside diffusion)
    nav_query_global_t = None
    nav_query = str(cfg.get("nav_query", "none"))
    if nav_query != "none":
        nav_query_field = str(cfg.get("nav_query_field", "dist"))
        count_thr = float(cfg.get("count_thr", 1.0))
        count = _load_nav_count(str(args.nav_file))
        if nav_query_field == "count":
            nav_query_global_t = torch.from_numpy(np.asarray(count, dtype=np.float32))[None, None, :, :].to(device=device)
        elif nav_query_field == "dist":
            if ndimage is None:
                raise ImportError("scipy is required for nav_query_field=dist (missing scipy.ndimage).")
            road = np.asarray(count >= float(count_thr), dtype=bool)
            dist = ndimage.distance_transform_edt(~road).astype(np.float32)  # 0 on-road, >0 offroad
            nav_query_global_t = torch.from_numpy(dist)[None, None, :, :].to(device=device)
        else:
            raise ValueError(f"Unknown nav_query_field: {nav_query_field}")

    preds_k_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    start_pos_list: list[np.ndarray] = []
    traj_idx_list: list[np.ndarray] = []
    start_t_list: list[np.ndarray] = []
    origin_pos_list: list[np.ndarray] = []
    dest_pos_list: list[np.ndarray] = []
    z_k_list: list[np.ndarray] = []
    z_k_grid_list: list[np.ndarray] = []

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if args.max_batches is not None and int(bidx) >= int(args.max_batches):
                break
            if need <= 0:
                break

            meta = batch.get("meta")
            if meta is None or not isinstance(meta, dict) or ("traj_idx" not in meta) or ("start_t" not in meta):
                raise RuntimeError("Dataset must provide meta.traj_idx/start_t for alignment.")
            tid = np.asarray(meta["traj_idx"].detach().cpu().numpy(), dtype=np.int64)
            t0 = np.asarray(meta["start_t"].detach().cpu().numpy(), dtype=np.int64)

            obs = batch["obs"].to(device)
            cond_trip_od = batch["cond"].to(device)  # (B,6)
            nav_patch = batch["nav_patch"].to(device)
            action = batch["action"]  # (B,12,2) vel_norm
            trip_o = batch["trip_o"].to(device)
            trip_d = batch["trip_d"].to(device)

            B = int(obs.shape[0])
            take = min(int(need), B)
            if take <= 0:
                break
            obs = obs[:take]
            cond_trip_od = cond_trip_od[:take]
            nav_patch = nav_patch[:take]
            action = action[:take]
            trip_o = trip_o[:take]
            trip_d = trip_d[:take]
            tid = tid[:take]
            t0 = t0[:take]

            # ---- Optional nav_patch override (map usage ablation) ----
            nav_patch_override = str(args.nav_patch_override)
            if nav_patch_override != "none":
                if nav_patch_override == "zeros":
                    nav_patch = torch.zeros_like(nav_patch)
                elif nav_patch_override == "shuffle":
                    perm = rng_patch.permutation(int(take)).astype(np.int64, copy=False)
                    perm_t = torch.from_numpy(perm).to(device=nav_patch.device, dtype=torch.long)
                    nav_patch = nav_patch.index_select(0, perm_t)
                elif nav_patch_override == "dir_zero":
                    nav_patch = nav_patch.clone()
                    nav_patch[:, 0:2].zero_()
                elif nav_patch_override == "ch2_zero":
                    nav_patch = nav_patch.clone()
                    nav_patch[:, 2:3].zero_()
                else:  # pragma: no cover
                    raise ValueError(f"Unknown --nav_patch_override: {nav_patch_override}")

            # start_pos grid
            start_pos = norm.denormalize_pos(obs[:, -1, :2].detach().cpu().numpy()).astype(np.float32, copy=False)
            start_pos_list.append(start_pos)

            # GT future positions (grid)
            gt_vel = norm.denormalize_vel(action.detach().cpu().numpy()).astype(np.float32, copy=False)
            gt_pos = _integrate_positions(start_pos, gt_vel).astype(np.float32, copy=False)
            targets_list.append(gt_pos)

            # Trip OD in grid (for visualization / sanity)
            origin_pos = norm.denormalize_pos(trip_o.detach().cpu().numpy()).astype(np.float32, copy=False)
            dest_pos = norm.denormalize_pos(trip_d.detach().cpu().numpy()).astype(np.float32, copy=False)
            origin_pos_list.append(origin_pos)
            dest_pos_list.append(dest_pos)

            traj_idx_list.append(tid.astype(np.int64, copy=False))
            start_t_list.append(t0.astype(np.int64, copy=False))

            # ---- Sample macro z candidates ----
            K_raw = int(K)
            if drivable is not None:
                K_raw = int(K) * max(int(args.gate_oversample), 1)

            obs_rep_raw = obs.repeat_interleave(int(K_raw), dim=0)
            cond_rep_raw = cond_trip_od.repeat_interleave(int(K_raw), dim=0)
            nav_rep_raw = nav_patch.repeat_interleave(int(K_raw), dim=0)
            z_rep = macro.sample_trajectory(
                obs_rep_raw,
                cond_rep_raw,
                horizon=3,
                nav_patch=nav_rep_raw,
                nav_global=nav_query_global_t,
            )  # (B*K_raw,3,2)
            z_k_raw = z_rep.view(int(take), int(K_raw), 3, 2)

            # Hard clip to the normalization box (prevents rare OOB).
            z_k_raw = torch.clamp(z_k_raw, -1.0, 1.0)

            # denormalize z to grid for gates/inspection
            z_grid_raw = ((z_k_raw + 1.0) * 0.5 * pos_range_t[None, None, None, :] + pos_min_t[None, None, None, :]).detach().cpu().numpy()

            # ---- Optional feasible gate: keep only non-colliding skeletons in output K ----
            if drivable is not None:
                collided_any = _collision_any_mask(
                    start_pos=start_pos,
                    z_grid=np.asarray(z_grid_raw, dtype=np.float32),
                    drivable=drivable,
                    sample_step=float(args.gate_sample_step),
                    max_samples_per_segment=int(args.gate_max_samples_per_segment),
                )
                sel_idx = _select_feasible_indices(collided_any, k_target=int(K), rng=rng)  # (B,K)
                sel_t = torch.from_numpy(sel_idx).to(device=device, dtype=torch.long)  # (B,K)
                z_k = torch.gather(
                    z_k_raw,
                    dim=1,
                    index=sel_t[:, :, None, None].expand(int(take), int(K), 3, 2),
                )
                sel_np = sel_idx[:, :, None, None]
                sel_np = np.broadcast_to(sel_np, (int(take), int(K), 3, 2))
                z_grid = np.take_along_axis(np.asarray(z_grid_raw, dtype=np.float32), sel_np, axis=1)
            else:
                z_k = z_k_raw
                z_grid = np.asarray(z_grid_raw, dtype=np.float32)

            z_k_list.append(z_k.detach().cpu().numpy().astype(np.float32, copy=False))
            z_k_grid_list.append(np.asarray(z_grid, dtype=np.float32, copy=False))

            # ---- Build cond_wp_end for skeleton prior + DetRes executor ----
            td = cond_trip_od[:, :2][:, None, :].expand(int(take), int(K), 2)  # (B,K,2)
            z_flat = z_k.reshape(int(take), int(K), 6)
            cond_wp_end = torch.cat([td, z_flat], dim=-1).reshape(int(take * K), 8)  # (B*K,8)

            obs_rep = obs.repeat_interleave(int(K), dim=0)
            prior_vel_norm = build_skeleton_prior_vel_norm_k2(
                obs=obs_rep,
                cond=cond_wp_end,
                pred_len=int(args.pred_len),
                num_waypoints=int(cond_spec.num_waypoints),
                pos_min=pos_min_t,
                pos_range=pos_range_t,
                vel_mean=vel_mean_t,
                vel_std=vel_std_t,
            )
            vel_norm = prior_vel_norm
            if micro is not None:
                res = micro.sample_trajectory(obs_rep, cond_wp_end, int(args.pred_len))
                vel_norm = vel_norm + res

            vel = norm.denormalize_vel(vel_norm.detach().cpu().numpy()).astype(np.float32, copy=False)
            start_pos_rep = np.repeat(start_pos, repeats=int(K), axis=0)
            pos_rep = _integrate_positions(start_pos_rep, vel).astype(np.float32, copy=False)
            pos_k = pos_rep.reshape(int(take), int(K), int(args.pred_len), 2)
            preds_k_list.append(pos_k)

            need -= int(take)

    out_dir = Path(f"data/experiments/{args.exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "samples.npz"

    preds_k = np.concatenate(preds_k_list, axis=0) if preds_k_list else np.zeros((0, int(K), int(args.pred_len), 2), dtype=np.float32)
    targets = np.concatenate(targets_list, axis=0) if targets_list else np.zeros((0, int(args.pred_len), 2), dtype=np.float32)
    start_pos = np.concatenate(start_pos_list, axis=0) if start_pos_list else np.zeros((0, 2), dtype=np.float32)
    origin_pos = np.concatenate(origin_pos_list, axis=0) if origin_pos_list else np.zeros((0, 2), dtype=np.float32)
    dest_pos = np.concatenate(dest_pos_list, axis=0) if dest_pos_list else np.zeros((0, 2), dtype=np.float32)
    traj_idx = np.concatenate(traj_idx_list, axis=0) if traj_idx_list else np.zeros((0,), dtype=np.int64)
    start_t = np.concatenate(start_t_list, axis=0) if start_t_list else np.zeros((0,), dtype=np.int64)
    z_k = np.concatenate(z_k_list, axis=0) if z_k_list else np.zeros((0, int(K), 3, 2), dtype=np.float32)
    z_k_grid = np.concatenate(z_k_grid_list, axis=0) if z_k_grid_list else np.zeros((0, int(K), 3, 2), dtype=np.float32)

    meta_out = {
        "macro_checkpoint": str(args.macro_checkpoint),
        "micro_checkpoint": (str(args.micro_checkpoint) if args.micro_checkpoint else None),
        "data_path": str(args.data_path),
        "nav_file": str(args.nav_file),
        "split": str(args.split),
        "obs_len": int(args.obs_len),
        "pred_len": int(args.pred_len),
        "patch_size": int(args.patch_size),
        "nav_patch_channel2": str(args.nav_patch_channel2),
        "nav_patch_override": str(args.nav_patch_override),
        "k_samples": int(args.k_samples),
        "diff_steps": int(diff_steps),
        "pred_type": str(pred_type),
        "seed": int(args.seed),
        "windows_npz": (str(args.windows_npz) if args.windows_npz else None),
        "feasible_gate": bool(args.feasible_gate),
        "gate_count_thr": float(args.gate_count_thr),
        "gate_sample_step": float(args.gate_sample_step),
        "gate_max_samples_per_segment": int(args.gate_max_samples_per_segment),
        "gate_oversample": int(args.gate_oversample),
    }

    np.savez_compressed(
        out_npz,
        preds_k=preds_k,
        targets=targets,
        start_pos=start_pos,
        origin_pos=origin_pos,
        dest_pos=dest_pos,
        traj_idx=traj_idx,
        start_t=start_t,
        z_k=z_k,
        z_k_grid=z_k_grid,
        meta=meta_out,
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    print(f"[OK] saved {out_npz} (N={int(targets.shape[0])}, K={int(args.k_samples)})")


if __name__ == "__main__":
    main()
