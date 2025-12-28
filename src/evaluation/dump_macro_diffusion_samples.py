from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets_diffusion import DiffusionDataset
from src.features.skeleton_prior import CondSpec, build_skeleton_prior_vel_norm_k2
from src.models.physics.physics_condition_diffusion import PhysicsConditionDiffusion
from src.models.seq.seq_baseline import SeqBaseline


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

    p.add_argument("--k_samples", type=int, default=20)
    p.add_argument("--save_samples", type=int, default=400)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--windows_npz", type=str, default=None, help="Optional GT windows npz; restrict sampling to these (traj_idx,start_t).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using seed: {int(args.seed)}")

    # Optional restriction to a predefined window set (e.g., test_detour_hard.npz)
    desired_keys = None
    if args.windows_npz:
        win = np.load(str(args.windows_npz))
        if "traj_idx" not in win.files or "start_t" not in win.files:
            raise ValueError(f"--windows_npz must contain traj_idx/start_t, got {win.files}")
        desired_keys = _keys_from_ids(np.asarray(win["traj_idx"]), np.asarray(win["start_t"]))

    traj_ids = None
    if str(args.split) != "all":
        processed_dir = Path(args.data_path).resolve().parents[1]
        splits_dir = Path(args.splits_dir) if args.splits_dir else (processed_dir / "splits")
        split_file = splits_dir / f"{args.split}_ids.npy"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        traj_ids = np.load(split_file).astype(np.int64)
        print(f"Using split={args.split}: {len(traj_ids)} trajectories ({split_file})")

    dataset = DiffusionDataset(
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
    norm = dataset.normalizer

    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

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

            if desired_keys is not None:
                keys = _keys_from_ids(tid, t0)
                mask = np.isin(keys, desired_keys)
                if not bool(np.any(mask)):
                    continue
                idx = np.nonzero(mask)[0]
                batch = {k: (v[idx] if isinstance(v, torch.Tensor) else v) for k, v in batch.items() if k != "meta"}
                # meta tensors need slicing too
                meta = {"traj_idx": torch.from_numpy(tid[idx]), "start_t": torch.from_numpy(t0[idx])}
                tid = tid[idx]
                t0 = t0[idx]

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

            # ---- Sample K macro z in one shot by replication ----
            obs_rep = obs.repeat_interleave(int(K), dim=0)
            cond_rep = cond_trip_od.repeat_interleave(int(K), dim=0)
            nav_rep = nav_patch.repeat_interleave(int(K), dim=0)
            z_rep = macro.sample_trajectory(obs_rep, cond_rep, horizon=3, nav_patch=nav_rep)  # (B*K,3,2) normalized pos
            z_k = z_rep.view(int(take), int(K), 3, 2)
            z_k_list.append(z_k.detach().cpu().numpy().astype(np.float32, copy=False))

            # denormalize z to grid for gates/inspection
            z_grid = ((z_k + 1.0) * 0.5 * pos_range_t[None, None, None, :] + pos_min_t[None, None, None, :]).detach().cpu().numpy()
            z_k_grid_list.append(np.asarray(z_grid, dtype=np.float32))

            # ---- Build cond_wp_end for skeleton prior + DetRes executor ----
            td = cond_trip_od[:, :2][:, None, :].expand(int(take), int(K), 2)  # (B,K,2)
            z_flat = z_k.reshape(int(take), int(K), 6)
            cond_wp_end = torch.cat([td, z_flat], dim=-1).reshape(int(take * K), 8)  # (B*K,8)

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
        "k_samples": int(args.k_samples),
        "diff_steps": int(diff_steps),
        "pred_type": str(pred_type),
        "seed": int(args.seed),
        "windows_npz": (str(args.windows_npz) if args.windows_npz else None),
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

