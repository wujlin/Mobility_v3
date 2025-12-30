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
from src.models.macro.macro_hardsupport_ar import MacroHardSupportARNet
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

    # SeqBaseline encoder input size = obs_dim + cond_dim
    # encoder.weight_ih_l0 has shape (4*hidden, obs_dim+cond_dim)
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


def _denorm_pos(pos_norm: torch.Tensor, *, pos_min: torch.Tensor, pos_range: torch.Tensor) -> torch.Tensor:
    return (pos_norm + 1.0) * 0.5 * pos_range + pos_min


def _integrate_positions(start_pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    return start_pos[:, None, :] + np.cumsum(vel, axis=1)


def _keys_from_ids(traj_idx: np.ndarray, start_t: np.ndarray) -> np.ndarray:
    traj_idx = traj_idx.astype(np.int64, copy=False)
    start_t = start_t.astype(np.int64, copy=False)
    return (traj_idx << np.int64(32)) + start_t


def _subset_from_windows(dataset: DiffusionDataset, windows_npz: Path) -> DiffusionDataset:
    data = np.load(windows_npz, allow_pickle=True)
    if "traj_idx" not in data.files or "start_t" not in data.files:
        raise ValueError(f"windows_npz must contain traj_idx/start_t, got {data.files}")
    keys = _keys_from_ids(np.asarray(data["traj_idx"]), np.asarray(data["start_t"]))
    key_set = set(map(int, keys.tolist()))
    kept = []
    for (tid, t0) in dataset.samples:
        k = (int(tid) << 32) + int(t0)
        if k in key_set:
            kept.append((int(tid), int(t0)))
    if not kept:
        raise RuntimeError(f"No matching windows found between dataset and {windows_npz}")
    dataset.samples = kept
    return dataset


def _one_hot_prev(y: torch.Tensor, x: torch.Tensor, *, K: int) -> torch.Tensor:
    B = int(y.shape[0])
    m = torch.zeros((B, int(K), int(K)), device=y.device, dtype=torch.float32)
    idx = torch.arange(B, device=y.device, dtype=torch.int64)
    m[idx, y.clamp(0, K - 1), x.clamp(0, K - 1)] = 1.0
    return m


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump Macro Hard Support AR samples: autoregressive z_k_grid for G1 gate.")
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--micro_checkpoint", type=str, default=None, help="Optional: deterministic residual executor (DetRes). Requires --emit_traj.")
    p.add_argument("--emit_traj", action="store_true", help="If set, also save preds/preds_k by running skeleton prior (+ optional DetRes).")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--nav_file", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--splits_dir", type=str, default=None)

    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12)

    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--nav_patch_channel2", type=str, choices=["count"], default="count")
    p.add_argument("--count_thr", type=float, default=1.0)

    p.add_argument("--k_samples", type=int, default=20)
    p.add_argument("--sample_mode", type=str, choices=["multinomial", "argmax"], default="multinomial")
    p.add_argument("--save_samples", type=int, default=400)
    p.add_argument("--max_batches", type=int, default=13)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--windows_npz", type=str, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using seed: {int(args.seed)}")
    if args.micro_checkpoint and not bool(args.emit_traj):
        raise ValueError("--micro_checkpoint requires --emit_traj (otherwise it is unused).")

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
        cond_mode="oracle_wp_end",
        waypoint_mode="rdp_dev",
        num_waypoints=2,
    )
    if args.windows_npz:
        dataset = _subset_from_windows(dataset, Path(args.windows_npz))

    nav_count = dataset.nav_field.count if dataset.nav_field is not None else None
    if nav_count is None:
        raise RuntimeError("nav_field.npz must contain count for strict mask.")

    thr_norm = float(np.log1p(float(args.count_thr)) / float(getattr(dataset, "_nav_count_log1p_max", 1.0)))

    state_dict, cfg = _load_checkpoint(str(args.checkpoint), device=device)
    model = MacroHardSupportARNet(
        obs_len=int(cfg.get("obs_len", args.obs_len)),
        obs_dim=4,
        cond_dim=6,
        patch_size=int(cfg.get("patch_size", args.patch_size)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        use_coord=bool(cfg.get("use_coord", False)),
    ).to(device=device)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    micro = _load_micro_model(str(args.micro_checkpoint), device=device) if args.micro_checkpoint else None
    cond_spec = CondSpec(cond_mode="oracle_wp_end", num_waypoints=2)

    pos_min = torch.tensor(dataset.normalizer.pos_min, dtype=torch.float32, device=device)
    pos_range = torch.tensor(dataset.normalizer.pos_range, dtype=torch.float32, device=device)
    vel_mean = torch.tensor(dataset.normalizer.vel_mean, dtype=torch.float32, device=device)
    vel_std = torch.tensor(dataset.normalizer.vel_std, dtype=torch.float32, device=device)

    pin_memory = bool(torch.cuda.is_available())
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        persistent_workers=(int(args.num_workers) > 0),
    )

    K = int(args.patch_size)
    r = float(K // 2)
    out_dir = Path("data/experiments") / str(args.exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_k_list = []
    z_k_grid_list = []
    z_k_list = []
    start_pos_list = []
    targets_list = []
    origin_pos_list = []
    dest_pos_list = []
    traj_idx_list = []
    start_t_list = []

    need = int(args.save_samples)
    with torch.no_grad():
        for _, batch in enumerate(loader):
            if need <= 0:
                break

            obs = batch["obs"].to(device)
            cond_oracle = batch["cond"].to(device)
            nav_patch = batch["nav_patch"].to(device)
            action = batch["action"]
            trip_o = batch["trip_o"].to(device)
            trip_d = batch["trip_d"].to(device)

            meta = batch.get("meta")
            if isinstance(meta, dict):
                tid = meta["traj_idx"]
                t0 = meta["start_t"]
            else:
                tid = [m["traj_idx"] for m in meta]
                t0 = [m["start_t"] for m in meta]

            if isinstance(tid, torch.Tensor):
                tid = tid.detach().cpu().numpy().astype(np.int64, copy=False)
            else:
                tid = np.asarray(tid, dtype=np.int64)
            if isinstance(t0, torch.Tensor):
                t0 = t0.detach().cpu().numpy().astype(np.int64, copy=False)
            else:
                t0 = np.asarray(t0, dtype=np.int64)

            B = int(obs.shape[0])
            take = min(int(need), B)
            if take <= 0:
                break

            obs = obs[:take]
            cond_oracle = cond_oracle[:take]
            nav_patch = nav_patch[:take]
            action = action[:take]
            trip_o = trip_o[:take]
            trip_d = trip_d[:take]
            tid = tid[:take]
            t0 = t0[:take]

            cond_trip_od = torch.cat([cond_oracle[:, :2], trip_o, trip_d], dim=-1)  # (B,6)
            strict = (nav_patch[:, 2] >= float(thr_norm))  # (B,K,K)
            empty = (strict.view(take, -1).sum(dim=1) == 0)
            if bool(torch.any(empty)):
                strict[empty] = True

            start_pos_grid = _denorm_pos(obs[:, -1, :2], pos_min=pos_min, pos_range=pos_range)  # (B,2)
            center = torch.floor(start_pos_grid)  # (B,2)

            # -------- Stage 0: wp1 (B,Ks) --------
            prev0 = torch.zeros((take, 2, K, K), device=device, dtype=torch.float32)
            logits0 = model(obs=obs, cond=cond_trip_od, nav_patch=nav_patch, prev_maps=prev0)
            logits0 = logits0.masked_fill(~strict, -1e9).view(take, K * K)
            if str(args.sample_mode) == "argmax":
                idx0 = torch.argmax(logits0, dim=-1, keepdim=True)
                if int(args.k_samples) > 1:
                    idx0 = idx0.expand(-1, int(args.k_samples)).contiguous()
            else:
                probs0 = torch.softmax(logits0, dim=-1)
                idx0 = torch.multinomial(probs0, num_samples=int(args.k_samples), replacement=True)
            y0 = (idx0 // K).to(torch.int64)
            x0 = (idx0 % K).to(torch.int64)

            # Expand for stage1: (B*Ks)
            Ks = int(args.k_samples)
            obs_k = obs[:, None, :, :].expand(take, Ks, -1, -1).reshape(take * Ks, obs.shape[1], obs.shape[2])
            cond_k = cond_trip_od[:, None, :].expand(take, Ks, -1).reshape(take * Ks, cond_trip_od.shape[1])
            nav_k = nav_patch[:, None, :, :, :].expand(take, Ks, -1, -1, -1).reshape(take * Ks, nav_patch.shape[1], K, K)
            strict_k = strict[:, None, :, :].expand(take, Ks, -1, -1).reshape(take * Ks, K, K)

            y0f = y0.reshape(take * Ks)
            x0f = x0.reshape(take * Ks)
            wp1_map = _one_hot_prev(y0f, x0f, K=K)
            prev1 = torch.stack([wp1_map, torch.zeros_like(wp1_map)], dim=1)

            # -------- Stage 1: wp2 (B*Ks,1) --------
            logits1 = model(obs=obs_k, cond=cond_k, nav_patch=nav_k, prev_maps=prev1)
            logits1 = logits1.masked_fill(~strict_k, -1e9).view(take * Ks, K * K)
            if str(args.sample_mode) == "argmax":
                idx1 = torch.argmax(logits1, dim=-1, keepdim=True)
            else:
                probs1 = torch.softmax(logits1, dim=-1)
                idx1 = torch.multinomial(probs1, num_samples=1, replacement=True)
            y1 = (idx1 // K).to(torch.int64).reshape(take, Ks)
            x1 = (idx1 % K).to(torch.int64).reshape(take, Ks)

            y1f = y1.reshape(take * Ks)
            x1f = x1.reshape(take * Ks)
            wp2_map = _one_hot_prev(y1f, x1f, K=K)
            prev2 = torch.stack([wp1_map, wp2_map], dim=1)

            # -------- Stage 2: end (B*Ks,1) --------
            logits2 = model(obs=obs_k, cond=cond_k, nav_patch=nav_k, prev_maps=prev2)
            logits2 = logits2.masked_fill(~strict_k, -1e9).view(take * Ks, K * K)
            if str(args.sample_mode) == "argmax":
                idx2 = torch.argmax(logits2, dim=-1, keepdim=True)
            else:
                probs2 = torch.softmax(logits2, dim=-1)
                idx2 = torch.multinomial(probs2, num_samples=1, replacement=True)
            y2 = (idx2 // K).to(torch.int64).reshape(take, Ks)
            x2 = (idx2 % K).to(torch.int64).reshape(take, Ks)

            # patch coords -> global grid
            z_patch = torch.stack(
                [
                    torch.stack([y0.to(torch.float32), x0.to(torch.float32)], dim=-1),
                    torch.stack([y1.to(torch.float32), x1.to(torch.float32)], dim=-1),
                    torch.stack([y2.to(torch.float32), x2.to(torch.float32)], dim=-1),
                ],
                dim=2,
            )  # (B,Ks,3,2)
            z_grid = center[:, None, None, :] + (z_patch - r)  # (B,Ks,3,2)
            z_k_grid_list.append(z_grid.detach().cpu().numpy().astype(np.float32, copy=False))

            # normalized z for downstream skeleton/micro: map grid coords -> [-1,1]
            z_norm = (z_grid - pos_min[None, None, None, :]) / pos_range[None, None, None, :]
            z_norm = z_norm * 2.0 - 1.0
            z_norm = torch.clamp(z_norm, -1.0, 1.0)
            z_k_list.append(z_norm.detach().cpu().numpy().astype(np.float32, copy=False))

            if bool(args.emit_traj):
                # Build cond_wp_end: [hour,day] + [wp1/wp2/end] (normalized pos)
                td = cond_trip_od[:, :2][:, None, :].expand(int(take), int(Ks), 2)  # (B,K,2)
                z_flat = z_norm.reshape(int(take), int(Ks), 6)  # (B,K,6)
                cond_wp_end = torch.cat([td, z_flat], dim=-1).reshape(int(take * Ks), 8)  # (B*K,8)

                obs_rep = obs.repeat_interleave(int(Ks), dim=0)
                prior_vel_norm = build_skeleton_prior_vel_norm_k2(
                    obs=obs_rep,
                    cond=cond_wp_end,
                    pred_len=int(args.pred_len),
                    num_waypoints=int(cond_spec.num_waypoints),
                    pos_min=pos_min,
                    pos_range=pos_range,
                    vel_mean=vel_mean,
                    vel_std=vel_std,
                )
                vel_norm = prior_vel_norm
                if micro is not None:
                    res = micro.sample_trajectory(obs_rep, cond_wp_end, int(args.pred_len))
                    vel_norm = vel_norm + res

                vel = (vel_norm * vel_std[None, None, :] + vel_mean[None, None, :]).detach().cpu().numpy().astype(np.float32, copy=False)
                start_pos_rep = start_pos_grid.detach().cpu().numpy().astype(np.float32, copy=False)
                start_pos_rep = np.repeat(start_pos_rep, repeats=int(Ks), axis=0)
                pos_rep = _integrate_positions(start_pos_rep, vel).astype(np.float32, copy=False)
                preds_k = pos_rep.reshape(int(take), int(Ks), int(args.pred_len), 2)
                preds_k_list.append(preds_k)

            start_pos = start_pos_grid.detach().cpu().numpy().astype(np.float32, copy=False)
            start_pos_list.append(start_pos)
            gt_vel = (action.to(device) * vel_std[None, None, :] + vel_mean[None, None, :]).detach().cpu().numpy().astype(np.float32, copy=False)
            targets_list.append(_integrate_positions(start_pos, gt_vel).astype(np.float32, copy=False))

            origin_pos = _denorm_pos(trip_o, pos_min=pos_min, pos_range=pos_range).detach().cpu().numpy().astype(np.float32, copy=False)
            dest_pos = _denorm_pos(trip_d, pos_min=pos_min, pos_range=pos_range).detach().cpu().numpy().astype(np.float32, copy=False)
            origin_pos_list.append(origin_pos)
            dest_pos_list.append(dest_pos)

            traj_idx_list.append(tid.astype(np.int64, copy=False))
            start_t_list.append(t0.astype(np.int64, copy=False))

            need -= int(take)

    z_k_grid = np.concatenate(z_k_grid_list, axis=0) if z_k_grid_list else np.zeros((0, int(args.k_samples), 3, 2), dtype=np.float32)
    z_k = np.concatenate(z_k_list, axis=0) if z_k_list else np.zeros((0, int(args.k_samples), 3, 2), dtype=np.float32)
    preds_k = None
    preds = None
    if bool(args.emit_traj):
        preds_k = np.concatenate(preds_k_list, axis=0) if preds_k_list else np.zeros((0, int(args.k_samples), int(args.pred_len), 2), dtype=np.float32)
        preds = preds_k[:, 0] if int(preds_k.shape[0]) > 0 and int(preds_k.shape[1]) > 0 else np.zeros((0, int(args.pred_len), 2), dtype=np.float32)
    start_pos = np.concatenate(start_pos_list, axis=0) if start_pos_list else np.zeros((0, 2), dtype=np.float32)
    targets = np.concatenate(targets_list, axis=0) if targets_list else np.zeros((0, int(args.pred_len), 2), dtype=np.float32)
    origin_pos = np.concatenate(origin_pos_list, axis=0) if origin_pos_list else np.zeros((0, 2), dtype=np.float32)
    dest_pos = np.concatenate(dest_pos_list, axis=0) if dest_pos_list else np.zeros((0, 2), dtype=np.float32)
    traj_idx = np.concatenate(traj_idx_list, axis=0) if traj_idx_list else np.zeros((0,), dtype=np.int64)
    start_t = np.concatenate(start_t_list, axis=0) if start_t_list else np.zeros((0,), dtype=np.int64)

    meta = {
        "checkpoint": str(args.checkpoint),
        "micro_checkpoint": (str(args.micro_checkpoint) if args.micro_checkpoint else None),
        "emit_traj": bool(args.emit_traj),
        "data_path": str(args.data_path),
        "nav_file": str(args.nav_file),
        "split": str(args.split),
        "obs_len": int(args.obs_len),
        "pred_len": int(args.pred_len),
        "patch_size": int(args.patch_size),
        "count_thr": float(args.count_thr),
        "thr_norm": float(thr_norm),
        "k_samples": int(args.k_samples),
        "sample_mode": str(args.sample_mode),
        "seed": int(args.seed),
        "windows_npz": (str(args.windows_npz) if args.windows_npz else None),
    }

    out_npz = out_dir / "samples.npz"
    payload = {
        "z_k_grid": z_k_grid,
        "z_k": z_k,
        "start_pos": start_pos,
        "targets": targets,
        "origin_pos": origin_pos,
        "dest_pos": dest_pos,
        "traj_idx": traj_idx,
        "start_t": start_t,
        "meta": meta,
    }
    if bool(args.emit_traj):
        payload["preds"] = preds
        payload["preds_k"] = preds_k
    np.savez_compressed(out_npz, **payload)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] saved {out_npz} (N={int(start_pos.shape[0])}, K={int(args.k_samples)})")


if __name__ == "__main__":
    main()
