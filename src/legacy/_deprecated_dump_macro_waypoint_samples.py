from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets_seq import SeqDataset
from src.features.skeleton_prior import CondSpec, build_skeleton_prior_vel_norm_k2
from src.legacy._deprecated_waypoint_mdn import WaypointMDN, WaypointMDNConfig
from src.models.seq.seq_baseline import SeqBaseline


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_checkpoint(path: str, device: torch.device) -> Tuple[dict, dict]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        cfg = ckpt.get("config", {})
        return ckpt["model_state_dict"], (cfg if isinstance(cfg, dict) else {})
    if isinstance(ckpt, dict):
        return ckpt, {}
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")


def _infer_hidden_dim_from_state_dict(state_dict: dict) -> int:
    w = state_dict.get("head_mu.weight")
    if hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[1])
    raise ValueError("Cannot infer hidden_dim from macro checkpoint (missing head_mu.weight).")


def _load_macro_model(path: str, device: torch.device) -> WaypointMDN:
    state_dict, cfg = _load_checkpoint(path, device=device)
    hidden_dim = int(cfg.get("hidden_dim")) if isinstance(cfg, dict) and cfg.get("hidden_dim") is not None else None
    n_components = int(cfg.get("n_components")) if isinstance(cfg, dict) and cfg.get("n_components") is not None else None

    if hidden_dim is None:
        hidden_dim = _infer_hidden_dim_from_state_dict(state_dict)
    if n_components is None:
        w = state_dict.get("head_logits.weight")
        n_components = int(w.shape[0]) if hasattr(w, "shape") and len(w.shape) == 2 else 8

    model = WaypointMDN(obs_dim=4, cond_dim=6, hidden_dim=int(hidden_dim), cfg=WaypointMDNConfig(z_dim=6, n_components=int(n_components)))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_micro_model(path: str, device: torch.device) -> SeqBaseline:
    ckpt = torch.load(path, map_location=device)
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
        raise ValueError(f"Cannot infer micro hidden_dim from checkpoint: {path}")

    cond_dim = None
    w_enc = state_dict.get("encoder.weight_ih_l0")
    if hasattr(w_enc, "shape") and len(w_enc.shape) == 2:
        cond_dim = int(w_enc.shape[1]) - 4
    if cond_dim is None or cond_dim <= 0:
        cond_dim = 8

    model = SeqBaseline(obs_dim=4, act_dim=2, cond_dim=int(cond_dim), hidden_dim=int(hidden_dim))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _integrate_positions(start_pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    return start_pos[:, None, :] + np.cumsum(vel, axis=1)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump macro waypoint MDN samples as skeleton-only (or +micro) trajectories.")
    p.add_argument("--exp_name", type=str, required=True, help="output dir under data/experiments/<exp_name>")
    p.add_argument("--macro_checkpoint", type=str, required=True)
    p.add_argument("--micro_checkpoint", type=str, default=None, help="optional: apply deterministic residual executor on top of skeleton")

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--splits_dir", type=str, default=None)
    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)

    p.add_argument("--k_samples", type=int, default=20, help="number of macro samples per condition")
    p.add_argument("--save_samples", type=int, default=400)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--waypoint_mode", type=str, choices=["rdp_dev"], default="rdp_dev")
    p.add_argument("--num_waypoints", type=int, default=2)
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

    cond_spec = CondSpec(cond_mode="oracle_wp_end", num_waypoints=int(args.num_waypoints))
    if int(cond_spec.num_waypoints) != 2:
        raise ValueError("--num_waypoints 当前仅支持 2（KISS）")

    dataset = SeqDataset(
        args.data_path,
        obs_len=int(args.obs_len),
        pred_len=int(args.pred_len),
        traj_ids=traj_ids,
        cond_mode=str(cond_spec.cond_mode),
        waypoint_mode=str(args.waypoint_mode),
        num_waypoints=int(cond_spec.num_waypoints),
    )
    norm = dataset.normalizer

    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    macro = _load_macro_model(str(args.macro_checkpoint), device=device)
    micro = _load_micro_model(str(args.micro_checkpoint), device=device) if args.micro_checkpoint else None

    pos_min_t = torch.tensor(norm.pos_min, dtype=torch.float32, device=device)
    pos_range_t = torch.tensor(norm.pos_range, dtype=torch.float32, device=device)
    vel_mean_t = torch.tensor(norm.vel_mean, dtype=torch.float32, device=device)
    vel_std_t = torch.tensor(norm.vel_std, dtype=torch.float32, device=device)

    preds_k_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    start_pos_list: list[np.ndarray] = []
    traj_idx_list: list[np.ndarray] = []
    start_t_list: list[np.ndarray] = []
    origin_pos_list: list[np.ndarray] = []
    dest_pos_list: list[np.ndarray] = []

    need = int(args.save_samples)
    K = int(args.k_samples)
    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if args.max_batches is not None and int(bidx) >= int(args.max_batches):
                break
            if need <= 0:
                break

            obs = batch["obs"].to(device)
            cond_oracle = batch["cond"].to(device)
            trip_o = batch["trip_o"].to(device)
            trip_d = batch["trip_d"].to(device)

            B = int(obs.shape[0])
            take = min(int(need), B)
            if take <= 0:
                break

            obs = obs[:take]
            cond_oracle = cond_oracle[:take]
            trip_o = trip_o[:take]
            trip_d = trip_d[:take]
            if "target_pos" in batch:
                target_pos = batch["target_pos"][:take]
            else:
                raise KeyError("SeqDataset must provide target_pos")

            cond_trip_od = torch.cat([cond_oracle[:, :2], trip_o, trip_d], dim=-1)
            z_k = macro.sample(obs, cond_trip_od, k=int(K))
            td = cond_oracle[:, :2][:, None, :].expand(int(take), int(K), 2)
            cond_k = torch.cat([td, z_k], dim=-1)

            obs_flat = obs.repeat_interleave(int(K), dim=0)
            cond_flat = cond_k.reshape(int(take * K), int(cond_k.shape[-1]))

            prior_vel_norm = build_skeleton_prior_vel_norm_k2(
                obs=obs_flat,
                cond=cond_flat,
                pred_len=int(args.pred_len),
                num_waypoints=int(cond_spec.num_waypoints),
                pos_min=pos_min_t,
                pos_range=pos_range_t,
                vel_mean=vel_mean_t,
                vel_std=vel_std_t,
            )

            vel_norm = prior_vel_norm
            if micro is not None:
                res = micro.sample_trajectory(obs_flat, cond_flat, int(args.pred_len))
                vel_norm = vel_norm + res

            vel = norm.denormalize_vel(vel_norm.detach().cpu().numpy()).astype(np.float32, copy=False)
            start_pos = norm.denormalize_pos(obs[:, -1, :2].detach().cpu().numpy()).astype(np.float32, copy=False)
            start_pos_flat = np.repeat(start_pos, repeats=int(K), axis=0)
            pos_flat = _integrate_positions(start_pos_flat, vel).astype(np.float32, copy=False)
            pos_k = pos_flat.reshape(int(take), int(K), int(args.pred_len), 2)

            targets = norm.denormalize_pos(target_pos.detach().cpu().numpy()).astype(np.float32, copy=False)
            preds_k_list.append(pos_k)
            targets_list.append(targets)
            start_pos_list.append(start_pos)

            meta = batch.get("meta")
            if meta is not None and isinstance(meta, dict) and ("traj_idx" in meta) and ("start_t" in meta):
                traj_idx_list.append(np.asarray(meta["traj_idx"][:take].detach().cpu().numpy(), dtype=np.int64))
                start_t_list.append(np.asarray(meta["start_t"][:take].detach().cpu().numpy(), dtype=np.int64))

            origin_pos = norm.denormalize_pos(batch["trip_o"][:take].detach().cpu().numpy()).astype(np.float32, copy=False)
            dest_pos = norm.denormalize_pos(batch["trip_d"][:take].detach().cpu().numpy()).astype(np.float32, copy=False)
            origin_pos_list.append(origin_pos)
            dest_pos_list.append(dest_pos)

            need -= int(take)

    out_dir = Path(f"data/experiments/{args.exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "samples.npz"

    preds_k = np.concatenate(preds_k_list, axis=0) if preds_k_list else np.zeros((0, int(K), int(args.pred_len), 2), dtype=np.float32)
    targets = np.concatenate(targets_list, axis=0) if targets_list else np.zeros((0, int(args.pred_len), 2), dtype=np.float32)
    start_pos = np.concatenate(start_pos_list, axis=0) if start_pos_list else np.zeros((0, 2), dtype=np.float32)

    npz = {
        "preds_k": preds_k,
        "targets": targets,
        "start_pos": start_pos,
        "origin_pos": (np.concatenate(origin_pos_list, axis=0) if origin_pos_list else np.zeros((0, 2), dtype=np.float32)),
        "dest_pos": (np.concatenate(dest_pos_list, axis=0) if dest_pos_list else np.zeros((0, 2), dtype=np.float32)),
    }
    if traj_idx_list and start_t_list:
        npz["traj_idx"] = np.concatenate(traj_idx_list, axis=0).astype(np.int64, copy=False)
        npz["start_t"] = np.concatenate(start_t_list, axis=0).astype(np.int64, copy=False)

    meta_out = {
        "macro_checkpoint": str(args.macro_checkpoint),
        "micro_checkpoint": (str(args.micro_checkpoint) if args.micro_checkpoint else None),
        "data_path": str(args.data_path),
        "split": str(args.split),
        "obs_len": int(args.obs_len),
        "pred_len": int(args.pred_len),
        "k_samples": int(args.k_samples),
        "waypoint_mode": str(args.waypoint_mode),
        "num_waypoints": int(args.num_waypoints),
        "seed": int(args.seed),
    }
    np.savez_compressed(out_npz, **npz, meta=meta_out)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    print(f"[OK] saved {out_npz} (N={int(targets.shape[0])}, K={int(args.k_samples)})")


if __name__ == "__main__":
    main()
