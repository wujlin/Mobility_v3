from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets_diffusion import DiffusionDataset
from src.data.datasets_seq import SeqDataset
from src.features.skeleton_prior import CondSpec, build_skeleton_prior_vel_norm_k2


def _integrate_positions(start_pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    return start_pos[:, None, :] + np.cumsum(vel, axis=1)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump skeleton-prior (oracle waypoint) samples.npz for Go/No-Go 3 baselines.")
    p.add_argument("--exp_name", type=str, required=True, help="output dir under data/experiments/<exp_name>")
    p.add_argument("--model_type", type=str, choices=["diffusion", "physics", "baseline"], default="physics", help="dataset type: physics/diffusion use DiffusionDataset; baseline uses SeqDataset")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--nav_file", type=str, default=None, help="required for --model_type physics")
    p.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--splits_dir", type=str, default=None)
    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_samples", type=int, default=400)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond_spec = CondSpec(cond_mode="oracle_wp_end", num_waypoints=2)

    traj_ids = None
    if str(args.split) != "all":
        processed_dir = Path(args.data_path).resolve().parents[1]
        splits_dir = Path(args.splits_dir) if args.splits_dir else (processed_dir / "splits")
        split_file = splits_dir / f"{args.split}_ids.npy"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        traj_ids = np.load(split_file).astype(np.int64)

    if str(args.model_type) == "baseline":
        dataset = SeqDataset(
            args.data_path,
            obs_len=int(args.obs_len),
            pred_len=int(args.pred_len),
            traj_ids=traj_ids,
            cond_mode=str(cond_spec.cond_mode),
            waypoint_mode="rdp_dev",
            num_waypoints=int(cond_spec.num_waypoints),
        )
    else:
        if str(args.model_type) == "physics" and not args.nav_file:
            raise ValueError("--nav_file is required for --model_type physics")
        nav_file = args.nav_file if str(args.model_type) == "physics" else None
        dataset = DiffusionDataset(
            args.data_path,
            obs_len=int(args.obs_len),
            pred_len=int(args.pred_len),
            nav_field_file=nav_file,
            nav_patch_size=32,
            nav_patch_channel2="speed",
            traj_ids=traj_ids,
            cond_mode=str(cond_spec.cond_mode),
            waypoint_mode="rdp_dev",
            num_waypoints=int(cond_spec.num_waypoints),
        )

    norm = dataset.normalizer
    pos_min_t = torch.tensor(norm.pos_min, dtype=torch.float32, device=device)
    pos_range_t = torch.tensor(norm.pos_range, dtype=torch.float32, device=device)
    vel_mean_t = torch.tensor(norm.vel_mean, dtype=torch.float32, device=device)
    vel_std_t = torch.tensor(norm.vel_std, dtype=torch.float32, device=device)

    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    start_pos_list: list[np.ndarray] = []
    origin_pos_list: list[np.ndarray] = []
    dest_pos_list: list[np.ndarray] = []

    need = int(args.save_samples)
    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if args.max_batches is not None and int(bidx) >= int(args.max_batches):
                break
            if need <= 0:
                break

            obs = batch["obs"].to(device)
            cond = batch["cond"].to(device)

            start_pos = norm.denormalize_pos(obs[:, -1, :2].detach().cpu().numpy()).astype(np.float32, copy=False)

            if str(args.model_type) == "baseline":
                gt_vel_norm = batch["target_vel"].cpu().numpy()
            else:
                gt_vel_norm = batch["action"].cpu().numpy()
            gt_vel = norm.denormalize_vel(gt_vel_norm).astype(np.float32, copy=False)
            gt_pos = _integrate_positions(start_pos, gt_vel).astype(np.float32, copy=False)

            prior_vel_norm = build_skeleton_prior_vel_norm_k2(
                obs=obs,
                cond=cond,
                pred_len=int(args.pred_len),
                num_waypoints=int(cond_spec.num_waypoints),
                pos_min=pos_min_t,
                pos_range=pos_range_t,
                vel_mean=vel_mean_t,
                vel_std=vel_std_t,
            )
            prior_vel = norm.denormalize_vel(prior_vel_norm.detach().cpu().numpy()).astype(np.float32, copy=False)
            prior_pos = _integrate_positions(start_pos, prior_vel).astype(np.float32, copy=False)

            take = min(int(need), int(gt_pos.shape[0]))
            preds.append(prior_pos[:take])
            targets.append(gt_pos[:take])
            start_pos_list.append(start_pos[:take])

            if "trip_o" in batch and "trip_d" in batch:
                origin_pos = norm.denormalize_pos(batch["trip_o"][:take].detach().cpu().numpy()).astype(np.float32, copy=False)
                dest_pos = norm.denormalize_pos(batch["trip_d"][:take].detach().cpu().numpy()).astype(np.float32, copy=False)
                origin_pos_list.append(origin_pos)
                dest_pos_list.append(dest_pos)

            need -= take

    out_dir = Path(f"data/experiments/{args.exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "samples.npz"
    npz = {
        "preds": np.concatenate(preds, axis=0) if preds else np.zeros((0, int(args.pred_len), 2), dtype=np.float32),
        "targets": np.concatenate(targets, axis=0) if targets else np.zeros((0, int(args.pred_len), 2), dtype=np.float32),
        "start_pos": np.concatenate(start_pos_list, axis=0) if start_pos_list else np.zeros((0, 2), dtype=np.float32),
    }
    if origin_pos_list and dest_pos_list:
        npz["origin_pos"] = np.concatenate(origin_pos_list, axis=0)
        npz["dest_pos"] = np.concatenate(dest_pos_list, axis=0)
    meta = {
        "model_type": str(args.model_type),
        "cond_mode": str(cond_spec.cond_mode),
        "num_waypoints": int(cond_spec.num_waypoints),
        "prior_mode": "skeleton_wp",
        "data_path": str(args.data_path),
        "nav_file": str(args.nav_file) if args.nav_file else None,
        "split": str(args.split),
        "obs_len": int(args.obs_len),
        "pred_len": int(args.pred_len),
        "seed": int(args.seed),
    }
    np.savez_compressed(out_npz, **npz, meta=meta)

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[OK] saved {out_npz} (N={int(npz['targets'].shape[0])})")


if __name__ == "__main__":
    main()

