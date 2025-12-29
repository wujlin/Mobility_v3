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
from src.models.macro.macro_hardsupport import MacroHardSupportNet


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


def _denorm_pos(pos_norm: torch.Tensor, *, pos_min: torch.Tensor, pos_range: torch.Tensor) -> torch.Tensor:
    return (pos_norm + 1.0) * 0.5 * pos_range + pos_min


def _integrate_positions(start_pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    return start_pos[:, None, :] + np.cumsum(vel, axis=1)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump Macro Hard Support samples: z_k_grid for G1 gate.")
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
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
    p.add_argument("--save_samples", type=int, default=400)
    p.add_argument("--max_batches", type=int, default=13)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--windows_npz", type=str, default=None, help="Optional: npz with traj_idx/start_t to restrict windows.")
    return p


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
    # Build a filtered samples list (KISS): iterate and keep matches.
    kept = []
    for (tid, t0) in dataset.samples:
        k = (int(tid) << 32) + int(t0)
        if k in key_set:
            kept.append((int(tid), int(t0)))
    if not kept:
        raise RuntimeError(f"No matching windows found between dataset and {windows_npz}")
    dataset.samples = kept
    return dataset


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
    model = MacroHardSupportNet(
        obs_len=int(cfg.get("obs_len", args.obs_len)),
        obs_dim=4,
        cond_dim=6,
        patch_size=int(cfg.get("patch_size", args.patch_size)),
        in_channels=3,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        use_coord=bool(cfg.get("use_coord", False)),
    ).to(device=device)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

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
    targets_list = []
    start_pos_list = []
    origin_pos_list = []
    dest_pos_list = []
    traj_idx_list = []
    start_t_list = []
    z_k_grid_list = []

    need = int(args.save_samples)
    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if need <= 0:
                break

            obs = batch["obs"].to(device)
            cond_oracle = batch["cond"].to(device)  # (B,8)
            nav_patch = batch["nav_patch"].to(device)  # (B,3,K,K)
            action = batch["action"]  # (B,F,2) vel_norm
            trip_o = batch["trip_o"].to(device)
            trip_d = batch["trip_d"].to(device)
            meta = batch.get("meta")
            if isinstance(meta, dict):
                if "traj_idx" not in meta or "start_t" not in meta:
                    raise TypeError(f"batch['meta'] missing keys traj_idx/start_t: keys={list(meta.keys())}")
                tid_v = meta["traj_idx"]
                t0_v = meta["start_t"]
                if isinstance(tid_v, torch.Tensor):
                    tid = tid_v.detach().cpu().numpy().astype(np.int64, copy=False)
                else:
                    tid = np.asarray(tid_v, dtype=np.int64)
                if isinstance(t0_v, torch.Tensor):
                    t0 = t0_v.detach().cpu().numpy().astype(np.int64, copy=False)
                else:
                    t0 = np.asarray(t0_v, dtype=np.int64)
            elif isinstance(meta, (list, tuple)):
                tid = np.asarray([m["traj_idx"] for m in meta], dtype=np.int64)
                t0 = np.asarray([m["start_t"] for m in meta], dtype=np.int64)
            else:
                raise TypeError(f"Expected batch['meta'] to be dict or list[dict], got: {type(meta)}")

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

            # strict mask from normalized count channel
            strict = (nav_patch[:, 2] >= float(thr_norm))  # (B,K,K)
            empty = (strict.view(take, -1).sum(dim=1) == 0)
            if bool(torch.any(empty)):
                strict[empty] = True

            logits = model(obs=obs, cond=cond_trip_od, nav_patch=nav_patch)  # (B,3,K,K)
            logits = logits.masked_fill(~strict[:, None, :, :], -1e9)
            probs = torch.softmax(logits.view(take * 3, K * K), dim=-1)
            idx = torch.multinomial(probs, num_samples=int(args.k_samples), replacement=True)  # (B*3,Ks)
            idx = idx.view(take, 3, int(args.k_samples)).permute(0, 2, 1)  # (B,Ks,3)
            yy = (idx // K).to(torch.float32)
            xx = (idx % K).to(torch.float32)
            z_patch = torch.stack([yy, xx], dim=-1)  # (B,Ks,3,2) patch coords

            # global coords anchored at floor(start_pos)
            start_pos_grid = _denorm_pos(obs[:, -1, :2], pos_min=pos_min, pos_range=pos_range)  # (B,2)
            center = torch.floor(start_pos_grid)  # (B,2)
            z_grid = center[:, None, None, :] + (z_patch - r)  # (B,Ks,3,2)
            z_k_grid = z_grid.detach().cpu().numpy().astype(np.float32, copy=False)
            z_k_grid_list.append(z_k_grid)

            # Save start_pos and GT targets for downstream (optional)
            start_pos = start_pos_grid.detach().cpu().numpy().astype(np.float32, copy=False)
            start_pos_list.append(start_pos)
            gt_vel = (action.to(device) * vel_std[None, None, :] + vel_mean[None, None, :]).detach().cpu().numpy().astype(np.float32, copy=False)
            targets = _integrate_positions(start_pos, gt_vel).astype(np.float32, copy=False)
            targets_list.append(targets)

            origin_pos = _denorm_pos(trip_o, pos_min=pos_min, pos_range=pos_range).detach().cpu().numpy().astype(np.float32, copy=False)
            dest_pos = _denorm_pos(trip_d, pos_min=pos_min, pos_range=pos_range).detach().cpu().numpy().astype(np.float32, copy=False)
            origin_pos_list.append(origin_pos)
            dest_pos_list.append(dest_pos)

            traj_idx_list.append(tid.astype(np.int64, copy=False))
            start_t_list.append(t0.astype(np.int64, copy=False))

            need -= int(take)
            if args.max_batches is not None and int(bidx) + 1 >= int(args.max_batches):
                break

    z_k_grid = np.concatenate(z_k_grid_list, axis=0) if z_k_grid_list else np.zeros((0, int(args.k_samples), 3, 2), dtype=np.float32)
    start_pos = np.concatenate(start_pos_list, axis=0) if start_pos_list else np.zeros((0, 2), dtype=np.float32)
    targets = np.concatenate(targets_list, axis=0) if targets_list else np.zeros((0, int(args.pred_len), 2), dtype=np.float32)
    origin_pos = np.concatenate(origin_pos_list, axis=0) if origin_pos_list else np.zeros((0, 2), dtype=np.float32)
    dest_pos = np.concatenate(dest_pos_list, axis=0) if dest_pos_list else np.zeros((0, 2), dtype=np.float32)
    traj_idx = np.concatenate(traj_idx_list, axis=0) if traj_idx_list else np.zeros((0,), dtype=np.int64)
    start_t = np.concatenate(start_t_list, axis=0) if start_t_list else np.zeros((0,), dtype=np.int64)

    meta = {
        "checkpoint": str(args.checkpoint),
        "data_path": str(args.data_path),
        "nav_file": str(args.nav_file),
        "split": str(args.split),
        "obs_len": int(args.obs_len),
        "pred_len": int(args.pred_len),
        "patch_size": int(args.patch_size),
        "count_thr": float(args.count_thr),
        "thr_norm": float(thr_norm),
        "k_samples": int(args.k_samples),
        "seed": int(args.seed),
        "windows_npz": (str(args.windows_npz) if args.windows_npz else None),
    }
    out_npz = out_dir / "samples.npz"
    np.savez_compressed(
        out_npz,
        z_k_grid=z_k_grid,
        start_pos=start_pos,
        targets=targets,
        origin_pos=origin_pos,
        dest_pos=dest_pos,
        traj_idx=traj_idx,
        start_t=start_t,
        meta=meta,
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] saved {out_npz} (N={int(start_pos.shape[0])}, K={int(args.k_samples)})")


if __name__ == "__main__":
    main()
