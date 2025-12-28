from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.datasets_seq import SeqDataset
from src.legacy._deprecated_waypoint_mdn import WaypointMDN, WaypointMDNConfig


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train macro waypoint MDN: p(z=[wp1,wp2,end_anchor] | obs, trip_od).")
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="train")
    p.add_argument("--splits_dir", type=str, default=None)
    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_components", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=None, help="limit batches per epoch (for quick iteration)")
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

    dataset = SeqDataset(
        args.data_path,
        obs_len=int(args.obs_len),
        pred_len=int(args.pred_len),
        traj_ids=traj_ids,
        cond_mode="oracle_wp_end",
        waypoint_mode=str(args.waypoint_mode),
        num_waypoints=int(args.num_waypoints),
    )
    g = torch.Generator()
    g.manual_seed(int(args.seed))
    pin_memory = bool(torch.cuda.is_available())
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        generator=g,
        pin_memory=pin_memory,
        persistent_workers=(int(args.num_workers) > 0),
    )

    cfg = WaypointMDNConfig(z_dim=6, n_components=int(args.n_components))
    model = WaypointMDN(obs_dim=4, cond_dim=6, hidden_dim=int(args.hidden_dim), cfg=cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))

    save_dir = Path(f"data/experiments/{args.exp_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model_type": "macro_waypoint_mdn",
        "data_path": str(args.data_path),
        "split": str(args.split),
        "splits_dir": str(args.splits_dir) if args.splits_dir else None,
        "obs_len": int(args.obs_len),
        "pred_len": int(args.pred_len),
        "hidden_dim": int(args.hidden_dim),
        "n_components": int(args.n_components),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "waypoint_mode": str(args.waypoint_mode),
        "num_waypoints": int(args.num_waypoints),
        "input_cond": "trip_od",
        "target_z": "oracle_wp_end[wp1,wp2,end]",
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    model.train()
    for epoch in range(int(args.epochs)):
        start = time.time()
        total = 0.0
        nb = 0

        for bidx, batch in enumerate(loader):
            obs = batch["obs"].to(device)
            cond_oracle = batch["cond"].to(device)
            trip_o = batch["trip_o"].to(device)
            trip_d = batch["trip_d"].to(device)

            cond_trip_od = torch.cat([cond_oracle[:, :2], trip_o, trip_d], dim=-1)
            z = cond_oracle[:, 2:]

            optimizer.zero_grad(set_to_none=True)
            loss = model.nll(obs, cond_trip_od, z)
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            nb += 1
            if bidx % 100 == 0:
                print(f"Epoch {epoch} | Batch {bidx} | NLL {loss.item():.4f}")
            if args.max_batches is not None and int(args.max_batches) > 0 and nb >= int(args.max_batches):
                break

        avg = total / float(max(nb, 1))
        dur = time.time() - start
        print(f"Epoch {epoch} Done. Avg NLL: {avg:.4f}. Time: {dur:.1f}s")

        ckpt = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": float(avg),
            "config": run_config,
        }
        torch.save(ckpt, save_dir / "last.pt")


if __name__ == "__main__":
    main()
