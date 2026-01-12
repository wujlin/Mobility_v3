from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):  # type: ignore[no-redef]
        return x

from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.features.temporal import encode_route_temporal_2d
from src.training.route_npz_utils import (
    RouteNorm,
    compute_vel_from_positions,
    estimate_vel_stats,
    load_route_windows_npz,
    make_default_pos_bounds,
    normalize_pos,
    normalize_vel,
)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass(frozen=True)
class TrainConfig:
    train_npz: str
    out_dir: str
    pos_max: int
    pos_max_y: Optional[int]
    pos_max_x: Optional[int]
    max_train_n: Optional[int]
    temporal_mode: str
    temporal_tz_offset_hours: float
    hidden_dim: int
    diff_steps: int
    pred_type: str
    batch_size: int
    epochs: int
    lr: float
    num_workers: int
    max_batches: Optional[int]
    seed: int


class RouteWindowsVelDataset(Dataset):
    def __init__(self, *, data: dict, norm: RouteNorm, temporal_mode: str, temporal_tz_offset_hours: float):
        self.start_pos = np.asarray(data["start_pos"], dtype=np.float32)
        self.targets = np.asarray(data["targets"], dtype=np.float32)
        self.dest_pos = np.asarray(data["dest_pos"], dtype=np.float32)
        self.traj_idx = np.asarray(data["traj_idx"], dtype=np.int64)
        self.start_t = np.asarray(data["start_t"], dtype=np.int64)
        self.norm = norm
        self.temporal, self.temporal_effective = encode_route_temporal_2d(
            self.start_t,
            tz_offset_hours=float(temporal_tz_offset_hours),
            mode=str(temporal_mode),
        )

        vel = compute_vel_from_positions(self.start_pos, self.targets)  # (N,F,2)
        self.vel_norm = normalize_vel(vel, norm)  # (N,F,2)

        self.start_pos_norm = normalize_pos(self.start_pos, norm)  # (N,2)
        self.dest_pos_norm = normalize_pos(self.dest_pos, norm)  # (N,2)

    def __len__(self) -> int:
        return int(self.start_pos.shape[0])

    def __getitem__(self, idx: int) -> dict:
        idx = int(idx)
        # obs_len=1: only the start position, with zero velocity (avoid leaking future heading).
        obs = np.concatenate([self.start_pos_norm[idx], np.zeros((2,), dtype=np.float32)], axis=0)[None, :]  # (1,4)
        t0, t1 = float(self.temporal[idx, 0]), float(self.temporal[idx, 1])
        cond = np.asarray(
            [t0, t1, float(self.start_pos_norm[idx, 0]), float(self.start_pos_norm[idx, 1]), float(self.dest_pos_norm[idx, 0]), float(self.dest_pos_norm[idx, 1])],
            dtype=np.float32,
        )
        return {
            "obs": torch.from_numpy(obs).float(),
            "cond": torch.from_numpy(cond).float(),
            "action": torch.from_numpy(self.vel_norm[idx]).float(),
            "meta": {"traj_idx": int(self.traj_idx[idx]), "start_t": int(self.start_t[idx])},
        }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train an end-to-end diffusion baseline directly from route windows npz (Detroit).")
    p.add_argument("--train_npz", type=str, required=True, help="npz with start_pos/targets/dest_pos/traj_idx/start_t")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--pos_max", type=int, default=1023, help="Grid max coordinate (assumes y/x in [0,pos_max])")
    p.add_argument("--pos_max_y", type=int, default=None, help="Optional y max (grid units) for non-square canvases (overrides --pos_max for y).")
    p.add_argument("--pos_max_x", type=int, default=None, help="Optional x max (grid units) for non-square canvases (overrides --pos_max for x).")
    p.add_argument("--max_train_n", type=int, default=None, help="Optional: subsample training windows for speed")
    p.add_argument("--temporal_mode", type=str, choices=["auto", "simple", "zeros"], default="auto", help="Temporal feature for the (hour,day) slots: auto/simple/zeros.")
    p.add_argument("--temporal_tz_offset_hours", type=float, default=-5.0, help="Timezone offset used when temporal_mode!=zeros (Detroit/Columbus: -5).")

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--diff_steps", type=int, default=100)
    p.add_argument("--pred_type", type=str, choices=["eps", "v"], default="eps")

    # CFG training (destination dropout) for later CFG inference.
    p.add_argument("--cfg_drop_dest_prob", type=float, default=0.0, help="Training-time destination dropout prob (0 disables).")
    p.add_argument("--cfg_uncond_dest_mode", type=str, choices=["origin", "zeros"], default="origin", help="How to replace destination when dropped.")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=None, help="Limit batches per epoch (smoke runs)")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = TrainConfig(
        train_npz=str(args.train_npz),
        out_dir=str(args.out_dir),
        pos_max=int(args.pos_max),
        pos_max_y=(int(args.pos_max_y) if args.pos_max_y is not None else None),
        pos_max_x=(int(args.pos_max_x) if args.pos_max_x is not None else None),
        max_train_n=(int(args.max_train_n) if args.max_train_n is not None else None),
        temporal_mode=str(args.temporal_mode),
        temporal_tz_offset_hours=float(args.temporal_tz_offset_hours),
        hidden_dim=int(args.hidden_dim),
        diff_steps=int(args.diff_steps),
        pred_type=str(args.pred_type),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        num_workers=int(args.num_workers),
        max_batches=(int(args.max_batches) if args.max_batches is not None else None),
        seed=int(args.seed),
    )
    _set_seed(int(cfg.seed))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "last.pt"
    summary_path = out_dir / "train_summary.json"

    data = load_route_windows_npz(cfg.train_npz, max_n=cfg.max_train_n, seed=int(cfg.seed))
    start_pos = np.asarray(data["start_pos"], dtype=np.float32)
    targets = np.asarray(data["targets"], dtype=np.float32)
    dest_pos = np.asarray(data["dest_pos"], dtype=np.float32)
    n = int(start_pos.shape[0])
    f = int(targets.shape[1])

    pos_min, pos_max_arr = make_default_pos_bounds(pos_max=int(cfg.pos_max), pos_max_y=cfg.pos_max_y, pos_max_x=cfg.pos_max_x)
    pos_range = (pos_max_arr - pos_min + 1e-6).astype(np.float32)
    vel = compute_vel_from_positions(start_pos, targets)
    vel_mean, vel_std = estimate_vel_stats(vel)
    norm = RouteNorm(
        pos_min=pos_min.astype(np.float32, copy=False),
        pos_max=pos_max_arr.astype(np.float32, copy=False),
        pos_range=pos_range.astype(np.float32, copy=False),
        vel_mean=vel_mean.astype(np.float32, copy=False),
        vel_std=vel_std.astype(np.float32, copy=False),
    )

    dataset = RouteWindowsVelDataset(data=data, norm=norm, temporal_mode=str(cfg.temporal_mode), temporal_tz_offset_hours=float(cfg.temporal_tz_offset_hours))
    g = torch.Generator()
    g.manual_seed(int(cfg.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        generator=g,
        pin_memory=bool(torch.cuda.is_available()),
        persistent_workers=(int(cfg.num_workers) > 0),
    )
    batches_per_epoch = int(min(len(loader), int(cfg.max_batches))) if cfg.max_batches is not None else int(len(loader))
    updates_total = int(cfg.epochs) * int(max(batches_per_epoch, 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionTrajectoryModel(
        obs_dim=4,
        act_dim=2,
        cond_dim=6,
        obs_len=1,
        pred_len=int(f),
        hidden_dim=int(cfg.hidden_dim),
        diffusion_steps=int(cfg.diff_steps),
        prediction_type=str(cfg.pred_type),
    ).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg.lr))

    model.train()
    start_wall = time.time()
    steps = 0
    for epoch in range(int(cfg.epochs)):
        epoch_loss = 0.0
        epoch_steps = 0
        total = int(cfg.max_batches) if cfg.max_batches is not None else len(loader)
        pbar = tqdm(enumerate(loader), total=total, desc=f"epoch {epoch+1}/{int(cfg.epochs)}", dynamic_ncols=True)
        for batch_idx, batch in pbar:
            if cfg.max_batches is not None and int(batch_idx) >= int(cfg.max_batches):
                break
            obs = batch["obs"].to(device=device, non_blocking=True)
            cond = batch["cond"].to(device=device, non_blocking=True)
            target = batch["action"].to(device=device, non_blocking=True)

            # CFG training: drop destination with probability p (per sample).
            if float(args.cfg_drop_dest_prob) > 0.0:
                p = float(args.cfg_drop_dest_prob)
                if p < 0.0 or p > 1.0:
                    raise ValueError("--cfg_drop_dest_prob must be in [0,1]")
                mask = (torch.rand((cond.shape[0],), device=cond.device) < p)
                if mask.any():
                    mode = str(args.cfg_uncond_dest_mode)
                    if mode == "origin":
                        cond[mask, 4:6] = cond[mask, 2:4]
                    elif mode == "zeros":
                        cond[mask, 4:6].zero_()
                    else:  # pragma: no cover
                        raise ValueError(f"Unknown cfg_uncond_dest_mode: {mode}")

            optimizer.zero_grad(set_to_none=True)
            loss = model(obs, cond, target=target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            l = float(loss.detach().cpu().item())
            epoch_loss += l
            epoch_steps += 1
            steps += 1
            if epoch_steps > 0 and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=float(l), avg=float(epoch_loss / max(epoch_steps, 1)))

        avg_loss = epoch_loss / max(epoch_steps, 1)
        torch.save(
            {
                "epoch": int(epoch),
                "loss": float(avg_loss),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "task": "route_e2e_diffusion_npz",
                    "F": int(f),
                    "model": {
                        "hidden_dim": int(cfg.hidden_dim),
                        "diff_steps": int(cfg.diff_steps),
                        "pred_type": str(cfg.pred_type),
                        "obs_len": 1,
                        "cond_dim": 6,
                    },
                    "cfg": {
                        "cfg_drop_dest_prob": float(args.cfg_drop_dest_prob),
                        "cfg_uncond_dest_mode": str(args.cfg_uncond_dest_mode),
                    },
                    "temporal": {
                        "mode": str(cfg.temporal_mode),
                        "tz_offset_hours": float(cfg.temporal_tz_offset_hours),
                        "effective": str(dataset.temporal_effective),
                    },
                    "norm": norm.as_jsonable(),
                },
            },
            ckpt_path,
        )

    elapsed_s = float(time.time() - start_wall)
    result = {
        "inputs": {"train_npz": str(Path(cfg.train_npz).resolve())},
        "config": {
            "pos_max": int(cfg.pos_max),
            "pos_max_y": (int(cfg.pos_max_y) if cfg.pos_max_y is not None else None),
            "pos_max_x": (int(cfg.pos_max_x) if cfg.pos_max_x is not None else None),
            "max_train_n": (int(cfg.max_train_n) if cfg.max_train_n is not None else None),
            "temporal_mode": str(cfg.temporal_mode),
            "temporal_tz_offset_hours": float(cfg.temporal_tz_offset_hours),
            "hidden_dim": int(cfg.hidden_dim),
            "diff_steps": int(cfg.diff_steps),
            "pred_type": str(cfg.pred_type),
            "cfg_drop_dest_prob": float(args.cfg_drop_dest_prob),
            "cfg_uncond_dest_mode": str(args.cfg_uncond_dest_mode),
            "batch_size": int(cfg.batch_size),
            "epochs": int(cfg.epochs),
            "lr": float(cfg.lr),
            "num_workers": int(cfg.num_workers),
            "max_batches": (int(cfg.max_batches) if cfg.max_batches is not None else None),
            "seed": int(cfg.seed),
        },
        "stats": {
            "N": int(n),
            "F": int(f),
            "pos_min": [float(x) for x in norm.pos_min.tolist()],
            "pos_max": [float(x) for x in norm.pos_max.tolist()],
            "vel_mean": [float(x) for x in norm.vel_mean.tolist()],
            "vel_std": [float(x) for x in norm.vel_std.tolist()],
            "dest_pos_present": bool(dest_pos is not None),
            "batches_per_epoch": int(batches_per_epoch),
            "updates_total": int(updates_total),
            "updates_done": int(steps),
        },
        "timing": {"elapsed_s": float(elapsed_s)},
        "outputs": {"out_dir": str(out_dir.resolve()), "checkpoint": str(ckpt_path.resolve()), "summary_json": str(summary_path.resolve())},
    }
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
