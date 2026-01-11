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

from src.features.skeleton_prior import build_skeleton_prior_vel_norm_k2
from src.features.waypoints import WaypointConfig, extract_oracle_waypoints_from_future
from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
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
    max_train_n: Optional[int]
    waypoint_mode: str
    waypoint_turn_alpha: float
    num_waypoints: int
    hidden_dim: int
    diff_steps: int
    pred_type: str
    batch_size: int
    epochs: int
    lr: float
    num_workers: int
    max_batches: Optional[int]
    seed: int
    precompute_device: str


class RouteExecResidualDataset(Dataset):
    def __init__(
        self,
        *,
        obs: np.ndarray,  # (N,1,4)
        cond: np.ndarray,  # (N,8)
        residual: np.ndarray,  # (N,F,2) normalized
        traj_idx: np.ndarray,  # (N,)
        start_t: np.ndarray,  # (N,)
    ) -> None:
        self.obs = np.asarray(obs, dtype=np.float32)
        self.cond = np.asarray(cond, dtype=np.float32)
        self.residual = np.asarray(residual, dtype=np.float32)
        self.traj_idx = np.asarray(traj_idx, dtype=np.int64)
        self.start_t = np.asarray(start_t, dtype=np.int64)

        if self.obs.ndim != 3 or self.obs.shape[1:] != (1, 4):
            raise ValueError(f"Expected obs (N,1,4), got {self.obs.shape}")
        if self.cond.ndim != 2 or self.cond.shape[1] != 8:
            raise ValueError(f"Expected cond (N,8), got {self.cond.shape}")
        if self.residual.ndim != 3 or self.residual.shape[-1] != 2:
            raise ValueError(f"Expected residual (N,F,2), got {self.residual.shape}")
        if self.obs.shape[0] != self.cond.shape[0] or self.obs.shape[0] != self.residual.shape[0]:
            raise ValueError("N mismatch among obs/cond/residual")

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, idx: int) -> dict:
        idx = int(idx)
        return {
            "obs": torch.from_numpy(self.obs[idx]).float(),
            "cond": torch.from_numpy(self.cond[idx]).float(),
            "action": torch.from_numpy(self.residual[idx]).float(),
            "meta": {"traj_idx": int(self.traj_idx[idx]), "start_t": int(self.start_t[idx])},
        }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train an execution-stage residual diffusion model conditioned on oracle waypoints (NPZ route windows)."
    )
    p.add_argument("--train_npz", type=str, required=True, help="npz with start_pos/targets/dest_pos/traj_idx/start_t")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--pos_max", type=int, default=1023, help="Grid max coordinate (assumes y/x in [0,pos_max])")
    p.add_argument("--max_train_n", type=int, default=None, help="Optional: subsample training windows for speed")

    p.add_argument("--waypoint_mode", type=str, choices=["rdp_dev", "rdp_turn"], default="rdp_dev")
    p.add_argument("--waypoint_turn_alpha", type=float, default=1.0, help="When waypoint_mode=rdp_turn: weight for turn-aware waypoint selection.")
    p.add_argument("--num_waypoints", type=int, default=2, help="Only supports 2 for skeleton prior (KISS).")
    p.add_argument("--precompute_device", type=str, choices=["cpu", "cuda"], default="cpu", help="Device for prior precompute.")

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--diff_steps", type=int, default=100)
    p.add_argument("--pred_type", type=str, choices=["eps", "v"], default="eps")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=None, help="Limit batches per epoch (smoke runs)")
    p.add_argument("--seed", type=int, default=0)
    return p


def _extract_waypoints_batch(
    *,
    start_pos: np.ndarray,  # (N,2)
    targets: np.ndarray,  # (N,F,2)
    cfg: WaypointConfig,
) -> np.ndarray:
    n = int(start_pos.shape[0])
    k = int(cfg.num_waypoints)
    out = np.zeros((n, k, 2), dtype=np.float32)
    for i in range(n):
        _, wp = extract_oracle_waypoints_from_future(start_pos=start_pos[i], future_pos=targets[i], cfg=cfg)
        if wp.shape != (k, 2):
            raise RuntimeError(f"Bad waypoint shape: {wp.shape}, expected {(k, 2)}")
        out[i] = wp
    return out


def main() -> None:
    args = build_argparser().parse_args()
    cfg = TrainConfig(
        train_npz=str(args.train_npz),
        out_dir=str(args.out_dir),
        pos_max=int(args.pos_max),
        max_train_n=(int(args.max_train_n) if args.max_train_n is not None else None),
        waypoint_mode=str(args.waypoint_mode),
        waypoint_turn_alpha=float(args.waypoint_turn_alpha),
        num_waypoints=int(args.num_waypoints),
        precompute_device=str(args.precompute_device),
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

    if int(cfg.num_waypoints) != 2:
        raise ValueError("--num_waypoints currently must be 2 (skeleton prior supports K=2 only).")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_route_windows_npz(cfg.train_npz, max_n=cfg.max_train_n, seed=int(cfg.seed))
    start_pos = np.asarray(data["start_pos"], dtype=np.float32)
    targets = np.asarray(data["targets"], dtype=np.float32)
    dest_pos = np.asarray(data["dest_pos"], dtype=np.float32)
    traj_idx = np.asarray(data["traj_idx"], dtype=np.int64)
    start_t = np.asarray(data["start_t"], dtype=np.int64)
    n = int(start_pos.shape[0])
    f = int(targets.shape[1])

    wp_cfg = WaypointConfig(mode=str(cfg.waypoint_mode), num_waypoints=int(cfg.num_waypoints), turn_alpha=float(cfg.waypoint_turn_alpha))
    waypoints = _extract_waypoints_batch(start_pos=start_pos, targets=targets, cfg=wp_cfg)  # (N,2,2)

    pos_min, pos_max_arr = make_default_pos_bounds(pos_max=int(cfg.pos_max))
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

    start_pos_norm = normalize_pos(start_pos, norm)  # (N,2)
    dest_pos_norm = normalize_pos(dest_pos, norm)  # (N,2)
    wp_norm = normalize_pos(waypoints.reshape(-1, 2), norm).reshape(n, int(cfg.num_waypoints), 2)  # (N,2,2)
    obs = np.concatenate([start_pos_norm, np.zeros((n, 2), dtype=np.float32)], axis=1)[:, None, :]  # (N,1,4)
    cond = np.concatenate(
        [
            np.zeros((n, 2), dtype=np.float32),  # [hour, day] placeholders
            wp_norm.reshape(n, -1),
            dest_pos_norm,
        ],
        axis=1,
    ).astype(np.float32, copy=False)  # (N,8)

    vel_norm = normalize_vel(vel, norm)  # (N,F,2)

    pre_dev = str(cfg.precompute_device)
    if pre_dev == "cuda" and not torch.cuda.is_available():
        pre_dev = "cpu"
    pre_device = torch.device(pre_dev)

    with torch.no_grad():
        obs_t = torch.from_numpy(obs).to(device=pre_device, dtype=torch.float32)
        cond_t = torch.from_numpy(cond).to(device=pre_device, dtype=torch.float32)
        pos_min_t = torch.tensor(norm.pos_min, device=pre_device, dtype=torch.float32)
        pos_range_t = torch.tensor(norm.pos_range, device=pre_device, dtype=torch.float32)
        vel_mean_t = torch.tensor(norm.vel_mean, device=pre_device, dtype=torch.float32)
        vel_std_t = torch.tensor(norm.vel_std, device=pre_device, dtype=torch.float32)
        prior_vel_norm = build_skeleton_prior_vel_norm_k2(
            obs=obs_t,
            cond=cond_t,
            pred_len=int(f),
            num_waypoints=int(cfg.num_waypoints),
            pos_min=pos_min_t,
            pos_range=pos_range_t,
            vel_mean=vel_mean_t,
            vel_std=vel_std_t,
        ).detach().cpu().numpy()  # (N,F,2)

    residual = (vel_norm - prior_vel_norm).astype(np.float32, copy=False)

    dataset = RouteExecResidualDataset(obs=obs, cond=cond, residual=residual, traj_idx=traj_idx, start_t=start_t)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionTrajectoryModel(
        obs_dim=4,
        act_dim=2,
        cond_dim=8,
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

    ckpt_path = out_dir / "last.pt"
    for epoch in range(int(cfg.epochs)):
        epoch_loss = 0.0
        epoch_steps = 0
        total = int(cfg.max_batches) if cfg.max_batches is not None else len(loader)
        pbar = tqdm(enumerate(loader), total=total, desc=f"epoch {epoch+1}/{int(cfg.epochs)}", dynamic_ncols=True)
        for batch_idx, batch in pbar:
            if cfg.max_batches is not None and int(batch_idx) >= int(cfg.max_batches):
                break
            obs_b = batch["obs"].to(device=device, non_blocking=True)
            cond_b = batch["cond"].to(device=device, non_blocking=True)
            target_b = batch["action"].to(device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = model(obs_b, cond_b, target=target_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())
            epoch_steps += 1
            steps += 1
            if epoch_steps > 0 and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=float(loss.detach().cpu().item()), avg=float(epoch_loss / max(epoch_steps, 1)))

        avg_loss = epoch_loss / max(epoch_steps, 1)
        torch.save(
            {
                "epoch": int(epoch),
                "loss": float(avg_loss),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "task": "route_exec_residual_diffusion_wp_npz",
                    "F": int(f),
                    "model": {
                        "hidden_dim": int(cfg.hidden_dim),
                        "diff_steps": int(cfg.diff_steps),
                        "pred_type": str(cfg.pred_type),
                        "obs_len": 1,
                        "cond_dim": 8,
                    },
                    "waypoints": {"mode": str(cfg.waypoint_mode), "turn_alpha": float(cfg.waypoint_turn_alpha), "num_waypoints": int(cfg.num_waypoints)},
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
            "max_train_n": (int(cfg.max_train_n) if cfg.max_train_n is not None else None),
            "waypoint_mode": str(cfg.waypoint_mode),
            "waypoint_turn_alpha": float(cfg.waypoint_turn_alpha),
            "num_waypoints": int(cfg.num_waypoints),
            "precompute_device": str(cfg.precompute_device),
            "hidden_dim": int(cfg.hidden_dim),
            "diff_steps": int(cfg.diff_steps),
            "pred_type": str(cfg.pred_type),
            "batch_size": int(cfg.batch_size),
            "epochs": int(cfg.epochs),
            "lr": float(cfg.lr),
            "num_workers": int(cfg.num_workers),
            "max_batches": (int(cfg.max_batches) if cfg.max_batches is not None else None),
            "seed": int(cfg.seed),
        },
        "stats": {"N": int(n), "F": int(f), "steps": int(steps), "elapsed_s": float(elapsed_s)},
        "outputs": {"checkpoint": str(ckpt_path.resolve())},
    }
    (out_dir / "train_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
