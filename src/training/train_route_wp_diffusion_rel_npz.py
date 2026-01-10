from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.features.semantic_od import fit_semantic_norm, load_poi_total_and_landuse_entropy, normalize_semantic, semantic_od_features
from src.features.waypoints import WaypointConfig, extract_oracle_waypoints_from_future
from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.training.route_npz_utils import RouteNorm, load_route_windows_npz, make_default_pos_bounds, normalize_pos


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _od_bin_center(pos: np.ndarray, *, bin_size: float) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    b = float(bin_size)
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("--od_bin must be > 0")
    return (np.floor(pos / b) + 0.5) * b


def _waypoints_rel_so(
    *,
    start: np.ndarray,  # (N,2)
    dest: np.ndarray,  # (N,2)
    waypoints: np.ndarray,  # (N,K,2)
    eps: float = 1e-6,
) -> np.ndarray:
    start = np.asarray(start, dtype=np.float32)
    dest = np.asarray(dest, dtype=np.float32)
    wp = np.asarray(waypoints, dtype=np.float32)
    v = dest - start  # (N,2)
    L = np.linalg.norm(v, axis=1).astype(np.float32)
    L = np.maximum(L, float(eps))
    e_par = v / L[:, None]  # (N,2)
    e_perp = np.stack([-e_par[:, 1], e_par[:, 0]], axis=1)  # (N,2)

    d = wp - start[:, None, :]  # (N,K,2)
    s = np.sum(d * e_par[:, None, :], axis=2) / L[:, None]
    o = np.sum(d * e_perp[:, None, :], axis=2) / L[:, None]
    rel = np.stack([s, o], axis=2).astype(np.float32, copy=False)  # (N,K,2)

    # Canonical ordering: sort by s (monotone along chord).
    order = np.argsort(rel[:, :, 0], axis=1)
    rel = np.take_along_axis(rel, order[:, :, None], axis=1)
    return rel.astype(np.float32, copy=False)


def _extract_oracle_waypoints(
    *,
    start_pos: np.ndarray,  # (N,2)
    targets: np.ndarray,  # (N,F,2)
    num_waypoints: int,
) -> np.ndarray:
    n = int(start_pos.shape[0])
    k = int(num_waypoints)
    cfg = WaypointConfig(mode="rdp_dev", num_waypoints=k)
    out = np.zeros((n, k, 2), dtype=np.float32)
    for i in range(n):
        _, wp = extract_oracle_waypoints_from_future(start_pos=start_pos[i], future_pos=targets[i], cfg=cfg)
        if wp.shape != (k, 2):
            raise RuntimeError(f"Bad oracle waypoint shape: {wp.shape}, expected {(k, 2)}")
        out[i] = wp
    return out.astype(np.float32, copy=False)


def _compute_rel_norm(rel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rel = np.asarray(rel, dtype=np.float32).reshape(-1, 2)
    mean = np.mean(rel, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(rel, axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-3).astype(np.float32, copy=False)
    return mean, std


def _normalize_rel(rel: np.ndarray, *, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    rel = np.asarray(rel, dtype=np.float32)
    return ((rel - mean[None, None, :]) / std[None, None, :]).astype(np.float32, copy=False)


@dataclass(frozen=True)
class TrainConfig:
    train_npz: str
    out_dir: str
    pos_max: int
    max_train_n: Optional[int]
    num_waypoints: int
    od_bin: float
    o_clip: float
    semantic_dir: Optional[str]
    hidden_dim: int
    diff_steps: int
    pred_type: str
    batch_size: int
    epochs: int
    lr: float
    num_workers: int
    max_batches: Optional[int]
    seed: int


class WaypointRelDataset(Dataset):
    def __init__(self, *, obs: np.ndarray, cond: np.ndarray, target_rel_norm: np.ndarray, traj_idx: np.ndarray, start_t: np.ndarray):
        self.obs = np.asarray(obs, dtype=np.float32)
        self.cond = np.asarray(cond, dtype=np.float32)
        self.target = np.asarray(target_rel_norm, dtype=np.float32)
        self.traj_idx = np.asarray(traj_idx, dtype=np.int64)
        self.start_t = np.asarray(start_t, dtype=np.int64)

        if self.obs.ndim != 3 or self.obs.shape[1:] != (1, 4):
            raise ValueError(f"Expected obs (N,1,4), got {self.obs.shape}")
        if self.cond.ndim != 2 or self.cond.shape[1] <= 0:
            raise ValueError(f"Expected cond (N,D), got {self.cond.shape}")
        if self.target.ndim != 3 or self.target.shape[-1] != 2:
            raise ValueError(f"Expected target (N,K,2), got {self.target.shape}")
        if self.obs.shape[0] != self.cond.shape[0] or self.obs.shape[0] != self.target.shape[0]:
            raise ValueError("N mismatch among obs/cond/target")

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, idx: int) -> dict:
        idx = int(idx)
        return {
            "obs": torch.from_numpy(self.obs[idx]).float(),
            "cond": torch.from_numpy(self.cond[idx]).float(),
            "action": torch.from_numpy(self.target[idx]).float(),
            "meta": {"traj_idx": int(self.traj_idx[idx]), "start_t": int(self.start_t[idx])},
        }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a diffusion decision model: p(waypoints | OD-bin) on route windows npz.")
    p.add_argument("--train_npz", type=str, required=True, help="npz with start_pos/targets/dest_pos/traj_idx/start_t")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--pos_max", type=int, default=1023)
    p.add_argument("--max_train_n", type=int, default=None)

    p.add_argument("--num_waypoints", type=int, default=2)
    p.add_argument("--od_bin", type=float, default=128.0, help="Bin size (grid units) for OD intent conditioning.")
    p.add_argument("--o_clip", type=float, default=2.0, help="Clip signed offset o (in chord-normalized units) for stability.")
    p.add_argument(
        "--semantic_dir",
        type=str,
        default=None,
        help="Optional directory containing poi_density_*.npy and landuse_entropy.npy (adds OD semantics to cond).",
    )

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--diff_steps", type=int, default=50)
    p.add_argument("--pred_type", type=str, choices=["eps", "v"], default="eps")

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
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
        max_train_n=(int(args.max_train_n) if args.max_train_n is not None else None),
        num_waypoints=int(args.num_waypoints),
        od_bin=float(args.od_bin),
        o_clip=float(args.o_clip),
        semantic_dir=(str(args.semantic_dir) if args.semantic_dir else None),
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
    traj_idx = np.asarray(data["traj_idx"], dtype=np.int64)
    start_t = np.asarray(data["start_t"], dtype=np.int64)

    n = int(start_pos.shape[0])
    k = int(cfg.num_waypoints)
    if k <= 0:
        raise ValueError("--num_waypoints must be > 0")

    # OD intent conditioning: bin centers (shared across nearby ODs) -> multi-modality supervision.
    start_ctr = _od_bin_center(start_pos, bin_size=float(cfg.od_bin))
    dest_ctr = _od_bin_center(dest_pos, bin_size=float(cfg.od_bin))

    pos_min, pos_max_arr = make_default_pos_bounds(pos_max=int(cfg.pos_max))
    pos_range = (pos_max_arr - pos_min + 1e-6).astype(np.float32)
    norm = RouteNorm(
        pos_min=pos_min.astype(np.float32, copy=False),
        pos_max=pos_max_arr.astype(np.float32, copy=False),
        pos_range=pos_range.astype(np.float32, copy=False),
        vel_mean=np.zeros((2,), dtype=np.float32),
        vel_std=np.ones((2,), dtype=np.float32),
    )

    start_ctr_norm = normalize_pos(start_ctr, norm)  # (N,2)
    dest_ctr_norm = normalize_pos(dest_ctr, norm)  # (N,2)

    # obs: start-bin only (avoid leaking per-window micro differences).
    obs = np.concatenate([start_ctr_norm, np.zeros((n, 2), dtype=np.float32)], axis=1)[:, None, :]  # (N,1,4)
    # cond: (hour,day placeholders) + (start_bin, dest_bin)
    base_cond = np.concatenate([np.zeros((n, 2), dtype=np.float32), start_ctr_norm, dest_ctr_norm], axis=1).astype(np.float32, copy=False)  # (N,6)

    sem_norm = None
    sem_keys = None
    sem_cfg = None
    if cfg.semantic_dir:
        poi_total, landuse_entropy = load_poi_total_and_landuse_entropy(cfg.semantic_dir)
        sem_raw, sem_keys = semantic_od_features(
            start_ctr=start_pos,
            dest_ctr=dest_pos,
            poi_total=poi_total,
            landuse_entropy=landuse_entropy,
            log_poi=True,
        )
        sem_cfg = fit_semantic_norm(sem_raw, keys=sem_keys)
        sem_norm = normalize_semantic(sem_raw, sem_cfg)
        cond = np.concatenate([base_cond, sem_norm], axis=1).astype(np.float32, copy=False)
    else:
        cond = base_cond

    # Oracle targets: waypoints -> chord-relative (s,o).
    wp_abs = _extract_oracle_waypoints(start_pos=start_pos, targets=targets, num_waypoints=int(cfg.num_waypoints))  # (N,K,2)
    rel = _waypoints_rel_so(start=start_pos, dest=dest_pos, waypoints=wp_abs)  # (N,K,2)
    rel[:, :, 0] = np.clip(rel[:, :, 0], 0.0, 1.0)
    if float(cfg.o_clip) > 0.0:
        rel[:, :, 1] = np.clip(rel[:, :, 1], -float(cfg.o_clip), float(cfg.o_clip))

    rel_mean, rel_std = _compute_rel_norm(rel)
    target_rel_norm = _normalize_rel(rel, mean=rel_mean, std=rel_std)

    dataset = WaypointRelDataset(obs=obs, cond=cond, target_rel_norm=target_rel_norm, traj_idx=traj_idx, start_t=start_t)
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
    cond_dim = int(cond.shape[1])
    model = DiffusionTrajectoryModel(
        obs_dim=4,
        act_dim=2,
        cond_dim=int(cond_dim),
        obs_len=1,
        pred_len=int(k),
        hidden_dim=int(cfg.hidden_dim),
        diffusion_steps=int(cfg.diff_steps),
        prediction_type=str(cfg.pred_type),
    ).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg.lr))

    start_wall = time.time()
    model.train()
    for epoch in range(int(cfg.epochs)):
        epoch_loss = 0.0
        epoch_steps = 0
        for batch_idx, batch in enumerate(loader):
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

        avg_loss = epoch_loss / max(epoch_steps, 1)
        torch.save(
            {
                "epoch": int(epoch),
                "loss": float(avg_loss),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "task": "route_wp_diffusion_rel_npz",
                    "K_waypoints": int(k),
                    "model": {
                        "hidden_dim": int(cfg.hidden_dim),
                        "diff_steps": int(cfg.diff_steps),
                        "pred_type": str(cfg.pred_type),
                        "obs_len": 1,
                        "cond_dim": int(cond_dim),
                    },
                    "od_bin": float(cfg.od_bin),
                    "o_clip": float(cfg.o_clip),
                    "pos_norm": {
                        "pos_min": [float(x) for x in norm.pos_min.tolist()],
                        "pos_max": [float(x) for x in norm.pos_max.tolist()],
                    },
                    "rel_norm": {"mean": [float(x) for x in rel_mean.tolist()], "std": [float(x) for x in rel_std.tolist()]},
                    "semantic_od_norm": (sem_cfg.to_json() if sem_cfg is not None else None),
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
            "num_waypoints": int(cfg.num_waypoints),
            "od_bin": float(cfg.od_bin),
            "o_clip": float(cfg.o_clip),
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
        "stats": {"N": int(n), "F": int(targets.shape[1]), "rel_mean": [float(x) for x in rel_mean.tolist()], "rel_std": [float(x) for x in rel_std.tolist()]},
        "outputs": {"checkpoint": str(ckpt_path.resolve())},
        "timing": {"elapsed_s": float(elapsed_s)},
    }
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
