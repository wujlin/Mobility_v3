from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RouteNorm:
    # Grid-space normalization (y,x) -> [-1,1]
    pos_min: np.ndarray  # (2,)
    pos_max: np.ndarray  # (2,)
    pos_range: np.ndarray  # (2,)
    # Velocity normalization (step displacement in grid space)
    vel_mean: np.ndarray  # (2,)
    vel_std: np.ndarray  # (2,)

    def as_jsonable(self) -> Dict[str, object]:
        return {
            "pos_min": [float(x) for x in self.pos_min.reshape(-1).tolist()],
            "pos_max": [float(x) for x in self.pos_max.reshape(-1).tolist()],
            "vel_mean": [float(x) for x in self.vel_mean.reshape(-1).tolist()],
            "vel_std": [float(x) for x in self.vel_std.reshape(-1).tolist()],
        }


def _to_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def compute_vel_from_positions(start_pos: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Convert GT positions to step displacement velocities.
    Args:
      start_pos: (N,2) grid [y,x]
      targets:   (N,F,2) grid [y,x] positions for steps 1..F
    Returns:
      vel: (N,F,2) where vel[:,t]=pos[t+1]-pos[t]
    """
    start_pos = _to_f32(start_pos).reshape(-1, 2)
    targets = _to_f32(targets)
    poly = np.concatenate([start_pos[:, None, :], targets], axis=1)  # (N,F+1,2)
    vel = poly[:, 1:, :] - poly[:, :-1, :]
    return vel.astype(np.float32, copy=False)


def estimate_vel_stats(vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vel = _to_f32(vel).reshape(-1, 2)
    vel_mean = np.mean(vel, axis=0, dtype=np.float64).astype(np.float32)
    vel_std = np.std(vel, axis=0, dtype=np.float64).astype(np.float32)
    vel_std = np.maximum(vel_std, 1e-3).astype(np.float32, copy=False)
    return vel_mean, vel_std


def make_default_pos_bounds(*, pos_max: int = 1023) -> Tuple[np.ndarray, np.ndarray]:
    pos_min = np.asarray([0.0, 0.0], dtype=np.float32)
    pos_max_arr = np.asarray([float(pos_max), float(pos_max)], dtype=np.float32)
    return pos_min, pos_max_arr


def normalize_pos(pos_grid: np.ndarray, norm: RouteNorm) -> np.ndarray:
    pos_grid = _to_f32(pos_grid)
    return (2.0 * (pos_grid - norm.pos_min) / norm.pos_range - 1.0).astype(np.float32, copy=False)


def denormalize_pos(pos_norm: np.ndarray, norm: RouteNorm) -> np.ndarray:
    pos_norm = _to_f32(pos_norm)
    return (((pos_norm + 1.0) * 0.5) * norm.pos_range + norm.pos_min).astype(np.float32, copy=False)


def normalize_vel(vel: np.ndarray, norm: RouteNorm) -> np.ndarray:
    vel = _to_f32(vel)
    return ((vel - norm.vel_mean) / norm.vel_std).astype(np.float32, copy=False)


def denormalize_vel(vel_norm: np.ndarray, norm: RouteNorm) -> np.ndarray:
    vel_norm = _to_f32(vel_norm)
    return (vel_norm * norm.vel_std + norm.vel_mean).astype(np.float32, copy=False)


def load_route_windows_npz(
    path: str,
    *,
    max_n: Optional[int],
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Load a route-windows npz (start_pos, targets, dest_pos, traj_idx, start_t).
    Optionally subsample to max_n (without replacement).
    """
    p = Path(path)
    data = np.load(str(p), allow_pickle=True)
    need = {"start_pos", "targets"}
    if not need.issubset(set(data.files)):
        raise ValueError(f"npz missing keys: {sorted(list(need - set(data.files)))}. got={sorted(list(data.files))}")

    start_pos = _to_f32(data["start_pos"]).reshape(-1, 2)
    targets = _to_f32(data["targets"])
    n = int(start_pos.shape[0])
    if targets.shape[0] != n:
        raise ValueError(f"Bad npz: start_pos N={n} != targets N={targets.shape[0]}")

    if "dest_pos" in data.files:
        dest_pos = _to_f32(data["dest_pos"]).reshape(-1, 2)
        if dest_pos.shape[0] != n:
            raise ValueError(f"Bad npz: dest_pos N={dest_pos.shape[0]} != {n}")
    else:
        dest_pos = targets[:, -1, :].astype(np.float32, copy=False)

    traj_idx = np.asarray(data["traj_idx"], dtype=np.int64).reshape(-1) if "traj_idx" in data.files else np.arange(n, dtype=np.int64)
    start_t = np.asarray(data["start_t"], dtype=np.int64).reshape(-1) if "start_t" in data.files else np.zeros((n,), dtype=np.int64)
    if traj_idx.shape[0] != n or start_t.shape[0] != n:
        raise ValueError(f"Bad npz: traj_idx/start_t N mismatch (traj_idx={traj_idx.shape[0]}, start_t={start_t.shape[0]}, expected={n})")

    if max_n is not None:
        k = int(max_n)
        if k > 0 and n > k:
            rng = np.random.default_rng(int(seed))
            pick = rng.choice(n, size=k, replace=False)
            pick = np.sort(pick.astype(np.int64, copy=False))
            start_pos = start_pos[pick]
            targets = targets[pick]
            dest_pos = dest_pos[pick]
            traj_idx = traj_idx[pick]
            start_t = start_t[pick]
            n = int(k)

    return {
        "start_pos": start_pos.astype(np.float32, copy=False),
        "targets": targets.astype(np.float32, copy=False),
        "dest_pos": dest_pos.astype(np.float32, copy=False),
        "traj_idx": traj_idx.astype(np.int64, copy=False),
        "start_t": start_t.astype(np.int64, copy=False),
    }

