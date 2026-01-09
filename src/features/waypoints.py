from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


WaypointMode = Literal["rdp_dev"]


@dataclass(frozen=True)
class WaypointConfig:
    mode: WaypointMode = "rdp_dev"
    num_waypoints: int = 2


def _distance_point_to_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = b - a
    ap = points - a[None, :]
    ab2 = float(np.sum(ab * ab))
    if not np.isfinite(ab2) or ab2 <= 1e-8:
        return np.linalg.norm(points - a[None, :], axis=-1).astype(np.float32)
    t = np.sum(ap * ab[None, :], axis=-1) / ab2
    t = np.clip(t, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - proj, axis=-1).astype(np.float32)


def pick_waypoint_indices_rdp_fixed_k(points: np.ndarray, *, k: int) -> np.ndarray:
    """
    Fixed-K RDP (largest deviation first).

    Args:
        points: (T,2) polyline vertices including endpoints.
        k: number of internal waypoint indices to pick.
    Returns:
        idx: (k,) int64 indices in [1, T-2], sorted ascending.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points (T,2), got {points.shape}")
    T = int(points.shape[0])
    if int(k) <= 0 or T < 3:
        return np.zeros((0,), dtype=np.int64)

    segs = [(0, T - 1)]
    picks: list[int] = []

    for _ in range(int(k)):
        best_dev = None
        best_seg_i = None
        best_mid = None
        for si, (a_idx, b_idx) in enumerate(segs):
            if int(b_idx) - int(a_idx) <= 1:
                continue
            a = points[int(a_idx)]
            b = points[int(b_idx)]
            mid_points = points[int(a_idx) + 1 : int(b_idx)]
            if mid_points.size == 0:
                continue
            d = _distance_point_to_segment(mid_points, a, b).reshape(-1)
            mid_off = int(np.argmax(d))
            dev = float(d[mid_off])
            mid_idx = int(a_idx) + 1 + int(mid_off)
            if best_dev is None or dev > float(best_dev):
                best_dev = dev
                best_seg_i = si
                best_mid = mid_idx
        if best_seg_i is None or best_mid is None:
            break
        if best_mid in picks:
            break
        picks.append(int(best_mid))
        a_idx, b_idx = segs.pop(int(best_seg_i))
        segs.append((int(a_idx), int(best_mid)))
        segs.append((int(best_mid), int(b_idx)))

    idx = [p for p in picks if 1 <= int(p) <= int(T - 2)]
    idx = sorted(set(int(p) for p in idx))[: int(k)]

    # Fallback: time quantiles if not enough picks.
    if len(idx) < int(k):
        cand = np.linspace(1, T - 2, num=int(k), dtype=np.float32)
        fill = [int(np.rint(x)) for x in cand.tolist()]
        for j in fill:
            if len(idx) >= int(k):
                break
            j = int(np.clip(j, 1, T - 2))
            if j not in idx:
                idx.append(j)
        idx = sorted(idx)[: int(k)]

    return np.asarray(idx, dtype=np.int64)


def extract_oracle_waypoints_from_future(
    start_pos: np.ndarray,  # (2,)
    future_pos: np.ndarray,  # (F,2)
    cfg: WaypointConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract oracle waypoint positions from GT future, using only geometry.

    Returns:
      idx_in_future: (K,) indices in [0, F-2] (relative to future_pos)
      waypoints: (K,2) positions in grid coordinates.
    """
    start_pos = np.asarray(start_pos, dtype=np.float32).reshape(2)
    future_pos = np.asarray(future_pos, dtype=np.float32)
    if future_pos.ndim != 2 or future_pos.shape[1] != 2:
        raise ValueError(f"Expected future_pos (F,2), got {future_pos.shape}")
    F = int(future_pos.shape[0])
    k = int(cfg.num_waypoints)
    if k <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
    if F < 2:
        return np.zeros((k,), dtype=np.int64), np.repeat(future_pos[:1], repeats=k, axis=0).astype(np.float32)

    if str(cfg.mode) != "rdp_dev":
        raise ValueError(f"Unknown waypoint mode: {cfg.mode}")

    poly = np.concatenate([start_pos[None, :], future_pos], axis=0)  # (F+1,2)
    idx_poly = pick_waypoint_indices_rdp_fixed_k(poly, k=k)  # indices in [1, F-1]
    if idx_poly.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
    idx_future = np.clip(idx_poly - 1, 0, max(F - 2, 0)).astype(np.int64)
    waypoints = future_pos[idx_future].astype(np.float32, copy=False)
    return idx_future.astype(np.int64, copy=False), waypoints
