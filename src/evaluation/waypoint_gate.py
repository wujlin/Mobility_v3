from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    from scipy import ndimage
    from scipy.interpolate import CubicSpline, PchipInterpolator
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "该脚本依赖 scipy（仅 CPU）：请在你的 conda 环境中安装 scipy 后再运行。"
    ) from e


WaypointMode = Literal["time", "arclen", "max_dev", "rdp_dev", "max_turn", "random"]
SkeletonKind = Literal["linear", "pchip", "cubic"]


@dataclass(frozen=True)
class GateConfig:
    count_thr: float
    dilate: int
    close: int
    index_mode: Literal["round", "floor"]
    sample_step: float
    max_samples_per_segment: int
    corner_sigma: float
    corner_k: float
    corner_pctl: float
    corner_nms: int


def _load_samples_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "targets" not in data.files or "start_pos" not in data.files:
        raise ValueError(f"Bad samples.npz: require keys ['targets','start_pos'], got {data.files}")
    targets = np.asarray(data["targets"], dtype=np.float32)  # (N,F,2) [y,x]
    start_pos = np.asarray(data["start_pos"], dtype=np.float32)  # (N,2) [y,x]
    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError(f"Expected targets (N,F,2), got {targets.shape}")
    if start_pos.ndim != 2 or start_pos.shape[-1] != 2:
        raise ValueError(f"Expected start_pos (N,2), got {start_pos.shape}")
    if targets.shape[0] != start_pos.shape[0]:
        raise ValueError(f"N mismatch: targets N={targets.shape[0]} vs start_pos N={start_pos.shape[0]}")
    return targets, start_pos


def _load_nav_count(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "count" not in data.files:
        raise ValueError(f"Bad nav_file: require key 'count', got {data.files}")
    count = np.asarray(data["count"], dtype=np.float32)  # (H,W)
    if count.ndim != 2:
        raise ValueError(f"Expected count (H,W), got {count.shape}")
    return count


def _build_drivable_mask(count: np.ndarray, *, cfg: GateConfig) -> np.ndarray:
    drivable = np.asarray(count >= float(cfg.count_thr), dtype=bool)
    if int(cfg.close) > 0:
        k = int(cfg.close)
        structure = np.ones((2 * k + 1, 2 * k + 1), dtype=bool)
        drivable = ndimage.binary_closing(drivable, structure=structure)
    if int(cfg.dilate) > 0:
        k = int(cfg.dilate)
        structure = np.ones((2 * k + 1, 2 * k + 1), dtype=bool)
        drivable = ndimage.binary_dilation(drivable, structure=structure)
    return np.asarray(drivable, dtype=bool)


def _index_points(pos: np.ndarray, *, cfg: GateConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map float grid positions to integer cell indices.

    Returns:
        y_idx, x_idx: int arrays
        in_bounds: bool array
    """
    if cfg.index_mode == "round":
        yy = np.rint(pos[..., 0]).astype(np.int64)
        xx = np.rint(pos[..., 1]).astype(np.int64)
    elif cfg.index_mode == "floor":
        yy = np.floor(pos[..., 0]).astype(np.int64)
        xx = np.floor(pos[..., 1]).astype(np.int64)
    else:
        raise ValueError(f"Unknown index_mode={cfg.index_mode}")

    return yy, xx, np.ones_like(yy, dtype=bool)  # bounds checked by caller


def _sample_linear_segment(
    a: np.ndarray,  # (N,2)
    b: np.ndarray,  # (N,2)
    *,
    sample_step: float,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized sampling for N line segments.

    Returns:
        pts: (N, M, 2)
        valid: (N, M) bool mask indicating which samples are valid per segment.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[1] != 2:
        raise ValueError(f"Expected a,b as (N,2), got {a.shape} and {b.shape}")

    d = b - a  # (N,2)
    seg_len = np.linalg.norm(d, axis=1)  # (N,)
    n = np.ceil(seg_len / max(float(sample_step), 1e-6)).astype(np.int64) + 1
    n = np.clip(n, 2, int(max_samples))
    m = int(np.max(n)) if n.size else 0
    if m <= 0:
        return np.zeros((a.shape[0], 0, 2), dtype=np.float32), np.zeros((a.shape[0], 0), dtype=bool)

    t = np.linspace(0.0, 1.0, num=m, dtype=np.float32)  # (M,)
    pts = a[:, None, :] + t[None, :, None] * d[:, None, :]  # (N,M,2)
    valid = (np.arange(m, dtype=np.int64)[None, :] < n[:, None])
    return pts, valid


def _segment_collision_stats(
    a: np.ndarray,
    b: np.ndarray,
    *,
    drivable: np.ndarray,
    cfg: GateConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collision stats for N segments (vectorized).

    Returns:
        collided_any: (N,) bool
        coll_points: (N,) int
        total_points: (N,) int
    """
    pts, valid = _sample_linear_segment(
        a,
        b,
        sample_step=float(cfg.sample_step),
        max_samples=int(cfg.max_samples_per_segment),
    )
    if pts.shape[1] == 0:
        n = int(a.shape[0])
        return np.zeros((n,), dtype=bool), np.zeros((n,), dtype=np.int64), np.zeros((n,), dtype=np.int64)

    H, W = int(drivable.shape[0]), int(drivable.shape[1])
    yy, xx, _ = _index_points(pts, cfg=cfg)
    in_bounds = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)

    ok = np.zeros_like(valid, dtype=bool)
    if np.any(in_bounds & valid):
        yi = np.clip(yy, 0, H - 1)
        xi = np.clip(xx, 0, W - 1)
        ok = drivable[yi, xi]

    bad = valid & (~in_bounds | ~ok)
    collided_any = np.any(bad, axis=1)
    coll_points = np.sum(bad, axis=1).astype(np.int64)
    total_points = np.sum(valid, axis=1).astype(np.int64)
    return collided_any, coll_points, total_points


def _build_polyline_vertices(
    *,
    start_pos: np.ndarray,  # (N,2)
    end_pos: np.ndarray,  # (N,2)
    waypoints: np.ndarray,  # (N,K,2)
) -> np.ndarray:
    start_pos = np.asarray(start_pos, dtype=np.float32)
    end_pos = np.asarray(end_pos, dtype=np.float32)
    waypoints = np.asarray(waypoints, dtype=np.float32)
    if waypoints.ndim != 3 or waypoints.shape[-1] != 2:
        raise ValueError(f"Expected waypoints (N,K,2), got {waypoints.shape}")
    if start_pos.shape != (waypoints.shape[0], 2) or end_pos.shape != (waypoints.shape[0], 2):
        raise ValueError("N mismatch among start/end/waypoints")
    return np.concatenate([start_pos[:, None, :], waypoints, end_pos[:, None, :]], axis=1)  # (N,K+2,2)


def _collision_rate_for_polyline(
    vertices: np.ndarray,  # (N,M,2)
    *,
    drivable: np.ndarray,
    cfg: GateConfig,
) -> Dict[str, float]:
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 3 or vertices.shape[-1] != 2:
        raise ValueError(f"Expected vertices (N,M,2), got {vertices.shape}")
    N, M = int(vertices.shape[0]), int(vertices.shape[1])
    if M < 2:
        return {"collision_rate_any": 0.0, "collision_point_rate": 0.0}

    any_coll = np.zeros((N,), dtype=bool)
    coll_pts = np.zeros((N,), dtype=np.int64)
    tot_pts = np.zeros((N,), dtype=np.int64)
    for j in range(M - 1):
        collided, cpts, tpts = _segment_collision_stats(vertices[:, j], vertices[:, j + 1], drivable=drivable, cfg=cfg)
        any_coll |= collided
        coll_pts += cpts
        tot_pts += tpts

    collision_rate_any = float(np.mean(any_coll)) if N > 0 else 0.0
    collision_point_rate = float(np.sum(coll_pts) / max(int(np.sum(tot_pts)), 1)) if N > 0 else 0.0
    return {"collision_rate_any": collision_rate_any, "collision_point_rate": collision_point_rate}


def _spline_points_through_vertices(
    vertices_1d: np.ndarray,  # (M,)
    t: np.ndarray,  # (M,)
    t_query: np.ndarray,
    *,
    kind: SkeletonKind,
) -> np.ndarray:
    if kind == "pchip":
        f = PchipInterpolator(t, vertices_1d, extrapolate=True)
        return f(t_query).astype(np.float32)
    if kind == "cubic":
        f = CubicSpline(t, vertices_1d, bc_type="natural", extrapolate=True)
        return f(t_query).astype(np.float32)
    raise ValueError(f"Unknown spline kind: {kind}")


def _collision_rate_for_spline(
    vertices: np.ndarray,  # (N,M,2)
    *,
    drivable: np.ndarray,
    cfg: GateConfig,
    kind: SkeletonKind,
    max_n: Optional[int] = None,
) -> Dict[str, float]:
    """
    Spline collision is computed per-trajectory (not fully vectorized).
    This is intended as a diagnostic complement to the linear check.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    N, M = int(vertices.shape[0]), int(vertices.shape[1])
    H, W = int(drivable.shape[0]), int(drivable.shape[1])
    if M < 2 or N == 0:
        return {"collision_rate_any": 0.0, "collision_point_rate": 0.0}

    take = N if max_n is None else min(N, int(max_n))
    any_coll: List[bool] = []
    coll_pts: List[int] = []
    tot_pts: List[int] = []

    for i in range(take):
        v = vertices[i]  # (M,2)
        # chord-length parameterization
        seg = v[1:] - v[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        t_nodes = np.concatenate([[0.0], np.cumsum(seg_len)], axis=0).astype(np.float32)
        # Ensure strictly increasing nodes required by PCHIP/CubicSpline.
        # Zero-length edges can happen when the trajectory lingers (repeated positions),
        # which would make t_nodes non-strictly increasing and break spline fitting.
        if t_nodes.size >= 2:
            t_keep: List[float] = [float(t_nodes[0])]
            v_keep: List[np.ndarray] = [v[0]]
            last = float(t_nodes[0])
            for j in range(1, int(t_nodes.size)):
                tj = float(t_nodes[j])
                if tj > last + 1e-6:
                    t_keep.append(tj)
                    v_keep.append(v[j])
                    last = tj
            v = np.stack(v_keep, axis=0).astype(np.float32)
            t_nodes = np.asarray(t_keep, dtype=np.float32)
        total = float(t_nodes[-1])
        if not np.isfinite(total) or total <= 1e-6:
            any_coll.append(False)
            coll_pts.append(0)
            tot_pts.append(0)
            continue

        n = int(np.ceil(total / max(float(cfg.sample_step), 1e-6))) + 1
        n = int(np.clip(n, 2, int(cfg.max_samples_per_segment) * max(M - 1, 1)))
        t_query = np.linspace(0.0, total, num=n, dtype=np.float32)
        y = _spline_points_through_vertices(v[:, 0], t_nodes, t_query, kind=kind)
        x = _spline_points_through_vertices(v[:, 1], t_nodes, t_query, kind=kind)
        pts = np.stack([y, x], axis=1)  # (n,2)

        yy, xx, _ = _index_points(pts, cfg=cfg)
        in_bounds = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        yi = np.clip(yy, 0, H - 1)
        xi = np.clip(xx, 0, W - 1)
        ok = drivable[yi, xi] & in_bounds
        bad = ~ok
        any_coll.append(bool(np.any(bad)))
        coll_pts.append(int(np.sum(bad)))
        tot_pts.append(int(bad.size))

    collision_rate_any = float(np.mean(any_coll)) if take > 0 else 0.0
    collision_point_rate = float(np.sum(coll_pts) / max(int(np.sum(tot_pts)), 1)) if take > 0 else 0.0
    return {"collision_rate_any": collision_rate_any, "collision_point_rate": collision_point_rate}


def _distance_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Distance from point(s) p to segment a-b.

    Shapes:
        p: (...,2)
        a,b: (...,2) broadcastable to p
    """
    p = np.asarray(p, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = b - a
    ap = p - a
    ab2 = np.sum(ab * ab, axis=-1, keepdims=True)  # (...,1)
    # handle degenerate segments
    ab2 = np.where(ab2 > 1e-8, ab2, 1e-8)
    t = np.sum(ap * ab, axis=-1, keepdims=True) / ab2
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    d = np.linalg.norm(p - proj, axis=-1)
    return d.astype(np.float32)


def _pick_waypoint_indices_time(F: int, *, K: int) -> List[int]:
    if int(K) <= 0:
        return []
    idx: List[int] = []
    for i in range(int(K)):
        frac = float(i + 1) / float(K + 1)
        j = int(np.rint(frac * float(F - 1)))
        j = int(np.clip(j, 1, max(F - 2, 1)))
        idx.append(j)
    # ensure strictly increasing (rare ties when F small)
    idx = sorted(set(idx))
    # if de-dup shrinks, fill greedily
    cand = [j for j in range(1, max(F - 1, 1))]
    for j in cand:
        if len(idx) >= int(K):
            break
        if j not in idx and j != 0 and j != F - 1:
            idx.append(j)
    return idx[: int(K)]


def _pick_waypoints(
    gt: np.ndarray,  # (N,F,2)
    start_pos: np.ndarray,  # (N,2)
    *,
    mode: WaypointMode,
    K: int,
    seed: int,
    min_sep: int,
    turn_min_speed: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        idx: (N,K) int indices into F
        waypoints: (N,K,2) positions
    """
    gt = np.asarray(gt, dtype=np.float32)
    start_pos = np.asarray(start_pos, dtype=np.float32)
    N, F = int(gt.shape[0]), int(gt.shape[1])
    if int(K) <= 0:
        return np.zeros((N, 0), dtype=np.int64), np.zeros((N, 0, 2), dtype=np.float32)

    end_pos = gt[:, -1, :]  # (N,2) horizon end as the segment destination
    internal = gt[:, 1:-1, :]  # (N,F-2,2)

    if mode == "time":
        idx_list = _pick_waypoint_indices_time(F, K=int(K))
        idx = np.tile(np.asarray(idx_list, dtype=np.int64)[None, :], (N, 1))
        wp = gt[np.arange(N)[:, None], idx]
        return idx, wp

    if mode == "arclen":
        # Select waypoints by arc-length quantiles (more robust than time indices under speed variation).
        poly = np.concatenate([start_pos[:, None, :], gt], axis=1)  # (N,F+1,2)
        seg = poly[:, 1:, :] - poly[:, :-1, :]
        seg_len = np.linalg.norm(seg, axis=-1)  # (N,F)
        s = np.cumsum(seg_len, axis=1)  # distance to each GT point (N,F)
        total = s[:, -1:]  # (N,1)
        # Avoid degenerate paths.
        total = np.where(total > 1e-6, total, 1e-6)
        idx = np.zeros((N, int(K)), dtype=np.int64)
        for kk in range(int(K)):
            frac = float(kk + 1) / float(int(K) + 1)
            tgt = total[:, 0] * frac  # (N,)
            j = np.argmin(np.abs(s - tgt[:, None]), axis=1).astype(np.int64)  # (N,), in [0..F-1]
            # keep away from endpoints (in GT index space: 0..F-1)
            j = np.clip(j, 1, max(F - 2, 1))
            idx[:, kk] = j
        idx = np.sort(idx, axis=1)
        wp = gt[np.arange(N)[:, None], idx]
        return idx, wp

    if mode == "random":
        rng = np.random.default_rng(int(seed))
        cand = np.arange(1, F - 1, dtype=np.int64)  # internal only
        if cand.size == 0:
            idx = np.ones((N, int(K)), dtype=np.int64)
        elif int(K) >= int(cand.size):
            idx = rng.choice(cand, size=(N, int(K)), replace=True)
        else:
            # Per-trajectory sample without replacement (vectorized): sort random scores per row.
            scores = rng.random((N, int(cand.size)), dtype=np.float32)
            order = np.argsort(scores, axis=1)
            idx = cand[order[:, : int(K)]]
        idx = np.sort(idx, axis=1)
        wp = gt[np.arange(N)[:, None], idx]
        return idx, wp

    if mode == "max_dev":
        # scores: distance to the straight segment from start_pos to end_pos
        d = _distance_point_to_segment(internal, start_pos[:, None, :], end_pos[:, None, :])  # (N,F-2)
        # pick top-K with minimum temporal separation (greedy)
        order = np.argsort(-d, axis=1)  # descending
        chosen = np.full((N, int(K)), -1, dtype=np.int64)
        for kk in range(int(K)):
            for rank in range(order.shape[1]):
                cand = order[:, rank] + 1  # shift back to [1..F-2]
                ok = cand >= 1
                for prev in range(kk):
                    ok &= (np.abs(cand - chosen[:, prev]) >= int(min_sep))
                # assign where not yet assigned
                need = chosen[:, kk] < 0
                take = need & ok
                chosen[take, kk] = cand[take]
            # fill any remaining with the best available (even if close)
            need = chosen[:, kk] < 0
            if np.any(need):
                chosen[need, kk] = (order[need, 0] + 1)
        idx = np.sort(chosen, axis=1)
        wp = gt[np.arange(N)[:, None], idx]
        return idx, wp

    if mode == "rdp_dev":
        # Fixed-K RDP-like selection by perpendicular deviation from segment chords.
        # Work on the full polyline indices: 0=start, 1..F are GT points 0..F-1.
        poly = np.concatenate([start_pos[:, None, :], gt], axis=1)  # (N,F+1,2)
        poly_len = int(poly.shape[1])  # F+1
        if poly_len < 3:
            idx = np.ones((N, int(K)), dtype=np.int64)
            wp = gt[np.arange(N)[:, None], idx]
            return idx, wp

        chosen_poly_idx = np.full((N, int(K)), -1, dtype=np.int64)
        for i in range(N):
            # segments are inclusive endpoints in poly index space
            segs: List[Tuple[int, int]] = [(0, poly_len - 1)]
            picks: List[int] = []
            for _ in range(int(K)):
                best = None  # (dev, seg_idx, mid_idx)
                for si, (a_idx, b_idx) in enumerate(segs):
                    if b_idx - a_idx <= 1:
                        continue
                    a = poly[i, a_idx]
                    b = poly[i, b_idx]
                    pts = poly[i, a_idx + 1 : b_idx]  # internal
                    d = _distance_point_to_segment(pts, a[None, :], b[None, :]).reshape(-1)
                    if d.size == 0:
                        continue
                    mid_off = int(np.argmax(d))
                    dev = float(d[mid_off])
                    mid_idx = a_idx + 1 + mid_off
                    if best is None or dev > best[0]:
                        best = (dev, si, mid_idx)
                if best is None:
                    break
                _, si, mid_idx = best
                if mid_idx in picks:
                    break
                picks.append(int(mid_idx))
                a_idx, b_idx = segs.pop(int(si))
                segs.append((a_idx, mid_idx))
                segs.append((mid_idx, b_idx))
            # map poly indices back to GT indices: poly idx 1..F-1 -> GT idx 0..F-2
            gt_idx = [p - 1 for p in picks if 1 <= p <= poly_len - 2]
            gt_idx = [int(np.clip(j, 1, max(F - 2, 1))) for j in gt_idx]  # keep away from endpoints
            gt_idx = sorted(set(gt_idx))
            # Fill to K (fall back to arclen if needed)
            if len(gt_idx) < int(K):
                fill = _pick_waypoint_indices_time(F, K=int(K))
                for j in fill:
                    if len(gt_idx) >= int(K):
                        break
                    if j not in gt_idx:
                        gt_idx.append(j)
                gt_idx = sorted(gt_idx)[: int(K)]
            chosen_poly_idx[i, : len(gt_idx)] = np.asarray(gt_idx, dtype=np.int64)

        # Replace any missing with a safe time-based fill (should be rare)
        need = chosen_poly_idx < 0
        if np.any(need):
            fill = _pick_waypoint_indices_time(F, K=int(K))
            chosen_poly_idx[need] = fill[0] if fill else 1
        idx = np.sort(chosen_poly_idx, axis=1)
        wp = gt[np.arange(N)[:, None], idx]
        return idx, wp

    if mode == "max_turn":
        disp = gt[:, 1:, :] - gt[:, :-1, :]  # (N,F-1,2)
        if disp.shape[1] < 2:
            idx = np.ones((N, int(K)), dtype=np.int64)
            wp = gt[np.arange(N)[:, None], idx]
            return idx, wp
        v1 = disp[:, :-1, :]
        v2 = disp[:, 1:, :]
        n1 = np.linalg.norm(v1, axis=-1)
        n2 = np.linalg.norm(v2, axis=-1)
        valid = (n1 > float(turn_min_speed)) & (n2 > float(turn_min_speed))
        dot = np.sum(v1 * v2, axis=-1)
        cos = dot / (n1 * n2 + 1e-8)
        cos = np.clip(cos, -1.0, 1.0)
        ang = np.arccos(cos).astype(np.float32)  # (N,F-2)
        ang[~valid] = -np.inf
        order = np.argsort(-ang, axis=1)
        chosen = np.full((N, int(K)), -1, dtype=np.int64)
        for kk in range(int(K)):
            for rank in range(order.shape[1]):
                cand = order[:, rank] + 1
                ok = cand >= 1
                for prev in range(kk):
                    ok &= (np.abs(cand - chosen[:, prev]) >= int(min_sep))
                need = chosen[:, kk] < 0
                take = need & ok
                chosen[take, kk] = cand[take]
            need = chosen[:, kk] < 0
            if np.any(need):
                chosen[need, kk] = (order[need, 0] + 1)
        idx = np.sort(chosen, axis=1)
        wp = gt[np.arange(N)[:, None], idx]
        return idx, wp

    raise ValueError(f"Unknown waypoint mode: {mode}")


def _compute_corner_map(
    drivable: np.ndarray,
    *,
    cfg: GateConfig,
) -> np.ndarray:
    """
    Approximate "obstacle vertices" via Harris corners on the drivable mask boundary.

    Returns:
        corners: (H,W) bool
    """
    drivable = np.asarray(drivable, dtype=bool)
    boundary = drivable & (~ndimage.binary_erosion(drivable))
    if not np.any(boundary):
        return np.zeros_like(drivable, dtype=bool)

    img = drivable.astype(np.float32)
    img = ndimage.gaussian_filter(img, sigma=float(cfg.corner_sigma))
    Ix = ndimage.sobel(img, axis=1, mode="constant")
    Iy = ndimage.sobel(img, axis=0, mode="constant")
    Ixx = ndimage.gaussian_filter(Ix * Ix, sigma=float(cfg.corner_sigma))
    Iyy = ndimage.gaussian_filter(Iy * Iy, sigma=float(cfg.corner_sigma))
    Ixy = ndimage.gaussian_filter(Ix * Iy, sigma=float(cfg.corner_sigma))
    det = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    R = det - float(cfg.corner_k) * (trace * trace)
    R = np.where(boundary, R, -np.inf)

    vals = R[np.isfinite(R)]
    if vals.size == 0:
        return np.zeros_like(drivable, dtype=bool)
    thr = float(np.percentile(vals, float(cfg.corner_pctl)))
    cand = R >= thr
    nms = int(cfg.corner_nms)
    max_f = ndimage.maximum_filter(R, size=(2 * nms + 1, 2 * nms + 1))
    corners = cand & (R == max_f) & np.isfinite(R)
    return np.asarray(corners, dtype=bool)


def _distance_maps(
    drivable: np.ndarray,
    *,
    cfg: GateConfig,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Returns:
        dist_to_obstacle: (H,W) distance to nearest obstacle cell (0 outside drivable)
        dist_to_corner: (H,W) distance to nearest "obstacle vertex" (corner); fallback to boundary if no corners found.
        num_corners: int
    """
    drivable = np.asarray(drivable, dtype=bool)
    dist_to_obstacle = ndimage.distance_transform_edt(drivable).astype(np.float32)
    corners = _compute_corner_map(drivable, cfg=cfg)
    num_corners = int(np.sum(corners))
    if num_corners == 0:
        # Fallback: use boundary pixels as "vertices" (coarser but always defined).
        boundary = drivable & (~ndimage.binary_erosion(drivable))
        corners = boundary
        num_corners = int(np.sum(corners))

    seeds = np.ones_like(drivable, dtype=np.uint8)
    seeds[corners] = 0
    dist_to_corner = ndimage.distance_transform_edt(seeds).astype(np.float32)
    return dist_to_obstacle, dist_to_corner, int(num_corners)


def _distance_stats_for_waypoints(
    waypoints: np.ndarray,  # (N,K,2)
    *,
    dist_to_obstacle: np.ndarray,
    dist_to_corner: np.ndarray,
    cfg: GateConfig,
) -> Dict[str, float]:
    H, W = int(dist_to_obstacle.shape[0]), int(dist_to_obstacle.shape[1])
    wp = np.asarray(waypoints, dtype=np.float32)
    if wp.size == 0:
        return {
            "median_clearance": float("nan"),
            "median_dist_corner": float("nan"),
            "mean_clearance": float("nan"),
            "mean_dist_corner": float("nan"),
        }

    yy, xx, _ = _index_points(wp, cfg=cfg)
    yy = np.clip(yy, 0, H - 1)
    xx = np.clip(xx, 0, W - 1)
    clearance = dist_to_obstacle[yy, xx].reshape(-1)
    dcorner = dist_to_corner[yy, xx].reshape(-1)
    return {
        "median_clearance": float(np.median(clearance)),
        "median_dist_corner": float(np.median(dcorner)),
        "mean_clearance": float(np.mean(clearance)),
        "mean_dist_corner": float(np.mean(dcorner)),
    }


def _point_biserial_corr(y01: np.ndarray, x: np.ndarray) -> float:
    """
    Correlation between a binary variable y in {0,1} and a continuous variable x.
    """
    y01 = np.asarray(y01, dtype=np.float32).reshape(-1)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if y01.size != x.size or y01.size == 0:
        return float("nan")
    p = float(np.mean(y01))
    if p <= 1e-6 or p >= 1.0 - 1e-6:
        return float("nan")
    x1 = x[y01 > 0.5]
    x0 = x[y01 <= 0.5]
    if x1.size == 0 or x0.size == 0:
        return float("nan")
    s = float(np.std(x))
    if s <= 1e-8:
        return 0.0
    q = 1.0 - p
    r = (float(np.mean(x1)) - float(np.mean(x0))) * np.sqrt(p * q) / s
    return float(r)


def run_gate(
    *,
    samples_npz: Path,
    nav_file: Path,
    waypoint_mode: WaypointMode,
    num_waypoints: int,
    skeletons: List[SkeletonKind],
    seed: int,
    min_sep: int,
    turn_min_speed: float,
    spline_max_n: Optional[int],
    cfg: GateConfig,
    max_n: Optional[int],
) -> Dict[str, object]:
    gt, start_pos = _load_samples_npz(samples_npz)
    if max_n is not None:
        gt = gt[: int(max_n)]
        start_pos = start_pos[: int(max_n)]
    end_pos = gt[:, -1, :]

    count = _load_nav_count(nav_file)
    drivable = _build_drivable_mask(count, cfg=cfg)
    dist_to_obstacle, dist_to_corner, num_corners = _distance_maps(drivable, cfg=cfg)

    idx_wp, wp = _pick_waypoints(
        gt,
        start_pos,
        mode=waypoint_mode,
        K=int(num_waypoints),
        seed=int(seed),
        min_sep=int(min_sep),
        turn_min_speed=float(turn_min_speed),
    )

    vertices = _build_polyline_vertices(start_pos=start_pos, end_pos=end_pos, waypoints=wp)

    out: Dict[str, object] = {
        "inputs": {"samples_npz": str(samples_npz), "nav_file": str(nav_file)},
        "config": {
            "waypoint_mode": str(waypoint_mode),
            "num_waypoints": int(num_waypoints),
            "skeletons": list(skeletons),
            "seed": int(seed),
            "min_sep": int(min_sep),
            "turn_min_speed": float(turn_min_speed),
            "count_thr": float(cfg.count_thr),
            "dilate": int(cfg.dilate),
            "close": int(cfg.close),
            "index_mode": str(cfg.index_mode),
            "sample_step": float(cfg.sample_step),
            "max_samples_per_segment": int(cfg.max_samples_per_segment),
            "corner_sigma": float(cfg.corner_sigma),
            "corner_k": float(cfg.corner_k),
            "corner_pctl": float(cfg.corner_pctl),
            "corner_nms": int(cfg.corner_nms),
            "spline_max_n": (int(spline_max_n) if spline_max_n is not None else None),
        },
        "stats": {
            "N": int(gt.shape[0]),
            "F": int(gt.shape[1]),
            "drivable_ratio": float(np.mean(drivable)),
            "num_corners": int(num_corners),
        },
        "results": {},
    }

    # ---- Validity check: collision rate ----
    res: Dict[str, Dict[str, float]] = {}
    for sk in skeletons:
        if sk == "linear":
            res[sk] = _collision_rate_for_polyline(vertices, drivable=drivable, cfg=cfg)
        else:
            res[sk] = _collision_rate_for_spline(
                vertices, drivable=drivable, cfg=cfg, kind=sk, max_n=spline_max_n
            )
    out["results"]["collision"] = res

    # ---- Learnability check: distance to obstacle/corner ----
    dist_stats = _distance_stats_for_waypoints(wp, dist_to_obstacle=dist_to_obstacle, dist_to_corner=dist_to_corner, cfg=cfg)
    out["results"]["waypoint_distance"] = dist_stats

    # Compare against time-uniform waypoints as a "no-geometry" baseline.
    idx_time, wp_time = _pick_waypoints(
        gt,
        start_pos,
        mode="time",
        K=int(num_waypoints),
        seed=int(seed),
        min_sep=int(min_sep),
        turn_min_speed=float(turn_min_speed),
    )
    dist_time = _distance_stats_for_waypoints(
        wp_time, dist_to_obstacle=dist_to_obstacle, dist_to_corner=dist_to_corner, cfg=cfg
    )
    out["results"]["baseline_time_distance"] = dist_time

    # Random baseline (same K).
    idx_rand, wp_rand = _pick_waypoints(
        gt,
        start_pos,
        mode="random",
        K=int(num_waypoints),
        seed=int(seed),
        min_sep=int(min_sep),
        turn_min_speed=float(turn_min_speed),
    )
    dist_rand = _distance_stats_for_waypoints(
        wp_rand, dist_to_obstacle=dist_to_obstacle, dist_to_corner=dist_to_corner, cfg=cfg
    )
    out["results"]["baseline_random_distance"] = dist_rand

    # Correlation: waypoint indicator vs (-distance to corner) over all internal timesteps.
    # If waypoints concentrate near geometric constraints, corr should be positive.
    H, W = int(dist_to_corner.shape[0]), int(dist_to_corner.shape[1])
    yy, xx, _ = _index_points(gt, cfg=cfg)
    yy = np.clip(yy, 0, H - 1)
    xx = np.clip(xx, 0, W - 1)
    dcorner_seq = dist_to_corner[yy, xx]  # (N,F)
    # focus on internal steps only to avoid trivial endpoints.
    internal_mask = np.zeros_like(dcorner_seq, dtype=bool)
    internal_mask[:, 1:-1] = True

    y = np.zeros_like(dcorner_seq, dtype=np.float32)
    rows = np.arange(int(gt.shape[0]), dtype=np.int64)[:, None]
    y[rows, idx_wp] = 1.0
    x = (-dcorner_seq).astype(np.float32)
    y_flat = y[internal_mask].reshape(-1)
    x_flat = x[internal_mask].reshape(-1)
    out["results"]["corner_corr_point_biserial"] = _point_biserial_corr(y_flat, x_flat)

    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hierarchical waypoint validity gate: collision + learnability checks (CPU-only).")
    p.add_argument("--samples_npz", type=str, required=True, help="samples.npz containing GT 'targets' and 'start_pos' (positions in grid space).")
    p.add_argument("--nav_file", type=str, required=True, help="nav_field.npz (train-only) containing 'count' used to derive a drivable mask.")

    p.add_argument(
        "--waypoint_mode",
        type=str,
        default="max_dev",
        choices=["time", "arclen", "max_dev", "rdp_dev", "max_turn"],
        help="How to extract GT waypoints for the skeleton.",
    )
    p.add_argument("--num_waypoints", type=int, default=1, help="Number of waypoints used in the skeleton (small K is the point of the gate).")
    p.add_argument("--min_sep", type=int, default=2, help="Minimum temporal separation (in steps) when selecting multiple waypoints.")
    p.add_argument("--turn_min_speed", type=float, default=0.1, help="Min speed threshold for max_turn waypoint mode (grid/step).")

    p.add_argument("--count_thr", type=float, default=1.0, help="Drivable mask threshold: count >= thr is drivable.")
    p.add_argument("--close", type=int, default=0, help="Optional binary closing radius (cells) to fill small holes in drivable mask.")
    p.add_argument("--dilate", type=int, default=0, help="Optional dilation radius (cells) to make the drivable mask more permissive.")
    p.add_argument("--index_mode", type=str, default="round", choices=["round", "floor"], help="How to map float positions to grid cells.")

    p.add_argument("--sample_step", type=float, default=0.5, help="Sampling step along skeleton segments (grid units).")
    p.add_argument("--max_samples_per_segment", type=int, default=256, help="Cap samples per segment to avoid huge allocations.")

    p.add_argument("--skeleton", type=str, default="linear", choices=["linear", "pchip", "cubic", "linear+pchip", "linear+cubic", "all"], help="Skeleton type(s) to collision-check.")
    p.add_argument("--spline_max_n", type=int, default=2000, help="When skeleton includes spline, limit to first N trajectories (spline check is per-trajectory).")

    p.add_argument("--corner_sigma", type=float, default=1.6, help="Gaussian sigma for corner detection.")
    p.add_argument("--corner_k", type=float, default=0.04, help="Harris k parameter.")
    p.add_argument("--corner_pctl", type=float, default=99.7, help="Percentile threshold for corner response on boundary.")
    p.add_argument("--corner_nms", type=int, default=2, help="Non-maximum suppression radius for corner detection.")

    p.add_argument("--seed", type=int, default=0, help="RNG seed (random baseline).")
    p.add_argument("--max_n", type=int, default=None, help="Optional limit on number of trajectories from samples.npz.")

    p.add_argument("--out_json", type=str, default=None, help="Optional output path to save the gate report JSON.")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    skeleton_arg = str(args.skeleton)
    if skeleton_arg == "linear":
        skeletons: List[SkeletonKind] = ["linear"]
    elif skeleton_arg == "pchip":
        skeletons = ["pchip"]
    elif skeleton_arg == "cubic":
        skeletons = ["cubic"]
    elif skeleton_arg == "linear+pchip":
        skeletons = ["linear", "pchip"]
    elif skeleton_arg == "linear+cubic":
        skeletons = ["linear", "cubic"]
    elif skeleton_arg == "all":
        skeletons = ["linear", "pchip", "cubic"]
    else:
        raise ValueError(f"Unknown --skeleton {skeleton_arg}")

    cfg = GateConfig(
        count_thr=float(args.count_thr),
        dilate=int(args.dilate),
        close=int(args.close),
        index_mode=str(args.index_mode),
        sample_step=float(args.sample_step),
        max_samples_per_segment=int(args.max_samples_per_segment),
        corner_sigma=float(args.corner_sigma),
        corner_k=float(args.corner_k),
        corner_pctl=float(args.corner_pctl),
        corner_nms=int(args.corner_nms),
    )

    report = run_gate(
        samples_npz=Path(args.samples_npz),
        nav_file=Path(args.nav_file),
        waypoint_mode=str(args.waypoint_mode),  # type: ignore[arg-type]
        num_waypoints=int(args.num_waypoints),
        skeletons=skeletons,
        seed=int(args.seed),
        min_sep=int(args.min_sep),
        turn_min_speed=float(args.turn_min_speed),
        spline_max_n=int(args.spline_max_n) if "pchip" in skeletons or "cubic" in skeletons else None,
        cfg=cfg,
        max_n=int(args.max_n) if args.max_n is not None else None,
    )

    print("[OK] Waypoint gate report")
    print(json.dumps(report["stats"], indent=2))
    print(json.dumps(report["results"], indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
