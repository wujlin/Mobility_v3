from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # optional
    pa = None
    pq = None

from src.evaluation.build_physical_reference_segments import AStarCfg, _astar_soft, _load_grid_from_osm_meta


@dataclass(frozen=True)
class TemplateCfg:
    dist_bins_m: Tuple[float, ...]
    dir_bins: int
    waypoint_fracs: Tuple[float, ...]
    min_bin_samples: int


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _mode_cluster_mask(v: np.ndarray, *, iters: int = 10) -> np.ndarray:
    """
    Return a boolean mask selecting the dominant cluster (mode) using a tiny 2-means.

    v: (N, D) float array.
    """
    x = np.asarray(v, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return np.ones((x.shape[0],), dtype=bool)

    # Deterministic init: pick two farthest points via a 2-step heuristic.
    c0 = x[0]
    d0 = np.sum((x - c0) ** 2, axis=1)
    i1 = int(np.argmax(d0))
    c1 = x[i1]
    d1 = np.sum((x - c1) ** 2, axis=1)
    i0 = int(np.argmax(d1))
    c0 = x[i0]

    for _ in range(int(iters)):
        d0 = np.sum((x - c0) ** 2, axis=1)
        d1 = np.sum((x - c1) ** 2, axis=1)
        a0 = d0 <= d1
        a1 = ~a0
        if not np.any(a0) or not np.any(a1):
            break
        c0 = np.mean(x[a0], axis=0)
        c1 = np.mean(x[a1], axis=0)

    # Pick the larger cluster as the mode.
    d0 = np.sum((x - c0) ** 2, axis=1)
    d1 = np.sum((x - c1) ** 2, axis=1)
    a0 = d0 <= d1
    if int(np.sum(a0)) >= int(np.sum(~a0)):
        return a0.astype(bool)
    return (~a0).astype(bool)


def _pick_medoid(stack: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Pick a representative *observed* template from stack (a medoid-like selection).

    Why medoid instead of median?
    - The median in waypoint space can lie between corridors (an interpolation),
      which turns an intended "mode" template into a diffuse support when mapped
      back to the city.
    - Picking an actual member of the dominant cluster keeps expected as "most
      likely corridor" rather than "average of corridors".

    stack: (N, M, 2)
    mask: optional boolean mask selecting a subset (e.g., dominant mode cluster).
    Returns: (M, 2) template from stack.
    """
    s = np.asarray(stack, dtype=np.float64)
    if s.ndim != 3 or s.shape[0] < 1:
        return np.zeros((0, 2), dtype=np.float32)
    if mask is None:
        idx = np.arange(int(s.shape[0]), dtype=np.int64)
    else:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        if m.size != int(s.shape[0]):
            idx = np.arange(int(s.shape[0]), dtype=np.int64)
        else:
            idx = np.where(m)[0].astype(np.int64)
            if idx.size == 0:
                idx = np.arange(int(s.shape[0]), dtype=np.int64)

    v = s[idx].reshape(int(idx.size), -1)  # (K, D)
    # Robust center in waypoint space; then pick the closest observed sample.
    center = np.median(v, axis=0, keepdims=True)  # (1, D)
    d2 = np.sum((v - center) ** 2, axis=1)
    j = int(idx[int(np.argmin(d2))])
    return np.asarray(stack[j], dtype=np.float32)


def _iter_segments_xy(parquet_path: Path, *, max_segments: int) -> Iterable[Tuple[str, np.ndarray, np.ndarray]]:
    if pq is None:
        raise SystemExit("pyarrow is required. Install: pip/conda install pyarrow")
    pf = pq.ParquetFile(str(parquet_path))
    cols = ["traj_csv", "y", "x"]
    scanned = 0
    for batch in pf.iter_batches(batch_size=128, columns=cols):
        d = batch.to_pydict()
        n = len(d["traj_csv"])
        for i in range(n):
            scanned += 1
            if max_segments and scanned > int(max_segments):
                return
            k = str(d["traj_csv"][i])
            y = np.asarray(d["y"][i], dtype=np.int64).reshape(-1)
            x = np.asarray(d["x"][i], dtype=np.int64).reshape(-1)
            if y.size < 2 or x.size < 2:
                continue
            yield k, y, x


def _chord_len_dir(
    y: np.ndarray,
    x: np.ndarray,
    *,
    res_y_m: float,
    res_x_m: float,
) -> Tuple[float, float]:
    dy_m = float(y[-1] - y[0]) * float(res_y_m)
    dx_m = float(x[-1] - x[0]) * float(res_x_m)
    chord_m = float(math.hypot(dy_m, dx_m))
    theta = float(math.atan2(dy_m, dx_m))  # [-pi, pi]
    return chord_m, theta


def _dist_bin_id(chord_m: float, edges: Tuple[float, ...]) -> int:
    # edges: (e0, e1, ..., eK) -> K bins
    if chord_m < float(edges[0]):
        return 0
    for i in range(len(edges) - 1):
        if float(edges[i]) <= chord_m < float(edges[i + 1]):
            return int(i)
    return int(len(edges) - 2)


def _dir_bin_id(theta: float, n: int) -> int:
    # Map [-pi, pi) -> [0, 2pi)
    t = float(theta)
    if t < -math.pi:
        t += 2.0 * math.pi
    if t >= math.pi:
        t -= 2.0 * math.pi
    u = (t + math.pi) / (2.0 * math.pi)  # [0,1)
    b = int(math.floor(u * float(n)))
    return int(min(int(n - 1), max(0, b)))


def _cell_bin_id(v: int, *, n: int, bins: int) -> int:
    """Map v in [0,n) to bin id in [0,bins)."""
    if int(bins) <= 1:
        return 0
    nn = int(max(1, n))
    vv = int(min(max(int(v), 0), nn - 1))
    return int(min(int(bins - 1), max(0, (vv * int(bins)) // nn)))


def _od_key(y0: int, x0: int, y1: int, x1: int, *, H: int, W: int, od_bins: int) -> Tuple[int, int]:
    """Return (o_id, d_id) where each is in [0, od_bins^2)."""
    oy = _cell_bin_id(int(y0), n=int(H), bins=int(od_bins))
    ox = _cell_bin_id(int(x0), n=int(W), bins=int(od_bins))
    dy = _cell_bin_id(int(y1), n=int(H), bins=int(od_bins))
    dx = _cell_bin_id(int(x1), n=int(W), bins=int(od_bins))
    o_id = int(oy * int(od_bins) + ox)
    d_id = int(dy * int(od_bins) + dx)
    return int(o_id), int(d_id)


def _sample_point_by_arclen(
    y_m: np.ndarray,
    x_m: np.ndarray,
    frac: float,
) -> Tuple[float, float]:
    if y_m.size < 2:
        return float(y_m[0]), float(x_m[0])
    dy = np.diff(y_m)
    dx = np.diff(x_m)
    seg = np.hypot(dy, dx)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(s[-1])
    if not np.isfinite(total) or total <= 1e-6:
        return float(y_m[0]), float(x_m[0])
    target = float(np.clip(frac, 0.0, 1.0)) * total
    j = int(np.searchsorted(s, target, side="right") - 1)
    j = int(min(max(j, 0), int(y_m.size - 2)))
    denom = float(s[j + 1] - s[j])
    t = 0.0 if denom <= 1e-9 else float((target - float(s[j])) / denom)
    y = float(y_m[j] * (1.0 - t) + y_m[j + 1] * t)
    x = float(x_m[j] * (1.0 - t) + x_m[j + 1] * t)
    return y, x


def _rotate(x: float, y: float, theta: float) -> Tuple[float, float]:
    c = float(math.cos(theta))
    s = float(math.sin(theta))
    xr = c * x - s * y
    yr = s * x + c * y
    return float(xr), float(yr)


def _build_templates(
    source_segments_parquet: Path,
    *,
    res_y_m: float,
    res_x_m: float,
    cfg: TemplateCfg,
    max_segments: int,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], int], Dict[Tuple[int, int], float]]:
    # templates[(dist_bin, dir_bin)] -> (M, 2) normalized waypoints in (x,y) where end is at (1,0).
    accum: Dict[Tuple[int, int], List[np.ndarray]] = {}
    counts: Dict[Tuple[int, int], int] = {}

    for _, y, x in _iter_segments_xy(source_segments_parquet, max_segments=max_segments):
        chord_m, theta = _chord_len_dir(y, x, res_y_m=res_y_m, res_x_m=res_x_m)
        if not np.isfinite(chord_m) or chord_m <= 1e-3:
            continue
        db = _dist_bin_id(float(chord_m), cfg.dist_bins_m)
        hb = _dir_bin_id(float(theta), int(cfg.dir_bins))

        # Metric coords (x east, y south)
        x_m = x.astype(np.float64) * float(res_x_m)
        y_m = y.astype(np.float64) * float(res_y_m)
        x0 = float(x_m[0])
        y0 = float(y_m[0])

        wps: List[List[float]] = []
        for f in cfg.waypoint_fracs:
            yy, xx = _sample_point_by_arclen(y_m, x_m, float(f))
            dx = float(xx - x0) / float(chord_m)
            dy = float(yy - y0) / float(chord_m)
            # Rotate into start-end frame
            xr, yr = _rotate(dx, dy, -float(theta))
            wps.append([float(xr), float(yr)])
        w = np.asarray(wps, dtype=np.float32)

        k = (int(db), int(hb))
        accum.setdefault(k, []).append(w)
        counts[k] = int(counts.get(k, 0) + 1)

    templates: Dict[Tuple[int, int], np.ndarray] = {}
    mode_frac: Dict[Tuple[int, int], float] = {}
    for k, items in accum.items():
        stack = np.stack(items, axis=0)  # (N, M, 2)
        n = int(stack.shape[0])
        # If a bin is multi-modal, a pointwise median can sit "between corridors" and inflate expected support.
        # We approximate a "mode" template by selecting the dominant cluster in waypoint space, then picking a
        # representative *observed* member (medoid-like) from that cluster.
        if n >= max(2 * int(cfg.min_bin_samples), 10):
            v = stack.reshape(n, -1)  # (N, 2M)
            m = _mode_cluster_mask(v, iters=10)
            m_cnt = int(np.sum(m))
            if m_cnt >= 2:
                templates[k] = _pick_medoid(stack, m)
                mode_frac[k] = float(m_cnt / max(1, n))
                continue
        templates[k] = _pick_medoid(stack, None)
        mode_frac[k] = 1.0
    return templates, counts, mode_frac


def _build_od_templates(
    source_segments_parquet: Path,
    *,
    H: int,
    W: int,
    res_y_m: float,
    res_x_m: float,
    cfg: TemplateCfg,
    od_bins: int,
    min_od_samples: int,
    max_segments: int,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], int], Dict[Tuple[int, int], float]]:
    # templates[(o_id, d_id)] -> (M, 2) normalized waypoints in (x,y) where end is at (1,0).
    accum: Dict[Tuple[int, int], List[np.ndarray]] = {}
    counts: Dict[Tuple[int, int], int] = {}

    for _, y, x in _iter_segments_xy(source_segments_parquet, max_segments=max_segments):
        y0, x0, y1, x1 = int(y[0]), int(x[0]), int(y[-1]), int(x[-1])
        o_id, d_id = _od_key(y0, x0, y1, x1, H=int(H), W=int(W), od_bins=int(od_bins))

        chord_m, theta = _chord_len_dir(y, x, res_y_m=res_y_m, res_x_m=res_x_m)
        if not np.isfinite(chord_m) or chord_m <= 1e-3:
            continue

        # Metric coords (x east, y south)
        x_m = x.astype(np.float64) * float(res_x_m)
        y_m = y.astype(np.float64) * float(res_y_m)
        x0_m = float(x_m[0])
        y0_m = float(y_m[0])

        wps: List[List[float]] = []
        for f in cfg.waypoint_fracs:
            yy, xx = _sample_point_by_arclen(y_m, x_m, float(f))
            dx = float(xx - x0_m) / float(chord_m)
            dy = float(yy - y0_m) / float(chord_m)
            xr, yr = _rotate(dx, dy, -float(theta))
            wps.append([float(xr), float(yr)])
        w = np.asarray(wps, dtype=np.float32)

        k = (int(o_id), int(d_id))
        accum.setdefault(k, []).append(w)
        counts[k] = int(counts.get(k, 0) + 1)

    templates: Dict[Tuple[int, int], np.ndarray] = {}
    mode_frac: Dict[Tuple[int, int], float] = {}
    for k, items in accum.items():
        if int(counts.get(k, 0)) < int(min_od_samples):
            continue
        stack = np.stack(items, axis=0)  # (N, M, 2)
        n = int(stack.shape[0])
        if n >= max(2 * int(min_od_samples), 10):
            v = stack.reshape(n, -1)  # (N, 2M)
            m = _mode_cluster_mask(v, iters=10)
            m_cnt = int(np.sum(m))
            if m_cnt >= 2:
                templates[k] = _pick_medoid(stack, m)
                mode_frac[k] = float(m_cnt / max(1, n))
                continue
        templates[k] = _pick_medoid(stack, None)
        mode_frac[k] = 1.0
    return templates, counts, mode_frac


def _pick_template(
    templates: Dict[Tuple[int, int], np.ndarray],
    counts: Dict[Tuple[int, int], int],
    *,
    db: int,
    hb: int,
    cfg: TemplateCfg,
) -> Tuple[np.ndarray, Tuple[int, int], bool]:
    """
    Return (template, chosen_key, used_fallback).
    Fallback searches nearest direction, then nearest distance.
    """
    k0 = (int(db), int(hb))
    if k0 in templates and int(counts.get(k0, 0)) >= int(cfg.min_bin_samples):
        return templates[k0], k0, False

    # 1) nearest direction within same dist bin
    for delta in range(1, int(cfg.dir_bins)):
        for sgn in (-1, 1):
            kk = (int(db), int((hb + sgn * delta) % int(cfg.dir_bins)))
            if kk in templates and int(counts.get(kk, 0)) >= int(cfg.min_bin_samples):
                return templates[kk], kk, True

    # 2) nearest distance, then nearest direction
    n_dist = int(len(cfg.dist_bins_m) - 1)
    for dd in range(1, n_dist):
        for sgn in (-1, 1):
            db2 = int(db + sgn * dd)
            if db2 < 0 or db2 >= n_dist:
                continue
            kk = (int(db2), int(hb))
            if kk in templates and int(counts.get(kk, 0)) >= int(cfg.min_bin_samples):
                return templates[kk], kk, True
            for delta in range(1, int(cfg.dir_bins)):
                for sgn2 in (-1, 1):
                    kk2 = (int(db2), int((hb + sgn2 * delta) % int(cfg.dir_bins)))
                    if kk2 in templates and int(counts.get(kk2, 0)) >= int(cfg.min_bin_samples):
                        return templates[kk2], kk2, True

    # 3) last resort: any template (largest bin by sample count)
    if templates:
        kk_best = max(templates.keys(), key=lambda k: int(counts.get(k, 0)))
        return templates[kk_best], kk_best, True
    return np.zeros((len(cfg.waypoint_fracs), 2), dtype=np.float32), k0, True


def _route_via_waypoints(
    road_prob: np.ndarray,
    *,
    grid,
    astar_cfg: AStarCfg,
    start: Tuple[int, int],
    waypoints: List[Tuple[int, int]],
    end: Tuple[int, int],
) -> Tuple[List[int], List[int], bool]:
    ys: List[int] = []
    xs: List[int] = []
    ok_all = True

    cur_y, cur_x = int(start[0]), int(start[1])
    for wy, wx in waypoints + [end]:
        py, px, meta = _astar_soft(
            road_prob,
            start_y=int(cur_y),
            start_x=int(cur_x),
            end_y=int(wy),
            end_x=int(wx),
            grid=grid,
            cfg=astar_cfg,
        )
        ok = bool(meta.get("success", False))
        if not ok:
            ok_all = False
            py = [int(cur_y), int(wy)]
            px = [int(cur_x), int(wx)]

        if ys:
            py = py[1:]
            px = px[1:]
        ys.extend(int(v) for v in py)
        xs.extend(int(v) for v in px)
        cur_y, cur_x = int(wy), int(wx)

    return ys, xs, bool(ok_all)


def _bresenham_line(y0: int, x0: int, y1: int, x1: int) -> List[Tuple[int, int]]:
    """Integer grid line drawing (Bresenham), inclusive of both endpoints."""
    y0, x0 = int(y0), int(x0)
    y1, x1 = int(y1), int(x1)
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy
    out: List[Tuple[int, int]] = []
    y, x = y0, x0
    while True:
        out.append((int(y), int(x)))
        if y == y1 and x == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return out


def _snap_point_to_road(
    road_prob: np.ndarray, *, y: int, x: int, radius: int, H: int, W: int
) -> Tuple[int, int]:
    r = int(max(0, radius))
    yy = int(min(max(0, int(y)), int(H - 1)))
    xx = int(min(max(0, int(x)), int(W - 1)))
    if r == 0:
        return yy, xx
    y0 = int(max(0, yy - r))
    y1 = int(min(H, yy + r + 1))
    x0 = int(max(0, xx - r))
    x1 = int(min(W, xx + r + 1))
    win = road_prob[y0:y1, x0:x1]
    if win.size == 0:
        return yy, xx
    j = int(np.argmax(win))
    wy, wx = np.unravel_index(j, win.shape)
    return int(y0 + wy), int(x0 + wx)


def _route_polyline(
    road_prob: np.ndarray,
    *,
    grid,
    start: Tuple[int, int],
    waypoints: List[Tuple[int, int]],
    end: Tuple[int, int],
    snap_radius: int,
) -> Tuple[List[int], List[int], bool]:
    """
    Deterministic polyline "landing" without A* search:
    start -> waypoints -> end, connected by Bresenham lines.

    This aims to keep expected concentrated (mode-like). Optionally snaps intermediate
    points to local maxima in road_prob within snap_radius to stay near the road manifold.
    """
    H, W = int(grid.H), int(grid.W)
    pts: List[Tuple[int, int]] = [tuple(map(int, start))] + [tuple(map(int, p)) for p in waypoints] + [
        tuple(map(int, end))
    ]
    if int(snap_radius) > 0 and len(pts) > 2:
        snapped: List[Tuple[int, int]] = [pts[0]]
        for y, x in pts[1:-1]:
            sy, sx = _snap_point_to_road(road_prob, y=int(y), x=int(x), radius=int(snap_radius), H=H, W=W)
            snapped.append((int(sy), int(sx)))
        snapped.append(pts[-1])
        pts = snapped

    ys: List[int] = []
    xs: List[int] = []
    for i in range(len(pts) - 1):
        y0, x0 = pts[i]
        y1, x1 = pts[i + 1]
        seg = _bresenham_line(y0, x0, y1, x1)
        if ys:
            seg = seg[1:]
        for y, x in seg:
            if 0 <= int(y) < H and 0 <= int(x) < W:
                ys.append(int(y))
                xs.append(int(x))
    if not ys:
        ys = [int(start[0]), int(end[0])]
        xs = [int(start[1]), int(end[1])]
    return ys, xs, True


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a behavioral reference as expected segments, using source-city route-shape templates + target-city soft OSM routing."
    )
    p.add_argument("--source_segments_parquet", type=Path, required=True, help="Functional city segments.parquet (traj_csv,y,x).")
    p.add_argument("--target_segments_parquet", type=Path, required=True, help="Target city observed segments.parquet (traj_csv,y,x).")
    p.add_argument("--target_osm_road_prob_npy", type=Path, required=True)
    p.add_argument("--target_osm_meta_json", type=Path, default=None, help="Optional osm_road_prob_meta.json for target (otherwise inferred).")
    p.add_argument("--out_parquet", type=Path, required=True)
    p.add_argument(
        "--od_bins",
        type=int,
        default=0,
        help="Optional OD binning per axis (e.g., 3 => 3x3 origin and 3x3 destination bins). When enabled, we try OD-conditioned templates first, then fallback to distance+direction bins.",
    )
    p.add_argument("--min_od_samples", type=int, default=50, help="Minimum source samples required for an OD bin (otherwise fallback).")
    p.add_argument("--dist_bins_m", type=float, nargs="+", default=[0, 2000, 5000, 10000, 20000, 40000])
    p.add_argument("--dir_bins", type=int, default=8)
    p.add_argument("--waypoint_fracs", type=float, nargs="+", default=[0.33, 0.66])
    p.add_argument("--min_bin_samples", type=int, default=50, help="Require at least this many source samples in a bin; otherwise fallback.")
    p.add_argument(
        "--landing",
        type=str,
        default="astar",
        choices=["astar", "polyline"],
        help="How to map templates onto the target city. 'astar' uses soft-OSM A*; 'polyline' connects (start, waypoints, end) without search (mode-like).",
    )
    p.add_argument(
        "--polyline_snap_radius",
        type=int,
        default=0,
        help="(polyline only) Snap intermediate points to local maxima in road_prob within this radius (in grid cells). 0 disables.",
    )
    p.add_argument("--lambda_offroad", type=float, default=20.0)
    p.add_argument("--max_expansions", type=int, default=350000)
    p.add_argument("--min_margin", type=int, default=64)
    p.add_argument("--max_margin", type=int, default=256)
    p.add_argument("--max_segments", type=int, default=0, help="Optional cap for speed/debug (0=no cap).")
    p.add_argument("--quiet", action="store_true", help="Write outputs only; do not print JSON.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if pq is None or pa is None:
        raise SystemExit("pyarrow is required. Install: pip/conda install pyarrow")

    osm_meta = args.target_osm_meta_json
    if osm_meta is None:
        cand = Path(args.target_osm_road_prob_npy).parent / "osm_road_prob_meta.json"
        osm_meta = cand if cand.exists() else None
    if osm_meta is None or not Path(osm_meta).exists():
        raise SystemExit("Missing target OSM meta: provide --target_osm_meta_json or place osm_road_prob_meta.json next to osm_road_prob.npy")

    grid = _load_grid_from_osm_meta(Path(osm_meta))
    road_prob = np.load(str(args.target_osm_road_prob_npy))
    if road_prob.shape != (int(grid.H), int(grid.W)):
        raise SystemExit(f"Shape mismatch: road_prob{tuple(road_prob.shape)} vs grid({grid.H},{grid.W})")

    res_y_m, res_x_m = grid.resolution_m()
    dist_bins = tuple(float(x) for x in args.dist_bins_m)
    if len(dist_bins) < 2 or any(not np.isfinite(v) for v in dist_bins):
        raise SystemExit("--dist_bins_m must be a list of finite numbers with len>=2")
    dist_bins = tuple(sorted(dist_bins))

    waypoint_fracs = tuple(float(x) for x in args.waypoint_fracs)
    if not waypoint_fracs:
        raise SystemExit("--waypoint_fracs must be non-empty (e.g., 0.33 0.66)")

    tcfg = TemplateCfg(
        dist_bins_m=dist_bins,
        dir_bins=int(args.dir_bins),
        waypoint_fracs=waypoint_fracs,
        min_bin_samples=int(args.min_bin_samples),
    )

    templates, counts, mode_frac = _build_templates(
        Path(args.source_segments_parquet),
        res_y_m=float(res_y_m),
        res_x_m=float(res_x_m),
        cfg=tcfg,
        max_segments=int(args.max_segments),
    )

    od_bins = int(args.od_bins)
    od_templates: Dict[Tuple[int, int], np.ndarray] = {}
    od_counts: Dict[Tuple[int, int], int] = {}
    od_mode_frac: Dict[Tuple[int, int], float] = {}
    if od_bins > 1:
        od_templates, od_counts, od_mode_frac = _build_od_templates(
            Path(args.source_segments_parquet),
            H=int(grid.H),
            W=int(grid.W),
            res_y_m=float(res_y_m),
            res_x_m=float(res_x_m),
            cfg=tcfg,
            od_bins=int(od_bins),
            min_od_samples=int(args.min_od_samples),
            max_segments=int(args.max_segments),
        )

    astar_cfg = AStarCfg(
        lambda_offroad=float(args.lambda_offroad),
        max_expansions=int(args.max_expansions),
        min_margin=int(args.min_margin),
        max_margin=int(args.max_margin),
    )

    out_parquet = Path(args.out_parquet)
    _ensure_dir(out_parquet.parent)
    report_path = out_parquet.with_suffix(".report.json")

    keys: List[str] = []
    ys: List[List[int]] = []
    xs: List[List[int]] = []
    start_y: List[int] = []
    start_x: List[int] = []
    end_y: List[int] = []
    end_x: List[int] = []
    ok_astar: List[bool] = []
    used_fallback: List[bool] = []
    tpl_from_od: List[bool] = []
    tpl_o_id: List[int] = []
    tpl_d_id: List[int] = []
    chosen_db: List[int] = []
    chosen_hb: List[int] = []

    scanned = 0
    fallback_n = 0
    od_used_n = 0
    od_fallback_n = 0
    ok_n = 0

    for k, y, x in _iter_segments_xy(Path(args.target_segments_parquet), max_segments=int(args.max_segments)):
        scanned += 1
        y0, x0, y1, x1 = int(y[0]), int(x[0]), int(y[-1]), int(x[-1])
        chord_m, theta = _chord_len_dir(y, x, res_y_m=float(res_y_m), res_x_m=float(res_x_m))
        db = _dist_bin_id(float(chord_m), tcfg.dist_bins_m)
        hb = _dir_bin_id(float(theta), int(tcfg.dir_bins))
        o_id, d_id = _od_key(y0, x0, y1, x1, H=int(grid.H), W=int(grid.W), od_bins=max(1, od_bins))
        used_od = False
        fb = False
        kk = (int(db), int(hb))
        if od_bins > 1:
            kk_od = (int(o_id), int(d_id))
            if kk_od in od_templates and int(od_counts.get(kk_od, 0)) >= int(args.min_od_samples):
                tpl = od_templates[kk_od]
                used_od = True
                od_used_n += 1
            else:
                tpl, kk, fb = _pick_template(templates, counts, db=db, hb=hb, cfg=tcfg)
                od_fallback_n += 1
        else:
            tpl, kk, fb = _pick_template(templates, counts, db=db, hb=hb, cfg=tcfg)

        if fb:
            fallback_n += 1

        # Inverse transform: normalized (x,y) -> target grid waypoint coords.
        # Metric start
        x0_m = float(x0) * float(res_x_m)
        y0_m = float(y0) * float(res_y_m)
        L = float(max(chord_m, 1e-6))

        wps_yx: List[Tuple[int, int]] = []
        for j in range(int(tpl.shape[0])):
            xn, yn = float(tpl[j, 0]), float(tpl[j, 1])
            dx_m, dy_m = _rotate(xn * L, yn * L, float(theta))
            wx_m = x0_m + float(dx_m)
            wy_m = y0_m + float(dy_m)
            wx = int(np.rint(wx_m / float(res_x_m)))
            wy = int(np.rint(wy_m / float(res_y_m)))
            wx = int(min(int(grid.W - 1), max(0, wx)))
            wy = int(min(int(grid.H - 1), max(0, wy)))
            wps_yx.append((wy, wx))

        if str(args.landing) == "polyline":
            py, px, ok = _route_polyline(
                road_prob,
                grid=grid,
                start=(y0, x0),
                waypoints=wps_yx,
                end=(y1, x1),
                snap_radius=int(args.polyline_snap_radius),
            )
        else:
            py, px, ok = _route_via_waypoints(
                road_prob,
                grid=grid,
                astar_cfg=astar_cfg,
                start=(y0, x0),
                waypoints=wps_yx,
                end=(y1, x1),
            )
        if ok:
            ok_n += 1

        keys.append(k)
        start_y.append(int(y0))
        start_x.append(int(x0))
        end_y.append(int(y1))
        end_x.append(int(x1))
        ys.append([int(v) for v in py] if py else [int(y0), int(y1)])
        xs.append([int(v) for v in px] if px else [int(x0), int(x1)])
        ok_astar.append(bool(ok))
        used_fallback.append(bool(fb))
        tpl_from_od.append(bool(used_od))
        tpl_o_id.append(int(o_id))
        tpl_d_id.append(int(d_id))
        chosen_db.append(int(kk[0]))
        chosen_hb.append(int(kk[1]))

    table = pa.table(
        {
            "traj_csv": pa.array(keys, type=pa.string()),
            "start_y": pa.array(start_y, type=pa.int32()),
            "start_x": pa.array(start_x, type=pa.int32()),
            "end_y": pa.array(end_y, type=pa.int32()),
            "end_x": pa.array(end_x, type=pa.int32()),
            "y": pa.array(ys, type=pa.list_(pa.int32())),
            "x": pa.array(xs, type=pa.list_(pa.int32())),
            "astar_success": pa.array(ok_astar, type=pa.bool_()),
            "used_fallback": pa.array(used_fallback, type=pa.bool_()),
            "tpl_from_od": pa.array(tpl_from_od, type=pa.bool_()),
            "tpl_o_id": pa.array(tpl_o_id, type=pa.int16()),
            "tpl_d_id": pa.array(tpl_d_id, type=pa.int16()),
            "tpl_dist_bin": pa.array(chosen_db, type=pa.int16()),
            "tpl_dir_bin": pa.array(chosen_hb, type=pa.int16()),
        }
    )
    pq.write_table(table, str(out_parquet))

    out = {
        "source_segments_parquet": str(Path(args.source_segments_parquet)),
        "target_segments_parquet": str(Path(args.target_segments_parquet)),
        "target_osm_road_prob_npy": str(Path(args.target_osm_road_prob_npy)),
        "target_osm_meta_json": str(Path(osm_meta)),
        "out_parquet": str(out_parquet),
        "grid": {"H": int(grid.H), "W": int(grid.W), "bbox": grid.bbox.__dict__},
        "template_cfg": {
            "template_repr": "medoid",
            "od_bins": int(od_bins),
            "min_od_samples": int(args.min_od_samples),
            "dist_bins_m": list(dist_bins),
            "dir_bins": int(tcfg.dir_bins),
            "waypoint_fracs": list(waypoint_fracs),
            "min_bin_samples": int(tcfg.min_bin_samples),
        },
        "landing": {
            "mode": str(args.landing),
            "polyline_snap_radius": int(args.polyline_snap_radius),
        },
        "astar_cfg": {
            "lambda_offroad": float(astar_cfg.lambda_offroad),
            "max_expansions": int(astar_cfg.max_expansions),
            "min_margin": int(astar_cfg.min_margin),
            "max_margin": int(astar_cfg.max_margin),
        },
        "stats": {
            "target_segments_scanned": int(scanned),
            "astar_success_rate": float(ok_n / max(1, scanned)),
            "template_fallback_rate": float(fallback_n / max(1, scanned)),
            "template_bins_total": int((len(dist_bins) - 1) * int(tcfg.dir_bins)),
            "template_bins_nonempty": int(len(templates)),
            "template_mode_frac": {
                "p50": float(np.percentile(list(mode_frac.values()), 50)) if mode_frac else float("nan"),
                "mean": float(np.mean(list(mode_frac.values()))) if mode_frac else float("nan"),
            },
            "template_od_bins_nonempty": int(len(od_templates)) if od_bins > 1 else 0,
            "template_od_used_rate": float(od_used_n / max(1, scanned)) if od_bins > 1 else 0.0,
            "template_od_fallback_rate": float(od_fallback_n / max(1, scanned)) if od_bins > 1 else 0.0,
            "template_od_mode_frac": {
                "p50": float(np.percentile(list(od_mode_frac.values()), 50)) if od_mode_frac else float("nan"),
                "mean": float(np.mean(list(od_mode_frac.values()))) if od_mode_frac else float("nan"),
            },
            "max_segments": int(args.max_segments),
        },
    }
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(args.quiet):
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Keep BLAS threads quiet in multiprocessing-heavy environments.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
