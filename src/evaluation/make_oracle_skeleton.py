from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    from scipy.interpolate import CubicSpline, PchipInterpolator
except Exception as e:  # pragma: no cover
    CubicSpline = None  # type: ignore[assignment]
    PchipInterpolator = None  # type: ignore[assignment]
    _SCIPY_IMPORT_ERROR = e
else:  # pragma: no cover
    _SCIPY_IMPORT_ERROR = None


WaypointMode = Literal["time", "arclen", "max_dev", "rdp_dev", "max_turn"]
SkeletonKind = Literal["linear", "pchip", "cubic"]


def _load_samples_npz(path: Path, *, max_n: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(str(path))
    if "targets" not in data.files or "start_pos" not in data.files:
        raise ValueError(f"Bad samples.npz: require keys ['targets','start_pos'], got {data.files}")
    targets = np.asarray(data["targets"], dtype=np.float32)  # (N,F,2)
    start_pos = np.asarray(data["start_pos"], dtype=np.float32)  # (N,2)
    if max_n is not None:
        targets = targets[: int(max_n)]
        start_pos = start_pos[: int(max_n)]
    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError(f"Expected targets (N,F,2), got {targets.shape}")
    if start_pos.ndim != 2 or start_pos.shape[-1] != 2:
        raise ValueError(f"Expected start_pos (N,2), got {start_pos.shape}")
    if targets.shape[0] != start_pos.shape[0]:
        raise ValueError("N mismatch between targets and start_pos")
    return targets, start_pos


def _distance_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = b - a
    ap = p - a
    ab2 = np.sum(ab * ab, axis=-1, keepdims=True)
    ab2 = np.where(ab2 > 1e-8, ab2, 1e-8)
    t = np.sum(ap * ab, axis=-1, keepdims=True) / ab2
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj, axis=-1).astype(np.float32)


def _pick_waypoint_indices_time(F: int, *, K: int) -> List[int]:
    if int(K) <= 0:
        return []
    idx: List[int] = []
    for i in range(int(K)):
        frac = float(i + 1) / float(K + 1)
        j = int(np.rint(frac * float(F - 1)))
        j = int(np.clip(j, 1, max(F - 2, 1)))
        idx.append(j)
    idx = sorted(set(idx))
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
    gt = np.asarray(gt, dtype=np.float32)
    start_pos = np.asarray(start_pos, dtype=np.float32)
    N, F = int(gt.shape[0]), int(gt.shape[1])
    if int(K) <= 0:
        return np.zeros((N, 0), dtype=np.int64), np.zeros((N, 0, 2), dtype=np.float32)

    end_pos = gt[:, -1, :]
    internal = gt[:, 1:-1, :]

    if mode == "time":
        idx_list = _pick_waypoint_indices_time(F, K=int(K))
        idx = np.tile(np.asarray(idx_list, dtype=np.int64)[None, :], (N, 1))
        wp = gt[np.arange(N)[:, None], idx]
        return idx, wp

    if mode == "arclen":
        poly = np.concatenate([start_pos[:, None, :], gt], axis=1)
        seg = poly[:, 1:, :] - poly[:, :-1, :]
        seg_len = np.linalg.norm(seg, axis=-1)
        s = np.cumsum(seg_len, axis=1)
        total = s[:, -1:]
        total = np.where(total > 1e-6, total, 1e-6)
        idx = np.zeros((N, int(K)), dtype=np.int64)
        for kk in range(int(K)):
            frac = float(kk + 1) / float(int(K) + 1)
            tgt = total[:, 0] * frac
            j = np.argmin(np.abs(s - tgt[:, None]), axis=1).astype(np.int64)
            j = np.clip(j, 1, max(F - 2, 1))
            idx[:, kk] = j
        idx = np.sort(idx, axis=1)
        wp = gt[np.arange(N)[:, None], idx]
        return idx, wp

    if mode == "max_dev":
        d = _distance_point_to_segment(internal, start_pos[:, None, :], end_pos[:, None, :])
        order = np.argsort(-d, axis=1)
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

    if mode == "rdp_dev":
        poly = np.concatenate([start_pos[:, None, :], gt], axis=1)  # (N,F+1,2)
        poly_len = int(poly.shape[1])
        chosen = np.full((N, int(K)), -1, dtype=np.int64)
        for i in range(N):
            segs: List[Tuple[int, int]] = [(0, poly_len - 1)]
            picks: List[int] = []
            for _ in range(int(K)):
                best = None
                for si, (a_idx, b_idx) in enumerate(segs):
                    if b_idx - a_idx <= 1:
                        continue
                    a = poly[i, a_idx]
                    b = poly[i, b_idx]
                    pts = poly[i, a_idx + 1 : b_idx]
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

            gt_idx = [p - 1 for p in picks if 1 <= p <= poly_len - 2]
            gt_idx = [int(np.clip(j, 1, max(F - 2, 1))) for j in gt_idx]
            gt_idx = sorted(set(gt_idx))[: int(K)]
            if len(gt_idx) < int(K):
                fill = _pick_waypoint_indices_time(F, K=int(K))
                for j in fill:
                    if len(gt_idx) >= int(K):
                        break
                    if j not in gt_idx:
                        gt_idx.append(j)
                gt_idx = sorted(gt_idx)[: int(K)]
            chosen[i, : len(gt_idx)] = np.asarray(gt_idx, dtype=np.int64)
        need = chosen < 0
        if np.any(need):
            fill = _pick_waypoint_indices_time(F, K=int(K))
            chosen[need] = fill[0] if fill else 1
        idx = np.sort(chosen, axis=1)
        wp = gt[np.arange(N)[:, None], idx]
        return idx, wp

    if mode == "max_turn":
        disp = gt[:, 1:, :] - gt[:, :-1, :]
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
        ang = np.arccos(cos).astype(np.float32)
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


def _poly_arclength(points: np.ndarray) -> Tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points (M,2), got {points.shape}")
    if points.shape[0] < 2:
        return np.zeros((points.shape[0],), dtype=np.float32), 0.0
    seg = points[1:] - points[:-1]
    seg_len = np.linalg.norm(seg, axis=1).astype(np.float32)
    s = np.concatenate([[0.0], np.cumsum(seg_len)], axis=0).astype(np.float32)
    return s, float(s[-1])


def _resample_by_arclength(points: np.ndarray, *, num: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    s_nodes, total = _poly_arclength(points)
    if int(num) <= 1:
        return points[:1]
    if not np.isfinite(total) or total <= 1e-6:
        return np.repeat(points[:1], repeats=int(num), axis=0)
    s_query = np.linspace(0.0, total, num=int(num), dtype=np.float32)
    y = np.interp(s_query, s_nodes, points[:, 0]).astype(np.float32)
    x = np.interp(s_query, s_nodes, points[:, 1]).astype(np.float32)
    return np.stack([y, x], axis=1).astype(np.float32)


def _spline_curve(vertices: np.ndarray, *, kind: SkeletonKind, dense: int) -> np.ndarray:
    if kind == "linear":
        return np.asarray(vertices, dtype=np.float32)
    if _SCIPY_IMPORT_ERROR is not None:
        raise RuntimeError("该脚本的 spline skeleton 依赖 scipy，请先安装 scipy。") from _SCIPY_IMPORT_ERROR
    assert PchipInterpolator is not None and CubicSpline is not None

    v = np.asarray(vertices, dtype=np.float32)
    seg = v[1:] - v[:-1]
    seg_len = np.linalg.norm(seg, axis=1).astype(np.float32)
    t_nodes = np.concatenate([[0.0], np.cumsum(seg_len)], axis=0).astype(np.float32)

    # Dedup non-strict nodes (lingering / repeated positions).
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
    if not np.isfinite(total) or total <= 1e-6 or t_nodes.size < 2:
        return np.asarray(vertices, dtype=np.float32)

    t_query = np.linspace(0.0, total, num=int(max(dense, int(t_nodes.size) * 16)), dtype=np.float32)
    if kind == "pchip":
        fy = PchipInterpolator(t_nodes, v[:, 0], extrapolate=True)
        fx = PchipInterpolator(t_nodes, v[:, 1], extrapolate=True)
        y = fy(t_query).astype(np.float32)
        x = fx(t_query).astype(np.float32)
    elif kind == "cubic":
        fy = CubicSpline(t_nodes, v[:, 0], bc_type="natural", extrapolate=True)
        fx = CubicSpline(t_nodes, v[:, 1], bc_type="natural", extrapolate=True)
        y = fy(t_query).astype(np.float32)
        x = fx(t_query).astype(np.float32)
    else:
        raise ValueError(f"Unknown spline kind: {kind}")
    return np.stack([y, x], axis=1).astype(np.float32)


def make_skeleton_npz(
    *,
    samples_npz: Path,
    out_npz: Path,
    waypoint_mode: WaypointMode,
    num_waypoints: int,
    skeleton: SkeletonKind,
    min_sep: int,
    turn_min_speed: float,
    seed: int,
    max_n: Optional[int],
    spline_dense: int,
) -> Dict[str, object]:
    targets, start_pos = _load_samples_npz(samples_npz, max_n=max_n)
    N, F, _ = targets.shape
    end_pos = targets[:, -1, :]

    idx_wp, wp = _pick_waypoints(
        targets,
        start_pos,
        mode=str(waypoint_mode),  # type: ignore[arg-type]
        K=int(num_waypoints),
        seed=int(seed),
        min_sep=int(min_sep),
        turn_min_speed=float(turn_min_speed),
    )

    vertices = np.concatenate([start_pos[:, None, :], wp, end_pos[:, None, :]], axis=1).astype(np.float32)  # (N,K+2,2)
    preds = np.zeros((N, F, 2), dtype=np.float32)
    for i in range(N):
        curve = _spline_curve(vertices[i], kind=str(skeleton), dense=int(spline_dense))  # type: ignore[arg-type]
        sampled = _resample_by_arclength(curve, num=int(F + 1))
        preds[i] = sampled[1:]

    meta = {
        "samples_npz": str(samples_npz),
        "waypoint_mode": str(waypoint_mode),
        "num_waypoints": int(num_waypoints),
        "skeleton": str(skeleton),
        "min_sep": int(min_sep),
        "turn_min_speed": float(turn_min_speed),
        "seed": int(seed),
        "max_n": (int(max_n) if max_n is not None else None),
        "spline_dense": int(spline_dense),
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, preds=preds, targets=targets, start_pos=start_pos, waypoint_idx=idx_wp, meta=meta)
    return {"N": int(N), "F": int(F), "out_npz": str(out_npz), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Make oracle skeleton-only samples.npz (GT waypoints -> skeleton -> arclen-resampled).")
    p.add_argument("--samples_npz", type=str, required=True, help="Input samples.npz with GT targets + start_pos.")
    p.add_argument("--out_npz", type=str, required=True, help="Output npz path (will contain preds/targets/start_pos).")

    p.add_argument("--waypoint_mode", type=str, default="rdp_dev", choices=["time", "arclen", "max_dev", "rdp_dev", "max_turn"])
    p.add_argument("--num_waypoints", type=int, default=2, help="Number of waypoints (0 means straight start->end).")
    p.add_argument("--min_sep", type=int, default=2)
    p.add_argument("--turn_min_speed", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--skeleton", type=str, default="linear", choices=["linear", "pchip", "cubic"])
    p.add_argument("--spline_dense", type=int, default=256, help="Dense samples along spline before arclen resampling.")

    p.add_argument("--max_n", type=int, default=None, help="Optional cap on number of samples.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = make_skeleton_npz(
        samples_npz=Path(args.samples_npz),
        out_npz=Path(args.out_npz),
        waypoint_mode=str(args.waypoint_mode),  # type: ignore[arg-type]
        num_waypoints=int(args.num_waypoints),
        skeleton=str(args.skeleton),  # type: ignore[arg-type]
        min_sep=int(args.min_sep),
        turn_min_speed=float(args.turn_min_speed),
        seed=int(args.seed),
        max_n=int(args.max_n) if args.max_n is not None else None,
        spline_dense=int(args.spline_dense),
    )
    print("[OK] saved oracle skeleton")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

