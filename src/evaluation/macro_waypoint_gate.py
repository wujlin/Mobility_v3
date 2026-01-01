from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _load_nav_count(nav_file: Path) -> np.ndarray:
    data = np.load(nav_file, allow_pickle=True)
    if "count" not in data.files:
        raise ValueError(f"nav_file must contain 'count', got {data.files}")
    count = np.asarray(data["count"], dtype=np.float32)
    if count.ndim != 2:
        raise ValueError(f"Expected count (H,W), got {count.shape}")
    return count


def _load_macro_samples(samples_npz: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(samples_npz, allow_pickle=True)
    if "start_pos" not in data.files:
        raise ValueError(f"samples.npz must contain start_pos, got {data.files}")
    if "z_k_grid" not in data.files:
        raise ValueError("samples.npz must contain z_k_grid (grid coords) for macro waypoint gate.")
    start_pos = np.asarray(data["start_pos"], dtype=np.float32)  # (N,2)
    z_k_grid = np.asarray(data["z_k_grid"], dtype=np.float32)  # (N,K,3,2)
    if start_pos.ndim != 2 or start_pos.shape[-1] != 2:
        raise ValueError(f"Expected start_pos (N,2), got {start_pos.shape}")
    if z_k_grid.ndim != 4 or z_k_grid.shape[-2:] != (3, 2):
        raise ValueError(f"Expected z_k_grid (N,K,3,2), got {z_k_grid.shape}")
    if start_pos.shape[0] != z_k_grid.shape[0]:
        raise ValueError("N mismatch between start_pos and z_k_grid")
    return start_pos, z_k_grid


def _index_round(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    yy = np.rint(pos[..., 0]).astype(np.int64)
    xx = np.rint(pos[..., 1]).astype(np.int64)
    return yy, xx


def _sample_segments(
    a: np.ndarray,  # (S,2)
    b: np.ndarray,  # (S,2)
    *,
    step: float,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    d = b - a
    seg_len = np.linalg.norm(d, axis=-1)  # (S,)
    # NOTE:
    # We must include the endpoint (t=1) for *every* segment, regardless of its sampled length.
    # A common pitfall is to use a shared linspace with max(m) samples and then truncate,
    # which causes short segments to miss their endpoints (under-counting collisions near b).
    n = np.ceil(seg_len / max(float(step), 1e-6)).astype(np.int64) + 1
    n = np.clip(n, 2, int(max_samples))
    m = int(np.max(n)) if int(n.size) else 0
    if m <= 0:
        return np.zeros((0, 0, 2), dtype=np.float32), np.zeros((0, 0), dtype=bool)

    idx = np.arange(int(m), dtype=np.float32)[None, :]  # (1,m)
    denom = np.maximum(n.astype(np.float32) - 1.0, 1.0)[:, None]  # (S,1)
    t = (idx / denom)[:, :, None]  # (S,m,1), ensures last valid index has t=1
    pts = a[:, None, :] + t * d[:, None, :]  # (S,m,2)
    valid = (np.arange(int(m), dtype=np.int64)[None, :] < n[:, None])  # (S,m)
    return pts.astype(np.float32, copy=False), valid


def run_gate(
    *,
    samples_npz: Path,
    nav_file: Path,
    count_thr: float,
    sample_step: float,
    max_samples_per_segment: int,
) -> Dict[str, object]:
    start_pos, z_k_grid = _load_macro_samples(samples_npz)
    count = _load_nav_count(nav_file)
    drivable = np.asarray(count >= float(count_thr), dtype=bool)
    H, W = int(drivable.shape[0]), int(drivable.shape[1])

    N, K = int(z_k_grid.shape[0]), int(z_k_grid.shape[1])
    start_k = np.repeat(start_pos[:, None, :], repeats=int(K), axis=1)  # (N,K,2)
    vertices = np.concatenate([start_k[:, :, None, :], z_k_grid], axis=2)  # (N,K,4,2)
    S = int(N * K)
    v = vertices.reshape(S, 4, 2)

    # Segment endpoints: (S*3,2)
    a = np.concatenate([v[:, 0], v[:, 1], v[:, 2]], axis=0)
    b = np.concatenate([v[:, 1], v[:, 2], v[:, 3]], axis=0)

    pts, valid = _sample_segments(a, b, step=float(sample_step), max_samples=int(max_samples_per_segment))  # (S*3,m,2)
    yy, xx = _index_round(pts)
    inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    yy_c = np.clip(yy, 0, H - 1)
    xx_c = np.clip(xx, 0, W - 1)
    drv = drivable[yy_c, xx_c]
    bad_point = valid & (~inb | ~drv)

    valid_points = int(np.sum(valid))
    bad_points = int(np.sum(bad_point))
    collision_point_rate = float(bad_points / max(valid_points, 1))

    seg_bad = np.any(bad_point, axis=1)  # (S*3,)
    seg_bad = seg_bad.reshape(3, S).T  # (S,3)
    collision_any = np.any(seg_bad, axis=1)  # (S,)
    collision_rate_any = float(np.mean(collision_any))

    # Waypoint-only offroad (wp1/wp2/end)
    wp = v[:, 1:, :]  # (S,3,2)
    y_wp, x_wp = _index_round(wp)
    inb_wp = (y_wp >= 0) & (y_wp < H) & (x_wp >= 0) & (x_wp < W)
    y_wp_c = np.clip(y_wp, 0, H - 1)
    x_wp_c = np.clip(x_wp, 0, W - 1)
    drv_wp = drivable[y_wp_c, x_wp_c]
    off_wp = (~inb_wp) | (~drv_wp)
    waypoint_offroad_rate = float(np.mean(off_wp))
    waypoint_any_offroad_rate = float(np.mean(np.any(off_wp, axis=1)))
    waypoint_oob_rate = float(np.mean(~inb_wp))
    wp1_offroad_rate = float(np.mean(off_wp[:, 0]))
    wp2_offroad_rate = float(np.mean(off_wp[:, 1]))
    end_offroad_rate = float(np.mean(off_wp[:, 2]))

    # "Cut-only" collision: the polyline collides but all waypoints are on-road.
    cut_only = collision_any & (~np.any(off_wp, axis=1))
    cut_only_rate = float(np.mean(cut_only))

    # Segment attribution: which segment(s) cause collision.
    seg0_rate = float(np.mean(seg_bad[:, 0]))  # start->wp1
    seg1_rate = float(np.mean(seg_bad[:, 1]))  # wp1->wp2
    seg2_rate = float(np.mean(seg_bad[:, 2]))  # wp2->end

    stats = {"N": int(N), "K": int(K), "S": int(S)}
    results = {
        "count_thr": float(count_thr),
        "sample_step": float(sample_step),
        "max_samples_per_segment": int(max_samples_per_segment),
        "collision_rate_any": float(collision_rate_any),
        "collision_point_rate": float(collision_point_rate),
        "waypoint_offroad_rate": float(waypoint_offroad_rate),
        "waypoint_any_offroad_rate": float(waypoint_any_offroad_rate),
        "waypoint_oob_rate": float(waypoint_oob_rate),
        "wp1_offroad_rate": float(wp1_offroad_rate),
        "wp2_offroad_rate": float(wp2_offroad_rate),
        "end_offroad_rate": float(end_offroad_rate),
        "cut_only_rate": float(cut_only_rate),
        "collision_seg0_rate": float(seg0_rate),
        "collision_seg1_rate": float(seg1_rate),
        "collision_seg2_rate": float(seg2_rate),
    }
    return {"stats": stats, "results": results}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Macro waypoint gate (CPU-only, no scipy): collision of predicted skeleton against nav_field count mask.")
    p.add_argument("--samples_npz", type=str, required=True, help="samples.npz containing start_pos and z_k_grid (grid coords).")
    p.add_argument("--nav_file", type=str, required=True, help="nav_field.npz containing count (train-only).")
    p.add_argument("--count_thr", type=float, default=1.0, help="Drivable mask: count >= thr is drivable.")
    p.add_argument("--sample_step", type=float, default=0.5, help="Sampling step along each segment (grid units).")
    p.add_argument("--max_samples_per_segment", type=int, default=256)
    p.add_argument("--out_json", type=str, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = run_gate(
        samples_npz=Path(args.samples_npz),
        nav_file=Path(args.nav_file),
        count_thr=float(args.count_thr),
        sample_step=float(args.sample_step),
        max_samples_per_segment=int(args.max_samples_per_segment),
    )
    print("[OK] Macro waypoint gate")
    print(json.dumps(report["stats"], indent=2))
    print(json.dumps(report["results"], indent=2))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[OK] saved: {out}")


if __name__ == "__main__":
    main()
