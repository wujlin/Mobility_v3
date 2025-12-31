from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:  # Optional; used for global nearest-drivable projection of GT end.
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None


@dataclass(frozen=True)
class Stats:
    p10: float
    p50: float
    p90: float
    mean: float


def _q(x: np.ndarray) -> Stats:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return Stats(p10=float("nan"), p50=float("nan"), p90=float("nan"), mean=float("nan"))
    p10, p50, p90 = np.percentile(x, [10, 50, 90]).tolist()
    return Stats(p10=float(p10), p50=float(p50), p90=float(p90), mean=float(np.mean(x)))


def _load_nav_count(nav_file: Path) -> np.ndarray:
    data = np.load(str(nav_file), allow_pickle=True)
    if "count" not in data.files:
        raise ValueError(f"nav_file must contain 'count', got {data.files}")
    count = np.asarray(data["count"], dtype=np.float32)
    if count.ndim != 2:
        raise ValueError(f"Expected count (H,W), got {count.shape}")
    return count


def _make_global_projector(nav_count: np.ndarray, *, count_thr: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if ndimage is None:
        return None
    drivable = np.asarray(nav_count >= float(count_thr), dtype=bool)
    offroad = ~drivable
    _, (iy, ix) = ndimage.distance_transform_edt(offroad, return_indices=True)
    return iy.astype(np.int64, copy=False), ix.astype(np.int64, copy=False)


def _project_to_drivable(
    pts: np.ndarray,  # (...,2)
    *,
    nav_count: np.ndarray,
    count_thr: float,
    projector: Optional[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    H, W = int(nav_count.shape[0]), int(nav_count.shape[1])
    y = np.clip(np.rint(pts[..., 0]).astype(np.int64), 0, H - 1)
    x = np.clip(np.rint(pts[..., 1]).astype(np.int64), 0, W - 1)
    drv = (nav_count[y, x] >= float(count_thr))
    if projector is None or np.all(drv):
        return np.stack([y, x], axis=-1).astype(np.float32)
    iy, ix = projector
    py = iy[y, x]
    px = ix[y, x]
    y2 = np.where(drv, y, py)
    x2 = np.where(drv, x, px)
    return np.stack([y2, x2], axis=-1).astype(np.float32)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End imprecision audit: decompose end-anchor error into along/cross components and GT distance; optional GT projection to strict drivable."
    )
    p.add_argument("--samples_npz", type=str, required=True, help="samples.npz with start_pos, targets, dest_pos, z_k_grid.")
    p.add_argument("--nav_file", type=str, required=True, help="nav_field.npz with count (for GT projection).")
    p.add_argument("--count_thr", type=float, default=1.0)
    p.add_argument("--k_index", type=int, default=0)
    p.add_argument("--use_gt_proj", action="store_true", help="Project GT end to nearest global drivable cell (requires scipy).")
    p.add_argument("--thr_along", type=float, default=8.0, help="Grid threshold for |Δalong| to call 'distance error'.")
    p.add_argument("--thr_cross", type=float, default=4.0, help="Grid threshold for |Δcross| to call 'corridor error'.")
    p.add_argument("--out_json", type=str, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    samples_npz = Path(args.samples_npz)
    nav_file = Path(args.nav_file)

    d = np.load(str(samples_npz), allow_pickle=True)
    need = {"start_pos", "targets", "dest_pos", "z_k_grid"}
    miss = [k for k in sorted(need) if k not in d.files]
    if miss:
        raise ValueError(f"samples_npz missing keys: {miss}. got={list(d.files)}")

    start = np.asarray(d["start_pos"], dtype=np.float32)
    targets = np.asarray(d["targets"], dtype=np.float32)
    dest = np.asarray(d["dest_pos"], dtype=np.float32)
    z = np.asarray(d["z_k_grid"], dtype=np.float32)
    if z.ndim != 4 or z.shape[-2:] != (3, 2):
        raise ValueError(f"Bad z_k_grid shape: {z.shape} (expected N,K,3,2)")
    k = int(args.k_index)
    if k < 0 or k >= int(z.shape[1]):
        raise ValueError(f"k_index out of range: {k}")
    end_pred = z[:, k, 2].astype(np.float32, copy=False)  # (N,2)

    end_gt = targets[:, -1].astype(np.float32, copy=False)
    nav_count = _load_nav_count(nav_file)
    projector = _make_global_projector(nav_count, count_thr=float(args.count_thr)) if bool(args.use_gt_proj) else None
    if bool(args.use_gt_proj) and projector is None:
        raise ImportError("--use_gt_proj requires scipy (missing scipy.ndimage).")
    end_gt_ref = _project_to_drivable(end_gt, nav_count=nav_count, count_thr=float(args.count_thr), projector=projector) if projector is not None else end_gt

    # --- progress to trip destination (direction correctness) ---
    dist0 = np.linalg.norm(start - dest, axis=-1)
    dist1 = np.linalg.norm(end_pred - dest, axis=-1)
    progress = dist0 - dist1

    # --- end accuracy vs GT end (raw/proj) ---
    err_end = np.linalg.norm(end_pred - end_gt_ref, axis=-1)

    # --- decompose w.r.t. start->dest direction ---
    v = (dest - start).astype(np.float64)
    vnorm = np.linalg.norm(v, axis=-1) + 1e-12
    u = v / vnorm[:, None]  # (N,2)
    u_perp = np.stack([u[:, 1], -u[:, 0]], axis=-1)  # rotate 90deg

    def proj_components(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dp = (p - start).astype(np.float64)
        along = np.sum(dp * u, axis=-1)
        cross = np.sum(dp * u_perp, axis=-1)
        return along, cross

    along_p, cross_p = proj_components(end_pred)
    along_g, cross_g = proj_components(end_gt_ref)
    dalong = along_p - along_g
    dcross = cross_p - cross_g

    # --- coarse type buckets ---
    thr_a = float(args.thr_along)
    thr_c = float(args.thr_cross)
    a_bad = np.abs(dalong) > thr_a
    c_bad = np.abs(dcross) > thr_c
    t_both = a_bad & c_bad
    t_dist = a_bad & (~c_bad)
    t_corr = c_bad & (~a_bad)
    t_fine = (~a_bad) & (~c_bad)

    out: Dict[str, object] = {
        "N": int(start.shape[0]),
        "meta": {
            "samples_npz": str(samples_npz),
            "nav_file": str(nav_file),
            "count_thr": float(args.count_thr),
            "k_index": int(args.k_index),
            "use_gt_proj": bool(args.use_gt_proj),
            "thr_along": thr_a,
            "thr_cross": thr_c,
        },
        "stats": {
            "progress_to_trip_dest": _q(progress).__dict__,
            "err_end_to_gt_end": _q(err_end).__dict__,
            "delta_along": _q(dalong).__dict__,
            "delta_cross": _q(dcross).__dict__,
            "abs_delta_along": _q(np.abs(dalong)).__dict__,
            "abs_delta_cross": _q(np.abs(dcross)).__dict__,
        },
        "types": {
            "fine_rate": float(np.mean(t_fine)),
            "dist_error_rate": float(np.mean(t_dist)),
            "corridor_error_rate": float(np.mean(t_corr)),
            "both_error_rate": float(np.mean(t_both)),
        },
    }

    print("============================================================")
    print("END IMPRECISION AUDIT")
    print("============================================================")
    print(f"N={out['N']}  use_gt_proj={bool(args.use_gt_proj)}  thr_along={thr_a} thr_cross={thr_c}")
    pr = out["stats"]["progress_to_trip_dest"]
    er = out["stats"]["err_end_to_gt_end"]
    print(f"Progress to trip dest (p10/p50/p90, mean): {pr['p10']:.2f}/{pr['p50']:.2f}/{pr['p90']:.2f}, mean={pr['mean']:.2f}")
    print(f"Err to GT end (p10/p50/p90, mean):        {er['p10']:.2f}/{er['p50']:.2f}/{er['p90']:.2f}, mean={er['mean']:.2f}")
    tp = out["types"]
    print(f"Type rates: fine={tp['fine_rate']:.3f}  dist={tp['dist_error_rate']:.3f}  corridor={tp['corridor_error_rate']:.3f}  both={tp['both_error_rate']:.3f}")
    print("Interpretation: 'corridor' => likely wrong parallel road; 'dist' => too short/too far along dest direction.")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()

