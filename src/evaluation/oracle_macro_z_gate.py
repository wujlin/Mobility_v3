from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:  # Optional: used only when --project_strict is enabled.
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from src.evaluation.macro_waypoint_gate import run_gate
from src.features.waypoints import WaypointConfig, extract_oracle_waypoints_from_future


def _load_samples_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "targets" not in data.files or "start_pos" not in data.files:
        raise ValueError(f"Input npz must contain keys ['targets','start_pos'], got {data.files}")
    targets = np.asarray(data["targets"], dtype=np.float32)  # (N,F,2)
    start_pos = np.asarray(data["start_pos"], dtype=np.float32)  # (N,2)
    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError(f"Expected targets (N,F,2), got {targets.shape}")
    if start_pos.ndim != 2 or start_pos.shape[-1] != 2:
        raise ValueError(f"Expected start_pos (N,2), got {start_pos.shape}")
    if int(targets.shape[0]) != int(start_pos.shape[0]):
        raise ValueError("N mismatch between targets and start_pos")
    return start_pos, targets


def _load_nav_count(nav_file: Path) -> np.ndarray:
    data = np.load(nav_file, allow_pickle=True)
    if "count" not in data.files:
        raise ValueError(f"nav_file must contain 'count', got {data.files}")
    count = np.asarray(data["count"], dtype=np.float32)
    if count.ndim != 2:
        raise ValueError(f"Expected nav count (H,W), got {count.shape}")
    return count


def _extract_z_oracle(
    *,
    start_pos: np.ndarray,  # (N,2)
    targets: np.ndarray,  # (N,F,2)
    num_waypoints: int,
) -> np.ndarray:
    cfg = WaypointConfig(mode="rdp_dev", num_waypoints=int(num_waypoints))
    N, F = int(targets.shape[0]), int(targets.shape[1])
    K = int(num_waypoints)
    if K != 2:
        raise ValueError("This script currently assumes num_waypoints=2 (wp1,wp2).")
    wp = np.zeros((N, K, 2), dtype=np.float32)
    for i in range(N):
        _, wpi = extract_oracle_waypoints_from_future(
            start_pos=start_pos[i],
            future_pos=targets[i],
            cfg=cfg,
        )
        if wpi.shape[0] >= 2:
            wp[i, :, :] = wpi[:2]
        elif wpi.shape[0] == 1:
            wp[i, 0, :] = wpi[0]
            wp[i, 1, :] = wpi[0]
        else:
            # Fallback: time quantiles in future positions (avoid crashing on degenerate windows).
            j1 = int(np.clip(np.rint((F - 1) / 3.0), 0, max(F - 1, 0)))
            j2 = int(np.clip(np.rint(2.0 * (F - 1) / 3.0), 0, max(F - 1, 0)))
            wp[i, 0, :] = targets[i, j1]
            wp[i, 1, :] = targets[i, j2]

    end = targets[:, -1, :].astype(np.float32, copy=False)  # (N,2)
    z = np.concatenate([wp, end[:, None, :]], axis=1).astype(np.float32)  # (N,3,2)
    return z


def _project_points_to_drivable(
    pts: np.ndarray,  # (...,2) float [y,x]
    *,
    drivable: np.ndarray,  # (H,W) bool
) -> np.ndarray:
    if ndimage is None:
        raise RuntimeError("scipy is required for --project_strict (missing scipy.ndimage).")
    H, W = int(drivable.shape[0]), int(drivable.shape[1])

    # Build a lookup from any cell -> nearest drivable cell (nearest 'False' in offroad mask).
    offroad = ~np.asarray(drivable, dtype=bool)
    _, (iy, ix) = ndimage.distance_transform_edt(offroad, return_indices=True)  # nearest drivable index for each cell

    out = np.asarray(pts, dtype=np.float32).copy()
    yy = np.rint(out[..., 0]).astype(np.int64)
    xx = np.rint(out[..., 1]).astype(np.int64)

    # Clip to bounds for indexing. Out-of-bounds points are snapped onto border then projected inward.
    yy_c = np.clip(yy, 0, H - 1)
    xx_c = np.clip(xx, 0, W - 1)

    ok = drivable[yy_c, xx_c]
    if np.any(~ok):
        py = iy[yy_c, xx_c]
        px = ix[yy_c, xx_c]
        out[..., 0] = np.where(ok, out[..., 0], py.astype(np.float32))
        out[..., 1] = np.where(ok, out[..., 1], px.astype(np.float32))
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Oracle GT z=[wp1,wp2,end] gate (with optional strict projection).")
    p.add_argument("--in_samples_npz", type=str, required=True, help="Input samples.npz containing targets/start_pos (e.g. dump_macro_diffusion_samples output).")
    p.add_argument("--nav_file", type=str, required=True, help="nav_field.npz containing count (used to define drivable mask).")
    p.add_argument("--count_thr", type=float, default=1.0, help="Drivable mask: count >= thr.")
    p.add_argument("--sample_step", type=float, default=0.5)
    p.add_argument("--max_samples_per_segment", type=int, default=256)
    p.add_argument("--num_waypoints", type=int, default=2)

    p.add_argument("--project_strict", action="store_true", help="Project wp1/wp2/end onto nearest drivable cell before gate.")
    p.add_argument("--out_npz", type=str, required=True, help="Output npz with start_pos + z_k_grid (K=1) for reproducibility.")
    p.add_argument("--out_json", type=str, default=None, help="Optional: save gate report json.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    in_npz = Path(args.in_samples_npz)
    nav_file = Path(args.nav_file)
    out_npz = Path(args.out_npz)

    start_pos, targets = _load_samples_npz(in_npz)
    z = _extract_z_oracle(start_pos=start_pos, targets=targets, num_waypoints=int(args.num_waypoints))  # (N,3,2)

    count = _load_nav_count(nav_file)
    drivable = np.asarray(count >= float(args.count_thr), dtype=bool)

    if bool(args.project_strict):
        z = _project_points_to_drivable(z, drivable=drivable)

    z_k_grid = z[:, None, :, :].astype(np.float32, copy=False)  # (N,1,3,2)

    meta = {
        "in_samples_npz": str(in_npz),
        "nav_file": str(nav_file),
        "count_thr": float(args.count_thr),
        "project_strict": bool(args.project_strict),
        "num_waypoints": int(args.num_waypoints),
        "sample_step": float(args.sample_step),
        "max_samples_per_segment": int(args.max_samples_per_segment),
    }
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, start_pos=start_pos.astype(np.float32, copy=False), z_k_grid=z_k_grid, meta=meta)

    report = run_gate(
        samples_npz=out_npz,
        nav_file=nav_file,
        count_thr=float(args.count_thr),
        sample_step=float(args.sample_step),
        max_samples_per_segment=int(args.max_samples_per_segment),
    )
    print("[OK] Oracle macro z gate")
    print(json.dumps(report["stats"], indent=2, ensure_ascii=False))
    print(json.dumps(report["results"], indent=2, ensure_ascii=False))

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[OK] saved: {out_json}")


if __name__ == "__main__":
    main()

