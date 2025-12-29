from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:  # Optional but recommended for fast nearest-drivable projection.
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from src.evaluation.macro_waypoint_gate import run_gate
from src.features.waypoints import WaypointConfig, extract_oracle_waypoints_from_future


def _load_windows_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "start_pos" not in data.files or "targets" not in data.files:
        raise ValueError(f"Expected keys start_pos/targets in {path}, got {data.files}")
    start_pos = np.asarray(data["start_pos"], dtype=np.float32)  # (N,2) global [y,x]
    targets = np.asarray(data["targets"], dtype=np.float32)  # (N,F,2) global [y,x]
    if start_pos.ndim != 2 or start_pos.shape[-1] != 2:
        raise ValueError(f"Expected start_pos (N,2), got {start_pos.shape}")
    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError(f"Expected targets (N,F,2), got {targets.shape}")
    if int(start_pos.shape[0]) != int(targets.shape[0]):
        raise ValueError("N mismatch between start_pos and targets")
    return start_pos, targets


def _load_nav_count(nav_file: Path) -> np.ndarray:
    data = np.load(nav_file, allow_pickle=True)
    if "count" not in data.files:
        raise ValueError(f"nav_file must contain 'count', got {data.files}")
    count = np.asarray(data["count"], dtype=np.float32)
    if count.ndim != 2:
        raise ValueError(f"Expected nav count (H,W), got {count.shape}")
    return count


def _extract_oracle_z(start_pos: np.ndarray, targets: np.ndarray, *, num_waypoints: int) -> np.ndarray:
    if int(num_waypoints) != 2:
        raise ValueError("KISS: currently only supports num_waypoints=2 (wp1,wp2,end).")
    cfg = WaypointConfig(mode="rdp_dev", num_waypoints=int(num_waypoints))
    N, F = int(targets.shape[0]), int(targets.shape[1])
    wp = np.zeros((N, 2, 2), dtype=np.float32)
    for i in range(N):
        _, wpi = extract_oracle_waypoints_from_future(start_pos=start_pos[i], future_pos=targets[i], cfg=cfg)
        if int(wpi.shape[0]) >= 2:
            wp[i] = wpi[:2]
        elif int(wpi.shape[0]) == 1:
            wp[i, 0] = wpi[0]
            wp[i, 1] = wpi[0]
        else:
            j1 = int(np.clip(np.rint((F - 1) / 3.0), 0, max(F - 1, 0)))
            j2 = int(np.clip(np.rint(2.0 * (F - 1) / 3.0), 0, max(F - 1, 0)))
            wp[i, 0] = targets[i, j1]
            wp[i, 1] = targets[i, j2]
    end = targets[:, -1, :].astype(np.float32, copy=False)
    return np.concatenate([wp, end[:, None, :]], axis=1).astype(np.float32, copy=False)  # (N,3,2)


def _nearest_drivable_projector(drivable: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if ndimage is None:
        raise RuntimeError("scipy is required for nearest-drivable projection (missing scipy.ndimage).")
    offroad = ~np.asarray(drivable, dtype=bool)
    _, (iy, ix) = ndimage.distance_transform_edt(offroad, return_indices=True)
    return iy.astype(np.int64, copy=False), ix.astype(np.int64, copy=False)


def _project_points(pts: np.ndarray, *, drivable: np.ndarray, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
    H, W = int(drivable.shape[0]), int(drivable.shape[1])
    out = np.asarray(pts, dtype=np.float32).copy()
    yy = np.rint(out[..., 0]).astype(np.int64)
    xx = np.rint(out[..., 1]).astype(np.int64)
    yy_c = np.clip(yy, 0, H - 1)
    xx_c = np.clip(xx, 0, W - 1)
    ok = drivable[yy_c, xx_c]
    if np.any(~ok):
        py = iy[yy_c, xx_c]
        px = ix[yy_c, xx_c]
        out[..., 0] = np.where(ok, out[..., 0], py.astype(np.float32))
        out[..., 1] = np.where(ok, out[..., 1], px.astype(np.float32))
    return out


def _extract_count_patch(
    count: np.ndarray, *, center_yx: np.ndarray, patch_size: int
) -> np.ndarray:
    H, W = int(count.shape[0]), int(count.shape[1])
    k = int(patch_size)
    r = int(k // 2)
    y, x = int(np.rint(float(center_yx[0]))), int(np.rint(float(center_yx[1])))
    y_min, y_max = y - r, y + r
    x_min, x_max = x - r, x + r
    patch = np.zeros((k, k), dtype=np.float32)
    img_y_min = max(0, y_min)
    img_y_max = min(H, y_max)
    img_x_min = max(0, x_min)
    img_x_max = min(W, x_max)
    py0 = img_y_min - y_min
    px0 = img_x_min - x_min
    py1 = py0 + (img_y_max - img_y_min)
    px1 = px0 + (img_x_max - img_x_min)
    if img_y_max > img_y_min and img_x_max > img_x_min:
        patch[py0:py1, px0:px1] = count[img_y_min:img_y_max, img_x_min:img_x_max]
    return patch


def _to_patch_xy(
    pts_global: np.ndarray, *, start_pos: np.ndarray, patch_size: int
) -> np.ndarray:
    k = int(patch_size)
    r = float(k // 2)
    rel = pts_global - start_pos[:, None, :]  # (N,3,2)
    patch_xy = rel + r
    return patch_xy.astype(np.float32, copy=False)


def _from_patch_xy(
    pts_patch: np.ndarray, *, start_pos: np.ndarray, patch_size: int
) -> np.ndarray:
    k = int(patch_size)
    r = float(k // 2)
    return (pts_patch - r) + start_pos[:, None, :]


def _quantize_coarse_only(
    pts_global: np.ndarray, *,
    start_pos: np.ndarray,
    patch_size: int,
    coarse_g: int,
) -> np.ndarray:
    """
    Ablation: keep only coarse cell identity.
    We map each point to its coarse cell in patch coords, then place it at the integer center
    of that coarse cell (nearest pixel center).
    """
    k = int(patch_size)
    g = int(coarse_g)
    if k % g != 0:
        raise ValueError(f"patch_size must be divisible by coarse_g, got {k} vs {g}")
    cell = int(k // g)
    r = float(k // 2)
    patch_xy = (pts_global - start_pos[:, None, :]) + r  # (N,3,2) float patch coords
    patch_xy_i = np.rint(patch_xy).astype(np.int64)
    patch_xy_i[..., 0] = np.clip(patch_xy_i[..., 0], 0, k - 1)
    patch_xy_i[..., 1] = np.clip(patch_xy_i[..., 1], 0, k - 1)
    cy = (patch_xy_i[..., 0] // cell).astype(np.int64)
    cx = (patch_xy_i[..., 1] // cell).astype(np.int64)
    # pick an integer "center" pixel inside the cell
    oy = (cy * cell + (cell // 2)).astype(np.float32)
    ox = (cx * cell + (cell // 2)).astype(np.float32)
    out_patch = np.stack([oy, ox], axis=-1).astype(np.float32)  # (N,3,2) patch coords
    return (out_patch - r) + start_pos[:, None, :]  # back to global


def _coarse_mask_recall(
    *,
    strict_patch_mask: np.ndarray,  # (N,K,K) bool
    pts_global: np.ndarray,  # (N,3,2) global
    start_pos: np.ndarray,  # (N,2)
    patch_size: int,
    coarse_g: int,
) -> Dict[str, float]:
    k = int(patch_size)
    g = int(coarse_g)
    if k % g != 0:
        raise ValueError(f"patch_size must be divisible by coarse_g, got {k} vs {g}")
    cell = int(k // g)
    # Coarse cell is drivable iff any strict pixel exists in the cell.
    m = strict_patch_mask.reshape(strict_patch_mask.shape[0], g, cell, g, cell).any(axis=(2, 4))  # (N,g,g)

    pts_patch = _to_patch_xy(pts_global, start_pos=start_pos, patch_size=patch_size)  # (N,3,2)
    pts_i = np.rint(pts_patch).astype(np.int64)
    pts_i[..., 0] = np.clip(pts_i[..., 0], 0, k - 1)
    pts_i[..., 1] = np.clip(pts_i[..., 1], 0, k - 1)
    cy = (pts_i[..., 0] // cell).astype(np.int64)
    cx = (pts_i[..., 1] // cell).astype(np.int64)
    ok = m[np.arange(m.shape[0])[:, None], cy, cx]  # (N,3) bool
    return {
        "recall_wp1": float(np.mean(ok[:, 0])),
        "recall_wp2": float(np.mean(ok[:, 1])),
        "recall_end": float(np.mean(ok[:, 2])),
        "recall_all_points": float(np.mean(ok)),
        "num_cells_total": float(g * g),
        "mean_drivable_cells_per_patch": float(np.mean(m.sum(axis=(1, 2)))),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline audits for Macro Hard Support (cheap, CPU-only).")
    p.add_argument("--in_samples_npz", type=str, required=True, help="Input windows npz containing start_pos/targets (e.g., dump_macro_diffusion_samples output).")
    p.add_argument("--nav_file", type=str, required=True, help="nav_field.npz with count (used for strict drivable mask).")
    p.add_argument("--count_thr", type=float, default=1.0)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--coarse_g", type=int, default=16)
    p.add_argument("--num_waypoints", type=int, default=2)
    p.add_argument("--sample_step", type=float, default=0.5)
    p.add_argument("--max_samples_per_segment", type=int, default=256)
    p.add_argument("--max_n", type=int, default=0, help="Optional: cap number of windows (0 = all).")
    p.add_argument("--out_dir", type=str, default="data/experiments/macro_hardsupport_offline_audit", help="Output directory.")
    p.add_argument("--out_json", type=str, default=None, help="Optional: save full audit json.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    in_npz = Path(args.in_samples_npz)
    nav_file = Path(args.nav_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_pos, targets = _load_windows_npz(in_npz)
    if int(args.max_n) > 0:
        n = int(min(int(args.max_n), int(start_pos.shape[0])))
        start_pos = start_pos[:n]
        targets = targets[:n]

    count = _load_nav_count(nav_file)
    drivable = np.asarray(count >= float(args.count_thr), dtype=bool)
    iy, ix = _nearest_drivable_projector(drivable)

    z_raw = _extract_oracle_z(start_pos, targets, num_waypoints=int(args.num_waypoints))  # (N,3,2)
    z_proj = _project_points(z_raw, drivable=drivable, iy=iy, ix=ix)

    # ---- 1) Empty strict-mask stats on per-sample patches ----
    k = int(args.patch_size)
    strict_sum = np.zeros((int(start_pos.shape[0]),), dtype=np.int64)
    corr_num = 0.0
    corr_den1 = 0.0
    corr_den2 = 0.0
    mean_on = []
    mean_off = []
    for i in range(int(start_pos.shape[0])):
        patch = _extract_count_patch(count, center_yx=start_pos[i], patch_size=k)
        m = patch >= float(args.count_thr)
        strict_sum[i] = int(np.sum(m))
        # correlation between log1p(count) and strict mask (binary) within patch
        c = np.log1p(patch).reshape(-1).astype(np.float64)
        b = m.reshape(-1).astype(np.float64)
        c0 = c - c.mean()
        b0 = b - b.mean()
        corr_num += float(np.sum(c0 * b0))
        corr_den1 += float(np.sum(c0 * c0))
        corr_den2 += float(np.sum(b0 * b0))
        if np.any(m):
            mean_on.append(float(np.mean(patch[m])))
        if np.any(~m):
            mean_off.append(float(np.mean(patch[~m])))

    empty_patch_rate = float(np.mean(strict_sum == 0))
    mean_strict_pixels = float(np.mean(strict_sum))
    corr = float(corr_num / max(np.sqrt(corr_den1 * corr_den2), 1e-12))
    nav_stats = {
        "empty_strict_patch_rate": empty_patch_rate,
        "mean_strict_pixels_per_patch": mean_strict_pixels,
        "p50_strict_pixels": float(np.percentile(strict_sum, 50)),
        "p10_strict_pixels": float(np.percentile(strict_sum, 10)),
        "p90_strict_pixels": float(np.percentile(strict_sum, 90)),
        "corr_log1p_count_vs_strict_mask": corr,
        "mean_count_onroad": float(np.mean(mean_on)) if mean_on else 0.0,
        "mean_count_offroad": float(np.mean(mean_off)) if mean_off else 0.0,
    }

    # ---- 2) Coarse recall (should be ~1.0 by construction if coarse mask is max-pooled strict) ----
    strict_patch_mask = np.stack(
        [_extract_count_patch(count, center_yx=start_pos[i], patch_size=k) >= float(args.count_thr) for i in range(int(start_pos.shape[0]))],
        axis=0,
    ).astype(bool)
    coarse_recall = _coarse_mask_recall(
        strict_patch_mask=strict_patch_mask,
        pts_global=z_proj,
        start_pos=start_pos,
        patch_size=k,
        coarse_g=int(args.coarse_g),
    )

    # ---- 3) Discretization / simplification ablation: coarse-only vs exact ----
    # Exact-pixel (oracle_proj) is already the best-case lower bound.
    # Coarse-only: keep only coarse-cell identity then place at cell center; finally project to drivable.
    z_coarse = _quantize_coarse_only(z_proj, start_pos=start_pos, patch_size=k, coarse_g=int(args.coarse_g))
    z_coarse_proj = _project_points(z_coarse, drivable=drivable, iy=iy, ix=ix)

    def _gate_from_z(name: str, z: np.ndarray) -> Dict[str, object]:
        tmp = out_dir / f"{name}.npz"
        np.savez_compressed(tmp, start_pos=start_pos.astype(np.float32, copy=False), z_k_grid=z[:, None, :, :].astype(np.float32, copy=False))
        return run_gate(
            samples_npz=tmp,
            nav_file=nav_file,
            count_thr=float(args.count_thr),
            sample_step=float(args.sample_step),
            max_samples_per_segment=int(args.max_samples_per_segment),
        )

    gate_oracle_raw = _gate_from_z("oracle_z_raw", z_raw)
    gate_oracle_proj = _gate_from_z("oracle_z_proj", z_proj)
    gate_coarse_proj = _gate_from_z("oracle_z_coarse_only_proj", z_coarse_proj)

    report = {
        "meta": {
            "in_samples_npz": str(in_npz),
            "nav_file": str(nav_file),
            "N": int(start_pos.shape[0]),
            "patch_size": int(args.patch_size),
            "count_thr": float(args.count_thr),
            "coarse_g": int(args.coarse_g),
            "sample_step": float(args.sample_step),
            "max_samples_per_segment": int(args.max_samples_per_segment),
        },
        "nav_stats": nav_stats,
        "coarse_recall": coarse_recall,
        "gate": {
            "oracle_raw": gate_oracle_raw["results"],
            "oracle_proj": gate_oracle_proj["results"],
            "oracle_proj_coarse_only": gate_coarse_proj["results"],
        },
    }

    print("[OK] Macro Hard Support offline audit")
    print(json.dumps(report["meta"], indent=2, ensure_ascii=False))
    print(json.dumps(report["nav_stats"], indent=2, ensure_ascii=False))
    print(json.dumps(report["coarse_recall"], indent=2, ensure_ascii=False))
    print(json.dumps(report["gate"], indent=2, ensure_ascii=False))

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[OK] saved: {out_json}")


if __name__ == "__main__":
    main()

