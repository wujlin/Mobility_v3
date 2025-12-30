from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:  # Optional; used for global distance transform (clearance) + nearest-drivable projection.
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

try:  # Optional; used for quick visualization.
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

from src.evaluation.distribution_metrics import compute_jsd_from_samples, jsd_from_hist
from src.features.waypoints import WaypointConfig, extract_oracle_waypoints_from_future


@dataclass(frozen=True)
class Inputs:
    start_pos: np.ndarray  # (N,2) float32 grid [y,x]
    targets: np.ndarray  # (N,F,2) float32 grid [y,x]
    z_k_grid: np.ndarray  # (N,K,3,2) float32 grid [y,x]
    meta: Dict[str, object]


def _load_nav_count(nav_file: Path) -> np.ndarray:
    data = np.load(nav_file, allow_pickle=True)
    if "count" not in data.files:
        raise ValueError(f"nav_file must contain 'count', got {data.files}")
    count = np.asarray(data["count"], dtype=np.float32)
    if count.ndim != 2:
        raise ValueError(f"Expected count (H,W), got {count.shape}")
    return count


def _load_inputs(samples_npz: Path) -> Inputs:
    data = np.load(samples_npz, allow_pickle=True)
    need = {"start_pos", "targets", "z_k_grid"}
    miss = [k for k in sorted(need) if k not in data.files]
    if miss:
        raise ValueError(f"samples_npz missing keys: {miss}. Got: {data.files}")

    start_pos = np.asarray(data["start_pos"], dtype=np.float32)
    targets = np.asarray(data["targets"], dtype=np.float32)
    z_k_grid = np.asarray(data["z_k_grid"], dtype=np.float32)
    meta = {}
    if "meta" in data.files:
        m = data["meta"]
        meta = m.item() if hasattr(m, "item") else (m if isinstance(m, dict) else {})

    if start_pos.ndim != 2 or start_pos.shape[1] != 2:
        raise ValueError(f"Expected start_pos (N,2), got {start_pos.shape}")
    if targets.ndim != 3 or targets.shape[2] != 2:
        raise ValueError(f"Expected targets (N,F,2), got {targets.shape}")
    if z_k_grid.ndim != 4 or z_k_grid.shape[-2:] != (3, 2):
        raise ValueError(f"Expected z_k_grid (N,K,3,2), got {z_k_grid.shape}")
    if int(z_k_grid.shape[0]) != int(start_pos.shape[0]):
        raise ValueError("N mismatch between start_pos and z_k_grid")
    if int(targets.shape[0]) != int(start_pos.shape[0]):
        raise ValueError("N mismatch between start_pos and targets")

    return Inputs(start_pos=start_pos, targets=targets, z_k_grid=z_k_grid, meta=meta)


def _make_global_projector(nav_count: np.ndarray, *, count_thr: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if ndimage is None:
        return None
    drivable = np.asarray(nav_count >= float(count_thr), dtype=bool)
    offroad = ~drivable
    _, (iy, ix) = ndimage.distance_transform_edt(offroad, return_indices=True)
    return iy.astype(np.int64, copy=False), ix.astype(np.int64, copy=False)


def _project_to_drivable(
    pts: np.ndarray,  # (...,2) float
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
        out = np.stack([y, x], axis=-1).astype(np.float32)
        return out
    iy, ix = projector
    py = iy[y, x]
    px = ix[y, x]
    y2 = np.where(drv, y, py)
    x2 = np.where(drv, x, px)
    return np.stack([y2, x2], axis=-1).astype(np.float32)


def _extract_gt_z_raw(start_pos: np.ndarray, future_pos: np.ndarray) -> np.ndarray:
    cfg = WaypointConfig(mode="rdp_dev", num_waypoints=2)
    _, wp = extract_oracle_waypoints_from_future(start_pos=start_pos, future_pos=future_pos, cfg=cfg)
    wp = np.asarray(wp, dtype=np.float32).reshape(-1, 2)
    if wp.shape[0] < 2:
        if wp.shape[0] == 0:
            wp = np.repeat(future_pos[:1], repeats=2, axis=0)
        else:
            wp = np.concatenate([wp, wp[-1:]], axis=0)
    end = np.asarray(future_pos[-1], dtype=np.float32).reshape(1, 2)
    return np.concatenate([wp[:2], end], axis=0)  # (3,2)


def _patch_center(start_pos: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(start_pos, dtype=np.float32)).astype(np.int64)


def _to_patch_px(
    pts: np.ndarray,  # (M,2) float grid
    *,
    center: np.ndarray,  # (2,) int64
    patch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = int(patch_size)
    r = int(K // 2)
    y = np.rint(np.asarray(pts[:, 0], dtype=np.float32)).astype(np.int64)
    x = np.rint(np.asarray(pts[:, 1], dtype=np.float32)).astype(np.int64)
    py = y - int(center[0]) + r
    px = x - int(center[1]) + r
    inb = (py >= 0) & (py < K) & (px >= 0) & (px < K)
    return py.astype(np.int64, copy=False), px.astype(np.int64, copy=False), inb


def _heatmap_counts(py: np.ndarray, px: np.ndarray, *, K: int) -> np.ndarray:
    c = np.zeros((int(K), int(K)), dtype=np.int64)
    if py.size == 0:
        return c
    np.add.at(c, (py, px), 1)
    return c


def _entropy_from_counts(counts: np.ndarray, *, eps: float = 1e-12) -> float:
    x = np.asarray(counts, dtype=np.float64).reshape(-1)
    x = x + float(eps)
    x = x / float(x.sum())
    return float(-(x * np.log(x)).sum())


def _plot_heatmaps(
    *,
    out_png: Path,
    K: int,
    stages: Tuple[str, str, str],
    gt_counts: Dict[str, np.ndarray],
    pred_counts: Dict[str, np.ndarray],
    rand_counts: Dict[str, np.ndarray],
) -> None:
    if plt is None:  # pragma: no cover
        return
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 9), constrained_layout=True)
    cols = [("GT", gt_counts), ("Pred", pred_counts), ("Random", rand_counts)]
    for i, name in enumerate(stages):
        for j, (title, blob) in enumerate(cols):
            ax = axes[i, j]
            mat = np.asarray(blob[name], dtype=np.float32)
            ax.imshow(np.log1p(mat), cmap="viridis", interpolation="nearest")
            ax.set_title(f"{name}: {title}")
            ax.set_xticks([])
            ax.set_yticks([])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mask-in distribution alignment audit for Macro waypoints (wp1/wp2/end).")
    p.add_argument("--samples_npz", type=str, required=True, help="macro samples.npz (must contain start_pos, targets, z_k_grid).")
    p.add_argument("--nav_file", type=str, required=True, help="nav_field.npz (must contain count).")
    p.add_argument("--count_thr", type=float, default=1.0)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--out_png", type=str, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    rng = np.random.default_rng(int(args.seed))

    samples_npz = Path(args.samples_npz)
    nav_file = Path(args.nav_file)
    K = int(args.patch_size)
    r = int(K // 2)

    inp = _load_inputs(samples_npz)
    nav_count = _load_nav_count(nav_file)
    H, W = int(nav_count.shape[0]), int(nav_count.shape[1])
    drivable = np.asarray(nav_count >= float(args.count_thr), dtype=bool)

    projector = _make_global_projector(nav_count, count_thr=float(args.count_thr))
    clearance_map = None
    if ndimage is not None:
        clearance_map = ndimage.distance_transform_edt(drivable).astype(np.float32)

    N = int(inp.start_pos.shape[0])
    Ks = int(inp.z_k_grid.shape[1])

    stages = ("wp1", "wp2", "end")
    pred_px: Dict[str, list[Tuple[int, int]]] = {k: [] for k in stages}
    gt_raw_px: Dict[str, list[Tuple[int, int]]] = {k: [] for k in stages}
    gt_proj_px: Dict[str, list[Tuple[int, int]]] = {k: [] for k in stages}
    rand_px: Dict[str, list[Tuple[int, int]]] = {k: [] for k in stages}

    pred_clear: Dict[str, list[float]] = {k: [] for k in stages}
    gt_raw_clear: Dict[str, list[float]] = {k: [] for k in stages}
    gt_proj_clear: Dict[str, list[float]] = {k: [] for k in stages}
    rand_clear: Dict[str, list[float]] = {k: [] for k in stages}

    pred_rdist: Dict[str, list[float]] = {k: [] for k in stages}
    gt_raw_rdist: Dict[str, list[float]] = {k: [] for k in stages}
    gt_proj_rdist: Dict[str, list[float]] = {k: [] for k in stages}
    rand_rdist: Dict[str, list[float]] = {k: [] for k in stages}

    pred_logc: Dict[str, list[float]] = {k: [] for k in stages}
    gt_raw_logc: Dict[str, list[float]] = {k: [] for k in stages}
    gt_proj_logc: Dict[str, list[float]] = {k: [] for k in stages}
    rand_logc: Dict[str, list[float]] = {k: [] for k in stages}

    avail = np.zeros((K, K), dtype=np.int64)  # how often a pixel is drivable (per-sample)
    drivable_pixels_per_sample: list[np.ndarray] = []

    # ---------- pass 1: cache per-sample drivable pixels + availability ----------
    for i in range(N):
        c = _patch_center(inp.start_pos[i])
        y0, y1 = int(c[0] - r), int(c[0] + r)
        x0, x1 = int(c[1] - r), int(c[1] + r)

        patch = np.zeros((K, K), dtype=bool)
        iy0, iy1 = max(0, y0), min(H, y1)
        ix0, ix1 = max(0, x0), min(W, x1)
        py0, px0 = int(iy0 - y0), int(ix0 - x0)
        py1, px1 = py0 + int(iy1 - iy0), px0 + int(ix1 - ix0)
        if iy1 > iy0 and ix1 > ix0:
            patch[py0:py1, px0:px1] = drivable[iy0:iy1, ix0:ix1]

        avail += patch.astype(np.int64)
        ys, xs = np.where(patch)
        if ys.size == 0:
            # offline audit should ensure this is ~0; still avoid crash.
            ys = np.asarray([r], dtype=np.int64)
            xs = np.asarray([r], dtype=np.int64)
        drivable_pixels_per_sample.append(np.stack([ys, xs], axis=1).astype(np.int64))

    # ---------- pass 2: collect points ----------
    pred_valid = 0
    pred_total = 0
    gt_raw_valid = 0
    gt_raw_total = 0
    gt_proj_valid = 0
    gt_proj_total = 0

    for i in range(N):
        center = _patch_center(inp.start_pos[i])
        future = inp.targets[i]

        z_gt_raw = _extract_gt_z_raw(inp.start_pos[i], future)
        z_gt_proj = _project_to_drivable(z_gt_raw, nav_count=nav_count, count_thr=float(args.count_thr), projector=projector)

        # Build per-sample strict drivable lookup for validity check (same as gate strict definition).
        drv_px = drivable_pixels_per_sample[i]  # (M,2) patch pixels
        strict = np.zeros((K, K), dtype=bool)
        strict[drv_px[:, 0], drv_px[:, 1]] = True

        # ---- Random baseline: sample 3 points per (i,k) from strict ----
        # (One per stage; independent.)
        for _k in range(Ks):
            for si, name in enumerate(stages):
                pick = drv_px[int(rng.integers(0, drv_px.shape[0]))]
                py, px = int(pick[0]), int(pick[1])
                rand_px[name].append((py, px))
                dy = float(py - r)
                dx = float(px - r)
                rand_rdist[name].append(float(np.hypot(dy, dx)))
                gy = int(center[0] + (py - r))
                gx = int(center[1] + (px - r))
                gy = int(np.clip(gy, 0, H - 1))
                gx = int(np.clip(gx, 0, W - 1))
                rand_logc[name].append(float(np.log1p(nav_count[gy, gx])))
                if clearance_map is not None:
                    rand_clear[name].append(float(clearance_map[gy, gx]))

        # ---- GT (raw / proj) ----
        for si, name in enumerate(stages):
            # raw
            gt_raw_total += 1
            py, px, inb = _to_patch_px(z_gt_raw[si : si + 1], center=center, patch_size=K)
            if bool(inb[0]) and bool(strict[int(py[0]), int(px[0])]):
                gt_raw_valid += 1
                gt_raw_px[name].append((int(py[0]), int(px[0])))
                dy = float(py[0] - r)
                dx = float(px[0] - r)
                gt_raw_rdist[name].append(float(np.hypot(dy, dx)))
                gy = int(np.clip(int(np.rint(z_gt_raw[si, 0])), 0, H - 1))
                gx = int(np.clip(int(np.rint(z_gt_raw[si, 1])), 0, W - 1))
                gt_raw_logc[name].append(float(np.log1p(nav_count[gy, gx])))
                if clearance_map is not None:
                    gt_raw_clear[name].append(float(clearance_map[gy, gx]))

            # proj
            gt_proj_total += 1
            py, px, inb = _to_patch_px(z_gt_proj[si : si + 1], center=center, patch_size=K)
            if bool(inb[0]) and bool(strict[int(py[0]), int(px[0])]):
                gt_proj_valid += 1
                gt_proj_px[name].append((int(py[0]), int(px[0])))
                dy = float(py[0] - r)
                dx = float(px[0] - r)
                gt_proj_rdist[name].append(float(np.hypot(dy, dx)))
                gy = int(np.clip(int(np.rint(z_gt_proj[si, 0])), 0, H - 1))
                gx = int(np.clip(int(np.rint(z_gt_proj[si, 1])), 0, W - 1))
                gt_proj_logc[name].append(float(np.log1p(nav_count[gy, gx])))
                if clearance_map is not None:
                    gt_proj_clear[name].append(float(clearance_map[gy, gx]))

        # ---- Pred (z_k_grid) ----
        for k in range(Ks):
            z_pred = inp.z_k_grid[i, k]  # (3,2)
            for si, name in enumerate(stages):
                pred_total += 1
                py, px, inb = _to_patch_px(z_pred[si : si + 1], center=center, patch_size=K)
                if bool(inb[0]) and bool(strict[int(py[0]), int(px[0])]):
                    pred_valid += 1
                    pred_px[name].append((int(py[0]), int(px[0])))
                    dy = float(py[0] - r)
                    dx = float(px[0] - r)
                    pred_rdist[name].append(float(np.hypot(dy, dx)))
                    gy = int(np.clip(int(np.rint(z_pred[si, 0])), 0, H - 1))
                    gx = int(np.clip(int(np.rint(z_pred[si, 1])), 0, W - 1))
                    pred_logc[name].append(float(np.log1p(nav_count[gy, gx])))
                    if clearance_map is not None:
                        pred_clear[name].append(float(clearance_map[gy, gx]))

    # ---------- aggregate metrics ----------
    def _counts_from_list(lst: list[Tuple[int, int]]) -> np.ndarray:
        if not lst:
            return np.zeros((K, K), dtype=np.int64)
        arr = np.asarray(lst, dtype=np.int64)
        return _heatmap_counts(arr[:, 0], arr[:, 1], K=K)

    pred_counts = {s: _counts_from_list(pred_px[s]) for s in stages}
    gt_raw_counts = {s: _counts_from_list(gt_raw_px[s]) for s in stages}
    gt_proj_counts = {s: _counts_from_list(gt_proj_px[s]) for s in stages}
    rand_counts = {s: _counts_from_list(rand_px[s]) for s in stages}

    # Availability-corrected preference map: P(select pixel | pixel is drivable).
    avail_f = avail.astype(np.float64)
    avail_pred = avail_f * float(max(Ks, 1))
    eps = 1e-9
    support = (avail_f.reshape(-1) > 0)  # pixels that are drivable in at least one sample
    if not bool(np.any(support)):  # pragma: no cover
        support = np.ones_like(support, dtype=bool)

    def _pref_dist(counts: np.ndarray, denom: np.ndarray) -> np.ndarray:
        c = np.asarray(counts, dtype=np.float64).reshape(-1)
        d = np.asarray(denom, dtype=np.float64).reshape(-1)
        c = c[support]
        d = d[support]
        x = c / (d + eps)
        x = x + 1e-12
        x = x / float(x.sum())
        return x

    report: Dict[str, object] = {
        "meta": {
            "samples_npz": str(samples_npz),
            "nav_file": str(nav_file),
            "count_thr": float(args.count_thr),
            "patch_size": int(K),
            "seed": int(args.seed),
            "N": int(N),
            "K_samples": int(Ks),
            "has_scipy": bool(ndimage is not None),
            "has_matplotlib": bool(plt is not None),
            "samples_meta": inp.meta,
        },
        "valid_rates": {
            "pred_valid_rate": float(pred_valid / max(pred_total, 1)),
            "gt_raw_valid_rate": float(gt_raw_valid / max(gt_raw_total, 1)),
            "gt_proj_valid_rate": float(gt_proj_valid / max(gt_proj_total, 1)),
            "avg_drivable_pixels_per_patch": float(np.mean([x.shape[0] for x in drivable_pixels_per_sample])),
        },
        "metrics": {},
    }

    def _summ_1d(x: list[float]) -> Dict[str, float]:
        a = np.asarray(x, dtype=np.float64).reshape(-1)
        if a.size == 0:
            return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
        return {
            "mean": float(np.mean(a)),
            "p50": float(np.percentile(a, 50.0)),
            "p90": float(np.percentile(a, 90.0)),
        }

    for si, name in enumerate(stages):
        c_pred = pred_counts[name].astype(np.int64, copy=False).reshape(-1)[support]
        c_gt_raw = gt_raw_counts[name].astype(np.int64, copy=False).reshape(-1)[support]
        c_gt_proj = gt_proj_counts[name].astype(np.int64, copy=False).reshape(-1)[support]
        c_rand = rand_counts[name].astype(np.int64, copy=False).reshape(-1)[support]

        # Raw count JSD (includes many zeros; OK as a global bias indicator).
        jsd_cnt_raw = jsd_from_hist(c_pred, c_gt_raw)
        jsd_cnt_proj = jsd_from_hist(c_pred, c_gt_proj)
        jsd_cnt_rand_raw = jsd_from_hist(c_rand, c_gt_raw)
        jsd_cnt_rand_proj = jsd_from_hist(c_rand, c_gt_proj)

        # Availability-corrected preference JSD (recommended for interpretation).
        pref_pred = _pref_dist(pred_counts[name], avail_pred)
        pref_gt_raw = _pref_dist(gt_raw_counts[name], avail_f)
        pref_gt_proj = _pref_dist(gt_proj_counts[name], avail_f)
        pref_rand = _pref_dist(rand_counts[name], avail_pred)

        jsd_pref_raw = jsd_from_hist(pref_pred, pref_gt_raw)
        jsd_pref_proj = jsd_from_hist(pref_pred, pref_gt_proj)
        jsd_pref_rand_raw = jsd_from_hist(pref_rand, pref_gt_raw)
        jsd_pref_rand_proj = jsd_from_hist(pref_rand, pref_gt_proj)

        # 1D distributions inside mask (distance/clearance/count).
        jsd_r_raw = compute_jsd_from_samples(pred_rdist[name], gt_raw_rdist[name], bins=40, clamp_min=0.0, clamp_max=float(r * np.sqrt(2)))
        jsd_r_proj = compute_jsd_from_samples(pred_rdist[name], gt_proj_rdist[name], bins=40, clamp_min=0.0, clamp_max=float(r * np.sqrt(2)))
        jsd_r_rand_raw = compute_jsd_from_samples(rand_rdist[name], gt_raw_rdist[name], bins=40, clamp_min=0.0, clamp_max=float(r * np.sqrt(2)))
        jsd_r_rand_proj = compute_jsd_from_samples(rand_rdist[name], gt_proj_rdist[name], bins=40, clamp_min=0.0, clamp_max=float(r * np.sqrt(2)))

        jsd_logc_raw = compute_jsd_from_samples(pred_logc[name], gt_raw_logc[name], bins=50)
        jsd_logc_proj = compute_jsd_from_samples(pred_logc[name], gt_proj_logc[name], bins=50)
        jsd_logc_rand_raw = compute_jsd_from_samples(rand_logc[name], gt_raw_logc[name], bins=50)
        jsd_logc_rand_proj = compute_jsd_from_samples(rand_logc[name], gt_proj_logc[name], bins=50)

        clear = {}
        if clearance_map is not None:
            clear = {
                "JSD_Clearance_raw": compute_jsd_from_samples(pred_clear[name], gt_raw_clear[name], bins=50, clamp_min=0.0),
                "JSD_Clearance_proj": compute_jsd_from_samples(pred_clear[name], gt_proj_clear[name], bins=50, clamp_min=0.0),
                "JSD_Clearance_rand_raw": compute_jsd_from_samples(rand_clear[name], gt_raw_clear[name], bins=50, clamp_min=0.0),
                "JSD_Clearance_rand_proj": compute_jsd_from_samples(rand_clear[name], gt_proj_clear[name], bins=50, clamp_min=0.0),
                "clearance_stats": {
                    "pred": _summ_1d(pred_clear[name]),
                    "gt_raw": _summ_1d(gt_raw_clear[name]),
                    "gt_proj": _summ_1d(gt_proj_clear[name]),
                    "rand": _summ_1d(rand_clear[name]),
                },
            }

        report["metrics"][name] = {
            "heatmap_jsd_counts": {
                "pred_vs_gt_raw": float(jsd_cnt_raw),
                "pred_vs_gt_proj": float(jsd_cnt_proj),
                "rand_vs_gt_raw": float(jsd_cnt_rand_raw),
                "rand_vs_gt_proj": float(jsd_cnt_rand_proj),
            },
            "heatmap_jsd_pref": {
                "pred_vs_gt_raw": float(jsd_pref_raw),
                "pred_vs_gt_proj": float(jsd_pref_proj),
                "rand_vs_gt_raw": float(jsd_pref_rand_raw),
                "rand_vs_gt_proj": float(jsd_pref_rand_proj),
            },
            "entropy": {
                "pred": float(_entropy_from_counts(c_pred)),
                "gt_raw": float(_entropy_from_counts(c_gt_raw)),
                "gt_proj": float(_entropy_from_counts(c_gt_proj)),
                "rand": float(_entropy_from_counts(c_rand)),
            },
            "JSD_Rdist_raw": float(jsd_r_raw),
            "JSD_Rdist_proj": float(jsd_r_proj),
            "JSD_Rdist_rand_raw": float(jsd_r_rand_raw),
            "JSD_Rdist_rand_proj": float(jsd_r_rand_proj),
            "rdist_stats": {
                "pred": _summ_1d(pred_rdist[name]),
                "gt_raw": _summ_1d(gt_raw_rdist[name]),
                "gt_proj": _summ_1d(gt_proj_rdist[name]),
                "rand": _summ_1d(rand_rdist[name]),
            },
            "JSD_LogCount_raw": float(jsd_logc_raw),
            "JSD_LogCount_proj": float(jsd_logc_proj),
            "JSD_LogCount_rand_raw": float(jsd_logc_rand_raw),
            "JSD_LogCount_rand_proj": float(jsd_logc_rand_proj),
            "logcount_stats": {
                "pred": _summ_1d(pred_logc[name]),
                "gt_raw": _summ_1d(gt_raw_logc[name]),
                "gt_proj": _summ_1d(gt_proj_logc[name]),
                "rand": _summ_1d(rand_logc[name]),
            },
            **clear,
        }

    if args.out_png:
        _plot_heatmaps(
            out_png=Path(args.out_png),
            K=K,
            stages=stages,
            gt_counts=gt_proj_counts,  # visualize projected GT for cleaner comparison
            pred_counts=pred_counts,
            rand_counts=rand_counts,
        )

    print("[OK] Macro mask alignment")
    print(json.dumps(report["valid_rates"], indent=2, ensure_ascii=False))
    # Print a compact summary (wp2/end usually most informative).
    for name in stages:
        m = report["metrics"][name]
        print(
            f"- {name}: "
            f"JSD_pref(pred,gt_proj)={m['heatmap_jsd_pref']['pred_vs_gt_proj']:.4f}, "
            f"JSD_rdist(pred,gt_proj)={m['JSD_Rdist_proj']:.4f}, "
            f"JSD_logc(pred,gt_proj)={m['JSD_LogCount_proj']:.4f}"
        )

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[OK] saved: {out}")


if __name__ == "__main__":
    main()
