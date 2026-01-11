from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.distribution_metrics import compute_jsd_from_samples
from src.plot_style import FIGSIZE_FULL, OKABE_ITO, add_panel_label, paper_style, save_figure
from src.training.route_npz_utils import load_route_windows_npz
from src.utils.geo_grid import BBox, GridSpec


@dataclass(frozen=True)
class Config:
    road_prob_thr: float
    hist_bins_len: int
    hist_bins_detour: int
    hist_bins_onroad: int
    smooth_sigma_bins: float
    seed: int


def _key64(traj_idx: np.ndarray, start_t: np.ndarray) -> np.ndarray:
    traj_idx = np.asarray(traj_idx, dtype=np.int64).reshape(-1)
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    return (traj_idx << np.int64(32)) | (start_t & np.int64(0xFFFFFFFF))


def _resolve_common_keys(
    gt: dict,
    model_npz_list: Sequence[np.lib.npyio.NpzFile],
) -> Tuple[List[int], List[np.ndarray]]:
    gt_key = _key64(np.asarray(gt["traj_idx"]), np.asarray(gt["start_t"]))
    gt_map = {int(k): int(i) for i, k in enumerate(gt_key.tolist())}
    common = set(gt_map.keys())

    for ms in model_npz_list:
        ms_key = _key64(np.asarray(ms["traj_idx"]), np.asarray(ms["start_t"]))
        ms_map = {int(k): int(i) for i, k in enumerate(ms_key.tolist())}
        common &= set(ms_map.keys())

    keys = sorted(list(common))
    if not keys:
        raise RuntimeError("No matched windows across gt_windows_npz and model_samples_npz (traj_idx/start_t mismatch).")

    gt_idx = [gt_map[int(k)] for k in keys]

    # Rebuild indices per model (keep ordering identical to keys).
    model_idx: List[np.ndarray] = []
    for ms in model_npz_list:
        ms_key = _key64(np.asarray(ms["traj_idx"]), np.asarray(ms["start_t"]))
        ms_map = {int(k): int(i) for i, k in enumerate(ms_key.tolist())}
        model_idx.append(np.asarray([ms_map[int(k)] for k in keys], dtype=np.int64))
    return gt_idx, model_idx


def _load_grid_from_osm_meta(osm_meta_json: Path) -> GridSpec:
    meta = json.loads(osm_meta_json.read_text(encoding="utf-8"))
    g = meta.get("grid", {})
    bbox = g.get("bbox", {})
    return GridSpec(
        H=int(g["H"]),
        W=int(g["W"]),
        bbox=BBox(
            min_lon=float(bbox["min_lon"]),
            min_lat=float(bbox["min_lat"]),
            max_lon=float(bbox["max_lon"]),
            max_lat=float(bbox["max_lat"]),
        ),
    )


def _gaussian_smooth_1d_fft(y: np.ndarray, sigma_bins: float) -> np.ndarray:
    sigma = float(sigma_bins)
    if sigma <= 0.0:
        return np.asarray(y, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(y.shape[0])
    f = np.fft.rfftfreq(n)
    kernel_ft = np.exp(-2.0 * (np.pi**2) * (sigma**2) * (f**2))
    out = np.fft.irfft(np.fft.rfft(y) * kernel_ft, n=n)
    return np.clip(out, 0.0, None)


def _choose_range(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    p_lo: float = 0.5,
    p_hi: float = 99.5,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
) -> Tuple[float, float]:
    x = np.concatenate([a.reshape(-1), b.reshape(-1), c.reshape(-1)], axis=0).astype(np.float64, copy=False)
    x = x[np.isfinite(x)]
    if x.size <= 0:
        return 0.0, 1.0
    lo, hi = np.percentile(x, [float(p_lo), float(p_hi)]).tolist()
    if clamp_min is not None:
        lo = max(float(lo), float(clamp_min))
    if clamp_max is not None:
        hi = min(float(hi), float(clamp_max))
    if not np.isfinite(lo) or not np.isfinite(hi) or float(hi) <= float(lo):
        lo = float(np.min(x))
        hi = float(np.max(x))
    if float(hi) <= float(lo):
        hi = float(lo) + 1e-6
    return float(lo), float(hi)


def _hist_density_smooth(samples: np.ndarray, *, bins: int, value_range: Tuple[float, float], sigma_bins: float) -> Tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]
    counts, edges = np.histogram(samples, bins=int(bins), range=value_range, density=True)
    counts_s = _gaussian_smooth_1d_fft(counts, sigma_bins=float(sigma_bins))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.astype(np.float64, copy=False), counts_s.astype(np.float64, copy=False)


def _path_length_m_gt(
    *,
    start_pos: np.ndarray,  # (N,2)
    targets: np.ndarray,  # (N,F,2)
    res_y_m: float,
    res_x_m: float,
) -> np.ndarray:
    start_pos = np.asarray(start_pos, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    first = targets[:, 0, :] - start_pos
    rest = targets[:, 1:, :] - targets[:, :-1, :]
    seg0 = np.linalg.norm(first * np.asarray([res_y_m, res_x_m], dtype=np.float32), axis=-1)
    seg_rest = np.linalg.norm(rest * np.asarray([res_y_m, res_x_m], dtype=np.float32), axis=-1).sum(axis=1)
    return (seg0 + seg_rest).astype(np.float64, copy=False)


def _path_length_m_pred(
    *,
    start_pos: np.ndarray,  # (N,2)
    preds_k: np.ndarray,  # (N,K,F,2)
    res_y_m: float,
    res_x_m: float,
) -> np.ndarray:
    start_pos = np.asarray(start_pos, dtype=np.float32)
    preds_k = np.asarray(preds_k, dtype=np.float32)
    first = preds_k[:, :, 0, :] - start_pos[:, None, :]
    rest = preds_k[:, :, 1:, :] - preds_k[:, :, :-1, :]
    scale = np.asarray([res_y_m, res_x_m], dtype=np.float32)
    seg0 = np.linalg.norm(first * scale[None, None, :], axis=-1)
    seg_rest = np.linalg.norm(rest * scale[None, None, None, :], axis=-1).sum(axis=2)
    return (seg0 + seg_rest).astype(np.float64, copy=False)  # (N,K)


def _chord_m(*, start_pos: np.ndarray, dest_pos: np.ndarray, res_y_m: float, res_x_m: float) -> np.ndarray:
    start_pos = np.asarray(start_pos, dtype=np.float32)
    dest_pos = np.asarray(dest_pos, dtype=np.float32)
    diff = dest_pos - start_pos
    scale = np.asarray([res_y_m, res_x_m], dtype=np.float32)
    d = np.linalg.norm(diff * scale[None, :], axis=-1)
    return d.astype(np.float64, copy=False)


def _safe_detour_ratio(length_m: np.ndarray, chord_m: np.ndarray) -> np.ndarray:
    length_m = np.asarray(length_m, dtype=np.float64)
    chord_m = np.asarray(chord_m, dtype=np.float64)
    denom = np.maximum(chord_m, 1e-6)
    return (length_m / denom).astype(np.float64, copy=False)


def _sample_road_prob(road_prob: np.ndarray, pos: np.ndarray) -> np.ndarray:
    road_prob = np.asarray(road_prob, dtype=np.float32)
    if road_prob.ndim != 2:
        raise ValueError(f"Expected road_prob (H,W), got {road_prob.shape}")
    H, W = int(road_prob.shape[0]), int(road_prob.shape[1])

    pos = np.asarray(pos, dtype=np.float32)
    yy = np.rint(pos[..., 0]).astype(np.int64)
    xx = np.rint(pos[..., 1]).astype(np.int64)
    inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    yy_c = np.clip(yy, 0, H - 1)
    xx_c = np.clip(xx, 0, W - 1)
    out = road_prob[yy_c, xx_c].astype(np.float32, copy=False)
    out = out * inb.astype(np.float32, copy=False)
    return out


def _route_onroad_rate_from_road_prob(road_prob_seq: np.ndarray, *, thr: float) -> np.ndarray:
    road_prob_seq = np.asarray(road_prob_seq, dtype=np.float32)
    on = (road_prob_seq >= float(thr)).astype(np.float32, copy=False)
    return np.mean(on, axis=-1).astype(np.float64, copy=False)


def _plot_map_panel(
    ax: plt.Axes,
    *,
    title: str,
    road_prob: Optional[np.ndarray],
    start_pos: np.ndarray,  # (2,)
    dest_pos: np.ndarray,  # (2,)
    gt_targets: np.ndarray,  # (F,2)
    preds_k: np.ndarray,  # (K,F,2)
    k_plot: int,
    crop_margin: int,
) -> Dict[str, object]:
    start_pos = np.asarray(start_pos, dtype=np.float32).reshape(2)
    dest_pos = np.asarray(dest_pos, dtype=np.float32).reshape(2)
    gt_targets = np.asarray(gt_targets, dtype=np.float32).reshape(-1, 2)
    preds_k = np.asarray(preds_k, dtype=np.float32).reshape(preds_k.shape[0], -1, 2)

    k_plot = int(min(int(k_plot), int(preds_k.shape[0])))
    pts = np.concatenate([start_pos[None, :], dest_pos[None, :], gt_targets, preds_k[:k_plot].reshape(-1, 2)], axis=0)
    y0 = int(np.floor(np.min(pts[:, 0]) - float(crop_margin)))
    y1 = int(np.ceil(np.max(pts[:, 0]) + float(crop_margin)))
    x0 = int(np.floor(np.min(pts[:, 1]) - float(crop_margin)))
    x1 = int(np.ceil(np.max(pts[:, 1]) + float(crop_margin)))
    if road_prob is not None:
        H, W = int(road_prob.shape[0]), int(road_prob.shape[1])
        y0 = max(0, min(H - 1, y0))
        y1 = max(0, min(H - 1, y1))
        x0 = max(0, min(W - 1, x0))
        x1 = max(0, min(W - 1, x1))
        if y1 <= y0:
            y1 = min(H - 1, y0 + 1)
        if x1 <= x0:
            x1 = min(W - 1, x0 + 1)

    if road_prob is not None:
        crop = road_prob[y0 : y1 + 1, x0 : x1 + 1]
        # extent=(left,right,bottom,top) with origin='upper' -> y axis goes downward (matches grid y).
        extent = (float(x0) - 0.5, float(x1) + 0.5, float(y1) + 0.5, float(y0) - 0.5)
        ax.imshow(crop, cmap="Greys", vmin=0.0, vmax=1.0, alpha=0.25, origin="upper", extent=extent, interpolation="nearest")

    # GT (gray) + Ours samples (blue).
    ax.plot(gt_targets[:, 1], gt_targets[:, 0], color=OKABE_ITO["gray"], lw=2.0, alpha=0.9, label="GT")
    for kk in range(k_plot):
        ax.plot(preds_k[kk, :, 1], preds_k[kk, :, 0], color=OKABE_ITO["blue"], lw=1.2, alpha=0.18)

    ax.scatter([start_pos[1]], [start_pos[0]], s=18, color=OKABE_ITO["black"], marker="o", zorder=5)
    ax.scatter([dest_pos[1]], [dest_pos[0]], s=22, color=OKABE_ITO["black"], marker="x", zorder=5)

    ax.set_title(str(title))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(x0), float(x1))
    ax.set_ylim(float(y1), float(y0))  # y axis downward
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    return {"crop": {"y0": int(y0), "y1": int(y1), "x0": int(x0), "x1": int(x1)}, "k_plot": int(k_plot)}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Figure 4: Realistic route generation validation (map + distribution alignment), JSON-only outputs.")
    p.add_argument("--gt_windows_npz", type=str, required=True)
    p.add_argument("--ours_samples_npz", type=str, required=True)
    p.add_argument("--e2e_samples_npz", type=str, required=True)
    p.add_argument("--semantic_dir", type=str, default=None, help="If set, uses osm_road_prob.npy (+ meta json if present) for on-road + basemap.")
    p.add_argument("--case_npz", type=str, action="append", default=[], help="Optional: repeatable case_XX/gt_case.npz for map panel.")
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--road_prob_thr", type=float, default=0.5)
    p.add_argument("--k_plot", type=int, default=20)
    p.add_argument("--crop_margin", type=int, default=40)
    p.add_argument("--smooth_sigma_bins", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config(
        road_prob_thr=float(args.road_prob_thr),
        hist_bins_len=60,
        hist_bins_detour=60,
        hist_bins_onroad=40,
        smooth_sigma_bins=float(args.smooth_sigma_bins),
        seed=int(args.seed),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_route_windows_npz(str(args.gt_windows_npz), max_n=None, seed=int(cfg.seed))
    gt_start = np.asarray(gt["start_pos"], dtype=np.float32)
    gt_targets = np.asarray(gt["targets"], dtype=np.float32)
    gt_dest = np.asarray(gt["dest_pos"], dtype=np.float32)
    gt_traj_idx = np.asarray(gt["traj_idx"], dtype=np.int64)
    gt_start_t = np.asarray(gt["start_t"], dtype=np.int64)

    ours = np.load(str(args.ours_samples_npz), allow_pickle=True)
    e2e = np.load(str(args.e2e_samples_npz), allow_pickle=True)
    for ms, name in ((ours, "ours"), (e2e, "e2e")):
        if "preds_k" not in ms.files:
            raise ValueError(f"{name}_samples_npz missing preds_k: {sorted(list(ms.files))}")
        if "traj_idx" not in ms.files or "start_t" not in ms.files or "start_pos" not in ms.files:
            raise ValueError(f"{name}_samples_npz missing traj_idx/start_t/start_pos: {sorted(list(ms.files))}")

    gt_idx, ms_idx_list = _resolve_common_keys(gt, [ours, e2e])
    ours_idx, e2e_idx = ms_idx_list[0], ms_idx_list[1]

    # Load optional OSM prior (for on-road + basemap).
    road_prob = None
    grid = None
    res_y_m, res_x_m = 1.0, 1.0
    semantic_dir = Path(args.semantic_dir) if args.semantic_dir else None
    if semantic_dir is not None:
        rp_path = semantic_dir / "osm_road_prob.npy"
        if not rp_path.exists():
            raise FileNotFoundError(rp_path)
        road_prob = np.load(str(rp_path)).astype(np.float32, copy=False)
        meta_path = semantic_dir / "osm_road_prob_meta.json"
        if meta_path.exists():
            grid = _load_grid_from_osm_meta(meta_path)
            res_y_m, res_x_m = grid.resolution_m()

    # ---- Distribution metrics (all matched windows) ----
    start_m = gt_start[np.asarray(gt_idx, dtype=np.int64)]
    targets_m = gt_targets[np.asarray(gt_idx, dtype=np.int64)]
    dest_m = gt_dest[np.asarray(gt_idx, dtype=np.int64)]
    ours_preds = np.asarray(ours["preds_k"], dtype=np.float32)[ours_idx]
    e2e_preds = np.asarray(e2e["preds_k"], dtype=np.float32)[e2e_idx]

    len_gt_m = _path_length_m_gt(start_pos=start_m, targets=targets_m, res_y_m=res_y_m, res_x_m=res_x_m)
    len_ours_m = _path_length_m_pred(start_pos=start_m, preds_k=ours_preds, res_y_m=res_y_m, res_x_m=res_x_m).reshape(-1)
    len_e2e_m = _path_length_m_pred(start_pos=start_m, preds_k=e2e_preds, res_y_m=res_y_m, res_x_m=res_x_m).reshape(-1)
    chord_m = _chord_m(start_pos=start_m, dest_pos=dest_m, res_y_m=res_y_m, res_x_m=res_x_m)
    det_gt = _safe_detour_ratio(len_gt_m, chord_m)
    det_ours = _safe_detour_ratio(len_ours_m, np.repeat(chord_m, repeats=int(ours_preds.shape[1]), axis=0))
    det_e2e = _safe_detour_ratio(len_e2e_m, np.repeat(chord_m, repeats=int(e2e_preds.shape[1]), axis=0))

    on_gt = None
    on_ours = None
    on_e2e = None
    if road_prob is not None:
        rp_gt = _sample_road_prob(road_prob, targets_m)  # (N,F)
        on_gt = _route_onroad_rate_from_road_prob(rp_gt, thr=float(cfg.road_prob_thr))  # (N,)
        rp_ours = _sample_road_prob(road_prob, ours_preds)  # (N,K,F)
        on_ours = _route_onroad_rate_from_road_prob(rp_ours, thr=float(cfg.road_prob_thr)).reshape(-1)  # (N*K,)
        rp_e2e = _sample_road_prob(road_prob, e2e_preds)  # (N,K,F)
        on_e2e = _route_onroad_rate_from_road_prob(rp_e2e, thr=float(cfg.road_prob_thr)).reshape(-1)

    # Hist ranges (robust, shared).
    len_range_m = _choose_range(len_gt_m, len_ours_m, len_e2e_m, clamp_min=0.0, clamp_max=None)
    det_range = _choose_range(det_gt, det_ours, det_e2e, clamp_min=1.0, clamp_max=None)
    on_range = (0.0, 1.0)

    jsd_len_ours = compute_jsd_from_samples(
        len_ours_m / 1000.0,
        len_gt_m / 1000.0,
        bins=int(cfg.hist_bins_len),
        value_range=(len_range_m[0] / 1000.0, len_range_m[1] / 1000.0),
    )
    jsd_len_e2e = compute_jsd_from_samples(
        len_e2e_m / 1000.0,
        len_gt_m / 1000.0,
        bins=int(cfg.hist_bins_len),
        value_range=(len_range_m[0] / 1000.0, len_range_m[1] / 1000.0),
    )
    jsd_det_ours = compute_jsd_from_samples(det_ours, det_gt, bins=int(cfg.hist_bins_detour), value_range=det_range)
    jsd_det_e2e = compute_jsd_from_samples(det_e2e, det_gt, bins=int(cfg.hist_bins_detour), value_range=det_range)
    jsd_on_ours = None
    jsd_on_e2e = None
    if on_gt is not None and on_ours is not None and on_e2e is not None:
        jsd_on_ours = compute_jsd_from_samples(on_ours, on_gt, bins=int(cfg.hist_bins_onroad), value_range=on_range, clamp_min=0.0, clamp_max=1.0)
        jsd_on_e2e = compute_jsd_from_samples(on_e2e, on_gt, bins=int(cfg.hist_bins_onroad), value_range=on_range, clamp_min=0.0, clamp_max=1.0)

    # Plot-ready hist densities.
    x_len, y_len_gt = _hist_density_smooth(len_gt_m / 1000.0, bins=int(cfg.hist_bins_len), value_range=(len_range_m[0] / 1000.0, len_range_m[1] / 1000.0), sigma_bins=float(cfg.smooth_sigma_bins))
    _, y_len_ours = _hist_density_smooth(len_ours_m / 1000.0, bins=int(cfg.hist_bins_len), value_range=(len_range_m[0] / 1000.0, len_range_m[1] / 1000.0), sigma_bins=float(cfg.smooth_sigma_bins))
    _, y_len_e2e = _hist_density_smooth(len_e2e_m / 1000.0, bins=int(cfg.hist_bins_len), value_range=(len_range_m[0] / 1000.0, len_range_m[1] / 1000.0), sigma_bins=float(cfg.smooth_sigma_bins))

    x_det, y_det_gt = _hist_density_smooth(det_gt, bins=int(cfg.hist_bins_detour), value_range=det_range, sigma_bins=float(cfg.smooth_sigma_bins))
    _, y_det_ours = _hist_density_smooth(det_ours, bins=int(cfg.hist_bins_detour), value_range=det_range, sigma_bins=float(cfg.smooth_sigma_bins))
    _, y_det_e2e = _hist_density_smooth(det_e2e, bins=int(cfg.hist_bins_detour), value_range=det_range, sigma_bins=float(cfg.smooth_sigma_bins))

    on_hist = None
    if on_gt is not None and on_ours is not None and on_e2e is not None:
        x_on, y_on_gt = _hist_density_smooth(on_gt, bins=int(cfg.hist_bins_onroad), value_range=on_range, sigma_bins=float(cfg.smooth_sigma_bins))
        _, y_on_ours = _hist_density_smooth(on_ours, bins=int(cfg.hist_bins_onroad), value_range=on_range, sigma_bins=float(cfg.smooth_sigma_bins))
        _, y_on_e2e = _hist_density_smooth(on_e2e, bins=int(cfg.hist_bins_onroad), value_range=on_range, sigma_bins=float(cfg.smooth_sigma_bins))
        on_hist = {"x": x_on, "gt": y_on_gt, "ours": y_on_ours, "e2e": y_on_e2e}

    # ---- Figure (a) map: 3 cases; (b) distributions: length/detour/on-road ----
    case_files = [Path(p) for p in (args.case_npz or [])]
    if not case_files:
        # Default: reuse Gate-0 audited cases if available locally in _sync.
        cand = []
        # Workstation A default (absolute paths).
        cand.extend(
            [
                Path("/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/E0_gt_baseline_detroit_F256_n200k_seed0_od128_n10_sep2/case_01/gt_case.npz"),
                Path("/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/E0_gt_baseline_detroit_F256_n200k_seed0_od128_n10_sep2/case_02/gt_case.npz"),
                Path("/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/E0_gt_baseline_detroit_F256_n200k_seed0_od128_n10_sep2/case_03/gt_case.npz"),
            ]
        )
        # Local WSL fallback (synced artifacts).
        cand.extend(
            [
                Path("_sync/wsA/icml2026_routegen/E0_gt_baseline_detroit_F256_n200k_seed0_od128_n10_sep2/case_01/gt_case.npz"),
                Path("_sync/wsA/icml2026_routegen/E0_gt_baseline_detroit_F256_n200k_seed0_od128_n10_sep2/case_02/gt_case.npz"),
                Path("_sync/wsA/icml2026_routegen/E0_gt_baseline_detroit_F256_n200k_seed0_od128_n10_sep2/case_03/gt_case.npz"),
            ]
        )
        case_files = [p for p in cand if p.exists()]

    case_reports: List[Dict[str, object]] = []
    ours_key = _key64(np.asarray(ours["traj_idx"]), np.asarray(ours["start_t"]))
    ours_map = {int(k): int(j) for j, k in enumerate(ours_key.tolist())}
    with paper_style():
        fig = plt.figure(figsize=FIGSIZE_FULL)
        gs = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[1.15, 1.0], wspace=0.20, hspace=0.22)

        map_axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        stat_axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]

        # Panel labels
        add_panel_label(map_axes[0], "a")
        add_panel_label(stat_axes[0], "b")

        # (a) Map panels
        for i, ax in enumerate(map_axes):
            if i >= len(case_files):
                ax.axis("off")
                continue
            p = case_files[i]
            case = np.load(str(p), allow_pickle=True)
            if not {"start_pos", "dest_pos", "targets", "traj_idx", "start_t"}.issubset(set(case.files)):
                raise ValueError(f"Bad case_npz: {p} missing required fields, got {sorted(list(case.files))}")
            # Use the first window in the case as representative.
            c_start = np.asarray(case["start_pos"], dtype=np.float32)[0]
            c_dest = np.asarray(case["dest_pos"], dtype=np.float32)[0]
            c_targets = np.asarray(case["targets"], dtype=np.float32)[0]
            c_traj = int(np.asarray(case["traj_idx"], dtype=np.int64)[0])
            c_t = int(np.asarray(case["start_t"], dtype=np.int64)[0])
            key = int(_key64(np.asarray([c_traj], dtype=np.int64), np.asarray([c_t], dtype=np.int64))[0])

            if key not in ours_map:
                raise RuntimeError(f"Case window not found in ours_samples_npz by (traj_idx,start_t)=({c_traj},{c_t}).")
            j = int(ours_map[int(key)])
            c_preds = np.asarray(ours["preds_k"], dtype=np.float32)[j]

            rep = _plot_map_panel(
                ax,
                title=f"case_{i+1:02d}",
                road_prob=road_prob,
                start_pos=c_start,
                dest_pos=c_dest,
                gt_targets=c_targets,
                preds_k=c_preds,
                k_plot=int(args.k_plot),
                crop_margin=int(args.crop_margin),
            )
            rep.update({"case_npz": str(p), "traj_idx": int(c_traj), "start_t": int(c_t)})
            case_reports.append(rep)

        # (b) Distribution panels
        col_gt = OKABE_ITO["black"]
        col_ours = OKABE_ITO["blue"]
        col_e2e = OKABE_ITO["vermillion"]

        ax0, ax1, ax2 = stat_axes

        ax0.plot(x_len, y_len_gt, color=col_gt, lw=2.0, label="GT")
        ax0.plot(x_len, y_len_ours, color=col_ours, lw=2.0, label="Ours")
        ax0.plot(x_len, y_len_e2e, color=col_e2e, lw=2.0, label="E2E")
        ax0.set_xlabel("Path length (km)")
        ax0.set_ylabel("Density")
        ax0.set_title("Length")

        ax1.plot(x_det, y_det_gt, color=col_gt, lw=2.0)
        ax1.plot(x_det, y_det_ours, color=col_ours, lw=2.0)
        ax1.plot(x_det, y_det_e2e, color=col_e2e, lw=2.0)
        ax1.set_xlabel("Detour factor (vs straight-line)")
        ax1.set_ylabel("Density")
        ax1.set_title("Detour")

        if on_hist is not None:
            ax2.plot(on_hist["x"], on_hist["gt"], color=col_gt, lw=2.0)
            ax2.plot(on_hist["x"], on_hist["ours"], color=col_ours, lw=2.0)
            ax2.plot(on_hist["x"], on_hist["e2e"], color=col_e2e, lw=2.0)
            ax2.set_xlabel("On-road rate (per route)")
            ax2.set_ylabel("Density")
            ax2.set_title("On-road")
        else:
            ax2.axis("off")

        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.62, 0.01), ncol=3, frameon=False)
        fig.subplots_adjust(bottom=0.12)

        fig_pdf = out_dir / "fig_realism_validation.pdf"
        fig_png = out_dir / "fig_realism_validation.png"
        save_figure(fig, fig_pdf)
        save_figure(fig, fig_png)
        plt.close(fig)

    # ---- JSON report (compact; no huge arrays) ----
    def _summ(x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
        return {"mean": float(np.mean(x)), "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90))}

    report: Dict[str, object] = {
        "gate": "E16 (Figure4 realism validation)",
        "inputs": {
            "gt_windows_npz": str(Path(args.gt_windows_npz).resolve()),
            "ours_samples_npz": str(Path(args.ours_samples_npz).resolve()),
            "e2e_samples_npz": str(Path(args.e2e_samples_npz).resolve()),
            "semantic_dir": (str(semantic_dir.resolve()) if semantic_dir is not None else None),
        },
        "config": {
            "road_prob_thr": float(cfg.road_prob_thr),
            "k_plot": int(args.k_plot),
            "crop_margin": int(args.crop_margin),
            "smooth_sigma_bins": float(cfg.smooth_sigma_bins),
            "seed": int(cfg.seed),
            "detour_definition": "length / straight_line_distance (both in meters from osm bbox resolution)",
        },
        "stats": {"num_windows_matched": int(len(gt_idx)), "K_ours": int(ours_preds.shape[1]), "K_e2e": int(e2e_preds.shape[1]), "F": int(ours_preds.shape[2])},
        "units": {"res_y_m": float(res_y_m), "res_x_m": float(res_x_m)},
        "distributions": {
            "length_km": {
                "jsd": {"ours": float(jsd_len_ours), "e2e": float(jsd_len_e2e)},
                "gt": _summ(len_gt_m / 1000.0),
                "ours": _summ(len_ours_m / 1000.0),
                "e2e": _summ(len_e2e_m / 1000.0),
            },
            "detour_factor": {
                "jsd": {"ours": float(jsd_det_ours), "e2e": float(jsd_det_e2e)},
                "gt": _summ(det_gt),
                "ours": _summ(det_ours),
                "e2e": _summ(det_e2e),
            },
            "onroad_rate": (
                {
                    "jsd": {"ours": (float(jsd_on_ours) if jsd_on_ours is not None else None), "e2e": (float(jsd_on_e2e) if jsd_on_e2e is not None else None)},
                    "gt": _summ(on_gt),
                    "ours": _summ(on_ours),
                    "e2e": _summ(on_e2e),
                }
                if on_gt is not None and on_ours is not None and on_e2e is not None
                else None
            ),
        },
        "figure": {"pdf": str(fig_pdf.resolve()), "png": str(fig_png.resolve())},
        "cases": case_reports,
    }

    out_json = out_dir / "report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "report_json": str(out_json.resolve()), "fig_pdf": str(fig_pdf.resolve()), "fig_png": str(fig_png.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
