"""
Storytelling case-study figure (journal style): fixed columns, multiple rows.

Motivation (PI review):
- Do NOT overlay different CFG settings in one subplot.
- Use "One case per row, different models per column" to make the narrative obvious:
  Prior (deterministic anchor) -> CFG2 (diverse) -> CFG3 (more macro-valid / controllable).
- Enforce a minimum map span (e.g., 3km x 3km) so short/long cases have consistent visual scale.
- Hide lat/lon ticks and use a scale bar (map language).
- Use a simple glow effect for spaghetti bundles (better aesthetics).

Inputs:
- --stats_path: data_stats.json with grid bbox mapping
- --sample "Label:/path/to/samples.npz" repeated; requires labels: Prior, CFG2, CFG3
  - Prior file needs `preds (N,F,2)` + `targets (N,F,2)`
  - CFG2/CFG3 ideally contain `preds_k (N,K,F,2)` saved by evaluate.py --save_all_k
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.visualization.basemap import BasemapStyle, draw_geojson_basemap
from src.visualization.style_config import get_color, set_style


@dataclass(frozen=True)
class GridConfig:
    H: int
    W: int
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


def _load_grid_config(stats_path: Path) -> GridConfig:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    grid = stats.get("grid_config") or {}
    required = ["H", "W", "min_lat", "max_lat", "min_lon", "max_lon"]
    missing = [k for k in required if k not in grid]
    if missing:
        raise ValueError(f"Missing grid_config keys {missing} in {stats_path}")
    return GridConfig(
        H=int(grid["H"]),
        W=int(grid["W"]),
        min_lat=float(grid["min_lat"]),
        max_lat=float(grid["max_lat"]),
        min_lon=float(grid["min_lon"]),
        max_lon=float(grid["max_lon"]),
    )


def _grid_yx_to_latlon(yx: np.ndarray, grid: GridConfig, flip_y: bool = False) -> np.ndarray:
    y = yx[..., 0].astype(np.float64, copy=False)
    x = yx[..., 1].astype(np.float64, copy=False)

    denom_y = max(grid.H - 1, 1)
    denom_x = max(grid.W - 1, 1)
    y01 = np.clip(y / denom_y, 0.0, 1.0)
    x01 = np.clip(x / denom_x, 0.0, 1.0)
    if flip_y:
        y01 = 1.0 - y01

    lat = grid.min_lat + y01 * (grid.max_lat - grid.min_lat)
    lon = grid.min_lon + x01 * (grid.max_lon - grid.min_lon)
    return np.stack([lat, lon], axis=-1)


def _aspect_for_latlon(grid: GridConfig) -> float:
    mean_lat = 0.5 * (grid.min_lat + grid.max_lat)
    return 1.0 / max(np.cos(np.deg2rad(mean_lat)), 1e-6)


def _parse_sample_arg(raw: str) -> Tuple[str, Path]:
    if ":" not in raw:
        raise ValueError(f"Invalid --sample '{raw}'. Expected 'Label:Path'.")
    label, path = raw.split(":", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --sample '{raw}'. Expected 'Label:Path'.")
    return label, Path(path)


@dataclass(frozen=True)
class SampleFile:
    name: str
    preds: np.ndarray  # (N,F,2) grid yx
    targets: np.ndarray  # (N,F,2)
    preds_k: Optional[np.ndarray]  # (N,K,F,2)


def _load_samples(path: Path, name: str) -> SampleFile:
    data = np.load(path)
    preds = np.asarray(data["preds"], dtype=np.float32)
    targets = np.asarray(data["targets"], dtype=np.float32)
    preds_k = np.asarray(data["preds_k"], dtype=np.float32) if "preds_k" in data.files else None
    return SampleFile(name=name, preds=preds, targets=targets, preds_k=preds_k)


def _km_per_deg_lat() -> float:
    return 111.32


def _km_per_deg_lon(mean_lat: float) -> float:
    return 111.32 * max(np.cos(np.deg2rad(mean_lat)), 1e-6)


def _e2e_disp_km(latlon: np.ndarray) -> float:
    start = latlon[0]
    end = latlon[-1]
    mean_lat = float(0.5 * (start[0] + end[0]))
    dx = float(end[1] - start[1]) * _km_per_deg_lon(mean_lat)
    dy = float(end[0] - start[0]) * _km_per_deg_lat()
    return float(np.sqrt(dx * dx + dy * dy))


def _endpoint_spread_km(preds_k_ll: np.ndarray) -> np.ndarray:
    """
    preds_k_ll: (N,K,F,2) lat/lon
    return: (N,) mean radius (km) of endpoints to their mean endpoint
    """
    endpoints = preds_k_ll[:, :, -1, :]  # (N,K,2)
    mean = endpoints.mean(axis=1, keepdims=True)
    mean_lat = endpoints[..., 0].mean(axis=1, keepdims=True)
    dx = (endpoints[..., 1] - mean[..., 1]) * _km_per_deg_lon(mean_lat)
    dy = (endpoints[..., 0] - mean[..., 0]) * _km_per_deg_lat()
    d = np.sqrt(dx * dx + dy * dy)  # (N,K)
    return d.mean(axis=1)


def _ensure_min_span(extent: Tuple[float, float, float, float], min_span_km: float) -> Tuple[float, float, float, float]:
    x0, x1, y0, y1 = extent  # lon_min, lon_max, lat_min, lat_max
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    mean_lat = float(cy)

    dlon_req = float(min_span_km) / _km_per_deg_lon(mean_lat)
    dlat_req = float(min_span_km) / _km_per_deg_lat()

    w = float(x1 - x0)
    h = float(y1 - y0)
    w_new = max(w, dlon_req)
    h_new = max(h, dlat_req)
    return (cx - 0.5 * w_new, cx + 0.5 * w_new, cy - 0.5 * h_new, cy + 0.5 * h_new)


def _compute_extent(latlon_list: List[np.ndarray], pad_frac: float, min_span_km: float) -> Tuple[float, float, float, float]:
    lat = np.concatenate([a[..., 0].reshape(-1) for a in latlon_list], axis=0)
    lon = np.concatenate([a[..., 1].reshape(-1) for a in latlon_list], axis=0)
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    dlat = max(lat_max - lat_min, 1e-6)
    dlon = max(lon_max - lon_min, 1e-6)
    lat_pad = float(pad_frac) * dlat
    lon_pad = float(pad_frac) * dlon
    extent = (lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad)
    return _ensure_min_span(extent, min_span_km=float(min_span_km))


def _add_scalebar(ax: plt.Axes, extent: Tuple[float, float, float, float], scalebar_km: float) -> None:
    x0, x1, y0, y1 = extent
    mean_lat = 0.5 * (y0 + y1)
    dlon = float(scalebar_km) / _km_per_deg_lon(mean_lat)
    # Place at bottom-right with small margins
    mx = 0.06 * (x1 - x0)
    my = 0.06 * (y1 - y0)
    xs = x1 - mx - dlon
    xe = x1 - mx
    y = y0 + my
    ax.plot([xs, xe], [y, y], color="black", lw=2.2, solid_capstyle="butt", zorder=10)
    ax.text(
        0.5 * (xs + xe),
        y - 0.02 * (y1 - y0),
        f"{float(scalebar_km):g} km",
        ha="center",
        va="top",
        fontsize=9,
        color="black",
        zorder=10,
    )


def _pick_k_idx(K: int, k_plot: int) -> np.ndarray:
    take_k = min(int(k_plot), int(K))
    if take_k <= 1:
        return np.array([0], dtype=int)
    return np.linspace(0, K - 1, num=take_k, dtype=int)


def _auto_select_cases(
    gt_ll_all: np.ndarray,  # (N,F,2)
    cfg3_preds_k_ll: Optional[np.ndarray],  # (N,K,F,2)
    num_rows: int,
    seed: int,
) -> List[int]:
    rng = np.random.default_rng(int(seed))
    N = int(gt_ll_all.shape[0])
    if N == 0:
        return []

    disp = np.array([_e2e_disp_km(gt_ll_all[i]) for i in range(N)], dtype=np.float64)

    candidates = np.arange(N, dtype=int)
    if cfg3_preds_k_ll is not None:
        spread = _endpoint_spread_km(cfg3_preds_k_ll)
        # Keep the top ~50% by branching to avoid boring cases.
        thr = float(np.quantile(spread, 0.5))
        candidates = np.where(spread >= thr)[0].astype(int)
        if candidates.size == 0:
            candidates = np.arange(N, dtype=int)

    # Target quantiles for (short, mid, long)
    qs = [0.2, 0.5, 0.8]
    target = [float(np.quantile(disp[candidates], q)) for q in qs]

    picked: List[int] = []
    remaining = set(int(i) for i in candidates.tolist())
    for t in target:
        if not remaining:
            break
        rem = np.array(sorted(remaining), dtype=int)
        i = int(rem[np.argmin(np.abs(disp[rem] - t))])
        picked.append(i)
        remaining.remove(i)

    # Fallback: fill with random distinct cases.
    while len(picked) < int(num_rows) and len(picked) < N:
        i = int(rng.integers(0, N))
        if i not in picked:
            picked.append(i)
    return picked[: int(num_rows)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_path", type=str, required=True)
    parser.add_argument("--sample", action="append", default=[], help="Repeatable 'Label:/path/to/samples.npz'")
    parser.add_argument("--out_dir", type=str, default="essay/figures/stage_cfg")
    parser.add_argument("--stem", type=str, default="fig_geo_story_grid")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--k_plot", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flip_y", action="store_true")
    parser.add_argument("--pad_frac", type=float, default=0.12)
    parser.add_argument("--min_span_km", type=float, default=3.0)
    parser.add_argument("--scalebar_km", type=float, default=1.0)
    parser.add_argument("--case_idx", action="append", default=[], help="Explicit case indices (repeatable).")

    parser.add_argument("--basemap_geojson", type=str, default=None)
    parser.add_argument("--basemap_edgecolor", type=str, default="#3A3A3A")
    parser.add_argument("--basemap_facecolor", type=str, default="none")
    parser.add_argument("--basemap_linewidth", type=float, default=0.7)
    parser.add_argument("--basemap_alpha", type=float, default=0.55)
    parser.add_argument("--basemap_labels", action="store_true")
    parser.add_argument("--basemap_label_size", type=int, default=8)
    parser.add_argument("--basemap_label_lang", type=str, choices=["en", "raw"], default="en")

    parser.add_argument("--png_only", action="store_true")
    parser.add_argument("--style", type=str, choices=["paper", "talk"], default="paper")
    args = parser.parse_args()

    if not args.sample:
        raise ValueError("No samples provided. Use --sample 'Label:Path' (repeatable).")

    set_style(context=str(args.style), font_scale=1.0)
    grid = _load_grid_config(Path(args.stats_path))
    aspect = _aspect_for_latlon(grid)

    # Load required models
    files: Dict[str, SampleFile] = {}
    target_ref: Optional[np.ndarray] = None
    for raw in args.sample:
        label, path = _parse_sample_arg(str(raw))
        sf = _load_samples(path, name=label)
        if target_ref is None:
            target_ref = sf.targets
        else:
            if sf.targets.shape != target_ref.shape or np.max(np.abs(sf.targets - target_ref)) > 1e-6:
                raise ValueError(f"targets mismatch across sample files; ensure aligned subsets. Offender: {path}")
        files[label] = sf

    for need in ("Prior", "CFG2", "CFG3"):
        if need not in files:
            raise ValueError(f"Missing required label '{need}'. Provide --sample '{need}:.../samples.npz'")

    assert target_ref is not None
    N = int(target_ref.shape[0])

    # Determine case indices
    if args.case_idx:
        raw_idx = [int(x) for x in args.case_idx]
        raw_idx = [i for i in raw_idx if 0 <= i < N]
        if not raw_idx:
            raise ValueError(f"--case_idx provided but all indices are out of range [0, {N-1}]")
        case_idx = raw_idx[: int(args.rows)]
    else:
        gt_ll_all = _grid_yx_to_latlon(target_ref, grid, flip_y=bool(args.flip_y))
        cfg3 = files["CFG3"]
        cfg3_k_ll = (
            _grid_yx_to_latlon(cfg3.preds_k, grid, flip_y=bool(args.flip_y)) if cfg3.preds_k is not None else None
        )
        case_idx = _auto_select_cases(gt_ll_all, cfg3_k_ll, num_rows=int(args.rows), seed=int(args.seed))

    rows = int(len(case_idx))
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.0 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.array([axes])
    basemap_geojson = Path(args.basemap_geojson) if args.basemap_geojson else None
    basemap_style = BasemapStyle(
        edgecolor=str(args.basemap_edgecolor),
        facecolor=str(args.basemap_facecolor),
        linewidth=float(args.basemap_linewidth),
        alpha=float(args.basemap_alpha),
        label=bool(args.basemap_labels),
        label_size=int(args.basemap_label_size),
        label_lang=str(args.basemap_label_lang),
    )

    col_titles = ["Prior (anchor)", "Residual (CFG=2)", "Residual (CFG=3)"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=12, pad=6)

    prior = files["Prior"]
    cfg2 = files["CFG2"]
    cfg3 = files["CFG3"]

    for i, idx in enumerate(case_idx):
        gt_ll = _grid_yx_to_latlon(prior.targets[idx], grid, flip_y=bool(args.flip_y))
        prior_ll = _grid_yx_to_latlon(prior.preds[idx], grid, flip_y=bool(args.flip_y))

        ll_all: List[np.ndarray] = [gt_ll, prior_ll]

        def get_spaghetti(sf: SampleFile) -> List[np.ndarray]:
            if sf.preds_k is None:
                return [_grid_yx_to_latlon(sf.preds[idx], grid, flip_y=bool(args.flip_y))]
            K = int(sf.preds_k.shape[1])
            k_idx = _pick_k_idx(K, int(args.k_plot))
            out = []
            for k in k_idx:
                out.append(_grid_yx_to_latlon(sf.preds_k[idx, k], grid, flip_y=bool(args.flip_y)))
            return out

        cfg2_ll = get_spaghetti(cfg2)
        cfg3_ll = get_spaghetti(cfg3)
        ll_all.extend(cfg2_ll)
        ll_all.extend(cfg3_ll)

        extent = _compute_extent(ll_all, pad_frac=float(args.pad_frac), min_span_km=float(args.min_span_km))

        # Row header inside the first axis (English)
        disp_km = _e2e_disp_km(gt_ll)
        axes[i, 0].text(
            0.02,
            0.98,
            f"Case #{int(idx)}  (GT disp ≈ {disp_km:.1f} km)",
            transform=axes[i, 0].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#111111",
        )

        # Draw each column
        for j, (name, spaghetti) in enumerate(
            [
                ("Prior", [prior_ll]),
                ("CFG2", cfg2_ll),
                ("CFG3", cfg3_ll),
            ]
        ):
            ax = axes[i, j]
            draw_geojson_basemap(ax, basemap_geojson, basemap_style, zorder_base=0)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect(aspect)

            # Map language: remove lat/lon ticks
            ax.set_axis_off()

            # GT (thick)
            ax.plot(gt_ll[:, 1], gt_ll[:, 0], color="#111111", lw=3.0, alpha=0.95, zorder=6)

            # Prior (dashed, background reference in all columns)
            ax.plot(prior_ll[:, 1], prior_ll[:, 0], color="#444444", ls="--", lw=2.0, alpha=0.85, zorder=5)

            # Spaghetti (glow)
            if name in ("CFG2", "CFG3"):
                color = get_color(name)
                for pr_ll in spaghetti:
                    ax.plot(pr_ll[:, 1], pr_ll[:, 0], color=color, lw=3.0, alpha=0.10, zorder=4)
                    ax.plot(pr_ll[:, 1], pr_ll[:, 0], color=color, lw=0.85, alpha=0.55, zorder=7)

            # Start/end markers (GT)
            ax.scatter(gt_ll[0, 1], gt_ll[0, 0], s=26, marker="*", color="#111111", zorder=8)
            ax.scatter(gt_ll[-1, 1], gt_ll[-1, 0], s=18, marker="o", color="#111111", zorder=8)

            _add_scalebar(ax, extent, scalebar_km=float(args.scalebar_km))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{str(args.stem)}.png"
    fig.savefig(png, dpi=300)
    print(f"[OK] saved {png}")
    if not bool(args.png_only):
        pdf = out_dir / f"{str(args.stem)}.pdf"
        fig.savefig(pdf)
        print(f"[OK] saved {pdf}")


if __name__ == "__main__":
    main()

