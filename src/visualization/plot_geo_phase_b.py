"""
Geographic-space visualizations for Phase B (dt-fixed=30s).

核心目标：
1) 把 evaluate.py 保存的 `samples.npz`（grid 坐标系 [y, x]）映射回经纬度；
2) 输出“子刊风格”的地理空间可视化图件（PDF + PNG），用于 essay/paper：
   - 轨迹叠图（GT vs Pred，多个样本）
   - 空间密度图（Pred heatmap + GT contour）

重要边界（v1）：
- 本项目 v1 仍是 grid-based，不做 map-matching；经纬度仅用于“地理空间展示”，
  映射采用 data_stats.json 中的 bbox 做线性投影（近似）。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.visualization.style_config import get_color, set_style
from src.visualization.basemap import BasemapStyle, draw_geojson_basemap


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
    """
    Convert grid coords [y, x] to lat/lon using a linear bbox mapping.

    Assumption (default):
      y=0 -> min_lat, y=H-1 -> max_lat
      x=0 -> min_lon, x=W-1 -> max_lon

    If your preprocessing used the opposite convention, set --flip_y.
    """
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


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str, png_only: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=300)
    print(f"[OK] saved {png}")
    if not bool(png_only):
        pdf = out_dir / f"{stem}.pdf"
        fig.savefig(pdf)
        print(f"[OK] saved {pdf}")


def _aspect_for_latlon(grid: GridConfig) -> float:
    # Rough correction: 1 degree lon ≈ cos(lat) * 1 degree lat
    mean_lat = 0.5 * (grid.min_lat + grid.max_lat)
    return 1.0 / max(np.cos(np.deg2rad(mean_lat)), 1e-6)


def _km_per_deg_lat() -> float:
    return 111.32


def _km_per_deg_lon(mean_lat: float) -> float:
    return 111.32 * max(np.cos(np.deg2rad(mean_lat)), 1e-6)


def _ensure_min_span(extent: Tuple[float, float, float, float], min_span_km: float) -> Tuple[float, float, float, float]:
    if float(min_span_km) <= 0:
        return extent
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


def _add_scalebar(ax: plt.Axes, extent: Tuple[float, float, float, float], scalebar_km: float) -> None:
    if float(scalebar_km) <= 0:
        return
    x0, x1, y0, y1 = extent
    mean_lat = 0.5 * (y0 + y1)
    dlon = float(scalebar_km) / _km_per_deg_lon(mean_lat)
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
        fontsize=10,
        color="black",
        zorder=10,
    )


def _gaussian_smooth_fft(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Dependency-free Gaussian smoothing via frequency-domain multiplication.
    sigma is in pixel units of the histogram grid.
    """
    sigma = float(sigma)
    if sigma <= 0:
        return img
    if img.ndim != 2:
        raise ValueError(f"Expected 2D array, got {img.shape}")
    ny, nx = img.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    Fy, Fx = np.meshgrid(fy, fx, indexing="ij")
    kernel_ft = np.exp(-2.0 * (np.pi**2) * (sigma**2) * (Fx**2 + Fy**2))
    out = np.fft.ifftn(np.fft.fftn(img) * kernel_ft).real
    # Numerical noise may create tiny negatives.
    return np.clip(out, 0.0, None)

@dataclass(frozen=True)
class Samples:
    name: str
    preds: np.ndarray  # (N, F, 2) [y, x]
    preds_k: Optional[np.ndarray]  # (N, K, F, 2) [y, x] (optional)
    targets: np.ndarray  # (N, F, 2) [y, x]


def _load_samples(path: Path, name: str) -> Samples:
    data = np.load(path)
    preds = np.asarray(data["preds"], dtype=np.float32)
    preds_k = np.asarray(data["preds_k"], dtype=np.float32) if "preds_k" in data.files else None
    targets = np.asarray(data["targets"], dtype=np.float32)
    return Samples(name=name, preds=preds, preds_k=preds_k, targets=targets)


def _parse_sample_arg(raw: str) -> Tuple[str, Path]:
    """
    Parse --sample "Label:/path/to/samples.npz".

    NOTE: split on the first ':' so Windows paths like 'E:\\...' still work
    (the second ':' remains in the path string).
    """
    if ":" not in raw:
        raise ValueError(f"Invalid --sample '{raw}'. Expected 'Label:Path'.")
    label, path = raw.split(":", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError(f"Invalid --sample '{raw}': empty label.")
    if not path:
        raise ValueError(f"Invalid --sample '{raw}': empty path.")
    return label, Path(path)



def plot_geo_overlays(
    samples_list: List[Samples],
    grid: GridConfig,
    out_dir: Path,
    num_trajs: int,
    seed: int,
    flip_y: bool,
    extent_mode: str,
    pad_frac: float,
    min_span_km: float,
    axis_off: bool,
    scalebar_km: float,
    style_context: str,
    basemap_geojson: Optional[Path],
    basemap_style: BasemapStyle,
    png_only: bool,
) -> None:
    # Geo figures are multi-panel; keep fonts readable for paper.
    set_style(context=str(style_context), font_scale=1.15)

    rng = np.random.default_rng(int(seed))
    ncols = len(samples_list)
    # Use tight_layout with a reserved top margin for a single shared legend.
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.8), constrained_layout=False)
    if ncols == 1:
        axes = [axes]

    aspect = _aspect_for_latlon(grid)

    # Use the same trajectory subset across models for comparability.
    N = min(int(s.preds.shape[0]) for s in samples_list)
    take = min(int(num_trajs), N)
    idx = rng.choice(N, size=take, replace=False) if take > 0 else np.array([], dtype=np.int64)

    # Optional zoom: compute extent from the selected samples (union of GT + all preds).
    extent: Optional[Tuple[float, float, float, float]] = None
    if extent_mode == "data" and take > 0:
        lons = []
        lats = []
        for s in samples_list:
            pr = _grid_yx_to_latlon(s.preds[idx].reshape(-1, 2), grid, flip_y=flip_y)
            gt = _grid_yx_to_latlon(s.targets[idx].reshape(-1, 2), grid, flip_y=flip_y)
            lats.append(pr[:, 0]); lons.append(pr[:, 1])
            lats.append(gt[:, 0]); lons.append(gt[:, 1])
        lat = np.concatenate(lats, axis=0)
        lon = np.concatenate(lons, axis=0)
        lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
        lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
        dlat = max(lat_max - lat_min, 1e-6)
        dlon = max(lon_max - lon_min, 1e-6)
        lat_pad = float(pad_frac) * dlat
        lon_pad = float(pad_frac) * dlon
        extent = (lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad)
        extent = _ensure_min_span(extent, min_span_km=float(min_span_km))

    legend_map: Dict[str, Any] = {}

    for ax, s in zip(axes, samples_list):
        draw_geojson_basemap(ax, basemap_geojson, basemap_style, zorder_base=0)
        preds_ll = _grid_yx_to_latlon(s.preds[idx], grid, flip_y=flip_y)  # (take, F, 2) [lat, lon]
        targets_ll = _grid_yx_to_latlon(s.targets[idx], grid, flip_y=flip_y)

        pred_color = get_color(s.name)

        # Plot a bundle (thin lines, high alpha cleanliness)
        for i in range(take):
            gt = targets_ll[i]
            pr = preds_ll[i]
            ax.plot(
                gt[:, 1],
                gt[:, 0],
                color="#222222",
                lw=1.2,
                alpha=0.30,
                label="GT" if i == 0 else None,
                zorder=3,
            )
            ax.plot(
                pr[:, 1],
                pr[:, 0],
                color=pred_color,
                lw=1.2,
                alpha=0.35,
                label=s.name if i == 0 else None,
                zorder=4,
            )

        n_total = int(s.preds.shape[0])
        if n_total == int(take):
            # Avoid "N=..." ambiguity: this is a plotted subset size, not necessarily dataset size.
            ax.set_title(f"{s.name} (n_plot={take})")
        else:
            ax.set_title(f"{s.name} (n_plot={take}/{n_total})")
        if bool(axis_off):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        if extent is None:
            ax.set_xlim(grid.min_lon, grid.max_lon)
            ax.set_ylim(grid.min_lat, grid.max_lat)
        else:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        ax.set_aspect(aspect)
        if not bool(axis_off):
            ax.grid(True, ls="--", alpha=0.25)
            ax.tick_params(axis="both", which="major", labelsize=9)
        _add_scalebar(ax, extent or (grid.min_lon, grid.max_lon, grid.min_lat, grid.max_lat), scalebar_km=float(scalebar_km))

        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll and ll not in legend_map:
                legend_map[str(ll)] = hh

    if legend_map:
        ordered_labels: List[str] = []
        if "GT" in legend_map:
            ordered_labels.append("GT")
        for s in samples_list:
            if s.name in legend_map and s.name not in ordered_labels:
                ordered_labels.append(s.name)
        # Fallback: include any extra labels not covered above.
        for ll in legend_map.keys():
            if ll not in ordered_labels:
                ordered_labels.append(ll)

        handles = [legend_map[ll] for ll in ordered_labels]
        fig.legend(
            handles,
            ordered_labels,
            loc="upper center",
            ncol=min(3, len(handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout()

    _save_fig(fig, out_dir, "fig_geo_traj_overlay", png_only=bool(png_only))


def plot_geo_density(
    samples_list: List[Samples],
    grid: GridConfig,
    out_dir: Path,
    bins: int,
    sigma: float,
    flip_y: bool,
    extent_mode: str,
    pad_frac: float,
    min_span_km: float,
    axis_off: bool,
    scalebar_km: float,
    style_context: str,
    basemap_geojson: Optional[Path],
    basemap_style: BasemapStyle,
    png_only: bool,
) -> None:
    """
    Density plot (macro, journal style):
    - First panel: GT density (targets).
    - Remaining panels: Pred density for each model, with GT contour as reference.

    Rendering:
    - High-res 2D histogram -> Gaussian smoothing (KDE-like) -> LogNorm color scaling.
    - One shared colorbar across panels (consistent physical meaning).
    """
    # Density maps need larger fonts (journal readability).
    set_style(context=str(style_context), font_scale=1.20)

    if not samples_list:
        return

    # Use targets from the first sample as GT reference (require aligned subsets).
    gt_ref = samples_list[0].targets
    for s in samples_list[1:]:
        if s.targets.shape != gt_ref.shape or np.max(np.abs(s.targets - gt_ref)) > 1e-6:
            raise ValueError("targets mismatch across sample files; ensure evaluations saved aligned subsets.")

    density_panels = ["GT"] + [s.name for s in samples_list]
    ncols = len(density_panels)
    # Use manual layout + a single shared colorbar (per-panel colorbars cause overlaps and wasted space).
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.8), constrained_layout=False)
    if ncols == 1:
        axes = [axes]

    aspect = _aspect_for_latlon(grid)
    if extent_mode == "data":
        lons = []
        lats = []
        # include GT + all preds for a tight extent, then enforce minimum span to keep visual scale stable.
        gt = _grid_yx_to_latlon(gt_ref.reshape(-1, 2), grid, flip_y=flip_y)
        lats.append(gt[:, 0]); lons.append(gt[:, 1])
        for s in samples_list:
            preds_for_extent = s.preds_k.reshape(-1, 2) if s.preds_k is not None else s.preds.reshape(-1, 2)
            pr = _grid_yx_to_latlon(preds_for_extent, grid, flip_y=flip_y)
            lats.append(pr[:, 0]); lons.append(pr[:, 1])
        lat = np.concatenate(lats, axis=0)
        lon = np.concatenate(lons, axis=0)
        lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
        lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
        dlat = max(lat_max - lat_min, 1e-6)
        dlon = max(lon_max - lon_min, 1e-6)
        lat_pad = float(pad_frac) * dlat
        lon_pad = float(pad_frac) * dlon
        extent = (lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad)
        extent = _ensure_min_span(extent, min_span_km=float(min_span_km))
    else:
        extent = (grid.min_lon, grid.max_lon, grid.min_lat, grid.max_lat)

    # Precompute histograms so all panels share the same color scale.
    # GT hist first
    gt_ll = _grid_yx_to_latlon(gt_ref.reshape(-1, 2), grid, flip_y=flip_y)
    gt_hist, lon_edges, lat_edges = np.histogram2d(
        gt_ll[:, 1],
        gt_ll[:, 0],
        bins=int(bins),
        range=[[extent[0], extent[1]], [extent[2], extent[3]]],
    )
    gt_smooth = _gaussian_smooth_fft(gt_hist, sigma=float(sigma))

    pred_smooth_list: List[np.ndarray] = []
    for s in samples_list:
        preds_for_density = s.preds_k.reshape(-1, 2) if s.preds_k is not None else s.preds.reshape(-1, 2)
        preds_ll = _grid_yx_to_latlon(preds_for_density, grid, flip_y=flip_y)
        pred_hist, _, _ = np.histogram2d(
            preds_ll[:, 1],
            preds_ll[:, 0],
            bins=[lon_edges, lat_edges],
        )
        # If preds_k exists (N,K,F,2), interpret density as the expected density of a single sampled trajectory:
        # using K samples improves estimation of the distribution, but should NOT multiply total "trip count".
        if s.preds_k is not None:
            K = int(s.preds_k.shape[1])
            pred_hist = pred_hist / max(K, 1)
        pred_smooth_list.append(_gaussian_smooth_fft(pred_hist, sigma=float(sigma)))

    # Shared LogNorm across all panels (GT + preds).
    all_vals = np.concatenate([gt_smooth.reshape(-1), *[p.reshape(-1) for p in pred_smooth_list]], axis=0)
    pos = all_vals[all_vals > 0]
    vmax = float(np.max(pos)) if pos.size else 1.0
    # Keep ~4 orders of dynamic range for long-tail mobility densities.
    vmin = max(vmax / 1e4, 1e-2)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    ims: List[Any] = []
    for ax in axes:
        draw_geojson_basemap(ax, basemap_geojson, basemap_style, zorder_base=4)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect(aspect)
        ax.grid(False)
        if bool(axis_off):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.tick_params(axis="both", which="major", labelsize=9)
        _add_scalebar(ax, extent, scalebar_km=float(scalebar_km))

    # Panel 0: GT density
    im0 = axes[0].imshow(
        gt_smooth.T,
        origin="lower",
        extent=extent,
        cmap="Blues",
        norm=norm,
        interpolation="bilinear",
        alpha=0.98,
        zorder=2,
    )
    ims.append(im0)
    axes[0].set_title("GT: Density", pad=4)

    # Pred panels: Pred density + GT contour
    levels = 7
    for j, (ax, s, pred_smooth) in enumerate(zip(axes[1:], samples_list, pred_smooth_list), start=1):
        im = ax.imshow(
            pred_smooth.T,
            origin="lower",
            extent=extent,
            cmap="Blues",
            norm=norm,
            interpolation="bilinear",
            alpha=0.98,
            zorder=2,
        )
        ims.append(im)
        if np.max(gt_smooth) > 0:
            ax.contour(
                gt_smooth.T,
                levels=levels,
                origin="lower",
                extent=extent,
                colors="black",
                linewidths=0.8,
                alpha=0.45,
                zorder=3,
            )
        ax.set_title(f"{s.name}: Density\n(GT contour)", pad=4)

    # Layout: reserve right margin for a shared colorbar.
    fig.subplots_adjust(right=0.90, wspace=0.22)
    cax = fig.add_axes([0.915, 0.18, 0.015, 0.66])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(ims[0], cax=cax)
    cbar.set_label("Smoothed density (count, log scale)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    _save_fig(fig, out_dir, "fig_geo_density", png_only=bool(png_only))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats_path",
        type=str,
        required=True,
        help="Path to data_stats.json containing grid_config bbox (min/max lat/lon).",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Add one panel: 'Label:/path/to/samples.npz' (repeatable).",
    )
    parser.add_argument("--baseline_samples", type=str, default=None, help="[deprecated] .../samples.npz")
    parser.add_argument("--diff_samples", type=str, default=None, help="[deprecated] .../samples.npz")
    parser.add_argument("--physics_samples", type=str, default=None, help="[deprecated] .../samples.npz")
    parser.add_argument("--out_dir", type=str, default="data/experiments/phase_b_report/figures_geo")
    parser.add_argument("--num_trajs", type=int, default=60, help="number of trajectories to overlay per model")
    parser.add_argument("--bins", type=int, default=220, help="2D histogram bins for density plot")
    parser.add_argument("--density_sigma", type=float, default=1.4, help="Gaussian smoothing sigma in bin pixels (KDE-like).")
    parser.add_argument(
        "--overlay_keep",
        action="append",
        default=[],
        help="Optional label filter for trajectory overlay (repeatable). Example: --overlay_keep CFG2",
    )
    parser.add_argument(
        "--density_keep",
        action="append",
        default=[],
        help="Optional label filter for density plot (repeatable). Example: --density_keep Prior --density_keep CFG2",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--extent",
        type=str,
        choices=["full", "data"],
        default="full",
        help="Plot extent: 'full' uses dataset bbox; 'data' zooms to the saved samples (union of GT+pred).",
    )
    parser.add_argument("--pad_frac", type=float, default=0.08, help="Padding fraction when --extent=data.")
    parser.add_argument("--min_span_km", type=float, default=0.0, help="Enforce a minimum map span (km) for consistent scale.")
    parser.add_argument("--axis_off", action="store_true", help="Hide axes/ticks (map style).")
    parser.add_argument("--scalebar_km", type=float, default=0.0, help="Draw a simple scale bar (km).")
    parser.add_argument("--style", type=str, choices=["paper", "talk"], default="paper", help="Matplotlib style preset.")
    parser.add_argument("--png_only", action="store_true", help="Only save PNG (skip PDF).")
    parser.add_argument(
        "--flip_y",
        action="store_true",
        help="flip y->lat mapping if your grid y axis is opposite (debug option)",
    )
    parser.add_argument(
        "--basemap_geojson",
        type=str,
        default=None,
        help="Optional GeoJSON (WGS84 lon/lat) to draw as a basemap (e.g., geo_map/Shenzhen_county.geojson).",
    )
    parser.add_argument("--basemap_edgecolor", type=str, default="#3A3A3A")
    parser.add_argument("--basemap_facecolor", type=str, default="none", help="Use 'none' for transparent fill.")
    parser.add_argument("--basemap_linewidth", type=float, default=0.7)
    parser.add_argument("--basemap_alpha", type=float, default=0.55)
    parser.add_argument("--basemap_labels", action="store_true", help="Draw district labels from GeoJSON properties['name'].")
    parser.add_argument("--basemap_label_size", type=int, default=8)
    parser.add_argument(
        "--basemap_label_lang",
        type=str,
        choices=["en", "raw"],
        default="en",
        help="Basemap label language: 'en' translates known Shenzhen district names; 'raw' keeps GeoJSON labels.",
    )
    args = parser.parse_args()

    grid = _load_grid_config(Path(args.stats_path))

    samples_list: List[Samples] = []
    for raw in args.sample:
        label, path = _parse_sample_arg(str(raw))
        samples_list.append(_load_samples(path, name=label))
    if args.baseline_samples:
        samples_list.append(_load_samples(Path(args.baseline_samples), name="Baseline"))
    if args.diff_samples:
        samples_list.append(_load_samples(Path(args.diff_samples), name="Diffusion"))
    if args.physics_samples:
        samples_list.append(_load_samples(Path(args.physics_samples), name="Physics"))

    if not samples_list:
        raise ValueError(
            "No samples provided. Use --sample 'Label:Path' (recommended) or legacy --baseline_samples/--diff_samples/--physics_samples."
        )

    overlay_samples = samples_list
    if args.overlay_keep:
        keep = {str(x) for x in args.overlay_keep}
        overlay_samples = [s for s in samples_list if s.name in keep]
        if not overlay_samples:
            raise ValueError(f"--overlay_keep provided but no samples matched: {sorted(keep)}")

    out_dir = Path(args.out_dir)
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

    plot_geo_overlays(
        samples_list=overlay_samples,
        grid=grid,
        out_dir=out_dir,
        num_trajs=int(args.num_trajs),
        seed=int(args.seed),
        flip_y=bool(args.flip_y),
        extent_mode=str(args.extent),
        pad_frac=float(args.pad_frac),
        min_span_km=float(args.min_span_km),
        axis_off=bool(args.axis_off),
        scalebar_km=float(args.scalebar_km),
        style_context=str(args.style),
        basemap_geojson=basemap_geojson,
        basemap_style=basemap_style,
        png_only=bool(args.png_only),
    )

    density_samples = samples_list
    if args.density_keep:
        keep = {str(x) for x in args.density_keep}
        density_samples = [s for s in samples_list if s.name in keep]
        if not density_samples:
            raise ValueError(f"--density_keep provided but no samples matched: {sorted(keep)}")

    plot_geo_density(
        samples_list=density_samples,
        grid=grid,
        out_dir=out_dir,
        bins=int(args.bins),
        sigma=float(args.density_sigma),
        flip_y=bool(args.flip_y),
        extent_mode=str(args.extent),
        pad_frac=float(args.pad_frac),
        min_span_km=float(args.min_span_km),
        axis_off=bool(args.axis_off),
        scalebar_km=float(args.scalebar_km),
        style_context=str(args.style),
        basemap_geojson=basemap_geojson,
        basemap_style=basemap_style,
        png_only=bool(args.png_only),
    )


if __name__ == "__main__":
    main()
