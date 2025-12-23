"""
Nav field visualization in geographic space (bbox linear mapping).

用途：
- 用一张图解释 physics-conditioned diffusion 的“物理先验”到底是什么：
  train-only nav_field = 城市尺度的局部平均流场（mean-flow prior）。

边界（v1）：
- grid-based，不做 map matching；经纬度用于展示。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.features.nav_field import NavField
from src.visualization.basemap import BasemapStyle, draw_geojson_basemap
from src.visualization.style_config import set_style


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


def _aspect_for_latlon(grid: GridConfig) -> float:
    mean_lat = 0.5 * (grid.min_lat + grid.max_lat)
    return 1.0 / max(np.cos(np.deg2rad(mean_lat)), 1e-6)


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(f"[OK] saved {pdf}")
    print(f"[OK] saved {png}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nav_file", type=str, required=True)
    parser.add_argument("--stats_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/experiments/phase_b_report/figures_geo_quick")
    parser.add_argument("--stride", type=int, default=18, help="downsample step for arrows")
    parser.add_argument("--flip_y", action="store_true", help="flip y->lat mapping if needed")
    parser.add_argument("--basemap_geojson", type=str, default=None, help="Optional GeoJSON overlay (WGS84 lon/lat).")
    parser.add_argument("--basemap_edgecolor", type=str, default="#3A3A3A")
    parser.add_argument("--basemap_facecolor", type=str, default="none")
    parser.add_argument("--basemap_linewidth", type=float, default=0.7)
    parser.add_argument("--basemap_alpha", type=float, default=0.55)
    parser.add_argument("--basemap_labels", action="store_true")
    parser.add_argument("--basemap_label_size", type=int, default=8)
    parser.add_argument(
        "--basemap_label_lang",
        type=str,
        choices=["en", "raw"],
        default="en",
        help="Basemap label language: 'en' translates known Shenzhen district names; 'raw' keeps GeoJSON labels.",
    )
    args = parser.parse_args()

    set_style(context="paper", font_scale=1.1)
    grid = _load_grid_config(Path(args.stats_path))
    nav = NavField(args.nav_file)

    # Heatmap uses nav.count if exists, else fall back to speed magnitude proxy.
    heat = nav.count if nav.count is not None else np.linalg.norm(nav.direction, axis=0)
    heat = heat.astype(np.float64, copy=False)

    extent = [grid.min_lon, grid.max_lon, grid.min_lat, grid.max_lat]
    aspect = _aspect_for_latlon(grid)

    # Build arrow grid
    stride = max(int(args.stride), 1)
    ys = np.arange(0, nav.H, stride)
    xs = np.arange(0, nav.W, stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")  # (Ny, Nx)

    # Map to lat/lon (linear)
    denom_y = max(grid.H - 1, 1)
    denom_x = max(grid.W - 1, 1)
    y01 = np.clip(yy.astype(np.float64) / denom_y, 0.0, 1.0)
    x01 = np.clip(xx.astype(np.float64) / denom_x, 0.0, 1.0)
    if bool(args.flip_y):
        y01 = 1.0 - y01
    lat = grid.min_lat + y01 * (grid.max_lat - grid.min_lat)
    lon = grid.min_lon + x01 * (grid.max_lon - grid.min_lon)

    # Direction vectors (grid space) -> delta in lat/lon
    dir_y = nav.direction[0][yy, xx]
    dir_x = nav.direction[1][yy, xx]
    lat_scale = (grid.max_lat - grid.min_lat) / denom_y
    lon_scale = (grid.max_lon - grid.min_lon) / denom_x
    dlat = dir_y * lat_scale * (-1.0 if bool(args.flip_y) else 1.0)
    dlon = dir_x * lon_scale

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.0), constrained_layout=True)
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
    im = ax.imshow(
        np.log1p(heat).T,
        origin="lower",
        extent=extent,
        cmap="Greys",
        alpha=0.95,
        zorder=2,
    )
    draw_geojson_basemap(ax, basemap_geojson, basemap_style, zorder_base=3)
    ax.quiver(
        lon,
        lat,
        dlon,
        dlat,
        angles="xy",
        scale_units="xy",
        scale=0.015,  # smaller -> longer arrows
        width=0.0018,
        alpha=0.55,
        color="#1f77b4",
        zorder=4,
    )
    ax.set_title("Navigation field (train-only mean-flow prior)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(grid.min_lon, grid.max_lon)
    ax.set_ylim(grid.min_lat, grid.max_lat)
    ax.set_aspect(aspect)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.85, aspect=25)
    cbar.set_label("log(1 + count)")

    _save_fig(fig, Path(args.out_dir), "fig_geo_nav_field")


if __name__ == "__main__":
    main()
