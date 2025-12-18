"""
OD hotspot visualization (grid -> lat/lon via bbox linear mapping).

用途（essay/paper 加分图）：
- 用一张“地图式”热力图直观展示城市出行的空间异质性：
  origin hotspots vs destination hotspots。

边界（v1）：
- 数据是 grid-based；经纬度仅用于可视化展示（bbox 线性映射），不做 road-level map matching。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py

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
    parser.add_argument("--data_path", type=str, required=True, help=".../trajectories/shenzhen_trajectories.h5")
    parser.add_argument("--stats_path", type=str, required=True, help=".../data_stats.json (contains bbox)")
    parser.add_argument("--out_dir", type=str, default="data/experiments/phase_b_report/figures_geo_quick")
    parser.add_argument("--bins", type=int, default=240)
    parser.add_argument("--max_trajs", type=int, default=200000, help="subsample trajectories for speed (0=all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flip_y", action="store_true", help="flip y->lat mapping if needed")
    args = parser.parse_args()

    set_style(context="paper", font_scale=1.1)
    grid = _load_grid_config(Path(args.stats_path))

    with h5py.File(args.data_path, "r") as f:
        pos = f["positions"][:]  # (Npoints, 2) [y, x]
        ptr = f["traj_ptr"][:]  # (Ntraj+1,)

    n_traj = int(ptr.shape[0] - 1)
    origin_idx = ptr[:-1].astype(np.int64, copy=False)
    dest_idx = (ptr[1:] - 1).astype(np.int64, copy=False)

    origins = pos[origin_idx]
    dests = pos[dest_idx]

    if int(args.max_trajs) > 0 and int(args.max_trajs) < n_traj:
        rng = np.random.default_rng(int(args.seed))
        sel = rng.choice(n_traj, size=int(args.max_trajs), replace=False)
        origins = origins[sel]
        dests = dests[sel]

    origins_ll = _grid_yx_to_latlon(origins, grid, flip_y=bool(args.flip_y))
    dests_ll = _grid_yx_to_latlon(dests, grid, flip_y=bool(args.flip_y))

    extent = [grid.min_lon, grid.max_lon, grid.min_lat, grid.max_lat]
    aspect = _aspect_for_latlon(grid)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    for ax, pts, title in [
        (axes[0], origins_ll, "Origin hotspots"),
        (axes[1], dests_ll, "Destination hotspots"),
    ]:
        lon = pts[:, 1]
        lat = pts[:, 0]
        hist, lon_edges, lat_edges = np.histogram2d(
            lon,
            lat,
            bins=int(args.bins),
            range=[[grid.min_lon, grid.max_lon], [grid.min_lat, grid.max_lat]],
        )
        im = ax.imshow(
            np.log1p(hist).T,
            origin="lower",
            extent=extent,
            cmap="magma",
            alpha=0.95,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(grid.min_lon, grid.max_lon)
        ax.set_ylim(grid.min_lat, grid.max_lat)
        ax.set_aspect(aspect)
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("log(1 + count)")

    _save_fig(fig, Path(args.out_dir), "fig_geo_od_hotspots")


if __name__ == "__main__":
    main()

