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


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(f"[OK] saved {pdf}")
    print(f"[OK] saved {png}")


def _aspect_for_latlon(grid: GridConfig) -> float:
    # Rough correction: 1 degree lon ≈ cos(lat) * 1 degree lat
    mean_lat = 0.5 * (grid.min_lat + grid.max_lat)
    return 1.0 / max(np.cos(np.deg2rad(mean_lat)), 1e-6)


@dataclass(frozen=True)
class Samples:
    name: str
    preds: np.ndarray  # (N, F, 2) [y, x]
    targets: np.ndarray  # (N, F, 2) [y, x]


def _load_samples(path: Path, name: str) -> Samples:
    data = np.load(path)
    preds = np.asarray(data["preds"], dtype=np.float32)
    targets = np.asarray(data["targets"], dtype=np.float32)
    return Samples(name=name, preds=preds, targets=targets)


def plot_geo_overlays(
    samples_list: List[Samples],
    grid: GridConfig,
    out_dir: Path,
    num_trajs: int,
    seed: int,
    flip_y: bool,
) -> None:
    set_style(context="paper", font_scale=1.1)

    rng = np.random.default_rng(int(seed))
    ncols = len(samples_list)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.8), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    aspect = _aspect_for_latlon(grid)

    for ax, s in zip(axes, samples_list):
        N = int(s.preds.shape[0])
        take = min(int(num_trajs), N)
        idx = rng.choice(N, size=take, replace=False) if take > 0 else np.array([], dtype=np.int64)

        preds_ll = _grid_yx_to_latlon(s.preds[idx], grid, flip_y=flip_y)  # (take, F, 2) [lat, lon]
        targets_ll = _grid_yx_to_latlon(s.targets[idx], grid, flip_y=flip_y)

        pred_color = get_color(s.name)

        # Plot a bundle (thin lines, high alpha cleanliness)
        for i in range(take):
            gt = targets_ll[i]
            pr = preds_ll[i]
            ax.plot(gt[:, 1], gt[:, 0], color="#222222", lw=1.2, alpha=0.30, label="GT" if i == 0 else None)
            ax.plot(pr[:, 1], pr[:, 0], color=pred_color, lw=1.2, alpha=0.35, label=s.name if i == 0 else None)

        ax.set_title(f"{s.name} (N={take})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(grid.min_lon, grid.max_lon)
        ax.set_ylim(grid.min_lat, grid.max_lat)
        ax.set_aspect(aspect)
        ax.grid(True, ls="--", alpha=0.25)
        ax.legend(loc="upper right")

    _save_fig(fig, out_dir, "fig_geo_traj_overlay")


def plot_geo_density(
    samples_list: List[Samples],
    grid: GridConfig,
    out_dir: Path,
    bins: int,
    flip_y: bool,
) -> None:
    """
    Density plot: per model, draw Pred heatmap (log1p counts) + GT contour.
    GT uses the same samples.npz targets (subset), so caption should state it's a subset.
    """
    set_style(context="paper", font_scale=1.1)

    ncols = len(samples_list)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.8), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    aspect = _aspect_for_latlon(grid)
    extent = [grid.min_lon, grid.max_lon, grid.min_lat, grid.max_lat]

    for ax, s in zip(axes, samples_list):
        preds_ll = _grid_yx_to_latlon(s.preds.reshape(-1, 2), grid, flip_y=flip_y)
        targets_ll = _grid_yx_to_latlon(s.targets.reshape(-1, 2), grid, flip_y=flip_y)

        pred_lon = preds_ll[:, 1]
        pred_lat = preds_ll[:, 0]
        gt_lon = targets_ll[:, 1]
        gt_lat = targets_ll[:, 0]

        pred_hist, lon_edges, lat_edges = np.histogram2d(
            pred_lon,
            pred_lat,
            bins=int(bins),
            range=[[grid.min_lon, grid.max_lon], [grid.min_lat, grid.max_lat]],
        )
        gt_hist, _, _ = np.histogram2d(
            gt_lon,
            gt_lat,
            bins=[lon_edges, lat_edges],
        )

        im = ax.imshow(
            np.log1p(pred_hist).T,
            origin="lower",
            extent=extent,
            cmap="Blues",
            alpha=0.95,
        )
        # GT contour as reference
        levels = 6
        if np.max(gt_hist) > 0:
            ax.contour(
                gt_hist.T,
                levels=levels,
                origin="lower",
                extent=extent,
                colors="black",
                linewidths=0.8,
                alpha=0.55,
            )

        ax.set_title(f"{s.name}: Pred density (GT contour)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(grid.min_lon, grid.max_lon)
        ax.set_ylim(grid.min_lat, grid.max_lat)
        ax.set_aspect(aspect)
        ax.grid(False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("log(1 + count)")

    _save_fig(fig, out_dir, "fig_geo_density")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats_path",
        type=str,
        required=True,
        help="Path to data_stats.json containing grid_config bbox (min/max lat/lon).",
    )
    parser.add_argument("--baseline_samples", type=str, default=None, help=".../samples.npz")
    parser.add_argument("--diff_samples", type=str, default=None, help=".../samples.npz")
    parser.add_argument("--physics_samples", type=str, default=None, help=".../samples.npz")
    parser.add_argument("--out_dir", type=str, default="data/experiments/phase_b_report/figures_geo")
    parser.add_argument("--num_trajs", type=int, default=60, help="number of trajectories to overlay per model")
    parser.add_argument("--bins", type=int, default=220, help="2D histogram bins for density plot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--flip_y",
        action="store_true",
        help="flip y->lat mapping if your grid y axis is opposite (debug option)",
    )
    args = parser.parse_args()

    grid = _load_grid_config(Path(args.stats_path))

    samples_list: List[Samples] = []
    if args.baseline_samples:
        samples_list.append(_load_samples(Path(args.baseline_samples), name="Baseline"))
    if args.diff_samples:
        samples_list.append(_load_samples(Path(args.diff_samples), name="Diffusion"))
    if args.physics_samples:
        samples_list.append(_load_samples(Path(args.physics_samples), name="Physics"))

    if not samples_list:
        raise ValueError("No samples provided. Use at least one of --baseline_samples/--diff_samples/--physics_samples.")

    out_dir = Path(args.out_dir)

    plot_geo_overlays(
        samples_list=samples_list,
        grid=grid,
        out_dir=out_dir,
        num_trajs=int(args.num_trajs),
        seed=int(args.seed),
        flip_y=bool(args.flip_y),
    )
    plot_geo_density(
        samples_list=samples_list,
        grid=grid,
        out_dir=out_dir,
        bins=int(args.bins),
        flip_y=bool(args.flip_y),
    )


if __name__ == "__main__":
    main()

