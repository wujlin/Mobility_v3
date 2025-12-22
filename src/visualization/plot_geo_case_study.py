"""
Micro-scale case study figure in geographic space.

This script draws a grid of small "local maps": each subplot is one condition (trajectory),
overlaying GT + multiple model predictions on top of a GeoJSON boundary basemap.

Why it exists:
- plot_geo_phase_b focuses on macro views (per-model panels). This figure focuses on micro comparisons:
  "for the same OD condition, how do different models behave?"

Inputs:
- --stats_path: data_stats.json (grid bbox mapping)
- --sample "Label:/path/to/samples.npz" repeated for multiple models
- Optional: --basemap_geojson geo_map/Shenzhen_county.geojson

Note:
- samples.npz stores only k=0 predictions; this figure is for qualitative comparison, not multi-sample fan-out.
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


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(f"[OK] saved {pdf}")
    print(f"[OK] saved {png}")


def _compute_extent(latlon_list: List[np.ndarray], pad_frac: float) -> Tuple[float, float, float, float]:
    lat = np.concatenate([a[..., 0].reshape(-1) for a in latlon_list], axis=0)
    lon = np.concatenate([a[..., 1].reshape(-1) for a in latlon_list], axis=0)
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    dlat = max(lat_max - lat_min, 1e-6)
    dlon = max(lon_max - lon_min, 1e-6)
    lat_pad = float(pad_frac) * dlat
    lon_pad = float(pad_frac) * dlon
    return (lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_path", type=str, required=True)
    parser.add_argument("--sample", action="append", default=[], help="Repeatable 'Label:/path/to/samples.npz'")
    parser.add_argument("--out_dir", type=str, default="essay/figures/stage_cfg")
    parser.add_argument("--stem", type=str, default="fig_geo_case_study")
    parser.add_argument("--num_cases", type=int, default=9)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pad_frac", type=float, default=0.12)
    parser.add_argument("--flip_y", action="store_true")

    parser.add_argument("--basemap_geojson", type=str, default=None)
    parser.add_argument("--basemap_edgecolor", type=str, default="#2f2f2f")
    parser.add_argument("--basemap_facecolor", type=str, default="none")
    parser.add_argument("--basemap_linewidth", type=float, default=0.7)
    parser.add_argument("--basemap_alpha", type=float, default=0.60)
    parser.add_argument("--basemap_labels", action="store_true")
    parser.add_argument("--basemap_label_size", type=int, default=7)

    parser.add_argument("--title", type=str, default="Qualitative case studies (geographic space)")
    parser.add_argument("--style", type=str, choices=["paper", "talk"], default="paper")
    args = parser.parse_args()

    if not args.sample:
        raise ValueError("No samples provided. Use --sample 'Label:Path' (repeatable).")

    set_style(context=str(args.style), font_scale=1.0)
    grid = _load_grid_config(Path(args.stats_path))
    aspect = _aspect_for_latlon(grid)

    # Load samples
    preds_by_model: Dict[str, np.ndarray] = {}
    target_ref: Optional[np.ndarray] = None
    N = None
    for raw in args.sample:
        label, path = _parse_sample_arg(str(raw))
        data = np.load(path)
        preds = np.asarray(data["preds"], dtype=np.float32)
        targets = np.asarray(data["targets"], dtype=np.float32)
        if preds.ndim != 3 or preds.shape[-1] != 2:
            raise ValueError(f"Invalid preds shape in {path}: {preds.shape}, expected (N,F,2)")
        if targets.ndim != 3 or targets.shape[-1] != 2:
            raise ValueError(f"Invalid targets shape in {path}: {targets.shape}, expected (N,F,2)")
        if target_ref is None:
            target_ref = targets
            N = int(targets.shape[0])
        else:
            if targets.shape != target_ref.shape or np.max(np.abs(targets - target_ref)) > 1e-6:
                raise ValueError(f"targets mismatch across sample files; ensure evaluations saved aligned subsets. Offender: {path}")
        preds_by_model[label] = preds

    assert target_ref is not None
    n_total = int(target_ref.shape[0])
    k = min(int(args.num_cases), n_total)
    rng = np.random.default_rng(int(args.seed))
    idx = rng.choice(n_total, size=k, replace=False) if k > 0 else np.array([], dtype=np.int64)

    cols = max(1, int(args.cols))
    rows = int(np.ceil(k / cols)) if k > 0 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.0 * rows), constrained_layout=False)
    axes = np.array(axes).reshape(-1)

    basemap_geojson = Path(args.basemap_geojson) if args.basemap_geojson else None
    basemap_style = BasemapStyle(
        edgecolor=str(args.basemap_edgecolor),
        facecolor=str(args.basemap_facecolor),
        linewidth=float(args.basemap_linewidth),
        alpha=float(args.basemap_alpha),
        label=bool(args.basemap_labels),
        label_size=int(args.basemap_label_size),
    )

    handles = None
    labels = None

    for i, si in enumerate(idx):
        ax = axes[i]
        draw_geojson_basemap(ax, basemap_geojson, basemap_style, zorder_base=0)

        gt_ll = _grid_yx_to_latlon(target_ref[si], grid, flip_y=bool(args.flip_y))
        ax.plot(gt_ll[:, 1], gt_ll[:, 0], color="#222222", lw=2.2, label="GT", zorder=3)

        for name, pred in preds_by_model.items():
            pr_ll = _grid_yx_to_latlon(pred[si], grid, flip_y=bool(args.flip_y))
            ax.plot(pr_ll[:, 1], pr_ll[:, 0], color=get_color(name), lw=1.9, ls="--", label=name, alpha=0.95, zorder=4)

        # Start/end markers (GT)
        ax.scatter(gt_ll[0, 1], gt_ll[0, 0], color="black", s=35, marker="*", zorder=5)
        ax.scatter(gt_ll[-1, 1], gt_ll[-1, 0], color="#222222", s=16, marker="o", zorder=5)

        # Per-case extent
        ll_all = [gt_ll] + [_grid_yx_to_latlon(preds_by_model[n][si], grid, flip_y=bool(args.flip_y)) for n in preds_by_model.keys()]
        x0, x1, y0, y1 = _compute_extent(ll_all, pad_frac=float(args.pad_frac))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect(aspect)
        ax.set_title(f"Case #{int(si)}", fontsize=10, pad=2)
        ax.grid(True, ls="--", alpha=0.18)
        ax.tick_params(axis="both", which="major", labelsize=8)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    for j in range(k, len(axes)):
        axes[j].axis("off")

    if args.title:
        fig.suptitle(str(args.title), y=0.99)

    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, 0.965))
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout()

    _save_fig(fig, out_dir=Path(args.out_dir), stem=str(args.stem))


if __name__ == "__main__":
    main()

