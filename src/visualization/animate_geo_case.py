"""
Animated micro-scale case study in geographic space (GIF or PNG frames).

Goal:
- Show "trajectory bundle" evolution over time for the same condition:
  GT vs (Prior / CFG2 / CFG3 / ...), highlighting multi-modality.

Inputs:
- --stats_path: data_stats.json (grid bbox mapping)
- --sample "Label:/path/to/samples.npz" repeated for multiple models
  If samples.npz contains `preds_k (N,K,F,2)`, we animate a spaghetti bundle.
  Otherwise we animate the single-path `preds (N,F,2)`.

Design:
- KISS: matplotlib-only. Default outputs per-frame PNGs.
- Optional GIF output if Pillow is available (matplotlib.animation.PillowWriter).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import ScalarFormatter

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


def _km_per_deg_lat() -> float:
    return 111.32


def _km_per_deg_lon(mean_lat: float) -> float:
    return 111.32 * max(np.cos(np.deg2rad(mean_lat)), 1e-6)


def _ensure_min_span(extent: Tuple[float, float, float, float], min_span_km: float) -> Tuple[float, float, float, float]:
    if float(min_span_km) <= 0:
        return extent
    x0, x1, y0, y1 = extent
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
        fontsize=9,
        color="black",
        zorder=10,
    )


def _disable_axis_offset(ax: plt.Axes) -> None:
    for axis in (ax.xaxis, ax.yaxis):
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        axis.set_major_formatter(fmt)


def _save_gif(anim: FuncAnimation, out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from matplotlib.animation import PillowWriter

        anim.save(out_path, writer=PillowWriter(fps=int(fps)))
        print(f"[OK] saved {out_path}")
    except Exception as e:
        raise RuntimeError(
            f"GIF save failed (need Pillow). Use --frames_only to export PNG frames. Error: {e}"
        ) from e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_path", type=str, required=True)
    parser.add_argument("--sample", action="append", default=[], help="Repeatable 'Label:/path/to/samples.npz'")
    parser.add_argument("--out_dir", type=str, default="essay/figures/stage_cfg/anim")
    parser.add_argument("--stem", type=str, default="anim_geo_case")
    parser.add_argument(
        "--case_idx",
        type=int,
        default=None,
        help="Index into the saved samples (0..N-1). If not set, pick a random case with --seed.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k_plot", type=int, default=12, help="If preds_k exists, animate up to k_plot spaghetti lines.")
    parser.add_argument("--pad_frac", type=float, default=0.12)
    parser.add_argument("--min_span_km", type=float, default=0.0, help="Enforce a minimum map span (km) for consistent scale.")
    parser.add_argument("--flip_y", action="store_true")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--frames_only", action="store_true", help="Only export PNG frames (skip GIF).")
    parser.add_argument("--encode_mp4", action="store_true", help="After saving frames, encode an mp4 with ffmpeg if available.")
    parser.add_argument("--axis_off", action="store_true", help="Hide axes/ticks (map style).")
    parser.add_argument("--scalebar_km", type=float, default=0.0, help="Draw a simple scale bar (km).")

    parser.add_argument("--basemap_geojson", type=str, default=None)
    parser.add_argument("--basemap_edgecolor", type=str, default="#2f2f2f")
    parser.add_argument("--basemap_facecolor", type=str, default="none")
    parser.add_argument("--basemap_linewidth", type=float, default=0.7)
    parser.add_argument("--basemap_alpha", type=float, default=0.60)
    parser.add_argument("--basemap_labels", action="store_true")
    parser.add_argument("--basemap_label_size", type=int, default=7)
    parser.add_argument(
        "--basemap_label_lang",
        type=str,
        choices=["en", "raw"],
        default="en",
        help="Basemap label language: 'en' translates known Shenzhen district names; 'raw' keeps GeoJSON labels.",
    )

    parser.add_argument("--title", type=str, default="Animated case study (geographic space)")
    parser.add_argument("--style", type=str, choices=["paper", "talk"], default="talk")
    args = parser.parse_args()

    if not args.sample:
        raise ValueError("No samples provided. Use --sample 'Label:Path' (repeatable).")

    set_style(context=str(args.style), font_scale=1.0)
    grid = _load_grid_config(Path(args.stats_path))
    aspect = _aspect_for_latlon(grid)

    # Load samples for each model; require aligned targets for cross-model animation.
    preds_by_model: Dict[str, np.ndarray] = {}
    preds_k_by_model: Dict[str, np.ndarray] = {}
    target_ref: Optional[np.ndarray] = None
    N = None
    for raw in args.sample:
        label, path = _parse_sample_arg(str(raw))
        data = np.load(path)
        preds = np.asarray(data["preds"], dtype=np.float32)
        preds_k = np.asarray(data["preds_k"], dtype=np.float32) if "preds_k" in data.files else None
        targets = np.asarray(data["targets"], dtype=np.float32)
        if target_ref is None:
            target_ref = targets
            N = int(targets.shape[0])
        else:
            if targets.shape != target_ref.shape or np.max(np.abs(targets - target_ref)) > 1e-6:
                raise ValueError(
                    "targets mismatch across sample files; ensure evaluations saved aligned subsets."
                )
        preds_by_model[label] = preds
        if preds_k is not None:
            preds_k_by_model[label] = preds_k

    assert target_ref is not None and N is not None
    rng = np.random.default_rng(int(args.seed))
    case_idx = int(args.case_idx) if args.case_idx is not None else int(rng.integers(0, N))

    # Precompute lat/lon trajectories.
    gt_ll = _grid_yx_to_latlon(target_ref[case_idx], grid, flip_y=bool(args.flip_y))
    ll_all: List[np.ndarray] = [gt_ll]

    model_names = list(preds_by_model.keys())
    per_model_ll: Dict[str, List[np.ndarray]] = {}
    k_plot = max(1, int(args.k_plot))
    for name in model_names:
        if name in preds_k_by_model:
            pk = preds_k_by_model[name]
            K = int(pk.shape[1])
            take_k = min(k_plot, K)
            k_idx = np.linspace(0, K - 1, num=take_k, dtype=int) if take_k > 1 else np.array([0], dtype=int)
            ll_list = []
            for kj in k_idx:
                pr_ll = _grid_yx_to_latlon(pk[case_idx, kj], grid, flip_y=bool(args.flip_y))
                ll_list.append(pr_ll)
                ll_all.append(pr_ll)
            per_model_ll[name] = ll_list
        else:
            pr_ll = _grid_yx_to_latlon(preds_by_model[name][case_idx], grid, flip_y=bool(args.flip_y))
            per_model_ll[name] = [pr_ll]
            ll_all.append(pr_ll)

    extent = _compute_extent(ll_all, pad_frac=float(args.pad_frac))
    extent = _ensure_min_span(extent, min_span_km=float(args.min_span_km))

    ncols = len(model_names)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.8), constrained_layout=False)
    if ncols == 1:
        axes = [axes]

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

    # Create artists.
    gt_full_lines = []
    gt_prefix_lines = []
    gt_heads = []
    pred_lines: Dict[str, List[plt.Line2D]] = {}
    pred_heads: Dict[str, List[plt.Line2D]] = {}

    F = int(gt_ll.shape[0])
    for ax, name in zip(axes, model_names):
        draw_geojson_basemap(ax, basemap_geojson, basemap_style, zorder_base=0)

        # Static GT (faint)
        gt_full = ax.plot(gt_ll[:, 1], gt_ll[:, 0], color="#222222", lw=1.6, alpha=0.25, zorder=2)[0]
        gt_full_lines.append(gt_full)

        # Animated GT prefix
        gt_prefix = ax.plot([], [], color="#111111", lw=2.6, alpha=0.95, zorder=5)[0]
        gt_prefix_lines.append(gt_prefix)
        gt_head = ax.plot([], [], marker="o", color="#111111", markersize=4, lw=0, zorder=6)[0]
        gt_heads.append(gt_head)

        # Preds
        lines = []
        heads = []
        color = get_color(name)
        for _ in per_model_ll[name]:
            ln = ax.plot([], [], color=color, lw=1.1, alpha=0.22, zorder=4)[0]
            hd = ax.plot([], [], marker="o", color=color, markersize=3, lw=0, alpha=0.40, zorder=6)[0]
            lines.append(ln)
            heads.append(hd)
        pred_lines[name] = lines
        pred_heads[name] = heads

        ax.set_title(name)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect(aspect)
        _disable_axis_offset(ax)
        if bool(args.axis_off):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            ax.grid(True, ls="--", alpha=0.18)
            ax.tick_params(axis="both", which="major", labelsize=9)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        _add_scalebar(ax, extent, scalebar_km=float(args.scalebar_km))

    if args.title:
        fig.suptitle(f"{args.title} (case #{case_idx})", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(frame: int):
        t = int(frame)
        t = max(0, min(F - 1, t))
        # GT update
        gt_seg = gt_ll[: t + 1]
        for ln, hd in zip(gt_prefix_lines, gt_heads):
            ln.set_data(gt_seg[:, 1], gt_seg[:, 0])
            hd.set_data([gt_seg[-1, 1]], [gt_seg[-1, 0]])

        # Pred update
        for name in model_names:
            for ln, hd, pr_ll in zip(pred_lines[name], pred_heads[name], per_model_ll[name]):
                seg = pr_ll[: t + 1]
                ln.set_data(seg[:, 1], seg[:, 0])
                hd.set_data([seg[-1, 1]], [seg[-1, 0]])
        return []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.frames_only):
        frames_dir = out_dir / f"{args.stem}_frames_case{case_idx}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for t in range(F):
            update(t)
            fp = frames_dir / f"frame_{t:03d}.png"
            fig.savefig(fp, dpi=int(args.dpi))
        print(f"[OK] saved frames to {frames_dir}")
        out_mp4 = out_dir / f"{args.stem}_case{case_idx}.mp4"
        ff_cmd = [
            "ffmpeg",
            "-y",
            "-r",
            str(int(args.fps)),
            "-i",
            "frame_%03d.png",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(out_mp4),
        ]
        if bool(args.encode_mp4):
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                try:
                    subprocess.run([ffmpeg, *ff_cmd[1:]], cwd=str(frames_dir), check=True)
                    print(f"[OK] saved {out_mp4}")
                except Exception as e:
                    print(f"[WARN] ffmpeg failed: {e}")
                    print(f"[TIP] ffmpeg: (cd {frames_dir} && {' '.join(ff_cmd)})")
            else:
                print("[WARN] ffmpeg not found; skipping mp4 encoding.")
                print(f"[TIP] ffmpeg: (cd {frames_dir} && {' '.join(ff_cmd)})")
        else:
            print(f"[TIP] ffmpeg: (cd {frames_dir} && {' '.join(ff_cmd)})")
    else:
        anim = FuncAnimation(fig, update, frames=list(range(F)), interval=int(1000 / max(int(args.fps), 1)), blit=False)
        gif_path = out_dir / f"{args.stem}_case{case_idx}.gif"
        _save_gif(anim, gif_path, fps=int(args.fps))


if __name__ == "__main__":
    main()
