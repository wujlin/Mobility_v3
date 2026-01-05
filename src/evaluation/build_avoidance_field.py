from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import numpy as np

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # optional
    pq = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plot_style import FIGSIZE_FULL, paper_style, save_figure


Weighting = Literal["points", "segment"]


@dataclass(frozen=True)
class BuildCfg:
    grid_h: int
    grid_w: int
    weighting: Weighting
    normalize: bool
    eps: float
    support_min_prob: float
    max_segments: int


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_segments_dict(parquet_path: Path, *, max_segments: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if pq is None:
        raise SystemExit("pyarrow is required. Install: pip/conda install pyarrow")

    pf = pq.ParquetFile(str(parquet_path))
    cols = ["traj_csv", "y", "x"]
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    scanned = 0
    for batch in pf.iter_batches(batch_size=128, columns=cols):
        d = batch.to_pydict()
        n_rows = len(d["traj_csv"])
        for i in range(n_rows):
            scanned += 1
            if max_segments and scanned > int(max_segments):
                break
            k = str(d["traj_csv"][i])
            y = np.asarray(d["y"][i], dtype=np.int64).reshape(-1)
            x = np.asarray(d["x"][i], dtype=np.int64).reshape(-1)
            if y.size == 0 or x.size == 0:
                continue
            out[k] = (y, x)
        if max_segments and scanned > int(max_segments):
            break
    return out


def _accum_heat(
    seg: Dict[str, Tuple[np.ndarray, np.ndarray]],
    keys: Iterable[str],
    *,
    H: int,
    W: int,
    weighting: Weighting,
) -> np.ndarray:
    flat = np.zeros((H * W,), dtype=np.float64)
    for k in keys:
        y, x = seg[k]
        mask = (y >= 0) & (y < H) & (x >= 0) & (x < W)
        if not np.any(mask):
            continue
        idx = (y[mask] * W + x[mask]).astype(np.int64)
        if idx.size == 0:
            continue
        if weighting == "segment":
            w = 1.0 / float(idx.size)
            np.add.at(flat, idx, w)
        else:
            np.add.at(flat, idx, 1.0)
    return flat.reshape(H, W)


def _safe_prob(x: np.ndarray, *, eps: float) -> np.ndarray:
    s = float(np.sum(x))
    if not np.isfinite(s) or s <= 0:
        return np.full_like(x, 1.0 / float(x.size) if x.size else 0.0)
    return x / (s + float(eps))


def _quantiles(x: np.ndarray) -> Dict[str, float]:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "mean": float("nan")}
    p10, p50, p90 = np.percentile(v, [10, 50, 90]).tolist()
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90), "mean": float(np.mean(v))}


def _plot_heat(x: np.ndarray, out_png: Path, *, title: str, cmap: str = "viridis") -> None:
    a = np.asarray(x, dtype=np.float64)
    v = np.log10(a + 1e-12)
    lo = float(np.percentile(v[np.isfinite(v)], 1)) if np.any(np.isfinite(v)) else float("nan")
    hi = float(np.percentile(v[np.isfinite(v)], 99)) if np.any(np.isfinite(v)) else float("nan")
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = None, None

    with paper_style():
        fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
        im = ax.imshow(v, cmap=cmap, origin="upper", vmin=lo, vmax=hi)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("log10(prob + eps)")
        fig.tight_layout()
        save_figure(fig, out_png)
        save_figure(fig, out_png.with_suffix(".pdf"))
        plt.close(fig)


def _plot_diverging(x: np.ndarray, out_png: Path, *, title: str) -> None:
    a = np.asarray(x, dtype=np.float64)
    v = np.clip(a, -3.0, 3.0)
    with paper_style():
        fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
        im = ax.imshow(v, cmap="RdBu_r", origin="upper", vmin=-3.0, vmax=3.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("clipped value")
        fig.tight_layout()
        save_figure(fig, out_png)
        save_figure(fig, out_png.with_suffix(".pdf"))
        plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Behavioral Avoidance Field: expected vs observed spatial footprints.")
    p.add_argument("--expected_segments_parquet", type=Path, required=True, help="Predicted/expected segments.parquet (must contain traj_csv,y,x).")
    p.add_argument("--observed_segments_parquet", type=Path, required=True, help="Observed segments.parquet (must contain traj_csv,y,x).")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--grid_h", type=int, default=1024)
    p.add_argument("--grid_w", type=int, default=1024)
    p.add_argument("--weighting", type=str, default="segment", choices=["points", "segment"])
    p.add_argument("--normalize", action="store_true", help="Normalize heatmaps into probabilities (recommended).")
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument(
        "--support_min_prob",
        type=float,
        default=1e-9,
        help="Support mask threshold on expected probability (cells below are ignored in summary stats).",
    )
    p.add_argument("--max_segments", type=int, default=0, help="Optional cap for speed/debug (0=no cap).")
    p.add_argument("--no_png", action="store_true", help="Do not write PNG quicklooks.")
    p.add_argument("--quiet", action="store_true", help="Write outputs only; do not print JSON.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    cfg = BuildCfg(
        grid_h=int(args.grid_h),
        grid_w=int(args.grid_w),
        weighting=str(args.weighting),  # type: ignore[arg-type]
        normalize=bool(args.normalize),
        eps=float(args.eps),
        support_min_prob=float(args.support_min_prob),
        max_segments=int(args.max_segments),
    )

    exp_seg = _load_segments_dict(Path(args.expected_segments_parquet), max_segments=cfg.max_segments)
    obs_seg = _load_segments_dict(Path(args.observed_segments_parquet), max_segments=cfg.max_segments)

    exp_keys = set(exp_seg.keys())
    obs_keys = set(obs_seg.keys())
    common = sorted(exp_keys & obs_keys)
    dropped = {"expected_only": int(len(exp_keys - obs_keys)), "observed_only": int(len(obs_keys - exp_keys))}

    H, W = int(cfg.grid_h), int(cfg.grid_w)
    exp_heat = _accum_heat(exp_seg, common, H=H, W=W, weighting=cfg.weighting)
    obs_heat = _accum_heat(obs_seg, common, H=H, W=W, weighting=cfg.weighting)

    if cfg.normalize:
        exp = _safe_prob(exp_heat, eps=cfg.eps)
        obs = _safe_prob(obs_heat, eps=cfg.eps)
    else:
        exp = exp_heat
        obs = obs_heat

    eps = float(cfg.eps)
    log_ratio = np.log((obs + eps) / (exp + eps))
    rel_diff = (obs - exp) / (exp + eps)

    support = exp > float(cfg.support_min_prob)
    supported_vals = log_ratio[support]
    supported_rel = rel_diff[support]

    out = {
        "expected_segments_parquet": str(Path(args.expected_segments_parquet)),
        "observed_segments_parquet": str(Path(args.observed_segments_parquet)),
        "cfg": {
            "grid_h": H,
            "grid_w": W,
            "weighting": cfg.weighting,
            "normalize": cfg.normalize,
            "eps": cfg.eps,
            "support_min_prob": cfg.support_min_prob,
            "max_segments": cfg.max_segments,
        },
        "alignment": {
            "common_n": int(len(common)),
            "dropped": dropped,
        },
        "summary": {
            "support_cells": int(np.sum(support)),
            "support_cells_ratio": float(np.mean(support.astype(np.float64))),
            "log_ratio_stats": _quantiles(supported_vals),
            "rel_diff_stats": _quantiles(supported_rel),
        },
        "artifacts": {
            "expected_npy": str(out_dir / "expected.npy"),
            "observed_npy": str(out_dir / "observed.npy"),
            "log_ratio_npy": str(out_dir / "avoidance_log_ratio.npy"),
            "rel_diff_npy": str(out_dir / "avoidance_rel_diff.npy"),
            "support_mask_npy": str(out_dir / "support_mask.npy"),
        },
    }

    np.save(out_dir / "expected.npy", exp.astype(np.float32))
    np.save(out_dir / "observed.npy", obs.astype(np.float32))
    np.save(out_dir / "avoidance_log_ratio.npy", log_ratio.astype(np.float32))
    np.save(out_dir / "avoidance_rel_diff.npy", rel_diff.astype(np.float32))
    np.save(out_dir / "support_mask.npy", support.astype(np.uint8))
    (out_dir / "avoidance_field.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if not bool(args.no_png):
        _plot_heat(exp, out_dir / "expected.png", title="Expected footprint (log prob)")
        _plot_heat(obs, out_dir / "observed.png", title="Observed footprint (log prob)")
        _plot_diverging(log_ratio, out_dir / "avoidance_log_ratio.png", title="Avoidance field: log(obs/exp) (clipped)")
        _plot_diverging(rel_diff, out_dir / "avoidance_rel_diff.png", title="Avoidance field: (obs-exp)/exp (clipped)")

    if not bool(args.quiet):
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
