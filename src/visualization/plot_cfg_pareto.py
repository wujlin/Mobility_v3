"""
CFG Pareto plot (micro vs macro trade-off).

Goal:
- Show cfg_scale as an interpretable inference-time knob (not a hyperparameter swamp).
- Left y-axis: micro metrics (ADE_best / FDE_best) lower is better.
- Right y-axis: macro validity ratios (MSD10_R / Rog_R) closer to 1 is better.

Inputs:
- A list of metrics.json files produced by src/training/evaluate.py (with cfg_scale field).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from src.visualization.style_config import set_style


@dataclass(frozen=True)
class Point:
    cfg: float
    ade: float
    fde: float
    msd10_r: float
    rog_r: float
    path: Path


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_cfg_from_path(path: Path) -> Optional[float]:
    m = re.search(r"cfg([0-9]+(?:\\.[0-9]+)?)", str(path))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _ratio(m: Dict[str, Any], a: str, b: str) -> float:
    denom = float(m.get(b, 0.0) or 0.0)
    if denom == 0.0:
        return float("nan")
    return float(m.get(a, 0.0) or 0.0) / denom


def load_points(paths: List[Path]) -> List[Point]:
    pts: List[Point] = []
    for p in paths:
        m = _load_json(p)
        cfg = m.get("cfg_scale", None)
        if cfg is None:
            cfg = _infer_cfg_from_path(p)
        if cfg is None:
            raise ValueError(f"Missing cfg_scale in metrics and cannot infer from path: {p}")
        pts.append(
            Point(
                cfg=float(cfg),
                ade=float(m.get("ADE_best", 0.0)),
                fde=float(m.get("FDE_best", 0.0)),
                msd10_r=_ratio(m, "MSD_10", "GT_MSD_10"),
                rog_r=_ratio(m, "Rog", "GT_Rog"),
                path=p,
            )
        )
    pts.sort(key=lambda x: x.cfg)
    return pts


def plot_pareto(points: List[Point], out_dir: Path, title: str, style: str, png_only: bool) -> None:
    set_style(context=str(style), font_scale=1.15)

    xs = [p.cfg for p in points]
    ade = [p.ade for p in points]
    fde = [p.fde for p in points]
    msd10_r = [p.msd10_r for p in points]
    rog_r = [p.rog_r for p in points]

    fig, ax_l = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax_r = ax_l.twinx()

    # Left axis (micro)
    ax_l.plot(xs, ade, marker="o", lw=2.2, color="#1f77b4", label="ADE@best (↓)")
    ax_l.plot(xs, fde, marker="s", lw=2.2, color="#ff7f0e", label="FDE@best (↓)")
    ax_l.set_xlabel("CFG scale")
    ax_l.set_ylabel("Micro error (grid units)")
    ax_l.grid(True, ls="--", alpha=0.35)

    # Right axis (macro ratios)
    ax_r.plot(xs, msd10_r, marker="^", lw=2.2, color="#2ca02c", label="MSD10 ratio (→1)")
    ax_r.plot(xs, rog_r, marker="D", lw=2.2, color="#d62728", label="Rog ratio (→1)")
    ax_r.axhline(1.0, color="black", lw=1.2, ls="--", alpha=0.55)
    ax_r.set_ylabel("Macro validity ratio (pred / GT)")

    if title:
        ax_l.set_title(title)

    # Single combined legend
    h1, l1 = ax_l.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax_l.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, frameon=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "fig_cfg_pareto.png"
    fig.savefig(png, dpi=300)
    print(f"[OK] saved {png}")
    if not bool(png_only):
        pdf = out_dir / "fig_cfg_pareto.pdf"
        fig.savefig(pdf)
        print(f"[OK] saved {pdf}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", action="append", default=[], help="Path to a metrics.json (repeatable).")
    parser.add_argument("--glob", type=str, default=None, help="Glob pattern to collect metrics.json files.")
    parser.add_argument("--out_dir", type=str, default="essay/figures", help="Output dir for fig_cfg_pareto.(pdf|png)")
    parser.add_argument("--title", type=str, default="CFG trade-off: micro vs macro validity")
    parser.add_argument("--style", type=str, choices=["paper", "talk"], default="paper")
    parser.add_argument("--png_only", action="store_true", help="Only save PNG (skip PDF).")
    args = parser.parse_args()

    paths: List[Path] = []
    for m in args.metrics:
        paths.append(Path(str(m)))
    if args.glob:
        paths.extend([Path(p) for p in sorted(Path().glob(str(args.glob)))])
    paths = [p for p in paths if str(p).endswith(".json")]
    if not paths:
        raise ValueError("No metrics provided. Use --metrics ... or --glob 'data/experiments/.../metrics.json'")

    points = load_points(paths)
    plot_pareto(points, out_dir=Path(args.out_dir), title=str(args.title), style=str(args.style), png_only=bool(args.png_only))


if __name__ == "__main__":
    main()
