#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.plot_style import OKABE_ITO, add_panel_label, paper_style, save_figure


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pretty_method_label(name: str) -> str:
    s = str(name).lower()
    if "betavae" in s or "flowmu" in s or "cascadetraj" in s:
        return "CascadeTraj"
    if "shortest" in s:
        return "Shortest Path"
    if "transformer" in s:
        return "Transformer AR"
    if "rnn" in s:
        return "RNN AR"
    return str(name)


def _method_color(name: str) -> str:
    s = str(name).lower()
    if "betavae" in s or "flowmu" in s or "cascadetraj" in s:
        return OKABE_ITO["vermillion"]
    if "shortest" in s:
        return OKABE_ITO["gray"]
    if "transformer" in s:
        return OKABE_ITO["bluish_green"]
    if "rnn" in s:
        return OKABE_ITO["blue"]
    return OKABE_ITO["black"]


def _method_order(name: str) -> Tuple[int, str]:
    s = str(name).lower()
    if "betavae" in s or "flowmu" in s or "cascadetraj" in s:
        return (0, s)
    if "rnn" in s:
        return (1, s)
    if "transformer" in s:
        return (2, s)
    if "shortest" in s:
        return (3, s)
    return (4, s)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Plot dataset-scale corridor fidelity figure.")
    ap.add_argument("--phasec_json", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--stem", type=str, default="fig2_dataset_scale_corridor_fidelity")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    d = _read_json(args.phasec_json)
    methods = d.get("methods", {})
    if not isinstance(methods, dict) or not methods:
        raise SystemExit(f"[FATAL] malformed methods in {args.phasec_json}")

    ordered = sorted(methods.items(), key=lambda kv: _method_order(kv[0]))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
        ax_curve, ax_scatter = axes

        # Panel (a): Coverage-tau curves.
        for raw_name, obj in ordered:
            cov_tau = obj.get("coverage_vs_tau", [])
            if not isinstance(cov_tau, list) or not cov_tau:
                continue
            tau = np.asarray([float(r["tau"]) for r in cov_tau], dtype=np.float64)
            mean = np.asarray([float(r["mean"]) for r in cov_tau], dtype=np.float64)
            label = _pretty_method_label(raw_name)
            color = _method_color(raw_name)
            lw = 2.8 if label == "CascadeTraj" else 2.0
            alpha = 0.98 if label == "CascadeTraj" else 0.92
            ls = "-" if label != "Shortest Path" else "--"
            ax_curve.plot(
                tau, mean,
                color=color,
                lw=lw,
                alpha=alpha,
                linestyle=ls,
                marker="o",
                markersize=4.0,
                label=label,
            )

        ax_curve.set_xlabel(r"Jaccard threshold $\tau$")
        ax_curve.set_ylabel(r"Coverage@$K$")
        ax_curve.set_xlim(0.08, 0.92)
        ax_curve.set_ylim(-0.01, 0.74)
        ax_curve.grid(True, alpha=0.18, linewidth=0.8)
        ax_curve.legend(frameon=False, fontsize=9, loc="upper right")
        add_panel_label(ax_curve, "a")

        # Panel (b): Arrival vs CovAUC positioning.
        xs: List[float] = []
        ys: List[float] = []
        names: List[str] = []
        colors: List[str] = []
        for raw_name, obj in ordered:
            xs.append(float(obj["arrival_rate"]))
            ys.append(float(obj["coverage_vs_tau_auc"]))
            names.append(_pretty_method_label(raw_name))
            colors.append(_method_color(raw_name))

        ax_scatter.scatter(xs, ys, c=colors, s=[160 if n == "CascadeTraj" else 95 for n in names], zorder=3)
        offsets = {
            "CascadeTraj": (0.012, 0.008),
            "RNN AR": (0.012, -0.002),
            "Transformer AR": (0.012, 0.002),
            "Shortest Path": (-0.21, 0.002),
        }
        for x, y, name, color in zip(xs, ys, names, colors):
            dx, dy = offsets.get(name, (0.012, 0.004))
            ax_scatter.text(x + dx, y + dy, name, color=color, fontsize=10, va="center")

        ax_scatter.set_xlabel("Arrival")
        ax_scatter.set_ylabel("CovAUC")
        ax_scatter.set_xlim(0.0, 1.05)
        ax_scatter.set_ylim(0.0, 0.24)
        ax_scatter.grid(True, alpha=0.18, linewidth=0.8)
        add_panel_label(ax_scatter, "b")

        out_png = out_dir / f"{args.stem}.png"
        out_pdf = out_dir / f"{args.stem}.pdf"
        save_figure(fig, out_png)
        save_figure(fig, out_pdf)
        plt.close(fig)

    meta = {
        "ok": True,
        "task": "waycasd_plot_dataset_scale_corridor_figure",
        "source_json": str(args.phasec_json),
        "methods": [
            {
                "raw_name": raw_name,
                "pretty_name": _pretty_method_label(raw_name),
                "arrival": float(obj["arrival_rate"]),
                "covtau_auc": float(obj["coverage_vs_tau_auc"]),
                "meanmaxj": float((obj.get("mean_max_jaccard_at_k", {}) or {}).get("mean", np.nan)),
                "diversity": float((obj.get("self_diversity_at_k", {}) or {}).get("mean", np.nan)),
            }
            for raw_name, obj in ordered
        ],
        "outputs": {
            "png": str(out_png),
            "pdf": str(out_pdf),
        },
    }
    (out_dir / f"{args.stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved: {out_png}")
    print(f"[OK] saved: {out_pdf}")
    print(f"[OK] saved: {out_dir / (args.stem + '.meta.json')}")


if __name__ == "__main__":
    main()
