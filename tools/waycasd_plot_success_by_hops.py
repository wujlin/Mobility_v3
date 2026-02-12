#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Allow running as file: `python tools/xxx.py ...`
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plot_style import OKABE_ITO, add_panel_label, paper_style, save_figure


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[FATAL] file not found: {path}")
    if not path.is_file():
        raise SystemExit(f"[FATAL] not a file: {path}")


def _parse_series(spec: str) -> Tuple[str, Path]:
    s = str(spec or "").strip()
    if "=" not in s:
        raise SystemExit(f"[FATAL] bad --series spec: {spec!r}; expect LABEL=PATH")
    name, p = s.split("=", 1)
    name = name.strip()
    path = Path(p.strip()).expanduser()
    if not name:
        raise SystemExit(f"[FATAL] empty label in --series spec: {spec!r}")
    return name, path


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _color_for_label(label: str) -> str:
    s = str(label).lower()
    if "oracle" in s:
        return OKABE_ITO["blue"]
    if "way-casd" in s or "waycasd" in s:
        return OKABE_ITO["vermillion"]
    if "rnn" in s:
        return OKABE_ITO["bluish_green"]
    if "transformer" in s or "tr-ar" in s:
        return OKABE_ITO["sky_blue"]
    return OKABE_ITO["gray"]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Figure C: success-vs-hops curves from binned eval json.")
    ap.add_argument("--series", action="append", required=True, help="Repeatable: LABEL=BINNED_JSON")
    ap.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    ap.add_argument("--metric", choices=["success_rate", "hit_wall_rate", "loop_rate"], default="success_rate")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--title", type=str, default="Success Rate vs GT Hops")
    ap.add_argument("--hide_bin_n", action="store_true", help="Hide per-bin sample size n on x-axis labels.")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xs_ref: List[str] = []
    curves: List[Tuple[str, List[float], List[int]]] = []

    for sp in list(args.series):
        name, path = _parse_series(sp)
        _require_file(path)
        d = _read_json(path)
        blk = d.get("overall", {}).get(str(args.decode))
        if not isinstance(blk, dict) or "bins" not in blk or "cells" not in blk:
            raise SystemExit(f"[FATAL] missing overall.{args.decode}.cells in {path}")
        bins = [str(x) for x in blk["bins"]]
        cells = blk["cells"]
        ys: List[float] = []
        ns: List[int] = []
        for b in bins:
            c = cells.get(b, {})
            y = c.get(str(args.metric), float("nan"))
            ys.append(float(y) if y is not None else float("nan"))
            ns.append(int(c.get("n", 0)))
        if not xs_ref:
            xs_ref = bins
        elif xs_ref != bins:
            raise SystemExit(f"[FATAL] bins mismatch in {path}")
        curves.append((name, ys, ns))

    with paper_style():
        fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
        add_panel_label(ax, "a")
        x = np.arange(len(xs_ref), dtype=np.float64)
        for name, ys, ns in curves:
            color = _color_for_label(name)
            yarr = np.asarray(ys, dtype=np.float64)
            ax.plot(
                x, yarr,
                marker="o",
                lw=2.0,
                ms=5.0,
                color=color,
                alpha=0.95,
                label=name,
            )
        ns_ref = curves[0][2] if curves else [0 for _ in xs_ref]
        if bool(args.hide_bin_n):
            xticks = xs_ref
        else:
            xticks = [f"{b}\n(n={int(ns_ref[i])})" for i, b in enumerate(xs_ref)]
        ax.set_xticks(x)
        ax.set_xticklabels(xticks)
        ax.set_ylim(-0.02, 1.02)
        ylabel = {
            "success_rate": "Success Rate",
            "hit_wall_rate": "Hit-Wall Rate",
            "loop_rate": "Loop Rate",
        }[str(args.metric)]
        ax.set_ylabel(ylabel)
        ax.set_xlabel("GT Hops Bins")
        ax.set_title(str(args.title))
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.legend(loc="best", framealpha=0.9)

        stem = f"figureC_{args.metric}_{args.decode}"
        out_png = out_dir / f"{stem}.png"
        out_pdf = out_dir / f"{stem}.pdf"
        save_figure(fig, out_png)
        save_figure(fig, out_pdf)
        plt.close(fig)

    meta = {
        "ok": True,
        "task": "waycasd_plot_success_by_hops",
        "decode": str(args.decode),
        "metric": str(args.metric),
        "bins": xs_ref,
        "series": [{"name": n, "n_per_bin": ns, "y": ys} for n, ys, ns in curves],
        "outputs": {"png": str(out_png), "pdf": str(out_pdf)},
    }
    out_json = out_dir / f"{stem}.meta.json"
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved: {out_png}")
    print(f"[OK] saved: {out_pdf}")
    print(f"[OK] saved: {out_json}")


if __name__ == "__main__":
    main()
