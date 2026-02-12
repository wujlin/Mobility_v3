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


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _mean_label_offset(label: str) -> Tuple[float, float]:
    s = str(label).lower()
    if "way-casd" in s or "waycasd" in s:
        return (0.012, 0.012)
    if "oracle" in s:
        return (0.012, -0.014)
    if "rnn" in s:
        return (0.012, 0.004)
    if "transformer" in s or "tr-ar" in s:
        return (0.012, -0.010)
    return (0.012, 0.0)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Figure B: OD-level GT Coverage@K vs Self-Diversity@K scatter from Phase C json.")
    ap.add_argument("--phasec_json", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--title", type=str, default="OD-level Coverage vs Diversity")
    ap.add_argument("--point_alpha", type=float, default=0.42)
    ap.add_argument("--point_size", type=float, default=14.0)
    ap.add_argument("--show_mean_label", action="store_true")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    _require_file(args.phasec_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = _read_json(args.phasec_json)
    methods = root.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise SystemExit("[FATAL] phasec json missing methods block")

    summary_table = root.get("summary_table", [])
    summary_by_method: Dict[str, Dict[str, Any]] = {}
    if isinstance(summary_table, list):
        for row in summary_table:
            if isinstance(row, dict) and "method" in row:
                summary_by_method[str(row["method"])] = row

    with paper_style():
        fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
        add_panel_label(ax, "a")

        legend_handles: List[Any] = []
        legend_labels: List[str] = []
        out_rows: List[Dict[str, Any]] = []

        for method_name, m in methods.items():
            if not isinstance(m, dict):
                continue
            per_od = m.get("per_od", [])
            if not isinstance(per_od, list) or not per_od:
                continue
            xs: List[float] = []
            ys: List[float] = []
            n_cov_finite = 0
            n_div_finite = 0
            for r in per_od:
                if not isinstance(r, dict):
                    continue
                x = _safe_float(r.get("gt_coverage_at_k"))
                y = _safe_float(r.get("self_diversity_at_k"))
                if np.isfinite(x):
                    n_cov_finite += 1
                if np.isfinite(y):
                    n_div_finite += 1
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(x)
                    ys.append(y)
            if not xs:
                continue

            color = _color_for_label(str(method_name))
            sc = ax.scatter(
                np.asarray(xs, dtype=np.float64),
                np.asarray(ys, dtype=np.float64),
                s=float(args.point_size),
                c=color,
                alpha=float(args.point_alpha),
                edgecolors="none",
                zorder=2,
            )
            legend_handles.append(sc)
            legend_labels.append(f"{method_name} (finite={len(xs)}/{len(per_od)})")

            # mean point from summary_table if available, otherwise from per_od.
            st = summary_by_method.get(str(method_name), {})
            mx = _safe_float(st.get("gt_coverage_at_k_mean", np.mean(xs)))
            my = _safe_float(st.get("self_diversity_at_k_mean", np.mean(ys)))
            if np.isfinite(mx) and np.isfinite(my):
                ax.scatter([mx], [my], s=95, marker="X", c=color, edgecolors="white", linewidths=0.7, zorder=4)
                if bool(args.show_mean_label):
                    off_x, off_y = _mean_label_offset(str(method_name))
                    tx = min(1.0, max(0.0, float(mx) + float(off_x)))
                    ty = min(1.0, max(0.0, float(my) + float(off_y)))
                    ax.text(tx, ty, f"{method_name}", fontsize=8, color=color, zorder=5)

            out_rows.append(
                {
                    "method": str(method_name),
                    "n_points_finite": int(len(xs)),
                    "n_od_groups_kept": int(len(per_od)),
                    "n_coverage_finite": int(n_cov_finite),
                    "n_diversity_finite": int(n_div_finite),
                    "coverage_mean_emp": float(np.mean(np.asarray(xs, dtype=np.float64))),
                    "diversity_mean_emp": float(np.mean(np.asarray(ys, dtype=np.float64))),
                }
            )

        ax.set_xlabel("GT Coverage@K")
        ax.set_ylabel("Self-Diversity@K")
        ax.set_title(str(args.title))
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.text(
            0.02, 0.98,
            "Only ODs with finite coverage & diversity are plotted.\n"
            "Diversity is undefined when successful predictions < 2.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#555555",
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2.5},
            zorder=6,
        )
        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc="lower right", framealpha=0.9)

        stem = "figureB_od_coverage_vs_diversity"
        out_png = out_dir / f"{stem}.png"
        out_pdf = out_dir / f"{stem}.pdf"
        save_figure(fig, out_png)
        save_figure(fig, out_pdf)
        plt.close(fig)

    out_json = out_dir / f"{stem}.meta.json"
    out = {
        "ok": True,
        "task": "waycasd_plot_od_diversity_scatter",
        "note": "Only ODs with finite coverage and finite diversity are plotted (diversity requires >=2 successful predictions).",
        "input_phasec_json": str(args.phasec_json),
        "summary_points": out_rows,
        "outputs": {"png": str(out_png), "pdf": str(out_pdf)},
    }
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved: {out_png}")
    print(f"[OK] saved: {out_pdf}")
    print(f"[OK] saved: {out_json}")


if __name__ == "__main__":
    main()
