#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Allow running as a file: `python tools/xxx.py ...` (so that `import src.*` works).
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plot_style import OKABE_ITO, add_panel_label, paper_style, save_figure

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[FATAL] file not found: {path}")
    if not path.is_file():
        raise SystemExit(f"[FATAL] not a file: {path}")


def _parse_name_path(spec: str) -> Tuple[str, Path]:
    s = str(spec or "").strip()
    if "=" not in s:
        raise ValueError(f"Bad spec (expect NAME=PATH): {spec!r}")
    name, path = s.split("=", 1)
    name = str(name).strip()
    p = Path(str(path).strip()).expanduser()
    if not name:
        raise ValueError(f"Bad spec (empty NAME): {spec!r}")
    return name, p


def _safe_float(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _get_cell(obj: dict, *, city: Optional[int], decode: str) -> Optional[dict]:
    if city is None:
        return obj.get("overall", {}).get(str(decode), None)
    for c in obj.get("per_city", []) or []:
        if isinstance(c, dict) and int(c.get("city", -1)) == int(city):
            return c.get(str(decode), None)
    return None


def _extract_bins(obj: dict, *, decode: str) -> List[str]:
    cell = _get_cell(obj, city=None, decode=decode)
    if not isinstance(cell, dict):
        return []
    bins = cell.get("bins", []) or []
    return [str(b) for b in bins if isinstance(b, str)]


def _extract_metric_value(cell: dict, *, bin_label: str, metric: str, stat: str) -> float:
    cells = cell.get("cells", None)
    if not isinstance(cells, dict):
        return float("nan")
    c = cells.get(str(bin_label), None)
    if not isinstance(c, dict):
        return float("nan")
    if metric.endswith("_rate") or metric == "success_rate":
        return _safe_float(c.get(metric, float("nan")))
    m = c.get(metric, None)
    if not isinstance(m, dict):
        return float("nan")
    return _safe_float(m.get(str(stat), float("nan")))


def _metric_label(metric: str) -> str:
    if metric == "success_rate":
        return "Success rate"
    if metric == "hit_wall_rate":
        return "Hit-wall rate"
    if metric == "loop_rate":
        return "Loop rate"
    if metric == "dead_end_rate":
        return "Dead-end rate"
    if metric == "frechet_m":
        return "Fréchet (km)"
    if metric == "dtw_m":
        return "DTW (km)"
    if metric == "final_error_m":
        return "Final error (km)"
    if metric == "len_ratio":
        return "Length ratio"
    return metric


def _metric_needs_km(metric: str) -> bool:
    return metric in ("frechet_m", "dtw_m", "final_error_m")


@dataclass(frozen=True)
class MethodSeries:
    name: str
    path: str
    bins: List[str]
    # city -> metric -> values
    per_city: Dict[int, Dict[str, List[float]]]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot shape metrics across hops bins from way_casd_binned_eval outputs.")
    p.add_argument("--out_dir", type=Path, default=Path("_sync/wsa/paper_figures/waycasd_v1/min5_s0/shape"))
    p.add_argument("--style", type=str, choices=["paper"], default="paper")
    p.add_argument(
        "--method",
        type=str,
        action="append",
        required=True,
        help="Repeatable: NAME=PATH to a binned_eval json.",
    )
    p.add_argument("--decode", type=str, choices=["beam", "greedy"], default="beam")
    p.add_argument("--stat", type=str, choices=["mean", "p50", "p75", "p95"], default="p50")
    p.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=["success_rate", "frechet_m", "len_ratio"],
        help="Metrics to plot (each becomes a column).",
    )
    p.add_argument("--cities", type=int, nargs="*", default=[0, 1], help="Which cities to plot (rows).")
    p.add_argument("--city_name", type=str, nargs="*", default=["0:Detroit", "1:Columbus"])
    p.add_argument("--fig_w", type=float, default=13.6)
    p.add_argument("--fig_h", type=float, default=6.2)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    city_names: Dict[int, str] = {}
    for raw in args.city_name or []:
        s = str(raw)
        if ":" not in s:
            continue
        k, v = s.split(":", 1)
        try:
            city_names[int(k)] = str(v)
        except Exception:
            continue

    decode = str(args.decode)
    stat = str(args.stat)
    metrics = [str(m) for m in (args.metrics or [])]

    methods: List[MethodSeries] = []
    ref_bins: Optional[List[str]] = None
    for spec in args.method or []:
        name, path = _parse_name_path(spec)
        _require_file(path)
        obj = _read_json(path)
        bins = _extract_bins(obj, decode=decode)
        if not bins:
            raise SystemExit(f"[FATAL] no bins found in {path} (decode={decode})")
        if ref_bins is None:
            ref_bins = list(bins)
        if ref_bins != list(bins):
            # KISS: use reference order, but allow missing bins by filling NaN.
            bins = list(ref_bins)

        per_city: Dict[int, Dict[str, List[float]]] = {}
        for city in args.cities or []:
            cell = _get_cell(obj, city=int(city), decode=decode)
            if not isinstance(cell, dict):
                raise SystemExit(f"[FATAL] missing per_city decode='{decode}' for city={city} in {path}")
            per_city[int(city)] = {}
            for metric in metrics:
                vals = [_extract_metric_value(cell, bin_label=bb, metric=str(metric), stat=stat) for bb in bins]
                if _metric_needs_km(str(metric)):
                    vals = [v / 1000.0 if np.isfinite(v) else float("nan") for v in vals]
                per_city[int(city)][str(metric)] = vals

        methods.append(MethodSeries(name=str(name), path=str(path), bins=list(bins), per_city=per_city))

    if ref_bins is None:
        raise SystemExit("[FATAL] no methods provided")

    color_cycle = [
        OKABE_ITO["black"],
        OKABE_ITO["vermillion"],
        OKABE_ITO["bluish_green"],
        OKABE_ITO["blue"],
        OKABE_ITO["orange"],
        OKABE_ITO["sky_blue"],
        OKABE_ITO.get("reddish_purple", OKABE_ITO["gray"]),
    ]
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]

    with paper_style() if args.style == "paper" else nullcontext():
        n_rows = len(args.cities or [])
        n_cols = len(metrics)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(float(args.fig_w), float(args.fig_h)), squeeze=False)

        x = np.arange(len(ref_bins), dtype=np.float64)
        legend_handles = []
        legend_labels = []
        for mi, m in enumerate(methods):
            c = color_cycle[int(mi) % len(color_cycle)]
            mk = marker_cycle[int(mi) % len(marker_cycle)]
            legend_handles.append(plt.Line2D([0], [0], color=c, marker=mk, lw=2.0))
            legend_labels.append(str(m.name))

        panel = 0
        for ri, city in enumerate(args.cities or []):
            city_title = city_names.get(int(city), f"city{int(city)}")
            for ci, metric in enumerate(metrics):
                ax = axes[ri][ci]
                for mi, m in enumerate(methods):
                    c = color_cycle[int(mi) % len(color_cycle)]
                    mk = marker_cycle[int(mi) % len(marker_cycle)]
                    y = np.asarray(m.per_city[int(city)][str(metric)], dtype=np.float64)
                    ax.plot(x, y, color=c, marker=mk, lw=2.0, ms=5.5, alpha=0.95)
                ax.set_xticks(x)
                ax.set_xticklabels(ref_bins, rotation=0)
                ax.grid(True, axis="y", alpha=0.25, lw=0.8)
                ax.set_ylabel(_metric_label(str(metric)))
                if ri == 0:
                    ax.set_title(_metric_label(str(metric)))
                if ci == 0:
                    ax.text(0.02, 0.96, city_title, transform=ax.transAxes, ha="left", va="top", fontsize=10)
                if str(metric).endswith("_rate") or str(metric) == "success_rate":
                    ax.set_ylim(-0.02, 1.02)

                add_panel_label(ax, chr(ord("a") + int(panel)))
                panel += 1

        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(len(legend_labels), 4),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

        save_figure(fig, out_dir / f"shape_by_hops_{decode}_{stat}.png")
        save_figure(fig, out_dir / f"shape_by_hops_{decode}_{stat}.pdf")
        plt.close(fig)

    # Write a compact extracted summary (for tables / quick diff).
    summary: Dict[str, Any] = {
        "ok": True,
        "task": "waycasd_plot_shape_macro_summary",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": {
            "decode": decode,
            "stat": stat,
            "metrics": metrics,
            "cities": [int(c) for c in (args.cities or [])],
        },
        "bins": list(ref_bins),
        "methods": [asdict(m) for m in methods],
    }
    (out_dir / f"shape_by_hops_{decode}_{stat}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
