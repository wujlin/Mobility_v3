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
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


def _seq_to_xy(seq: Sequence[int], *, way_center_x: np.ndarray, way_center_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cx = np.asarray(way_center_x, dtype=np.float64).reshape(-1)
    cy = np.asarray(way_center_y, dtype=np.float64).reshape(-1)
    s = np.asarray([int(x) for x in seq], dtype=np.int64).reshape(-1)
    if s.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    s = np.clip(s, 0, cx.shape[0] - 1)
    return cx[s], cy[s]


def _compute_bbox(xs: List[np.ndarray], ys: List[np.ndarray], *, pad_frac: float = 0.2) -> Tuple[float, float, float, float]:
    x = np.concatenate([a.reshape(-1) for a in xs if a.size > 0], axis=0)
    y = np.concatenate([a.reshape(-1) for a in ys if a.size > 0], axis=0)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)
    px = float(pad_frac) * dx
    py = float(pad_frac) * dy
    return xmin - px, xmax + px, ymin - py, ymax + py


def _fmt_km(m: Optional[float]) -> str:
    if m is None:
        return "n/a"
    try:
        v = float(m)
    except Exception:
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    if abs(v) >= 10000.0:
        return f"{v/1000.0:.1f}km"
    return f"{v:.0f}m"


@dataclass(frozen=True)
class RouteRecord:
    route_id: int
    city: int
    gt_hops: int
    gt_way_ids: List[int]
    start_way: int
    dest_way: int
    pred_way_ids: List[int]
    success: bool
    hit_wall: bool
    dead_end: bool
    has_loop: bool
    jaccard: float
    len_ratio: float
    frechet_m: float
    dtw_m: float
    final_error_m: float


def _extract_route_records(*, rep: dict, decode: str) -> Dict[int, Dict[int, RouteRecord]]:
    per_route = rep.get("per_route", []) or []
    if not isinstance(per_route, list):
        raise ValueError("per_route must be a list")

    out: Dict[int, Dict[int, RouteRecord]] = {}
    for r in per_route:
        if not isinstance(r, dict):
            continue
        city = int(r.get("city", -1))
        rid = r.get("route_id", None)
        if rid is None:
            continue
        rid_i = int(rid)

        gt = r.get("gt_way_ids", None)
        if not isinstance(gt, list) or not gt:
            raise ValueError("per_route record missing gt_way_ids; re-run eval with --dump_way_seqs")
        gt_ids = [int(x) for x in gt]

        start_way = int(r.get("start_way", gt_ids[0]))
        dest_way = int(r.get("dest_way", gt_ids[-1]))

        dec = r.get(str(decode), None)
        if not isinstance(dec, dict):
            continue
        pred = dec.get("pred_way_ids", None)
        if not isinstance(pred, list) or not pred:
            raise ValueError(f"per_route record missing {decode}.pred_way_ids; re-run eval with --dump_way_seqs")
        pred_ids = [int(x) for x in pred]

        out.setdefault(city, {})[rid_i] = RouteRecord(
            route_id=int(rid_i),
            city=int(city),
            gt_hops=int(r.get("gt_hops", max(0, len(gt_ids) - 1))),
            gt_way_ids=gt_ids,
            start_way=int(start_way),
            dest_way=int(dest_way),
            pred_way_ids=pred_ids,
            success=bool(dec.get("success", False)),
            hit_wall=bool(dec.get("hit_wall", False)),
            dead_end=bool(dec.get("dead_end", False)),
            has_loop=bool(dec.get("has_loop", False)),
            jaccard=float(dec.get("jaccard", float("nan"))),
            len_ratio=float(dec.get("len_ratio", float("nan"))),
            frechet_m=float(dec.get("frechet_m", float("nan"))),
            dtw_m=float(dec.get("dtw_m", float("nan"))),
            final_error_m=float(dec.get("final_error_m", float("nan"))),
        )

    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Flow end-to-end micro case study: plot GT vs multiple methods (from per_route json with dumped sequences).")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--style", type=str, choices=["paper"], default="paper")
    p.add_argument("--method", type=str, action="append", required=True, help="Repeatable: NAME=PATH to per_route json (must include gt_way_ids and pred_way_ids).")
    p.add_argument("--decode", type=str, choices=["beam", "greedy"], default="beam")
    p.add_argument("--way_features_npz", type=Path, default=None, help="Optional override (otherwise inferred from the first method json inputs).")
    p.add_argument("--city_name", type=str, nargs="*", default=["0:Detroit", "1:Columbus"])
    p.add_argument("--pad_frac", type=float, default=0.2)
    p.add_argument("--bg_alpha", type=float, default=0.35)
    p.add_argument("--bg_s", type=float, default=0.8)

    # Case selection
    p.add_argument("--easy_min_hops", type=int, default=5)
    p.add_argument("--easy_max_hops", type=int, default=20)
    p.add_argument("--improved_min_hops", type=int, default=20)
    p.add_argument("--hard_min_hops", type=int, default=40)
    p.add_argument("--baseline", type=str, default=None, help="Baseline method name (default: first --method).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if not args.method or len(args.method) < 2:
        raise SystemExit("[FATAL] Need at least 2 --method entries (baseline + treatment).")

    methods: List[Tuple[str, Path, dict]] = []
    for spec in args.method:
        name, path = _parse_name_path(str(spec))
        _require_file(path)
        rep = _read_json(path)
        methods.append((name, path, rep))

    decode = str(args.decode)
    baseline_name = str(args.baseline) if args.baseline is not None else str(methods[0][0])
    name_to_idx = {name: i for i, (name, _, _) in enumerate(methods)}
    if baseline_name not in name_to_idx:
        raise SystemExit(f"[FATAL] baseline {baseline_name!r} not found in --method list.")

    # Infer way_features_npz
    if args.way_features_npz is not None:
        way_features_npz = Path(args.way_features_npz)
    else:
        inputs0 = methods[0][2].get("inputs") or {}
        wf0 = inputs0.get("way_features_npz", None)
        if wf0 is None:
            raise SystemExit("[FATAL] cannot infer way_features_npz; pass --way_features_npz explicitly.")
        way_features_npz = Path(str(wf0))
    _require_file(way_features_npz)
    wf = np.load(str(way_features_npz), allow_pickle=True)
    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)

    city_names: Dict[int, str] = {}
    for raw in args.city_name or []:
        s = str(raw)
        if ":" not in s:
            continue
        k, v = s.split(":", 1)
        try:
            city_names[int(k)] = str(v).strip()
        except Exception:
            continue
    city_names.setdefault(0, "Detroit")
    city_names.setdefault(1, "Columbus")

    # Extract records per method
    rec_by_method: List[Dict[int, Dict[int, RouteRecord]]] = []
    for name, path, rep in methods:
        try:
            rec_by_method.append(_extract_route_records(rep=rep, decode=decode))
        except Exception as e:
            raise SystemExit(f"[FATAL] failed to parse {path}: {e}")

    base_i = name_to_idx[baseline_name]

    # Pick cases per city
    picked: Dict[Tuple[int, str], int] = {}
    for city in (0, 1):
        # Intersect route ids across all methods (same sampled set expected).
        rid_sets = []
        for mrec in rec_by_method:
            rid_sets.append(set(mrec.get(int(city), {}).keys()))
        common = set.intersection(*rid_sets) if rid_sets else set()
        if not common:
            raise SystemExit(f"[FATAL] no common route_ids across methods for city={city}.")

        def base_rec(rid: int) -> RouteRecord:
            return rec_by_method[base_i][int(city)][int(rid)]

        def any_treat_success(rid: int) -> bool:
            for mi in range(len(methods)):
                if mi == base_i:
                    continue
                if rec_by_method[mi][int(city)][int(rid)].success:
                    return True
            return False

        # Easy: baseline success, mid-length [min,max]
        easy = [
            rid
            for rid in common
            if base_rec(rid).success and int(args.easy_min_hops) <= int(base_rec(rid).gt_hops) <= int(args.easy_max_hops)
        ]
        if not easy:
            easy = [rid for rid in common if base_rec(rid).success]
        if not easy:
            raise SystemExit(f"[FATAL] no easy candidates for city={city} (baseline never succeeds).")
        target = int((int(args.easy_min_hops) + int(args.easy_max_hops)) // 2)
        easy.sort(key=lambda rid: (abs(int(base_rec(rid).gt_hops) - target), -int(base_rec(rid).gt_hops)))
        picked[(int(city), "easy")] = int(easy[0])

        # Improved: baseline fail, treatment success, prefer long.
        improved = [
            rid
            for rid in common
            if (not base_rec(rid).success)
            and any_treat_success(rid)
            and int(base_rec(rid).gt_hops) >= int(args.improved_min_hops)
        ]
        if not improved:
            improved = [rid for rid in common if (not base_rec(rid).success) and any_treat_success(rid)]
        if improved:
            improved.sort(key=lambda rid: (-int(base_rec(rid).gt_hops), float(base_rec(rid).frechet_m)))
            picked[(int(city), "improved")] = int(improved[0])
        else:
            picked[(int(city), "improved")] = int(easy[0])

        # Hard: all fail, prefer long.
        hard = [
            rid
            for rid in common
            if all((not rec_by_method[mi][int(city)][int(rid)].success) for mi in range(len(methods)))
            and int(base_rec(rid).gt_hops) >= int(args.hard_min_hops)
        ]
        if not hard:
            hard = [rid for rid in common if all((not rec_by_method[mi][int(city)][int(rid)].success) for mi in range(len(methods)))]
        if not hard:
            hard = [rid for rid in common if not base_rec(rid).success]
        hard.sort(key=lambda rid: -int(base_rec(rid).gt_hops))
        picked[(int(city), "hard")] = int(hard[0])

    # Plot
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = [
        OKABE_ITO["vermillion"],
        OKABE_ITO["bluish_green"],
        OKABE_ITO["orange"],
        OKABE_ITO["sky_blue"],
        OKABE_ITO.get("reddish_purple", OKABE_ITO["gray"]),
        OKABE_ITO["black"],
    ]
    linestyles = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]

    with paper_style() if args.style == "paper" else plt.rc_context({}):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13.6, 7.6), constrained_layout=True)
        legend_handles: List[Any] = []
        legend_labels: List[str] = []

        # Legend: GT + methods + O/D + background
        legend_handles.append(Line2D([0], [0], color=OKABE_ITO["blue"], lw=2.4, ls="-"))
        legend_labels.append("GT")
        for mi, (name, _, _) in enumerate(methods):
            c = colors[int(mi) % len(colors)]
            ls = linestyles[int(mi) % len(linestyles)]
            legend_handles.append(Line2D([0], [0], color=c, lw=2.0, ls=ls))
            legend_labels.append(str(name))
        legend_handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#000000", markersize=7))
        legend_labels.append("O")
        legend_handles.append(Line2D([0], [0], marker="*", color="none", markerfacecolor="#000000", markersize=9))
        legend_labels.append("D")
        legend_handles.append(Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#DDDDDD", markeredgecolor="none", markersize=6, alpha=0.8))
        legend_labels.append("Road graph")

        cat_order = ["easy", "improved", "hard"]
        panel = 0
        for r, city in enumerate([0, 1]):
            for cci, cat in enumerate(cat_order):
                ax = axes[r, cci]
                rid = picked[(int(city), str(cat))]

                # Use baseline record as source of GT seq.
                gt = rec_by_method[base_i][int(city)][int(rid)].gt_way_ids
                gt_x, gt_y = _seq_to_xy(gt, way_center_x=way_center_x, way_center_y=way_center_y)

                # O/D at way-centers (align with trajectories)
                sx, sy = (float(gt_x[0]), float(gt_y[0])) if gt_x.size else (0.0, 0.0)
                dx, dy = (float(gt_x[-1]), float(gt_y[-1])) if gt_x.size else (0.0, 0.0)

                xs = [gt_x, np.asarray([sx, dx], dtype=np.float64)]
                ys = [gt_y, np.asarray([sy, dy], dtype=np.float64)]

                pred_xy: List[Tuple[np.ndarray, np.ndarray]] = []
                for mi in range(len(methods)):
                    pred = rec_by_method[mi][int(city)][int(rid)].pred_way_ids
                    px, py = _seq_to_xy(pred, way_center_x=way_center_x, way_center_y=way_center_y)
                    pred_xy.append((px, py))
                    xs.append(px)
                    ys.append(py)

                xmin, xmax, ymin, ymax = _compute_bbox(xs, ys, pad_frac=float(args.pad_frac))
                mask = (way_center_x >= xmin) & (way_center_x <= xmax) & (way_center_y >= ymin) & (way_center_y <= ymax)
                ax.scatter(
                    way_center_x[mask],
                    way_center_y[mask],
                    s=float(args.bg_s),
                    c="#DDDDDD",
                    alpha=float(args.bg_alpha),
                    linewidths=0,
                    zorder=1,
                )

                ax.plot(gt_x, gt_y, color=OKABE_ITO["blue"], lw=2.4, ls="-", zorder=4)
                for mi, (px, py) in enumerate(pred_xy):
                    col = colors[int(mi) % len(colors)]
                    ls = linestyles[int(mi) % len(linestyles)]
                    ax.plot(px, py, color=col, lw=2.0, ls=ls, zorder=5)

                ax.scatter([sx], [sy], s=80, c="#000000", marker="o", edgecolors="white", linewidths=0.8, zorder=6)
                ax.scatter([dx], [dy], s=90, c="#000000", marker="*", edgecolors="white", linewidths=0.8, zorder=6)

                city_title = city_names.get(int(city), f"city{int(city)}")
                cat_title = {"easy": "Easy", "improved": "Improved", "hard": "Hard"}[str(cat)]
                ax.set_title(f"{city_title} · {cat_title} (rid={int(rid)})")

                # Annotation
                gt_hops = int(rec_by_method[base_i][int(city)][int(rid)].gt_hops)
                lines = [f"GT hops: {gt_hops}"]
                for mi, (name, _, _) in enumerate(methods):
                    rr = rec_by_method[mi][int(city)][int(rid)]
                    tag = (
                        "✓" if rr.success else ("HW" if rr.hit_wall else ("DE" if rr.dead_end else ("Loop" if rr.has_loop else "Fail")))
                    )
                    lines.append(
                        f"{name}: {tag} | Fr={_fmt_km(rr.frechet_m)} | Len={rr.len_ratio:.2f} | Err={_fmt_km(rr.final_error_m)}"
                    )
                ax.text(
                    0.98,
                    0.02,
                    "\n".join(lines),
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="#222222",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
                    zorder=10,
                )

                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_aspect("equal", adjustable="box")
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])
                add_panel_label(ax, chr(ord("a") + int(panel)))
                panel += 1

        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(legend_labels), 6),
            frameon=False,
        )

        out_png = out_dir / f"flow_micro_compare_{decode}.png"
        out_pdf = out_dir / f"flow_micro_compare_{decode}.pdf"
        save_figure(fig, out_png, dpi=300)
        save_figure(fig, out_pdf)
        plt.close(fig)

    meta = {
        "ok": True,
        "task": "waycasd_plot_flow_micro_compare",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "decode": decode,
        "baseline": baseline_name,
        "methods": [{"name": n, "path": str(p)} for n, p, _ in methods],
        "picked": {f"{city}:{cat}": int(rid) for (city, cat), rid in picked.items()},
        "way_features_npz": str(way_features_npz),
    }
    (out_dir / "flow_micro_compare_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] saved {out_png}")
    print(f"[OK] saved {out_pdf}")


if __name__ == "__main__":
    main()

