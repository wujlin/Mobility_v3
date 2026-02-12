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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plot_style import OKABE_ITO, add_panel_label, paper_style, save_figure


@dataclass(frozen=True)
class MethodSpec:
    label: str
    decode: str
    path: Path


@dataclass(frozen=True)
class RouteRec:
    route_id: int
    city: int
    start_way: int
    dest_way: int
    gt_hops: int
    gt_way_ids: List[int]
    pred_way_ids: List[int]
    success: bool


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[FATAL] file not found: {path}")
    if not path.is_file():
        raise SystemExit(f"[FATAL] not a file: {path}")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_method_spec(spec: str) -> MethodSpec:
    s = str(spec or "").strip()
    if "=" not in s:
        raise SystemExit(f"[FATAL] bad --method spec: {spec!r}; expect LABEL|DECODE=PATH")
    left, right = s.split("=", 1)
    left = left.strip()
    path = Path(right.strip()).expanduser()
    if "|" in left:
        label, decode = left.rsplit("|", 1)
    else:
        label, decode = left, "greedy"
    label = label.strip()
    decode = decode.strip().lower()
    if decode not in {"greedy", "beam"}:
        raise SystemExit(f"[FATAL] bad decode in --method: {decode!r}, expect greedy/beam")
    if not label:
        raise SystemExit(f"[FATAL] empty label in --method spec: {spec!r}")
    return MethodSpec(label=label, decode=decode, path=path)


def _extract_records(per_route_json: Path, decode: str) -> List[RouteRec]:
    root = _read_json(per_route_json)
    per_route = root.get("per_route")
    if not isinstance(per_route, list):
        raise SystemExit(f"[FATAL] {per_route_json}: missing per_route list")
    out: List[RouteRec] = []
    for rec in per_route:
        if not isinstance(rec, dict):
            continue
        gt = rec.get("gt_way_ids")
        if not isinstance(gt, list) or not gt:
            continue
        dec = rec.get(str(decode))
        if not isinstance(dec, dict):
            continue
        pred = dec.get("pred_way_ids")
        if not isinstance(pred, list) or not pred:
            continue
        out.append(
            RouteRec(
                route_id=int(rec.get("route_id", -1)),
                city=int(rec.get("city", -1)),
                start_way=int(rec.get("start_way", gt[0])),
                dest_way=int(rec.get("dest_way", gt[-1])),
                gt_hops=int(rec.get("gt_hops", max(0, len(gt) - 1))),
                gt_way_ids=[int(x) for x in gt],
                pred_way_ids=[int(x) for x in pred],
                success=bool(dec.get("success", False)),
            )
        )
    if not out:
        raise SystemExit(
            f"[FATAL] no valid rows in {per_route_json} (decode={decode}); "
            f"make sure eval used --out_per_route_json + --dump_way_seqs."
        )
    return out


def _seq_to_xy(seq: Sequence[int], x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray([int(v) for v in seq], dtype=np.int64)
    s = np.clip(s, 0, x.shape[0] - 1)
    return x[s], y[s]


def _bbox(xs: List[np.ndarray], ys: List[np.ndarray], pad_frac: float) -> Tuple[float, float, float, float]:
    xx = np.concatenate([a.reshape(-1) for a in xs if a.size > 0], axis=0)
    yy = np.concatenate([a.reshape(-1) for a in ys if a.size > 0], axis=0)
    xmin, xmax = float(np.min(xx)), float(np.max(xx))
    ymin, ymax = float(np.min(yy)), float(np.max(yy))
    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)
    return (
        xmin - float(pad_frac) * dx,
        xmax + float(pad_frac) * dx,
        ymin - float(pad_frac) * dy,
        ymax + float(pad_frac) * dy,
    )


def _pick_od_from_phasec(
    *,
    phasec_json: Path,
    hero_label: str,
    min_gt_routes: int,
    min_pred_success: int,
    min_self_diversity: float,
) -> List[Dict[str, Any]]:
    d = _read_json(phasec_json)
    methods = d.get("methods", {})
    if not isinstance(methods, dict) or hero_label not in methods:
        avail = list(methods.keys()) if isinstance(methods, dict) else []
        raise SystemExit(f"[FATAL] hero label not found in phaseC json: {hero_label}; available={avail}")
    per_od = methods[hero_label].get("per_od", [])
    if not isinstance(per_od, list) or not per_od:
        raise SystemExit("[FATAL] phaseC json has no per_od. re-run with --save_per_od.")
    cands: List[Dict[str, Any]] = []
    for r in per_od:
        if not isinstance(r, dict):
            continue
        n_gt = int(r.get("n_gt_routes", 0))
        n_pred = int(r.get("n_pred_success_used", 0))
        div = r.get("self_diversity_at_k", None)
        if div is None:
            continue
        divf = float(div)
        if n_gt < int(min_gt_routes):
            continue
        if n_pred < int(min_pred_success):
            continue
        if divf < float(min_self_diversity):
            continue
        cands.append(r)
    if not cands:
        raise SystemExit(
            "[FATAL] no OD candidate passed phaseC filters. "
            "Try relaxing --min_gt_routes/--min_pred_success/--min_self_diversity."
        )
    # Prefer stronger diversity first, then coverage, then gt count.
    cands.sort(
        key=lambda z: (
            float(z.get("self_diversity_at_k", -1.0)),
            float(z.get("gt_coverage_at_k", -1.0)),
            int(z.get("n_gt_routes", 0)),
        ),
        reverse=True,
    )
    return cands


def _method_rank_for_hero(label: str) -> int:
    s = str(label).lower()
    if "way-casd" in s or "waycasd" in s:
        return 0
    if "rnn" in s:
        return 1
    if "transformer" in s or "tr-ar" in s:
        return 2
    return 3


def _select_gt_rows_for_od(rows: Sequence[RouteRec], city: int, sw: int, dw: int) -> List[RouteRec]:
    return [
        r for r in rows
        if int(r.city) == int(city) and int(r.start_way) == int(sw) and int(r.dest_way) == int(dw)
    ]


def _median_gt_hops(rows: Sequence[RouteRec]) -> float:
    if not rows:
        return float("nan")
    vals = np.asarray([int(r.gt_hops) for r in rows], dtype=np.int64)
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def _dedup_keep_order(seqs: List[List[int]], limit: int) -> List[List[int]]:
    out: List[List[int]] = []
    seen: set[Tuple[int, ...]] = set()
    for s in seqs:
        key = tuple(int(x) for x in s)
        if key in seen:
            continue
        seen.add(key)
        out.append([int(x) for x in s])
        if len(out) >= int(limit):
            break
    return out


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Figure A hero: same OD, GT vs multiple methods at way-level.")
    ap.add_argument("--phasec_json", type=Path, required=True, help="Phase C json with per_od (must run with --save_per_od).")
    ap.add_argument("--hero_label", type=str, required=True, help="Method label in phaseC used for OD selection.")
    ap.add_argument("--method", action="append", required=True, help="Repeatable: LABEL|DECODE=PER_ROUTE_JSON.")
    ap.add_argument("--way_features_npz", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--city", type=int, default=0)
    ap.add_argument("--min_gt_routes", type=int, default=10)
    ap.add_argument("--min_pred_success", type=int, default=4)
    ap.add_argument("--min_self_diversity", type=float, default=0.6)
    ap.add_argument("--hops_min", type=int, default=20)
    ap.add_argument("--hops_max", type=int, default=40)
    ap.add_argument("--k_pred_per_method", type=int, default=10, help="Max successful predicted routes shown per method panel.")
    ap.add_argument("--max_gt_draw", type=int, default=80)
    ap.add_argument("--pad_frac", type=float, default=0.18)
    ap.add_argument("--bg_alpha", type=float, default=0.16)
    ap.add_argument("--bg_s", type=float, default=0.6)
    ap.add_argument("--od_start_way", type=int, default=None, help="Manual override start_way.")
    ap.add_argument("--od_dest_way", type=int, default=None, help="Manual override dest_way.")
    ap.add_argument("--keep_method_order", action="store_true", help="Keep --method order; default auto: Way-CASD, RNN, Transformer.")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _require_file(Path(args.way_features_npz))
    _require_file(Path(args.phasec_json))

    methods = [_parse_method_spec(s) for s in list(args.method)]
    if len(methods) < 3:
        raise SystemExit("[FATAL] need at least 3 --method specs for panels (b,c,d).")
    methods = methods[:3]
    if not bool(args.keep_method_order):
        methods = sorted(methods, key=lambda m: (_method_rank_for_hero(m.label), m.label.lower()))
    for m in methods:
        _require_file(m.path)

    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)
    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)

    rec_by_method: Dict[str, List[RouteRec]] = {}
    for m in methods:
        rec_by_method[m.label] = _extract_records(m.path, m.decode)

    gt_carrier_rows = rec_by_method[methods[0].label]
    if args.od_start_way is None or args.od_dest_way is None:
        od_cands = _pick_od_from_phasec(
            phasec_json=Path(args.phasec_json),
            hero_label=str(args.hero_label),
            min_gt_routes=int(args.min_gt_routes),
            min_pred_success=int(args.min_pred_success),
            min_self_diversity=float(args.min_self_diversity),
        )
        chosen = None
        rej_msgs: List[str] = []
        for od_row in od_cands:
            sw_try = int(od_row["start_way"])
            dw_try = int(od_row["dest_way"])
            gt_rows_try = _select_gt_rows_for_od(gt_carrier_rows, int(args.city), sw_try, dw_try)
            if not gt_rows_try:
                rej_msgs.append(f"OD=({sw_try},{dw_try}) reason=no_gt_rows")
                continue
            med_try = _median_gt_hops(gt_rows_try)
            if (not np.isfinite(med_try)) or (med_try < float(args.hops_min)) or (med_try > float(args.hops_max)):
                rej_msgs.append(
                    f"OD=({sw_try},{dw_try}) reason=gt_hops_med={med_try:.1f} "
                    f"outside[{int(args.hops_min)},{int(args.hops_max)}]"
                )
                continue
            chosen = (sw_try, dw_try, od_row, gt_rows_try, med_try)
            break
        if chosen is None:
            preview = "; ".join(rej_msgs[:6])
            raise SystemExit(
                "[FATAL] no phaseC candidate passed GT hops gate "
                f"[{int(args.hops_min)},{int(args.hops_max)}] for city={int(args.city)}. "
                f"Preview rejects: {preview}"
            )
        sw, dw, od_row, gt_rows_all, gt_hops_med = chosen
    else:
        sw, dw = int(args.od_start_way), int(args.od_dest_way)
        od_row = {"start_way": sw, "dest_way": dw, "manual": True}
        gt_rows_all = _select_gt_rows_for_od(gt_carrier_rows, int(args.city), int(sw), int(dw))
        gt_hops_med = _median_gt_hops(gt_rows_all)
    if not gt_rows_all:
        raise SystemExit(f"[FATAL] no GT rows found for city={int(args.city)} OD=({sw},{dw})")
    gt_rows_all.sort(key=lambda r: int(r.route_id))
    gt_rows = gt_rows_all[: int(args.max_gt_draw)]

    gt_seqs = [r.gt_way_ids for r in gt_rows]
    method_pred: Dict[str, List[List[int]]] = {}
    method_meta: Dict[str, Dict[str, Any]] = {}
    for m in methods:
        rows = [
            r for r in rec_by_method[m.label]
            if int(r.city) == int(args.city) and int(r.start_way) == int(sw) and int(r.dest_way) == int(dw)
        ]
        succ = [r.pred_way_ids for r in rows if bool(r.success)]
        succ = _dedup_keep_order(succ, limit=int(args.k_pred_per_method))
        method_pred[m.label] = succ
        method_meta[m.label] = {
            "n_routes_od": int(len(rows)),
            "n_success_od": int(sum(1 for r in rows if bool(r.success))),
            "n_unique_drawn": int(len(succ)),
        }

    # Build global bbox (all trajectories).
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for s in gt_seqs:
        x, y = _seq_to_xy(s, way_center_x, way_center_y)
        xs.append(x)
        ys.append(y)
    for ms in method_pred.values():
        for s in ms:
            x, y = _seq_to_xy(s, way_center_x, way_center_y)
            xs.append(x)
            ys.append(y)
    xmin, xmax, ymin, ymax = _bbox(xs, ys, float(args.pad_frac))

    # Start/dest marker from GT first route.
    sxy = _seq_to_xy(gt_rows[0].gt_way_ids, way_center_x, way_center_y)
    sx, sy = float(sxy[0][0]), float(sxy[1][0])
    dx, dy = float(sxy[0][-1]), float(sxy[1][-1])

    # Plot 2x2
    method_colors = [
        OKABE_ITO["vermillion"],
        OKABE_ITO["bluish_green"],
        OKABE_ITO["blue"],
    ]
    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
        axs = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

        # Common background.
        mask = (
            (way_center_x >= xmin) & (way_center_x <= xmax) &
            (way_center_y >= ymin) & (way_center_y <= ymax)
        )
        for ax in axs:
            ax.scatter(
                way_center_x[mask], way_center_y[mask],
                s=float(args.bg_s), c="#DADADA", alpha=float(args.bg_alpha), linewidths=0, zorder=1
            )
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])

        # (a) all GT routes
        ax = axs[0]
        for s in gt_seqs:
            x, y = _seq_to_xy(s, way_center_x, way_center_y)
            ax.plot(x, y, color=OKABE_ITO["black"], lw=1.2, alpha=0.35, zorder=3)
        ax.scatter([sx], [sy], s=80, c="#000000", marker="o", edgecolors="white", linewidths=0.8, zorder=4)
        ax.scatter([dx], [dy], s=90, c="#000000", marker="*", edgecolors="white", linewidths=0.8, zorder=4)
        ax.set_title(f"Ground Truth (n={len(gt_seqs)})")
        add_panel_label(ax, "a")

        # (b,c,d) methods
        for i, m in enumerate(methods, start=1):
            ax = axs[i]
            # Faint GT context
            for s in gt_seqs:
                x, y = _seq_to_xy(s, way_center_x, way_center_y)
                ax.plot(x, y, color=OKABE_ITO["gray"], lw=0.9, alpha=0.16, zorder=2)
            color = method_colors[(i - 1) % len(method_colors)]
            for s in method_pred[m.label]:
                x, y = _seq_to_xy(s, way_center_x, way_center_y)
                ax.plot(x, y, color=color, lw=2.1, alpha=0.9, zorder=4)
            ax.scatter([sx], [sy], s=80, c="#000000", marker="o", edgecolors="white", linewidths=0.8, zorder=5)
            ax.scatter([dx], [dy], s=90, c="#000000", marker="*", edgecolors="white", linewidths=0.8, zorder=5)
            mm = method_meta[m.label]
            ax.set_title(
                f"{m.label} ({m.decode})\n"
                f"success_in_OD={mm['n_success_od']}/{mm['n_routes_od']}, shown={mm['n_unique_drawn']}"
            )
            add_panel_label(ax, chr(ord("a") + i))

        # Global title
        fig.suptitle(
            f"Hero OD Comparison (city={int(args.city)}, OD=({sw},{dw}), "
            f"gt_hops_med={gt_hops_med:.1f})",
            y=1.01,
        )

        stem = f"hero_od_city{int(args.city)}_{int(sw)}_{int(dw)}"
        out_png = out_dir / f"{stem}.png"
        out_pdf = out_dir / f"{stem}.pdf"
        save_figure(fig, out_png)
        save_figure(fig, out_pdf)
        plt.close(fig)

    meta = {
        "ok": True,
        "task": "waycasd_plot_od_hero_figure",
        "selected_od": {
            "city": int(args.city),
            "start_way": int(sw),
            "dest_way": int(dw),
            "gt_hops_median": (float(gt_hops_med) if np.isfinite(gt_hops_med) else None),
            "phasec_row": od_row,
        },
        "methods": [{"label": m.label, "decode": m.decode, "path": str(m.path)} for m in methods],
        "method_meta": method_meta,
        "inputs": {
            "phasec_json": str(args.phasec_json),
            "way_features_npz": str(args.way_features_npz),
        },
        "outputs": {
            "png": str(out_png),
            "pdf": str(out_pdf),
        },
    }
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved: {out_png}")
    print(f"[OK] saved: {out_pdf}")
    print(f"[OK] saved: {out_dir / (stem + '.meta.json')}")


if __name__ == "__main__":
    main()
