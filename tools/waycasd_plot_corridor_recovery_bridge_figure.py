#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv, to_hex, to_rgb

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


def _mix_with_white(color: str, frac: float) -> str:
    rgb = np.asarray(to_rgb(color), dtype=np.float64)
    out = (1.0 - float(frac)) * rgb + float(frac) * np.ones(3, dtype=np.float64)
    return to_hex(np.clip(out, 0.0, 1.0))


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


def _crop_white_border(img: np.ndarray, thr: float = 0.985) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        mask = arr < thr
    else:
        rgb = arr[..., :3]
        mask = np.any(rgb < thr, axis=-1)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return arr
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return arr[y0:y1, x0:x1]


def _split_representative_panels(rep_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cropped = _crop_white_border(rep_img)
    w = cropped.shape[1]
    mid = w // 2
    left = _crop_white_border(cropped[:, :mid])
    right = _crop_white_border(cropped[:, mid:])
    return left, right


def _strip_baked_annotations(panel_img: np.ndarray) -> np.ndarray:
    h, w = panel_img.shape[:2]
    top = int(round(0.19 * h))
    left = int(round(0.055 * w))
    right = int(round(0.015 * w))
    bottom = int(round(0.015 * h))
    y0 = min(max(top, 0), h - 2)
    x0 = min(max(left, 0), w - 2)
    y1 = max(y0 + 1, h - max(bottom, 0))
    x1 = max(x0 + 1, w - max(right, 0))
    return panel_img[y0:y1, x0:x1]


def _fade_panel(panel_img: np.ndarray, white_frac: float = 0.42) -> np.ndarray:
    arr = np.asarray(panel_img).copy()
    arr[..., :3] = (1.0 - white_frac) * arr[..., :3] + white_frac * 1.0
    return np.clip(arr, 0.0, 1.0)


def _compose_route_set_panel(gt_panel: np.ndarray, pred_panel: np.ndarray, route_color: str) -> np.ndarray:
    h = min(gt_panel.shape[0], pred_panel.shape[0])
    w = min(gt_panel.shape[1], pred_panel.shape[1])
    base = _fade_panel(np.asarray(gt_panel)[:h, :w], white_frac=0.48)
    pred = np.asarray(pred_panel)[:h, :w]
    hsv = rgb_to_hsv(np.clip(pred[..., :3], 0.0, 1.0))
    sat = hsv[..., 1]
    val = hsv[..., 2]
    route_mask = (sat > 0.28) & (val < 0.98)
    out = base.copy()
    out[..., :3][route_mask] = np.asarray(to_rgb(route_color), dtype=np.float64)
    return np.clip(out, 0.0, 1.0)


def _crop_fraction(panel_img: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    h, w = panel_img.shape[:2]
    ix0 = max(0, min(w - 2, int(round(x0 * w))))
    ix1 = max(ix0 + 1, min(w, int(round(x1 * w))))
    iy0 = max(0, min(h - 2, int(round(y0 * h))))
    iy1 = max(iy0 + 1, min(h, int(round(y1 * h))))
    return panel_img[iy0:iy1, ix0:ix1]


def _extract_companion_from_legacy_bridge(legacy_png: Path) -> List[np.ndarray]:
    legacy = mpimg.imread(str(legacy_png))
    # Crop only the embedded map regions from the previous 4-panel layout.
    boxes = [
        (0.60, 0.69, 0.77, 0.87),
        (0.84, 0.54, 0.98, 0.95),
    ]
    out: List[np.ndarray] = []
    for x0, y0, x1, y1 in boxes:
        crop = _crop_fraction(legacy, x0, y0, x1, y1)
        crop = _crop_white_border(crop)
        out.append(crop)
    return out


def _load_route_panel_from_hero(hero_png: Path, route_color: str) -> np.ndarray:
    hero_img = mpimg.imread(str(hero_png))
    left, right = _split_representative_panels(hero_img)
    left = _strip_baked_annotations(left)
    right = _strip_baked_annotations(right)
    return _compose_route_set_panel(left, right, route_color)


def _load_main_route_panel(main_png: Path, route_color: str) -> np.ndarray:
    meta_path = main_png.with_suffix(".meta.json")
    if meta_path.is_file():
        try:
            meta = _read_json(meta_path)
            figure_mode = str(meta.get("inputs", {}).get("figure_mode", "")).strip().lower()
            if figure_mode == "overlay_only":
                return _crop_white_border(mpimg.imread(str(main_png)))
        except Exception:
            pass
    return _load_route_panel_from_hero(main_png, route_color)


def _parse_od_from_name(name: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"(\d+)_(\d+)\.png$", str(name))
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def _lookup_od_row(rows: List[Dict[str, Any]], start_way: int, dest_way: int) -> Optional[Dict[str, Any]]:
    for row in rows:
        if int(row.get("start_way", -1)) == int(start_way) and int(row.get("dest_way", -1)) == int(dest_way):
            return row
    return None


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Plot bridge figure: route-space corridor view + dataset-scale coverage.")
    ap.add_argument("--main_png", type=Path, default=None)
    ap.add_argument("--aux_png", action="append", default=[])
    ap.add_argument("--representative_png", type=Path, default=None, help="Backward-compatible alias for --main_png.")
    ap.add_argument("--legacy_bridge_png", type=Path, default=None, help="Existing bridge figure used to recover companion maps when aux PNGs are unavailable.")
    ap.add_argument("--legacy_bridge_meta", type=Path, default=None, help="Meta JSON of the existing bridge figure.")
    ap.add_argument("--phasec_json", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--stem", type=str, default="fig2_corridor_recovery_bridge")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    rep_png_arg = args.main_png if args.main_png is not None else args.representative_png
    if rep_png_arg is None:
        raise SystemExit("[FATAL] need --main_png (or legacy --representative_png).")
    rep_png = Path(rep_png_arg)
    phasec_json = Path(args.phasec_json)
    out_dir = Path(args.out_dir)
    if not rep_png.is_file():
        raise SystemExit(f"[FATAL] representative image not found: {rep_png}")
    if not phasec_json.is_file():
        raise SystemExit(f"[FATAL] phaseC json not found: {phasec_json}")
    out_dir.mkdir(parents=True, exist_ok=True)

    d = _read_json(phasec_json)
    methods = d.get("methods", {})
    if not isinstance(methods, dict) or not methods:
        raise SystemExit(f"[FATAL] malformed methods in {phasec_json}")
    ordered = sorted(methods.items(), key=lambda kv: _method_order(kv[0]))
    cascade_key = None
    for raw_name, _obj in ordered:
        if _pretty_method_label(raw_name) == "CascadeTraj":
            cascade_key = raw_name
            break
    if cascade_key is None:
        cascade_key = ordered[0][0]

    rep_b = _load_main_route_panel(rep_png, _mix_with_white(OKABE_ITO["vermillion"], 0.02))
    rep_meta_path = rep_png.with_suffix(".meta.json")
    if not rep_meta_path.is_file():
        raise SystemExit(f"[FATAL] representative meta not found: {rep_meta_path}")
    rep_meta = _read_json(rep_meta_path)
    sel_od = rep_meta.get("selected_od", {})
    sw = int(sel_od.get("start_way"))
    dw = int(sel_od.get("dest_way"))
    od_rows = methods[cascade_key].get("per_od", [])
    if not isinstance(od_rows, list) or not od_rows:
        raise SystemExit(f"[FATAL] missing per_od for {cascade_key}")
    od_pool = [r for r in od_rows if int(r.get("n_gt_routes", 0)) >= 4]
    if not od_pool:
        raise SystemExit("[FATAL] no multi-route ODs (n_gt_routes >= 4) found for ECDF panel")
    sel_row = None
    for r in od_pool:
        if int(r.get("start_way", -1)) == sw and int(r.get("dest_way", -1)) == dw:
            sel_row = r
            break
    if sel_row is None:
        raise SystemExit(f"[FATAL] selected OD ({sw}, {dw}) not found in multi-route OD pool")
    x_all = np.sort(np.asarray([float(r["gt_coverage_at_k"]) for r in od_pool], dtype=np.float64))

    example_specs = [{"name": "Illustrative", "od": (sw, dw), "marker": "o"}]
    example_points: List[Dict[str, Any]] = []
    for spec in example_specs:
        if spec["od"] is None:
            continue
        row = _lookup_od_row(od_pool, spec["od"][0], spec["od"][1])
        if row is None:
            continue
        x_cov = float(row["gt_coverage_at_k"])
        rank = int(np.sum(x_all <= x_cov))
        example_points.append(
            {
                "name": spec["name"],
                "coverage": x_cov,
                "ecdf": rank / float(x_all.size),
                "marker": spec["marker"],
                "start_way": int(spec["od"][0]),
                "dest_way": int(spec["od"][1]),
            }
        )

    with paper_style():
        fig = plt.figure(figsize=(14.2, 5.8), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.76, 0.94], height_ratios=[1.0, 0.78])
        ax_route = fig.add_subplot(gs[:, 0])
        ax_curve = fig.add_subplot(gs[0, 1])
        ax_ecdf = fig.add_subplot(gs[1, 1])

        ax_route.imshow(rep_b)
        ax_route.axis("off")
        add_panel_label(ax_route, "a", fontsize=20)

        for raw_name, obj in ordered:
            cov_tau = obj.get("coverage_vs_tau", [])
            if not isinstance(cov_tau, list) or not cov_tau:
                continue
            tau = np.asarray([float(r["tau"]) for r in cov_tau], dtype=np.float64)
            mean = np.asarray([float(r["mean"]) for r in cov_tau], dtype=np.float64)
            label = _pretty_method_label(raw_name)
            base_color = _method_color(raw_name)
            is_ours = label == "CascadeTraj"
            is_sp = label == "Shortest Path"
            if is_ours:
                color = _mix_with_white(base_color, 0.08)
            elif is_sp:
                color = _mix_with_white(base_color, 0.15)
            else:
                color = _mix_with_white(base_color, 0.22)
            lw = 3.0 if is_ours else 1.7
            alpha = 0.96 if is_ours else 0.82
            ls = "-" if not is_sp else "--"
            ms = 4.2 if is_ours else 3.6
            ax_curve.plot(
                tau, mean,
                color=color,
                lw=lw,
                alpha=alpha,
                linestyle=ls,
                marker="o",
                markersize=ms,
                label=label,
            )

        ax_curve.set_xlabel(r"Jaccard threshold $\tau$", fontsize=16)
        ax_curve.set_ylabel(r"Coverage@$K$", fontsize=16)
        ax_curve.set_xlim(0.08, 0.92)
        ax_curve.set_ylim(-0.01, 0.74)
        ax_curve.grid(False)
        ax_curve.tick_params(axis="both", labelsize=13)
        ax_curve.legend(frameon=False, fontsize=12, loc="upper right")
        add_panel_label(ax_curve, "b", fontsize=20)

        ecdf_color = _mix_with_white(OKABE_ITO["black"], 0.42)
        y_all = np.arange(1, x_all.size + 1, dtype=np.float64) / float(x_all.size)
        ax_ecdf.step(x_all, y_all, where="post", color=ecdf_color, lw=2.2)
        point_color = _mix_with_white(OKABE_ITO["vermillion"], 0.10)
        for pt in example_points:
            ax_ecdf.scatter(
                [pt["coverage"]],
                [pt["ecdf"]],
                s=58,
                marker=pt["marker"],
                facecolor=point_color,
                edgecolor="white",
                linewidth=0.9,
                zorder=4,
                label=pt["name"],
            )
        ax_ecdf.set_xlabel(r"Coverage@$K$ ($\tau=0.3$)", fontsize=15)
        ax_ecdf.set_ylabel("ECDF", fontsize=15)
        ax_ecdf.set_xlim(-0.02, 1.02)
        ax_ecdf.set_ylim(-0.02, 1.02)
        ax_ecdf.grid(False)
        ax_ecdf.tick_params(axis="both", labelsize=13)
        handles, labels = ax_ecdf.get_legend_handles_labels()
        if handles:
            uniq: Dict[str, Any] = {}
            for h, lab in zip(handles, labels):
                if lab not in uniq:
                    uniq[lab] = h
            ax_ecdf.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=11, loc="lower right")
        add_panel_label(ax_ecdf, "c", fontsize=20)

        out_png = out_dir / f"{args.stem}.png"
        out_pdf = out_dir / f"{args.stem}.pdf"
        save_figure(fig, out_png)
        save_figure(fig, out_pdf)
        plt.close(fig)

    meta = {
        "ok": True,
        "task": "waycasd_plot_corridor_recovery_bridge_figure",
        "representative_png": str(rep_png),
        "representative_meta": str(rep_meta_path),
        "source_json": str(phasec_json),
        "selected_od": {
            "start_way": sw,
            "dest_way": dw,
            "coverage_at_k": float(sel_row["gt_coverage_at_k"]),
            "coverage_percentile": float(np.sum(x_all <= float(sel_row["gt_coverage_at_k"])) / float(x_all.size)),
            "n_pool": len(od_pool),
        },
        "example_points": example_points,
        "aux_pngs": [],
        "methods": [
            {
                "raw_name": raw_name,
                "pretty_name": _pretty_method_label(raw_name),
                "arrival": float(obj["arrival_rate"]),
                "covtau_auc": float(obj["coverage_vs_tau_auc"]),
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
