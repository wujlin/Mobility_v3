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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.plot_style import OKABE_ITO, add_panel_label, paper_style, save_figure

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[FATAL] file not found: {path}")
    if not path.is_file():
        raise SystemExit(f"[FATAL] not a file: {path}")


def _build_city_index(rep: dict) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for c in rep.get("per_city", []) or []:
        if not isinstance(c, dict):
            continue
        city = int(c.get("city", -1))
        succ = [int(x) for x in (c.get("success_route_ids") or [])]
        fails = [f for f in (c.get("failures") or []) if isinstance(f, dict) and f.get("route_id") is not None]
        fail_ids = [int(f.get("route_id")) for f in fails]
        out[city] = {"success_ids": succ, "fail_ids": fail_ids}
    return out


def _bbox_from_points(x: np.ndarray, y: np.ndarray, *, pad_frac: float = 0.06) -> Tuple[float, float, float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return (0.0, 1.0, 0.0, 1.0)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)
    px = float(pad_frac) * dx
    py = float(pad_frac) * dy
    return xmin - px, xmax + px, ymin - py, ymax + py


def _hist2d(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    bins: int,
) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float64).reshape(-1)
    ys = np.asarray(ys, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size == 0:
        return np.zeros((bins, bins), dtype=np.float32)
    h, _, _ = np.histogram2d(ys, xs, bins=bins, range=[[ymin, ymax], [xmin, xmax]])
    return h.astype(np.float32, copy=False)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WayCASD macro overview: failure clustering + difficulty partition (by start location).")
    p.add_argument("--eval_dir", type=Path, required=True, help="Strong-ckpt eval dir (oracle_decode_greedy/beam10 json).")
    p.add_argument("--out_dir", type=Path, default=Path("_sync/wsa/paper_figures/waycasd_v1/macro"))
    p.add_argument("--style", type=str, choices=["paper"], default="paper")
    p.add_argument("--greedy_json", type=Path, default=None)
    p.add_argument("--beam10_json", type=Path, default=None)
    p.add_argument("--way_routes_npz", type=Path, default=None)
    p.add_argument("--way_features_npz", type=Path, default=None)
    p.add_argument("--city_name", type=str, nargs="*", default=["0:Detroit", "1:Columbus"])
    p.add_argument("--bins_fail", type=int, default=60, help="2D bins for failure density.")
    p.add_argument("--bins_rate", type=int, default=22, help="2D bins for success-rate partition (small N).")
    p.add_argument("--bg_alpha", type=float, default=0.25)
    p.add_argument("--bg_s", type=float, default=0.6)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    eval_dir = Path(args.eval_dir)

    greedy_json = Path(args.greedy_json) if args.greedy_json is not None else (eval_dir / "oracle_decode_greedy_n200.json")
    beam10_json = Path(args.beam10_json) if args.beam10_json is not None else (eval_dir / "oracle_decode_beam10_n200.json")
    _require_file(greedy_json)
    _require_file(beam10_json)
    greedy_rep = _read_json(greedy_json)
    beam10_rep = _read_json(beam10_json)

    inputs = greedy_rep.get("inputs") or {}
    way_routes_npz = Path(args.way_routes_npz) if args.way_routes_npz is not None else Path(inputs["way_routes_npz"])
    way_features_npz = Path(args.way_features_npz) if args.way_features_npz is not None else Path(inputs["way_features_npz"])
    _require_file(way_routes_npz)
    _require_file(way_features_npz)

    city_names: Dict[int, str] = {}
    for raw in args.city_name or []:
        if ":" not in str(raw):
            continue
        k, v = str(raw).split(":", 1)
        try:
            city_names[int(k)] = v.strip()
        except Exception:
            continue
    city_names.setdefault(0, "Detroit")
    city_names.setdefault(1, "Columbus")

    routes = load_way_routes_npz(Path(way_routes_npz))
    wf = np.load(str(way_features_npz), allow_pickle=True)
    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)

    g = _build_city_index(greedy_rep)
    b10 = _build_city_index(beam10_rep)

    # Collect per-city eval route IDs (200 each) from greedy json.
    per_city_routes: Dict[int, List[int]] = {}
    for city in (0, 1):
        succ = [int(x) for x in g.get(int(city), {}).get("success_ids", [])]
        fail = [int(x) for x in g.get(int(city), {}).get("fail_ids", [])]
        all_ids = sorted(set(succ + fail))
        per_city_routes[int(city)] = all_ids

    # Collect per-city start points and success flags (greedy), plus beam10 failure starts.
    city_data: Dict[int, Dict[str, Any]] = {}
    for city in (0, 1):
        rids = per_city_routes[int(city)]
        if not rids:
            continue
        starts_x: List[float] = []
        starts_y: List[float] = []
        greedy_succ_set = set(int(x) for x in g.get(int(city), {}).get("success_ids", []))
        succ_flags: List[int] = []
        for rid in rids:
            rid = int(rid)
            if int(routes.route_city[rid]) != int(city):
                continue
            sp = routes.start_pos[rid].astype(np.float64, copy=False).reshape(2)
            starts_y.append(float(sp[0]))
            starts_x.append(float(sp[1]))
            succ_flags.append(1 if int(rid) in greedy_succ_set else 0)

        fail10_ids = set(int(x) for x in b10.get(int(city), {}).get("fail_ids", []))
        fail10_x: List[float] = []
        fail10_y: List[float] = []
        for rid in rids:
            rid = int(rid)
            if rid not in fail10_ids:
                continue
            sp = routes.start_pos[rid].astype(np.float64, copy=False).reshape(2)
            fail10_y.append(float(sp[0]))
            fail10_x.append(float(sp[1]))

        starts_xa = np.asarray(starts_x, dtype=np.float64)
        starts_ya = np.asarray(starts_y, dtype=np.float64)
        xmin, xmax, ymin, ymax = _bbox_from_points(starts_xa, starts_ya, pad_frac=0.08)

        city_data[int(city)] = {
            "rids": rids,
            "starts_x": starts_xa,
            "starts_y": starts_ya,
            "succ_flags": np.asarray(succ_flags, dtype=np.int64),
            "fail10_x": np.asarray(fail10_x, dtype=np.float64),
            "fail10_y": np.asarray(fail10_y, dtype=np.float64),
            "bbox": (xmin, xmax, ymin, ymax),
        }

    # Precompute a shared vmax for failure density (log1p).
    vmax_fail = 0.0
    fail_hists: Dict[int, np.ndarray] = {}
    rate_hists: Dict[int, np.ndarray] = {}
    rate_counts: Dict[int, np.ndarray] = {}
    for city, d in city_data.items():
        xmin, xmax, ymin, ymax = d["bbox"]
        h_fail = _hist2d(d["fail10_x"], d["fail10_y"], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, bins=int(args.bins_fail))
        h_fail_log = np.log1p(h_fail)
        h_fail_log[h_fail <= 0] = np.nan
        fail_hists[int(city)] = h_fail_log
        if np.isfinite(h_fail_log).any():
            vmax_fail = max(vmax_fail, float(np.nanmax(h_fail_log)))

        # success-rate grid (greedy): sum(success)/count per bin.
        h_cnt = _hist2d(d["starts_x"], d["starts_y"], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, bins=int(args.bins_rate))
        # weighted by success flag
        xs = d["starts_x"]
        ys = d["starts_y"]
        mask = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[mask]
        ys = ys[mask]
        sf = d["succ_flags"][mask]
        h_succ = np.zeros_like(h_cnt, dtype=np.float32)
        if xs.size > 0:
            # Map points to bins (same binning as histogram2d).
            bx = np.clip(((xs - xmin) / max(1e-9, (xmax - xmin)) * float(args.bins_rate)).astype(np.int64), 0, int(args.bins_rate) - 1)
            by = np.clip(((ys - ymin) / max(1e-9, (ymax - ymin)) * float(args.bins_rate)).astype(np.int64), 0, int(args.bins_rate) - 1)
            for ix, iy, v in zip(bx.tolist(), by.tolist(), sf.tolist()):
                h_succ[int(iy), int(ix)] += float(v)
        rate = np.divide(h_succ, np.maximum(1.0, h_cnt), dtype=np.float32)
        rate[h_cnt <= 0] = np.nan
        rate_hists[int(city)] = rate
        rate_counts[int(city)] = h_cnt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "ok": True,
        "task": "waycasd_plot_city_macro_overview",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "eval_dir": str(eval_dir),
        "greedy_json": str(greedy_json),
        "beam10_json": str(beam10_json),
        "inputs": {
            "way_routes_npz": str(way_routes_npz),
            "way_features_npz": str(way_features_npz),
        },
        "bins_fail": int(args.bins_fail),
        "bins_rate": int(args.bins_rate),
        "per_city": {},
    }
    for city, d in city_data.items():
        meta["per_city"][str(int(city))] = {
            "city_name": city_names.get(int(city), f"city{int(city)}"),
            "n_eval_routes": int(len(d["rids"])),
            "n_fail_beam10": int(d["fail10_x"].size),
            "bbox": {"xmin": d["bbox"][0], "xmax": d["bbox"][1], "ymin": d["bbox"][2], "ymax": d["bbox"][3]},
        }
    (out_dir / "waycasd_city_macro_overview_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Plot: rows=cities, cols=[fail density, greedy success rate].
    with paper_style():
        cmap_fail = plt.get_cmap("magma").copy()
        cmap_fail.set_bad(color=(1.0, 1.0, 1.0, 0.0))
        cmap_rate = plt.get_cmap("viridis").copy()
        cmap_rate.set_bad(color=(1.0, 1.0, 1.0, 0.0))

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.6, 7.2), constrained_layout=True)
        panel = 0

        ims_fail = []
        ims_rate = []
        for r, city in enumerate([0, 1]):
            if int(city) not in city_data:
                for cc in (0, 1):
                    axes[r, cc].axis("off")
                continue
            d = city_data[int(city)]
            xmin, xmax, ymin, ymax = d["bbox"]
            mask_bg = (way_center_x >= xmin) & (way_center_x <= xmax) & (way_center_y >= ymin) & (way_center_y <= ymax)

            # A: beam10 failure density (start points)
            ax0 = axes[r, 0]
            ax0.scatter(
                way_center_x[mask_bg],
                way_center_y[mask_bg],
                s=float(args.bg_s),
                c="#DDDDDD",
                alpha=float(args.bg_alpha),
                linewidths=0,
                zorder=1,
            )
            im0 = ax0.imshow(
                fail_hists[int(city)],
                origin="upper",
                extent=(xmin, xmax, ymax, ymin),
                cmap=cmap_fail,
                vmin=0.0,
                vmax=vmax_fail if vmax_fail > 0 else None,
                alpha=0.88,
                zorder=2,
                interpolation="nearest",
            )
            # Overlay failure points (small N; helps interpret heatmap)
            if d["fail10_x"].size > 0:
                ax0.scatter(d["fail10_x"], d["fail10_y"], s=28, c=OKABE_ITO["vermillion"], alpha=0.9, linewidths=0, zorder=3)
            ax0.text(
                0.02,
                0.02,
                f"n_fail={int(d['fail10_x'].size)}/{int(len(d['rids']))}",
                transform=ax0.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
                zorder=10,
            )
            ax0.set_title(f"{city_names.get(int(city), f'city{city}')} · Beam-10 failures (start density)")
            ax0.set_xlim(xmin, xmax)
            ax0.set_ylim(ymin, ymax)
            ax0.set_aspect("equal", adjustable="box")
            ax0.invert_yaxis()
            ax0.set_xticks([])
            ax0.set_yticks([])
            add_panel_label(ax0, chr(ord("a") + panel))
            panel += 1
            ims_fail.append(im0)

            # B: greedy success rate partition (by start bin)
            ax1 = axes[r, 1]
            ax1.scatter(
                way_center_x[mask_bg],
                way_center_y[mask_bg],
                s=float(args.bg_s),
                c="#DDDDDD",
                alpha=float(args.bg_alpha),
                linewidths=0,
                zorder=1,
            )
            im1 = ax1.imshow(
                rate_hists[int(city)],
                origin="upper",
                extent=(xmin, xmax, ymax, ymin),
                cmap=cmap_rate,
                vmin=0.0,
                vmax=1.0,
                alpha=0.92,
                zorder=2,
                interpolation="nearest",
            )
            sr = float(np.mean(d["succ_flags"])) if int(d["succ_flags"].size) > 0 else float("nan")
            sr_s = f"{sr:.1%}" if np.isfinite(sr) else "n/a"
            ax1.text(
                0.02,
                0.02,
                f"greedy={sr_s} (n={int(d['succ_flags'].size)})",
                transform=ax1.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
                zorder=10,
            )
            ax1.set_title(f"{city_names.get(int(city), f'city{city}')} · Greedy success rate (start bins)")
            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(ymin, ymax)
            ax1.set_aspect("equal", adjustable="box")
            ax1.invert_yaxis()
            ax1.set_xticks([])
            ax1.set_yticks([])
            add_panel_label(ax1, chr(ord("a") + panel))
            panel += 1
            ims_rate.append(im1)

        # Shared colorbars (one per column).
        if ims_fail:
            cbar0 = fig.colorbar(ims_fail[0], ax=[axes[0, 0], axes[1, 0]], fraction=0.030, pad=0.02)
            cbar0.set_label("log(1 + count)")
        if ims_rate:
            cbar1 = fig.colorbar(ims_rate[0], ax=[axes[0, 1], axes[1, 1]], fraction=0.030, pad=0.02)
            cbar1.set_label("success rate")

        out_pdf = out_dir / "waycasd_city_macro_overview.pdf"
        out_png = out_dir / "waycasd_city_macro_overview.png"
        save_figure(fig, out_pdf)
        save_figure(fig, out_png, dpi=300)
        plt.close(fig)

    print(f"[OK] saved {out_pdf}")
    print(f"[OK] saved {out_png}")


if __name__ == "__main__":
    main()
