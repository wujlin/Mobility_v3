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


def _first_loop_segment(seq: Sequence[int]) -> Optional[Tuple[int, int]]:
    first: Dict[int, int] = {}
    for i, x in enumerate(seq):
        xx = int(x)
        if xx in first:
            return int(first[xx]), int(i)
        first[xx] = int(i)
    return None


def _add_direction_arrows(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    n_arrows: int = 2,
    size: float = 9.0,
    lw: float = 1.2,
    alpha: float = 0.9,
    zorder: int = 7,
) -> None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 3 or y.size < 3:
        return
    n = int(x.size)
    fracs = [0.35, 0.65] if int(n_arrows) >= 2 else [0.5]
    idxs: List[int] = []
    for f in fracs[: int(n_arrows)]:
        i = int(max(0, min(n - 2, round(float(f) * float(n - 2)))))
        if i not in idxs:
            idxs.append(int(i))
    for i in idxs:
        ax.annotate(
            "",
            xy=(float(x[i + 1]), float(y[i + 1])),
            xytext=(float(x[i]), float(y[i])),
            arrowprops=dict(
                arrowstyle="-|>",
                color=str(color),
                lw=float(lw),
                alpha=float(alpha),
                mutation_scale=float(size),
                shrinkA=0.0,
                shrinkB=0.0,
            ),
            zorder=int(zorder),
        )


@dataclass(frozen=True)
class RouteRecord:
    route_id: int
    city: int
    gt_hops: int
    gt_way_ids: List[int]
    start_way: int
    dest_way: int
    pred_way_ids: List[int]
    region_seq: Optional[List[int]]
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
            region_seq=([int(x) for x in (r.get("region_seq") or [])] if isinstance(r.get("region_seq"), list) else None),
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
    p.add_argument("--way_regions_npz", type=Path, default=None, help="Optional: way->region mapping for corridor visualization (expects key 'way_region').")
    p.add_argument("--city_name", type=str, nargs="*", default=["0:Detroit", "1:Columbus"])
    p.add_argument("--pad_frac", type=float, default=0.2)
    p.add_argument("--bg_alpha", type=float, default=0.35)
    p.add_argument("--bg_s", type=float, default=0.8)
    p.add_argument("--corridor_alpha", type=float, default=0.20, help="Alpha for corridor region highlight.")
    p.add_argument("--corridor_draw", action="store_true", help="If set, overlay region corridor (region_seq) when available.")
    p.add_argument("--no_arrows", action="store_true", help="Disable direction arrows on trajectories.")
    p.add_argument("--arrow_n", type=int, default=2, help="Number of direction arrows per trajectory.")
    p.add_argument("--arrow_size", type=float, default=9.0, help="Arrow size (mutation_scale).")
    p.add_argument("--no_loop_overlay", action="store_true", help="Disable overlay for detected loop segments.")

    # Case selection
    p.add_argument("--easy_min_hops", type=int, default=5)
    p.add_argument("--easy_max_hops", type=int, default=20)
    p.add_argument("--improved_min_hops", type=int, default=20)
    p.add_argument("--hard_min_hops", type=int, default=40)
    p.add_argument("--baseline", type=str, default=None, help="Baseline method name (default: first --method).")
    p.add_argument("--n_easy", type=int, default=1, help="How many easy cases per city.")
    p.add_argument("--n_improved", type=int, default=1, help="How many improved cases per city.")
    p.add_argument("--n_hard", type=int, default=1, help="How many hard cases per city.")
    return p


def _pick_top_unique(cands: List[int], *, k: int, used: set[int]) -> List[int]:
    out: List[int] = []
    for rid in cands:
        rid_i = int(rid)
        if rid_i in used:
            continue
        out.append(rid_i)
        used.add(rid_i)
        if len(out) >= int(k):
            break
    return out


def _compute_region_centroids(*, way_region: np.ndarray, way_center_x: np.ndarray, way_center_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    way_region = np.asarray(way_region, dtype=np.int64).reshape(-1)
    x = np.asarray(way_center_x, dtype=np.float64).reshape(-1)
    y = np.asarray(way_center_y, dtype=np.float64).reshape(-1)
    if way_region.size != x.size or way_region.size != y.size:
        raise ValueError("way_region and way_center_x/y must have same length")
    mask = way_region >= 0
    if not np.any(mask):
        raise ValueError("way_region has no valid assignments (all <0)")
    reg = way_region[mask]
    n_regions = int(np.max(reg)) + 1
    cnt = np.bincount(reg, minlength=n_regions).astype(np.float64, copy=False)
    sum_x = np.bincount(reg, weights=x[mask], minlength=n_regions).astype(np.float64, copy=False)
    sum_y = np.bincount(reg, weights=y[mask], minlength=n_regions).astype(np.float64, copy=False)
    denom = np.maximum(1.0, cnt)
    return (sum_x / denom).astype(np.float64, copy=False), (sum_y / denom).astype(np.float64, copy=False)


def _plot_one_case(
    *,
    ax,
    city: int,
    rid: int,
    city_title: str,
    cat_title: str,
    methods: List[Tuple[str, Path, dict]],
    rec_by_method: List[Dict[int, Dict[int, RouteRecord]]],
    base_i: int,
    way_center_x: np.ndarray,
    way_center_y: np.ndarray,
    way_region: Optional[np.ndarray],
    reg_cx: Optional[np.ndarray],
    reg_cy: Optional[np.ndarray],
    corridor_draw: bool,
    corridor_alpha: float,
    no_arrows: bool,
    arrow_n: int,
    arrow_size: float,
    no_loop_overlay: bool,
    pad_frac: float,
    bg_alpha: float,
    bg_s: float,
    colors: List[str],
    linestyles: List[Any],
) -> Tuple[List[Any], List[str]]:
    rr0 = rec_by_method[base_i][int(city)][int(rid)]
    gt = rr0.gt_way_ids
    gt_x, gt_y = _seq_to_xy(gt, way_center_x=way_center_x, way_center_y=way_center_y)

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

    xmin, xmax, ymin, ymax = _compute_bbox(xs, ys, pad_frac=float(pad_frac))

    # Background: road graph as way-centers.
    mask = (way_center_x >= xmin) & (way_center_x <= xmax) & (way_center_y >= ymin) & (way_center_y <= ymax)
    ax.scatter(
        way_center_x[mask],
        way_center_y[mask],
        s=float(bg_s),
        c="#DDDDDD",
        alpha=float(bg_alpha),
        linewidths=0,
        zorder=1,
    )

    # Corridor overlay (optional): highlight regions in region_seq.
    if bool(corridor_draw) and way_region is not None and reg_cx is not None and reg_cy is not None:
        seq = rr0.region_seq
        if isinstance(seq, list) and seq:
            seq_i = [int(x) for x in seq if int(x) >= 0 and int(x) < int(reg_cx.size)]
            if seq_i:
                # Highlight corridor ways.
                corr_mask = mask & np.isin(way_region, np.asarray(seq_i, dtype=np.int64))
                ax.scatter(
                    way_center_x[corr_mask],
                    way_center_y[corr_mask],
                    s=float(bg_s) * 1.4,
                    c=OKABE_ITO["sky_blue"],
                    alpha=float(corridor_alpha),
                    linewidths=0,
                    zorder=2,
                )
                # Region centroid polyline.
                cx = reg_cx[np.asarray(seq_i, dtype=np.int64)]
                cy = reg_cy[np.asarray(seq_i, dtype=np.int64)]
                ax.plot(cx, cy, color=OKABE_ITO["gray"], lw=3.0, alpha=0.35, ls="-", zorder=3)
                ax.scatter(cx, cy, s=14, c=OKABE_ITO["gray"], alpha=0.6, zorder=3)

    # Trajectories
    ln_gt = ax.plot(gt_x, gt_y, color=OKABE_ITO["blue"], lw=2.4, ls="-", zorder=4)[0]
    if not bool(no_arrows):
        _add_direction_arrows(ax, gt_x, gt_y, color=OKABE_ITO["blue"], n_arrows=int(arrow_n), size=float(arrow_size), zorder=6)

    any_loop = False
    for mi, (px, py) in enumerate(pred_xy):
        col = colors[int(mi) % len(colors)]
        ls = linestyles[int(mi) % len(linestyles)]
        ax.plot(px, py, color=col, lw=2.0, ls=ls, zorder=5)
        if not bool(no_arrows):
            _add_direction_arrows(ax, px, py, color=str(col), n_arrows=int(arrow_n), size=float(arrow_size), zorder=6)
        if not bool(no_loop_overlay):
            rr = rec_by_method[int(mi)][int(city)][int(rid)]
            if bool(rr.has_loop):
                seg = _first_loop_segment(rr.pred_way_ids)
                if seg is not None:
                    i0, i1 = seg
                    if 0 <= i0 < i1 < int(px.size):
                        ax.plot(
                            px[i0 : i1 + 1],
                            py[i0 : i1 + 1],
                            color=OKABE_ITO["black"],
                            lw=2.2,
                            ls="--",
                            alpha=0.85,
                            zorder=7,
                        )
                        any_loop = True

    mk_o = ax.scatter([sx], [sy], s=80, c="#000000", marker="o", edgecolors="white", linewidths=0.8, zorder=6)
    mk_d = ax.scatter([dx], [dy], s=90, c="#000000", marker="*", edgecolors="white", linewidths=0.8, zorder=6)

    ax.set_title(f"{city_title} · {cat_title} (rid={int(rid)})")

    lines = [f"GT hops: {int(rr0.gt_hops)}"]
    for mi, (name, _, _) in enumerate(methods):
        rr = rec_by_method[mi][int(city)][int(rid)]
        tag = "✓" if rr.success else ("HW" if rr.hit_wall else ("DE" if rr.dead_end else ("Loop" if rr.has_loop else "Fail")))
        lines.append(f"{name}: {tag} | Fr={_fmt_km(rr.frechet_m)} | Len={rr.len_ratio:.2f} | Err={_fmt_km(rr.final_error_m)}")
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

    # Legend handles (only need once per figure)
    handles: List[Any] = []
    labels: List[str] = []
    handles.append(Line2D([0], [0], color=OKABE_ITO["blue"], lw=2.4, ls="-"))
    labels.append("GT")
    for mi, (name, _, _) in enumerate(methods):
        c = colors[int(mi) % len(colors)]
        ls = linestyles[int(mi) % len(linestyles)]
        handles.append(Line2D([0], [0], color=c, lw=2.0, ls=ls))
        labels.append(str(name))
    if any_loop:
        handles.append(Line2D([0], [0], color=OKABE_ITO["black"], lw=2.2, ls="--"))
        labels.append("Loop segment")
    if bool(corridor_draw) and way_region is not None:
        handles.append(Line2D([0], [0], color=OKABE_ITO["gray"], lw=3.0, ls="-", alpha=0.35))
        labels.append("Corridor (region_seq)")
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#000000", markersize=7))
    labels.append("O")
    handles.append(Line2D([0], [0], marker="*", color="none", markerfacecolor="#000000", markersize=9))
    labels.append("D")
    handles.append(Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#DDDDDD", markeredgecolor="none", markersize=6, alpha=0.8))
    labels.append("Road graph")
    if bool(corridor_draw) and way_region is not None:
        handles.append(Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=OKABE_ITO["sky_blue"], markeredgecolor="none", markersize=6, alpha=float(corridor_alpha)))
        labels.append("Corridor regions")

    return handles, labels


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

    way_region: Optional[np.ndarray] = None
    reg_cx: Optional[np.ndarray] = None
    reg_cy: Optional[np.ndarray] = None
    if bool(args.corridor_draw):
        if args.way_regions_npz is None:
            # best-effort infer from first method json
            p0 = methods[0][2].get("inputs", {}).get("way_regions_npz", None)
            if p0 is not None:
                args.way_regions_npz = Path(str(p0))
        if args.way_regions_npz is not None:
            _require_file(Path(args.way_regions_npz))
            wr = np.load(str(Path(args.way_regions_npz)), allow_pickle=True)
            if "way_region" not in wr.files:
                raise SystemExit("[FATAL] way_regions_npz missing key: way_region")
            way_region = np.asarray(wr["way_region"], dtype=np.int64).reshape(-1)
            reg_cx, reg_cy = _compute_region_centroids(way_region=way_region, way_center_x=way_center_x, way_center_y=way_center_y)

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

    # Pick cases per city (possibly multiple each category).
    picked: Dict[Tuple[int, str], List[int]] = {}
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

        used: set[int] = set()

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
        picked[(int(city), "easy")] = _pick_top_unique(easy, k=int(args.n_easy), used=used)

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
        improved.sort(key=lambda rid: (-int(base_rec(rid).gt_hops), float(base_rec(rid).frechet_m)))
        picked[(int(city), "improved")] = _pick_top_unique(improved, k=int(args.n_improved), used=used)

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
        picked[(int(city), "hard")] = _pick_top_unique(hard, k=int(args.n_hard), used=used)
        # If any category is empty (rare), fallback to easy picks.
        if not picked[(int(city), "improved")]:
            picked[(int(city), "improved")] = picked[(int(city), "easy")][:1]
        if not picked[(int(city), "hard")]:
            picked[(int(city), "hard")] = picked[(int(city), "easy")][:1]

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

    # 1) Overview: 2 cities × 3 categories using the first case in each list.
    cat_order = ["easy", "improved", "hard"]
    picked_first = {(city, cat): int((picked[(city, cat)] or [])[0]) for city in (0, 1) for cat in cat_order}

    with paper_style() if args.style == "paper" else plt.rc_context({}):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13.6, 7.6), constrained_layout=True)
        panel = 0
        legend_handles: Optional[List[Any]] = None
        legend_labels: Optional[List[str]] = None
        for r, city in enumerate([0, 1]):
            for cci, cat in enumerate(cat_order):
                ax = axes[r, cci]
                rid = picked_first[(int(city), str(cat))]
                city_title = city_names.get(int(city), f"city{int(city)}")
                cat_title = {"easy": "Easy", "improved": "Improved", "hard": "Hard"}[str(cat)]
                h, lab = _plot_one_case(
                    ax=ax,
                    city=int(city),
                    rid=int(rid),
                    city_title=str(city_title),
                    cat_title=str(cat_title),
                    methods=methods,
                    rec_by_method=rec_by_method,
                    base_i=int(base_i),
                    way_center_x=way_center_x,
                    way_center_y=way_center_y,
                    way_region=way_region,
                    reg_cx=reg_cx,
                    reg_cy=reg_cy,
                    corridor_draw=bool(args.corridor_draw),
                    corridor_alpha=float(args.corridor_alpha),
                    no_arrows=bool(args.no_arrows),
                    arrow_n=int(args.arrow_n),
                    arrow_size=float(args.arrow_size),
                    no_loop_overlay=bool(args.no_loop_overlay),
                    pad_frac=float(args.pad_frac),
                    bg_alpha=float(args.bg_alpha),
                    bg_s=float(args.bg_s),
                    colors=colors,
                    linestyles=linestyles,
                )
                if legend_handles is None:
                    legend_handles, legend_labels = h, lab
                add_panel_label(ax, chr(ord("a") + int(panel)))
                panel += 1

        if legend_handles is not None and legend_labels is not None:
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

    # 2) Individual cases (more than 3): save one png/pdf per case for easy browsing.
    with paper_style() if args.style == "paper" else plt.rc_context({}):
        for city in (0, 1):
            city_title = city_names.get(int(city), f"city{int(city)}")
            for cat in cat_order:
                cat_title = {"easy": "Easy", "improved": "Improved", "hard": "Hard"}[str(cat)]
                rids = picked[(int(city), str(cat))]
                for ii, rid in enumerate(rids):
                    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
                    h, lab = _plot_one_case(
                        ax=ax,
                        city=int(city),
                        rid=int(rid),
                        city_title=str(city_title),
                        cat_title=f"{cat_title} #{int(ii)}",
                        methods=methods,
                        rec_by_method=rec_by_method,
                        base_i=int(base_i),
                        way_center_x=way_center_x,
                        way_center_y=way_center_y,
                        way_region=way_region,
                        reg_cx=reg_cx,
                        reg_cy=reg_cy,
                        corridor_draw=bool(args.corridor_draw),
                        corridor_alpha=float(args.corridor_alpha),
                        no_arrows=bool(args.no_arrows),
                        arrow_n=int(args.arrow_n),
                        arrow_size=float(args.arrow_size),
                        no_loop_overlay=bool(args.no_loop_overlay),
                        pad_frac=float(args.pad_frac),
                        bg_alpha=float(args.bg_alpha),
                        bg_s=float(args.bg_s),
                        colors=colors,
                        linestyles=linestyles,
                    )
                    fig.legend(
                        handles=h, labels=lab, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=min(len(lab), 4), frameon=False
                    )
                    out_png = out_dir / f"flow_case_city{int(city)}_{str(cat)}_{int(ii)}_rid{int(rid)}_{decode}.png"
                    out_pdf = out_dir / f"flow_case_city{int(city)}_{str(cat)}_{int(ii)}_rid{int(rid)}_{decode}.pdf"
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
        "picked": {f"{city}:{cat}": [int(x) for x in rids] for (city, cat), rids in picked.items()},
        "way_features_npz": str(way_features_npz),
        "way_regions_npz": (str(args.way_regions_npz) if args.way_regions_npz is not None else None),
        "corridor_draw": bool(args.corridor_draw),
    }
    (out_dir / "flow_micro_compare_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] saved {out_png}")
    print(f"[OK] saved {out_pdf}")


if __name__ == "__main__":
    main()
