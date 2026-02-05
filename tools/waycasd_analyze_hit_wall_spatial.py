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
import math
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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[FATAL] file not found: {path}")
    if not path.is_file():
        raise SystemExit(f"[FATAL] not a file: {path}")


def _parse_bin(spec: str) -> Tuple[int, Optional[int]]:
    s = str(spec).strip()
    if not (s.startswith("[") and s.endswith(")")):
        raise ValueError(f"bad bin spec: {spec!r} (expect like [40,60))")
    core = s[1:-1]
    a, b = core.split(",", 1)
    lo = int(a.strip())
    b = b.strip()
    if b.startswith("+"):
        return lo, None
    hi = int(b)
    return lo, hi


def _in_bin(hops: int, lo: int, hi: Optional[int]) -> bool:
    h = int(hops)
    if h < int(lo):
        return False
    if hi is None:
        return True
    return h < int(hi)


def _compress_consecutive_int(seq: List[int]) -> List[int]:
    out: List[int] = []
    last: Optional[int] = None
    for x in seq:
        xx = int(x)
        if last is None or xx != int(last):
            out.append(xx)
            last = xx
    return out


def _first_diverge_step(gt: List[int], pred: List[int]) -> Optional[int]:
    n = min(len(gt), len(pred))
    for i in range(n):
        if int(gt[i]) != int(pred[i]):
            return int(i)
    if len(gt) != len(pred):
        return int(n)
    return None


def _quantiles_int(values: np.ndarray, qs: Tuple[int, ...] = (0, 50, 90, 95, 99, 100)) -> Dict[str, int | None]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"p{q:02d}": None for q in qs}
    out: Dict[str, int | None] = {}
    for q in qs:
        out[f"p{q:02d}"] = int(np.percentile(v, float(q)))
    return out


def _hist2d(xs: np.ndarray, ys: np.ndarray, *, bins: int, xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float64).reshape(-1)
    ys = np.asarray(ys, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size == 0:
        return np.zeros((bins, bins), dtype=np.float32)
    h, _, _ = np.histogram2d(ys, xs, bins=bins, range=[[ymin, ymax], [xmin, xmax]])
    return h.astype(np.float32, copy=False)


@dataclass(frozen=True)
class Cfg:
    key: str
    hops_bin: str
    topk_ways: int
    bins: int
    bg_alpha: float
    bg_s: float


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze hit-wall spatial pattern for a given hops bin (expects per-route json with dumped way seqs).")
    p.add_argument("--per_route_json", type=Path, required=True, help="per-route json from way_casd_binned_eval.py (--out_per_route_json).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--key", type=str, default="beam", choices=["greedy", "beam"])
    p.add_argument("--hops_bin", type=str, default="[40,60)")
    p.add_argument("--topk_ways", type=int, default=20)
    p.add_argument("--bins", type=int, default=60)
    p.add_argument("--bg_alpha", type=float, default=0.25)
    p.add_argument("--bg_s", type=float, default=0.6)
    args = p.parse_args()

    cfg = Cfg(
        key=str(args.key),
        hops_bin=str(args.hops_bin),
        topk_ways=int(args.topk_ways),
        bins=int(args.bins),
        bg_alpha=float(args.bg_alpha),
        bg_s=float(args.bg_s),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _require_file(Path(args.per_route_json))
    _require_file(Path(args.way_routes_npz))
    _require_file(Path(args.way_graph_npz))
    _require_file(Path(args.way_features_npz))

    recs = _read_json(Path(args.per_route_json))
    if not isinstance(recs, list):
        raise SystemExit("[FATAL] per_route_json must be a JSON list (records).")

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    outdeg = (np.asarray(wg["way_adj_ptr"], dtype=np.int64)[1:] - np.asarray(wg["way_adj_ptr"], dtype=np.int64)[:-1]).astype(np.int64, copy=False)
    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)

    lo, hi = _parse_bin(str(cfg.hops_bin))
    key = str(cfg.key)

    # Per-city collections.
    per_city: List[Dict[str, Any]] = []
    figs: List[Path] = []

    paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.6), dpi=160)
    panels = {(0, 0): "a", (0, 1): "b", (1, 0): "c", (1, 1): "d"}

    for city in (0, 1):
        city_recs = [r for r in recs if isinstance(r, dict) and int(r.get("city", -1)) == int(city)]
        # filter by hops bin
        city_recs = [r for r in city_recs if _in_bin(int(r.get("gt_hops", 0)), lo, hi)]
        n_total = int(len(city_recs))

        # Require seq dumps for last_way / divergence diagnostics.
        have_pred = 0
        last_way_ids: List[int] = []
        last_xy: List[Tuple[float, float]] = []
        start_xy: List[Tuple[float, float]] = []
        last_outdeg: List[int] = []
        div_outdeg: List[int] = []
        div_xy: List[Tuple[float, float]] = []

        n_hit_wall = 0
        for r in city_recs:
            dec = r.get(key, {}) if isinstance(r.get(key, {}), dict) else {}
            hit_wall = bool(dec.get("hit_wall", False))
            if not hit_wall:
                continue
            n_hit_wall += 1
            rid = int(r.get("route_id", -1))
            if rid < 0 or rid >= int(routes.route_city.size):
                continue
            sp = routes.start_pos[rid].astype(np.float64, copy=False).reshape(2)
            start_xy.append((float(sp[1]), float(sp[0])))

            pred = dec.get("pred_way_ids", None)
            gt = r.get("gt_way_ids", None)
            if not isinstance(pred, list) or not pred:
                continue
            have_pred += 1
            pred_ids = [int(x) for x in pred]
            last = int(pred_ids[-1])
            last_way_ids.append(last)
            if 0 <= last < int(way_center_x.size):
                last_xy.append((float(way_center_x[last]), float(way_center_y[last])))
            if 0 <= last < int(outdeg.size):
                last_outdeg.append(int(outdeg[last]))

            if isinstance(gt, list) and gt:
                gt_ids = [int(x) for x in gt]
                div = _first_diverge_step(gt_ids, pred_ids)
                if div is not None and 0 <= int(div) < int(len(pred_ids)):
                    w = int(pred_ids[int(div)])
                    if 0 <= w < int(outdeg.size):
                        div_outdeg.append(int(outdeg[w]))
                    if 0 <= w < int(way_center_x.size):
                        div_xy.append((float(way_center_x[w]), float(way_center_y[w])))

        hit_wall_rate = float(n_hit_wall) / float(max(1, n_total)) if n_total > 0 else float("nan")

        # Top last-way ids
        top_list: List[Dict[str, Any]] = []
        if last_way_ids:
            vals, cnts = np.unique(np.asarray(last_way_ids, dtype=np.int64), return_counts=True)
            order = np.argsort(cnts)[::-1]
            for w, c in zip(vals[order][: int(cfg.topk_ways)].tolist(), cnts[order][: int(cfg.topk_ways)].tolist()):
                w = int(w)
                recw = {"way": w, "count": int(c)}
                if 0 <= w < int(outdeg.size):
                    recw["outdeg"] = int(outdeg[w])
                if 0 <= w < int(way_center_x.size):
                    recw["center_x"] = float(way_center_x[w])
                    recw["center_y"] = float(way_center_y[w])
                top_list.append(recw)

        rep_city: Dict[str, Any] = {
            "city": int(city),
            "n_total_in_bin": int(n_total),
            "n_hit_wall": int(n_hit_wall),
            "hit_wall_rate": float(hit_wall_rate),
            "n_with_pred_way_ids": int(have_pred),
            "last_outdeg": {
                "n": int(len(last_outdeg)),
                "quantiles": _quantiles_int(np.asarray(last_outdeg, dtype=np.int64)),
                "frac_gt2": float(np.mean((np.asarray(last_outdeg, dtype=np.int64) > 2).astype(np.float64))) if last_outdeg else None,
                "frac_gt4": float(np.mean((np.asarray(last_outdeg, dtype=np.int64) > 4).astype(np.float64))) if last_outdeg else None,
            },
            "div_outdeg": {
                "n": int(len(div_outdeg)),
                "quantiles": _quantiles_int(np.asarray(div_outdeg, dtype=np.int64)),
            }
            if div_outdeg
            else None,
            "top_last_ways": top_list,
        }
        per_city.append(rep_city)

        # Plot: start density and last-way density (grid coords), over background ways (from GT ways).
        ax_start = axes[int(city), 0]
        ax_last = axes[int(city), 1]

        # Background: GT ways from sampled routes in this bin (requires gt_way_ids).
        bg_ways: List[int] = []
        for r in city_recs:
            gt = r.get("gt_way_ids", None)
            if isinstance(gt, list) and gt:
                bg_ways.extend(int(x) for x in gt)
        bg_ways = _compress_consecutive_int(bg_ways)
        if bg_ways:
            uniq = np.unique(np.asarray(bg_ways, dtype=np.int64))
            uniq = uniq[(uniq >= 0) & (uniq < int(way_center_x.size))]
            ax_start.scatter(
                way_center_x[uniq],
                way_center_y[uniq],
                s=float(cfg.bg_s),
                c="#DDDDDD",
                alpha=float(cfg.bg_alpha),
                linewidths=0.0,
                zorder=0,
            )
            ax_last.scatter(
                way_center_x[uniq],
                way_center_y[uniq],
                s=float(cfg.bg_s),
                c="#DDDDDD",
                alpha=float(cfg.bg_alpha),
                linewidths=0.0,
                zorder=0,
            )

        # Use bbox from background ways if available, else default [0,1024].
        if bg_ways:
            xs = way_center_x[uniq]
            ys = way_center_y[uniq]
            xmin, xmax = float(np.min(xs)), float(np.max(xs))
            ymin, ymax = float(np.min(ys)), float(np.max(ys))
        else:
            xmin, xmax, ymin, ymax = (0.0, 1024.0, 0.0, 1024.0)

        # Start density
        if start_xy:
            xs = np.asarray([x for x, _y in start_xy], dtype=np.float64)
            ys = np.asarray([y for _x, y in start_xy], dtype=np.float64)
            h = _hist2d(xs, ys, bins=int(cfg.bins), xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            img = np.log1p(h)
            img[h <= 0] = np.nan
            ax_start.imshow(img, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="magma", alpha=0.85, zorder=1)

        # Last density
        if last_xy:
            xs = np.asarray([x for x, _y in last_xy], dtype=np.float64)
            ys = np.asarray([y for _x, y in last_xy], dtype=np.float64)
            h = _hist2d(xs, ys, bins=int(cfg.bins), xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            img = np.log1p(h)
            img[h <= 0] = np.nan
            ax_last.imshow(img, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="magma", alpha=0.85, zorder=1)

        city_name = "Detroit" if int(city) == 0 else "Columbus"
        ax_start.set_title(f"{city_name} start density (HW, {cfg.hops_bin})\\nN={n_hit_wall}/{n_total} ({hit_wall_rate*100:.1f}%)")
        ax_last.set_title(f"{city_name} last-way density (HW, {cfg.hops_bin})\\nN={have_pred} w/ pred_way_ids")
        for ax in (ax_start, ax_last):
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
        add_panel_label(ax_start, panels[(int(city), 0)])
        add_panel_label(ax_last, panels[(int(city), 1)])

    fig.tight_layout()
    fig_path = out_dir / "hit_wall_spatial.png"
    save_figure(fig, fig_path)
    plt.close(fig)
    figs.append(fig_path)

    out = {
        "ok": True,
        "task": "waycasd_hit_wall_spatial_audit",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "per_route_json": str(args.per_route_json),
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
        },
        "per_city": per_city,
        "figures": [str(p) for p in figs],
        "notes": [
            "This audit expects per_route_json generated with --dump_way_seqs so that pred_way_ids and gt_way_ids are available.",
            "Intersection-type proxy uses out-degree of way graph at last/diverge way.",
        ],
    }
    (out_dir / "hit_wall_spatial_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_dir / "hit_wall_spatial_audit.json"))


if __name__ == "__main__":
    main()

