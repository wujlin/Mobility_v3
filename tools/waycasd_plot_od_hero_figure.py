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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex, to_rgb
from matplotlib.transforms import ScaledTranslation

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


def _pretty_method_label(label: str) -> str:
    s = str(label).lower()
    if "cascadetraj" in s or "betavae" in s or "flowmu" in s:
        return "CascadeTraj"
    if "way-casd" in s or "waycasd" in s:
        return "Way-CASD"
    if "oracle" in s:
        return "Oracle"
    if "rnn" in s:
        return "RNN"
    if "transformer" in s or "tr-ar" in s:
        return "Transformer"
    return str(label)


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


def _parse_city_kv(spec: str) -> Tuple[int, Path]:
    s = str(spec or "").strip()
    if "=" in s:
        k, v = s.split("=", 1)
    elif ":" in s:
        k, v = s.split(":", 1)
    else:
        raise ValueError(f"Bad spec (expect CITY=PATH): {spec!r}")
    return int(str(k).strip()), Path(str(v).strip()).expanduser()


def _decode_meta(meta_obj: object) -> Dict[str, Any] | None:
    if meta_obj is None:
        return None
    if isinstance(meta_obj, np.ndarray):
        if meta_obj.size != 1:
            return None
        meta_obj = meta_obj.item()
    return meta_obj if isinstance(meta_obj, dict) else None


def _grid_bbox_from_meta(meta: Dict[str, Any]) -> Tuple[int, int, float, float, float, float] | None:
    grid = meta.get("grid", {}) if isinstance(meta, dict) else {}
    if not isinstance(grid, dict):
        return None
    bbox = grid.get("bbox", {})
    if not isinstance(bbox, dict):
        return None
    try:
        H = int(grid["H"])
        W = int(grid["W"])
        min_lon = float(bbox["min_lon"])
        min_lat = float(bbox["min_lat"])
        max_lon = float(bbox["max_lon"])
        max_lat = float(bbox["max_lat"])
    except Exception:
        return None
    if H <= 0 or W <= 0:
        return None
    return (H, W, min_lon, min_lat, max_lon, max_lat)


def _meta_from_city_grid_meta(path: Path) -> Dict[str, Any]:
    if str(path).endswith(".npz"):
        wf = np.load(str(path), allow_pickle=True)
        meta = _decode_meta(wf.get("meta", None))
        if meta is None:
            raise ValueError(f"{path} missing meta (need grid.H/W/bbox).")
    else:
        meta = _read_json(path)
    if _grid_bbox_from_meta(meta) is None:
        if isinstance(meta, dict) and ("H" in meta) and ("W" in meta) and ("bbox" in meta):
            meta = {"grid": {"H": meta["H"], "W": meta["W"], "bbox": meta["bbox"]}}
    if _grid_bbox_from_meta(meta) is None:
        raise ValueError(f"{path} missing grid meta (need grid.H/W/bbox).")
    return meta


def _grid_xy_to_lonlat(y: np.ndarray, x: np.ndarray, *, meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    bb = _grid_bbox_from_meta(meta)
    if bb is None:
        raise ValueError("city meta missing grid bbox")
    H, W, min_lon, min_lat, max_lon, max_lat = bb
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    lon = min_lon + (x / float(W)) * (max_lon - min_lon)
    lat = max_lat - (y / float(H)) * (max_lat - min_lat)
    return lon, lat


def _aspect_for_latlon(meta: Dict[str, Any]) -> float:
    bb = _grid_bbox_from_meta(meta)
    if bb is None:
        return 1.0
    _, _, _, min_lat, _, max_lat = bb
    mean_lat = 0.5 * (min_lat + max_lat)
    c = max(math.cos(math.radians(float(mean_lat))), 1e-6)
    return 1.0 / c


def _resolve_ctx_provider(ctx: Any, provider_name: str) -> Any:
    cur: Any = ctx.providers
    for p in str(provider_name).split("."):
        if hasattr(cur, p):
            cur = getattr(cur, p)
        elif isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            raise ValueError(f"Unknown provider path: {provider_name}")
    return cur


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
    if "cascadetraj" in s or "betavae" in s or "flowmu" in s:
        return 0
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


def _jaccard_set(a: Sequence[int], b: Sequence[int]) -> float:
    sa = {int(x) for x in a}
    sb = {int(x) for x in b}
    union = sa | sb
    if not union:
        return 0.0
    return float(len(sa & sb)) / float(len(union))


def _corridor_palette(n: int) -> List[str]:
    base = [
        OKABE_ITO["vermillion"],
        OKABE_ITO["blue"],
        OKABE_ITO["bluish_green"],
        OKABE_ITO["orange"],
        OKABE_ITO["reddish_purple"],
        OKABE_ITO["sky_blue"],
    ]
    if n <= len(base):
        return base[:n]
    out: List[str] = []
    for i in range(int(n)):
        out.append(base[i % len(base)])
    return out


def _mix_with_white(color: str, frac: float) -> str:
    rgb = np.asarray(to_rgb(color), dtype=np.float64)
    out = (1.0 - float(frac)) * rgb + float(frac) * np.ones(3, dtype=np.float64)
    return to_hex(np.clip(out, 0.0, 1.0))


def _cluster_sequences_by_jaccard(seqs: Sequence[Sequence[int]], thr: float = 0.3) -> List[List[int]]:
    uniq = _dedup_keep_order([[int(x) for x in s] for s in seqs], limit=10**9)
    n = len(uniq)
    vis = [False] * n
    comps: List[List[int]] = []
    for i in range(n):
        if vis[i]:
            continue
        stack = [i]
        vis[i] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if vis[v]:
                    continue
                if _jaccard_set(uniq[u], uniq[v]) >= float(thr):
                    vis[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def _representative_display_offsets(n: int, mag_pt: float = 2.4) -> List[Tuple[float, float]]:
    if n <= 1:
        return [(0.0, 0.0)]
    if n == 2:
        return [(-mag_pt, 0.78 * mag_pt), (mag_pt, -0.78 * mag_pt)]
    if n == 3:
        return [(-mag_pt, 0.78 * mag_pt), (0.0, 0.0), (mag_pt, -0.78 * mag_pt)]
    if n == 4:
        return [
            (-mag_pt, 0.78 * mag_pt),
            (mag_pt, -0.78 * mag_pt),
            (-0.52 * mag_pt, -0.90 * mag_pt),
            (0.52 * mag_pt, 0.90 * mag_pt),
        ]
    base = [
        (-mag_pt, 0.78 * mag_pt),
        (mag_pt, -0.78 * mag_pt),
        (-0.52 * mag_pt, -0.90 * mag_pt),
        (0.52 * mag_pt, 0.90 * mag_pt),
    ]
    out: List[Tuple[float, float]] = []
    for i in range(int(n)):
        out.append(base[i % len(base)])
    return out


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Figure A hero: same OD, GT vs multiple methods at way-level.")
    ap.add_argument("--phasec_json", type=Path, required=True, help="Phase C json with per_od (must run with --save_per_od).")
    ap.add_argument("--hero_label", type=str, required=True, help="Method label in phaseC used for OD selection.")
    ap.add_argument("--method", action="append", required=True, help="Repeatable: LABEL|DECODE=PER_ROUTE_JSON.")
    ap.add_argument("--way_features_npz", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument(
        "--city_grid_meta",
        action="append",
        default=[],
        help="Optional CITY=PATH mapping to osm_road_prob_meta.json (needed for --coord_mode latlon).",
    )
    ap.add_argument("--coord_mode", choices=["grid", "latlon"], default="grid")
    ap.add_argument("--use_basemap", action="store_true", help="Draw web basemap (only in latlon mode).")
    ap.add_argument("--basemap_provider", type=str, default="CartoDB.Positron")
    ap.add_argument("--basemap_zoom", type=int, default=-1, help="<=0 means auto zoom.")
    ap.add_argument("--city", type=int, default=0)
    ap.add_argument("--min_gt_routes", type=int, default=10)
    ap.add_argument("--min_pred_success", type=int, default=4)
    ap.add_argument("--min_self_diversity", type=float, default=0.6)
    ap.add_argument("--hops_min", type=int, default=20)
    ap.add_argument("--hops_max", type=int, default=40)
    ap.add_argument("--k_pred_per_method", type=int, default=10, help="Max successful predicted routes shown per method panel.")
    ap.add_argument("--max_gt_draw", type=int, default=80)
    ap.add_argument("--pad_frac", type=float, default=0.18)
    ap.add_argument("--pad_frac_x", type=float, default=None, help="Optional x-direction padding override.")
    ap.add_argument("--pad_frac_y", type=float, default=None, help="Optional y-direction padding override.")
    ap.add_argument("--bg_alpha", type=float, default=0.16)
    ap.add_argument("--bg_s", type=float, default=0.6)
    ap.add_argument(
        "--ours_panel_mode",
        choices=["match_gt", "single_color", "cluster_representatives"],
        default="match_gt",
        help="How to render the CascadeTraj panel.",
    )
    ap.add_argument(
        "--ours_color",
        type=str,
        default=OKABE_ITO["vermillion"],
        help="Main color used when --ours_panel_mode=single_color.",
    )
    ap.add_argument("--corridor_cluster_thr", type=float, default=0.3)
    ap.add_argument("--od_start_way", type=int, default=None, help="Manual override start_way.")
    ap.add_argument("--od_dest_way", type=int, default=None, help="Manual override dest_way.")
    ap.add_argument("--keep_method_order", action="store_true", help="Keep --method order; default auto: Way-CASD, RNN, Transformer.")
    ap.add_argument(
        "--figure_title",
        type=str,
        default="",
        help="Figure-level title. Use empty string to suppress.",
    )
    ap.add_argument(
        "--figure_mode",
        choices=["gt_vs_ours", "overlay_only"],
        default="gt_vs_ours",
        help="gt_vs_ours renders the original 2-panel figure; overlay_only renders only the CascadeTraj overlay panel.",
    )
    ap.add_argument("--fig_w", type=float, default=None, help="Optional figure width override.")
    ap.add_argument("--fig_h", type=float, default=None, help="Optional figure height override.")
    ap.add_argument("--no_panel_labels", action="store_true", help="Disable a/b/c/d panel labels.")
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

    city_meta: Dict[int, Dict[str, Any]] = {}
    city_meta_src: Dict[int, str] = {}
    for spec in list(args.city_grid_meta or []):
        try:
            c, p = _parse_city_kv(spec)
            _require_file(p)
            city_meta[int(c)] = _meta_from_city_grid_meta(p)
            city_meta_src[int(c)] = str(p)
        except Exception as e:
            raise SystemExit(f"[FATAL] bad --city_grid_meta {spec!r}: {e}") from e

    coord_mode = str(args.coord_mode)
    if coord_mode == "latlon":
        if int(args.city) not in city_meta:
            raise SystemExit(
                f"[FATAL] coord_mode=latlon requires --city_grid_meta for city={int(args.city)}."
            )
        lon_all, lat_all = _grid_xy_to_lonlat(way_center_y, way_center_x, meta=city_meta[int(args.city)])
        plot_x_all = lon_all
        plot_y_all = lat_all
        latlon_aspect = _aspect_for_latlon(city_meta[int(args.city)])
    else:
        plot_x_all = way_center_x
        plot_y_all = way_center_y
        latlon_aspect = 1.0

    use_basemap = bool(args.use_basemap)
    if use_basemap and coord_mode != "latlon":
        raise SystemExit("[FATAL] --use_basemap requires --coord_mode latlon.")

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
        x, y = _seq_to_xy(s, plot_x_all, plot_y_all)
        xs.append(x)
        ys.append(y)
    for ms in method_pred.values():
        for s in ms:
            x, y = _seq_to_xy(s, plot_x_all, plot_y_all)
            xs.append(x)
            ys.append(y)
    base_pad = float(args.pad_frac)
    pad_x = base_pad if args.pad_frac_x is None else float(args.pad_frac_x)
    pad_y = base_pad if args.pad_frac_y is None else float(args.pad_frac_y)
    xmin, xmax, ymin, ymax = _bbox(xs, ys, base_pad)
    xx = np.concatenate([a.reshape(-1) for a in xs if a.size > 0], axis=0)
    yy = np.concatenate([a.reshape(-1) for a in ys if a.size > 0], axis=0)
    xmin0, xmax0 = float(np.min(xx)), float(np.max(xx))
    ymin0, ymax0 = float(np.min(yy)), float(np.max(yy))
    dx0 = max(1e-6, xmax0 - xmin0)
    dy0 = max(1e-6, ymax0 - ymin0)
    xmin = xmin0 - pad_x * dx0
    xmax = xmax0 + pad_x * dx0
    ymin = ymin0 - pad_y * dy0
    ymax = ymax0 + pad_y * dy0

    # Start/dest marker from GT first route.
    sxy = _seq_to_xy(gt_rows[0].gt_way_ids, plot_x_all, plot_y_all)
    sx, sy = float(sxy[0][0]), float(sxy[1][0])
    dx, dy = float(sxy[0][-1]), float(sxy[1][-1])

    corridor_colors = _corridor_palette(len(gt_seqs))
    ours_method = methods[0]
    ours_pred = method_pred[ours_method.label]
    ours_matched: List[Tuple[List[int], int, float]] = []
    for seq in ours_pred:
        if not gt_seqs:
            ours_matched.append((seq, 0, 0.0))
            continue
        scores = np.asarray([_jaccard_set(seq, gt) for gt in gt_seqs], dtype=np.float64)
        best_idx = int(np.argmax(scores))
        ours_matched.append((seq, best_idx, float(scores[best_idx])))
    gt_clusters = _cluster_sequences_by_jaccard(gt_seqs, thr=float(args.corridor_cluster_thr))

    with paper_style():
        fig_mode = str(args.figure_mode)
        default_w, default_h = ((12.8, 4.4) if fig_mode == "gt_vs_ours" else (8.6, 5.2))
        fig_w = float(args.fig_w) if args.fig_w is not None else default_w
        fig_h = float(args.fig_h) if args.fig_h is not None else default_h
        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
        if fig_mode == "overlay_only":
            gs = fig.add_gridspec(1, 1)
            ax_ours = fig.add_subplot(gs[0, 0])
            ax_gt = None
            axs = [ax_ours]
        else:
            gs = fig.add_gridspec(1, 2)
            ax_gt = fig.add_subplot(gs[0, 0])
            ax_ours = fig.add_subplot(gs[0, 1])
            axs = [ax_gt, ax_ours]
        basemap_active = bool(use_basemap)
        basemap_warned = False

        # Common background.
        mask = (
            (plot_x_all >= xmin) & (plot_x_all <= xmax) &
            (plot_y_all >= ymin) & (plot_y_all <= ymax)
        )
        for ax in axs:
            if not basemap_active:
                ax.scatter(
                    plot_x_all[mask], plot_y_all[mask],
                    s=float(args.bg_s), c="#DADADA", alpha=float(args.bg_alpha), linewidths=0, zorder=1
                )
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            if coord_mode == "latlon":
                ax.set_aspect(latlon_aspect, adjustable="box")
            else:
                ax.set_aspect("equal", adjustable="box")
                ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            if basemap_active:
                try:
                    import contextily as ctx  # type: ignore[import-not-found]

                    src = _resolve_ctx_provider(ctx, str(args.basemap_provider))
                    if int(args.basemap_zoom) > 0:
                        ctx.add_basemap(
                            ax,
                            source=src,
                            crs="EPSG:4326",
                            zoom=int(args.basemap_zoom),
                            attribution=False,
                            alpha=0.95,
                        )
                    else:
                        # Let contextily pick zoom automatically.
                        ctx.add_basemap(
                            ax,
                            source=src,
                            crs="EPSG:4326",
                            attribution=False,
                            alpha=0.95,
                        )
                except Exception as e:
                    if not basemap_warned:
                        print(f"[WARN] basemap render failed, fallback to local road background: {e}")
                        basemap_warned = True
                    basemap_active = False
                    ax.scatter(
                        plot_x_all[mask], plot_y_all[mask],
                        s=float(args.bg_s), c="#DADADA", alpha=float(args.bg_alpha), linewidths=0, zorder=1
                    )

        if fig_mode != "overlay_only":
            # (a) ground-truth corridors
            ax = ax_gt
            for idx, s in enumerate(gt_seqs):
                x, y = _seq_to_xy(s, plot_x_all, plot_y_all)
                color = corridor_colors[idx]
                ax.plot(x, y, color="white", lw=3.8, alpha=0.90, zorder=2)
                ax.plot(x, y, color=color, lw=2.2, alpha=0.92, zorder=3)
            ax.scatter([sx], [sy], s=80, c="#000000", marker="o", edgecolors="white", linewidths=0.8, zorder=4)
            ax.scatter([dx], [dy], s=90, c="#000000", marker="*", edgecolors="white", linewidths=0.8, zorder=4)
            ax.set_title(f"Ground-truth corridors ({len(gt_seqs)} routes)")
            if not bool(args.no_panel_labels):
                add_panel_label(ax, "a")

        # (b) ours
        ax = ax_ours
        if str(args.ours_panel_mode) == "single_color":
            for idx, s in enumerate(gt_seqs):
                x, y = _seq_to_xy(s, plot_x_all, plot_y_all)
                ref_color = _mix_with_white(corridor_colors[idx], 0.40)
                ax.plot(x, y, color=ref_color, lw=1.6, alpha=0.42, zorder=2)
            for seq, _gt_idx, _score in ours_matched:
                x, y = _seq_to_xy(seq, plot_x_all, plot_y_all)
                ax.plot(x, y, color="white", lw=4.2, alpha=0.92, zorder=3)
                ax.plot(x, y, color=str(args.ours_color), lw=2.5, alpha=0.96, zorder=4)
        elif str(args.ours_panel_mode) == "cluster_representatives":
            for idx, s in enumerate(gt_seqs):
                x, y = _seq_to_xy(s, plot_x_all, plot_y_all)
                ref_color = _mix_with_white(corridor_colors[idx], 0.32)
                ax.plot(x, y, color="white", lw=2.8, alpha=0.58, zorder=2)
                ax.plot(x, y, color=ref_color, lw=1.85, alpha=0.58, zorder=2.1)
            # First show the full successful route support as a translucent band:
            # overlap then reads as evidence concentration rather than clutter.
            for seq, _gt_idx, _score in ours_matched:
                x, y = _seq_to_xy(seq, plot_x_all, plot_y_all)
                ax.plot(x, y, color="white", lw=3.8, alpha=0.12, zorder=2.6)
                ax.plot(x, y, color=str(args.ours_color), lw=2.0, alpha=0.10, zorder=2.8)
            best_by_cluster: Dict[int, Tuple[List[int], float]] = {}
            for seq, gt_idx, score in ours_matched:
                cluster_idx = 0
                for ci, comp in enumerate(gt_clusters):
                    if int(gt_idx) in comp:
                        cluster_idx = ci
                        break
                cur = best_by_cluster.get(cluster_idx)
                if cur is None or float(score) > float(cur[1]):
                    best_by_cluster[cluster_idx] = (seq, float(score))
            rep_cluster_ids = sorted(best_by_cluster.keys())
            rep_offsets = _representative_display_offsets(len(rep_cluster_ids), mag_pt=2.6)
            rep_colors: List[str] = [
                str(args.ours_color),
                _mix_with_white(str(args.ours_color), 0.22),
                _mix_with_white(str(args.ours_color), 0.38),
            ]
            rep_dash = (0, (7.0, 2.4))
            for rep_ord, cluster_idx in enumerate(rep_cluster_ids):
                seq, _score = best_by_cluster[cluster_idx]
                x, y = _seq_to_xy(seq, plot_x_all, plot_y_all)
                dx_pt, dy_pt = rep_offsets[rep_ord]
                rep_trans = ax.transData + ScaledTranslation(
                    float(dx_pt) / 72.0,
                    float(dy_pt) / 72.0,
                    fig.dpi_scale_trans,
                )
                ax.plot(
                    x, y,
                    color="white",
                    lw=4.6,
                    alpha=0.96,
                    zorder=3,
                    transform=rep_trans,
                )
                ax.plot(
                    x, y,
                    color=rep_colors[rep_ord % len(rep_colors)],
                    lw=3.0,
                    alpha=0.98,
                    linestyle=rep_dash,
                    zorder=4,
                    transform=rep_trans,
                )
        else:
            for s in gt_seqs:
                x, y = _seq_to_xy(s, plot_x_all, plot_y_all)
                ax.plot(x, y, color=OKABE_ITO["gray"], lw=1.0, alpha=0.18, zorder=2)
            for seq, gt_idx, _score in ours_matched:
                x, y = _seq_to_xy(seq, plot_x_all, plot_y_all)
                color = corridor_colors[gt_idx]
                ax.plot(x, y, color="white", lw=4.0, alpha=0.90, zorder=3)
                ax.plot(x, y, color=color, lw=2.4, alpha=0.96, zorder=4)
        ax.scatter([sx], [sy], s=80, c="#000000", marker="o", edgecolors="white", linewidths=0.8, zorder=5)
        ax.scatter([dx], [dy], s=90, c="#000000", marker="*", edgecolors="white", linewidths=0.8, zorder=5)
        if fig_mode != "overlay_only":
            if str(args.ours_panel_mode) == "cluster_representatives":
                ax.set_title("CascadeTraj recovered corridors")
            else:
                ax.set_title("CascadeTraj route set")
            if not bool(args.no_panel_labels):
                add_panel_label(ax, "b")

        fig_title = str(args.figure_title).strip()
        if fig_title:
            fig.suptitle(fig_title, y=0.995)

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
        "pretty_method_labels": {m.label: _pretty_method_label(m.label) for m in methods},
        "method_meta": method_meta,
        "inputs": {
            "phasec_json": str(args.phasec_json),
            "way_features_npz": str(args.way_features_npz),
            "coord_mode": coord_mode,
            "use_basemap": bool(use_basemap),
            "basemap_active_final": bool(basemap_active),
            "city_grid_meta": {str(k): str(v) for k, v in sorted(city_meta_src.items(), key=lambda kv: kv[0])},
            "figure_mode": fig_mode,
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
