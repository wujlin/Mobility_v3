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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz


TZ_SHANGHAI = timezone(timedelta(hours=8))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[FATAL] file not found: {path}")
    if not path.is_file():
        raise SystemExit(f"[FATAL] not a file: {path}")


def _as_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _nanquantiles(x: np.ndarray, qs: Iterable[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    out: Dict[str, float] = {}
    if x.size == 0:
        for q in qs:
            out[f"p{int(round(100 * float(q)))}"] = float("nan")
        return out
    for q in qs:
        out[f"p{int(round(100 * float(q)))}"] = float(np.quantile(x, float(q)))
    return out


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def _decode_meta(meta_obj: object) -> Optional[dict]:
    if meta_obj is None:
        return None
    if isinstance(meta_obj, np.ndarray):
        if meta_obj.size != 1:
            return None
        meta_obj = meta_obj.item()
    return meta_obj if isinstance(meta_obj, dict) else None


def _grid_bbox_from_meta(meta: dict) -> Optional[Tuple[int, int, float, float, float, float]]:
    grid = meta.get("grid", {}) if isinstance(meta, dict) else {}
    if not isinstance(grid, dict):
        return None
    H = grid.get("H", None)
    W = grid.get("W", None)
    bbox = grid.get("bbox", None)
    if not isinstance(bbox, dict):
        return None
    try:
        H_i = int(H)
        W_i = int(W)
        min_lon = float(bbox["min_lon"])
        min_lat = float(bbox["min_lat"])
        max_lon = float(bbox["max_lon"])
        max_lat = float(bbox["max_lat"])
    except Exception:
        return None
    if H_i <= 0 or W_i <= 0:
        return None
    return (H_i, W_i, min_lon, min_lat, max_lon, max_lat)


def _yx_to_latlon(y: np.ndarray, x: np.ndarray, meta: Optional[dict]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if meta is None:
        return None
    bb = _grid_bbox_from_meta(meta)
    if bb is None:
        return None
    H, W, min_lon, min_lat, max_lon, max_lat = bb
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    lon = min_lon + (x / float(W)) * (max_lon - min_lon)
    lat = max_lat - (y / float(H)) * (max_lat - min_lat)
    return lat, lon


def _haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    r = 6371000.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (np.sin(dl / 2.0) ** 2)
    return 2.0 * r * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _dist_to_dest_m(
    *,
    pred_last_way: int,
    dest_pos_yx: np.ndarray,
    way_center_y: np.ndarray,
    way_center_x: np.ndarray,
    meta: Optional[dict],
) -> Tuple[float, str]:
    """
    Final position error: distance between pred_last_way center and destination position.
    Units:
      - meters if meta (grid bbox) is available
      - otherwise grid-units (L2 in y/x)
    """
    M = int(way_center_y.size)
    w = int(pred_last_way)
    if w < 0 or w >= M:
        return float("nan"), "invalid_way_id"
    py = float(way_center_y[w])
    px = float(way_center_x[w])
    dy = float(dest_pos_yx[0])
    dx = float(dest_pos_yx[1])
    ll1 = _yx_to_latlon(np.asarray([py]), np.asarray([px]), meta)
    ll2 = _yx_to_latlon(np.asarray([dy]), np.asarray([dx]), meta)
    if ll1 is not None and ll2 is not None:
        lat1, lon1 = ll1
        lat2, lon2 = ll2
        return float(_haversine_m(lat1, lon1, lat2, lon2)[0]), "meters"
    # fallback: grid-units
    return float(math.hypot(px - dx, py - dy)), "grid_units"


def _dist_way_to_way_m(
    *,
    a_way: int,
    b_way: int,
    way_center_y: np.ndarray,
    way_center_x: np.ndarray,
    meta: Optional[dict],
) -> Tuple[float, str]:
    M = int(way_center_y.size)
    a = int(a_way)
    b = int(b_way)
    if a < 0 or a >= M or b < 0 or b >= M:
        return float("nan"), "invalid_way_id"
    ay, ax = float(way_center_y[a]), float(way_center_x[a])
    by, bx = float(way_center_y[b]), float(way_center_x[b])
    ll1 = _yx_to_latlon(np.asarray([ay]), np.asarray([ax]), meta)
    ll2 = _yx_to_latlon(np.asarray([by]), np.asarray([bx]), meta)
    if ll1 is not None and ll2 is not None:
        lat1, lon1 = ll1
        lat2, lon2 = ll2
        return float(_haversine_m(lat1, lon1, lat2, lon2)[0]), "meters"
    return float(math.hypot(ax - bx, ay - by)), "grid_units"


def _index_eval(rep: dict) -> Dict[int, Dict[str, Any]]:
    """
    Build a per-route index from oracle_decode_*_n200.json.
    For successes we only store route_id (no per-route record in json).
    For failures we keep the full failure record (contains last_k steps).
    """
    out: Dict[int, Dict[str, Any]] = {}
    for c in rep.get("per_city", []) or []:
        if not isinstance(c, dict):
            continue
        succ = [int(x) for x in (c.get("success_route_ids") or [])]
        fails = [f for f in (c.get("failures") or []) if isinstance(f, dict) and f.get("route_id") is not None]
        for rid in succ:
            out[int(rid)] = {"success": True, "fail": None}
        for f in fails:
            rid = int(f["route_id"])
            out[rid] = {"success": False, "fail": f}
    return out


def _infer_pred_last_way(routes, rid: int, info: Dict[str, Any]) -> int:
    if bool(info.get("success", False)):
        return int(routes.dest_way[int(rid)])
    f = info.get("fail", None)
    if isinstance(f, dict):
        last_k = f.get("last_k", {}) if isinstance(f.get("last_k", {}), dict) else {}
        steps = last_k.get("steps", []) if isinstance(last_k.get("steps", []), list) else []
        if steps:
            last = steps[-1] if isinstance(steps[-1], dict) else {}
            if "next" in last:
                return int(last["next"])
            if "cur" in last:
                return int(last["cur"])
        if "start_way" in f:
            return int(f["start_way"])
    return int(routes.start_way[int(rid)])


def _bin_label(v: float, bins: List[Tuple[float, float, str]]) -> str:
    x = float(v)
    for lo, hi, name in bins:
        if x >= float(lo) and x < float(hi):
            return str(name)
    return str(bins[-1][2])


def main() -> None:
    p = argparse.ArgumentParser(description="WayCASD evaluation granularity stats: way length + final position error + stratified success table.")
    p.add_argument("--eval_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("_sync/wsa/paper_figures/waycasd_v1/metrics"))
    p.add_argument("--greedy_json", type=Path, default=None)
    p.add_argument("--beam10_json", type=Path, default=None)
    p.add_argument("--way_routes_npz", type=Path, default=None)
    p.add_argument("--way_features_npz", type=Path, default=None)
    p.add_argument("--filter_avg_way_len_gt_m", type=float, default=2000.0, help="Optional strict eval: remove routes with avg_way_len_m > threshold.")
    args = p.parse_args()

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

    routes = load_way_routes_npz(way_routes_npz)
    wf = np.load(str(way_features_npz), allow_pickle=True)
    way_len_m = np.asarray(wf["way_len_m"], dtype=np.float64).reshape(-1)
    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)
    meta = _decode_meta(wf.get("meta", None))

    idx_g = _index_eval(greedy_rep)
    idx_b = _index_eval(beam10_rep)
    rids_g = set(idx_g.keys())
    rids_b = set(idx_b.keys())
    mism = {
        "only_in_greedy": sorted(list(rids_g - rids_b))[:50],
        "only_in_beam10": sorted(list(rids_b - rids_g))[:50],
        "n_only_in_greedy": int(len(rids_g - rids_b)),
        "n_only_in_beam10": int(len(rids_b - rids_g)),
    }

    rids = sorted(list(rids_g & rids_b))
    if not rids:
        raise SystemExit("[FATAL] no overlapping route_ids between greedy and beam10 reports.")

    # Per-route derived stats (based on GT way sequence).
    recs: List[Dict[str, Any]] = []
    units_seen: set[str] = set()
    for rid in rids:
        rid_i = int(rid)
        city = int(routes.route_city[rid_i])
        L = int(routes.way_seq_len[rid_i])
        s = int(routes.way_seq_ptr[rid_i])
        gt = routes.way_seq_idx[s : s + L].astype(np.int64, copy=False)
        gt = gt[gt >= 0]
        gt_hops = max(0, int(gt.size) - 1)
        gt_way_lens = way_len_m[np.clip(gt, 0, way_len_m.size - 1)] if gt.size else np.asarray([], dtype=np.float64)
        avg_way_len = float(np.mean(gt_way_lens)) if gt_way_lens.size else float("nan")

        dest_way = int(routes.dest_way[rid_i])
        dest_way_len = float(way_len_m[dest_way]) if 0 <= dest_way < int(way_len_m.size) else float("nan")
        dest_pos = routes.dest_pos[rid_i].astype(np.float64, copy=False).reshape(2)

        # Final position error (meters if possible).
        g_last = _infer_pred_last_way(routes, rid_i, idx_g[rid_i])
        b_last = _infer_pred_last_way(routes, rid_i, idx_b[rid_i])
        g_pos_err, g_unit = _dist_to_dest_m(
            pred_last_way=g_last, dest_pos_yx=dest_pos, way_center_y=way_center_y, way_center_x=way_center_x, meta=meta
        )
        b_pos_err, b_unit = _dist_to_dest_m(
            pred_last_way=b_last, dest_pos_yx=dest_pos, way_center_y=way_center_y, way_center_x=way_center_x, meta=meta
        )
        g_center_err, g_unit2 = _dist_way_to_way_m(a_way=g_last, b_way=dest_way, way_center_y=way_center_y, way_center_x=way_center_x, meta=meta)
        b_center_err, b_unit3 = _dist_way_to_way_m(a_way=b_last, b_way=dest_way, way_center_y=way_center_y, way_center_x=way_center_x, meta=meta)
        units_seen.add(str(g_unit))
        units_seen.add(str(b_unit))
        units_seen.add(str(g_unit2))
        units_seen.add(str(b_unit3))

        recs.append(
            {
                "route_id": int(rid_i),
                "city": int(city),
                "gt_hops": int(gt_hops),
                "avg_way_len_m": float(avg_way_len),
                "dest_way_len_m": float(dest_way_len),
                "greedy_success": bool(idx_g[rid_i]["success"]),
                "beam10_success": bool(idx_b[rid_i]["success"]),
                "greedy_pred_last_way": int(g_last),
                "beam10_pred_last_way": int(b_last),
                "greedy_final_pos_error_to_dest_pos": float(g_pos_err),
                "beam10_final_pos_error_to_dest_pos": float(b_pos_err),
                "greedy_final_center_error_to_dest_way": float(g_center_err),
                "beam10_final_center_error_to_dest_way": float(b_center_err),
                "final_pos_error_unit": str(g_unit) if str(g_unit) == str(b_unit) else f"{g_unit}|{b_unit}",
            }
        )

    # Global way length stats.
    way_stats = {
        "n_way": int(way_len_m.size),
        "mean_m": _nanmean(way_len_m),
        "quantiles_m": _nanquantiles(way_len_m, [0.25, 0.50, 0.75, 0.95]),
        "frac_ge_1000m": float(np.mean(way_len_m >= 1000.0)),
        "frac_ge_2000m": float(np.mean(way_len_m >= 2000.0)),
        "max_m": float(np.nanmax(way_len_m)) if np.isfinite(way_len_m).any() else float("nan"),
    }

    # Per-route avg way length stats.
    avg_way = np.asarray([_as_float(r.get("avg_way_len_m")) for r in recs], dtype=np.float64)
    dest_way_len = np.asarray([_as_float(r.get("dest_way_len_m")) for r in recs], dtype=np.float64)
    hops = np.asarray([int(r.get("gt_hops", 0)) for r in recs], dtype=np.int64)

    # Compare 1-5 hops routes vs all.
    mask_1_5 = (hops >= 1) & (hops <= 5)
    avg_stats = {
        "all": {"n": int(avg_way.size), "mean_m": _nanmean(avg_way), "quantiles_m": _nanquantiles(avg_way, [0.25, 0.50, 0.75, 0.95])},
        "hops_1_5": {
            "n": int(np.sum(mask_1_5)),
            "mean_m": _nanmean(avg_way[mask_1_5]),
            "quantiles_m": _nanquantiles(avg_way[mask_1_5], [0.25, 0.50, 0.75, 0.95]),
        },
    }
    dest_stats = {"n": int(dest_way_len.size), "mean_m": _nanmean(dest_way_len), "quantiles_m": _nanquantiles(dest_way_len, [0.25, 0.50, 0.75, 0.95])}

    # Final position error stats (success vs failure), per method.
    def _err_stats(key_success: str, key_err: str) -> dict:
        succ = np.asarray([float(r[key_err]) for r in recs if bool(r[key_success])], dtype=np.float64)
        fail = np.asarray([float(r[key_err]) for r in recs if not bool(r[key_success])], dtype=np.float64)
        return {
            "success": {"n": int(succ.size), "mean": _nanmean(succ), "quantiles": _nanquantiles(succ, [0.50, 0.75, 0.95])},
            "failure": {"n": int(fail.size), "mean": _nanmean(fail), "quantiles": _nanquantiles(fail, [0.50, 0.75, 0.95])},
        }

    final_err = {
        "units_seen": sorted(list(u for u in units_seen if u and u != "invalid_way_id")),
        "to_dest_pos": {
            "greedy": _err_stats("greedy_success", "greedy_final_pos_error_to_dest_pos"),
            "beam10": _err_stats("beam10_success", "beam10_final_pos_error_to_dest_pos"),
        },
        "to_dest_way_center": {
            "greedy": _err_stats("greedy_success", "greedy_final_center_error_to_dest_way"),
            "beam10": _err_stats("beam10_success", "beam10_final_center_error_to_dest_way"),
        },
    }

    # Stratified success table: avg way len bins x hop bins.
    way_bins = [(0.0, 500.0, "<500m"), (500.0, 1000.0, "500m-1km"), (1000.0, float("inf"), ">1km")]
    hop_bins = [(0.0, 10.0, "<10"), (10.0, 20.0, "10-20"), (20.0, 50.0, "20-50"), (50.0, float("inf"), ">50")]

    def _strat(method_key: str) -> dict:
        counts: Dict[str, Dict[str, Dict[str, float]]] = {}
        for wlo, whi, wname in way_bins:
            counts[wname] = {}
            for hlo, hhi, hname in hop_bins:
                counts[wname][hname] = {"n": 0.0, "succ": 0.0, "rate": float("nan")}

        for r in recs:
            wl = float(r.get("avg_way_len_m", float("nan")))
            hp = float(r.get("gt_hops", 0))
            wlab = _bin_label(wl, way_bins)
            hlab = _bin_label(hp, hop_bins)
            cell = counts[wlab][hlab]
            cell["n"] += 1.0
            cell["succ"] += 1.0 if bool(r[method_key]) else 0.0

        for wname in counts:
            for hname in counts[wname]:
                n = float(counts[wname][hname]["n"])
                s = float(counts[wname][hname]["succ"])
                counts[wname][hname]["rate"] = float(s / n) if n > 0 else float("nan")

        return {"way_bins": [b[2] for b in way_bins], "hop_bins": [b[2] for b in hop_bins], "cells": counts}

    strat = {"greedy": _strat("greedy_success"), "beam10": _strat("beam10_success")}

    # Optional strict eval: filter out long-avg-way routes.
    thr = float(args.filter_avg_way_len_gt_m)
    filt_mask = np.isfinite(avg_way) & (avg_way <= thr)
    n_keep = int(np.sum(filt_mask))
    n_total = int(avg_way.size)
    if n_keep <= 0:
        strict = {"threshold_avg_way_len_m": float(thr), "removed_n": int(n_total), "removed_frac": 1.0, "note": "no routes kept under threshold"}
    else:
        g_succ = np.asarray([1.0 if bool(recs[i]["greedy_success"]) else 0.0 for i in range(len(recs))], dtype=np.float64)[filt_mask]
        b_succ = np.asarray([1.0 if bool(recs[i]["beam10_success"]) else 0.0 for i in range(len(recs))], dtype=np.float64)[filt_mask]
        strict = {
            "threshold_avg_way_len_m": float(thr),
            "kept_n": int(n_keep),
            "removed_n": int(n_total - n_keep),
            "removed_frac": float((n_total - n_keep) / max(1, n_total)),
            "greedy_success_rate": float(np.mean(g_succ)),
            "beam10_success_rate": float(np.mean(b_succ)),
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "ok": True,
        "task": "waycasd_eval_granularity_stats",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "eval_dir": str(eval_dir),
        "inputs": {
            "greedy_json": str(greedy_json),
            "beam10_json": str(beam10_json),
            "way_routes_npz": str(way_routes_npz),
            "way_features_npz": str(way_features_npz),
        },
        "mismatches": mism,
        "way_len_m": way_stats,
        "dest_way_len_m_on_eval_routes": dest_stats,
        "route_avg_way_len_m_on_eval_routes": avg_stats,
        "final_pos_error": final_err,
        "stratified_success_rate": strat,
        "strict_eval_filter_avg_way_len": strict,
        "n_eval_routes": int(len(recs)),
    }

    out_json = out_dir / "waycasd_eval_granularity_stats.json"
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Per-route table (csv) for quick slicing.
    out_csv = out_dir / "waycasd_eval_granularity_routes.csv"
    cols = [
        "route_id",
        "city",
        "gt_hops",
        "avg_way_len_m",
        "dest_way_len_m",
        "greedy_success",
        "beam10_success",
        "greedy_final_pos_error_to_dest_pos",
        "beam10_final_pos_error_to_dest_pos",
        "greedy_final_center_error_to_dest_way",
        "beam10_final_center_error_to_dest_way",
        "final_pos_error_unit",
    ]
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in recs:
            row = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, bool):
                    row.append("1" if v else "0")
                else:
                    s = str(v)
                    if "," in s:
                        s = s.replace(",", " ")
                    row.append(s)
            f.write(",".join(row) + "\n")

    # Paper snippet (markdown) with key numbers filled.
    q = way_stats["quantiles_m"]
    qd = dest_stats["quantiles_m"]
    qe = final_err["to_dest_pos"]["beam10"]["success"]["quantiles"]
    unit = (final_err["units_seen"][0] if final_err.get("units_seen") else "unknown")
    snippet = f"""## Way granularity & evaluation definition (auto-generated)

- Way segment length distribution (meters): median={q.get('p50', float('nan')):.0f}m (p25={q.get('p25', float('nan')):.0f}m, p75={q.get('p75', float('nan')):.0f}m, p95={q.get('p95', float('nan')):.0f}m).
- Destination-way length on eval routes: median={qd.get('p50', float('nan')):.0f}m (p95={qd.get('p95', float('nan')):.0f}m).
- Success is defined as reaching the destination way (street-level *segment* granularity). We additionally report final position error as the distance between the predicted last-way center and the destination position (unit: {unit}).
- Beam-10 final position error (successful cases): median={qe.get('p50', float('nan')):.0f} ({unit}), p95={qe.get('p95', float('nan')):.0f} ({unit}).
"""
    out_md = out_dir / "waycasd_eval_granularity_paper_snippet.md"
    out_md.write_text(snippet, encoding="utf-8")

    print(f"[OK] saved: {out_json}")
    print(f"[OK] saved: {out_csv}")
    print(f"[OK] saved: {out_md}")
    # Quick console summary (for PI blocking checks).
    q = out["way_len_m"]["quantiles_m"]
    mx = out["way_len_m"]["max_m"]
    qd = out["dest_way_len_m_on_eval_routes"]["quantiles_m"]
    qa = out["route_avg_way_len_m_on_eval_routes"]["all"]["quantiles_m"]
    unit = (out["final_pos_error"]["units_seen"][0] if out["final_pos_error"].get("units_seen") else "unknown")
    b_succ = out["final_pos_error"]["to_dest_pos"]["beam10"]["success"]["quantiles"]
    b_fail = out["final_pos_error"]["to_dest_pos"]["beam10"]["failure"]["quantiles"]
    strict = out["strict_eval_filter_avg_way_len"]
    print(
        "[WayLen(m)] "
        f"p25={q.get('p25', float('nan')):.0f} "
        f"p50={q.get('p50', float('nan')):.0f} "
        f"p75={q.get('p75', float('nan')):.0f} "
        f"p95={q.get('p95', float('nan')):.0f} "
        f"max={mx:.0f}"
    )
    print(
        "[DestWayLen(m)] "
        f"p50={qd.get('p50', float('nan')):.0f} "
        f"p95={qd.get('p95', float('nan')):.0f}"
    )
    print(
        "[RouteAvgWayLen(m)] "
        f"p25={qa.get('p25', float('nan')):.0f} "
        f"p50={qa.get('p50', float('nan')):.0f} "
        f"p75={qa.get('p75', float('nan')):.0f} "
        f"p95={qa.get('p95', float('nan')):.0f}"
    )
    print(
        f"[FinalPosErr -> dest_pos] unit={unit} "
        f"beam10_success_p50={b_succ.get('p50', float('nan')):.0f} "
        f"p95={b_succ.get('p95', float('nan')):.0f} | "
        f"beam10_fail_p50={b_fail.get('p50', float('nan')):.0f} "
        f"p95={b_fail.get('p95', float('nan')):.0f}"
    )
    if isinstance(strict, dict) and "kept_n" in strict:
        print(
            "[StrictFilter avg_way_len] "
            f"thr={float(strict.get('threshold_avg_way_len_m', float('nan'))):.0f}m "
            f"kept={int(strict.get('kept_n', 0))} "
            f"removed={int(strict.get('removed_n', 0))} "
            f"greedy={float(strict.get('greedy_success_rate', float('nan'))):.4f} "
            f"beam10={float(strict.get('beam10_success_rate', float('nan'))):.4f}"
        )


if __name__ == "__main__":
    main()
