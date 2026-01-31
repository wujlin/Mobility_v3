#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[FATAL] file not found: {path}")
    if not path.is_file():
        raise SystemExit(f"[FATAL] not a file: {path}")


def _summary_stats(values: List[Optional[float]]) -> dict:
    xs = [float(x) for x in values if x is not None]
    if not xs:
        return {"n": 0}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _slice_csr(ptr: np.ndarray, idx: np.ndarray, u: int) -> np.ndarray:
    s = int(ptr[u])
    e = int(ptr[u + 1])
    if e <= s:
        return np.asarray([], dtype=np.int64)
    return np.asarray(idx[s:e], dtype=np.int64)


def _shortest_hops_forward_bfs(
    *,
    ptr: np.ndarray,
    idx: np.ndarray,
    start: int,
    dest: int,
    max_visits: int,
) -> Optional[int]:
    n = int(ptr.size) - 1
    s = int(start)
    d = int(dest)
    if s < 0 or s >= n or d < 0 or d >= n:
        return None
    if s == d:
        return 0

    visited = set([s])
    frontier = [s]
    depth = 0
    visits = 0

    while frontier:
        depth += 1
        nxt: List[int] = []
        for u in frontier:
            for v in _slice_csr(ptr, idx, int(u)).tolist():
                vv = int(v)
                if vv == d:
                    return int(depth)
                if vv not in visited:
                    visited.add(vv)
                    nxt.append(vv)
            visits += 1
            if visits >= int(max_visits):
                return None
        frontier = nxt
    return None


@dataclass(frozen=True)
class HardCaseRow:
    city: int
    route_id: int

    hour: Optional[int]
    dow: Optional[int]

    gt_len: Optional[int]
    pred_len: Optional[int]
    gt_hops: Optional[int]
    shortest_hops: Optional[int]
    gt_over_shortest_hops: Optional[float]

    pred_has_loop: Optional[bool]
    pred_first_repeat_step: Optional[int]
    gt_has_loop: Optional[bool]
    gt_first_repeat_step: Optional[int]

    jaccard: Optional[float]
    prefix_match_len: Optional[int]
    diverge_step: Optional[int]

    start_way: Optional[int]
    dest_way: Optional[int]

    start_y: Optional[float]
    start_x: Optional[float]
    dest_y: Optional[float]
    dest_x: Optional[float]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PI verify: analyze remaining hard cases (beam hit_wall) from oracle_decode json.")
    p.add_argument("--beam_json", type=Path, required=True, help="oracle_decode_beam*_n*.json")
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--way_graph_npz", type=Path, default=None, help="Optional: compute shortest hop distance.")
    p.add_argument("--way_features_npz", type=Path, default=None, help="Optional: dump start/dest coordinates.")
    p.add_argument("--max_bfs_visits", type=int, default=200000, help="Safety cap for BFS visits per case.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    beam_json = Path(args.beam_json)
    _require_file(beam_json)

    out_dir = Path(args.out_dir) if args.out_dir is not None else (beam_json.parent / "hard_cases_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = _read_json(beam_json)
    per_city = rep.get("per_city", []) or []

    ptr = None
    idx = None
    if args.way_graph_npz is not None:
        wg = np.load(str(Path(args.way_graph_npz)), allow_pickle=True)
        ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64)
        idx = np.asarray(wg["way_adj_idx"], dtype=np.int64)

    way_center_y = None
    way_center_x = None
    if args.way_features_npz is not None:
        wf = np.load(str(Path(args.way_features_npz)), allow_pickle=True)
        way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
        way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)

    rows: List[HardCaseRow] = []
    route_ids_by_city: Dict[str, List[int]] = {}

    for c in per_city:
        if not isinstance(c, dict):
            continue
        city = int(c.get("city", -1))
        for f in c.get("failures", []) or []:
            if not isinstance(f, dict):
                continue
            if not bool(f.get("hit_wall", False)):
                continue
            if bool(f.get("dead_end", False)):
                continue
            if bool(f.get("success", False)):
                continue

            rid = int(f.get("route_id"))
            route_ids_by_city.setdefault(str(city), []).append(int(rid))

            start_way = f.get("start_way", None)
            dest_way = f.get("dest_way", None)
            if start_way is not None:
                start_way = int(start_way)
            if dest_way is not None:
                dest_way = int(dest_way)

            gt_len = f.get("gt_len", None)
            pred_len = f.get("pred_len", None)
            gt_len_i = int(gt_len) if gt_len is not None else None
            pred_len_i = int(pred_len) if pred_len is not None else None
            gt_hops = (gt_len_i - 1) if (gt_len_i is not None and gt_len_i > 0) else None

            shortest_hops = None
            if ptr is not None and idx is not None and start_way is not None and dest_way is not None:
                shortest_hops = _shortest_hops_forward_bfs(
                    ptr=ptr,
                    idx=idx,
                    start=int(start_way),
                    dest=int(dest_way),
                    max_visits=int(args.max_bfs_visits),
                )

            ratio = None
            if gt_hops is not None and shortest_hops is not None and shortest_hops > 0:
                ratio = float(gt_hops) / float(shortest_hops)

            def _safe_coord(arr: Optional[np.ndarray], i: Optional[int]) -> Optional[float]:
                if arr is None or i is None:
                    return None
                if i < 0 or i >= int(arr.size):
                    return None
                return float(arr[int(i)])

            rows.append(
                HardCaseRow(
                    city=int(city),
                    route_id=int(rid),
                    hour=(int(f["hour"]) if f.get("hour") is not None else None),
                    dow=(int(f["dow"]) if f.get("dow") is not None else None),
                    gt_len=gt_len_i,
                    pred_len=pred_len_i,
                    gt_hops=gt_hops,
                    shortest_hops=shortest_hops,
                    gt_over_shortest_hops=ratio,
                    pred_has_loop=(bool(f["pred_has_loop"]) if f.get("pred_has_loop") is not None else None),
                    pred_first_repeat_step=(int(f["pred_first_repeat_step"]) if f.get("pred_first_repeat_step") is not None else None),
                    gt_has_loop=(bool(f["gt_has_loop"]) if f.get("gt_has_loop") is not None else None),
                    gt_first_repeat_step=(int(f["gt_first_repeat_step"]) if f.get("gt_first_repeat_step") is not None else None),
                    jaccard=(float(f["jaccard"]) if f.get("jaccard") is not None else None),
                    prefix_match_len=(int(f["prefix_match_len"]) if f.get("prefix_match_len") is not None else None),
                    diverge_step=(int(f["diverge_step"]) if f.get("diverge_step") is not None else None),
                    start_way=start_way,
                    dest_way=dest_way,
                    start_y=_safe_coord(way_center_y, start_way),
                    start_x=_safe_coord(way_center_x, start_way),
                    dest_y=_safe_coord(way_center_y, dest_way),
                    dest_x=_safe_coord(way_center_x, dest_way),
                )
            )

    rows.sort(key=lambda r: (r.city, r.route_id))
    route_ids_by_city = {k: sorted(v) for k, v in route_ids_by_city.items()}

    # Summaries
    hours = [r.hour for r in rows if r.hour is not None]
    dows = [r.dow for r in rows if r.dow is not None]
    gt_lens = [float(r.gt_len) if r.gt_len is not None else None for r in rows]
    gt_hops_list = [float(r.gt_hops) if r.gt_hops is not None else None for r in rows]
    pred_lens = [float(r.pred_len) if r.pred_len is not None else None for r in rows]
    shortest_hops_list = [float(r.shortest_hops) if r.shortest_hops is not None else None for r in rows]
    ratios = [r.gt_over_shortest_hops for r in rows]

    out_summary = {
        "ok": True,
        "task": "pi_analyze_beam_hard_cases",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "beam_json": str(beam_json),
            "way_graph_npz": (str(args.way_graph_npz) if args.way_graph_npz is not None else None),
            "way_features_npz": (str(args.way_features_npz) if args.way_features_npz is not None else None),
        },
        "n_hard_cases": int(len(rows)),
        "route_ids_by_city": route_ids_by_city,
        "gt_len_stats": _summary_stats(gt_lens),
        "gt_hops_stats": _summary_stats(gt_hops_list),
        "pred_len_stats": _summary_stats(pred_lens),
        "shortest_hops_stats": _summary_stats(shortest_hops_list),
        "gt_over_shortest_hops_stats": _summary_stats(ratios),
        "hour_counts": dict(sorted(Counter(hours).items(), key=lambda kv: kv[0])),
        "dow_counts": dict(sorted(Counter(dows).items(), key=lambda kv: kv[0])),
        "by_city_counts": dict(sorted(Counter([r.city for r in rows]).items(), key=lambda kv: kv[0])),
    }

    out_json = out_dir / "hard_cases_summary.json"
    out_json.write_text(json.dumps(out_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_ids = out_dir / "hard_cases_route_ids.json"
    out_ids.write_text(json.dumps(route_ids_by_city, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_csv = out_dir / "hard_cases.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "city",
                "route_id",
                "hour",
                "dow",
                "gt_len",
                "pred_len",
                "gt_hops",
                "shortest_hops",
                "gt_over_shortest_hops",
                "pred_has_loop",
                "pred_first_repeat_step",
                "gt_has_loop",
                "gt_first_repeat_step",
                "jaccard",
                "prefix_match_len",
                "diverge_step",
                "start_way",
                "dest_way",
                "start_y",
                "start_x",
                "dest_y",
                "dest_x",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "city": r.city,
                    "route_id": r.route_id,
                    "hour": r.hour,
                    "dow": r.dow,
                    "gt_len": r.gt_len,
                    "pred_len": r.pred_len,
                    "gt_hops": r.gt_hops,
                    "shortest_hops": r.shortest_hops,
                    "gt_over_shortest_hops": r.gt_over_shortest_hops,
                    "pred_has_loop": r.pred_has_loop,
                    "pred_first_repeat_step": r.pred_first_repeat_step,
                    "gt_has_loop": r.gt_has_loop,
                    "gt_first_repeat_step": r.gt_first_repeat_step,
                    "jaccard": r.jaccard,
                    "prefix_match_len": r.prefix_match_len,
                    "diverge_step": r.diverge_step,
                    "start_way": r.start_way,
                    "dest_way": r.dest_way,
                    "start_y": r.start_y,
                    "start_x": r.start_x,
                    "dest_y": r.dest_y,
                    "dest_x": r.dest_x,
                }
            )

    print(f"[saved] {out_json}")
    print(f"[saved] {out_csv}")
    print(json.dumps(out_summary.get('by_city_counts', {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

