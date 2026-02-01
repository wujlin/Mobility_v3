from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.evaluation.shape_metrics import dtw_distance, frechet_distance, summarize

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_city_kv(spec: str) -> Tuple[int, Path]:
    s = str(spec or "").strip()
    if "=" in s:
        k, v = s.split("=", 1)
    elif ":" in s:
        k, v = s.split(":", 1)
    else:
        raise ValueError(f"Bad spec (expect CITY=PATH): {spec!r}")
    city = int(str(k).strip())
    path = Path(str(v).strip()).expanduser()
    return city, path


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


def _meta_from_city_grid_meta(path: Path) -> dict:
    if str(path).endswith(".npz"):
        wf = np.load(str(path), allow_pickle=True)
        meta = _decode_meta(wf.get("meta", None))
        if meta is None:
            raise ValueError(f"{path} missing meta (need meta.grid.H/W/bbox).")
    else:
        meta = _read_json(path)

    if _grid_bbox_from_meta(meta) is None:
        if isinstance(meta, dict) and ("H" in meta) and ("W" in meta) and ("bbox" in meta):
            meta = {"grid": {"H": meta["H"], "W": meta["W"], "bbox": meta["bbox"]}}
    if _grid_bbox_from_meta(meta) is None:
        raise ValueError(f"{path} missing grid meta (need grid.H/grid.W/grid.bbox).")
    return meta


def _grid_yx_to_xy_m(y: np.ndarray, x: np.ndarray, *, meta: dict) -> np.ndarray:
    bb = _grid_bbox_from_meta(meta)
    if bb is None:
        raise ValueError("meta missing grid bbox")
    H, W, min_lon, min_lat, max_lon, max_lat = bb
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    lon = min_lon + (x / float(W)) * (max_lon - min_lon)
    lat = max_lat - (y / float(H)) * (max_lat - min_lat)

    lat0 = 0.5 * (min_lat + max_lat)
    lon0 = 0.5 * (min_lon + max_lon)
    r = 6371000.0
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    lat0_r = math.radians(float(lat0))
    lon0_r = math.radians(float(lon0))
    x_m = (lon_r - lon0_r) * math.cos(lat0_r) * r
    y_m = (lat_r - lat0_r) * r
    return np.stack([x_m, y_m], axis=1).astype(np.float64, copy=False)


def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb)) / float(len(sa | sb))


def _sum_way_len_m(way_len_m: np.ndarray, seq: Sequence[int]) -> float:
    if not seq:
        return float("nan")
    ids = np.asarray([int(x) for x in seq], dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < int(way_len_m.size))]
    if ids.size == 0:
        return float("nan")
    return float(np.sum(way_len_m[ids].astype(np.float64, copy=False)))


def _slice_csr(ptr: np.ndarray, idx: np.ndarray, u: int) -> np.ndarray:
    s = int(ptr[u])
    e = int(ptr[u + 1])
    if e <= s:
        return np.asarray([], dtype=np.int64)
    return np.asarray(idx[s:e], dtype=np.int64)


def dijkstra_way_path(
    *,
    ptr: np.ndarray,
    idx: np.ndarray,
    way_len_m: np.ndarray,
    start: int,
    dest: int,
    max_visits: int = 500000,
) -> List[int]:
    """
    Length-weighted shortest path on the directed way graph.

    Cost definition (node-weighted):
      path_cost = sum(way_len_m[way] for way in path)
    Implemented by edge cost u->v = way_len_m[v], plus init cost way_len_m[start].
    """
    n = int(ptr.size) - 1
    s = int(start)
    d = int(dest)
    if s < 0 or s >= n or d < 0 or d >= n:
        return []
    if s == d:
        return [s]

    dist = np.full((n,), np.inf, dtype=np.float64)
    parent = np.full((n,), -1, dtype=np.int64)
    init = float(way_len_m[s]) if 0 <= s < int(way_len_m.size) and math.isfinite(float(way_len_m[s])) else 0.0
    dist[s] = init
    heap: List[Tuple[float, int]] = [(float(dist[s]), s)]
    seen = 0

    while heap:
        du, u = heapq.heappop(heap)
        if du != float(dist[u]):
            continue
        if u == d:
            break
        for v in _slice_csr(ptr, idx, u).tolist():
            vv = int(v)
            if vv < 0 or vv >= n:
                continue
            w = float(way_len_m[vv]) if 0 <= vv < int(way_len_m.size) and math.isfinite(float(way_len_m[vv])) else 0.0
            nd = du + w
            if nd < float(dist[vv]):
                dist[vv] = nd
                parent[vv] = u
                heapq.heappush(heap, (nd, vv))
        seen += 1
        if seen >= int(max_visits):
            break

    if not math.isfinite(float(dist[d])):
        return []

    # Reconstruct
    path: List[int] = []
    cur = d
    while cur != -1:
        path.append(int(cur))
        if cur == s:
            break
        cur = int(parent[cur])
    if not path or int(path[-1]) != s:
        return []
    path.reverse()
    return path


@dataclass(frozen=True)
class Cfg:
    seed: int
    n_routes: int
    min_hops: int
    max_way_len: int
    max_decode_len: int
    dijkstra_max_visits: int
    dump_way_seqs: bool


def _require_city_meta(city_meta: Dict[int, dict], cities: Iterable[int]) -> None:
    missing = [int(c) for c in cities if int(c) not in city_meta]
    if missing:
        raise SystemExit(f"[FATAL] missing --city_grid_meta for cities={missing} (PI: meters is mandatory).")


def main() -> None:
    p = argparse.ArgumentParser(description="Shortest-path baseline on way graph (length-weighted Dijkstra, meters).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_routes", type=int, default=200, help="Per city.")
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--max_decode_len", type=int, default=160)
    p.add_argument("--dijkstra_max_visits", type=int, default=500000)
    p.add_argument("--dump_way_seqs", action="store_true", help="Include gt_way_ids and sp_way_ids in per-route records (JSON larger).")

    p.add_argument(
        "--city_grid_meta",
        type=str,
        action="append",
        default=[],
        help="Per-city grid meta for meters conversion, format CITY=PATH (osm_road_prob_meta.json or single-city way_features.npz).",
    )
    args = p.parse_args()

    cfg = Cfg(
        seed=int(args.seed),
        n_routes=int(args.n_routes),
        min_hops=int(args.min_hops),
        max_way_len=int(args.max_way_len),
        max_decode_len=int(args.max_decode_len),
        dijkstra_max_visits=int(args.dijkstra_max_visits),
        dump_way_seqs=bool(args.dump_way_seqs),
    )

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64)
    idx = np.asarray(wg["way_adj_idx"], dtype=np.int64)
    way_len_m = np.asarray(wf["way_len_m"], dtype=np.float64).reshape(-1)
    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)

    # Per-city meta.
    city_meta: Dict[int, dict] = {}
    city_meta_src: Dict[int, str] = {}
    for spec in list(args.city_grid_meta or []):
        c, path = _parse_city_kv(str(spec))
        if not path.exists():
            raise SystemExit(f"[FATAL] file not found: {path}")
        city_meta[int(c)] = _meta_from_city_grid_meta(path)
        city_meta_src[int(c)] = str(path)
    cities_obs = sorted(set(int(x) for x in routes.route_city.astype(np.int64).tolist()))
    _require_city_meta(city_meta, cities_obs)

    # Precompute way center coords (meters) per city meta.
    way_xy_m: Dict[int, np.ndarray] = {}
    for c in cities_obs:
        way_xy_m[int(c)] = _grid_yx_to_xy_m(way_center_y, way_center_x, meta=city_meta[int(c)])

    # Sample routes per city.
    picks: Dict[int, np.ndarray] = {}
    for city in cities_obs:
        keep = (
            (routes.route_city.astype(np.int64) == int(city))
            & (routes.way_seq_len >= (int(cfg.min_hops) + 1))
            & (routes.way_seq_len <= int(cfg.max_way_len))
        )
        ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
        rng = np.random.default_rng(int(cfg.seed) + 101 * int(city))
        rng.shuffle(ids)
        picks[int(city)] = ids[: min(int(cfg.n_routes), int(ids.size))]

    per_route: List[Dict[str, Any]] = []
    for city in cities_obs:
        pick = picks[int(city)]
        if pick.size == 0:
            continue

        xy_way = way_xy_m[int(city)]
        for rid in pick.tolist():
            rid_i = int(rid)
            L = int(routes.way_seq_len[rid_i])
            s = int(routes.way_seq_ptr[rid_i])
            gt = routes.way_seq_idx[s : s + L].astype(np.int64, copy=False).tolist()
            gt_ids = [int(x) for x in gt]

            sw = int(routes.start_way[rid_i])
            dw = int(routes.dest_way[rid_i])
            dpos_yx = routes.dest_pos[rid_i].astype(np.float64, copy=False).reshape(2)
            dpos_xy_m = _grid_yx_to_xy_m(np.asarray([dpos_yx[0]]), np.asarray([dpos_yx[1]]), meta=city_meta[int(city)])[0]

            sp_full = dijkstra_way_path(
                ptr=ptr,
                idx=idx,
                way_len_m=way_len_m,
                start=sw,
                dest=dw,
                max_visits=int(cfg.dijkstra_max_visits),
            )
            sp = list(sp_full)
            if sp and len(sp) > int(cfg.max_decode_len) + 1:
                sp = sp[: int(cfg.max_decode_len) + 1]

            success = bool(sp and int(sp[-1]) == int(dw))
            hit_wall = bool((not success) and sp and (len(sp) >= int(cfg.max_decode_len) + 1))
            outdeg_last = int(ptr[int(sp[-1]) + 1] - ptr[int(sp[-1])]) if sp and 0 <= int(sp[-1]) + 1 < int(ptr.size) else 0
            dead_end = bool((not success) and (not hit_wall) and (outdeg_last == 0))

            gt_len_m = _sum_way_len_m(way_len_m, gt_ids)
            sp_len_m = _sum_way_len_m(way_len_m, sp)

            gt_xy = xy_way[np.asarray(gt_ids, dtype=np.int64)]
            sp_xy = xy_way[np.asarray(sp, dtype=np.int64)] if sp else np.zeros((0, 2), dtype=np.float64)
            dtw_m = dtw_distance(sp_xy, gt_xy)
            fre_m = frechet_distance(sp_xy, gt_xy)

            err_m = float(np.linalg.norm(sp_xy[-1] - dpos_xy_m)) if sp else float("nan")

            rec: Dict[str, Any] = {
                "route_id": int(rid_i),
                "city": int(city),
                "gt_hops": int(max(0, len(gt_ids) - 1)),
                "sp_hops": int(max(0, len(sp) - 1)) if sp else 0,
                "success": bool(success),
                "hit_wall": bool(hit_wall),
                "dead_end": bool(dead_end),
                "jaccard": float(_jaccard(gt_ids, sp)) if sp else float("nan"),
                "gt_len_m": float(gt_len_m),
                "sp_len_m": float(sp_len_m),
                "len_ratio_sp_over_gt": float(sp_len_m / gt_len_m) if (math.isfinite(gt_len_m) and gt_len_m > 0 and math.isfinite(sp_len_m)) else float("nan"),
                "detour_gt_over_sp": float(gt_len_m / sp_len_m) if (math.isfinite(sp_len_m) and sp_len_m > 0 and math.isfinite(gt_len_m)) else float("nan"),
                "dtw_m": float(dtw_m),
                "frechet_m": float(fre_m),
                "final_error_m": float(err_m),
            }
            if bool(cfg.dump_way_seqs):
                rec["gt_way_ids"] = gt_ids
                rec["sp_way_ids"] = [int(x) for x in sp]
            per_route.append(rec)

        print(f"[city{int(city)}] done={int(pick.size)} routes")

    # Summaries.
    def _agg(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        succ = [1.0 if bool(r.get("success", False)) else 0.0 for r in recs]
        return {
            "n": int(len(recs)),
            "success_rate": float(np.mean(np.asarray(succ, dtype=np.float64))) if succ else float("nan"),
            "dtw_m": summarize([float(r.get("dtw_m", float("nan"))) for r in recs]),
            "frechet_m": summarize([float(r.get("frechet_m", float("nan"))) for r in recs]),
            "len_ratio_sp_over_gt": summarize([float(r.get("len_ratio_sp_over_gt", float("nan"))) for r in recs]),
            "detour_gt_over_sp": summarize([float(r.get("detour_gt_over_sp", float("nan"))) for r in recs]),
            "final_error_m": summarize([float(r.get("final_error_m", float("nan"))) for r in recs]),
        }

    per_city = []
    for city in cities_obs:
        recs = [r for r in per_route if int(r.get("city", -1)) == int(city)]
        per_city.append({"city": int(city), "summary": _agg(recs)})
    out = {
        "ok": True,
        "task": "shortest_path_baseline",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "city_grid_meta": {str(int(k)): str(v) for k, v in sorted(city_meta_src.items(), key=lambda kv: int(kv[0]))},
        },
        "per_city": per_city,
        "overall": _agg(per_route),
        "per_route": per_route,
        "notes": {
            "path_cost_definition": "node-weighted: sum(way_len_m[way]) along path; implemented as edge cost u->v = way_len_m[v] plus init way_len_m[start].",
            "shape_metric": "DTW/Fréchet on way-center sequences (meters, equirectangular projection from bbox).",
            "max_decode_len_applied": True,
        },
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {out_json}")


if __name__ == "__main__":
    main()

