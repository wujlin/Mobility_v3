from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.baselines.rnn_ar import WayRNNAR, WayRNNARCfg
from src.baselines.transformer_ar import WayTransformerAR, WayTransformerARCfg
from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.evaluation.shape_metrics import dtw_distance, frechet_distance, summarize
from src.sota.difftraj import DiffTrajCfg, DiffTrajModel
from src.sota.gtg import GTGCostNet, GTGCostNetCfg
from src.utils.time_unix import dow_from_unix, hour_from_unix
from src.utils.way_csr import build_truncated_successors_first, infer_n_ways_from_ptr, out_degree, slice_csr

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def _require_city_meta(city_meta: Dict[int, dict], cities: Iterable[int]) -> None:
    missing = [int(c) for c in cities if int(c) not in city_meta]
    if missing:
        raise SystemExit(f"[FATAL] missing --city_grid_meta for cities={missing} (PI: meters is mandatory).")


def _hops_bins() -> List[Tuple[int, Optional[int], str]]:
    return [
        (5, 10, "[5,10)"),
        (10, 20, "[10,20)"),
        (20, 30, "[20,30)"),
        (30, 40, "[30,40)"),
        (40, 60, "[40,60)"),
        (60, None, "[60,+)"),
    ]


def _bin_label(hops: int) -> str:
    hh = int(hops)
    for lo, hi, name in _hops_bins():
        if hh < int(lo):
            continue
        if hi is None or hh < int(hi):
            return str(name)
    return str(_hops_bins()[-1][2])


def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb)) / float(len(sa | sb))


def _has_loop(seq: Sequence[int]) -> bool:
    seen: set[int] = set()
    for x in seq:
        xx = int(x)
        if xx in seen:
            return True
        seen.add(xx)
    return False


def _sum_way_len_m(way_len_m: np.ndarray, seq: Sequence[int]) -> float:
    if not seq:
        return float("nan")
    ids = np.asarray([int(x) for x in seq], dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < int(way_len_m.size))]
    if ids.size == 0:
        return float("nan")
    return float(np.sum(way_len_m[ids].astype(np.float64, copy=False)))


def _nanmean(x: Sequence[float]) -> float:
    a = np.asarray(list(x), dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float("nan")


def _best_sample_index(samples: List[Dict[str, object]]) -> int:
    if not samples:
        return 0

    def _key(m: Dict[str, object]) -> Tuple[int, float, float]:
        succ = 0 if bool(m.get("success", False)) else 1
        dtw = float(m.get("dtw_m", float("nan")))
        fre = float(m.get("frechet_m", float("nan")))
        dtw = dtw if math.isfinite(dtw) else float("inf")
        fre = fre if math.isfinite(fre) else float("inf")
        return (succ, dtw, fre)

    best_i = 0
    best_k = _key(samples[0])
    for i in range(1, len(samples)):
        k = _key(samples[i])
        if k < best_k:
            best_k = k
            best_i = int(i)
    return int(best_i)


def _compress_consecutive(seq: Sequence[int]) -> List[int]:
    out: List[int] = []
    last: Optional[int] = None
    for x in seq:
        xx = int(x)
        if last is None or xx != int(last):
            out.append(xx)
            last = xx
    return out


def dijkstra_way_path(
    *,
    ptr: np.ndarray,
    idx: np.ndarray,
    way_len_m: np.ndarray,
    start: int,
    dest: int,
    max_visits: int = 500000,
) -> List[int]:
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
        succ = slice_csr(ptr, idx, u)
        for v in succ.tolist():
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


def dijkstra_learned_cost(
    *,
    ptr: np.ndarray,
    idx: np.ndarray,
    model: GTGCostNet,
    route_cond: Dict[str, torch.Tensor],
    start: int,
    dest: int,
    max_candidates: int,
    max_visits: int = 500000,
) -> List[int]:
    n = int(ptr.size) - 1
    s = int(start)
    d = int(dest)
    if s < 0 or s >= n or d < 0 or d >= n:
        return []
    if s == d:
        return [s]

    dist = np.full((n,), np.inf, dtype=np.float64)
    parent = np.full((n,), -1, dtype=np.int64)
    dist[s] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, s)]
    seen = 0

    while heap:
        du, u = heapq.heappop(heap)
        if du != float(dist[u]):
            continue
        if u == d:
            break

        succ = slice_csr(ptr, idx, u)
        if succ.size == 0:
            seen += 1
            continue
        if int(max_candidates) > 0:
            succ = succ[: int(max_candidates)]
        costs = model.edge_costs_numpy(u=u, v_list=succ, route_cond=route_cond)  # (K,)
        for v, w in zip(succ.tolist(), costs.tolist()):
            vv = int(v)
            if vv < 0 or vv >= n:
                continue
            ww = float(w)
            if not math.isfinite(ww) or ww < 0.0:
                continue
            nd = du + ww
            if nd < float(dist[vv]):
                dist[vv] = nd
                parent[vv] = u
                heapq.heappush(heap, (nd, vv))

        seen += 1
        if seen >= int(max_visits):
            break

    if not math.isfinite(float(dist[d])):
        return []

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
    device: str
    tz_offset_hours: float
    method: str
    ckpt: Optional[str]

    n_routes: int
    min_hops: int
    max_way_len: int
    max_decode_len: int

    decode_max_candidates: int
    beam_size: int
    compare_beam: bool
    eval_batch_size: int
    fast_metrics: bool

    n_samples_per_route: int
    sample_select: str

    difftraj_sample_steps: Optional[int]
    difftraj_disconnected_fail: float
    snap_cell_size: int
    snap_max_radius: int

    split_json: Optional[str] = None
    split_part: Optional[str] = None


def _load_state_dict(ckpt_obj: object) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        state = ckpt_obj["model_state_dict"]
        cfg = ckpt_obj.get("cfg", {})
        if not isinstance(state, dict):
            raise TypeError("ckpt['model_state_dict'] must be a dict")
        cfg = cfg if isinstance(cfg, dict) else {}
        return state, cfg
    if isinstance(ckpt_obj, dict):
        # assume raw state_dict
        return ckpt_obj, {}
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt_obj)}")


def _filter_cfg_dict(cfg_d: Dict[str, object], cls) -> Dict[str, object]:
    allowed = {f.name for f in fields(cls)}
    return {str(k): v for k, v in cfg_d.items() if str(k) in allowed}


def _infer_n_route_cities_from_routes(routes) -> int:
    cities = np.asarray(routes.route_city, dtype=np.int64).reshape(-1)
    uniq = sorted(set(int(x) for x in cities.tolist() if int(x) >= 0))
    return max(1, len(uniq))


def _infer_way_city_from_routes(routes, n_ways: int) -> np.ndarray:
    """
    Infer a per-way city ID by scanning GT routes.
    -1 means unknown (never appeared in any route).
    """
    way_city = np.full((int(n_ways),), -1, dtype=np.int64)
    conflicts = 0
    N = int(routes.way_seq_len.size)
    for r in range(N):
        L = int(routes.way_seq_len[r])
        if L <= 0:
            continue
        s = int(routes.way_seq_ptr[r])
        e = s + L
        if e > int(routes.way_seq_idx.size):
            continue
        c = int(routes.route_city[r])
        seq = np.asarray(routes.way_seq_idx[s:e], dtype=np.int64)
        for w in seq.tolist():
            wi = int(w)
            if wi < 0 or wi >= int(n_ways):
                continue
            prev = int(way_city[wi])
            if prev == -1:
                way_city[wi] = np.int64(c)
            elif prev != c:
                conflicts += 1
    if conflicts > 0:
        # keep going; this only affects DiffTraj snapping restriction
        pass
    return way_city


class _SnapIndex:
    def __init__(
        self,
        *,
        way_ids: np.ndarray,  # (M_city,)
        way_center_y: np.ndarray,  # (M_all,)
        way_center_x: np.ndarray,  # (M_all,)
        cell_size: int,
        max_radius: int,
    ) -> None:
        self.cell_size = max(1, int(cell_size))
        self.max_radius = max(0, int(max_radius))
        self.way_ids = np.asarray(way_ids, dtype=np.int64).reshape(-1)
        self.wy = np.asarray(way_center_y, dtype=np.float64).reshape(-1)
        self.wx = np.asarray(way_center_x, dtype=np.float64).reshape(-1)

        grid: Dict[Tuple[int, int], List[int]] = {}
        for wid in self.way_ids.tolist():
            w = int(wid)
            if w < 0 or w >= int(self.wy.size):
                continue
            cy = int(math.floor(float(self.wy[w]) / float(self.cell_size)))
            cx = int(math.floor(float(self.wx[w]) / float(self.cell_size)))
            key = (cy, cx)
            grid.setdefault(key, []).append(w)
        self.grid = grid

    def nearest_way(self, y: float, x: float) -> int:
        cy = int(math.floor(float(y) / float(self.cell_size)))
        cx = int(math.floor(float(x) / float(self.cell_size)))
        best_w = -1
        best_d2 = float("inf")

        for r in range(self.max_radius + 1):
            cand: List[int] = []
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    cand.extend(self.grid.get((cy + dy, cx + dx), []))
            if not cand:
                continue
            for w in cand:
                dy0 = float(self.wy[w]) - float(y)
                dx0 = float(self.wx[w]) - float(x)
                d2 = dy0 * dy0 + dx0 * dx0
                if d2 < best_d2:
                    best_d2 = d2
                    best_w = int(w)
            if best_w >= 0:
                return int(best_w)

        # Fallback: brute-force over city ways (rare).
        if self.way_ids.size == 0:
            return -1
        ys = self.wy[self.way_ids]
        xs = self.wx[self.way_ids]
        d2 = (ys - float(y)) ** 2 + (xs - float(x)) ** 2
        j = int(np.argmin(d2))
        return int(self.way_ids[j])


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Unified binned evaluation for baselines/SOTA (meters, DTW/Fréchet).")
    p.add_argument("--method", choices=["shortest_path", "rnn_ar", "transformer_ar", "gtg", "difftraj"], required=True)
    p.add_argument("--ckpt", type=Path, default=None, help="Checkpoint for methods that require a model (rnn_ar/transformer_ar/gtg/difftraj).")

    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)

    p.add_argument("--n_routes", type=int, default=200, help="Per city.")
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--max_decode_len", type=int, default=160)

    p.add_argument("--decode_max_candidates", type=int, default=-1, help="-1=use ckpt cfg; 0=all successors; >0=override.")
    p.add_argument("--beam_size", type=int, default=10)
    p.add_argument("--no_compare_beam", action="store_true")
    p.add_argument("--eval_batch_size", type=int, default=256, help="Batch size for baseline batched decode paths (rnn_ar/transformer_ar).")
    p.add_argument("--fast_metrics", action="store_true", help="Skip DTW/Frechet/len_ratio/final_error_m for faster eval.")

    # Multi-sample (for diffusion-like methods; also useful to keep schema aligned).
    p.add_argument("--n_samples_per_route", type=int, default=1)
    p.add_argument("--sample_select", choices=["first", "best"], default="first")

    # DiffTraj options
    p.add_argument("--difftraj_sample_steps", type=int, default=None, help="Optional sampler steps (< diffusion_steps).")
    p.add_argument("--difftraj_disconnected_fail", type=float, default=0.5, help="If disconnected_rate > thr => force hit_wall (fail).")
    p.add_argument("--snap_cell_size", type=int, default=16)
    p.add_argument("--snap_max_radius", type=int, default=8)

    p.add_argument("--dump_way_seqs", action="store_true")
    p.add_argument(
        "--out_per_route_json",
        type=Path,
        default=None,
        help="Optional: dump per-route records into a standalone JSON (same schema as way_casd_binned_eval).",
    )
    p.add_argument(
        "--split_json",
        type=Path,
        default=None,
        help="Optional: restrict evaluated routes to a split json (expects splits.train/val/test route_ids).",
    )
    p.add_argument(
        "--split_part",
        choices=["train", "val", "test"],
        default=None,
        help="Which split to evaluate (default: test when --split_json is set).",
    )
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
        device=str(args.device),
        tz_offset_hours=float(args.tz_offset_hours),
        method=str(args.method),
        ckpt=(str(args.ckpt) if args.ckpt is not None else None),
        n_routes=int(args.n_routes),
        min_hops=int(args.min_hops),
        max_way_len=int(args.max_way_len),
        max_decode_len=int(args.max_decode_len),
        decode_max_candidates=int(args.decode_max_candidates),
        beam_size=int(args.beam_size),
        compare_beam=(not bool(args.no_compare_beam)),
        eval_batch_size=max(1, int(args.eval_batch_size)),
        fast_metrics=bool(args.fast_metrics),
        n_samples_per_route=max(1, int(args.n_samples_per_route)),
        sample_select=str(args.sample_select),
        difftraj_sample_steps=(int(args.difftraj_sample_steps) if args.difftraj_sample_steps is not None else None),
        difftraj_disconnected_fail=float(args.difftraj_disconnected_fail),
        snap_cell_size=int(args.snap_cell_size),
        snap_max_radius=int(args.snap_max_radius),
        split_json=(str(args.split_json) if args.split_json is not None else None),
        split_part=(str(args.split_part) if args.split_part is not None else (("test" if args.split_json is not None else None))),
    )

    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    _set_seed(int(cfg.seed))

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64)
    idx = np.asarray(wg["way_adj_idx"], dtype=np.int64)
    n_ways = infer_n_ways_from_ptr(ptr)
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

    # Way centers (meters) per city.
    way_xy_m: Dict[int, np.ndarray] = {}
    for c in cities_obs:
        way_xy_m[int(c)] = _grid_yx_to_xy_m(way_center_y, way_center_x, meta=city_meta[int(c)])

    # Optional: restrict evaluated routes to a predefined split (route_id list).
    split_ids: Optional[np.ndarray] = None
    if cfg.split_json is not None:
        if cfg.split_part is None:
            raise SystemExit("[FATAL] --split_part is required when --split_json is set.")
        split_path = Path(str(cfg.split_json))
        if not split_path.exists():
            raise SystemExit(f"[FATAL] file not found: {split_path}")
        split_obj = _read_json(split_path)
        splits = split_obj.get("splits", split_obj)
        ids_raw = splits.get(str(cfg.split_part), None) if isinstance(splits, dict) else None
        if ids_raw is None:
            raise SystemExit(f"[FATAL] split_json missing part={cfg.split_part!r} (expects splits.train/val/test).")
        split_ids = np.asarray([int(x) for x in list(ids_raw)], dtype=np.int64).reshape(-1)
        if int(split_ids.size) == 0:
            raise SystemExit(f"[FATAL] split {cfg.split_part!r} is empty in {split_path}")

    # Sample routes per city.
    picks: Dict[int, np.ndarray] = {}
    for city in cities_obs:
        keep = (
            (routes.route_city.astype(np.int64) == int(city))
            & (routes.way_seq_len >= (int(cfg.min_hops) + 1))
            & (routes.way_seq_len <= int(cfg.max_way_len))
        )
        ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
        if split_ids is not None:
            ids = ids[np.isin(ids, split_ids, assume_unique=False)]
        rng = np.random.default_rng(int(cfg.seed) + 101 * int(city))
        rng.shuffle(ids)
        picks[int(city)] = ids[: min(int(cfg.n_routes), int(ids.size))]

    # Load model (if needed).
    method = str(cfg.method)
    model_obj: Optional[object] = None
    ckpt_cfg: Dict[str, object] = {}
    ckpt_load_ok = True

    if method != "shortest_path":
        if cfg.ckpt is None:
            raise SystemExit("[FATAL] --ckpt is required for method != shortest_path")
        if not Path(cfg.ckpt).exists():
            raise SystemExit(f"[FATAL] file not found: {cfg.ckpt}")
        ckpt_obj = torch.load(str(Path(cfg.ckpt)), map_location=device)
        state, cfg_d = _load_state_dict(ckpt_obj)
        ckpt_cfg = dict(cfg_d)
        try:
            if method == "rnn_ar":
                cfg_use = _filter_cfg_dict(cfg_d, WayRNNARCfg) if isinstance(cfg_d, dict) else {}
                cfg_use["n_ways"] = int(n_ways)
                mcfg = WayRNNARCfg(**cfg_use)
                model = WayRNNAR(cfg=mcfg).to(device)
                model.load_state_dict(state, strict=True)
                model_obj = model
            elif method == "transformer_ar":
                cfg_use = _filter_cfg_dict(cfg_d, WayTransformerARCfg) if isinstance(cfg_d, dict) else {}
                cfg_use["n_ways"] = int(n_ways)
                mcfg = WayTransformerARCfg(**cfg_use)
                model = WayTransformerAR(cfg=mcfg).to(device)
                model.load_state_dict(state, strict=True)
                model_obj = model
            elif method == "gtg":
                cfg_use = _filter_cfg_dict(cfg_d, GTGCostNetCfg) if isinstance(cfg_d, dict) else {}
                cfg_use["n_ways"] = int(n_ways)
                if "n_route_cities" not in cfg_use:
                    cfg_use["n_route_cities"] = int(_infer_n_route_cities_from_routes(routes))
                mcfg = GTGCostNetCfg(**cfg_use)
                model = GTGCostNet(cfg=mcfg).to(device)
                model.load_state_dict(state, strict=True)
                model_obj = model
            elif method == "difftraj":
                cfg_use = _filter_cfg_dict(cfg_d, DiffTrajCfg) if isinstance(cfg_d, dict) else {}
                if "n_route_cities" not in cfg_use:
                    cfg_use["n_route_cities"] = int(_infer_n_route_cities_from_routes(routes))
                mcfg = DiffTrajCfg(**cfg_use)
                model = DiffTrajModel(cfg=mcfg).to(device)
                model.load_state_dict(state, strict=True)
                model_obj = model
            else:
                raise SystemExit(f"[FATAL] unsupported method: {method}")
        except Exception:
            ckpt_load_ok = False
            raise

    # DiffTraj snap indices (per city).
    snap_index: Dict[int, _SnapIndex] = {}
    if method == "difftraj":
        way_city = _infer_way_city_from_routes(routes, n_ways=int(n_ways))
        for c in cities_obs:
            city_way_ids = np.nonzero(way_city == int(c))[0].astype(np.int64, copy=False)
            snap_index[int(c)] = _SnapIndex(
                way_ids=city_way_ids,
                way_center_y=way_center_y,
                way_center_x=way_center_x,
                cell_size=int(cfg.snap_cell_size),
                max_radius=int(cfg.snap_max_radius),
            )

    # Per-route evaluation.
    per_route: List[Dict[str, Any]] = []
    for city in cities_obs:
        pick = picks[int(city)]
        if pick.size == 0:
            continue

        xy_way = way_xy_m[int(city)]
        rnn_pred_g: Dict[int, List[int]] = {}
        rnn_pred_b: Dict[int, List[int]] = {}
        tr_pred_g: Dict[int, List[int]] = {}
        tr_pred_b: Dict[int, List[int]] = {}
        if method == "rnn_ar":
            m = model_obj
            assert isinstance(m, WayRNNAR)
            max_candidates = None if int(cfg.decode_max_candidates) < 0 else int(cfg.decode_max_candidates)
            mc_use = max_candidates
            succ_pad_t: Optional[torch.Tensor] = None
            succ_mask_t: Optional[torch.Tensor] = None
            if mc_use is not None and int(mc_use) > 0:
                succ_pad_np, succ_mask_np = build_truncated_successors_first(ptr, idx, max_candidates=int(mc_use))
                succ_pad_t = torch.as_tensor(succ_pad_np, dtype=torch.long, device=device)
                succ_mask_t = torch.as_tensor(succ_mask_np, dtype=torch.bool, device=device)
            bs = max(1, int(cfg.eval_batch_size))
            total = int(pick.size)
            for st_i in range(0, total, bs):
                ed_i = min(total, st_i + bs)
                rid_chunk = pick[st_i:ed_i].astype(np.int64, copy=False)
                sw_chunk = routes.start_way[rid_chunk].astype(np.int64, copy=False)
                dw_chunk = routes.dest_way[rid_chunk].astype(np.int64, copy=False)
                st_chunk = routes.start_t[rid_chunk].astype(np.int64, copy=False)
                hr_chunk = hour_from_unix(st_chunk, tz_offset_hours=float(cfg.tz_offset_hours)).astype(np.int64, copy=False)
                dow_chunk = dow_from_unix(st_chunk, tz_offset_hours=float(cfg.tz_offset_hours)).astype(np.int64, copy=False)
                route_cond_chunk = {
                    "start_pos": torch.as_tensor(routes.start_pos[rid_chunk].astype(np.float32, copy=False), dtype=torch.float32, device=device),
                    "dest_pos": torch.as_tensor(routes.dest_pos[rid_chunk].astype(np.float32, copy=False), dtype=torch.float32, device=device),
                    "hour": torch.as_tensor(hr_chunk, dtype=torch.long, device=device),
                    "dow": torch.as_tensor(dow_chunk, dtype=torch.long, device=device),
                    "route_city": torch.as_tensor(routes.route_city[rid_chunk].astype(np.int64, copy=False), dtype=torch.long, device=device),
                }
                pred_g_list = m.greedy_decode_batch(
                    way_adj_ptr=ptr,
                    way_adj_idx=idx,
                    start_way=sw_chunk,
                    dest_way=dw_chunk,
                    route_cond=route_cond_chunk,
                    max_len=int(cfg.max_decode_len),
                    max_candidates=mc_use,
                    succ_pad=succ_pad_t,
                    succ_mask=succ_mask_t,
                )
                pred_b_list: Optional[List[List[int]]] = None
                if bool(cfg.compare_beam):
                    pred_b_list = m.beam_search_batch(
                        way_adj_ptr=ptr,
                        way_adj_idx=idx,
                        start_way=sw_chunk,
                        dest_way=dw_chunk,
                        route_cond=route_cond_chunk,
                        beam_size=int(cfg.beam_size),
                        max_len=int(cfg.max_decode_len),
                        max_candidates=mc_use,
                        state_batch_size=max(1024, int(bs) * int(cfg.beam_size)),
                        succ_pad=succ_pad_t,
                        succ_mask=succ_mask_t,
                    )
                for j, rid_j in enumerate(rid_chunk.tolist()):
                    rj = int(rid_j)
                    rnn_pred_g[rj] = [int(x) for x in pred_g_list[j]]
                    if pred_b_list is not None:
                        rnn_pred_b[rj] = [int(x) for x in pred_b_list[j]]
                print(f"[city{int(city)}][rnn_batched] {int(ed_i)}/{int(total)} decoded")
        elif method == "transformer_ar":
            m = model_obj
            assert isinstance(m, WayTransformerAR)
            max_candidates = None if int(cfg.decode_max_candidates) < 0 else int(cfg.decode_max_candidates)
            mc_use = max_candidates
            succ_pad_t: Optional[torch.Tensor] = None
            succ_mask_t: Optional[torch.Tensor] = None
            if mc_use is not None and int(mc_use) > 0:
                succ_pad_np, succ_mask_np = build_truncated_successors_first(ptr, idx, max_candidates=int(mc_use))
                succ_pad_t = torch.as_tensor(succ_pad_np, dtype=torch.long, device=device)
                succ_mask_t = torch.as_tensor(succ_mask_np, dtype=torch.bool, device=device)
            bs = max(1, int(cfg.eval_batch_size))
            total = int(pick.size)
            for st_i in range(0, total, bs):
                ed_i = min(total, st_i + bs)
                rid_chunk = pick[st_i:ed_i].astype(np.int64, copy=False)
                sw_chunk = routes.start_way[rid_chunk].astype(np.int64, copy=False)
                dw_chunk = routes.dest_way[rid_chunk].astype(np.int64, copy=False)
                st_chunk = routes.start_t[rid_chunk].astype(np.int64, copy=False)
                hr_chunk = hour_from_unix(st_chunk, tz_offset_hours=float(cfg.tz_offset_hours)).astype(np.int64, copy=False)
                dow_chunk = dow_from_unix(st_chunk, tz_offset_hours=float(cfg.tz_offset_hours)).astype(np.int64, copy=False)
                route_cond_chunk = {
                    "start_pos": torch.as_tensor(routes.start_pos[rid_chunk].astype(np.float32, copy=False), dtype=torch.float32, device=device),
                    "dest_pos": torch.as_tensor(routes.dest_pos[rid_chunk].astype(np.float32, copy=False), dtype=torch.float32, device=device),
                    "hour": torch.as_tensor(hr_chunk, dtype=torch.long, device=device),
                    "dow": torch.as_tensor(dow_chunk, dtype=torch.long, device=device),
                    "route_city": torch.as_tensor(routes.route_city[rid_chunk].astype(np.int64, copy=False), dtype=torch.long, device=device),
                }
                pred_g_list = m.greedy_decode_batch(
                    way_adj_ptr=ptr,
                    way_adj_idx=idx,
                    start_way=sw_chunk,
                    dest_way=dw_chunk,
                    route_cond=route_cond_chunk,
                    max_len=int(cfg.max_decode_len),
                    max_candidates=mc_use,
                    succ_pad=succ_pad_t,
                    succ_mask=succ_mask_t,
                )
                pred_b_list: Optional[List[List[int]]] = None
                if bool(cfg.compare_beam):
                    pred_b_list = m.beam_search_batch(
                        way_adj_ptr=ptr,
                        way_adj_idx=idx,
                        start_way=sw_chunk,
                        dest_way=dw_chunk,
                        route_cond=route_cond_chunk,
                        beam_size=int(cfg.beam_size),
                        max_len=int(cfg.max_decode_len),
                        max_candidates=mc_use,
                        state_batch_size=max(512, int(bs) * int(cfg.beam_size)),
                        succ_pad=succ_pad_t,
                        succ_mask=succ_mask_t,
                    )
                for j, rid_j in enumerate(rid_chunk.tolist()):
                    rj = int(rid_j)
                    tr_pred_g[rj] = [int(x) for x in pred_g_list[j]]
                    if pred_b_list is not None:
                        tr_pred_b[rj] = [int(x) for x in pred_b_list[j]]
                print(f"[city{int(city)}][transformer_batched] {int(ed_i)}/{int(total)} decoded")

        for rid in pick.tolist():
            rid_i = int(rid)
            L = int(routes.way_seq_len[rid_i])
            s = int(routes.way_seq_ptr[rid_i])
            gt = routes.way_seq_idx[s : s + L].astype(np.int64, copy=False).tolist()
            gt_ids = [int(x) for x in gt]

            sw = int(routes.start_way[rid_i])
            dw = int(routes.dest_way[rid_i])
            st = int(routes.start_t[rid_i])
            start_pos = routes.start_pos[rid_i].astype(np.float64, copy=False).reshape(2)
            dest_pos = routes.dest_pos[rid_i].astype(np.float64, copy=False).reshape(2)
            hr = int(hour_from_unix(np.asarray([st], dtype=np.int64), tz_offset_hours=float(cfg.tz_offset_hours))[0])
            dow = int(dow_from_unix(np.asarray([st], dtype=np.int64), tz_offset_hours=float(cfg.tz_offset_hours))[0])

            route_cond_1 = {
                "start_pos": torch.as_tensor(start_pos.astype(np.float32, copy=False), dtype=torch.float32, device=device),
                "dest_pos": torch.as_tensor(dest_pos.astype(np.float32, copy=False), dtype=torch.float32, device=device),
                "hour": torch.as_tensor(np.asarray(hr, dtype=np.int64), dtype=torch.long, device=device),
                "dow": torch.as_tensor(np.asarray(dow, dtype=np.int64), dtype=torch.long, device=device),
                "route_city": torch.as_tensor(np.asarray(int(city), dtype=np.int64), dtype=torch.long, device=device),
            }
            route_cond_b = {
                "start_pos": route_cond_1["start_pos"].reshape(1, 2),
                "dest_pos": route_cond_1["dest_pos"].reshape(1, 2),
                "hour": route_cond_1["hour"].reshape(1),
                "dow": route_cond_1["dow"].reshape(1),
                "route_city": route_cond_1["route_city"].reshape(1),
            }

            gt_hops = int(max(0, len(gt_ids) - 1))
            if bool(cfg.fast_metrics):
                gt_len_m = float("nan")
                gt_xy = None
                dpos_xy_m = None
            else:
                gt_len_m = _sum_way_len_m(way_len_m, gt_ids)
                gt_xy = xy_way[np.asarray(gt_ids, dtype=np.int64)]
                dpos_xy_m = _grid_yx_to_xy_m(np.asarray([dest_pos[0]]), np.asarray([dest_pos[1]]), meta=city_meta[int(city)])[0]

            def _eval_pred(pred: List[int], *, force_hit_wall: bool = False, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
                if not pred:
                    out0 = {
                        "success": False,
                        "hit_wall": True,
                        "dead_end": False,
                        "has_loop": False,
                        "hops": 0,
                        "jaccard": float("nan"),
                        "len_m": float("nan"),
                        "len_ratio": float("nan"),
                        "dtw_m": float("nan"),
                        "frechet_m": float("nan"),
                        "final_error_m": float("nan"),
                    }
                    if extra:
                        out0.update(extra)
                    return out0
                success = bool(int(pred[-1]) == int(dw))
                max_len_hit = int(cfg.max_decode_len) + 1
                hit_wall = bool((not success) and (len(pred) >= max_len_hit))
                if bool(force_hit_wall) and (not success):
                    hit_wall = True
                outdeg_last = out_degree(ptr, int(pred[-1]))
                dead_end = bool((not success) and (not hit_wall) and (outdeg_last == 0))
                if bool(cfg.fast_metrics):
                    pred_len_m = float("nan")
                    dtw_m = float("nan")
                    fre_m = float("nan")
                    err_m = float("nan")
                else:
                    pred_len_m = _sum_way_len_m(way_len_m, pred)
                    pred_xy = xy_way[np.asarray(pred, dtype=np.int64)] if pred else np.zeros((0, 2), dtype=np.float64)
                    dtw_m = dtw_distance(pred_xy, gt_xy)
                    fre_m = frechet_distance(pred_xy, gt_xy)
                    last_xy = pred_xy[-1].astype(np.float64, copy=False)
                    err_m = float(np.linalg.norm(last_xy - dpos_xy_m.astype(np.float64, copy=False)))
                out1 = {
                    "success": bool(success),
                    "hit_wall": bool(hit_wall),
                    "dead_end": bool(dead_end),
                    "has_loop": bool(_has_loop(pred)),
                    "hops": int(max(0, len(pred) - 1)),
                    "jaccard": float(_jaccard(gt_ids, pred)),
                    "len_m": float(pred_len_m),
                    "len_ratio": float(pred_len_m / gt_len_m) if (math.isfinite(gt_len_m) and gt_len_m > 0 and math.isfinite(pred_len_m)) else float("nan"),
                    "dtw_m": float(dtw_m),
                    "frechet_m": float(fre_m),
                    "final_error_m": float(err_m),
                }
                if extra:
                    out1.update(extra)
                return out1

            max_candidates = None if int(cfg.decode_max_candidates) < 0 else int(cfg.decode_max_candidates)
            # For method wrappers: mc=0 means all successors.
            mc_use = max_candidates

            # Greedy and "beam"
            pred_g: List[int] = []
            pred_b: Optional[List[int]] = None
            g_extras: Optional[Dict[str, object]] = None
            b_extras: Optional[Dict[str, object]] = None
            force_hit_wall_g = False
            force_hit_wall_b = False

            if method == "shortest_path":
                pred_g = dijkstra_way_path(ptr=ptr, idx=idx, way_len_m=way_len_m, start=sw, dest=dw)
                if pred_g and len(pred_g) > int(cfg.max_decode_len) + 1:
                    pred_g = pred_g[: int(cfg.max_decode_len) + 1]
            elif method == "rnn_ar":
                pred_g = rnn_pred_g.get(int(rid_i), [int(sw)])
                if bool(cfg.compare_beam):
                    pred_b = rnn_pred_b.get(int(rid_i), list(pred_g))
            elif method == "transformer_ar":
                pred_g = tr_pred_g.get(int(rid_i), [int(sw)])
                if bool(cfg.compare_beam):
                    pred_b = tr_pred_b.get(int(rid_i), list(pred_g))
            elif method == "gtg":
                m = model_obj
                assert isinstance(m, GTGCostNet)
                mc = int(m.cfg.max_candidates) if mc_use is None else int(mc_use)
                pred_g = dijkstra_learned_cost(
                    ptr=ptr,
                    idx=idx,
                    model=m,
                    route_cond=route_cond_1,
                    start=sw,
                    dest=dw,
                    max_candidates=mc,
                )
                if pred_g and len(pred_g) > int(cfg.max_decode_len) + 1:
                    pred_g = pred_g[: int(cfg.max_decode_len) + 1]
            elif method == "difftraj":
                m = model_obj
                assert isinstance(m, DiffTrajModel)
                K = int(cfg.n_samples_per_route)
                samples: List[List[int]] = []
                metas: List[Dict[str, object]] = []

                for _k in range(K):
                    traj_rel = m.sample(route_cond=route_cond_b, steps=cfg.difftraj_sample_steps)[0].detach().cpu().numpy().astype(np.float64, copy=False)  # (T,2) rel
                    coord_scale = float(getattr(m.cfg, "coord_scale", 1024.0))
                    abs_yx = traj_rel * coord_scale + start_pos[None, :]  # (T,2)

                    snap = snap_index[int(city)]
                    way_seq = [int(snap.nearest_way(float(y), float(x))) for y, x in abs_yx.tolist()]
                    way_seq = _compress_consecutive(way_seq)
                    if way_seq and len(way_seq) > int(cfg.max_decode_len) + 1:
                        way_seq = way_seq[: int(cfg.max_decode_len) + 1]

                    # disconnected stats on deduped sequence
                    disc = 0
                    gaps: List[float] = []
                    for u, v in zip(way_seq, way_seq[1:]):
                        succ = slice_csr(ptr, idx, int(u))
                        ok = int(v) in set(int(x) for x in succ.tolist())
                        if not bool(ok):
                            disc += 1
                            if 0 <= int(u) < int(xy_way.shape[0]) and 0 <= int(v) < int(xy_way.shape[0]):
                                gaps.append(float(np.linalg.norm(xy_way[int(v)] - xy_way[int(u)])))
                    denom = max(1, int(len(way_seq) - 1))
                    disc_rate = float(disc / float(denom)) if len(way_seq) >= 2 else 0.0
                    avg_gap = float(np.mean(np.asarray(gaps, dtype=np.float64))) if gaps else 0.0
                    meta = {"disconnected_rate": float(disc_rate), "avg_gap_m": float(avg_gap)}

                    samples.append(way_seq)
                    metas.append(meta)

                # Evaluate all samples, then select.
                metrics_list = [_eval_pred(s, force_hit_wall=(float(metas[i]["disconnected_rate"]) > float(cfg.difftraj_disconnected_fail)), extra=metas[i]) for i, s in enumerate(samples)]
                if K > 1:
                    k_sel = 0 if str(cfg.sample_select) == "first" else _best_sample_index(metrics_list)
                else:
                    k_sel = 0
                pred_g = samples[int(k_sel)]
                g_extras = dict(metas[int(k_sel)])
                force_hit_wall_g = bool(float(g_extras.get("disconnected_rate", 0.0)) > float(cfg.difftraj_disconnected_fail))

                # Add sample stats into greedy dict (align with way_casd_binned_eval behavior).
                if K > 1:
                    g_extras.update(
                        {
                            "n_samples": int(K),
                            "selected_k": int(k_sel),
                            "route_any_success": bool(any(bool(m0.get("success", False)) for m0 in metrics_list)),
                            "sample_success_rate": float(np.mean([1.0 if bool(m0.get("success", False)) else 0.0 for m0 in metrics_list])),
                            "sample_hit_wall_rate": float(np.mean([1.0 if bool(m0.get("hit_wall", False)) else 0.0 for m0 in metrics_list])),
                            "sample_dead_end_rate": float(np.mean([1.0 if bool(m0.get("dead_end", False)) else 0.0 for m0 in metrics_list])),
                            "sample_loop_rate": float(np.mean([1.0 if bool(m0.get("has_loop", False)) else 0.0 for m0 in metrics_list])),
                            "sample_dtw_m_mean": _nanmean([float(m0.get("dtw_m", float("nan"))) for m0 in metrics_list]),
                            "sample_frechet_m_mean": _nanmean([float(m0.get("frechet_m", float("nan"))) for m0 in metrics_list]),
                            "sample_len_ratio_mean": _nanmean([float(m0.get("len_ratio", float("nan"))) for m0 in metrics_list]),
                            "sample_final_error_m_mean": _nanmean([float(m0.get("final_error_m", float("nan"))) for m0 in metrics_list]),
                            "sample_disconnected_rate_mean": _nanmean([float(m0.get("disconnected_rate", float("nan"))) for m0 in metrics_list]),
                            "sample_avg_gap_m_mean": _nanmean([float(m0.get("avg_gap_m", float("nan"))) for m0 in metrics_list]),
                        }
                    )
            else:
                raise SystemExit(f"[FATAL] unsupported method: {method}")

            if pred_b is None and bool(cfg.compare_beam):
                pred_b = list(pred_g)
                b_extras = {"beam_not_applicable": True} if method in {"shortest_path", "gtg", "difftraj"} else None

            mg = _eval_pred(pred_g, force_hit_wall=bool(force_hit_wall_g), extra=(g_extras if g_extras else None))
            if bool(args.dump_way_seqs):
                mg = dict(mg)
                mg["pred_way_ids"] = [int(x) for x in pred_g]

            mb: Optional[Dict[str, object]] = None
            if pred_b is not None:
                mb = _eval_pred(pred_b, force_hit_wall=bool(force_hit_wall_b), extra=(b_extras if b_extras else None))
                if bool(args.dump_way_seqs):
                    mb = dict(mb)
                    mb["pred_way_ids"] = [int(x) for x in pred_b]

            rec: Dict[str, Any] = {
                "route_id": int(rid_i),
                "city": int(city),
                "gt_hops": int(gt_hops),
                "gt_len_m": float(gt_len_m),
                "gt_avg_way_len_m": float(gt_len_m / float(max(1, len(gt_ids)))) if math.isfinite(gt_len_m) else float("nan"),
                "dest_way_len_m": float(way_len_m[int(dw)]) if 0 <= int(dw) < int(way_len_m.size) else float("nan"),
                "greedy": mg,
            }
            if bool(args.dump_way_seqs):
                rec["gt_way_ids"] = [int(x) for x in gt_ids]
                rec["start_way"] = int(sw)
                rec["dest_way"] = int(dw)
            if mb is not None:
                rec["beam"] = mb
            per_route.append(rec)

        print(f"[city{int(city)}] done={int(pick.size)} routes")

    # Bin aggregation.
    def _agg(records: List[Dict[str, Any]], *, key: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for _lo, _hi, name in _hops_bins():
            out[str(name)] = {"n": 0, "success": [], "dtw_m": [], "frechet_m": [], "len_ratio": [], "final_error_m": [], "hit_wall": [], "dead_end": [], "has_loop": []}
        for r in records:
            hops = int(r.get("gt_hops", 0))
            lab = _bin_label(hops)
            cell = out[lab]
            cell["n"] += 1
            m = r.get(key, {}) if isinstance(r.get(key, {}), dict) else {}
            cell["success"].append(1.0 if bool(m.get("success", False)) else 0.0)
            cell["dtw_m"].append(float(m.get("dtw_m", float("nan"))))
            cell["frechet_m"].append(float(m.get("frechet_m", float("nan"))))
            cell["len_ratio"].append(float(m.get("len_ratio", float("nan"))))
            cell["final_error_m"].append(float(m.get("final_error_m", float("nan"))))
            cell["hit_wall"].append(1.0 if bool(m.get("hit_wall", False)) else 0.0)
            cell["dead_end"].append(1.0 if bool(m.get("dead_end", False)) else 0.0)
            cell["has_loop"].append(1.0 if bool(m.get("has_loop", False)) else 0.0)

        rep: Dict[str, Any] = {"bins": [b[2] for b in _hops_bins()], "cells": {}}
        for lab, cell in out.items():
            n = int(cell["n"])
            rep["cells"][lab] = {
                "n": int(n),
                "success_rate": float(np.mean(np.asarray(cell["success"], dtype=np.float64))) if n else float("nan"),
                "dtw_m": summarize(cell["dtw_m"]),
                "frechet_m": summarize(cell["frechet_m"]),
                "len_ratio": summarize(cell["len_ratio"]),
                "final_error_m": summarize(cell["final_error_m"]),
                "hit_wall_rate": float(np.mean(np.asarray(cell["hit_wall"], dtype=np.float64))) if n else float("nan"),
                "dead_end_rate": float(np.mean(np.asarray(cell["dead_end"], dtype=np.float64))) if n else float("nan"),
                "loop_rate": float(np.mean(np.asarray(cell["has_loop"], dtype=np.float64))) if n else float("nan"),
            }
        return rep

    per_city: List[Dict[str, Any]] = []
    for city in cities_obs:
        recs = [r for r in per_route if int(r.get("city", -1)) == int(city)]
        city_out: Dict[str, Any] = {"city": int(city), "n": int(len(recs)), "greedy": _agg(recs, key="greedy")}
        if bool(cfg.compare_beam):
            city_out["beam"] = _agg(recs, key="beam")
            delta: Dict[str, Any] = {"bins": city_out["greedy"]["bins"], "cells": {}}
            for lab in city_out["greedy"]["cells"]:
                g = city_out["greedy"]["cells"][lab]
                b = city_out["beam"]["cells"][lab]
                delta["cells"][lab] = {"delta_success_rate": float(b["success_rate"]) - float(g["success_rate"])}
            city_out["beam_gain"] = delta
        per_city.append(city_out)

    overall: Dict[str, Any] = {"n": int(len(per_route)), "greedy": _agg(per_route, key="greedy")}
    if bool(cfg.compare_beam):
        overall["beam"] = _agg(per_route, key="beam")
        delta = {"bins": overall["greedy"]["bins"], "cells": {}}
        for lab in overall["greedy"]["cells"]:
            g = overall["greedy"]["cells"][lab]
            b = overall["beam"]["cells"][lab]
            delta["cells"][lab] = {"delta_success_rate": float(b["success_rate"]) - float(g["success_rate"])}
        overall["beam_gain"] = delta

    out = {
        "ok": True,
        "task": "unified_binned_eval",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ckpt": (str(args.ckpt) if args.ckpt is not None else None),
            "city_grid_meta": {str(int(k)): str(v) for k, v in sorted(city_meta_src.items(), key=lambda kv: int(kv[0]))},
            "split_json": (str(args.split_json) if args.split_json is not None else None),
            "split_part": (str(cfg.split_part) if cfg.split_part is not None else None),
        },
        "ckpt_strict_load_ok": bool(ckpt_load_ok),
        "per_city": per_city,
        "overall": overall,
        "notes": {
            "shape_metric": "DTW/Fréchet on way-center sequences (meters, equirectangular projection from osm_road_prob_meta.json bbox).",
            "bins": [b[2] for b in _hops_bins()],
            "candidate_policy": "first (successors[:max_candidates])",
            "beam_semantics": "For shortest_path/gtg/difftraj: beam is copied from greedy (not applicable).",
            "difftraj_fail_rule": f"if disconnected_rate > {float(cfg.difftraj_disconnected_fail):.2f} => force hit_wall (fail).",
        },
        "per_route": per_route,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {out_json}")

    if args.out_per_route_json is not None:
        out_rec = Path(args.out_per_route_json)
        out_rec.parent.mkdir(parents=True, exist_ok=True)
        out_rec.write_text(
            json.dumps(
                {
                    "ok": True,
                    "task": str(out.get("task", "")),
                    "created_at": str(out.get("created_at", "")),
                    "cfg": dict(out.get("cfg", {})),
                    "inputs": dict(out.get("inputs", {})),
                    "per_route": per_route,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[OK] saved: {out_rec}")


if __name__ == "__main__":
    main()
