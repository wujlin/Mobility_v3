from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz


TZ_SHANGHAI = timezone(timedelta(hours=8))


def _quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, float(q)))


def _stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    return {
        "mean": float(np.mean(x)) if x.size else float("nan"),
        "p10": _quantile(x, 0.10),
        "p50": _quantile(x, 0.50),
        "p90": _quantile(x, 0.90),
        "p95": _quantile(x, 0.95),
        "p99": _quantile(x, 0.99),
        "min": float(np.min(x)) if x.size else float("nan"),
        "max": float(np.max(x)) if x.size else float("nan"),
    }


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine distance (meters).
    Good enough for bbox-scale pixel->meter estimation.
    """
    r = 6371000.0
    p1 = np.deg2rad(float(lat1))
    p2 = np.deg2rad(float(lat2))
    dlat = p2 - p1
    dlon = np.deg2rad(float(lon2) - float(lon1))
    a = np.sin(dlat * 0.5) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon * 0.5) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(max(1e-12, 1.0 - a)))
    return float(r * c)


def _load_bbox_grid(meta_path: Path) -> Tuple[int, int, Dict[str, float]]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    g = meta.get("grid", meta)
    bbox = g.get("bbox", meta.get("bbox", meta.get("bounds", None)))
    if bbox is None:
        raise ValueError(f"{meta_path} missing grid.bbox/bbox/bounds")
    # bbox could be dict or list
    if isinstance(bbox, dict):
        b = {
            "min_lon": float(bbox["min_lon"]),
            "min_lat": float(bbox["min_lat"]),
            "max_lon": float(bbox["max_lon"]),
            "max_lat": float(bbox["max_lat"]),
        }
    else:
        # [-83.25, 42.25, -82.95, 42.5]
        bb = list(bbox)
        if len(bb) != 4:
            raise ValueError(f"{meta_path} unexpected bbox: {bbox}")
        b = {"min_lon": float(bb[0]), "min_lat": float(bb[1]), "max_lon": float(bb[2]), "max_lat": float(bb[3])}
    H = int(g["H"]) if "H" in g else int(meta["H"])
    W = int(g["W"]) if "W" in g else int(meta["W"])
    return H, W, b


def _pixel_scale_m(meta_path: Path) -> Tuple[float, float]:
    H, W, b = _load_bbox_grid(meta_path)
    lat_mid = 0.5 * (b["min_lat"] + b["max_lat"])
    lon_mid = 0.5 * (b["min_lon"] + b["max_lon"])
    width_m = _haversine_m(lat_mid, b["min_lon"], lat_mid, b["max_lon"])
    height_m = _haversine_m(b["min_lat"], lon_mid, b["max_lat"], lon_mid)
    x_m_per_pix = float(width_m) / max(1.0, float(W))
    y_m_per_pix = float(height_m) / max(1.0, float(H))
    return y_m_per_pix, x_m_per_pix


def _edge_exists(ptr: np.ndarray, idx: np.ndarray, u: int, v: int) -> bool:
    s = int(ptr[u])
    e = int(ptr[u + 1])
    if e <= s:
        return False
    nbrs = np.asarray(idx[s:e], dtype=np.int64)
    return bool(np.any(nbrs == int(v)))


@dataclass(frozen=True)
class AuditCfg:
    max_way_len: int
    min_way_len: int
    # Sweeps for impact analysis (optional).
    max_step_m: List[float]
    max_loop_ratio: List[float]
    max_missing_frac: List[float]
    min_valid_transition_ratio: List[float]


def _ensure_sorted_unique(vals: Sequence[float]) -> List[float]:
    out = sorted(set(float(x) for x in vals))
    return out


def audit(
    *,
    way_routes_npz: Path,
    way_features_npz: Path,
    way_graph_npz: Path,
    city_meta_json: Dict[int, Path],
    cfg: AuditCfg,
    out_bad_json: Optional[Path],
    out_bad_max: int,
) -> Dict[str, object]:
    routes = load_way_routes_npz(Path(way_routes_npz))
    wf = np.load(str(way_features_npz), allow_pickle=True)
    wg = np.load(str(way_graph_npz), allow_pickle=True)

    need_wf = {"way_center_y", "way_center_x", "way_len_m"}
    missing_wf = sorted(list(need_wf - set(wf.files)))
    if missing_wf:
        raise ValueError(f"way_features_npz missing keys: {missing_wf}")
    need_wg = {"way_adj_ptr", "way_adj_idx"}
    missing_wg = sorted(list(need_wg - set(wg.files)))
    if missing_wg:
        raise ValueError(f"way_graph_npz missing keys: {missing_wg}")

    center_y = np.asarray(wf["way_center_y"], dtype=np.float32).reshape(-1)
    center_x = np.asarray(wf["way_center_x"], dtype=np.float32).reshape(-1)
    way_len_m = np.asarray(wf["way_len_m"], dtype=np.float32).reshape(-1)
    M = int(center_y.size)

    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(wg["way_adj_idx"], dtype=np.int64).reshape(-1)
    if int(ptr.size) != M + 1:
        raise ValueError(f"way_graph ptr size mismatch: ptr={ptr.size} vs M+1={M+1}")
    out_deg = (ptr[1:] - ptr[:-1]).astype(np.int64, copy=False)

    # Per-city pixel scales (optional).
    city_scale: Dict[int, Tuple[float, float]] = {}
    for c, meta_path in city_meta_json.items():
        try:
            city_scale[int(c)] = _pixel_scale_m(Path(meta_path))
        except Exception:
            continue

    N = int(routes.way_seq_len.size)

    # Per-route arrays.
    seq_len = routes.way_seq_len.astype(np.int64, copy=False)
    route_city = routes.route_city.astype(np.int64, copy=False)
    unique_ways = np.zeros((N,), dtype=np.int64)
    loop_ratio = np.zeros((N,), dtype=np.float64)
    missing_frac = np.zeros((N,), dtype=np.float64)
    max_step_grid = np.zeros((N,), dtype=np.float64)
    p95_step_grid = np.zeros((N,), dtype=np.float64)
    max_step_m = np.full((N,), float("nan"), dtype=np.float64)
    p95_step_m = np.full((N,), float("nan"), dtype=np.float64)
    valid_trans_ratio = np.ones((N,), dtype=np.float64)
    dead_end_frac = np.zeros((N,), dtype=np.float64)

    # Iterate routes.
    for r in range(N):
        L = int(seq_len[r])
        if L <= 0:
            unique_ways[r] = 0
            loop_ratio[r] = float("nan")
            missing_frac[r] = float("nan")
            max_step_grid[r] = float("nan")
            p95_step_grid[r] = float("nan")
            valid_trans_ratio[r] = float("nan")
            dead_end_frac[r] = float("nan")
            continue
        s = int(routes.way_seq_ptr[r])
        e = s + L
        if e > int(routes.way_seq_idx.size):
            # Corrupted pointers.
            unique_ways[r] = 0
            loop_ratio[r] = float("nan")
            missing_frac[r] = float("nan")
            max_step_grid[r] = float("nan")
            p95_step_grid[r] = float("nan")
            valid_trans_ratio[r] = float("nan")
            dead_end_frac[r] = float("nan")
            continue
        seq = np.asarray(routes.way_seq_idx[s:e], dtype=np.int64)
        seq = np.clip(seq, 0, M - 1)

        uniq = int(np.unique(seq).size)
        unique_ways[r] = np.int64(uniq)
        loop_ratio[r] = float(1.0 - float(uniq) / max(1.0, float(L)))

        miss = way_len_m[seq] <= 0.0
        missing_frac[r] = float(np.mean(miss.astype(np.float64))) if miss.size else float("nan")

        # Step distance.
        if L >= 2:
            cy = center_y[seq].astype(np.float64, copy=False)
            cx = center_x[seq].astype(np.float64, copy=False)
            dy = cy[1:] - cy[:-1]
            dx = cx[1:] - cx[:-1]
            d = np.sqrt(dy * dy + dx * dx)
            max_step_grid[r] = float(np.max(d)) if d.size else float("nan")
            p95_step_grid[r] = float(np.quantile(d, 0.95)) if d.size else float("nan")

            cc = int(route_city[r])
            if cc in city_scale:
                y_mpp, x_mpp = city_scale[cc]
                dm = np.sqrt((dy * y_mpp) ** 2 + (dx * x_mpp) ** 2)
                max_step_m[r] = float(np.max(dm)) if dm.size else float("nan")
                p95_step_m[r] = float(np.quantile(dm, 0.95)) if dm.size else float("nan")

            # Valid transition ratio vs way_graph edges (sanity check).
            bad = 0
            total = L - 1
            de = 0
            for u, v in zip(seq[:-1].tolist(), seq[1:].tolist()):
                u = int(u)
                v = int(v)
                if out_deg[u] <= 0:
                    de += 1
                if not _edge_exists(ptr, idx, u, v):
                    bad += 1
            valid_trans_ratio[r] = float(1.0 - float(bad) / max(1.0, float(total)))
            dead_end_frac[r] = float(float(de) / max(1.0, float(total)))
        else:
            max_step_grid[r] = 0.0
            p95_step_grid[r] = 0.0
            valid_trans_ratio[r] = 1.0
            dead_end_frac[r] = 0.0

    # Keep only routes within max_way_len for decision training (same as training filter).
    keep_len = (seq_len >= int(cfg.min_way_len)) & (seq_len <= int(cfg.max_way_len))

    def _city_mask(city_id: int) -> np.ndarray:
        return keep_len & (route_city == int(city_id))

    # Global / per-city stats.
    def _pack_stats(mask: np.ndarray) -> Dict[str, object]:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        return {
            "n_routes": int(np.sum(mask)),
            "way_seq_len": _stats(seq_len[mask]),
            "unique_ways": _stats(unique_ways[mask]),
            "loop_ratio": _stats(loop_ratio[mask]),
            "missing_frac": _stats(missing_frac[mask]),
            "max_step_grid": _stats(max_step_grid[mask]),
            "p95_step_grid": _stats(p95_step_grid[mask]),
            "max_step_m": _stats(max_step_m[mask]) if np.any(np.isfinite(max_step_m[mask])) else None,
            "p95_step_m": _stats(p95_step_m[mask]) if np.any(np.isfinite(p95_step_m[mask])) else None,
            "valid_transition_ratio": _stats(valid_trans_ratio[mask]),
            "dead_end_frac": _stats(dead_end_frac[mask]),
        }

    cities = sorted(list(set(int(x) for x in route_city.tolist())))
    by_city = {str(c): _pack_stats(_city_mask(c)) for c in cities}

    # Filter impact sweeps (apply on keep_len subset).
    def _count_after(mask: np.ndarray) -> Dict[str, object]:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        out = {"n_routes": int(np.sum(mask))}
        per = {}
        for c in cities:
            mc = mask & (route_city == int(c))
            per[str(c)] = int(np.sum(mc))
        out["n_by_city"] = per
        return out

    # Default thresholds (PI suggestion).
    default_gate = {
        "max_step_m": 2000.0,
        "max_loop_ratio": 0.30,
        "max_missing_frac": 0.00,
        "min_valid_transition_ratio": 0.90,
    }

    base = keep_len.copy()
    m_default = base.copy()
    if np.any(np.isfinite(max_step_m)):
        m_default &= np.isfinite(max_step_m) & (max_step_m <= float(default_gate["max_step_m"]))
    m_default &= np.isfinite(loop_ratio) & (loop_ratio <= float(default_gate["max_loop_ratio"]))
    m_default &= np.isfinite(missing_frac) & (missing_frac <= float(default_gate["max_missing_frac"]))
    m_default &= np.isfinite(valid_trans_ratio) & (valid_trans_ratio >= float(default_gate["min_valid_transition_ratio"]))
    default_after = _count_after(m_default)

    sweeps: Dict[str, object] = {}
    if cfg.max_step_m and np.any(np.isfinite(max_step_m)):
        ss = {}
        for thr in _ensure_sorted_unique(cfg.max_step_m):
            m = base & np.isfinite(max_step_m) & (max_step_m <= float(thr))
            ss[str(float(thr))] = _count_after(m)
        sweeps["max_step_m"] = ss
    if cfg.max_loop_ratio:
        ss = {}
        for thr in _ensure_sorted_unique(cfg.max_loop_ratio):
            m = base & np.isfinite(loop_ratio) & (loop_ratio <= float(thr))
            ss[str(float(thr))] = _count_after(m)
        sweeps["max_loop_ratio"] = ss
    if cfg.max_missing_frac:
        ss = {}
        for thr in _ensure_sorted_unique(cfg.max_missing_frac):
            m = base & np.isfinite(missing_frac) & (missing_frac <= float(thr))
            ss[str(float(thr))] = _count_after(m)
        sweeps["max_missing_frac"] = ss
    if cfg.min_valid_transition_ratio:
        ss = {}
        for thr in _ensure_sorted_unique(cfg.min_valid_transition_ratio):
            m = base & np.isfinite(valid_trans_ratio) & (valid_trans_ratio >= float(thr))
            ss[str(float(thr))] = _count_after(m)
        sweeps["min_valid_transition_ratio"] = ss

    # Worst cases (within keep_len).
    def _topk(metric: np.ndarray, k: int, *, desc: bool = True) -> List[Dict[str, object]]:
        metric = np.asarray(metric, dtype=np.float64).reshape(-1)
        m = np.isfinite(metric) & keep_len
        ids = np.nonzero(m)[0]
        if ids.size == 0:
            return []
        vals = metric[ids]
        order = np.argsort(vals)
        if desc:
            order = order[::-1]
        take = ids[order[: int(k)]]
        out: List[Dict[str, object]] = []
        for rid in take.tolist():
            out.append(
                {
                    "route_id": int(rid),
                    "route_city": int(route_city[rid]),
                    "way_seq_len": int(seq_len[rid]),
                    "unique_ways": int(unique_ways[rid]),
                    "loop_ratio": float(loop_ratio[rid]),
                    "missing_frac": float(missing_frac[rid]),
                    "max_step_m": float(max_step_m[rid]) if np.isfinite(max_step_m[rid]) else None,
                    "max_step_grid": float(max_step_grid[rid]) if np.isfinite(max_step_grid[rid]) else None,
                    "valid_transition_ratio": float(valid_trans_ratio[rid]) if np.isfinite(valid_trans_ratio[rid]) else None,
                    "dead_end_frac": float(dead_end_frac[rid]) if np.isfinite(dead_end_frac[rid]) else None,
                }
            )
        return out

    worst = {
        "max_step_m": _topk(max_step_m, out_bad_max) if np.any(np.isfinite(max_step_m)) else [],
        "max_step_grid": _topk(max_step_grid, out_bad_max),
        "missing_frac": _topk(missing_frac, out_bad_max),
        "loop_ratio": _topk(loop_ratio, out_bad_max),
        "dead_end_frac": _topk(dead_end_frac, out_bad_max),
        "invalid_transition_ratio": _topk(1.0 - valid_trans_ratio, out_bad_max),
    }

    # Optional: emit bad routes list for default thresholds.
    if out_bad_json is not None:
        bad_ids = np.nonzero(base & ~m_default)[0].astype(np.int64, copy=False).tolist()
        payload = {
            "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
            "thresholds": default_gate,
            "max_way_len": int(cfg.max_way_len),
            "min_way_len": int(cfg.min_way_len),
            "n_bad": int(len(bad_ids)),
            "bad_route_ids": bad_ids,
        }
        out_bad_json.parent.mkdir(parents=True, exist_ok=True)
        out_bad_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out = {
        "ok": True,
        "task": "audit_way_routes_quality",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "way_routes_npz": str(way_routes_npz),
            "way_features_npz": str(way_features_npz),
            "way_graph_npz": str(way_graph_npz),
            "city_meta_json": {str(k): str(v) for k, v in city_meta_json.items()},
        },
        "cfg": asdict(cfg),
        "note": (
            "valid_transition_ratio is computed against way_graph adjacency. "
            "If your way_graph is built from route transitions (current default), this is a sanity check and is expected to be ~1.0."
        ),
        "stats_global": _pack_stats(keep_len),
        "stats_by_city": by_city,
        "filter_impact": {"default_gate": default_gate, "default_after": default_after, "sweeps": sweeps},
        "worst_cases": worst,
    }
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit way_routes quality and estimate filtering impact (Decision stage).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_bad_json", type=Path, default=None, help="Optional: dump bad route_ids under default gate thresholds.")
    p.add_argument("--out_bad_max", type=int, default=50, help="Worst-case list size per metric in report.")

    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--min_way_len", type=int, default=3)

    p.add_argument("--city_meta_json", type=Path, nargs="*", default=None, help="One or more osm_road_prob_meta.json (ordered by city id: 0,1,2...).")

    p.add_argument("--max_step_m", type=float, nargs="*", default=[1000.0, 2000.0, 3000.0])
    p.add_argument("--max_loop_ratio", type=float, nargs="*", default=[0.2, 0.3, 0.4])
    p.add_argument("--max_missing_frac", type=float, nargs="*", default=[0.0, 0.05, 0.1])
    p.add_argument("--min_valid_transition_ratio", type=float, nargs="*", default=[0.9, 1.0])
    return p


def main() -> None:
    args = build_argparser().parse_args()
    city_meta: Dict[int, Path] = {}
    if args.city_meta_json:
        for i, p in enumerate(args.city_meta_json):
            city_meta[int(i)] = Path(p)

    rep = audit(
        way_routes_npz=Path(args.way_routes_npz),
        way_features_npz=Path(args.way_features_npz),
        way_graph_npz=Path(args.way_graph_npz),
        city_meta_json=city_meta,
        cfg=AuditCfg(
            max_way_len=int(args.max_way_len),
            min_way_len=int(args.min_way_len),
            max_step_m=[float(x) for x in (args.max_step_m or [])],
            max_loop_ratio=[float(x) for x in (args.max_loop_ratio or [])],
            max_missing_frac=[float(x) for x in (args.max_missing_frac or [])],
            min_valid_transition_ratio=[float(x) for x in (args.min_valid_transition_ratio or [])],
        ),
        out_bad_json=Path(args.out_bad_json) if args.out_bad_json is not None else None,
        out_bad_max=int(args.out_bad_max),
    )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Print a compact 1-screen summary.
    g = rep["stats_global"]
    print(f"[way_routes] {args.way_routes_npz}")
    print(f"[N] {g['n_routes']} (len in [{args.min_way_len},{args.max_way_len}])")
    wl = g["way_seq_len"]
    print(f"[way_seq_len] p50={wl['p50']:.1f} p90={wl['p90']:.1f} max={wl['max']:.0f}")
    lr = g["loop_ratio"]
    print(f"[loop_ratio] p50={lr['p50']:.3f} p90={lr['p90']:.3f} p99={lr['p99']:.3f}")
    mf = g["missing_frac"]
    print(f"[missing_frac] p50={mf['p50']:.3f} p90={mf['p90']:.3f} p99={mf['p99']:.3f}")
    vt = g["valid_transition_ratio"]
    print(f"[valid_trans_ratio] p50={vt['p50']:.3f} p90={vt['p90']:.3f} min={vt['min']:.3f}")
    de = g["dead_end_frac"]
    print(f"[dead_end_frac] p50={de['p50']:.3f} p90={de['p90']:.3f} max={de['max']:.3f}")
    if g.get("max_step_m") is not None:
        sd = g["max_step_m"]
        print(f"[max_step_m] p50={sd['p50']:.1f} p95={sd['p95']:.1f} p99={sd['p99']:.1f} max={sd['max']:.1f}")
    else:
        sd = g["max_step_grid"]
        print(f"[max_step_grid] p50={sd['p50']:.1f} p95={sd['p95']:.1f} p99={sd['p99']:.1f} max={sd['max']:.1f}")
    print(f"[saved] {out_path}")
    if args.out_bad_json is not None:
        print(f"[saved] {args.out_bad_json}")


if __name__ == "__main__":
    main()
