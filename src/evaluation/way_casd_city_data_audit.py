"""
Way-CASD Data Audit (Detroit vs Columbus)

PI 诊断2：排查两城数据差异是否导致 hit-wall/成功率差异。

输出（JSON）：
1) 训练/验证集两城样本数量比例（使用与 train_way_casd_autoencoder.py 相同的 split 口径）
2) way graph 出度分布（按“该城 routes 用到的 ways”统计）
3) route way_seq_len 分布（按城市/按 split）

注意：我们不假设 way_features 里有 way_city；按 route_city 聚合其用到的 way 集合即可。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    seed: int
    val_ratio: float
    max_way_len: int


def _split_dataset(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n))
    n_val = int(round(float(val_ratio) * float(n)))
    n_val = max(1, min(n_val, n - 1))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx.astype(np.int64, copy=False), val_idx.astype(np.int64, copy=False)


def _quantiles_int(values: np.ndarray, qs: Tuple[int, ...] = (0, 50, 90, 95, 99, 100)) -> Dict[str, int | None]:
    if values.size == 0:
        return {f"p{q:02d}": None for q in qs}
    v = np.asarray(values, dtype=np.float64)
    out: Dict[str, int | None] = {}
    for q in qs:
        out[f"p{q:02d}"] = int(np.percentile(v, float(q)))
    return out


def _quantiles_float(values: np.ndarray, qs: Tuple[int, ...] = (0, 50, 90, 95, 99, 100)) -> Dict[str, float | None]:
    if values.size == 0:
        return {f"p{q:02d}": None for q in qs}
    v = np.asarray(values, dtype=np.float64)
    out: Dict[str, float | None] = {}
    for q in qs:
        out[f"p{q:02d}"] = float(np.percentile(v, float(q)))
    return out


def _flatten_way_tokens(routes, route_ids: np.ndarray) -> np.ndarray:
    route_ids = np.asarray(route_ids, dtype=np.int64).reshape(-1)
    if route_ids.size == 0:
        return np.zeros((0,), dtype=np.int64)
    lens = routes.way_seq_len[route_ids].astype(np.int64)
    total = int(lens.sum())
    out = np.empty((total,), dtype=np.int64)
    cur = 0
    for rid, L in zip(route_ids.tolist(), lens.tolist()):
        rid = int(rid)
        L = int(L)
        s = int(routes.way_seq_ptr[rid])
        e = s + L
        out[cur : cur + L] = routes.way_seq_idx[s:e].astype(np.int64, copy=False)
        cur += L
    return out


def _route_ids_filtered(routes, max_way_len: int) -> np.ndarray:
    keep = (routes.way_seq_len > 0) & (routes.way_seq_len <= int(max_way_len))
    return np.nonzero(keep)[0].astype(np.int64, copy=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Way-CASD audit: train split ratio + per-city outdeg/len distributions.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_regions_npz", type=Path, default=None, help="Optional: include per-city region granularity stats.")
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--max_way_len", type=int, default=160)
    args = p.parse_args()

    cfg = Cfg(seed=int(args.seed), val_ratio=float(args.val_ratio), max_way_len=int(args.max_way_len))

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64).reshape(-1)
    outdeg_all = (ptr[1:] - ptr[:-1]).astype(np.int64, copy=False)

    way_region: np.ndarray | None = None
    if args.way_regions_npz is not None:
        wr = np.load(str(Path(args.way_regions_npz)), allow_pickle=True)
        if "way_region" not in wr.files:
            raise SystemExit("[FATAL] way_regions_npz missing key: way_region")
        way_region = np.asarray(wr["way_region"], dtype=np.int64).reshape(-1)

    route_ids_all = _route_ids_filtered(routes, max_way_len=int(cfg.max_way_len))
    n = int(route_ids_all.size)
    if n < 2:
        raise SystemExit(f"Not enough routes after filter: n={n}")

    # Training split uses indices over the filtered dataset order (same as train script).
    train_idx, val_idx = _split_dataset(n, float(cfg.val_ratio), int(cfg.seed))
    train_rids = route_ids_all[train_idx]
    val_rids = route_ids_all[val_idx]

    n_cities = int(np.max(routes.route_city.astype(np.int64))) + 1

    def _by_city(rids: np.ndarray) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for c in range(n_cities):
            mask = routes.route_city[rids].astype(np.int64) == int(c)
            out[str(c)] = rids[mask]
        return out

    splits = {
        "all": route_ids_all,
        "train": train_rids,
        "val": val_rids,
    }

    out: Dict[str, object] = {
        "ok": True,
        "task": "way_casd_city_data_audit",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_regions_npz": (str(args.way_regions_npz) if args.way_regions_npz is not None else None),
        },
        "n_cities": int(n_cities),
        "splits": {},
    }

    for split_name, rids in splits.items():
        rids = np.asarray(rids, dtype=np.int64).reshape(-1)
        by_city = _by_city(rids)
        split_rep: Dict[str, object] = {
            "n_routes": int(rids.size),
            "by_city": {},
        }
        for c_str, rids_c in by_city.items():
            rids_c = np.asarray(rids_c, dtype=np.int64).reshape(-1)
            lens = routes.way_seq_len[rids_c].astype(np.int64, copy=False)
            toks = _flatten_way_tokens(routes, rids_c)
            uniq = np.unique(toks) if toks.size > 0 else np.zeros((0,), dtype=np.int64)
            deg = outdeg_all[uniq] if uniq.size > 0 else np.zeros((0,), dtype=np.int64)
            dead_end = (deg <= 0).astype(np.float64, copy=False)

            region_rep: Dict[str, object] | None = None
            if way_region is not None and uniq.size > 0:
                reg = way_region[uniq].astype(np.int64, copy=False)
                reg = reg[reg >= 0]
                if reg.size > 0:
                    reg_ids, reg_counts = np.unique(reg, return_counts=True)
                    region_rep = {
                        "n_regions": int(reg_ids.size),
                        "region_size_ways": {
                            "p50": int(np.percentile(reg_counts, 50)) if reg_counts.size > 0 else None,
                            "p90": int(np.percentile(reg_counts, 90)) if reg_counts.size > 0 else None,
                            "max": int(reg_counts.max()) if reg_counts.size > 0 else None,
                            "quantiles": _quantiles_int(reg_counts.astype(np.int64, copy=False)),
                        },
                    }
            split_rep["by_city"][c_str] = {
                "n_routes": int(rids_c.size),
                "route_len": {
                    "p50": int(np.percentile(lens, 50)) if lens.size > 0 else None,
                    "p90": int(np.percentile(lens, 90)) if lens.size > 0 else None,
                    "max": int(lens.max()) if lens.size > 0 else None,
                    "quantiles": _quantiles_int(lens),
                },
                "ways_used": int(uniq.size),
                "outdeg": {
                    "p50": int(np.percentile(deg, 50)) if deg.size > 0 else None,
                    "p90": int(np.percentile(deg, 90)) if deg.size > 0 else None,
                    "max": int(deg.max()) if deg.size > 0 else None,
                    "quantiles": _quantiles_int(deg),
                    "dead_end_frac": float(np.mean(dead_end)) if dead_end.size > 0 else None,
                    "frac_gt2": float(np.mean((deg > 2).astype(np.float64))) if deg.size > 0 else None,
                    "frac_gt8": float(np.mean((deg > 8).astype(np.float64))) if deg.size > 0 else None,
                    "frac_gt32": float(np.mean((deg > 32).astype(np.float64))) if deg.size > 0 else None,
                },
                "tokens": {
                    "n_tokens": int(toks.size),
                    "tokens_per_route_mean": float(toks.size) / float(max(1, rids_c.size)),
                },
                "regions": region_rep,
            }
        out["splits"][split_name] = split_rep

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(str(args.out_json))


if __name__ == "__main__":
    main()
