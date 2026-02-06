from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.data.way_graph.way_sequence_dataset import WayRoutes, load_way_routes_npz

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class SplitCfg:
    seed: int = 0
    val_ratio: float = 0.05
    test_ratio: float = 0.1
    min_hops: int = 5
    max_way_len: int = 160
    per_city: bool = True


def _od_key(routes: WayRoutes, rid: int) -> Tuple[int, int, int]:
    return (int(routes.route_city[int(rid)]), int(routes.start_way[int(rid)]), int(routes.dest_way[int(rid)]))


def _filter_route_ids(routes: WayRoutes, *, min_hops: int, max_way_len: int) -> np.ndarray:
    keep = (routes.way_seq_len >= (int(min_hops) + 1)) & (routes.way_seq_len <= int(max_way_len))
    return np.nonzero(keep)[0].astype(np.int64, copy=False)


def _split_od_keys(keys: List[Tuple[int, int, int]], *, seed: int, val_ratio: float, test_ratio: float) -> Tuple[List, List, List]:
    if not keys:
        return [], [], []
    rng = np.random.default_rng(int(seed))
    keys = list(keys)
    rng.shuffle(keys)
    n = int(len(keys))
    n_test = int(round(float(test_ratio) * float(n)))
    n_val = int(round(float(val_ratio) * float(n)))
    # Ensure at least 1 key in each split when feasible.
    if n >= 3:
        n_test = max(1, min(n - 2, n_test))
        n_val = max(1, min(n - n_test - 1, n_val))
    else:
        n_test = 0
        n_val = max(1, min(n - 1, n_val)) if n >= 2 else 0
    test_k = keys[:n_test]
    val_k = keys[n_test : n_test + n_val]
    train_k = keys[n_test + n_val :]
    return train_k, val_k, test_k


def make_od_disjoint_split(routes: WayRoutes, *, cfg: SplitCfg) -> Dict[str, object]:
    kept = _filter_route_ids(routes, min_hops=int(cfg.min_hops), max_way_len=int(cfg.max_way_len))
    if kept.size == 0:
        return {
            "ok": False,
            "task": "od_disjoint_split",
            "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
            "cfg": asdict(cfg),
            "error": "no routes after filtering",
            "splits": {"train": [], "val": [], "test": []},
        }

    # Group routes by exact OD key (city, start_way, dest_way).
    od_to_rids: Dict[Tuple[int, int, int], List[int]] = {}
    for rid in kept.tolist():
        k = _od_key(routes, int(rid))
        od_to_rids.setdefault(k, []).append(int(rid))

    # Split OD keys (optionally per city).
    train_ids: List[int] = []
    val_ids: List[int] = []
    test_ids: List[int] = []
    city_stats: Dict[str, Dict[str, int]] = {}

    if bool(cfg.per_city):
        cities = sorted(set(int(k[0]) for k in od_to_rids.keys()))
        for c in cities:
            keys_c = [k for k in od_to_rids.keys() if int(k[0]) == int(c)]
            tr_k, va_k, te_k = _split_od_keys(
                keys_c, seed=int(cfg.seed) + 101 * int(c), val_ratio=float(cfg.val_ratio), test_ratio=float(cfg.test_ratio)
            )
            for k in tr_k:
                train_ids.extend(od_to_rids[k])
            for k in va_k:
                val_ids.extend(od_to_rids[k])
            for k in te_k:
                test_ids.extend(od_to_rids[k])
            city_stats[str(int(c))] = {
                "n_od_total": int(len(keys_c)),
                "n_od_train": int(len(tr_k)),
                "n_od_val": int(len(va_k)),
                "n_od_test": int(len(te_k)),
                "n_routes_total": int(sum(len(od_to_rids[k]) for k in keys_c)),
                "n_routes_train": int(sum(len(od_to_rids[k]) for k in tr_k)),
                "n_routes_val": int(sum(len(od_to_rids[k]) for k in va_k)),
                "n_routes_test": int(sum(len(od_to_rids[k]) for k in te_k)),
            }
    else:
        keys_all = list(od_to_rids.keys())
        tr_k, va_k, te_k = _split_od_keys(
            keys_all, seed=int(cfg.seed), val_ratio=float(cfg.val_ratio), test_ratio=float(cfg.test_ratio)
        )
        for k in tr_k:
            train_ids.extend(od_to_rids[k])
        for k in va_k:
            val_ids.extend(od_to_rids[k])
        for k in te_k:
            test_ids.extend(od_to_rids[k])

    # De-dup and sort route ids.
    train_ids = sorted(set(int(x) for x in train_ids))
    val_ids = sorted(set(int(x) for x in val_ids))
    test_ids = sorted(set(int(x) for x in test_ids))

    # Safety: check disjointness on OD keys.
    def _od_set(rids: List[int]) -> set:
        return set(_od_key(routes, rid) for rid in rids)

    od_tr = _od_set(train_ids)
    od_va = _od_set(val_ids)
    od_te = _od_set(test_ids)

    ok = True
    err: List[str] = []
    if od_tr & od_va:
        ok = False
        err.append("train/val OD overlap detected")
    if od_tr & od_te:
        ok = False
        err.append("train/test OD overlap detected")
    if od_va & od_te:
        ok = False
        err.append("val/test OD overlap detected")

    counts = {
        "kept_routes": int(kept.size),
        "n_od_total": int(len(od_to_rids)),
        "train": {"n_routes": int(len(train_ids)), "n_od": int(len(od_tr))},
        "val": {"n_routes": int(len(val_ids)), "n_od": int(len(od_va))},
        "test": {"n_routes": int(len(test_ids)), "n_od": int(len(od_te))},
    }

    out: Dict[str, object] = {
        "ok": bool(ok),
        "task": "od_disjoint_split",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "counts": counts,
        "per_city": city_stats,
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
    }
    if err:
        out["errors"] = err
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build OD-disjoint train/val/test split (exact OD: city,start_way,dest_way).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--no_per_city", action="store_true", help="If set, split OD keys globally (not per city).")
    args = p.parse_args()

    cfg = SplitCfg(
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        min_hops=int(args.min_hops),
        max_way_len=int(args.max_way_len),
        per_city=(not bool(args.no_per_city)),
    )
    routes = load_way_routes_npz(Path(args.way_routes_npz))
    rep = make_od_disjoint_split(routes, cfg=cfg)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()

