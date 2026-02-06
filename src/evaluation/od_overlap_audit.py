from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.data.way_graph.way_sequence_dataset import WayRoutes, load_way_routes_npz

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    seed: int = 0
    val_ratio: float = 0.05
    min_hops: int = 5
    max_way_len: int = 160
    test_n_routes: int = 200
    use_split_json: bool = False
    split_part_train: str = "train"
    split_part_test: str = "test"
    split_part_val: str = "val"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _od_key(routes: WayRoutes, rid: int) -> Tuple[int, int, int]:
    return (int(routes.route_city[int(rid)]), int(routes.start_way[int(rid)]), int(routes.dest_way[int(rid)]))


def _filter_route_ids(routes: WayRoutes, *, min_hops: int, max_way_len: int) -> np.ndarray:
    keep = (routes.way_seq_len >= (int(min_hops) + 1)) & (routes.way_seq_len <= int(max_way_len))
    return np.nonzero(keep)[0].astype(np.int64, copy=False)


def _split_route_level(rids: np.ndarray, *, seed: int, val_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    rids = np.asarray(rids, dtype=np.int64).reshape(-1)
    n = int(rids.size)
    if n <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_val = int(round(float(val_ratio) * float(n)))
    n_val = max(1, min(n - 1, n_val)) if n >= 2 else 0
    va = rids[idx[:n_val]]
    tr = rids[idx[n_val:]]
    return tr, va


def _sample_test_routes(routes: WayRoutes, *, kept: np.ndarray, seed: int, n_routes: int) -> np.ndarray:
    kept = np.asarray(kept, dtype=np.int64).reshape(-1)
    if kept.size == 0:
        return kept
    cities = sorted(set(int(routes.route_city[int(r)]) for r in kept.tolist()))
    out: List[int] = []
    for c in cities:
        ids_c = kept[(routes.route_city[kept].astype(np.int64) == int(c))]
        rng = np.random.default_rng(int(seed) + 101 * int(c))
        ids_c = ids_c.copy()
        rng.shuffle(ids_c)
        out.extend(ids_c[: min(int(n_routes), int(ids_c.size))].tolist())
    return np.asarray(sorted(set(int(x) for x in out)), dtype=np.int64)


def _get_way_seq(routes: WayRoutes, rid: int) -> np.ndarray:
    L = int(routes.way_seq_len[int(rid)])
    s = int(routes.way_seq_ptr[int(rid)])
    e = s + L
    return routes.way_seq_idx[s:e].astype(np.int64, copy=False)


def _transitions(seq: Sequence[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if len(seq) <= 1:
        return out
    for i in range(int(len(seq)) - 1):
        out.append((int(seq[i]), int(seq[i + 1])))
    return out


def _quantiles(x: Sequence[float], qs: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(list(x), dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"p{int(round(float(q) * 100))}": float("nan") for q in qs}
    out: Dict[str, float] = {}
    for q in qs:
        out[f"p{int(round(float(q) * 100))}"] = float(np.quantile(a, float(q)))
    return out


def audit(
    *,
    routes: WayRoutes,
    kept: np.ndarray,
    train_rids: np.ndarray,
    test_rids: np.ndarray,
    val_rids: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    kept_set = set(int(x) for x in np.asarray(kept, dtype=np.int64).tolist())
    train = np.asarray(train_rids, dtype=np.int64).reshape(-1)
    test = np.asarray(test_rids, dtype=np.int64).reshape(-1)
    val = np.asarray(val_rids, dtype=np.int64).reshape(-1) if val_rids is not None else np.zeros((0,), dtype=np.int64)

    # Ensure all ids are within kept set.
    train = np.asarray([int(x) for x in train.tolist() if int(x) in kept_set], dtype=np.int64)
    test = np.asarray([int(x) for x in test.tolist() if int(x) in kept_set], dtype=np.int64)
    val = np.asarray([int(x) for x in val.tolist() if int(x) in kept_set], dtype=np.int64)

    train_set = set(int(x) for x in train.tolist())
    test_set = set(int(x) for x in test.tolist())
    val_set = set(int(x) for x in val.tolist())

    # OD overlap.
    od_train = set(_od_key(routes, int(r)) for r in train.tolist())
    od_test = set(_od_key(routes, int(r)) for r in test.tolist())

    # Transition coverage.
    trans_train: set[Tuple[int, int, int]] = set()
    for rid in train.tolist():
        city = int(routes.route_city[int(rid)])
        seq = _get_way_seq(routes, int(rid))
        for u, v in _transitions(seq.tolist()):
            trans_train.add((city, int(u), int(v)))

    trans_test: set[Tuple[int, int, int]] = set()
    per_route_cov: List[float] = []
    per_route_len: List[int] = []
    per_route_is_full: List[float] = []
    for rid in test.tolist():
        city = int(routes.route_city[int(rid)])
        seq = _get_way_seq(routes, int(rid))
        tr = [(city, int(u), int(v)) for u, v in _transitions(seq.tolist())]
        if not tr:
            continue
        per_route_len.append(int(len(tr)))
        hit = sum(1 for t in tr if t in trans_train)
        cov = float(hit) / float(len(tr))
        per_route_cov.append(float(cov))
        per_route_is_full.append(1.0 if cov >= 1.0 - 1e-12 else 0.0)
        for t in tr:
            trans_test.add(t)

    # Summary.
    route_overlap_train = int(len(train_set & test_set))
    route_overlap_val = int(len(val_set & test_set))
    od_overlap = int(len(od_train & od_test))
    trans_cov = float(len(trans_train & trans_test)) / float(max(1, len(trans_test)))

    rep: Dict[str, object] = {
        "n_kept": int(len(kept_set)),
        "train": {"n_routes": int(train.size), "n_od": int(len(od_train)), "n_trans": int(len(trans_train))},
        "val": {"n_routes": int(val.size)},
        "test": {"n_routes": int(test.size), "n_od": int(len(od_test)), "n_trans": int(len(trans_test))},
        "overlap": {
            "route_overlap_test_in_train": int(route_overlap_train),
            "route_overlap_test_in_val": int(route_overlap_val),
            "route_overlap_rate_test_in_train": float(route_overlap_train) / float(max(1, int(test.size))),
            "od_overlap_test_in_train": int(od_overlap),
            "od_overlap_rate_test_in_train": float(od_overlap) / float(max(1, int(len(od_test)))),
            "transition_coverage_test_in_train": float(trans_cov),
        },
        "per_test_route_transition_coverage": {
            "n_routes": int(len(per_route_cov)),
            "mean": float(np.mean(np.asarray(per_route_cov, dtype=np.float64))) if per_route_cov else float("nan"),
            "quantiles": _quantiles(per_route_cov, qs=[0.0, 0.5, 0.9, 0.95, 1.0]),
            "frac_full_coverage": float(np.mean(np.asarray(per_route_is_full, dtype=np.float64))) if per_route_is_full else float("nan"),
            "route_len_quantiles": _quantiles(per_route_len, qs=[0.0, 0.5, 0.9, 0.95, 1.0]),
        },
    }
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description="Audit OD overlap + transition coverage between train and test subsets.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--test_n_routes", type=int, default=200, help="Per city, when split_json is not provided.")

    p.add_argument("--split_json", type=Path, default=None, help="Optional split json (expects splits.train/val/test route_ids).")
    p.add_argument("--split_part_train", type=str, default="train")
    p.add_argument("--split_part_val", type=str, default="val")
    p.add_argument("--split_part_test", type=str, default="test")
    args = p.parse_args()

    cfg = Cfg(
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        min_hops=int(args.min_hops),
        max_way_len=int(args.max_way_len),
        test_n_routes=int(args.test_n_routes),
        use_split_json=bool(args.split_json is not None),
        split_part_train=str(args.split_part_train),
        split_part_val=str(args.split_part_val),
        split_part_test=str(args.split_part_test),
    )

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    kept = _filter_route_ids(routes, min_hops=int(cfg.min_hops), max_way_len=int(cfg.max_way_len))

    if args.split_json is None:
        tr, va = _split_route_level(kept, seed=int(cfg.seed), val_ratio=float(cfg.val_ratio))
        te = _sample_test_routes(routes, kept=kept, seed=int(cfg.seed), n_routes=int(cfg.test_n_routes))
    else:
        split = _read_json(Path(args.split_json))
        splits = split.get("splits", split)
        tr = np.asarray(splits.get(str(cfg.split_part_train), []), dtype=np.int64).reshape(-1)
        va = np.asarray(splits.get(str(cfg.split_part_val), []), dtype=np.int64).reshape(-1)
        te = np.asarray(splits.get(str(cfg.split_part_test), []), dtype=np.int64).reshape(-1)

    rep = {
        "ok": True,
        "task": "od_overlap_audit",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {"way_routes_npz": str(args.way_routes_npz), "split_json": (str(args.split_json) if args.split_json is not None else None)},
        "report": audit(routes=routes, kept=kept, train_rids=tr, val_rids=va, test_rids=te),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()

