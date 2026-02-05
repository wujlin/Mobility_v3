from __future__ import annotations

"""Analyze how many GT routes match shortest path (hop-count) on the way graph."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def load_way_graph_csr(way_graph_npz: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a directed way-graph in CSR format.

    Expected keys (new): way_adj_ptr / way_adj_idx
    Fallback keys (legacy): adj_ptr / adj_idx
    """
    data = np.load(str(way_graph_npz), allow_pickle=True)
    if "way_adj_ptr" in data.files and "way_adj_idx" in data.files:
        ptr = np.asarray(data["way_adj_ptr"], dtype=np.int64).reshape(-1)
        idx = np.asarray(data["way_adj_idx"], dtype=np.int64).reshape(-1)
        return ptr, idx
    if "adj_ptr" in data.files and "adj_idx" in data.files:
        ptr = np.asarray(data["adj_ptr"], dtype=np.int64).reshape(-1)
        idx = np.asarray(data["adj_idx"], dtype=np.int64).reshape(-1)
        return ptr, idx
    raise ValueError(f"way_graph_npz missing CSR keys. Got: {sorted(list(data.files))}")


class ShortestHopsBFS:
    def __init__(self, ptr: np.ndarray, idx: np.ndarray) -> None:
        self.ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
        self.idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        n = int(self.ptr.size) - 1
        if n <= 0:
            raise ValueError("Bad CSR ptr: need ptr.size>=2")
        self.n = int(n)
        self._seen = np.zeros((self.n,), dtype=np.int32)
        self._dist = np.zeros((self.n,), dtype=np.int32)
        self._stamp = 0

    def shortest_hops(self, start: int, dest: int, *, max_visits: int = 200000) -> Optional[int]:
        s = int(start)
        d = int(dest)
        if s < 0 or d < 0 or s >= int(self.n) or d >= int(self.n):
            return None
        if s == d:
            return 0

        from collections import deque

        self._stamp += 1
        stamp = int(self._stamp)
        q = deque([s])
        self._seen[s] = stamp
        self._dist[s] = 0

        seen = 0
        while q:
            u = int(q.popleft())
            du = int(self._dist[u])
            ss = int(self.ptr[u])
            ee = int(self.ptr[u + 1])
            for v in self.idx[ss:ee]:
                vv = int(v)
                if vv < 0 or vv >= int(self.n):
                    continue
                if int(self._seen[vv]) == stamp:
                    continue
                self._seen[vv] = stamp
                self._dist[vv] = du + 1
                if vv == d:
                    return int(du + 1)
                q.append(vv)
            seen += 1
            if seen >= int(max_visits):
                return None
        return None


def analyze_shortest_match(
    *,
    way_routes_npz: str,
    way_graph_npz: str,
    max_visits: int = 200000,
    max_routes: Optional[int] = None,
    seed: int = 0,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    log.info("Loading graph...")
    ptr, idx = load_way_graph_csr(way_graph_npz)
    log.info(f"Graph CSR: n_ways={int(ptr.size) - 1} edges={int(idx.size)}")
    bfs = ShortestHopsBFS(ptr, idx)

    log.info("Loading routes...")
    routes = load_way_routes_npz(Path(way_routes_npz))
    n_routes = int(routes.way_seq_len.size)
    log.info(f"Loaded {n_routes} routes")

    route_ids = np.arange(n_routes, dtype=np.int64)
    if max_routes is not None and int(max_routes) < n_routes:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(route_ids)
        route_ids = route_ids[: int(max_routes)]
        log.info(f"Subsampled routes: {int(route_ids.size)}/{n_routes} (seed={int(seed)})")

    sp_cache: Dict[Tuple[int, int], Optional[int]] = {}
    match_count = 0
    reachable_count = 0
    per_city: Dict[int, Dict[str, int]] = {}

    for k, rid in enumerate(route_ids.tolist()):
        rr = int(rid)
        city = int(routes.route_city[rr])
        if city not in per_city:
            per_city[city] = {"total": 0, "reachable": 0, "match": 0}
        per_city[city]["total"] += 1

        L = int(routes.way_seq_len[rr])
        gt_hops = max(0, L - 1)
        sw = int(routes.start_way[rr])
        dw = int(routes.dest_way[rr])

        key = (int(sw), int(dw))
        if key not in sp_cache:
            sp_cache[key] = bfs.shortest_hops(int(sw), int(dw), max_visits=int(max_visits))
        sp_hops = sp_cache[key]

        if sp_hops is not None:
            reachable_count += 1
            per_city[city]["reachable"] += 1
            if int(gt_hops) == int(sp_hops):
                match_count += 1
                per_city[city]["match"] += 1

        if (k + 1) % 2000 == 0:
            log.info(f"Processed {k + 1}/{int(route_ids.size)} routes (cache_size={len(sp_cache)})")

    results: Dict[str, Any] = {
        "config": {
            "way_routes_npz": str(way_routes_npz),
            "way_graph_npz": str(way_graph_npz),
            "max_visits": int(max_visits),
            "max_routes": int(max_routes) if max_routes is not None else None,
            "seed": int(seed),
        },
        "overall": {
            "n_routes": int(route_ids.size),
            "n_reachable": int(reachable_count),
            "reachable_rate": float(reachable_count / int(route_ids.size)) if int(route_ids.size) > 0 else 0.0,
            "n_match": int(match_count),
            "match_rate": float(match_count / int(route_ids.size)) if int(route_ids.size) > 0 else 0.0,
            "match_rate_among_reachable": float(match_count / int(reachable_count)) if int(reachable_count) > 0 else 0.0,
        },
        "per_city": {},
    }

    for city, stats in sorted(per_city.items(), key=lambda kv: int(kv[0])):
        total = int(stats["total"])
        reachable = int(stats["reachable"])
        match = int(stats["match"])
        results["per_city"][str(int(city))] = {
            "n_routes": int(total),
            "n_reachable": int(reachable),
            "reachable_rate": float(reachable / total) if total > 0 else 0.0,
            "n_match": int(match),
            "match_rate": float(match / total) if total > 0 else 0.0,
            "match_rate_among_reachable": float(match / reachable) if reachable > 0 else 0.0,
        }

    log.info(f"Overall match rate: {results['overall']['match_rate']:.1%} (reachable={results['overall']['reachable_rate']:.1%})")

    if output_path:
        out = Path(str(output_path))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.info(f"Saved to {out}")

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze GT shortest-path (hop-count) match rate on a way graph.")
    p.add_argument("--way_routes_npz", type=str, required=True)
    p.add_argument("--way_graph_npz", type=str, required=True)
    p.add_argument("--max_visits", type=int, default=200000)
    p.add_argument("--max_routes", type=int, default=None, help="Debug-only: subsample total routes (random).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()

    analyze_shortest_match(
        way_routes_npz=str(args.way_routes_npz),
        way_graph_npz=str(args.way_graph_npz),
        max_visits=int(args.max_visits),
        max_routes=int(args.max_routes) if args.max_routes is not None else None,
        seed=int(args.seed),
        output_path=str(args.output),
    )


if __name__ == "__main__":
    main()
