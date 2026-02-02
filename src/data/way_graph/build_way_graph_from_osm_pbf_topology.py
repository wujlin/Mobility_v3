"""
Build a way-level adjacency graph by combining:

  (1) OSM topology: two ways are adjacent iff they share at least one OSM node
  (2) Behavior edges: transitions observed in way_routes.npz sequences

This is intended to fix severe fragmentation of a purely GT-transition graph,
which blocks hierarchical planning (Region->Way).

Output format matches existing way_graph.npz:
  - way_osm_id: (M,) int64
  - way_adj_ptr: (M+1,) int64 CSR ptr
  - way_adj_idx: (E,) int32 CSR idx (directed)
  - meta: dict with build stats + connectivity summary

Usage (WSL):
  python -m src.data.way_graph.build_way_graph_from_osm_pbf_topology \
    --way_routes_npz /path/to/way_routes.npz \
    --osm_pbf /path/to/detroit.pbf \
    --osm_pbf /path/to/columbus.pbf \
    --out_npz /path/to/way_graph_topo.npz

Dependencies:
  - osmium (pyosmium): `conda install -c conda-forge pyosmium` or `pip install osmium`
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _import_osmium():
    try:
        import osmium  # type: ignore

        return osmium
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "[FATAL] missing dependency: osmium (pyosmium). Install via `conda install -c conda-forge pyosmium` "
            "or `pip install osmium`."
        ) from e


@dataclass(frozen=True)
class Cfg:
    include_gt_transitions: bool
    max_ways_per_node: int  # 0 = no cap (debug safety)


def _p(x: np.ndarray, q: float) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, q))


def _load_way_city_from_routes(routes: np.lib.npyio.NpzFile, M: int) -> Tuple[np.ndarray, int]:
    """
    Infer way->city id from route sequences.
    Assumption: each way belongs to exactly one city in our dataset.
    """
    ptr = np.asarray(routes["way_seq_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(routes["way_seq_idx"], dtype=np.int64).reshape(-1)
    lens = np.asarray(routes["way_seq_len"], dtype=np.int64).reshape(-1)
    route_city = np.asarray(routes["route_city"], dtype=np.int64).reshape(-1)
    N = int(lens.size)

    way_city = np.full((M,), -1, dtype=np.int64)
    conflicts = 0
    for r in range(N):
        L = int(lens[r])
        if L <= 0:
            continue
        s = int(ptr[r])
        e = s + L
        if e > int(idx.size):
            continue
        c = int(route_city[r])
        seq = np.asarray(idx[s:e], dtype=np.int64)
        # Dedup per route to reduce writes.
        for w in np.unique(seq).tolist():
            wi = int(w)
            if wi < 0 or wi >= M:
                continue
            prev = int(way_city[wi])
            if prev == -1:
                way_city[wi] = np.int64(c)
            elif prev != c:
                conflicts += 1
    return way_city, int(conflicts)


def _gt_transition_edges(routes: np.lib.npyio.NpzFile, M: int) -> List[Set[int]]:
    ptr = np.asarray(routes["way_seq_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(routes["way_seq_idx"], dtype=np.int64).reshape(-1)
    lens = np.asarray(routes["way_seq_len"], dtype=np.int64).reshape(-1)
    N = int(lens.size)

    adj: List[Set[int]] = [set() for _ in range(M)]
    for r in range(N):
        L = int(lens[r])
        if L <= 1:
            continue
        s = int(ptr[r])
        e = s + L
        if e > int(idx.size):
            continue
        seq = np.asarray(idx[s:e], dtype=np.int64)
        for j in range(L - 1):
            a = int(seq[j])
            b = int(seq[j + 1])
            if a < 0 or b < 0 or a >= M or b >= M or a == b:
                continue
            adj[a].add(b)
    return adj


class _WayNodeCollector:
    def __init__(
        self,
        *,
        osmium_mod,
        wanted_way_ids: Set[int],
        way_to_idx: Dict[int, int],
        node_to_ways: Dict[int, List[int]],
        processed_way_idx: Set[int],
        max_ways_per_node: int,
    ) -> None:
        self.osmium = osmium_mod
        self.wanted_way_ids = wanted_way_ids
        self.way_to_idx = way_to_idx
        self.node_to_ways = node_to_ways
        self.processed_way_idx = processed_way_idx
        self.max_ways_per_node = int(max_ways_per_node)

        class Handler(osmium_mod.SimpleHandler):  # type: ignore[misc]
            def __init__(self, outer: "_WayNodeCollector") -> None:
                super().__init__()
                self.outer = outer

            def way(self, w):  # type: ignore[no-untyped-def]
                outer = self.outer
                wid = int(getattr(w, "id", -1))
                if wid not in outer.wanted_way_ids:
                    return
                wi = outer.way_to_idx.get(wid)
                if wi is None:
                    return
                if int(wi) in outer.processed_way_idx:
                    return
                outer.processed_way_idx.add(int(wi))
                # Collect node refs (dedup within way).
                try:
                    nodes = getattr(w, "nodes", None)
                    if nodes is None:
                        return
                    node_ids = {int(n.ref) for n in nodes}  # type: ignore[attr-defined]
                except Exception:
                    return
                for nid in node_ids:
                    lst = outer.node_to_ways.get(nid)
                    if lst is None:
                        outer.node_to_ways[nid] = [int(wi)]
                        continue
                    if outer.max_ways_per_node > 0 and len(lst) >= outer.max_ways_per_node:
                        # Safety cap: avoid exploding cliques on extremely high-degree nodes.
                        continue
                    lst.append(int(wi))

        self.handler = Handler(self)

    def apply(self, pbf: Path) -> None:
        self.handler.apply_file(str(pbf), locations=False)  # type: ignore[attr-defined]


def _csr_from_adj(adj: Sequence[Set[int]]) -> Tuple[np.ndarray, np.ndarray]:
    M = int(len(adj))
    out_ptr = np.zeros((M + 1,), dtype=np.int64)
    out_idx: List[int] = []
    out_deg = np.zeros((M,), dtype=np.int64)
    for i in range(M):
        nbrs = sorted(int(x) for x in adj[i] if int(x) != int(i))
        out_deg[i] = int(len(nbrs))
        out_idx.extend(nbrs)
        out_ptr[i + 1] = np.int64(len(out_idx))
    return out_ptr, np.asarray(out_idx, dtype=np.int32)


def _connectivity_stats_from_adj(adj: Sequence[Set[int]], way_city: Optional[np.ndarray]) -> Dict[str, object]:
    M = int(len(adj))
    parent = np.arange(M, dtype=np.int64)
    size = np.ones((M,), dtype=np.int64)
    has_nbr = np.zeros((M,), dtype=bool)

    def find(x: int) -> int:
        xx = int(x)
        while parent[xx] != xx:
            parent[xx] = parent[parent[xx]]
            xx = int(parent[xx])
        return int(xx)

    def union(a: int, b: int) -> None:
        ra = find(int(a))
        rb = find(int(b))
        if ra == rb:
            return
        if int(size[ra]) < int(size[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for u in range(M):
        for v in adj[u]:
            uu = int(u)
            vv = int(v)
            if uu == vv:
                continue
            has_nbr[uu] = True
            has_nbr[vv] = True
            union(uu, vv)

    roots = np.asarray([find(i) for i in range(M)], dtype=np.int64)
    # component sizes
    uniq, inv = np.unique(roots, return_inverse=True)
    comp_sizes = np.bincount(inv.astype(np.int64), minlength=int(uniq.size)).astype(np.int64)
    n_comp = int(comp_sizes.size)
    largest = int(comp_sizes.max()) if comp_sizes.size else 0
    isolate_n = int(np.sum(~has_nbr))

    out: Dict[str, object] = {
        "n_connected_components_undirected": int(n_comp),
        "largest_cc_n": int(largest),
        "largest_cc_frac": float(largest / max(1, M)),
        "isolated_deg0_n": int(isolate_n),
        "isolated_deg0_frac": float(isolate_n / max(1, M)),
    }

    if way_city is None:
        return out

    city = np.asarray(way_city, dtype=np.int64).reshape(-1)
    if city.size != M:
        return out

    def connectivity_induced(mask: np.ndarray) -> Dict[str, object]:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.size != M:
            return {}
        idxs = np.nonzero(mask)[0]
        n = int(idxs.size)
        if n <= 0:
            return {
                "n_connected_components_undirected": 0,
                "largest_cc_n": 0,
                "largest_cc_frac": float("nan"),
                "isolated_deg0_n": 0,
                "isolated_deg0_frac": float("nan"),
            }

        g2l = np.full((M,), -1, dtype=np.int64)
        g2l[idxs] = np.arange(n, dtype=np.int64)
        parent = np.arange(n, dtype=np.int64)
        size = np.ones((n,), dtype=np.int64)
        has_nbr = np.zeros((n,), dtype=bool)

        def find(x: int) -> int:
            xx = int(x)
            while parent[xx] != xx:
                parent[xx] = parent[parent[xx]]
                xx = int(parent[xx])
            return int(xx)

        def union(a: int, b: int) -> None:
            ra = find(int(a))
            rb = find(int(b))
            if ra == rb:
                return
            if int(size[ra]) < int(size[rb]):
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

        for ug in idxs.tolist():
            u = int(ug)
            lu = int(g2l[u])
            for vg in adj[u]:
                v = int(vg)
                if v < 0 or v >= M or (not bool(mask[v])):
                    continue
                lv = int(g2l[v])
                if lu == lv:
                    continue
                has_nbr[lu] = True
                has_nbr[lv] = True
                union(lu, lv)

        roots = np.asarray([find(i) for i in range(n)], dtype=np.int64)
        uniq, inv = np.unique(roots, return_inverse=True)
        comp_sizes = np.bincount(inv.astype(np.int64), minlength=int(uniq.size)).astype(np.int64)
        n_comp = int(comp_sizes.size)
        largest = int(comp_sizes.max()) if comp_sizes.size else 0
        isolate_n = int(np.sum(~has_nbr))
        return {
            "n_connected_components_undirected": int(n_comp),
            "largest_cc_n": int(largest),
            "largest_cc_frac": float(largest / max(1, n)),
            "isolated_deg0_n": int(isolate_n),
            "isolated_deg0_frac": float(isolate_n / max(1, n)),
        }

    per_city: Dict[str, object] = {}
    for c in sorted(set(int(x) for x in city.tolist() if int(x) >= 0)):
        mask = city == int(c)
        stats = connectivity_induced(mask)
        stats["n_ways"] = int(np.sum(mask))
        per_city[str(int(c))] = stats

    unknown_n = int(np.sum(city < 0))
    out["per_city"] = per_city
    out["unknown_city_n"] = int(unknown_n)
    out["unknown_city_frac"] = float(unknown_n / max(1, M))
    return out


def build_way_graph(
    *,
    way_routes_npz: Path,
    osm_pbfs: Sequence[Path],
    out_npz: Path,
    cfg: Cfg,
) -> Dict[str, object]:
    routes = np.load(str(way_routes_npz), allow_pickle=True)
    need = {"way_osm_id", "way_seq_ptr", "way_seq_idx", "way_seq_len", "route_city"}
    missing = sorted(list(need - set(routes.files)))
    if missing:
        raise SystemExit(f"[FATAL] way_routes_npz missing keys: {missing}")

    way_osm_id = np.asarray(routes["way_osm_id"], dtype=np.int64).reshape(-1)
    M = int(way_osm_id.size)
    way_to_idx = {int(w): int(i) for i, w in enumerate(way_osm_id.tolist())}
    wanted_way_ids = set(int(w) for w in way_osm_id.tolist())

    way_city, city_conflicts = _load_way_city_from_routes(routes, M)

    # Base adjacency
    adj: List[Set[int]] = [set() for _ in range(M)]
    n_edges_gt = 0
    if bool(cfg.include_gt_transitions):
        gt_adj = _gt_transition_edges(routes, M)
        for i in range(M):
            if gt_adj[i]:
                adj[i].update(gt_adj[i])
                n_edges_gt += int(len(gt_adj[i]))

    # OSM topology adjacency via shared nodes
    osmium = _import_osmium()
    node_to_ways: Dict[int, List[int]] = {}
    processed_way_idx: Set[int] = set()
    collector = _WayNodeCollector(
        osmium_mod=osmium,
        wanted_way_ids=wanted_way_ids,
        way_to_idx=way_to_idx,
        node_to_ways=node_to_ways,
        processed_way_idx=processed_way_idx,
        max_ways_per_node=int(cfg.max_ways_per_node),
    )
    for pbf in osm_pbfs:
        if not Path(pbf).exists():
            raise SystemExit(f"[FATAL] file not found: {pbf}")
        collector.apply(Path(pbf))

    # Add clique edges per shared node (bidirectional)
    n_nodes = int(len(node_to_ways))
    n_osm_pairs = 0
    for ways in node_to_ways.values():
        s = set(int(x) for x in ways)
        if len(s) < 2:
            continue
        # each way connects to all others at this node (bidirectional)
        for a in s:
            before = int(len(adj[a]))
            adj[a].update(s)
            after = int(len(adj[a]))
            n_osm_pairs += max(0, after - before)
    for i in range(M):
        adj[i].discard(i)

    # Build CSR
    out_ptr, out_idx = _csr_from_adj(adj)
    out_deg = (out_ptr[1:] - out_ptr[:-1]).astype(np.int64, copy=False)

    conn = _connectivity_stats_from_adj(adj, way_city)

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "build_way_graph_from_osm_pbf_topology",
        "inputs": {
            "way_routes_npz": str(way_routes_npz),
            "osm_pbfs": [str(Path(p)) for p in osm_pbfs],
        },
        "config": {
            "include_gt_transitions": bool(cfg.include_gt_transitions),
            "max_ways_per_node": int(cfg.max_ways_per_node),
        },
        "stats": {
            "n_ways": int(M),
            "n_osm_ways_found": int(len(processed_way_idx)),
            "osm_way_coverage_frac": float(len(processed_way_idx) / max(1, M)),
            "n_osm_nodes_touched": int(n_nodes),
            "n_edges_gt_directed": int(n_edges_gt),
            "n_edges_directed": int(out_idx.size),
            "out_deg": {"p50": _p(out_deg, 50), "p90": _p(out_deg, 90), "max": int(np.max(out_deg) if out_deg.size else 0)},
            "connectivity": conn,
            "route_city_conflicts": int(city_conflicts),
        },
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_npz),
        way_osm_id=way_osm_id,
        way_adj_ptr=out_ptr,
        way_adj_idx=out_idx,
        meta=meta,
    )
    return {"ok": True, "out_npz": str(out_npz), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build way_graph.npz from OSM topology (shared nodes) + GT transitions.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--osm_pbf", type=Path, action="append", required=True, help="Repeatable; one or more .pbf files covering the cities.")
    p.add_argument("--out_npz", type=Path, required=True)
    p.add_argument("--no_gt_transitions", action="store_true", help="If set, do NOT include GT transition edges from way_routes.")
    p.add_argument(
        "--max_ways_per_node",
        type=int,
        default=0,
        help="Safety cap to skip extremely high-degree nodes when building cliques (0=no cap).",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = build_way_graph(
        way_routes_npz=Path(args.way_routes_npz),
        osm_pbfs=[Path(p) for p in list(args.osm_pbf or [])],
        out_npz=Path(args.out_npz),
        cfg=Cfg(
            include_gt_transitions=(not bool(args.no_gt_transitions)),
            max_ways_per_node=int(args.max_ways_per_node),
        ),
    )
    meta = report["meta"]
    st = meta["stats"]
    conn = st["connectivity"]
    compact = {
        "ok": True,
        "out_npz": report["out_npz"],
        "n_ways": int(st["n_ways"]),
        "n_edges_directed": int(st["n_edges_directed"]),
        "osm_way_coverage_frac": float(st["osm_way_coverage_frac"]),
        "largest_cc_frac": float(conn.get("largest_cc_frac", float("nan"))),
        "isolated_deg0_frac": float(conn.get("isolated_deg0_frac", float("nan"))),
        "per_city": conn.get("per_city", None),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
