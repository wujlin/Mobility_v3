from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.way_graph.build_region_graph import (
    _build_region_adj_csr,
    _build_region_way_csr,
    _csr_to_edge_weights,
    _import_louvain,
    _remap_labels,
)

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    seed: int
    resolution: float
    min_component_size: int
    keep_only_largest_cc: bool
    directed_region_edges: bool
    fill_unknown_by_neighbor: bool


def _q_int(x: np.ndarray, q: float) -> int:
    a = np.asarray(x, dtype=np.int64).reshape(-1)
    if a.size == 0:
        return 0
    return int(np.quantile(a, float(q)))


def _infer_way_city_from_routes(*, way_routes_npz: Path, way_osm_id_graph: np.ndarray) -> Tuple[np.ndarray, int]:
    routes = np.load(str(way_routes_npz), allow_pickle=True)
    need = {"way_osm_id", "way_seq_ptr", "way_seq_idx", "way_seq_len", "route_city"}
    missing = sorted(list(need - set(routes.files)))
    if missing:
        raise SystemExit(f"[FATAL] way_routes_npz missing keys: {missing}")

    way_osm_id_routes = np.asarray(routes["way_osm_id"], dtype=np.int64).reshape(-1)
    ptr = np.asarray(routes["way_seq_ptr"], dtype=np.int64).reshape(-1)
    seq_idx = np.asarray(routes["way_seq_idx"], dtype=np.int64).reshape(-1)
    lens = np.asarray(routes["way_seq_len"], dtype=np.int64).reshape(-1)
    route_city = np.asarray(routes["route_city"], dtype=np.int64).reshape(-1)

    M = int(way_osm_id_graph.size)
    if way_osm_id_routes.size == way_osm_id_graph.size and np.array_equal(way_osm_id_routes, way_osm_id_graph):
        map_routes_to_graph = None  # identity
    else:
        osm_to_graph = {int(w): int(i) for i, w in enumerate(way_osm_id_graph.tolist())}
        map_routes_to_graph = np.full((int(way_osm_id_routes.size),), -1, dtype=np.int64)
        for i, wid in enumerate(way_osm_id_routes.tolist()):
            gi = osm_to_graph.get(int(wid))
            if gi is not None:
                map_routes_to_graph[int(i)] = np.int64(gi)

    way_city = np.full((M,), -1, dtype=np.int64)
    conflicts = 0
    N = int(lens.size)
    for r in range(N):
        L = int(lens[r])
        if L <= 0:
            continue
        s = int(ptr[r])
        e = s + L
        if e > int(seq_idx.size):
            continue
        c = int(route_city[r])
        seq = np.asarray(seq_idx[s:e], dtype=np.int64)
        if map_routes_to_graph is not None:
            seq = map_routes_to_graph[seq]
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


def _neighbor_fill_unknown_regions(*, ptr: np.ndarray, idx: np.ndarray, way_region: np.ndarray) -> int:
    """
    Assign region for way_region==-1 by majority vote among neighbors with known region.
    Uses undirected neighbor set (out + in) by scanning CSR both directions (cheap for our sizes).
    """
    ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    reg = np.asarray(way_region, dtype=np.int64).reshape(-1)
    n = int(reg.size)

    # Build incoming adjacency lists (to approximate undirected neighbors).
    in_rows: List[List[int]] = [[] for _ in range(n)]
    for u in range(n):
        s = int(ptr[u])
        e = int(ptr[u + 1])
        for v in idx[s:e].tolist():
            vv = int(v)
            if 0 <= vv < n and vv != u:
                in_rows[vv].append(int(u))

    filled = 0
    for u in range(n):
        if int(reg[u]) != -1:
            continue
        votes: Dict[int, int] = {}
        # out neighbors
        s = int(ptr[u])
        e = int(ptr[u + 1])
        for v in idx[s:e].tolist():
            vv = int(v)
            if 0 <= vv < n:
                rv = int(reg[vv])
                if rv >= 0:
                    votes[rv] = int(votes.get(rv, 0) + 1)
        # in neighbors
        for v in in_rows[u]:
            rv = int(reg[int(v)])
            if rv >= 0:
                votes[rv] = int(votes.get(rv, 0) + 1)
        if not votes:
            continue
        best_r = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        reg[u] = int(best_r)
        filled += 1

    way_region[:] = reg.astype(np.int64, copy=False)
    return int(filled)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Louvain per-city and merge into a single way_regions.npz.")
    ap.add_argument("--way_graph_npz", type=Path, required=True)
    ap.add_argument("--way_routes_npz", type=Path, required=True, help="Used to infer per-way city id from GT routes.")
    ap.add_argument("--out_npz", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=float, default=0.2)
    ap.add_argument("--min_component_size", type=int, default=2)
    ap.add_argument("--keep_only_largest_cc", action="store_true")
    ap.add_argument("--directed_region_edges", action="store_true")
    ap.add_argument(
        "--fill_unknown_by_neighbor",
        action="store_true",
        help="If set, assign region ids to unknown ways (region=-1) by neighbor majority vote (no new regions).",
    )
    args = ap.parse_args()

    cfg = Cfg(
        seed=int(args.seed),
        resolution=float(args.resolution),
        min_component_size=int(args.min_component_size),
        keep_only_largest_cc=bool(args.keep_only_largest_cc),
        directed_region_edges=bool(args.directed_region_edges),
        fill_unknown_by_neighbor=bool(args.fill_unknown_by_neighbor),
    )

    data = np.load(str(args.way_graph_npz), allow_pickle=True)
    need = {"way_osm_id", "way_adj_ptr", "way_adj_idx"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise SystemExit(f"[FATAL] way_graph.npz missing keys: {missing}")
    way_osm_id = np.asarray(data["way_osm_id"], dtype=np.int64).reshape(-1)
    ptr = np.asarray(data["way_adj_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(data["way_adj_idx"], dtype=np.int64).reshape(-1)
    n_ways = int(way_osm_id.size)

    way_city, conflicts = _infer_way_city_from_routes(way_routes_npz=Path(args.way_routes_npz), way_osm_id_graph=way_osm_id)
    city_ids = sorted(set(int(x) for x in way_city.tolist() if int(x) >= 0))
    unknown_n = int(np.sum(way_city < 0))

    nx, cl = _import_louvain()

    # Build undirected weighted graph once, then subgraph by city.
    ew = _csr_to_edge_weights(ptr=ptr, idx=idx)
    G = nx.Graph()
    G.add_nodes_from(range(n_ways))
    for (u, v), w in ew.items():
        if int(u) != int(v) and int(w) > 0:
            G.add_edge(int(u), int(v), weight=float(w))

    global_way_region = np.full((n_ways,), -1, dtype=np.int64)
    region_sizes_all: List[np.ndarray] = []
    per_city: Dict[str, Any] = {}
    offset = 0

    for city in city_ids:
        nodes = np.nonzero(way_city == int(city))[0].astype(np.int64, copy=False)
        H0 = G.subgraph(nodes.tolist()).copy()

        comps = list(nx.connected_components(H0))
        comps = sorted(comps, key=lambda s: len(s), reverse=True)
        if bool(cfg.keep_only_largest_cc):
            base_nodes = set(comps[0]) if comps else set()
        else:
            base_nodes = set()
            for c in comps:
                if len(c) >= int(cfg.min_component_size):
                    base_nodes |= set(c)
        H = H0.subgraph(base_nodes).copy() if base_nodes else H0

        try:
            part = cl.best_partition(H, weight="weight", random_state=int(cfg.seed), resolution=float(cfg.resolution))
        except TypeError:
            part = cl.best_partition(H, weight="weight", random_state=int(cfg.seed))

        lab = np.full((n_ways,), -1, dtype=np.int64)
        for u, c in part.items():
            uu = int(u)
            if 0 <= uu < n_ways:
                lab[uu] = int(c)

        local_way_region, n_regions, region_sizes = _remap_labels(lab)
        # Shift to global region id space.
        assigned = local_way_region >= 0
        global_way_region[assigned] = (local_way_region[assigned] + int(offset)).astype(np.int64, copy=False)

        region_sizes_all.append(region_sizes.astype(np.int64, copy=False))
        per_city[str(int(city))] = {
            "n_ways": int(nodes.size),
            "n_connected_components_undirected": int(len(comps)),
            "largest_cc_n": int(len(comps[0])) if comps else 0,
            "largest_cc_frac": float((len(comps[0]) if comps else 0) / max(1, int(nodes.size))),
            "n_regions": int(n_regions),
            "region_size": {
                "p50": _q_int(region_sizes, 0.50),
                "p90": _q_int(region_sizes, 0.90),
                "p95": _q_int(region_sizes, 0.95),
                "max": int(np.max(region_sizes)) if region_sizes.size else 0,
            },
            "assigned_frac": float(np.mean(local_way_region[nodes] >= 0)) if nodes.size else 0.0,
            "region_id_offset": int(offset),
        }
        offset += int(n_regions)

        print(
            f"[city={city}] n_regions={int(n_regions)} "
            f"p50={per_city[str(int(city))]['region_size']['p50']} "
            f"p90={per_city[str(int(city))]['region_size']['p90']} "
            f"assigned={per_city[str(int(city))]['assigned_frac']:.1%}"
        )

    n_regions_total = int(offset)
    region_sizes_cat = np.concatenate(region_sizes_all, axis=0) if region_sizes_all else np.zeros((0,), dtype=np.int64)

    filled_unknown = 0
    if bool(cfg.fill_unknown_by_neighbor) and int(unknown_n) > 0:
        filled_unknown = _neighbor_fill_unknown_regions(ptr=ptr, idx=idx, way_region=global_way_region)

    region_way_ptr, region_way_idx = _build_region_way_csr(global_way_region, n_regions_total)
    region_adj_ptr, region_adj_idx, region_adj_w = _build_region_adj_csr(
        ptr=ptr,
        idx=idx,
        way_region=global_way_region,
        n_regions=int(n_regions_total),
        directed=bool(cfg.directed_region_edges),
    )

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "task": "build_way_regions_louvain_per_city",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {"way_graph_npz": str(args.way_graph_npz), "way_routes_npz": str(args.way_routes_npz)},
        "n_ways": int(n_ways),
        "unknown_city_n": int(unknown_n),
        "unknown_city_frac": float(unknown_n / max(1, int(n_ways))),
        "route_city_conflicts": int(conflicts),
        "n_regions": int(n_regions_total),
        "filled_unknown_by_neighbor_n": int(filled_unknown),
        "per_city": per_city,
    }
    np.savez_compressed(
        str(out_npz),
        way_region=global_way_region.astype(np.int32, copy=False),
        region_sizes=region_sizes_cat.astype(np.int32, copy=False),
        region_way_ptr=region_way_ptr.astype(np.int64, copy=False),
        region_way_idx=region_way_idx.astype(np.int64, copy=False),
        region_adj_ptr=region_adj_ptr.astype(np.int64, copy=False),
        region_adj_idx=region_adj_idx.astype(np.int64, copy=False),
        region_adj_w=region_adj_w.astype(np.int64, copy=False),
        meta=meta,
    )
    print(f"[OK] saved: {out_npz} (n_regions_total={int(n_regions_total)})")

    if args.out_json is not None:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        rep: Dict[str, Any] = {
            "ok": True,
            "task": meta["task"],
            "created_at": meta["created_at"],
            "cfg": asdict(cfg),
            "inputs": meta["inputs"],
            "n_ways": int(n_ways),
            "unknown_city_n": int(unknown_n),
            "unknown_city_frac": float(meta["unknown_city_frac"]),
            "route_city_conflicts": int(conflicts),
            "filled_unknown_by_neighbor_n": int(filled_unknown),
            "n_regions": int(n_regions_total),
            "per_city": per_city,
        }
        out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] saved: {out_json}")


if __name__ == "__main__":
    main()

