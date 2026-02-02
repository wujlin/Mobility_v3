from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _p(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def _connectivity(ptr: np.ndarray, idx: np.ndarray) -> Dict[str, object]:
    """
    Undirected connectivity stats by symmetrizing directed CSR edges.
    """
    ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    n = int(ptr.size) - 1
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

    for u in range(n):
        s = int(ptr[u])
        e = int(ptr[u + 1])
        for v in idx[s:e].tolist():
            vv = int(v)
            if vv < 0 or vv >= n or vv == u:
                continue
            has_nbr[u] = True
            has_nbr[vv] = True
            union(u, vv)

    roots = np.asarray([find(i) for i in range(n)], dtype=np.int64)
    uniq, inv = np.unique(roots, return_inverse=True)
    comp_sizes = np.bincount(inv.astype(np.int64), minlength=int(uniq.size)).astype(np.int64)
    n_comp = int(comp_sizes.size)
    largest = int(comp_sizes.max()) if comp_sizes.size else 0
    isolate_n = int(np.sum(~has_nbr))
    return {
        "n_connected_components_undirected": int(n_comp),
        "largest_cc_n": int(largest),
        "largest_cc_frac": float(largest / max(1, int(n))),
        "isolated_deg0_n": int(isolate_n),
        "isolated_deg0_frac": float(isolate_n / max(1, int(n))),
    }


def _connectivity_induced(ptr: np.ndarray, idx: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
    """
    Undirected connectivity stats on an induced subgraph (mask over nodes).
    """
    ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    n_total = int(ptr.size) - 1
    if mask.size != n_total:
        return {}
    nodes = np.nonzero(mask)[0]
    n = int(nodes.size)
    if n <= 0:
        return {
            "n_connected_components_undirected": 0,
            "largest_cc_n": 0,
            "largest_cc_frac": float("nan"),
            "isolated_deg0_n": 0,
            "isolated_deg0_frac": float("nan"),
        }

    g2l = np.full((n_total,), -1, dtype=np.int64)
    g2l[nodes] = np.arange(n, dtype=np.int64)
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

    for ug in nodes.tolist():
        u = int(ug)
        lu = int(g2l[u])
        s = int(ptr[u])
        e = int(ptr[u + 1])
        for v in idx[s:e].tolist():
            vv = int(v)
            if vv < 0 or vv >= n_total or vv == u or (not bool(mask[vv])):
                continue
            lv = int(g2l[vv])
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


def _infer_way_city_from_routes(*, way_routes_npz: Path, way_osm_id_graph: np.ndarray) -> Dict[str, object]:
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
    return {"way_city": way_city, "conflicts": int(conflicts)}


def audit(*, way_graph_npz: Path, way_routes_npz: Optional[Path] = None) -> Dict[str, object]:
    if not Path(way_graph_npz).exists():
        raise SystemExit(f"[FATAL] file not found: {way_graph_npz}")
    data = np.load(str(way_graph_npz), allow_pickle=True)
    need = {"way_osm_id", "way_adj_ptr", "way_adj_idx"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"way_graph.npz missing keys: {missing}")
    way_osm_id = np.asarray(data["way_osm_id"]).reshape(-1)
    ptr = np.asarray(data["way_adj_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(data["way_adj_idx"], dtype=np.int64).reshape(-1)
    M = int(way_osm_id.shape[0])
    deg = (ptr[1:] - ptr[:-1]).astype(np.int64, copy=False)
    conn = _connectivity(ptr, idx)
    out: Dict[str, object] = {
        "ok": True,
        "way_graph_npz": str(way_graph_npz),
        "n_ways": int(M),
        "n_edges_directed": int(idx.size),
        "out_deg": {"p50": _p(deg, 50), "p90": _p(deg, 90), "max": int(np.max(deg) if deg.size else 0)},
        "connectivity": conn,
    }
    if way_routes_npz is not None:
        inf = _infer_way_city_from_routes(way_routes_npz=Path(way_routes_npz), way_osm_id_graph=way_osm_id.astype(np.int64))
        way_city = np.asarray(inf["way_city"], dtype=np.int64).reshape(-1)
        per_city: Dict[str, object] = {}
        for c in sorted(set(int(x) for x in way_city.tolist() if int(x) >= 0)):
            mask = way_city == int(c)
            stats = _connectivity_induced(ptr, idx, mask)
            stats["n_ways"] = int(np.sum(mask))
            per_city[str(int(c))] = stats
        out["per_city"] = per_city
        out["unknown_city_n"] = int(np.sum(way_city < 0))
        out["unknown_city_frac"] = float(out["unknown_city_n"] / max(1, int(M)))
        out["route_city_conflicts"] = int(inf["conflicts"])
        out["way_routes_npz"] = str(way_routes_npz)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit way_graph.npz (compact).")
    ap.add_argument("--way_graph_npz", type=Path, required=True)
    ap.add_argument("--way_routes_npz", type=Path, default=None, help="Optional; used to infer per-city connectivity.")
    args = ap.parse_args()
    report = audit(way_graph_npz=Path(args.way_graph_npz), way_routes_npz=(Path(args.way_routes_npz) if args.way_routes_npz else None))
    deg = report["out_deg"]
    conn = report.get("connectivity", {})
    print(f"[way_graph] {report['way_graph_npz']}")
    print(f"[ways] {report['n_ways']} edges_directed={report['n_edges_directed']}")
    print(f"[out_deg] p50={deg['p50']:.1f} p90={deg['p90']:.1f} max={int(deg['max'])}")
    if isinstance(conn, dict):
        print(
            "[connectivity] "
            f"n_cc={int(conn.get('n_connected_components_undirected', -1))} "
            f"largest_cc={float(conn.get('largest_cc_frac', float('nan'))):.1%} "
            f"isolated={float(conn.get('isolated_deg0_frac', float('nan'))):.1%}"
        )
    per_city = report.get("per_city", None)
    if isinstance(per_city, dict) and per_city:
        unknown = float(report.get("unknown_city_frac", float("nan")))
        conflicts = int(report.get("route_city_conflicts", 0))
        print(f"[per_city] unknown_frac={unknown:.1%} conflicts={conflicts}")
        for k in sorted(per_city.keys()):
            st = per_city[k]
            if not isinstance(st, dict):
                continue
            print(
                f"  [city {k}] n_ways={int(st.get('n_ways', -1))} "
                f"largest_cc={float(st.get('largest_cc_frac', float('nan'))):.1%} "
                f"isolated={float(st.get('isolated_deg0_frac', float('nan'))):.1%} "
                f"n_cc={int(st.get('n_connected_components_undirected', -1))}"
            )


if __name__ == "__main__":
    main()
