from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    seed: int
    directed_region_edges: bool
    resolution: float
    min_component_size: int
    keep_only_largest_cc: bool


def _import_louvain():
    try:
        import networkx as nx  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "[FATAL] missing dependency: networkx. Install in your conda env, e.g. `pip install networkx`."
        ) from e
    try:
        import community as community_louvain  # type: ignore
        # python-louvain exposes best_partition at top-level in many installs.
        if hasattr(community_louvain, "best_partition"):
            return nx, community_louvain
    except Exception:
        community_louvain = None
    try:
        from community import community_louvain as cl  # type: ignore

        return nx, cl
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "[FATAL] missing dependency: python-louvain. Install in your conda env, e.g. `pip install python-louvain`."
        ) from e


def _sym_edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def _csr_to_edge_weights(ptr: np.ndarray, idx: np.ndarray) -> Dict[Tuple[int, int], int]:
    """
    Build undirected edge weights by symmetrizing directed edges.
    Weight = count of directed edges contributing to the undirected pair.
    """
    ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    n = int(ptr.size) - 1
    out: Dict[Tuple[int, int], int] = {}
    for u in range(n):
        s = int(ptr[u])
        e = int(ptr[u + 1])
        for v in idx[s:e].tolist():
            vv = int(v)
            if vv < 0 or vv >= n:
                continue
            a, b = _sym_edge_key(int(u), int(vv))
            out[(a, b)] = int(out.get((a, b), 0) + 1)
    return out


def _union_find_components(n: int, edges: Iterable[Tuple[int, int]]) -> Tuple[int, np.ndarray]:
    parent = np.arange(int(n), dtype=np.int64)
    size = np.ones((int(n),), dtype=np.int64)

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

    for u, v in edges:
        uu = int(u)
        vv = int(v)
        if uu == vv:
            continue
        union(uu, vv)

    roots = np.asarray([find(i) for i in range(int(n))], dtype=np.int64)
    _, inv = np.unique(roots, return_inverse=True)
    comp_sizes = np.bincount(inv.astype(np.int64), minlength=int(inv.max()) + 1 if inv.size else 0).astype(np.int64)
    n_comp = int(comp_sizes.size)
    return n_comp, comp_sizes


def _remap_labels(labels: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Remap community labels to a compact range [0, n_regions).

    Important:
      - Negative labels (e.g. -1) are treated as "unassigned" and preserved as -1.
      - region_sizes only counts assigned regions (excludes -1).
    """
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    assigned = lab >= 0
    if not bool(np.any(assigned)):
        out = np.full_like(lab, -1, dtype=np.int64)
        return out, 0, np.zeros((0,), dtype=np.int64)

    uniq = np.unique(lab[assigned].astype(np.int64, copy=False))
    mapping = {int(old): int(i) for i, old in enumerate(uniq.tolist())}
    out = np.full_like(lab, -1, dtype=np.int64)
    for i, x in enumerate(lab.tolist()):
        xx = int(x)
        if xx >= 0:
            out[i] = int(mapping.get(xx, -1))
    counts = np.bincount(out[assigned].astype(np.int64, copy=False), minlength=int(uniq.size)).astype(np.int64, copy=False)
    return out.astype(np.int64, copy=False), int(uniq.size), counts


def _build_region_way_csr(way_region: np.ndarray, n_regions: int) -> Tuple[np.ndarray, np.ndarray]:
    r = np.asarray(way_region, dtype=np.int64).reshape(-1)
    n_regions = int(n_regions)
    ptr = np.zeros((n_regions + 1,), dtype=np.int64)
    valid = (r >= 0) & (r < n_regions)
    for rr in r[valid].tolist():
        ptr[int(rr) + 1] += 1
    ptr = np.cumsum(ptr, dtype=np.int64)
    idx = np.full((int(np.sum(valid)),), -1, dtype=np.int64)
    cur = ptr[:-1].copy()
    for way_id, rr in enumerate(r.tolist()):
        if 0 <= int(rr) < n_regions:
            j = int(cur[int(rr)])
            idx[j] = int(way_id)
            cur[int(rr)] += 1
    return ptr, idx


def _build_region_adj_csr(
    *,
    ptr: np.ndarray,
    idx: np.ndarray,
    way_region: np.ndarray,
    n_regions: int,
    directed: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build region->region adjacency from way-level directed edges.
    """
    ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    reg = np.asarray(way_region, dtype=np.int64).reshape(-1)
    n_way = int(reg.size)
    n_regions = int(n_regions)

    w: Dict[Tuple[int, int], int] = {}
    n = int(ptr.size) - 1
    for u in range(n):
        ru = int(reg[u]) if 0 <= u < n_way else -1
        if ru < 0:
            continue
        s = int(ptr[u])
        e = int(ptr[u + 1])
        for v in idx[s:e].tolist():
            vv = int(v)
            if vv < 0 or vv >= n_way:
                continue
            rv = int(reg[vv])
            if rv < 0 or rv == ru:
                continue
            a, b = (ru, rv) if directed else _sym_edge_key(ru, rv)
            w[(int(a), int(b))] = int(w.get((int(a), int(b)), 0) + 1)

    # CSR (directed or undirected-as-directed with symmetric key)
    rows: List[List[Tuple[int, int]]] = [[] for _ in range(n_regions)]
    for (a, b), ww in w.items():
        if 0 <= int(a) < n_regions and 0 <= int(b) < n_regions:
            rows[int(a)].append((int(b), int(ww)))
            if (not directed) and int(a) != int(b):
                rows[int(b)].append((int(a), int(ww)))

    out_ptr = np.zeros((n_regions + 1,), dtype=np.int64)
    for i in range(n_regions):
        out_ptr[i + 1] = out_ptr[i] + int(len(rows[i]))
    out_idx = np.full((int(out_ptr[-1]),), -1, dtype=np.int64)
    out_w = np.zeros((int(out_ptr[-1]),), dtype=np.int64)
    k = 0
    for i in range(n_regions):
        for j, ww in rows[i]:
            out_idx[k] = int(j)
            out_w[k] = int(ww)
            k += 1
    return out_ptr, out_idx, out_w


def main() -> None:
    ap = argparse.ArgumentParser(description="Build region graph via Louvain communities on way_graph.npz.")
    ap.add_argument("--way_graph_npz", type=Path, required=True)
    ap.add_argument("--out_npz", type=Path, required=True, help="Output region_graph.npz (mapping + region CSR).")
    ap.add_argument("--out_json", type=Path, default=None, help="Optional: save a small report json next to out_npz.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution (smaller => fewer/larger communities).")
    ap.add_argument(
        "--min_component_size",
        type=int,
        default=2,
        help="Ignore tiny connected components (< this size) during Louvain; their nodes are assigned region=-1.",
    )
    ap.add_argument(
        "--keep_only_largest_cc",
        action="store_true",
        help="If set, run Louvain only on the largest connected component; others assigned region=-1.",
    )
    ap.add_argument(
        "--directed_region_edges",
        action="store_true",
        help="If set, build directed region adjacency (ru->rv). Default: undirected (symmetrized).",
    )
    args = ap.parse_args()

    cfg = Cfg(
        seed=int(args.seed),
        directed_region_edges=bool(args.directed_region_edges),
        resolution=float(args.resolution),
        min_component_size=int(args.min_component_size),
        keep_only_largest_cc=bool(args.keep_only_largest_cc),
    )

    data = np.load(str(args.way_graph_npz), allow_pickle=True)
    need = {"way_osm_id", "way_adj_ptr", "way_adj_idx"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise SystemExit(f"[FATAL] way_graph.npz missing keys: {missing}")
    way_osm_id = np.asarray(data["way_osm_id"]).reshape(-1)
    ptr = np.asarray(data["way_adj_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(data["way_adj_idx"], dtype=np.int64).reshape(-1)
    n_ways = int(way_osm_id.size)

    nx, cl = _import_louvain()

    # Build undirected weighted graph for Louvain.
    ew = _csr_to_edge_weights(ptr=ptr, idx=idx)
    edges = [(int(u), int(v)) for (u, v) in ew.keys() if int(u) != int(v)]
    undeg = np.zeros((n_ways,), dtype=np.int64)
    for u, v in edges:
        undeg[int(u)] += 1
        undeg[int(v)] += 1
    n_comp, comp_sizes = _union_find_components(n_ways, edges)
    largest_cc = int(comp_sizes.max()) if comp_sizes.size else 0
    isolate_n = int(np.sum(undeg == 0))

    G = nx.Graph()
    G.add_nodes_from(range(n_ways))
    for (u, v), w in ew.items():
        if int(u) != int(v) and int(w) > 0:
            G.add_edge(int(u), int(v), weight=float(w))

    # Restrict nodes for Louvain (optional): remove tiny CCs and/or keep only largest CC.
    nodes_for_louvain = None
    if bool(cfg.keep_only_largest_cc) or int(cfg.min_component_size) > 1:
        # Use networkx CCs for exact node set (graph already built).
        comps = list(nx.connected_components(G))
        if comps:
            comps = sorted(comps, key=lambda s: len(s), reverse=True)
            if bool(cfg.keep_only_largest_cc):
                comps = [comps[0]]
            if int(cfg.min_component_size) > 1:
                comps = [c for c in comps if len(c) >= int(cfg.min_component_size)]
            nodes_for_louvain = set().union(*comps) if comps else set()

    if nodes_for_louvain is not None:
        H = G.subgraph(nodes_for_louvain).copy()
    else:
        H = G

    try:
        part = cl.best_partition(H, weight="weight", random_state=int(cfg.seed), resolution=float(cfg.resolution))
    except TypeError:
        # Older python-louvain may not accept `resolution`.
        part = cl.best_partition(H, weight="weight", random_state=int(cfg.seed))
    lab = np.full((n_ways,), -1, dtype=np.int64)
    for u, c in part.items():
        if 0 <= int(u) < n_ways:
            lab[int(u)] = int(c)

    way_region, n_regions, region_sizes = _remap_labels(lab)
    assigned_n = int(np.sum(way_region >= 0))
    assigned_frac = float(assigned_n / max(1, int(n_ways)))
    region_way_ptr, region_way_idx = _build_region_way_csr(way_region, n_regions)
    region_adj_ptr, region_adj_idx, region_adj_w = _build_region_adj_csr(
        ptr=ptr,
        idx=idx,
        way_region=way_region,
        n_regions=int(n_regions),
        directed=bool(cfg.directed_region_edges),
    )

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_npz),
        way_region=way_region.astype(np.int32, copy=False),
        region_sizes=region_sizes.astype(np.int32, copy=False),
        region_way_ptr=region_way_ptr.astype(np.int64, copy=False),
        region_way_idx=region_way_idx.astype(np.int64, copy=False),
        region_adj_ptr=region_adj_ptr.astype(np.int64, copy=False),
        region_adj_idx=region_adj_idx.astype(np.int64, copy=False),
        region_adj_w=region_adj_w.astype(np.int64, copy=False),
        meta={
            "task": "build_region_graph",
            "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
            "cfg": asdict(cfg),
            "inputs": {"way_graph_npz": str(args.way_graph_npz)},
            "n_ways": int(n_ways),
            "n_regions": int(n_regions),
            "assigned_n": int(assigned_n),
            "assigned_frac": float(assigned_frac),
            "graph": {
                "n_connected_components_undirected": int(n_comp),
                "largest_cc_n": int(largest_cc),
                "largest_cc_frac": float(largest_cc / max(1, int(n_ways))),
                "isolated_outdeg0_n": int(isolate_n),
                "isolated_outdeg0_frac": float(isolate_n / max(1, int(n_ways))),
                "louvain_nodes_n": int(H.number_of_nodes()),
                "louvain_nodes_frac": float(H.number_of_nodes() / max(1, int(n_ways))),
            },
        },
    )
    print(f"[OK] saved: {out_npz} (n_regions={int(n_regions)})")

    if args.out_json is not None:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        rep = {
            "ok": True,
            "task": "build_region_graph",
            "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
            "cfg": asdict(cfg),
            "inputs": {"way_graph_npz": str(args.way_graph_npz)},
            "outputs": {"out_npz": str(out_npz)},
            "n_ways": int(n_ways),
            "n_regions": int(n_regions),
            "assigned_n": int(assigned_n),
            "assigned_frac": float(assigned_frac),
            "graph": {
                "n_connected_components_undirected": int(n_comp),
                "largest_cc_n": int(largest_cc),
                "largest_cc_frac": float(largest_cc / max(1, int(n_ways))),
                "isolated_outdeg0_n": int(isolate_n),
                "isolated_outdeg0_frac": float(isolate_n / max(1, int(n_ways))),
                "louvain_nodes_n": int(H.number_of_nodes()),
                "louvain_nodes_frac": float(H.number_of_nodes() / max(1, int(n_ways))),
            },
            "region_size": {
                "p50": int(np.quantile(region_sizes, 0.50)) if region_sizes.size else 0,
                "p90": int(np.quantile(region_sizes, 0.90)) if region_sizes.size else 0,
                "max": int(np.max(region_sizes)) if region_sizes.size else 0,
            },
        }
        out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] saved: {out_json}")


if __name__ == "__main__":
    main()
