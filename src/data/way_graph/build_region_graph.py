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


def _remap_labels(labels: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    uniq = np.unique(lab)
    uniq = uniq[np.isfinite(uniq)]
    mapping = {int(old): int(i) for i, old in enumerate(uniq.tolist())}
    out = np.full_like(lab, -1)
    for i, x in enumerate(lab.tolist()):
        out[i] = int(mapping.get(int(x), -1))
    counts = np.zeros((len(uniq),), dtype=np.int64)
    for x in out.tolist():
        if int(x) >= 0:
            counts[int(x)] += 1
    return out.astype(np.int64, copy=False), int(len(uniq)), counts.astype(np.int64, copy=False)


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
    ap.add_argument(
        "--directed_region_edges",
        action="store_true",
        help="If set, build directed region adjacency (ru->rv). Default: undirected (symmetrized).",
    )
    args = ap.parse_args()

    cfg = Cfg(seed=int(args.seed), directed_region_edges=bool(args.directed_region_edges))

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
    G = nx.Graph()
    G.add_nodes_from(range(n_ways))
    for (u, v), w in ew.items():
        if int(u) != int(v) and int(w) > 0:
            G.add_edge(int(u), int(v), weight=float(w))

    part = cl.best_partition(G, weight="weight", random_state=int(cfg.seed))
    lab = np.full((n_ways,), -1, dtype=np.int64)
    for u, c in part.items():
        if 0 <= int(u) < n_ways:
            lab[int(u)] = int(c)

    way_region, n_regions, region_sizes = _remap_labels(lab)
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

