from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

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


def audit(*, way_graph_npz: Path) -> Dict[str, object]:
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
    return {
        "ok": True,
        "way_graph_npz": str(way_graph_npz),
        "n_ways": int(M),
        "n_edges_directed": int(idx.size),
        "out_deg": {"p50": _p(deg, 50), "p90": _p(deg, 90), "max": int(np.max(deg) if deg.size else 0)},
        "connectivity": conn,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit way_graph.npz (compact).")
    ap.add_argument("--way_graph_npz", type=Path, required=True)
    args = ap.parse_args()
    report = audit(way_graph_npz=Path(args.way_graph_npz))
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


if __name__ == "__main__":
    main()
