from __future__ import annotations

"""Corridor extraction via clustering on LCS similarity."""

from typing import Any, Dict, List
import warnings

import numpy as np

try:
    from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore
    from scipy.spatial.distance import squareform  # type: ignore
except Exception as e:  # pragma: no cover
    linkage = None  # type: ignore[assignment]
    fcluster = None  # type: ignore[assignment]
    squareform = None  # type: ignore[assignment]
    _SCIPY_ERR = e

from src.evaluation.corridor_metrics import compute_pairwise_similarity


def _uf_cluster_labels(sim: np.ndarray, *, threshold: float) -> np.ndarray:
    n = int(sim.shape[0])
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros((n,), dtype=np.int8)

    def find(x: int) -> int:
        xx = int(x)
        while parent[xx] != xx:
            parent[xx] = parent[int(parent[xx])]
            xx = int(parent[xx])
        return xx

    def union(a: int, b: int) -> None:
        ra = find(int(a))
        rb = find(int(b))
        if ra == rb:
            return
        if int(rank[ra]) < int(rank[rb]):
            parent[ra] = rb
        elif int(rank[ra]) > int(rank[rb]):
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] = np.int8(int(rank[ra]) + 1)

    thr = float(threshold)
    for i in range(n):
        for j in range(i + 1, n):
            if float(sim[i, j]) >= thr:
                union(i, j)

    roots = np.asarray([find(i) for i in range(n)], dtype=np.int64)
    uniq = {}
    labels = np.empty((n,), dtype=np.int64)
    next_lab = 0
    for i, r in enumerate(roots.tolist()):
        rr = int(r)
        if rr not in uniq:
            uniq[rr] = next_lab
            next_lab += 1
        labels[i] = int(uniq[rr])
    return labels


def extract_corridors(routes: List[np.ndarray], threshold: float = 0.5, method: str = "average") -> Dict[str, Any]:
    """
    从一组routes中提取corridors。

    Args:
        routes: List of way sequences (each is (Li,) array)
        threshold: LCS similarity threshold for clustering
        method: clustering method:
          - "average"/"complete"/"single": hierarchical clustering (requires scipy)
          - "graph": threshold graph connected-components (no scipy, ~single-link)

    Returns:
        {
            "n_corridors": int,
            "labels": (N,) int array, corridor label for each route
            "corridor_sizes": List[int], number of routes in each corridor
            "corridor_prototypes": List[int], index of prototype route for each corridor
            "method_used": str,
        }
    """
    if len(routes) == 0:
        return {
            "n_corridors": 0,
            "labels": np.asarray([], dtype=np.int64),
            "corridor_sizes": [],
            "corridor_prototypes": [],
            "method_used": str(method),
        }
    if len(routes) == 1:
        return {
            "n_corridors": 1,
            "labels": np.asarray([0], dtype=np.int64),
            "corridor_sizes": [1],
            "corridor_prototypes": [0],
            "method_used": str(method),
        }

    sim_matrix = compute_pairwise_similarity(routes)
    method_req = str(method or "").strip().lower()
    method_used = method_req

    if method_req == "graph":
        labels = _uf_cluster_labels(sim_matrix, threshold=float(threshold))
    elif method_req in ("average", "complete", "single"):
        if linkage is None or fcluster is None or squareform is None:  # pragma: no cover
            warnings.warn(
                f"scipy is missing, fallback to method='graph'. Error: {_SCIPY_ERR}",
                RuntimeWarning,
                stacklevel=2,
            )
            method_used = "graph"
            labels = _uf_cluster_labels(sim_matrix, threshold=float(threshold))
        else:
            dist_matrix = 1.0 - sim_matrix.astype(np.float64, copy=False)
            np.fill_diagonal(dist_matrix, 0.0)
            condensed = squareform(dist_matrix, checks=False)
            Z = linkage(condensed, method=method_req)
            labels = fcluster(Z, t=1.0 - float(threshold), criterion="distance").astype(np.int64, copy=False)
            labels = labels - 1  # 1-indexed -> 0-indexed
    else:
        raise ValueError(f"Unknown clustering method: {method!r} (use average/complete/single/graph)")

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    n_corridors = int(np.max(labels)) + 1 if labels.size else 0
    corridor_sizes = [int(np.sum(labels == c)) for c in range(n_corridors)]

    corridor_prototypes: List[int] = []
    for c in range(n_corridors):
        members = np.where(labels == c)[0]
        if int(members.size) == 1:
            corridor_prototypes.append(int(members[0]))
            continue
        # 平均相似度最高的member作为prototype
        best_m = int(members[0])
        best_avg = -1.0
        for m in members.tolist():
            mm = int(m)
            # Exclude self to avoid trivially favoring longer clusters.
            sims = sim_matrix[mm, members].astype(np.float64, copy=False)
            if sims.size <= 1:
                avg = 1.0
            else:
                avg = float((np.sum(sims) - float(sim_matrix[mm, mm])) / float(sims.size - 1))
            if avg > best_avg:
                best_avg = avg
                best_m = mm
        corridor_prototypes.append(int(best_m))

    return {
        "n_corridors": int(n_corridors),
        "labels": labels,
        "corridor_sizes": corridor_sizes,
        "corridor_prototypes": corridor_prototypes,
        "method_used": str(method_used),
    }

