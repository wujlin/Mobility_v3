from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception as e:  # pragma: no cover
    cKDTree = None  # type: ignore[assignment]
    _KD_ERR = e

from src.utils.geo_grid import BBox, GridSpec


TZ_SHANGHAI = timezone(timedelta(hours=8))


def _load_grid_from_graph_meta(meta: dict) -> GridSpec:
    g = meta.get("grid", {})
    bbox = g.get("bbox", {})
    return GridSpec(
        H=int(g["H"]),
        W=int(g["W"]),
        bbox=BBox(
            min_lon=float(bbox["min_lon"]),
            min_lat=float(bbox["min_lat"]),
            max_lon=float(bbox["max_lon"]),
            max_lat=float(bbox["max_lat"]),
        ),
    )


def _bresenham(y0: int, x0: int, y1: int, x1: int) -> Iterable[Tuple[int, int]]:
    dy = abs(int(y1) - int(y0))
    dx = abs(int(x1) - int(x0))
    sy = 1 if int(y0) < int(y1) else -1
    sx = 1 if int(x0) < int(x1) else -1
    err = dx - dy
    y, x = int(y0), int(x0)
    while True:
        yield y, x
        if y == int(y1) and x == int(x1):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _poly_cells(points_yx: np.ndarray, *, H: int, W: int) -> np.ndarray:
    """
    points_yx: (T,2) float grid positions.
    Return sorted unique cell ids (y*W + x).
    """
    pts = np.asarray(points_yx, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return np.zeros((0,), dtype=np.int64)
    y = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, int(H) - 1)
    x = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, int(W) - 1)
    ids = y * int(W) + x
    ids = np.unique(ids)
    return np.sort(ids.astype(np.int64, copy=False))


def _path_cells(nodes_yx: np.ndarray, *, H: int, W: int) -> np.ndarray:
    """
    nodes_yx: (M,2) float grid node coords.
    Rasterize node-to-node segments with Bresenham to approximate road-following cells.
    """
    nodes = np.asarray(nodes_yx, dtype=np.float32).reshape(-1, 2)
    if nodes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if nodes.shape[0] == 1:
        return _poly_cells(nodes, H=H, W=W)
    cells: List[int] = []
    for i in range(nodes.shape[0] - 1):
        y0, x0 = nodes[i]
        y1, x1 = nodes[i + 1]
        for yy, xx in _bresenham(int(round(float(y0))), int(round(float(x0))), int(round(float(y1))), int(round(float(x1)))):
            if 0 <= yy < int(H) and 0 <= xx < int(W):
                cells.append(int(yy) * int(W) + int(xx))
    if not cells:
        return np.zeros((0,), dtype=np.int64)
    ids = np.unique(np.asarray(cells, dtype=np.int64))
    return np.sort(ids.astype(np.int64, copy=False))


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.int64).reshape(-1)
    b = np.asarray(b, dtype=np.int64).reshape(-1)
    if a.size == 0 and b.size == 0:
        return 1.0
    if a.size == 0 or b.size == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True)
    denom = int(a.size + b.size - inter.size)
    return float(inter.size) / float(max(1, denom))


@dataclass(frozen=True)
class Graph:
    node_y: np.ndarray  # (N,)
    node_x: np.ndarray  # (N,)
    adj: List[List[Tuple[int, float]]]  # u -> [(v, w_m), ...]
    grid: GridSpec
    y_m: np.ndarray  # (N,)
    x_m: np.ndarray  # (N,)
    edge_cost: Dict[Tuple[int, int], float]


def _load_graph_npz(path: Path) -> Graph:
    data = np.load(str(path), allow_pickle=True)
    need = {"node_y", "node_x", "edge_u", "edge_v", "edge_w_m", "meta"}
    if not need.issubset(set(data.files)):
        raise ValueError(f"road_graph.npz missing keys: {sorted(list(need - set(data.files)))}")
    meta = data["meta"].item() if isinstance(data["meta"], np.ndarray) and data["meta"].shape == () else data["meta"]
    if not isinstance(meta, dict):
        raise ValueError("road_graph.npz meta must be a dict.")
    grid = _load_grid_from_graph_meta(meta)
    node_y = np.asarray(data["node_y"], dtype=np.float32).reshape(-1)
    node_x = np.asarray(data["node_x"], dtype=np.float32).reshape(-1)
    n = int(node_y.shape[0])
    eu = np.asarray(data["edge_u"], dtype=np.int32).reshape(-1)
    ev = np.asarray(data["edge_v"], dtype=np.int32).reshape(-1)
    ew = np.asarray(data["edge_w_m"], dtype=np.float32).reshape(-1)
    if eu.shape[0] != ev.shape[0] or eu.shape[0] != ew.shape[0]:
        raise ValueError("edge_u/edge_v/edge_w_m length mismatch.")

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    edge_cost: Dict[Tuple[int, int], float] = {}
    for u, v, w in zip(eu.tolist(), ev.tolist(), ew.tolist()):
        uu = int(u)
        vv = int(v)
        ww = float(w)
        if 0 <= uu < n and 0 <= vv < n and math.isfinite(ww) and ww > 0:
            adj[uu].append((vv, ww))
            edge_cost[(uu, vv)] = ww

    res_y_m, res_x_m = grid.resolution_m()
    y_m = node_y.astype(np.float64) * float(res_y_m)
    x_m = node_x.astype(np.float64) * float(res_x_m)
    return Graph(
        node_y=node_y.astype(np.float32, copy=False),
        node_x=node_x.astype(np.float32, copy=False),
        adj=adj,
        grid=grid,
        y_m=y_m.astype(np.float64, copy=False),
        x_m=x_m.astype(np.float64, copy=False),
        edge_cost=edge_cost,
    )


def _astar(
    g: Graph,
    *,
    start: int,
    goal: int,
    disabled_edges: Optional[Set[Tuple[int, int]]] = None,
    banned_nodes: Optional[Set[int]] = None,
) -> Tuple[float, List[int]]:
    n = int(g.node_y.shape[0])
    if not (0 <= int(start) < n and 0 <= int(goal) < n):
        return float("inf"), []
    if banned_nodes and int(start) in banned_nodes:
        return float("inf"), []
    if banned_nodes and int(goal) in banned_nodes:
        # Goal must remain reachable.
        return float("inf"), []

    def h(u: int) -> float:
        dy = float(g.y_m[int(u)] - g.y_m[int(goal)])
        dx = float(g.x_m[int(u)] - g.x_m[int(goal)])
        return float(math.hypot(dy, dx))

    gscore = np.full((n,), np.inf, dtype=np.float64)
    parent = np.full((n,), -1, dtype=np.int32)
    gscore[int(start)] = 0.0
    heap: List[Tuple[float, int]] = [(h(int(start)), int(start))]
    visited = np.zeros((n,), dtype=np.uint8)

    dis = disabled_edges or set()
    ban = banned_nodes or set()

    while heap:
        f_u, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = 1
        if u == int(goal):
            break
        if u in ban:
            continue
        gu = float(gscore[u])
        if not math.isfinite(gu):
            continue
        for v, w in g.adj[u]:
            vv = int(v)
            if vv in ban:
                continue
            if (u, vv) in dis:
                continue
            alt = gu + float(w)
            if alt < float(gscore[vv]):
                gscore[vv] = alt
                parent[vv] = int(u)
                heapq.heappush(heap, (alt + h(vv), vv))

    cost = float(gscore[int(goal)])
    if not math.isfinite(cost):
        return float("inf"), []
    # Reconstruct.
    path = []
    cur = int(goal)
    while cur != -1:
        path.append(cur)
        if cur == int(start):
            break
        cur = int(parent[cur])
    if not path or path[-1] != int(start):
        return float("inf"), []
    path.reverse()
    return cost, path


def _path_cost(edge_cost: Dict[Tuple[int, int], float], path: Sequence[int]) -> float:
    if len(path) < 2:
        return 0.0
    c = 0.0
    for a, b in zip(path[:-1], path[1:]):
        w = edge_cost.get((int(a), int(b)))
        if w is None:
            return float("inf")
        c += float(w)
    return float(c)


def k_shortest_paths_yen(g: Graph, *, start: int, goal: int, K: int) -> List[List[int]]:
    """
    Yen's algorithm (K shortest loopless paths).
    KISS constraints:
      - K is small (<=10).
      - Uses A* as the shortest-path oracle.
    """
    K = int(K)
    if K <= 0:
        return []

    cost0, p0 = _astar(g, start=int(start), goal=int(goal))
    if not p0:
        return []
    A: List[List[int]] = [p0]
    A_set: Set[Tuple[int, ...]] = {tuple(p0)}

    # Candidate paths heap: (total_cost, path_tuple)
    B: List[Tuple[float, Tuple[int, ...]]] = []

    for k in range(1, K):
        prev = A[-1]
        for i in range(len(prev) - 1):
            spur = int(prev[i])
            root = prev[: i + 1]

            disabled_edges: Set[Tuple[int, int]] = set()
            for p in A:
                if len(p) > i and p[: i + 1] == root:
                    disabled_edges.add((int(p[i]), int(p[i + 1])))

            banned_nodes: Set[int] = set(int(x) for x in root[:-1])
            spur_cost, spur_path = _astar(g, start=spur, goal=int(goal), disabled_edges=disabled_edges, banned_nodes=banned_nodes)
            if not spur_path:
                continue

            total = root[:-1] + spur_path
            t = tuple(int(x) for x in total)
            if t in A_set:
                continue
            total_cost = _path_cost(g.edge_cost, total)
            if not math.isfinite(total_cost):
                # Fallback: root cost + spur cost (approx).
                total_cost = _path_cost(g.edge_cost, root[:-1] + [spur]) + float(spur_cost)
            heapq.heappush(B, (float(total_cost), t))

        # Pick next.
        next_path = None
        while B:
            _, cand = heapq.heappop(B)
            if cand not in A_set:
                next_path = list(cand)
                break
        if next_path is None:
            break
        A.append(next_path)
        A_set.add(tuple(next_path))

    return A


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gate: build top-K candidate graph paths per OD and measure coverage/diversity against GT routes.")
    p.add_argument("--routes_npz", type=str, required=True, help="Fixed-length route npz (segment-level recommended).")
    p.add_argument("--road_graph_npz", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--min_jaccard", type=float, default=0.5, help="A trajectory is covered if its best candidate Jaccard >= this.")
    p.add_argument("--coverage_threshold", type=float, default=0.85, help="Gate passes if covered trajectory fraction >= this.")
    p.add_argument("--coverage_thr", type=float, default=None, help="DEPRECATED alias for --min_jaccard (kept for compatibility).")
    p.add_argument("--min_traj_per_od", type=int, default=2, help="Only compute OD diversity for groups with >= this many trajectories.")
    p.add_argument("--multimodal_thr", type=float, default=0.3, help="OD is multimodal if mean pairwise Jaccard distance >= this.")
    p.add_argument("--max_od", type=int, default=0, help="Optional cap on unique OD pairs for speed (0=all).")
    p.add_argument("--write_candidates_jsonl", action="store_true", help="Write candidates.jsonl (paths per OD) for downstream classifier training.")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if cKDTree is None:  # pragma: no cover
        raise SystemExit(f"Missing scipy.spatial.cKDTree (scipy). Error: {_KD_ERR}")

    min_jacc = float(args.min_jaccard)
    if args.coverage_thr is not None:
        # Backward-compat: old flag name.
        min_jacc = float(args.coverage_thr)
    cov_thr = float(args.coverage_threshold)
    mm_thr = float(args.multimodal_thr)
    if not (0.0 <= min_jacc <= 1.0 and 0.0 <= cov_thr <= 1.0):
        raise ValueError("--min_jaccard/--coverage_threshold must be in [0,1].")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g = _load_graph_npz(Path(args.road_graph_npz))
    H, W = int(g.grid.H), int(g.grid.W)

    data = np.load(str(Path(args.routes_npz)), allow_pickle=True)
    need = {"start_pos", "targets", "dest_pos", "traj_idx", "start_t"}
    if not need.issubset(set(data.files)):
        raise ValueError(f"routes_npz missing keys: {sorted(list(need - set(data.files)))}")
    start_pos = np.asarray(data["start_pos"], dtype=np.float32).reshape(-1, 2)
    dest_pos = np.asarray(data["dest_pos"], dtype=np.float32).reshape(-1, 2)
    targets = np.asarray(data["targets"], dtype=np.float32)
    traj_idx = np.asarray(data["traj_idx"], dtype=np.int64).reshape(-1)
    start_t = np.asarray(data["start_t"], dtype=np.int64).reshape(-1)
    n = int(start_pos.shape[0])

    node_xy = np.stack([g.node_y, g.node_x], axis=1).astype(np.float32, copy=False)
    tree = cKDTree(node_xy.astype(np.float64, copy=False))

    # Snap OD endpoints to nearest nodes.
    s_dist, s_idx = tree.query(start_pos.astype(np.float64, copy=False), k=1)
    t_dist, t_idx = tree.query(dest_pos.astype(np.float64, copy=False), k=1)
    s_idx = np.asarray(s_idx, dtype=np.int32).reshape(-1)
    t_idx = np.asarray(t_idx, dtype=np.int32).reshape(-1)
    s_dist = np.asarray(s_dist, dtype=np.float64).reshape(-1)
    t_dist = np.asarray(t_dist, dtype=np.float64).reshape(-1)

    # Cache candidates per OD.
    od_to_paths: Dict[Tuple[int, int], List[List[int]]] = {}
    od_list: List[Tuple[int, int]] = []
    for si, ti in zip(s_idx.tolist(), t_idx.tolist()):
        od_list.append((int(si), int(ti)))
    unique_ods = sorted(set(od_list))
    if int(args.max_od) > 0:
        unique_ods = unique_ods[: int(args.max_od)]

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        def tqdm(x, *a, **k):  # type: ignore
            return x

    for od in tqdm(unique_ods, desc="k-shortest", dynamic_ncols=True):
        si, ti = int(od[0]), int(od[1])
        paths = k_shortest_paths_yen(g, start=si, goal=ti, K=int(args.K))
        od_to_paths[(si, ti)] = paths

    # Evaluate coverage per trajectory (match candidates against GT).
    best_j = np.zeros((n,), dtype=np.float32)
    best_k = np.full((n,), -1, dtype=np.int32)
    gt_cells_all: List[np.ndarray] = []
    for i in range(n):
        poly = np.concatenate([start_pos[i : i + 1], targets[i]], axis=0)
        gt_cells_all.append(_poly_cells(poly, H=H, W=W))

    for i in tqdm(range(n), desc="coverage", dynamic_ncols=True):
        od = (int(s_idx[i]), int(t_idx[i]))
        paths = od_to_paths.get(od, [])
        gt_cells = gt_cells_all[i]
        bj = 0.0
        bk = -1
        for kk, p in enumerate(paths):
            nodes_yx = np.stack([g.node_y[np.asarray(p, dtype=np.int32)], g.node_x[np.asarray(p, dtype=np.int32)]], axis=1)
            cand_cells = _path_cells(nodes_yx, H=H, W=W)
            j = _jaccard(gt_cells, cand_cells)
            if j > bj:
                bj = float(j)
                bk = int(kk)
        best_j[i] = float(bj)
        best_k[i] = int(bk)

    covered = best_j >= float(min_jacc)

    # OD diversity based on GT (not candidates).
    od_to_indices: Dict[Tuple[int, int], List[int]] = {}
    for i, od in enumerate(od_list):
        if od not in od_to_indices:
            od_to_indices[od] = []
        od_to_indices[od].append(int(i))

    od_div = []
    od_div_is_mm = []
    for od, idxs in od_to_indices.items():
        if len(idxs) < int(args.min_traj_per_od):
            continue
        # Pairwise jaccard distance = 1 - jaccard
        dvals = []
        for a in range(len(idxs)):
            ca = gt_cells_all[idxs[a]]
            for b in range(a + 1, len(idxs)):
                cb = gt_cells_all[idxs[b]]
                dvals.append(1.0 - _jaccard(ca, cb))
        if dvals:
            m = float(np.mean(dvals))
            od_div.append(m)
            od_div_is_mm.append(bool(m >= float(mm_thr)))
    od_div = np.asarray(od_div, dtype=np.float32)
    od_div_is_mm = np.asarray(od_div_is_mm, dtype=np.uint8)

    # OD-level coverage: use per-OD median best_jaccard over its trajectories.
    od_best_p50 = []
    od_best_mean = []
    od_any = []
    for od, idxs in od_to_indices.items():
        bj = best_j[np.asarray(idxs, dtype=np.int64)]
        if bj.size == 0:
            continue
        od_best_p50.append(float(np.quantile(bj, 0.5)))
        od_best_mean.append(float(np.mean(bj)))
        od_any.append(bool(np.max(bj) >= float(min_jacc)))
    od_best_p50 = np.asarray(od_best_p50, dtype=np.float32)
    od_best_mean = np.asarray(od_best_mean, dtype=np.float32)
    od_any = np.asarray(od_any, dtype=np.uint8)

    def q(x: np.ndarray, p: float) -> Optional[float]:
        if x.size == 0:
            return None
        return float(np.quantile(x, float(p)))

    traj_covered_frac = float(np.mean(covered.astype(np.float32)))
    gate_passed = bool(traj_covered_frac >= float(cov_thr))

    report = {
        "ok": True,
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "gate": "candidate_paths_graph_gate",
        "gate_passed": bool(gate_passed),
        "inputs": {"routes_npz": str(Path(args.routes_npz)), "road_graph_npz": str(Path(args.road_graph_npz))},
        "config": {
            "K": int(args.K),
            "min_jaccard": float(min_jacc),
            "coverage_threshold": float(cov_thr),
            "min_traj_per_od": int(args.min_traj_per_od),
            "multimodal_thr": float(mm_thr),
            "max_od": int(args.max_od),
        },
        "stats": {
            "N": int(n),
            "num_unique_ods": int(len(unique_ods)),
            "snap_dist_grid": {
                "start_p50": float(np.quantile(s_dist, 0.5)),
                "start_p90": float(np.quantile(s_dist, 0.9)),
                "dest_p50": float(np.quantile(t_dist, 0.5)),
                "dest_p90": float(np.quantile(t_dist, 0.9)),
            },
            "traj_coverage": {
                "covered_frac": float(traj_covered_frac),
                "best_jaccard_mean": float(np.mean(best_j)),
                "best_jaccard_p50": q(best_j, 0.5),
                "best_jaccard_p90": q(best_j, 0.9),
            },
            "od_coverage": {
                "od_any_frac": float(np.mean(od_any.astype(np.float32))) if od_any.size else None,
                "od_best_jaccard_mean_mean": float(np.mean(od_best_mean)) if od_best_mean.size else None,
                "od_best_jaccard_p50_mean": q(od_best_mean, 0.5),
                "od_best_jaccard_p50_p50": q(od_best_p50, 0.5),
            },
            "gt_od_diversity_dist": {
                "num_ods": int(od_div.size),
                "mean": float(np.mean(od_div)) if od_div.size else None,
                "p50": q(od_div, 0.5),
                "p90": q(od_div, 0.9),
            },
            "gt_multimodal_ods": {
                "num_ods": int(od_div_is_mm.size),
                "num_multimodal_ods": int(np.sum(od_div_is_mm).item()) if od_div_is_mm.size else 0,
                "multimodal_rate": float(np.mean(od_div_is_mm.astype(np.float32))) if od_div_is_mm.size else None,
            },
        },
        "outputs": {"report_json": str((out_dir / "report.json").resolve())},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if bool(args.write_candidates_jsonl):
        path_out = out_dir / "candidates.jsonl"
        with path_out.open("w", encoding="utf-8") as f:
            for od, paths in od_to_paths.items():
                rec = {"s_idx": int(od[0]), "t_idx": int(od[1]), "paths": [list(map(int, p)) for p in paths]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        report["outputs"]["candidates_jsonl"] = str(path_out.resolve())
        (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ["ok", "gate", "gate_passed", "stats", "outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
