from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(x, *args, **kwargs):  # type: ignore[no-redef]
        return x

from src.features.waypoints import pick_waypoint_indices_rdp_fixed_k, pick_waypoint_indices_rdp_turn_fixed_k


TZ_SHANGHAI = timezone(timedelta(hours=8))


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


@dataclass(frozen=True)
class DumpCfg:
    num_waypoints: int
    mode: str
    turn_alpha: float
    branch_degree_thr: int
    od_bin: int
    min_traj_per_od: int
    min_choice_count: int
    seed: int
    progress: str
    log_every: int


def _load_graph_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    need = {"node_y", "node_x", "meta"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"road_graph.npz missing keys: {missing}")
    meta = data["meta"]
    if isinstance(meta, np.ndarray) and meta.shape == ():
        meta = meta.item()
    if not isinstance(meta, dict):
        raise ValueError("road_graph.npz meta must be a dict.")
    return {
        "node_y": np.asarray(data["node_y"], dtype=np.float32).reshape(-1),
        "node_x": np.asarray(data["node_x"], dtype=np.float32).reshape(-1),
        "edge_u": (np.asarray(data["edge_u"], dtype=np.int32).reshape(-1) if "edge_u" in data.files else None),
        "edge_v": (np.asarray(data["edge_v"], dtype=np.int32).reshape(-1) if "edge_v" in data.files else None),
        "meta": meta,
    }


def _load_paths_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    need = {"start_t", "start_node", "dest_node", "node_seq_pad", "node_seq_len", "traj_idx"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"paths_graph.npz missing keys: {missing}")
    meta = data["meta"] if "meta" in data.files else None
    if isinstance(meta, np.ndarray) and meta.shape == ():
        meta = meta.item()
    return {
        "start_t": np.asarray(data["start_t"], dtype=np.int64).reshape(-1),
        "start_node": np.asarray(data["start_node"], dtype=np.int32).reshape(-1),
        "dest_node": np.asarray(data["dest_node"], dtype=np.int32).reshape(-1),
        "node_seq_pad": np.asarray(data["node_seq_pad"], dtype=np.int32),
        "node_seq_len": np.asarray(data["node_seq_len"], dtype=np.int32).reshape(-1),
        "traj_idx": np.asarray(data["traj_idx"], dtype=np.int64).reshape(-1),
        "route_city": np.asarray(data["route_city"], dtype=np.int8).reshape(-1) if "route_city" in data.files else None,
        "start_pos": np.asarray(data["start_pos"], dtype=np.float32).reshape(-1, 2) if "start_pos" in data.files else None,
        "dest_pos": np.asarray(data["dest_pos"], dtype=np.float32).reshape(-1, 2) if "dest_pos" in data.files else None,
        "meta": meta if isinstance(meta, dict) else None,
    }


def _pick_idx(points: np.ndarray, *, cfg: DumpCfg) -> np.ndarray:
    k = int(cfg.num_waypoints)
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    mode = str(cfg.mode)
    if mode == "rdp_dev":
        return pick_waypoint_indices_rdp_fixed_k(points, k=k)
    if mode == "rdp_turn":
        return pick_waypoint_indices_rdp_turn_fixed_k(points, k=k, turn_alpha=float(cfg.turn_alpha))
    if mode == "branch":
        # Branch mode needs graph degrees and is handled in run_dump().
        return np.zeros((0,), dtype=np.int64)
    if mode == "decision_branch":
        # Decision-branch mode needs OD-bin groups and is handled in run_dump().
        return np.zeros((0,), dtype=np.int64)
    raise ValueError(f"Unknown mode {cfg.mode!r} (expected rdp_dev|rdp_turn|branch|decision_branch)")


def _unique_undirected_degree(*, n_nodes: int, edge_u: np.ndarray, edge_v: np.ndarray) -> np.ndarray:
    """
    Undirected neighbor degree = number of unique neighbors per node.
    Robust to one-way streets and avoids double-counting bidirectional edges.
    """
    n = int(n_nodes)
    eu = np.asarray(edge_u, dtype=np.int64).reshape(-1)
    ev = np.asarray(edge_v, dtype=np.int64).reshape(-1)
    if eu.size != ev.size:
        raise ValueError("edge_u/edge_v length mismatch")
    if eu.size == 0:
        return np.zeros((n,), dtype=np.int32)

    u = np.concatenate([eu, ev], axis=0)
    v = np.concatenate([ev, eu], axis=0)
    m = (u >= 0) & (u < n) & (v >= 0) & (v < n) & (u != v)
    u = u[m]
    v = v[m]
    if u.size == 0:
        return np.zeros((n,), dtype=np.int32)
    order = np.lexsort((v, u))
    u = u[order]
    v = v[order]
    uniq = np.ones((u.size,), dtype=np.bool_)
    uniq[1:] = (u[1:] != u[:-1]) | (v[1:] != v[:-1])
    u_uniq = u[uniq]
    deg = np.bincount(u_uniq.astype(np.int64, copy=False), minlength=n).astype(np.int32, copy=False)
    return deg


def _pick_idx_branch_from_seq(*, seq: np.ndarray, deg: np.ndarray, cfg: DumpCfg) -> np.ndarray:
    """
    Pick internal waypoint positions based on branch nodes (degree >= threshold).
    Returns indices into `seq` (positions), excluding endpoints.
    """
    k = int(cfg.num_waypoints)
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    L = int(seq.size)
    if L < 3:
        return np.zeros((0,), dtype=np.int64)
    thr = int(cfg.branch_degree_thr)
    internal = seq[1:-1].astype(np.int64, copy=False)
    m = deg[internal] >= thr
    pos = (np.nonzero(m)[0] + 1).astype(np.int64, copy=False)
    if pos.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if pos.size <= k:
        return pos
    q = np.linspace(0.0, float(pos.size - 1), num=k, dtype=np.float64)
    sel = np.clip(np.rint(q).astype(np.int64), 0, int(pos.size) - 1)
    return pos[sel].astype(np.int64, copy=False)


def _fill_to_k(*, idx: np.ndarray, L: int, k: int) -> np.ndarray:
    """
    Fill selected internal indices to length k using evenly spaced positions (excluding endpoints).
    Keeps existing indices first.
    """
    k = int(k)
    L = int(L)
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    idx_list = [int(x) for x in np.asarray(idx, dtype=np.int64).reshape(-1).tolist() if 1 <= int(x) <= int(L) - 2]
    seen = set(idx_list)
    if len(idx_list) >= k:
        return np.asarray(idx_list[:k], dtype=np.int64)
    fill = np.linspace(1, int(L) - 2, num=k, dtype=np.float64)
    fill = np.clip(np.rint(fill).astype(np.int64), 1, int(L) - 2)
    for j in fill.tolist():
        jj = int(j)
        if jj not in seen:
            idx_list.append(jj)
            seen.add(jj)
        if len(idx_list) >= k:
            break
    if not idx_list:
        idx_list = [1]
    while len(idx_list) < k:
        idx_list.append(int(idx_list[-1]))
    idx_list = idx_list[:k]
    idx_list.sort()
    return np.asarray(idx_list, dtype=np.int64)


def _od_bin_key(*, start_pos: np.ndarray, dest_pos: np.ndarray, route_city: np.ndarray, od_bin: int) -> List[Tuple[int, int, int, int, int]]:
    start_pos = np.asarray(start_pos, dtype=np.float32).reshape(-1, 2)
    dest_pos = np.asarray(dest_pos, dtype=np.float32).reshape(-1, 2)
    route_city = np.asarray(route_city, dtype=np.int64).reshape(-1)
    b = int(max(1, od_bin))
    s_bin = np.floor(start_pos / float(b)).astype(np.int32)
    d_bin = np.floor(dest_pos / float(b)).astype(np.int32)
    out = []
    for c, (sy, sx), (dy, dx) in zip(route_city.tolist(), s_bin.tolist(), d_bin.tolist()):
        out.append((int(c), int(sy), int(sx), int(dy), int(dx)))
    return out


def _decision_nodes_by_group(
    *,
    od_keys: Sequence[Tuple[int, int, int, int, int]],
    node_seq_pad: np.ndarray,
    node_seq_len: np.ndarray,
    min_traj_per_od: int,
    min_choice_count: int,
) -> Tuple[Dict[Tuple[int, int, int, int, int], set[int]], Dict[Tuple[int, int, int, int, int], int]]:
    """
    Data-driven decision nodes:
    Within each OD-bin group, a node u is a decision node if there exist >=2 distinct next nodes v
    such that transition (u->v) is observed in >=min_choice_count trajectories in this group.
    """
    min_traj_per_od = int(max(1, min_traj_per_od))
    min_choice_count = int(max(1, min_choice_count))

    # Pass 1: group sizes
    group_n: Dict[Tuple[int, int, int, int, int], int] = {}
    for k in od_keys:
        group_n[k] = int(group_n.get(k, 0)) + 1
    eligible = {k for k, n in group_n.items() if int(n) >= int(min_traj_per_od)}

    # Pass 2: accumulate per-group observed next-node transitions (counted per-trajectory, not per-step).
    counts: Dict[Tuple[int, int, int, int, int], Dict[int, Dict[int, int]]] = {k: {} for k in eligible}
    decision_nodes: Dict[Tuple[int, int, int, int, int], set[int]] = {k: set() for k in eligible}

    for i, key in enumerate(od_keys):
        if key not in eligible:
            continue
        L = int(node_seq_len[i])
        if L < 2:
            continue
        seq = node_seq_pad[i, :L].astype(np.int64, copy=False)
        # Unique transitions for this trajectory.
        seen = set()
        for a, b in zip(seq[:-1].tolist(), seq[1:].tolist()):
            aa = int(a)
            bb = int(b)
            if aa < 0 or bb < 0 or aa == bb:
                continue
            seen.add((aa << 32) | (bb & 0xFFFFFFFF))
        if not seen:
            continue

        per_u = counts[key]
        decided = decision_nodes[key]
        for code in seen:
            u = int(code >> 32)
            v = int(code & 0xFFFFFFFF)
            if u in decided:
                continue
            m = per_u.get(u)
            if m is None:
                m = {}
                per_u[u] = m
            c = int(m.get(v, 0)) + 1
            if c > min_choice_count:
                c = min_choice_count
            m[v] = c
            # Early-stop for this u once it becomes a decision node.
            n_opts = 0
            for vv, cc in m.items():
                if int(cc) >= min_choice_count:
                    n_opts += 1
                    if n_opts >= 2:
                        decided.add(u)
                        per_u.pop(u, None)
                        break
    return decision_nodes, group_n


def run_dump(*, paths_graph_npz: Path, road_graph_npz: Path, out_dir: Path, cfg: DumpCfg, viz_cases: int) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "report.json"
    out_npz = out_dir / "waypoints_graph.npz"

    g = _load_graph_npz(road_graph_npz)
    node_y = g["node_y"]
    node_x = g["node_x"]
    edge_u = g.get("edge_u", None)
    edge_v = g.get("edge_v", None)
    meta_g = g["meta"]

    p = _load_paths_npz(paths_graph_npz)
    node_seq_pad = p["node_seq_pad"]
    node_seq_len = p["node_seq_len"]
    start_t = p["start_t"]
    traj_idx = p["traj_idx"]
    start_node = p["start_node"]
    dest_node = p["dest_node"]
    route_city = p["route_city"]
    meta_p = p["meta"]
    start_pos = p["start_pos"]
    dest_pos = p["dest_pos"]

    N = int(start_node.size)
    print(json.dumps({"event": "loaded", "N": int(N)}, ensure_ascii=False), flush=True)
    K = int(cfg.num_waypoints)
    wp_seq = np.full((N, K + 2), -1, dtype=np.int32)
    wp_len = np.full((N,), K + 2, dtype=np.int32)
    gt_len = np.asarray(node_seq_len, dtype=np.int32, copy=False).reshape(-1)

    deg = None
    if str(cfg.mode) == "branch":
        if edge_u is None or edge_v is None:
            raise ValueError("mode=branch requires road_graph.npz to contain edge_u/edge_v.")
        print(json.dumps({"event": "compute_degree", "mode": "branch", "thr": int(cfg.branch_degree_thr)}, ensure_ascii=False), flush=True)
        deg = _unique_undirected_degree(n_nodes=int(node_y.size), edge_u=edge_u, edge_v=edge_v)
        print(json.dumps({"event": "computed_degree", "deg_p50": float(np.percentile(deg, 50)), "deg_p90": float(np.percentile(deg, 90))}, ensure_ascii=False), flush=True)

    decision_nodes = None
    group_n = None
    decision_counts = []
    if str(cfg.mode) == "decision_branch":
        if start_pos is None or dest_pos is None:
            raise ValueError("mode=decision_branch requires paths_graph.npz to contain start_pos/dest_pos.")
        rc = route_city if route_city is not None else np.zeros((int(N),), dtype=np.int8)
        od_keys = _od_bin_key(start_pos=start_pos, dest_pos=dest_pos, route_city=rc, od_bin=int(cfg.od_bin))
        print(
            json.dumps(
                {
                    "event": "compute_decision_branch",
                    "od_bin": int(cfg.od_bin),
                    "min_traj_per_od": int(cfg.min_traj_per_od),
                    "min_choice_count": int(cfg.min_choice_count),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        decision_nodes, group_n = _decision_nodes_by_group(
            od_keys=od_keys,
            node_seq_pad=node_seq_pad,
            node_seq_len=node_seq_len,
            min_traj_per_od=int(cfg.min_traj_per_od),
            min_choice_count=int(cfg.min_choice_count),
        )
        num_groups = len(group_n or {})
        num_elig = len(decision_nodes or {})
        num_nodes = int(sum(len(v) for v in (decision_nodes or {}).values()))
        print(
            json.dumps(
                {"event": "computed_decision_branch", "num_groups": int(num_groups), "num_eligible_groups": int(num_elig), "num_decision_nodes": int(num_nodes)},
                ensure_ascii=False,
            ),
            flush=True,
        )

    good = 0
    branch_counts = []
    progress_mode = str(cfg.progress)
    if progress_mode == "auto":
        # tqdm's carriage-return output is not friendly when piped to files (tee). Use JSON lines in that case.
        progress_mode = "tqdm" if bool(sys.stderr.isatty()) else "json"
    if progress_mode not in {"tqdm", "json", "none"}:
        raise ValueError(f"--progress must be one of auto|tqdm|json|none, got {cfg.progress!r}")

    it = range(N)
    if progress_mode == "tqdm":
        it = tqdm(it, desc="dump_waypoints", dynamic_ncols=True)  # type: ignore[assignment]

    for i in it:
        L = int(node_seq_len[i])
        if L < 2:
            continue
        seq = node_seq_pad[i, :L].astype(np.int64, copy=False)
        # Build polyline points (y,x).
        yy = node_y[seq]
        xx = node_x[seq]
        pts = np.stack([yy, xx], axis=1).astype(np.float32, copy=False)

        if L <= 2:
            # Degenerate path: only [start, dest]. Repeat dest as internal waypoints.
            nodes = [int(seq[0])] + [int(seq[-1])] * int(K) + [int(seq[-1])]
        else:
            if str(cfg.mode) == "branch":
                assert deg is not None
                branch_counts.append(int((deg[seq[1:-1].astype(np.int64, copy=False)] >= int(cfg.branch_degree_thr)).sum()))
                idx = _pick_idx_branch_from_seq(seq=seq, deg=deg, cfg=cfg)
                idx = _fill_to_k(idx=idx, L=int(L), k=int(K))
            elif str(cfg.mode) == "decision_branch":
                assert decision_nodes is not None
                assert group_n is not None
                rc = int(route_city[i]) if route_city is not None else 0
                sy, sx = (start_pos[i] if start_pos is not None else np.asarray([0.0, 0.0], dtype=np.float32)).tolist()
                dy, dx = (dest_pos[i] if dest_pos is not None else np.asarray([0.0, 0.0], dtype=np.float32)).tolist()
                b = int(max(1, int(cfg.od_bin)))
                key = (int(rc), int(math.floor(float(sy) / float(b))), int(math.floor(float(sx) / float(b))), int(math.floor(float(dy) / float(b))), int(math.floor(float(dx) / float(b))))
                decided = decision_nodes.get(key)
                cand_pos = []
                if decided is not None and int(group_n.get(key, 0)) >= int(cfg.min_traj_per_od):
                    for pos in range(1, int(L) - 1):
                        nid = int(seq[pos])
                        if nid >= 0 and nid in decided:
                            cand_pos.append(int(pos))
                decision_counts.append(int(len(cand_pos)))
                if cand_pos:
                    pos_arr = np.asarray(sorted(set(cand_pos)), dtype=np.int64)
                    if int(pos_arr.size) > int(K):
                        q = np.linspace(0.0, float(pos_arr.size - 1), num=int(K), dtype=np.float64)
                        sel = np.clip(np.rint(q).astype(np.int64), 0, int(pos_arr.size) - 1)
                        idx = pos_arr[sel].astype(np.int64, copy=False)
                    else:
                        idx = pos_arr.astype(np.int64, copy=False)
                    idx = _fill_to_k(idx=idx, L=int(L), k=int(K))
                else:
                    idx = _fill_to_k(idx=np.zeros((0,), dtype=np.int64), L=int(L), k=int(K))
            elif L < (K + 2):
                # Fallback: time quantiles (duplicates allowed; fixed K).
                hi = int(max(1, L - 2))
                fill = np.linspace(1, hi, num=K, dtype=np.float32)
                idx = np.clip(np.rint(fill), 1, hi).astype(np.int64, copy=False)[: int(K)]
            else:
                idx = _pick_idx(pts, cfg=cfg)
                if idx.size < K:
                    # Ensure fixed K (prefer RDP picks, then fill by time quantiles; duplicates allowed).
                    fill = np.linspace(1, L - 2, num=K, dtype=np.float32)
                    fill = np.clip(np.rint(fill), 1, L - 2).astype(np.int64, copy=False)
                    idx = np.concatenate([idx.astype(np.int64, copy=False), fill.astype(np.int64, copy=False)], axis=0)[: int(K)]
                if idx.size < K:
                    idx = np.pad(idx, (0, int(K) - int(idx.size)), mode="edge")
                idx = idx[: int(K)]

            nodes = [int(seq[0])] + [int(seq[int(j)]) for j in idx.tolist()] + [int(seq[-1])]
        wp_seq[i, :] = np.asarray(nodes, dtype=np.int32)
        good += 1
        if progress_mode == "json" and (int(i) % int(max(1, cfg.log_every)) == 0 or int(i) == int(N) - 1):
            print(
                json.dumps(
                    {"task": "dump_waypoints_from_paths_graph_npz", "i": int(i), "N": int(N), "done": int(good), "pct": float(good) / float(max(1, int(N)))},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "paths_graph_npz": str(paths_graph_npz),
        "road_graph_npz": str(road_graph_npz),
        "cfg": {
            "num_waypoints": K,
            "mode": str(cfg.mode),
            "turn_alpha": float(cfg.turn_alpha),
            "branch_degree_thr": int(cfg.branch_degree_thr),
            "od_bin": int(cfg.od_bin),
            "min_traj_per_od": int(cfg.min_traj_per_od),
            "min_choice_count": int(cfg.min_choice_count),
            "seed": int(cfg.seed),
        },
        "graph_meta": meta_g,
        "paths_meta": meta_p,
    }
    print(json.dumps({"event": "extracted", "done": int(good)}, ensure_ascii=False), flush=True)
    print(json.dumps({"event": "saving", "out_npz": str(out_npz)}, ensure_ascii=False), flush=True)
    # NOTE: This file is small (~KB/MB). Prefer uncompressed npz for robustness and speed on different filesystems.
    np.savez(
        out_npz,
        wp_seq=wp_seq,
        wp_len=wp_len,
        gt_len=gt_len,
        start_t=start_t.astype(np.int64, copy=False),
        traj_idx=traj_idx.astype(np.int64, copy=False),
        start_node=start_node.astype(np.int32, copy=False),
        dest_node=dest_node.astype(np.int32, copy=False),
        route_city=(route_city.astype(np.int8, copy=False) if route_city is not None else None),
        meta=meta,
    )
    print(json.dumps({"event": "saved", "out_npz": str(out_npz)}, ensure_ascii=False), flush=True)

    report: Dict[str, object] = {
        "ok": True,
        "task": "dump_waypoints_from_paths_graph_npz",
        "inputs": {"paths_graph_npz": str(paths_graph_npz), "road_graph_npz": str(road_graph_npz)},
        "config": {
            "num_waypoints": K,
            "mode": str(cfg.mode),
            "turn_alpha": float(cfg.turn_alpha),
            "branch_degree_thr": int(cfg.branch_degree_thr),
            "od_bin": int(cfg.od_bin),
            "min_traj_per_od": int(cfg.min_traj_per_od),
            "min_choice_count": int(cfg.min_choice_count),
            "seed": int(cfg.seed),
            "viz_cases": int(viz_cases),
        },
        "stats": {
            "n_routes": int(N),
            "n_good": int(good),
            "branch_candidates": (
                {
                    "mean": float(np.mean(np.asarray(branch_counts, dtype=np.float64))) if branch_counts else None,
                    "p50": float(np.percentile(np.asarray(branch_counts, dtype=np.float64), 50)) if branch_counts else None,
                    "p90": float(np.percentile(np.asarray(branch_counts, dtype=np.float64), 90)) if branch_counts else None,
                }
                if str(cfg.mode) == "branch"
                else None
            ),
            "decision_branch_candidates": (
                {
                    "mean": float(np.mean(np.asarray(decision_counts, dtype=np.float64))) if decision_counts else None,
                    "p50": float(np.percentile(np.asarray(decision_counts, dtype=np.float64), 50)) if decision_counts else None,
                    "p90": float(np.percentile(np.asarray(decision_counts, dtype=np.float64), 90)) if decision_counts else None,
                }
                if str(cfg.mode) == "decision_branch"
                else None
            ),
            "gt_len": {
                "p50": float(np.percentile(gt_len, 50)),
                "p90": float(np.percentile(gt_len, 90)),
            },
        },
        "outputs": {"out_npz": str(out_npz), "report_json": str(report_json)},
        "meta": {"created_at": meta["created_at"]},
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump fixed-K waypoint node sequences from GT graph paths (paths_graph.npz).")
    p.add_argument("--paths_graph_npz", type=Path, required=True)
    p.add_argument("--road_graph_npz", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--num_waypoints", type=int, default=4, help="Number of INTERNAL waypoints (excluding start/dest).")
    p.add_argument("--mode", type=str, default="rdp_turn", choices=["rdp_dev", "rdp_turn", "branch", "decision_branch"])
    p.add_argument("--turn_alpha", type=float, default=1.0)
    p.add_argument("--branch_degree_thr", type=int, default=3, help="Only used when --mode=branch. Branch node = degree >= thr.")
    p.add_argument("--od_bin", type=int, default=128, help="Only used when --mode=decision_branch. OD bin size in grid cells.")
    p.add_argument("--min_traj_per_od", type=int, default=5, help="Only used when --mode=decision_branch. Require at least this many routes per OD-bin group.")
    p.add_argument("--min_choice_count", type=int, default=2, help="Only used when --mode=decision_branch. Require each branch transition to appear in >= this many trajectories.")
    p.add_argument("--progress", type=str, default="auto", choices=["auto", "tqdm", "json", "none"])
    p.add_argument("--log_every", type=int, default=200, help="Only used when --progress=json.")
    p.add_argument("--viz_cases", type=int, default=0, help="Reserved for future; kept for naming consistency.")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))
    cfg = DumpCfg(
        num_waypoints=int(args.num_waypoints),
        mode=str(args.mode),
        turn_alpha=float(args.turn_alpha),
        branch_degree_thr=int(args.branch_degree_thr),
        od_bin=int(args.od_bin),
        min_traj_per_od=int(args.min_traj_per_od),
        min_choice_count=int(args.min_choice_count),
        seed=int(args.seed),
        progress=str(args.progress),
        log_every=int(args.log_every),
    )
    report = run_dump(
        paths_graph_npz=Path(args.paths_graph_npz),
        road_graph_npz=Path(args.road_graph_npz),
        out_dir=Path(args.out_dir),
        cfg=cfg,
        viz_cases=int(args.viz_cases),
    )
    compact = {"ok": True, "out_npz": report["outputs"]["out_npz"], "n_routes": report["stats"]["n_routes"], "report_json": report["outputs"]["report_json"]}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
