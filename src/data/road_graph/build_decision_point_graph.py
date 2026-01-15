"""
Build Decision Point Graph from GT trajectories.

A decision point is a node where:
1. The node appears in GT trajectories
2. From this node, GT trajectories diverge to >= 2 different next nodes
3. Each divergent choice is observed in >= min_choice_count trajectories

Output:
- decision_points: array of node ids that are decision points
- dp_to_idx: mapping from node_id to decision point index
- dp_successors[dp_idx]: list of successor decision points (or dest) reachable from dp
- dp_successor_counts[dp_idx]: how many times each successor was observed
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class BuildCfg:
    min_choice_count: int  # min times a (u -> v) transition must be observed
    min_out_degree: int  # min number of distinct successors for a decision point
    seed: int


def _load_paths_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    need = {"start_t", "start_node", "dest_node", "node_seq_pad", "node_seq_len", "traj_idx"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"paths_graph.npz missing keys: {missing}")
    return {
        "start_t": np.asarray(data["start_t"], dtype=np.int64).reshape(-1),
        "start_node": np.asarray(data["start_node"], dtype=np.int32).reshape(-1),
        "dest_node": np.asarray(data["dest_node"], dtype=np.int32).reshape(-1),
        "node_seq_pad": np.asarray(data["node_seq_pad"], dtype=np.int32),
        "node_seq_len": np.asarray(data["node_seq_len"], dtype=np.int32).reshape(-1),
        "traj_idx": np.asarray(data["traj_idx"], dtype=np.int64).reshape(-1),
        "route_city": np.asarray(data["route_city"], dtype=np.int8).reshape(-1) if "route_city" in data.files else None,
    }


def _load_graph_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    return {
        "node_y": np.asarray(data["node_y"], dtype=np.float32).reshape(-1),
        "node_x": np.asarray(data["node_x"], dtype=np.float32).reshape(-1),
        "edge_u": np.asarray(data["edge_u"], dtype=np.int32).reshape(-1),
        "edge_v": np.asarray(data["edge_v"], dtype=np.int32).reshape(-1),
        "edge_tier": np.asarray(data["edge_tier"], dtype=np.uint8).reshape(-1) if "edge_tier" in data.files else None,
    }


def find_decision_points(
    *,
    node_seq_pad: np.ndarray,
    node_seq_len: np.ndarray,
    cfg: BuildCfg,
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, Dict[int, int]]]:
    """
    Find decision points from GT trajectories.
    
    Returns:
        decision_points: (D,) array of node ids
        dp_to_idx: dict mapping node_id -> dp_idx
        transition_counts: {node_id: {next_node: count}}
    """
    n_routes = int(node_seq_len.shape[0])
    
    # Count transitions: node -> {next_node: count}
    transition_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    
    for rid in range(n_routes):
        L = int(node_seq_len[rid])
        if L < 2:
            continue
        seq = node_seq_pad[rid, :L].astype(np.int64, copy=False)
        for i in range(L - 1):
            u = int(seq[i])
            v = int(seq[i + 1])
            if u >= 0 and v >= 0 and u != v:
                transition_counts[u][v] += 1
    
    # Find decision points: nodes with >= min_out_degree successors,
    # each observed >= min_choice_count times
    decision_points_set: Set[int] = set()
    for u, successors in transition_counts.items():
        # Filter successors by min_choice_count
        valid_successors = {v: c for v, c in successors.items() if c >= cfg.min_choice_count}
        if len(valid_successors) >= cfg.min_out_degree:
            decision_points_set.add(u)
    
    # Also add all start and dest nodes as decision points
    for rid in range(n_routes):
        L = int(node_seq_len[rid])
        if L >= 2:
            seq = node_seq_pad[rid, :L].astype(np.int64, copy=False)
            decision_points_set.add(int(seq[0]))   # start
            decision_points_set.add(int(seq[-1]))  # dest
    
    decision_points = np.array(sorted(decision_points_set), dtype=np.int64)
    dp_to_idx = {int(dp): i for i, dp in enumerate(decision_points.tolist())}
    
    return decision_points, dp_to_idx, dict(transition_counts)


def build_dp_sequences(
    *,
    node_seq_pad: np.ndarray,
    node_seq_len: np.ndarray,
    start_node: np.ndarray,
    dest_node: np.ndarray,
    decision_points: np.ndarray,
    dp_to_idx: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert GT node sequences to decision point sequences.
    
    For each GT trajectory, extract only the decision points it passes through.
    
    Returns:
        dp_seq_pad: (N, max_dp_len) padded decision point index sequences
        dp_seq_len: (N,) lengths
    """
    n_routes = int(node_seq_len.shape[0])
    dp_set = set(decision_points.tolist())
    
    # First pass: find max dp sequence length
    dp_seqs: List[List[int]] = []
    for rid in range(n_routes):
        L = int(node_seq_len[rid])
        if L < 2:
            dp_seqs.append([])
            continue
        seq = node_seq_pad[rid, :L].astype(np.int64, copy=False)
        dp_seq = []
        for nid in seq.tolist():
            if int(nid) in dp_set:
                dp_idx = dp_to_idx[int(nid)]
                # Avoid consecutive duplicates
                if not dp_seq or dp_seq[-1] != dp_idx:
                    dp_seq.append(dp_idx)
        dp_seqs.append(dp_seq)
    
    max_len = max(len(s) for s in dp_seqs) if dp_seqs else 1
    max_len = max(max_len, 2)
    
    dp_seq_pad = np.full((n_routes, max_len), -1, dtype=np.int32)
    dp_seq_len = np.zeros((n_routes,), dtype=np.int32)
    
    for rid, dp_seq in enumerate(dp_seqs):
        if dp_seq:
            dp_seq_len[rid] = len(dp_seq)
            dp_seq_pad[rid, :len(dp_seq)] = np.array(dp_seq, dtype=np.int32)
    
    return dp_seq_pad, dp_seq_len


def build_dp_transitions(
    *,
    dp_seq_pad: np.ndarray,
    dp_seq_len: np.ndarray,
    n_decision_points: int,
) -> Tuple[Dict[int, Dict[int, int]], np.ndarray]:
    """
    Build transition counts between decision points.
    
    Returns:
        dp_transitions: {dp_idx: {next_dp_idx: count}}
        dp_out_degree: (D,) number of distinct successors for each dp
    """
    dp_transitions: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    
    n_routes = int(dp_seq_len.shape[0])
    for rid in range(n_routes):
        L = int(dp_seq_len[rid])
        if L < 2:
            continue
        seq = dp_seq_pad[rid, :L].astype(np.int64, copy=False)
        for i in range(L - 1):
            u = int(seq[i])
            v = int(seq[i + 1])
            if u >= 0 and v >= 0:
                dp_transitions[u][v] += 1
    
    dp_out_degree = np.zeros((n_decision_points,), dtype=np.int32)
    for u, succs in dp_transitions.items():
        dp_out_degree[u] = len(succs)
    
    return dict(dp_transitions), dp_out_degree


def run(
    *,
    paths_graph_npz: Path,
    road_graph_npz: Path,
    out_dir: Path,
    cfg: BuildCfg,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "decision_point_graph.npz"
    report_json = out_dir / "report.json"
    
    print(json.dumps({"event": "loading_data"}), flush=True)
    p = _load_paths_npz(paths_graph_npz)
    g = _load_graph_npz(road_graph_npz)
    
    node_seq_pad = p["node_seq_pad"]
    node_seq_len = p["node_seq_len"]
    start_node = p["start_node"]
    dest_node = p["dest_node"]
    start_t = p["start_t"]
    route_city = p["route_city"]
    n_routes = int(node_seq_len.shape[0])
    
    print(json.dumps({"event": "finding_decision_points", "n_routes": n_routes}), flush=True)
    decision_points, dp_to_idx, transition_counts = find_decision_points(
        node_seq_pad=node_seq_pad,
        node_seq_len=node_seq_len,
        cfg=cfg,
    )
    n_dp = int(decision_points.shape[0])
    print(json.dumps({"event": "found_decision_points", "n_decision_points": n_dp}), flush=True)
    
    print(json.dumps({"event": "building_dp_sequences"}), flush=True)
    dp_seq_pad, dp_seq_len = build_dp_sequences(
        node_seq_pad=node_seq_pad,
        node_seq_len=node_seq_len,
        start_node=start_node,
        dest_node=dest_node,
        decision_points=decision_points,
        dp_to_idx=dp_to_idx,
    )
    
    print(json.dumps({"event": "building_dp_transitions"}), flush=True)
    dp_transitions, dp_out_degree = build_dp_transitions(
        dp_seq_pad=dp_seq_pad,
        dp_seq_len=dp_seq_len,
        n_decision_points=n_dp,
    )
    
    # Build CSR format for dp successors
    dp_succ_ptr = np.zeros((n_dp + 1,), dtype=np.int64)
    dp_succ_idx_list: List[int] = []
    dp_succ_cnt_list: List[int] = []
    for i in range(n_dp):
        succs = dp_transitions.get(i, {})
        dp_succ_ptr[i + 1] = dp_succ_ptr[i] + len(succs)
        for v, c in sorted(succs.items()):
            dp_succ_idx_list.append(v)
            dp_succ_cnt_list.append(c)
    dp_succ_idx = np.array(dp_succ_idx_list, dtype=np.int32) if dp_succ_idx_list else np.zeros((0,), dtype=np.int32)
    dp_succ_cnt = np.array(dp_succ_cnt_list, dtype=np.int32) if dp_succ_cnt_list else np.zeros((0,), dtype=np.int32)
    
    # Decision point positions
    dp_y = g["node_y"][decision_points]
    dp_x = g["node_x"][decision_points]
    
    # Statistics
    dp_seq_lens = dp_seq_len[dp_seq_len >= 2]
    avg_dp_seq_len = float(np.mean(dp_seq_lens)) if dp_seq_lens.size > 0 else 0.0
    
    print(json.dumps({"event": "saving", "out_npz": str(out_npz)}), flush=True)
    np.savez_compressed(
        str(out_npz),
        # Decision points
        decision_points=decision_points,
        dp_y=dp_y,
        dp_x=dp_x,
        dp_out_degree=dp_out_degree,
        # DP successors (CSR format)
        dp_succ_ptr=dp_succ_ptr,
        dp_succ_idx=dp_succ_idx,
        dp_succ_cnt=dp_succ_cnt,
        # DP sequences for training
        dp_seq_pad=dp_seq_pad,
        dp_seq_len=dp_seq_len,
        # Original route info (for time features, etc.)
        start_t=start_t,
        start_node=start_node,
        dest_node=dest_node,
        route_city=route_city if route_city is not None else np.zeros((n_routes,), dtype=np.int8),
        # Config
        min_choice_count=np.int32(cfg.min_choice_count),
        min_out_degree=np.int32(cfg.min_out_degree),
    )
    
    report = {
        "ok": True,
        "task": "build_decision_point_graph",
        "inputs": {
            "paths_graph_npz": str(paths_graph_npz),
            "road_graph_npz": str(road_graph_npz),
        },
        "config": {
            "min_choice_count": int(cfg.min_choice_count),
            "min_out_degree": int(cfg.min_out_degree),
            "seed": int(cfg.seed),
        },
        "stats": {
            "n_routes": int(n_routes),
            "n_decision_points": int(n_dp),
            "avg_dp_seq_len": float(avg_dp_seq_len),
            "dp_seq_len_p50": float(np.median(dp_seq_lens)) if dp_seq_lens.size > 0 else 0.0,
            "dp_seq_len_p90": float(np.percentile(dp_seq_lens, 90)) if dp_seq_lens.size > 0 else 0.0,
            "dp_out_degree_mean": float(np.mean(dp_out_degree)),
            "dp_out_degree_max": int(np.max(dp_out_degree)) if dp_out_degree.size > 0 else 0,
            "n_routes_with_dp_seq": int(np.sum(dp_seq_len >= 2)),
        },
        "outputs": {
            "out_npz": str(out_npz),
        },
        "meta": {"created_at": datetime.now(tz=TZ_SHANGHAI).isoformat()},
    }
    
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Build Decision Point Graph from GT trajectories")
    p.add_argument("--paths_graph_npz", type=Path, required=True)
    p.add_argument("--road_graph_npz", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--min_choice_count", type=int, default=2, help="Min times a transition must be observed")
    p.add_argument("--min_out_degree", type=int, default=2, help="Min distinct successors for a decision point")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    
    cfg = BuildCfg(
        min_choice_count=int(args.min_choice_count),
        min_out_degree=int(args.min_out_degree),
        seed=int(args.seed),
    )
    run(
        paths_graph_npz=args.paths_graph_npz,
        road_graph_npz=args.road_graph_npz,
        out_dir=args.out_dir,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
