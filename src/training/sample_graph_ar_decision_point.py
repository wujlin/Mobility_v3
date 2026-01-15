"""
Evaluate Decision Point AR Model (Proposal C).

Evaluation flow:
1. Load trained DP AR model
2. For each test route:
   a. Start from origin decision point
   b. Autoregressively predict next decision points until reaching dest
   c. Use A* to connect consecutive decision points into full node path
3. Compare predicted path vs GT path using Jaccard, etc.

This is the full pipeline: DP AR + A* connection.
"""
from __future__ import annotations

import argparse
import heapq
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.models.road_graph.ar_decision_point import (
    DecisionPointARModelSimple,
    DPARConfig,
)

TZ_SHANGHAI = timezone(timedelta(hours=8))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


@dataclass
class EvalCfg:
    max_dp_steps: int  # max decision point steps before giving up
    top_k: int         # sample from top-k candidates (1 = greedy)
    temperature: float
    seed: int
    device: str
    tz_offset_hours: float


def _load_graph_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    return {
        "node_y": np.asarray(data["node_y"], dtype=np.float32).reshape(-1),
        "node_x": np.asarray(data["node_x"], dtype=np.float32).reshape(-1),
        "edge_u": np.asarray(data["edge_u"], dtype=np.int32).reshape(-1),
        "edge_v": np.asarray(data["edge_v"], dtype=np.int32).reshape(-1),
        "edge_length_m": np.asarray(data["edge_length_m"], dtype=np.float32).reshape(-1) if "edge_length_m" in data.files else None,
    }


def _load_paths_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    return {
        "start_t": np.asarray(data["start_t"], dtype=np.int64).reshape(-1),
        "start_node": np.asarray(data["start_node"], dtype=np.int32).reshape(-1),
        "dest_node": np.asarray(data["dest_node"], dtype=np.int32).reshape(-1),
        "node_seq_pad": np.asarray(data["node_seq_pad"], dtype=np.int32),
        "node_seq_len": np.asarray(data["node_seq_len"], dtype=np.int32).reshape(-1),
        "traj_idx": np.asarray(data["traj_idx"], dtype=np.int64).reshape(-1),
    }


def build_adjacency(edge_u: np.ndarray, edge_v: np.ndarray, edge_length: Optional[np.ndarray], n_nodes: int) -> Dict[int, List[Tuple[int, float]]]:
    """Build adjacency list for A* search."""
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_nodes)}
    for i in range(len(edge_u)):
        u = int(edge_u[i])
        v = int(edge_v[i])
        w = float(edge_length[i]) if edge_length is not None else 1.0
        adj[u].append((v, w))
    return adj


def astar(
    adj: Dict[int, List[Tuple[int, float]]],
    node_y: np.ndarray,
    node_x: np.ndarray,
    start: int,
    goal: int,
    max_steps: int = 10000,
) -> Optional[List[int]]:
    """A* search from start to goal."""
    if start == goal:
        return [start]
    
    def heuristic(n: int) -> float:
        dy = node_y[goal] - node_y[n]
        dx = node_x[goal] - node_x[n]
        return float(np.sqrt(dy*dy + dx*dx)) * 111000  # rough meters
    
    # Priority queue: (f_score, g_score, node, path)
    pq = [(heuristic(start), 0.0, start, [start])]
    visited: Set[int] = set()
    
    steps = 0
    while pq and steps < max_steps:
        steps += 1
        f, g, node, path = heapq.heappop(pq)
        
        if node == goal:
            return path
        
        if node in visited:
            continue
        visited.add(node)
        
        for neighbor, weight in adj[node]:
            if neighbor not in visited:
                new_g = g + weight
                new_f = new_g + heuristic(neighbor)
                heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))
    
    return None


def sample_next_dp(
    model: torch.nn.Module,
    current_dp: int,
    dest_dp: int,
    hour: int,
    dp_y: np.ndarray,
    dp_x: np.ndarray,
    dp_succ_ptr: np.ndarray,
    dp_succ_idx: np.ndarray,
    max_candidates: int,
    device: torch.device,
    top_k: int = 1,
    temperature: float = 1.0,
) -> Optional[int]:
    """Sample next decision point given current state."""
    # Get successors
    start_ptr = int(dp_succ_ptr[current_dp])
    end_ptr = int(dp_succ_ptr[current_dp + 1])
    successors = dp_succ_idx[start_ptr:end_ptr]
    
    if len(successors) == 0:
        return None
    
    n_cand = min(len(successors), max_candidates)
    
    # Build input tensors
    cand_dp = np.full((max_candidates,), -1, dtype=np.int64)
    cand_dp[:n_cand] = successors[:n_cand]
    cand_mask = np.zeros((max_candidates,), dtype=bool)
    cand_mask[:n_cand] = True
    
    cand_y = np.zeros((max_candidates,), dtype=np.float32)
    cand_x = np.zeros((max_candidates,), dtype=np.float32)
    for i in range(n_cand):
        c = int(cand_dp[i])
        cand_y[i] = dp_y[c]
        cand_x[i] = dp_x[c]
    
    # To tensors
    current_dp_t = torch.tensor([current_dp], dtype=torch.long, device=device)
    dest_dp_t = torch.tensor([dest_dp], dtype=torch.long, device=device)
    hour_t = torch.tensor([hour], dtype=torch.long, device=device)
    current_y_t = torch.tensor([dp_y[current_dp]], dtype=torch.float32, device=device)
    current_x_t = torch.tensor([dp_x[current_dp]], dtype=torch.float32, device=device)
    dest_y_t = torch.tensor([dp_y[dest_dp]], dtype=torch.float32, device=device)
    dest_x_t = torch.tensor([dp_x[dest_dp]], dtype=torch.float32, device=device)
    cand_dp_t = torch.tensor(cand_dp, dtype=torch.long, device=device).unsqueeze(0)
    cand_y_t = torch.tensor(cand_y, dtype=torch.float32, device=device).unsqueeze(0)
    cand_x_t = torch.tensor(cand_x, dtype=torch.float32, device=device).unsqueeze(0)
    cand_mask_t = torch.tensor(cand_mask, dtype=torch.bool, device=device).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(
            current_dp=current_dp_t,
            dest_dp=dest_dp_t,
            hour=hour_t,
            current_pos=(current_y_t, current_x_t),
            dest_pos=(dest_y_t, dest_x_t),
            cand_dp=cand_dp_t,
            cand_pos=(cand_y_t, cand_x_t),
            cand_mask=cand_mask_t,
        )  # (1, max_candidates)
    
    logits = logits[0]  # (max_candidates,)
    
    # Apply temperature and sample
    if top_k == 1:
        # Greedy
        idx = logits.argmax().item()
    else:
        # Top-k sampling
        probs = F.softmax(logits / temperature, dim=-1)
        # Zero out non-top-k
        topk_vals, topk_idxs = torch.topk(probs, min(top_k, n_cand))
        probs_filtered = torch.zeros_like(probs)
        probs_filtered[topk_idxs] = topk_vals
        probs_filtered = probs_filtered / probs_filtered.sum()
        idx = torch.multinomial(probs_filtered, 1).item()
    
    return int(cand_dp[idx])


def generate_route(
    model: torch.nn.Module,
    start_node: int,
    dest_node: int,
    hour: int,
    decision_points: np.ndarray,  # node_id -> is_dp
    dp_to_idx: Dict[int, int],
    idx_to_dp: Dict[int, int],
    dp_y: np.ndarray,
    dp_x: np.ndarray,
    dp_succ_ptr: np.ndarray,
    dp_succ_idx: np.ndarray,
    adj: Dict[int, List[Tuple[int, float]]],
    node_y: np.ndarray,
    node_x: np.ndarray,
    cfg: EvalCfg,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    """
    Generate a route using DP AR + A*.
    
    Returns:
        dp_sequence: list of decision point node_ids
        full_path: list of node_ids (A* connected)
    """
    # Find closest decision points to start and dest
    # In practice, start_node and dest_node should be in decision_points (added during build)
    start_dp_idx = dp_to_idx.get(start_node)
    dest_dp_idx = dp_to_idx.get(dest_node)
    
    if start_dp_idx is None or dest_dp_idx is None:
        # Fallback: find nearest dp
        dp_node_set = set(decision_points.tolist())
        if start_dp_idx is None:
            # A* to nearest dp
            for dp_node in decision_points:
                path = astar(adj, node_y, node_x, start_node, int(dp_node), max_steps=1000)
                if path is not None:
                    start_dp_idx = dp_to_idx[int(dp_node)]
                    break
        if dest_dp_idx is None:
            for dp_node in decision_points:
                path = astar(adj, node_y, node_x, int(dp_node), dest_node, max_steps=1000)
                if path is not None:
                    dest_dp_idx = dp_to_idx[int(dp_node)]
                    break
    
    if start_dp_idx is None or dest_dp_idx is None:
        return [], []
    
    # Autoregressive generation of decision point sequence
    dp_sequence_idx = [start_dp_idx]
    current_dp_idx = start_dp_idx
    
    for step in range(cfg.max_dp_steps):
        if current_dp_idx == dest_dp_idx:
            break
        
        next_dp_idx = sample_next_dp(
            model=model,
            current_dp=current_dp_idx,
            dest_dp=dest_dp_idx,
            hour=hour,
            dp_y=dp_y,
            dp_x=dp_x,
            dp_succ_ptr=dp_succ_ptr,
            dp_succ_idx=dp_succ_idx,
            max_candidates=cfg.top_k * 2 + 10,  # enough candidates
            device=device,
            top_k=cfg.top_k,
            temperature=cfg.temperature,
        )
        
        if next_dp_idx is None:
            break
        
        if next_dp_idx in dp_sequence_idx:
            # Loop detected, stop
            break
        
        dp_sequence_idx.append(next_dp_idx)
        current_dp_idx = next_dp_idx
    
    # Convert dp indices to node ids
    dp_sequence = [int(decision_points[i]) for i in dp_sequence_idx]
    
    # A* connect consecutive decision points
    full_path: List[int] = []
    for i in range(len(dp_sequence) - 1):
        u = dp_sequence[i]
        v = dp_sequence[i + 1]
        segment = astar(adj, node_y, node_x, u, v, max_steps=5000)
        if segment is None:
            # Can't connect, skip this segment
            continue
        if full_path and segment[0] == full_path[-1]:
            segment = segment[1:]
        full_path.extend(segment)
    
    # Connect last dp to dest if needed
    if full_path and full_path[-1] != dest_node:
        segment = astar(adj, node_y, node_x, full_path[-1], dest_node, max_steps=5000)
        if segment is not None:
            if segment[0] == full_path[-1]:
                segment = segment[1:]
            full_path.extend(segment)
    
    # Connect start if needed
    if full_path and full_path[0] != start_node:
        segment = astar(adj, node_y, node_x, start_node, full_path[0], max_steps=5000)
        if segment is not None:
            full_path = segment[:-1] + full_path
    
    return dp_sequence, full_path


def jaccard(pred: List[int], gt: List[int]) -> float:
    pred_set = set(pred)
    gt_set = set(gt)
    if not pred_set and not gt_set:
        return 1.0
    if not pred_set or not gt_set:
        return 0.0
    return len(pred_set & gt_set) / len(pred_set | gt_set)


def run(
    *,
    model_path: Path,
    dp_graph_npz: Path,
    road_graph_npz: Path,
    paths_graph_npz: Path,
    out_dir: Path,
    cfg: EvalCfg,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "report.json"
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    
    # Load model
    log.info(f"Loading model from {model_path}")
    ckpt = torch.load(str(model_path), map_location=device)
    model_cfg = DPARConfig(**ckpt["model_cfg"])
    n_dp = ckpt["n_decision_points"]
    model = DecisionPointARModelSimple(model_cfg, n_decision_points=n_dp)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Load decision point graph
    log.info(f"Loading DP graph from {dp_graph_npz}")
    dp_data = np.load(str(dp_graph_npz), allow_pickle=True)
    decision_points = np.asarray(dp_data["decision_points"], dtype=np.int64)
    dp_y = np.asarray(dp_data["dp_y"], dtype=np.float32)
    dp_x = np.asarray(dp_data["dp_x"], dtype=np.float32)
    dp_succ_ptr = np.asarray(dp_data["dp_succ_ptr"], dtype=np.int64)
    dp_succ_idx = np.asarray(dp_data["dp_succ_idx"], dtype=np.int32)
    
    dp_to_idx = {int(dp): i for i, dp in enumerate(decision_points.tolist())}
    idx_to_dp = {i: int(dp) for i, dp in enumerate(decision_points.tolist())}
    
    # Load road graph for A*
    log.info(f"Loading road graph from {road_graph_npz}")
    g = _load_graph_npz(road_graph_npz)
    node_y = g["node_y"]
    node_x = g["node_x"]
    n_nodes = len(node_y)
    adj = build_adjacency(g["edge_u"], g["edge_v"], g["edge_length_m"], n_nodes)
    
    # Load test paths
    log.info(f"Loading paths from {paths_graph_npz}")
    p = _load_paths_npz(paths_graph_npz)
    node_seq_pad = p["node_seq_pad"]
    node_seq_len = p["node_seq_len"]
    start_node = p["start_node"]
    dest_node = p["dest_node"]
    start_t = p["start_t"]
    n_routes = node_seq_len.shape[0]
    
    # Evaluate
    log.info(f"Evaluating {n_routes} routes...")
    results = []
    
    for rid in range(n_routes):
        L = int(node_seq_len[rid])
        if L < 2:
            continue
        
        gt_seq = node_seq_pad[rid, :L].astype(np.int64, copy=False).tolist()
        s = int(start_node[rid])
        d = int(dest_node[rid])
        ts = int(start_t[rid])
        offset_sec = int(round(float(cfg.tz_offset_hours) * 3600.0))
        sec = int((ts + offset_sec) % 86400)
        hour = int(sec // 3600)
        
        dp_seq, pred_path = generate_route(
            model=model,
            start_node=s,
            dest_node=d,
            hour=hour,
            decision_points=decision_points,
            dp_to_idx=dp_to_idx,
            idx_to_dp=idx_to_dp,
            dp_y=dp_y,
            dp_x=dp_x,
            dp_succ_ptr=dp_succ_ptr,
            dp_succ_idx=dp_succ_idx,
            adj=adj,
            node_y=node_y,
            node_x=node_x,
            cfg=cfg,
            device=device,
        )
        
        jac = jaccard(pred_path, gt_seq)
        reached_dest = pred_path and pred_path[-1] == d
        
        results.append({
            "rid": rid,
            "jaccard": jac,
            "reached_dest": reached_dest,
            "n_dp_steps": len(dp_seq),
            "pred_len": len(pred_path),
            "gt_len": len(gt_seq),
        })
        
        if (rid + 1) % 100 == 0:
            avg_jac = np.mean([r["jaccard"] for r in results])
            reach_rate = np.mean([r["reached_dest"] for r in results])
            log.info(f"Progress: {rid+1}/{n_routes}, Avg Jaccard: {avg_jac:.4f}, Reach rate: {reach_rate:.4f}")
    
    # Aggregate stats
    jaccards = [r["jaccard"] for r in results]
    reach_rates = [r["reached_dest"] for r in results]
    dp_steps = [r["n_dp_steps"] for r in results]
    
    stats = {
        "n_routes_eval": len(results),
        "jaccard_mean": float(np.mean(jaccards)),
        "jaccard_std": float(np.std(jaccards)),
        "jaccard_median": float(np.median(jaccards)),
        "reach_dest_rate": float(np.mean(reach_rates)),
        "dp_steps_mean": float(np.mean(dp_steps)),
        "dp_steps_max": int(np.max(dp_steps)),
    }
    
    report = {
        "ok": True,
        "task": "sample_graph_ar_decision_point",
        "inputs": {
            "model_path": str(model_path),
            "dp_graph_npz": str(dp_graph_npz),
            "road_graph_npz": str(road_graph_npz),
            "paths_graph_npz": str(paths_graph_npz),
        },
        "config": {
            "max_dp_steps": cfg.max_dp_steps,
            "top_k": cfg.top_k,
            "temperature": cfg.temperature,
            "seed": cfg.seed,
        },
        "stats": stats,
        "meta": {"created_at": datetime.now(tz=TZ_SHANGHAI).isoformat()},
    }
    
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Evaluation complete.")
    log.info(f"Jaccard: {stats['jaccard_mean']:.4f} ± {stats['jaccard_std']:.4f}")
    log.info(f"Reach dest rate: {stats['reach_dest_rate']:.4f}")
    log.info(f"Avg DP steps: {stats['dp_steps_mean']:.2f}")
    
    print(json.dumps(report, ensure_ascii=False))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Decision Point AR Model")
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--dp_graph_npz", type=Path, required=True)
    p.add_argument("--road_graph_npz", type=Path, required=True)
    p.add_argument("--paths_graph_npz", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--max_dp_steps", type=int, default=30)
    p.add_argument("--top_k", type=int, default=1, help="1=greedy, >1=top-k sampling")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    args = p.parse_args()
    
    cfg = EvalCfg(
        max_dp_steps=args.max_dp_steps,
        top_k=args.top_k,
        temperature=args.temperature,
        seed=args.seed,
        device=args.device,
        tz_offset_hours=float(args.tz_offset_hours),
    )
    
    run(
        model_path=args.model_path,
        dp_graph_npz=args.dp_graph_npz,
        road_graph_npz=args.road_graph_npz,
        paths_graph_npz=args.paths_graph_npz,
        out_dir=args.out_dir,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
