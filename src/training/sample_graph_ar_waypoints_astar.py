from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(x, *args, **kwargs):  # type: ignore[no-redef]
        return x

from src.data.road_graph.gate_candidate_paths_from_routes_npz import _astar, _load_graph_npz, k_shortest_paths_yen
from src.features.waypoints import pick_waypoint_indices_rdp_turn_fixed_k
from src.models.road_graph import ARGraphWaypointBin, WaypointBinARConfig
from src.training.train_graph_ar_waypoint_bins import _time_features
from src.plot_style import OKABE_ITO, paper_style, save_figure


TZ_SHANGHAI = timezone(timedelta(hours=8))
_ASTAR_G = None


def _set_seed(seed: int, *, seed_cuda: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    # IMPORTANT: if using multiprocessing with "fork", do NOT touch torch.cuda.* before the pool starts.
    # (Calling torch.cuda.manual_seed_all() may initialize CUDA and make fork unsafe.)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _seq_from_pad(pad: np.ndarray, lens: np.ndarray, i: int) -> List[int]:
    L = int(lens[i])
    if L <= 0:
        return []
    s = pad[i, :L].astype(np.int64, copy=False).tolist()
    return [int(x) for x in s if int(x) >= 0]


def _edge_set(seq: Sequence[int]) -> set[Tuple[int, int]]:
    out: set[Tuple[int, int]] = set()
    for a, b in zip(seq[:-1], seq[1:]):
        aa = int(a)
        bb = int(b)
        if aa >= 0 and bb >= 0 and aa != bb:
            out.add((aa, bb))
    return out


def _jaccard_edges(a: set[Tuple[int, int]], b: set[Tuple[int, int]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    denom = len(a) + len(b) - inter
    return float(inter) / float(max(1, denom))


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros((0,), dtype=np.float64)
    x = x - float(np.max(x))
    ex = np.exp(np.clip(x, -60.0, 60.0))
    s = float(np.sum(ex))
    if not np.isfinite(s) or s <= 0:
        return np.full_like(ex, 1.0 / float(max(1, ex.size)))
    return (ex / s).astype(np.float64, copy=False)


def _build_node_tier_min(n_nodes: int, edge_u: np.ndarray, edge_tier: np.ndarray) -> np.ndarray:
    out = np.full((int(n_nodes),), 3, dtype=np.int64)
    u = np.asarray(edge_u, dtype=np.int64).reshape(-1)
    t = np.asarray(edge_tier, dtype=np.int64).reshape(-1)
    for uu, tt in zip(u.tolist(), t.tolist()):
        if 0 <= int(uu) < int(n_nodes):
            out[int(uu)] = min(int(out[int(uu)]), int(np.clip(int(tt), 0, 3)))
    return out


def _bin_id_from_yx(y: float, x: float, *, wp_bin: int, H: int, W: int) -> Tuple[int, int, int]:
    wp_bin = int(wp_bin)
    n_by = int((int(H) + wp_bin - 1) // wp_bin)
    n_bx = int((int(W) + wp_bin - 1) // wp_bin)
    by = int(np.clip(math.floor(float(y) / float(wp_bin)), 0, n_by - 1))
    bx = int(np.clip(math.floor(float(x) / float(wp_bin)), 0, n_bx - 1))
    cls = by * int(n_bx) + bx
    return cls, n_by, n_bx


def _build_bin_to_nodes(*, node_y: np.ndarray, node_x: np.ndarray, wp_bin: int, H: int, W: int) -> Tuple[List[np.ndarray], int, int]:
    wp_bin = int(wp_bin)
    n_by = int((int(H) + wp_bin - 1) // wp_bin)
    n_bx = int((int(W) + wp_bin - 1) // wp_bin)
    bins: List[List[int]] = [[] for _ in range(int(n_by * n_bx))]
    for nid, (yy, xx) in enumerate(zip(node_y.tolist(), node_x.tolist())):
        cls, _, _ = _bin_id_from_yx(float(yy), float(xx), wp_bin=wp_bin, H=H, W=W)
        bins[int(cls)].append(int(nid))
    out = [np.asarray(v, dtype=np.int64) if v else np.zeros((0,), dtype=np.int64) for v in bins]
    return out, int(n_by), int(n_bx)


def _build_bin_to_nodes_by_city(
    *,
    node_y: np.ndarray,
    node_x: np.ndarray,
    node_city: np.ndarray,
    wp_bin: int,
    H: int,
    W: int,
) -> Tuple[List[List[np.ndarray]], int, int]:
    wp_bin = int(wp_bin)
    n_by = int((int(H) + wp_bin - 1) // wp_bin)
    n_bx = int((int(W) + wp_bin - 1) // wp_bin)
    node_city_i = np.asarray(node_city, dtype=np.int64).reshape(-1)
    num_cities = int(np.max(node_city_i)) + 1 if int(node_city_i.size) > 0 else 1

    bins: List[List[List[int]]] = [[[] for _ in range(int(n_by * n_bx))] for _ in range(int(num_cities))]
    for nid, (yy, xx, cc) in enumerate(zip(node_y.tolist(), node_x.tolist(), node_city_i.tolist())):
        city = int(cc)
        if city < 0 or city >= int(num_cities):
            continue
        cls, _, _ = _bin_id_from_yx(float(yy), float(xx), wp_bin=wp_bin, H=H, W=W)
        bins[int(city)][int(cls)].append(int(nid))

    out_by_city: List[List[np.ndarray]] = []
    for c in range(int(num_cities)):
        out_by_city.append([np.asarray(v, dtype=np.int64) if v else np.zeros((0,), dtype=np.int64) for v in bins[int(c)]])
    return out_by_city, int(n_by), int(n_bx)


def _pick_node_in_bin(
    *,
    bin_nodes: np.ndarray,
    node_y: np.ndarray,
    node_x: np.ndarray,
    by: int,
    bx: int,
    wp_bin: int,
    rng: np.random.Generator,
) -> Optional[int]:
    if bin_nodes.size == 0:
        return None
    # Prefer nodes near bin center to avoid sampling tiny dead-ends.
    cy = float(by * int(wp_bin) + int(wp_bin) * 0.5)
    cx = float(bx * int(wp_bin) + int(wp_bin) * 0.5)
    yy = node_y[bin_nodes].astype(np.float64, copy=False)
    xx = node_x[bin_nodes].astype(np.float64, copy=False)
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    order = np.argsort(d2, kind="mergesort")
    top = bin_nodes[order[: min(int(order.size), 64)]]
    return int(rng.choice(top))


def _pick_node_in_bin_tier_dir(
    *,
    bin_nodes: np.ndarray,
    node_y: np.ndarray,
    node_x: np.ndarray,
    node_tier_min: np.ndarray,
    prev_y: float,
    prev_x: float,
    dest_y: float,
    dest_x: float,
    by: int,
    bx: int,
    wp_bin: int,
    rng: np.random.Generator,
) -> Optional[int]:
    if bin_nodes.size == 0:
        return None

    tiers = node_tier_min[bin_nodes].astype(np.int64, copy=False)
    major_mask = tiers <= 1
    candidates = bin_nodes[major_mask] if bool(np.any(major_mask)) else bin_nodes

    # Keep a small set near the bin center to avoid tiny dead-ends.
    cy = float(by * int(wp_bin) + int(wp_bin) * 0.5)
    cx = float(bx * int(wp_bin) + int(wp_bin) * 0.5)
    yy = node_y[candidates].astype(np.float64, copy=False)
    xx = node_x[candidates].astype(np.float64, copy=False)
    d2_center = (yy - cy) ** 2 + (xx - cx) ** 2
    order = np.argsort(d2_center, kind="mergesort")
    candidates = candidates[order[: min(int(order.size), 128)]]
    if candidates.size == 0:
        return None

    # Prefer nodes aligned with the (prev -> dest) direction.
    dy = float(dest_y - prev_y)
    dx = float(dest_x - prev_x)
    norm = float(math.hypot(dy, dx)) + 1e-6
    dy /= norm
    dx /= norm
    yy = node_y[candidates].astype(np.float64, copy=False)
    xx = node_x[candidates].astype(np.float64, copy=False)
    proj = (yy - float(prev_y)) * dy + (xx - float(prev_x)) * dx

    top_k = min(8, int(candidates.size))
    if top_k <= 1:
        return int(candidates[0])
    idx = np.argsort(-proj, kind="mergesort")[:top_k]
    return int(rng.choice(candidates[idx]))


@dataclass(frozen=True)
class SampleCfg:
    K: int
    temperature: float
    max_steps: int


def _astar_worker(pair: Tuple[int, int]) -> List[int]:
    global _ASTAR_G
    if _ASTAR_G is None:
        raise RuntimeError("A* worker graph is not initialized.")
    s, d = pair
    _, path = _astar(_ASTAR_G, start=int(s), goal=int(d))
    return [int(x) for x in path]


def _kshortest_worker(args: Tuple[int, int, int]) -> List[List[int]]:
    """Worker for parallel K-shortest paths computation."""
    global _ASTAR_G
    if _ASTAR_G is None:
        raise RuntimeError("A* worker graph is not initialized.")
    s, d, K = args
    return k_shortest_paths_yen(_ASTAR_G, start=int(s), goal=int(d), K=int(K))


def _plot_case(
    *,
    out_png: Path,
    out_pdf: Path,
    node_y: np.ndarray,
    node_x: np.ndarray,
    gt_bucket_seqs: Optional[Sequence[Sequence[int]]],
    gt_seq: Sequence[int],
    pred_paths: Sequence[Sequence[int]],
    gt_wp: Optional[Sequence[int]],
    pred_wps: Sequence[Sequence[int]],
    title: str,
) -> None:
    gt = np.asarray(gt_seq, dtype=np.int64)
    with paper_style():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.2))
        # Optional: overlay other GT routes from the same OD-bin bucket.
        # This avoids misreading "single GT vs many samples" as corridor-diversity evidence.
        if gt_bucket_seqs is not None:
            for s in gt_bucket_seqs:
                ss = np.asarray(list(s), dtype=np.int64)
                if ss.size < 2:
                    continue
                ax.plot(node_x[ss], node_y[ss], color=OKABE_ITO["gray"], lw=1.0, alpha=0.12)
        ax.plot(node_x[gt], node_y[gt], color="black", lw=2.0, alpha=0.9, label="GT")
        for p in pred_paths:
            pp = np.asarray(p, dtype=np.int64)
            ax.plot(node_x[pp], node_y[pp], color=OKABE_ITO["blue"], lw=1.0, alpha=0.22)
        # GT waypoints (optional)
        if gt_wp is not None and len(gt_wp) >= 2:
            w = np.asarray(gt_wp, dtype=np.int64)
            ax.scatter(node_x[w], node_y[w], s=20, c="black", edgecolors="white", linewidths=0.5, zorder=10, label="GT-WP")
        # Pred waypoint clouds
        for wps in pred_wps:
            if not wps:
                continue
            w = np.asarray(wps, dtype=np.int64)
            ax.scatter(node_x[w], node_y[w], s=8, c=OKABE_ITO["sky_blue"], alpha=0.25, linewidths=0.0)

        ax.scatter([node_x[gt[0]]], [node_y[gt[0]]], s=60, c="black", edgecolors="white", linewidths=1.0, zorder=10)
        ax.scatter([node_x[gt[-1]]], [node_y[gt[-1]]], s=60, c="black", marker="s", edgecolors="white", linewidths=1.0, zorder=10)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
        save_figure(fig, out_png, dpi=250)
        save_figure(fig, out_pdf)
        plt.close(fig)


def run(
    *,
    checkpoint: Path,
    road_graph_npz: Path,
    paths_graph_npz: Path,
    waypoints_npz: Optional[Path],
    out_dir: Path,
    K: int,
    temperature: float,
    num_routes: int,
    baseline_k: int,
    oracle: bool,
    pick_strategy: str,
    astar_workers: int,
    tz_offset_hours: float,
    seed: int,
    viz_cases: int,
    viz_gt_od_bin: int,
    viz_gt_max: int,
    progress: str,
    log_every: int,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "report.json"

    # Graph (A* + optional Yen baseline)
    g = _load_graph_npz(road_graph_npz)
    H = int(g.grid.H)
    W = int(g.grid.W)

    pool = None
    astar_workers_i = int(astar_workers)
    if astar_workers_i < 0:
        astar_workers_i = max(1, int(mp.cpu_count()) - 2)
    if astar_workers_i > 0:
        # IMPORTANT: use fork and start the pool BEFORE any CUDA initialization.
        # Workers only run A* (pure Python CPU), so fork is safe and avoids pickling the graph.
        ctx = mp.get_context("fork")
        global _ASTAR_G
        _ASTAR_G = g
        pool = ctx.Pool(processes=int(astar_workers_i))

    raw = np.load(str(road_graph_npz), allow_pickle=True)
    node_y = np.asarray(raw["node_y"], dtype=np.float32).reshape(-1)
    node_x = np.asarray(raw["node_x"], dtype=np.float32).reshape(-1)
    edge_u = np.asarray(raw["edge_u"], dtype=np.int32).reshape(-1)
    edge_tier = np.asarray(raw["edge_tier"], dtype=np.uint8).reshape(-1)
    node_city = np.asarray(raw["node_city"], dtype=np.int8).reshape(-1) if "node_city" in raw.files else None
    n_nodes = int(node_y.size)

    node_tier_min = _build_node_tier_min(n_nodes, edge_u=edge_u, edge_tier=edge_tier)

    p = np.load(str(paths_graph_npz), allow_pickle=True)
    node_seq_pad = np.asarray(p["node_seq_pad"], dtype=np.int32)
    node_seq_len = np.asarray(p["node_seq_len"], dtype=np.int32).reshape(-1)
    start_t = np.asarray(p["start_t"], dtype=np.int64).reshape(-1)
    start_pos = np.asarray(p["start_pos"], dtype=np.float32).reshape(-1, 2) if "start_pos" in p.files else None
    dest_pos = np.asarray(p["dest_pos"], dtype=np.float32).reshape(-1, 2) if "dest_pos" in p.files else None
    start_node = np.asarray(p["start_node"], dtype=np.int32).reshape(-1)
    dest_node = np.asarray(p["dest_node"], dtype=np.int32).reshape(-1)
    route_city = np.asarray(p["route_city"], dtype=np.int8).reshape(-1) if "route_city" in p.files else np.zeros_like(start_node, dtype=np.int8)
    n_routes_total = int(start_node.size)

    gt_wp_seq = None
    if waypoints_npz is not None and waypoints_npz.exists():
        w = np.load(str(waypoints_npz), allow_pickle=True)
        if "wp_seq" in w.files:
            gt_wp_seq = np.asarray(w["wp_seq"], dtype=np.int32)

    progress_mode = str(progress)
    if progress_mode == "auto":
        progress_mode = "tqdm" if bool(sys.stderr.isatty()) else "json"
    if progress_mode not in {"tqdm", "json", "none"}:
        raise ValueError(f"--progress must be one of auto|tqdm|json|none, got {progress!r}")

    try:
        # Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(checkpoint), map_location="cpu")
        model_cfg = ckpt.get("model_cfg") or {}
        num_steps = int(model_cfg.get("num_steps", 4))
        n_classes = int(model_cfg.get("n_classes", 1024))
        wp_bin = int(model_cfg.get("wp_bin", 32))
        num_cities = int(model_cfg.get("num_cities", 2))

        model = ARGraphWaypointBin(cfg=WaypointBinARConfig(hidden_dim=int(model_cfg.get("hidden_dim", 256)), num_cities=int(num_cities)), n_classes=n_classes, num_steps=num_steps).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        node_yx = torch.from_numpy(np.stack([node_y, node_x], axis=1).astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)
        node_tier_min_t = torch.from_numpy(node_tier_min.astype(np.int64, copy=False)).to(device=device, dtype=torch.long)
        tf = _time_features(start_t, tz_offset_hours=float(tz_offset_hours))

        # Bin index (for sampling bins -> nodes). If `node_city` exists, filter candidates by route city
        # to avoid mixing disjoint city subgraphs that share the same normalized grid coordinate system.
        bin_to_nodes = None
        bin_to_nodes_by_city = None
        if node_city is not None and int(node_city.size) == int(n_nodes):
            bin_to_nodes_by_city, n_by, n_bx = _build_bin_to_nodes_by_city(node_y=node_y, node_x=node_x, node_city=node_city, wp_bin=wp_bin, H=H, W=W)
        else:
            bin_to_nodes, n_by, n_bx = _build_bin_to_nodes(node_y=node_y, node_x=node_x, wp_bin=wp_bin, H=H, W=W)
        if int(n_by * n_bx) != int(n_classes):
            # Keep going but report mismatch.
            pass

        rng = np.random.default_rng(int(seed))

        # For visualization only: multi-GT overlay within the same OD-bin bucket.
        gt_bucket_keys = None
        gt_bucket_map = None
        od_bin_i = int(viz_gt_od_bin)
        if od_bin_i > 0:
            if start_pos is None or dest_pos is None:
                raise ValueError("--viz_gt_od_bin requires start_pos/dest_pos in paths_graph_npz")
            b = float(max(1, od_bin_i))
            s_bin = np.floor(start_pos.astype(np.float64) / b).astype(np.int32)
            d_bin = np.floor(dest_pos.astype(np.float64) / b).astype(np.int32)
            gt_bucket_keys = np.concatenate([route_city[:, None].astype(np.int32), s_bin, d_bin], axis=1)  # (N,5)
            view = gt_bucket_keys.view([("", gt_bucket_keys.dtype)] * gt_bucket_keys.shape[1]).reshape(-1)
            order = np.argsort(view, kind="mergesort")
            keys_s = gt_bucket_keys[order]
            idx_s = order.astype(np.int64, copy=False)
            gt_bucket_map = {}
            i = 0
            n0 = int(keys_s.shape[0])
            while i < n0:
                j = i + 1
                while j < n0 and np.array_equal(keys_s[j], keys_s[i]):
                    j += 1
                gt_bucket_map[tuple(int(x) for x in keys_s[i].tolist())] = idx_s[i:j].copy()
                i = j
        pick = rng.choice(n_routes_total, size=int(min(int(num_routes), n_routes_total)), replace=False)
        pick = np.sort(pick.astype(np.int64))
        pick_list = pick.tolist()
        total = int(len(pick_list))
        t0 = time.time()

        od_cache: Dict[Tuple[int, int, int], List[List[int]]] = {}

        # Pre-compute K-shortest paths in parallel if baseline_k > 0
        if int(baseline_k) > 0 and pool is not None:
            # Collect unique OD pairs for K-shortest computation
            kshortest_args = []
            kshortest_keys = []
            for rid in pick_list:
                s = int(start_node[int(rid)])
                d = int(dest_node[int(rid)])
                key = (s, d, int(baseline_k))
                if key not in od_cache and key not in kshortest_keys:
                    kshortest_keys.append(key)
                    kshortest_args.append((s, d, int(baseline_k)))
            if kshortest_args:
                print(json.dumps({"event": "precompute_kshortest", "n_od_pairs": len(kshortest_args), "K": int(baseline_k)}, ensure_ascii=False), flush=True)
                kshortest_results = pool.map(_kshortest_worker, kshortest_args)
                for key, paths in zip(kshortest_keys, kshortest_results):
                    od_cache[key] = paths
                print(json.dumps({"event": "precompute_kshortest_done", "n_od_pairs": len(kshortest_args)}, ensure_ascii=False), flush=True)

        rows = []
        best_j_list = []
        best_j_succ_list = []
        succ_rate_list = []
        base_best_list = []

        cfg_s = SampleCfg(K=int(K), temperature=float(temperature), max_steps=4096)

        viz = 0
        it = pick_list
        if progress_mode == "tqdm":
            it = tqdm(it, desc="sample_wp_astar", dynamic_ncols=True)  # type: ignore[assignment]
        for i_idx, rid in enumerate(it):
            gt_seq = _seq_from_pad(node_seq_pad, node_seq_len, int(rid))
            if len(gt_seq) < 2:
                continue
            gt_es = _edge_set(gt_seq)
            s = int(start_node[int(rid)])
            d = int(dest_node[int(rid)])
            city = int(route_city[int(rid)])
            time_feat = torch.from_numpy(tf[int(rid) : int(rid) + 1].astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)

            gt_wp_bins: Optional[List[int]] = None
            if bool(oracle):
                gt_wp_full: Optional[List[int]] = None
                if gt_wp_seq is not None and int(rid) < int(gt_wp_seq.shape[0]):
                    gt_wp_full = [int(x) for x in gt_wp_seq[int(rid)].astype(np.int64, copy=False).tolist() if int(x) >= 0]

                if gt_wp_full is None or len(gt_wp_full) < int(num_steps) + 2:
                    pts = np.stack([node_y[np.asarray(gt_seq, dtype=np.int64)], node_x[np.asarray(gt_seq, dtype=np.int64)]], axis=1).astype(np.float32, copy=False)
                    idx = pick_waypoint_indices_rdp_turn_fixed_k(pts, k=int(num_steps), turn_alpha=1.0)
                    gt_wp_full = [int(gt_seq[0])] + [int(gt_seq[int(j)]) for j in idx.tolist()] + [int(gt_seq[-1])]

                # Expect: [O, w1, ..., wK, D]
                internal = gt_wp_full[1:-1]
                if len(internal) != int(num_steps):
                    # If a mismatch still happens, fall back to "no oracle bins" for this route.
                    gt_wp_bins = None
                else:
                    gt_wp_bins = []
                    for nid in internal:
                        cls, _, _ = _bin_id_from_yx(float(node_y[int(nid)]), float(node_x[int(nid)]), wp_bin=wp_bin, H=H, W=W)
                        gt_wp_bins.append(int(cls))

            # 1) sample K waypoint sequences (cheap; GPU/CPU)
            wps_list: List[List[int]] = []
            ok_mask: List[bool] = []
            for _ in range(int(cfg_s.K)):
                cur = int(s)
                wps: List[int] = []
                ok = True
                for step in range(int(num_steps)):
                    if bool(oracle):
                        if gt_wp_bins is None:
                            ok = False
                            break
                        cls = int(gt_wp_bins[int(step)])
                    else:
                        with torch.no_grad():
                            logits, _ = model(
                                node_yx=node_yx,
                                node_tier_min=node_tier_min_t,
                                cur=torch.tensor([cur], device=device, dtype=torch.long),
                                dest=torch.tensor([d], device=device, dtype=torch.long),
                                time_feat=time_feat,
                                route_city=torch.tensor([city], device=device, dtype=torch.long),
                                step_idx=torch.tensor([step], device=device, dtype=torch.long),
                            )
                            logits_np = logits.detach().cpu().numpy().reshape(-1)

                        temp = float(cfg_s.temperature)
                        if temp <= 0:
                            cls = int(np.argmax(logits_np))
                        else:
                            prob = _softmax_np(logits_np / max(1e-6, temp))
                            cls = int(rng.choice(int(prob.size), p=prob))

                    by = int(cls // int(n_bx))
                    bx = int(cls % int(n_bx))
                    if bin_to_nodes_by_city is not None:
                        city_clamped = int(np.clip(int(city), 0, int(len(bin_to_nodes_by_city)) - 1))
                        cand_nodes = bin_to_nodes_by_city[int(city_clamped)][int(cls)]
                    else:
                        cand_nodes = bin_to_nodes[int(cls)] if bin_to_nodes is not None else np.zeros((0,), dtype=np.int64)

                    if str(pick_strategy) == "tier_dir":
                        nxt = _pick_node_in_bin_tier_dir(
                            bin_nodes=cand_nodes,
                            node_y=node_y,
                            node_x=node_x,
                            node_tier_min=node_tier_min,
                            prev_y=float(node_y[int(cur)]),
                            prev_x=float(node_x[int(cur)]),
                            dest_y=float(node_y[int(d)]),
                            dest_x=float(node_x[int(d)]),
                            by=by,
                            bx=bx,
                            wp_bin=wp_bin,
                            rng=rng,
                        )
                    else:
                        nxt = _pick_node_in_bin(bin_nodes=cand_nodes, node_y=node_y, node_x=node_x, by=by, bx=bx, wp_bin=wp_bin, rng=rng)
                    if nxt is None:
                        ok = False
                        break
                    wps.append(int(nxt))
                    cur = int(nxt)
                wps_list.append(wps)
                ok_mask.append(bool(ok))

            # 2) Connect with A* (CPU-heavy). Deduplicate segment pairs per route.
            pair_to_idx: Dict[Tuple[int, int], int] = {}
            pairs: List[Tuple[int, int]] = []
            seg_pairs_by_sample: List[List[Tuple[int, int]]] = []
            for wps, ok in zip(wps_list, ok_mask):
                if not ok:
                    seg_pairs_by_sample.append([])
                    continue
                segs = []
                prev = int(s)
                for nxt in wps + [int(d)]:
                    pair = (int(prev), int(nxt))
                    segs.append(pair)
                    if pair not in pair_to_idx:
                        pair_to_idx[pair] = len(pairs)
                        pairs.append(pair)
                    prev = int(nxt)
                seg_pairs_by_sample.append(segs)

            seg_paths: List[List[int]] = [[] for _ in range(len(pairs))]
            if pairs:
                if pool is not None:
                    seg_paths = pool.map(_astar_worker, pairs)
                else:
                    for j, (uu, vv) in enumerate(pairs):
                        _, path = _astar(g, start=int(uu), goal=int(vv))
                        seg_paths[j] = [int(x) for x in path]

            # 3) assemble K full paths and compute metrics
            pred_paths: List[List[int]] = []
            pred_wps = wps_list
            succ = 0
            best_j = 0.0
            best_j_succ = 0.0
            for wps, ok, segs in zip(wps_list, ok_mask, seg_pairs_by_sample):
                if not ok or not segs:
                    pred_paths.append([int(s)])
                    continue
                full: List[int] = [int(s)]
                ok2 = True
                for uu, vv in segs:
                    idx_seg = pair_to_idx[(int(uu), int(vv))]
                    seg = seg_paths[int(idx_seg)]
                    if not seg:
                        ok2 = False
                        break
                    full.extend([int(x) for x in seg[1:]])
                is_succ = bool(ok2 and full and full[-1] == int(d))
                if is_succ:
                    succ += 1
                pred_paths.append(full)
                j = _jaccard_edges(_edge_set(full), gt_es)
                best_j = float(max(best_j, j))
                if is_succ:
                    best_j_succ = float(max(best_j_succ, j))

            succ_rate = float(succ) / float(max(1, int(cfg_s.K)))

            base_best = None
            if int(baseline_k) > 0:
                key = (s, d, int(baseline_k))
                if key not in od_cache:
                    od_cache[key] = k_shortest_paths_yen(g, start=s, goal=d, K=int(baseline_k))
                bj = 0.0
                for path in od_cache[key]:
                    bj = max(bj, _jaccard_edges(_edge_set(path), gt_es))
                base_best = float(bj)

            rows.append(
                {
                    "route_id": int(rid),
                    "city": int(city),
                    "start": int(s),
                    "dest": int(d),
                    "gt_len": int(len(gt_seq)),
                    "best_jaccard": float(best_j),
                    "best_jaccard_success": float(best_j_succ),
                    "success_rate": float(succ_rate),
                    "baseline_best_jaccard_kshortest": (float(base_best) if base_best is not None else None),
                }
            )
            best_j_list.append(float(best_j))
            best_j_succ_list.append(float(best_j_succ))
            succ_rate_list.append(float(succ_rate))
            if base_best is not None:
                base_best_list.append(float(base_best))

            if int(viz_cases) > 0 and viz < int(viz_cases):
                name = f"case_{int(viz):02d}_rid{int(rid)}"
                gt_wp = None
                if gt_wp_seq is not None and int(rid) < int(gt_wp_seq.shape[0]):
                    gt_wp = [int(x) for x in gt_wp_seq[int(rid)].astype(np.int64, copy=False).tolist() if int(x) >= 0]
                else:
                    pts = np.stack([node_y[np.asarray(gt_seq, dtype=np.int64)], node_x[np.asarray(gt_seq, dtype=np.int64)]], axis=1).astype(np.float32, copy=False)
                    idx = pick_waypoint_indices_rdp_turn_fixed_k(pts, k=int(num_steps), turn_alpha=1.0)
                    gt_wp = [int(gt_seq[0])] + [int(gt_seq[int(j)]) for j in idx.tolist()] + [int(gt_seq[-1])]

                gt_bucket_seqs = None
                if gt_bucket_map is not None and gt_bucket_keys is not None:
                    key = tuple(int(x) for x in gt_bucket_keys[int(rid)].tolist())
                    idx2 = gt_bucket_map.get(key)
                    if idx2 is not None and int(idx2.size) > 1:
                        other = idx2[idx2 != int(rid)]
                        if other.size > 0:
                            max_k = int(max(0, viz_gt_max))
                            if max_k > 0 and int(other.size) > max_k:
                                pick2 = rng.choice(other, size=max_k, replace=False)
                                other = np.sort(pick2)
                            gt_bucket_seqs = [_seq_from_pad(node_seq_pad, node_seq_len, int(ii)) for ii in other.tolist()]

                _plot_case(
                    out_png=out_dir / f"{name}.png",
                    out_pdf=out_dir / f"{name}.pdf",
                    node_y=node_y,
                    node_x=node_x,
                    gt_bucket_seqs=gt_bucket_seqs,
                    gt_seq=gt_seq,
                    pred_paths=pred_paths,
                    gt_wp=gt_wp,
                    pred_wps=pred_wps,
                    title=f"rid={int(rid)} bestJ={best_j:.3f} bestJ_s={best_j_succ:.3f} succ={succ_rate:.2f}",
                )
                viz += 1

            if progress_mode == "json" and (int(i_idx) % int(max(1, log_every)) == 0 or int(i_idx) == int(total) - 1):
                print(
                    json.dumps(
                        {
                            "task": "sample_graph_ar_waypoints_astar",
                            "done": int(i_idx) + 1,
                            "total": int(total),
                            "pct": float(int(i_idx) + 1) / float(max(1, int(total))),
                            "elapsed_s": float(time.time() - t0),
                            "best_j_mean_sofar": float(np.mean(np.asarray(best_j_list, dtype=np.float64))) if best_j_list else None,
                            "best_j_succ_mean_sofar": float(np.mean(np.asarray(best_j_succ_list, dtype=np.float64))) if best_j_succ_list else None,
                            "succ_rate_mean_sofar": float(np.mean(np.asarray(succ_rate_list, dtype=np.float64))) if succ_rate_list else None,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    def _q(a: List[float], p: float) -> Optional[float]:
        if not a:
            return None
        return float(np.percentile(np.asarray(a, dtype=np.float64), p))

    report = {
        "ok": True,
        "task": "sample_graph_ar_waypoints_astar",
        "inputs": {
            "checkpoint": str(checkpoint),
            "road_graph_npz": str(road_graph_npz),
            "paths_graph_npz": str(paths_graph_npz),
            "waypoints_npz": (str(waypoints_npz) if waypoints_npz is not None else None),
        },
        "config": {
            "K": int(K),
            "temperature": float(temperature),
            "wp_bin": int(wp_bin),
            "num_waypoints": int(num_steps),
            "num_routes": int(num_routes),
            "baseline_k": int(baseline_k),
            "oracle": bool(oracle),
            "pick_strategy": str(pick_strategy),
            "bin_filter_city": bool(bin_to_nodes_by_city is not None),
            "astar_workers": int(astar_workers_i),
            "seed": int(seed),
            "tz_offset_hours": float(tz_offset_hours),
            "viz_gt_od_bin": int(viz_gt_od_bin),
            "viz_gt_max": int(viz_gt_max),
            "progress": str(progress_mode),
            "log_every": int(log_every),
        },
        "stats": {
            "num_routes_sampled": int(len(rows)),
            "success_rate": {"mean": float(np.mean(np.asarray(succ_rate_list, dtype=np.float64))) if succ_rate_list else None, "p50": _q(succ_rate_list, 50), "p90": _q(succ_rate_list, 90)},
            "best_jaccard": {"mean": float(np.mean(np.asarray(best_j_list, dtype=np.float64))) if best_j_list else None, "p50": _q(best_j_list, 50), "p90": _q(best_j_list, 90)},
            "best_jaccard_success": {"mean": float(np.mean(np.asarray(best_j_succ_list, dtype=np.float64))) if best_j_succ_list else None, "p50": _q(best_j_succ_list, 50), "p90": _q(best_j_succ_list, 90)},
            "baseline_best_jaccard_kshortest": {
                "mean": float(np.mean(np.asarray(base_best_list, dtype=np.float64))) if base_best_list else None,
                "p50": _q(base_best_list, 50) if base_best_list else None,
                "p90": _q(base_best_list, 90) if base_best_list else None,
                "n": int(len(base_best_list)),
            },
        },
        "rows": rows[:200],
        "outputs": {"report_json": str(report_json), "out_dir": str(out_dir)},
        "meta": {"created_at": datetime.now(tz=TZ_SHANGHAI).isoformat()},
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample waypoint-AR (bin) model, then connect waypoints with A* to form full graph corridors.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--road_graph_npz", type=Path, required=True)
    p.add_argument("--paths_graph_npz", type=Path, required=True)
    p.add_argument("--waypoints_npz", type=Path, default=None, help="Optional GT waypoint npz for visualization.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--num_routes", type=int, default=200)
    p.add_argument("--baseline_k", type=int, default=0)
    p.add_argument(
        "--oracle",
        action="store_true",
        help="Oracle upper bound: use GT-derived waypoint bins instead of model predictions (still samples a node within each bin).",
    )
    p.add_argument(
        "--pick_strategy",
        type=str,
        default="tier_dir",
        choices=["tier_dir", "center"],
        help="How to instantiate a graph node inside a predicted bin. tier_dir=prefer major roads + OD-aligned direction; center=original bin-center sampling.",
    )
    p.add_argument(
        "--astar_workers",
        type=int,
        default=0,
        help="Parallelize A* connections with multiprocessing (fork). 0=disable, -1=auto(cpu_count-2).",
    )
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--viz_cases", type=int, default=10)
    p.add_argument("--viz_gt_od_bin", type=int, default=0, help="If >0, overlay GT routes from the same OD-bin bucket (viz only).")
    p.add_argument("--viz_gt_max", type=int, default=50, help="Max number of extra GT routes to overlay when --viz_gt_od_bin>0.")
    p.add_argument("--progress", type=str, default="auto", choices=["auto", "tqdm", "json", "none"])
    p.add_argument("--log_every", type=int, default=20, help="Only used when --progress=json.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    # IMPORTANT: keep CUDA untouched before the optional "fork" pool starts (A* workers).
    _set_seed(int(args.seed), seed_cuda=False)
    report = run(
        checkpoint=Path(args.checkpoint),
        road_graph_npz=Path(args.road_graph_npz),
        paths_graph_npz=Path(args.paths_graph_npz),
        waypoints_npz=(Path(args.waypoints_npz) if args.waypoints_npz is not None else None),
        out_dir=Path(args.out_dir),
        K=int(args.K),
        temperature=float(args.temperature),
        num_routes=int(args.num_routes),
        baseline_k=int(args.baseline_k),
        oracle=bool(args.oracle),
        pick_strategy=str(args.pick_strategy),
        astar_workers=int(args.astar_workers),
        tz_offset_hours=float(args.tz_offset_hours),
        seed=int(args.seed),
        viz_cases=int(args.viz_cases),
        viz_gt_od_bin=int(args.viz_gt_od_bin),
        viz_gt_max=int(args.viz_gt_max),
        progress=str(args.progress),
        log_every=int(args.log_every),
    )
    compact = {
        "ok": True,
        "out_dir": report["outputs"]["out_dir"],
        "num_routes_sampled": report["stats"]["num_routes_sampled"],
        "best_jaccard_mean": report["stats"]["best_jaccard"]["mean"],
        "best_jaccard_success_mean": report["stats"]["best_jaccard_success"]["mean"],
        "success_rate_mean": report["stats"]["success_rate"]["mean"],
        "report_json": report["outputs"]["report_json"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
