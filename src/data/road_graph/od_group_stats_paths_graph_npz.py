from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    od_bin: int
    min_traj_per_od: int
    multimodal_dist_thr: float
    max_groups: int
    max_pairs: int
    seed: int


def _as_view(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys)
    if keys.ndim != 2:
        raise ValueError(f"Expected keys (N,D), got {keys.shape}")
    return keys.view([("", keys.dtype)] * keys.shape[1]).reshape(-1)


def _iter_groups(keys: np.ndarray) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    keys = np.asarray(keys)
    view = _as_view(keys)
    order = np.argsort(view, kind="mergesort")
    keys_s = keys[order]
    idx_s = order.astype(np.int64, copy=False)
    i = 0
    n = int(keys_s.shape[0])
    while i < n:
        j = i + 1
        while j < n and np.array_equal(keys_s[j], keys_s[i]):
            j += 1
        yield keys_s[i].copy(), idx_s[i:j].copy()
        i = j


def _count_summary(counts: np.ndarray) -> Dict[str, object]:
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if counts.size == 0:
        return {"num_groups": 0}
    def q(p: float) -> float:
        return float(np.quantile(counts.astype(np.float64), p))
    return {
        "num_groups": int(counts.size),
        "count_p50": float(np.median(counts)),
        "count_p90": q(0.9),
        "count_p99": q(0.99),
        "max": int(np.max(counts)),
        "num_ge_2": int(np.sum(counts >= 2)),
        "num_ge_3": int(np.sum(counts >= 3)),
        "num_ge_5": int(np.sum(counts >= 5)),
        "num_ge_10": int(np.sum(counts >= 10)),
        "num_ge_20": int(np.sum(counts >= 20)),
    }


def _edge_set_from_seq(seq: Sequence[int]) -> set[int]:
    out: set[int] = set()
    for a, b in zip(seq[:-1], seq[1:]):
        aa = int(a)
        bb = int(b)
        if aa >= 0 and bb >= 0 and aa != bb:
            out.add((aa << 32) | (bb & 0xFFFFFFFF))
    return out


def _jaccard_edges(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    denom = len(a) + len(b) - inter
    return float(inter) / float(max(1, denom))


def _sample_pairwise_dist(
    idx: np.ndarray,
    edge_sets: Sequence[set[int]],
    *,
    rng: np.random.Generator,
    max_pairs: int,
) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    m = int(idx.size)
    if m < 2:
        return np.zeros((0,), dtype=np.float32)
    if m * (m - 1) // 2 <= int(max_pairs):
        d = []
        for i in range(m):
            for j in range(i + 1, m):
                a = edge_sets[int(idx[i])]
                b = edge_sets[int(idx[j])]
                d.append(1.0 - _jaccard_edges(a, b))
        return np.asarray(d, dtype=np.float32)
    d = []
    for _ in range(int(max_pairs)):
        i = int(rng.integers(0, m))
        j = int(rng.integers(0, m - 1))
        if j >= i:
            j += 1
        a = edge_sets[int(idx[i])]
        b = edge_sets[int(idx[j])]
        d.append(1.0 - _jaccard_edges(a, b))
    return np.asarray(d, dtype=np.float32)


def _dist_summary(d: np.ndarray) -> Optional[Dict[str, object]]:
    d = np.asarray(d, dtype=np.float32).reshape(-1)
    if d.size == 0:
        return None
    def q(p: float) -> float:
        return float(np.quantile(d.astype(np.float64), p))
    return {"mean": float(np.mean(d)), "p50": float(np.median(d)), "p90": q(0.9), "p99": q(0.99)}


def run(*, paths_graph_npz: Path, out_json: Path, cfg: Cfg) -> Dict[str, object]:
    p = np.load(str(paths_graph_npz), allow_pickle=True)
    start_node = np.asarray(p["start_node"], dtype=np.int64).reshape(-1)
    dest_node = np.asarray(p["dest_node"], dtype=np.int64).reshape(-1)
    route_city = np.asarray(p["route_city"], dtype=np.int64).reshape(-1) if "route_city" in p.files else np.zeros_like(start_node)
    start_pos = np.asarray(p["start_pos"], dtype=np.float64).reshape(-1, 2)
    dest_pos = np.asarray(p["dest_pos"], dtype=np.float64).reshape(-1, 2)
    node_seq_pad = np.asarray(p["node_seq_pad"], dtype=np.int64)
    node_seq_len = np.asarray(p["node_seq_len"], dtype=np.int64).reshape(-1)
    n = int(start_node.size)

    # Exact OD keys: (city, start_node, dest_node)
    exact_keys = np.stack([route_city, start_node, dest_node], axis=1).astype(np.int64, copy=False)
    exact_view = _as_view(exact_keys)
    _, exact_first, exact_counts = np.unique(exact_view, return_index=True, return_counts=True)
    exact_counts = exact_counts.astype(np.int64, copy=False)
    exact_keys_uniq = exact_keys[exact_first]
    order = np.argsort(-exact_counts, kind="mergesort")
    top_exact = []
    for k in order[: min(20, int(order.size))].tolist():
        city, s, d = (int(x) for x in exact_keys_uniq[int(k)].tolist())
        top_exact.append({"route_city": city, "start_node": s, "dest_node": d, "n": int(exact_counts[int(k)])})

    # Binned OD keys: (city, floor(start_pos/od_bin), floor(dest_pos/od_bin))
    b = float(max(1, int(cfg.od_bin)))
    s_bin = np.floor(start_pos / b).astype(np.int64)
    d_bin = np.floor(dest_pos / b).astype(np.int64)
    bin_keys = np.concatenate([route_city[:, None], s_bin, d_bin], axis=1).astype(np.int64, copy=False)  # (N,5)
    bin_view = _as_view(bin_keys)
    bin_uniq_view, bin_first, bin_counts = np.unique(bin_view, return_index=True, return_counts=True)
    bin_counts = bin_counts.astype(np.int64, copy=False)
    bin_keys_uniq = bin_keys[bin_first]
    order_b = np.argsort(-bin_counts, kind="mergesort")
    top_bins = []
    for k in order_b[: min(20, int(order_b.size))].tolist():
        key = bin_keys_uniq[int(k)].tolist()
        top_bins.append({"od_key": [int(x) for x in key], "n": int(bin_counts[int(k)])})

    # Precompute edge sets (for diversity audit)
    edge_sets: List[set[int]] = []
    for i in range(n):
        L = int(node_seq_len[i])
        if L <= 0:
            edge_sets.append(set())
            continue
        seq = node_seq_pad[i, :L].astype(np.int64, copy=False).tolist()
        edge_sets.append(_edge_set_from_seq(seq))

    rng = np.random.default_rng(int(cfg.seed))
    group_rows = []
    considered = 0
    multimodal = 0
    dist_all = []

    # Iterate groups by size (descending), but cap max_groups for speed.
    groups = list(_iter_groups(bin_keys))
    groups.sort(key=lambda kv: int(kv[1].size), reverse=True)
    for key_arr, idx in groups:
        if int(idx.size) < int(cfg.min_traj_per_od):
            continue
        considered += 1
        if considered > int(cfg.max_groups):
            break
        d = _sample_pairwise_dist(idx, edge_sets, rng=rng, max_pairs=int(cfg.max_pairs))
        ds = _dist_summary(d)
        if ds is None:
            continue
        dist_all.append(d.astype(np.float64, copy=False))
        is_mm = float(ds["p90"]) >= float(cfg.multimodal_dist_thr)
        if is_mm:
            multimodal += 1
        # include a compact row (no huge arrays)
        sample_ids = idx[: min(10, int(idx.size))].astype(np.int64, copy=False).tolist()
        group_rows.append(
            {
                "od_key": [int(x) for x in key_arr.tolist()],
                "n": int(idx.size),
                "dist": ds,
                "multimodal": bool(is_mm),
                "route_ids_head": [int(x) for x in sample_ids],
            }
        )

    dist_concat = np.concatenate(dist_all, axis=0) if dist_all else np.zeros((0,), dtype=np.float64)
    dist_global = _dist_summary(dist_concat) if dist_concat.size else None

    report: Dict[str, object] = {
        "ok": True,
        "task": "od_group_stats_paths_graph_npz",
        "paths_graph_npz": str(paths_graph_npz),
        "N_routes": int(n),
        "meta": {"created_at": datetime.now(TZ_SHANGHAI).isoformat()},
        "exact_od": {
            "summary": _count_summary(exact_counts),
            "top_groups": top_exact,
        },
        "od_bin": {
            "od_bin": int(cfg.od_bin),
            "min_traj_per_od": int(cfg.min_traj_per_od),
            "multimodal_dist_thr": float(cfg.multimodal_dist_thr),
            "summary": _count_summary(bin_counts),
            "top_groups": top_bins,
            "diversity_audit": {
                "groups_considered": int(considered),
                "groups_multimodal": int(multimodal),
                "multimodal_rate": float(multimodal) / float(max(1, considered)),
                "dist_global": dist_global,
                "group_rows_head": group_rows[: min(20, len(group_rows))],
            },
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="OD repetition + multimodality stats for paths_graph*.npz")
    p.add_argument("--paths_graph_npz", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--od_bin", type=int, default=128)
    p.add_argument("--min_traj_per_od", type=int, default=5)
    p.add_argument("--multimodal_dist_thr", type=float, default=0.3)
    p.add_argument("--max_groups", type=int, default=200)
    p.add_argument("--max_pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = Cfg(
        od_bin=int(args.od_bin),
        min_traj_per_od=int(args.min_traj_per_od),
        multimodal_dist_thr=float(args.multimodal_dist_thr),
        max_groups=int(args.max_groups),
        max_pairs=int(args.max_pairs),
        seed=int(args.seed),
    )
    run(paths_graph_npz=Path(args.paths_graph_npz), out_json=Path(args.out_json), cfg=cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
