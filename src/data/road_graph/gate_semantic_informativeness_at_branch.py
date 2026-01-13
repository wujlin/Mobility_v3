from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.data.road_graph.gate_candidate_paths_from_routes_npz import _load_graph_npz
from src.features.semantic_od import (
    load_osm_road_prob_major,
    load_osm_road_prob_minor,
    load_osm_road_prob_service,
)


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class GateCfg:
    od_bin: int
    min_traj_per_od: int
    min_multimodal_dist: float
    tier_hops: int
    tz_offset_hours: float
    test_frac: float
    l2: float
    lr: float
    steps: int
    seed: int
    max_od_groups: int


def _time_features(start_t: np.ndarray, *, tz_offset_hours: float) -> np.ndarray:
    """
    Return per-sample temporal features.
    - hour_sin, hour_cos
    - dow_sin, dow_cos
    - is_weekend
    """
    t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    # Apply timezone shift in seconds.
    t = (t + int(round(float(tz_offset_hours) * 3600.0))).astype(np.int64, copy=False)
    sec = np.mod(t, 86400).astype(np.float64, copy=False)
    hour = sec / 86400.0 * (2.0 * math.pi)
    day = (t // 86400).astype(np.int64, copy=False)
    dow = np.mod(day, 7).astype(np.float64, copy=False) / 7.0 * (2.0 * math.pi)
    is_weekend = (np.mod(day, 7) >= 5).astype(np.float64, copy=False)
    return np.stack(
        [
            np.sin(hour),
            np.cos(hour),
            np.sin(dow),
            np.cos(dow),
            is_weekend,
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _od_bin_key(start_pos: np.ndarray, dest_pos: np.ndarray, *, od_bin: int) -> np.ndarray:
    s = np.asarray(start_pos, dtype=np.float32).reshape(-1, 2)
    d = np.asarray(dest_pos, dtype=np.float32).reshape(-1, 2)
    b = int(max(1, od_bin))
    s_bin = np.floor(s / float(b)).astype(np.int32)
    d_bin = np.floor(d / float(b)).astype(np.int32)
    key = np.concatenate([s_bin, d_bin], axis=1).astype(np.int32, copy=False)
    return key


def _iter_groups(keys: np.ndarray) -> Iterable[Tuple[Tuple[int, ...], np.ndarray]]:
    """
    Group indices by key row, yielding (key_tuple, idx_array).
    """
    keys = np.asarray(keys, dtype=np.int32)
    if keys.ndim != 2:
        raise ValueError(f"Expected keys (N,D), got {keys.shape}")
    view = keys.view([("", keys.dtype)] * keys.shape[1])
    order = np.argsort(view.reshape(-1), kind="mergesort")
    keys_sorted = keys[order]
    idx_sorted = order
    i = 0
    n = int(keys_sorted.shape[0])
    while i < n:
        j = i + 1
        while j < n and np.array_equal(keys_sorted[j], keys_sorted[i]):
            j += 1
        key_tuple = tuple(int(x) for x in keys_sorted[i].tolist())
        yield key_tuple, idx_sorted[i:j].astype(np.int64, copy=False)
        i = j


def _seq_from_pad(node_seq_pad: np.ndarray, node_seq_len: np.ndarray, i: int) -> List[int]:
    L = int(node_seq_len[i])
    if L <= 0:
        return []
    seq = node_seq_pad[i, :L].astype(np.int64, copy=False).tolist()
    return [int(x) for x in seq if int(x) >= 0]


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
    if denom <= 0:
        return 0.0
    return float(inter) / float(denom)


def _pick_most_different_pair(seqs: List[List[int]]) -> Tuple[int, int, float]:
    """
    Return (i,j,dist) where dist = 1 - jaccard_edges and is maximized.
    """
    m = int(len(seqs))
    if m < 2:
        return -1, -1, 0.0
    edge_sets = [_edge_set(s) for s in seqs]
    best = (-1, -1, -1.0)
    for i in range(m):
        for j in range(i + 1, m):
            sim = _jaccard_edges(edge_sets[i], edge_sets[j])
            dist = 1.0 - float(sim)
            if dist > best[2]:
                best = (i, j, dist)
    return int(best[0]), int(best[1]), float(best[2])


def _first_branch(a: Sequence[int], b: Sequence[int]) -> Optional[Tuple[int, int, int, int]]:
    """
    Given two node sequences, return (branch_node, next_a, next_b, depth)
    where depth is the common-prefix length (number of nodes shared).
    """
    if not a or not b:
        return None
    m = min(len(a), len(b))
    k = 0
    while k < m and int(a[k]) == int(b[k]):
        k += 1
    if k <= 0:
        # Diverge at start: treat the first node as branch, but we still need distinct next.
        if len(a) < 2 or len(b) < 2:
            return None
        branch = int(a[0])
        na = int(a[1])
        nb = int(b[1])
        if na == nb:
            return None
        return branch, na, nb, 1
    # If one is a prefix of the other, no clear branch decision.
    if k >= len(a) or k >= len(b):
        return None
    branch = int(a[k - 1])
    na = int(a[k])
    nb = int(b[k])
    if na == nb:
        return None
    return branch, na, nb, int(k)


def _sample_tier_means_for_prefix(
    node_seq: Sequence[int],
    *,
    g,
    start_at_node: int,
    hops: int,
    road_major: np.ndarray,
    road_minor: np.ndarray,
    road_service: np.ndarray,
) -> np.ndarray:
    """
    Compute mean tier-road probabilities over a prefix starting at `start_at_node`.
    Uses node positions; hops counts edges (so nodes = hops+1, clipped by available length).
    Returns: (3,) float32 [major, minor, service]
    """
    if hops <= 0:
        return np.zeros((3,), dtype=np.float32)
    # Find first occurrence.
    try:
        idx0 = list(map(int, node_seq)).index(int(start_at_node))
    except ValueError:
        return np.zeros((3,), dtype=np.float32)

    end = min(len(node_seq), idx0 + int(hops) + 1)
    nodes = np.asarray([int(x) for x in node_seq[idx0:end] if int(x) >= 0], dtype=np.int64)
    if nodes.size <= 0:
        return np.zeros((3,), dtype=np.float32)
    y = np.rint(g.node_y[nodes]).astype(np.int64, copy=False)
    x = np.rint(g.node_x[nodes]).astype(np.int64, copy=False)
    H, W = road_major.shape
    y = np.clip(y, 0, int(H) - 1)
    x = np.clip(x, 0, int(W) - 1)
    maj = road_major[y, x].astype(np.float32, copy=False)
    mi = road_minor[y, x].astype(np.float32, copy=False)
    sv = road_service[y, x].astype(np.float32, copy=False)
    return np.asarray([float(np.mean(maj)), float(np.mean(mi)), float(np.mean(sv))], dtype=np.float32)


def _train_test_split_groups(groups: Sequence[Tuple[int, ...]], *, test_frac: float, seed: int) -> Tuple[set, set]:
    groups = list(groups)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(groups)
    n = len(groups)
    n_test = int(round(float(test_frac) * float(n)))
    n_test = int(max(1, min(n - 1, n_test))) if n >= 2 else 0
    test = set(groups[:n_test])
    train = set(groups[n_test:])
    return train, test


def _standardize(X: np.ndarray, *, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32, copy=False)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logreg(X: np.ndarray, y: np.ndarray, *, l2: float, lr: float, steps: int) -> Tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, d = X.shape
    w = np.zeros((d,), dtype=np.float64)
    b = 0.0
    lam = float(max(0.0, l2))
    lr = float(lr)
    steps = int(max(10, steps))
    for _ in range(steps):
        z = X @ w + b
        p = _sigmoid(z)
        g = (p - y)  # (n,)
        grad_w = (X.T @ g) / float(max(1, n)) + lam * w
        grad_b = float(np.mean(g)) if n > 0 else 0.0
        w -= lr * grad_w
        b -= lr * grad_b
    return w.astype(np.float64, copy=False), float(b)


def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    n = int(a.size)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty((n,), dtype=np.float64)
    s = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        r = (float(i + j) / 2.0) + 1.0
        ranks[order[i : j + 1]] = r
        i = j + 1
    return ranks


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.size != y_score.size or y_true.size == 0:
        return None
    pos = y_true == 1
    n_pos = int(np.sum(pos).item())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rankdata(y_score)
    sum_ranks_pos = float(np.sum(ranks[pos]).item())
    auc = (sum_ranks_pos - float(n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def _fit_eval_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Tuple[int, ...]],
    *,
    train_groups: set,
    test_groups: set,
    l2: float,
    lr: float,
    steps: int,
) -> Dict[str, object]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32).reshape(-1)
    if X.shape[0] != y.size or X.shape[0] != len(groups):
        raise ValueError("X/y/groups length mismatch")
    mask_train = np.asarray([g in train_groups for g in groups], dtype=np.uint8)
    mask_test = np.asarray([g in test_groups for g in groups], dtype=np.uint8)
    i_tr = np.nonzero(mask_train)[0].astype(np.int64, copy=False)
    i_te = np.nonzero(mask_test)[0].astype(np.int64, copy=False)
    if i_tr.size == 0 or i_te.size == 0:
        return {"auc": None, "n_train": int(i_tr.size), "n_test": int(i_te.size)}
    X_tr = X[i_tr]
    y_tr = y[i_tr].astype(np.float32, copy=False)
    mu = np.mean(X_tr, axis=0).astype(np.float32)
    sig = np.std(X_tr, axis=0).astype(np.float32)
    sig = np.maximum(sig, 1e-3).astype(np.float32, copy=False)
    X_trn = _standardize(X_tr, mean=mu, std=sig)
    X_ten = _standardize(X[i_te], mean=mu, std=sig)
    w, b = _fit_logreg(X_trn, y_tr, l2=float(l2), lr=float(lr), steps=int(steps))
    score = (X_ten.astype(np.float64) @ w + float(b)).astype(np.float64, copy=False)
    auc = _auc(y[i_te], score)
    return {"auc": auc, "n_train": int(i_tr.size), "n_test": int(i_te.size)}


def run_gate(
    *,
    paths_graph_npz: Path,
    road_graph_npz: Path,
    semantic_dir: Path,
    out_dir: Path,
    cfg: GateCfg,
) -> Dict[str, object]:
    g = _load_graph_npz(Path(road_graph_npz))
    data = np.load(str(paths_graph_npz), allow_pickle=True)
    need = {"start_t", "start_pos", "dest_pos", "node_seq_pad", "node_seq_len"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"paths_graph.npz missing keys: {missing}")
    start_t = np.asarray(data["start_t"], dtype=np.int64).reshape(-1)
    start_pos = np.asarray(data["start_pos"], dtype=np.float32).reshape(-1, 2)
    dest_pos = np.asarray(data["dest_pos"], dtype=np.float32).reshape(-1, 2)
    node_seq_pad = np.asarray(data["node_seq_pad"], dtype=np.int32)
    node_seq_len = np.asarray(data["node_seq_len"], dtype=np.int32).reshape(-1)
    n = int(start_t.size)

    # Load tier-road rasters (semantic minimal set).
    road_major = load_osm_road_prob_major(semantic_dir)
    road_minor = load_osm_road_prob_minor(semantic_dir)
    road_service = load_osm_road_prob_service(semantic_dir)
    if road_major.shape != road_minor.shape or road_major.shape != road_service.shape:
        raise ValueError("tier-road raster shapes mismatch")
    H, W = road_major.shape
    if int(g.grid.H) != int(H) or int(g.grid.W) != int(W):
        # Hard fail: a mismatch here means bbox/grid alignment is inconsistent.
        raise ValueError(f"grid mismatch: road_graph grid={g.grid.H}x{g.grid.W}, semantic rasters={H}x{W}")

    # Group by OD bin to ensure enough repeated ODs.
    od_keys = _od_bin_key(start_pos, dest_pos, od_bin=int(cfg.od_bin))
    groups = list(_iter_groups(od_keys))
    num_groups_raw = int(len(groups))
    groups = [(k, idx) for (k, idx) in groups if int(idx.size) >= int(cfg.min_traj_per_od)]
    num_groups_min = int(len(groups))

    # Optional cap for speed.
    if cfg.max_od_groups > 0 and num_groups_min > int(cfg.max_od_groups):
        rng = np.random.default_rng(int(cfg.seed))
        pick = rng.choice(num_groups_min, size=int(cfg.max_od_groups), replace=False)
        pick = np.sort(pick.astype(np.int64))
        groups = [groups[i] for i in pick.tolist()]

    # Build decision samples.
    X_time: List[np.ndarray] = []
    X_tier: List[np.ndarray] = []
    y: List[int] = []
    group_id: List[Tuple[int, ...]] = []
    branch_depths: List[int] = []

    events_jsonl = []
    num_groups_multimodal = 0
    num_groups_used = 0
    num_groups_skipped_no_branch = 0
    num_groups_skipped_one_class = 0

    tfeat_all = _time_features(start_t, tz_offset_hours=float(cfg.tz_offset_hours))

    for od_key, idx in groups:
        seqs = [_seq_from_pad(node_seq_pad, node_seq_len, int(i)) for i in idx.tolist()]
        if len(seqs) < 2:
            continue
        ia, ib, dist = _pick_most_different_pair(seqs)
        if ia < 0 or ib < 0:
            continue
        if float(dist) < float(cfg.min_multimodal_dist):
            continue
        num_groups_multimodal += 1

        br = _first_branch(seqs[ia], seqs[ib])
        if br is None:
            num_groups_skipped_no_branch += 1
            continue
        branch_node, next_a, next_b, depth = br

        tier_a = _sample_tier_means_for_prefix(
            seqs[ia],
            g=g,
            start_at_node=int(branch_node),
            hops=int(cfg.tier_hops),
            road_major=road_major,
            road_minor=road_minor,
            road_service=road_service,
        )
        tier_b = _sample_tier_means_for_prefix(
            seqs[ib],
            g=g,
            start_at_node=int(branch_node),
            hops=int(cfg.tier_hops),
            road_major=road_major,
            road_minor=road_minor,
            road_service=road_service,
        )
        tier_diff = (tier_a - tier_b).astype(np.float32, copy=False)

        # Collect per-trajectory choices at the branch.
        local_y = []
        local_rows = []
        for jj in idx.tolist():
            seq = _seq_from_pad(node_seq_pad, node_seq_len, int(jj))
            if not seq:
                continue
            try:
                k0 = seq.index(int(branch_node))
            except ValueError:
                continue
            if k0 + 1 >= len(seq):
                continue
            nxt = int(seq[k0 + 1])
            if nxt == int(next_a):
                lbl = 0
            elif nxt == int(next_b):
                lbl = 1
            else:
                continue
            local_y.append(int(lbl))
            local_rows.append(int(jj))

        if len(local_y) < 4:
            # Too few trajectories that actually use one of the two branches.
            continue
        n1 = int(np.sum(np.asarray(local_y, dtype=np.int32) == 1).item())
        n0 = int(len(local_y) - n1)
        if n0 == 0 or n1 == 0:
            num_groups_skipped_one_class += 1
            continue
        num_groups_used += 1

        for jj, lbl in zip(local_rows, local_y):
            X_time.append(tfeat_all[int(jj)].astype(np.float32, copy=False))
            X_tier.append(tier_diff.astype(np.float32, copy=False))
            y.append(int(lbl))
            group_id.append(tuple(int(x) for x in od_key))
            branch_depths.append(int(depth))

        events_jsonl.append(
            {
                "od_key": [int(x) for x in od_key],
                "n_total": int(idx.size),
                "n_used": int(len(local_y)),
                "branch_node": int(branch_node),
                "next_a": int(next_a),
                "next_b": int(next_b),
                "tier_a": [float(x) for x in tier_a.tolist()],
                "tier_b": [float(x) for x in tier_b.tolist()],
                "tier_diff": [float(x) for x in tier_diff.tolist()],
                "branch_depth": int(depth),
                "label_counts": {"0": int(n0), "1": int(n1)},
                "pair_dist": float(dist),
            }
        )

    X_time_arr = np.asarray(X_time, dtype=np.float32) if X_time else np.zeros((0, 5), dtype=np.float32)
    X_tier_arr = np.asarray(X_tier, dtype=np.float32) if X_tier else np.zeros((0, 3), dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int32).reshape(-1)
    groups_arr = list(group_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "report.json"
    events_path = out_dir / "events.jsonl"

    if events_jsonl:
        with events_path.open("w", encoding="utf-8") as f:
            for row in events_jsonl:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Evaluate AUC with group split.
    unique_groups_used = sorted(set(groups_arr))
    train_g, test_g = _train_test_split_groups(unique_groups_used, test_frac=float(cfg.test_frac), seed=int(cfg.seed))

    X_time_tier = np.concatenate([X_time_arr, X_tier_arr], axis=1) if X_time_arr.size else np.zeros((0, 8), dtype=np.float32)
    res_time = _fit_eval_auc(
        X_time_arr,
        y_arr,
        groups_arr,
        train_groups=train_g,
        test_groups=test_g,
        l2=float(cfg.l2),
        lr=float(cfg.lr),
        steps=int(cfg.steps),
    )
    res_tier = _fit_eval_auc(
        X_tier_arr,
        y_arr,
        groups_arr,
        train_groups=train_g,
        test_groups=test_g,
        l2=float(cfg.l2),
        lr=float(cfg.lr),
        steps=int(cfg.steps),
    )
    res_tt = _fit_eval_auc(
        X_time_tier,
        y_arr,
        groups_arr,
        train_groups=train_g,
        test_groups=test_g,
        l2=float(cfg.l2),
        lr=float(cfg.lr),
        steps=int(cfg.steps),
    )

    auc_time = res_time.get("auc")
    auc_tier = res_tier.get("auc")
    auc_tt = res_tt.get("auc")

    # Gate decision (use time+tier AUC if available; else fallback to time-only).
    auc_for_gate = None
    if auc_tt is not None:
        auc_for_gate = float(auc_tt)
    elif auc_time is not None:
        auc_for_gate = float(auc_time)

    decision = "NO_GO"
    if auc_for_gate is None:
        decision = "NO_DATA"
    elif auc_for_gate < 0.55:
        decision = "NO_GO"
    elif auc_for_gate < 0.60:
        decision = "CONDITIONAL_GO"
    elif auc_for_gate < 0.70:
        decision = "GO"
    else:
        decision = "STRONG_GO"

    report = {
        "ok": True,
        "gate": "G3_branch_semantic_informativeness",
        "inputs": {
            "paths_graph_npz": str(paths_graph_npz),
            "road_graph_npz": str(road_graph_npz),
            "semantic_dir": str(semantic_dir),
        },
        "config": {
            "od_bin": int(cfg.od_bin),
            "min_traj_per_od": int(cfg.min_traj_per_od),
            "min_multimodal_dist": float(cfg.min_multimodal_dist),
            "tier_hops": int(cfg.tier_hops),
            "tz_offset_hours": float(cfg.tz_offset_hours),
            "test_frac": float(cfg.test_frac),
            "l2": float(cfg.l2),
            "lr": float(cfg.lr),
            "steps": int(cfg.steps),
            "seed": int(cfg.seed),
            "max_od_groups": int(cfg.max_od_groups),
        },
        "stats": {
            "N_routes": int(n),
            "od_groups": {
                "raw": int(num_groups_raw),
                "min_traj": int(num_groups_min),
                "multimodal": int(num_groups_multimodal),
                "used": int(num_groups_used),
                "skipped_no_branch": int(num_groups_skipped_no_branch),
                "skipped_one_class": int(num_groups_skipped_one_class),
            },
            "samples": {
                "n": int(y_arr.size),
                "pos": int(np.sum(y_arr == 1).item()) if y_arr.size else 0,
                "pos_frac": float(np.mean(y_arr.astype(np.float32))) if y_arr.size else None,
            },
            "branch_depth": {
                "p50": float(np.percentile(np.asarray(branch_depths, dtype=np.float32), 50)) if branch_depths else None,
                "p90": float(np.percentile(np.asarray(branch_depths, dtype=np.float32), 90)) if branch_depths else None,
            },
            "split": {"n_groups_train": int(len(train_g)), "n_groups_test": int(len(test_g))},
        },
        "auc": {
            "time_only": res_time,
            "tier_only": res_tier,
            "time_tier": res_tt,
        },
        "decision": {"auc_used": auc_for_gate, "label": decision},
        "outputs": {"report_json": str(report_json), "events_jsonl": str(events_path)},
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gate (T2): measure semantic informativeness (time + tier-road) at the first branch decision point.")
    p.add_argument("--paths_graph_npz", type=Path, required=True)
    p.add_argument("--road_graph_npz", type=Path, required=True)
    p.add_argument("--semantic_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--od_bin", type=int, default=128, help="OD bin size in grid cells (used to form OD groups).")
    p.add_argument("--min_traj_per_od", type=int, default=10)
    p.add_argument("--min_multimodal_dist", type=float, default=0.3, help="Require max pairwise (1-Jaccard) >= this in each OD group.")
    p.add_argument("--tier_hops", type=int, default=32, help="How many graph edges to look ahead from the branch node for tier-road summary.")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0, help="Timezone offset for temporal features (Detroit ~ -5).")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--l2", type=float, default=1e-2)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_od_groups", type=int, default=0, help="Optional cap on OD groups for speed (0=all).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = GateCfg(
        od_bin=int(args.od_bin),
        min_traj_per_od=int(args.min_traj_per_od),
        min_multimodal_dist=float(args.min_multimodal_dist),
        tier_hops=int(args.tier_hops),
        tz_offset_hours=float(args.tz_offset_hours),
        test_frac=float(args.test_frac),
        l2=float(args.l2),
        lr=float(args.lr),
        steps=int(args.steps),
        seed=int(args.seed),
        max_od_groups=int(args.max_od_groups),
    )
    report = run_gate(
        paths_graph_npz=Path(args.paths_graph_npz),
        road_graph_npz=Path(args.road_graph_npz),
        semantic_dir=Path(args.semantic_dir),
        out_dir=Path(args.out_dir),
        cfg=cfg,
    )
    compact = {
        "ok": True,
        "gate": report["gate"],
        "decision": report["decision"],
        "auc_time_tier": report["auc"]["time_tier"]["auc"],
        "n_samples": report["stats"]["samples"]["n"],
        "n_od_used": report["stats"]["od_groups"]["used"],
        "report_json": report["outputs"]["report_json"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

