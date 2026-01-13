from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore
except Exception as e:  # pragma: no cover
    linkage = None  # type: ignore[assignment]
    fcluster = None  # type: ignore[assignment]
    _HCLUST_ERR = e

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
    cluster_dist_thr: float
    min_cluster_size: int
    tz_offset_hours: float
    l2: float
    lr: float
    steps: int
    seed: int
    max_od_groups: int


def _time_features(start_t: np.ndarray, *, tz_offset_hours: float) -> np.ndarray:
    t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    t = (t + int(round(float(tz_offset_hours) * 3600.0))).astype(np.int64, copy=False)
    sec = np.mod(t, 86400).astype(np.float64, copy=False)
    hour = sec / 86400.0 * (2.0 * math.pi)
    day = (t // 86400).astype(np.int64, copy=False)
    dow = np.mod(day, 7).astype(np.float64, copy=False) / 7.0 * (2.0 * math.pi)
    is_weekend = (np.mod(day, 7) >= 5).astype(np.float64, copy=False)
    return np.stack([np.sin(hour), np.cos(hour), np.sin(dow), np.cos(dow), is_weekend], axis=1).astype(np.float32, copy=False)


def _od_bin_key(start_pos: np.ndarray, dest_pos: np.ndarray, *, od_bin: int) -> np.ndarray:
    s = np.asarray(start_pos, dtype=np.float32).reshape(-1, 2)
    d = np.asarray(dest_pos, dtype=np.float32).reshape(-1, 2)
    b = int(max(1, od_bin))
    s_bin = np.floor(s / float(b)).astype(np.int32)
    d_bin = np.floor(d / float(b)).astype(np.int32)
    return np.concatenate([s_bin, d_bin], axis=1).astype(np.int32, copy=False)


def _iter_groups(keys: np.ndarray) -> Iterable[Tuple[Tuple[int, ...], np.ndarray]]:
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
        yield tuple(int(x) for x in keys_sorted[i].tolist()), idx_sorted[i:j].astype(np.int64, copy=False)
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


def _condensed_from_dist(dist: np.ndarray) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float64)
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"Expected square dist matrix, got {dist.shape}")
    m = int(dist.shape[0])
    out = []
    for i in range(m):
        for j in range(i + 1, m):
            out.append(float(dist[i, j]))
    return np.asarray(out, dtype=np.float64)


def _cluster_labels(dist: np.ndarray, *, thr: float) -> np.ndarray:
    if linkage is None or fcluster is None:  # pragma: no cover
        raise SystemExit(f"Missing scipy.cluster.hierarchy (scipy). Error: {_HCLUST_ERR}")
    cd = _condensed_from_dist(dist)
    if cd.size == 0:
        return np.ones((dist.shape[0],), dtype=np.int32)
    Z = linkage(cd, method="average")
    lab = fcluster(Z, t=float(thr), criterion="distance").astype(np.int32, copy=False)
    return lab


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
        g = (p - y)
        grad_w = (X.T @ g) / float(max(1, n)) + lam * w
        grad_b = float(np.mean(g)) if n > 0 else 0.0
        w -= lr * grad_w
        b -= lr * grad_b
    return w.astype(np.float64, copy=False), float(b)


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    mu = np.mean(X, axis=0).astype(np.float32)
    sig = np.std(X, axis=0).astype(np.float32)
    sig = np.maximum(sig, 1e-3).astype(np.float32, copy=False)
    Xn = ((X - mu) / sig).astype(np.float32, copy=False)
    return Xn, mu, sig


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return ((np.asarray(X, dtype=np.float32) - mu) / sig).astype(np.float32, copy=False)


def _loocv_auc(X: np.ndarray, y: np.ndarray, *, l2: float, lr: float, steps: int, seed: int) -> Optional[float]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32).reshape(-1)
    n = int(y.size)
    if n < 6:
        return None
    if int(np.sum(y == 1).item()) == 0 or int(np.sum(y == 0).item()) == 0:
        return None

    rng = np.random.default_rng(int(seed))
    order = np.arange(n, dtype=np.int64)
    rng.shuffle(order)
    score = np.full((n,), np.nan, dtype=np.float64)
    for k in range(n):
        i = int(order[k])
        tr = np.ones((n,), dtype=np.uint8)
        tr[i] = 0
        idx_tr = np.nonzero(tr)[0].astype(np.int64, copy=False)
        X_tr, mu, sig = _standardize_fit(X[idx_tr])
        y_tr = y[idx_tr].astype(np.float32, copy=False)
        w, b = _fit_logreg(X_tr, y_tr, l2=float(l2), lr=float(lr), steps=int(steps))
        x_i = _standardize_apply(X[i : i + 1], mu, sig).astype(np.float64)
        score[i] = float(x_i @ w + float(b))
    auc = _auc(y, score)
    if auc is None:
        return None
    # Label orientation is arbitrary across OD groups; use symmetric AUC.
    return float(max(float(auc), 1.0 - float(auc)))


def _pick_top2(labels: np.ndarray, *, min_cluster_size: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    uniq, cnt = np.unique(labels, return_counts=True)
    order = np.argsort(-cnt)
    uniq = uniq[order]
    cnt = cnt[order]
    if uniq.size < 2:
        return None
    a, b = int(uniq[0]), int(uniq[1])
    if int(cnt[0]) < int(min_cluster_size) or int(cnt[1]) < int(min_cluster_size):
        return None
    mask = (labels == a) | (labels == b)
    y = (labels[mask] == b).astype(np.int32, copy=False)
    return mask.astype(np.uint8, copy=False), y


def _tier_at_points(
    yx: np.ndarray,
    *,
    major: np.ndarray,
    minor: np.ndarray,
    service: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(yx, dtype=np.float32).reshape(-1, 2)
    H, W = major.shape
    y = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, int(H) - 1)
    x = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, int(W) - 1)
    return np.stack([major[y, x], minor[y, x], service[y, x]], axis=1).astype(np.float32, copy=False)


def run_gate(
    *,
    paths_graph_npz: Path,
    semantic_dir: Path,
    road_graph_npz: Path,
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

    major = load_osm_road_prob_major(semantic_dir)
    minor = load_osm_road_prob_minor(semantic_dir)
    service = load_osm_road_prob_service(semantic_dir)
    if major.shape != minor.shape or major.shape != service.shape:
        raise ValueError("tier-road raster shapes mismatch")
    if int(g.grid.H) != int(major.shape[0]) or int(g.grid.W) != int(major.shape[1]):
        raise ValueError(
            f"grid mismatch: road_graph grid={g.grid.H}x{g.grid.W}, semantic rasters={major.shape[0]}x{major.shape[1]}"
        )

    tf = _time_features(start_t, tz_offset_hours=float(cfg.tz_offset_hours))
    tier_o = _tier_at_points(start_pos, major=major, minor=minor, service=service)
    tier_d = _tier_at_points(dest_pos, major=major, minor=minor, service=service)

    keys = _od_bin_key(start_pos, dest_pos, od_bin=int(cfg.od_bin))
    groups = list(_iter_groups(keys))
    num_groups_raw = int(len(groups))
    groups = [(k, idx) for (k, idx) in groups if int(idx.size) >= int(cfg.min_traj_per_od)]
    num_groups_min = int(len(groups))

    if cfg.max_od_groups > 0 and num_groups_min > int(cfg.max_od_groups):
        rng = np.random.default_rng(int(cfg.seed))
        pick = rng.choice(num_groups_min, size=int(cfg.max_od_groups), replace=False)
        pick = np.sort(pick.astype(np.int64))
        groups = [groups[i] for i in pick.tolist()]

    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "report.json"
    events_jsonl = out_dir / "events.jsonl"

    n_clusters_list: List[int] = []
    auc_time: List[float] = []
    auc_tier: List[float] = []
    auc_tt: List[float] = []

    events = []

    n_used = 0
    n_skipped_single = 0
    n_skipped_small = 0

    for od_key, idx in groups:
        rows = idx.tolist()
        seqs = [_seq_from_pad(node_seq_pad, node_seq_len, int(i)) for i in rows]
        if len(seqs) < int(cfg.min_traj_per_od):
            continue
        edge_sets = [_edge_set(s) for s in seqs]
        m = int(len(edge_sets))
        dist = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(i + 1, m):
                sim = _jaccard_edges(edge_sets[i], edge_sets[j])
                d = 1.0 - float(sim)
                dist[i, j] = d
                dist[j, i] = d

        lab = _cluster_labels(dist, thr=float(cfg.cluster_dist_thr))
        n_cl = int(np.unique(lab).size)
        n_clusters_list.append(n_cl)
        if n_cl < 2:
            n_skipped_single += 1
            continue

        picked = _pick_top2(lab, min_cluster_size=int(cfg.min_cluster_size))
        if picked is None:
            n_skipped_small += 1
            continue
        mask, y = picked
        rows_sel = [rows[i] for i in range(m) if int(mask[i]) == 1]

        X_time = tf[np.asarray(rows_sel, dtype=np.int64)]
        X_tier = np.concatenate([tier_o[np.asarray(rows_sel, dtype=np.int64)], tier_d[np.asarray(rows_sel, dtype=np.int64)]], axis=1)
        X_tt = np.concatenate([X_time, X_tier], axis=1)

        auc1 = _loocv_auc(X_time, y, l2=float(cfg.l2), lr=float(cfg.lr), steps=int(cfg.steps), seed=int(cfg.seed))
        auc2 = _loocv_auc(X_tier, y, l2=float(cfg.l2), lr=float(cfg.lr), steps=int(cfg.steps), seed=int(cfg.seed))
        auc3 = _loocv_auc(X_tt, y, l2=float(cfg.l2), lr=float(cfg.lr), steps=int(cfg.steps), seed=int(cfg.seed))
        if auc1 is None or auc2 is None or auc3 is None:
            # If the group is too small after filtering, skip.
            n_skipped_small += 1
            continue

        auc_time.append(float(auc1))
        auc_tier.append(float(auc2))
        auc_tt.append(float(auc3))
        n_used += 1

        # Compact per-OD event (no huge arrays).
        uniq, cnt = np.unique(lab, return_counts=True)
        sizes = {str(int(u)): int(c) for u, c in zip(uniq.tolist(), cnt.tolist())}
        events.append(
            {
                "od_key": [int(x) for x in od_key],
                "n_total": int(len(rows)),
                "n_clusters": int(n_cl),
                "cluster_sizes": sizes,
                "selected_top2": int(len(rows_sel)),
                "auc": {"time_only": float(auc1), "tier_od": float(auc2), "time_tier": float(auc3)},
            }
        )

    if events:
        with events_jsonl.open("w", encoding="utf-8") as f:
            for row in events:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _summ(a: List[float]) -> Dict[str, object]:
        if not a:
            return {"mean": None, "p50": None, "p90": None, "n": 0}
        arr = np.asarray(a, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "n": int(arr.size),
        }

    def _hist_int(vals: List[int]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for v in vals:
            k = str(int(v))
            out[k] = int(out.get(k, 0)) + 1
        return out

    auc_used = _summ(auc_tt)["mean"]
    decision = "NO_DATA"
    if auc_used is not None:
        a = float(auc_used)
        if a < 0.55:
            decision = "NO_GO"
        elif a < 0.60:
            decision = "CONDITIONAL_GO"
        elif a < 0.70:
            decision = "GO"
        else:
            decision = "STRONG_GO"

    report = {
        "ok": True,
        "gate": "G3b_cluster_semantic_informativeness",
        "inputs": {"paths_graph_npz": str(paths_graph_npz), "road_graph_npz": str(road_graph_npz), "semantic_dir": str(semantic_dir)},
        "config": {
            "od_bin": int(cfg.od_bin),
            "min_traj_per_od": int(cfg.min_traj_per_od),
            "cluster_dist_thr": float(cfg.cluster_dist_thr),
            "min_cluster_size": int(cfg.min_cluster_size),
            "tz_offset_hours": float(cfg.tz_offset_hours),
            "l2": float(cfg.l2),
            "lr": float(cfg.lr),
            "steps": int(cfg.steps),
            "seed": int(cfg.seed),
            "max_od_groups": int(cfg.max_od_groups),
        },
        "stats": {
            "N_routes": int(n),
            "od_groups": {"raw": int(num_groups_raw), "min_traj": int(num_groups_min), "used": int(n_used), "skipped_single_cluster": int(n_skipped_single), "skipped_small": int(n_skipped_small)},
            "n_clusters": {"hist": _hist_int(n_clusters_list)},
            "auc": {"time_only": _summ(auc_time), "tier_od": _summ(auc_tier), "time_tier": _summ(auc_tt)},
        },
        "decision": {"auc_used_mean_time_tier": auc_used, "label": decision},
        "outputs": {"report_json": str(report_json), "events_jsonl": str(events_jsonl)},
        "meta": {"created_at": datetime.now(tz=TZ_SHANGHAI).isoformat()},
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gate (T2 revised): corridor label via hierarchical clustering within OD groups; predict cluster with time+tier (OD endpoint tiers)."
    )
    p.add_argument("--paths_graph_npz", type=Path, required=True)
    p.add_argument("--road_graph_npz", type=Path, required=True)
    p.add_argument("--semantic_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--od_bin", type=int, default=128)
    p.add_argument("--min_traj_per_od", type=int, default=10)
    p.add_argument("--cluster_dist_thr", type=float, default=0.5, help="Distance threshold t for fcluster (distance=1-Jaccard).")
    p.add_argument("--min_cluster_size", type=int, default=3)
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--l2", type=float, default=1e-2)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_od_groups", type=int, default=0, help="Optional cap on OD groups for speed (0=all).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = GateCfg(
        od_bin=int(args.od_bin),
        min_traj_per_od=int(args.min_traj_per_od),
        cluster_dist_thr=float(args.cluster_dist_thr),
        min_cluster_size=int(args.min_cluster_size),
        tz_offset_hours=float(args.tz_offset_hours),
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
        "used_groups": report["stats"]["od_groups"]["used"],
        "auc_time_tier_mean": report["stats"]["auc"]["time_tier"]["mean"],
        "report_json": report["outputs"]["report_json"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

