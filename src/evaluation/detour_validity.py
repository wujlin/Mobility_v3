from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.evaluation.distribution_metrics import jsd_from_hist


@dataclass(frozen=True)
class BinConfig:
    turn_bins: int
    dev_bins: int
    len_bins: int
    range_percentiles: Tuple[float, float]


def _parse_inputs(items: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in items:
        if ":" not in raw:
            raise ValueError(f"Invalid --inputs item '{raw}'. Expected 'Label:Path'.")
        label, path = raw.split(":", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid --inputs item '{raw}'. Expected 'Label:Path'.")
        out[label] = path
    return out


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    for k in ["preds", "preds_k", "targets", "start_pos", "traj_idx", "start_t"]:
        if k in data.files:
            out[k] = np.asarray(data[k])
    if "targets" not in out:
        raise ValueError(f"{path} missing required key 'targets'. keys={list(data.files)}")
    if "preds" not in out and "preds_k" not in out:
        raise ValueError(f"{path} missing required key 'preds' or 'preds_k'. keys={list(data.files)}")
    return out


def _window_keys(data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """
    Stable window identifier for alignment across methods.
    key = (traj_idx << 32) | start_t
    """
    if "traj_idx" not in data or "start_t" not in data:
        return None
    traj_idx = np.asarray(data["traj_idx"], dtype=np.int64).reshape(-1)
    start_t = np.asarray(data["start_t"], dtype=np.int64).reshape(-1)
    if traj_idx.shape[0] != start_t.shape[0]:
        raise ValueError(f"Bad traj_idx/start_t shapes: traj_idx={traj_idx.shape} start_t={start_t.shape}")
    return (traj_idx << np.int64(32)) | (start_t & np.int64(0xFFFFFFFF))


def _align_loaded_by_window_ids(loaded: Dict[str, Dict[str, np.ndarray]]) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, object]]:
    """
    Align different method npz files to the common set of (traj_idx,start_t) windows.
    Preserves the order of the first input.
    """
    names = list(loaded.keys())
    if not names:
        raise ValueError("No inputs.")

    keys_by_name: Dict[str, np.ndarray] = {}
    for name in names:
        keys = _window_keys(loaded[name])
        if keys is None:
            raise ValueError(f"Missing traj_idx/start_t in {name}; cannot align.")
        keys = np.asarray(keys, dtype=np.int64).reshape(-1)
        if int(np.unique(keys).size) != int(keys.size):
            raise ValueError(f"Duplicate window ids detected in {name}; cannot align safely.")
        keys_by_name[name] = keys

    common = keys_by_name[names[0]]
    for name in names[1:]:
        common = np.intersect1d(common, keys_by_name[name], assume_unique=False)

    if common.size == 0:
        raise ValueError("No common windows across inputs (traj_idx/start_t intersection is empty).")

    # Preserve reference order (the first input).
    ref_keys = keys_by_name[names[0]]
    keep_mask = np.isin(ref_keys, common)
    order = ref_keys[keep_mask]

    stats: Dict[str, object] = {
        "aligned_by": "traj_idx_start_t",
        "common_N": int(order.size),
        "dropped": {},
    }

    aligned: Dict[str, Dict[str, np.ndarray]] = {}
    for name in names:
        data = loaded[name]
        keys = keys_by_name[name]
        idx_map = {int(k): int(i) for i, k in enumerate(keys)}
        idx = np.asarray([idx_map[int(k)] for k in order], dtype=np.int64)
        stats["dropped"][name] = int(keys.size - idx.size)

        out: Dict[str, np.ndarray] = {}
        for k, v in data.items():
            if k in ("preds", "targets", "start_pos", "traj_idx", "start_t"):
                out[k] = np.asarray(v)[idx]
            elif k == "preds_k":
                out[k] = np.asarray(v)[idx]
            else:
                out[k] = v
        aligned[name] = out

    return aligned, stats


def _targets_hash(targets: np.ndarray) -> str:
    return _array_hash(targets)


def _array_hash(x: np.ndarray) -> str:
    import hashlib

    arr = np.asarray(x, dtype=np.float32)
    h = hashlib.sha256(arr.tobytes()).hexdigest()
    return h[:16]


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _polyline_arclength(points: np.ndarray) -> Tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points (T,2), got {points.shape}")
    if points.shape[0] < 2:
        return np.zeros((points.shape[0],), dtype=np.float32), 0.0
    seg = points[1:] - points[:-1]
    seg_len = np.linalg.norm(seg, axis=1).astype(np.float32)
    s = np.concatenate([[0.0], np.cumsum(seg_len)], axis=0).astype(np.float32)
    return s, float(s[-1])


def _interp_polyline(points: np.ndarray, s_nodes: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    s_nodes = np.asarray(s_nodes, dtype=np.float32)
    s_query = np.asarray(s_query, dtype=np.float32)
    y = np.interp(s_query, s_nodes, points[:, 0]).astype(np.float32)
    x = np.interp(s_query, s_nodes, points[:, 1]).astype(np.float32)
    return np.stack([y, x], axis=-1).astype(np.float32)


def _turn_angles_spatial(
    points: np.ndarray,
    *,
    ds: float,
    lag: float,
    offset_fracs: Sequence[float],
    min_vec_len: float,
) -> np.ndarray:
    """
    Spatial-scale heading change:
      Δθ(L) = wrap(angle(p(s+L)-p(s)) - angle(p(s)-p(s-L)))
    Returned as |Δθ| in [0, π].
    """
    points = np.asarray(points, dtype=np.float32)
    s_nodes, total = _polyline_arclength(points)
    if not np.isfinite(total) or total <= 0:
        return np.zeros((0,), dtype=np.float32)
    if float(lag) <= 0 or float(ds) <= 0:
        raise ValueError("lag and ds must be > 0")
    if total <= 2.0 * float(lag):
        return np.zeros((0,), dtype=np.float32)

    out: List[np.ndarray] = []
    for frac in offset_fracs:
        frac_f = float(frac)
        if not np.isfinite(frac_f):
            continue
        offset = frac_f * float(ds)
        s = np.arange(float(lag) + offset, total - float(lag) + 1e-6, float(ds), dtype=np.float32)
        if s.size == 0:
            continue
        p0 = _interp_polyline(points, s_nodes, s)
        pm = _interp_polyline(points, s_nodes, s - float(lag))
        pp = _interp_polyline(points, s_nodes, s + float(lag))
        v_fwd = pp - p0
        v_back = p0 - pm
        n1 = np.linalg.norm(v_fwd, axis=1)
        n2 = np.linalg.norm(v_back, axis=1)
        valid = (n1 > float(min_vec_len)) & (n2 > float(min_vec_len))
        if not np.any(valid):
            continue
        ang_fwd = np.arctan2(v_fwd[:, 0], v_fwd[:, 1])
        ang_back = np.arctan2(v_back[:, 0], v_back[:, 1])
        dtheta = _wrap_pi(ang_fwd - ang_back)
        out.append(np.abs(dtheta[valid]).astype(np.float32))
    if not out:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(out, axis=0).astype(np.float32)


def _max_lateral_deviation_ratio(points: np.ndarray) -> float:
    """
    Max perpendicular deviation from the chord line (start->end), normalized by chord length.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return 0.0
    a = points[0].astype(np.float64)
    b = points[-1].astype(np.float64)
    ab = b - a
    denom = float(np.linalg.norm(ab)) + 1e-12
    chord = float(np.linalg.norm(ab))
    if chord <= 1e-6:
        return 0.0
    ap = points.astype(np.float64) - a[None, :]
    cross = np.abs(ab[0] * ap[:, 1] - ab[1] * ap[:, 0])
    d = cross / denom
    d[0] = 0.0
    d[-1] = 0.0
    return float(np.max(d) / chord)


def _path_length_ratio(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return 1.0
    seg = points[1:] - points[:-1]
    length = float(np.sum(np.linalg.norm(seg, axis=1)))
    chord = float(np.linalg.norm(points[-1] - points[0])) + 1e-12
    return float(length / chord)


def _hist_edges_from_gt(values: np.ndarray, *, bins: int, pctl: Tuple[float, float], clamp_min: float, clamp_max: Optional[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        lo, hi = float(clamp_min), float(clamp_min) + 1e-6
    else:
        lo, hi = np.percentile(values, [float(pctl[0]), float(pctl[1])]).tolist()
        lo = max(float(lo), float(clamp_min))
        if clamp_max is not None:
            hi = min(float(hi), float(clamp_max))
        if not np.isfinite(lo) or not np.isfinite(hi) or float(hi) <= float(lo):
            lo, hi = float(np.min(values)), float(np.max(values))
        if float(hi) <= float(lo):
            hi = float(lo) + 1e-6
    return np.linspace(float(lo), float(hi), num=int(bins) + 1, dtype=np.float64)


def _per_condition_hist_turn(
    polylines: Sequence[np.ndarray],
    *,
    ds: float,
    lag: float,
    offset_fracs: Sequence[float],
    min_vec_len: float,
    turn_edges: np.ndarray,
) -> np.ndarray:
    n = len(polylines)
    h = np.zeros((n, int(turn_edges.size - 1)), dtype=np.int64)
    for i, pts in enumerate(polylines):
        ang = _turn_angles_spatial(pts, ds=float(ds), lag=float(lag), offset_fracs=offset_fracs, min_vec_len=float(min_vec_len))
        if ang.size == 0:
            continue
        h[i], _ = np.histogram(ang, bins=turn_edges)
    return h


def _per_condition_hist_scalar(values_per_cond: Sequence[np.ndarray], edges: np.ndarray) -> np.ndarray:
    n = len(values_per_cond)
    h = np.zeros((n, int(edges.size - 1)), dtype=np.int64)
    for i, vals in enumerate(values_per_cond):
        v = np.asarray(vals, dtype=np.float64).reshape(-1)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        h[i], _ = np.histogram(v, bins=edges)
    return h


def _bootstrap_jsd(
    pred_hist: np.ndarray,
    gt_hist: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    pred_hist = np.asarray(pred_hist, dtype=np.int64)
    gt_hist = np.asarray(gt_hist, dtype=np.int64)
    if pred_hist.ndim != 2 or gt_hist.ndim != 2 or pred_hist.shape != gt_hist.shape:
        raise ValueError("Expected pred_hist and gt_hist as (N,B) with same shape")
    n = int(pred_hist.shape[0])
    if n == 0:
        return {"mean": 0.0, "p2_5": 0.0, "p97_5": 0.0}
    if int(n_boot) <= 0:
        jsd = jsd_from_hist(np.sum(pred_hist, axis=0), np.sum(gt_hist, axis=0))
        return {"mean": float(jsd), "p2_5": float(jsd), "p97_5": float(jsd)}

    rng = np.random.default_rng(int(seed))
    vals = np.zeros((int(n_boot),), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=(n,), endpoint=False)
        p = np.sum(pred_hist[idx], axis=0)
        q = np.sum(gt_hist[idx], axis=0)
        vals[b] = jsd_from_hist(p, q)
    return {
        "mean": float(np.mean(vals)),
        "p2_5": float(np.percentile(vals, 2.5)),
        "p97_5": float(np.percentile(vals, 97.5)),
    }


def _noise_floor_jsd_split_half(gt_hist: np.ndarray, *, n_splits: int, seed: int) -> Dict[str, float]:
    gt_hist = np.asarray(gt_hist, dtype=np.int64)
    if gt_hist.ndim != 2:
        raise ValueError("Expected gt_hist (N,B)")
    n = int(gt_hist.shape[0])
    if n < 2:
        return {"mean": 0.0, "p2_5": 0.0, "p97_5": 0.0}
    if int(n_splits) <= 0:
        return {"mean": 0.0, "p2_5": 0.0, "p97_5": 0.0}

    rng = np.random.default_rng(int(seed))
    vals = np.zeros((int(n_splits),), dtype=np.float64)
    for i in range(int(n_splits)):
        perm = rng.permutation(n)
        a = perm[: n // 2]
        b = perm[n // 2 :]
        pa = np.sum(gt_hist[a], axis=0)
        pb = np.sum(gt_hist[b], axis=0)
        vals[i] = jsd_from_hist(pa, pb)
    return {
        "mean": float(np.mean(vals)),
        "p2_5": float(np.percentile(vals, 2.5)),
        "p97_5": float(np.percentile(vals, 97.5)),
    }


def _stack_polyline(start_pos: np.ndarray, traj: np.ndarray) -> np.ndarray:
    start_pos = np.asarray(start_pos, dtype=np.float32).reshape(1, 2)
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] != 2:
        raise ValueError(f"Expected traj (F,2), got {traj.shape}")
    return np.concatenate([start_pos, traj], axis=0).astype(np.float32)


def _build_condition_payload(
    data: Dict[str, np.ndarray],
    *,
    use_all_k: bool,
    k_max: int,
) -> Tuple[List[List[np.ndarray]], np.ndarray, np.ndarray]:
    targets = np.asarray(data["targets"], dtype=np.float32)  # (N,F,2)
    start_pos = np.asarray(data.get("start_pos", targets[:, 0]), dtype=np.float32)  # (N,2)
    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError(f"Expected targets (N,F,2), got {targets.shape}")
    if start_pos.ndim != 2 or start_pos.shape[-1] != 2:
        raise ValueError(f"Expected start_pos (N,2), got {start_pos.shape}")
    if targets.shape[0] != start_pos.shape[0]:
        raise ValueError("N mismatch between targets and start_pos")

    N = int(targets.shape[0])

    if bool(use_all_k) and "preds_k" in data:
        preds_k = np.asarray(data["preds_k"], dtype=np.float32)  # (N,K,F,2)
        if preds_k.ndim != 4 or preds_k.shape[0] != N or preds_k.shape[-1] != 2:
            raise ValueError(f"Expected preds_k (N,K,F,2), got {preds_k.shape}")
        if int(k_max) > 0:
            preds_k = preds_k[:, : int(k_max)]
        pred_polylines_per_cond: List[List[np.ndarray]] = []
        for i in range(N):
            polys = [_stack_polyline(start_pos[i], preds_k[i, k]) for k in range(int(preds_k.shape[1]))]
            pred_polylines_per_cond.append(polys)
        return pred_polylines_per_cond, targets, start_pos

    preds = np.asarray(data.get("preds"), dtype=np.float32)  # (N,F,2)
    if preds.ndim != 3 or preds.shape[0] != N or preds.shape[-1] != 2:
        raise ValueError(f"Expected preds (N,F,2), got {preds.shape}")
    pred_polylines_per_cond = [[_stack_polyline(start_pos[i], preds[i])] for i in range(N)]
    return pred_polylines_per_cond, targets, start_pos


def compute_report(
    inputs: Dict[str, str],
    *,
    use_all_k: bool,
    k_max: int,
    ds: float,
    lags: Sequence[float],
    offset_fracs: Sequence[float],
    min_vec_len: float,
    detour_pct: float,
    detour_score: str,
    detour_lag: Optional[float],
    bins: BinConfig,
    n_boot: int,
    n_splits: int,
    seed: int,
    max_n: Optional[int],
) -> Dict[str, object]:
    loaded: Dict[str, Dict[str, np.ndarray]] = {}
    targets_hash_by_name: Dict[str, str] = {}

    for name, path_str in inputs.items():
        data = _load_npz(Path(path_str))
        if max_n is not None:
            n = int(max_n)
            data["targets"] = np.asarray(data["targets"][:n])
            if "start_pos" in data:
                data["start_pos"] = np.asarray(data["start_pos"][:n])
            if "preds" in data:
                data["preds"] = np.asarray(data["preds"][:n])
            if "preds_k" in data:
                data["preds_k"] = np.asarray(data["preds_k"][:n])
            if "traj_idx" in data:
                data["traj_idx"] = np.asarray(data["traj_idx"][:n])
            if "start_t" in data:
                data["start_t"] = np.asarray(data["start_t"][:n])
        loaded[name] = data

        targets_hash_by_name[name] = _targets_hash(data["targets"])

    # If targets mismatch, try aligning by (traj_idx,start_t) when available.
    unique_hashes = set(targets_hash_by_name.values())
    align_stats: Optional[Dict[str, object]] = None
    if len(unique_hashes) != 1:
        if all(("traj_idx" in d and "start_t" in d) for d in loaded.values()):
            loaded, align_stats = _align_loaded_by_window_ids(loaded)
            # After alignment, allow tiny float noise (normalization/denorm + integration) but catch real mismatches.
            ref_name = next(iter(loaded.keys()))
            ref_targets = np.asarray(loaded[ref_name]["targets"], dtype=np.float32)
            ref_start = np.asarray(loaded[ref_name].get("start_pos", ref_targets[:, 0]), dtype=np.float32)
            for name, data in loaded.items():
                cur_targets = np.asarray(data["targets"], dtype=np.float32)
                max_abs = float(np.max(np.abs(cur_targets - ref_targets)))
                if max_abs > 1e-3:
                    raise ValueError(
                        "Aligned by (traj_idx,start_t) but targets still differ "
                        f"(method={name}, max|Δ|={max_abs:.6g}). "
                        "This usually means different processed data or inconsistent normalization stats."
                    )
                # Canonicalize targets to the reference (targets are only used for GT / bookkeeping).
                data["targets"] = ref_targets
                if "start_pos" in data:
                    cur_start = np.asarray(data["start_pos"], dtype=np.float32)
                    max_abs_s = float(np.max(np.abs(cur_start - ref_start)))
                    if max_abs_s > 1e-3:
                        raise ValueError(
                            "Aligned by (traj_idx,start_t) but start_pos differs "
                            f"(method={name}, max|Δ|={max_abs_s:.6g}). "
                            "This usually means different processed data or inconsistent normalization stats."
                        )
            targets_hash_by_name = {k: _targets_hash(v["targets"]) for k, v in loaded.items()}
        else:
            detail = {k: v for k, v in targets_hash_by_name.items()}
            raise ValueError(
                "GT mismatch across inputs (different sampled windows).\n"
                f"targets_hash={json.dumps(detail, ensure_ascii=False)}\n"
                "Fix: re-dump samples.npz with stable ids (traj_idx/start_t) and ensure all methods use the same window set."
            )

    # Set GT from the first input (after optional alignment).
    first_name = next(iter(loaded.keys()))
    gt_targets = np.asarray(loaded[first_name]["targets"], dtype=np.float32)
    gt_start = np.asarray(loaded[first_name].get("start_pos", gt_targets[:, 0]), dtype=np.float32)
    gt_hash = targets_hash_by_name[first_name]
    N = int(gt_targets.shape[0])

    # --- Build GT polylines + per-condition scalar scores (for detour subset) ---
    gt_polylines = [_stack_polyline(gt_start[i], gt_targets[i]) for i in range(N)]
    gt_dev_ratio = np.asarray([_max_lateral_deviation_ratio(p) for p in gt_polylines], dtype=np.float64)
    gt_len_ratio = np.asarray([_path_length_ratio(p) for p in gt_polylines], dtype=np.float64)

    if detour_score == "max_dev_ratio":
        score = gt_dev_ratio
    elif detour_score == "len_ratio":
        score = gt_len_ratio
    elif detour_score == "turn_abs_mean":
        if detour_lag is None:
            raise ValueError("--detour_score turn_abs_mean requires --detour_lag")
        vals = []
        for p in gt_polylines:
            ang = _turn_angles_spatial(p, ds=float(ds), lag=float(detour_lag), offset_fracs=offset_fracs, min_vec_len=float(min_vec_len))
            vals.append(float(np.mean(ang)) if ang.size else 0.0)
        score = np.asarray(vals, dtype=np.float64)
    else:
        raise ValueError(f"Unknown detour_score={detour_score}")

    detour_pct_f = float(detour_pct)
    if detour_pct_f <= 0 or detour_pct_f >= 100:
        detour_mask = np.ones((N,), dtype=bool)
        detour_thr = float("nan")
    else:
        detour_thr = float(np.percentile(score, 100.0 - detour_pct_f))
        detour_mask = score >= detour_thr

    # --- Build bin edges (locked by GT only) ---
    turn_edges = np.linspace(0.0, float(np.pi), num=int(bins.turn_bins) + 1, dtype=np.float64)
    dev_edges = _hist_edges_from_gt(
        gt_dev_ratio,
        bins=int(bins.dev_bins),
        pctl=bins.range_percentiles,
        clamp_min=0.0,
        clamp_max=None,
    )
    len_edges = _hist_edges_from_gt(
        gt_len_ratio,
        bins=int(bins.len_bins),
        pctl=bins.range_percentiles,
        clamp_min=1.0,
        clamp_max=None,
    )

    # Precompute GT per-condition histograms (shared for all methods)
    gt_turn_hists: Dict[str, np.ndarray] = {}
    for lag in lags:
        gt_turn_hists[str(lag)] = _per_condition_hist_turn(
            gt_polylines,
            ds=float(ds),
            lag=float(lag),
            offset_fracs=offset_fracs,
            min_vec_len=float(min_vec_len),
            turn_edges=turn_edges,
        )
    gt_dev_hist = _per_condition_hist_scalar([np.asarray([v]) for v in gt_dev_ratio], dev_edges)
    gt_len_hist = _per_condition_hist_scalar([np.asarray([v]) for v in gt_len_ratio], len_edges)

    # Noise floors (GT split-half) for each metric
    noise: Dict[str, Dict[str, float]] = {}
    for lag in lags:
        noise[f"turn@{lag}"] = _noise_floor_jsd_split_half(gt_turn_hists[str(lag)], n_splits=int(n_splits), seed=int(seed) + 17)
    noise["max_dev_ratio"] = _noise_floor_jsd_split_half(gt_dev_hist, n_splits=int(n_splits), seed=int(seed) + 29)
    noise["len_ratio"] = _noise_floor_jsd_split_half(gt_len_hist, n_splits=int(n_splits), seed=int(seed) + 31)

    out: Dict[str, object] = {
        "config": {
            "use_all_k": bool(use_all_k),
            "k_max": int(k_max),
            "ds": float(ds),
            "lags": [float(x) for x in lags],
            "offset_fracs": [float(x) for x in offset_fracs],
            "min_vec_len": float(min_vec_len),
            "detour_pct": float(detour_pct),
            "detour_score": str(detour_score),
            "detour_lag": (float(detour_lag) if detour_lag is not None else None),
            "bins": {
                "turn_bins": int(bins.turn_bins),
                "dev_bins": int(bins.dev_bins),
                "len_bins": int(bins.len_bins),
                "range_percentiles": [float(bins.range_percentiles[0]), float(bins.range_percentiles[1])],
            },
            "bootstrap": {"n_boot": int(n_boot), "n_splits": int(n_splits), "seed": int(seed)},
        },
        "stats": {
            "N": int(N),
            "targets_hash": str(gt_hash),
            "detour": {"size": int(np.sum(detour_mask)), "thr": float(detour_thr)},
        },
        "bin_edges": {
            "turn": turn_edges.tolist(),
            "max_dev_ratio": dev_edges.tolist(),
            "len_ratio": len_edges.tolist(),
        },
        "noise_floor": noise,
        "metrics": {},
    }
    if align_stats is not None:
        out["stats"]["alignment"] = align_stats

    # --- Evaluate each method vs GT (overall + detour subset) ---
    for name, data in loaded.items():
        pred_polys_per_cond, _, _ = _build_condition_payload(data, use_all_k=use_all_k, k_max=int(k_max))
        # Turn hist per condition for each lag: sum over K per condition.
        metrics: Dict[str, object] = {"overall": {}, "detour": {}}

        for lag in lags:
            pred_hist_cond = np.zeros_like(gt_turn_hists[str(lag)], dtype=np.int64)
            for i in range(N):
                h_i = _per_condition_hist_turn(
                    pred_polys_per_cond[i],
                    ds=float(ds),
                    lag=float(lag),
                    offset_fracs=offset_fracs,
                    min_vec_len=float(min_vec_len),
                    turn_edges=turn_edges,
                )
                pred_hist_cond[i] = np.sum(h_i, axis=0)

            metrics["overall"][f"JSD_turn@{lag}"] = _bootstrap_jsd(
                pred_hist_cond,
                gt_turn_hists[str(lag)],
                n_boot=int(n_boot),
                seed=int(seed) + 101,
            )
            if int(np.sum(detour_mask)) > 0:
                metrics["detour"][f"JSD_turn@{lag}"] = _bootstrap_jsd(
                    pred_hist_cond[detour_mask],
                    gt_turn_hists[str(lag)][detour_mask],
                    n_boot=int(n_boot),
                    seed=int(seed) + 201,
                )

        # Scalars: max_dev_ratio, len_ratio
        pred_dev_vals: List[np.ndarray] = []
        pred_len_vals: List[np.ndarray] = []
        for i in range(N):
            polys = pred_polys_per_cond[i]
            pred_dev_vals.append(np.asarray([_max_lateral_deviation_ratio(p) for p in polys], dtype=np.float64))
            pred_len_vals.append(np.asarray([_path_length_ratio(p) for p in polys], dtype=np.float64))

        pred_dev_hist = _per_condition_hist_scalar(pred_dev_vals, dev_edges)
        pred_len_hist = _per_condition_hist_scalar(pred_len_vals, len_edges)
        metrics["overall"]["JSD_max_dev_ratio"] = _bootstrap_jsd(pred_dev_hist, gt_dev_hist, n_boot=int(n_boot), seed=int(seed) + 303)
        metrics["overall"]["JSD_len_ratio"] = _bootstrap_jsd(pred_len_hist, gt_len_hist, n_boot=int(n_boot), seed=int(seed) + 307)
        if int(np.sum(detour_mask)) > 0:
            metrics["detour"]["JSD_max_dev_ratio"] = _bootstrap_jsd(
                pred_dev_hist[detour_mask],
                gt_dev_hist[detour_mask],
                n_boot=int(n_boot),
                seed=int(seed) + 403,
            )
            metrics["detour"]["JSD_len_ratio"] = _bootstrap_jsd(
                pred_len_hist[detour_mask],
                gt_len_hist[detour_mask],
                n_boot=int(n_boot),
                seed=int(seed) + 409,
            )

        out["metrics"][name] = metrics

    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detour validity metrics (spatial-scale, CI, detour subset). CPU-only.")
    p.add_argument("--inputs", type=str, nargs="+", required=True, help="Repeatable: 'Label:/path/to/samples.npz'")
    p.add_argument("--use_all_k", action="store_true", help="If preds_k exists, use all K (flattened per condition).")
    p.add_argument("--k_max", type=int, default=10, help="Cap K when --use_all_k is on (<=0 means all).")

    p.add_argument("--ds", type=float, default=0.5, help="Arc-length sampling step in grid units.")
    p.add_argument("--lags", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0], help="Spatial lags in grid units.")
    p.add_argument("--offset_fracs", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75], help="Offsets as fractions of ds (phase aggregation).")
    p.add_argument("--min_vec_len", type=float, default=1e-3, help="Min vector length to define heading (grid units).")

    p.add_argument("--detour_pct", type=float, default=10.0, help="Define detour subset as top pct (e.g., 10 => top 10%).")
    p.add_argument(
        "--detour_score",
        type=str,
        default="max_dev_ratio",
        choices=["max_dev_ratio", "len_ratio", "turn_abs_mean"],
        help="Which GT-derived score defines the detour subset.",
    )
    p.add_argument("--detour_lag", type=float, default=None, help="When detour_score=turn_abs_mean, use this lag.")

    p.add_argument("--turn_bins", type=int, default=60)
    p.add_argument("--dev_bins", type=int, default=80)
    p.add_argument("--len_bins", type=int, default=80)
    p.add_argument("--range_percentiles", type=float, nargs=2, default=[0.5, 99.5], help="Percentiles for scalar bin edges (GT).")

    p.add_argument("--bootstrap", type=int, default=200, help="Bootstrap replicates for JSD CI (<=0 disables).")
    p.add_argument("--noise_splits", type=int, default=200, help="GT split-half replicates for noise floor (<=0 disables).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_n", type=int, default=None, help="Optional limit on number of conditions.")

    p.add_argument("--out_json", type=str, default=None, help="Optional output JSON path.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    inputs = _parse_inputs(args.inputs)

    bins = BinConfig(
        turn_bins=int(args.turn_bins),
        dev_bins=int(args.dev_bins),
        len_bins=int(args.len_bins),
        range_percentiles=(float(args.range_percentiles[0]), float(args.range_percentiles[1])),
    )

    report = compute_report(
        inputs,
        use_all_k=bool(args.use_all_k),
        k_max=int(args.k_max),
        ds=float(args.ds),
        lags=[float(x) for x in args.lags],
        offset_fracs=[float(x) for x in args.offset_fracs],
        min_vec_len=float(args.min_vec_len),
        detour_pct=float(args.detour_pct),
        detour_score=str(args.detour_score),
        detour_lag=(float(args.detour_lag) if args.detour_lag is not None else None),
        bins=bins,
        n_boot=int(args.bootstrap),
        n_splits=int(args.noise_splits),
        seed=int(args.seed),
        max_n=(int(args.max_n) if args.max_n is not None else None),
    )

    # Compact stdout summary (stable keys; detailed report in JSON)
    print("[OK] Detour validity report")
    print(json.dumps(report["stats"], indent=2))
    for name, m in report["metrics"].items():
        overall = m["overall"]
        detour = m["detour"]
        print(f"- {name}:")
        if overall:
            for k in sorted(overall.keys()):
                v = overall[k]
                print(f"  overall {k}: mean={v['mean']:.4f} (CI {v['p2_5']:.4f}..{v['p97_5']:.4f})")
        if detour:
            for k in sorted(detour.keys()):
                v = detour[k]
                print(f"  detour  {k}: mean={v['mean']:.4f} (CI {v['p2_5']:.4f}..{v['p97_5']:.4f})")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
