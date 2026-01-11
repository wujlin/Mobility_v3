from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.features.waypoints import WaypointConfig, extract_oracle_waypoints_from_future
from src.plot_style import FIGSIZE_FULL, OKABE_ITO, add_panel_label, paper_style, save_figure

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class AuditConfig:
    od_bin: float
    min_bucket_n: int
    min_cluster_frac: float
    sep_thr: float
    max_buckets: int
    num_cases: int
    num_unimodal_cases: int
    max_traj_per_case: int
    seed: int
    jacc_cell: float
    waypoint_mode: str
    waypoint_turn_alpha: float
    num_waypoints: int
    save_case_npz: bool
    save_png_preview: bool


def _hour_and_dow_from_epoch(start_t: np.ndarray, *, tz_offset_hours: float) -> Tuple[np.ndarray, np.ndarray]:
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    start_t_local = start_t + int(float(tz_offset_hours) * 3600)
    seconds_per_day = 86400
    days_since_epoch = start_t_local // seconds_per_day
    seconds_in_day = start_t_local % seconds_per_day
    hour = (seconds_in_day // 3600).astype(np.int64)
    dow = ((days_since_epoch + 3) % 7).astype(np.int64)  # 0=Mon,...,6=Sun
    return hour, dow


def _od_stats(
    *,
    start_pos: np.ndarray,
    dest_pos: Optional[np.ndarray],
) -> Dict[str, object]:
    start_pos = np.asarray(start_pos, dtype=np.float64)
    out: Dict[str, object] = {
        "start_mean_yx": [float(np.mean(start_pos[:, 0])), float(np.mean(start_pos[:, 1]))],
        "start_std_yx": [float(np.std(start_pos[:, 0])), float(np.std(start_pos[:, 1]))],
        "start_min_yx": [float(np.min(start_pos[:, 0])), float(np.min(start_pos[:, 1]))],
        "start_max_yx": [float(np.max(start_pos[:, 0])), float(np.max(start_pos[:, 1]))],
    }
    if dest_pos is None:
        return out
    dest_pos = np.asarray(dest_pos, dtype=np.float64)
    out.update(
        {
            "dest_mean_yx": [float(np.mean(dest_pos[:, 0])), float(np.mean(dest_pos[:, 1]))],
            "dest_std_yx": [float(np.std(dest_pos[:, 0])), float(np.std(dest_pos[:, 1]))],
            "dest_min_yx": [float(np.min(dest_pos[:, 0])), float(np.min(dest_pos[:, 1]))],
            "dest_max_yx": [float(np.max(dest_pos[:, 0])), float(np.max(dest_pos[:, 1]))],
        }
    )
    return out


def _time_stats(start_t: Optional[np.ndarray], *, tz_offset_hours: float) -> Optional[Dict[str, object]]:
    if start_t is None:
        return None
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    if start_t.size == 0:
        return None
    # Heuristic: only treat as epoch seconds when values look like unix seconds.
    if int(np.max(start_t)) < 1_000_000_000:
        return {"effective": "zeros", "reason": "start_t_not_epoch_seconds"}
    hour, dow = _hour_and_dow_from_epoch(start_t, tz_offset_hours=float(tz_offset_hours))
    return {
        "effective": "simple",
        "tz_offset_hours": float(tz_offset_hours),
        "hour_hist_24": np.bincount(hour, minlength=24).astype(np.int64).tolist(),
        "dow_hist_7": np.bincount(dow, minlength=7).astype(np.int64).tolist(),
        "start_t_min": int(np.min(start_t)),
        "start_t_max": int(np.max(start_t)),
    }


def _keys_from_od(start_pos: np.ndarray, end_pos: np.ndarray, *, od_bin: float) -> np.ndarray:
    b = max(float(od_bin), 1e-6)
    s = np.rint(start_pos / b).astype(np.int64)
    e = np.rint(end_pos / b).astype(np.int64)
    return np.concatenate([s, e], axis=1)  # (N,4) [sy,sx,ey,ex]


def _polyline_features_to_dest(
    start_pos: np.ndarray,
    targets: np.ndarray,
    dest_pos: np.ndarray,
) -> np.ndarray:
    """
    Features for corridor clustering with global destination as endpoint.
    Returns (N,3): signed_dev_ratio, progress_ratio, len_ratio
    """
    start_pos = np.asarray(start_pos, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    dest_pos = np.asarray(dest_pos, dtype=np.float32)
    N, F, _ = targets.shape

    poly = np.concatenate([start_pos[:, None, :], targets], axis=1)  # (N,F+1,2)
    a = start_pos.astype(np.float64)
    b = dest_pos.astype(np.float64)
    ab = b - a
    chord = np.linalg.norm(ab, axis=1) + 1e-12

    ap = poly.astype(np.float64) - a[:, None, :]
    cross = ab[:, None, 0] * ap[:, :, 1] - ab[:, None, 1] * ap[:, :, 0]
    dist_signed = cross / chord[:, None]
    dist_signed[:, 0] = 0.0
    idx = np.argmax(np.abs(dist_signed), axis=1)
    dev_signed = dist_signed[np.arange(N), idx]
    signed_dev_ratio = (dev_signed / chord).astype(np.float32)

    end_seg = poly[:, -1, :].astype(np.float64)
    proj = np.sum((end_seg - a) * ab, axis=1) / (chord * chord)
    progress_ratio = proj.astype(np.float32)

    seg = poly[:, 1:, :] - poly[:, :-1, :]
    seg_len = np.linalg.norm(seg, axis=2).astype(np.float64)
    path_len = np.sum(seg_len, axis=1)
    len_ratio = (path_len / chord).astype(np.float32)

    return np.stack([signed_dev_ratio, progress_ratio, len_ratio], axis=1)


def _polyline_features_segment_end(start_pos: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Fallback features with segment end as endpoint.
    Returns (N,3): signed_dev_ratio, s_frac, len_ratio
    """
    start_pos = np.asarray(start_pos, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    N, F, _ = targets.shape

    poly = np.concatenate([start_pos[:, None, :], targets], axis=1)  # (N,F+1,2)
    a = poly[:, 0, :].astype(np.float64)
    b = poly[:, -1, :].astype(np.float64)
    ab = b - a
    chord = np.linalg.norm(ab, axis=1) + 1e-12

    ap = poly.astype(np.float64) - a[:, None, :]
    cross = ab[:, None, 0] * ap[:, :, 1] - ab[:, None, 1] * ap[:, :, 0]
    dist_signed = cross / chord[:, None]
    dist_signed[:, 0] = 0.0
    dist_signed[:, -1] = 0.0
    idx = np.argmax(np.abs(dist_signed), axis=1)
    dev_signed = dist_signed[np.arange(N), idx]
    signed_dev_ratio = (dev_signed / chord).astype(np.float32)

    seg = poly[:, 1:, :] - poly[:, :-1, :]
    seg_len = np.linalg.norm(seg, axis=2).astype(np.float64)
    s = np.concatenate([np.zeros((N, 1), dtype=np.float64), np.cumsum(seg_len, axis=1)], axis=1)
    total = s[:, -1] + 1e-12
    s_frac = (s[np.arange(N), idx] / total).astype(np.float32)

    path_len = np.sum(seg_len, axis=1)
    len_ratio = (path_len / chord).astype(np.float32)

    return np.stack([signed_dev_ratio, s_frac, len_ratio], axis=1)


def _kmeans2(x: np.ndarray, *, seed: int, iters: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n < 2:
        return np.zeros((n,), dtype=np.int64), np.zeros((2, d), dtype=np.float64)

    i0 = int(np.argmin(x[:, 0]))
    i1 = int(np.argmax(x[:, 0]))
    if i0 == i1:
        rng = np.random.default_rng(int(seed))
        i1 = int(rng.integers(0, n))
    c = np.stack([x[i0], x[i1]], axis=0)

    labels = np.zeros((n,), dtype=np.int64)
    for _ in range(int(iters)):
        d0 = np.sum((x - c[0]) ** 2, axis=1)
        d1 = np.sum((x - c[1]) ** 2, axis=1)
        new_labels = (d1 < d0).astype(np.int64)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for k in (0, 1):
            mask = labels == k
            if not np.any(mask):
                continue
            c[k] = np.mean(x[mask], axis=0)
    return labels.astype(np.int64, copy=False), c


def _cluster_two_modes(
    feats: np.ndarray,
    *,
    min_cluster_frac: float,
    sep_thr: float,
    seed: int,
) -> Dict[str, object]:
    feats = np.asarray(feats, dtype=np.float64)
    n = int(feats.shape[0])
    if n < 2:
        return {"multimodal": False}

    mu = np.mean(feats, axis=0)
    sig = np.std(feats, axis=0) + 1e-6
    x = (feats - mu) / sig

    labels, centers = _kmeans2(x, seed=int(seed))
    n0 = int(np.sum(labels == 0))
    n1 = int(n - n0)
    frac0 = float(n0) / float(n)
    frac1 = float(n1) / float(n)
    if frac0 < float(min_cluster_frac) or frac1 < float(min_cluster_frac):
        return {
            "multimodal": False,
            "reason": "cluster_too_small",
            "n0": n0,
            "n1": n1,
            "frac0": float(frac0),
            "frac1": float(frac1),
        }

    c0, c1 = centers[0], centers[1]
    sep = float(np.linalg.norm(c0 - c1))
    w0 = x[labels == 0] - c0[None, :]
    w1 = x[labels == 1] - c1[None, :]
    rms0 = float(np.sqrt(np.mean(np.sum(w0 * w0, axis=1)))) if w0.size else 0.0
    rms1 = float(np.sqrt(np.mean(np.sum(w1 * w1, axis=1)))) if w1.size else 0.0
    scatter = float(max(rms0, rms1, 1e-6))
    score = float(sep / scatter)
    multimodal = bool(score >= float(sep_thr))
    return {
        "multimodal": multimodal,
        "score": float(score),
        "sep": float(sep),
        "scatter": float(scatter),
        "n0": n0,
        "n1": n1,
        "frac0": float(frac0),
        "frac1": float(frac1),
        "labels": labels.astype(np.int64, copy=False),
        "centers": centers.astype(np.float64, copy=False),
        "mu": mu.astype(np.float64, copy=False),
        "sig": sig.astype(np.float64, copy=False),
    }


def _occupancy_set_from_polyline(start_pos: np.ndarray, targets: np.ndarray, *, cell: float) -> set[int]:
    c = max(float(cell), 1e-6)
    pts = np.concatenate([start_pos.reshape(1, 2), targets.reshape(-1, 2)], axis=0).astype(np.float64, copy=False)
    yy = np.floor(pts[:, 0] / c).astype(np.int64)
    xx = np.floor(pts[:, 1] / c).astype(np.int64)
    h = (yy << np.int64(32)) ^ (xx & np.int64(0xFFFFFFFF))
    return set(int(v) for v in h.tolist())


def _pairwise_jaccard_distance(sets: List[set[int]], *, max_pairs: int = 20000, seed: int = 0) -> Dict[str, float]:
    n = int(len(sets))
    if n < 2:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    if len(pairs) > int(max_pairs):
        rng = np.random.default_rng(int(seed))
        pick = rng.choice(len(pairs), size=int(max_pairs), replace=False)
        pairs = [pairs[int(k)] for k in pick.tolist()]

    d = np.zeros((len(pairs),), dtype=np.float64)
    for t, (i, j) in enumerate(pairs):
        a = sets[i]
        b = sets[j]
        inter = len(a & b)
        uni = len(a | b)
        jac = 0.0 if uni <= 0 else float(inter) / float(uni)
        d[t] = 1.0 - jac

    return {
        "mean": float(np.mean(d)),
        "p50": float(np.percentile(d, 50)),
        "p90": float(np.percentile(d, 90)),
        "n_pairs": int(d.size),
    }


def _plot_case_gt(
    *,
    out_path: Path,
    start_pos: np.ndarray,  # (N,2)
    targets: np.ndarray,  # (N,F,2)
    dest_pos: Optional[np.ndarray],  # (N,2)
    labels: np.ndarray,  # (N,)
    cfg: AuditConfig,
    case_id: int,
) -> None:
    with paper_style():
        fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
        colors = [OKABE_ITO["blue"], OKABE_ITO["vermillion"]]
        for k in (0, 1):
            idx = np.where(labels == k)[0]
            if idx.size == 0:
                continue
            for i in idx.tolist():
                p0 = start_pos[i]
                path = targets[i]
                ax.plot(path[:, 1], path[:, 0], color=colors[k], alpha=0.12, linewidth=1.2)
                ax.scatter([p0[1]], [p0[0]], color=colors[k], s=8, alpha=0.35)
        if dest_pos is not None:
            # Plot a few destination markers to indicate endpoint region (not per-line to avoid clutter).
            d = dest_pos
            ax.scatter(d[:, 1], d[:, 0], color=OKABE_ITO["gray"], s=4, alpha=0.10)

        # Waypoint audit: plot oracle waypoints for a few random trajectories per cluster.
        rng = np.random.default_rng(int(cfg.seed) + int(case_id))
        wcfg = WaypointConfig(mode=str(cfg.waypoint_mode), num_waypoints=int(cfg.num_waypoints), turn_alpha=float(cfg.waypoint_turn_alpha))
        for k in (0, 1):
            idx = np.where(labels == k)[0]
            if idx.size == 0:
                continue
            pick_n = int(min(6, idx.size))
            pick = rng.choice(idx, size=pick_n, replace=False)
            for i in pick.tolist():
                p0 = start_pos[i]
                fut = targets[i]
                _, wp = extract_oracle_waypoints_from_future(start_pos=p0, future_pos=fut, cfg=wcfg)
                if wp.size > 0:
                    ax.scatter(wp[:, 1], wp[:, 0], color=colors[k], s=18, alpha=0.85, marker="x")

        ax.set_xlabel("x (grid)")
        ax.set_ylabel("y (grid)")
        ax.invert_yaxis()
        add_panel_label(ax, f"case{case_id}")
        save_figure(fig, out_path)
        if bool(cfg.save_png_preview) and str(out_path).lower().endswith(".pdf"):
            save_figure(fig, out_path.with_suffix(".png"))
        plt.close(fig)


def run_audit(*, samples_npz: Path, out_dir: Path, cfg: AuditConfig) -> Dict[str, object]:
    data = np.load(str(samples_npz), allow_pickle=True)
    need = {"start_pos", "targets"}
    if not need.issubset(set(data.files)):
        raise ValueError(f"samples_npz must contain {sorted(need)}, got {sorted(list(data.files))}")

    start_pos = np.asarray(data["start_pos"], dtype=np.float32)
    targets = np.asarray(data["targets"], dtype=np.float32)
    dest_pos = np.asarray(data["dest_pos"], dtype=np.float32) if "dest_pos" in data.files else None
    traj_idx = np.asarray(data["traj_idx"], dtype=np.int64) if "traj_idx" in data.files else None
    start_t = np.asarray(data["start_t"], dtype=np.int64) if "start_t" in data.files else None

    if start_pos.ndim != 2 or start_pos.shape[1] != 2:
        raise ValueError(f"Bad start_pos shape: {start_pos.shape}")
    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError(f"Bad targets shape: {targets.shape}")
    if dest_pos is not None and (dest_pos.ndim != 2 or dest_pos.shape[1] != 2):
        raise ValueError(f"Bad dest_pos shape: {dest_pos.shape}")

    if dest_pos is None:
        end_pos = targets[:, -1, :]
        feats = _polyline_features_segment_end(start_pos, targets)
        od_end = "segment_end"
    else:
        end_pos = dest_pos
        feats = _polyline_features_to_dest(start_pos, targets, dest_pos)
        od_end = "dest_pos"

    keys = _keys_from_od(start_pos, end_pos, od_bin=float(cfg.od_bin))
    buckets: Dict[Tuple[int, int, int, int], List[int]] = {}
    for i in range(int(keys.shape[0])):
        k = tuple(int(x) for x in keys[i].tolist())
        buckets.setdefault(k, []).append(i)

    items = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    if int(cfg.max_buckets) > 0:
        items = items[: int(cfg.max_buckets)]

    rng = np.random.default_rng(int(cfg.seed))
    bucket_reports: List[Dict[str, object]] = []
    for k, idxs in items:
        n = int(len(idxs))
        if n < int(cfg.min_bucket_n):
            continue
        f = feats[np.asarray(idxs, dtype=np.int64)]
        rep = _cluster_two_modes(
            f,
            min_cluster_frac=float(cfg.min_cluster_frac),
            sep_thr=float(cfg.sep_thr),
            seed=int(rng.integers(0, 1_000_000)),
        )
        rep_out = {kk: rep[kk] for kk in rep.keys() if kk not in ("labels", "centers", "mu", "sig")}
        rep_out.update({"key": list(k), "n": int(n)})
        bucket_reports.append(rep_out)

    multimodal = [b for b in bucket_reports if bool(b.get("multimodal"))]
    multimodal = sorted(multimodal, key=lambda r: (int(r.get("n", 0)), float(r.get("score", 0.0))), reverse=True)
    selected_multi = multimodal[: int(cfg.num_cases)]

    # Unimodal controls: require stable two-cluster fit (no 'reason'), but score below sep_thr.
    unimodal = [b for b in bucket_reports if (not bool(b.get("multimodal"))) and (b.get("reason") is None)]
    unimodal = sorted(unimodal, key=lambda r: (float(r.get("score", 0.0)), -int(r.get("n", 0))))
    selected_uni = unimodal[: int(cfg.num_unimodal_cases)]

    selected = [(b, "multimodal") for b in selected_multi] + [(b, "unimodal") for b in selected_uni]

    out_dir.mkdir(parents=True, exist_ok=True)
    cases: List[Dict[str, object]] = []
    for ci, (b, case_type) in enumerate(selected):
        k = tuple(int(x) for x in b["key"])
        idxs_all = buckets.get(k, [])
        if not idxs_all:
            continue
        idxs = np.asarray(idxs_all, dtype=np.int64)

        # Subsample for metrics/plot (keep reproducible).
        if int(cfg.max_traj_per_case) > 0 and idxs.size > int(cfg.max_traj_per_case):
            pick = rng.choice(idxs, size=int(cfg.max_traj_per_case), replace=False)
            idxs = np.sort(pick.astype(np.int64))

        sp = start_pos[idxs]
        tg = targets[idxs]
        dp = dest_pos[idxs] if dest_pos is not None else None

        # Cluster labels for this case (re-fit on case samples for audit stability).
        f_case = feats[idxs]
        rep = _cluster_two_modes(
            f_case,
            min_cluster_frac=float(cfg.min_cluster_frac),
            sep_thr=float(cfg.sep_thr),
            seed=int(rng.integers(0, 1_000_000)),
        )
        labels = np.asarray(rep.get("labels", np.zeros((idxs.size,), dtype=np.int64)), dtype=np.int64)

        occ_sets = [_occupancy_set_from_polyline(sp[i], tg[i], cell=float(cfg.jacc_cell)) for i in range(int(idxs.size))]
        jac = _pairwise_jaccard_distance(occ_sets, seed=int(cfg.seed) + int(ci))

        case_dir = out_dir / f"case_{ci:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        fig_path = case_dir / "gt_corridor_clusters.pdf"
        _plot_case_gt(
            out_path=fig_path,
            start_pos=sp,
            targets=tg,
            dest_pos=dp,
            labels=labels,
            cfg=cfg,
            case_id=int(ci),
        )

        ids = None
        if traj_idx is not None and start_t is not None:
            ids = {
                "traj_idx": traj_idx[idxs].astype(np.int64, copy=False),
                "start_t": start_t[idxs].astype(np.int64, copy=False),
            }
            if bool(cfg.save_case_npz):
                npz_kwargs = {
                    "start_pos": sp.astype(np.float32, copy=False),
                    "targets": tg.astype(np.float32, copy=False),
                    "dest_pos": (dp.astype(np.float32, copy=False) if dp is not None else None),
                    "traj_idx": ids["traj_idx"],
                    "start_t": ids["start_t"],
                }
                npz_kwargs = {kk: vv for kk, vv in npz_kwargs.items() if vv is not None}
                np.savez_compressed(case_dir / "gt_case.npz", **npz_kwargs)

        cases.append(
            {
                "case_id": int(ci),
                "case_type": str(case_type),
                "od_key": list(k),
                "od_end": str(od_end),
                "n_used": int(idxs.size),
                "bucket": {kk: b.get(kk) for kk in ("multimodal", "score", "sep", "scatter", "n0", "n1", "frac0", "frac1", "n", "key", "reason")},
                "cluster": {kk: rep.get(kk) for kk in ("multimodal", "score", "sep", "scatter", "n0", "n1", "frac0", "frac1")},
                "gt_jaccard_distance": jac,
                "od_stats": _od_stats(start_pos=sp, dest_pos=dp),
                "time_stats": _time_stats((ids["start_t"] if ids is not None else None), tz_offset_hours=-5.0),
                "paths": {"gt_corridor_clusters_pdf": str(fig_path)},
                "window_ids": (
                    {
                        "traj_idx": ids["traj_idx"].tolist(),
                        "start_t": ids["start_t"].tolist(),
                    }
                    if ids is not None
                    else None
                ),
            }
        )

    report = {
        "inputs": {"samples_npz": str(samples_npz)},
        "config": {
            "od_end": str(od_end),
            "od_bin": float(cfg.od_bin),
            "min_bucket_n": int(cfg.min_bucket_n),
            "min_cluster_frac": float(cfg.min_cluster_frac),
            "sep_thr": float(cfg.sep_thr),
            "max_buckets": int(cfg.max_buckets),
            "num_cases": int(cfg.num_cases),
            "num_unimodal_cases": int(cfg.num_unimodal_cases),
            "max_traj_per_case": int(cfg.max_traj_per_case),
            "seed": int(cfg.seed),
            "jacc_cell": float(cfg.jacc_cell),
            "waypoint_mode": str(cfg.waypoint_mode),
            "waypoint_turn_alpha": float(cfg.waypoint_turn_alpha),
            "num_waypoints": int(cfg.num_waypoints),
            "save_case_npz": bool(cfg.save_case_npz),
            "save_png_preview": bool(cfg.save_png_preview),
        },
        "stats": {
            "N": int(targets.shape[0]),
            "F": int(targets.shape[1]),
            "num_od_buckets_total": int(len(buckets)),
            "num_buckets_reported": int(len(bucket_reports)),
            "num_buckets_multimodal": int(len(multimodal)),
            "num_buckets_unimodal": int(len(unimodal)),
            "num_cases_multimodal": int(len(selected_multi)),
            "num_cases_unimodal": int(len(selected_uni)),
        },
        "top_buckets": bucket_reports[: min(len(bucket_reports), 50)],
        "selected_cases": cases,
    }

    out_json = out_dir / "report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["outputs"] = {"report_json": str(out_json)}
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="E0: GT baseline for route multimodality (case selection + clustering + Jaccard diversity).")
    p.add_argument("--samples_npz", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--od_bin", type=float, default=8.0)
    p.add_argument("--min_bucket_n", type=int, default=30)
    p.add_argument("--min_cluster_frac", type=float, default=0.2)
    p.add_argument("--sep_thr", type=float, default=2.5)
    p.add_argument("--max_buckets", type=int, default=500)

    p.add_argument("--num_cases", type=int, default=5)
    p.add_argument("--num_unimodal_cases", type=int, default=0, help="Also pick unimodal control OD buckets (score < sep_thr).")
    p.add_argument("--max_traj_per_case", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--jacc_cell", type=float, default=8.0, help="Occupancy cell size in grid units for Jaccard diversity.")

    p.add_argument("--waypoint_mode", type=str, default="rdp_dev", choices=["rdp_dev", "rdp_turn"])
    p.add_argument("--waypoint_turn_alpha", type=float, default=1.0, help="When waypoint_mode=rdp_turn: weight for turn-aware waypoint selection.")
    p.add_argument("--num_waypoints", type=int, default=2)
    p.add_argument("--save_case_npz", action="store_true", help="Save per-case gt_case.npz with traj_idx/start_t subset.")
    p.add_argument("--save_png_preview", action="store_true", help="Also save a .png preview alongside each .pdf figure.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = AuditConfig(
        od_bin=float(args.od_bin),
        min_bucket_n=int(args.min_bucket_n),
        min_cluster_frac=float(args.min_cluster_frac),
        sep_thr=float(args.sep_thr),
        max_buckets=int(args.max_buckets),
        num_cases=int(args.num_cases),
        num_unimodal_cases=int(args.num_unimodal_cases),
        max_traj_per_case=int(args.max_traj_per_case),
        seed=int(args.seed),
        jacc_cell=float(args.jacc_cell),
        waypoint_mode=str(args.waypoint_mode),
        waypoint_turn_alpha=float(args.waypoint_turn_alpha),
        num_waypoints=int(args.num_waypoints),
        save_case_npz=bool(args.save_case_npz),
        save_png_preview=bool(args.save_png_preview),
    )
    report = run_audit(samples_npz=Path(args.samples_npz), out_dir=Path(args.out_dir), cfg=cfg)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
