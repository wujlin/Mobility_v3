from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GateConfig:
    od_bin: float
    min_bucket_n: int
    min_cluster_frac: float
    sep_thr: float
    max_buckets: int
    max_n: Optional[int]
    seed: int


def _load_gt(path: Path, *, max_n: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(str(path))
    if "targets" not in data.files or "start_pos" not in data.files:
        raise ValueError(f"Bad samples.npz: require keys ['targets','start_pos'], got {data.files}")
    targets = np.asarray(data["targets"], dtype=np.float32)
    start_pos = np.asarray(data["start_pos"], dtype=np.float32)
    dest_pos = np.asarray(data["dest_pos"], dtype=np.float32) if "dest_pos" in data.files else None
    if max_n is not None:
        targets = targets[: int(max_n)]
        start_pos = start_pos[: int(max_n)]
        if dest_pos is not None:
            dest_pos = dest_pos[: int(max_n)]
    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError(f"Expected targets (N,F,2), got {targets.shape}")
    if start_pos.ndim != 2 or start_pos.shape[-1] != 2:
        raise ValueError(f"Expected start_pos (N,2), got {start_pos.shape}")
    if targets.shape[0] != start_pos.shape[0]:
        raise ValueError("N mismatch between targets and start_pos")
    if dest_pos is not None:
        if dest_pos.ndim != 2 or dest_pos.shape[-1] != 2:
            raise ValueError(f"Expected dest_pos (N,2), got {dest_pos.shape}")
        if dest_pos.shape[0] != targets.shape[0]:
            raise ValueError("N mismatch between dest_pos and targets")
    return targets, start_pos, dest_pos


def _od_key(start_pos: np.ndarray, end_pos: np.ndarray, *, od_bin: float) -> np.ndarray:
    """
    Quantize OD to coarse bins to get enough repeats.
    Returns keys as (N,4) int: [sy,sx,ey,ex]
    """
    b = max(float(od_bin), 1e-6)
    s = np.rint(start_pos / b).astype(np.int64)
    e = np.rint(end_pos / b).astype(np.int64)
    return np.concatenate([s, e], axis=1)


def _polyline_features(start_pos: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract coarse geometry features for clustering:
      - signed_dev_ratio: signed max lateral deviation / chord_len
      - s_frac: arc-length fraction where max deviation occurs
      - len_ratio: path_len / chord_len
    """
    start_pos = np.asarray(start_pos, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    N, F, _ = targets.shape
    poly = np.concatenate([start_pos[:, None, :], targets], axis=1)  # (N,F+1,2)
    a = poly[:, 0, :].astype(np.float64)
    b = poly[:, -1, :].astype(np.float64)
    ab = b - a
    chord = np.linalg.norm(ab, axis=1) + 1e-12

    # Signed perpendicular distance to the chord line (infinite line).
    ap = poly.astype(np.float64) - a[:, None, :]
    cross = ab[:, None, 0] * ap[:, :, 1] - ab[:, None, 1] * ap[:, :, 0]
    dist_signed = cross / chord[:, None]  # signed distance in grid units
    dist_signed[:, 0] = 0.0
    dist_signed[:, -1] = 0.0
    idx = np.argmax(np.abs(dist_signed), axis=1)
    dev_signed = dist_signed[np.arange(N), idx]  # (N,)
    signed_dev_ratio = dev_signed / chord  # (N,)

    # Arc-length fraction for idx.
    seg = poly[:, 1:, :] - poly[:, :-1, :]
    seg_len = np.linalg.norm(seg, axis=2).astype(np.float64)  # (N,F)
    s = np.concatenate([np.zeros((N, 1), dtype=np.float64), np.cumsum(seg_len, axis=1)], axis=1)  # (N,F+1)
    total = s[:, -1] + 1e-12
    s_frac = s[np.arange(N), idx] / total

    # Length ratio.
    path_len = np.sum(seg_len, axis=1)
    len_ratio = path_len / chord

    return signed_dev_ratio.astype(np.float32), s_frac.astype(np.float32), len_ratio.astype(np.float32)


def _polyline_features_to_dest(
    start_pos: np.ndarray,
    targets: np.ndarray,
    dest_pos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract coarse geometry features for clustering with *global destination* as the OD endpoint.

    This is closer to the real task setting where condition includes trip-level destination.
    Features:
      - signed_dev_ratio: signed max lateral deviation to chord (start->dest) / chord_len
      - progress_ratio: signed progress along chord, normalized by chord length
      - len_ratio: segment path_len / chord_len
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
    signed_dev_ratio = dev_signed / chord

    # Progress along chord direction (negative => moving away from destination).
    end_seg = poly[:, -1, :].astype(np.float64)
    proj = np.sum((end_seg - a) * ab, axis=1) / (chord * chord)
    progress_ratio = proj.astype(np.float32)

    # Segment path length (not full trip).
    seg = poly[:, 1:, :] - poly[:, :-1, :]
    seg_len = np.linalg.norm(seg, axis=2).astype(np.float64)
    path_len = np.sum(seg_len, axis=1)
    len_ratio = (path_len / chord).astype(np.float32)

    return signed_dev_ratio.astype(np.float32), progress_ratio, len_ratio


def _kmeans2(x: np.ndarray, *, seed: int, iters: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Very small 2-means for gating (no sklearn).
    Returns:
      labels: (n,)
      centers: (2,d)
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n < 2:
        return np.zeros((n,), dtype=np.int64), np.zeros((2, d), dtype=np.float64)

    # init: extremes along first dimension (signed_dev_ratio by default)
    i0 = int(np.argmin(x[:, 0]))
    i1 = int(np.argmax(x[:, 0]))
    if i0 == i1:
        rng = np.random.default_rng(int(seed))
        i1 = int(rng.integers(0, n))
    c = np.stack([x[i0], x[i1]], axis=0)

    labels = np.zeros((n,), dtype=np.int64)
    for _ in range(int(iters)):
        # assign
        d0 = np.sum((x - c[0]) ** 2, axis=1)
        d1 = np.sum((x - c[1]) ** 2, axis=1)
        new_labels = (d1 < d0).astype(np.int64)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # update
        for k in (0, 1):
            mask = labels == k
            if not np.any(mask):
                continue
            c[k] = np.mean(x[mask], axis=0)
    return labels, c


def _cluster_gate(
    feats: np.ndarray,
    *,
    min_cluster_frac: float,
    sep_thr: float,
    seed: int,
) -> Dict[str, object]:
    """
    Decide whether a bucket is meaningfully multi-modal.
    """
    feats = np.asarray(feats, dtype=np.float64)
    n = int(feats.shape[0])
    if n < 2:
        return {"multimodal": False}

    # normalize per bucket
    mu = np.mean(feats, axis=0)
    sig = np.std(feats, axis=0) + 1e-6
    x = (feats - mu) / sig

    labels, centers = _kmeans2(x, seed=int(seed))
    n0 = int(np.sum(labels == 0))
    n1 = int(n - n0)
    frac0 = float(n0) / float(n)
    frac1 = float(n1) / float(n)
    if frac0 < float(min_cluster_frac) or frac1 < float(min_cluster_frac):
        return {"multimodal": False, "reason": "cluster_too_small", "n0": n0, "n1": n1}

    # separation vs within scatter
    c0, c1 = centers[0], centers[1]
    sep = float(np.linalg.norm(c0 - c1))
    w0 = x[labels == 0] - c0[None, :]
    w1 = x[labels == 1] - c1[None, :]
    rms0 = float(np.sqrt(np.mean(np.sum(w0 * w0, axis=1)))) if w0.size else 0.0
    rms1 = float(np.sqrt(np.mean(np.sum(w1 * w1, axis=1)))) if w1.size else 0.0
    scatter = float(max(rms0, rms1, 1e-6))
    score = sep / scatter
    multimodal = bool(score >= float(sep_thr))
    return {
        "multimodal": multimodal,
        "score": float(score),
        "sep": float(sep),
        "scatter": float(scatter),
        "frac0": float(frac0),
        "frac1": float(frac1),
    }


def run_gate(*, samples_npz: Path, cfg: GateConfig) -> Dict[str, object]:
    targets, start_pos, dest_pos = _load_gt(samples_npz, max_n=cfg.max_n)
    use_dest = dest_pos is not None
    end_pos = dest_pos if dest_pos is not None else targets[:, -1, :]
    keys = _od_key(start_pos, end_pos, od_bin=float(cfg.od_bin))

    if use_dest:
        sd, pr, lr = _polyline_features_to_dest(start_pos, targets, dest_pos=dest_pos)  # type: ignore[arg-type]
        feats = np.stack([sd, pr, lr], axis=1)  # (N,3)
    else:
        sd, sf, lr = _polyline_features(start_pos, targets)
        feats = np.stack([sd, sf, lr], axis=1)  # (N,3)

    # group by key
    buckets: Dict[Tuple[int, int, int, int], List[int]] = {}
    for i in range(int(keys.shape[0])):
        k = tuple(int(x) for x in keys[i].tolist())
        buckets.setdefault(k, []).append(i)

    # sort buckets by size
    items = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    if int(cfg.max_buckets) > 0:
        items = items[: int(cfg.max_buckets)]

    considered = 0
    multimodal = 0
    weighted_considered = 0
    weighted_multimodal = 0
    per_bucket: List[Dict[str, object]] = []

    rng = np.random.default_rng(int(cfg.seed))
    for k, idxs in items:
        n = len(idxs)
        if n < int(cfg.min_bucket_n):
            continue
        considered += 1
        weighted_considered += n
        f = feats[np.asarray(idxs, dtype=np.int64)]
        rep = _cluster_gate(f, min_cluster_frac=float(cfg.min_cluster_frac), sep_thr=float(cfg.sep_thr), seed=int(rng.integers(0, 1_000_000)))
        rep.update({"key": list(k), "n": int(n)})
        per_bucket.append(rep)
        if bool(rep.get("multimodal")):
            multimodal += 1
            weighted_multimodal += n

    return {
        "inputs": {"samples_npz": str(samples_npz)},
        "config": {
            "od_bin": float(cfg.od_bin),
            "od_end": ("dest_pos" if use_dest else "segment_end"),
            "min_bucket_n": int(cfg.min_bucket_n),
            "min_cluster_frac": float(cfg.min_cluster_frac),
            "sep_thr": float(cfg.sep_thr),
            "max_buckets": int(cfg.max_buckets),
            "max_n": (int(cfg.max_n) if cfg.max_n is not None else None),
            "seed": int(cfg.seed),
        },
        "stats": {
            "N": int(targets.shape[0]),
            "F": int(targets.shape[1]),
            "num_od_buckets_total": int(len(buckets)),
            "num_buckets_considered": int(considered),
            "num_buckets_multimodal": int(multimodal),
            "bucket_multimodal_ratio": float(multimodal / max(considered, 1)),
            "weighted_considered": int(weighted_considered),
            "weighted_multimodal": int(weighted_multimodal),
            "weighted_multimodal_ratio": float(weighted_multimodal / max(weighted_considered, 1)),
        },
        # Keep only a small number of bucket records to avoid massive JSON.
        "top_buckets": per_bucket[: min(len(per_bucket), 50)],
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Go/No-Go #1: OD multimodality gate (GT-only, CPU-only).")
    p.add_argument("--samples_npz", type=str, required=True, help="samples.npz with GT 'targets' and 'start_pos'.")
    p.add_argument("--od_bin", type=float, default=8.0, help="OD quantization bin in grid units (bigger => more repeats).")
    p.add_argument("--min_bucket_n", type=int, default=30, help="Min samples per OD bucket to evaluate.")
    p.add_argument("--min_cluster_frac", type=float, default=0.2, help="Min fraction per cluster for 2-means split.")
    p.add_argument("--sep_thr", type=float, default=2.5, help="Separation score threshold (sep / within-scatter).")
    p.add_argument("--max_buckets", type=int, default=500, help="Evaluate only top-K biggest buckets (<=0 means all).")
    p.add_argument("--max_n", type=int, default=None, help="Optional cap on number of samples loaded from npz.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument("--out_json", type=str, default=None, help="Optional output JSON path.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = GateConfig(
        od_bin=float(args.od_bin),
        min_bucket_n=int(args.min_bucket_n),
        min_cluster_frac=float(args.min_cluster_frac),
        sep_thr=float(args.sep_thr),
        max_buckets=int(args.max_buckets),
        max_n=int(args.max_n) if args.max_n is not None else None,
        seed=int(args.seed),
    )
    report = run_gate(samples_npz=Path(args.samples_npz), cfg=cfg)
    print("[OK] OD multimodality gate")
    print(json.dumps(report["stats"], indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
