from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np


TZ_SHANGHAI = timezone(timedelta(hours=8))


def _stack_polyline(start_pos: np.ndarray, targets: np.ndarray) -> np.ndarray:
    start_pos = np.asarray(start_pos, dtype=np.float32).reshape(2)
    targets = np.asarray(targets, dtype=np.float32)
    return np.concatenate([start_pos[None, :], targets], axis=0)  # (F+1,2)


def _max_lateral_deviation_ratio(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=np.float32)
    a = poly[0]
    b = poly[-1]
    ab = b - a
    ab2 = float(np.sum(ab * ab))
    chord = float(np.sqrt(max(ab2, 0.0)))
    if not np.isfinite(chord) or chord < 1e-6:
        return 0.0
    ap = poly - a[None, :]
    t = np.sum(ap * ab[None, :], axis=-1) / max(ab2, 1e-6)
    t = np.clip(t, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    dev = np.linalg.norm(poly - proj, axis=-1)
    return float(np.max(dev) / chord)


def _path_length_ratio(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=np.float32)
    a = poly[0]
    b = poly[-1]
    chord = float(np.linalg.norm(b - a))
    if not np.isfinite(chord) or chord < 1e-6:
        return 1.0
    seg = poly[1:] - poly[:-1]
    length = float(np.sum(np.linalg.norm(seg, axis=-1)))
    return float(length / chord)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a detour-hard subset samples.npz (CPU-only).")
    p.add_argument("--in_npz", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--score", type=str, choices=["max_dev_ratio", "len_ratio"], default="max_dev_ratio")
    p.add_argument("--top_pct", type=float, default=10.0, help="Keep top pct windows by the chosen score.")
    p.add_argument("--max_n", type=int, default=None, help="Optional cap on output size (sample from the selected pool).")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    data = np.load(str(args.in_npz), allow_pickle=True)
    if "targets" not in data.files or "start_pos" not in data.files:
        raise ValueError(f"in_npz must contain targets/start_pos, got {data.files}")

    targets = np.asarray(data["targets"], dtype=np.float32)
    start_pos = np.asarray(data["start_pos"], dtype=np.float32)
    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError(f"Expected targets (N,F,2), got {targets.shape}")
    if start_pos.ndim != 2 or start_pos.shape[-1] != 2:
        raise ValueError(f"Expected start_pos (N,2), got {start_pos.shape}")
    if int(targets.shape[0]) != int(start_pos.shape[0]):
        raise ValueError("N mismatch between targets and start_pos")

    N = int(targets.shape[0])
    score = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        poly = _stack_polyline(start_pos[i], targets[i])
        if str(args.score) == "len_ratio":
            score[i] = float(_path_length_ratio(poly))
        else:
            score[i] = float(_max_lateral_deviation_ratio(poly))

    top_pct = float(args.top_pct)
    if not (0.0 < top_pct <= 100.0):
        raise ValueError("--top_pct must be in (0,100]")
    thr = float(np.percentile(score, 100.0 - top_pct))
    idx = np.nonzero(score >= thr)[0]

    rng = np.random.default_rng(int(args.seed))
    if args.max_n is not None and int(args.max_n) > 0 and idx.size > int(args.max_n):
        idx = rng.choice(idx, size=int(args.max_n), replace=False)
        idx = np.sort(idx)

    out: Dict[str, object] = {}
    for k in data.files:
        v = data[k]
        if k == "meta":
            continue
        if hasattr(v, "shape") and len(v.shape) >= 1 and int(v.shape[0]) == N:
            out[k] = np.asarray(v)[idx]
        else:
            out[k] = v

    meta_in = None
    if "meta" in data.files:
        try:
            meta_in = data["meta"].item() if hasattr(data["meta"], "item") else data["meta"]
        except Exception:
            meta_in = None

    meta_out = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "in_npz": str(args.in_npz),
        "score": str(args.score),
        "top_pct": float(args.top_pct),
        "thr": float(thr),
        "seed": int(args.seed),
        "max_n": (int(args.max_n) if args.max_n is not None else None),
        "N_in": int(N),
        "N_out": int(idx.size),
        "meta_in": meta_in,
        "score_stats": {
            "min": float(np.min(score)),
            "p50": float(np.percentile(score, 50)),
            "p90": float(np.percentile(score, 90)),
            "max": float(np.max(score)),
        },
    }
    out["meta"] = meta_out

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print("[OK] detour_hard subset")
    print(json.dumps(meta_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

