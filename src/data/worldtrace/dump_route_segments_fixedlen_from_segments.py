from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover
    pq = None


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class DumpConfig:
    pred_len: int
    seed: int
    max_segments: Optional[int]
    min_points: int
    chord_min: float
    detour_min: float
    total_min: float


def _load_segments_columns(segments_parquet: Path) -> Tuple[list[list[int]], list[list[int]], list[list[int]]]:
    if pq is None:
        raise ModuleNotFoundError("pyarrow is required to read segments.parquet (please install pyarrow).")
    table = pq.read_table(str(segments_parquet), columns=["y", "x", "t"])
    y_col = table.column("y").to_pylist()
    x_col = table.column("x").to_pylist()
    t_col = table.column("t").to_pylist()
    if not (len(y_col) == len(x_col) == len(t_col)):
        raise RuntimeError("segments.parquet columns length mismatch")
    return y_col, x_col, t_col


def _segment_metrics(points: np.ndarray) -> Tuple[float, float, float]:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return 0.0, 0.0, 0.0
    seg = points[1:] - points[:-1]
    step = np.linalg.norm(seg.astype(np.float64), axis=1)
    total = float(np.sum(step))
    chord = float(np.linalg.norm((points[-1] - points[0]).astype(np.float64)))
    detour = float(total / max(chord, 1e-6))
    return chord, total, detour


def _resample_by_arclength(points: np.ndarray, *, num: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points (T,2), got {points.shape}")
    T = int(points.shape[0])
    if int(num) <= 1:
        return points[:1]
    if T < 2:
        return np.repeat(points[:1], repeats=int(num), axis=0).astype(np.float32)

    seg = points[1:] - points[:-1]
    seg_len = np.linalg.norm(seg.astype(np.float64), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)], axis=0).astype(np.float64)
    total = float(s[-1])
    if not np.isfinite(total) or total <= 1e-6:
        return np.repeat(points[:1], repeats=int(num), axis=0).astype(np.float32)
    q = np.linspace(0.0, total, num=int(num), dtype=np.float64)
    y = np.interp(q, s, points[:, 0].astype(np.float64)).astype(np.float32)
    x = np.interp(q, s, points[:, 1].astype(np.float64)).astype(np.float32)
    return np.stack([y, x], axis=1).astype(np.float32, copy=False)


def run_dump(*, segments_parquet: Path, out_npz: Path, cfg: DumpConfig) -> Dict[str, object]:
    if int(cfg.pred_len) <= 0:
        raise ValueError("--pred_len must be > 0")
    y_list, x_list, t_list = _load_segments_columns(segments_parquet)
    n_total = int(len(y_list))

    # Build candidate segment indices.
    cand = []
    for i in range(n_total):
        if len(y_list[i]) < int(cfg.min_points):
            continue
        pts = np.stack([np.asarray(y_list[i], dtype=np.float32), np.asarray(x_list[i], dtype=np.float32)], axis=1)
        chord, total, detour = _segment_metrics(pts)
        if chord < float(cfg.chord_min) or total < float(cfg.total_min) or detour < float(cfg.detour_min):
            continue
        cand.append(i)

    rng = np.random.default_rng(int(cfg.seed))
    cand = np.asarray(cand, dtype=np.int64)
    if cand.size == 0:
        raise RuntimeError("No segments pass filters (min_points/chord_min/total_min/detour_min).")
    if cfg.max_segments is not None:
        m = int(cfg.max_segments)
        m = max(1, min(m, int(cand.size)))
        pick = rng.choice(int(cand.size), size=m, replace=False)
        cand = cand[np.sort(pick)]

    window_len = int(cfg.pred_len) + 1
    n = int(cand.size)
    start_pos = np.empty((n, 2), dtype=np.float32)
    dest_pos = np.empty((n, 2), dtype=np.float32)
    targets = np.empty((n, int(cfg.pred_len), 2), dtype=np.float32)
    traj_idx = cand.astype(np.int64, copy=False)
    start_t = np.empty((n,), dtype=np.int64)

    chord_arr = np.zeros((n,), dtype=np.float32)
    total_arr = np.zeros((n,), dtype=np.float32)
    detour_arr = np.zeros((n,), dtype=np.float32)

    for j, i in enumerate(cand.tolist()):
        pts = np.stack([np.asarray(y_list[int(i)], dtype=np.float32), np.asarray(x_list[int(i)], dtype=np.float32)], axis=1)
        chord, total, detour = _segment_metrics(pts)
        chord_arr[j] = float(chord)
        total_arr[j] = float(total)
        detour_arr[j] = float(detour)

        res = _resample_by_arclength(pts, num=int(window_len))
        start_pos[j] = res[0]
        targets[j] = res[1:]
        dest_pos[j] = res[-1]
        ti = t_list[int(i)]
        start_t[j] = int(ti[0]) if (isinstance(ti, list) and ti) else 0

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "segments_parquet": str(segments_parquet),
        "config": {
            "pred_len": int(cfg.pred_len),
            "seed": int(cfg.seed),
            "max_segments": (int(cfg.max_segments) if cfg.max_segments is not None else None),
            "min_points": int(cfg.min_points),
            "chord_min": float(cfg.chord_min),
            "total_min": float(cfg.total_min),
            "detour_min": float(cfg.detour_min),
            "resample": "arclength_linear_interp",
        },
        "stats": {
            "n_seg_total": int(n_total),
            "n_seg_candidates": int(cand.size),
            "window_len": int(window_len),
            "chord_len": {
                "p50": float(np.percentile(chord_arr, 50)),
                "p90": float(np.percentile(chord_arr, 90)),
            },
            "detour_ratio": {
                "p50": float(np.percentile(detour_arr, 50)),
                "p90": float(np.percentile(detour_arr, 90)),
            },
            "total_len": {
                "p50": float(np.percentile(total_arr, 50)),
                "p90": float(np.percentile(total_arr, 90)),
            },
        },
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        start_pos=start_pos.astype(np.float32, copy=False),
        targets=targets.astype(np.float32, copy=False),
        dest_pos=dest_pos.astype(np.float32, copy=False),
        traj_idx=traj_idx.astype(np.int64, copy=False),
        start_t=start_t.astype(np.int64, copy=False),
        meta=meta,
    )
    return {"ok": True, "out_npz": str(out_npz), "N": int(n), "F": int(cfg.pred_len), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump segment-level route npz by resampling full segments to fixed length (F).")
    p.add_argument("--segments_parquet", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--pred_len", type=int, default=256, help="F (number of future positions); output length is F+1 including start.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_segments", type=int, default=None, help="Optional cap on number of segments (sample without replacement).")
    p.add_argument("--min_points", type=int, default=300, help="Minimum raw points per segment before resampling.")
    # NOTE: y/x in segments.parquet are grid cells (not meters). For Detroit 1024x1024, ~25m/cell.
    p.add_argument("--chord_min", type=float, default=40.0, help="Minimum chord length (grid cells). ~40 ~= 1km in Detroit.")
    p.add_argument("--total_min", type=float, default=60.0, help="Minimum total path length (grid cells). ~60 ~= 1.5km in Detroit.")
    p.add_argument("--detour_min", type=float, default=1.2, help="Minimum detour ratio (total/chord).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = DumpConfig(
        pred_len=int(args.pred_len),
        seed=int(args.seed),
        max_segments=(int(args.max_segments) if args.max_segments is not None else None),
        min_points=int(args.min_points),
        chord_min=float(args.chord_min),
        total_min=float(args.total_min),
        detour_min=float(args.detour_min),
    )
    report = run_dump(segments_parquet=Path(args.segments_parquet), out_npz=Path(args.out_npz), cfg=cfg)
    # Compact stdout.
    compact = {
        "ok": True,
        "out_npz": report["out_npz"],
        "N": report["N"],
        "F": report["F"],
        "n_seg_total": report["meta"]["stats"]["n_seg_total"],
        "n_seg_candidates": report["meta"]["stats"]["n_seg_candidates"],
        "chord_p50": report["meta"]["stats"]["chord_len"]["p50"],
        "detour_p50": report["meta"]["stats"]["detour_ratio"]["p50"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
