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
class Config:
    prefix_len: int
    min_points: int
    chord_min: float
    detour_min: float
    seed: int
    hist_bins: int


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


def _segment_metrics(points: np.ndarray) -> Tuple[float, float, float, float]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points (T,2), got {points.shape}")
    T = int(points.shape[0])
    if T < 2:
        return 0.0, 0.0, 0.0, 0.0
    seg = points[1:] - points[:-1]
    step = np.linalg.norm(seg.astype(np.float64), axis=1)
    total = float(np.sum(step))
    chord = float(np.linalg.norm((points[-1] - points[0]).astype(np.float64)))
    detour = float(total / max(chord, 1e-6))
    step_mean = float(total / max(T - 1, 1))
    return chord, total, detour, step_mean


def _summ(a: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(a)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def _hist(a: np.ndarray, *, bins: int) -> Dict[str, object]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"bins": [], "counts": []}
    b = int(bins)
    b = max(10, min(200, b))
    counts, edges = np.histogram(a, bins=b)
    return {"bins": [float(x) for x in edges.tolist()], "counts": [int(x) for x in counts.tolist()]}


def run_stats(*, segments_parquet: Path, out_json: Path, cfg: Config) -> Dict[str, object]:
    y_list, x_list, t_list = _load_segments_columns(segments_parquet)
    n = int(len(y_list))
    n_points = np.asarray([len(v) for v in y_list], dtype=np.int64)

    # Full segment metrics.
    chord = np.zeros((n,), dtype=np.float32)
    total = np.zeros((n,), dtype=np.float32)
    detour = np.zeros((n,), dtype=np.float32)
    step_mean = np.zeros((n,), dtype=np.float32)

    # Prefix metrics (first 1+prefix_len points).
    pre_chord = np.full((n,), np.nan, dtype=np.float32)
    pre_total = np.full((n,), np.nan, dtype=np.float32)
    pre_detour = np.full((n,), np.nan, dtype=np.float32)
    pre_step_mean = np.full((n,), np.nan, dtype=np.float32)

    pre_len = int(cfg.prefix_len)
    win_len = int(pre_len) + 1 if pre_len > 0 else 0

    for i in range(n):
        yi = np.asarray(y_list[i], dtype=np.float32)
        xi = np.asarray(x_list[i], dtype=np.float32)
        pts = np.stack([yi, xi], axis=1)
        c, tot, d, sm = _segment_metrics(pts)
        chord[i] = c
        total[i] = tot
        detour[i] = d
        step_mean[i] = sm
        if win_len > 1 and int(pts.shape[0]) >= win_len:
            p = pts[:win_len]
            c, tot, d, sm = _segment_metrics(p)
            pre_chord[i] = c
            pre_total[i] = tot
            pre_detour[i] = d
            pre_step_mean[i] = sm

    keep = (n_points >= int(cfg.min_points)) & (chord >= float(cfg.chord_min)) & (detour >= float(cfg.detour_min))
    n_keep = int(np.sum(keep))

    report: Dict[str, object] = {
        "inputs": {"segments_parquet": str(segments_parquet)},
        "config": {
            "prefix_len": int(cfg.prefix_len),
            "min_points": int(cfg.min_points),
            "chord_min": float(cfg.chord_min),
            "detour_min": float(cfg.detour_min),
            "hist_bins": int(cfg.hist_bins),
            "seed": int(cfg.seed),
        },
        "stats": {
            "num_segments": int(n),
            "num_segments_prefix_valid": int(np.sum(np.isfinite(pre_chord))),
            "num_segments_keep": int(n_keep),
        },
        "full": {
            "n_points": _summ(n_points),
            "chord_len": _summ(chord),
            "total_len": _summ(total),
            "detour_ratio": _summ(detour),
            "step_len_mean": _summ(step_mean),
            "hist": {
                "chord_len": _hist(chord, bins=int(cfg.hist_bins)),
                "detour_ratio": _hist(detour, bins=int(cfg.hist_bins)),
            },
        },
        "prefix": {
            "window_len": int(win_len),
            "chord_len": _summ(pre_chord),
            "total_len": _summ(pre_total),
            "detour_ratio": _summ(pre_detour),
            "step_len_mean": _summ(pre_step_mean),
            "hist": {
                "chord_len": _hist(pre_chord, bins=int(cfg.hist_bins)),
                "detour_ratio": _hist(pre_detour, bins=int(cfg.hist_bins)),
            },
        },
        "meta": {"created_at": datetime.now(tz=TZ_SHANGHAI).isoformat()},
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P0: segment-level geometry stats from WorldTrace segments.parquet (JSON).")
    p.add_argument("--segments_parquet", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--prefix_len", type=int, default=256, help="Also report prefix stats for the first 1+prefix_len points (to diagnose window artifacts).")
    p.add_argument("--min_points", type=int, default=0, help="Filter threshold (report only): min points per segment.")
    p.add_argument("--chord_min", type=float, default=0.0, help="Filter threshold (report only): min chord length.")
    p.add_argument("--detour_min", type=float, default=0.0, help="Filter threshold (report only): min detour ratio.")
    p.add_argument("--hist_bins", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config(
        prefix_len=int(args.prefix_len),
        min_points=int(args.min_points),
        chord_min=float(args.chord_min),
        detour_min=float(args.detour_min),
        seed=int(args.seed),
        hist_bins=int(args.hist_bins),
    )
    report = run_stats(segments_parquet=Path(args.segments_parquet), out_json=Path(args.out_json), cfg=cfg)
    # Compact stdout: summary only.
    compact = {
        "ok": True,
        "segments_parquet": report["inputs"]["segments_parquet"],
        "num_segments": report["stats"]["num_segments"],
        "full": {
            "n_points_p50": report["full"]["n_points"]["p50"],
            "chord_p50": report["full"]["chord_len"]["p50"],
            "detour_p50": report["full"]["detour_ratio"]["p50"],
        },
        "prefix": {
            "prefix_len": report["prefix"]["window_len"] - 1,
            "chord_p50": report["prefix"]["chord_len"]["p50"],
            "detour_p50": report["prefix"]["detour_ratio"]["p50"],
        },
        "keep": report["stats"]["num_segments_keep"],
        "out_json": str(Path(args.out_json).resolve()),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

