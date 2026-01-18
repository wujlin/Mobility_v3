from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import pyarrow.parquet as pq  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pq = None

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class AuditCfg:
    seed: int
    n_routes: int
    prefer_matched: bool
    point_stride: int
    prob_thr: List[float]


def _quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, float(q)))


def _stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    return {
        "mean": float(np.mean(x)) if x.size else float("nan"),
        "p10": _quantile(x, 0.10),
        "p50": _quantile(x, 0.50),
        "p90": _quantile(x, 0.90),
        "p99": _quantile(x, 0.99),
        "min": float(np.min(x)) if x.size else float("nan"),
        "max": float(np.max(x)) if x.size else float("nan"),
    }


def _road_prob_global_stats(rp: np.ndarray, thr_list: Sequence[float]) -> Dict[str, object]:
    rp = np.asarray(rp, dtype=np.float32)
    out: Dict[str, object] = {
        "shape": [int(rp.shape[0]), int(rp.shape[1])],
        "value": _stats(rp.reshape(-1)),
        "coverage": {str(float(thr)): float(np.mean(rp >= float(thr))) for thr in thr_list},
    }
    return out


def _onroad_for_points(rp: np.ndarray, y: np.ndarray, x: np.ndarray, thr_list: Sequence[float]) -> Dict[str, float]:
    H, W = map(int, rp.shape)
    yy = np.clip(np.rint(y).astype(np.int64, copy=False), 0, H - 1)
    xx = np.clip(np.rint(x).astype(np.int64, copy=False), 0, W - 1)
    p = rp[yy, xx].astype(np.float32, copy=False)
    out: Dict[str, float] = {"prob_mean": float(np.mean(p)) if p.size else float("nan")}
    for thr in thr_list:
        out[f"onroad@{float(thr)}"] = float(np.mean(p >= float(thr))) if p.size else float("nan")
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit osm_road_prob.npy coverage and GT on-road rate from segments_with_wayid.parquet.")
    p.add_argument("--segments_parquet", type=Path, required=True)
    p.add_argument("--road_prob_npy", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_routes", type=int, default=5000, help="Randomly sample N routes from parquet (0=all).")
    p.add_argument("--prefer_matched", action="store_true", help="If is_matched exists, only use matched points.")
    p.add_argument("--point_stride", type=int, default=1, help="Downsample points along each route for speed (>=1).")
    p.add_argument("--prob_thr", type=float, nargs="+", default=[0.5], help="One or more thresholds for coverage/on-road.")
    return p


def main() -> None:
    if pq is None:
        raise SystemExit("pyarrow is required. Install: conda/pip install pyarrow")

    args = build_argparser().parse_args()
    cfg = AuditCfg(
        seed=int(args.seed),
        n_routes=int(args.n_routes),
        prefer_matched=bool(args.prefer_matched),
        point_stride=max(1, int(args.point_stride)),
        prob_thr=[float(x) for x in args.prob_thr],
    )

    rp = np.load(str(args.road_prob_npy))
    if rp.ndim != 2:
        raise SystemExit(f"road_prob_npy must be 2D, got shape={rp.shape}")
    rp = np.asarray(rp, dtype=np.float32)

    cols = ["y", "x"]
    if bool(cfg.prefer_matched):
        cols.append("is_matched")
    table = pq.read_table(str(args.segments_parquet), columns=cols)
    y_col = table.column("y").to_pylist()
    x_col = table.column("x").to_pylist()
    m_col = table.column("is_matched").to_pylist() if (bool(cfg.prefer_matched) and "is_matched" in table.column_names) else None

    N = len(y_col)
    ids = np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(int(cfg.seed))
    rng.shuffle(ids)
    if int(cfg.n_routes) > 0:
        ids = ids[: min(int(cfg.n_routes), int(ids.size))]

    per_route_prob_mean: List[float] = []
    per_route_onroad: Dict[str, List[float]] = {str(float(thr)): [] for thr in cfg.prob_thr}
    per_route_matched_ratio: List[float] = []
    total_points = 0

    for rid in ids.tolist():
        ys0 = y_col[int(rid)] or []
        xs0 = x_col[int(rid)] or []
        if not (ys0 and xs0):
            continue

        if m_col is not None:
            mm = m_col[int(rid)] or []
            if len(mm) == len(ys0):
                keep = np.asarray([int(v) != 0 for v in mm], dtype=bool)
                per_route_matched_ratio.append(float(np.mean(keep)))
                ys = np.asarray(ys0, dtype=np.float32)[keep]
                xs = np.asarray(xs0, dtype=np.float32)[keep]
            else:
                per_route_matched_ratio.append(float("nan"))
                ys = np.asarray(ys0, dtype=np.float32)
                xs = np.asarray(xs0, dtype=np.float32)
        else:
            per_route_matched_ratio.append(float("nan"))
            ys = np.asarray(ys0, dtype=np.float32)
            xs = np.asarray(xs0, dtype=np.float32)

        if ys.size <= 0 or xs.size <= 0:
            continue
        ys = ys[:: int(cfg.point_stride)]
        xs = xs[:: int(cfg.point_stride)]
        total_points += int(ys.size)

        stats = _onroad_for_points(rp, ys, xs, cfg.prob_thr)
        per_route_prob_mean.append(float(stats["prob_mean"]))
        for thr in cfg.prob_thr:
            per_route_onroad[str(float(thr))].append(float(stats[f"onroad@{float(thr)}"]))

    out = {
        "ok": True,
        "task": "audit_osm_road_prob",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {"segments_parquet": str(args.segments_parquet), "road_prob_npy": str(args.road_prob_npy)},
        "road_prob_global": _road_prob_global_stats(rp, cfg.prob_thr),
        "gt_sample": {
            "n_routes_sampled": int(ids.size),
            "n_routes_used": int(len(per_route_prob_mean)),
            "total_points_used": int(total_points),
            "matched_ratio": _stats(np.asarray(per_route_matched_ratio, dtype=np.float64)),
            "prob_mean": _stats(np.asarray(per_route_prob_mean, dtype=np.float64)),
            "onroad_rate": {k: _stats(np.asarray(v, dtype=np.float64)) for k, v in per_route_onroad.items()},
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

