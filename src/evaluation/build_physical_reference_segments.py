from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # optional
    pa = None
    pq = None

from src.utils.geo_grid import BBox, GridSpec


@dataclass(frozen=True)
class AStarCfg:
    lambda_offroad: float
    max_expansions: int
    min_margin: int
    max_margin: int


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_grid_from_osm_meta(osm_meta_json: Path) -> GridSpec:
    meta = json.loads(osm_meta_json.read_text(encoding="utf-8"))
    g = meta.get("grid", {})
    bbox = g.get("bbox", {})
    return GridSpec(
        H=int(g["H"]),
        W=int(g["W"]),
        bbox=BBox(min_lon=float(bbox["min_lon"]), min_lat=float(bbox["min_lat"]), max_lon=float(bbox["max_lon"]), max_lat=float(bbox["max_lat"])),
    )


def _dynamic_margin(start_y: int, start_x: int, end_y: int, end_x: int, *, cfg: AStarCfg) -> int:
    d = int(max(abs(int(end_y) - int(start_y)), abs(int(end_x) - int(start_x))))
    m = int(0.5 * float(d))
    return int(min(int(cfg.max_margin), max(int(cfg.min_margin), m)))


def _neighbors_8(res_y_m: float, res_x_m: float) -> List[Tuple[int, int, float]]:
    moves: List[Tuple[int, int, float]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            step_m = float(math.hypot(float(dy) * float(res_y_m), float(dx) * float(res_x_m)))
            moves.append((dy, dx, step_m))
    return moves


def _astar_soft(
    road_prob: np.ndarray,
    *,
    start_y: int,
    start_x: int,
    end_y: int,
    end_x: int,
    grid: GridSpec,
    cfg: AStarCfg,
) -> Tuple[List[int], List[int], Dict[str, object]]:
    H, W = int(grid.H), int(grid.W)
    if not (0 <= start_y < H and 0 <= start_x < W and 0 <= end_y < H and 0 <= end_x < W):
        return [], [], {"success": False, "reason": "oob_start_or_end"}

    res_y_m, res_x_m = grid.resolution_m()
    moves = _neighbors_8(res_y_m, res_x_m)

    margin = _dynamic_margin(start_y, start_x, end_y, end_x, cfg=cfg)
    y0 = max(0, min(start_y, end_y) - margin)
    y1 = min(H - 1, max(start_y, end_y) + margin)
    x0 = max(0, min(start_x, end_x) - margin)
    x1 = min(W - 1, max(start_x, end_x) + margin)

    sub = np.asarray(road_prob[y0 : y1 + 1, x0 : x1 + 1], dtype=np.float32)
    subH, subW = int(sub.shape[0]), int(sub.shape[1])

    sy, sx = int(start_y - y0), int(start_x - x0)
    gy, gx = int(end_y - y0), int(end_x - x0)
    start_i = sy * subW + sx
    goal_i = gy * subW + gx

    # Weight: 1 on-road (road_prob≈1), higher off-road; stays "soft" (no hard mask).
    w = 1.0 + float(cfg.lambda_offroad) * (1.0 - np.clip(sub.reshape(-1), 0.0, 1.0))
    w = w.astype(np.float32, copy=False)

    gscore = np.full((subH * subW,), np.inf, dtype=np.float64)
    came = np.full((subH * subW,), -1, dtype=np.int32)
    gscore[start_i] = 0.0

    import heapq

    def heuristic(i: int) -> float:
        yy = i // subW
        xx = i - yy * subW
        return float(math.hypot((yy - gy) * float(res_y_m), (xx - gx) * float(res_x_m)))

    heap: List[Tuple[float, int]] = [(heuristic(start_i), int(start_i))]
    expansions = 0

    while heap:
        _, i = heapq.heappop(heap)
        gi = float(gscore[i])
        if not np.isfinite(gi):
            continue
        if i == goal_i:
            break

        expansions += 1
        if cfg.max_expansions and expansions > int(cfg.max_expansions):
            return [], [], {
                "success": False,
                "reason": "max_expansions",
                "expansions": int(expansions),
                "margin": int(margin),
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
            }

        y = i // subW
        x = i - y * subW
        # stale queue entry
        if gi != float(gscore[i]):
            continue
        for dy, dx, step_m in moves:
            ny = y + int(dy)
            nx = x + int(dx)
            if ny < 0 or ny >= subH or nx < 0 or nx >= subW:
                continue
            ni = ny * subW + nx
            ng = gi + float(step_m) * float(w[ni])
            if ng < float(gscore[ni]):
                gscore[ni] = ng
                came[ni] = int(i)
                heapq.heappush(heap, (ng + heuristic(ni), int(ni)))

    if goal_i != start_i and came[goal_i] < 0:
        return [], [], {
            "success": False,
            "reason": "no_path",
            "expansions": int(expansions),
            "margin": int(margin),
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
        }

    # Reconstruct
    path_idx: List[int] = []
    cur = int(goal_i)
    path_idx.append(cur)
    while cur != start_i:
        cur = int(came[cur])
        if cur < 0:
            break
        path_idx.append(cur)
    path_idx.reverse()

    py: List[int] = []
    px: List[int] = []
    for i in path_idx:
        yy = i // subW
        xx = i - yy * subW
        py.append(int(yy + y0))
        px.append(int(xx + x0))

    return py, px, {
        "success": True,
        "expansions": int(expansions),
        "margin": int(margin),
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "cost": float(gscore[goal_i]) if np.isfinite(gscore[goal_i]) else float("nan"),
        "path_len": int(len(py)),
    }


def _iter_observed_endpoints(segments_parquet: Path, *, max_segments: int) -> Iterable[Tuple[str, int, int, int, int]]:
    if pq is None:
        raise SystemExit("pyarrow is required. Install: pip/conda install pyarrow")
    pf = pq.ParquetFile(str(segments_parquet))
    cols = ["traj_csv", "y", "x"]
    scanned = 0
    for batch in pf.iter_batches(batch_size=128, columns=cols):
        d = batch.to_pydict()
        n = len(d["traj_csv"])
        for i in range(n):
            scanned += 1
            if max_segments and scanned > int(max_segments):
                return
            k = str(d["traj_csv"][i])
            y = np.asarray(d["y"][i], dtype=np.int64).reshape(-1)
            x = np.asarray(d["x"][i], dtype=np.int64).reshape(-1)
            if y.size < 2 or x.size < 2:
                continue
            yield k, int(y[0]), int(x[0]), int(y[-1]), int(x[-1])


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a physical reference (soft OSM shortest path) as expected segments.parquet.")
    p.add_argument("--observed_segments_parquet", type=Path, required=True)
    p.add_argument("--osm_road_prob_npy", type=Path, required=True)
    p.add_argument("--osm_meta_json", type=Path, default=None, help="Optional osm_road_prob_meta.json (otherwise inferred next to npy).")
    p.add_argument("--out_parquet", type=Path, required=True)
    p.add_argument("--lambda_offroad", type=float, default=20.0, help="Penalty strength for off-road (higher => closer to roads).")
    p.add_argument("--max_expansions", type=int, default=350000, help="Fail a query if A* expands too many nodes (0=disabled).")
    p.add_argument("--min_margin", type=int, default=64)
    p.add_argument("--max_margin", type=int, default=256)
    p.add_argument("--max_segments", type=int, default=0, help="Optional cap for speed/debug (0=no cap).")
    p.add_argument("--quiet", action="store_true", help="Write outputs only; do not print JSON.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if pq is None or pa is None:
        raise SystemExit("pyarrow is required. Install: pip/conda install pyarrow")

    osm_meta = args.osm_meta_json
    if osm_meta is None:
        cand = Path(args.osm_road_prob_npy).parent / "osm_road_prob_meta.json"
        osm_meta = cand if cand.exists() else None
    if osm_meta is None or not Path(osm_meta).exists():
        raise SystemExit("Missing OSM meta: provide --osm_meta_json or place osm_road_prob_meta.json next to osm_road_prob.npy")

    grid = _load_grid_from_osm_meta(Path(osm_meta))
    road_prob = np.load(str(args.osm_road_prob_npy))
    if road_prob.shape != (int(grid.H), int(grid.W)):
        raise SystemExit(f"Shape mismatch: road_prob{tuple(road_prob.shape)} vs grid({grid.H},{grid.W})")

    cfg = AStarCfg(
        lambda_offroad=float(args.lambda_offroad),
        max_expansions=int(args.max_expansions),
        min_margin=int(args.min_margin),
        max_margin=int(args.max_margin),
    )

    out_parquet = Path(args.out_parquet)
    _ensure_dir(out_parquet.parent)
    report_path = out_parquet.with_suffix(".report.json")

    keys: List[str] = []
    ys: List[List[int]] = []
    xs: List[List[int]] = []
    start_y: List[int] = []
    start_x: List[int] = []
    end_y: List[int] = []
    end_x: List[int] = []
    success: List[bool] = []
    expansions: List[int] = []
    margin: List[int] = []
    cost: List[float] = []
    path_len: List[int] = []
    reasons: Dict[str, int] = {}

    scanned = 0
    for k, y0, x0, y1, x1 in _iter_observed_endpoints(Path(args.observed_segments_parquet), max_segments=int(args.max_segments)):
        scanned += 1
        py, px, meta = _astar_soft(
            road_prob,
            start_y=y0,
            start_x=x0,
            end_y=y1,
            end_x=x1,
            grid=grid,
            cfg=cfg,
        )
        keys.append(k)
        start_y.append(int(y0))
        start_x.append(int(x0))
        end_y.append(int(y1))
        end_x.append(int(x1))
        ok = bool(meta.get("success", False))
        success.append(bool(ok))
        expansions.append(int(meta.get("expansions", 0)))
        margin.append(int(meta.get("margin", -1)))
        cost.append(float(meta.get("cost", float("nan"))))
        path_len.append(int(meta.get("path_len", len(py))))
        ys.append([int(v) for v in py] if py else [int(y0), int(y1)])
        xs.append([int(v) for v in px] if px else [int(x0), int(x1)])
        if not ok:
            r = str(meta.get("reason", "unknown"))
            reasons[r] = reasons.get(r, 0) + 1

    table = pa.table(
        {
            "traj_csv": pa.array(keys, type=pa.string()),
            "start_y": pa.array(start_y, type=pa.int32()),
            "start_x": pa.array(start_x, type=pa.int32()),
            "end_y": pa.array(end_y, type=pa.int32()),
            "end_x": pa.array(end_x, type=pa.int32()),
            "y": pa.array(ys, type=pa.list_(pa.int32())),
            "x": pa.array(xs, type=pa.list_(pa.int32())),
            "success": pa.array(success, type=pa.bool_()),
            "expansions": pa.array(expansions, type=pa.int32()),
            "margin": pa.array(margin, type=pa.int32()),
            "cost": pa.array(cost, type=pa.float64()),
            "path_len": pa.array(path_len, type=pa.int32()),
        }
    )
    pq.write_table(table, str(out_parquet))

    out = {
        "observed_segments_parquet": str(Path(args.observed_segments_parquet)),
        "osm_road_prob_npy": str(Path(args.osm_road_prob_npy)),
        "osm_meta_json": str(Path(osm_meta)),
        "out_parquet": str(out_parquet),
        "grid": {"H": int(grid.H), "W": int(grid.W), "bbox": grid.bbox.__dict__},
        "cfg": {
            "lambda_offroad": cfg.lambda_offroad,
            "max_expansions": cfg.max_expansions,
            "min_margin": cfg.min_margin,
            "max_margin": cfg.max_margin,
            "max_segments": int(args.max_segments),
        },
        "stats": {
            "segments_scanned": int(scanned),
            "success_rate": float(np.mean(np.asarray(success, dtype=np.float64))) if success else float("nan"),
            "fail_reasons": reasons,
            "path_len_p50": float(np.percentile(np.asarray(path_len, dtype=np.float64), 50)) if path_len else float("nan"),
            "expansions_p50": float(np.percentile(np.asarray(expansions, dtype=np.float64), 50)) if expansions else float("nan"),
        },
    }
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if not bool(args.quiet):
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Keep BLAS threads quiet in multiprocessing-heavy environments.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()

