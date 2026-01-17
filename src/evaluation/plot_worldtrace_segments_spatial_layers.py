from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # optional
    pq = None

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    H: int
    W: int
    min_od_dist_km: float
    max_segments: int
    vmax_pct: float


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dlat = p2 - p1
    dlon = math.radians(float(lon2) - float(lon1))
    a = math.sin(dlat / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return float(r * c)


def _load_road_prob(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    a = np.load(str(p))
    if a.ndim != 2:
        return None
    return np.asarray(a, dtype=np.float32)


def _accum_points(heat: np.ndarray, y: np.ndarray, x: np.ndarray) -> None:
    H, W = map(int, heat.shape)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    x = np.asarray(x, dtype=np.int64).reshape(-1)
    if y.size == 0:
        return
    mask = (y >= 0) & (y < H) & (x >= 0) & (x < W)
    if not np.any(mask):
        return
    idx = y[mask] * W + x[mask]
    np.add.at(heat.reshape(-1), idx, 1)


def _endpoint_idx(osm_way_id: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Return (first_idx, last_idx) where osm_way_id is valid (>0).
    If none valid, return None.
    """
    w = np.asarray(osm_way_id, dtype=np.int64).reshape(-1)
    good = np.nonzero(w > 0)[0]
    if good.size == 0:
        return None
    return int(good[0]), int(good[-1])


def _plot_layers(
    *,
    out_dir: Path,
    heat: np.ndarray,
    od_o: np.ndarray,
    od_d: np.ndarray,
    road_prob: Optional[np.ndarray],
    cfg: Cfg,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("Missing dependency: matplotlib (needed for plotting).") from e

    out_dir.mkdir(parents=True, exist_ok=True)

    def _vmax(arr: np.ndarray) -> float:
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return 1.0
        return float(np.percentile(a, float(cfg.vmax_pct)))

    # Layer 1: trajectory heatmap
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=200)
    if road_prob is not None:
        rp = np.clip(np.asarray(road_prob, dtype=np.float32), 0.0, 1.0)
        ax.imshow(rp, cmap="Greys", origin="upper", alpha=0.35, vmin=0.0, vmax=1.0)
    h = np.log1p(heat.astype(np.float32))
    ax.imshow(h, cmap="magma", origin="upper", alpha=0.85, vmin=0.0, vmax=_vmax(h))
    ax.set_title("Layer1: Trajectory visit heatmap (log1p)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "detroit_trajectory_heatmap.png")
    plt.close(fig)

    # Layer 2: OD distributions
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0), dpi=200)
    for ax, arr, title in zip(axes, [od_o, od_d], ["Origins (log1p count)", "Destinations (log1p count)"]):
        if road_prob is not None:
            rp = np.clip(np.asarray(road_prob, dtype=np.float32), 0.0, 1.0)
            ax.imshow(rp, cmap="Greys", origin="upper", alpha=0.35, vmin=0.0, vmax=1.0)
        a = np.log1p(arr.astype(np.float32))
        ax.imshow(a, cmap="viridis", origin="upper", alpha=0.85, vmin=0.0, vmax=_vmax(a))
        ax.set_title(f"Layer2: {title}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "detroit_od_scatter.png")
    plt.close(fig)


def run(*, segments_parquet: Path, out_dir: Path, road_prob_npy: Optional[Path], cfg: Cfg) -> Dict[str, object]:
    if pq is None:
        raise SystemExit("pyarrow is required. Install: pip/conda install pyarrow")

    road_prob = _load_road_prob(road_prob_npy)
    if road_prob is not None:
        H, W = map(int, road_prob.shape)
    else:
        H, W = int(cfg.H), int(cfg.W)

    heat = np.zeros((H, W), dtype=np.int64)
    od_o = np.zeros((H, W), dtype=np.int64)
    od_d = np.zeros((H, W), dtype=np.int64)

    cols = ["traj_csv", "t", "lat", "lon", "y", "x", "osm_way_id"]
    pf = pq.ParquetFile(str(segments_parquet))
    scanned = 0
    kept = 0
    dropped_short = 0
    dropped_no_way = 0

    for batch in pf.iter_batches(batch_size=128, columns=cols):
        d = batch.to_pydict()
        n_rows = len(d["traj_csv"])
        for i in range(n_rows):
            scanned += 1
            if int(cfg.max_segments) > 0 and scanned > int(cfg.max_segments):
                break

            y = np.asarray(d["y"][i], dtype=np.int64)
            x = np.asarray(d["x"][i], dtype=np.int64)
            if y.size < 2 or x.size < 2:
                continue

            osm = np.asarray(d["osm_way_id"][i], dtype=np.int64)
            end_idx = _endpoint_idx(osm)
            if end_idx is None:
                dropped_no_way += 1
                continue
            i0, i1 = end_idx

            lat = np.asarray(d["lat"][i], dtype=np.float64)
            lon = np.asarray(d["lon"][i], dtype=np.float64)
            if lat.size <= max(i0, i1) or lon.size <= max(i0, i1):
                continue
            od_km = _haversine_km(float(lat[i0]), float(lon[i0]), float(lat[i1]), float(lon[i1]))
            if float(od_km) < float(cfg.min_od_dist_km):
                dropped_short += 1
                continue

            _accum_points(heat, y, x)

            oy, ox = int(y[i0]), int(x[i0])
            dy, dx = int(y[i1]), int(x[i1])
            if 0 <= oy < H and 0 <= ox < W:
                od_o[oy, ox] += 1
            if 0 <= dy < H and 0 <= dx < W:
                od_d[dy, dx] += 1
            kept += 1

        if int(cfg.max_segments) > 0 and scanned > int(cfg.max_segments):
            break

    _plot_layers(out_dir=out_dir, heat=heat, od_o=od_o, od_d=od_d, road_prob=road_prob, cfg=cfg)

    report = {
        "ok": True,
        "task": "plot_worldtrace_segments_spatial_layers",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {"segments_parquet": str(segments_parquet), "road_prob_npy": (str(road_prob_npy) if road_prob_npy else None)},
        "cfg": {"H": int(H), "W": int(W), "min_od_dist_km": float(cfg.min_od_dist_km), "max_segments": int(cfg.max_segments), "vmax_pct": float(cfg.vmax_pct)},
        "stats": {
            "scanned": int(scanned),
            "kept_after_filters": int(kept),
            "dropped_short_od": int(dropped_short),
            "dropped_no_way_id": int(dropped_no_way),
        },
        "artifacts": {
            "trajectory_heatmap_png": str(out_dir / "detroit_trajectory_heatmap.png"),
            "od_scatter_png": str(out_dir / "detroit_od_scatter.png"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WorldTrace spatial visualization (Layer1-2) from segments_with_wayid.parquet.")
    p.add_argument("--segments_parquet", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--road_prob_npy", type=Path, default=None, help="Optional: osm_road_prob.npy for grey road background.")

    p.add_argument("--H", type=int, default=1024)
    p.add_argument("--W", type=int, default=1024)
    p.add_argument("--min_od_dist_km", type=float, default=1.0)
    p.add_argument("--max_segments", type=int, default=0, help="Debug cap for speed (0=no cap).")
    p.add_argument("--vmax_pct", type=float, default=99.0, help="Color scale cap percentile for log heatmaps.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Cfg(
        H=int(args.H),
        W=int(args.W),
        min_od_dist_km=float(args.min_od_dist_km),
        max_segments=int(args.max_segments),
        vmax_pct=float(args.vmax_pct),
    )
    rep = run(
        segments_parquet=Path(args.segments_parquet),
        out_dir=Path(args.out_dir),
        road_prob_npy=(Path(args.road_prob_npy) if args.road_prob_npy is not None else None),
        cfg=cfg,
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

