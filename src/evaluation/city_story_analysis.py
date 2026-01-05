from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # optional
    pq = None

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plot_style import FIGSIZE_HALF, OKABE_ITO, paper_style, save_figure
from src.utils.geo_grid import BBox, GridSpec


@dataclass(frozen=True)
class DefaultDetroitCore:
    H: int = 1024
    W: int = 1024
    min_lon: float = -83.25
    max_lon: float = -82.95
    min_lat: float = 42.25
    max_lat: float = 42.50

    def grid(self) -> GridSpec:
        return GridSpec(
            H=int(self.H),
            W=int(self.W),
            bbox=BBox(min_lon=float(self.min_lon), max_lon=float(self.max_lon), min_lat=float(self.min_lat), max_lat=float(self.max_lat)),
        )


def _quantiles(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "mean": float("nan")}
    p10, p50, p90 = np.percentile(x, [10, 50, 90]).tolist()
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90), "mean": float(np.mean(x))}


def _path_len_and_chord_m(x: np.ndarray, y: np.ndarray, *, res_x_m: float, res_y_m: float) -> Tuple[float, float]:
    if x.size < 2:
        return 0.0, 0.0
    dx = (x[1:] - x[:-1]) * float(res_x_m)
    dy = (y[1:] - y[:-1]) * float(res_y_m)
    path = float(np.sum(np.hypot(dx, dy)))
    chord = float(np.hypot((x[-1] - x[0]) * float(res_x_m), (y[-1] - y[0]) * float(res_y_m)))
    return path, chord


def _max_dev_ratio(x: np.ndarray, y: np.ndarray, *, res_x_m: float, res_y_m: float) -> float:
    if x.size < 2:
        return 0.0
    px = x.astype(np.float64) * float(res_x_m)
    py = y.astype(np.float64) * float(res_y_m)
    a = np.array([px[0], py[0]], dtype=np.float64)
    b = np.array([px[-1], py[-1]], dtype=np.float64)
    ab = b - a
    chord = float(np.linalg.norm(ab))
    if chord <= 1e-6:
        return 0.0
    apx = px - a[0]
    apy = py - a[1]
    cross = np.abs(ab[0] * apy - ab[1] * apx)
    d = cross / (chord + 1e-12)
    d[0] = 0.0
    d[-1] = 0.0
    return float(np.max(d) / chord)


def _hour_of_day_utc_seconds(t0: int, *, tz_name: str) -> Optional[int]:
    if ZoneInfo is None:
        return None
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return None
    import datetime as dt

    try:
        dt0 = dt.datetime.fromtimestamp(int(t0), tz=dt.timezone.utc).astimezone(tz)
        return int(dt0.hour)
    except Exception:
        return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _plot_detour_hist(len_ratio: np.ndarray, out_png: Path, *, city_name: str) -> None:
    x = np.asarray(len_ratio, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return
    # Keep the bulk readable; tail is not the story here.
    hi = float(np.percentile(x, 99.5))
    hi = max(1.5, min(hi, 4.0))
    x = np.clip(x, 1.0, hi)

    with paper_style():
        fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
        ax.hist(x, bins=50, color=OKABE_ITO["blue"], alpha=0.85, edgecolor="none")
        ax.set_xlim(1.0, hi)
        ax.set_xlabel("Detour ratio (path length / straight distance)")
        ax.set_ylabel("Count")
        ax.set_title(f"{city_name} (WorldTrace) detour ratio distribution")
        fig.tight_layout()
        save_figure(fig, out_png)
        save_figure(fig, out_png.with_suffix(".pdf"))
        plt.close(fig)


def _plot_detour_by_hour(len_ratio: np.ndarray, hour: np.ndarray, out_png: Path, *, tz_name: str) -> None:
    mask = np.isfinite(len_ratio) & (hour >= 0) & (hour <= 23)
    if int(np.sum(mask)) == 0:
        return
    x = len_ratio[mask].astype(np.float64)
    h = hour[mask].astype(np.int64)
    med = np.full((24,), np.nan, dtype=np.float64)
    for hh in range(24):
        v = x[h == hh]
        if v.size >= 10:
            med[hh] = float(np.median(v))

    with paper_style():
        fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
        ax.plot(np.arange(24), med, marker="o", color=OKABE_ITO["vermillion"])
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xlabel(f"Local hour ({tz_name})")
        ax.set_ylabel("Median detour ratio")
        ax.set_title("Detour ratio by hour (median, n>=10)")
        fig.tight_layout()
        save_figure(fig, out_png)
        save_figure(fig, out_png.with_suffix(".pdf"))
        plt.close(fig)


def _od_bins_xy(x0: int, y0: int, x1: int, y1: int, *, H: int, W: int, od_bins: int) -> Tuple[int, int, int, int, int, int]:
    b = int(od_bins)
    cell_h = int(math.ceil(H / b))
    cell_w = int(math.ceil(W / b))
    oy = min(b - 1, max(0, int(y0) // cell_h))
    ox = min(b - 1, max(0, int(x0) // cell_w))
    dy = min(b - 1, max(0, int(y1) // cell_h))
    dx = min(b - 1, max(0, int(x1) // cell_w))
    o = oy * b + ox
    d = dy * b + dx
    od = o * (b * b) + d
    return ox, oy, dx, dy, o, od


def _accum_density(segments_xy: Iterable[Tuple[np.ndarray, np.ndarray]], *, H: int, W: int) -> np.ndarray:
    heat = np.zeros((H * W,), dtype=np.int64)
    for y, x in segments_xy:
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        x = np.asarray(x, dtype=np.int64).reshape(-1)
        if y.size == 0:
            continue
        idx = y * W + x
        idx = idx[(idx >= 0) & (idx < H * W)]
        np.add.at(heat, idx, 1)
    return heat.reshape(H, W)


def _plot_route_choice_heatmaps(
    pairs: List[Dict[str, object]],
    by_pair_segments: Dict[int, List[int]],
    xs: List[List[int]],
    ys: List[List[int]],
    *,
    H: int,
    W: int,
    out_png: Path,
    city_name: str,
) -> None:
    if not pairs:
        return
    n = len(pairs)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig_h = 2.4 * float(nrows)
    with paper_style():
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5, fig_h))
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])
        axes = axes.reshape(-1)

        vmax = 0.0
        heatmaps: List[np.ndarray] = []
        for p in pairs:
            pid = int(p["pair_id"])  # type: ignore[arg-type]
            seg_idx = by_pair_segments[pid]
            heat = _accum_density(((np.asarray(ys[i]), np.asarray(xs[i])) for i in seg_idx), H=H, W=W)
            heat = np.log1p(heat.astype(np.float32))
            heatmaps.append(heat)
            vmax = max(vmax, float(np.max(heat)))

        for ax, p, heat in zip(axes, pairs, heatmaps):
            im = ax.imshow(heat, cmap="viridis", origin="upper", vmin=0.0, vmax=vmax if vmax > 0 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(p.get("title", ""))
            sx = float(p.get("start_x_mean", float("nan")))
            sy = float(p.get("start_y_mean", float("nan")))
            ex = float(p.get("end_x_mean", float("nan")))
            ey = float(p.get("end_y_mean", float("nan")))
            if np.isfinite(sx) and np.isfinite(sy):
                ax.scatter([sx], [sy], s=28, c=OKABE_ITO["bluish_green"], marker="o", edgecolors="white", linewidths=0.6)
            if np.isfinite(ex) and np.isfinite(ey):
                ax.scatter([ex], [ey], s=34, c=OKABE_ITO["vermillion"], marker="*", edgecolors="white", linewidths=0.6)

        for ax in axes[len(pairs) :]:
            ax.axis("off")

        cbar = fig.colorbar(im, ax=axes[: len(pairs)].tolist(), fraction=0.03, pad=0.02)
        cbar.set_label("log(1 + visit count)", rotation=90)
        fig.suptitle(f"{city_name}: top OD route-choice patterns")
        fig.tight_layout()
        save_figure(fig, out_png)
        save_figure(fig, out_png.with_suffix(".pdf"))
        plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="City story analysis: detour distribution + OD route-choice patterns (WorldTrace segments.parquet).")
    p.add_argument("--segments_parquet", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--city_name", type=str, default="Detroit")
    p.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    p.add_argument("--grid_h", type=int, default=1024)
    p.add_argument("--grid_w", type=int, default=1024)
    p.add_argument("--timezone", type=str, default="America/Detroit")
    p.add_argument("--od_bins", type=int, default=8)
    p.add_argument("--top_od", type=int, default=6)
    p.add_argument("--min_od_n", type=int, default=30)
    p.add_argument("--max_segments", type=int, default=0, help="Optional cap for speed/debug (0=no cap).")
    p.add_argument("--quiet", action="store_true", help="Do not print anything (write JSON only).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if pq is None:
        raise SystemExit("pyarrow is required. Install: pip/conda install pyarrow")

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    if args.bbox is None:
        grid = DefaultDetroitCore().grid()
    else:
        min_lon, min_lat, max_lon, max_lat = map(float, args.bbox)
        grid = GridSpec(H=int(args.grid_h), W=int(args.grid_w), bbox=BBox(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat))
    res_y_m, res_x_m = grid.resolution_m()

    cols = ["traj_csv", "n_points", "t", "y", "x"]
    pf = pq.ParquetFile(str(args.segments_parquet))

    len_ratio: List[float] = []
    max_dev_ratio: List[float] = []
    chord_m: List[float] = []
    hour: List[int] = []

    start_xs: List[int] = []
    start_ys: List[int] = []
    end_xs: List[int] = []
    end_ys: List[int] = []
    xs: List[List[int]] = []
    ys: List[List[int]] = []

    scanned = 0
    for batch in pf.iter_batches(batch_size=128, columns=cols):
        d = batch.to_pydict()
        n_rows = len(d["traj_csv"])
        for i in range(n_rows):
            scanned += 1
            if args.max_segments and scanned > int(args.max_segments):
                break

            x = np.asarray(d["x"][i], dtype=np.int64)
            y = np.asarray(d["y"][i], dtype=np.int64)
            t = np.asarray(d["t"][i], dtype=np.int64)
            if x.size < 2 or y.size < 2:
                continue

            path_m, chord0 = _path_len_and_chord_m(x, y, res_x_m=res_x_m, res_y_m=res_y_m)
            if chord0 <= 1e-6:
                continue

            lr = float(path_m / chord0)
            len_ratio.append(lr)
            chord_m.append(float(chord0))
            max_dev_ratio.append(_max_dev_ratio(x, y, res_x_m=res_x_m, res_y_m=res_y_m))

            h = _hour_of_day_utc_seconds(int(t[0]), tz_name=str(args.timezone))
            hour.append(int(h) if h is not None else -1)

            start_xs.append(int(x[0]))
            start_ys.append(int(y[0]))
            end_xs.append(int(x[-1]))
            end_ys.append(int(y[-1]))
            xs.append(x.astype(int).tolist())
            ys.append(y.astype(int).tolist())

        if args.max_segments and scanned > int(args.max_segments):
            break

    len_ratio_np = np.asarray(len_ratio, dtype=np.float64)
    max_dev_ratio_np = np.asarray(max_dev_ratio, dtype=np.float64)
    chord_np = np.asarray(chord_m, dtype=np.float64)
    hour_np = np.asarray(hour, dtype=np.int64)

    detour_hist_png = out_dir / "detour_ratio_hist.png"
    detour_by_hour_png = out_dir / "detour_ratio_by_hour.png"
    _plot_detour_hist(len_ratio_np, detour_hist_png, city_name=str(args.city_name))
    _plot_detour_by_hour(len_ratio_np, hour_np, detour_by_hour_png, tz_name=str(args.timezone))

    H, W = int(grid.H), int(grid.W)
    od_bins = int(args.od_bins)
    top_od = int(args.top_od)
    min_od_n = int(args.min_od_n)

    pair_id: List[int] = []
    pair_key: List[Tuple[int, int, int, int]] = []
    for sx, sy, ex, ey in zip(start_xs, start_ys, end_xs, end_ys):
        ox, oy, dx, dy, _, od = _od_bins_xy(sx, sy, ex, ey, H=H, W=W, od_bins=od_bins)
        pair_id.append(int(od))
        pair_key.append((ox, oy, dx, dy))

    pair_id_np = np.asarray(pair_id, dtype=np.int64)
    uniq, cnt = np.unique(pair_id_np, return_counts=True)
    order = np.argsort(-cnt)

    picked: List[Dict[str, object]] = []
    by_pair_segments: Dict[int, List[int]] = {}
    for j in order:
        if len(picked) >= top_od:
            break
        pid = int(uniq[j])
        c = int(cnt[j])
        if c < min_od_n:
            continue
        idx = np.where(pair_id_np == pid)[0].astype(int).tolist()
        if not idx:
            continue
        ox, oy, dx, dy = pair_key[idx[0]]
        sx_mean = float(np.mean(np.asarray([start_xs[k] for k in idx], dtype=np.float64)))
        sy_mean = float(np.mean(np.asarray([start_ys[k] for k in idx], dtype=np.float64)))
        ex_mean = float(np.mean(np.asarray([end_xs[k] for k in idx], dtype=np.float64)))
        ey_mean = float(np.mean(np.asarray([end_ys[k] for k in idx], dtype=np.float64)))
        title = f"OD({ox},{oy})->({dx},{dy}) n={c}"
        picked.append(
            {
                "pair_id": pid,
                "od_bins": od_bins,
                "origin_bin": {"x": ox, "y": oy},
                "dest_bin": {"x": dx, "y": dy},
                "n_segments": c,
                "start_x_mean": sx_mean,
                "start_y_mean": sy_mean,
                "end_x_mean": ex_mean,
                "end_y_mean": ey_mean,
                "title": title,
            }
        )
        by_pair_segments[pid] = idx

    route_png = out_dir / "route_choice_top_od.png"
    _plot_route_choice_heatmaps(
        picked,
        by_pair_segments,
        xs,
        ys,
        H=H,
        W=W,
        out_png=route_png,
        city_name=str(args.city_name),
    )
    route_json = out_dir / "route_choice_top_od.json"

    out = {
        "segments_parquet": str(args.segments_parquet),
        "city_name": str(args.city_name),
        "grid": {"H": H, "W": W, "bbox": grid.bbox.__dict__, "res_m": {"x": float(res_x_m), "y": float(res_y_m)}},
        "n_segments_used": int(len(len_ratio)),
        "detour_ratio": {
            "len_ratio": _quantiles(len_ratio_np),
            "max_dev_ratio": _quantiles(max_dev_ratio_np),
            "straight_dist_m": _quantiles(chord_np),
        },
        "by_hour": {"timezone": str(args.timezone), "hour_valid_n": int(np.sum(hour_np >= 0))},
        "route_choice": {"od_bins": od_bins, "top_od": top_od, "min_od_n": min_od_n, "picked": picked},
        "artifacts": {
            "detour_ratio_hist_png": str(detour_hist_png),
            "detour_ratio_by_hour_png": str(detour_by_hour_png),
            "route_choice_top_od_png": str(route_png),
            "route_choice_top_od_json": str(route_json),
        },
    }

    (out_dir / "story_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    route_json.write_text(json.dumps({"picked": picked}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
