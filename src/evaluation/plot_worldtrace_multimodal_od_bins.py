from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class StateBBoxes:
    # Loose bboxes (from user notes) for choosing which pbf to draw.
    mi: Tuple[float, float, float, float] = (-90.4, 41.7, -82.4, 48.3)
    oh: Tuple[float, float, float, float] = (-84.8, 38.4, -80.5, 42.0)

    def which(self, lat: float, lon: float) -> str:
        mi0, mi1, mi2, mi3 = self.mi
        if (mi1 <= lat <= mi3) and (mi0 <= lon <= mi2):
            return "MI"
        oh0, oh1, oh2, oh3 = self.oh
        if (oh1 <= lat <= oh3) and (oh0 <= lon <= oh2):
            return "OH"
        return "OTHER"


def _safe_float(v: object) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pick_latlon(row: Dict[str, str], *, prefer_matched: bool) -> Tuple[Optional[float], Optional[float]]:
    if prefer_matched:
        lat_m = _safe_float(row.get("matched_latitude", "")) or _safe_float(row.get("matched_lat", ""))
        lon_m = _safe_float(row.get("matched_longitude", "")) or _safe_float(row.get("matched_lon", ""))
        if lat_m is not None and lon_m is not None:
            return float(lat_m), float(lon_m)
    lat = _safe_float(row.get("latitude", "")) or _safe_float(row.get("lat", ""))
    lon = _safe_float(row.get("longitude", "")) or _safe_float(row.get("lon", ""))
    if lat is None or lon is None:
        return None, None
    return float(lat), float(lon)


def _read_traj_from_zip(zf: zipfile.ZipFile, member: str, *, prefer_matched: bool, downsample_step: int) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    step = max(1, int(downsample_step))
    with zf.open(member, "r") as f:
        text = io.TextIOWrapper(f, encoding="utf-8", errors="ignore", newline="")
        reader = csv.DictReader(text)
        for i, row in enumerate(reader):
            if (i % step) != 0:
                continue
            lat, lon = _pick_latlon(row, prefer_matched=bool(prefer_matched))
            if lat is None or lon is None:
                continue
            pts.append((float(lat), float(lon)))
    return pts


def _bin_int(x: float, bin_deg: float) -> int:
    return int(math.floor(float(x) / float(bin_deg)))


def _bins_from_points(points: Sequence[Tuple[float, float]], *, bin_deg: float) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    last: Optional[Tuple[int, int]] = None
    for lat, lon in points:
        b = (_bin_int(lon, bin_deg), _bin_int(lat, bin_deg))
        if last is None or b != last:
            out.append(b)
            last = b
    return out


def _union_bins(trajs: Sequence[Sequence[Tuple[float, float]]], *, bin_deg: float) -> set[Tuple[int, int]]:
    u: set[Tuple[int, int]] = set()
    for pts in trajs:
        u.update(_bins_from_points(pts, bin_deg=float(bin_deg)))
    return u


def _odbin_center(od_bin: Sequence[int], *, bin_deg: float) -> Tuple[float, float, float, float]:
    o_lon_bin, o_lat_bin, d_lon_bin, d_lat_bin = [int(x) for x in od_bin]
    o_lon = (o_lon_bin + 0.5) * float(bin_deg)
    o_lat = (o_lat_bin + 0.5) * float(bin_deg)
    d_lon = (d_lon_bin + 0.5) * float(bin_deg)
    d_lat = (d_lat_bin + 0.5) * float(bin_deg)
    return o_lat, o_lon, d_lat, d_lon


def _plot_roads(ax, *, osm_pbf: Path, bbox: Tuple[float, float, float, float]) -> None:
    try:
        from pyrosm import OSM  # type: ignore
    except ModuleNotFoundError:
        print("[WARN] pyrosm not installed; skip road background.", file=sys.stderr)
        return

    min_lon, min_lat, max_lon, max_lat = map(float, bbox)
    try:
        osm = OSM(str(osm_pbf), bounding_box=[min_lon, min_lat, max_lon, max_lat])
    except Exception as e:
        print(f"[WARN] pyrosm init failed ({osm_pbf}): {e}", file=sys.stderr)
        return

    # pyrosm can be brittle with very small/empty bounding boxes; be defensive and skip on failure.
    net = None
    try:
        try:
            net = osm.get_network(network_type="driving", nodes=False)
        except TypeError:
            net = osm.get_network(network_type="driving")
    except Exception as e:
        print(f"[WARN] pyrosm get_network failed ({osm_pbf}): {e}", file=sys.stderr)
        return
    roads = net[0] if isinstance(net, tuple) else net
    if roads is None or not hasattr(roads, "geometry"):
        return

    # Draw light grey road lines.
    def _iter_line_coords(g) -> Iterable[np.ndarray]:
        if g is None:
            return
        # MultiLineString / GeometryCollection: recurse into parts.
        parts = getattr(g, "geoms", None)
        if parts is not None:
            for part in parts:
                yield from _iter_line_coords(part)
            return
        # LineString: has coords. Multi-part geometries raise NotImplementedError for coords.
        try:
            coords = np.asarray(getattr(g, "coords"), dtype=np.float64)
        except NotImplementedError:
            return
        except Exception:
            coords = np.asarray(getattr(g, "coords", []), dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] < 2:
            return
        yield coords

    for geom in roads.geometry:
        for coords in _iter_line_coords(geom):
            # coords: (N,2)=(lon,lat)
            ax.plot(coords[:, 0], coords[:, 1], color="#cfcfcf", linewidth=0.6, alpha=0.6, zorder=0)


def _bbox_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    a0, a1, a2, a3 = map(float, a)
    b0, b1, b2, b3 = map(float, b)
    if a2 < b0 or b2 < a0:
        return False
    if a3 < b1 or b3 < a1:
        return False
    return True


def _plot_one_od(
    *,
    out_png: Path,
    od_bin: Sequence[int],
    n_routes: int,
    cluster_sizes: Sequence[int],
    top2_jaccard_dist: float,
    clusters_trajs: List[List[List[Tuple[float, float]]]],
    od_bin_deg: float,
    route_bin_deg: float,
    osm_pbf_mi: Optional[Path],
    osm_pbf_oh: Optional[Path],
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("Missing dependency: matplotlib (needed for plotting).") from e

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))
    ax_map, ax_bins = axes.tolist()

    # Collect all points for view bbox.
    pts_all = [p for cl in clusters_trajs for tr in cl for p in tr]
    if pts_all:
        lats = np.asarray([p[0] for p in pts_all], dtype=np.float64)
        lons = np.asarray([p[1] for p in pts_all], dtype=np.float64)
        min_lat = float(np.quantile(lats, 0.01))
        max_lat = float(np.quantile(lats, 0.99))
        min_lon = float(np.quantile(lons, 0.01))
        max_lon = float(np.quantile(lons, 0.99))
        pad_lat = max(0.01, 0.15 * (max_lat - min_lat))
        pad_lon = max(0.01, 0.15 * (max_lon - min_lon))
        view_bbox = (min_lon - pad_lon, min_lat - pad_lat, max_lon + pad_lon, max_lat + pad_lat)
    else:
        view_bbox = None

    # Road background (optional).
    if view_bbox is not None and (osm_pbf_mi is not None or osm_pbf_oh is not None):
        sbox = StateBBoxes()
        # Prefer the pbf whose state bbox intersects the local view bbox.
        mi_bbox = sbox.mi  # (min_lon,min_lat,max_lon,max_lat)
        oh_bbox = sbox.oh
        if (osm_pbf_oh is not None) and _bbox_intersects(view_bbox, oh_bbox):
            _plot_roads(ax_map, osm_pbf=osm_pbf_oh, bbox=view_bbox)
        elif (osm_pbf_mi is not None) and _bbox_intersects(view_bbox, mi_bbox):
            _plot_roads(ax_map, osm_pbf=osm_pbf_mi, bbox=view_bbox)
        else:
            # Outside MI/OH core boxes: skip background to avoid pyrosm empty-bbox crashes.
            pass

    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    labels = [f"C{i} (rep={len(clusters_trajs[i])}, n≈{int(cluster_sizes[i])})" for i in range(min(len(clusters_trajs), len(cluster_sizes)))]

    # Panel A: raw trajectories.
    for ci, trajs in enumerate(clusters_trajs):
        for tr in trajs:
            if len(tr) < 2:
                continue
            lat = np.asarray([p[0] for p in tr], dtype=np.float64)
            lon = np.asarray([p[1] for p in tr], dtype=np.float64)
            ax_map.plot(lon, lat, color=colors[ci % len(colors)], linewidth=1.8, alpha=0.75)
        if trajs:
            # one proxy for legend
            ax_map.plot([], [], color=colors[ci % len(colors)], linewidth=2.5, label=labels[ci])

    # Start/dest markers from OD bin center.
    o_lat, o_lon, d_lat, d_lon = _odbin_center(od_bin, bin_deg=float(od_bin_deg))
    ax_map.scatter([o_lon], [o_lat], s=80, c="white", edgecolors="black", linewidths=2.0, zorder=5, label="O (bin center)")
    ax_map.scatter([d_lon], [d_lat], s=80, c="black", marker="s", edgecolors="white", linewidths=1.5, zorder=5, label="D (bin center)")

    if view_bbox is not None:
        min_lon, min_lat, max_lon, max_lat = view_bbox
        ax_map.set_xlim(min_lon, max_lon)
        ax_map.set_ylim(min_lat, max_lat)
    ax_map.set_title("Trajectories (rep files) on roads (if provided)")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.legend(loc="lower left", frameon=False, fontsize=9)

    # Panel B: corridor footprint (route_bin_deg occupancy).
    u0 = _union_bins(clusters_trajs[0], bin_deg=float(route_bin_deg)) if len(clusters_trajs) > 0 else set()
    u1 = _union_bins(clusters_trajs[1], bin_deg=float(route_bin_deg)) if len(clusters_trajs) > 1 else set()
    ov = u0 & u1

    def _scatter_bins(ax, bins: Iterable[Tuple[int, int]], *, color: str, label: str, alpha: float) -> None:
        bins = list(bins)
        if not bins:
            return
        lon = np.asarray([(b[0] + 0.5) * float(route_bin_deg) for b in bins], dtype=np.float64)
        lat = np.asarray([(b[1] + 0.5) * float(route_bin_deg) for b in bins], dtype=np.float64)
        ax.scatter(lon, lat, s=28, c=color, alpha=float(alpha), linewidths=0, label=label, marker="s")

    _scatter_bins(ax_bins, u0 - ov, color=colors[0], label="C0 bins", alpha=0.55)
    _scatter_bins(ax_bins, u1 - ov, color=colors[1], label="C1 bins", alpha=0.55)
    _scatter_bins(ax_bins, ov, color="#55a868", label="Overlap bins", alpha=0.75)

    if view_bbox is not None:
        min_lon, min_lat, max_lon, max_lat = view_bbox
        ax_bins.set_xlim(min_lon, max_lon)
        ax_bins.set_ylim(min_lat, max_lat)
    ax_bins.set_title(f"Corridor footprint (bin={route_bin_deg:.02f}°) top2_dist={top2_jaccard_dist:.2f}")
    ax_bins.set_aspect("equal", adjustable="box")
    ax_bins.set_xticks([])
    ax_bins.set_yticks([])
    ax_bins.legend(loc="lower left", frameon=False, fontsize=9)

    fig.suptitle(f"OD bin={list(map(int, od_bin))} n_routes={int(n_routes)} clusters={list(map(int, cluster_sizes))}", fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize multimodal OD bins from scan_multimodal_od_region.py report.json.")
    p.add_argument("--scan_report_json", type=Path, required=True)
    p.add_argument("--trajectory_zip", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--top_k", type=int, default=30, help="Plot top-K multimodal OD bins (sorted by n_routes, then separation).")
    p.add_argument("--od_indices", type=int, nargs="*", default=None, help="Optional: explicit indices into multimodal_od_bins list.")
    p.add_argument("--max_files_per_cluster", type=int, default=3, help="Max representative files to plot per cluster.")
    p.add_argument("--prefer_matched", action="store_true", help="Prefer matched_latitude/longitude if present.")
    p.add_argument("--downsample_step", type=int, default=5, help="Downsample points when plotting each trajectory.")

    p.add_argument("--osm_pbf_michigan", type=Path, default=None, help="Optional: michigan-latest.osm.pbf for road background.")
    p.add_argument("--osm_pbf_ohio", type=Path, default=None, help="Optional: ohio-latest.osm.pbf for road background.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    rep = json.loads(Path(args.scan_report_json).read_text(encoding="utf-8"))
    mm = rep.get("multimodal_od_bins", [])
    if not isinstance(mm, list) or not mm:
        raise SystemExit(f"No multimodal_od_bins found in: {args.scan_report_json}")

    cfg = rep.get("scan_config", {}) or {}
    od_bin_deg = float(cfg.get("od_bin_deg", 0.01))
    route_bin_deg = float(cfg.get("route_bin_deg", 0.05))

    if args.od_indices:
        pick = [int(i) for i in args.od_indices if 0 <= int(i) < len(mm)]
    else:
        pick = list(range(min(int(args.top_k), len(mm))))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    osm_mi = Path(args.osm_pbf_michigan) if args.osm_pbf_michigan is not None else None
    osm_oh = Path(args.osm_pbf_ohio) if args.osm_pbf_ohio is not None else None

    summary = {
        "ok": True,
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "scan_report_json": str(args.scan_report_json),
            "trajectory_zip": str(args.trajectory_zip),
            "osm_pbf_michigan": (str(osm_mi) if osm_mi is not None else None),
            "osm_pbf_ohio": (str(osm_oh) if osm_oh is not None else None),
        },
        "config": {
            "od_bin_deg": float(od_bin_deg),
            "route_bin_deg": float(route_bin_deg),
            "top_k": int(args.top_k),
            "max_files_per_cluster": int(args.max_files_per_cluster),
            "prefer_matched": bool(args.prefer_matched),
            "downsample_step": int(args.downsample_step),
        },
        "plotted": [],
    }

    with zipfile.ZipFile(str(args.trajectory_zip), "r") as zf:
        for rank, mi in enumerate(pick):
            ent = mm[int(mi)]
            od_bin = ent.get("od_bin", None)
            if not isinstance(od_bin, list) or len(od_bin) != 4:
                continue
            n_routes = int(ent.get("n_routes", 0))
            cluster_sizes = ent.get("cluster_sizes", [])
            top2 = float(ent.get("top2_jaccard_dist", float("nan")))
            rep_files = ent.get("cluster_rep_files", [])
            if not isinstance(rep_files, list) or len(rep_files) < 2:
                continue

            clusters_trajs: List[List[List[Tuple[float, float]]]] = []
            used_files: List[List[str]] = []
            for ci, files in enumerate(rep_files[:2]):  # only top-2 for quick check
                f_list = []
                trajs = []
                if isinstance(files, list):
                    for member in files[: int(args.max_files_per_cluster)]:
                        if not isinstance(member, str):
                            continue
                        try:
                            zf.getinfo(member)
                        except KeyError:
                            continue
                        pts = _read_traj_from_zip(
                            zf,
                            member,
                            prefer_matched=bool(args.prefer_matched),
                            downsample_step=int(args.downsample_step),
                        )
                        if len(pts) >= 2:
                            trajs.append(pts)
                            f_list.append(member)
                clusters_trajs.append(trajs)
                used_files.append(f_list)

            # Ensure we have something to plot.
            if not clusters_trajs or all(len(t) == 0 for t in clusters_trajs):
                continue

            od_name = "_".join(str(int(x)) for x in od_bin)
            out_png = out_dir / f"od_{rank:03d}_n{n_routes}_{od_name}.png"
            _plot_one_od(
                out_png=out_png,
                od_bin=od_bin,
                n_routes=int(n_routes),
                cluster_sizes=[int(x) for x in (cluster_sizes or [])],
                top2_jaccard_dist=float(top2),
                clusters_trajs=clusters_trajs,
                od_bin_deg=float(od_bin_deg),
                route_bin_deg=float(route_bin_deg),
                osm_pbf_mi=osm_mi,
                osm_pbf_oh=osm_oh,
            )
            summary["plotted"].append(
                {
                    "rank": int(rank),
                    "od_bin": [int(x) for x in od_bin],
                    "n_routes": int(n_routes),
                    "cluster_sizes": [int(x) for x in (cluster_sizes or [])],
                    "top2_jaccard_dist": float(top2),
                    "used_files": used_files,
                    "out_png": str(out_png),
                }
            )

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out_dir": str(out_dir), "n_plotted": int(len(summary["plotted"]))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
