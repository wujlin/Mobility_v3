from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from src.utils.geo_grid import BBox, GridSpec


def _default_detroit_core_grid() -> GridSpec:
    bbox = BBox(min_lon=-83.25, max_lon=-82.95, min_lat=42.25, max_lat=42.50)
    return GridSpec(H=1024, W=1024, bbox=bbox)


ROAD_TYPES_A = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "residential",
}

ROAD_TYPES_B = ROAD_TYPES_A | {"service", "unclassified"}


def _iter_geom_coords(geom) -> Iterator[np.ndarray]:
    """
    Yield Nx2 arrays of (lon, lat) coords for LineString/MultiLineString geometries.
    """
    if geom is None:
        return
    gtype = getattr(geom, "geom_type", None)
    if gtype == "LineString":
        yield np.asarray(geom.coords, dtype=np.float64)
    elif gtype == "MultiLineString":
        for part in geom.geoms:
            yield np.asarray(part.coords, dtype=np.float64)


def _bresenham(y0: int, x0: int, y1: int, x1: int) -> Iterator[Tuple[int, int]]:
    """
    Bresenham line rasterization in (y,x).
    """
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy

    y, x = y0, x0
    while True:
        yield y, x
        if y == y1 and x == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def rasterize_roads_to_mask(geoms: Iterable, grid: GridSpec) -> np.ndarray:
    mask = np.zeros((grid.H, grid.W), dtype=bool)
    for geom in geoms:
        for coords in _iter_geom_coords(geom):
            if coords.shape[0] < 2:
                continue
            lon = coords[:, 0]
            lat = coords[:, 1]
            y, x = grid.latlon_to_yx(lat, lon)
            inb = grid.in_bounds(y, x)
            y = y[inb]
            x = x[inb]
            if y.size < 2:
                continue
            for i in range(y.size - 1):
                for yy, xx in _bresenham(int(y[i]), int(x[i]), int(y[i + 1]), int(x[i + 1])):
                    if 0 <= yy < grid.H and 0 <= xx < grid.W:
                        mask[yy, xx] = True
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description="Build OSM-based road_mask/dist_to_road/road_prob for Detroit core grid.")
    ap.add_argument("--osm_pbf", type=Path, required=True, help="OSM .pbf file path (Detroit region)")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output directory (e.g. data/processed_worldtrace_detroit)")
    ap.add_argument("--road_types", choices=["A", "B"], default="B", help="Road types set (A=conservative, B=more complete)")
    ap.add_argument("--buffer_m", type=float, default=15.0, help="Road width buffer (meters) via dilation")
    ap.add_argument("--road_prob_sigma_m", type=float, default=50.0, help="Sigma for exp(-dist/sigma)")
    args = ap.parse_args()

    try:
        from pyrosm import OSM  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit("Missing dependency: pyrosm. Install via conda/pip (plus shapely/geopandas).") from e

    grid = _default_detroit_core_grid()
    road_types = ROAD_TYPES_A if args.road_types == "A" else ROAD_TYPES_B

    bbox = grid.bbox
    osm = OSM(str(args.osm_pbf), bounding_box=[bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat])
    # pyrosm returns a GeoDataFrame with a 'geometry' column
    roads = osm.get_network(network_type="driving")
    if roads is None or len(roads) == 0:
        raise SystemExit("No roads extracted from OSM within bbox. Check osm_pbf/bbox/road_types.")

    if "highway" in roads.columns:
        roads = roads[roads["highway"].isin(road_types)]

    if "geometry" not in roads.columns:
        raise SystemExit("Unexpected pyrosm output: missing 'geometry' column.")

    mask = rasterize_roads_to_mask(roads["geometry"].values, grid)

    res_y_m, res_x_m = grid.resolution_m()
    iters = int(np.ceil(float(args.buffer_m) / max(1e-6, min(res_x_m, res_y_m))))
    if iters > 0:
        mask = binary_dilation(mask, iterations=iters)

    dist_m = distance_transform_edt(~mask, sampling=(res_y_m, res_x_m)).astype(np.float32)
    road_prob = np.exp(-dist_m / float(args.road_prob_sigma_m)).astype(np.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "osm_road_mask.npy", np.asarray(mask, np.uint8))
    np.save(args.out_dir / "osm_dist_to_road_m.npy", dist_m)
    np.save(args.out_dir / "osm_road_prob.npy", road_prob)

    meta = {
        "grid": {"H": grid.H, "W": grid.W, "bbox": bbox.__dict__},
        "road_types": args.road_types,
        "buffer_m": float(args.buffer_m),
        "road_prob_sigma_m": float(args.road_prob_sigma_m),
        "res_x_m": res_x_m,
        "res_y_m": res_y_m,
    }
    (args.out_dir / "osm_road_prob_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), **meta}, indent=2))


if __name__ == "__main__":
    main()
