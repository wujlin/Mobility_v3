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

TIER_MAJOR = {"motorway", "trunk", "primary", "secondary"}
TIER_MINOR = {"tertiary", "residential"}
TIER_SERVICE = {"service", "unclassified"}


def _load_grid_from_semantic_dir(semantic_dir: Path) -> GridSpec:
    meta_path = Path(semantic_dir) / "osm_road_prob_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path} (needed for bbox/grid).")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    g = meta.get("grid", {})
    bbox = g.get("bbox", {})
    return GridSpec(
        H=int(g["H"]),
        W=int(g["W"]),
        bbox=BBox(
            min_lon=float(bbox["min_lon"]),
            min_lat=float(bbox["min_lat"]),
            max_lon=float(bbox["max_lon"]),
            max_lat=float(bbox["max_lat"]),
        ),
    )


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


def _normalize_highway_tag(tag: object) -> str:
    s = str(tag or "").strip()
    if not s:
        return ""
    # Some OSM extracts contain *_link variants; treat them as the base type.
    if s.endswith("_link"):
        s = s[: -len("_link")]
    return s


def _tier_prob_from_roads(
    roads,
    *,
    grid: GridSpec,
    buffer_m: float,
    sigma_m: float,
    tier_weights: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a tiered road probability field using three masks:
    - major roads: high probability
    - minor roads: medium probability
    - service roads: low probability

    Returns: (mask_union, dist_m, road_prob, road_prob_major, road_prob_minor, road_prob_service)
    """
    if "highway" not in roads.columns:
        raise SystemExit("Expected 'highway' column in OSM roads GeoDataFrame.")

    hw = roads["highway"].apply(_normalize_highway_tag)
    major = roads[hw.isin(TIER_MAJOR)]
    minor = roads[hw.isin(TIER_MINOR)]
    service = roads[hw.isin(TIER_SERVICE)]

    # Rasterize each tier.
    mask_major = rasterize_roads_to_mask(major["geometry"].values, grid) if len(major) else np.zeros((grid.H, grid.W), dtype=bool)
    mask_minor = rasterize_roads_to_mask(minor["geometry"].values, grid) if len(minor) else np.zeros((grid.H, grid.W), dtype=bool)
    mask_service = rasterize_roads_to_mask(service["geometry"].values, grid) if len(service) else np.zeros((grid.H, grid.W), dtype=bool)
    mask_union = mask_major | mask_minor | mask_service

    res_y_m, res_x_m = grid.resolution_m()
    iters = int(np.ceil(float(buffer_m) / max(1e-6, min(res_x_m, res_y_m))))
    if iters > 0:
        mask_major = binary_dilation(mask_major, iterations=iters)
        mask_minor = binary_dilation(mask_minor, iterations=iters)
        mask_service = binary_dilation(mask_service, iterations=iters)
        mask_union = binary_dilation(mask_union, iterations=iters)

    # Distances to each tier (meters).
    dist_major = distance_transform_edt(~mask_major, sampling=(res_y_m, res_x_m)).astype(np.float32) if np.any(mask_major) else None
    dist_minor = distance_transform_edt(~mask_minor, sampling=(res_y_m, res_x_m)).astype(np.float32) if np.any(mask_minor) else None
    dist_service = distance_transform_edt(~mask_service, sampling=(res_y_m, res_x_m)).astype(np.float32) if np.any(mask_service) else None

    # Always report distance to the union mask for diagnostics.
    dist_union = distance_transform_edt(~mask_union, sampling=(res_y_m, res_x_m)).astype(np.float32)

    w_major, w_minor, w_service = (float(tier_weights[0]), float(tier_weights[1]), float(tier_weights[2]))
    if not (0.0 < w_service <= w_minor <= w_major <= 1.0):
        raise SystemExit("--tier_weights must satisfy 0 < service <= minor <= major <= 1")

    zeros = np.zeros((grid.H, grid.W), dtype=np.float32)
    prob_major = (w_major * np.exp(-dist_major / float(sigma_m))).astype(np.float32) if dist_major is not None else zeros
    prob_minor = (w_minor * np.exp(-dist_minor / float(sigma_m))).astype(np.float32) if dist_minor is not None else zeros
    prob_service = (w_service * np.exp(-dist_service / float(sigma_m))).astype(np.float32) if dist_service is not None else zeros
    prob = np.maximum(np.maximum(prob_major, prob_minor), prob_service).astype(np.float32, copy=False)

    return np.asarray(mask_union, np.uint8), dist_union, prob, prob_major, prob_minor, prob_service


def main() -> None:
    ap = argparse.ArgumentParser(description="Build OSM-based road_mask/dist_to_road/road_prob for a city grid (default: Detroit core).")
    ap.add_argument("--osm_pbf", type=Path, required=True, help="OSM .pbf file path (Detroit region)")
    ap.add_argument(
        "--semantic_dir",
        type=Path,
        default=None,
        help="Optional city semantic_dir containing osm_road_prob_meta.json; if set, bbox/H/W will be loaded from it. (out_dir defaults to semantic_dir)",
    )
    ap.add_argument("--out_dir", type=Path, default=None, help="Output directory (default: semantic_dir if provided)")
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Override bbox in EPSG:4326. Default: Detroit core bbox.",
    )
    ap.add_argument("--grid_h", type=int, default=None, help="Override grid H (default: 1024)")
    ap.add_argument("--grid_w", type=int, default=None, help="Override grid W (default: 1024)")
    ap.add_argument("--road_types", choices=["A", "B"], default="B", help="Road types set (A=conservative, B=more complete)")
    ap.add_argument("--buffer_m", type=float, default=15.0, help="Road width buffer (meters) via dilation")
    ap.add_argument("--road_prob_sigma_m", type=float, default=50.0, help="Sigma for exp(-dist/sigma)")
    ap.add_argument(
        "--road_prob_variant",
        choices=["distance", "tiered"],
        default="distance",
        help="How to define road_prob: 'distance' treats all drivable roads equally; 'tiered' assigns lower probability to minor/service roads to better reflect corridor preference.",
    )
    ap.add_argument(
        "--tier_weights",
        type=float,
        nargs=3,
        default=[1.0, 0.7, 0.4],
        metavar=("MAJOR", "MINOR", "SERVICE"),
        help="Weights for tiered road_prob (must satisfy 0 < SERVICE <= MINOR <= MAJOR <= 1).",
    )
    ap.add_argument(
        "--save_tier_probs",
        action="store_true",
        help="When road_prob_variant=tiered: also save osm_road_prob_{major,minor,service}.npy for downstream semantic conditioning.",
    )
    ap.add_argument(
        "--tier_only",
        action="store_true",
        help="When road_prob_variant=tiered and save_tier_probs: only save tier prob rasters (do not overwrite osm_road_prob.npy/meta).",
    )
    args = ap.parse_args()

    try:
        from pyrosm import OSM  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit("Missing dependency: pyrosm. Install via conda/pip (plus shapely/geopandas).") from e

    if args.out_dir is None and args.semantic_dir is None:
        raise SystemExit("Provide --out_dir or --semantic_dir (to infer out_dir and grid).")
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.semantic_dir)  # type: ignore[arg-type]

    if args.semantic_dir is not None:
        grid = _load_grid_from_semantic_dir(Path(args.semantic_dir))
    elif args.bbox is None and args.grid_h is None and args.grid_w is None:
        grid = _default_detroit_core_grid()
    else:
        bbox_vals = args.bbox or [-83.25, 42.25, -82.95, 42.50]
        bbox = BBox(min_lon=float(bbox_vals[0]), min_lat=float(bbox_vals[1]), max_lon=float(bbox_vals[2]), max_lat=float(bbox_vals[3]))
        H = int(args.grid_h or 1024)
        W = int(args.grid_w or 1024)
        grid = GridSpec(H=H, W=W, bbox=bbox)
    road_types = ROAD_TYPES_A if args.road_types == "A" else ROAD_TYPES_B

    bbox = grid.bbox
    osm = OSM(str(args.osm_pbf), bounding_box=[bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat])
    # pyrosm returns a GeoDataFrame with a 'geometry' column
    roads = osm.get_network(network_type="driving")
    if roads is None or len(roads) == 0:
        raise SystemExit("No roads extracted from OSM within bbox. Check osm_pbf/bbox/road_types.")

    if "highway" in roads.columns:
        hw = roads["highway"].apply(_normalize_highway_tag)
        roads = roads[hw.isin(road_types)]

    if "geometry" not in roads.columns:
        raise SystemExit("Unexpected pyrosm output: missing 'geometry' column.")

    res_y_m, res_x_m = grid.resolution_m()
    if args.road_prob_variant == "tiered":
        mask_u8, dist_m, road_prob, prob_major, prob_minor, prob_service = _tier_prob_from_roads(
            roads,
            grid=grid,
            buffer_m=float(args.buffer_m),
            sigma_m=float(args.road_prob_sigma_m),
            tier_weights=(float(args.tier_weights[0]), float(args.tier_weights[1]), float(args.tier_weights[2])),
        )
        mask = mask_u8.astype(bool)
    else:
        mask = rasterize_roads_to_mask(roads["geometry"].values, grid)
        iters = int(np.ceil(float(args.buffer_m) / max(1e-6, min(res_x_m, res_y_m))))
        if iters > 0:
            mask = binary_dilation(mask, iterations=iters)
        dist_m = distance_transform_edt(~mask, sampling=(res_y_m, res_x_m)).astype(np.float32)
        road_prob = np.exp(-dist_m / float(args.road_prob_sigma_m)).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.tier_only):
        if args.road_prob_variant != "tiered" or not bool(args.save_tier_probs):
            raise SystemExit("--tier_only requires --road_prob_variant tiered and --save_tier_probs.")
        np.save(out_dir / "osm_road_prob_major.npy", np.asarray(prob_major, np.float32))
        np.save(out_dir / "osm_road_prob_minor.npy", np.asarray(prob_minor, np.float32))
        np.save(out_dir / "osm_road_prob_service.npy", np.asarray(prob_service, np.float32))
        print(
            json.dumps(
                {
                    "ok": True,
                    "out_dir": str(out_dir),
                    "tier_only": True,
                    "grid": {"H": int(grid.H), "W": int(grid.W), "bbox": bbox.__dict__},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    np.save(out_dir / "osm_road_mask.npy", np.asarray(mask, np.uint8))
    np.save(out_dir / "osm_dist_to_road_m.npy", dist_m)
    np.save(out_dir / "osm_road_prob.npy", road_prob)
    if bool(args.save_tier_probs) and args.road_prob_variant == "tiered":
        np.save(out_dir / "osm_road_prob_major.npy", np.asarray(prob_major, np.float32))
        np.save(out_dir / "osm_road_prob_minor.npy", np.asarray(prob_minor, np.float32))
        np.save(out_dir / "osm_road_prob_service.npy", np.asarray(prob_service, np.float32))

    meta = {
        "grid": {"H": grid.H, "W": grid.W, "bbox": bbox.__dict__},
        "road_types": args.road_types,
        "buffer_m": float(args.buffer_m),
        "road_prob_sigma_m": float(args.road_prob_sigma_m),
        "road_prob_variant": str(args.road_prob_variant),
        "tier_weights": [float(x) for x in args.tier_weights],
        "save_tier_probs": bool(args.save_tier_probs),
        "res_x_m": res_x_m,
        "res_y_m": res_y_m,
    }
    (out_dir / "osm_road_prob_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), **meta}, indent=2))


if __name__ == "__main__":
    main()
