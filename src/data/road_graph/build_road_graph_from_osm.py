from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from src.utils.geo_grid import BBox, GridSpec


TZ_SHANGHAI = timezone(timedelta(hours=8))

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


def _normalize_highway_tag(tag: object) -> str:
    s = str(tag or "").strip()
    if not s:
        return ""
    if s.endswith("_link"):
        s = s[: -len("_link")]
    return s


def _tier_id(highway: str) -> int:
    if highway in TIER_MAJOR:
        return 0
    if highway in TIER_MINOR:
        return 1
    if highway in TIER_SERVICE:
        return 2
    return 3


def _iter_geom_coords_lonlat(geom) -> Iterator[np.ndarray]:
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
            coords = np.asarray(part.coords, dtype=np.float64)
            if coords.shape[0] >= 2:
                yield coords


def _bresenham(y0: int, x0: int, y1: int, x1: int) -> Iterator[Tuple[int, int]]:
    """
    Bresenham line rasterization in (y,x).
    """
    dy = abs(int(y1) - int(y0))
    dx = abs(int(x1) - int(x0))
    sy = 1 if int(y0) < int(y1) else -1
    sx = 1 if int(x0) < int(x1) else -1
    err = dx - dy

    y, x = int(y0), int(x0)
    while True:
        yield y, x
        if y == int(y1) and x == int(x1):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _load_grid_from_semantic_dir(semantic_dir: Path) -> GridSpec:
    meta_path = semantic_dir / "osm_road_prob_meta.json"
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


@dataclass(frozen=True)
class BuildCfg:
    road_types: str


def build_graph(
    *,
    osm_pbf: Path,
    semantic_dir: Path,
    out_dir: Path,
    city: Optional[str],
    cfg: BuildCfg,
) -> Dict[str, object]:
    try:
        from pyrosm import OSM  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit("Missing dependency: pyrosm. Install via conda/pip (plus shapely/geopandas).") from e

    grid = _load_grid_from_semantic_dir(semantic_dir)
    bbox = grid.bbox
    roads_allow = ROAD_TYPES_A if str(cfg.road_types) == "A" else ROAD_TYPES_B

    osm = OSM(str(osm_pbf), bounding_box=[bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat])
    roads = osm.get_network(network_type="driving")
    if roads is None or len(roads) == 0:
        raise SystemExit("No roads extracted from OSM within bbox. Check osm_pbf/bbox.")
    if "highway" not in roads.columns or "geometry" not in roads.columns:
        raise SystemExit("Unexpected pyrosm output: missing highway/geometry columns.")

    hw = roads["highway"].apply(_normalize_highway_tag)
    roads = roads[hw.isin(roads_allow)]
    if len(roads) == 0:
        raise SystemExit("No roads after filtering by road_types. Try --road_types B.")

    # We build a graph in grid space:
    # - node_id is the grid cell id (y*W + x)
    # - nodes are unique road cells visited by OSM polylines
    # - edges connect consecutive rasterized cells along road polylines
    H, W = int(grid.H), int(grid.W)
    res_y_m, res_x_m = grid.resolution_m()

    cell_to_idx: Dict[int, int] = {}
    node_id_list: list[int] = []

    # Directed edge map: (u_idx,v_idx) -> (len_m, tier_id)
    edge_map: Dict[Tuple[int, int], Tuple[float, int]] = {}

    def _node_idx(y: int, x: int) -> Tuple[int, int]:
        cid = int(y) * int(W) + int(x)
        idx = cell_to_idx.get(cid)
        if idx is None:
            idx = int(len(node_id_list))
            cell_to_idx[cid] = idx
            node_id_list.append(int(cid))
        return int(idx), int(cid)

    def _add_edge(u: int, v: int, *, w_m: float, tier: int) -> None:
        key = (int(u), int(v))
        prev = edge_map.get(key)
        if prev is None:
            edge_map[key] = (float(w_m), int(tier))
            return
        w0, t0 = prev
        edge_map[key] = (float(min(float(w0), float(w_m))), int(min(int(t0), int(tier))))

    n_edge_rows = 0
    n_geom = 0
    n_raster_steps = 0
    for _, row in roads.iterrows():
        n_edge_rows += 1
        geom = row.get("geometry")
        highway = _normalize_highway_tag(row.get("highway"))
        if not highway:
            continue
        tier = _tier_id(highway)
        for coords in _iter_geom_coords_lonlat(geom):
            n_geom += 1
            if coords.shape[0] < 2:
                continue
            lon = coords[:, 0]
            lat = coords[:, 1]
            yy, xx = grid.latlon_to_yx(lat, lon)
            inb = grid.in_bounds(yy, xx)
            yy = yy[inb]
            xx = xx[inb]
            if yy.size < 2:
                continue

            prev_cell: Optional[Tuple[int, int]] = None
            for i in range(int(yy.size) - 1):
                y0, x0 = int(yy[i]), int(xx[i])
                y1, x1 = int(yy[i + 1]), int(xx[i + 1])
                for yx in _bresenham(y0, x0, y1, x1):
                    cy, cx = int(yx[0]), int(yx[1])
                    if not (0 <= cy < H and 0 <= cx < W):
                        continue
                    if prev_cell is not None and (cy, cx) != prev_cell:
                        py, px = prev_cell
                        u_idx, _ = _node_idx(py, px)
                        v_idx, _ = _node_idx(cy, cx)
                        dy = abs(int(cy) - int(py))
                        dx = abs(int(cx) - int(px))
                        step_m = float(np.hypot(float(dy) * float(res_y_m), float(dx) * float(res_x_m)))
                        if step_m > 0:
                            _add_edge(u_idx, v_idx, w_m=step_m, tier=int(tier))
                            _add_edge(v_idx, u_idx, w_m=step_m, tier=int(tier))
                            n_raster_steps += 1
                    prev_cell = (cy, cx)

    if not node_id_list or not edge_map:
        raise SystemExit("Built an empty road graph (0 nodes or 0 edges). Check OSM extract / bbox / road_types.")

    node_id = np.asarray(node_id_list, dtype=np.int64)
    n_nodes = int(node_id.shape[0])
    node_y = (node_id // int(W)).astype(np.float32, copy=False)
    node_x = (node_id % int(W)).astype(np.float32, copy=False)

    # Unpack edges.
    u_idx = np.asarray([k[0] for k in edge_map.keys()], dtype=np.int32)
    v_idx = np.asarray([k[1] for k in edge_map.keys()], dtype=np.int32)
    edge_len_m = np.asarray([v[0] for v in edge_map.values()], dtype=np.float32)
    edge_tier = np.asarray([v[1] for v in edge_map.values()], dtype=np.uint8)
    u_node = node_id[u_idx].astype(np.int64, copy=False)
    v_node = node_id[v_idx].astype(np.int64, copy=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "road_graph.npz"
    report_json = out_dir / "report.json"

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "build_road_graph_from_osm",
        "inputs": {"osm_pbf": str(osm_pbf), "semantic_dir": str(semantic_dir), "city": (str(city) if city else None)},
        "config": {"road_types": str(cfg.road_types)},
        "grid": {"H": int(grid.H), "W": int(grid.W), "bbox": grid.bbox.__dict__},
        "stats": {
            "n_edge_rows": int(n_edge_rows),
            "n_nodes": int(n_nodes),
            "n_edges_directed": int(len(u_idx)),
            "n_geom": int(n_geom),
            "n_raster_steps": int(n_raster_steps),
        },
    }

    np.savez_compressed(
        out_npz,
        node_id=node_id,
        node_y=node_y,
        node_x=node_x,
        node_yx=np.stack([node_y, node_x], axis=1).astype(np.float32, copy=False),
        edge_u=u_idx,
        edge_v=v_idx,
        edge_uv=np.stack([u_idx, v_idx], axis=1).astype(np.int32, copy=False),
        edge_u_node_id=u_node,
        edge_v_node_id=v_node,
        edge_w_m=edge_len_m,
        edge_len_m=edge_len_m,
        edge_tier=edge_tier,
        meta=meta,
    )
    report_json.write_text(json.dumps({"ok": True, "out_npz": str(out_npz), "meta": meta}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "out_npz": str(out_npz), "report_json": str(report_json), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a directed road graph (nodes+edges) from OSM .pbf within the semantic_dir grid bbox.")
    p.add_argument("--osm_pbf", type=Path, required=True)
    p.add_argument("--semantic_dir", type=Path, required=True, help="City semantic dir containing osm_road_prob_meta.json (bbox/grid).")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--city", type=str, default=None)
    p.add_argument("--road_types", choices=["A", "B"], default="B")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = build_graph(
        osm_pbf=Path(args.osm_pbf),
        semantic_dir=Path(args.semantic_dir),
        out_dir=Path(args.out_dir),
        city=(str(args.city) if args.city else None),
        cfg=BuildCfg(road_types=str(args.road_types)),
    )
    compact = {
        "ok": True,
        "out_npz": report["out_npz"],
        "report_json": report["report_json"],
        "n_nodes": int(report["meta"]["stats"]["n_nodes"]),
        "n_edges_directed": int(report["meta"]["stats"]["n_edges_directed"]),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
