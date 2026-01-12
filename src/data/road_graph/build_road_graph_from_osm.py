from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

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


def _iter_geom_endpoints_lonlat(geom) -> Iterator[Tuple[float, float, float, float]]:
    """
    Yield (lon0, lat0, lon1, lat1) for LineString geometries.
    """
    if geom is None:
        return
    gtype = getattr(geom, "geom_type", None)
    if gtype == "LineString":
        coords = np.asarray(geom.coords, dtype=np.float64)
        if coords.shape[0] >= 2:
            lon0, lat0 = float(coords[0, 0]), float(coords[0, 1])
            lon1, lat1 = float(coords[-1, 0]), float(coords[-1, 1])
            yield lon0, lat0, lon1, lat1
    elif gtype == "MultiLineString":
        # Fall back: take the first and last segment endpoints.
        first = None
        last = None
        for part in geom.geoms:
            coords = np.asarray(part.coords, dtype=np.float64)
            if coords.shape[0] < 2:
                continue
            if first is None:
                first = (float(coords[0, 0]), float(coords[0, 1]))
            last = (float(coords[-1, 0]), float(coords[-1, 1]))
        if first is not None and last is not None:
            yield first[0], first[1], last[0], last[1]


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

    # Node coordinate registry (OSM node id -> (y,x) in grid).
    node_xy: Dict[int, Tuple[float, float]] = {}

    # Directed edges as (u_id, v_id, weight_m, tier_id).
    edges_u: list[int] = []
    edges_v: list[int] = []
    edges_w: list[float] = []
    edges_tier: list[int] = []

    n_edge_rows = 0
    for _, row in roads.iterrows():
        n_edge_rows += 1
        try:
            u_id = int(row.get("u"))
            v_id = int(row.get("v"))
        except Exception:
            continue
        geom = row.get("geometry")
        highway = _normalize_highway_tag(row.get("highway"))
        if not highway:
            continue
        tier = _tier_id(highway)
        length_m = row.get("length")
        w_m = float(length_m) if length_m is not None and float(length_m) > 0 else None

        # Determine endpoints in grid space.
        lon0 = lat0 = lon1 = lat1 = None
        for a, b, c, d in _iter_geom_endpoints_lonlat(geom):
            lon0, lat0, lon1, lat1 = a, b, c, d
            break
        if lon0 is None or lat0 is None or lon1 is None or lat1 is None:
            continue
        y0, x0 = grid.latlon_to_yx(np.asarray([lat0]), np.asarray([lon0]))
        y1, x1 = grid.latlon_to_yx(np.asarray([lat1]), np.asarray([lon1]))
        y0f, x0f = float(y0[0]), float(x0[0])
        y1f, x1f = float(y1[0]), float(x1[0])
        node_xy.setdefault(u_id, (y0f, x0f))
        node_xy.setdefault(v_id, (y1f, x1f))

        if w_m is None:
            # Fallback: grid-distance in meters (approx).
            res_y_m, res_x_m = grid.resolution_m()
            dy_m = float(abs(y1f - y0f)) * float(res_y_m)
            dx_m = float(abs(x1f - x0f)) * float(res_x_m)
            w_m = float(np.hypot(dy_m, dx_m))

        oneway = row.get("oneway")
        is_oneway = bool(oneway) if oneway is not None else False

        edges_u.append(u_id)
        edges_v.append(v_id)
        edges_w.append(float(w_m))
        edges_tier.append(int(tier))
        if not is_oneway:
            edges_u.append(v_id)
            edges_v.append(u_id)
            edges_w.append(float(w_m))
            edges_tier.append(int(tier))

    # Build node index mapping.
    node_ids = np.asarray(sorted(node_xy.keys()), dtype=np.int64)
    n_nodes = int(node_ids.shape[0])
    id_to_idx = {int(nid): int(i) for i, nid in enumerate(node_ids.tolist())}
    node_y = np.zeros((n_nodes,), dtype=np.float32)
    node_x = np.zeros((n_nodes,), dtype=np.float32)
    for i, nid in enumerate(node_ids.tolist()):
        yy, xx = node_xy[int(nid)]
        node_y[i] = float(yy)
        node_x[i] = float(xx)

    # Map edges to indices (drop edges with missing endpoints).
    u_idx = []
    v_idx = []
    w_m = []
    tier_id = []
    u_node = []
    v_node = []
    dropped = 0
    for uu, vv, ww, tt in zip(edges_u, edges_v, edges_w, edges_tier):
        iu = id_to_idx.get(int(uu))
        iv = id_to_idx.get(int(vv))
        if iu is None or iv is None:
            dropped += 1
            continue
        u_idx.append(int(iu))
        v_idx.append(int(iv))
        u_node.append(int(uu))
        v_node.append(int(vv))
        w_m.append(float(ww))
        tier_id.append(int(tt))

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
            "dropped_edges": int(dropped),
        },
    }

    np.savez_compressed(
        out_npz,
        node_id=node_ids,
        node_y=node_y,
        node_x=node_x,
        node_yx=np.stack([node_y, node_x], axis=1).astype(np.float32, copy=False),
        edge_u=np.asarray(u_idx, dtype=np.int32),
        edge_v=np.asarray(v_idx, dtype=np.int32),
        edge_uv=np.stack([np.asarray(u_idx, dtype=np.int32), np.asarray(v_idx, dtype=np.int32)], axis=1).astype(np.int32, copy=False),
        edge_u_node_id=np.asarray(u_node, dtype=np.int64),
        edge_v_node_id=np.asarray(v_node, dtype=np.int64),
        edge_w_m=np.asarray(w_m, dtype=np.float32),
        edge_len_m=np.asarray(w_m, dtype=np.float32),
        edge_tier=np.asarray(tier_id, dtype=np.uint8),
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
