from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    if isinstance(tag, (list, tuple, set)):
        if not tag:
            return ""
        tag = next(iter(tag))
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


def _latlon_to_grid_yx_float(grid: GridSpec, *, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    x01 = (lon - float(grid.bbox.min_lon)) / float(grid.bbox.max_lon - grid.bbox.min_lon)
    y01 = (float(grid.bbox.max_lat) - lat) / float(grid.bbox.max_lat - grid.bbox.min_lat)
    x = (x01 * float(grid.W)).astype(np.float32)
    y = (y01 * float(grid.H)).astype(np.float32)
    return y, x


def _percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(np.asarray(x, dtype=np.float64), q))


@dataclass(frozen=True)
class BuildCfg:
    road_types: str
    keep_tier3: bool


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

    t0 = time.time()
    grid = _load_grid_from_semantic_dir(semantic_dir)
    bbox = grid.bbox
    roads_allow = ROAD_TYPES_A if str(cfg.road_types) == "A" else ROAD_TYPES_B

    osm = OSM(str(osm_pbf), bounding_box=[bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat])
    roads = osm.get_network(network_type="driving")
    if roads is None or len(roads) == 0:
        raise SystemExit("No roads extracted from OSM within bbox. Check osm_pbf/bbox.")

    for col in ("u", "v", "highway"):
        if col not in roads.columns:
            raise SystemExit(f"Unexpected pyrosm output: missing '{col}' column.")
    if "length" not in roads.columns:
        raise SystemExit("Unexpected pyrosm output: missing 'length' column (meters).")

    hw = roads["highway"].apply(_normalize_highway_tag)
    roads = roads[hw.isin(roads_allow)]
    if not bool(cfg.keep_tier3):
        roads = roads[hw.apply(_tier_id).astype(int) <= 2]
    if len(roads) == 0:
        raise SystemExit("No roads after filtering by road_types. Try --road_types B or --keep_tier3.")

    # Map u/v osm node ids -> dense indices by looking up node lat/lon.
    u_osm = np.asarray(roads["u"].values, dtype=np.int64).reshape(-1)
    v_osm = np.asarray(roads["v"].values, dtype=np.int64).reshape(-1)
    osm_ids = np.unique(np.concatenate([u_osm, v_osm], axis=0).astype(np.int64, copy=False))
    if osm_ids.size == 0:
        raise SystemExit("No node ids found in pyrosm roads table.")

    nodes = osm.get_node_data()
    if nodes is None or len(nodes) == 0:
        raise SystemExit("pyrosm.get_node_data() returned empty; cannot build native road graph.")
    for col in ("id", "lat", "lon"):
        if col not in nodes.columns:
            raise SystemExit(f"Unexpected pyrosm node table: missing '{col}' column.")

    # Filter to needed ids only.
    nodes = nodes[nodes["id"].isin(osm_ids)]
    if nodes is None or len(nodes) == 0:
        raise SystemExit("No nodes found for u/v ids (bbox mismatch?).")

    node_osm_id = np.asarray(nodes["id"].values, dtype=np.int64).reshape(-1)
    node_lat = np.asarray(nodes["lat"].values, dtype=np.float64).reshape(-1)
    node_lon = np.asarray(nodes["lon"].values, dtype=np.float64).reshape(-1)
    order = np.argsort(node_osm_id, kind="mergesort")
    node_osm_id = node_osm_id[order]
    node_lat = node_lat[order]
    node_lon = node_lon[order]

    # Map endpoints via searchsorted.
    pos_u = np.searchsorted(node_osm_id, u_osm)
    pos_v = np.searchsorted(node_osm_id, v_osm)
    valid = (pos_u < node_osm_id.size) & (pos_v < node_osm_id.size) & (node_osm_id[pos_u] == u_osm) & (node_osm_id[pos_v] == v_osm)
    if not bool(np.any(valid)):
        raise SystemExit("All edges refer to missing nodes. Check node table and bbox.")

    u_idx = pos_u[valid].astype(np.int32, copy=False)
    v_idx = pos_v[valid].astype(np.int32, copy=False)

    length_m = np.asarray(roads["length"].values, dtype=np.float64).reshape(-1)[valid]
    length_m = np.asarray(length_m, dtype=np.float32)
    length_m = np.clip(length_m, 1e-3, np.finfo(np.float32).max)

    hw_valid = np.asarray(hw.values, dtype=object).reshape(-1)[valid]
    tier = np.asarray([_tier_id(str(s)) for s in hw_valid.tolist()], dtype=np.uint8)

    oneway = None
    if "oneway" in roads.columns:
        oneway = np.asarray(roads["oneway"].values, dtype=object).reshape(-1)[valid]

    # Node coordinates in grid space (float).
    node_y, node_x = _latlon_to_grid_yx_float(grid, lat=node_lat, lon=node_lon)
    inb = (node_x >= 0.0) & (node_x < float(grid.W)) & (node_y >= 0.0) & (node_y < float(grid.H))

    # Keep only in-bounds nodes; drop edges pointing to removed nodes.
    keep_node = inb.astype(bool)
    if not bool(np.all(keep_node)):
        remap = np.full((node_osm_id.size,), -1, dtype=np.int32)
        remap[keep_node] = np.arange(int(keep_node.sum()), dtype=np.int32)
        u2 = remap[u_idx.astype(np.int64, copy=False)]
        v2 = remap[v_idx.astype(np.int64, copy=False)]
        keep_e = (u2 >= 0) & (v2 >= 0)
        u_idx = u2[keep_e].astype(np.int32, copy=False)
        v_idx = v2[keep_e].astype(np.int32, copy=False)
        length_m = length_m[keep_e].astype(np.float32, copy=False)
        tier = tier[keep_e].astype(np.uint8, copy=False)
        if oneway is not None:
            oneway = oneway[keep_e]

        node_osm_id = node_osm_id[keep_node]
        node_lat = node_lat[keep_node]
        node_lon = node_lon[keep_node]
        node_y = node_y[keep_node]
        node_x = node_x[keep_node]

    n_nodes = int(node_osm_id.size)
    if n_nodes <= 0 or int(u_idx.size) <= 0:
        raise SystemExit("Built an empty native graph after in-bounds filtering.")

    # Expand to directed edges (respect oneway when available; otherwise add both directions).
    if oneway is None:
        eu = np.concatenate([u_idx, v_idx], axis=0)
        ev = np.concatenate([v_idx, u_idx], axis=0)
        ew = np.concatenate([length_m, length_m], axis=0)
        et = np.concatenate([tier, tier], axis=0)
    else:
        one = np.asarray([(bool(x) if not (x is None or (isinstance(x, float) and np.isnan(x))) else False) for x in oneway.tolist()], dtype=bool)
        eu_f = u_idx
        ev_f = v_idx
        ew_f = length_m
        et_f = tier
        eu_b = v_idx[~one]
        ev_b = u_idx[~one]
        ew_b = length_m[~one]
        et_b = tier[~one]
        eu = np.concatenate([eu_f, eu_b], axis=0)
        ev = np.concatenate([ev_f, ev_b], axis=0)
        ew = np.concatenate([ew_f, ew_b], axis=0)
        et = np.concatenate([et_f, et_b], axis=0)

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "build_road_graph_native_from_osm",
        "inputs": {"osm_pbf": str(osm_pbf), "semantic_dir": str(semantic_dir), "city": (str(city) if city else None)},
        "config": {"road_types": str(cfg.road_types), "keep_tier3": bool(cfg.keep_tier3)},
        "grid": {"H": int(grid.H), "W": int(grid.W), "bbox": grid.bbox.__dict__},
        "stats": {
            "n_nodes": int(n_nodes),
            "n_edges_directed": int(eu.size),
            "edge_len_m_p50": _percentile(ew, 50),
            "edge_len_m_p90": _percentile(ew, 90),
            "tier_counts": np.bincount(np.clip(et.astype(np.int64), 0, 3), minlength=4).astype(np.int64).tolist(),
            "elapsed_s": float(time.time() - t0),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "road_graph.npz"
    report_json = out_dir / "report.json"

    np.savez_compressed(
        out_npz,
        node_osm_id=node_osm_id.astype(np.int64, copy=False),
        node_lat=node_lat.astype(np.float64, copy=False),
        node_lon=node_lon.astype(np.float64, copy=False),
        node_y=node_y.astype(np.float32, copy=False),
        node_x=node_x.astype(np.float32, copy=False),
        edge_u=eu.astype(np.int32, copy=False),
        edge_v=ev.astype(np.int32, copy=False),
        edge_w_m=ew.astype(np.float32, copy=False),
        edge_len_m=ew.astype(np.float32, copy=False),
        edge_tier=et.astype(np.uint8, copy=False),
        meta=meta,
    )
    report_json.write_text(json.dumps({"ok": True, "out_npz": str(out_npz), "meta": meta}, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"ok": True, "out_npz": str(out_npz), "report_json": str(report_json), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a native (non-raster) road graph from OSM .pbf using pyrosm.")
    p.add_argument("--osm_pbf", type=Path, required=True)
    p.add_argument("--semantic_dir", type=Path, required=True, help="City semantic dir containing osm_road_prob_meta.json (bbox/grid).")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--city", type=str, default=None)
    p.add_argument("--road_types", choices=["A", "B"], default="B")
    p.add_argument("--keep_tier3", action="store_true", help="Keep tier=3 roads (unknown/other). Default: drop them.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = build_graph(
        osm_pbf=Path(args.osm_pbf),
        semantic_dir=Path(args.semantic_dir),
        out_dir=Path(args.out_dir),
        city=(str(args.city) if args.city else None),
        cfg=BuildCfg(road_types=str(args.road_types), keep_tier3=bool(args.keep_tier3)),
    )
    meta = report["meta"]
    compact = {
        "ok": True,
        "out_npz": report["out_npz"],
        "n_nodes": int(meta["stats"]["n_nodes"]),
        "n_edges_directed": int(meta["stats"]["n_edges_directed"]),
        "edge_len_p50_m": float(meta["stats"]["edge_len_m_p50"]),
        "edge_len_p90_m": float(meta["stats"]["edge_len_m_p90"]),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

