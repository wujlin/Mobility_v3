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


def _split_pyrosm_network(net) -> Tuple[object, Optional[object]]:
    if isinstance(net, tuple) and len(net) == 2:
        a, b = net
        a_cols = set(getattr(a, "columns", []))
        b_cols = set(getattr(b, "columns", []))
        a_is_edges = ("geometry" in a_cols) and ("highway" in a_cols)
        b_is_edges = ("geometry" in b_cols) and ("highway" in b_cols)
        if a_is_edges and not b_is_edges:
            return a, b
        if b_is_edges and not a_is_edges:
            return b, a
        return a, b
    return net, None


@dataclass(frozen=True)
class Config:
    road_types: str
    semantic_channels: str


def build_way_features(*, osm_pbf: Path, semantic_dir: Path, way_routes_npz: Path, out_npz: Path, cfg: Config) -> Dict[str, object]:
    try:
        from pyrosm import OSM  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("Missing dependency: pyrosm. Install via conda/pip (plus shapely/geopandas).") from e

    t0 = time.time()
    grid = _load_grid_from_semantic_dir(Path(semantic_dir))
    bbox = grid.bbox

    routes = np.load(str(way_routes_npz), allow_pickle=True)
    if "way_osm_id" not in routes.files:
        raise ValueError(f"way_routes_npz missing way_osm_id: {way_routes_npz}")
    way_osm_id = np.asarray(routes["way_osm_id"], dtype=np.int64).reshape(-1)
    M = int(way_osm_id.size)
    way_to_idx = {int(w): int(i) for i, w in enumerate(way_osm_id.tolist())}

    # Build a stable highway vocab (small, fixed).
    roads_allow = ROAD_TYPES_A if str(cfg.road_types) == "A" else ROAD_TYPES_B
    hw_vocab = sorted(list(roads_allow))
    hw_to_code = {h: int(i) for i, h in enumerate(hw_vocab)}

    # Load OSM roads within bbox.
    osm = OSM(str(osm_pbf), bounding_box=[bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat])
    net = None
    try:
        net = osm.get_network(network_type="driving", nodes=True)
    except TypeError:
        net = osm.get_network(network_type="driving")
    roads, _nodes = _split_pyrosm_network(net)

    # Robust way-id column.
    way_id_col = None
    for k in ("id", "osm_id", "osmid", "way_id"):
        if k in roads.columns:
            way_id_col = k
            break
    if way_id_col is None:
        raise SystemExit(f"Unexpected pyrosm output: missing way-id column. Got: {sorted(list(roads.columns))}")
    if "highway" not in roads.columns:
        raise SystemExit(f"Unexpected pyrosm output: missing 'highway' column. Got: {sorted(list(roads.columns))}")

    # Normalize + filter road types.
    roads = roads.copy()
    roads["_highway_norm"] = roads["highway"].apply(_normalize_highway_tag)
    roads = roads[roads["_highway_norm"].isin(roads_allow)]

    # Keep only ways we need (fast membership via python set).
    wanted = set(way_to_idx.keys())
    roads = roads[roads[way_id_col].isin(wanted)]

    # Length column (pyrosm usually provides meters).
    length_col = None
    for k in ("length", "length_m", "edge_len_m"):
        if k in roads.columns:
            length_col = k
            break
    if length_col is None:
        raise SystemExit(f"Unexpected pyrosm output: missing length column. Need one of length/length_m. Got: {sorted(list(roads.columns))}")

    # Precompute scalar transforms for speed.
    H, W = int(grid.H), int(grid.W)
    lon0 = float(bbox.min_lon)
    lat1 = float(bbox.max_lat)
    inv_lon = float(W) / max(float(bbox.max_lon - bbox.min_lon), 1e-12)
    inv_lat = float(H) / max(float(bbox.max_lat - bbox.min_lat), 1e-12)

    def _latlon_to_yx(lat: float, lon: float) -> Tuple[float, float]:
        x = (float(lon) - lon0) * inv_lon
        y = (lat1 - float(lat)) * inv_lat
        return y, x

    way_len_m = np.zeros((M,), dtype=np.float64)
    way_center_y = np.zeros((M,), dtype=np.float64)
    way_center_x = np.zeros((M,), dtype=np.float64)
    way_dir_y = np.zeros((M,), dtype=np.float64)
    way_dir_x = np.zeros((M,), dtype=np.float64)
    tier_sum = np.zeros((M, 4), dtype=np.float64)
    hw_sum = np.zeros((M, len(hw_vocab)), dtype=np.float64)

    # Iterate (subset columns to avoid pandas itertuples name quirks).
    roads_sub = roads[[way_id_col, length_col, "_highway_norm", "geometry"]]
    for wid, length_val, hw_norm, geom in roads_sub.itertuples(index=False, name=None):
        wi = way_to_idx.get(int(wid))
        if wi is None:
            continue
        hw = _normalize_highway_tag(hw_norm)
        if hw not in roads_allow:
            continue
        l = float(length_val or 0.0)
        if not np.isfinite(l) or l <= 0.0:
            continue
        if geom is None:
            continue
        coords = np.asarray(getattr(geom, "coords", []), dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] < 2:
            continue
        # coords: (N,2) = (lon,lat)
        lon = coords[:, 0]
        lat = coords[:, 1]
        c_lat = float(np.mean(lat))
        c_lon = float(np.mean(lon))
        cy, cx = _latlon_to_yx(c_lat, c_lon)

        y0, x0 = _latlon_to_yx(float(lat[0]), float(lon[0]))
        y1_, x1_ = _latlon_to_yx(float(lat[-1]), float(lon[-1]))
        dy = y1_ - y0
        dx = x1_ - x0
        norm = float((dy * dy + dx * dx) ** 0.5)
        if norm > 1e-6:
            dy /= norm
            dx /= norm
        else:
            dy = 0.0
            dx = 0.0

        way_len_m[wi] += l
        way_center_y[wi] += l * cy
        way_center_x[wi] += l * cx
        way_dir_y[wi] += l * dy
        way_dir_x[wi] += l * dx
        tier_sum[wi, _tier_id(hw)] += l
        hw_sum[wi, hw_to_code[hw]] += l

    missing = int(np.sum(way_len_m <= 0))
    valid = way_len_m > 0
    way_center_y[valid] /= way_len_m[valid]
    way_center_x[valid] /= way_len_m[valid]

    # Normalize direction.
    dy = way_dir_y.copy()
    dx = way_dir_x.copy()
    dn = np.sqrt(dy * dy + dx * dx)
    ok = dn > 1e-6
    way_dir_y[ok] /= dn[ok]
    way_dir_x[ok] /= dn[ok]
    way_dir_y[~ok] = 0.0
    way_dir_x[~ok] = 0.0

    way_tier = np.argmax(tier_sum, axis=1).astype(np.int64, copy=False)
    way_hw_code = np.argmax(hw_sum, axis=1).astype(np.int64, copy=False)

    # ---------------- Semantic raster sampling (optional) ----------------
    semantic_keys = [x.strip() for x in str(cfg.semantic_channels or "").split(",") if str(x).strip()]
    semantic_keys = list(dict.fromkeys(semantic_keys))  # de-dup, keep order
    semantic_keys_ok = {"road_prob_major", "road_prob_minor", "road_prob_service", "entropy", "poi_total"}
    for k in semantic_keys:
        if k not in semantic_keys_ok:
            raise ValueError(f"Bad semantic channel: {k} (expected one of {sorted(semantic_keys_ok)})")

    way_semantic = None
    if semantic_keys:
        # Load rasters (H,W) float32
        def _load_raster(name: str, path: Path) -> np.ndarray:
            if not path.exists():
                raise FileNotFoundError(f"Missing {name} raster: {path}")
            a = np.load(path).astype(np.float32, copy=False)
            if a.ndim != 2 or a.shape != (H, W):
                raise ValueError(f"Bad {name} shape in {path}: {a.shape} (expected {(H, W)})")
            return a

        rasters: Dict[str, np.ndarray] = {}
        for k in semantic_keys:
            if k == "road_prob_major":
                rasters[k] = _load_raster(k, Path(semantic_dir) / "osm_road_prob_major.npy")
            elif k == "road_prob_minor":
                rasters[k] = _load_raster(k, Path(semantic_dir) / "osm_road_prob_minor.npy")
            elif k == "road_prob_service":
                rasters[k] = _load_raster(k, Path(semantic_dir) / "osm_road_prob_service.npy")
            elif k == "entropy":
                rasters[k] = _load_raster(k, Path(semantic_dir) / "landuse_entropy.npy")
            elif k == "poi_total":
                poi_paths = sorted(Path(semantic_dir).glob("poi_density_*.npy"))
                if not poi_paths:
                    raise FileNotFoundError(f"No poi_density_*.npy under: {semantic_dir}")
                poi_total = None
                for pp in poi_paths:
                    a = np.load(pp).astype(np.float32, copy=False)
                    if a.ndim != 2 or a.shape != (H, W):
                        raise ValueError(f"Bad poi_density shape in {pp}: {a.shape} (expected {(H, W)})")
                    poi_total = a if poi_total is None else (poi_total + a)
                assert poi_total is not None
                rasters[k] = poi_total.astype(np.float32, copy=False)

        C = int(len(semantic_keys))
        way_semantic = np.zeros((M, C), dtype=np.float32)
        # Sample nearest pixel at way center (y,x in grid coordinates).
        yy = np.rint(way_center_y).astype(np.int64, copy=False)
        xx = np.rint(way_center_x).astype(np.int64, copy=False)
        yy = np.clip(yy, 0, H - 1)
        xx = np.clip(xx, 0, W - 1)
        valid_way = way_len_m > 0
        for ci, k in enumerate(semantic_keys):
            r = rasters[k]
            v = r[yy, xx]
            # Keep missing ways at 0
            v = np.where(valid_way, v, 0.0).astype(np.float32, copy=False)
            way_semantic[:, ci] = v

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "build_way_features_from_osm_pbf",
        "inputs": {"osm_pbf": str(osm_pbf), "semantic_dir": str(semantic_dir), "way_routes_npz": str(way_routes_npz)},
        "config": {"road_types": str(cfg.road_types), "semantic_channels": str(cfg.semantic_channels)},
        "grid": {"H": int(grid.H), "W": int(grid.W), "bbox": grid.bbox.__dict__},
        "vocab": {"highway": hw_vocab},
        "stats": {
            "n_way_vocab": int(M),
            "n_missing_way": int(missing),
            "missing_frac": float(missing / max(1, M)),
            "way_len_m": {
                "p50": float(np.percentile(way_len_m[valid], 50) if np.any(valid) else float("nan")),
                "p90": float(np.percentile(way_len_m[valid], 90) if np.any(valid) else float("nan")),
            },
            "elapsed_s": float(time.time() - t0),
        },
    }
    if semantic_keys:
        assert way_semantic is not None
        meta["semantic"] = {
            "keys": list(semantic_keys),
            "sampling": "nearest_center",
            "stats": {
                k: {
                    "mean": float(np.mean(way_semantic[:, i])) if M else float("nan"),
                    "p90": float(np.percentile(way_semantic[:, i], 90)) if M else float("nan"),
                    "max": float(np.max(way_semantic[:, i])) if M else float("nan"),
                }
                for i, k in enumerate(semantic_keys)
            },
        }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_kwargs = dict(
        way_osm_id=way_osm_id.astype(np.int64, copy=False),
        way_len_m=way_len_m.astype(np.float32, copy=False),
        way_center_y=way_center_y.astype(np.float32, copy=False),
        way_center_x=way_center_x.astype(np.float32, copy=False),
        way_dir_y=way_dir_y.astype(np.float32, copy=False),
        way_dir_x=way_dir_x.astype(np.float32, copy=False),
        way_tier=way_tier.astype(np.int64, copy=False),
        way_highway_code=way_hw_code.astype(np.int64, copy=False),
        meta=meta,
    )
    if way_semantic is not None:
        out_kwargs["way_semantic"] = way_semantic.astype(np.float32, copy=False)
    np.savez_compressed(out_npz, **out_kwargs)
    return {"ok": True, "out_npz": str(out_npz), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build way-level features (tier/length/center/dir) from OSM .pbf via pyrosm.")
    p.add_argument("--osm_pbf", type=Path, required=True)
    p.add_argument("--semantic_dir", type=Path, required=True)
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--out_npz", type=Path, required=True)
    p.add_argument("--road_types", choices=["A", "B"], default="B")
    p.add_argument(
        "--semantic_channels",
        type=str,
        default="",
        help="Optional comma-separated semantic raster channels to sample at way center: road_prob_major,road_prob_minor,road_prob_service,entropy,poi_total. Empty=disable.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = build_way_features(
        osm_pbf=Path(args.osm_pbf),
        semantic_dir=Path(args.semantic_dir),
        way_routes_npz=Path(args.way_routes_npz),
        out_npz=Path(args.out_npz),
        cfg=Config(road_types=str(args.road_types), semantic_channels=str(args.semantic_channels)),
    )
    meta = report["meta"]
    st = meta["stats"]
    compact = {
        "ok": True,
        "out_npz": report["out_npz"],
        "n_way_vocab": int(st["n_way_vocab"]),
        "missing_frac": float(st["missing_frac"]),
        "way_len_p50_m": float(st["way_len_m"]["p50"]),
        "way_len_p90_m": float(st["way_len_m"]["p90"]),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
