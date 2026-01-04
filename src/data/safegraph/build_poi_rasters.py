from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.utils.geo_grid import BBox, GridSpec


@dataclass(frozen=True)
class POIConfig:
    vintage: str = "2024-01"  # YYYY-MM


def _default_detroit_core_grid() -> GridSpec:
    bbox = BBox(min_lon=-83.25, max_lon=-82.95, min_lat=42.25, max_lat=42.50)
    return GridSpec(H=1024, W=1024, bbox=bbox)


def _parse_yyyymm(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = str(s).strip()
    if len(s) != 7 or s[4] != "-":
        return None
    return s


def poi_active(vintage: str, opened_on: Optional[str], closed_on: Optional[str]) -> bool:
    o = _parse_yyyymm(opened_on)
    c = _parse_yyyymm(closed_on)
    if c == "1900-01":  # SafeGraph special: permanent closed but unknown month
        return False
    if o is not None and o > vintage:
        return False
    if c is not None and c <= vintage:
        return False
    return True


def naics_to_coarse_cat(naics: Optional[object]) -> str:
    if naics is None or (isinstance(naics, float) and np.isnan(naics)):
        return "other"
    s = str(int(naics)) if str(naics).strip().isdigit() else str(naics).strip()
    if len(s) < 2 or not s[:2].isdigit():
        return "other"
    n2 = int(s[:2])
    if n2 == 72:
        return "food"
    if n2 in {44, 45}:
        return "retail"
    if n2 == 62:
        return "medical"
    if n2 == 61:
        return "education"
    if n2 in {48, 49}:
        return "transport"
    if n2 == 71:
        return "leisure"
    if n2 in {52, 53, 54, 55, 56}:
        return "office"
    if n2 in {31, 32, 33}:
        return "industrial"
    if n2 == 92:
        return "public"
    return "other"


def _canon_col(name: str) -> str:
    return str(name).strip().strip('"').strip().lower()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_canon_col(c) for c in df.columns]
    return df


def _read_poi_chunks(
    base_paths: Sequence[Path],
    bbox: BBox,
    *,
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    need = {"placekey", "latitude", "longitude", "naics_code", "opened_on", "closed_on"}
    for csv_path in base_paths:
        for chunk in pd.read_csv(
            csv_path,
            chunksize=chunksize,
            usecols=lambda c: _canon_col(c) in need,
        ):
            chunk = _normalize_columns(chunk)
            if "placekey" not in chunk.columns or "latitude" not in chunk.columns or "longitude" not in chunk.columns:
                continue
            # normalize coords
            chunk = chunk.rename(columns={"latitude": "lat", "longitude": "lon"})
            lat = pd.to_numeric(chunk["lat"], errors="coerce")
            lon = pd.to_numeric(chunk["lon"], errors="coerce")
            in_bbox = (lon >= bbox.min_lon) & (lon <= bbox.max_lon) & (lat >= bbox.min_lat) & (lat <= bbox.max_lat)
            sub = chunk.loc[in_bbox].copy()
            if len(sub):
                sub["lat"] = lat[in_bbox]
                sub["lon"] = lon[in_bbox]
                yield sub


def _load_rich_for_keys(rich_csv: Path, keys: Set[str], *, chunksize: int = 500_000) -> pd.DataFrame:
    usecols = {"placekey", "opened_on", "closed_on"}
    rows: List[pd.DataFrame] = []
    for chunk in pd.read_csv(rich_csv, usecols=lambda c: _canon_col(c) in usecols, chunksize=chunksize):
        chunk = _normalize_columns(chunk)
        if "placekey" not in chunk.columns:
            continue
        sub = chunk[chunk["placekey"].astype(str).isin(keys)]
        if len(sub):
            rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["placekey", "opened_on", "closed_on"])
    return pd.concat(rows, ignore_index=True).drop_duplicates(subset=["placekey"], keep="last")


def rasterize_pois(df: pd.DataFrame, grid: GridSpec, categories: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    C = len(categories)
    counts = np.zeros((C, grid.H, grid.W), dtype=np.float32)

    lat = df["lat"].to_numpy(np.float64)
    lon = df["lon"].to_numpy(np.float64)
    y, x = grid.latlon_to_yx(lat, lon)
    inb = grid.in_bounds(y, x)
    y = y[inb]
    x = x[inb]
    cats = df.loc[inb, "coarse_cat"].astype(str).to_list()

    for yy, xx, cat in zip(y.tolist(), x.tolist(), cats):
        idx = cat_to_idx.get(cat, cat_to_idx["other"])
        counts[idx, yy, xx] += 1.0

    total = counts.sum(axis=0)  # (H,W)
    dom = np.full((grid.H, grid.W), fill_value=-1, dtype=np.int16)
    entropy = np.zeros((grid.H, grid.W), dtype=np.float32)

    nonzero = total > 0
    if np.any(nonzero):
        dom[nonzero] = np.argmax(counts[:, nonzero], axis=0).astype(np.int16)
        p = counts[:, nonzero] / (total[nonzero][None, :] + 1e-9)
        ent = -np.sum(p * np.log(p + 1e-9), axis=0)
        entropy[nonzero] = (ent / np.log(C)).astype(np.float32)  # normalized to [0,1]

    return counts, dom, entropy


def main() -> None:
    ap = argparse.ArgumentParser(description="Rasterize SafeGraph POI to Detroit core grid (WGS84).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--base_csv", type=Path, default=None, help="SafeGraph Places CSV (single file)")
    src.add_argument("--base_dir", type=Path, default=None, help="Directory containing many SafeGraph CSV shards")
    ap.add_argument("--base_glob", type=str, default="*.csv", help="When using --base_dir, glob pattern for shards")
    ap.add_argument("--rich_csv", type=Path, default=None, help="Optional separate rich CSV (opened_on/closed_on)")
    ap.add_argument("--vintage", type=str, default="2024-01", help="POI vintage YYYY-MM")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output dir (e.g. data/processed_worldtrace_detroit)")
    ap.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"), help="Override bbox (WGS84 lon/lat)")
    ap.add_argument("--grid_h", type=int, default=None, help="Override grid H (default=1024 for Detroit core)")
    ap.add_argument("--grid_w", type=int, default=None, help="Override grid W (default=1024 for Detroit core)")
    ap.add_argument("--chunksize", type=int, default=500_000, help="CSV read chunksize")
    args = ap.parse_args()

    if args.bbox is None and args.grid_h is None and args.grid_w is None:
        grid = _default_detroit_core_grid()
    else:
        bbox_vals = args.bbox or [-83.25, 42.25, -82.95, 42.50]
        bbox = BBox(min_lon=float(bbox_vals[0]), min_lat=float(bbox_vals[1]), max_lon=float(bbox_vals[2]), max_lat=float(bbox_vals[3]))
        H = int(args.grid_h or 1024)
        W = int(args.grid_w or 1024)
        grid = GridSpec(H=H, W=W, bbox=bbox)
    bbox = grid.bbox
    cfg = POIConfig(vintage=str(args.vintage))

    if args.base_dir is not None:
        base_paths = sorted(args.base_dir.glob(args.base_glob))
        if not base_paths:
            raise SystemExit(f"No CSV shards matched: {args.base_dir}/{args.base_glob}")
    else:
        assert args.base_csv is not None
        base_paths = [args.base_csv]

    categories = ["food", "retail", "medical", "education", "transport", "leisure", "office", "industrial", "public", "other"]
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    counts = np.zeros((len(categories), grid.H, grid.W), dtype=np.float32)
    n_poi_total = 0
    n_poi_active = 0

    for chunk in _read_poi_chunks(base_paths, bbox, chunksize=int(args.chunksize)):
        # Optional enrichment from separate rich CSV is supported only when base has placekey columns (slow but bounded by bbox filter).
        if args.rich_csv is not None and ("opened_on" not in chunk.columns or "closed_on" not in chunk.columns):
            keys = set(chunk["placekey"].astype(str).to_list())
            rich = _load_rich_for_keys(args.rich_csv, keys, chunksize=int(args.chunksize))
            if len(rich):
                chunk = chunk.merge(rich, on="placekey", how="left")

        if "opened_on" not in chunk.columns:
            chunk["opened_on"] = None
        if "closed_on" not in chunk.columns:
            chunk["closed_on"] = None

        n_poi_total += int(len(chunk))
        chunk["coarse_cat"] = chunk.get("naics_code").apply(naics_to_coarse_cat) if "naics_code" in chunk.columns else "other"
        active = chunk.apply(lambda r: poi_active(cfg.vintage, r.get("opened_on"), r.get("closed_on")), axis=1)
        chunk = chunk.loc[active].reset_index(drop=True)
        if len(chunk) == 0:
            continue
        n_poi_active += int(len(chunk))

        lat = chunk["lat"].to_numpy(np.float64)
        lon = chunk["lon"].to_numpy(np.float64)
        y, x = grid.latlon_to_yx(lat, lon)
        inb = grid.in_bounds(y, x)
        if not np.any(inb):
            continue
        y = y[inb].astype(np.int64)
        x = x[inb].astype(np.int64)
        cats = chunk.loc[inb, "coarse_cat"].astype(str).to_list()
        idx = np.asarray([cat_to_idx.get(c, cat_to_idx["other"]) for c in cats], dtype=np.int64)
        np.add.at(counts, (idx, y, x), 1.0)

    if n_poi_active == 0:
        raise SystemExit("No active POI in bbox after filtering. Check bbox/vintage/inputs.")

    # derive dom/entropy
    total = counts.sum(axis=0)  # (H,W)
    dom = np.full((grid.H, grid.W), fill_value=-1, dtype=np.int16)
    entropy = np.zeros((grid.H, grid.W), dtype=np.float32)

    nonzero = total > 0
    if np.any(nonzero):
        dom[nonzero] = np.argmax(counts[:, nonzero], axis=0).astype(np.int16)
        p = counts[:, nonzero] / (total[nonzero][None, :] + 1e-9)
        ent = -np.sum(p * np.log(p + 1e-9), axis=0)
        entropy[nonzero] = (ent / np.log(len(categories))).astype(np.float32)  # normalized to [0,1]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, cat in enumerate(categories):
        np.save(args.out_dir / f"poi_density_{cat}.npy", counts[i])
    np.save(args.out_dir / "landuse_dom.npy", dom)
    np.save(args.out_dir / "landuse_entropy.npy", entropy)

    meta = {
        "grid": {"H": grid.H, "W": grid.W, "bbox": bbox.__dict__},
        "vintage": cfg.vintage,
        "categories": categories,
        "n_poi_total_in_bbox": int(n_poi_total),
        "n_poi_active": int(n_poi_active),
        "src": {
            "base_paths_n": int(len(base_paths)),
            "base_paths_sample": [str(p) for p in base_paths[:3]],
            "rich_csv": str(args.rich_csv) if args.rich_csv is not None else None,
        },
    }
    (args.out_dir / "poi_raster_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
