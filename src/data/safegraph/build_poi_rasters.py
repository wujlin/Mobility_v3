from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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


def _filter_base_places(
    base_csv: Path, bbox: BBox, *, chunksize: int = 500_000
) -> pd.DataFrame:
    usecols = ["placekey", "latitude", "longitude", "naics_code"]
    rows: List[pd.DataFrame] = []
    for chunk in pd.read_csv(base_csv, usecols=lambda c: c in usecols, chunksize=chunksize):
        chunk = chunk.rename(columns={"latitude": "lat", "longitude": "lon"})
        in_bbox = (chunk["lon"] >= bbox.min_lon) & (chunk["lon"] <= bbox.max_lon) & (chunk["lat"] >= bbox.min_lat) & (chunk["lat"] <= bbox.max_lat)
        sub = chunk.loc[in_bbox, ["placekey", "lat", "lon", "naics_code"]]
        if len(sub):
            rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["placekey", "lat", "lon", "naics_code"])
    return pd.concat(rows, ignore_index=True)


def _load_rich_for_keys(rich_csv: Path, keys: Set[str], *, chunksize: int = 500_000) -> pd.DataFrame:
    usecols = ["placekey", "opened_on", "closed_on"]
    rows: List[pd.DataFrame] = []
    for chunk in pd.read_csv(rich_csv, usecols=lambda c: c in usecols, chunksize=chunksize):
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
    ap.add_argument("--base_csv", type=Path, required=True, help="SafeGraph Places base CSV (must include placekey/lat/lon/naics_code)")
    ap.add_argument("--rich_csv", type=Path, default=None, help="SafeGraph Places rich CSV (optional; for opened_on/closed_on)")
    ap.add_argument("--vintage", type=str, default="2024-01", help="POI vintage YYYY-MM")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output dir (e.g. data/processed_worldtrace_detroit)")
    args = ap.parse_args()

    grid = _default_detroit_core_grid()
    bbox = grid.bbox
    cfg = POIConfig(vintage=str(args.vintage))

    base = _filter_base_places(args.base_csv, bbox)
    if len(base) == 0:
        raise SystemExit("No POI in bbox after filtering base CSV.")

    if args.rich_csv is not None:
        keys = set(base["placekey"].astype(str).to_list())
        rich = _load_rich_for_keys(args.rich_csv, keys)
        base = base.merge(rich, on="placekey", how="left")
    else:
        base["opened_on"] = None
        base["closed_on"] = None

    base["coarse_cat"] = base["naics_code"].apply(naics_to_coarse_cat)
    active = base.apply(lambda r: poi_active(cfg.vintage, r.get("opened_on"), r.get("closed_on")), axis=1)
    base = base.loc[active].reset_index(drop=True)
    if len(base) == 0:
        raise SystemExit("No active POI after applying opened_on/closed_on filter.")

    categories = ["food", "retail", "medical", "education", "transport", "leisure", "office", "industrial", "public", "other"]
    counts, dom, entropy = rasterize_pois(base, grid, categories)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, cat in enumerate(categories):
        np.save(args.out_dir / f"poi_density_{cat}.npy", counts[i])
    np.save(args.out_dir / "landuse_dom.npy", dom)
    np.save(args.out_dir / "landuse_entropy.npy", entropy)

    meta = {
        "grid": {"H": grid.H, "W": grid.W, "bbox": bbox.__dict__},
        "vintage": cfg.vintage,
        "categories": categories,
        "n_poi": int(len(base)),
    }
    (args.out_dir / "poi_raster_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

