from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pstats(s: pd.Series) -> Dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce")
    # Negative values are invalid for our covariates (counts/income/ratios) and
    # often represent missing sentinels in ACS.
    x = x.where(x >= 0)
    out: Dict[str, Any] = {"count": int(x.notna().sum()), "na_rate": float(x.isna().mean())}
    if out["count"] <= 0:
        out.update({"min": None, "p50": None, "p90": None, "mean": None})
        return out
    out.update(
        {
            "min": float(x.min()),
            "p50": float(x.quantile(0.5)),
            "p90": float(x.quantile(0.9)),
            "mean": float(x.mean()),
        }
    )
    return out


def _parse_bbox(vals: Optional[list[float]]) -> Optional[Tuple[float, float, float, float]]:
    if not vals:
        return None
    if len(vals) != 4:
        raise ValueError("--bbox requires 4 floats: min_lon min_lat max_lon max_lat")
    min_lon, min_lat, max_lon, max_lat = map(float, vals)
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError("Invalid bbox: expected min < max.")
    return (min_lon, min_lat, max_lon, max_lat)


def _load_tracts(
    *,
    tract_parquet: Optional[Path],
    tract_geojson: Optional[Path],
    bbox: Optional[Tuple[float, float, float, float]],
    out_crs: int,
) -> tuple[pd.DataFrame, Dict[str, Any], bool]:
    """
    Returns:
      - tracts dataframe (may include 'geometry' if geopandas available)
      - meta dict
      - has_geometry flag
    """
    if tract_parquet is None and tract_geojson is None:
        raise ValueError("Need one of --tract_parquet or --tract_geojson.")
    if tract_parquet is not None and tract_geojson is not None:
        raise ValueError("Provide only one of --tract_parquet or --tract_geojson.")

    try:
        import geopandas as gpd  # type: ignore
        from shapely.geometry import box  # type: ignore
    except ModuleNotFoundError:
        gpd = None
        box = None

    if gpd is None:
        # Fallback: no geometry operations. Use INTPTLAT/INTPTLON for bbox filtering if present.
        if tract_parquet is None:
            raise ModuleNotFoundError("geopandas is required for --tract_geojson input.")
        df = pd.read_parquet(tract_parquet)
        meta: Dict[str, Any] = {"loader": "pandas", "has_geometry": bool("geometry" in df.columns)}
        if bbox is not None and "INTPTLAT" in df.columns and "INTPTLON" in df.columns:
            min_lon, min_lat, max_lon, max_lat = bbox
            lat = pd.to_numeric(df["INTPTLAT"], errors="coerce")
            lon = pd.to_numeric(df["INTPTLON"], errors="coerce")
            keep = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
            meta["bbox_filter"] = {"mode": "intpt", "kept": int(keep.sum()), "total": int(len(df))}
            df = df.loc[keep].reset_index(drop=True)
        return df, meta, bool("geometry" in df.columns)

    # Geo path
    if tract_parquet is not None:
        gdf = gpd.read_parquet(tract_parquet)
        src_path = tract_parquet
        src_kind = "parquet"
    else:
        gdf = gpd.read_file(tract_geojson)  # type: ignore[arg-type]
        src_path = tract_geojson  # type: ignore[assignment]
        src_kind = "geojson"

    meta = {"loader": "geopandas", "src_kind": src_kind, "src_path": str(src_path), "crs_in": str(gdf.crs)}
    if gdf.crs is None:
        # Be conservative: TIGER is usually NAD83 (EPSG:4269). If CRS missing, assume 4326.
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
        meta["crs_inferred"] = "EPSG:4326"

    # Normalize CRS for bbox and downstream joins/plots
    try:
        gdf = gdf.to_crs(epsg=int(out_crs))
    except Exception as e:
        meta["crs_convert_error"] = f"{type(e).__name__}: {e}"

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
        keep = gdf.geometry.intersects(bbox_geom)
        meta["bbox_filter"] = {"mode": "intersects", "kept": int(keep.sum()), "total": int(len(gdf))}
        gdf = gdf.loc[keep].reset_index(drop=True)

    return pd.DataFrame(gdf), meta, True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Join ACS tract indicators with TIGER tract geometry (optionally clip to bbox) and write (Geo)Parquet."
    )
    ap.add_argument("--acs_csv", type=Path, required=True, help="ACS tract CSV from download_acs_tract.py")
    ap.add_argument("--tract_parquet", type=Path, default=None, help="TIGER tract GeoParquet (preferred)")
    ap.add_argument("--tract_geojson", type=Path, default=None, help="TIGER tract GeoJSON (requires geopandas)")
    ap.add_argument("--bbox", type=float, nargs=4, default=None, help="min_lon min_lat max_lon max_lat (optional)")
    ap.add_argument("--out_parquet", type=Path, required=True, help="Output Parquet/GeoParquet path")
    ap.add_argument("--out_meta", type=Path, default=None, help="Meta JSON (default: <out_parquet>.meta.json)")
    ap.add_argument("--out_crs", type=int, default=4326, help="Output CRS EPSG for geometry (default: 4326)")
    args = ap.parse_args()

    bbox = _parse_bbox(list(args.bbox) if args.bbox is not None else None)

    acs = pd.read_csv(args.acs_csv)
    if "geoid" not in acs.columns:
        raise SystemExit("ACS CSV missing 'geoid' column. Expect output from download_acs_tract.py.")
    acs["geoid"] = acs["geoid"].astype(str).str.zfill(11)

    tracts, tr_meta, has_geometry = _load_tracts(
        tract_parquet=args.tract_parquet,
        tract_geojson=args.tract_geojson,
        bbox=bbox,
        out_crs=int(args.out_crs),
    )

    if "GEOID" not in tracts.columns:
        raise SystemExit("Tract table missing 'GEOID' column.")
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)

    # One-to-one join expected for a single-year ACS + TIGER.
    out = tracts.merge(acs, left_on="GEOID", right_on="geoid", how="left", suffixes=("", "_acs"))
    missing = int(out["geoid"].isna().sum())

    # Clean negative ACS sentinels inside the merged table (defensive; should also be handled upstream).
    for col in ("B25002_001E", "B25002_003E", "B01003_001E", "B19013_001E"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out.loc[out[col] < 0, col] = None
    if "vacancy_rate" in out.columns:
        out["vacancy_rate"] = pd.to_numeric(out["vacancy_rate"], errors="coerce")
        out.loc[out["vacancy_rate"] < 0, "vacancy_rate"] = None

    # Write parquet
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    write_meta: Dict[str, Any] = {}
    if has_geometry:
        try:
            import geopandas as gpd  # type: ignore

            gdf = gpd.GeoDataFrame(out, geometry="geometry", crs=f"EPSG:{int(args.out_crs)}")
            gdf.to_parquet(args.out_parquet, index=False)
            write_meta["writer"] = "geopandas"
        except Exception as e:
            # Fallback to plain parquet without guaranteeing geometry semantics.
            out.to_parquet(args.out_parquet, index=False)
            write_meta["writer"] = "pandas_fallback"
            write_meta["geopandas_write_error"] = f"{type(e).__name__}: {e}"
    else:
        out.to_parquet(args.out_parquet, index=False)
        write_meta["writer"] = "pandas"

    stats = {
        "vacancy_rate": _pstats(out.get("vacancy_rate", pd.Series(dtype="float64"))),
        "population": _pstats(out.get("B01003_001E", pd.Series(dtype="float64"))),
        "median_income": _pstats(out.get("B19013_001E", pd.Series(dtype="float64"))),
    }

    out_meta = args.out_meta or args.out_parquet.with_suffix(args.out_parquet.suffix + ".meta.json")
    meta = {
        "created_at": _now_iso(),
        "acs_csv": str(args.acs_csv),
        "tract_parquet": str(args.tract_parquet) if args.tract_parquet is not None else None,
        "tract_geojson": str(args.tract_geojson) if args.tract_geojson is not None else None,
        "bbox": {"min_lon": bbox[0], "min_lat": bbox[1], "max_lon": bbox[2], "max_lat": bbox[3]} if bbox else None,
        "out_crs": int(args.out_crs),
        "rows_out": int(len(out)),
        "missing_acs_rows": missing,
        "has_geometry": bool(has_geometry),
        "tract_loader": tr_meta,
        "writer": write_meta,
        "stats": stats,
        "out_parquet": str(args.out_parquet),
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"out_parquet": str(args.out_parquet), "out_meta": str(out_meta), "rows": int(len(out)), "missing_acs_rows": missing}, indent=2))


if __name__ == "__main__":
    main()
