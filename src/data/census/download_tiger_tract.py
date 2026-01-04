from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(out_path)


def _try_convert_to_geoparquet(shp_path: Path, out_parquet: Path) -> Optional[str]:
    try:
        import geopandas as gpd  # type: ignore
    except ModuleNotFoundError:
        return "geopandas_not_installed"

    gdf = gpd.read_file(shp_path)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_parquet, index=False)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Download TIGER/Line tract boundaries (zip) and unzip locally.")
    ap.add_argument("--year", type=int, required=True, help="TIGER year (e.g., 2023)")
    ap.add_argument("--state_fips", type=str, default="26", help="State FIPS (default: 26=MI)")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output directory")
    ap.add_argument("--convert_geoparquet", action="store_true", help="If geopandas installed, convert shapefile to geoparquet")
    args = ap.parse_args()

    year = int(args.year)
    state = str(args.state_fips).zfill(2)
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state}_tract.zip"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.out_dir / f"tl_{year}_{state}_tract.zip"
    _download(url, zip_path)

    extract_dir = args.out_dir / f"tl_{year}_{state}_tract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    shp = next(iter(extract_dir.glob("*.shp")), None)
    conv_status = None
    out_parquet = None
    if args.convert_geoparquet and shp is not None:
        out_parquet = args.out_dir / f"tl_{year}_{state}_tract.parquet"
        conv_status = _try_convert_to_geoparquet(shp, out_parquet)

    meta = {
        "created_at": _now_iso(),
        "url": url,
        "zip_path": str(zip_path),
        "extract_dir": str(extract_dir),
        "shp_path": str(shp) if shp is not None else None,
        "converted_geoparquet": str(out_parquet) if out_parquet is not None and conv_status is None else None,
        "convert_status": conv_status,
    }
    (args.out_dir / "tiger_tract_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

