from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _init_session(user_agent: str, *, max_retries: int) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=max(0, int(max_retries)),
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": user_agent})
    return session


def _download(session: requests.Session, url: str, out_path: Path, *, proxies: Optional[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=300, proxies=proxies) as r:
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
    ap.add_argument("--http_proxy", type=str, default=None, help="HTTP proxy (overrides env HTTP_PROXY)")
    ap.add_argument("--https_proxy", type=str, default=None, help="HTTPS proxy (overrides env HTTPS_PROXY)")
    ap.add_argument(
        "--user_agent",
        type=str,
        default="GeoExplicitSFM/TIGERDownloader (requests)",
        help="HTTP User-Agent",
    )
    ap.add_argument("--max_retries", type=int, default=3, help="HTTP retries for transient errors (default: 3)")
    args = ap.parse_args()

    year = int(args.year)
    state = str(args.state_fips).zfill(2)
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state}_tract.zip"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.out_dir / f"tl_{year}_{state}_tract.zip"

    http_proxy = args.http_proxy or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = args.https_proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    proxies = None
    if http_proxy or https_proxy:
        proxies = {}
        if http_proxy:
            proxies["http"] = http_proxy
        if https_proxy:
            proxies["https"] = https_proxy

    session = _init_session(str(args.user_agent), max_retries=int(args.max_retries))
    try:
        _download(session, url, zip_path, proxies=proxies)
    except Exception as e:
        print(
            json.dumps(
                {
                    "created_at": _now_iso(),
                    "url": url,
                    "zip_path": str(zip_path),
                    "error": f"{type(e).__name__}: {e}",
                    "proxy_env": {
                        "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
                        "HTTPS_PROXY": os.environ.get("HTTPS_PROXY"),
                    },
                    "proxy_args": {"http_proxy": args.http_proxy, "https_proxy": args.https_proxy},
                    "user_agent": str(args.user_agent),
                },
                indent=2,
            )
        )
        raise SystemExit(2) from e

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
