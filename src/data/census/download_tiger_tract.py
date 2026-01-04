from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

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
    session.headers.update(
        {
            "Accept": "application/zip,application/octet-stream,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
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


def _http_json(session: requests.Session, url: str, *, proxies: Optional[dict], params: Optional[dict] = None) -> Dict[str, Any]:
    res = session.get(url, params=params, timeout=120, proxies=proxies)
    res.raise_for_status()
    data = res.json()
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected JSON response (not a dict).")
    return data


def _tigerweb_fetch_tracts_geojson(
    session: requests.Session,
    *,
    year: int,
    state_fips: str,
    out_geojson: Path,
    proxies: Optional[dict],
) -> Dict[str, Any]:
    """
    Fallback: query TIGERweb ArcGIS REST service and save a GeoJSON FeatureCollection.
    """
    service = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer"
    svc = _http_json(session, service, proxies=proxies, params={"f": "pjson"})
    layers = svc.get("layers") or []
    if not isinstance(layers, list) or not layers:
        raise RuntimeError("TIGERweb service has no layers list.")

    def score(name: str) -> int:
        s = (name or "").lower()
        if "tract" not in s:
            return -999
        # prefer explicit "census tract(s)" over other layers
        bonus = 0
        if "census" in s:
            bonus += 10
        if "block" in s:
            bonus -= 10
        return bonus

    tract_layer_id = None
    best = -10_000
    for it in layers:
        if not isinstance(it, dict):
            continue
        lid = it.get("id")
        name = str(it.get("name", ""))
        sc = score(name)
        if sc > best and lid is not None:
            best = sc
            tract_layer_id = int(lid)

    if tract_layer_id is None:
        raise RuntimeError("Failed to locate a tract layer in TIGERweb service metadata.")

    layer_url = f"{service}/{tract_layer_id}"
    layer = _http_json(session, layer_url, proxies=proxies, params={"f": "pjson"})

    fields = layer.get("fields") or []
    field_names = [str(f.get("name", "")) for f in fields if isinstance(f, dict)]
    field_set = {n.upper() for n in field_names}

    # Pick a state FIPS field name
    candidates = ["STATE", "STATEFP", "STATEFP20", "STATEFP10", "STATEFP00"]
    state_field = None
    for c in candidates:
        if c in field_set:
            # recover original casing if possible
            for n in field_names:
                if n.upper() == c:
                    state_field = n
                    break
            if state_field is not None:
                break
    if state_field is None:
        raise RuntimeError(f"Failed to find a state field in TIGERweb layer fields: {field_names[:20]}")

    where = f"{state_field}='{str(state_fips).zfill(2)}'"

    q_url = f"{layer_url}/query"
    count_obj = _http_json(
        session,
        q_url,
        proxies=proxies,
        params={"where": where, "returnCountOnly": "true", "f": "pjson"},
    )
    total = int(count_obj.get("count") or 0)
    if total <= 0:
        raise RuntimeError("TIGERweb returned 0 features for the given state.")

    max_rc = int(layer.get("maxRecordCount") or svc.get("maxRecordCount") or 2000)
    page_size = min(2000, max_rc) if max_rc > 0 else 2000

    fc: Dict[str, Any] = {"type": "FeatureCollection", "features": []}
    feats: list = fc["features"]
    for offset in range(0, total, page_size):
        params = {
            "where": where,
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
            "resultOffset": str(offset),
            "resultRecordCount": str(page_size),
        }
        res = session.get(q_url, params=params, timeout=300, proxies=proxies)
        res.raise_for_status()
        data = res.json()
        if not isinstance(data, dict) or "features" not in data:
            raise RuntimeError("Unexpected TIGERweb geojson page response.")
        page_feats = data.get("features") or []
        if isinstance(page_feats, list):
            feats.extend(page_feats)

    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_geojson.write_text(json.dumps(fc), encoding="utf-8")

    return {
        "method": "tigerweb_geojson",
        "service": service,
        "layer_id": tract_layer_id,
        "state_field": state_field,
        "where": where,
        "features": int(len(feats)),
        "page_size": page_size,
        "out_geojson": str(out_geojson),
    }


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
        default="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        help="HTTP User-Agent",
    )
    ap.add_argument("--max_retries", type=int, default=3, help="HTTP retries for transient errors (default: 3)")
    ap.add_argument(
        "--fallback",
        choices=["none", "tigerweb"],
        default="tigerweb",
        help="If TIGER zip download fails (e.g., 403), fallback strategy (default: tigerweb).",
    )
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
    download_method = "tiger_zip"
    tigerweb_meta: Optional[Dict[str, Any]] = None
    zip_ok = False
    err_obj: Optional[Dict[str, Any]] = None
    try:
        _download(session, url, zip_path, proxies=proxies)
        zip_ok = True
    except Exception as e:
        # Record error first; then try fallback if enabled.
        err_obj = {
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
        }
        if str(args.fallback) == "tigerweb":
            try:
                out_geojson = args.out_dir / f"tl_{year}_{state}_tract.geojson"
                tigerweb_meta = _tigerweb_fetch_tracts_geojson(
                    session,
                    year=year,
                    state_fips=state,
                    out_geojson=out_geojson,
                    proxies=proxies,
                )
                download_method = "tigerweb_geojson"
            except Exception as e2:
                err_obj["fallback_error"] = f"{type(e2).__name__}: {e2}"
                print(json.dumps(err_obj, indent=2))
                raise SystemExit(2) from e2
        else:
            print(json.dumps(err_obj, indent=2))
            raise SystemExit(2) from e

    extract_dir = None
    shp = None
    conv_status = None
    out_parquet = None
    geojson_path = None
    if zip_ok:
        extract_dir = args.out_dir / f"tl_{year}_{state}_tract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        shp = next(iter(extract_dir.glob("*.shp")), None)
        if args.convert_geoparquet and shp is not None:
            out_parquet = args.out_dir / f"tl_{year}_{state}_tract.parquet"
            conv_status = _try_convert_to_geoparquet(shp, out_parquet)
    else:
        if tigerweb_meta is not None:
            geojson_path = Path(tigerweb_meta["out_geojson"])
            if args.convert_geoparquet:
                try:
                    import geopandas as gpd  # type: ignore
                except ModuleNotFoundError:
                    conv_status = "geopandas_not_installed"
                else:
                    out_parquet = args.out_dir / f"tl_{year}_{state}_tract.parquet"
                    gdf = gpd.read_file(geojson_path)
                    gdf.to_parquet(out_parquet, index=False)

    meta = {
        "created_at": _now_iso(),
        "download_method": download_method,
        "url": url,
        "zip_path": str(zip_path),
        "extract_dir": str(extract_dir) if extract_dir is not None else None,
        "shp_path": str(shp) if shp is not None else None,
        "geojson_path": str(geojson_path) if geojson_path is not None else None,
        "converted_geoparquet": str(out_parquet) if out_parquet is not None and conv_status is None else None,
        "convert_status": conv_status,
        "tigerweb": tigerweb_meta,
        "download_error": err_obj,
    }
    (args.out_dir / "tiger_tract_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
