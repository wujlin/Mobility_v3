from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CONFIG_URL_DEFAULT = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
METADATA_URL_TPL_DEFAULT = "https://s3-us-west-2.amazonaws.com/wayback-tilemap-console/metadata/edge/tile/{z}/{y}/{x}.json"


@dataclass(frozen=True)
class ReleaseInfo:
    release_id: int
    release_date: str
    item_url_template: str


def _parse_iso_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    # common: YYYY-MM-DD
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def _init_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "GeoExplicitSFM/WaybackScraper (requests)",
        }
    )
    return session


def lon_lat_to_tile_xy(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    n = 2.0 ** zoom
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    lat_rad = math.radians(lat)
    y = int(math.floor((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n))
    return x, y


def bbox_to_tile_range(bbox: Tuple[float, float, float, float], zoom: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    bbox = (min_lon, min_lat, max_lon, max_lat)
    Returns: ((x_min, x_max), (y_min, y_max)) inclusive.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    x1, y1 = lon_lat_to_tile_xy(min_lon, max_lat, zoom)  # NW
    x2, y2 = lon_lat_to_tile_xy(max_lon, min_lat, zoom)  # SE
    return (min(x1, x2), max(x1, x2)), (min(y1, y2), max(y1, y2))


def load_release_map(session: requests.Session, config_url: str) -> Dict[int, ReleaseInfo]:
    res = session.get(config_url, timeout=20)
    res.raise_for_status()
    data = res.json()
    # Old schema: {"archive": [{"releaseNum":..,"releaseDate":..,"itemURL":..}, ...]}
    if isinstance(data, dict) and isinstance(data.get("archive"), list):
        archive = data.get("archive", [])
        release_map: Dict[int, ReleaseInfo] = {}
        for item in archive:
            try:
                rid = int(item["releaseNum"])
            except Exception:
                continue
            release_map[rid] = ReleaseInfo(
                release_id=rid,
                release_date=str(item.get("releaseDate", "")),
                item_url_template=str(item.get("itemURL", "")),
            )
        return release_map

    # New schema (observed 2026-01): top-level dict keyed by release_id (string),
    # value is either:
    # - dict containing releaseDate/itemURL (or similar), or
    # - direct URL template string.
    def _iter_strings(obj) -> List[str]:
        out: List[str] = []
        if isinstance(obj, str):
            out.append(obj)
        elif isinstance(obj, dict):
            for vv in obj.values():
                out.extend(_iter_strings(vv))
        elif isinstance(obj, list):
            for vv in obj:
                out.extend(_iter_strings(vv))
        return out

    def _looks_like_tile_template(s: str) -> bool:
        s2 = s.lower()
        if not (s2.startswith("http://") or s2.startswith("https://")):
            return False
        # ArcGIS tile endpoints typically contain "tile" + z/x/y placeholders.
        if "tile" not in s2:
            return False
        if any(tok in s2 for tok in ("{level}", "{z}", "{zoom}", "{col}", "{x}", "{row}", "{y}")):
            return True
        # Some templates may already have concrete z/x/y for a sample; keep as fallback.
        return "/tile/" in s2 or "/mapserver/tile/" in s2

    def _pick_first_template(obj) -> str:
        candidates = [s for s in _iter_strings(obj) if _looks_like_tile_template(s)]
        # prefer "most structured" templates (more placeholders / longer)
        def score(s: str) -> Tuple[int, int]:
            num_placeholders = sum(s.count(tok) for tok in ("{level}", "{z}", "{zoom}", "{col}", "{x}", "{row}", "{y}"))
            return (num_placeholders, len(s))

        candidates.sort(key=score, reverse=True)
        return candidates[0] if candidates else ""

    def _pick_first_date(obj) -> str:
        # Look for any ISO-like date string in nested values.
        for s in _iter_strings(obj):
            d = _parse_iso_date(s)
            if d is not None:
                return d.isoformat()
        return ""

    release_map: Dict[int, ReleaseInfo] = {}
    if not isinstance(data, dict):
        return release_map

    for k, v in data.items():
        try:
            rid = int(k)
        except Exception:
            continue

        if isinstance(v, str):
            item_url = v
            rel_date = ""
        elif isinstance(v, dict):
            rel_date = str(
                v.get("releaseDate")
                or v.get("date")
                or v.get("d")
                or ""
            )
            item_url = str(
                v.get("itemURL")
                or v.get("itemUrl")
                or v.get("url")
                or v.get("template")
                or ""
            )
            if not item_url:
                item_url = _pick_first_template(v)
            if not rel_date:
                rel_date = _pick_first_date(v)
        else:
            continue

        if not item_url:
            continue
        release_map[rid] = ReleaseInfo(release_id=rid, release_date=rel_date, item_url_template=item_url)
    return release_map


def get_tile_changes(
    session: requests.Session,
    release_map: Dict[int, ReleaseInfo],
    metadata_url_tpl: str,
    *,
    x: int,
    y: int,
    z: int,
) -> List[int]:
    url = metadata_url_tpl.format(z=z, y=y, x=x)
    res = session.get(url, timeout=10)
    if res.status_code == 404:
        # Distinguish "no metadata for this tile" vs "endpoint/bucket no longer exists".
        # If the bucket is gone, returning [] would silently produce 0 tasks and mislead debugging.
        if b"NoSuchBucket" in res.content:
            raise RuntimeError(
                "Wayback metadata endpoint is invalid (S3 NoSuchBucket). "
                "Please switch to --mode fixed_releases (use --list_releases to pick ids), "
                "or update --metadata_url_tpl to the new endpoint once discovered."
            )
        return []
    res.raise_for_status()
    arr = res.json()
    rids: List[int] = []
    for it in arr:
        rid = it.get("r")
        if rid is None:
            continue
        try:
            rid_int = int(rid)
        except (TypeError, ValueError):
            continue
        if rid_int in release_map:
            rids.append(rid_int)
    return rids


def _build_tile_url(info: ReleaseInfo, *, x: int, y: int, z: int) -> str:
    # itemURL commonly uses placeholders like:
    # - {level}/{col}/{row} (legacy)
    # - {z}/{x}/{y} (common web mercator tile naming)
    tpl = info.item_url_template
    for k in ("{level}", "{z}", "{zoom}"):
        tpl = tpl.replace(k, str(z))
    for k in ("{col}", "{x}"):
        tpl = tpl.replace(k, str(x))
    for k in ("{row}", "{y}"):
        tpl = tpl.replace(k, str(y))
    return tpl


def _download_one(session: requests.Session, url: str, out_path: Path) -> str:
    if out_path.exists():
        return "SKIP"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        res = session.get(url, timeout=30)
        if res.status_code != 200:
            return f"HTTP_{res.status_code}"
        tmp.write_bytes(res.content)
        tmp.replace(out_path)
        return "OK"
    except Exception as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return f"ERR_{type(e).__name__}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download ArcGIS Wayback imagery tiles for a bbox (metadata-first).")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output directory")
    ap.add_argument("--bbox", type=float, nargs=4, default=[-83.25, 42.25, -82.95, 42.50], metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    ap.add_argument("--zoom", type=int, default=16)
    ap.add_argument("--max_threads", type=int, default=16)
    ap.add_argument("--max_tiles", type=int, default=0, help="Debug: limit number of spatial tiles scanned (0=no limit)")
    ap.add_argument("--dry_run", action="store_true", help="Only scan metadata and report task count (no downloads)")
    ap.add_argument(
        "--log_every",
        type=int,
        default=0,
        help="If >0, print a JSON progress line every N completed downloads (to stderr).",
    )
    ap.add_argument("--config_url", type=str, default=CONFIG_URL_DEFAULT, help="Wayback config URL")
    ap.add_argument("--metadata_url_tpl", type=str, default=METADATA_URL_TPL_DEFAULT, help="Wayback metadata URL template with {z}/{y}/{x}")
    ap.add_argument("--mode", choices=["metadata", "fixed_releases"], default="metadata", help="Download mode")
    ap.add_argument("--release_ids", type=int, nargs="*", default=None, help="Release IDs for fixed_releases mode")
    ap.add_argument("--list_releases", type=int, default=0, help="List first N releases (JSON) then exit")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    bbox = tuple(float(v) for v in args.bbox)
    zoom = int(args.zoom)

    session = _init_session()
    release_map = load_release_map(session, str(args.config_url))
    if not release_map:
        raise SystemExit(
            "Wayback config loaded but no releases parsed. "
            "This usually means config schema changed. "
            "Please inspect the downloaded JSON and update the parser."
        )

    if args.list_releases:
        n = int(args.list_releases)
        items = list(release_map.values())
        items.sort(key=lambda r: (_parse_iso_date(r.release_date) or date.min, r.release_id))
        out = {
            "num_releases_total": len(release_map),
            "has_release_date": sum(1 for r in release_map.values() if bool(_parse_iso_date(r.release_date))),
            "sample": [
                {
                    "release_id": r.release_id,
                    "release_date": r.release_date,
                    "item_url_template_head": r.item_url_template[:120],
                }
                for r in items[:n]
            ],
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    (x_min, x_max), (y_min, y_max) = bbox_to_tile_range(bbox, zoom)
    total_tiles_geo = (x_max - x_min + 1) * (y_max - y_min + 1)

    # stage 1: build download queue
    tasks: List[Tuple[str, Path]] = []
    scanned_tiles = 0
    if args.mode == "metadata":
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                scanned_tiles += 1
                if args.max_tiles and scanned_tiles > int(args.max_tiles):
                    break
                rids = get_tile_changes(session, release_map, str(args.metadata_url_tpl), x=x, y=y, z=zoom)
                if not rids:
                    continue
                for rid in rids:
                    info = release_map[rid]
                    url = _build_tile_url(info, x=x, y=y, z=zoom)
                    out_path = out_dir / f"z{zoom}" / f"{zoom}_{x}_{y}" / f"rid_{rid}.jpg"
                    tasks.append((url, out_path))
            if args.max_tiles and scanned_tiles > int(args.max_tiles):
                break
    else:
        release_ids = args.release_ids or []
        if not release_ids:
            raise SystemExit("fixed_releases mode requires --release_ids (use --list_releases to inspect available ids).")
        missing = [rid for rid in release_ids if rid not in release_map]
        if missing:
            raise SystemExit(f"Unknown release_ids (not in config): {missing[:20]}")

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                scanned_tiles += 1
                if args.max_tiles and scanned_tiles > int(args.max_tiles):
                    break
                for rid in release_ids:
                    info = release_map[rid]
                    url = _build_tile_url(info, x=x, y=y, z=zoom)
                    out_path = out_dir / f"z{zoom}" / f"{zoom}_{x}_{y}" / f"rid_{rid}.jpg"
                    tasks.append((url, out_path))
            if args.max_tiles and scanned_tiles > int(args.max_tiles):
                break

    meta = {
        "bbox": {"min_lon": bbox[0], "min_lat": bbox[1], "max_lon": bbox[2], "max_lat": bbox[3]},
        "zoom": zoom,
        "tile_range": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "total_tiles_geo": total_tiles_geo,
        "scanned_tiles": scanned_tiles,
        "num_releases_total": len(release_map),
        "download_tasks": len(tasks),
        "dry_run": bool(args.dry_run),
        "mode": str(args.mode),
    }
    if args.mode == "fixed_releases":
        meta["release_ids"] = list(args.release_ids or [])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wayback_scan_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(meta, indent=2))
        return

    # stage 2: download
    t0 = time.time()
    counts: Dict[str, int] = {"OK": 0, "SKIP": 0, "FAIL": 0}
    status_counts: Dict[str, int] = {}
    done = 0
    log_every = max(0, int(args.log_every))
    with ThreadPoolExecutor(max_workers=int(args.max_threads)) as ex:
        futs = {ex.submit(_download_one, session, url, path): (url, path) for (url, path) in tasks}
        for fut in as_completed(futs):
            status = fut.result()
            done += 1
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "OK":
                counts["OK"] += 1
            elif status == "SKIP":
                counts["SKIP"] += 1
            else:
                counts["FAIL"] += 1
            if log_every and (done % log_every == 0):
                print(
                    json.dumps(
                        {
                            "done": done,
                            "total": len(tasks),
                            "ok": counts["OK"],
                            "skip": counts["SKIP"],
                            "fail": counts["FAIL"],
                            "elapsed_s": time.time() - t0,
                        }
                    ),
                    file=os.sys.stderr,
                    flush=True,
                )

    report = {**meta, "download": counts, "status_counts": status_counts, "elapsed_s": time.time() - t0}
    (out_dir / "wayback_download_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
