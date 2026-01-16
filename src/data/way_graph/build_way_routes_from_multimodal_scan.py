from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.geo_grid import BBox, haversine_m


TZ_SHANGHAI = timezone(timedelta(hours=8))
WAY_ID_KEYS = ("osm_way_id", "osm_wayid", "way_id")


@dataclass(frozen=True)
class StateBBoxes:
    # Loose boxes (from project notes). Used only for route_city coarse labeling / optional filtering.
    mi: Tuple[float, float, float, float] = (-90.4, 41.7, -82.4, 48.3)
    oh: Tuple[float, float, float, float] = (-84.8, 38.4, -80.5, 42.0)

    def which(self, *, lat: float, lon: float) -> int:
        mi0, mi1, mi2, mi3 = self.mi
        if (mi1 <= lat <= mi3) and (mi0 <= lon <= mi2):
            return 0  # MI
        oh0, oh1, oh2, oh3 = self.oh
        if (oh1 <= lat <= oh3) and (oh0 <= lon <= oh2):
            return 1  # OH
        return 2  # OTHER

    def in_mi_or_oh(self, *, lat: float, lon: float) -> bool:
        return self.which(lat=lat, lon=lon) in (0, 1)


@dataclass(frozen=True)
class BuildCfg:
    bbox: BBox
    od_bin_deg: float
    min_points_in_bbox_ratio: float
    min_od_dist_km: float
    min_seq_len: int
    coord_scale: float
    od_filter: str  # "all" or "mi_oh"


def _safe_float(v: object) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: object) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _bin_int(x: float, bin_deg: float) -> int:
    return int(math.floor(float(x) / float(bin_deg)))


def _pick_latlon(row: Dict[str, str], *, prefer_matched: bool) -> Tuple[Optional[float], Optional[float]]:
    if prefer_matched:
        lat_m = _safe_float(row.get("matched_latitude", "")) or _safe_float(row.get("matched_lat", ""))
        lon_m = _safe_float(row.get("matched_longitude", "")) or _safe_float(row.get("matched_lon", ""))
        if lat_m is not None and lon_m is not None:
            return float(lat_m), float(lon_m)
    lat = _safe_float(row.get("latitude", "")) or _safe_float(row.get("lat", ""))
    lon = _safe_float(row.get("longitude", "")) or _safe_float(row.get("lon", ""))
    if lat is None or lon is None:
        return None, None
    return float(lat), float(lon)


def _pick_osm_way_id(row: Dict[str, str]) -> Optional[int]:
    for k in WAY_ID_KEYS:
        v = row.get(k, "")
        wi = _safe_int(v)
        if wi is not None and int(wi) > 0:
            return int(wi)
    return None


def _parse_time_s(v: object) -> Optional[int]:
    # Follow build_detroit_segments.py: accept epoch (s/ms) or ISO-like strings.
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        fv = float(s)
        if fv > 1e12:
            fv = fv / 1000.0
        return int(fv)
    except (TypeError, ValueError):
        pass
    # Minimal ISO fallback (avoid heavy parsing on non-ISO strings).
    try:
        from datetime import datetime as _dt  # local import

        s2 = s.replace("Z", "+00:00")
        t = _dt.fromisoformat(s2)
        if t.tzinfo is None:
            return int(t.replace(tzinfo=timezone.utc).timestamp())
        return int(t.timestamp())
    except Exception:
        return None


def _iter_csv_rows_from_zip(zf: zipfile.ZipFile, member: str) -> Iterator[Dict[str, str]]:
    with zf.open(member, "r") as f:
        text = io.TextIOWrapper(f, encoding="utf-8", errors="ignore", newline="")
        reader = csv.DictReader(text)
        for row in reader:
            yield row


def _latlon_to_yx(lat: float, lon: float, *, bbox: BBox, coord_scale: float) -> Tuple[float, float]:
    lon0 = float(bbox.min_lon)
    lat1 = float(bbox.max_lat)
    inv_lon = float(coord_scale) / max(float(bbox.max_lon - bbox.min_lon), 1e-12)
    inv_lat = float(coord_scale) / max(float(bbox.max_lat - bbox.min_lat), 1e-12)
    x = (float(lon) - lon0) * inv_lon
    y = (lat1 - float(lat)) * inv_lat
    return float(y), float(x)


def _dedup_consecutive_int(seq: Iterable[int]) -> List[int]:
    out: List[int] = []
    last: Optional[int] = None
    for x in seq:
        xi = int(x)
        if last is None or xi != int(last):
            out.append(xi)
            last = xi
    return out


def _process_member_chunk(
    zip_path: str,
    members: List[str],
    cfg_dict: Dict[str, object],
    target_od_bins: List[Tuple[int, int, int, int]],
    *,
    seed: int,
    prefer_matched: bool,
) -> Dict[str, object]:
    cfg_dict = dict(cfg_dict)
    bbox_v = cfg_dict.get("bbox")
    if isinstance(bbox_v, dict):
        cfg_dict["bbox"] = BBox(**bbox_v)  # type: ignore[arg-type]
    cfg = BuildCfg(**cfg_dict)  # type: ignore[arg-type]

    target = set(tuple(int(x) for x in k) for k in target_od_bins)
    sb = StateBBoxes()
    rnd = random.Random(int(seed))

    scanned = 0
    any_in_bbox = 0
    pass_ratio = 0
    dropped_short_od = 0
    dropped_no_time = 0
    dropped_short_seq = 0
    kept = 0

    out_routes: List[Dict[str, object]] = []

    bbox = cfg.bbox
    min_lon, max_lon = float(bbox.min_lon), float(bbox.max_lon)
    min_lat, max_lat = float(bbox.min_lat), float(bbox.max_lat)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in members:
            scanned += 1
            tot_n = 0
            in_n = 0
            first: Optional[Tuple[float, float]] = None
            last: Optional[Tuple[float, float]] = None
            first_t: Optional[int] = None
            # OD computed from first/last point with valid way_id (consistent with scan + signature).
            first_way_coord: Optional[Tuple[float, float]] = None
            last_way_coord: Optional[Tuple[float, float]] = None

            # Build way sequence in one pass (consecutive dedup).
            way_seq: List[int] = []
            last_way: Optional[int] = None

            rows = _iter_csv_rows_from_zip(zf, member)
            for row in rows:
                lat, lon = _pick_latlon(row, prefer_matched=bool(prefer_matched))
                if lat is None or lon is None:
                    continue
                tot_n += 1
                if (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat):
                    in_n += 1
                if first is None:
                    first = (lat, lon)
                if first_t is None:
                    first_t = _parse_time_s(row.get("time", ""))
                last = (lat, lon)

                wid = _pick_osm_way_id(row)
                if wid is None:
                    continue
                if int(wid) <= 0:
                    continue
                if first_way_coord is None:
                    first_way_coord = (lat, lon)
                last_way_coord = (lat, lon)
                if last_way is None or int(wid) != int(last_way):
                    way_seq.append(int(wid))
                    last_way = int(wid)

            if tot_n <= 1 or first is None or last is None:
                continue
            if in_n > 0:
                any_in_bbox += 1

            ratio = float(in_n) / float(max(1, tot_n))
            if ratio >= float(cfg.min_points_in_bbox_ratio):
                pass_ratio += 1
            else:
                continue

            if first_way_coord is None or last_way_coord is None:
                dropped_short_seq += 1
                continue
            o_lat, o_lon = first_way_coord
            d_lat, d_lon = last_way_coord

            # Optional: keep only MI/OH OD pairs (avoid needing extra-state OSM pbfs).
            if str(cfg.od_filter) == "mi_oh":
                if not (sb.in_mi_or_oh(lat=o_lat, lon=o_lon) and sb.in_mi_or_oh(lat=d_lat, lon=d_lon)):
                    continue

            od_km = float(haversine_m(o_lat, o_lon, d_lat, d_lon)) / 1000.0
            if od_km < float(cfg.min_od_dist_km):
                dropped_short_od += 1
                continue

            od_key = (
                _bin_int(o_lon, cfg.od_bin_deg),
                _bin_int(o_lat, cfg.od_bin_deg),
                _bin_int(d_lon, cfg.od_bin_deg),
                _bin_int(d_lat, cfg.od_bin_deg),
            )
            if od_key not in target:
                continue

            # Require time for conditioning.
            if first_t is None:
                dropped_no_time += 1
                continue

            dedup = _dedup_consecutive_int(way_seq)
            if len(dedup) < int(cfg.min_seq_len):
                dropped_short_seq += 1
                continue

            start_pos = _latlon_to_yx(o_lat, o_lon, bbox=bbox, coord_scale=float(cfg.coord_scale))
            dest_pos = _latlon_to_yx(d_lat, d_lon, bbox=bbox, coord_scale=float(cfg.coord_scale))
            route_city = int(sb.which(lat=o_lat, lon=o_lon))

            out_routes.append(
                {
                    "member": member,
                    "od_bin": [int(x) for x in od_key],
                    "route_city": int(route_city),
                    "start_t": int(first_t),
                    "start_pos": [float(start_pos[0]), float(start_pos[1])],
                    "dest_pos": [float(dest_pos[0]), float(dest_pos[1])],
                    "way_seq_osm": [int(x) for x in dedup],
                }
            )
            kept += 1

    _ = rnd  # keep seed plumbed for future sampling
    return {
        "scanned": int(scanned),
        "any_in_bbox": int(any_in_bbox),
        "pass_ratio": int(pass_ratio),
        "dropped_short_od": int(dropped_short_od),
        "dropped_no_time": int(dropped_no_time),
        "dropped_short_seq": int(dropped_short_seq),
        "kept": int(kept),
        "routes": out_routes,
    }


def _p(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def _write_semantic_stub(*, out_dir: Path, bbox: BBox, coord_scale: float) -> Path:
    stub = {
        "grid": {"H": int(round(float(coord_scale))), "W": int(round(float(coord_scale))), "bbox": asdict(bbox)},
        "note": "Stub meta for mapping lat/lon -> (y,x) in way_features builder. No raster arrays included.",
    }
    sem_dir = out_dir / "semantic_stub"
    sem_dir.mkdir(parents=True, exist_ok=True)
    (sem_dir / "osm_road_prob_meta.json").write_text(json.dumps(stub, ensure_ascii=False, indent=2), encoding="utf-8")
    return sem_dir


def build_way_routes_from_scan(
    *,
    scan_report_json: Path,
    trajectory_zip: Path,
    out_npz: Path,
    num_workers: int,
    mp_start: str,
    chunk_size: int,
    limit_files: int,
    seed: int,
    prefer_matched: bool,
    od_filter: str,
    min_seq_len: int,
    coord_scale: float,
) -> Dict[str, object]:
    rep = json.loads(Path(scan_report_json).read_text(encoding="utf-8"))
    cfg0 = rep.get("scan_config", {}) or {}
    bbox0 = cfg0.get("bbox", None)
    if not isinstance(bbox0, dict):
        raise SystemExit(f"scan_report_json missing scan_config.bbox: {scan_report_json}")
    bbox = BBox(**{k: float(v) for k, v in bbox0.items()})

    od_bin_deg = float(cfg0.get("od_bin_deg", 0.01))
    min_ratio = float(cfg0.get("min_points_in_bbox_ratio", 0.8))
    min_od_km = float(cfg0.get("min_od_dist_km", 1.0))

    mm = rep.get("multimodal_od_bins", [])
    if not isinstance(mm, list) or not mm:
        raise SystemExit(f"scan_report_json has no multimodal_od_bins: {scan_report_json}")
    target_bins = []
    for ent in mm:
        od_bin = ent.get("od_bin", None) if isinstance(ent, dict) else None
        if isinstance(od_bin, list) and len(od_bin) == 4:
            target_bins.append(tuple(int(x) for x in od_bin))
    if not target_bins:
        raise SystemExit(f"No valid od_bin entries found in scan report: {scan_report_json}")

    cfg = BuildCfg(
        bbox=bbox,
        od_bin_deg=float(od_bin_deg),
        min_points_in_bbox_ratio=float(min_ratio),
        min_od_dist_km=float(min_od_km),
        min_seq_len=int(min_seq_len),
        coord_scale=float(coord_scale),
        od_filter=str(od_filter),
    )

    # Enumerate members.
    zip_path = str(trajectory_zip)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith(".csv")]
    if int(limit_files) > 0:
        members = members[: int(limit_files)]
    total_files = int(len(members))
    if total_files == 0:
        raise SystemExit(f"No csv members found in: {trajectory_zip}")

    num_workers = int(num_workers)
    if num_workers <= 0:
        num_workers = os.cpu_count() or 1
    chunk_size = max(1, int(chunk_size))
    chunks = [members[i : i + chunk_size] for i in range(0, total_files, chunk_size)]
    cfg_dict = asdict(cfg)

    # progress
    t0 = time.time()
    scanned = 0
    any_in_bbox = 0
    pass_ratio = 0
    dropped_short_od = 0
    dropped_no_time = 0
    dropped_short_seq = 0
    kept = 0

    out_routes: List[Dict[str, object]] = []

    next_report = 50000

    def _report() -> None:
        nonlocal next_report
        while scanned >= next_report:
            elapsed = max(1e-6, float(time.time() - t0))
            pct = 100.0 * float(scanned) / float(max(1, total_files))
            rps = float(scanned) / elapsed
            kps = float(kept) / elapsed
            print(f"[INFO] scanned={scanned}/{total_files} ({pct:.1f}%) kept={kept} rps={rps:.1f} kept_ps={kps:.2f}", file=sys.stderr)
            next_report += 50000

    mp_ctx = None
    if str(mp_start) in {"fork", "spawn"}:
        import multiprocessing as mp

        mp_ctx = mp.get_context(str(mp_start))

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as ex:
        futs = []
        for ci, chunk in enumerate(chunks):
            futs.append(
                ex.submit(
                    _process_member_chunk,
                    zip_path,
                    chunk,
                    cfg_dict,
                    target_bins,
                    seed=int(seed) + int(ci),
                    prefer_matched=bool(prefer_matched),
                )
            )
        for fut in as_completed(futs):
            r = fut.result()
            scanned += int(r["scanned"])
            any_in_bbox += int(r.get("any_in_bbox", 0))
            pass_ratio += int(r.get("pass_ratio", 0))
            dropped_short_od += int(r.get("dropped_short_od", 0))
            dropped_no_time += int(r.get("dropped_no_time", 0))
            dropped_short_seq += int(r.get("dropped_short_seq", 0))
            kept += int(r.get("kept", 0))
            out_routes.extend(list(r.get("routes") or []))
            _report()

    elapsed = max(1e-6, float(time.time() - t0))
    print(f"[INFO] scanned={scanned}/{total_files} kept={kept} elapsed_s={elapsed:.1f}", file=sys.stderr)

    # Build vocab + CSR encoding.
    seqs: List[List[int]] = []
    start_t: List[int] = []
    start_pos: List[List[float]] = []
    dest_pos: List[List[float]] = []
    route_city: List[int] = []
    od_bins: List[Tuple[int, int, int, int]] = []

    way_vocab: set[int] = set()
    for r in out_routes:
        ways = [int(x) for x in (r.get("way_seq_osm") or [])]
        if len(ways) < int(cfg.min_seq_len):
            continue
        seqs.append(ways)
        start_t.append(int(r["start_t"]))
        start_pos.append([float(r["start_pos"][0]), float(r["start_pos"][1])])
        dest_pos.append([float(r["dest_pos"][0]), float(r["dest_pos"][1])])
        route_city.append(int(r["route_city"]))
        od_bins.append(tuple(int(x) for x in r["od_bin"]))
        way_vocab.update(ways)

    way_osm_id = np.asarray(sorted(list(way_vocab)), dtype=np.int64).reshape(-1)
    way_to_idx = {int(w): int(i) for i, w in enumerate(way_osm_id.tolist())}

    N = int(len(seqs))
    ptr = np.zeros((N + 1,), dtype=np.int64)
    lens = np.zeros((N,), dtype=np.int32)
    start_way = np.zeros((N,), dtype=np.int32)
    dest_way = np.zeros((N,), dtype=np.int32)
    flat: List[int] = []
    for i, s in enumerate(seqs):
        enc = [way_to_idx[int(w)] for w in s]
        L = int(len(enc))
        lens[i] = np.int32(L)
        start_way[i] = np.int32(enc[0])
        dest_way[i] = np.int32(enc[-1])
        flat.extend(enc)
        ptr[i + 1] = np.int64(len(flat))
    way_seq_idx = np.asarray(flat, dtype=np.int32)

    start_t_arr = np.asarray(start_t, dtype=np.int64).reshape(-1)
    start_pos_arr = np.asarray(start_pos, dtype=np.float32).reshape(-1, 2)
    dest_pos_arr = np.asarray(dest_pos, dtype=np.float32).reshape(-1, 2)
    route_city_arr = np.asarray(route_city, dtype=np.int8).reshape(-1)
    corridor_type = np.full((N,), -1, dtype=np.int8)  # optional later labeling

    # OD-bin stats (top-K by count) for sanity.
    od_count: Dict[Tuple[int, int, int, int], int] = {}
    for k in od_bins:
        od_count[k] = od_count.get(k, 0) + 1
    top_od = sorted(od_count.items(), key=lambda kv: -int(kv[1]))[:20]

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "build_way_routes_from_multimodal_scan",
        "inputs": {"scan_report_json": str(scan_report_json), "trajectory_zip": str(trajectory_zip)},
        "config": {
            "od_filter": str(cfg.od_filter),
            "prefer_matched": bool(prefer_matched),
            "min_seq_len": int(cfg.min_seq_len),
            "coord_scale": float(cfg.coord_scale),
            "bbox": asdict(cfg.bbox),
            "od_bin_deg": float(cfg.od_bin_deg),
            "min_points_in_bbox_ratio": float(cfg.min_points_in_bbox_ratio),
            "min_od_dist_km": float(cfg.min_od_dist_km),
            "num_workers": int(num_workers),
            "chunk_size": int(chunk_size),
            "mp_start": str(mp_start),
        },
        "stats": {
            "total_files_scanned": int(scanned),
            "files_any_in_bbox": int(any_in_bbox),
            "files_pass_ratio_gate": int(pass_ratio),
            "files_dropped_short_od_km": int(dropped_short_od),
            "files_dropped_no_time": int(dropped_no_time),
            "files_dropped_short_way_seq": int(dropped_short_seq),
            "n_routes": int(N),
            "n_way_vocab": int(way_osm_id.size),
            "way_seq_len": {"p50": _p(lens, 50), "p90": _p(lens, 90), "max": int(np.max(lens) if lens.size else 0)},
            "route_city_counts": np.bincount(np.clip(route_city_arr.astype(np.int64), 0, 15), minlength=4).astype(np.int64).tolist(),
            "top_od_bins": [{"od_bin": list(map(int, k)), "n_routes": int(v)} for k, v in top_od],
            "elapsed_s": float(elapsed),
        },
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        way_osm_id=way_osm_id,
        way_seq_ptr=ptr,
        way_seq_idx=way_seq_idx,
        way_seq_len=lens,
        start_way=start_way,
        dest_way=dest_way,
        start_t=start_t_arr,
        route_city=route_city_arr,
        corridor_type=corridor_type,
        start_pos=start_pos_arr,
        dest_pos=dest_pos_arr,
        meta=meta,
    )

    # Also dump members list for reproducibility (small).
    members_jsonl = out_npz.parent / "members.jsonl"
    with members_jsonl.open("w", encoding="utf-8") as f:
        for r in out_routes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sem_dir = _write_semantic_stub(out_dir=out_npz.parent, bbox=bbox, coord_scale=float(coord_scale))
    report = {"ok": True, "out_npz": str(out_npz), "members_jsonl": str(members_jsonl), "semantic_stub_dir": str(sem_dir), "meta": meta}
    (out_npz.parent / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build way_routes.npz from Trajectory.zip for OD bins found by scan_multimodal_od_region.py.")
    p.add_argument("--scan_report_json", type=Path, required=True)
    p.add_argument("--trajectory_zip", type=Path, required=True)
    p.add_argument("--out_npz", type=Path, required=True)

    p.add_argument("--od_filter", choices=["all", "mi_oh"], default="mi_oh", help="Filter OD pairs to MI/OH only (KISS default) or keep all bins from scan.")
    p.add_argument("--prefer_matched", action="store_true", help="Prefer matched_latitude/longitude for bbox/OD.")
    p.add_argument("--min_seq_len", type=int, default=2, help="Drop routes whose deduped osm_way_id length < this.")
    p.add_argument("--coord_scale", type=float, default=1024.0, help="Map bbox lat/lon into [0,coord_scale] for start_pos/dest_pos and way_features.")

    p.add_argument("--num_workers", type=int, default=48)
    p.add_argument("--mp_start", choices=["fork", "spawn"], default="fork")
    p.add_argument("--chunk_size", type=int, default=2000)
    p.add_argument("--limit_files", type=int, default=0, help="Debug: limit number of zip members scanned (0=no limit).")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = build_way_routes_from_scan(
        scan_report_json=Path(args.scan_report_json),
        trajectory_zip=Path(args.trajectory_zip),
        out_npz=Path(args.out_npz),
        num_workers=int(args.num_workers),
        mp_start=str(args.mp_start),
        chunk_size=int(args.chunk_size),
        limit_files=int(args.limit_files),
        seed=int(args.seed),
        prefer_matched=bool(args.prefer_matched),
        od_filter=str(args.od_filter),
        min_seq_len=int(args.min_seq_len),
        coord_scale=float(args.coord_scale),
    )
    st = report["meta"]["stats"]
    wl = st["way_seq_len"]
    compact = {
        "ok": True,
        "out_npz": report["out_npz"],
        "n_routes": int(st["n_routes"]),
        "n_way_vocab": int(st["n_way_vocab"]),
        "way_seq_len_p50": float(wl["p50"]),
        "way_seq_len_p90": float(wl["p90"]),
        "way_seq_len_max": int(wl["max"]),
        "semantic_stub_dir": report["semantic_stub_dir"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
