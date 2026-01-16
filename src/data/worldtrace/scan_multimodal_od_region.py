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
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from src.utils.geo_grid import BBox, haversine_m


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class ScanConfig:
    bbox: BBox
    od_bin_deg: float = 0.01
    route_bin_deg: float = 0.05
    sig_subsample_step: int = 5
    max_sig_len: int = 256
    min_points_in_bbox_ratio: float = 0.80
    min_od_dist_km: float = 1.0
    min_routes_per_od: int = 5
    min_cluster_frac: float = 0.20
    cluster_sep_thr: float = 0.35  # Jaccard distance in [0,1]
    merge_dist_thr: float = 0.20  # merge near-identical signatures
    max_sigs_per_od: int = 32
    max_rep_files: int = 3


@dataclass
class SigEntry:
    count: int
    reps: List[str]


@dataclass
class ODEntry:
    n_routes: int
    sigs: Dict[Tuple[int, ...], SigEntry]


def _safe_float(v: object) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pick_latlon(row: Dict[str, str]) -> Tuple[Optional[float], Optional[float]]:
    # Prefer matched coords if present.
    lat_m = _safe_float(row.get("matched_latitude", "")) or _safe_float(row.get("matched_lat", ""))
    lon_m = _safe_float(row.get("matched_longitude", "")) or _safe_float(row.get("matched_lon", ""))
    if lat_m is not None and lon_m is not None:
        return float(lat_m), float(lon_m)
    lat = _safe_float(row.get("latitude", "")) or _safe_float(row.get("lat", ""))
    lon = _safe_float(row.get("longitude", "")) or _safe_float(row.get("lon", ""))
    if lat is None or lon is None:
        return None, None
    return float(lat), float(lon)


def _iter_csv_rows_from_zip(zf: zipfile.ZipFile, member: str) -> Iterator[Dict[str, str]]:
    with zf.open(member, "r") as f:
        text = io.TextIOWrapper(f, encoding="utf-8", errors="ignore", newline="")
        reader = csv.DictReader(text)
        for row in reader:
            yield row


def _bin_int(x: float, bin_deg: float) -> int:
    # Use floor for stable binning across +/- degrees.
    return int(math.floor(float(x) / float(bin_deg)))


def _pack_bin2(lon_bin: int, lat_bin: int) -> int:
    # Pack two signed 32-bit ints into one unsigned 64-bit space (still fits in Python int).
    return ((int(lon_bin) & 0xFFFFFFFF) << 32) | (int(lat_bin) & 0xFFFFFFFF)


def _route_signature_from_stream(
    *,
    rows: Iterable[Dict[str, str]],
    cfg: ScanConfig,
    member: str,
) -> Tuple[bool, bool, Optional[Tuple[int, int, int, int]], Optional[Tuple[int, ...]], Dict[str, float]]:
    bbox = cfg.bbox
    in_n = 0
    tot_n = 0
    first: Optional[Tuple[float, float]] = None
    last: Optional[Tuple[float, float]] = None

    sig: List[int] = []
    last_bin: Optional[int] = None
    step = max(1, int(cfg.sig_subsample_step))

    for row in rows:
        lat, lon = _pick_latlon(row)
        if lat is None or lon is None:
            continue
        tot_n += 1

        if bbox.min_lon <= lon <= bbox.max_lon and bbox.min_lat <= lat <= bbox.max_lat:
            in_n += 1

        if first is None:
            first = (lat, lon)
        last = (lat, lon)

        # signature (subsample + consecutive dedup)
        if (tot_n % step) == 1:
            lon_bin = _bin_int(lon, cfg.route_bin_deg)
            lat_bin = _bin_int(lat, cfg.route_bin_deg)
            b = _pack_bin2(lon_bin, lat_bin)
            if last_bin is None or b != int(last_bin):
                sig.append(int(b))
                last_bin = int(b)
                if int(cfg.max_sig_len) > 0 and len(sig) >= int(cfg.max_sig_len):
                    # Cap signature length for memory safety.
                    break

    has_any_in_bbox = bool(in_n > 0)
    if tot_n <= 1 or first is None or last is None:
        return has_any_in_bbox, False, None, None, {"points_total": float(tot_n), "points_in_bbox_ratio": float(in_n) / float(max(1, tot_n))}

    ratio = float(in_n) / float(max(1, tot_n))
    pass_ratio = bool(ratio >= float(cfg.min_points_in_bbox_ratio))
    if not pass_ratio:
        return has_any_in_bbox, False, None, None, {"points_total": float(tot_n), "points_in_bbox_ratio": float(ratio)}

    o_lat, o_lon = first
    d_lat, d_lon = last
    od_km = float(haversine_m(o_lat, o_lon, d_lat, d_lon)) / 1000.0
    if od_km < float(cfg.min_od_dist_km):
        return has_any_in_bbox, False, None, None, {"od_km": float(od_km), "points_total": float(tot_n), "points_in_bbox_ratio": float(ratio)}

    o_lon_bin = _bin_int(o_lon, cfg.od_bin_deg)
    o_lat_bin = _bin_int(o_lat, cfg.od_bin_deg)
    d_lon_bin = _bin_int(d_lon, cfg.od_bin_deg)
    d_lat_bin = _bin_int(d_lat, cfg.od_bin_deg)
    od_key = (o_lon_bin, o_lat_bin, d_lon_bin, d_lat_bin)

    stats = {"od_km": float(od_km), "points_total": float(tot_n), "points_in_bbox_ratio": float(ratio)}
    return has_any_in_bbox, True, od_key, tuple(sig), stats


def _update_sig_table(
    *,
    sigs: Dict[Tuple[int, ...], SigEntry],
    sig: Tuple[int, ...],
    member: str,
    max_sigs: int,
    max_rep_files: int,
) -> None:
    ent = sigs.get(sig)
    if ent is not None:
        ent.count += 1
        if len(ent.reps) < int(max_rep_files):
            ent.reps.append(member)
        return
    sigs[sig] = SigEntry(count=1, reps=[member])
    if len(sigs) <= int(max_sigs):
        return
    # Drop the smallest-count signature (exact top-K heuristic, K small so O(K) is fine).
    s_min = min(sigs.keys(), key=lambda k: int(sigs[k].count))
    if s_min in sigs:
        del sigs[s_min]


def _merge_od_entry(
    *,
    dst: ODEntry,
    src: ODEntry,
    max_sigs: int,
    max_rep_files: int,
) -> None:
    dst.n_routes += int(src.n_routes)
    for sig, ent in src.sigs.items():
        if sig in dst.sigs:
            dst_ent = dst.sigs[sig]
            dst_ent.count += int(ent.count)
            # merge reps
            for r in ent.reps:
                if len(dst_ent.reps) >= int(max_rep_files):
                    break
                if r not in dst_ent.reps:
                    dst_ent.reps.append(r)
        else:
            dst.sigs[sig] = SigEntry(count=int(ent.count), reps=list(ent.reps)[: int(max_rep_files)])
    # Keep bounded.
    while len(dst.sigs) > int(max_sigs):
        s_min = min(dst.sigs.keys(), key=lambda k: int(dst.sigs[k].count))
        del dst.sigs[s_min]


def _process_member_chunk(
    zip_path: str,
    members: List[str],
    cfg_dict: Dict[str, object],
    *,
    seed: int,
) -> Dict[str, object]:
    cfg = ScanConfig(**cfg_dict)
    rnd = random.Random(int(seed))
    buckets: Dict[Tuple[int, int, int, int], ODEntry] = {}

    scanned = 0
    kept = 0
    any_in_bbox = 0
    pass_ratio = 0
    dropped_short = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in members:
            scanned += 1
            rows = _iter_csv_rows_from_zip(zf, member)
            has_any, ok, od_key, sig, stats = _route_signature_from_stream(rows=rows, cfg=cfg, member=member)
            if has_any:
                any_in_bbox += 1
            if stats.get("points_in_bbox_ratio", 0.0) >= float(cfg.min_points_in_bbox_ratio):
                pass_ratio += 1
            else:
                # below ratio
                pass
            if not ok or od_key is None or sig is None:
                if float(stats.get("points_in_bbox_ratio", 0.0)) >= float(cfg.min_points_in_bbox_ratio) and float(stats.get("od_km", 0.0)) < float(cfg.min_od_dist_km):
                    dropped_short += 1
                continue
            kept += 1
            e = buckets.get(od_key)
            if e is None:
                e = ODEntry(n_routes=0, sigs={})
                buckets[od_key] = e
            e.n_routes += 1
            _update_sig_table(
                sigs=e.sigs,
                sig=sig,
                member=member,
                max_sigs=int(cfg.max_sigs_per_od),
                max_rep_files=int(cfg.max_rep_files),
            )

    out_buckets: Dict[Tuple[int, int, int, int], Tuple[int, List[Tuple[Tuple[int, ...], int, List[str]]]]] = {}
    for k, v in buckets.items():
        out_buckets[k] = (
            int(v.n_routes),
            [(sig, int(ent.count), list(ent.reps)) for sig, ent in v.sigs.items()],
        )

    _ = rnd  # silence lint
    return {
        "scanned": int(scanned),
        "any_in_bbox": int(any_in_bbox),
        "pass_ratio": int(pass_ratio),
        "dropped_short": int(dropped_short),
        "kept": int(kept),
        "buckets": out_buckets,
    }


def _jaccard_dist(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    uni = len(sa | sb)
    return 1.0 - (float(inter) / float(max(1, uni)))


def _cluster_signatures(
    sig_items: List[Tuple[Tuple[int, ...], SigEntry]],
    *,
    merge_dist_thr: float,
    max_rep_files: int,
) -> List[Dict[str, object]]:
    # Greedy merge similar signatures to reduce noise.
    clusters: List[Dict[str, object]] = []
    for sig, ent in sig_items:
        placed = False
        for c in clusters:
            if _jaccard_dist(sig, c["rep_sig"]) < float(merge_dist_thr):
                c["count"] += int(ent.count)
                for r in ent.reps:
                    if len(c["reps"]) >= int(max_rep_files):
                        break
                    if r not in c["reps"]:
                        c["reps"].append(r)
                placed = True
                break
        if not placed:
            clusters.append({"rep_sig": sig, "count": int(ent.count), "reps": list(ent.reps)[: int(max_rep_files)]})
    clusters.sort(key=lambda x: -int(x["count"]))
    return clusters


def scan_multimodal(*, trajectory_zip: Path, out_json: Path, cfg: ScanConfig, num_workers: int, mp_start: str, chunk_size: int, limit_files: int, seed: int, max_out_bins: int) -> Dict[str, object]:
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

    t0 = time.time()
    scanned = 0
    kept = 0
    files_any_in_bbox = 0
    files_pass_ratio = 0
    files_dropped_short = 0
    buckets: Dict[Tuple[int, int, int, int], ODEntry] = {}

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

    ctx = mp_start
    mp_ctx = None
    if ctx == "spawn":
        import multiprocessing as mp
        mp_ctx = mp.get_context("spawn")
    elif ctx == "fork":
        import multiprocessing as mp
        mp_ctx = mp.get_context("fork")

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as ex:
        futs = []
        for ci, chunk in enumerate(chunks):
            futs.append(ex.submit(_process_member_chunk, zip_path, chunk, cfg_dict, seed=int(seed) + int(ci)))
        for fut in as_completed(futs):
            r = fut.result()
            scanned += int(r["scanned"])
            kept += int(r["kept"])
            files_any_in_bbox += int(r.get("any_in_bbox", 0))
            files_pass_ratio += int(r.get("pass_ratio", 0))
            files_dropped_short += int(r.get("dropped_short", 0))

            # merge buckets
            for od_key, v in (r.get("buckets") or {}).items():
                n_routes, sig_rows = v
                src = ODEntry(n_routes=int(n_routes), sigs={})
                for sig, cnt, reps in sig_rows:
                    src.sigs[tuple(int(x) for x in sig)] = SigEntry(count=int(cnt), reps=list(reps)[: int(cfg.max_rep_files)])
                dst = buckets.get(od_key)
                if dst is None:
                    buckets[od_key] = src
                else:
                    _merge_od_entry(dst=dst, src=src, max_sigs=int(cfg.max_sigs_per_od), max_rep_files=int(cfg.max_rep_files))

            _report()

    # Final report line
    elapsed = max(1e-6, float(time.time() - t0))
    print(f"[INFO] scanned={scanned}/{total_files} kept={kept} elapsed_s={elapsed:.1f}", file=sys.stderr)

    # Analyze OD bins
    od_counts = np.asarray([int(v.n_routes) for v in buckets.values()], dtype=np.int64)
    uniq_od_bins = int(len(buckets))
    n_ge_5 = int(np.sum(od_counts >= 5))
    n_ge_10 = int(np.sum(od_counts >= 10))

    multimodal: List[Dict[str, object]] = []
    for od_key, ent in buckets.items():
        n_routes = int(ent.n_routes)
        if n_routes < int(cfg.min_routes_per_od):
            continue
        sig_items = list(ent.sigs.items())
        sig_items.sort(key=lambda kv: -int(kv[1].count))
        clusters = _cluster_signatures(sig_items, merge_dist_thr=float(cfg.merge_dist_thr), max_rep_files=int(cfg.max_rep_files))
        if len(clusters) < 2:
            continue
        # Minimum minority cluster fraction
        if float(clusters[1]["count"]) < float(cfg.min_cluster_frac) * float(n_routes):
            continue
        # Separation between top-2 clusters
        d01 = _jaccard_dist(clusters[0]["rep_sig"], clusters[1]["rep_sig"])
        if d01 < float(cfg.cluster_sep_thr):
            continue
        multimodal.append(
            {
                "od_bin": [int(x) for x in od_key],
                "n_routes": int(n_routes),
                "n_clusters": int(len(clusters)),
                "cluster_sizes": [int(c["count"]) for c in clusters],
                "cluster_rep_files": [list(c["reps"]) for c in clusters],
                "top2_jaccard_dist": float(d01),
            }
        )

    multimodal.sort(key=lambda x: (-int(x["n_routes"]), -float(x["top2_jaccard_dist"])))
    total_mm = int(len(multimodal))
    max_out_bins = max(1, int(max_out_bins))
    mm_out = multimodal[:max_out_bins]
    truncated = total_mm > len(mm_out)

    report = {
        "ok": True,
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "scan_config": {
            "bbox": asdict(cfg.bbox),
            "od_bin_deg": float(cfg.od_bin_deg),
            "route_bin_deg": float(cfg.route_bin_deg),
            "sig_subsample_step": int(cfg.sig_subsample_step),
            "max_sig_len": int(cfg.max_sig_len),
            "min_points_in_bbox_ratio": float(cfg.min_points_in_bbox_ratio),
            "min_od_dist_km": float(cfg.min_od_dist_km),
            "min_routes_per_od": int(cfg.min_routes_per_od),
            "min_cluster_frac": float(cfg.min_cluster_frac),
            "cluster_sep_thr": float(cfg.cluster_sep_thr),
            "merge_dist_thr": float(cfg.merge_dist_thr),
            "max_sigs_per_od": int(cfg.max_sigs_per_od),
            "max_rep_files": int(cfg.max_rep_files),
        },
        "summary": {
            "total_files_scanned": int(scanned),
            "files_any_in_bbox": int(files_any_in_bbox),
            "files_pass_ratio_gate": int(files_pass_ratio),
            "files_dropped_short_od_km": int(files_dropped_short),
            "files_kept_after_filter": int(kept),
            "unique_od_bins": int(uniq_od_bins),
            "od_bins_with_n_gte_5": int(n_ge_5),
            "od_bins_with_n_gte_10": int(n_ge_10),
            "od_bins_multimodal": int(total_mm),
            "multimodal_truncated": bool(truncated),
        },
        "multimodal_od_bins": mm_out,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scan WorldTrace Trajectory.zip and find multimodal OD bins in a bbox region.")
    p.add_argument("--trajectory_zip", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)

    p.add_argument("--bbox", type=float, nargs=4, default=[-90.4, 38.4, -80.5, 48.3], metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    p.add_argument("--od_bin_deg", type=float, default=0.01)
    p.add_argument("--route_bin_deg", type=float, default=0.05)
    p.add_argument("--sig_subsample_step", type=int, default=5)
    p.add_argument("--max_sig_len", type=int, default=256)
    p.add_argument("--min_points_in_bbox_ratio", type=float, default=0.8)
    p.add_argument("--min_od_dist_km", type=float, default=1.0)

    p.add_argument("--min_routes_per_od", type=int, default=5)
    p.add_argument("--min_cluster_frac", type=float, default=0.2)
    p.add_argument("--cluster_sep_thr", type=float, default=0.35, help="Jaccard distance threshold in [0,1].")
    p.add_argument("--merge_dist_thr", type=float, default=0.20, help="Merge near-identical signatures if dist < this.")
    p.add_argument("--max_sigs_per_od", type=int, default=32)
    p.add_argument("--max_rep_files", type=int, default=3)
    p.add_argument("--max_out_bins", type=int, default=1000)

    p.add_argument("--num_workers", type=int, default=24)
    p.add_argument("--mp_start", choices=["fork", "spawn"], default="fork")
    p.add_argument("--chunk_size", type=int, default=2000)
    p.add_argument("--limit_files", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    min_lon, min_lat, max_lon, max_lat = map(float, args.bbox)
    cfg = ScanConfig(
        bbox=BBox(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat),
        od_bin_deg=float(args.od_bin_deg),
        route_bin_deg=float(args.route_bin_deg),
        sig_subsample_step=int(args.sig_subsample_step),
        max_sig_len=int(args.max_sig_len),
        min_points_in_bbox_ratio=float(args.min_points_in_bbox_ratio),
        min_od_dist_km=float(args.min_od_dist_km),
        min_routes_per_od=int(args.min_routes_per_od),
        min_cluster_frac=float(args.min_cluster_frac),
        cluster_sep_thr=float(args.cluster_sep_thr),
        merge_dist_thr=float(args.merge_dist_thr),
        max_sigs_per_od=int(args.max_sigs_per_od),
        max_rep_files=int(args.max_rep_files),
    )
    # Clamp distance thresholds to [0,1] for safety.
    if cfg.cluster_sep_thr > 1.0:
        print(f"[WARN] cluster_sep_thr={cfg.cluster_sep_thr} > 1.0; clamped to 1.0 (Jaccard dist).", file=sys.stderr)
        cfg = ScanConfig(**{**asdict(cfg), "cluster_sep_thr": 1.0})
    if cfg.merge_dist_thr > 1.0:
        print(f"[WARN] merge_dist_thr={cfg.merge_dist_thr} > 1.0; clamped to 1.0.", file=sys.stderr)
        cfg = ScanConfig(**{**asdict(cfg), "merge_dist_thr": 1.0})

    report = scan_multimodal(
        trajectory_zip=Path(args.trajectory_zip),
        out_json=Path(args.out_json),
        cfg=cfg,
        num_workers=int(args.num_workers),
        mp_start=str(args.mp_start),
        chunk_size=int(args.chunk_size),
        limit_files=int(args.limit_files),
        seed=int(args.seed),
        max_out_bins=int(args.max_out_bins),
    )
    s = report["summary"]
    print(
        f"[done] scanned={s['total_files_scanned']} kept={s['files_kept_after_filter']} "
        f"unique_od={s['unique_od_bins']} multimodal={s['od_bins_multimodal']} saved={args.out_json}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
