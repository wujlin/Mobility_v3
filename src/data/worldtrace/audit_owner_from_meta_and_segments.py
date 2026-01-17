from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # optional
    pq = None

try:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
except ModuleNotFoundError:  # pragma: no cover
    mp = None
    ProcessPoolExecutor = None  # type: ignore[assignment]
    as_completed = None  # type: ignore[assignment]

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class OwnerAuditCfg:
    od_bin_deg: List[float]
    min_od_dist_km: float
    max_rows_per_segments: int
    dump_top_owner_n: int
    num_workers: int
    mp_start: str
    log_every: int


@dataclass(frozen=True)
class TripRow:
    city: str
    traj_base: str
    start_t: int
    o_lat: float
    o_lon: float
    d_lat: float
    d_lon: float
    owner: Optional[str] = None


def _sha1_8(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:8]


def _normalize_owner(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, (list, dict)):
        try:
            s = json.dumps(v, ensure_ascii=False, sort_keys=True)
        except TypeError:
            s = str(v)
        s = s.strip()
        if s in ("", "[]", "{}"):
            return None
        return s
    s = str(v).strip()
    return s if s else None


def _extract_owner(obj: Dict[str, Any]) -> Optional[str]:
    # Common keys (data card + variants)
    for k in ("Owner", "owner", "OWNER"):
        if k in obj:
            return _normalize_owner(obj.get(k))
    # Some datasets wrap meta in nested dicts.
    for parent in ("properties", "meta", "metadata", "info"):
        v = obj.get(parent, None)
        if isinstance(v, dict):
            for k in ("Owner", "owner", "OWNER"):
                if k in v:
                    return _normalize_owner(v.get(k))
    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dlat = p2 - p1
    dlon = math.radians(float(lon2) - float(lon1))
    a = math.sin(dlat / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return float(r * c)


def _od_bin(lat: float, lon: float, deg: float) -> Tuple[int, int]:
    deg = float(deg)
    if deg <= 0:
        raise ValueError(f"od_bin_deg must be > 0, got {deg}")
    return int(math.floor(float(lon) / deg)), int(math.floor(float(lat) / deg))


def _pick_city_label(path: Path) -> str:
    # Prefer parent folder (e.g., detroit_core_v1), else file stem.
    p = Path(path)
    if p.parent.name:
        return str(p.parent.name)
    return str(p.stem)


def _read_segments_parquet(path: Path, *, city: str, max_rows: int) -> List[TripRow]:
    if pq is None:
        raise SystemExit("Missing dependency: pyarrow (required to read .parquet). Install: conda/pip install pyarrow")
    table = pq.read_table(str(path), columns=["traj_csv", "t", "lat", "lon"])
    rows = table.to_pydict()
    traj = rows.get("traj_csv", [])
    ts = rows.get("t", [])
    lats = rows.get("lat", [])
    lons = rows.get("lon", [])
    n = min(len(traj), len(ts), len(lats), len(lons))
    if max_rows > 0:
        n = min(n, int(max_rows))

    out: List[TripRow] = []
    for i in range(int(n)):
        member = traj[i]
        if not isinstance(member, str) or not member:
            continue
        t_list = ts[i]
        lat_list = lats[i]
        lon_list = lons[i]
        if not isinstance(t_list, list) or not isinstance(lat_list, list) or not isinstance(lon_list, list):
            continue
        if len(t_list) == 0 or len(lat_list) == 0 or len(lon_list) == 0:
            continue
        if len(lat_list) != len(lon_list):
            continue
        start_t = int(t_list[0])
        o_lat = float(lat_list[0])
        o_lon = float(lon_list[0])
        d_lat = float(lat_list[-1])
        d_lon = float(lon_list[-1])
        out.append(
            TripRow(
                city=str(city),
                traj_base=Path(member).name,
                start_t=int(start_t),
                o_lat=float(o_lat),
                o_lon=float(o_lon),
                d_lat=float(d_lat),
                d_lon=float(d_lon),
            )
        )
    return out


def _meta_member_candidates(traj_base: str) -> List[str]:
    base = str(traj_base)
    stem = str(Path(base).stem)
    # Try common patterns: same stem with .json across possible zip layouts.
    names = [
        f"{stem}.json",
        f"Meta/{stem}.json",
        f"meta/{stem}.json",
        # WorldTrace official HF layout often has nested directories.
        f"data/yuanshao/OpenTrace/Meta/{stem}.json",
        f"data/yuanshao/OpenTrace/meta/{stem}.json",
        f"OpenTrace/Meta/{stem}.json",
    ]
    # Dedup while preserving order.
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def _load_owner_map_from_meta_zip(meta_zip: Path, wanted_traj_base: Sequence[str]) -> Tuple[Dict[str, str], int]:
    wanted = {str(x) for x in wanted_traj_base if str(x)}
    owner_by_base: Dict[str, str] = {}
    n_json_read = 0

    with zipfile.ZipFile(meta_zip, "r") as zf:
        # 1) Try direct member lookup (fast if naming matches).
        for tb in list(wanted):
            for cand in _meta_member_candidates(tb):
                try:
                    with zf.open(cand, "r") as f:
                        obj = json.load(f)
                    n_json_read += 1
                except KeyError:
                    continue
                except json.JSONDecodeError:
                    continue
                owner = _extract_owner(obj)
                if owner:
                    owner_by_base[tb] = owner
                    break

        missing = wanted - set(owner_by_base.keys())
        if not missing:
            return owner_by_base, int(n_json_read)

        # 2) Fallback: scan meta zip until we find all missing.
        for info in zf.infolist():
            if not info.filename.endswith(".json"):
                continue
            try:
                with zf.open(info, "r") as f:
                    obj = json.load(f)
                n_json_read += 1
            except json.JSONDecodeError:
                continue
            filename = obj.get("Filename", None)
            if not isinstance(filename, str) or not filename:
                continue
            base = Path(filename).name
            if base not in missing:
                continue
            owner = _extract_owner(obj)
            if owner:
                owner_by_base[base] = owner
            missing.discard(base)
            if not missing:
                break

    return owner_by_base, int(n_json_read)


def _process_traj_base_chunk(meta_zip: str, keys: List[str]) -> Tuple[Dict[str, str], List[str], int]:
    owner_by_key: Dict[str, str] = {}
    found_keys: List[str] = []
    n_json_read = 0
    with zipfile.ZipFile(meta_zip, "r") as zf:
        for key in keys:
            found = False
            for cand in _meta_member_candidates(key):
                try:
                    with zf.open(cand, "r") as f:
                        obj = json.load(f)
                    n_json_read += 1
                    found = True
                except KeyError:
                    continue
                except json.JSONDecodeError:
                    continue
                owner = _extract_owner(obj)
                if owner:
                    owner_by_key[str(key)] = owner
                break
            if found:
                found_keys.append(str(key))
    return owner_by_key, found_keys, int(n_json_read)


def _load_owner_map_with_progress(
    *,
    meta_zip: Path,
    wanted_keys: Sequence[str],
    num_workers: int,
    mp_start: str,
    log_every: int,
) -> Tuple[Dict[str, str], int, int, Dict[str, int]]:
    """
    Load Owner mapping for a list of trajectory basenames.

    Fast path (recommended): direct lookup by member name candidates, parallelized by chunk.
    Fallback: scan Meta.zip for remaining missing (still with progress).
    """
    wanted = sorted({str(x) for x in wanted_keys if str(x)})
    n_total = int(len(wanted))
    if n_total == 0:
        return {}, 0

    num_workers = int(num_workers)
    if num_workers <= 0:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, int(num_workers))
    log_every = max(1, int(log_every))

    # Heuristic: small jobs are faster sequentially.
    use_mp = bool(num_workers > 1 and n_total >= 500 and mp is not None and ProcessPoolExecutor is not None)
    t0 = time.time()
    owner_by_key: Dict[str, str] = {}
    found_set: set[str] = set()
    n_json_read = 0
    match_mode = {"by_meta_id": 0, "by_filename_stem": 0}

    if not use_mp:
        with zipfile.ZipFile(meta_zip, "r") as zf:
            found = 0
            for i, tb in enumerate(wanted, start=1):
                meta_found = False
                for cand in _meta_member_candidates(tb):
                    try:
                        with zf.open(cand, "r") as f:
                            obj = json.load(f)
                        n_json_read += 1
                        meta_found = True
                    except KeyError:
                        continue
                    except json.JSONDecodeError:
                        continue
                    owner = _extract_owner(obj)
                    if owner:
                        owner_by_key[tb] = owner
                        found += 1
                    break
                if meta_found:
                    found_set.add(tb)
                if (i % log_every) == 0 or i == n_total:
                    elapsed = max(1e-6, float(time.time() - t0))
                    rps = float(i) / elapsed
                    print(
                        f"[INFO] owner_lookup processed={i}/{n_total} found={found} elapsed_s={elapsed:.1f} rps={rps:.1f}",
                        file=sys.stderr,
                    )
    else:
        # Chunk wanted list; each worker opens Meta.zip once.
        chunks: List[List[str]] = []
        chunk_size = max(200, int(math.ceil(float(n_total) / float(num_workers * 8))))
        for s in range(0, n_total, chunk_size):
            chunks.append(wanted[s : s + chunk_size])
        n_chunks = int(len(chunks))

        ctx = mp.get_context(str(mp_start))
        done = 0
        found = 0
        with ProcessPoolExecutor(max_workers=int(num_workers), mp_context=ctx) as ex:
            futs = [ex.submit(_process_traj_base_chunk, str(meta_zip), ch) for ch in chunks]
            for fut in as_completed(futs):
                d, found_bases, n_read = fut.result()
                done += 1
                n_json_read += int(n_read)
                owner_by_key.update(d)
                found_set.update(found_bases)
                found = int(len(owner_by_key))
                if (done % max(1, (log_every // chunk_size))) == 0 or done == n_chunks:
                    elapsed = max(1e-6, float(time.time() - t0))
                    cps = float(done) / elapsed
                    print(
                        f"[INFO] owner_lookup chunks={done}/{n_chunks} found={found} elapsed_s={elapsed:.1f} chunks_ps={cps:.2f}",
                        file=sys.stderr,
                    )

    # Only fallback-scan Meta.zip if we failed to locate the corresponding meta json by filename candidates.
    missing = set(wanted) - set(found_set)
    if not missing:
        return owner_by_key, int(n_json_read), int(len(found_set)), match_mode

    # Fallback: scan Meta.zip until we find all missing.
    print(f"[WARN] direct lookup missed {len(missing)}/{n_total}; fallback scanning Meta.zip...", file=sys.stderr)
    with zipfile.ZipFile(meta_zip, "r") as zf:
        infos = zf.infolist()
        total = int(len(infos))
        scanned = 0
        next_log = log_every
        for info in infos:
            scanned += 1
            if not info.filename.endswith(".json"):
                continue
            try:
                with zf.open(info, "r") as f:
                    obj = json.load(f)
                n_json_read += 1
            except json.JSONDecodeError:
                continue
            filename = obj.get("Filename", obj.get("filename", None))
            k_id = Path(info.filename).stem
            k_fname = Path(filename).stem if isinstance(filename, str) and filename else ""

            hit = False
            if k_id and k_id in missing:
                found_set.add(k_id)
                owner = _extract_owner(obj)
                if owner:
                    owner_by_key[k_id] = owner
                missing.discard(k_id)
                match_mode["by_meta_id"] += 1
                hit = True
            if k_fname and k_fname in missing:
                found_set.add(k_fname)
                owner = _extract_owner(obj)
                if owner:
                    owner_by_key[k_fname] = owner
                missing.discard(k_fname)
                match_mode["by_filename_stem"] += 1
                hit = True
            if hit and not missing:
                break
            if scanned >= next_log:
                elapsed = max(1e-6, float(time.time() - t0))
                pct = 100.0 * float(scanned) / float(max(1, total))
                rps = float(scanned) / elapsed
                print(
                    f"[INFO] meta_scan scanned={scanned}/{total} ({pct:.1f}%) missing={len(missing)} elapsed_s={elapsed:.1f} rps={rps:.1f}",
                    file=sys.stderr,
                )
                next_log += log_every

    return owner_by_key, int(n_json_read), int(len(found_set)), match_mode


def _percentiles(x: np.ndarray, ps: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{int(p)}": float("nan") for p in ps}
    return {f"p{int(p)}": float(np.percentile(x, float(p))) for p in ps}


def _topk_counts(counts: Dict[str, int], k: int) -> List[Dict[str, object]]:
    items = sorted(((o, int(n)) for o, n in counts.items()), key=lambda t: (-t[1], t[0]))
    out = []
    for o, n in items[: max(0, int(k))]:
        out.append({"owner_hash": _sha1_8(o), "n_trips": int(n)})
    return out


def _owner_gate_summary(*, trips_per_owner: Dict[str, int]) -> Dict[str, object]:
    total = int(sum(int(v) for v in trips_per_owner.values()))
    if total <= 0:
        return {"top10_share": float("nan"), "top10_n_owners": 0}
    items = sorted((int(v) for v in trips_per_owner.values()), reverse=True)
    top10 = int(sum(items[:10]))
    return {"top10_share": float(top10) / float(total), "top10_n_owners": int(min(10, len(items)))}


def _od_coverage_stats(rows: Sequence[TripRow], *, od_bin_deg: float) -> Dict[str, object]:
    deg = float(od_bin_deg)
    buckets: Dict[Tuple[int, int, int, int], List[str]] = {}
    for r in rows:
        if not r.owner:
            continue
        o = _od_bin(r.o_lat, r.o_lon, deg)
        d = _od_bin(r.d_lat, r.d_lon, deg)
        key = (o[0], o[1], d[0], d[1])
        buckets.setdefault(key, []).append(str(r.owner))

    n_od = int(len(buckets))
    n_trips = int(sum(len(v) for v in buckets.values()))
    n_od_1trip = int(sum(1 for v in buckets.values() if len(v) == 1))
    n_od_ge2owners = int(sum(1 for v in buckets.values() if len(set(v)) >= 2))
    n_od_1owner_ge2trips = int(sum(1 for v in buckets.values() if len(v) >= 2 and len(set(v)) == 1))

    # For debugging/interpretation: distribution of trips per OD and owners per OD.
    trips_per_od = np.asarray([len(v) for v in buckets.values()], dtype=np.int64)
    owners_per_od = np.asarray([len(set(v)) for v in buckets.values()], dtype=np.int64)

    return {
        "od_bin_deg": float(deg),
        "n_od_bins": int(n_od),
        "n_trips_in_od_bins": int(n_trips),
        "n_od_bins_1trip": int(n_od_1trip),
        "n_od_bins_ge2owners": int(n_od_ge2owners),
        "n_od_bins_1owner_ge2trips": int(n_od_1owner_ge2trips),
        "trips_per_od": {"p10": float(np.percentile(trips_per_od, 10)) if trips_per_od.size else float("nan"),
                         "p50": float(np.percentile(trips_per_od, 50)) if trips_per_od.size else float("nan"),
                         "p90": float(np.percentile(trips_per_od, 90)) if trips_per_od.size else float("nan"),
                         "max": int(trips_per_od.max()) if trips_per_od.size else 0},
        "owners_per_od": {"p10": float(np.percentile(owners_per_od, 10)) if owners_per_od.size else float("nan"),
                          "p50": float(np.percentile(owners_per_od, 50)) if owners_per_od.size else float("nan"),
                          "p90": float(np.percentile(owners_per_od, 90)) if owners_per_od.size else float("nan"),
                          "max": int(owners_per_od.max()) if owners_per_od.size else 0},
    }


def run_audit(*, meta_zip: Path, segments_parquet: Sequence[Path], segments_label: Optional[Sequence[str]], cfg: OwnerAuditCfg) -> Dict[str, object]:
    # 1) Load trips from segments parquet(s).
    trips: List[TripRow] = []
    for i, sp in enumerate(list(segments_parquet)):
        label = None
        if segments_label and i < len(segments_label) and segments_label[i]:
            label = str(segments_label[i])
        city = label or _pick_city_label(Path(sp))
        trips.extend(_read_segments_parquet(Path(sp), city=city, max_rows=int(cfg.max_rows_per_segments)))

    # 2) Filter trivial OD (optional).
    if float(cfg.min_od_dist_km) > 0:
        keep: List[TripRow] = []
        for r in trips:
            if _haversine_km(r.o_lat, r.o_lon, r.d_lat, r.d_lon) >= float(cfg.min_od_dist_km):
                keep.append(r)
        trips = keep

    # 3) Join owner from meta zip.
    wanted_keys = sorted({Path(t.traj_base).stem for t in trips})
    owner_by_key, n_meta_json_read, n_meta_found, match_mode = _load_owner_map_with_progress(
        meta_zip=Path(meta_zip),
        wanted_keys=wanted_keys,
        num_workers=int(cfg.num_workers),
        mp_start=str(cfg.mp_start),
        log_every=int(cfg.log_every),
    )
    n_owner_found = 0
    for i, r in enumerate(trips):
        owner = owner_by_key.get(Path(r.traj_base).stem, None)
        if owner:
            n_owner_found += 1
        trips[i] = TripRow(**{**asdict(r), "owner": owner})

    # 4) Owner-level counts and proxies.
    trips_per_owner: Dict[str, int] = {}
    owner_min_t: Dict[str, int] = {}
    owner_max_t: Dict[str, int] = {}
    trips_with_owner = 0
    for r in trips:
        if not r.owner:
            continue
        trips_with_owner += 1
        o = str(r.owner)
        trips_per_owner[o] = int(trips_per_owner.get(o, 0)) + 1
        owner_min_t[o] = int(min(owner_min_t.get(o, r.start_t), r.start_t))
        owner_max_t[o] = int(max(owner_max_t.get(o, r.start_t), r.start_t))

    owner_counts = np.asarray(list(trips_per_owner.values()), dtype=np.int64)
    unique_owner_count = int(owner_counts.size)
    trips_dist = _percentiles(owner_counts, ps=[10, 50, 90])
    trips_dist["max"] = int(owner_counts.max()) if owner_counts.size else 0

    spans = []
    for o in trips_per_owner.keys():
        spans.append(float(owner_max_t[o] - owner_min_t[o]))
    spans = np.asarray(spans, dtype=np.float64)
    span_stats = _percentiles(spans / 3600.0, ps=[10, 50, 90])
    span_stats["max"] = float(np.max(spans / 3600.0)) if spans.size else 0.0

    n_gt_1000 = int(sum(1 for v in trips_per_owner.values() if int(v) > 1000))
    n_span_lt_1d = int(sum(1 for o in trips_per_owner.keys() if (owner_max_t[o] - owner_min_t[o]) < 86400))

    # 5) OD coverage stats (for each od_bin_deg).
    od_stats = [_od_coverage_stats(trips, od_bin_deg=float(deg)) for deg in cfg.od_bin_deg]

    # 6) Go/No-Go suggestions (Detroit-first gate; user can apply per city if needed).
    gate_notes = []
    for s in od_stats:
        n2 = int(s.get("n_od_bins_ge2owners", 0))
        deg = float(s.get("od_bin_deg", float("nan")))
        gate_notes.append(
            {
                "od_bin_deg": deg,
                "rule": "n_od_bins_ge2owners >= 100",
                "value": n2,
                "pass": bool(n2 >= 100),
            }
        )

    gate_owner_conc = _owner_gate_summary(trips_per_owner=trips_per_owner)
    gate_owner_conc_pass = not (float(gate_owner_conc.get("top10_share", 0.0)) >= 0.95 and unique_owner_count > 0)

    report = {
        "ok": True,
        "task": "worldtrace_owner_audit",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "meta_zip": str(meta_zip),
            "segments_parquet": [str(p) for p in segments_parquet],
            "segments_label": list(segments_label) if segments_label else None,
        },
        "cfg": asdict(cfg),
        "join": {
            "traj_key": "Path(traj_csv).stem",
            "meta_key": "Path(Meta.Filename).stem OR Path(meta_json_path).stem",
            "meta_json_read": int(n_meta_json_read),
            "meta_found": int(n_meta_found),
            "meta_found_ratio": float(n_meta_found) / float(max(1, len(wanted_keys))),
            "match_mode": match_mode,
            "trips_total": int(len(trips)),
            "trips_kept_after_od_filter": int(len(trips)),
            "trips_with_owner": int(trips_with_owner),
            "owner_found_ratio": float(trips_with_owner) / float(max(1, len(trips))),
        },
        "owner": {
            "unique_owner_count": int(unique_owner_count),
            "trips_per_owner": trips_dist,
            "time_span_hours_per_owner": span_stats,
            "n_owner_trips_gt_1000": int(n_gt_1000),
            "n_owner_span_lt_1day": int(n_span_lt_1d),
            "topk": _topk_counts(trips_per_owner, k=int(cfg.dump_top_owner_n)),
            "concentration": gate_owner_conc,
        },
        "od_coverage": od_stats,
        "go_no_go": {
            "od_multi_owner_gate": gate_notes,
            "owner_concentration_rule": "top10_share < 0.95",
            "owner_concentration_pass": bool(gate_owner_conc_pass),
        },
    }
    return report


def _render_md(report: Dict[str, object]) -> str:
    owner = report.get("owner", {}) if isinstance(report.get("owner", {}), dict) else {}
    join = report.get("join", {}) if isinstance(report.get("join", {}), dict) else {}
    od_cov = report.get("od_coverage", [])
    if not isinstance(od_cov, list):
        od_cov = []

    lines: List[str] = []
    lines.append("# WorldTrace Owner 审计（Phase 2 / Step 2.1）")
    lines.append("")
    lines.append("## 结论先行（Go/No-Go）")
    lines.append("")
    conc = owner.get("concentration", {}) if isinstance(owner.get("concentration", {}), dict) else {}
    top10_share = conc.get("top10_share", None)
    lines.append(f"- `top10_share`（Top-10 Owners 占比）：{top10_share}")
    for g in report.get("go_no_go", {}).get("od_multi_owner_gate", []):  # type: ignore[union-attr]
        if not isinstance(g, dict):
            continue
        lines.append(f"- `od_bin_deg={g.get('od_bin_deg')}`：`n_od_bins_ge2owners={g.get('value')}`（阈值 100） -> pass={g.get('pass')}")
    lines.append("")
    lines.append("## 关键统计")
    lines.append("")
    lines.append(f"- trips_total={join.get('trips_total')}, trips_with_owner={join.get('trips_with_owner')}, owner_found_ratio={join.get('owner_found_ratio')}")
    lines.append(f"- unique_owner_count={owner.get('unique_owner_count')}")
    tpo = owner.get("trips_per_owner", {}) if isinstance(owner.get("trips_per_owner", {}), dict) else {}
    lines.append(f"- trips_per_owner: p10={tpo.get('p10')}, p50={tpo.get('p50')}, p90={tpo.get('p90')}, max={tpo.get('max')}")
    tsp = owner.get("time_span_hours_per_owner", {}) if isinstance(owner.get("time_span_hours_per_owner", {}), dict) else {}
    lines.append(f"- owner_time_span_hours: p10={tsp.get('p10')}, p50={tsp.get('p50')}, p90={tsp.get('p90')}, max={tsp.get('max')}")
    lines.append(f"- n_owner_trips_gt_1000={owner.get('n_owner_trips_gt_1000')}, n_owner_span_lt_1day={owner.get('n_owner_span_lt_1day')}")
    lines.append("")
    lines.append("## OD 覆盖（按 od_bin_deg）")
    lines.append("")
    for s in od_cov:
        if not isinstance(s, dict):
            continue
        lines.append(
            "- "
            + f"od_bin_deg={s.get('od_bin_deg')}: "
            + f"n_od_bins={s.get('n_od_bins')}, "
            + f"n_od_bins_1trip={s.get('n_od_bins_1trip')}, "
            + f"n_od_bins_ge2owners={s.get('n_od_bins_ge2owners')}, "
            + f"n_od_bins_1owner_ge2trips={s.get('n_od_bins_1owner_ge2trips')}"
        )
    lines.append("")
    lines.append("## 可复现命令")
    lines.append("")
    lines.append("建议在工作站 conda 环境（含 `pyarrow`）运行：")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m src.data.worldtrace.audit_owner_from_meta_and_segments \\")
    lines.append("  --meta_zip \"$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Meta.zip\" \\")
    lines.append("  --segments_parquet \"$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet\" \\")
    lines.append("  --out_json \"$EXP_ROOT/A_owner_audit_detroit/report.json\" \\")
    lines.append("  --out_md  \"$EXP_ROOT/A_owner_audit_detroit/report.md\" \\")
    lines.append("  --od_bin_deg 0.01 0.02 \\")
    lines.append("  --min_od_dist_km 1.0 \\")
    lines.append("  --num_top_owners 20")
    lines.append("```")
    lines.append("")
    lines.append("## 备注（隐私）")
    lines.append("")
    lines.append("- 默认只输出 `owner_hash`（sha1 前 8 位）与计数，不输出原始 Owner 字符串。")
    lines.append("")
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit WorldTrace Owner coverage using Meta.zip + city segments.parquet.")
    p.add_argument("--meta_zip", type=Path, required=True, help="Path to Meta.zip (contains Owner).")
    p.add_argument("--segments_parquet", type=Path, nargs="+", required=True, help="One or more segments(.parquet) files.")
    p.add_argument("--segments_label", type=str, nargs="*", default=None, help="Optional labels for each segments_parquet.")
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_md", type=Path, default=None)

    p.add_argument("--od_bin_deg", type=float, nargs="*", default=[0.01, 0.02], help="OD bin size(s) in degrees.")
    p.add_argument("--min_od_dist_km", type=float, default=1.0, help="Filter trips with OD distance < this threshold.")
    p.add_argument("--max_rows_per_segments", type=int, default=0, help="Debug cap per segments parquet (0=no cap).")
    p.add_argument("--num_top_owners", type=int, default=20, help="How many top owners to include (hashed).")
    p.add_argument("--num_workers", type=int, default=24, help="Parallel workers for Owner join (Meta.zip lookups).")
    p.add_argument("--mp_start", type=str, default="fork", choices=["fork", "spawn"], help="Multiprocessing start method.")
    p.add_argument("--log_every", type=int, default=2000, help="Progress log frequency (items; used for lookup/scan).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md = Path(args.out_md) if args.out_md is not None else None
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)

    cfg = OwnerAuditCfg(
        od_bin_deg=[float(x) for x in (args.od_bin_deg or [0.02])],
        min_od_dist_km=float(args.min_od_dist_km),
        max_rows_per_segments=int(args.max_rows_per_segments),
        dump_top_owner_n=int(args.num_top_owners),
        num_workers=int(args.num_workers),
        mp_start=str(args.mp_start),
        log_every=int(args.log_every),
    )

    report = run_audit(
        meta_zip=Path(args.meta_zip),
        segments_parquet=[Path(p) for p in args.segments_parquet],
        segments_label=list(args.segments_label) if args.segments_label else None,
        cfg=cfg,
    )
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_json}", file=sys.stderr)

    if out_md is not None:
        out_md.write_text(_render_md(report), encoding="utf-8")
        print(f"[saved] {out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
