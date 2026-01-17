from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # optional
    pq = None

from src.data.worldtrace.audit_owner_from_meta_and_segments import _extract_owner, _meta_member_candidates, _sha1_8

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    tz_offset_hours: float
    min_od_dist_km: float
    max_rows: int
    owner: str  # raw owner string (exact match); empty means top-1 owner
    include_owner_raw: bool


def _hour_from_unix(t: np.ndarray, tz_offset_hours: float) -> np.ndarray:
    t = np.asarray(t, dtype=np.int64).reshape(-1)
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    sec = ((t + tz_sec) % 86400).astype(np.int64, copy=False)
    return (sec // 3600).astype(np.int64, copy=False)


def _dow_from_unix(t: np.ndarray, tz_offset_hours: float) -> np.ndarray:
    t = np.asarray(t, dtype=np.int64).reshape(-1)
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    days = ((t + tz_sec) // 86400).astype(np.int64, copy=False)
    return ((days + 3) % 7).astype(np.int64, copy=False)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dlat = p2 - p1
    dlon = math.radians(float(lon2) - float(lon1))
    a = math.sin(dlat / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return float(r * c)


def _endpoint_idx(osm_way_id: np.ndarray) -> Optional[Tuple[int, int]]:
    w = np.asarray(osm_way_id, dtype=np.int64).reshape(-1)
    good = np.nonzero(w > 0)[0]
    if good.size == 0:
        return None
    return int(good[0]), int(good[-1])


def _owner_by_traj_key(meta_zip: Path, keys: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with zipfile.ZipFile(meta_zip, "r") as zf:
        for k in keys:
            for cand in _meta_member_candidates(k):
                try:
                    with zf.open(cand, "r") as f:
                        obj = json.load(f)
                except KeyError:
                    continue
                except json.JSONDecodeError:
                    continue
                owner = _extract_owner(obj)
                if owner:
                    out[str(k)] = str(owner)
                break
    return out


def _summ(a: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "max": float("nan"), "mean": float("nan")}
    p10, p50, p90 = np.percentile(a, [10, 50, 90]).tolist()
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90), "max": float(np.max(a)), "mean": float(np.mean(a))}


def run(*, segments_parquet: Path, meta_zip: Path, out_json: Path, cfg: Cfg) -> Dict[str, object]:
    if pq is None:
        raise SystemExit("pyarrow is required. Install: pip/conda install pyarrow")

    cols = ["traj_csv", "t", "lat", "lon", "osm_way_id"]
    pf = pq.ParquetFile(str(segments_parquet))

    traj_keys: List[str] = []
    starts: List[int] = []
    ends: List[int] = []
    dists_km: List[float] = []
    durs_min: List[float] = []

    # First pass: extract per-trip stats keyed by traj stem (only for those passing gates).
    scanned = 0
    kept = 0
    for batch in pf.iter_batches(batch_size=128, columns=cols):
        d = batch.to_pydict()
        n_rows = len(d["traj_csv"])
        for i in range(n_rows):
            scanned += 1
            if int(cfg.max_rows) > 0 and scanned > int(cfg.max_rows):
                break
            key = Path(str(d["traj_csv"][i])).stem
            osm = np.asarray(d["osm_way_id"][i], dtype=np.int64)
            end_idx = _endpoint_idx(osm)
            if end_idx is None:
                continue
            i0, i1 = end_idx
            t = np.asarray(d["t"][i], dtype=np.int64)
            lat = np.asarray(d["lat"][i], dtype=np.float64)
            lon = np.asarray(d["lon"][i], dtype=np.float64)
            if t.size < 2 or lat.size <= max(i0, i1) or lon.size <= max(i0, i1):
                continue
            od_km = _haversine_km(float(lat[i0]), float(lon[i0]), float(lat[i1]), float(lon[i1]))
            if float(od_km) < float(cfg.min_od_dist_km):
                continue
            dur_min = float(int(t[-1]) - int(t[0])) / 60.0
            traj_keys.append(str(key))
            starts.append(int(t[0]))
            ends.append(int(t[-1]))
            dists_km.append(float(od_km))
            durs_min.append(float(dur_min))
            kept += 1
        if int(cfg.max_rows) > 0 and scanned > int(cfg.max_rows):
            break

    if kept <= 0:
        raise SystemExit("No trips kept after filters (check min_od_dist_km / max_rows).")

    owner_map = _owner_by_traj_key(Path(meta_zip), traj_keys)
    owners = [owner_map.get(k, "") for k in traj_keys]
    if not owners or all(o == "" for o in owners):
        raise SystemExit("Owner join failed (0 owners found). Check Meta.zip path and naming layout.")

    # Pick target owner.
    counts: Dict[str, int] = {}
    for o in owners:
        if o:
            counts[o] = int(counts.get(o, 0)) + 1
    if not counts:
        raise SystemExit("Owner join failed (no non-empty owners).")
    if str(cfg.owner).strip():
        owner_target = str(cfg.owner).strip()
        if owner_target not in counts:
            raise SystemExit(f"--owner not found in this subset: {owner_target}")
    else:
        owner_target = max(counts.items(), key=lambda kv: kv[1])[0]

    mask = np.asarray([o == owner_target for o in owners], dtype=bool)
    st = np.asarray(starts, dtype=np.int64)[mask]
    en = np.asarray(ends, dtype=np.int64)[mask]
    dist = np.asarray(dists_km, dtype=np.float64)[mask]
    dur = np.asarray(durs_min, dtype=np.float64)[mask]

    hour = _hour_from_unix(st, cfg.tz_offset_hours)
    dow = _dow_from_unix(st, cfg.tz_offset_hours)

    hour_counts = {f"{h:02d}": int(np.sum(hour == h)) for h in range(24)}
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_counts = {dow_labels[i]: int(np.sum(dow == i)) for i in range(7)}

    report: Dict[str, object] = {
        "ok": True,
        "task": "worldtrace_owner_profile",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {"segments_parquet": str(segments_parquet), "meta_zip": str(meta_zip)},
        "cfg": asdict(cfg),
        "subset": {
            "scanned": int(scanned),
            "kept": int(kept),
            "min_od_dist_km": float(cfg.min_od_dist_km),
            "unique_owner_count": int(len(counts)),
        },
        "owner": {
            "owner_hash": _sha1_8(owner_target),
            "n_trips": int(np.sum(mask)),
            "topk_owner_hash": [{"owner_hash": _sha1_8(o), "n_trips": int(n)} for o, n in sorted(counts.items(), key=lambda kv: -kv[1])[:10]],
        },
        "time": {
            "tz_offset_hours": float(cfg.tz_offset_hours),
            "hour_counts": hour_counts,
            "dow_counts": dow_counts,
        },
        "distance_km": _summ(dist),
        "duration_min": _summ(dur),
        "time_span_hours": float((int(np.max(en)) - int(np.min(st))) / 3600.0) if st.size else float("nan"),
    }
    if bool(cfg.include_owner_raw):
        report["owner"]["owner_raw"] = owner_target

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WorldTrace Owner profile (hour/dow + OD distance/duration) from segments_with_wayid.parquet.")
    p.add_argument("--segments_parquet", type=Path, required=True)
    p.add_argument("--meta_zip", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--min_od_dist_km", type=float, default=1.0)
    p.add_argument("--max_rows", type=int, default=0, help="Debug cap (0=no cap).")
    p.add_argument("--owner", type=str, default="", help="Exact owner string to analyze; empty means top-1 owner.")
    p.add_argument("--include_owner_raw", action="store_true", help="Include raw owner string in output JSON (default: hash only).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Cfg(
        tz_offset_hours=float(args.tz_offset_hours),
        min_od_dist_km=float(args.min_od_dist_km),
        max_rows=int(args.max_rows),
        owner=str(args.owner),
        include_owner_raw=bool(args.include_owner_raw),
    )
    rep = run(segments_parquet=Path(args.segments_parquet), meta_zip=Path(args.meta_zip), out_json=Path(args.out_json), cfg=cfg)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
