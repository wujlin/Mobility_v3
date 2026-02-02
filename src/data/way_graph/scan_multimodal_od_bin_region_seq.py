from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.geo_grid import haversine_m

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    od_bin_deg: float
    min_od_km: float
    min_routes_per_bin: int
    max_sigs_per_bin: int
    max_rep_routes_per_sig: int
    max_out_bins: int
    lcs_sep_thr: float
    lcs_max_patterns: int


def _p(x: np.ndarray, q: float) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, float(q)))


def _bin_int(x: float, bin_deg: float) -> int:
    return int(math.floor(float(x) / float(bin_deg)))


def _lcs_len(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    # DP with 2 rows; sequences are very short (<= ~13).
    n = int(len(a))
    m = int(len(b))
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        ai = int(a[i - 1])
        for j in range(1, m + 1):
            if ai == int(b[j - 1]):
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
        prev, cur = cur, prev
    return int(prev[m])


def _lcs_dist(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    denom = float(max(len(a), len(b), 1))
    return float(1.0 - (_lcs_len(a, b) / denom))


def _update_sig_table(
    *,
    sigs: Dict[Tuple[int, ...], Dict[str, Any]],
    sig: Tuple[int, ...],
    route_id: int,
    max_sigs: int,
    max_rep: int,
) -> None:
    ent = sigs.get(sig)
    if ent is not None:
        ent["count"] = int(ent["count"]) + 1
        reps = ent["reps"]
        if isinstance(reps, list) and len(reps) < int(max_rep):
            reps.append(int(route_id))
        return

    sigs[sig] = {"count": 1, "reps": [int(route_id)]}
    if len(sigs) <= int(max_sigs):
        return
    # Drop smallest-count signature (K small so O(K) is fine).
    s_min = min(sigs.keys(), key=lambda k: int(sigs[k]["count"]))
    if s_min in sigs:
        del sigs[s_min]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_city_kv(spec: str) -> Tuple[int, Path]:
    s = str(spec or "").strip()
    if "=" in s:
        k, v = s.split("=", 1)
    elif ":" in s:
        k, v = s.split(":", 1)
    else:
        raise ValueError(f"Bad spec (expect CITY=PATH): {spec!r}")
    city = int(str(k).strip())
    path = Path(str(v).strip()).expanduser()
    return city, path


def _decode_meta(meta_obj: object) -> Optional[dict]:
    if meta_obj is None:
        return None
    if isinstance(meta_obj, np.ndarray):
        if meta_obj.size != 1:
            return None
        meta_obj = meta_obj.item()
    return meta_obj if isinstance(meta_obj, dict) else None


def _grid_bbox_from_meta(meta: dict) -> Optional[Tuple[int, int, float, float, float, float]]:
    grid = meta.get("grid", {}) if isinstance(meta, dict) else {}
    if not isinstance(grid, dict):
        return None
    H = grid.get("H", None)
    W = grid.get("W", None)
    bbox = grid.get("bbox", None)
    if not isinstance(bbox, dict):
        return None
    try:
        H_i = int(H)
        W_i = int(W)
        min_lon = float(bbox["min_lon"])
        min_lat = float(bbox["min_lat"])
        max_lon = float(bbox["max_lon"])
        max_lat = float(bbox["max_lat"])
    except Exception:
        return None
    if H_i <= 0 or W_i <= 0:
        return None
    return (H_i, W_i, min_lon, min_lat, max_lon, max_lat)


def _meta_from_city_grid_meta(path: Path) -> dict:
    if str(path).endswith(".npz"):
        wf = np.load(str(path), allow_pickle=True)
        meta = _decode_meta(wf.get("meta", None))
        if meta is None:
            raise ValueError(f"{path} missing meta (need meta.grid.H/W/bbox).")
    else:
        meta = _read_json(path)

    if _grid_bbox_from_meta(meta) is None:
        if isinstance(meta, dict) and ("H" in meta) and ("W" in meta) and ("bbox" in meta):
            meta = {"grid": {"H": meta["H"], "W": meta["W"], "bbox": meta["bbox"]}}
    if _grid_bbox_from_meta(meta) is None:
        raise ValueError(f"{path} missing grid meta (need grid.H/grid.W/grid.bbox).")
    return meta


def _grid_yx_to_latlon(y: float, x: float, *, meta: dict) -> Tuple[float, float]:
    bb = _grid_bbox_from_meta(meta)
    if bb is None:
        raise ValueError("meta missing grid bbox")
    H, W, min_lon, min_lat, max_lon, max_lat = bb
    lon = float(min_lon + (float(x) / float(W)) * (max_lon - min_lon))
    lat = float(max_lat - (float(y) / float(H)) * (max_lat - min_lat))
    return lat, lon


def _require_city_meta(city_meta: Dict[int, dict], cities: List[int]) -> None:
    missing = [int(c) for c in cities if int(c) not in city_meta]
    if missing:
        raise SystemExit(f"[FATAL] missing --city_grid_meta for cities={missing} (need osm_road_prob_meta.json for OD bins).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan coarse OD bins and measure region_seq corridor diversity.")
    ap.add_argument("--way_routes_npz", type=Path, required=True)
    ap.add_argument("--region_seq_npz", type=Path, required=True, help="Output of extract_region_seq_stats (region_seq_*.npz).")
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--od_bin_deg", type=float, default=0.02, help="OD bin size in degrees (~2km at 0.02).")
    ap.add_argument("--min_od_km", type=float, default=1.0, help="Drop routes with OD distance < this.")
    ap.add_argument("--min_routes_per_bin", type=int, default=5)
    ap.add_argument("--max_sigs_per_bin", type=int, default=32)
    ap.add_argument("--max_rep_routes_per_sig", type=int, default=3)
    ap.add_argument("--max_out_bins", type=int, default=200)
    ap.add_argument("--lcs_sep_thr", type=float, default=0.50, help="Mark a bin as 'separated' if max LCS-dist >= this.")
    ap.add_argument("--lcs_max_patterns", type=int, default=6, help="Only compute LCS over top-K patterns by count.")
    ap.add_argument(
        "--city_grid_meta",
        type=str,
        action="append",
        default=[],
        help="Per-city grid meta for yx->latlon conversion, format CITY=PATH (osm_road_prob_meta.json or single-city way_features.npz).",
    )
    args = ap.parse_args()

    cfg = Cfg(
        od_bin_deg=float(args.od_bin_deg),
        min_od_km=float(args.min_od_km),
        min_routes_per_bin=int(args.min_routes_per_bin),
        max_sigs_per_bin=int(args.max_sigs_per_bin),
        max_rep_routes_per_sig=int(args.max_rep_routes_per_sig),
        max_out_bins=int(args.max_out_bins),
        lcs_sep_thr=float(args.lcs_sep_thr),
        lcs_max_patterns=int(args.lcs_max_patterns),
    )

    routes = np.load(str(args.way_routes_npz), allow_pickle=True)
    need = {"start_pos", "dest_pos", "route_city"}
    missing = sorted(list(need - set(routes.files)))
    if missing:
        raise SystemExit(f"[FATAL] way_routes_npz missing keys: {missing}")
    start_pos = np.asarray(routes["start_pos"], dtype=np.float64).reshape(-1, 2)  # (y,x)
    dest_pos = np.asarray(routes["dest_pos"], dtype=np.float64).reshape(-1, 2)  # (y,x)
    route_city = np.asarray(routes["route_city"], dtype=np.int64).reshape(-1)

    cities_obs = sorted(set(int(x) for x in route_city.tolist()))
    city_meta: Dict[int, dict] = {}
    city_meta_src: Dict[int, str] = {}
    for spec in list(args.city_grid_meta or []):
        c, path = _parse_city_kv(str(spec))
        if not path.exists():
            raise SystemExit(f"[FATAL] file not found: {path}")
        city_meta[int(c)] = _meta_from_city_grid_meta(path)
        city_meta_src[int(c)] = str(path)
    _require_city_meta(city_meta, cities_obs)

    seq = np.load(str(args.region_seq_npz), allow_pickle=True)
    need = {"route_id", "region_seq_ptr", "region_seq_idx"}
    missing = sorted(list(need - set(seq.files)))
    if missing:
        raise SystemExit(f"[FATAL] region_seq_npz missing keys: {missing}")
    route_id = np.asarray(seq["route_id"], dtype=np.int64).reshape(-1)
    ptr = np.asarray(seq["region_seq_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(seq["region_seq_idx"], dtype=np.int64).reshape(-1)

    K = int(route_id.size)
    if ptr.size != K + 1:
        raise SystemExit(f"[FATAL] region_seq_ptr shape mismatch: got {ptr.size}, expect {K+1}")

    # OD bin -> entry {n_routes, sigs{sig: {count,reps}}, city}
    table: Dict[Tuple[int, int, int, int, int], Dict[str, Any]] = {}
    skipped_short_od = 0
    skipped_oob = 0

    od_km_all: List[float] = []
    for i in range(K):
        rid = int(route_id[i])
        if rid < 0 or rid >= int(start_pos.shape[0]):
            skipped_oob += 1
            continue

        city = int(route_city[rid])
        meta = city_meta.get(int(city))
        if meta is None:
            raise SystemExit(f"[FATAL] missing city meta for city={city}")

        oy, ox = float(start_pos[rid, 0]), float(start_pos[rid, 1])
        dy, dx = float(dest_pos[rid, 0]), float(dest_pos[rid, 1])
        o_lat, o_lon = _grid_yx_to_latlon(oy, ox, meta=meta)
        d_lat, d_lon = _grid_yx_to_latlon(dy, dx, meta=meta)

        od_km = float(haversine_m(o_lat, o_lon, d_lat, d_lon)) / 1000.0
        od_km_all.append(float(od_km))
        if float(od_km) < float(cfg.min_od_km):
            skipped_short_od += 1
            continue

        o_lon_bin = _bin_int(o_lon, cfg.od_bin_deg)
        o_lat_bin = _bin_int(o_lat, cfg.od_bin_deg)
        d_lon_bin = _bin_int(d_lon, cfg.od_bin_deg)
        d_lat_bin = _bin_int(d_lat, cfg.od_bin_deg)
        od_key = (int(o_lon_bin), int(o_lat_bin), int(d_lon_bin), int(d_lat_bin), int(city))

        s = int(ptr[i])
        e = int(ptr[i + 1])
        sig = tuple(int(x) for x in idx[s:e].tolist())

        ent = table.get(od_key)
        if ent is None:
            ent = {"n_routes": 0, "sigs": {}}
            table[od_key] = ent
        ent["n_routes"] = int(ent["n_routes"]) + 1
        _update_sig_table(
            sigs=ent["sigs"],
            sig=sig,
            route_id=rid,
            max_sigs=cfg.max_sigs_per_bin,
            max_rep=cfg.max_rep_routes_per_sig,
        )

    # Summaries
    n_bins = int(len(table))
    routes_per_bin = np.asarray([int(v["n_routes"]) for v in table.values()], dtype=np.int64)
    bins_ge_min = [k for k, v in table.items() if int(v["n_routes"]) >= int(cfg.min_routes_per_bin)]

    uniq_per_bin = np.asarray([int(len(table[k]["sigs"])) for k in bins_ge_min], dtype=np.int64)
    multimodal_bins = [k for k in bins_ge_min if int(len(table[k]["sigs"])) >= 2]

    # LCS separation among top patterns (by count)
    separated_bins = 0
    sep_score_by_bin: Dict[Tuple[int, int, int, int, int], float] = {}
    for k in multimodal_bins:
        sigs: Dict[Tuple[int, ...], Dict[str, Any]] = table[k]["sigs"]
        # Sort patterns by count desc
        pats = sorted(sigs.items(), key=lambda kv: int(kv[1]["count"]), reverse=True)
        pats = pats[: int(cfg.lcs_max_patterns)]
        best = 0.0
        for i in range(len(pats)):
            for j in range(i + 1, len(pats)):
                d = _lcs_dist(pats[i][0], pats[j][0])
                if d > best:
                    best = float(d)
        sep_score_by_bin[k] = float(best)
        if float(best) >= float(cfg.lcs_sep_thr):
            separated_bins += 1

    # Prepare top bins for output (multimodal first, then by n_routes)
    def rank_key(k: Tuple[int, int, int, int, int]) -> Tuple[int, int]:
        return (int(len(table[k]["sigs"])), int(table[k]["n_routes"]))

    top_bins = sorted(multimodal_bins, key=rank_key, reverse=True)[: int(cfg.max_out_bins)]
    out_bins: List[Dict[str, Any]] = []
    for k in top_bins:
        o_lon_bin, o_lat_bin, d_lon_bin, d_lat_bin, city = k
        sigs = table[k]["sigs"]
        pats = sorted(sigs.items(), key=lambda kv: int(kv[1]["count"]), reverse=True)
        out_bins.append(
            {
                "od_bin": {"o_lon": int(o_lon_bin), "o_lat": int(o_lat_bin), "d_lon": int(d_lon_bin), "d_lat": int(d_lat_bin)},
                "city": int(city),
                "n_routes": int(table[k]["n_routes"]),
                "n_patterns": int(len(sigs)),
                "lcs_sep_score_max": float(sep_score_by_bin.get(k, float("nan"))),
                "patterns": [{"seq": list(map(int, sig)), "count": int(ent["count"]), "rep_route_ids": [int(x) for x in ent["reps"]]} for sig, ent in pats],
            }
        )

    # Per-city summary
    per_city: Dict[str, Any] = {}
    for city in sorted(set(int(k[-1]) for k in table.keys())):
        keys = [k for k in table.keys() if int(k[-1]) == int(city)]
        n_bins_c = int(len(keys))
        routes_per_bin_c = np.asarray([int(table[k]["n_routes"]) for k in keys], dtype=np.int64)
        bins_ge_min_c = [k for k in keys if int(table[k]["n_routes"]) >= int(cfg.min_routes_per_bin)]
        uniq_c = np.asarray([int(len(table[k]["sigs"])) for k in bins_ge_min_c], dtype=np.int64)
        multi_c = [k for k in bins_ge_min_c if int(len(table[k]["sigs"])) >= 2]
        per_city[str(int(city))] = {
            "n_od_bins": int(n_bins_c),
            "routes_per_bin": {"p50": _p(routes_per_bin_c, 50), "p90": _p(routes_per_bin_c, 90), "p95": _p(routes_per_bin_c, 95), "max": int(routes_per_bin_c.max()) if routes_per_bin_c.size else 0},
            "n_bins_ge_min_routes": int(len(bins_ge_min_c)),
            "multimodal_bins": {
                "n_bins_multi": int(len(multi_c)),
                "frac_multi_over_ge_min": float(len(multi_c) / max(1, len(bins_ge_min_c))),
            },
            "uniq_patterns_per_bin_ge_min": {"p50": _p(uniq_c, 50), "p90": _p(uniq_c, 90), "p95": _p(uniq_c, 95), "max": int(uniq_c.max()) if uniq_c.size else 0},
        }

    rep: Dict[str, Any] = {
        "ok": True,
        "task": "scan_multimodal_od_bin_region_seq",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "region_seq_npz": str(args.region_seq_npz),
            "city_grid_meta": city_meta_src,
        },
        "n_routes_in_region_seq_npz": int(K),
        "skipped_short_od": int(skipped_short_od),
        "skipped_oob": int(skipped_oob),
        "od_km": {
            "units": "km",
            "p50": _p(np.asarray(od_km_all, dtype=np.float64), 50) if od_km_all else float("nan"),
            "p90": _p(np.asarray(od_km_all, dtype=np.float64), 90) if od_km_all else float("nan"),
        },
        "od_bins": {
            "n_bins": int(n_bins),
            "routes_per_bin": {"p50": _p(routes_per_bin, 50), "p90": _p(routes_per_bin, 90), "p95": _p(routes_per_bin, 95), "max": int(routes_per_bin.max()) if routes_per_bin.size else 0},
            "n_bins_ge_min_routes": int(len(bins_ge_min)),
        },
        "multimodal": {
            "n_bins_multi": int(len(multimodal_bins)),
            "frac_multi_over_ge_min": float(len(multimodal_bins) / max(1, len(bins_ge_min))),
            "n_bins_sep_lcs": int(separated_bins),
            "frac_sep_over_multi": float(separated_bins / max(1, len(multimodal_bins))),
        },
        "uniq_patterns_per_bin_ge_min": {"p50": _p(uniq_per_bin, 50), "p90": _p(uniq_per_bin, 90), "p95": _p(uniq_per_bin, 95), "max": int(uniq_per_bin.max()) if uniq_per_bin.size else 0},
        "per_city": per_city,
        "top_multimodal_bins": out_bins,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {out_json}")


if __name__ == "__main__":
    main()
