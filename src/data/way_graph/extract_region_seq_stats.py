from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    min_hops: int
    max_way_len: int
    compress_consecutive: bool


def _p(x: np.ndarray, q: float) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, float(q)))


def _compress_consecutive(seq: np.ndarray) -> np.ndarray:
    s = np.asarray(seq, dtype=np.int64).reshape(-1)
    if s.size <= 1:
        return s
    keep = np.ones((int(s.size),), dtype=bool)
    keep[1:] = s[1:] != s[:-1]
    return s[keep]


def _encode_seq(seq: np.ndarray) -> str:
    # Stable encoding for "unique pattern" counting.
    return ",".join(str(int(x)) for x in np.asarray(seq, dtype=np.int64).reshape(-1).tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract region_seq from GT way_seq and compute corridor diversity stats.")
    ap.add_argument("--way_routes_npz", type=Path, required=True)
    ap.add_argument("--way_regions_npz", type=Path, required=True, help="Output of build_way_regions_louvain_per_city.")
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--out_npz", type=Path, default=None, help="Optional: save region_seq.npz (CSR) for later modeling.")
    ap.add_argument("--min_hops", type=int, default=5)
    ap.add_argument("--max_way_len", type=int, default=160)
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--compress_consecutive",
        action="store_true",
        help="Compress consecutive same-region steps (default behavior).",
    )
    g.add_argument(
        "--no_compress_consecutive",
        action="store_true",
        help="If set, do NOT compress consecutive same-region steps.",
    )
    args = ap.parse_args()

    cfg = Cfg(
        min_hops=int(args.min_hops),
        max_way_len=int(args.max_way_len),
        compress_consecutive=(not bool(args.no_compress_consecutive)),
    )

    routes_npz = np.load(str(Path(args.way_routes_npz)), allow_pickle=True)
    need = {"way_seq_ptr", "way_seq_idx", "way_seq_len", "start_way", "dest_way", "route_city"}
    missing = sorted(list(need - set(routes_npz.files)))
    if missing:
        raise SystemExit(f"[FATAL] way_routes_npz missing keys: {missing}")
    way_seq_ptr = np.asarray(routes_npz["way_seq_ptr"], dtype=np.int64).reshape(-1)
    way_seq_idx = np.asarray(routes_npz["way_seq_idx"], dtype=np.int64).reshape(-1)
    way_seq_len = np.asarray(routes_npz["way_seq_len"], dtype=np.int64).reshape(-1)
    start_way = np.asarray(routes_npz["start_way"], dtype=np.int64).reshape(-1)
    dest_way = np.asarray(routes_npz["dest_way"], dtype=np.int64).reshape(-1)
    route_city = np.asarray(routes_npz["route_city"], dtype=np.int64).reshape(-1)
    reg_data = np.load(str(args.way_regions_npz), allow_pickle=True)
    if "way_region" not in reg_data.files:
        raise SystemExit("[FATAL] way_regions_npz missing key: way_region")
    way_region = np.asarray(reg_data["way_region"], dtype=np.int64).reshape(-1)

    # Filter routes by hops/len (same as training/eval filtering).
    keep = way_seq_len >= (int(cfg.min_hops) + 1)
    keep &= way_seq_len <= int(cfg.max_way_len)
    rids = np.nonzero(keep)[0].astype(np.int64, copy=False)

    region_seq_lens: List[int] = []
    has_repeat: List[int] = []
    patterns_by_od: Dict[Tuple[int, int, int], set] = {}
    lens_by_city: Dict[int, List[int]] = {}
    rep_by_city: Dict[int, List[int]] = {}

    # Optional CSR output
    out_ptr = np.zeros((int(rids.size) + 1,), dtype=np.int64)
    flat: List[int] = []

    n_missing_region = 0
    for i, rid in enumerate(rids.tolist()):
        L = int(way_seq_len[int(rid)])
        s = int(way_seq_ptr[int(rid)])
        e = s + L
        way_seq = way_seq_idx[s:e].astype(np.int64, copy=False)

        reg_seq = way_region[way_seq]
        if bool(cfg.compress_consecutive):
            reg_seq = _compress_consecutive(reg_seq)

        # If any -1 appears, keep it (honest), but count it for debugging.
        if int(np.sum(reg_seq < 0)) > 0:
            n_missing_region += 1

        region_seq_lens.append(int(reg_seq.size))
        rep = 1 if int(np.unique(reg_seq).size) < int(reg_seq.size) else 0
        has_repeat.append(int(rep))

        city = int(route_city[int(rid)])
        lens_by_city.setdefault(int(city), []).append(int(reg_seq.size))
        rep_by_city.setdefault(int(city), []).append(int(rep))
        od = (int(start_way[int(rid)]), int(dest_way[int(rid)]), int(city))
        key = _encode_seq(reg_seq)
        patterns_by_od.setdefault(od, set()).add(key)

        out_ptr[i + 1] = out_ptr[i] + int(reg_seq.size)
        flat.extend(int(x) for x in reg_seq.tolist())

    lens = np.asarray(region_seq_lens, dtype=np.int64)
    rep_rate = float(np.mean(np.asarray(has_repeat, dtype=np.float64))) if has_repeat else float("nan")

    # Diversity stats by OD
    uniq_counts = np.asarray([len(v) for v in patterns_by_od.values()], dtype=np.int64)
    od_multi_frac = float(np.mean(uniq_counts >= 2)) if uniq_counts.size else float("nan")

    per_city: Dict[str, Any] = {}
    for city, lens_list in sorted(lens_by_city.items(), key=lambda kv: kv[0]):
        lens_c = np.asarray(lens_list, dtype=np.int64)
        rep_c = np.asarray(rep_by_city.get(int(city), []), dtype=np.float64)
        # OD patterns for this city (od key includes city)
        uniq_c = np.asarray([len(v) for (o, d, c), v in patterns_by_od.items() if int(c) == int(city)], dtype=np.int64)
        per_city[str(int(city))] = {
            "n_routes": int(lens_c.size),
            "region_seq_len": {"p25": _p(lens_c, 25), "p50": _p(lens_c, 50), "p75": _p(lens_c, 75), "p95": _p(lens_c, 95), "max": int(lens_c.max()) if lens_c.size else 0},
            "backtrack_rate": float(np.mean(rep_c)) if rep_c.size else float("nan"),
            "od_diversity": {
                "n_od": int(uniq_c.size),
                "od_multi_frac": float(np.mean(uniq_c >= 2)) if uniq_c.size else float("nan"),
                "uniq_patterns_per_od": {
                    "p50": int(np.quantile(uniq_c, 0.50)) if uniq_c.size else 0,
                    "p90": int(np.quantile(uniq_c, 0.90)) if uniq_c.size else 0,
                    "p95": int(np.quantile(uniq_c, 0.95)) if uniq_c.size else 0,
                    "max": int(uniq_c.max()) if uniq_c.size else 0,
                },
            },
        }

    rep: Dict[str, Any] = {
        "ok": True,
        "task": "extract_region_seq_stats",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {"way_routes_npz": str(args.way_routes_npz), "way_regions_npz": str(args.way_regions_npz)},
        "n_routes_total": int(way_seq_len.size),
        "n_routes_kept": int(rids.size),
        "n_routes_with_missing_region": int(n_missing_region),
        "region_seq_len": {"p25": _p(lens, 25), "p50": _p(lens, 50), "p75": _p(lens, 75), "p95": _p(lens, 95), "max": int(lens.max()) if lens.size else 0},
        "backtrack_rate": float(rep_rate),
        "od_diversity": {
            "n_od": int(uniq_counts.size),
            "od_multi_frac": float(od_multi_frac),
            "uniq_patterns_per_od": {
                "p50": int(np.quantile(uniq_counts, 0.50)) if uniq_counts.size else 0,
                "p90": int(np.quantile(uniq_counts, 0.90)) if uniq_counts.size else 0,
                "p95": int(np.quantile(uniq_counts, 0.95)) if uniq_counts.size else 0,
                "max": int(uniq_counts.max()) if uniq_counts.size else 0,
            },
        },
        "per_city": per_city,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {out_json}")

    if args.out_npz is not None:
        out_npz = Path(args.out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        region_seq_idx = np.asarray(flat, dtype=np.int32)
        region_seq_ptr = out_ptr.astype(np.int64, copy=False)
        region_seq_len = (region_seq_ptr[1:] - region_seq_ptr[:-1]).astype(np.int32, copy=False)
        route_id = rids.astype(np.int64, copy=False)
        np.savez_compressed(
            str(out_npz),
            route_id=route_id,
            region_seq_ptr=region_seq_ptr,
            region_seq_idx=region_seq_idx,
            region_seq_len=region_seq_len,
            meta={
                "task": "extract_region_seq_stats",
                "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
                "cfg": asdict(cfg),
                "inputs": {"way_routes_npz": str(args.way_routes_npz), "way_regions_npz": str(args.way_regions_npz)},
            },
        )
        print(f"[OK] saved: {out_npz}")


if __name__ == "__main__":
    main()
