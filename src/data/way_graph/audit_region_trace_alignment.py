from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _compress_consecutive(seq: Sequence[int]) -> List[int]:
    out: List[int] = []
    last: Optional[int] = None
    for x in seq:
        xx = int(x)
        if xx < 0:
            continue
        if last is None or int(xx) != int(last):
            out.append(int(xx))
            last = int(xx)
    return out


def _count_backtracks(seq: Sequence[int]) -> int:
    """
    Count immediate backtracks A->B->A on the *compressed* region sequence.
    """
    s = _compress_consecutive(seq)
    n = 0
    for i in range(2, len(s)):
        if int(s[i]) == int(s[i - 2]) and int(s[i]) != int(s[i - 1]):
            n += 1
    return int(n)


def _hops_bins() -> List[Tuple[int, Optional[int], str]]:
    return [
        (5, 10, "[5,10)"),
        (10, 20, "[10,20)"),
        (20, 30, "[20,30)"),
        (30, 40, "[30,40)"),
        (40, 60, "[40,60)"),
        (60, None, "[60,+)"),
    ]


def _bin_label(hops: int) -> str:
    hh = int(hops)
    for lo, hi, name in _hops_bins():
        if hh < int(lo):
            continue
        if hi is None or hh < int(hi):
            return str(name)
    return str(_hops_bins()[-1][2])


def _p(x: np.ndarray, q: float) -> float:
    if int(x.size) == 0:
        return float("nan")
    return float(np.percentile(x.astype(np.float64, copy=False), float(q)))


def _summarize(x: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(x), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if int(arr.size) == 0:
        return {"n": 0, "p25": float("nan"), "p50": float("nan"), "p75": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "p25": _p(arr, 25),
        "p50": _p(arr, 50),
        "p75": _p(arr, 75),
        "p95": _p(arr, 95),
        "max": float(np.max(arr)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit GT way_seq -> region_trace alignment (backtracks, compression, long-route stats).")
    ap.add_argument("--way_routes_npz", type=Path, required=True)
    ap.add_argument("--way_regions_npz", type=Path, required=True, help="Need key: way_region.")
    ap.add_argument("--min_hops", type=int, default=5)
    ap.add_argument("--max_way_len", type=int, default=160)
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    wr = np.load(str(Path(args.way_regions_npz)), allow_pickle=True)
    if "way_region" not in wr.files:
        raise SystemExit("[FATAL] way_regions_npz missing key: way_region")
    way_region = np.asarray(wr["way_region"], dtype=np.int64).reshape(-1)

    min_hops = int(args.min_hops)
    max_way_len = int(args.max_way_len)

    keep = (routes.way_seq_len >= (min_hops + 1)) & (routes.way_seq_len <= max_way_len)
    ids = np.nonzero(keep)[0].astype(np.int64, copy=False)

    # per-route stats
    recs: List[Dict[str, Any]] = []
    for rid in ids.tolist():
        L = int(routes.way_seq_len[int(rid)])
        s = int(routes.way_seq_ptr[int(rid)])
        e = s + L
        way_seq = routes.way_seq_idx[s:e].astype(np.int64, copy=False)
        reg_trace = []
        for w in way_seq.tolist():
            wi = int(w)
            if 0 <= wi < int(way_region.size):
                reg_trace.append(int(way_region[wi]))
        reg_seq = _compress_consecutive(reg_trace)
        reg_len = int(len(reg_seq))
        hops = int(max(0, L - 1))
        back_n = _count_backtracks(reg_seq)
        ratio = float(L) / float(reg_len) if reg_len > 0 else float("inf")
        recs.append(
            {
                "route_id": int(rid),
                "city": int(routes.route_city[int(rid)]),
                "gt_hops": int(hops),
                "way_len": int(L),
                "region_seq_len": int(reg_len),
                "compression_ratio": float(ratio),
                "backtracks": int(back_n),
                "has_backtrack": bool(back_n > 0),
            }
        )

    def agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        reg_lens = [float(r["region_seq_len"]) for r in rows]
        ratios = [float(r["compression_ratio"]) for r in rows if np.isfinite(float(r["compression_ratio"]))]
        back = [1.0 if bool(r["has_backtrack"]) else 0.0 for r in rows]
        return {
            "n": int(len(rows)),
            "region_seq_len": _summarize(reg_lens),
            "compression_ratio_way_over_region": _summarize(ratios),
            "backtrack_rate": float(np.mean(np.asarray(back, dtype=np.float64))) if rows else float("nan"),
        }

    # by city + bins
    per_city: Dict[str, Any] = {}
    for city in sorted(set(int(r["city"]) for r in recs)):
        rows_c = [r for r in recs if int(r["city"]) == int(city)]
        out_c: Dict[str, Any] = {"overall": agg(rows_c), "bins": {}}
        for _lo, _hi, name in _hops_bins():
            rows_b = [r for r in rows_c if _bin_label(int(r["gt_hops"])) == str(name)]
            out_c["bins"][str(name)] = agg(rows_b)
        per_city[str(int(city))] = out_c

    out = {
        "ok": True,
        "task": "audit_region_trace_alignment",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": {"min_hops": int(min_hops), "max_way_len": int(max_way_len), "bins": [b[2] for b in _hops_bins()]},
        "inputs": {"way_routes_npz": str(args.way_routes_npz), "way_regions_npz": str(args.way_regions_npz)},
        "overall": agg(recs),
        "per_city": per_city,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()

