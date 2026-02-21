from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TZ_SHANGHAI = timezone(timedelta(hours=8))


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


def _fallback_flag(dec: Dict[str, Any]) -> Optional[bool]:
    if "n_samples" not in dec:
        return None
    try:
        n_samples = int(dec.get("n_samples", 0))
    except Exception:
        return None
    if n_samples <= 1:
        return None
    if "sample_select_fallback" in dec:
        return bool(dec.get("sample_select_fallback", False))
    ssr = dec.get("sample_success_rate", None)
    if ssr is None:
        return None
    try:
        return bool(float(ssr) <= 0.0)
    except Exception:
        return None


def _safe_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return float("nan")
    return float(num) / float(den)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze sample-select fallback frequency from per_route JSON.")
    ap.add_argument("--per_route_json", type=Path, required=True)
    ap.add_argument("--mode", choices=["greedy", "beam"], default="greedy")
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    obj = json.loads(args.per_route_json.read_text(encoding="utf-8"))
    rows = obj.get("per_route", [])
    if not isinstance(rows, list):
        raise SystemExit("[FATAL] per_route_json missing list field: per_route")

    bins: Dict[str, Dict[str, int]] = {b[2]: {"n": 0, "eligible": 0, "fallback": 0} for b in _hops_bins()}
    overall = {"n": 0, "eligible": 0, "fallback": 0}

    for r in rows:
        if not isinstance(r, dict):
            continue
        dec = r.get(str(args.mode), None)
        if not isinstance(dec, dict):
            continue
        hops = int(r.get("gt_hops", 0))
        lab = _bin_label(hops)
        bins[lab]["n"] += 1
        overall["n"] += 1
        fb = _fallback_flag(dec)
        if fb is None:
            continue
        bins[lab]["eligible"] += 1
        overall["eligible"] += 1
        if bool(fb):
            bins[lab]["fallback"] += 1
            overall["fallback"] += 1

    out = {
        "ok": True,
        "task": "waycasd_analyze_fallback_rate",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "per_route_json": str(args.per_route_json),
            "mode": str(args.mode),
        },
        "overall": {
            **overall,
            "fallback_rate": _safe_rate(int(overall["fallback"]), int(overall["eligible"])),
        },
        "per_bin": {
            k: {
                **v,
                "fallback_rate": _safe_rate(int(v["fallback"]), int(v["eligible"])),
            }
            for k, v in bins.items()
        },
        "notes": {
            "eligible": "records with n_samples>1 (or inferred from sample_success_rate).",
            "fallback": "selector fallback because no successful sample exists in K candidates.",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {args.out_json}")
    print(
        f"overall: eligible={int(overall['eligible'])} fallback={int(overall['fallback'])} "
        f"rate={_safe_rate(int(overall['fallback']), int(overall['eligible'])):.4f}"
    )


if __name__ == "__main__":
    main()

