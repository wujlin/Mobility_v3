from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_bin(spec: str) -> Tuple[int, Optional[int]]:
    s = str(spec or "").strip()
    if not s:
        raise ValueError("empty hops_bin")
    if s.startswith("[") and s.endswith(")"):
        inner = s[1:-1]
    else:
        inner = s
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) != 2:
        raise ValueError(f"bad hops_bin: {spec!r}")
    lo = int(parts[0])
    hi_s = parts[1]
    if hi_s in {"+", "+inf", "inf", "None", ""} or ("+" in hi_s):
        return lo, None
    return lo, int(hi_s)


def _in_bin(hops: int, lo: int, hi: Optional[int]) -> bool:
    if int(hops) < int(lo):
        return False
    if hi is None:
        return True
    return int(hops) < int(hi)


def main() -> None:
    p = argparse.ArgumentParser(description="Extract route_ids from per_route json for hard-mining/curriculum.")
    p.add_argument("--per_route_json", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--key", type=str, default="beam", choices=["greedy", "beam"])
    p.add_argument("--only_fail", action="store_true", help="Keep only routes with success=false.")
    p.add_argument("--only_hit_wall", action="store_true", help="Keep only routes with hit_wall=true.")
    p.add_argument("--only_loop", action="store_true", help="Keep only routes with has_loop=true.")
    p.add_argument("--hops_bin", type=str, default="", help='Optional filter, e.g. "[40,60)" or "[60,+)".')
    p.add_argument("--max_n", type=int, default=0, help="0=keep all; >0=cap number of ids.")
    args = p.parse_args()

    rep = _read_json(Path(args.per_route_json))
    per_route = rep.get("per_route", rep.get("routes", None))
    if not isinstance(per_route, list):
        raise SystemExit("[FATAL] per_route_json missing list field: per_route")

    lo = hi = None
    if str(args.hops_bin).strip():
        lo, hi = _parse_bin(str(args.hops_bin))

    out_ids: List[int] = []
    key = str(args.key)
    for r in per_route:
        if not isinstance(r, dict) or r.get("route_id") is None:
            continue
        rid = int(r["route_id"])
        hops = int(r.get("gt_hops", 0))
        if lo is not None and not _in_bin(hops, int(lo), hi):
            continue

        m = r.get(key, None)
        if not isinstance(m, dict):
            continue
        succ = bool(m.get("success", False))
        hit_wall = bool(m.get("hit_wall", False))
        loop = bool(m.get("has_loop", False))
        if bool(args.only_fail) and succ:
            continue
        if bool(args.only_hit_wall) and (not hit_wall):
            continue
        if bool(args.only_loop) and (not loop):
            continue
        out_ids.append(rid)

    if int(args.max_n) > 0:
        out_ids = out_ids[: int(args.max_n)]

    out = {
        "ok": True,
        "task": "waycasd_extract_route_ids",
        "inputs": {
            "per_route_json": str(args.per_route_json),
            "key": str(args.key),
            "only_fail": bool(args.only_fail),
            "only_hit_wall": bool(args.only_hit_wall),
            "only_loop": bool(args.only_loop),
            "hops_bin": (str(args.hops_bin) if str(args.hops_bin).strip() else None),
            "max_n": int(args.max_n),
        },
        "n": int(len(out_ids)),
        "route_ids": [int(x) for x in out_ids],
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved: {args.out_json} (n={len(out_ids)})")


if __name__ == "__main__":
    main()

