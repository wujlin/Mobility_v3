from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bin_label(hops: int) -> str:
    hh = int(hops)
    bins = [
        (5, 10, "[5,10)"),
        (10, 20, "[10,20)"),
        (20, 30, "[20,30)"),
        (30, 40, "[30,40)"),
        (40, 60, "[40,60)"),
        (60, None, "[60,+)"),
    ]
    for lo, hi, name in bins:
        if hh < int(lo):
            continue
        if hi is None or hh < int(hi):
            return str(name)
    return str(bins[-1][2])


def _index_by_route(records: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for r in records:
        try:
            rid = int(r.get("route_id"))
        except Exception:
            continue
        out[int(rid)] = r
    return out


def _success(rec: Dict[str, Any], *, which: str) -> bool:
    m = rec.get(which, {})
    return bool(isinstance(m, dict) and bool(m.get("success", False)))


def _metric(rec: Dict[str, Any], *, which: str) -> Dict[str, Any]:
    m = rec.get(which, {})
    return m if isinstance(m, dict) else {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_json", type=Path, required=True, help="Per-route JSON from binned_eval baseline.")
    p.add_argument("--constrained_json", type=Path, required=True, help="Per-route JSON from binned_eval constrained.")
    p.add_argument("--gt_constrained_json", type=Path, default=None, help="Optional: per-route JSON from GT-region constrained.")
    p.add_argument("--city", type=int, default=1)
    p.add_argument("--bin", type=str, default="[60,+)")
    p.add_argument("--which", choices=["beam", "greedy"], default="beam")
    p.add_argument("--out_json", type=Path, default=None)
    args = p.parse_args()

    base = _read_json(Path(args.baseline_json))
    con = _read_json(Path(args.constrained_json))
    gt = _read_json(Path(args.gt_constrained_json)) if args.gt_constrained_json is not None else None

    base_idx = _index_by_route(list(base.get("per_route", [])))
    con_idx = _index_by_route(list(con.get("per_route", [])))
    gt_idx = _index_by_route(list(gt.get("per_route", []))) if isinstance(gt, dict) else None

    city = int(args.city)
    bin_label = str(args.bin)
    which = str(args.which)

    rows: List[Dict[str, Any]] = []
    for rid, br in base_idx.items():
        cr = con_idx.get(int(rid))
        if cr is None:
            continue
        if int(br.get("city", -1)) != int(city):
            continue
        if int(cr.get("city", -1)) != int(city):
            continue
        if _bin_label(int(br.get("gt_hops", 0))) != bin_label:
            continue

        base_ok = _success(br, which=which)
        con_ok = _success(cr, which=which)
        if base_ok and (not con_ok):
            item: Dict[str, Any] = {
                "route_id": int(rid),
                "city": int(city),
                "gt_hops": int(br.get("gt_hops", 0)),
                "bin": bin_label,
                "baseline": _metric(br, which=which),
                "constrained": _metric(cr, which=which),
                "constrained_region_seq": cr.get("region_seq"),
            }
            if gt_idx is not None and int(rid) in gt_idx:
                gr = gt_idx[int(rid)]
                item["gt_constrained_success"] = _success(gr, which=which)
                item["gt_constrained_region_seq"] = gr.get("region_seq")
            rows.append(item)

    rows.sort(key=lambda x: int(x["route_id"]))
    out = {
        "ok": True,
        "city": int(city),
        "bin": bin_label,
        "which": which,
        "n_total_in_bin": sum(
            1
            for r in base_idx.values()
            if int(r.get("city", -1)) == int(city) and _bin_label(int(r.get("gt_hops", 0))) == bin_label
        ),
        "n_baseline_success_constrained_fail": int(len(rows)),
        "rows": rows,
        "inputs": {
            "baseline_json": str(args.baseline_json),
            "constrained_json": str(args.constrained_json),
            "gt_constrained_json": (str(args.gt_constrained_json) if args.gt_constrained_json is not None else None),
        },
    }

    print(
        f"[OK] city={int(city)} bin={bin_label} which={which} "
        f"baseline_success_constrained_fail={int(len(rows))}/{int(out['n_total_in_bin'])}",
        flush=True,
    )

    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()

