from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _get_p50(summary: object) -> float:
    if not isinstance(summary, dict):
        return float("nan")
    return _f(summary.get("p50", float("nan")))


def _fmt(x: float, *, digits: int = 0) -> str:
    if not math.isfinite(float(x)):
        return "nan"
    q = 10**int(digits)
    return str(int(round(float(x) * q)) / q)


def _extract_way_casd(obj: dict, *, city: Optional[int]) -> Dict[str, Any]:
    if city is None:
        return obj.get("overall", {}) if isinstance(obj.get("overall", {}), dict) else {}
    for c in obj.get("per_city", []) if isinstance(obj.get("per_city", []), list) else []:
        if isinstance(c, dict) and int(c.get("city", -1)) == int(city):
            return c
    return {}


def _extract_sp(obj: dict, *, city: Optional[int]) -> Dict[str, Any]:
    if city is None:
        return obj.get("overall_by_gt_hops", {}) if isinstance(obj.get("overall_by_gt_hops", {}), dict) else {}
    for c in obj.get("per_city", []) if isinstance(obj.get("per_city", []), list) else []:
        if isinstance(c, dict) and int(c.get("city", -1)) == int(city):
            return c.get("by_gt_hops", {}) if isinstance(c.get("by_gt_hops", {}), dict) else {}
    return {}


def _cells(obj: dict) -> Dict[str, dict]:
    cells = obj.get("cells", {}) if isinstance(obj, dict) else {}
    return cells if isinstance(cells, dict) else {}


def _md_table(title: str, rows: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append(
        "| bin | n | SP dtw p50 (m) | Greedy dtw p50 | Beam10 dtw p50 | SP frechet p50 (m) | Greedy frechet p50 | Beam10 frechet p50 |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {bin} | {n} | {sp_dtw} | {g_dtw} | {b_dtw} | {sp_fre} | {g_fre} | {b_fre} |".format(
                bin=str(r["bin"]),
                n=int(r.get("n", 0)),
                sp_dtw=_fmt(float(r.get("sp_dtw_p50", float("nan"))), digits=0),
                g_dtw=_fmt(float(r.get("greedy_dtw_p50", float("nan"))), digits=0),
                b_dtw=_fmt(float(r.get("beam_dtw_p50", float("nan"))), digits=0),
                sp_fre=_fmt(float(r.get("sp_frechet_p50", float("nan"))), digits=0),
                g_fre=_fmt(float(r.get("greedy_frechet_p50", float("nan"))), digits=0),
                b_fre=_fmt(float(r.get("beam_frechet_p50", float("nan"))), digits=0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Way-CASD vs Shortest-Path baseline on shape metrics (meters), aligned by gt_hops bins.")
    ap.add_argument("--way_casd_binned_json", type=Path, required=True)
    ap.add_argument("--sp_baseline_json", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--out_md", type=Path, default=None)
    args = ap.parse_args()

    w = _read_json(Path(args.way_casd_binned_json))
    sp = _read_json(Path(args.sp_baseline_json))

    # Use bins from Way-CASD output as source of truth.
    bins = []
    overall = w.get("overall", {}) if isinstance(w.get("overall", {}), dict) else {}
    greedy = overall.get("greedy", {}) if isinstance(overall.get("greedy", {}), dict) else {}
    if isinstance(greedy.get("bins", []), list):
        bins = [str(x) for x in greedy.get("bins", [])]
    if not bins:
        raise SystemExit("[FATAL] cannot infer bins from way_casd_binned_json")

    def _one(city: Optional[int]) -> Dict[str, Any]:
        w_city = _extract_way_casd(w, city=city)
        sp_city = _extract_sp(sp, city=city)

        w_g = w_city.get("greedy", {}) if isinstance(w_city.get("greedy", {}), dict) else {}
        w_b = w_city.get("beam", {}) if isinstance(w_city.get("beam", {}), dict) else {}
        sp_cells = _cells(sp_city)
        g_cells = _cells(w_g)
        b_cells = _cells(w_b)

        rows: List[Dict[str, Any]] = []
        for bname in bins:
            sp_cell = sp_cells.get(str(bname), {}) if isinstance(sp_cells.get(str(bname), {}), dict) else {}
            g_cell = g_cells.get(str(bname), {}) if isinstance(g_cells.get(str(bname), {}), dict) else {}
            b_cell = b_cells.get(str(bname), {}) if isinstance(b_cells.get(str(bname), {}), dict) else {}

            n = int(g_cell.get("n", sp_cell.get("n", 0))) if isinstance(g_cell, dict) else int(sp_cell.get("n", 0))
            rows.append(
                {
                    "bin": str(bname),
                    "n": int(n),
                    "sp_success_rate": _f(sp_cell.get("success_rate", float("nan"))),
                    "sp_dtw_p50": _get_p50(sp_cell.get("dtw_m", {})),
                    "sp_frechet_p50": _get_p50(sp_cell.get("frechet_m", {})),
                    "greedy_success_rate": _f(g_cell.get("success_rate", float("nan"))),
                    "greedy_dtw_p50": _get_p50(g_cell.get("dtw_m", {})),
                    "greedy_frechet_p50": _get_p50(g_cell.get("frechet_m", {})),
                    "beam_success_rate": _f(b_cell.get("success_rate", float("nan"))),
                    "beam_dtw_p50": _get_p50(b_cell.get("dtw_m", {})),
                    "beam_frechet_p50": _get_p50(b_cell.get("frechet_m", {})),
                    # Positive means "closer to GT than SP" (improvement)
                    "delta_dtw_sp_minus_greedy": _get_p50(sp_cell.get("dtw_m", {})) - _get_p50(g_cell.get("dtw_m", {})),
                    "delta_dtw_sp_minus_beam": _get_p50(sp_cell.get("dtw_m", {})) - _get_p50(b_cell.get("dtw_m", {})),
                    "delta_frechet_sp_minus_greedy": _get_p50(sp_cell.get("frechet_m", {})) - _get_p50(g_cell.get("frechet_m", {})),
                    "delta_frechet_sp_minus_beam": _get_p50(sp_cell.get("frechet_m", {})) - _get_p50(b_cell.get("frechet_m", {})),
                }
            )
        return {"city": city, "bins": bins, "rows": rows}

    per_city = [_one(0), _one(1)]
    overall_cmp = _one(None)

    out = {
        "ok": True,
        "task": "way_casd_vs_sp_shape_compare",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "way_casd_binned_json": str(args.way_casd_binned_json),
            "sp_baseline_json": str(args.sp_baseline_json),
        },
        "bins": bins,
        "per_city": per_city,
        "overall": overall_cmp,
        "notes": {
            "dtw_frechet": "Lower is better; deltas are computed as (SP p50 - Model p50), positive means model is closer to GT than shortest-path baseline.",
            "binning": "Aligned by gt_hops bins from way_casd_binned_json.",
        },
    }

    if args.out_json is not None:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] saved: {out_json}")

    if args.out_md is not None:
        chunks = []
        chunks.append(f"# Way-CASD vs Shortest-Path (shape metrics)\n")
        chunks.append(_md_table("Overall", overall_cmp["rows"]))
        for c in per_city:
            name = "Detroit" if int(c["city"]) == 0 else "Columbus" if int(c["city"]) == 1 else f"City{c['city']}"
            chunks.append(_md_table(name, c["rows"]))
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(chunks), encoding="utf-8")
        print(f"[OK] saved: {out_md}")


if __name__ == "__main__":
    main()

