from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bin_label(hops: int) -> str:
    h = int(hops)
    if 5 <= h < 10:
        return "[5,10)"
    if 10 <= h < 20:
        return "[10,20)"
    if 20 <= h < 30:
        return "[20,30)"
    if 30 <= h < 40:
        return "[30,40)"
    if 40 <= h < 60:
        return "[40,60)"
    if h >= 60:
        return "[60,+)"
    return "<5"


def _mcnemar_exact_p(n01: int, n10: int) -> float:
    """
    Exact two-sided McNemar p-value (binomial test under H0: p=0.5).
    """
    n01 = int(n01)
    n10 = int(n10)
    n = int(n01 + n10)
    if n <= 0:
        return float("nan")
    k = int(min(n01, n10))
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(n, i) * (0.5**n)
    return float(min(1.0, 2.0 * p))


def _safe_f(x: object) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _bootstrap_ci_median(x: np.ndarray, *, n_boot: int, seed: int = 0) -> Tuple[float, float]:
    """
    Bootstrap 95% CI for median(x).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(int(seed))
    n = int(x.size)
    meds = np.empty((int(n_boot),), dtype=np.float64)
    for i in range(int(n_boot)):
        samp = x[rng.integers(0, n, size=n)]
        meds[i] = np.median(samp)
    return (float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975)))


def _collect(per_route: Sequence[Dict[str, Any]], *, key: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for r in per_route:
        rid = int(r.get("route_id", -1))
        if rid < 0:
            continue
        out[rid] = r
    if out:
        sample = next(iter(out.values()))
        if key not in sample:
            raise SystemExit(f"[FATAL] per_route missing key={key!r}; sample keys={sorted(sample.keys())}")
    return out


def _fmt_pct(x: float) -> str:
    return f"{float(x)*100:.1f}%" if np.isfinite(float(x)) else "nan"


def main() -> None:
    p = argparse.ArgumentParser(description="Paired comparison (McNemar + shape deltas) from per_route json dumps.")
    p.add_argument("--a_json", type=Path, required=True)
    p.add_argument("--b_json", type=Path, required=True)
    p.add_argument("--a_name", type=str, default="A")
    p.add_argument("--b_name", type=str, default="B")
    p.add_argument("--key", choices=["greedy", "beam"], default="beam")
    p.add_argument("--n_boot", type=int, default=200, help="Bootstrap replicates for CI of median(Δ) (<=0 disables).")
    p.add_argument("--out_md", type=Path, default=None)
    args = p.parse_args()

    a = _read_json(Path(args.a_json))
    b = _read_json(Path(args.b_json))
    a_pr = a.get("per_route", [])
    b_pr = b.get("per_route", [])
    if not isinstance(a_pr, list) or not isinstance(b_pr, list):
        raise SystemExit("[FATAL] per_route missing or not a list in inputs.")

    key = str(args.key)
    a_map = _collect(a_pr, key=key)
    b_map = _collect(b_pr, key=key)

    rids_a = set(a_map.keys())
    rids_b = set(b_map.keys())
    rids = sorted(rids_a & rids_b)
    if not rids:
        raise SystemExit("[FATAL] no overlapping route_id between a_json and b_json.")
    if rids_a != rids_b:
        print(
            f"[WARN] route_id mismatch: only_in_a={len(rids_a - rids_b)} only_in_b={len(rids_b - rids_a)} "
            f"using_intersection={len(rids)}",
            flush=True,
        )

    # Group rids by gt_hops bins.
    groups: Dict[str, List[int]] = {"overall": []}
    for lab in ["[5,10)", "[10,20)", "[20,30)", "[30,40)", "[40,60)", "[60,+)"]:
        groups[lab] = []
    for rid in rids:
        hops = int(a_map[rid].get("gt_hops", 0))
        lab = _bin_label(hops)
        groups["overall"].append(rid)
        if lab in groups:
            groups[lab].append(rid)

    def _succ(rec: Dict[str, Any]) -> bool:
        m = rec.get(key, {})
        return bool(m.get("success", False)) if isinstance(m, dict) else False

    def _metric(rec: Dict[str, Any], name: str) -> float:
        m = rec.get(key, {})
        if not isinstance(m, dict):
            return float("nan")
        return _safe_f(m.get(name, float("nan")))

    # Markdown report.
    lines: List[str] = []
    lines.append(f"# Paired Compare ({args.a_name} vs {args.b_name})")
    lines.append("")
    lines.append(f"- a_json: `{args.a_json}`")
    lines.append(f"- b_json: `{args.b_json}`")
    lines.append(f"- key: `{key}`")
    lines.append(f"- n_routes (intersection): `{len(rids)}`")
    lines.append("")

    lines.append("|Bin|n|succ(A)|succ(B)|Δsucc(B-A)|n01(A0,B1)|n10(A1,B0)|p(McNemar)|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for lab in ["overall", "[5,10)", "[10,20)", "[20,30)", "[30,40)", "[40,60)", "[60,+)"]:
        ids = groups.get(lab, [])
        if not ids:
            continue
        a_s = np.asarray([1.0 if _succ(a_map[r]) else 0.0 for r in ids], dtype=np.float64)
        b_s = np.asarray([1.0 if _succ(b_map[r]) else 0.0 for r in ids], dtype=np.float64)
        succ_a = float(np.mean(a_s)) if a_s.size else float("nan")
        succ_b = float(np.mean(b_s)) if b_s.size else float("nan")
        n01 = int(np.sum((a_s < 0.5) & (b_s > 0.5)))
        n10 = int(np.sum((a_s > 0.5) & (b_s < 0.5)))
        pval = _mcnemar_exact_p(n01, n10)
        lines.append(
            f"|{lab}|{int(a_s.size)}|{_fmt_pct(succ_a)}|{_fmt_pct(succ_b)}|{(succ_b - succ_a)*100:+.1f}pp|{n01}|{n10}|{pval:.4f}|"
        )

    lines.append("")
    lines.append("## Shape（仅在 A 与 B 同时成功的 route 上做配对）")
    lines.append("")

    shape_metrics: List[Tuple[str, str]] = [
        ("frechet_m", "↓ Fréchet(m)"),
        ("dtw_m", "↓ DTW(m)"),
        ("final_error_m", "↓ FinalErr(m)"),
        ("len_ratio", "→ LenRatio(=1)"),
    ]

    for lab in ["overall", "[40,60)", "[60,+)"]:
        ids = groups.get(lab, [])
        both = [r for r in ids if _succ(a_map[r]) and _succ(b_map[r])]
        lines.append(f"- {lab}: n_pair_success={len(both)}")
        if not both:
            continue
        for m, title in shape_metrics:
            xa = np.asarray([_metric(a_map[r], m) for r in both], dtype=np.float64)
            xb = np.asarray([_metric(b_map[r], m) for r in both], dtype=np.float64)
            mask = np.isfinite(xa) & np.isfinite(xb)
            xa = xa[mask]
            xb = xb[mask]
            if xa.size == 0:
                lines.append(f"  - {title}: n=0")
                continue

            if m == "len_ratio":
                ea = np.abs(xa - 1.0)
                eb = np.abs(xb - 1.0)
                diff = eb - ea  # <0 means B closer to 1
                msg = (
                    f"  - {title}: median(|A-1|)={float(np.median(ea)):.3f}, "
                    f"median(|B-1|)={float(np.median(eb)):.3f}, "
                    f"median(Δ=B-A)={float(np.median(diff)):+.3f}, "
                    f"frac(B better)={float(np.mean(diff < 0.0)):.2f}"
                )
            else:
                diff = xb - xa  # <0 means B smaller/better
                msg = (
                    f"  - {title}: median(A)={float(np.median(xa)):.1f}, "
                    f"median(B)={float(np.median(xb)):.1f}, "
                    f"median(Δ=B-A)={float(np.median(diff)):+.1f}, "
                    f"frac(B better)={float(np.mean(diff < 0.0)):.2f}"
                )

            if int(args.n_boot) > 0 and xa.size >= 8:
                lo, hi = _bootstrap_ci_median(diff, n_boot=int(args.n_boot), seed=0)
                msg += f", CI95%(median Δ)=[{lo:+.1f},{hi:+.1f}]"
            lines.append(msg)

    md = "\n".join(lines) + "\n"
    if args.out_md is not None:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"[OK] saved: {out}")
    else:
        print(md)


if __name__ == "__main__":
    main()

