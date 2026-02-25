from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass
class MethodSpec:
    label: str
    decode: str
    path: str


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_method_spec(spec: str) -> MethodSpec:
    if "=" not in spec:
        raise SystemExit(f"[FATAL] invalid --method spec: {spec!r}; expected LABEL|DECODE=PATH")
    left, right = spec.split("=", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise SystemExit(f"[FATAL] invalid --method spec: {spec!r}; empty label/decode/path")
    if "|" in left:
        label, decode = left.rsplit("|", 1)
    else:
        label, decode = left, "greedy"
    label = label.strip()
    decode = decode.strip().lower()
    if decode not in {"greedy", "beam"}:
        raise SystemExit(f"[FATAL] invalid decode mode in --method: {spec!r}; decode must be greedy/beam")
    return MethodSpec(label=label, decode=decode, path=right)


def _as_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _seq_jaccard(a: List[int], b: List[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    den = len(sa | sb)
    if den == 0:
        return 1.0
    return float(len(sa & sb) / float(den))


def _pairwise_diversity(seqs: List[List[int]]) -> float:
    m = len(seqs)
    if m < 2:
        return float("nan")
    vals: List[float] = []
    for i in range(m):
        for j in range(i + 1, m):
            vals.append(1.0 - _seq_jaccard(seqs[i], seqs[j]))
    return float(np.mean(np.asarray(vals, dtype=np.float64))) if vals else float("nan")


def _extract_per_route(root: Dict[str, Any]) -> List[Dict[str, Any]]:
    per_route = root.get("per_route")
    if not isinstance(per_route, list):
        raise SystemExit("[FATAL] input json missing per_route list; re-run eval with per-route dumping enabled.")
    return [r for r in per_route if isinstance(r, dict)]


def _extract_records(per_route: List[Dict[str, Any]], decode: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    miss_gt = 0
    miss_decode = 0
    miss_pred = 0
    for rec in per_route:
        gt = rec.get("gt_way_ids")
        if not isinstance(gt, list) or len(gt) == 0:
            miss_gt += 1
            continue
        pred_obj = rec.get(decode)
        if not isinstance(pred_obj, dict):
            miss_decode += 1
            continue
        pred = pred_obj.get("pred_way_ids")
        if not isinstance(pred, list) or len(pred) == 0:
            miss_pred += 1
            continue
        sw = rec.get("start_way", gt[0])
        dw = rec.get("dest_way", gt[-1])
        rows.append(
            {
                "route_id": _as_int(rec.get("route_id", -1), -1),
                "start_way": _as_int(sw, -1),
                "dest_way": _as_int(dw, -1),
                "gt_way_ids": [int(x) for x in gt],
                "pred_way_ids": [int(x) for x in pred],
                "success": bool(pred_obj.get("success", False)),
            }
        )
    rows.sort(key=lambda x: x["route_id"])
    if len(rows) == 0:
        raise SystemExit(
            "[FATAL] no valid route records found; "
            f"decode={decode}, missing_gt_way_ids={miss_gt}, missing_{decode}_block={miss_decode}, "
            f"missing_pred_way_ids={miss_pred}. Re-run eval with --dump_way_seqs and per-route dumping enabled."
        )
    return rows


def _mean(xs: List[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _pct(xs: List[float], q: float) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _analyze_method(rows: List[Dict[str, Any]], *, k: int, min_routes_per_od: int, jacc_th: float) -> Dict[str, Any]:
    od_groups: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in rows:
        key = (int(r["start_way"]), int(r["dest_way"]))
        g = od_groups.setdefault(key, {"gt": [], "pred_success": []})
        g["gt"].append(r["gt_way_ids"])
        if bool(r["success"]):
            g["pred_success"].append(r["pred_way_ids"])

    kept = {k0: g for k0, g in od_groups.items() if len(g["gt"]) >= int(min_routes_per_od)}

    coverage_vals: List[float] = []
    diversity_vals: List[float] = []
    n_no_success = 0
    n_div_valid = 0
    per_od_rows: List[Dict[str, Any]] = []

    for (sw, dw), g in kept.items():
        gt_list = g["gt"]
        pred_succ = g["pred_success"][: int(k)]
        if len(pred_succ) == 0:
            n_no_success += 1
        matched = 0
        for gt_seq in gt_list:
            ok = any(_seq_jaccard(gt_seq, pr) >= float(jacc_th) for pr in pred_succ)
            matched += 1 if ok else 0
        cov = float(matched / max(1, len(gt_list)))
        coverage_vals.append(cov)

        div = _pairwise_diversity(pred_succ)
        if math.isfinite(div):
            diversity_vals.append(div)
            n_div_valid += 1

        per_od_rows.append(
            {
                "start_way": int(sw),
                "dest_way": int(dw),
                "n_gt_routes": int(len(gt_list)),
                "n_pred_success_used": int(len(pred_succ)),
                "gt_coverage_at_k": float(cov),
                "self_diversity_at_k": (float(div) if math.isfinite(div) else None),
            }
        )

    total = len(rows)
    succ = sum(1 for r in rows if bool(r["success"]))
    out = {
        "n_routes": int(total),
        "arrival_rate": float(succ / max(1, total)),
        "n_od_groups_all": int(len(od_groups)),
        "n_od_groups_kept": int(len(kept)),
        "n_od_groups_no_success": int(n_no_success),
        "gt_coverage_at_k": {
            "mean": _mean(coverage_vals),
            "p25": _pct(coverage_vals, 25),
            "p50": _pct(coverage_vals, 50),
            "p75": _pct(coverage_vals, 75),
            "n": int(len(coverage_vals)),
        },
        "self_diversity_at_k": {
            "mean": _mean(diversity_vals),
            "p25": _pct(diversity_vals, 25),
            "p50": _pct(diversity_vals, 50),
            "p75": _pct(diversity_vals, 75),
            "n": int(n_div_valid),
        },
        "per_od": per_od_rows,
    }
    return out


def _parse_tau_values(spec: str) -> List[float]:
    s = str(spec or "").strip()
    if not s:
        return [round(0.1 * i, 1) for i in range(1, 10)]
    vals: List[float] = []
    for tok in s.split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            x = float(t)
        except Exception:
            raise SystemExit(f"[FATAL] invalid tau in --tau_values: {tok!r}") from None
        if x < 0.0 or x > 1.0:
            raise SystemExit(f"[FATAL] tau must be in [0,1], got {x}")
        vals.append(float(x))
    if not vals:
        raise SystemExit("[FATAL] --tau_values parsed empty list.")
    # unique + sorted
    vals = sorted(set(vals))
    return vals


def _analyze_method_threshold_free(
    rows: List[Dict[str, Any]], *, k: int, min_routes_per_od: int, tau_values: List[float]
) -> Dict[str, Any]:
    od_groups: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in rows:
        key = (int(r["start_way"]), int(r["dest_way"]))
        g = od_groups.setdefault(key, {"gt": [], "pred_success": []})
        g["gt"].append(r["gt_way_ids"])
        if bool(r["success"]):
            g["pred_success"].append(r["pred_way_ids"])

    kept = {k0: g for k0, g in od_groups.items() if len(g["gt"]) >= int(min_routes_per_od)}

    # per-OD aggregates
    per_od_mean_max_jacc: List[float] = []
    per_od_cov_by_tau: Dict[float, List[float]] = {float(t): [] for t in tau_values}
    per_od_rows: List[Dict[str, Any]] = []

    # global (route-level) threshold-free aggregates
    all_gt_max_jacc: List[float] = []

    for (sw, dw), g in kept.items():
        gt_list = g["gt"]
        pred_succ = g["pred_success"][: int(k)]

        max_jacc_per_gt: List[float] = []
        for gt_seq in gt_list:
            if len(pred_succ) == 0:
                max_j = 0.0
            else:
                max_j = max(_seq_jaccard(gt_seq, pr) for pr in pred_succ)
            max_jacc_per_gt.append(float(max_j))
            all_gt_max_jacc.append(float(max_j))

        mean_max_j = _mean(max_jacc_per_gt)
        per_od_mean_max_jacc.append(float(mean_max_j))

        cov_tau_row: Dict[str, float] = {}
        for t in tau_values:
            cov_t = float(np.mean(np.asarray([1.0 if x >= float(t) else 0.0 for x in max_jacc_per_gt], dtype=np.float64))) if max_jacc_per_gt else float("nan")
            per_od_cov_by_tau[float(t)].append(float(cov_t))
            cov_tau_row[f"{float(t):.3f}"] = float(cov_t)

        # Only threshold-free + curve metadata, keep compact.
        per_od_rows.append(
            {
                "start_way": int(sw),
                "dest_way": int(dw),
                "n_gt_routes": int(len(gt_list)),
                "n_pred_success_used": int(len(pred_succ)),
                "mean_max_jaccard_at_k": float(mean_max_j),
                "coverage_vs_tau": cov_tau_row,
            }
        )

    # coverage-τ mean curve and AUC
    curve_rows: List[Dict[str, float]] = []
    xs: List[float] = []
    ys: List[float] = []
    for t in tau_values:
        covs = per_od_cov_by_tau[float(t)]
        y = _mean(covs)
        xs.append(float(t))
        ys.append(float(y))
        curve_rows.append(
            {
                "tau": float(t),
                "mean": float(y),
                "p25": _pct(covs, 25),
                "p50": _pct(covs, 50),
                "p75": _pct(covs, 75),
                "n": int(len(covs)),
            }
        )

    auc = float("nan")
    if len(xs) >= 2:
        x_arr = np.asarray(xs, dtype=np.float64)
        y_arr = np.asarray(ys, dtype=np.float64)
        span = float(np.max(x_arr) - np.min(x_arr))
        if span > 0:
            auc = float(np.trapezoid(y_arr, x_arr) / span)

    return {
        "mean_max_jaccard_at_k": {
            "mean": _mean(all_gt_max_jacc),
            "p25": _pct(all_gt_max_jacc, 25),
            "p50": _pct(all_gt_max_jacc, 50),
            "p75": _pct(all_gt_max_jacc, 75),
            "p90": _pct(all_gt_max_jacc, 90),
            "n": int(len(all_gt_max_jacc)),
        },
        "mean_max_jaccard_at_k_per_od": {
            "mean": _mean(per_od_mean_max_jacc),
            "p25": _pct(per_od_mean_max_jacc, 25),
            "p50": _pct(per_od_mean_max_jacc, 50),
            "p75": _pct(per_od_mean_max_jacc, 75),
            "p90": _pct(per_od_mean_max_jacc, 90),
            "n": int(len(per_od_mean_max_jacc)),
        },
        "coverage_vs_tau": curve_rows,
        "coverage_vs_tau_auc": float(auc),
        "per_od_threshold_free": per_od_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase C OD-level GT coverage and self-diversity analysis from per-route decode outputs.")
    ap.add_argument(
        "--method",
        type=str,
        action="append",
        required=True,
        help="Method spec: LABEL|DECODE=PATH, where DECODE is greedy or beam. Example: WayCASD_E2|greedy=.../per_route.json",
    )
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--k", type=int, default=16, help="Use at most K successful predicted routes per OD group.")
    ap.add_argument("--min_routes_per_od", type=int, default=3, help="Keep OD groups with at least this many GT routes.")
    ap.add_argument("--jaccard_threshold", type=float, default=0.5, help="GT route is covered if max Jaccard >= threshold.")
    ap.add_argument(
        "--tau_values",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated τ values for Coverage-τ curve and AUC.",
    )
    ap.add_argument("--save_per_od", action="store_true", help="If set, keep per-OD detail rows in output JSON.")
    # Backward-compatible passthrough args: accepted but not used in this script.
    ap.add_argument("--way_routes_npz", type=Path, default=None, help="Unused in this script; accepted for compatibility.")
    ap.add_argument("--split_json", type=Path, default=None, help="Unused in this script; accepted for compatibility.")
    ap.add_argument("--split_part", choices=["train", "val", "test"], default=None, help="Unused in this script; accepted for compatibility.")
    args = ap.parse_args()

    specs = [_parse_method_spec(s) for s in list(args.method)]
    tau_values = _parse_tau_values(str(args.tau_values))
    results: Dict[str, Any] = {}
    table: List[Dict[str, Any]] = []

    for s in specs:
        p = Path(s.path)
        if not p.exists():
            raise SystemExit(f"[FATAL] file not found: {p}")
        root = _read_json(p)
        per_route = _extract_per_route(root)
        rows = _extract_records(per_route, decode=s.decode)
        res = _analyze_method(
            rows,
            k=int(args.k),
            min_routes_per_od=int(args.min_routes_per_od),
            jacc_th=float(args.jaccard_threshold),
        )
        tf = _analyze_method_threshold_free(
            rows,
            k=int(args.k),
            min_routes_per_od=int(args.min_routes_per_od),
            tau_values=tau_values,
        )
        res.update(tf)
        if not bool(args.save_per_od):
            res.pop("per_od", None)
            res.pop("per_od_threshold_free", None)
        results[s.label] = {
            "decode": s.decode,
            "source_json": str(p),
            **res,
        }
        table.append(
            {
                "method": s.label,
                "decode": s.decode,
                "arrival_rate": float(res["arrival_rate"]),
                "gt_coverage_at_k_mean": float(res["gt_coverage_at_k"]["mean"]),
                "self_diversity_at_k_mean": float(res["self_diversity_at_k"]["mean"]),
                "mean_max_jaccard_at_k_mean": float(res["mean_max_jaccard_at_k"]["mean"]),
                "coverage_vs_tau_auc": float(res["coverage_vs_tau_auc"]),
                "n_od_groups_kept": int(res["n_od_groups_kept"]),
            }
        )

    out = {
        "ok": True,
        "task": "od_coverage_diversity_eval",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": {
            "k": int(args.k),
            "min_routes_per_od": int(args.min_routes_per_od),
            "jaccard_threshold": float(args.jaccard_threshold),
            "tau_values": [float(x) for x in tau_values],
            "save_per_od": bool(args.save_per_od),
            "compat_way_routes_npz": (str(args.way_routes_npz) if args.way_routes_npz is not None else None),
            "compat_split_json": (str(args.split_json) if args.split_json is not None else None),
            "compat_split_part": (str(args.split_part) if args.split_part is not None else None),
            "methods": [asdict(s) for s in specs],
        },
        "summary_table": table,
        "methods": results,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {out_json}")
    print("Method | Decode | Arrival | GT Coverage@K | Self-Diversity@K | MeanMaxJ | CovTauAUC | n_OD")
    print("------ | ------ | ------- | ------------- | ---------------- | -------- | --------- | ----")
    for row in table:
        print(
            f"{row['method']} | {row['decode']} | {row['arrival_rate']:.4f} | {row['gt_coverage_at_k_mean']:.4f} | "
            f"{row['self_diversity_at_k_mean']:.4f} | {row['mean_max_jaccard_at_k_mean']:.4f} | "
            f"{row['coverage_vs_tau_auc']:.4f} | {row['n_od_groups_kept']}"
        )


if __name__ == "__main__":
    main()
