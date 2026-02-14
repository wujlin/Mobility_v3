#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MethodSpec:
    label: str
    decode: str
    path: Path


@dataclass(frozen=True)
class RouteRec:
    city: int
    start_way: int
    dest_way: int
    gt_hops: int
    gt_way_ids: Tuple[int, ...]
    success: bool


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[FATAL] file not found: {path}")
    if not path.is_file():
        raise SystemExit(f"[FATAL] not a file: {path}")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_method_spec(spec: str) -> MethodSpec:
    s = str(spec or "").strip()
    if "=" not in s:
        raise SystemExit(f"[FATAL] bad --method spec: {spec!r}; expect LABEL|DECODE=PATH")
    left, right = s.split("=", 1)
    left = left.strip()
    path = Path(right.strip()).expanduser()
    if "|" in left:
        label, decode = left.rsplit("|", 1)
    else:
        label, decode = left, "greedy"
    label = label.strip()
    decode = decode.strip().lower()
    if decode not in {"greedy", "beam"}:
        raise SystemExit(f"[FATAL] bad decode in --method: {decode!r}, expect greedy/beam")
    if not label:
        raise SystemExit(f"[FATAL] empty label in --method spec: {spec!r}")
    return MethodSpec(label=label, decode=decode, path=path)


def _extract_records(per_route_json: Path, decode: str) -> List[RouteRec]:
    root = _read_json(per_route_json)
    per_route = root.get("per_route")
    if not isinstance(per_route, list):
        raise SystemExit(f"[FATAL] {per_route_json}: missing per_route list")
    out: List[RouteRec] = []
    missing_decode = 0
    missing_gt = 0
    for rec in per_route:
        if not isinstance(rec, dict):
            continue
        gt = rec.get("gt_way_ids")
        if not isinstance(gt, list) or not gt:
            missing_gt += 1
            continue
        dec = rec.get(str(decode))
        if not isinstance(dec, dict):
            missing_decode += 1
            continue
        out.append(
            RouteRec(
                city=int(rec.get("city", -1)),
                start_way=int(rec.get("start_way", gt[0])),
                dest_way=int(rec.get("dest_way", gt[-1])),
                gt_hops=int(rec.get("gt_hops", max(0, len(gt) - 1))),
                gt_way_ids=tuple(int(x) for x in gt),
                success=bool(dec.get("success", False)),
            )
        )
    if not out:
        raise SystemExit(
            f"[FATAL] no valid rows in {per_route_json} (decode={decode}); "
            f"missing_gt={missing_gt}, missing_decode={missing_decode}"
        )
    return out


def _infer_hero_method_label(methods: Sequence[MethodSpec]) -> str:
    for m in methods:
        s = m.label.lower()
        if "way-casd" in s or "waycasd" in s:
            return m.label
    return methods[0].label


def _jaccard_dist(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 0.0
    return 1.0 - float(len(sa & sb)) / float(max(1, len(sa | sb)))


def _mean_pairwise_dist(
    seqs: Sequence[Tuple[int, ...]],
    *,
    max_pairs: int,
    seed: int,
) -> float:
    uniq = list({tuple(int(x) for x in s) for s in seqs})
    n = len(uniq)
    if n < 2:
        return float("nan")
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    if len(pairs) > int(max_pairs):
        rnd = random.Random(int(seed))
        pairs = rnd.sample(pairs, int(max_pairs))
    vals = [_jaccard_dist(uniq[i], uniq[j]) for i, j in pairs]
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Select representative Hero ODs for Figure A under strict filters.")
    ap.add_argument("--phasec_json", type=Path, required=True)
    ap.add_argument("--hero_label", type=str, required=True, help="Method label in phaseC json (e.g., Way-CASD_E2e100_K16).")
    ap.add_argument("--method", action="append", required=True, help="Repeatable: LABEL|DECODE=PER_ROUTE_JSON")
    ap.add_argument("--hero_method_label", type=str, default="", help="Which --method label provides GT routes and main success (default auto Way-CASD).")
    ap.add_argument("--city", type=int, default=0)
    ap.add_argument("--hops_min", type=int, default=20)
    ap.add_argument("--hops_max", type=int, default=40)
    ap.add_argument("--min_gt_routes", type=int, default=5)
    ap.add_argument("--min_hero_success", type=int, default=3, help="Minimum successful predictions for hero method.")
    ap.add_argument("--max_baseline_success", type=int, default=2, help="Maximum successful predictions allowed for each baseline (RNN/Transformer).")
    ap.add_argument("--min_gt_jaccard_dist", type=float, default=0.5, help="Minimum mean pairwise GT Jaccard distance.")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max_gt_pairs", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", type=Path, required=True)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    _require_file(Path(args.phasec_json))
    methods = [_parse_method_spec(s) for s in list(args.method)]
    for m in methods:
        _require_file(m.path)

    hero_method_label = str(args.hero_method_label).strip() or _infer_hero_method_label(methods)
    labels = {m.label for m in methods}
    if hero_method_label not in labels:
        raise SystemExit(f"[FATAL] hero_method_label={hero_method_label!r} not in --method labels: {sorted(labels)}")

    # Load phaseC per_od for the target hero label.
    phasec = _read_json(Path(args.phasec_json))
    methods_block = phasec.get("methods", {})
    if not isinstance(methods_block, dict) or str(args.hero_label) not in methods_block:
        raise SystemExit(f"[FATAL] hero_label={args.hero_label!r} not found in phaseC methods")
    per_od = methods_block[str(args.hero_label)].get("per_od", [])
    if not isinstance(per_od, list) or not per_od:
        raise SystemExit("[FATAL] phaseC per_od is missing/empty (need --save_per_od when running Phase C).")
    phasec_od: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in per_od:
        if not isinstance(r, dict):
            continue
        try:
            k = (int(r["start_way"]), int(r["dest_way"]))
        except Exception:
            continue
        phasec_od[k] = r

    # Load per-route records.
    recs_by_method: Dict[str, List[RouteRec]] = {}
    for m in methods:
        rs = [r for r in _extract_records(m.path, m.decode) if int(r.city) == int(args.city)]
        recs_by_method[m.label] = rs

    # Aggregate per OD statistics.
    od_keys: set[Tuple[int, int]] = set()
    for rs in recs_by_method.values():
        for r in rs:
            od_keys.add((int(r.start_way), int(r.dest_way)))

    # Baseline labels to constrain.
    baseline_labels: List[str] = []
    for m in methods:
        s = m.label.lower()
        if "rnn" in s or "transformer" in s or "tr-ar" in s:
            baseline_labels.append(m.label)

    candidates: List[Dict[str, Any]] = []
    for sw, dw in sorted(od_keys):
        od = (int(sw), int(dw))
        if od not in phasec_od:
            continue

        hero_rows = [r for r in recs_by_method[hero_method_label] if int(r.start_way) == sw and int(r.dest_way) == dw]
        if not hero_rows:
            continue

        n_gt_emp = int(len(hero_rows))
        n_gt_phasec = int(phasec_od[od].get("n_gt_routes", 0))
        n_gt = max(n_gt_emp, n_gt_phasec)
        if n_gt < int(args.min_gt_routes):
            continue

        gt_hops = np.asarray([int(r.gt_hops) for r in hero_rows], dtype=np.int64)
        if gt_hops.size <= 0:
            continue
        gt_hops_median = float(np.median(gt_hops))
        if gt_hops_median < float(args.hops_min) or gt_hops_median > float(args.hops_max):
            continue

        gt_dist = _mean_pairwise_dist(
            [r.gt_way_ids for r in hero_rows],
            max_pairs=int(args.max_gt_pairs),
            seed=int(args.seed),
        )
        if (not np.isfinite(gt_dist)) or gt_dist < float(args.min_gt_jaccard_dist):
            continue

        succ_by_method: Dict[str, int] = {}
        for m in methods:
            rows = [r for r in recs_by_method[m.label] if int(r.start_way) == sw and int(r.dest_way) == dw]
            succ_by_method[m.label] = int(sum(1 for r in rows if bool(r.success)))

        hero_success = int(succ_by_method.get(hero_method_label, 0))
        if hero_success < int(args.min_hero_success):
            continue

        baseline_ok = True
        for bl in baseline_labels:
            if int(succ_by_method.get(bl, 0)) > int(args.max_baseline_success):
                baseline_ok = False
                break
        if not baseline_ok:
            continue

        pr = phasec_od[od]
        cov = _safe_float(pr.get("gt_coverage_at_k"))
        div = _safe_float(pr.get("self_diversity_at_k"))
        n_pred_success_used = int(pr.get("n_pred_success_used", 0))

        candidates.append(
            {
                "start_way": int(sw),
                "dest_way": int(dw),
                "n_gt_routes": int(n_gt),
                "gt_hops_median": float(gt_hops_median),
                "gt_jaccard_dist_mean": float(gt_dist),
                "hero_success": int(hero_success),
                "baseline_success": {k: int(v) for k, v in succ_by_method.items() if k in baseline_labels},
                "phasec_gt_coverage_at_k": (float(cov) if np.isfinite(cov) else None),
                "phasec_self_diversity_at_k": (float(div) if np.isfinite(div) else None),
                "phasec_n_pred_success_used": int(n_pred_success_used),
            }
        )

    # Rank: hero success > GT diversity > phasec diversity > coverage > n_gt.
    candidates.sort(
        key=lambda r: (
            int(r.get("hero_success", 0)),
            float(r.get("gt_jaccard_dist_mean", -1.0)),
            float(r.get("phasec_self_diversity_at_k", -1.0) if r.get("phasec_self_diversity_at_k") is not None else -1.0),
            float(r.get("phasec_gt_coverage_at_k", -1.0) if r.get("phasec_gt_coverage_at_k") is not None else -1.0),
            int(r.get("n_gt_routes", 0)),
        ),
        reverse=True,
    )
    topk = max(1, int(args.topk))
    top_rows = candidates[:topk]

    out = {
        "ok": True,
        "task": "waycasd_select_hero_od_candidates",
        "cfg": {
            "phasec_json": str(args.phasec_json),
            "hero_label": str(args.hero_label),
            "hero_method_label": str(hero_method_label),
            "city": int(args.city),
            "hops_min": int(args.hops_min),
            "hops_max": int(args.hops_max),
            "min_gt_routes": int(args.min_gt_routes),
            "min_hero_success": int(args.min_hero_success),
            "max_baseline_success": int(args.max_baseline_success),
            "min_gt_jaccard_dist": float(args.min_gt_jaccard_dist),
            "topk": int(args.topk),
            "max_gt_pairs": int(args.max_gt_pairs),
            "seed": int(args.seed),
            "baseline_labels": baseline_labels,
            "methods": [{"label": m.label, "decode": m.decode, "path": str(m.path)} for m in methods],
        },
        "n_candidates_total": int(len(candidates)),
        "top_candidates": top_rows,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved: {out_path}")
    if top_rows:
        first = top_rows[0]
        print(
            "[TOP1]",
            f"OD=({int(first['start_way'])},{int(first['dest_way'])})",
            f"hero_success={int(first['hero_success'])}",
            f"gt_jaccard_dist_mean={float(first['gt_jaccard_dist_mean']):.3f}",
            f"gt_hops_median={float(first['gt_hops_median']):.1f}",
        )
    else:
        print("[WARN] no candidate passed current filters; relax thresholds.")


if __name__ == "__main__":
    main()
