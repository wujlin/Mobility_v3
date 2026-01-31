#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[FATAL] file not found: {path}")
    if not path.is_file():
        raise SystemExit(f"[FATAL] not a file: {path}")


def _default_step_json(run_dir: Path) -> Path:
    return run_dir / "oracle_step_diagnose_n200.json"


def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _turn_cat(deg: float, *, thr: float) -> str:
    if abs(float(deg)) < float(thr):
        return "straight"
    return "left" if float(deg) > 0.0 else "right"


def _stats(x: List[float]) -> Dict[str, Optional[float]]:
    a = np.asarray(list(x), dtype=np.float64).reshape(-1)
    if a.size == 0:
        return {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "mean": float(np.mean(a)),
        "p50": float(np.quantile(a, 0.50)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
    }


def _safe_rate(num: int, den: int) -> Optional[float]:
    if int(den) <= 0:
        return None
    return float(num) / float(den)


def run(
    *,
    oracle_step_json: Path,
    way_features_npz: Path,
    turn_threshold_deg: float,
    out_json: Path,
) -> dict:
    step = _read_json(Path(oracle_step_json))
    wf = np.load(str(way_features_npz), allow_pickle=True)

    dir_y = np.asarray(wf["way_dir_y"], dtype=np.float64).reshape(-1)
    dir_x = np.asarray(wf["way_dir_x"], dtype=np.float64).reshape(-1)
    tier = np.asarray(wf["way_tier"], dtype=np.int64).reshape(-1)
    hw = np.asarray(wf["way_highway_code"], dtype=np.int64).reshape(-1)

    def ang(w: int) -> float:
        return math.atan2(float(dir_y[w]), float(dir_x[w]))

    abs_err_all: List[float] = []
    abs_err_fail: List[float] = []
    abs_err_succ_div: List[float] = []
    outdeg_all: List[float] = []

    # Accumulators per city/outcome
    acc: Dict[str, Dict[str, Any]] = {}

    for r in step.get("per_route", []) or []:
        if not isinstance(r, dict):
            continue
        fd = r.get("first_div_transition")
        if not isinstance(fd, dict):
            continue

        success = bool(r.get("success", False))
        diverged = r.get("diverge_idx") is not None
        if success and (not diverged):
            continue
        outcome = "succ_diverged" if success else "fail"
        city = int(r.get("city", -1))

        cur = int(fd.get("cur_way", -1))
        gt = int(fd.get("gt_next", -1))
        pred = int(fd.get("pred_next", -1))
        if cur < 0 or gt < 0 or pred < 0:
            continue
        if cur >= dir_y.size or gt >= dir_y.size or pred >= dir_y.size:
            continue

        a_cur = ang(cur)
        deg_gt = math.degrees(_wrap_pi(ang(gt) - a_cur))
        deg_pr = math.degrees(_wrap_pi(ang(pred) - a_cur))
        cat_gt = _turn_cat(deg_gt, thr=float(turn_threshold_deg))
        cat_pr = _turn_cat(deg_pr, thr=float(turn_threshold_deg))
        abs_err_deg = abs(math.degrees(_wrap_pi(math.radians(deg_pr) - math.radians(deg_gt))))

        turn_mis = int(cat_gt != cat_pr)
        tier_mis = int(int(tier[gt]) != int(tier[pred])) if (gt < tier.size and pred < tier.size) else 0
        hw_mis = int(int(hw[gt]) != int(hw[pred])) if (gt < hw.size and pred < hw.size) else 0

        outdeg = int(fd.get("succ_full_n", -1))

        abs_err_all.append(float(abs_err_deg))
        if outcome == "fail":
            abs_err_fail.append(float(abs_err_deg))
        else:
            abs_err_succ_div.append(float(abs_err_deg))
        if outdeg >= 0:
            outdeg_all.append(float(outdeg))

        d = acc.setdefault(str(city), {})
        d[f"{outcome}_n"] = int(d.get(f"{outcome}_n", 0)) + 1
        d[f"{outcome}_turn_mismatch_n"] = int(d.get(f"{outcome}_turn_mismatch_n", 0)) + int(turn_mis)
        d[f"{outcome}_tier_mismatch_n"] = int(d.get(f"{outcome}_tier_mismatch_n", 0)) + int(tier_mis)
        d[f"{outcome}_highway_mismatch_n"] = int(d.get(f"{outcome}_highway_mismatch_n", 0)) + int(hw_mis)
        d.setdefault(f"{outcome}_abs_turn_err_deg", []).append(float(abs_err_deg))
        if outdeg >= 0:
            d.setdefault(f"{outcome}_first_div_outdeg", []).append(float(outdeg))

    by_city: Dict[str, Any] = {}
    for city, d in acc.items():
        out_c: Dict[str, Any] = {"city": int(city)}
        for outcome in ("fail", "succ_diverged"):
            n = int(d.get(f"{outcome}_n", 0))
            out_c[f"{outcome}_n"] = n
            out_c[f"{outcome}_turn_mismatch_rate"] = _safe_rate(int(d.get(f"{outcome}_turn_mismatch_n", 0)), n)
            out_c[f"{outcome}_tier_mismatch_rate"] = _safe_rate(int(d.get(f"{outcome}_tier_mismatch_n", 0)), n)
            out_c[f"{outcome}_highway_mismatch_rate"] = _safe_rate(int(d.get(f"{outcome}_highway_mismatch_n", 0)), n)
            out_c[f"{outcome}_abs_turn_err_deg_stats"] = _stats(list(d.get(f"{outcome}_abs_turn_err_deg", [])))
            out_c[f"{outcome}_first_div_outdeg_stats"] = _stats(list(d.get(f"{outcome}_first_div_outdeg", [])))
        by_city[str(city)] = out_c

    out = {
        "ok": True,
        "task": "pi_verify_nonspatial_first_div_diag",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "oracle_step_json": str(oracle_step_json),
            "way_features_npz": str(way_features_npz),
            "turn_threshold_deg": float(turn_threshold_deg),
        },
        "n_events": int(len(abs_err_all)),
        "overall": {
            "abs_turn_err_deg_stats": _stats(abs_err_all),
            "abs_turn_err_deg_stats_fail": _stats(abs_err_fail),
            "abs_turn_err_deg_stats_succ_diverged": _stats(abs_err_succ_div),
            "first_div_outdeg_stats": _stats(outdeg_all),
        },
        "by_city": by_city,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PI verify: non-spatial first-div diagnosis from oracle_step_diagnose json + way_features.npz.")
    p.add_argument("--run_dir", type=Path, default=None, help="If set, read oracle_step_diagnose_n200.json under this directory.")
    p.add_argument("--oracle_step_json", type=Path, default=None)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--turn_threshold_deg", type=float, default=30.0)
    p.add_argument("--out_json", type=Path, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.run_dir is not None:
        step_json = Path(args.oracle_step_json) if args.oracle_step_json is not None else _default_step_json(Path(args.run_dir))
        out_json = Path(args.out_json) if args.out_json is not None else (Path(args.run_dir) / "nonspatial_first_div_diag.json")
    else:
        if args.oracle_step_json is None:
            raise SystemExit("[FATAL] need --run_dir or --oracle_step_json")
        step_json = Path(args.oracle_step_json)
        out_json = Path(args.out_json) if args.out_json is not None else (step_json.parent / "nonspatial_first_div_diag.json")

    _require_file(step_json)
    _require_file(Path(args.way_features_npz))

    rep = run(
        oracle_step_json=step_json,
        way_features_npz=Path(args.way_features_npz),
        turn_threshold_deg=float(args.turn_threshold_deg),
        out_json=out_json,
    )
    print(f"[saved] {out_json}")
    print(json.dumps(rep.get("overall", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

