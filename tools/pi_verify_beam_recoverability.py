#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[FATAL] file not found: {path}")
    if not path.is_file():
        raise SystemExit(f"[FATAL] not a file: {path}")


def _default_paths(run_dir: Path) -> Tuple[Path, Path, Path]:
    return (
        run_dir / "oracle_decode_greedy_n200.json",
        run_dir / "oracle_decode_beam3_n200.json",
        run_dir / "oracle_decode_beam10_n200.json",
    )


def _cfg_sig(cfg: object) -> Dict[str, object]:
    if not isinstance(cfg, dict):
        return {}
    keys = [
        "seed",
        "n_routes",
        "max_way_len",
        "max_decode_len",
        "decode_max_candidates",
        "decode_candidate_policy",
        "decode_include_dest_if_successor",
        "tz_offset_hours",
    ]
    return {k: cfg.get(k) for k in keys}


@dataclass(frozen=True)
class CityIndex:
    n_eval: int
    success_rate: Optional[float]
    succ: Set[int]
    fail: Set[int]


def _index_by_city(rep: dict) -> Dict[int, CityIndex]:
    out: Dict[int, CityIndex] = {}
    for c in rep.get("per_city", []) or []:
        if not isinstance(c, dict):
            continue
        city = int(c.get("city", -1))
        succ = set(int(x) for x in (c.get("success_route_ids") or []) if x is not None)
        fail = set(
            int(f.get("route_id"))
            for f in (c.get("failures") or [])
            if isinstance(f, dict) and f.get("route_id") is not None
        )
        n_eval = int(c.get("n_eval", len(succ) + len(fail)))
        sr = c.get("success_rate", None)
        out[city] = CityIndex(n_eval=n_eval, success_rate=(float(sr) if sr is not None else None), succ=succ, fail=fail)
    return out


def _safe_rate(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


def _build_report(*, greedy: dict, beam3: dict, beam10: dict, out_json: Path, paths: Dict[str, str]) -> dict:
    ig = _index_by_city(greedy)
    i3 = _index_by_city(beam3)
    i10 = _index_by_city(beam10)

    cities = sorted(set(ig.keys()) | set(i3.keys()) | set(i10.keys()))
    mismatches: List[str] = []

    cfg_g = _cfg_sig(greedy.get("cfg"))
    cfg_3 = _cfg_sig(beam3.get("cfg"))
    cfg_10 = _cfg_sig(beam10.get("cfg"))
    if cfg_g != cfg_3:
        mismatches.append("cfg_mismatch(greedy vs beam3)")
    if cfg_g != cfg_10:
        mismatches.append("cfg_mismatch(greedy vs beam10)")

    by_city: Dict[str, Any] = {}
    overall = {
        "n_eval": 0,
        "greedy_fail_n": 0,
        "beam3_fail_n": 0,
        "beam10_fail_n": 0,
        "beam3_recovered_from_greedy_fail_n": 0,
        "beam10_recovered_from_greedy_fail_n": 0,
        "beam3_regress_from_greedy_success_n": 0,
        "beam10_regress_from_greedy_success_n": 0,
    }

    for city in cities:
        g = ig.get(city, CityIndex(n_eval=0, success_rate=None, succ=set(), fail=set()))
        b3 = i3.get(city, CityIndex(n_eval=0, success_rate=None, succ=set(), fail=set()))
        b10 = i10.get(city, CityIndex(n_eval=0, success_rate=None, succ=set(), fail=set()))

        if not (g.n_eval == b3.n_eval == b10.n_eval):
            mismatches.append(f"n_eval_mismatch(city={city} greedy={g.n_eval} beam3={b3.n_eval} beam10={b10.n_eval})")

        greedy_fail = set(g.fail)
        greedy_succ = set(g.succ)
        recov3 = len(greedy_fail & set(b3.succ))
        recov10 = len(greedy_fail & set(b10.succ))
        regress3 = len(greedy_succ - set(b3.succ))
        regress10 = len(greedy_succ - set(b10.succ))

        by_city[str(int(city))] = {
            "n_eval": int(g.n_eval),
            "success_rate_greedy": g.success_rate,
            "success_rate_beam3": b3.success_rate,
            "success_rate_beam10": b10.success_rate,
            "failure_beam_recoverable": {
                "beam1_fail_n": int(len(greedy_fail)),
                "beam3_fail_n": int(len(b3.fail)),
                "beam10_fail_n": int(len(b10.fail)),
                "beam3_recovered_from_greedy_fail_n": int(recov3),
                "beam10_recovered_from_greedy_fail_n": int(recov10),
                "beam3_recovered_from_greedy_fail_rate": _safe_rate(recov3, len(greedy_fail)),
                "beam10_recovered_from_greedy_fail_rate": _safe_rate(recov10, len(greedy_fail)),
                "beam3_regress_from_greedy_success_n": int(regress3),
                "beam10_regress_from_greedy_success_n": int(regress10),
                "beam3_regress_from_greedy_success_rate": _safe_rate(regress3, len(greedy_succ)),
                "beam10_regress_from_greedy_success_rate": _safe_rate(regress10, len(greedy_succ)),
            },
        }

        overall["n_eval"] += int(g.n_eval)
        overall["greedy_fail_n"] += int(len(greedy_fail))
        overall["beam3_fail_n"] += int(len(b3.fail))
        overall["beam10_fail_n"] += int(len(b10.fail))
        overall["beam3_recovered_from_greedy_fail_n"] += int(recov3)
        overall["beam10_recovered_from_greedy_fail_n"] += int(recov10)
        overall["beam3_regress_from_greedy_success_n"] += int(regress3)
        overall["beam10_regress_from_greedy_success_n"] += int(regress10)

    overall["beam3_recovered_from_greedy_fail_rate"] = _safe_rate(
        int(overall["beam3_recovered_from_greedy_fail_n"]), int(overall["greedy_fail_n"])
    )
    overall["beam10_recovered_from_greedy_fail_rate"] = _safe_rate(
        int(overall["beam10_recovered_from_greedy_fail_n"]), int(overall["greedy_fail_n"])
    )

    out = {
        "ok": True,
        "task": "pi_verify_beam_recoverability",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": paths,
        "cfg_signature": {"greedy": cfg_g, "beam3": cfg_3, "beam10": cfg_10},
        "mismatches": mismatches,
        "by_city": by_city,
        "overall": overall,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PI verify: beam recoverability summary from oracle_decode_* jsons.")
    p.add_argument("--run_dir", type=Path, default=None, help="If set, use default filenames under this directory.")
    p.add_argument("--greedy_json", type=Path, default=None)
    p.add_argument("--beam3_json", type=Path, default=None)
    p.add_argument("--beam10_json", type=Path, default=None)
    p.add_argument("--out_json", type=Path, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.run_dir is not None:
        g, b3, b10 = _default_paths(Path(args.run_dir))
        greedy_json = Path(args.greedy_json) if args.greedy_json is not None else g
        beam3_json = Path(args.beam3_json) if args.beam3_json is not None else b3
        beam10_json = Path(args.beam10_json) if args.beam10_json is not None else b10
        out_json = Path(args.out_json) if args.out_json is not None else (Path(args.run_dir) / "beam_recoverability_summary.json")
    else:
        if args.greedy_json is None or args.beam3_json is None or args.beam10_json is None:
            raise SystemExit("[FATAL] need --run_dir or all of --greedy_json/--beam3_json/--beam10_json")
        greedy_json = Path(args.greedy_json)
        beam3_json = Path(args.beam3_json)
        beam10_json = Path(args.beam10_json)
        out_json = Path(args.out_json) if args.out_json is not None else (greedy_json.parent / "beam_recoverability_summary.json")

    for p in (greedy_json, beam3_json, beam10_json):
        _require_file(p)

    greedy = _read_json(greedy_json)
    beam3 = _read_json(beam3_json)
    beam10 = _read_json(beam10_json)

    rep = _build_report(
        greedy=greedy,
        beam3=beam3,
        beam10=beam10,
        out_json=out_json,
        paths={
            "greedy_json": str(greedy_json),
            "beam3_json": str(beam3_json),
            "beam10_json": str(beam10_json),
        },
    )
    print(f"[saved] {out_json}")
    print(json.dumps(rep.get("overall", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
