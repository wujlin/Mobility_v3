from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class GateDir:
    path: Path


def _parse_dirs(s: str) -> Tuple[GateDir, ...]:
    items = [x.strip() for x in str(s).split(",") if str(x).strip()]
    if not items:
        raise ValueError("--gate_dirs must be a non-empty comma-separated list.")
    out = []
    for x in items:
        p = Path(x)
        out.append(GateDir(path=p))
    return tuple(out)


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_events_auc(events_jsonl: Path) -> Dict[str, np.ndarray]:
    auc_time: List[float] = []
    auc_tier: List[float] = []
    auc_tt: List[float] = []
    for line in events_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        auc = ev.get("auc") or {}
        auc_time.append(float(auc.get("time_only", 0.0)))
        auc_tier.append(float(auc.get("tier_od", 0.0)))
        auc_tt.append(float(auc.get("time_tier", 0.0)))
    return {
        "time_only": np.asarray(auc_time, dtype=np.float64),
        "tier_od": np.asarray(auc_tier, dtype=np.float64),
        "time_tier": np.asarray(auc_tt, dtype=np.float64),
    }


def _summ(a: np.ndarray) -> Dict[str, object]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return {"mean": None, "p50": None, "p90": None, "n": 0}
    return {
        "mean": float(np.mean(a)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "n": int(a.size),
    }


def run(*, gate_dirs: Sequence[GateDir], out_json: Optional[Path]) -> Dict[str, object]:
    per_city = []
    all_auc_time = []
    all_auc_tier = []
    all_auc_tt = []

    for gd in gate_dirs:
        rep_path = gd.path / "report.json"
        ev_path = gd.path / "events.jsonl"
        if not rep_path.exists():
            raise FileNotFoundError(f"Missing report.json under gate_dir: {gd.path}")
        rep = _read_json(rep_path)
        city = None
        try:
            city = Path(str(((rep.get("inputs") or {}).get("semantic_dir") or ""))).name
        except Exception:
            city = None

        auc = _read_events_auc(ev_path) if ev_path.exists() else {"time_only": np.zeros((0,)), "tier_od": np.zeros((0,)), "time_tier": np.zeros((0,))}
        all_auc_time.append(auc["time_only"])
        all_auc_tier.append(auc["tier_od"])
        all_auc_tt.append(auc["time_tier"])

        per_city.append(
            {
                "gate_dir": str(gd.path),
                "city": city,
                "decision": rep.get("decision"),
                "used_groups": int(((rep.get("stats") or {}).get("od_groups") or {}).get("used") or 0),
                "auc": {
                    "time_only": _summ(auc["time_only"]),
                    "tier_od": _summ(auc["tier_od"]),
                    "time_tier": _summ(auc["time_tier"]),
                },
            }
        )

    all_time = np.concatenate(all_auc_time, axis=0) if all_auc_time else np.zeros((0,), dtype=np.float64)
    all_tier = np.concatenate(all_auc_tier, axis=0) if all_auc_tier else np.zeros((0,), dtype=np.float64)
    all_tt = np.concatenate(all_auc_tt, axis=0) if all_auc_tt else np.zeros((0,), dtype=np.float64)

    report = {
        "ok": True,
        "tool": "aggregate_cluster_gate_reports",
        "inputs": {"gate_dirs": [str(x.path) for x in gate_dirs]},
        "stats": {"used_groups_total": int(all_tt.size), "auc": {"time_only": _summ(all_time), "tier_od": _summ(all_tier), "time_tier": _summ(all_tt)}},
        "per_city": per_city,
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["outputs"] = {"out_json": str(out_json)}
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate multiple G3b cluster-gate outputs (report.json + events.jsonl) into a single summary.")
    p.add_argument("--gate_dirs", type=str, required=True, help="Comma-separated gate output dirs (each contains report.json + events.jsonl).")
    p.add_argument("--out_json", type=Path, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = run(gate_dirs=_parse_dirs(args.gate_dirs), out_json=(Path(args.out_json) if args.out_json is not None else None))
    compact = {
        "ok": True,
        "used_groups_total": report["stats"]["used_groups_total"],
        "auc_time_tier_mean": report["stats"]["auc"]["time_tier"]["mean"],
        "out_json": (report.get("outputs") or {}).get("out_json"),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

