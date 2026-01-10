from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.evaluation.route_mode_collapse_metrics import _fit_two_corridors, _polyline_features_segment_end_single, _polyline_features_to_dest_single


@dataclass(frozen=True)
class Config:
    mult: float
    percentile: float
    seed: int


def _od_bin_center(pos: np.ndarray, *, bin_size: float) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    b = float(bin_size)
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("--od_bin must be > 0")
    return (np.floor(pos / b) + 0.5) * b


def _max_abs_dist_to_chord(poly: np.ndarray, *, a: np.ndarray, b: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    ab = b - a
    chord = float(np.linalg.norm(ab)) + 1e-12
    ap = poly - a[None, :]
    cross = ab[0] * ap[:, 1] - ab[1] * ap[:, 0]
    dist_signed = cross / chord
    return float(np.max(np.abs(dist_signed))) if dist_signed.size > 0 else 0.0


def _summarize(values: np.ndarray, *, q: float) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, float(q))),
        "max": float(np.max(v)),
    }


def estimate_from_e0(*, e0_dir: Path, cfg: Config) -> Dict[str, object]:
    report_path = e0_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sel = report.get("selected_cases") or []
    if not isinstance(sel, list) or not sel:
        raise ValueError("E0 report.json missing selected_cases")

    od_bin = float((report.get("config") or {}).get("od_bin", 128.0))
    od_end = str((report.get("config") or {}).get("od_end", "dest_pos"))
    seed = int((report.get("config") or {}).get("seed", cfg.seed))

    cases_out: List[Dict[str, object]] = []
    rec_extents: List[int] = []

    for c in sel:
        case_id = int(c.get("case_id", -1))
        case_dir = e0_dir / f"case_{case_id:02d}"
        case_npz = case_dir / "gt_case.npz"
        data = np.load(str(case_npz), allow_pickle=True)
        need = {"start_pos", "targets", "dest_pos"}
        if not need.issubset(set(data.files)):
            raise ValueError(f"{case_npz} must contain {sorted(need)}, got {sorted(list(data.files))}")

        start_pos = np.asarray(data["start_pos"], dtype=np.float32)
        targets = np.asarray(data["targets"], dtype=np.float32)
        dest_pos = np.asarray(data["dest_pos"], dtype=np.float32)

        start_ctr = _od_bin_center(start_pos, bin_size=od_bin)
        dest_ctr = _od_bin_center(dest_pos, bin_size=od_bin)
        a = start_ctr[0].astype(np.float32, copy=False)
        b = dest_ctr[0].astype(np.float32, copy=False)
        chord_len = float(np.linalg.norm((b - a).astype(np.float64)))

        # Per-window max abs deviation (to OD-bin chord).
        dev_abs = np.zeros((int(start_pos.shape[0]),), dtype=np.float64)
        feats = []
        for i in range(int(start_pos.shape[0])):
            poly = np.concatenate([start_pos[i : i + 1], targets[i]], axis=0)
            dev_abs[i] = _max_abs_dist_to_chord(poly, a=a, b=b)
            if od_end == "dest_pos":
                feats.append(_polyline_features_to_dest_single(start_pos[i], targets[i], dest_pos[i]))
            else:
                feats.append(_polyline_features_segment_end_single(start_pos[i], targets[i]))
        feats_arr = np.stack(feats, axis=0)
        cl = _fit_two_corridors(feats_arr, seed=int(seed))
        labels = np.asarray(cl["labels"], dtype=np.int64).reshape(-1)

        dev0 = dev_abs[labels == 0]
        dev1 = dev_abs[labels == 1]
        s_all = _summarize(dev_abs, q=float(cfg.percentile))
        s0 = _summarize(dev0, q=float(cfg.percentile))
        s1 = _summarize(dev1, q=float(cfg.percentile))

        p95_max = max(float(s0["p95"]), float(s1["p95"]), float(s_all["p95"]))
        rec = int(math.ceil(float(cfg.mult) * p95_max))
        rec_extents.append(int(rec))

        cases_out.append(
            {
                "case_id": int(case_id),
                "N": int(start_pos.shape[0]),
                "F": int(targets.shape[1]),
                "od_bin": float(od_bin),
                "od_end": str(od_end),
                "chord": {"start_ctr": [float(a[0]), float(a[1])], "dest_ctr": [float(b[0]), float(b[1])], "len": float(chord_len)},
                "dev_abs": {"all": s_all, "corr0": s0, "corr1": s1},
                "recommended_extent": int(rec),
            }
        )

    out = {
        "inputs": {"e0_dir": str(e0_dir.resolve())},
        "config": {"mult": float(cfg.mult), "percentile": float(cfg.percentile), "seed": int(seed)},
        "cases": cases_out,
        "recommended": {
            "extent_max": int(max(rec_extents)) if rec_extents else 0,
            "extent_p50": int(np.percentile(np.asarray(rec_extents, dtype=np.float64), 50)) if rec_extents else 0,
        },
    }
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Estimate OD-aligned semantic patch extent from E0 fixed GT cases (JSON-only).")
    p.add_argument("--e0_dir", type=str, required=True, help="E0 directory containing report.json and case_XX/gt_case.npz")
    p.add_argument("--mult", type=float, default=1.5, help="extent = ceil(mult * max_p95_dev)")
    p.add_argument("--percentile", type=float, default=95.0, help="percentile for deviation summary (e.g., 95)")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config(mult=float(args.mult), percentile=float(args.percentile), seed=int(args.seed))
    out = estimate_from_e0(e0_dir=Path(args.e0_dir), cfg=cfg)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

