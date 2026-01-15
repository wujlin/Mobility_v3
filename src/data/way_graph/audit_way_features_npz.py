from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np


def _p(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def audit(*, way_features_npz: Path) -> Dict[str, object]:
    data = np.load(str(way_features_npz), allow_pickle=True)
    need = {"way_osm_id", "way_len_m", "way_tier"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"way_features.npz missing keys: {missing}")
    way_len = np.asarray(data["way_len_m"], dtype=np.float64).reshape(-1)
    ok = np.isfinite(way_len) & (way_len > 0)
    report = {
        "ok": True,
        "way_features_npz": str(way_features_npz),
        "n_way_vocab": int(np.asarray(data["way_osm_id"]).reshape(-1).shape[0]),
        "missing_frac": float(1.0 - float(np.mean(ok.astype(np.float64))) if way_len.size else 1.0),
        "way_len_m": {"p50": _p(way_len[ok], 50), "p90": _p(way_len[ok], 90), "max": float(np.max(way_len[ok]) if np.any(ok) else float("nan"))},
        "tier_counts": np.bincount(np.clip(np.asarray(data["way_tier"], dtype=np.int64).reshape(-1), 0, 3), minlength=4).astype(np.int64).tolist(),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit way_features.npz (compact).")
    ap.add_argument("--way_features_npz", type=Path, required=True)
    args = ap.parse_args()
    report = audit(way_features_npz=Path(args.way_features_npz))
    wl = report["way_len_m"]
    c = report["tier_counts"]
    print(f"[way_features] {report['way_features_npz']}")
    print(f"[ways] {report['n_way_vocab']} missing_frac={report['missing_frac']:.3f}")
    print(f"[way_len_m] p50={wl['p50']:.1f} p90={wl['p90']:.1f} max={wl['max']:.1f}")
    print(f"[tier_counts] 0:{c[0]} 1:{c[1]} 2:{c[2]} 3:{c[3]}")


if __name__ == "__main__":
    main()

