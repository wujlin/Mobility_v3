from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def _p(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def audit(*, routes_npz: Path) -> Dict[str, object]:
    data = np.load(str(routes_npz), allow_pickle=True)
    need = {"way_osm_id", "way_seq_ptr", "way_seq_idx", "way_seq_len", "start_way", "dest_way", "start_t", "route_city"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"way_routes.npz missing keys: {missing}")
    n_routes = int(np.asarray(data["way_seq_len"]).reshape(-1).shape[0])
    lens = np.asarray(data["way_seq_len"], dtype=np.float64).reshape(-1)
    corridor = data["corridor_type"] if "corridor_type" in data.files else None
    report = {
        "ok": True,
        "routes_npz": str(routes_npz),
        "n_routes": int(n_routes),
        "n_way_vocab": int(np.asarray(data["way_osm_id"]).reshape(-1).shape[0]),
        "way_seq_len": {"p50": _p(lens, 50), "p90": _p(lens, 90), "max": int(np.max(lens) if lens.size else 0), "mean": float(np.mean(lens) if lens.size else 0.0)},
        "corridor_type_counts": (
            np.bincount(np.clip(np.asarray(corridor, dtype=np.int64).reshape(-1), 0, 3), minlength=4).astype(np.int64).tolist()
            if corridor is not None
            else None
        ),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit way_routes.npz (compact).")
    ap.add_argument("--routes_npz", type=Path, required=True)
    args = ap.parse_args()
    report = audit(routes_npz=Path(args.routes_npz))
    lens = report["way_seq_len"]
    print(f"[way_routes] {report['routes_npz']}")
    print(f"[N] {report['n_routes']} vocab={report['n_way_vocab']}")
    print(f"[way_seq_len] p50={lens['p50']:.1f} p90={lens['p90']:.1f} max={int(lens['max'])} mean={lens['mean']:.1f}")
    if report.get("corridor_type_counts") is not None:
        c = report["corridor_type_counts"]
        print(f"[corridor_type] 0:{c[0]} 1:{c[1]} 2:{c[2]} 3:{c[3]}")


if __name__ == "__main__":
    main()

