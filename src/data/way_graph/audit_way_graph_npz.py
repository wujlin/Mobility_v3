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


def audit(*, way_graph_npz: Path) -> Dict[str, object]:
    data = np.load(str(way_graph_npz), allow_pickle=True)
    need = {"way_osm_id", "way_adj_ptr", "way_adj_idx"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"way_graph.npz missing keys: {missing}")
    way_osm_id = np.asarray(data["way_osm_id"]).reshape(-1)
    ptr = np.asarray(data["way_adj_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(data["way_adj_idx"], dtype=np.int64).reshape(-1)
    M = int(way_osm_id.shape[0])
    deg = (ptr[1:] - ptr[:-1]).astype(np.int64, copy=False)
    return {
        "ok": True,
        "way_graph_npz": str(way_graph_npz),
        "n_ways": int(M),
        "n_edges_directed": int(idx.size),
        "out_deg": {"p50": _p(deg, 50), "p90": _p(deg, 90), "max": int(np.max(deg) if deg.size else 0)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit way_graph.npz (compact).")
    ap.add_argument("--way_graph_npz", type=Path, required=True)
    args = ap.parse_args()
    report = audit(way_graph_npz=Path(args.way_graph_npz))
    deg = report["out_deg"]
    print(f"[way_graph] {report['way_graph_npz']}")
    print(f"[ways] {report['n_ways']} edges_directed={report['n_edges_directed']}")
    print(f"[out_deg] p50={deg['p50']:.1f} p90={deg['p90']:.1f} max={int(deg['max'])}")


if __name__ == "__main__":
    main()

