from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect a road_graph.npz and print compact JSON stats for debugging.")
    p.add_argument("--road_graph_npz", type=Path, required=True)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    data = np.load(str(args.road_graph_npz), allow_pickle=True)
    need = {"node_y", "node_x", "edge_u", "edge_v", "edge_w_m", "meta"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        print(json.dumps({"ok": False, "error": "missing_keys", "missing": missing}, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    node_y = np.asarray(data["node_y"], dtype=np.float64).reshape(-1)
    node_x = np.asarray(data["node_x"], dtype=np.float64).reshape(-1)
    eu = np.asarray(data["edge_u"], dtype=np.int64).reshape(-1)
    ev = np.asarray(data["edge_v"], dtype=np.int64).reshape(-1)
    ew = np.asarray(data["edge_w_m"], dtype=np.float64).reshape(-1)

    n_nodes = int(node_y.shape[0])
    n_edges = int(eu.shape[0])
    finite_nodes = bool(np.isfinite(node_y).all() and np.isfinite(node_x).all())
    finite_edges = bool(np.isfinite(ew).all())

    meta = data["meta"].item() if isinstance(data["meta"], np.ndarray) and data["meta"].shape == () else data["meta"]
    grid = None
    if isinstance(meta, dict) and isinstance(meta.get("grid"), dict):
        grid = meta["grid"]

    out = {
        "ok": True,
        "road_graph_npz": str(args.road_graph_npz),
        "n_nodes": n_nodes,
        "n_edges_directed": n_edges,
        "finite_nodes": finite_nodes,
        "finite_edges": finite_edges,
        "node_y_minmax": [float(np.min(node_y)), float(np.max(node_y))] if n_nodes else None,
        "node_x_minmax": [float(np.min(node_x)), float(np.max(node_x))] if n_nodes else None,
        "edge_w_minmax_m": [float(np.min(ew)), float(np.max(ew))] if n_edges else None,
        "grid": grid,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

