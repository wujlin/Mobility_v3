from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _load_meta(data: np.lib.npyio.NpzFile) -> Optional[dict]:
    if "meta" not in data.files:
        return None
    meta = data["meta"]
    if isinstance(meta, np.ndarray) and meta.shape == ():
        meta = meta.item()
    return meta if isinstance(meta, dict) else None


def _p(x: np.ndarray, q: float) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x.astype(np.float64, copy=False), q))


def _format_counts(counts: np.ndarray, *, prefix: str = "") -> str:
    c = np.asarray(counts, dtype=np.int64).reshape(-1)
    return " ".join([f"{prefix}{i}:{int(c[i])}" for i in range(int(c.size))])


def audit(*, paths_graph_npz: Path) -> Dict[str, Any]:
    data = np.load(str(paths_graph_npz), allow_pickle=True)
    need = {"node_seq_len", "traj_idx"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise SystemExit(f"paths_graph.npz missing keys: {missing} ({paths_graph_npz})")

    node_seq_len = np.asarray(data["node_seq_len"], dtype=np.int32).reshape(-1)
    n = int(node_seq_len.size)
    edge_steps = np.maximum(node_seq_len.astype(np.int64, copy=False) - 1, 0).astype(np.int64, copy=False)

    route_city = np.asarray(data["route_city"], dtype=np.int8).reshape(-1) if "route_city" in data.files else None
    city_counts = None
    if route_city is not None and int(route_city.size) == n:
        city_counts = np.bincount(np.clip(route_city.astype(np.int64, copy=False), 0, 16), minlength=2).astype(np.int64).tolist()

    meta = _load_meta(data)
    task = meta.get("task") if isinstance(meta, dict) else None
    stats_meta = meta.get("stats") if isinstance(meta, dict) else None

    report: Dict[str, Any] = {
        "inputs": {"paths_graph_npz": str(paths_graph_npz)},
        "meta": {"task": task, "stats": stats_meta},
        "stats": {
            "n_routes": int(n),
            "node_seq_len": {"p50": _p(node_seq_len, 50), "p90": _p(node_seq_len, 90), "max": int(node_seq_len.max()) if n else 0},
            "edge_steps": {"p50": _p(edge_steps, 50), "p90": _p(edge_steps, 90), "max": int(edge_steps.max()) if n else 0},
            "route_city_counts": city_counts,
        },
    }
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit paths_graph.npz stats (sequence length) with compact prints.")
    p.add_argument("--paths_graph_npz", type=Path, required=True)
    p.add_argument("--out_json", type=Path, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = audit(paths_graph_npz=Path(args.paths_graph_npz))
    stats = report["stats"]
    print(f"[paths_graph] {report['inputs']['paths_graph_npz']}")
    if report.get("meta", {}).get("task"):
        print(f"[meta] task={report['meta']['task']}")
    print(f"[N] {int(stats['n_routes'])}")
    ns = stats["node_seq_len"]
    es = stats["edge_steps"]
    print(f"[node_seq_len] p50={ns['p50']:.1f} p90={ns['p90']:.1f} max={int(ns['max'])}")
    print(f"[edge_steps] p50={es['p50']:.1f} p90={es['p90']:.1f} max={int(es['max'])}")
    if stats.get("route_city_counts") is not None:
        print(f"[route_city] {_format_counts(np.asarray(stats['route_city_counts'], dtype=np.int64))}")

    if args.out_json is not None:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()

