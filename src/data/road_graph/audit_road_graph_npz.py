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


def audit(*, road_graph_npz: Path) -> Dict[str, Any]:
    data = np.load(str(road_graph_npz), allow_pickle=True)
    need = {"node_y", "node_x", "edge_u", "edge_v", "edge_w_m"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise SystemExit(f"road_graph.npz missing keys: {missing} ({road_graph_npz})")

    node_y = np.asarray(data["node_y"], dtype=np.float32).reshape(-1)
    node_x = np.asarray(data["node_x"], dtype=np.float32).reshape(-1)
    edge_u = np.asarray(data["edge_u"], dtype=np.int32).reshape(-1)
    edge_v = np.asarray(data["edge_v"], dtype=np.int32).reshape(-1)
    edge_w_m = np.asarray(data["edge_w_m"], dtype=np.float32).reshape(-1)
    edge_tier = np.asarray(data["edge_tier"], dtype=np.uint8).reshape(-1) if "edge_tier" in data.files else None
    node_city = np.asarray(data["node_city"], dtype=np.int8).reshape(-1) if "node_city" in data.files else None
    edge_city = np.asarray(data["edge_city"], dtype=np.int8).reshape(-1) if "edge_city" in data.files else None

    n_nodes = int(node_y.size)
    n_edges = int(edge_u.size)
    if not (int(node_x.size) == n_nodes and int(edge_v.size) == n_edges and int(edge_w_m.size) == n_edges):
        raise SystemExit(f"shape mismatch inside road_graph: N={n_nodes} E={n_edges} ({road_graph_npz})")

    out_deg = np.bincount(np.clip(edge_u.astype(np.int64, copy=False), 0, max(0, n_nodes - 1)), minlength=n_nodes).astype(np.int64)
    edge_stats = {
        "p50_m": _p(edge_w_m, 50),
        "p90_m": _p(edge_w_m, 90),
        "mean_m": float(np.mean(edge_w_m.astype(np.float64, copy=False))) if edge_w_m.size else float("nan"),
        "min_m": float(np.min(edge_w_m)) if edge_w_m.size else float("nan"),
        "max_m": float(np.max(edge_w_m)) if edge_w_m.size else float("nan"),
    }

    tier_counts = None
    if edge_tier is not None and int(edge_tier.size) == n_edges:
        tier_counts = np.bincount(np.clip(edge_tier.astype(np.int64, copy=False), 0, 3), minlength=4).astype(np.int64).tolist()

    meta = _load_meta(data)
    task = meta.get("task") if isinstance(meta, dict) else None
    inputs = meta.get("inputs") if isinstance(meta, dict) else None

    report: Dict[str, Any] = {
        "inputs": {"road_graph_npz": str(road_graph_npz)},
        "meta": {"task": task, "inputs": inputs},
        "stats": {
            "n_nodes": int(n_nodes),
            "n_edges_directed": int(n_edges),
            "edge_len_m": edge_stats,
            "out_deg": {"p50": _p(out_deg, 50), "p90": _p(out_deg, 90), "max": int(out_deg.max()) if out_deg.size else 0},
            "tier_counts": tier_counts,
            "node_city_counts": np.bincount(np.clip(node_city.astype(np.int64, copy=False), 0, 16), minlength=2).astype(np.int64).tolist()
            if node_city is not None
            else None,
            "edge_city_counts": np.bincount(np.clip(edge_city.astype(np.int64, copy=False), 0, 16), minlength=2).astype(np.int64).tolist()
            if edge_city is not None
            else None,
        },
    }
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit road_graph.npz stats (edge length, tiers, degrees) with compact prints.")
    p.add_argument("--road_graph_npz", type=Path, required=True)
    p.add_argument("--out_json", type=Path, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = audit(road_graph_npz=Path(args.road_graph_npz))
    stats = report["stats"]
    print(f"[road_graph] {report['inputs']['road_graph_npz']}")
    if report.get("meta", {}).get("task"):
        print(f"[meta] task={report['meta']['task']}")
    print(f"[nodes] {int(stats['n_nodes'])}")
    print(f"[edges_directed] {int(stats['n_edges_directed'])}")
    el = stats["edge_len_m"]
    print(f"[edge_len_m] p50={el['p50_m']:.1f} p90={el['p90_m']:.1f} mean={el['mean_m']:.1f} min={el['min_m']:.1f} max={el['max_m']:.1f}")
    od = stats["out_deg"]
    print(f"[out_deg] p50={od['p50']:.1f} p90={od['p90']:.1f} max={int(od['max'])}")
    if stats.get("tier_counts") is not None:
        print(f"[tier_counts] {_format_counts(np.asarray(stats['tier_counts'], dtype=np.int64))}")
    if stats.get("node_city_counts") is not None:
        print(f"[node_city] {_format_counts(np.asarray(stats['node_city_counts'], dtype=np.int64))}")
    if stats.get("edge_city_counts") is not None:
        print(f"[edge_city] {_format_counts(np.asarray(stats['edge_city_counts'], dtype=np.int64))}")

    if args.out_json is not None:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()

