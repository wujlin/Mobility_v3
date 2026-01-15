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


def audit(*, routes_npz: Path) -> Dict[str, Any]:
    data = np.load(str(routes_npz), allow_pickle=True)
    need = {"seg_seq_len", "corridor_type"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise SystemExit(f"segments_graph_routes.npz missing keys: {missing} ({routes_npz})")

    seg_seq_len = np.asarray(data["seg_seq_len"], dtype=np.int32).reshape(-1)
    corridor_type = np.asarray(data["corridor_type"], dtype=np.int8).reshape(-1)
    n = int(seg_seq_len.size)

    corridor_counts = np.bincount(np.clip(corridor_type.astype(np.int64, copy=False), 0, 3), minlength=4).astype(np.int64).tolist()
    route_city = np.asarray(data["route_city"], dtype=np.int8).reshape(-1) if "route_city" in data.files else None
    city_counts = None
    if route_city is not None and int(route_city.size) == n:
        city_counts = np.bincount(np.clip(route_city.astype(np.int64, copy=False), 0, 16), minlength=2).astype(np.int64).tolist()

    meta = _load_meta(data)
    stats_meta = meta.get("stats") if isinstance(meta, dict) else None

    report: Dict[str, Any] = {
        "inputs": {"routes_npz": str(routes_npz)},
        "meta": {"stats": stats_meta},
        "stats": {
            "n_routes": int(n),
            "seg_seq_len": {"p50": _p(seg_seq_len, 50), "p90": _p(seg_seq_len, 90), "max": int(seg_seq_len.max()) if n else 0},
            "corridor_type_counts": corridor_counts,
            "route_city_counts": city_counts,
            "missing_edge_frac": (float(stats_meta["missing_edge_frac"]) if isinstance(stats_meta, dict) and "missing_edge_frac" in stats_meta else None),
        },
    }
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit segments_graph_routes.npz stats (segment length, corridor type) with compact prints.")
    p.add_argument("--routes_npz", type=Path, required=True)
    p.add_argument("--out_json", type=Path, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = audit(routes_npz=Path(args.routes_npz))
    stats = report["stats"]
    print(f"[segment_routes] {report['inputs']['routes_npz']}")
    print(f"[N] {int(stats['n_routes'])}")
    sl = stats["seg_seq_len"]
    print(f"[seg_seq_len] p50={sl['p50']:.1f} p90={sl['p90']:.1f} max={int(sl['max'])}")
    print(f"[corridor_type] {_format_counts(np.asarray(stats['corridor_type_counts'], dtype=np.int64))}")
    if stats.get("route_city_counts") is not None:
        print(f"[route_city] {_format_counts(np.asarray(stats['route_city_counts'], dtype=np.int64))}")
    if stats.get("missing_edge_frac") is not None:
        print(f"[missing_edge_frac] {float(stats['missing_edge_frac']):.4f}")

    if args.out_json is not None:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()

