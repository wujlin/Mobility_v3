from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Config:
    make_undirected: bool


def _p(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def build_way_graph(*, way_routes_npz: Path, out_npz: Path, cfg: Config) -> Dict[str, object]:
    data = np.load(str(way_routes_npz), allow_pickle=True)
    need = {"way_osm_id", "way_seq_ptr", "way_seq_idx", "way_seq_len"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"way_routes.npz missing keys: {missing}")

    way_osm_id = np.asarray(data["way_osm_id"], dtype=np.int64).reshape(-1)
    ptr = np.asarray(data["way_seq_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(data["way_seq_idx"], dtype=np.int64).reshape(-1)
    lens = np.asarray(data["way_seq_len"], dtype=np.int64).reshape(-1)

    M = int(way_osm_id.size)
    adj: List[set[int]] = [set() for _ in range(M)]
    n_routes = int(lens.size)
    for r in range(n_routes):
        L = int(lens[r])
        if L <= 1:
            continue
        s = int(ptr[r])
        e = s + L
        seq = idx[s:e]
        for j in range(L - 1):
            a = int(seq[j])
            b = int(seq[j + 1])
            if a < 0 or b < 0 or a >= M or b >= M:
                continue
            if a == b:
                continue
            adj[a].add(b)
            if bool(cfg.make_undirected):
                adj[b].add(a)

    out_ptr = np.zeros((M + 1,), dtype=np.int64)
    out_idx: List[int] = []
    out_deg = np.zeros((M,), dtype=np.int64)
    for i in range(M):
        nbrs = sorted(list(adj[i]))
        out_deg[i] = int(len(nbrs))
        out_idx.extend(nbrs)
        out_ptr[i + 1] = np.int64(len(out_idx))

    out_idx_arr = np.asarray(out_idx, dtype=np.int32)

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "build_way_graph_from_way_routes_npz",
        "inputs": {"way_routes_npz": str(way_routes_npz)},
        "config": {"make_undirected": bool(cfg.make_undirected)},
        "stats": {
            "n_ways": int(M),
            "n_edges_directed": int(out_idx_arr.size),
            "out_deg": {"p50": _p(out_deg, 50), "p90": _p(out_deg, 90), "max": int(np.max(out_deg) if out_deg.size else 0)},
        },
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        way_osm_id=way_osm_id,
        way_adj_ptr=out_ptr,
        way_adj_idx=out_idx_arr,
        meta=meta,
    )
    return {"ok": True, "out_npz": str(out_npz), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build way adjacency (CSR) from way_routes.npz transitions.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--out_npz", type=Path, required=True)
    p.add_argument("--make_undirected", action="store_true", help="Also add reverse edges (KISS default: directed).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = build_way_graph(
        way_routes_npz=Path(args.way_routes_npz),
        out_npz=Path(args.out_npz),
        cfg=Config(make_undirected=bool(args.make_undirected)),
    )
    meta = report["meta"]
    st = meta["stats"]
    compact = {
        "ok": True,
        "out_npz": report["out_npz"],
        "n_ways": int(st["n_ways"]),
        "n_edges_directed": int(st["n_edges_directed"]),
        "out_deg_p50": float(st["out_deg"]["p50"]),
        "out_deg_p90": float(st["out_deg"]["p90"]),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

