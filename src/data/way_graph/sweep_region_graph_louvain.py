from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.data.way_graph.build_region_graph import (
    _build_region_adj_csr,
    _build_region_way_csr,
    _csr_to_edge_weights,
    _import_louvain,
    _remap_labels,
)

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    seed: int
    resolutions: List[float]
    min_component_size: int
    keep_only_largest_cc: bool
    directed_region_edges: bool
    save_npz: bool


def _q_int(x: np.ndarray, q: float) -> int:
    a = np.asarray(x, dtype=np.int64).reshape(-1)
    if a.size == 0:
        return 0
    return int(np.quantile(a, float(q)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep Louvain resolutions for way_graph -> regions (debugging utility).")
    ap.add_argument("--way_graph_npz", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolutions", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5])
    ap.add_argument("--min_component_size", type=int, default=2)
    ap.add_argument("--keep_only_largest_cc", action="store_true")
    ap.add_argument("--directed_region_edges", action="store_true")
    ap.add_argument("--save_npz", action="store_true", help="If set, save region_graph npz for each resolution.")
    args = ap.parse_args()

    cfg = Cfg(
        seed=int(args.seed),
        resolutions=[float(x) for x in list(args.resolutions or [])],
        min_component_size=int(args.min_component_size),
        keep_only_largest_cc=bool(args.keep_only_largest_cc),
        directed_region_edges=bool(args.directed_region_edges),
        save_npz=bool(args.save_npz),
    )

    data = np.load(str(args.way_graph_npz), allow_pickle=True)
    need = {"way_osm_id", "way_adj_ptr", "way_adj_idx"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise SystemExit(f"[FATAL] way_graph.npz missing keys: {missing}")
    way_osm_id = np.asarray(data["way_osm_id"]).reshape(-1)
    ptr = np.asarray(data["way_adj_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(data["way_adj_idx"], dtype=np.int64).reshape(-1)
    n_ways = int(way_osm_id.size)

    nx, cl = _import_louvain()

    ew = _csr_to_edge_weights(ptr=ptr, idx=idx)
    G = nx.Graph()
    G.add_nodes_from(range(n_ways))
    for (u, v), w in ew.items():
        if int(u) != int(v) and int(w) > 0:
            G.add_edge(int(u), int(v), weight=float(w))

    # Connectivity stats.
    comps = list(nx.connected_components(G))
    comps = sorted(comps, key=lambda s: len(s), reverse=True)
    n_comp = int(len(comps))
    largest_cc_n = int(len(comps[0])) if comps else 0
    deg0 = [n for n, d in G.degree() if int(d) == 0]
    isolate_n = int(len(deg0))

    # Determine node set for clustering.
    if bool(cfg.keep_only_largest_cc):
        base_nodes = set(comps[0]) if comps else set()
    else:
        base_nodes = set()
        for c in comps:
            if len(c) >= int(cfg.min_component_size):
                base_nodes |= set(c)
    H = G.subgraph(base_nodes).copy() if base_nodes else G

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for res in cfg.resolutions:
        try:
            part = cl.best_partition(H, weight="weight", random_state=int(cfg.seed), resolution=float(res))
        except TypeError:
            part = cl.best_partition(H, weight="weight", random_state=int(cfg.seed))

        lab = np.full((n_ways,), -1, dtype=np.int64)
        for u, c in part.items():
            uu = int(u)
            if 0 <= uu < n_ways:
                lab[uu] = int(c)

        way_region, n_regions, region_sizes = _remap_labels(lab)
        singleton_frac = float(np.mean(region_sizes == 1)) if region_sizes.size else 0.0
        assigned_n = int(np.sum(way_region >= 0))
        assigned_frac = float(assigned_n / max(1, n_ways))

        row = {
            "resolution": float(res),
            "n_regions": int(n_regions),
            "assigned_n": int(assigned_n),
            "assigned_frac": float(assigned_frac),
            "region_size": {"p50": _q_int(region_sizes, 0.50), "p90": _q_int(region_sizes, 0.90), "p95": _q_int(region_sizes, 0.95), "max": int(np.max(region_sizes)) if region_sizes.size else 0},
            "singleton_frac": float(singleton_frac),
        }
        rows.append(row)

        if bool(cfg.save_npz):
            region_way_ptr, region_way_idx = _build_region_way_csr(way_region, n_regions)
            region_adj_ptr, region_adj_idx, region_adj_w = _build_region_adj_csr(
                ptr=ptr,
                idx=idx,
                way_region=way_region,
                n_regions=int(n_regions),
                directed=bool(cfg.directed_region_edges),
            )
            out_npz = out_dir / f"way_regions_louvain_res{res:g}_seed{cfg.seed}.npz"
            np.savez_compressed(
                str(out_npz),
                way_region=way_region.astype(np.int32, copy=False),
                region_sizes=region_sizes.astype(np.int32, copy=False),
                region_way_ptr=region_way_ptr.astype(np.int64, copy=False),
                region_way_idx=region_way_idx.astype(np.int64, copy=False),
                region_adj_ptr=region_adj_ptr.astype(np.int64, copy=False),
                region_adj_idx=region_adj_idx.astype(np.int64, copy=False),
                region_adj_w=region_adj_w.astype(np.int64, copy=False),
                meta={
                    "task": "build_region_graph",
                    "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
                    "cfg": {
                        **asdict(cfg),
                        "resolution": float(res),
                    },
                    "inputs": {"way_graph_npz": str(args.way_graph_npz)},
                    "n_ways": int(n_ways),
                    "n_regions": int(n_regions),
                },
            )

        print(
            f"[res={res:g}] n_regions={row['n_regions']} "
            f"p50={row['region_size']['p50']} p90={row['region_size']['p90']} "
            f"assigned={row['assigned_frac']:.1%} singleton={row['singleton_frac']:.1%}"
        )

    rep = {
        "ok": True,
        "task": "sweep_region_graph_louvain",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {"way_graph_npz": str(args.way_graph_npz)},
        "graph": {
            "n_ways": int(n_ways),
            "n_connected_components_undirected": int(n_comp),
            "largest_cc_n": int(largest_cc_n),
            "largest_cc_frac": float(largest_cc_n / max(1, int(n_ways))),
            "isolated_deg0_n": int(isolate_n),
            "isolated_deg0_frac": float(isolate_n / max(1, int(n_ways))),
            "cluster_nodes_n": int(len(H.nodes())),
            "cluster_nodes_frac": float(len(H.nodes()) / max(1, int(n_ways))),
        },
        "rows": rows,
    }
    out_json = out_dir / f"sweep_region_graph_louvain_seed{cfg.seed}.json"
    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {out_json}")


if __name__ == "__main__":
    main()

