from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.data.way_graph.build_region_graph import _csr_to_edge_weights, _import_louvain
from src.data.way_graph.build_way_regions_louvain_per_city import _infer_way_city_from_routes

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    seed: int
    resolutions: List[float]
    min_component_size: int
    keep_only_largest_cc: bool


def _q_int(x: np.ndarray, q: float) -> int:
    a = np.asarray(x, dtype=np.int64).reshape(-1)
    if a.size == 0:
        return 0
    return int(np.quantile(a, float(q)))


def _region_sizes_from_partition(part: Dict[int, int], nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
    lab = np.full((int(nodes.size),), -1, dtype=np.int64)
    for i, u in enumerate(nodes.tolist()):
        c = part.get(int(u))
        if c is not None:
            lab[int(i)] = int(c)
    assigned = lab >= 0
    if not bool(np.any(assigned)):
        return np.zeros((0,), dtype=np.int64)
    uniq = np.unique(lab[assigned].astype(np.int64, copy=False))
    mapping = {int(old): int(i) for i, old in enumerate(uniq.tolist())}
    reg = np.full_like(lab, -1, dtype=np.int64)
    for i, x in enumerate(lab.tolist()):
        if int(x) >= 0:
            reg[int(i)] = int(mapping[int(x)])
    sizes = np.bincount(reg[assigned].astype(np.int64, copy=False), minlength=int(uniq.size)).astype(np.int64, copy=False)
    return sizes


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep Louvain resolutions per-city and report region count/size stats.")
    ap.add_argument("--way_graph_npz", type=Path, required=True)
    ap.add_argument("--way_routes_npz", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolutions", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--min_component_size", type=int, default=2)
    ap.add_argument("--keep_only_largest_cc", action="store_true")
    args = ap.parse_args()

    cfg = Cfg(
        seed=int(args.seed),
        resolutions=[float(x) for x in list(args.resolutions or [])],
        min_component_size=int(args.min_component_size),
        keep_only_largest_cc=bool(args.keep_only_largest_cc),
    )

    data = np.load(str(args.way_graph_npz), allow_pickle=True)
    need = {"way_osm_id", "way_adj_ptr", "way_adj_idx"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise SystemExit(f"[FATAL] way_graph.npz missing keys: {missing}")
    way_osm_id = np.asarray(data["way_osm_id"], dtype=np.int64).reshape(-1)
    ptr = np.asarray(data["way_adj_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(data["way_adj_idx"], dtype=np.int64).reshape(-1)
    n_ways = int(way_osm_id.size)

    way_city, conflicts = _infer_way_city_from_routes(way_routes_npz=Path(args.way_routes_npz), way_osm_id_graph=way_osm_id)
    city_ids = sorted(set(int(x) for x in way_city.tolist() if int(x) >= 0))
    unknown_n = int(np.sum(way_city < 0))

    nx, cl = _import_louvain()

    ew = _csr_to_edge_weights(ptr=ptr, idx=idx)
    G = nx.Graph()
    G.add_nodes_from(range(n_ways))
    for (u, v), w in ew.items():
        if int(u) != int(v) and int(w) > 0:
            G.add_edge(int(u), int(v), weight=float(w))

    rows: List[Dict[str, Any]] = []
    for res in cfg.resolutions:
        row: Dict[str, Any] = {"resolution": float(res), "per_city": {}}
        for city in city_ids:
            nodes = np.nonzero(way_city == int(city))[0].astype(np.int64, copy=False)
            H0 = G.subgraph(nodes.tolist()).copy()
            comps = list(nx.connected_components(H0))
            comps = sorted(comps, key=lambda s: len(s), reverse=True)
            if bool(cfg.keep_only_largest_cc):
                base_nodes = set(comps[0]) if comps else set()
            else:
                base_nodes = set()
                for c in comps:
                    if len(c) >= int(cfg.min_component_size):
                        base_nodes |= set(c)
            H = H0.subgraph(base_nodes).copy() if base_nodes else H0

            try:
                part = cl.best_partition(H, weight="weight", random_state=int(cfg.seed), resolution=float(res))
            except TypeError:
                part = cl.best_partition(H, weight="weight", random_state=int(cfg.seed))

            sizes = _region_sizes_from_partition(part, nodes)
            n_regions = int(sizes.size)
            row["per_city"][str(int(city))] = {
                "n_ways": int(nodes.size),
                "n_regions": int(n_regions),
                "region_size": {"p50": _q_int(sizes, 0.50), "p90": _q_int(sizes, 0.90), "p95": _q_int(sizes, 0.95), "max": int(sizes.max()) if sizes.size else 0},
            }
        rows.append(row)

        # Print a compact summary per resolution for quick scan.
        parts = []
        for city in city_ids:
            pc = row["per_city"].get(str(int(city)), {})
            parts.append(f"c{city}: nR={pc.get('n_regions')} p50={pc.get('region_size', {}).get('p50')}")
        print(f"[res={res:g}] " + " | ".join(parts))

    rep: Dict[str, Any] = {
        "ok": True,
        "task": "sweep_way_regions_louvain_per_city",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {"way_graph_npz": str(args.way_graph_npz), "way_routes_npz": str(args.way_routes_npz)},
        "n_ways": int(n_ways),
        "unknown_city_n": int(unknown_n),
        "unknown_city_frac": float(unknown_n / max(1, int(n_ways))),
        "route_city_conflicts": int(conflicts),
        "rows": rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {out_json}")


if __name__ == "__main__":
    main()

