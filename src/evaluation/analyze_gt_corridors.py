from __future__ import annotations

"""Analyze corridor distribution in GT way-route data (per OD, LCS clustering)."""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np

from src.data.way_graph.way_sequence_dataset import WayRoutes, load_way_routes_npz
from src.evaluation.corridor_clustering import extract_corridors

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def _route_seq(routes: WayRoutes, rid: int) -> np.ndarray:
    rid_i = int(rid)
    L = int(routes.way_seq_len[rid_i])
    if L <= 0:
        return np.asarray([], dtype=np.int64)
    s = int(routes.way_seq_ptr[rid_i])
    e = s + L
    return np.asarray(routes.way_seq_idx[s:e], dtype=np.int64)


def analyze_corridors(
    *,
    way_routes_npz: str,
    min_routes_per_od: int = 5,
    lcs_threshold: float = 0.5,
    cluster_method: str = "average",
    max_routes: Optional[int] = None,
    seed: int = 0,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    分析GT数据中的corridor分布（按(city, OD)分组）。

    Args:
        way_routes_npz: 路径到way_routes.npz（CSR: way_seq_ptr/idx/len）
        min_routes_per_od: OD至少有多少条route才分析
        lcs_threshold: LCS相似度阈值
        cluster_method: "average"/"complete"/"single"(scipy) 或 "graph"(无scipy)
        max_routes: 仅用于调试：最多取多少条routes参与统计（随机采样）
        seed: max_routes采样随机种子
        output_path: 输出JSON路径（可选）

    Returns:
        分析结果字典
    """
    log.info(f"Loading routes from {way_routes_npz}")
    routes = load_way_routes_npz(Path(way_routes_npz))
    n_routes = int(routes.way_seq_len.size)
    log.info(f"Loaded {n_routes} routes")

    route_ids = np.arange(n_routes, dtype=np.int64)
    if max_routes is not None and int(max_routes) < n_routes:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(route_ids)
        route_ids = route_ids[: int(max_routes)]
        log.info(f"Subsampled routes: {int(route_ids.size)}/{n_routes} (seed={int(seed)})")

    od_groups: DefaultDict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for rid in route_ids.tolist():
        rr = int(rid)
        city = int(routes.route_city[rr])
        start = int(routes.start_way[rr])
        dest = int(routes.dest_way[rr])
        od_groups[(city, start, dest)].append(rr)
    log.info(f"Found {len(od_groups)} unique (city, start_way, dest_way) pairs")

    valid_ods = {k: v for k, v in od_groups.items() if int(len(v)) >= int(min_routes_per_od)}
    log.info(f"Found {len(valid_ods)} ODs with >= {int(min_routes_per_od)} routes")

    results: Dict[str, Any] = {
        "config": {
            "way_routes_npz": str(way_routes_npz),
            "min_routes_per_od": int(min_routes_per_od),
            "lcs_threshold": float(lcs_threshold),
            "cluster_method": str(cluster_method),
            "max_routes": int(max_routes) if max_routes is not None else None,
            "seed": int(seed),
        },
        "summary": {},
        "per_od": [],
    }

    all_n_corridors: List[int] = []
    all_effective_k: List[float] = []

    for (city, start, dest), rids in valid_ods.items():
        seqs = [_route_seq(routes, rid) for rid in rids]
        corridor_result = extract_corridors(seqs, threshold=float(lcs_threshold), method=str(cluster_method))

        n_corridors = int(corridor_result["n_corridors"])
        corridor_sizes = [int(x) for x in corridor_result["corridor_sizes"]]

        if n_corridors > 0 and sum(corridor_sizes) > 0:
            probs = np.asarray(corridor_sizes, dtype=np.float64) / float(sum(corridor_sizes))
            entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
            effective_k = float(np.exp(entropy))
        else:
            effective_k = 0.0

        all_n_corridors.append(int(n_corridors))
        all_effective_k.append(float(effective_k))

        labels = np.asarray(corridor_result["labels"], dtype=np.int64).reshape(-1)
        proto_local = [int(x) for x in corridor_result["corridor_prototypes"]]
        proto_global = [int(rids[p]) for p in proto_local]

        results["per_od"].append(
            {
                "city": int(city),
                "start_way": int(start),
                "dest_way": int(dest),
                "od_key": f"{int(start)}_{int(dest)}",
                "n_routes": int(len(rids)),
                "n_corridors": int(n_corridors),
                "corridor_sizes": corridor_sizes,
                "effective_k": float(effective_k),
                "route_indices": [int(x) for x in rids],
                "corridor_labels": labels.tolist(),
                "prototype_indices": proto_global,
                "cluster_method_used": str(corridor_result.get("method_used", str(cluster_method))),
            }
        )

    dist = {
        "1": int(sum(1 for n in all_n_corridors if int(n) == 1)),
        "2": int(sum(1 for n in all_n_corridors if int(n) == 2)),
        "3": int(sum(1 for n in all_n_corridors if int(n) == 3)),
        "4+": int(sum(1 for n in all_n_corridors if int(n) >= 4)),
    }
    results["summary"] = {
        "n_valid_ods": int(len(valid_ods)),
        "avg_n_corridors": float(np.mean(np.asarray(all_n_corridors, dtype=np.float64))) if all_n_corridors else 0.0,
        "median_n_corridors": float(np.median(np.asarray(all_n_corridors, dtype=np.float64))) if all_n_corridors else 0.0,
        "avg_effective_k": float(np.mean(np.asarray(all_effective_k, dtype=np.float64))) if all_effective_k else 0.0,
        "median_effective_k": float(np.median(np.asarray(all_effective_k, dtype=np.float64))) if all_effective_k else 0.0,
        "n_corridors_distribution": dist,
    }

    log.info(
        "Summary: n_valid_ods=%d avg_n_corridors=%.2f avg_effective_k=%.2f",
        int(results["summary"]["n_valid_ods"]),
        float(results["summary"]["avg_n_corridors"]),
        float(results["summary"]["avg_effective_k"]),
    )

    if output_path:
        out = Path(str(output_path))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.info(f"Saved to {out}")

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze GT corridor distribution (LCS clustering per OD).")
    p.add_argument("--way_routes_npz", type=str, required=True)
    p.add_argument("--min_routes_per_od", type=int, default=5)
    p.add_argument("--lcs_threshold", type=float, default=0.5)
    p.add_argument("--cluster_method", type=str, default="average", help="average/complete/single (scipy) or graph (no scipy).")
    p.add_argument("--max_routes", type=int, default=None, help="Debug-only: subsample total routes (random).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()

    analyze_corridors(
        way_routes_npz=str(args.way_routes_npz),
        min_routes_per_od=int(args.min_routes_per_od),
        lcs_threshold=float(args.lcs_threshold),
        cluster_method=str(args.cluster_method),
        max_routes=int(args.max_routes) if args.max_routes is not None else None,
        seed=int(args.seed),
        output_path=str(args.output),
    )


if __name__ == "__main__":
    main()

