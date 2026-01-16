"""
Merge multiple way_routes.npz files from different cities into a single npz.

Usage:
    python -m src.data.way_graph.merge_way_routes_multi_city \
        --inputs city0/way_routes.npz city1/way_routes.npz \
        --route_cities 0 1 \
        --out_npz merged/way_routes.npz
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np


TZ_SHANGHAI = timezone(timedelta(hours=8))


def _p(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


@dataclass
class CityRoutes:
    way_osm_id: np.ndarray  # (M_i,) int64
    way_seq_ptr: np.ndarray  # (N_i+1,) int64
    way_seq_idx: np.ndarray  # (total_i,) int32
    way_seq_len: np.ndarray  # (N_i,) int32
    corridor_type: np.ndarray  # (N_i,) int8
    start_way: np.ndarray  # (N_i,) int32
    dest_way: np.ndarray  # (N_i,) int32
    start_t: np.ndarray  # (N_i,) int64
    route_city: np.ndarray  # (N_i,) int8
    start_pos: np.ndarray  # (N_i,2) float32
    dest_pos: np.ndarray  # (N_i,2) float32


def load_city_routes(path: Path) -> CityRoutes:
    data = np.load(str(path), allow_pickle=True)
    need = {
        "way_osm_id", "way_seq_ptr", "way_seq_idx", "way_seq_len",
        "start_way", "dest_way", "start_t", "route_city", "start_pos", "dest_pos",
    }
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"{path} missing keys: {missing}")

    corridor_type = data["corridor_type"] if "corridor_type" in data.files else np.full(
        (data["way_seq_len"].shape[0],), -1, dtype=np.int8
    )

    return CityRoutes(
        way_osm_id=np.asarray(data["way_osm_id"], dtype=np.int64).reshape(-1),
        way_seq_ptr=np.asarray(data["way_seq_ptr"], dtype=np.int64),
        way_seq_idx=np.asarray(data["way_seq_idx"], dtype=np.int32),
        way_seq_len=np.asarray(data["way_seq_len"], dtype=np.int32).reshape(-1),
        corridor_type=np.asarray(corridor_type, dtype=np.int8).reshape(-1),
        start_way=np.asarray(data["start_way"], dtype=np.int32).reshape(-1),
        dest_way=np.asarray(data["dest_way"], dtype=np.int32).reshape(-1),
        start_t=np.asarray(data["start_t"], dtype=np.int64).reshape(-1),
        route_city=np.asarray(data["route_city"], dtype=np.int8).reshape(-1),
        start_pos=np.asarray(data["start_pos"], dtype=np.float32).reshape(-1, 2),
        dest_pos=np.asarray(data["dest_pos"], dtype=np.float32).reshape(-1, 2),
    )


def merge_routes(inputs: List[Path], route_cities: List[int], out_npz: Path) -> Dict[str, object]:
    """
    Merge multiple city way_routes.npz into one.

    Key insight: way_osm_id is globally unique (OSM IDs are global), so we can build
    a unified way vocab by taking the union of all per-city way vocabs.
    """
    if len(inputs) != len(route_cities):
        raise ValueError(f"inputs ({len(inputs)}) and route_cities ({len(route_cities)}) must have same length")

    # Load all cities.
    city_data: List[CityRoutes] = []
    for p in inputs:
        city_data.append(load_city_routes(p))

    # Build unified way vocab (union of all osm_way_ids).
    all_way_ids: set[int] = set()
    for cd in city_data:
        all_way_ids.update(cd.way_osm_id.tolist())
    merged_way_osm_id = np.asarray(sorted(list(all_way_ids)), dtype=np.int64)
    M = int(merged_way_osm_id.size)
    global_way_to_idx = {int(w): int(i) for i, w in enumerate(merged_way_osm_id.tolist())}

    # Remap each city's sequences to global indices.
    all_seqs: List[np.ndarray] = []
    all_lens: List[int] = []
    all_start_way: List[int] = []
    all_dest_way: List[int] = []
    all_start_t: List[int] = []
    all_route_city: List[int] = []
    all_corridor_type: List[int] = []
    all_start_pos: List[np.ndarray] = []
    all_dest_pos: List[np.ndarray] = []

    for ci, cd in enumerate(city_data):
        city_id = int(route_cities[ci])
        local_way_to_global = {int(i): global_way_to_idx[int(w)] for i, w in enumerate(cd.way_osm_id.tolist())}

        N_i = int(cd.way_seq_len.size)
        for r in range(N_i):
            L = int(cd.way_seq_len[r])
            s = int(cd.way_seq_ptr[r])
            e = s + L
            local_seq = cd.way_seq_idx[s:e]
            global_seq = np.asarray([local_way_to_global[int(x)] for x in local_seq], dtype=np.int32)

            all_seqs.append(global_seq)
            all_lens.append(L)
            all_start_way.append(int(global_way_to_idx[int(cd.way_osm_id[cd.start_way[r]])]))
            all_dest_way.append(int(global_way_to_idx[int(cd.way_osm_id[cd.dest_way[r]])]))
            all_start_t.append(int(cd.start_t[r]))
            all_route_city.append(city_id)
            all_corridor_type.append(int(cd.corridor_type[r]))
            all_start_pos.append(cd.start_pos[r])
            all_dest_pos.append(cd.dest_pos[r])

    # Build CSR.
    N = len(all_seqs)
    ptr = np.zeros((N + 1,), dtype=np.int64)
    flat: List[int] = []
    for i, seq in enumerate(all_seqs):
        flat.extend(seq.tolist())
        ptr[i + 1] = np.int64(len(flat))

    way_seq_idx = np.asarray(flat, dtype=np.int32)
    way_seq_len = np.asarray(all_lens, dtype=np.int32)
    start_way = np.asarray(all_start_way, dtype=np.int32)
    dest_way = np.asarray(all_dest_way, dtype=np.int32)
    start_t = np.asarray(all_start_t, dtype=np.int64)
    route_city = np.asarray(all_route_city, dtype=np.int8)
    corridor_type = np.asarray(all_corridor_type, dtype=np.int8)
    start_pos = np.stack(all_start_pos, axis=0).astype(np.float32, copy=False)
    dest_pos = np.stack(all_dest_pos, axis=0).astype(np.float32, copy=False)

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "merge_way_routes_multi_city",
        "inputs": [str(p) for p in inputs],
        "route_cities": route_cities,
        "stats": {
            "n_routes": int(N),
            "n_way_vocab": int(M),
            "way_seq_len": {"p50": _p(way_seq_len, 50), "p90": _p(way_seq_len, 90), "max": int(np.max(way_seq_len) if way_seq_len.size else 0)},
            "per_city": {int(c): int(np.sum(route_city == c)) for c in route_cities},
        },
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        way_osm_id=merged_way_osm_id,
        way_seq_ptr=ptr,
        way_seq_idx=way_seq_idx,
        way_seq_len=way_seq_len,
        corridor_type=corridor_type,
        start_way=start_way,
        dest_way=dest_way,
        start_t=start_t,
        route_city=route_city,
        start_pos=start_pos,
        dest_pos=dest_pos,
        meta=meta,
    )
    return {"ok": True, "out_npz": str(out_npz), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge multiple city way_routes.npz into one unified npz.")
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="List of way_routes.npz files to merge.")
    p.add_argument("--route_cities", type=int, nargs="+", required=True, help="Route city IDs for each input (same order).")
    p.add_argument("--out_npz", type=Path, required=True, help="Output merged way_routes.npz.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = merge_routes(
        inputs=[Path(p) for p in args.inputs],
        route_cities=[int(c) for c in args.route_cities],
        out_npz=Path(args.out_npz),
    )
    meta = report["meta"]
    st = meta["stats"]
    compact = {
        "ok": True,
        "out_npz": report["out_npz"],
        "n_routes": int(st["n_routes"]),
        "n_way_vocab": int(st["n_way_vocab"]),
        "way_seq_len_p50": float(st["way_seq_len"]["p50"]),
        "way_seq_len_p90": float(st["way_seq_len"]["p90"]),
        "per_city": st["per_city"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
