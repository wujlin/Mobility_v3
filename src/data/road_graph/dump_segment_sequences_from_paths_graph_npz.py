from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class DumpCfg:
    max_routes: Optional[int]


def _percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x.astype(np.float64), q))


def run_dump(
    *,
    paths_graph_npz: Path,
    road_graph_npz: Path,
    segment_graph_npz: Path,
    out_dir: Path,
    cfg: DumpCfg,
) -> Dict[str, object]:
    t0 = time.time()
    p = np.load(str(paths_graph_npz), allow_pickle=True)
    node_seq_pad = np.asarray(p["node_seq_pad"], dtype=np.int32)
    node_seq_len = np.asarray(p["node_seq_len"], dtype=np.int32).reshape(-1)
    start_node = np.asarray(p["start_node"], dtype=np.int32).reshape(-1)
    dest_node = np.asarray(p["dest_node"], dtype=np.int32).reshape(-1)
    start_t = np.asarray(p["start_t"], dtype=np.int64).reshape(-1)
    traj_idx = np.asarray(p["traj_idx"], dtype=np.int64).reshape(-1) if "traj_idx" in p.files else np.arange(int(start_node.size), dtype=np.int64)
    route_city = np.asarray(p["route_city"], dtype=np.int8).reshape(-1) if "route_city" in p.files else np.zeros_like(start_node, dtype=np.int8)
    start_pos = np.asarray(p["start_pos"], dtype=np.float32).reshape(-1, 2) if "start_pos" in p.files else None
    dest_pos = np.asarray(p["dest_pos"], dtype=np.float32).reshape(-1, 2) if "dest_pos" in p.files else None

    n_routes = int(start_node.size)
    if cfg.max_routes is not None:
        n_routes = min(n_routes, int(cfg.max_routes))
        node_seq_pad = node_seq_pad[:n_routes]
        node_seq_len = node_seq_len[:n_routes]
        start_node = start_node[:n_routes]
        dest_node = dest_node[:n_routes]
        start_t = start_t[:n_routes]
        traj_idx = traj_idx[:n_routes]
        route_city = route_city[:n_routes]
        if start_pos is not None:
            start_pos = start_pos[:n_routes]
        if dest_pos is not None:
            dest_pos = dest_pos[:n_routes]

    rg = np.load(str(road_graph_npz), allow_pickle=True)
    edge_u = np.asarray(rg["edge_u"], dtype=np.int32).reshape(-1)
    edge_v = np.asarray(rg["edge_v"], dtype=np.int32).reshape(-1)
    n_edges = int(edge_u.size)

    sg = np.load(str(segment_graph_npz), allow_pickle=True)
    edge_to_seg = np.asarray(sg["edge_to_seg"], dtype=np.int32).reshape(-1)
    seg_tier = np.asarray(sg["seg_tier"], dtype=np.uint8).reshape(-1) if "seg_tier" in sg.files else None
    if int(edge_to_seg.size) != int(n_edges):
        raise ValueError(f"edge_to_seg size mismatch: {int(edge_to_seg.size)} vs road_graph n_edges={int(n_edges)}")

    # Precompute (u,v)->edge_idx by sorting 64-bit keys: key=(u<<32)|v.
    keys = (edge_u.astype(np.uint64) << np.uint64(32)) | edge_v.astype(np.uint64)
    order = np.argsort(keys, kind="mergesort")
    keys_sorted = keys[order]

    seg_seq_ptr = np.zeros((n_routes + 1,), dtype=np.int64)
    seg_seq_len = np.zeros((n_routes,), dtype=np.int32)
    corridor_type = np.full((n_routes,), -1, dtype=np.int8)
    seg_seq_chunks: list[np.ndarray] = []

    missing_edges = 0
    total_steps = 0
    total_segments = 0

    for i in tqdm(range(n_routes), desc="dump_seg_seq", total=n_routes):
        L = int(node_seq_len[i])
        if L < 2:
            seg_seq_len[i] = 0
            seg_seq_ptr[i + 1] = seg_seq_ptr[i]
            continue
        nodes = node_seq_pad[i, :L].astype(np.int64, copy=False)
        u = nodes[:-1].astype(np.uint64, copy=False)
        v = nodes[1:].astype(np.uint64, copy=False)
        k = int(u.size)
        total_steps += k
        key_r = (u << np.uint64(32)) | v
        pos = np.searchsorted(keys_sorted, key_r)
        valid = (pos < n_edges) & (keys_sorted[pos] == key_r)
        if not bool(np.all(valid)):
            missing_edges += int((~valid).sum())
        if not bool(np.any(valid)):
            seg_seq_len[i] = 0
            seg_seq_ptr[i + 1] = seg_seq_ptr[i]
            continue

        edge_idx = order[pos[valid]].astype(np.int64, copy=False)
        seg_ids = edge_to_seg[edge_idx].astype(np.int32, copy=False)
        if seg_ids.size == 0:
            seg_seq_len[i] = 0
            seg_seq_ptr[i + 1] = seg_seq_ptr[i]
            continue

        # Collapse consecutive duplicates (many edges belong to the same segment).
        keep = np.ones((int(seg_ids.size),), dtype=bool)
        keep[1:] = seg_ids[1:] != seg_ids[:-1]
        seg_comp = seg_ids[keep]
        seg_seq_chunks.append(seg_comp.astype(np.int32, copy=False))
        seg_seq_len[i] = int(seg_comp.size)
        total_segments += int(seg_comp.size)
        seg_seq_ptr[i + 1] = seg_seq_ptr[i] + int(seg_comp.size)

        # Corridor type (KISS): dominant segment tier (>50%) among {0,1,2}; else mixed (3).
        if seg_tier is not None and int(seg_comp.size) > 0:
            tiers = seg_tier[seg_comp.astype(np.int64, copy=False)]
            cnt = np.bincount(tiers.astype(np.int64, copy=False), minlength=4).astype(np.int64, copy=False)
            dom = int(np.argmax(cnt))
            frac = float(cnt[dom] / float(max(int(seg_comp.size), 1)))
            if frac > 0.5 and dom in (0, 1, 2):
                corridor_type[i] = int(dom)
            else:
                corridor_type[i] = 3

    if seg_seq_chunks:
        seg_seq_idx = np.concatenate(seg_seq_chunks, axis=0).astype(np.int32, copy=False)
    else:
        seg_seq_idx = np.zeros((0,), dtype=np.int32)

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "dump_segment_sequences_from_paths_graph_npz",
        "inputs": {
            "paths_graph_npz": str(paths_graph_npz),
            "road_graph_npz": str(road_graph_npz),
            "segment_graph_npz": str(segment_graph_npz),
        },
        "config": {"max_routes": (int(cfg.max_routes) if cfg.max_routes is not None else None)},
        "stats": {
            "n_routes": int(n_routes),
            "total_steps": int(total_steps),
            "missing_edges": int(missing_edges),
            "missing_edge_frac": float(missing_edges / max(total_steps, 1)),
            "seg_len_p50": _percentile(seg_seq_len.astype(np.float32), 50),
            "seg_len_p90": _percentile(seg_seq_len.astype(np.float32), 90),
            "mean_seg_len": float(total_segments / max(n_routes, 1)),
            "corridor_type_counts": np.bincount(np.clip(corridor_type, 0, 3).astype(np.int64, copy=False), minlength=4).astype(np.int64).tolist(),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "segments_graph_routes.npz"
    report_json = out_dir / "report.json"
    np.savez_compressed(
        out_npz,
        seg_seq_ptr=seg_seq_ptr,
        seg_seq_idx=seg_seq_idx,
        seg_seq_len=seg_seq_len,
        corridor_type=corridor_type,
        start_node=start_node,
        dest_node=dest_node,
        start_t=start_t,
        traj_idx=traj_idx,
        route_city=route_city,
        start_pos=start_pos if start_pos is not None else np.zeros((n_routes, 2), dtype=np.float32),
        dest_pos=dest_pos if dest_pos is not None else np.zeros((n_routes, 2), dtype=np.float32),
        meta=meta,
    )

    report = {
        "ok": True,
        "out_npz": str(out_npz),
        "report_json": str(report_json),
        "stats": meta["stats"],
        "timing": {"elapsed_s": float(time.time() - t0)},
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump segment-id sequences per route by mapping node_seq to road segments.")
    p.add_argument("--paths_graph_npz", type=Path, required=True)
    p.add_argument("--road_graph_npz", type=Path, required=True)
    p.add_argument("--segment_graph_npz", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--max_routes", type=int, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = run_dump(
        paths_graph_npz=Path(args.paths_graph_npz),
        road_graph_npz=Path(args.road_graph_npz),
        segment_graph_npz=Path(args.segment_graph_npz),
        out_dir=Path(args.out_dir),
        cfg=DumpCfg(max_routes=(int(args.max_routes) if args.max_routes else None)),
    )
    compact = {
        "ok": True,
        "out_npz": report["out_npz"],
        "n_routes": int(report["stats"]["n_routes"]),
        "seg_len_p50": float(report["stats"]["seg_len_p50"]),
        "missing_edge_frac": float(report["stats"]["missing_edge_frac"]),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
