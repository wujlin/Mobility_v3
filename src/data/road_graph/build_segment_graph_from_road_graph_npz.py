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
class BuildCfg:
    max_edges: Optional[int]
    paths_graph_npz: Optional[Path]
    mode: str = "collapse"


def _percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x.astype(np.float64), q))


def build_segment_graph(*, road_graph_npz: Path, out_dir: Path, cfg: BuildCfg) -> Dict[str, object]:
    t0 = time.time()
    raw = np.load(str(road_graph_npz), allow_pickle=True)
    node_y = np.asarray(raw["node_y"], dtype=np.float32).reshape(-1)
    node_x = np.asarray(raw["node_x"], dtype=np.float32).reshape(-1)
    node_city = np.asarray(raw["node_city"], dtype=np.int8).reshape(-1) if "node_city" in raw.files else None

    edge_u = np.asarray(raw["edge_u"], dtype=np.int32).reshape(-1)
    edge_v = np.asarray(raw["edge_v"], dtype=np.int32).reshape(-1)
    edge_w_m = np.asarray(raw["edge_w_m"], dtype=np.float32).reshape(-1)
    edge_tier = np.asarray(raw["edge_tier"], dtype=np.uint8).reshape(-1)
    edge_city = np.asarray(raw["edge_city"], dtype=np.int8).reshape(-1) if "edge_city" in raw.files else None

    n_nodes = int(node_y.size)
    n_edges = int(edge_u.size)
    if cfg.max_edges is not None:
        n_edges = min(n_edges, int(cfg.max_edges))
        edge_u = edge_u[:n_edges]
        edge_v = edge_v[:n_edges]
        edge_w_m = edge_w_m[:n_edges]
        edge_tier = edge_tier[:n_edges]
        if edge_city is not None:
            edge_city = edge_city[:n_edges]

    if n_nodes <= 0 or n_edges <= 0:
        raise ValueError(f"Empty road graph: n_nodes={n_nodes}, n_edges={n_edges}")

    terminal_mask = None
    n_terminal_nodes = 0
    if cfg.paths_graph_npz is not None:
        p = np.load(str(cfg.paths_graph_npz), allow_pickle=True)
        if "start_node" not in p.files or "dest_node" not in p.files:
            raise ValueError(f"paths_graph_npz missing start_node/dest_node: {str(cfg.paths_graph_npz)}")
        start_node = np.asarray(p["start_node"], dtype=np.int64).reshape(-1)
        dest_node = np.asarray(p["dest_node"], dtype=np.int64).reshape(-1)
        terminal_nodes = np.unique(np.concatenate([start_node, dest_node], axis=0))
        terminal_nodes = terminal_nodes[(terminal_nodes >= 0) & (terminal_nodes < n_nodes)]
        terminal_mask = np.zeros((n_nodes,), dtype=bool)
        terminal_mask[terminal_nodes.astype(np.int64, copy=False)] = True
        n_terminal_nodes = int(terminal_nodes.size)

    # Out-degree (CSR over directed edges).
    out_deg = np.bincount(edge_u.astype(np.int64), minlength=n_nodes).astype(np.int32, copy=False)

    seg_mode = str(cfg.mode).strip().lower()
    if seg_mode not in {"collapse", "edge"}:
        raise ValueError(f"Unknown --mode: {cfg.mode} (expected 'collapse' or 'edge').")

    if seg_mode == "edge":
        # Each directed edge is a segment: seg_id == edge_id.
        seg_u = edge_u.astype(np.int32, copy=False)
        seg_v = edge_v.astype(np.int32, copy=False)
        seg_len_m = edge_w_m.astype(np.float32, copy=False)
        seg_tier = edge_tier.astype(np.uint8, copy=False)
        if edge_city is not None:
            seg_city = edge_city.astype(np.int8, copy=False)
        elif node_city is not None:
            seg_city = node_city[seg_u.astype(np.int64, copy=False)].astype(np.int8, copy=False)
        else:
            seg_city = np.zeros((n_edges,), dtype=np.int8)

        edge_to_seg = np.arange(n_edges, dtype=np.int32)
        seg_edges = np.arange(n_edges, dtype=np.int32)
        seg_ptr = np.arange(n_edges + 1, dtype=np.int64)

        n_segs = int(seg_u.size)
        if n_segs <= 0:
            raise RuntimeError("No segments built (edge mode, n_segs=0).")
    else:
        order = np.argsort(edge_u, kind="mergesort")
        counts = out_deg.astype(np.int64, copy=False)
        ptr = np.zeros(n_nodes + 1, dtype=np.int64)
        np.cumsum(counts, out=ptr[1:])
        edge_v_sorted = edge_v[order]

        edge_to_seg = np.full((n_edges,), -1, dtype=np.int32)
        seg_u_list: list[int] = []
        seg_v_list: list[int] = []
        seg_len_list: list[float] = []
        seg_tier_list: list[int] = []
        seg_city_list: list[int] = []

        seg_ptr_list: list[int] = [0]
        seg_edges = np.empty((n_edges,), dtype=np.int32)
        write_pos = 0

        it = range(n_edges)
        for e0 in tqdm(it, desc="build_seg_graph", total=n_edges):
            if int(edge_to_seg[int(e0)]) != -1:
                continue

            u0 = int(edge_u[int(e0)])
            v0 = int(edge_v[int(e0)])
            seg_id = int(len(seg_u_list))

            seg_u_list.append(u0)
            tier_min = int(edge_tier[int(e0)])
            len_sum = float(edge_w_m[int(e0)])
            if edge_city is not None:
                seg_city_list.append(int(edge_city[int(e0)]))
            elif node_city is not None:
                seg_city_list.append(int(node_city[u0]))
            else:
                seg_city_list.append(0)

            edge_to_seg[int(e0)] = seg_id
            seg_edges[write_pos] = int(e0)
            write_pos += 1

            prev = int(u0)
            cur = int(v0)

            # Follow through degree-2 chain (grid-road raster graph).
            while True:
                if cur == u0:
                    break  # loop closed
                if terminal_mask is not None and bool(terminal_mask[cur]):
                    break  # force boundary at route terminal nodes (start/dest)
                if int(out_deg[cur]) != 2:
                    break  # hit junction / dead-end
                s = int(ptr[cur])
                e = int(ptr[cur + 1])
                if e - s != 2:
                    break

                eidx_a = int(order[s + 0])
                eidx_b = int(order[s + 1])
                va = int(edge_v_sorted[s + 0])
                vb = int(edge_v_sorted[s + 1])

                if va == prev:
                    nxt = eidx_b
                elif vb == prev:
                    nxt = eidx_a
                else:
                    # Unexpected: degree==2 but neither edge goes back to prev.
                    break

                if int(edge_to_seg[nxt]) != -1:
                    break  # already assigned (should be rare); terminate here

                edge_to_seg[nxt] = seg_id
                seg_edges[write_pos] = int(nxt)
                write_pos += 1

                tier_min = min(tier_min, int(edge_tier[nxt]))
                len_sum += float(edge_w_m[nxt])

                prev, cur = cur, int(edge_v[nxt])

            seg_v_list.append(int(cur))
            seg_len_list.append(float(len_sum))
            seg_tier_list.append(int(tier_min))
            seg_ptr_list.append(int(write_pos))

        seg_edges = seg_edges[:write_pos]
        seg_u = np.asarray(seg_u_list, dtype=np.int32)
        seg_v = np.asarray(seg_v_list, dtype=np.int32)
        seg_len_m = np.asarray(seg_len_list, dtype=np.float32)
        seg_tier = np.asarray(seg_tier_list, dtype=np.uint8)
        seg_city = np.asarray(seg_city_list, dtype=np.int8)
        seg_ptr = np.asarray(seg_ptr_list, dtype=np.int64)

        n_segs = int(seg_u.size)
        if n_segs <= 0:
            raise RuntimeError("No segments built (n_segs=0).")

    # Segment geometric features (KISS): endpoint-based center + direction.
    uy = node_y[seg_u]
    ux = node_x[seg_u]
    vy = node_y[seg_v]
    vx = node_x[seg_v]
    seg_center_y = ((uy + vy) * 0.5).astype(np.float32, copy=False)
    seg_center_x = ((ux + vx) * 0.5).astype(np.float32, copy=False)
    dy = (vy - uy).astype(np.float32, copy=False)
    dx = (vx - ux).astype(np.float32, copy=False)
    nrm = np.sqrt(dy * dy + dx * dx).astype(np.float32, copy=False)
    nrm = np.maximum(nrm, np.float32(1e-6))
    seg_dir_y = (dy / nrm).astype(np.float32, copy=False)
    seg_dir_x = (dx / nrm).astype(np.float32, copy=False)

    # Segment adjacency (CSR) via node -> outgoing segments.
    seg_order = np.argsort(seg_u, kind="mergesort")
    seg_counts_by_node = np.bincount(seg_u.astype(np.int64), minlength=n_nodes).astype(np.int64, copy=False)
    node_seg_ptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(seg_counts_by_node, out=node_seg_ptr[1:])
    node_seg_idx = seg_order.astype(np.int32, copy=False)

    succ_counts = (node_seg_ptr[seg_v.astype(np.int64) + 1] - node_seg_ptr[seg_v.astype(np.int64)]).astype(np.int64, copy=False)
    succ_ptr = np.zeros(n_segs + 1, dtype=np.int64)
    np.cumsum(succ_counts, out=succ_ptr[1:])
    succ_idx = np.empty((int(succ_ptr[-1]),), dtype=np.int32)

    w = 0
    for i in range(n_segs):
        v = int(seg_v[i])
        s = int(node_seg_ptr[v])
        e = int(node_seg_ptr[v + 1])
        k = int(e - s)
        if k > 0:
            succ_idx[w : w + k] = node_seg_idx[s:e]
            w += k
    if w != int(succ_idx.size):
        succ_idx = succ_idx[:w]
        succ_ptr[-1] = int(w)

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "build_segment_graph_from_road_graph_npz",
        "inputs": {"road_graph_npz": str(road_graph_npz), "paths_graph_npz": (str(cfg.paths_graph_npz) if cfg.paths_graph_npz is not None else None)},
        "config": {"max_edges": (int(cfg.max_edges) if cfg.max_edges is not None else None), "mode": str(seg_mode)},
        "stats": {
            "n_nodes": int(n_nodes),
            "n_edges_directed": int(n_edges),
            "n_segments": int(n_segs),
            "n_terminal_nodes": int(n_terminal_nodes),
            "seg_len_m_p50": _percentile(seg_len_m, 50),
            "seg_len_m_p90": _percentile(seg_len_m, 90),
            "seg_edges_per_seg_p50": _percentile(np.diff(seg_ptr).astype(np.float32), 50),
            "seg_edges_per_seg_p90": _percentile(np.diff(seg_ptr).astype(np.float32), 90),
            "deg_p50": _percentile(out_deg.astype(np.float32), 50),
            "deg_p90": _percentile(out_deg.astype(np.float32), 90),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "segment_graph.npz"
    report_json = out_dir / "report.json"
    np.savez_compressed(
        out_npz,
        # segment topology
        seg_u=seg_u,
        seg_v=seg_v,
        seg_len_m=seg_len_m,
        seg_tier=seg_tier,
        seg_city=seg_city,
        # segment geometry (feature-based, no segment ID embedding required)
        seg_center_y=seg_center_y,
        seg_center_x=seg_center_x,
        seg_dir_y=seg_dir_y,
        seg_dir_x=seg_dir_x,
        # mapping for reconstruction / debugging
        seg_ptr=seg_ptr,
        seg_edges=seg_edges,
        edge_to_seg=edge_to_seg,
        # adjacency (CSR)
        seg_succ_ptr=succ_ptr,
        seg_succ_idx=succ_idx,
        # node->seg index (optional helper)
        node_seg_ptr=node_seg_ptr,
        node_seg_idx=node_seg_idx,
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
    p = argparse.ArgumentParser(description="Build a segment graph from road_graph.npz (collapse degree-2 chains, or edge-as-segment).")
    p.add_argument("--road_graph_npz", type=Path, required=True)
    p.add_argument("--paths_graph_npz", type=Path, default=None, help="Optional: in collapse mode, force segment boundaries at all route start/dest nodes from paths_graph.npz.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--max_edges", type=int, default=None, help="Debug: only use the first N directed edges.")
    p.add_argument("--mode", choices=["collapse", "edge"], default="collapse", help="Segment definition: collapse degree-2 chains, or treat each directed edge as a segment.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = build_segment_graph(
        road_graph_npz=Path(args.road_graph_npz),
        out_dir=Path(args.out_dir),
        cfg=BuildCfg(
            max_edges=(int(args.max_edges) if args.max_edges else None),
            paths_graph_npz=(Path(args.paths_graph_npz) if args.paths_graph_npz else None),
            mode=str(args.mode),
        ),
    )
    compact = {
        "ok": True,
        "out_npz": report["out_npz"],
        "n_segments": int(report["stats"]["n_segments"]),
        "seg_len_m_p50": float(report["stats"]["seg_len_m_p50"]),
        "seg_edges_per_seg_p50": float(report["stats"]["seg_edges_per_seg_p50"]),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
