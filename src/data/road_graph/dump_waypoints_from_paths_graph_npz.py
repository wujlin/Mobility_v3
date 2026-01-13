from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(x, *args, **kwargs):  # type: ignore[no-redef]
        return x

from src.features.waypoints import pick_waypoint_indices_rdp_fixed_k, pick_waypoint_indices_rdp_turn_fixed_k


TZ_SHANGHAI = timezone(timedelta(hours=8))


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


@dataclass(frozen=True)
class DumpCfg:
    num_waypoints: int
    mode: str
    turn_alpha: float
    seed: int


def _load_graph_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    need = {"node_y", "node_x", "meta"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"road_graph.npz missing keys: {missing}")
    meta = data["meta"]
    if isinstance(meta, np.ndarray) and meta.shape == ():
        meta = meta.item()
    if not isinstance(meta, dict):
        raise ValueError("road_graph.npz meta must be a dict.")
    return {
        "node_y": np.asarray(data["node_y"], dtype=np.float32).reshape(-1),
        "node_x": np.asarray(data["node_x"], dtype=np.float32).reshape(-1),
        "meta": meta,
    }


def _load_paths_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    need = {"start_t", "start_node", "dest_node", "node_seq_pad", "node_seq_len", "traj_idx"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"paths_graph.npz missing keys: {missing}")
    meta = data["meta"] if "meta" in data.files else None
    if isinstance(meta, np.ndarray) and meta.shape == ():
        meta = meta.item()
    return {
        "start_t": np.asarray(data["start_t"], dtype=np.int64).reshape(-1),
        "start_node": np.asarray(data["start_node"], dtype=np.int32).reshape(-1),
        "dest_node": np.asarray(data["dest_node"], dtype=np.int32).reshape(-1),
        "node_seq_pad": np.asarray(data["node_seq_pad"], dtype=np.int32),
        "node_seq_len": np.asarray(data["node_seq_len"], dtype=np.int32).reshape(-1),
        "traj_idx": np.asarray(data["traj_idx"], dtype=np.int64).reshape(-1),
        "route_city": np.asarray(data["route_city"], dtype=np.int8).reshape(-1) if "route_city" in data.files else None,
        "meta": meta if isinstance(meta, dict) else None,
    }


def _pick_idx(points: np.ndarray, *, cfg: DumpCfg) -> np.ndarray:
    k = int(cfg.num_waypoints)
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    mode = str(cfg.mode)
    if mode == "rdp_dev":
        return pick_waypoint_indices_rdp_fixed_k(points, k=k)
    if mode == "rdp_turn":
        return pick_waypoint_indices_rdp_turn_fixed_k(points, k=k, turn_alpha=float(cfg.turn_alpha))
    raise ValueError(f"Unknown mode {cfg.mode!r} (expected rdp_dev|rdp_turn)")


def run_dump(*, paths_graph_npz: Path, road_graph_npz: Path, out_dir: Path, cfg: DumpCfg, viz_cases: int) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "report.json"
    out_npz = out_dir / "waypoints_graph.npz"

    g = _load_graph_npz(road_graph_npz)
    node_y = g["node_y"]
    node_x = g["node_x"]
    meta_g = g["meta"]

    p = _load_paths_npz(paths_graph_npz)
    node_seq_pad = p["node_seq_pad"]
    node_seq_len = p["node_seq_len"]
    start_t = p["start_t"]
    traj_idx = p["traj_idx"]
    start_node = p["start_node"]
    dest_node = p["dest_node"]
    route_city = p["route_city"]
    meta_p = p["meta"]

    N = int(start_node.size)
    K = int(cfg.num_waypoints)
    wp_seq = np.full((N, K + 2), -1, dtype=np.int32)
    wp_len = np.full((N,), K + 2, dtype=np.int32)
    gt_len = np.asarray(node_seq_len, dtype=np.int32, copy=False).reshape(-1)

    good = 0
    for i in tqdm(range(N), desc="dump_waypoints", dynamic_ncols=True):
        L = int(node_seq_len[i])
        if L < 2:
            continue
        seq = node_seq_pad[i, :L].astype(np.int64, copy=False)
        # Build polyline points (y,x).
        yy = node_y[seq]
        xx = node_x[seq]
        pts = np.stack([yy, xx], axis=1).astype(np.float32, copy=False)

        if L < (K + 2):
            # Fallback: time quantiles (always include endpoints).
            idx = np.linspace(1, max(1, L - 2), num=K, dtype=np.float32)
            idx = np.clip(np.rint(idx), 1, max(1, L - 2)).astype(np.int64, copy=False)
            idx = np.unique(idx)[:K]
            while idx.size < K:
                idx = np.unique(np.concatenate([idx, idx[-1:]], axis=0))
            idx = idx[:K]
        else:
            idx = _pick_idx(pts, cfg=cfg)
            if idx.size < K:
                # Ensure fixed K.
                fill = np.linspace(1, L - 2, num=K, dtype=np.float32)
                fill = np.clip(np.rint(fill), 1, L - 2).astype(np.int64, copy=False)
                idx = np.unique(np.concatenate([idx, fill], axis=0))[:K]
                if idx.size < K:
                    idx = np.pad(idx, (0, K - idx.size), mode="edge")
                idx = idx[:K]

        nodes = [int(seq[0])] + [int(seq[int(j)]) for j in idx.tolist()] + [int(seq[-1])]
        wp_seq[i, :] = np.asarray(nodes, dtype=np.int32)
        good += 1

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "paths_graph_npz": str(paths_graph_npz),
        "road_graph_npz": str(road_graph_npz),
        "cfg": {"num_waypoints": K, "mode": str(cfg.mode), "turn_alpha": float(cfg.turn_alpha), "seed": int(cfg.seed)},
        "graph_meta": meta_g,
        "paths_meta": meta_p,
    }
    np.savez_compressed(
        out_npz,
        wp_seq=wp_seq,
        wp_len=wp_len,
        gt_len=gt_len,
        start_t=start_t.astype(np.int64, copy=False),
        traj_idx=traj_idx.astype(np.int64, copy=False),
        start_node=start_node.astype(np.int32, copy=False),
        dest_node=dest_node.astype(np.int32, copy=False),
        route_city=(route_city.astype(np.int8, copy=False) if route_city is not None else None),
        meta=meta,
    )

    report: Dict[str, object] = {
        "ok": True,
        "task": "dump_waypoints_from_paths_graph_npz",
        "inputs": {"paths_graph_npz": str(paths_graph_npz), "road_graph_npz": str(road_graph_npz)},
        "config": {"num_waypoints": K, "mode": str(cfg.mode), "turn_alpha": float(cfg.turn_alpha), "seed": int(cfg.seed), "viz_cases": int(viz_cases)},
        "stats": {
            "n_routes": int(N),
            "n_good": int(good),
            "gt_len": {
                "p50": float(np.percentile(gt_len, 50)),
                "p90": float(np.percentile(gt_len, 90)),
            },
        },
        "outputs": {"out_npz": str(out_npz), "report_json": str(report_json)},
        "meta": {"created_at": meta["created_at"]},
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump fixed-K waypoint node sequences from GT graph paths (paths_graph.npz).")
    p.add_argument("--paths_graph_npz", type=Path, required=True)
    p.add_argument("--road_graph_npz", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--num_waypoints", type=int, default=4, help="Number of INTERNAL waypoints (excluding start/dest).")
    p.add_argument("--mode", type=str, default="rdp_turn", choices=["rdp_dev", "rdp_turn"])
    p.add_argument("--turn_alpha", type=float, default=1.0)
    p.add_argument("--viz_cases", type=int, default=0, help="Reserved for future; kept for naming consistency.")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))
    cfg = DumpCfg(num_waypoints=int(args.num_waypoints), mode=str(args.mode), turn_alpha=float(args.turn_alpha), seed=int(args.seed))
    report = run_dump(
        paths_graph_npz=Path(args.paths_graph_npz),
        road_graph_npz=Path(args.road_graph_npz),
        out_dir=Path(args.out_dir),
        cfg=cfg,
        viz_cases=int(args.viz_cases),
    )
    compact = {"ok": True, "out_npz": report["outputs"]["out_npz"], "n_routes": report["stats"]["n_routes"], "report_json": report["outputs"]["report_json"]}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

