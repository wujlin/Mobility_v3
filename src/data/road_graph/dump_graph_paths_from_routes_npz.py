from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception as e:  # pragma: no cover
    cKDTree = None  # type: ignore[assignment]
    _KD_ERR = e

from src.data.road_graph.gate_candidate_paths_from_routes_npz import _astar, _load_graph_npz


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class DumpCfg:
    subsample_step: int
    debounce: bool
    max_bridge_steps: int
    max_total_steps: int
    max_routes: Optional[int]
    seed: int


def _dedup_consecutive(seq: Sequence[int]) -> List[int]:
    out: List[int] = []
    last: Optional[int] = None
    for x in seq:
        xi = int(x)
        if last is None or xi != int(last):
            out.append(int(xi))
            last = int(xi)
    return out


def _debounce_aba(seq: Sequence[int]) -> List[int]:
    seq = list(map(int, seq))
    if len(seq) < 3:
        return list(seq)
    keep = np.ones((len(seq),), dtype=np.uint8)
    for i in range(1, len(seq) - 1):
        if seq[i - 1] == seq[i + 1] and seq[i] != seq[i - 1]:
            keep[i] = 0
    out = [seq[i] for i in range(len(seq)) if int(keep[i]) == 1]
    return _dedup_consecutive(out)


def _subsample_points(points: np.ndarray, *, step: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    step = int(max(1, step))
    if step == 1 or points.shape[0] <= 2:
        return points
    idx = np.arange(0, points.shape[0], step, dtype=np.int64)
    if idx[-1] != points.shape[0] - 1:
        idx = np.concatenate([idx, np.asarray([points.shape[0] - 1], dtype=np.int64)], axis=0)
    return points[idx]


def _bridge_snapped_sequence(
    g,
    snapped: Sequence[int],
    *,
    max_bridge_steps: int,
    max_total_steps: int,
) -> Tuple[bool, List[int], int]:
    """
    Ensure adjacency by inserting shortest graph paths between non-adjacent snapped nodes.
    Returns: (ok, path_nodes, bridged_jumps)
    """
    max_bridge_steps = int(max(2, max_bridge_steps))
    max_total_steps = int(max(4, max_total_steps))
    seq = _dedup_consecutive(snapped)
    if not seq:
        return False, [], 0
    out = [int(seq[0])]
    bridged = 0
    for v in seq[1:]:
        u = int(out[-1])
        vv = int(v)
        if vv == u:
            continue
        if (u, vv) in g.edge_cost:
            out.append(vv)
        else:
            cost, path = _astar(g, start=u, goal=vv)
            if not path:
                return False, [], bridged
            if len(path) > int(max_bridge_steps):
                return False, [], bridged
            out.extend(list(map(int, path[1:])))
            bridged += 1
        if len(out) > int(max_total_steps):
            return False, [], bridged
    return True, out, int(bridged)


def run_dump(*, routes_npz: Path, road_graph_npz: Path, out_dir: Path, cfg: DumpCfg) -> Dict[str, object]:
    if cKDTree is None:  # pragma: no cover
        raise SystemExit(f"Missing scipy.spatial.cKDTree (scipy). Error: {_KD_ERR}")

    g = _load_graph_npz(Path(road_graph_npz))
    node_xy = np.stack([g.node_y, g.node_x], axis=1).astype(np.float64, copy=False)
    if node_xy.shape[0] < 10:
        raise RuntimeError("road_graph has too few nodes; rebuild road_graph.npz")
    tree = cKDTree(node_xy)

    data = np.load(str(routes_npz), allow_pickle=True)
    need = {"start_pos", "targets", "dest_pos", "traj_idx", "start_t"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"routes_npz missing keys: {missing}")
    start_pos = np.asarray(data["start_pos"], dtype=np.float32).reshape(-1, 2)
    dest_pos = np.asarray(data["dest_pos"], dtype=np.float32).reshape(-1, 2)
    targets = np.asarray(data["targets"], dtype=np.float32)
    traj_idx = np.asarray(data["traj_idx"], dtype=np.int64).reshape(-1)
    start_t = np.asarray(data["start_t"], dtype=np.int64).reshape(-1)
    n = int(start_pos.shape[0])
    F = int(targets.shape[1])

    if cfg.max_routes is not None:
        m = int(max(1, min(int(cfg.max_routes), n)))
        rng = np.random.default_rng(int(cfg.seed))
        pick = rng.choice(n, size=m, replace=False)
        pick = np.sort(pick.astype(np.int64))
    else:
        pick = np.arange(n, dtype=np.int64)

    snapped_dist_all = []
    kept = []
    node_seqs: List[List[int]] = []
    bridged_jumps = []

    n_fail_bridge = 0
    n_fail_empty = 0

    for ii in pick.tolist():
        pts = np.concatenate([start_pos[ii : ii + 1], targets[ii], dest_pos[ii : ii + 1]], axis=0)
        pts = _subsample_points(pts, step=int(cfg.subsample_step))
        dist, idx = tree.query(pts.astype(np.float64, copy=False), k=1)
        dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        idx = np.asarray(idx, dtype=np.int32).reshape(-1)
        snapped_dist_all.append(dist)

        seq = idx.tolist()
        seq = _dedup_consecutive(seq)
        if cfg.debounce:
            seq = _debounce_aba(seq)
        if len(seq) < 2:
            n_fail_empty += 1
            continue

        ok, path, n_br = _bridge_snapped_sequence(
            g,
            seq,
            max_bridge_steps=int(cfg.max_bridge_steps),
            max_total_steps=int(cfg.max_total_steps),
        )
        if not ok or len(path) < 2:
            n_fail_bridge += 1
            continue

        kept.append(int(ii))
        node_seqs.append(list(map(int, path)))
        bridged_jumps.append(int(n_br))

    kept = np.asarray(kept, dtype=np.int64)
    n_kept = int(kept.size)
    if n_kept == 0:
        raise RuntimeError("No routes kept after snapping/bridging. Check road_graph compatibility.")

    # Pad sequences.
    lens = np.asarray([len(s) for s in node_seqs], dtype=np.int32)
    Lmax = int(np.max(lens).item())
    pad_val = -1
    node_seq_pad = np.full((n_kept, Lmax), pad_val, dtype=np.int32)
    for i, seq in enumerate(node_seqs):
        node_seq_pad[i, : len(seq)] = np.asarray(seq, dtype=np.int32)

    start_node = node_seq_pad[:, 0].astype(np.int32, copy=False)
    dest_node = np.take_along_axis(node_seq_pad, (lens - 1).reshape(-1, 1), axis=1).reshape(-1).astype(np.int32, copy=False)

    snapped_dist_all = np.concatenate(snapped_dist_all, axis=0).astype(np.float64, copy=False) if snapped_dist_all else np.zeros((0,), dtype=np.float64)
    bridged_jumps = np.asarray(bridged_jumps, dtype=np.int32)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "paths_graph.npz"
    report_json = out_dir / "report.json"

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "dump_graph_paths_from_routes_npz",
        "inputs": {"routes_npz": str(routes_npz), "road_graph_npz": str(road_graph_npz)},
        "config": {
            "subsample_step": int(cfg.subsample_step),
            "debounce": bool(cfg.debounce),
            "max_bridge_steps": int(cfg.max_bridge_steps),
            "max_total_steps": int(cfg.max_total_steps),
            "max_routes": (int(cfg.max_routes) if cfg.max_routes is not None else None),
            "seed": int(cfg.seed),
        },
        "stats": {
            "N_in": int(n),
            "N_pick": int(pick.size),
            "N_kept": int(n_kept),
            "F": int(F),
            "seq_len": {"p50": float(np.percentile(lens, 50)), "p90": float(np.percentile(lens, 90)), "max": int(Lmax)},
            "bridged_jumps_per_route": {
                "mean": float(np.mean(bridged_jumps.astype(np.float32))),
                "p50": float(np.percentile(bridged_jumps, 50)),
                "p90": float(np.percentile(bridged_jumps, 90)),
            },
            "snap_dist_grid": {
                "p50": float(np.percentile(snapped_dist_all, 50)) if snapped_dist_all.size else None,
                "p90": float(np.percentile(snapped_dist_all, 90)) if snapped_dist_all.size else None,
            },
            "failures": {"empty_seq": int(n_fail_empty), "bridge_fail": int(n_fail_bridge)},
        },
    }

    np.savez_compressed(
        out_npz,
        kept_index=kept.astype(np.int64, copy=False),
        traj_idx=traj_idx[kept].astype(np.int64, copy=False),
        start_t=start_t[kept].astype(np.int64, copy=False),
        start_pos=start_pos[kept].astype(np.float32, copy=False),
        dest_pos=dest_pos[kept].astype(np.float32, copy=False),
        start_node=start_node.astype(np.int32, copy=False),
        dest_node=dest_node.astype(np.int32, copy=False),
        node_seq_pad=node_seq_pad.astype(np.int32, copy=False),
        node_seq_len=lens.astype(np.int32, copy=False),
        meta=meta,
    )
    report_json.write_text(json.dumps({"ok": True, "out_npz": str(out_npz), "meta": meta}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "out_npz": str(out_npz), "report_json": str(report_json), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump graph-aligned node sequences (teacher-forcing ready) by snapping routes_npz to road_graph.")
    p.add_argument("--routes_npz", type=Path, required=True)
    p.add_argument("--road_graph_npz", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--subsample_step", type=int, default=1)
    p.add_argument("--no_debounce", action="store_true", help="Disable simple ABA debouncing on snapped node sequence.")
    p.add_argument("--max_bridge_steps", type=int, default=2048)
    p.add_argument("--max_total_steps", type=int, default=8192)
    p.add_argument("--max_routes", type=int, default=None, help="Optional cap on number of routes (random subset).")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = DumpCfg(
        subsample_step=int(args.subsample_step),
        debounce=not bool(args.no_debounce),
        max_bridge_steps=int(args.max_bridge_steps),
        max_total_steps=int(args.max_total_steps),
        max_routes=(int(args.max_routes) if args.max_routes is not None else None),
        seed=int(args.seed),
    )
    report = run_dump(routes_npz=Path(args.routes_npz), road_graph_npz=Path(args.road_graph_npz), out_dir=Path(args.out_dir), cfg=cfg)
    meta = report["meta"]
    compact = {
        "ok": True,
        "out_npz": report["out_npz"],
        "N_kept": int(meta["stats"]["N_kept"]),
        "seq_len_p50": float(meta["stats"]["seq_len"]["p50"]),
        "snap_p90": float(meta["stats"]["snap_dist_grid"]["p90"]) if meta["stats"]["snap_dist_grid"]["p90"] is not None else None,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

