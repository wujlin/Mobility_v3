from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(x, *args, **kwargs):  # type: ignore[no-redef]
        return x


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class DumpCfg:
    subsample_step: int
    debounce: bool
    max_bridge_steps: int
    max_total_steps: int
    max_routes: Optional[int]
    seed: int
    progress: str
    log_every: int
    num_workers: int
    mp_start: str
    chunk_size: int
    snap_sample_k: int


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


def _sample_snap_dist(dist: np.ndarray, *, k: int) -> List[float]:
    """
    Sample k distances from a vector deterministically (evenly spaced).
    Used to estimate snap distance percentiles without storing all per-point distances.
    """
    k = int(k)
    d = np.asarray(dist, dtype=np.float64).reshape(-1)
    if k <= 0 or d.size == 0:
        return []
    if d.size <= k:
        return [float(x) for x in d.tolist()]
    q = np.linspace(0.0, float(d.size - 1), num=k, dtype=np.float64)
    sel = np.clip(np.rint(q).astype(np.int64), 0, int(d.size) - 1)
    return [float(x) for x in d[sel].tolist()]


def _percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x.astype(np.float64, copy=False), q))


def _resolve_progress_mode(progress: str) -> str:
    mode = str(progress)
    if mode == "auto":
        # tqdm carriage-return is not friendly when piped to tee/log files.
        return "tqdm" if bool(sys.stderr.isatty()) else "json"
    if mode not in {"tqdm", "json", "none"}:
        raise ValueError(f"--progress must be one of auto|tqdm|json|none, got {progress!r}")
    return mode


def _chunk_indices(pick: np.ndarray, *, chunk_size: int) -> List[np.ndarray]:
    pick = np.asarray(pick, dtype=np.int64).reshape(-1)
    cs = int(max(1, chunk_size))
    return [pick[i : i + cs] for i in range(0, int(pick.size), cs)]


# Globals for multiprocessing (fork).
_MM_G = None
_MM_TREE = None
_MM_START_POS = None
_MM_DEST_POS = None
_MM_TARGETS = None
_MM_CFG = None


def _process_route(ii: int) -> Tuple[bool, Optional[List[int]], int, List[float], str]:
    assert _MM_G is not None
    assert _MM_TREE is not None
    assert _MM_START_POS is not None
    assert _MM_DEST_POS is not None
    assert _MM_TARGETS is not None
    assert _MM_CFG is not None

    cfg: DumpCfg = _MM_CFG  # type: ignore[assignment]
    start_pos = _MM_START_POS
    dest_pos = _MM_DEST_POS
    targets = _MM_TARGETS
    tree = _MM_TREE
    g = _MM_G

    pts = np.concatenate([start_pos[ii : ii + 1], targets[ii], dest_pos[ii : ii + 1]], axis=0)
    pts = _subsample_points(pts, step=int(cfg.subsample_step))
    dist, idx = tree.query(pts.astype(np.float64, copy=False), k=1)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int32).reshape(-1)
    snap_sample = _sample_snap_dist(dist, k=int(cfg.snap_sample_k))

    seq = idx.tolist()
    seq = _dedup_consecutive(seq)
    if cfg.debounce:
        seq = _debounce_aba(seq)
    if len(seq) < 2:
        return False, None, 0, snap_sample, "empty"

    ok, path, n_br = _bridge_snapped_sequence(
        g,
        seq,
        max_bridge_steps=int(cfg.max_bridge_steps),
        max_total_steps=int(cfg.max_total_steps),
    )
    if not ok or len(path) < 2:
        return False, None, 0, snap_sample, "bridge"
    return True, list(map(int, path)), int(n_br), snap_sample, "ok"


def _process_chunk(chunk: np.ndarray) -> Dict[str, object]:
    chunk = np.asarray(chunk, dtype=np.int64).reshape(-1)
    kept = []
    node_seqs: List[List[int]] = []
    bridged_jumps = []
    snap_samples: List[float] = []
    n_fail_empty = 0
    n_fail_bridge = 0

    for ii in chunk.tolist():
        ok, path, n_br, snap_sample, status = _process_route(int(ii))
        snap_samples.extend([float(x) for x in snap_sample])
        if not ok or path is None:
            if status == "empty":
                n_fail_empty += 1
            else:
                n_fail_bridge += 1
            continue
        kept.append(int(ii))
        node_seqs.append(path)
        bridged_jumps.append(int(n_br))

    return {
        "processed": int(chunk.size),
        "kept": kept,
        "node_seqs": node_seqs,
        "bridged_jumps": bridged_jumps,
        "snap_samples": snap_samples,
        "fail_empty": int(n_fail_empty),
        "fail_bridge": int(n_fail_bridge),
    }


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

    progress_mode = _resolve_progress_mode(cfg.progress)
    t0 = time.time()

    # Install globals (for fork-based multiprocessing).
    global _MM_G, _MM_TREE, _MM_START_POS, _MM_DEST_POS, _MM_TARGETS, _MM_CFG
    _MM_G = g
    _MM_TREE = tree
    _MM_START_POS = start_pos
    _MM_DEST_POS = dest_pos
    _MM_TARGETS = targets
    _MM_CFG = cfg

    kept: List[int] = []
    node_seqs: List[List[int]] = []
    bridged_jumps: List[int] = []
    snap_samples: List[float] = []
    n_fail_bridge = 0
    n_fail_empty = 0

    num_workers = int(cfg.num_workers)
    chunk_size = int(max(1, cfg.chunk_size))

    if num_workers <= 1:
        it: object = pick.tolist()
        if progress_mode == "tqdm":
            it = tqdm(it, desc="dump_paths", dynamic_ncols=True)
        for j, ii in enumerate(it):  # type: ignore[arg-type]
            ok, path, n_br, snap_sample, status = _process_route(int(ii))
            snap_samples.extend([float(x) for x in snap_sample])
            if not ok or path is None:
                if status == "empty":
                    n_fail_empty += 1
                else:
                    n_fail_bridge += 1
            else:
                kept.append(int(ii))
                node_seqs.append(path)
                bridged_jumps.append(int(n_br))

            if progress_mode == "json" and (int(j) % int(max(1, cfg.log_every)) == 0 or int(j) == int(pick.size) - 1):
                elapsed = float(time.time() - t0)
                print(
                    json.dumps(
                        {
                            "task": "dump_graph_paths_from_routes_npz",
                            "i": int(j),
                            "N": int(pick.size),
                            "kept": int(len(kept)),
                            "fail_empty": int(n_fail_empty),
                            "fail_bridge": int(n_fail_bridge),
                            "elapsed_s": elapsed,
                            "rate_rps": float((j + 1) / max(elapsed, 1e-6)),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    else:
        mp_start = str(cfg.mp_start)
        if mp_start != "fork":
            raise SystemExit("Multi-process mode requires --mp_start fork to avoid copying large arrays; use --num_workers 0/1 otherwise.")
        chunks = _chunk_indices(pick, chunk_size=chunk_size)
        mp_ctx = mp.get_context("fork")
        fut_to_cid = {}
        results: List[Optional[Dict[str, object]]] = [None for _ in range(len(chunks))]

        pbar = None
        if progress_mode == "tqdm":
            pbar = tqdm(total=int(pick.size), desc="dump_paths_mp", dynamic_ncols=True)

        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as ex:
            for cid, ch in enumerate(chunks):
                fut = ex.submit(_process_chunk, ch)
                fut_to_cid[fut] = int(cid)
            done_routes = 0
            kept_chunks = 0
            fail_empty_chunks = 0
            fail_bridge_chunks = 0
            log_every_routes = int(max(1, cfg.log_every))
            next_log_at = int(log_every_routes)
            for fut in as_completed(list(fut_to_cid.keys())):
                cid = int(fut_to_cid[fut])
                r = fut.result()
                results[cid] = r
                processed = int(r.get("processed", 0))
                done_routes += int(processed)
                kept_chunks += int(len(r.get("kept") or []))
                fail_empty_chunks += int(r.get("fail_empty", 0))
                fail_bridge_chunks += int(r.get("fail_bridge", 0))
                if pbar is not None:
                    pbar.update(int(processed))
                    pbar.set_postfix(kept=int(kept_chunks), fail_empty=int(fail_empty_chunks), fail_bridge=int(fail_bridge_chunks))  # type: ignore[arg-type]
                if progress_mode == "json" and (int(done_routes) >= int(next_log_at) or int(done_routes) >= int(pick.size)):
                    elapsed = float(time.time() - t0)
                    print(
                        json.dumps(
                            {
                                "task": "dump_graph_paths_from_routes_npz",
                                "done_routes": int(done_routes),
                                "N": int(pick.size),
                                "kept": int(kept_chunks),
                                "fail_empty": int(fail_empty_chunks),
                                "fail_bridge": int(fail_bridge_chunks),
                                "elapsed_s": elapsed,
                                "rate_rps": float(done_routes / max(elapsed, 1e-6)),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    while int(done_routes) >= int(next_log_at):
                        next_log_at += int(log_every_routes)
        if pbar is not None:
            pbar.close()

        # Merge in submission order for determinism.
        for rr in results:
            if rr is None:
                continue
            kept.extend(list(map(int, rr.get("kept") or [])))
            node_seqs.extend(list(rr.get("node_seqs") or []))
            bridged_jumps.extend(list(map(int, rr.get("bridged_jumps") or [])))
            snap_samples.extend([float(x) for x in (rr.get("snap_samples") or [])])
            n_fail_empty += int(rr.get("fail_empty", 0))
            n_fail_bridge += int(rr.get("fail_bridge", 0))

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

    snap_samples_arr = np.asarray(snap_samples, dtype=np.float64).reshape(-1)
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
            "progress": str(cfg.progress),
            "log_every": int(cfg.log_every),
            "num_workers": int(cfg.num_workers),
            "mp_start": str(cfg.mp_start),
            "chunk_size": int(cfg.chunk_size),
            "snap_sample_k": int(cfg.snap_sample_k),
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
                "p50": _percentile(snap_samples_arr, 50) if snap_samples_arr.size else None,
                "p90": _percentile(snap_samples_arr, 90) if snap_samples_arr.size else None,
                "sample_n": int(snap_samples_arr.size),
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
    p.add_argument("--progress", type=str, default="auto", choices=["auto", "tqdm", "json", "none"])
    p.add_argument("--log_every", type=int, default=2000, help="Only used when --progress=json.")
    p.add_argument("--num_workers", type=int, default=0, help="Set >1 to enable multi-process map-matching (Linux fork).")
    p.add_argument("--mp_start", type=str, default="fork", choices=["fork", "spawn"])
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--snap_sample_k", type=int, default=4, help="Sample k snap distances per route for percentile stats (0 to disable).")
    return p


def _die_missing_file(*, label: str, path: Path) -> None:
    print(f"[error] missing {label}: {path}", file=sys.stderr, flush=True)
    # Best-effort hints: search basename under nearest existing parent.
    root = path.parent
    while root != root.parent and not root.exists():
        root = root.parent
    if root.exists():
        hits = []
        try:
            for cand in root.rglob(path.name):
                hits.append(cand)
                if len(hits) >= 5:
                    break
        except Exception:
            hits = []
        if hits:
            print("[hint] found candidates (pick one as --routes_npz):", file=sys.stderr, flush=True)
            for h in hits:
                print(f"  - {h}", file=sys.stderr, flush=True)
    print("[hint] 建议先建立工作站别名目录（软链接）以稳定路径：", file=sys.stderr, flush=True)
    print("       python tools/routegen_make_ws_aliases.py --raw_root \"$RAW_ROOT\"", file=sys.stderr, flush=True)
    raise SystemExit(2)


def main() -> None:
    args = build_argparser().parse_args()
    routes_npz = Path(args.routes_npz)
    road_graph_npz = Path(args.road_graph_npz)
    if not routes_npz.exists():
        _die_missing_file(label="routes_npz", path=routes_npz)
    if not road_graph_npz.exists():
        _die_missing_file(label="road_graph_npz", path=road_graph_npz)
    cfg = DumpCfg(
        subsample_step=int(args.subsample_step),
        debounce=not bool(args.no_debounce),
        max_bridge_steps=int(args.max_bridge_steps),
        max_total_steps=int(args.max_total_steps),
        max_routes=(int(args.max_routes) if args.max_routes is not None else None),
        seed=int(args.seed),
        progress=str(args.progress),
        log_every=int(args.log_every),
        num_workers=int(args.num_workers),
        mp_start=str(args.mp_start),
        chunk_size=int(args.chunk_size),
        snap_sample_k=int(args.snap_sample_k),
    )
    report = run_dump(routes_npz=routes_npz, road_graph_npz=road_graph_npz, out_dir=Path(args.out_dir), cfg=cfg)
    meta = report["meta"]
    compact = {
        "ok": True,
        "out_npz": report["out_npz"],
        "N_kept": int(meta["stats"]["N_kept"]),
        "seq_len_p50": float(meta["stats"]["seq_len"]["p50"]),
        "snap_p90": float(meta["stats"]["snap_dist_grid"]["p90"]) if meta["stats"]["snap_dist_grid"]["p90"] is not None else None,
        "snap_sample_n": int(meta["stats"]["snap_dist_grid"]["sample_n"]) if meta["stats"]["snap_dist_grid"]["sample_n"] is not None else 0,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
