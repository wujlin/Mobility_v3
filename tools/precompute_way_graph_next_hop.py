"""
预计算 way graph 的 shortest-path next-hop 查表（供 DAgger shortest_path expert 使用）。

输出:
  - next_hop.npy: shape=(N, M), dtype=uint16/uint32
    next_hop[src, j] = 从 src 前往 dest_ids[j] 的第一跳 way_id
    若不可达则为 sentinel（默认 65535）
  - dest_ids.npy (可选): shape=(M,), 每列对应的 dest_way id

说明:
  1) 若不提供 --dest_ids_npy，则默认对所有 dest_way 预计算 (M=N)。
  2) 该表可在训练中通过 mmap O(1) 查询，避免在线 Dijkstra（CPU 瓶颈）。
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_G_REV_PTR: np.ndarray | None = None
_G_REV_IDX: np.ndarray | None = None
_G_WAY_LEN_M: np.ndarray | None = None
_G_SENTINEL: int = 65535
_G_DTYPE: np.dtype = np.uint16
_G_MAX_VISITS: int = 0


def _build_reverse_graph(ptr: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(ptr.size) - 1
    in_deg = np.zeros((n,), dtype=np.int64)
    for v in idx.tolist():
        in_deg[int(v)] += 1
    rev_ptr = np.zeros((n + 1,), dtype=np.int64)
    np.cumsum(in_deg, out=rev_ptr[1:])
    rev_idx = np.empty((int(idx.size),), dtype=np.int64)
    pos = rev_ptr[:-1].copy()
    for u in range(n):
        s = int(ptr[u])
        e = int(ptr[u + 1])
        for v in idx[s:e]:
            vv = int(v)
            rev_idx[int(pos[vv])] = int(u)
            pos[vv] += 1
    return rev_ptr, rev_idx


def _init_worker(
    rev_ptr: np.ndarray,
    rev_idx: np.ndarray,
    way_len_m: np.ndarray,
    sentinel: int,
    dtype_name: str,
    max_visits: int,
) -> None:
    global _G_REV_PTR, _G_REV_IDX, _G_WAY_LEN_M, _G_SENTINEL, _G_DTYPE, _G_MAX_VISITS
    _G_REV_PTR = rev_ptr
    _G_REV_IDX = rev_idx
    _G_WAY_LEN_M = way_len_m
    _G_SENTINEL = int(sentinel)
    _G_DTYPE = np.dtype(dtype_name)
    _G_MAX_VISITS = int(max_visits)


def _reverse_dijkstra_next_hop(dest: int) -> np.ndarray:
    """
    反向 Dijkstra:
      dist[u] = way_len[u] + min_{u->v} dist[v], dist[dest]=way_len[dest]
    同时记录每个 src 的第一跳 next_hop[src]。
    """
    if _G_REV_PTR is None or _G_REV_IDX is None or _G_WAY_LEN_M is None:
        raise RuntimeError("worker globals are not initialized")

    n = int(_G_REV_PTR.size) - 1
    d = int(dest)
    out = np.full((n,), int(_G_SENTINEL), dtype=_G_DTYPE)
    if d < 0 or d >= n:
        return out

    dist = np.full((n,), np.inf, dtype=np.float64)
    base = float(_G_WAY_LEN_M[d]) if np.isfinite(float(_G_WAY_LEN_M[d])) else 0.0
    dist[d] = base
    out[d] = np.asarray(d, dtype=_G_DTYPE)
    heap: List[Tuple[float, int]] = [(float(dist[d]), int(d))]
    seen = 0

    while heap:
        du, u = heapq.heappop(heap)
        if du != float(dist[u]):
            continue
        s = int(_G_REV_PTR[u])
        e = int(_G_REV_PTR[u + 1])
        for pred in _G_REV_IDX[s:e]:
            p = int(pred)
            w = float(_G_WAY_LEN_M[p]) if np.isfinite(float(_G_WAY_LEN_M[p])) else 0.0
            nd = du + w
            if nd < float(dist[p]):
                dist[p] = nd
                out[p] = np.asarray(u, dtype=_G_DTYPE)  # 从 p 去 dest 的第一跳是 u
                heapq.heappush(heap, (nd, p))
        seen += 1
        if int(_G_MAX_VISITS) > 0 and seen >= int(_G_MAX_VISITS):
            break
    return out


def _worker_chunk(dests: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    if _G_REV_PTR is None:
        raise RuntimeError("worker globals are not initialized")
    n = int(_G_REV_PTR.size) - 1
    if len(dests) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((n, 0), dtype=_G_DTYPE)
    cols = np.empty((n, len(dests)), dtype=_G_DTYPE)
    for j, dest in enumerate(dests):
        cols[:, j] = _reverse_dijkstra_next_hop(int(dest))
    return np.asarray(dests, dtype=np.int64), cols


def _iter_chunks(seq: np.ndarray, chunk_size: int) -> Iterable[List[int]]:
    c = max(1, int(chunk_size))
    n = int(seq.size)
    for s in range(0, n, c):
        e = min(n, s + c)
        yield [int(x) for x in seq[s:e].tolist()]


def _pick_dtype(n_ways: int, dtype_name: str, sentinel: int) -> np.dtype:
    name = str(dtype_name).strip().lower()
    if name == "uint16":
        dt = np.uint16
    elif name == "uint32":
        dt = np.uint32
    elif name == "auto":
        dt = np.uint16 if int(n_ways) <= 65535 and int(sentinel) <= 65535 else np.uint32
    else:
        raise ValueError(f"unsupported dtype: {dtype_name!r}")
    maxv = np.iinfo(dt).max
    if int(sentinel) < 0 or int(sentinel) > int(maxv):
        raise ValueError(f"sentinel={sentinel} out of range for dtype={np.dtype(dt).name} (max={maxv})")
    return np.dtype(dt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute shortest-path next-hop table for way graph.")
    ap.add_argument("--way_graph_npz", type=Path, required=True)
    ap.add_argument("--way_features_npz", type=Path, required=True)
    ap.add_argument("--out_next_hop_npy", type=Path, required=True)
    ap.add_argument("--out_dest_ids_npy", type=Path, default=None, help="Optional output dest_ids.npy.")
    ap.add_argument("--out_meta_json", type=Path, default=None, help="Optional output meta json.")
    ap.add_argument("--dest_ids_npy", type=Path, default=None, help="Optional input subset of destination way ids (shape [M]).")
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--chunk_size", type=int, default=32)
    ap.add_argument("--start_method", type=str, default="auto", choices=["auto", "fork", "spawn", "forkserver"])
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "uint16", "uint32"])
    ap.add_argument("--sentinel", type=int, default=65535)
    ap.add_argument("--max_visits", type=int, default=0, help="Per-destination node visit cap in reverse Dijkstra (0=unlimited).")
    args = ap.parse_args()

    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(wg["way_adj_idx"], dtype=np.int64).reshape(-1)
    n = int(ptr.size) - 1

    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    way_len_m = np.asarray(wf["way_len_m"], dtype=np.float64).reshape(-1)
    if int(way_len_m.size) != int(n):
        raise SystemExit(f"[FATAL] way_len_m size mismatch: {int(way_len_m.size)} vs n_ways={n}")

    if args.dest_ids_npy is None:
        dest_ids = np.arange(n, dtype=np.int64)
    else:
        dest_ids = np.load(str(args.dest_ids_npy), allow_pickle=False).reshape(-1).astype(np.int64, copy=False)
        dest_ids = np.unique(dest_ids[(dest_ids >= 0) & (dest_ids < n)])
        if int(dest_ids.size) == 0:
            raise SystemExit("[FATAL] empty dest_ids after filtering")
    m = int(dest_ids.size)

    out_dtype = _pick_dtype(n_ways=n, dtype_name=str(args.dtype), sentinel=int(args.sentinel))
    sentinel = int(args.sentinel)

    log.info(f"graph: N={n}, E={int(idx.size)}")
    log.info(f"destinations: M={m}")
    log.info(f"dtype={out_dtype.name} sentinel={sentinel}")
    log.info("building reverse graph ...")
    t0 = time.time()
    rev_ptr, rev_idx = _build_reverse_graph(ptr, idx)
    log.info(f"reverse graph built in {time.time()-t0:.1f}s")

    out = np.full((n, m), sentinel, dtype=out_dtype)
    dest_to_col = {int(d): int(i) for i, d in enumerate(dest_ids.tolist())}

    workers = max(1, int(args.num_workers))
    chunk_size = max(1, int(args.chunk_size))
    done = 0
    t1 = time.time()
    if workers <= 1:
        _init_worker(rev_ptr, rev_idx, way_len_m, sentinel, out_dtype.name, int(args.max_visits))
        for chunk in _iter_chunks(dest_ids, chunk_size):
            d_arr, cols = _worker_chunk(chunk)
            for j, d in enumerate(d_arr.tolist()):
                out[:, dest_to_col[int(d)]] = cols[:, j]
            done += int(d_arr.size)
            elapsed = max(1e-6, time.time() - t1)
            rate = float(done) / elapsed
            eta = float(m - done) / max(1e-6, rate)
            log.info(f"next-hop {done}/{m} ({rate:.1f}/s, ETA {eta/60.0:.1f}m)")
    else:
        method = str(args.start_method).strip().lower()
        if method == "auto":
            method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        mp_ctx = mp.get_context(method)
        chunks = list(_iter_chunks(dest_ids, chunk_size))
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_ctx,
            initializer=_init_worker,
            initargs=(rev_ptr, rev_idx, way_len_m, sentinel, out_dtype.name, int(args.max_visits)),
        ) as ex:
            futs = [ex.submit(_worker_chunk, ch) for ch in chunks]
            for fut in as_completed(futs):
                d_arr, cols = fut.result()
                for j, d in enumerate(d_arr.tolist()):
                    out[:, dest_to_col[int(d)]] = cols[:, j]
                done += int(d_arr.size)
                elapsed = max(1e-6, time.time() - t1)
                rate = float(done) / elapsed
                eta = float(m - done) / max(1e-6, rate)
                if done % max(1, int(chunk_size) * 4) == 0 or done >= m:
                    log.info(f"next-hop {done}/{m} ({rate:.1f}/s, ETA {eta/60.0:.1f}m)")

    args.out_next_hop_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.out_next_hop_npy), out)
    if args.out_dest_ids_npy is not None:
        args.out_dest_ids_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(args.out_dest_ids_npy), dest_ids.astype(np.int64, copy=False))

    meta = {
        "ok": True,
        "task": "precompute_way_graph_next_hop",
        "way_graph_npz": str(args.way_graph_npz),
        "way_features_npz": str(args.way_features_npz),
        "out_next_hop_npy": str(args.out_next_hop_npy),
        "out_dest_ids_npy": (str(args.out_dest_ids_npy) if args.out_dest_ids_npy is not None else None),
        "n_ways": int(n),
        "n_dests": int(m),
        "dtype": out_dtype.name,
        "sentinel": int(sentinel),
        "num_workers": int(workers),
        "chunk_size": int(chunk_size),
        "start_method": str(method if workers > 1 else "none"),
        "max_visits": int(args.max_visits),
        "elapsed_sec": float(time.time() - t1),
    }
    if args.out_meta_json is not None:
        args.out_meta_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

