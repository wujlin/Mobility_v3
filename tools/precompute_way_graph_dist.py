"""
预计算 way graph 上任意 dest_way 到所有 way 的最短 hop-count 距离。

用法:
    python -m tools.precompute_way_graph_dist \
        --way_graph_npz <path> \
        --out_npz <path> \
        [--max_hops 200]

输出 npz 包含:
    - dist_matrix: (N, N) int16, dist_matrix[src, dest] = hop count from src to dest
      不可达的标记为 max_hops+1 (= 201)

对 Porto 35K ways: ~35K × 35K × 2 bytes ≈ 2.4 GB，需要足够内存。
如果太大，可以改成只对 test set 中出现的 dest_way 做 BFS（lazy 模式）。

但 Porto 35K ways 的 BFS 是 O(N*(N+E))，约 35K * 70K ≈ 2.5e9 ops。
为了可行性，我们改成 **on-demand BFS**: 只对每个 batch 中的 dest_way 做反向 BFS，
结果缓存在 dict 中。这样不需要预计算全量矩阵。

然而，训练脚本中需要快速查表。所以我们提供两种模式:
1. full: 预计算全量 (N,N) 矩阵（N<=40K 时可行，~2.4GB）
2. dest_only: 只对指定 dest_ways 做 BFS（适用于更大图）
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_G_REV_PTR: np.ndarray | None = None
_G_REV_IDX: np.ndarray | None = None
_G_MAX_HOPS: int = 200


def reverse_bfs_single(
    dest: int,
    rev_ptr: np.ndarray,
    rev_idx: np.ndarray,
    max_hops: int,
) -> np.ndarray:
    """从 dest 出发做反向 BFS，返回 dist[src] = 从 src 到 dest 的最短 hop count。"""
    N = len(rev_ptr) - 1
    dist = np.full(N, max_hops + 1, dtype=np.int16)
    dist[dest] = 0
    queue = deque([dest])
    while queue:
        u = queue.popleft()
        d_u = int(dist[u])
        if d_u >= max_hops:
            continue
        s = int(rev_ptr[u])
        e = int(rev_ptr[u + 1])
        for v in rev_idx[s:e]:
            v = int(v)
            if dist[v] > d_u + 1:
                dist[v] = d_u + 1
                queue.append(v)
    return dist


def _init_worker(rev_ptr: np.ndarray, rev_idx: np.ndarray, max_hops: int) -> None:
    global _G_REV_PTR, _G_REV_IDX, _G_MAX_HOPS
    _G_REV_PTR = rev_ptr
    _G_REV_IDX = rev_idx
    _G_MAX_HOPS = int(max_hops)


def _worker_bfs_chunk(dests: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    if _G_REV_PTR is None or _G_REV_IDX is None:
        raise RuntimeError("Worker globals are not initialized")
    if len(dests) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((len(_G_REV_PTR) - 1, 0), dtype=np.int16)
    N = len(_G_REV_PTR) - 1
    cols = np.empty((N, len(dests)), dtype=np.int16)
    for j, dest in enumerate(dests):
        cols[:, j] = reverse_bfs_single(int(dest), _G_REV_PTR, _G_REV_IDX, int(_G_MAX_HOPS))
    return np.asarray(dests, dtype=np.int64), cols


def build_reverse_graph(ptr: np.ndarray, idx: np.ndarray) -> tuple:
    """从 CSR 正向图构建 CSR 反向图。"""
    N = len(ptr) - 1
    # 统计每个节点的入度
    in_deg = np.zeros(N, dtype=np.int64)
    for v in idx:
        in_deg[int(v)] += 1
    rev_ptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(in_deg, out=rev_ptr[1:])
    rev_idx = np.empty(len(idx), dtype=np.int64)
    pos = rev_ptr[:-1].copy()
    for u in range(N):
        s = int(ptr[u])
        e = int(ptr[u + 1])
        for v in idx[s:e]:
            v = int(v)
            rev_idx[int(pos[v])] = u
            pos[v] += 1
    return rev_ptr, rev_idx


def _iter_chunks(n: int, chunk_size: int) -> Iterable[List[int]]:
    c = max(1, int(chunk_size))
    for s in range(0, int(n), c):
        e = min(int(n), s + c)
        yield list(range(s, e))


def precompute_full(
    ptr: np.ndarray,
    idx: np.ndarray,
    max_hops: int = 200,
    num_workers: int = 1,
    chunk_size: int = 64,
) -> np.ndarray:
    """预计算全量 (N,N) 距离矩阵。N<=40K 时可行。"""
    N = len(ptr) - 1
    log.info(f"Building reverse graph (N={N}, E={len(idx)})...")
    rev_ptr, rev_idx = build_reverse_graph(ptr, idx)

    n_workers = max(1, int(num_workers))
    log.info(f"Running BFS for all {N} destinations (max_hops={max_hops}, workers={n_workers}, chunk_size={int(chunk_size)})...")
    # dist_matrix[src, dest] = hop count from src to dest
    dist_matrix = np.full((N, N), max_hops + 1, dtype=np.int16)

    t0 = time.time()
    if n_workers <= 1:
        for dest in range(N):
            dist_col = reverse_bfs_single(dest, rev_ptr, rev_idx, max_hops)
            dist_matrix[:, dest] = dist_col
            if (dest + 1) % 5000 == 0 or dest == N - 1:
                elapsed = time.time() - t0
                rate = (dest + 1) / elapsed
                eta = (N - dest - 1) / rate if rate > 0 else 0
                log.info(f"  BFS {dest+1}/{N} done ({elapsed:.1f}s, {rate:.1f}/s, ETA {eta:.0f}s)")
    else:
        done = 0
        # Linux/WSL 下优先 fork，避免重复拷贝大数组到子进程。
        try:
            mp_ctx = mp.get_context("fork")
        except Exception:
            mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_ctx,
            initializer=_init_worker,
            initargs=(rev_ptr, rev_idx, int(max_hops)),
        ) as ex:
            futs = [ex.submit(_worker_bfs_chunk, chunk) for chunk in _iter_chunks(N, int(chunk_size))]
            for fut in as_completed(futs):
                dests, cols = fut.result()
                if int(dests.size) > 0:
                    dist_matrix[:, dests] = cols
                    done += int(dests.size)
                if done % 5000 == 0 or done >= N:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (N - done) / rate if rate > 0 else 0.0
                    log.info(f"  BFS {done}/{N} done ({elapsed:.1f}s, {rate:.1f}/s, ETA {eta:.0f}s)")

    return dist_matrix


def main():
    ap = argparse.ArgumentParser(description="Precompute way graph hop-count distances.")
    ap.add_argument("--way_graph_npz", type=Path, required=True)
    ap.add_argument("--out_npz", type=Path, required=True)
    ap.add_argument("--max_hops", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=1, help="BFS worker processes (1=single process).")
    ap.add_argument("--chunk_size", type=int, default=64, help="Destinations per worker task in multiprocessing mode.")
    ap.add_argument("--mode", type=str, default="full", choices=["full"],
                    help="full: precompute (N,N) matrix")
    ap.add_argument("--no_compress", action="store_true", help="Use np.savez (faster, larger file) instead of np.savez_compressed.")
    args = ap.parse_args()

    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(wg["way_adj_idx"], dtype=np.int64).reshape(-1)
    N = len(ptr) - 1
    log.info(f"Loaded graph: N={N}, E={len(idx)}")

    if args.mode == "full":
        dist_matrix = precompute_full(
            ptr,
            idx,
            max_hops=int(args.max_hops),
            num_workers=int(args.num_workers),
            chunk_size=int(args.chunk_size),
        )
        args.out_npz.parent.mkdir(parents=True, exist_ok=True)
        if bool(args.no_compress):
            np.savez(
                str(args.out_npz),
                dist_matrix=dist_matrix,
                max_hops=np.array(args.max_hops, dtype=np.int32),
                n_ways=np.array(N, dtype=np.int32),
            )
        else:
            np.savez_compressed(
                str(args.out_npz),
                dist_matrix=dist_matrix,
                max_hops=np.array(args.max_hops, dtype=np.int32),
                n_ways=np.array(N, dtype=np.int32),
            )
        # Stats
        reachable = (dist_matrix <= args.max_hops)
        log.info(f"Saved to {args.out_npz}")
        log.info(f"  shape={dist_matrix.shape}, dtype={dist_matrix.dtype}")
        log.info(f"  reachable fraction: {reachable.mean():.4f}")
        log.info(f"  dist p50={np.median(dist_matrix[reachable]):.0f}, "
                 f"p90={np.percentile(dist_matrix[reachable], 90):.0f}, "
                 f"max={dist_matrix[reachable].max()}")
        size_mb = dist_matrix.nbytes / 1024**2
        log.info(f"  matrix size: {size_mb:.1f} MB (int16)")


if __name__ == "__main__":
    main()
