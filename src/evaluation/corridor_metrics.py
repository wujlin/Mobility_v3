from __future__ import annotations

"""Corridor diversity metrics based on LCS similarity."""

from typing import List

import numpy as np


def lcs_length(seq_a: np.ndarray, seq_b: np.ndarray) -> int:
    """
    计算两个way sequence的最长公共子序列长度（LCS）。

    Args:
        seq_a: (L1,) int array, way_id序列（不含padding）
        seq_b: (L2,) int array, way_id序列（不含padding）

    Returns:
        LCS长度
    """
    a = np.asarray(seq_a).reshape(-1)
    b = np.asarray(seq_b).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0

    # Use O(min(L1,L2)) memory DP by making b the shorter one.
    if int(b.size) > int(a.size):
        a, b = b, a
    b_list = [int(x) for x in b.tolist()]
    m = int(len(b_list))

    prev = np.zeros((m + 1,), dtype=np.int32)
    cur = np.zeros((m + 1,), dtype=np.int32)

    # Standard DP:
    # dp[i][j] = dp[i-1][j-1] + 1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    for i in range(int(a.size)):
        cur[0] = 0
        aii = int(a[i])
        for j in range(m):
            if aii == b_list[j]:
                cur[j + 1] = prev[j] + 1
            else:
                x = cur[j]
                y = prev[j + 1]
                cur[j + 1] = x if x >= y else y
        prev, cur = cur, prev
    return int(prev[m])


def lcs_similarity(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    计算两个way sequence的LCS相似度。

    similarity = LCS_length / min(len(seq_a), len(seq_b))

    Returns:
        float in [0, 1], 1表示完全相同
    """
    lcs_len = lcs_length(seq_a, seq_b)
    min_len = min(int(np.asarray(seq_a).size), int(np.asarray(seq_b).size))
    if min_len <= 0:
        return 0.0
    return float(lcs_len) / float(min_len)


def compute_pairwise_similarity(routes: List[np.ndarray]) -> np.ndarray:
    """
    计算一组routes的两两LCS相似度矩阵。

    Args:
        routes: List of (Li,) arrays, 每个是一条route的way序列

    Returns:
        (N, N) float array, similarity matrix
    """
    n = int(len(routes))
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            s = float(lcs_similarity(routes[i], routes[j]))
            sim[i, j] = np.float32(s)
            sim[j, i] = np.float32(s)
    return sim
