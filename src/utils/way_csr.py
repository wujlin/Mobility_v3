from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def slice_csr(ptr: np.ndarray, idx: np.ndarray, u: int) -> np.ndarray:
    ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    uu = int(u)
    if uu < 0 or uu + 1 >= int(ptr.size):
        return np.asarray([], dtype=np.int64)
    s = int(ptr[uu])
    e = int(ptr[uu + 1])
    if e <= s:
        return np.asarray([], dtype=np.int64)
    return np.asarray(idx[s:e], dtype=np.int64)


def out_degree(ptr: np.ndarray, u: int) -> int:
    ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
    uu = int(u)
    if uu < 0 or uu + 1 >= int(ptr.size):
        return 0
    return int(ptr[uu + 1] - ptr[uu])


@dataclass(frozen=True)
class CandidateRow:
    cand: np.ndarray  # (C,) int64, -1 padded
    mask: np.ndarray  # (C,) bool
    target_idx: Optional[int]  # None if target not provided


def build_candidate_row(
    succ: np.ndarray,
    *,
    max_candidates: int,
    target: Optional[int] = None,
) -> CandidateRow:
    """
    Candidate policy: "first" (successors[:max_candidates]).

    If target is provided, we guarantee it appears in the row (for training):
      - if succ empty: row=[target]
      - else if target in succ[:C]: ok
      - else if succ shorter than C: append target
      - else replace last element with target
    """
    C = int(max_candidates)
    if C <= 0:
        raise ValueError("max_candidates must be > 0")

    s = np.asarray(succ, dtype=np.int64).reshape(-1)
    if s.size > C:
        s = s[:C]

    row = np.full((C,), -1, dtype=np.int64)
    mask = np.zeros((C,), dtype=bool)

    if target is None:
        n = int(min(s.size, C))
        if n > 0:
            row[:n] = s[:n]
            mask[:n] = True
        return CandidateRow(cand=row, mask=mask, target_idx=None)

    tgt = int(target)
    if s.size == 0:
        row[0] = tgt
        mask[0] = True
        return CandidateRow(cand=row, mask=mask, target_idx=0)

    s_list = s.tolist()
    if tgt in set(int(x) for x in s_list):
        n = int(min(s.size, C))
        row[:n] = s[:n]
        mask[:n] = True
        pos = int(np.where(row == tgt)[0][0])
        return CandidateRow(cand=row, mask=mask, target_idx=pos)

    if int(s.size) < C:
        row[: int(s.size)] = s
        row[int(s.size)] = tgt
        mask[: int(s.size) + 1] = True
        return CandidateRow(cand=row, mask=mask, target_idx=int(s.size))

    # full but missing target: replace last
    row[:] = s[:C]
    row[-1] = tgt
    mask[:] = True
    return CandidateRow(cand=row, mask=mask, target_idx=C - 1)


def infer_n_ways_from_ptr(ptr: np.ndarray) -> int:
    ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
    if ptr.size < 2:
        raise ValueError("way_adj_ptr too small")
    return int(ptr.size) - 1


def build_truncated_successors_first(
    ptr: np.ndarray,
    idx: np.ndarray,
    *,
    max_candidates: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute padded successor rows for candidate policy "first" = successors[:max_candidates].

    Returns:
      succ_pad: (n_ways, C) int64, -1 padded
      succ_mask: (n_ways, C) bool
    """
    ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    C = int(max_candidates)
    if C <= 0:
        raise ValueError("max_candidates must be > 0")
    if ptr.size < 2:
        raise ValueError("way_adj_ptr too small")
    n_ways = int(ptr.size) - 1
    succ_pad = np.full((n_ways, C), -1, dtype=np.int64)
    succ_mask = np.zeros((n_ways, C), dtype=bool)
    for u in range(n_ways):
        s = int(ptr[u])
        e = int(ptr[u + 1])
        if e <= s:
            continue
        e2 = int(min(e, s + C))
        n = int(e2 - s)
        if n <= 0:
            continue
        succ_pad[u, :n] = idx[s:e2]
        succ_mask[u, :n] = True
    return succ_pad, succ_mask
