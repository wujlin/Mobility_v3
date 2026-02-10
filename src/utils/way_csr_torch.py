from __future__ import annotations

from typing import Tuple

import torch


def build_candidates_first_with_target(
    *,
    succ_pad: torch.Tensor,  # (n_ways, C) int64, -1 padded
    succ_mask: torch.Tensor,  # (n_ways, C) bool
    cur_way: torch.Tensor,  # (B,) int64
    next_way: torch.Tensor,  # (B,) int64
    valid: torch.Tensor,  # (B,) bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Torch equivalent of src.utils.way_csr.build_candidate_row with candidate policy "first".

    This is used in AR baselines training to avoid per-step CPU loops and CPU->GPU copies.

    Guarantees: for each i where valid[i] is True, next_way[i] appears in the returned candidate row
    (either already in successors[:C], appended to the first free slot, or replacing the last slot).
    """
    if succ_pad.ndim != 2 or succ_mask.ndim != 2:
        raise ValueError("succ_pad/succ_mask must be 2D tensors")
    if cur_way.ndim != 1 or next_way.ndim != 1 or valid.ndim != 1:
        raise ValueError("cur_way/next_way/valid must be 1D tensors")

    device = succ_pad.device
    if succ_mask.device != device:
        raise ValueError("succ_pad and succ_mask must be on the same device")
    if cur_way.device != device or next_way.device != device or valid.device != device:
        raise ValueError("cur_way/next_way/valid must be on the same device as succ_pad")

    # Align dtypes
    cur_way = cur_way.to(dtype=torch.long)
    next_way = next_way.to(dtype=torch.long)
    valid = valid.to(dtype=torch.bool)
    succ_mask = succ_mask.to(dtype=torch.bool)

    B = int(cur_way.numel())
    C = int(succ_pad.shape[1])
    if C <= 0:
        raise ValueError("succ_pad has invalid candidate dimension")

    # Gather candidates for each current way (invalid cur_way gets clamped, then masked out by valid).
    cur_idx = torch.clamp(cur_way, min=0)
    cand = succ_pad[cur_idx].clone()
    mask = succ_mask[cur_idx].clone()

    tgt_mask = valid & (cur_way >= 0) & (next_way >= 0)
    tgt = torch.where(tgt_mask, next_way, torch.zeros_like(next_way))

    idx_range = torch.arange(C, device=device, dtype=torch.long).view(1, C)
    big = torch.full((1, C), C, device=device, dtype=torch.long)

    has_tgt = (cand == tgt[:, None]) & mask & tgt_mask[:, None]
    has_any = has_tgt.any(dim=1)
    tgt_pos = torch.where(has_tgt, idx_range, big).min(dim=1).values

    free = (~mask) & tgt_mask[:, None]
    has_free = free.any(dim=1)
    first_free = torch.where(free, idx_range, big).min(dim=1).values
    insert_pos = torch.where(has_free, first_free, torch.full((B,), C - 1, device=device, dtype=torch.long))

    need_insert = tgt_mask & (~has_any)
    if bool(need_insert.any()):
        rows = need_insert.nonzero(as_tuple=False).view(-1)
        cand[rows, insert_pos[rows]] = tgt[rows]
        mask[rows, insert_pos[rows]] = True

    tgt_idx = torch.where(has_any, tgt_pos, insert_pos)
    tgt_idx = torch.where(tgt_mask, tgt_idx, torch.zeros_like(tgt_idx))
    return cand, mask, tgt_idx, tgt_mask

