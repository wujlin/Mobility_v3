from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SegmentRoutes:
    seg_seq_ptr: np.ndarray  # (N+1,) int64
    seg_seq_idx: np.ndarray  # (total,) int32
    seg_seq_len: np.ndarray  # (N,) int32
    corridor_type: np.ndarray  # (N,) int8
    start_node: np.ndarray  # (N,) int32
    dest_node: np.ndarray  # (N,) int32
    start_t: np.ndarray  # (N,) int64
    route_city: np.ndarray  # (N,) int8
    start_pos: np.ndarray  # (N,2) float32 (y,x)
    dest_pos: np.ndarray  # (N,2) float32 (y,x)


def load_segment_routes_npz(path: Path) -> SegmentRoutes:
    data = np.load(str(path), allow_pickle=True)
    need = {
        "seg_seq_ptr",
        "seg_seq_idx",
        "seg_seq_len",
        "corridor_type",
        "start_node",
        "dest_node",
        "start_t",
        "route_city",
        "start_pos",
        "dest_pos",
    }
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"segments_graph_routes.npz missing keys: {missing}")
    return SegmentRoutes(
        seg_seq_ptr=np.asarray(data["seg_seq_ptr"], dtype=np.int64),
        seg_seq_idx=np.asarray(data["seg_seq_idx"], dtype=np.int32),
        seg_seq_len=np.asarray(data["seg_seq_len"], dtype=np.int32).reshape(-1),
        corridor_type=np.asarray(data["corridor_type"], dtype=np.int8).reshape(-1),
        start_node=np.asarray(data["start_node"], dtype=np.int32).reshape(-1),
        dest_node=np.asarray(data["dest_node"], dtype=np.int32).reshape(-1),
        start_t=np.asarray(data["start_t"], dtype=np.int64).reshape(-1),
        route_city=np.asarray(data["route_city"], dtype=np.int8).reshape(-1),
        start_pos=np.asarray(data["start_pos"], dtype=np.float32).reshape(-1, 2),
        dest_pos=np.asarray(data["dest_pos"], dtype=np.float32).reshape(-1, 2),
    )


class SegmentRouteDataset(Dataset):
    """Dataset of per-route segment sequences."""

    def __init__(
        self,
        routes: SegmentRoutes,
        *,
        max_routes: Optional[int] = None,
        max_seg_len: Optional[int] = None,
    ) -> None:
        self.routes = routes
        n = int(routes.seg_seq_len.shape[0])
        keep = routes.seg_seq_len > 0
        if max_seg_len is not None:
            keep &= routes.seg_seq_len <= int(max_seg_len)
        ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
        if max_routes is not None:
            ids = ids[: int(max_routes)]
        self.route_ids = ids

    def __len__(self) -> int:
        return int(self.route_ids.size)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        rid = int(self.route_ids[int(idx)])
        L = int(self.routes.seg_seq_len[rid])
        s = int(self.routes.seg_seq_ptr[rid])
        e = s + L
        seg_seq = self.routes.seg_seq_idx[s:e].astype(np.int64, copy=False)
        return {
            "route_id": np.asarray(rid, dtype=np.int64),
            "seg_seq": seg_seq,
            "seg_len": np.asarray(L, dtype=np.int64),
            "corridor_type": np.asarray(int(self.routes.corridor_type[rid]), dtype=np.int64),
            "start_node": np.asarray(int(self.routes.start_node[rid]), dtype=np.int64),
            "dest_node": np.asarray(int(self.routes.dest_node[rid]), dtype=np.int64),
            "start_t": np.asarray(int(self.routes.start_t[rid]), dtype=np.int64),
            "route_city": np.asarray(int(self.routes.route_city[rid]), dtype=np.int64),
            "start_pos": self.routes.start_pos[rid].astype(np.float32, copy=False),
            "dest_pos": self.routes.dest_pos[rid].astype(np.float32, copy=False),
        }


def _hour_from_unix(start_t: np.ndarray, tz_offset_hours: float) -> np.ndarray:
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    sec = ((start_t + tz_sec) % 86400).astype(np.int64, copy=False)
    return (sec // 3600).astype(np.int64, copy=False)


def make_casd_collate_fn(
    *,
    node_seg_ptr: np.ndarray,
    node_seg_idx: np.ndarray,
    seg_succ_ptr: np.ndarray,
    seg_succ_idx: np.ndarray,
    max_candidates: int,
    tz_offset_hours: float,
) -> Callable[[List[Dict[str, np.ndarray]]], Dict[str, torch.Tensor]]:
    max_candidates = int(max_candidates)
    node_seg_ptr = np.asarray(node_seg_ptr, dtype=np.int64)
    node_seg_idx = np.asarray(node_seg_idx, dtype=np.int64)
    seg_succ_ptr = np.asarray(seg_succ_ptr, dtype=np.int64)
    seg_succ_idx = np.asarray(seg_succ_idx, dtype=np.int64)

    def _node_cands(node: int) -> np.ndarray:
        s = int(node_seg_ptr[node])
        e = int(node_seg_ptr[node + 1])
        return node_seg_idx[s:e].copy()

    def _succ_cands(seg: int) -> np.ndarray:
        s = int(seg_succ_ptr[seg])
        e = int(seg_succ_ptr[seg + 1])
        return seg_succ_idx[s:e].copy()

    def _ensure_target(cands: np.ndarray, target: int) -> np.ndarray:
        if cands.size == 0:
            return np.asarray([target], dtype=np.int64)
        if int(target) in set(cands.tolist()):
            return cands
        if cands.size < max_candidates:
            return np.concatenate([cands, np.asarray([target], dtype=np.int64)], axis=0)
        cands = cands.copy()
        cands[-1] = int(target)
        return cands

    def collate(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        seg_lens = np.asarray([int(b["seg_len"]) for b in batch], dtype=np.int64)
        Kmax = int(seg_lens.max()) if B > 0 else 1
        seg_pad = np.full((B, Kmax), -1, dtype=np.int64)
        for i, b in enumerate(batch):
            L = int(b["seg_len"])
            seg_pad[i, :L] = np.asarray(b["seg_seq"], dtype=np.int64)[:L]

        start_t = np.asarray([int(b["start_t"]) for b in batch], dtype=np.int64)
        hour = _hour_from_unix(start_t, tz_offset_hours=float(tz_offset_hours))

        route_cond = {
            "start_pos": torch.as_tensor(np.stack([b["start_pos"] for b in batch], axis=0), dtype=torch.float32),
            "dest_pos": torch.as_tensor(np.stack([b["dest_pos"] for b in batch], axis=0), dtype=torch.float32),
            "hour": torch.as_tensor(hour, dtype=torch.long),
            "route_city": torch.as_tensor(np.asarray([int(b["route_city"]) for b in batch], dtype=np.int64), dtype=torch.long),
            "corridor_type": torch.as_tensor(np.asarray([int(b["corridor_type"]) for b in batch], dtype=np.int64), dtype=torch.long),
            "start_node": torch.as_tensor(np.asarray([int(b["start_node"]) for b in batch], dtype=np.int64), dtype=torch.long),
            "dest_node": torch.as_tensor(np.asarray([int(b["dest_node"]) for b in batch], dtype=np.int64), dtype=torch.long),
        }

        # Build packed transitions across the batch.
        route_idx: List[int] = []
        cur_seg: List[int] = []
        target_idx: List[int] = []
        cand_seg_rows: List[np.ndarray] = []
        cand_mask_rows: List[np.ndarray] = []

        for bi in range(B):
            L = int(seg_lens[bi])
            if L <= 0:
                continue
            seq = seg_pad[bi, :L]
            sn = int(route_cond["start_node"][bi].item())
            for j in range(L):
                tgt = int(seq[j])
                if j == 0:
                    cands = _node_cands(sn)
                    cur = -1
                else:
                    prev = int(seq[j - 1])
                    cands = _succ_cands(prev)
                    cur = prev
                cands = _ensure_target(cands[:max_candidates], tgt)
                C = min(int(cands.size), max_candidates)
                row = np.full((max_candidates,), -1, dtype=np.int64)
                row[:C] = cands[:C]
                mask = np.zeros((max_candidates,), dtype=bool)
                mask[:C] = True
                pos = int(np.where(row == tgt)[0][0])
                route_idx.append(bi)
                cur_seg.append(cur)
                target_idx.append(pos)
                cand_seg_rows.append(row)
                cand_mask_rows.append(mask)

        trans = {
            "route_idx": torch.as_tensor(np.asarray(route_idx, dtype=np.int64), dtype=torch.long),
            "cur_seg": torch.as_tensor(np.asarray(cur_seg, dtype=np.int64), dtype=torch.long),
            "cand_seg": torch.as_tensor(np.stack(cand_seg_rows, axis=0), dtype=torch.long),
            "cand_mask": torch.as_tensor(np.stack(cand_mask_rows, axis=0), dtype=torch.bool),
            "target_idx": torch.as_tensor(np.asarray(target_idx, dtype=np.int64), dtype=torch.long),
        }

        return {
            "seg_seq_pad": torch.as_tensor(seg_pad, dtype=torch.long),
            "seg_seq_len": torch.as_tensor(seg_lens, dtype=torch.long),
            "route_cond": route_cond,
            "trans": trans,
        }

    return collate

