from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WayRoutes:
    way_osm_id: np.ndarray  # (M,) int64
    way_seq_ptr: np.ndarray  # (N+1,) int64
    way_seq_idx: np.ndarray  # (total,) int32
    way_seq_len: np.ndarray  # (N,) int32
    corridor_type: np.ndarray  # (N,) int8
    start_way: np.ndarray  # (N,) int32
    dest_way: np.ndarray  # (N,) int32
    start_t: np.ndarray  # (N,) int64
    route_city: np.ndarray  # (N,) int8
    start_pos: np.ndarray  # (N,2) float32 (y,x)
    dest_pos: np.ndarray  # (N,2) float32 (y,x)


def load_way_routes_npz(path: Path) -> WayRoutes:
    data = np.load(str(path), allow_pickle=True)
    need = {
        "way_osm_id",
        "way_seq_ptr",
        "way_seq_idx",
        "way_seq_len",
        "corridor_type",
        "start_way",
        "dest_way",
        "start_t",
        "route_city",
        "start_pos",
        "dest_pos",
    }
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"way_routes.npz missing keys: {missing}")
    return WayRoutes(
        way_osm_id=np.asarray(data["way_osm_id"], dtype=np.int64).reshape(-1),
        way_seq_ptr=np.asarray(data["way_seq_ptr"], dtype=np.int64),
        way_seq_idx=np.asarray(data["way_seq_idx"], dtype=np.int32),
        way_seq_len=np.asarray(data["way_seq_len"], dtype=np.int32).reshape(-1),
        corridor_type=np.asarray(data["corridor_type"], dtype=np.int8).reshape(-1),
        start_way=np.asarray(data["start_way"], dtype=np.int32).reshape(-1),
        dest_way=np.asarray(data["dest_way"], dtype=np.int32).reshape(-1),
        start_t=np.asarray(data["start_t"], dtype=np.int64).reshape(-1),
        route_city=np.asarray(data["route_city"], dtype=np.int8).reshape(-1),
        start_pos=np.asarray(data["start_pos"], dtype=np.float32).reshape(-1, 2),
        dest_pos=np.asarray(data["dest_pos"], dtype=np.float32).reshape(-1, 2),
    )


class WayRouteDataset(Dataset):
    """Dataset of per-route way sequences (CSR-backed)."""

    def __init__(
        self,
        routes: WayRoutes,
        *,
        max_routes: Optional[int] = None,
        max_way_len: Optional[int] = None,
        min_hops: int = 1,
    ) -> None:
        self.routes = routes
        keep = routes.way_seq_len > 0
        # A route with L ways has (L-1) transitions (hops). We filter by transitions.
        keep &= routes.way_seq_len >= (int(min_hops) + 1)
        if max_way_len is not None:
            keep &= routes.way_seq_len <= int(max_way_len)
        ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
        if max_routes is not None:
            ids = ids[: int(max_routes)]
        self.route_ids = ids

    def __len__(self) -> int:
        return int(self.route_ids.size)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        rid = int(self.route_ids[int(idx)])
        L = int(self.routes.way_seq_len[rid])
        s = int(self.routes.way_seq_ptr[rid])
        e = s + L
        way_seq = self.routes.way_seq_idx[s:e].astype(np.int64, copy=False)
        return {
            "route_id": np.asarray(rid, dtype=np.int64),
            "way_seq": way_seq,
            "way_len": np.asarray(L, dtype=np.int64),
            "corridor_type": np.asarray(int(self.routes.corridor_type[rid]), dtype=np.int64),
            "start_way": np.asarray(int(self.routes.start_way[rid]), dtype=np.int64),
            "dest_way": np.asarray(int(self.routes.dest_way[rid]), dtype=np.int64),
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


def _dow_from_unix(start_t: np.ndarray, tz_offset_hours: float) -> np.ndarray:
    """
    Day-of-week with Monday=0..Sunday=6 (same as datetime.weekday()).

    1970-01-01 is Thursday (weekday=3), so:
      dow = (days_since_epoch + 3) % 7
    """
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    days = ((start_t + tz_sec) // 86400).astype(np.int64, copy=False)
    return ((days + 3) % 7).astype(np.int64, copy=False)


def make_way_casd_collate_fn(
    *,
    way_adj_ptr: np.ndarray,
    way_adj_idx: np.ndarray,
    max_candidates: int,
    tz_offset_hours: float,
    past_k: int = 8,  # Number of past steps to include for past context
) -> Callable[[List[Dict[str, np.ndarray]]], Dict[str, torch.Tensor]]:
    way_adj_ptr = np.asarray(way_adj_ptr, dtype=np.int64)
    way_adj_idx = np.asarray(way_adj_idx, dtype=np.int64)
    max_candidates = int(max_candidates)
    past_k = int(past_k)

    def _succ_cands(way: int) -> np.ndarray:
        s = int(way_adj_ptr[way])
        e = int(way_adj_ptr[way + 1])
        return way_adj_idx[s:e].copy()

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
        way_lens = np.asarray([int(b["way_len"]) for b in batch], dtype=np.int64)
        Kmax = int(way_lens.max()) if B > 0 else 1
        way_pad = np.full((B, Kmax), -1, dtype=np.int64)
        for i, b in enumerate(batch):
            L = int(b["way_len"])
            way_pad[i, :L] = np.asarray(b["way_seq"], dtype=np.int64)[:L]

        start_t = np.asarray([int(b["start_t"]) for b in batch], dtype=np.int64)
        hour = _hour_from_unix(start_t, tz_offset_hours=float(tz_offset_hours))
        dow = _dow_from_unix(start_t, tz_offset_hours=float(tz_offset_hours))

        route_cond = {
            "start_pos": torch.as_tensor(np.stack([b["start_pos"] for b in batch], axis=0), dtype=torch.float32),
            "dest_pos": torch.as_tensor(np.stack([b["dest_pos"] for b in batch], axis=0), dtype=torch.float32),
            "hour": torch.as_tensor(hour, dtype=torch.long),
            "dow": torch.as_tensor(dow, dtype=torch.long),
            "route_city": torch.as_tensor(np.asarray([int(b["route_city"]) for b in batch], dtype=np.int64), dtype=torch.long),
            "corridor_type": torch.as_tensor(np.asarray([int(b["corridor_type"]) for b in batch], dtype=np.int64), dtype=torch.long),
            "start_way": torch.as_tensor(np.asarray([int(b["start_way"]) for b in batch], dtype=np.int64), dtype=torch.long),
            "dest_way": torch.as_tensor(np.asarray([int(b["dest_way"]) for b in batch], dtype=np.int64), dtype=torch.long),
        }
        route_id = torch.as_tensor(np.asarray([int(b["route_id"]) for b in batch], dtype=np.int64), dtype=torch.long)

        # Packed transitions: for each step j>=1, predict way[j] from way[j-1] candidates.
        route_idx: List[int] = []
        cur_way: List[int] = []
        target_idx: List[int] = []
        step: List[int] = []
        cand_way_rows: List[np.ndarray] = []
        cand_mask_rows: List[np.ndarray] = []
        past_way_rows: List[np.ndarray] = []
        past_mask_rows: List[np.ndarray] = []

        for bi in range(B):
            L = int(way_lens[bi])
            if L <= 1:
                continue
            seq = way_pad[bi, :L]
            for j in range(1, L):
                prev = int(seq[j - 1])
                tgt = int(seq[j])
                cands = _succ_cands(prev)[:max_candidates]
                cands = _ensure_target(cands, tgt)
                C = min(int(cands.size), max_candidates)
                row = np.full((max_candidates,), -1, dtype=np.int64)
                row[:C] = cands[:C]
                mask = np.zeros((max_candidates,), dtype=bool)
                mask[:C] = True
                pos = int(np.where(row == tgt)[0][0])

                # Build past_way: last K ways before current step (seq[0:j-1])
                # Right-aligned: most recent at the end
                past_seq = seq[:j - 1] if j > 1 else np.array([], dtype=np.int64)
                past_len = min(int(past_seq.size), past_k)
                past_row = np.full((past_k,), -1, dtype=np.int64)
                if past_len > 0:
                    offset = past_k - past_len
                    past_row[offset:] = past_seq[-past_len:]
                past_m = (past_row >= 0)

                route_idx.append(bi)
                cur_way.append(prev)
                target_idx.append(pos)
                step.append(int(j - 1))
                cand_way_rows.append(row)
                cand_mask_rows.append(mask)
                past_way_rows.append(past_row)
                past_mask_rows.append(past_m)

        if not cand_way_rows:
            cand_way = torch.zeros((0, int(max_candidates)), dtype=torch.long)
            cand_mask = torch.zeros((0, int(max_candidates)), dtype=torch.bool)
            past_way = torch.zeros((0, int(past_k)), dtype=torch.long)
            past_mask = torch.zeros((0, int(past_k)), dtype=torch.bool)
        else:
            cand_way = torch.as_tensor(np.stack(cand_way_rows, axis=0), dtype=torch.long)
            cand_mask = torch.as_tensor(np.stack(cand_mask_rows, axis=0), dtype=torch.bool)
            past_way = torch.as_tensor(np.stack(past_way_rows, axis=0), dtype=torch.long)
            past_mask = torch.as_tensor(np.stack(past_mask_rows, axis=0), dtype=torch.bool)

        trans = {
            "route_idx": torch.as_tensor(np.asarray(route_idx, dtype=np.int64), dtype=torch.long),
            "cur_way": torch.as_tensor(np.asarray(cur_way, dtype=np.int64), dtype=torch.long),
            "cand_way": cand_way,
            "cand_mask": cand_mask,
            "target_idx": torch.as_tensor(np.asarray(target_idx, dtype=np.int64), dtype=torch.long),
            "step": torch.as_tensor(np.asarray(step, dtype=np.int64), dtype=torch.long),
            "past_way": past_way,
            "past_mask": past_mask,
        }

        return {
            "route_id": route_id,
            "way_seq_pad": torch.as_tensor(way_pad, dtype=torch.long),
            "way_seq_len": torch.as_tensor(way_lens, dtype=torch.long),
            "route_cond": route_cond,
            "trans": trans,
        }

    return collate
