from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.way_graph.way_sequence_dataset import WayRoutes, load_way_routes_npz


@dataclass(frozen=True)
class RegionSeqCSR:
    route_id: np.ndarray  # (K,) int64
    region_seq_ptr: np.ndarray  # (K+1,) int64
    region_seq_idx: np.ndarray  # (total,) int32
    region_seq_len: np.ndarray  # (K,) int32


def load_region_seq_npz(path: Path) -> RegionSeqCSR:
    data = np.load(str(path), allow_pickle=True)
    need = {"route_id", "region_seq_ptr", "region_seq_idx", "region_seq_len"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"region_seq.npz missing keys: {missing}")
    rid = np.asarray(data["route_id"], dtype=np.int64).reshape(-1)
    ptr = np.asarray(data["region_seq_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(data["region_seq_idx"], dtype=np.int32).reshape(-1)
    ln = np.asarray(data["region_seq_len"], dtype=np.int32).reshape(-1)
    if ptr.size != rid.size + 1:
        raise ValueError(f"region_seq_ptr shape mismatch: got {ptr.size}, expect {rid.size + 1}")
    return RegionSeqCSR(route_id=rid, region_seq_ptr=ptr, region_seq_idx=idx, region_seq_len=ln)


def _hour_from_unix(start_t: int, tz_offset_hours: float) -> int:
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    sec = int((int(start_t) + tz_sec) % 86400)
    return int(sec // 3600)


def _dow_from_unix(start_t: int, tz_offset_hours: float) -> int:
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    days = int((int(start_t) + tz_sec) // 86400)
    # 1970-01-01 is Thursday (weekday=3)
    return int((days + 3) % 7)


class RegionRouteDataset(Dataset):
    """
    Per-route region sequences with route-level conditioning (same as Way-CASD).

    region_seq is usually produced by:
      src.data.way_graph.extract_region_seq_stats --out_npz ...
    """

    def __init__(
        self,
        *,
        way_routes: WayRoutes,
        region_seq: RegionSeqCSR,
        way_region: np.ndarray,  # (n_ways,) int
        tz_offset_hours: float,
        max_routes: Optional[int] = None,
    ) -> None:
        self.routes = way_routes
        self.region_seq = region_seq
        self.way_region = np.asarray(way_region, dtype=np.int64).reshape(-1)
        self.tz_offset_hours = float(tz_offset_hours)

        ids = np.arange(int(region_seq.route_id.size), dtype=np.int64)
        if max_routes is not None:
            ids = ids[: int(max_routes)]
        self.row_ids = ids

    def __len__(self) -> int:
        return int(self.row_ids.size)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        row = int(self.row_ids[int(idx)])
        rid = int(self.region_seq.route_id[row])
        s = int(self.region_seq.region_seq_ptr[row])
        e = int(self.region_seq.region_seq_ptr[row + 1])
        reg_seq = self.region_seq.region_seq_idx[s:e].astype(np.int64, copy=False)

        city = int(self.routes.route_city[rid])
        start_t = int(self.routes.start_t[rid])
        hour = _hour_from_unix(start_t, tz_offset_hours=self.tz_offset_hours)
        dow = _dow_from_unix(start_t, tz_offset_hours=self.tz_offset_hours)

        sw = int(self.routes.start_way[rid])
        dw = int(self.routes.dest_way[rid])
        o_region = int(self.way_region[sw])
        d_region = int(self.way_region[dw])

        return {
            "route_id": np.asarray(rid, dtype=np.int64),
            "region_seq": reg_seq,
            "region_len": np.asarray(int(reg_seq.size), dtype=np.int64),
            "o_region": np.asarray(o_region, dtype=np.int64),
            "d_region": np.asarray(d_region, dtype=np.int64),
            "start_pos": self.routes.start_pos[rid].astype(np.float32, copy=False),
            "dest_pos": self.routes.dest_pos[rid].astype(np.float32, copy=False),
            "hour": np.asarray(int(hour), dtype=np.int64),
            "dow": np.asarray(int(dow), dtype=np.int64),
            "route_city": np.asarray(int(city), dtype=np.int64),
        }


def make_region_ar_collate_fn(*, max_len: int) -> callable:
    max_len = int(max_len)

    def collate(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        B = int(len(batch))
        lens = np.asarray([int(b["region_len"]) for b in batch], dtype=np.int64)
        T = int(min(int(lens.max()) if B else 1, max_len))

        pad = np.full((B, T), -1, dtype=np.int64)
        for i, b in enumerate(batch):
            seq = np.asarray(b["region_seq"], dtype=np.int64).reshape(-1)
            L = int(min(int(seq.size), T))
            pad[i, :L] = seq[:L]

        route_cond = {
            "start_pos": torch.as_tensor(np.stack([b["start_pos"] for b in batch], axis=0), dtype=torch.float32),
            "dest_pos": torch.as_tensor(np.stack([b["dest_pos"] for b in batch], axis=0), dtype=torch.float32),
            "hour": torch.as_tensor(np.asarray([int(b["hour"]) for b in batch], dtype=np.int64), dtype=torch.long),
            "dow": torch.as_tensor(np.asarray([int(b["dow"]) for b in batch], dtype=np.int64), dtype=torch.long),
            "route_city": torch.as_tensor(np.asarray([int(b["route_city"]) for b in batch], dtype=np.int64), dtype=torch.long),
        }

        return {
            "route_id": torch.as_tensor(np.asarray([int(b["route_id"]) for b in batch], dtype=np.int64), dtype=torch.long),
            "region_seq_pad": torch.as_tensor(pad, dtype=torch.long),
            "o_region": torch.as_tensor(np.asarray([int(b["o_region"]) for b in batch], dtype=np.int64), dtype=torch.long),
            "d_region": torch.as_tensor(np.asarray([int(b["d_region"]) for b in batch], dtype=np.int64), dtype=torch.long),
            "route_cond": route_cond,
        }

    return collate


def load_region_ar_dataset(
    *,
    way_routes_npz: Path,
    region_seq_npz: Path,
    way_regions_npz: Path,
    tz_offset_hours: float,
    max_routes: Optional[int] = None,
) -> RegionRouteDataset:
    routes = load_way_routes_npz(Path(way_routes_npz))
    reg_seq = load_region_seq_npz(Path(region_seq_npz))
    wr = np.load(str(way_regions_npz), allow_pickle=True)
    if "way_region" not in wr.files:
        raise ValueError(f"{way_regions_npz} missing key: way_region")
    way_region = np.asarray(wr["way_region"], dtype=np.int64).reshape(-1)
    return RegionRouteDataset(
        way_routes=routes,
        region_seq=reg_seq,
        way_region=way_region,
        tz_offset_hours=float(tz_offset_hours),
        max_routes=max_routes,
    )

