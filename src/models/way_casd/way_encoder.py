from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class WayFeatureTensors:
    way_center_y: torch.Tensor  # (M,)
    way_center_x: torch.Tensor  # (M,)
    way_dir_y: torch.Tensor  # (M,)
    way_dir_x: torch.Tensor  # (M,)
    way_len_m: torch.Tensor  # (M,)
    way_tier: torch.Tensor  # (M,)
    way_highway_code: torch.Tensor  # (M,)


def _as_tensor(x, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def make_way_feature_tensors(
    *,
    way_center_y,
    way_center_x,
    way_dir_y,
    way_dir_x,
    way_len_m,
    way_tier,
    way_highway_code,
    device: Optional[torch.device] = None,
) -> WayFeatureTensors:
    out = WayFeatureTensors(
        way_center_y=_as_tensor(way_center_y, dtype=torch.float32),
        way_center_x=_as_tensor(way_center_x, dtype=torch.float32),
        way_dir_y=_as_tensor(way_dir_y, dtype=torch.float32),
        way_dir_x=_as_tensor(way_dir_x, dtype=torch.float32),
        way_len_m=_as_tensor(way_len_m, dtype=torch.float32),
        way_tier=_as_tensor(way_tier, dtype=torch.long),
        way_highway_code=_as_tensor(way_highway_code, dtype=torch.long),
    )
    if device is None:
        return out
    return WayFeatureTensors(
        way_center_y=out.way_center_y.to(device=device),
        way_center_x=out.way_center_x.to(device=device),
        way_dir_y=out.way_dir_y.to(device=device),
        way_dir_x=out.way_dir_x.to(device=device),
        way_len_m=out.way_len_m.to(device=device),
        way_tier=out.way_tier.to(device=device),
        way_highway_code=out.way_highway_code.to(device=device),
    )


class WayEncoder(nn.Module):
    """
    Feature-based way encoder (NO learnable way ID embedding).

    Input:
      way_ids: (B,K) int64, padded with -1
    Output:
      emb: (B,K,d_model)
      mask: (B,K) bool, True where way_ids valid
    """

    def __init__(
        self,
        *,
        features: WayFeatureTensors,
        d_model: int = 256,
        n_tiers: int = 4,
        n_highway_types: int = 16,
        coord_scale: float = 1024.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.coord_scale = float(coord_scale)

        self.register_buffer("way_center_y", features.way_center_y, persistent=False)
        self.register_buffer("way_center_x", features.way_center_x, persistent=False)
        self.register_buffer("way_dir_y", features.way_dir_y, persistent=False)
        self.register_buffer("way_dir_x", features.way_dir_x, persistent=False)
        self.register_buffer("way_len_m", features.way_len_m, persistent=False)
        self.register_buffer("way_tier", features.way_tier, persistent=False)
        self.register_buffer("way_highway_code", features.way_highway_code, persistent=False)

        self.geom_mlp = nn.Sequential(
            nn.Linear(5, int(d_model)),
            nn.SiLU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        self.tier_embed = nn.Embedding(int(n_tiers), int(d_model))
        self.highway_embed = nn.Embedding(int(n_highway_types), int(d_model))
        self.out_ln = nn.LayerNorm(int(d_model))

    def _lookup(self, way_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        way_ids = way_ids.to(dtype=torch.long)
        mask = way_ids >= 0
        ids = torch.clamp(way_ids, min=0)

        cy = self.way_center_y[ids]
        cx = self.way_center_x[ids]
        dy = self.way_dir_y[ids]
        dx = self.way_dir_x[ids]
        ln = self.way_len_m[ids]
        tier = self.way_tier[ids]
        hw = self.way_highway_code[ids]

        if cy.dtype != torch.float32:
            cy = cy.float()
            cx = cx.float()
            dy = dy.float()
            dx = dx.float()
            ln = ln.float()

        if self.coord_scale > 0:
            cy = cy / self.coord_scale
            cx = cx / self.coord_scale

        ln = torch.log1p(torch.clamp_min(ln, 0.0))

        if not bool(mask.all()):
            z = torch.zeros((), dtype=cy.dtype, device=cy.device)
            cy = torch.where(mask, cy, z)
            cx = torch.where(mask, cx, z)
            dy = torch.where(mask, dy, z)
            dx = torch.where(mask, dx, z)
            ln = torch.where(mask, ln, z)
            tier = torch.where(mask, tier, torch.zeros((), dtype=tier.dtype, device=tier.device))
            hw = torch.where(mask, hw, torch.zeros((), dtype=hw.dtype, device=hw.device))

        geom = torch.stack([cy, cx, dy, dx, ln], dim=-1)
        return geom, tier, hw

    def forward(self, way_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        geom, tier, hw = self._lookup(way_ids)
        x = self.geom_mlp(geom)
        x = x + self.tier_embed(torch.clamp(tier, 0, self.tier_embed.num_embeddings - 1))
        x = x + self.highway_embed(torch.clamp(hw, 0, self.highway_embed.num_embeddings - 1))
        x = self.out_ln(x)
        mask = way_ids >= 0
        return x, mask

