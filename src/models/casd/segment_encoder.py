from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SegmentFeatureTensors:
    seg_center_y: torch.Tensor  # (S,)
    seg_center_x: torch.Tensor  # (S,)
    seg_dir_y: torch.Tensor  # (S,)
    seg_dir_x: torch.Tensor  # (S,)
    seg_len_m: torch.Tensor  # (S,)
    seg_tier: torch.Tensor  # (S,)
    seg_city: torch.Tensor  # (S,)


def _as_tensor(x, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def make_segment_feature_tensors(
    *,
    seg_center_y,
    seg_center_x,
    seg_dir_y,
    seg_dir_x,
    seg_len_m,
    seg_tier,
    seg_city,
    device: Optional[torch.device] = None,
) -> SegmentFeatureTensors:
    out = SegmentFeatureTensors(
        seg_center_y=_as_tensor(seg_center_y, dtype=torch.float32),
        seg_center_x=_as_tensor(seg_center_x, dtype=torch.float32),
        seg_dir_y=_as_tensor(seg_dir_y, dtype=torch.float32),
        seg_dir_x=_as_tensor(seg_dir_x, dtype=torch.float32),
        seg_len_m=_as_tensor(seg_len_m, dtype=torch.float32),
        seg_tier=_as_tensor(seg_tier, dtype=torch.long),
        seg_city=_as_tensor(seg_city, dtype=torch.long),
    )
    if device is None:
        return out
    return SegmentFeatureTensors(
        seg_center_y=out.seg_center_y.to(device=device),
        seg_center_x=out.seg_center_x.to(device=device),
        seg_dir_y=out.seg_dir_y.to(device=device),
        seg_dir_x=out.seg_dir_x.to(device=device),
        seg_len_m=out.seg_len_m.to(device=device),
        seg_tier=out.seg_tier.to(device=device),
        seg_city=out.seg_city.to(device=device),
    )


class SegmentEncoder(nn.Module):
    """
    Feature-based segment encoder (NO learnable segment ID embedding).

    Input:
      seg_ids: (B,K) int64, padded with -1
    Output:
      emb: (B,K,d_model)
      mask: (B,K) bool, True where seg_ids valid
    """

    def __init__(
        self,
        *,
        features: SegmentFeatureTensors,
        d_model: int = 256,
        n_tiers: int = 4,
        n_cities: int = 4,
        coord_scale: float = 1024.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.coord_scale = float(coord_scale)

        self.register_buffer("seg_center_y", features.seg_center_y, persistent=False)
        self.register_buffer("seg_center_x", features.seg_center_x, persistent=False)
        self.register_buffer("seg_dir_y", features.seg_dir_y, persistent=False)
        self.register_buffer("seg_dir_x", features.seg_dir_x, persistent=False)
        self.register_buffer("seg_len_m", features.seg_len_m, persistent=False)
        self.register_buffer("seg_tier", features.seg_tier, persistent=False)
        self.register_buffer("seg_city", features.seg_city, persistent=False)

        self.geom_mlp = nn.Sequential(
            nn.Linear(5, int(d_model)),
            nn.SiLU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        self.tier_embed = nn.Embedding(int(n_tiers), int(d_model))
        self.city_embed = nn.Embedding(int(n_cities), int(d_model))
        self.out_ln = nn.LayerNorm(int(d_model))

    def _lookup(self, seg_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seg_ids = seg_ids.to(dtype=torch.long)
        mask = seg_ids >= 0
        ids = torch.clamp(seg_ids, min=0)

        cy = self.seg_center_y[ids]
        cx = self.seg_center_x[ids]
        dy = self.seg_dir_y[ids]
        dx = self.seg_dir_x[ids]
        ln = self.seg_len_m[ids]
        tier = self.seg_tier[ids]
        city = self.seg_city[ids]

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
            city = torch.where(mask, city, torch.zeros((), dtype=city.dtype, device=city.device))

        geom = torch.stack([cy, cx, dy, dx, ln], dim=-1)
        return geom, tier, city

    def forward(self, seg_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        geom, tier, city = self._lookup(seg_ids)
        x = self.geom_mlp(geom)
        x = x + self.tier_embed(torch.clamp(tier, 0, self.tier_embed.num_embeddings - 1))
        x = x + self.city_embed(torch.clamp(city, 0, self.city_embed.num_embeddings - 1))
        x = self.out_ln(x)
        mask = seg_ids >= 0
        return x, mask

