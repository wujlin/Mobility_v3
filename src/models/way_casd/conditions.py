from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ConditionEncoderCfg:
    d_model: int = 256
    n_route_cities: int = 4
    n_corridor_types: int = 4
    coord_scale: float = 1024.0


class ConditionEncoder(nn.Module):
    """
    Encode route-level conditions into a single vector.

    Required fields:
      - start_pos: (B,2) in (y,x)
      - dest_pos:  (B,2) in (y,x)
      - hour:      (B,) int [0,23]
      - dow:       (B,) int [0,6] (Mon=0)
      - route_city:(B,) int
      - corridor_type: (B,) int in [0,3] or -1 for 'null'
    """

    def __init__(self, cfg: ConditionEncoderCfg) -> None:
        super().__init__()
        self.cfg = cfg
        d_model = int(cfg.d_model)
        self.coord_scale = float(cfg.coord_scale)

        self.pos_fc = nn.Linear(4, d_model)
        self.time_fc = nn.Sequential(
            nn.Linear(4, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.route_city_embed = nn.Embedding(int(cfg.n_route_cities), d_model)
        self.corridor_embed = nn.Embedding(int(cfg.n_corridor_types), d_model)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        *,
        start_pos: torch.Tensor,
        dest_pos: torch.Tensor,
        hour: torch.Tensor,
        dow: torch.Tensor,
        route_city: torch.Tensor,
        corridor_type: Optional[torch.Tensor],
    ) -> torch.Tensor:
        start_pos = start_pos.to(dtype=torch.float32)
        dest_pos = dest_pos.to(dtype=torch.float32)
        hour = hour.to(dtype=torch.long)
        dow = dow.to(dtype=torch.long)
        route_city = route_city.to(dtype=torch.long)

        pos = torch.cat([start_pos, dest_pos], dim=-1)
        if self.coord_scale > 0:
            pos = pos / self.coord_scale
        pos_emb = self.pos_fc(pos)

        hr = hour.float() * (2.0 * 3.141592653589793 / 24.0)
        dw = dow.float() * (2.0 * 3.141592653589793 / 7.0)
        time_feat = torch.stack([torch.sin(hr), torch.cos(hr), torch.sin(dw), torch.cos(dw)], dim=-1)
        time_emb = self.time_fc(time_feat)

        city = torch.clamp(route_city, 0, self.route_city_embed.num_embeddings - 1)
        city_emb = self.route_city_embed(city)

        if corridor_type is None:
            corr_emb = torch.zeros_like(city_emb)
        else:
            corridor_type = corridor_type.to(dtype=torch.long)
            is_null = corridor_type < 0
            corr = torch.clamp(corridor_type, 0, self.corridor_embed.num_embeddings - 1)
            corr_emb = self.corridor_embed(corr)
            if bool(is_null.any()):
                corr_emb = corr_emb.clone()
                corr_emb[is_null] = 0.0

        out = pos_emb + time_emb + city_emb + corr_emb
        return self.out_ln(out)

