from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class WaypointBinARConfig:
    hidden_dim: int = 256
    tier_emb_dim: int = 8
    city_emb_dim: int = 8
    step_emb_dim: int = 8
    num_cities: int = 2


class ARGraphWaypointBin(nn.Module):
    """
    Minimal waypoint-level autoregressive model.

    Factorization (fixed K steps):
      p(bin_{t+1} | wp_t, dest, time, city, node_tier, step_idx)

    Output space is a coarse grid bin (wp_bin x wp_bin), NOT node-level (too long-horizon).
    """

    def __init__(self, *, cfg: WaypointBinARConfig, n_classes: int, num_steps: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_classes = int(n_classes)
        self.num_steps = int(num_steps)

        self.tier_emb = nn.Embedding(4, int(cfg.tier_emb_dim))
        self.city_emb = nn.Embedding(int(cfg.num_cities), int(cfg.city_emb_dim))
        self.step_emb = nn.Embedding(int(num_steps), int(cfg.step_emb_dim))

        in_dim = 0
        # geometry: cur_yx,dest_yx,delta_yx,dist = 7
        in_dim += 7
        # time: 5
        in_dim += 5
        # tier emb for (cur,dest)
        in_dim += 2 * int(cfg.tier_emb_dim)
        # city + step
        in_dim += int(cfg.city_emb_dim) + int(cfg.step_emb_dim)

        h = int(cfg.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, int(n_classes)),
        )

    @staticmethod
    def _norm_xy(yx: torch.Tensor, *, denom: float = 1023.0) -> torch.Tensor:
        return yx / float(denom)

    def forward(
        self,
        *,
        node_yx: torch.Tensor,  # (N,2) float
        node_tier_min: torch.Tensor,  # (N,) long in [0,3]
        cur: torch.Tensor,  # (B,) long
        dest: torch.Tensor,  # (B,) long
        time_feat: torch.Tensor,  # (B,5) float
        route_city: torch.Tensor,  # (B,) long
        step_idx: torch.Tensor,  # (B,) long in [0,num_steps-1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits, mask) where mask is always True (kept for interface symmetry).
        """
        device = node_yx.device
        cur = cur.to(device=device, dtype=torch.long)
        dest = dest.to(device=device, dtype=torch.long)
        time_feat = time_feat.to(device=device, dtype=torch.float32)
        route_city = route_city.to(device=device, dtype=torch.long)
        step_idx = step_idx.to(device=device, dtype=torch.long)
        node_tier_min = node_tier_min.to(device=device, dtype=torch.long)

        cur_yx = node_yx[cur]  # (B,2)
        dest_yx = node_yx[dest]  # (B,2)
        cur_yx_n = self._norm_xy(cur_yx)
        dest_yx_n = self._norm_xy(dest_yx)
        delta = dest_yx_n - cur_yx_n
        dist = torch.sqrt(torch.sum(delta * delta, dim=-1, keepdim=True) + 1e-6)
        geom = torch.cat([cur_yx_n, dest_yx_n, delta, dist], dim=-1)  # (B,7)

        cur_tier = torch.clamp(node_tier_min[cur], 0, 3)
        dest_tier = torch.clamp(node_tier_min[dest], 0, 3)
        te = torch.cat([self.tier_emb(cur_tier), self.tier_emb(dest_tier)], dim=-1)

        ce = self.city_emb(torch.clamp(route_city, 0, int(self.cfg.num_cities) - 1))
        se = self.step_emb(torch.clamp(step_idx, 0, int(self.num_steps) - 1))

        x = torch.cat([geom, time_feat, te, ce, se], dim=-1)
        logits = self.mlp(x)
        mask = torch.ones((logits.shape[0],), device=logits.device, dtype=torch.bool)
        return logits, mask

