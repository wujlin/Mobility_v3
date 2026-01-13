from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ARDecisionConfig:
    hidden_dim: int = 256
    edge_tier_emb_dim: int = 8


class ARGraphDecisionMarkov(nn.Module):
    """
    Minimal autoregressive decision model on a road graph.

    Factorization (1st-order Markov):
      p(n_{t+1} | n_t, dest, time, edge_tier, geometry)

    At each step, it scores ONLY the neighbor set of the current node.
    """

    def __init__(self, *, cfg: ARDecisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Edge tier in build_road_graph_from_osm: 0=major,1=minor,2=service,3=other
        self.edge_tier_emb = nn.Embedding(4, int(cfg.edge_tier_emb_dim))

        in_dim = 0
        # geometry: (u_y,u_x,d_y,d_x,v_y,v_x, duv_y,duv_x, dvd_y,dvd_x) = 10
        in_dim += 10
        # time features: 5 (sin/cos hour, sin/cos dow, is_weekend)
        in_dim += 5
        # edge tier emb
        in_dim += int(cfg.edge_tier_emb_dim)

        h = int(cfg.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    @staticmethod
    def _norm_xy(yx: torch.Tensor, *, denom: float = 1023.0) -> torch.Tensor:
        return yx / float(denom)

    def score_neighbors(
        self,
        *,
        node_yx: torch.Tensor,  # (N,2) float
        cur: torch.Tensor,  # (B,) long
        dest: torch.Tensor,  # (B,) long
        neigh: torch.Tensor,  # (B,M) long; -1 is padding
        neigh_tier: torch.Tensor,  # (B,M) long; undefined for padding
        time_feat: torch.Tensor,  # (B,5) float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits, mask) with logits shape (B,M), mask shape (B,M) bool.
        """
        device = node_yx.device
        cur = cur.to(device=device, dtype=torch.long)
        dest = dest.to(device=device, dtype=torch.long)
        neigh = neigh.to(device=device, dtype=torch.long)
        neigh_tier = neigh_tier.to(device=device, dtype=torch.long)
        time_feat = time_feat.to(device=device, dtype=torch.float32)

        B, M = neigh.shape
        mask = neigh >= 0

        # Gather positions.
        cur_yx = node_yx[cur]  # (B,2)
        dest_yx = node_yx[dest]  # (B,2)
        # For padded neighbors, clamp to 0 to allow gather (will be masked out).
        neigh_safe = torch.clamp(neigh, min=0)
        neigh_yx = node_yx[neigh_safe]  # (B,M,2)

        # Geometry features (normalized).
        cur_yx_n = self._norm_xy(cur_yx).unsqueeze(1).expand(B, M, 2)
        dest_yx_n = self._norm_xy(dest_yx).unsqueeze(1).expand(B, M, 2)
        neigh_yx_n = self._norm_xy(neigh_yx)
        duv = self._norm_xy(neigh_yx - cur_yx.unsqueeze(1))
        dvd = self._norm_xy(dest_yx.unsqueeze(1) - neigh_yx)
        geom = torch.cat([cur_yx_n, dest_yx_n, neigh_yx_n, duv, dvd], dim=-1)  # (B,M,10)

        # Time features.
        tf = time_feat.unsqueeze(1).expand(B, M, 5)

        # Edge tier embedding (pad -> 0, masked anyway).
        tier = torch.clamp(neigh_tier, min=0, max=3)
        te = self.edge_tier_emb(tier)  # (B,M,E)

        x = torch.cat([geom, tf, te], dim=-1)  # (B,M,D)
        logits = self.mlp(x).squeeze(-1)  # (B,M)
        # Mask out invalid neighbors.
        logits = logits.masked_fill(~mask, -1e9)
        return logits, mask

