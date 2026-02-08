from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class WayValueFnCfg:
    d_model: int = 256
    hidden_dim: int = 256
    dropout: float = 0.1
    use_z_mean: bool = True
    use_cond_emb: bool = True
    use_dest_dist: bool = True


class WayValueFn(nn.Module):
    """
    A lightweight value function for lookahead scoring:

      V(cur_way, z, dest) -> logit(prob_reach_dest)

    This module is intentionally simple: it consumes embeddings (not IDs),
    so it can be used with feature-based WayEncoder (no ID embedding).
    """

    def __init__(self, *, cfg: WayValueFnCfg) -> None:
        super().__init__()
        self.cfg = cfg
        d = int(cfg.d_model)
        h = int(cfg.hidden_dim)

        self.cur_proj = nn.Linear(d, h)
        self.z_proj = nn.Linear(d, h) if bool(cfg.use_z_mean) else None
        self.cond_proj = nn.Linear(d, h) if bool(cfg.use_cond_emb) else None

        in_dim = int(h)
        if bool(cfg.use_z_mean):
            in_dim += int(h)
        if bool(cfg.use_cond_emb):
            in_dim += int(h)
        if bool(cfg.use_dest_dist):
            in_dim += 1

        self.mlp = nn.Sequential(
            nn.Linear(int(in_dim), int(h)),
            nn.SiLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(h), 1),
        )

    def forward(
        self,
        *,
        cur_emb: torch.Tensor,  # (..., d_model)
        z_mean: Optional[torch.Tensor] = None,  # (..., d_model)
        cond_emb: Optional[torch.Tensor] = None,  # (..., d_model)
        dest_dist: Optional[torch.Tensor] = None,  # (..., 1)
    ) -> torch.Tensor:
        x = [self.cur_proj(cur_emb)]
        if bool(self.cfg.use_z_mean):
            if z_mean is None:
                raise ValueError("WayValueFn requires z_mean (cfg.use_z_mean=True).")
            x.append(self.z_proj(z_mean) if self.z_proj is not None else z_mean)
        if bool(self.cfg.use_cond_emb):
            if cond_emb is None:
                raise ValueError("WayValueFn requires cond_emb (cfg.use_cond_emb=True).")
            x.append(self.cond_proj(cond_emb) if self.cond_proj is not None else cond_emb)
        if bool(self.cfg.use_dest_dist):
            if dest_dist is None:
                raise ValueError("WayValueFn requires dest_dist (cfg.use_dest_dist=True).")
            # Expect dest_dist to be broadcastable to cur_emb[..., 0]
            # Typical: cur_emb is (T,C,D) while dest_dist is (T,C) -> unsqueeze to (T,C,1).
            if dest_dist.ndim == (cur_emb.ndim - 1):
                dest_dist = dest_dist.unsqueeze(-1)
            elif dest_dist.ndim != cur_emb.ndim:
                raise ValueError(f"dest_dist.ndim={dest_dist.ndim} not compatible with cur_emb.ndim={cur_emb.ndim}.")
            x.append(dest_dist.to(dtype=cur_emb.dtype))
        h = torch.cat(x, dim=-1)
        return self.mlp(h).squeeze(-1)
