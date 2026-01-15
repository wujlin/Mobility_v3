from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.casd.conditions import ConditionEncoder, ConditionEncoderCfg


@dataclass(frozen=True)
class LatentFlowCfg:
    d_model: int = 256
    n_latent: int = 128
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    time_scale: float = 1000.0
    noise_sigma: float = 1.0
    solver_steps: int = 20
    cfg_drop_prob: float = 0.1


class LatentFlowMatching(nn.Module):
    """
    Rectified flow / flow matching in latent token space.

    This module learns v_theta(z_t, t | cond) where:
      z_0 ~ N(0, sigma^2 I), z_1 is target latent from a frozen encoder.
      z_t = (1-t) z_0 + t z_1, v_target = z_1 - z_0
    """

    def __init__(self, *, cfg: LatentFlowCfg, cond_cfg: ConditionEncoderCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cond_enc = ConditionEncoder(cond_cfg)

        d_model = int(cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(cfg.n_heads),
            dim_feedforward=d_model * 4,
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(layer, num_layers=int(cfg.n_layers))
        self.time_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.out_ln = nn.LayerNorm(d_model)

        self.register_buffer("rf_time_scale", torch.tensor(float(cfg.time_scale), dtype=torch.float32))
        self.register_buffer("rf_noise_sigma", torch.tensor(float(cfg.noise_sigma), dtype=torch.float32))

    def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(dtype=torch.float32)
        ang = t * (2.0 * 3.141592653589793)
        feat = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.time_mlp(feat)

    def compute_loss(
        self,
        *,
        z1: torch.Tensor,  # (B,L,d_model)
        route_cond: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B = int(z1.shape[0])
        device = z1.device
        sigma = self.rf_noise_sigma.to(device=device, dtype=z1.dtype)
        z0 = torch.randn_like(z1) * sigma
        t = torch.rand((B,), device=device, dtype=z1.dtype)
        t_ = t[:, None, None]
        zt = (1.0 - t_) * z0 + t_ * z1
        v_target = z1 - z0

        corridor_type = route_cond.get("corridor_type", None)
        if corridor_type is not None:
            corridor_type = corridor_type.to(device=device)
            if self.training and float(self.cfg.cfg_drop_prob) > 0.0:
                drop = torch.rand((B,), device=device) < float(self.cfg.cfg_drop_prob)
                corridor_type = corridor_type.clone()
                corridor_type[drop] = -1

        cond_emb = self.cond_enc(
            start_pos=route_cond["start_pos"],
            dest_pos=route_cond["dest_pos"],
            hour=route_cond["hour"],
            route_city=route_cond["route_city"],
            corridor_type=corridor_type,
        )
        time_emb = self._time_emb(t)
        x = zt + cond_emb[:, None, :] + time_emb[:, None, :]
        v_pred = self.out_ln(self.net(x))
        loss = ((v_pred - v_target) ** 2).mean()

        stats = {"loss": float(loss.item())}
        return loss, stats

    @torch.no_grad()
    def sample(
        self,
        *,
        route_cond: Dict[str, torch.Tensor],
        cfg_scale: float = 1.5,
        solver_steps: Optional[int] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        B = int(route_cond["start_pos"].shape[0])
        L = int(self.cfg.n_latent)
        d = int(self.cfg.d_model)
        steps = int(solver_steps) if solver_steps is not None else int(self.cfg.solver_steps)
        steps = max(1, steps)
        dt = 1.0 / float(steps)

        sigma = self.rf_noise_sigma.to(device=device, dtype=torch.float32)
        z = torch.randn((B, L, d), device=device, dtype=torch.float32) * sigma

        corr = route_cond.get("corridor_type", None)
        corr_uncond = None
        if corr is not None:
            corr_uncond = corr.new_full(corr.shape, -1)

        for i in range(steps):
            t = torch.full((B,), (float(i) + 0.5) * dt, device=device, dtype=torch.float32)
            v_u = self._v(z, t, route_cond, corridor_type=corr_uncond)
            v_c = self._v(z, t, route_cond, corridor_type=corr)
            v = v_u + float(cfg_scale) * (v_c - v_u)
            z = z + dt * v

        return z

    def _v(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        route_cond: Dict[str, torch.Tensor],
        *,
        corridor_type: Optional[torch.Tensor],
    ) -> torch.Tensor:
        device = zt.device
        cond_emb = self.cond_enc(
            start_pos=route_cond["start_pos"].to(device=device),
            dest_pos=route_cond["dest_pos"].to(device=device),
            hour=route_cond["hour"].to(device=device),
            route_city=route_cond["route_city"].to(device=device),
            corridor_type=(corridor_type.to(device=device) if corridor_type is not None else None),
        )
        time_emb = self._time_emb(t.to(device=device))
        x = zt + cond_emb[:, None, :] + time_emb[:, None, :]
        return self.out_ln(self.net(x))

