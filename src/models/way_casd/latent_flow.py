from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.way_casd.conditions import ConditionEncoder, ConditionEncoderCfg


@dataclass(frozen=True)
class LatentFlowCfg:
    d_model: int = 256
    n_latent: int = 64
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    noise_sigma: float = 1.0
    solver_steps: int = 20
    # Condition injection into latent tokens.
    # - "add": broadcast-add a single condition vector to all latent tokens (baseline behavior).
    # - "xattn": cross-attend latent tokens to condition tokens (stronger conditioning).
    cond_inject: str = "add"  # {"add","xattn"}
    # Optional: provide a region_seq (coarse corridor guidance) as extra condition.
    use_region_seq: bool = False
    n_regions: int = 154
    region_max_len: int = 16


class LatentFlowMatching(nn.Module):
    """
    Rectified flow / flow matching in latent token space.
    """

    def __init__(self, *, cfg: LatentFlowCfg, cond_cfg: ConditionEncoderCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cond_enc = ConditionEncoder(cond_cfg)

        d_model = int(cfg.d_model)
        cond_inject = str(cfg.cond_inject)
        if cond_inject not in {"add", "xattn"}:
            raise ValueError(f"Unknown cond_inject={cond_inject!r} (expected 'add' or 'xattn').")
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

        self.register_buffer("rf_noise_sigma", torch.tensor(float(cfg.noise_sigma), dtype=torch.float32))

        # Optional: region_seq conditioning (coarse corridor guidance).
        self.use_region_seq = bool(cfg.use_region_seq)
        self.n_regions = int(cfg.n_regions)
        self.region_max_len = int(cfg.region_max_len)
        if self.use_region_seq:
            self.region_emb = nn.Embedding(int(self.n_regions), int(d_model))
            self.region_pos_emb = nn.Embedding(int(self.region_max_len), int(d_model))
            self.region_ln = nn.LayerNorm(int(d_model))
        else:
            self.region_emb = None
            self.region_pos_emb = None
            self.region_ln = None

        # Optional: condition cross-attention injection.
        self.cond_inject = cond_inject
        if str(self.cond_inject) == "xattn":
            self.cond_xattn = nn.MultiheadAttention(int(d_model), int(cfg.n_heads), dropout=float(cfg.dropout), batch_first=True)
            self.cond_xattn_ln = nn.LayerNorm(int(d_model))
        else:
            self.cond_xattn = None
            self.cond_xattn_ln = None

    def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(dtype=torch.float32)
        ang = t * (2.0 * 3.141592653589793)
        feat = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.time_mlp(feat)

    def _encode_region_tokens(self, *, region_seq_pad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          region_seq_pad: (B,S) long, padded with -1
        Returns:
          tok: (B,S,D)
          key_padding: (B,S) bool, True for pads
        """
        if not self.use_region_seq or self.region_emb is None or self.region_pos_emb is None or self.region_ln is None:
            raise RuntimeError("_encode_region_tokens called but use_region_seq is disabled.")
        seq = region_seq_pad.to(dtype=torch.long)
        key_padding = (seq < 0)
        seq = torch.clamp(seq, 0, int(self.n_regions) - 1)
        tok = self.region_emb(seq)
        S = int(tok.shape[1])
        pos = torch.arange(S, device=tok.device, dtype=torch.long).clamp(max=int(self.region_max_len) - 1)
        tok = tok + self.region_pos_emb(pos)[None, :, :]
        tok = self.region_ln(tok)
        return tok, key_padding

    def _apply_condition(
        self,
        *,
        zt: torch.Tensor,  # (B,L,D)
        t: torch.Tensor,  # (B,)
        route_cond: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        device = zt.device
        B = int(zt.shape[0])

        cond0 = self.cond_enc(
            start_pos=route_cond["start_pos"].to(device=device),
            dest_pos=route_cond["dest_pos"].to(device=device),
            hour=route_cond["hour"].to(device=device),
            dow=route_cond["dow"].to(device=device),
            route_city=route_cond["route_city"].to(device=device),
        )  # (B,D)
        time_emb = self._time_emb(t.to(device=device))  # (B,D)

        region_tok: Optional[torch.Tensor] = None
        region_pad: Optional[torch.Tensor] = None
        if self.use_region_seq:
            if "region_seq_pad" not in route_cond:
                raise KeyError("Flow requires route_cond['region_seq_pad'] when cfg.use_region_seq=True.")
            region_tok, region_pad = self._encode_region_tokens(region_seq_pad=route_cond["region_seq_pad"].to(device=device))

        if str(self.cond_inject) == "add":
            cond_vec = cond0
            if region_tok is not None and region_pad is not None:
                valid = (~region_pad).to(dtype=region_tok.dtype)  # (B,S)
                denom = valid.sum(dim=1).clamp(min=1.0)  # (B,)
                mean = (region_tok * valid[:, :, None]).sum(dim=1) / denom[:, None]
                cond_vec = cond_vec + mean
            return zt + cond_vec[:, None, :] + time_emb[:, None, :]

        # Cross-attend latent tokens to condition tokens.
        if self.cond_xattn is None or self.cond_xattn_ln is None:
            raise RuntimeError("cfg.cond_inject='xattn' but cond_xattn modules are missing.")

        cond_tokens = cond0[:, None, :]  # (B,1,D)
        key_padding = None
        if region_tok is not None and region_pad is not None:
            cond_tokens = torch.cat([cond_tokens, region_tok], dim=1)  # (B,1+S,D)
            key_padding = torch.cat(
                [torch.zeros((B, 1), device=device, dtype=torch.bool), region_pad.to(device=device, dtype=torch.bool)],
                dim=1,
            )

        h, _ = self.cond_xattn(
            zt,
            cond_tokens,
            cond_tokens,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        x = self.cond_xattn_ln(zt + h)
        return x + time_emb[:, None, :]

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

        x = self._apply_condition(zt=zt, t=t, route_cond=route_cond)
        v_pred = self.out_ln(self.net(x))
        loss = ((v_pred - v_target) ** 2).mean()
        return loss, {"loss": float(loss.item())}

    @torch.no_grad()
    def sample(
        self,
        *,
        route_cond: Dict[str, torch.Tensor],
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

        for i in range(steps):
            t = torch.full((B,), (float(i) + 0.5) * dt, device=device, dtype=torch.float32)
            v = self._v(z, t, route_cond)
            z = z + dt * v

        return z

    def _v(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        route_cond: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = self._apply_condition(zt=zt, t=t, route_cond=route_cond)
        return self.out_ln(self.net(x))
