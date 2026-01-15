from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PerceiverCfg:
    d_model: int = 256
    n_latent: int = 128
    n_heads: int = 8
    dropout: float = 0.1


class PerceiverCompressor(nn.Module):
    """
    Perceiver-style compression: variable-length tokens -> fixed-length latent tokens.

    Inputs:
      tokens: (B,K,d_model)
      mask:   (B,K) bool, True where valid token
    Outputs:
      latent: (B,L,d_model)
    """

    def __init__(self, cfg: PerceiverCfg) -> None:
        super().__init__()
        self.cfg = cfg
        d_model = int(cfg.d_model)
        n_latent = int(cfg.n_latent)
        self.latent_queries = nn.Parameter(torch.randn(n_latent, d_model) / (d_model**0.5))
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=int(cfg.n_heads),
            dropout=float(cfg.dropout),
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, tokens: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = int(tokens.shape[0])
        q = self.latent_queries.unsqueeze(0).expand(B, -1, -1)
        q = self.ln_q(q)
        kv = self.ln_kv(tokens)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.to(dtype=torch.bool)
        latent, _ = self.cross_attn(query=q, key=kv, value=kv, key_padding_mask=key_padding_mask, need_weights=False)
        latent = latent + self.ff(latent)
        return latent

