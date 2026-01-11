from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GridCrossAttnConfig:
    in_channels: int
    act_dim: int
    model_dim: int
    num_heads: int
    diff_steps: int
    weight: float


class GridCrossAttentionControlMid(nn.Module):
    """
    Scheme-A: Cross-attention conditioning (ControlNet-like) for 1D UNet diffusion.

    We compute a mid-level control tensor:
      query   = Conv1d(x_t) + timestep embedding
      key/val = Linear(flatten(grid_patch))
      control_mid = CrossAttn(query, key, val)

    The caller should pass `control_mid` into UNet1D via `unet_kwargs_fn`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        act_dim: int,
        model_dim: int,
        num_heads: int = 4,
        diff_steps: int = 50,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        a = int(act_dim)
        d = int(model_dim)
        h = int(num_heads)
        tmax = int(diff_steps)
        w = float(weight)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if a <= 0:
            raise ValueError("act_dim must be > 0")
        if d <= 0:
            raise ValueError("model_dim must be > 0")
        if h <= 0 or (d % h) != 0:
            raise ValueError("num_heads must be > 0 and divide model_dim")
        if tmax <= 0:
            raise ValueError("diff_steps must be > 0")
        if not torch.isfinite(torch.tensor(w)) or w < 0.0:
            raise ValueError("weight must be finite and >= 0")

        self.cfg = GridCrossAttnConfig(in_channels=c_in, act_dim=a, model_dim=d, num_heads=h, diff_steps=tmax, weight=w)

        self.q_proj = nn.Conv1d(a, d, kernel_size=1)
        self.kv_proj = nn.Linear(c_in, d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.t_embed = nn.Embedding(tmax, d)

        # Cache for 2D sinusoidal position embeddings (keyed by patch size and device/dtype).
        self._pos_cache: Dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

        self.record_attn: bool = False
        self.last_attn: Optional[torch.Tensor] = None  # (B,H,L,N_tokens) if record_attn

    @staticmethod
    def _sincos_1d(pos: torch.Tensor, dim: int) -> torch.Tensor:
        """
        1D sinusoidal position embedding.
          pos: (N,) float tensor
          returns: (N, dim)
        """
        dim = int(dim)
        if dim <= 0:
            return torch.zeros((pos.shape[0], 0), device=pos.device, dtype=pos.dtype)
        # Use even dimensions for sin/cos pairs, pad if needed.
        dim2 = (dim // 2) * 2
        if dim2 <= 0:
            out = torch.zeros((pos.shape[0], dim), device=pos.device, dtype=pos.dtype)
            return out
        idx = torch.arange(0, dim2, 2, device=pos.device, dtype=pos.dtype)  # (dim2/2,)
        inv_freq = torch.pow(torch.tensor(10000.0, device=pos.device, dtype=pos.dtype), -idx / float(dim2))
        sinusoid_inp = pos[:, None] * inv_freq[None, :]  # (N, dim2/2)
        emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=1)  # (N, dim2)
        if dim2 < dim:
            pad = torch.zeros((pos.shape[0], dim - dim2), device=pos.device, dtype=pos.dtype)
            emb = torch.cat([emb, pad], dim=1)
        return emb

    def _pos_emb_2d(self, *, s: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        2D sinusoidal position embedding for a square SxS patch.
          returns: (S*S, D) where D = model_dim
        """
        key = (int(s), device, dtype)
        cached = self._pos_cache.get(key)
        if cached is not None:
            return cached

        s = int(s)
        d = int(self.cfg.model_dim)
        # Compute in float32 for stability, then cast.
        y = torch.arange(s, device=device, dtype=torch.float32)
        x = torch.arange(s, device=device, dtype=torch.float32)
        try:
            yy, xx = torch.meshgrid(y, x, indexing="ij")
        except TypeError:  # torch<1.10
            yy, xx = torch.meshgrid(y, x)
        pos_y = yy.reshape(-1)
        pos_x = xx.reshape(-1)

        d_half = d // 2
        emb_y = self._sincos_1d(pos_y, d_half)
        emb_x = self._sincos_1d(pos_x, d - d_half)
        emb = torch.cat([emb_y, emb_x], dim=1)  # (S*S, D)
        emb = emb.to(dtype=dtype)

        self._pos_cache[key] = emb
        return emb

    def forward(
        self,
        x_t: torch.Tensor,  # (B,act_dim,L)
        timesteps: torch.Tensor,  # (B,)
        *,
        grid_patch: torch.Tensor,  # (B,C,S,S)
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be (B,act_dim,L), got {tuple(x_t.shape)}")
        b = int(x_t.shape[0])
        if int(x_t.shape[1]) != int(self.cfg.act_dim):
            raise ValueError(f"act_dim mismatch: expected {int(self.cfg.act_dim)} got {int(x_t.shape[1])}")
        if grid_patch.ndim != 4 or int(grid_patch.shape[0]) != b or int(grid_patch.shape[1]) != int(self.cfg.in_channels):
            raise ValueError(f"grid_patch must be (B,{int(self.cfg.in_channels)},S,S), got {tuple(grid_patch.shape)}")

        # Query: (B,act_dim,L) -> (B,L,model_dim)
        q = self.q_proj(x_t.to(dtype=torch.float32)).transpose(1, 2)  # (B,L,D)
        t = timesteps.to(device=q.device, dtype=torch.long).clamp(min=0, max=int(self.cfg.diff_steps) - 1)
        q = q + self.t_embed(t)[:, None, :]

        # KV tokens: (B,C,S,S) -> (B,S*S,D)
        tok = grid_patch.to(dtype=torch.float32).flatten(2).transpose(1, 2)  # (B,N,C)
        kv = self.kv_proj(tok)  # (B,N,D)
        s = int(grid_patch.shape[2])
        if int(grid_patch.shape[3]) != s:
            raise ValueError(f"grid_patch must be square (B,C,S,S), got {tuple(grid_patch.shape)}")
        kv = kv + self._pos_emb_2d(s=s, device=kv.device, dtype=kv.dtype)[None, :, :]

        attn_out, attn_w = self.attn(q, kv, kv, need_weights=(bool(need_weights) or bool(self.record_attn)), average_attn_weights=False)
        # attn_out: (B,L,D) -> control_mid: (B,D,L)
        control_mid = attn_out.transpose(1, 2).contiguous() * float(self.cfg.weight)

        if bool(self.record_attn):
            self.last_attn = attn_w.detach()
        return control_mid, (attn_w.detach() if bool(need_weights) else None)
