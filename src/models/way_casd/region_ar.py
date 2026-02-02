from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.way_casd.conditions import ConditionEncoder, ConditionEncoderCfg


@dataclass(frozen=True)
class RegionARCfg:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    max_len: int = 16
    n_regions: int = 154
    n_route_cities: int = 2
    coord_scale: float = 1024.0


class RegionEmbedding(nn.Module):
    """
    Region embedding = learned ID + city embedding + projected static features.

    We keep it simple (sum, then LayerNorm) so downstream layers always see d_model.
    """

    def __init__(
        self,
        *,
        n_regions: int,
        d_model: int,
        n_route_cities: int,
        region_city: torch.Tensor,  # (R,) long
        region_static: torch.Tensor,  # (R,S) float
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_regions = int(n_regions)
        self.d_model = int(d_model)

        self.id_emb = nn.Embedding(int(n_regions), int(d_model))
        self.city_emb = nn.Embedding(int(n_route_cities), int(d_model))
        self.static_proj = nn.Linear(int(region_static.shape[1]), int(d_model), bias=False)
        self.ln = nn.LayerNorm(int(d_model))
        self.drop = nn.Dropout(float(dropout))

        self.register_buffer("region_city", region_city.to(dtype=torch.long), persistent=False)
        self.register_buffer("region_static", region_static.to(dtype=torch.float32), persistent=False)

    def forward(self, region_ids: torch.Tensor) -> torch.Tensor:
        rid = region_ids.to(dtype=torch.long)
        rid = torch.clamp(rid, 0, int(self.n_regions) - 1)
        h = self.id_emb(rid)

        city = self.region_city[rid]
        city = torch.clamp(city, 0, self.city_emb.num_embeddings - 1)
        h = h + self.city_emb(city)

        sf = self.region_static[rid]
        h = h + self.static_proj(sf)
        return self.drop(self.ln(h))


class _DecoderLayer(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(int(d_model), int(n_heads), dropout=float(dropout), batch_first=True)
        self.cross_attn = nn.MultiheadAttention(int(d_model), int(n_heads), dropout=float(dropout), batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(int(d_model), 4 * int(d_model)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * int(d_model), int(d_model)),
        )
        self.ln1 = nn.LayerNorm(int(d_model))
        self.ln2 = nn.LayerNorm(int(d_model))
        self.ln3 = nn.LayerNorm(int(d_model))
        self.drop = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,  # (B,T,D)
        *,
        memory: torch.Tensor,  # (B,M,D)
        self_attn_mask: Optional[torch.Tensor],  # (T,T) bool
        key_padding_mask: Optional[torch.Tensor],  # (B,T) bool
    ) -> torch.Tensor:
        # Self-attn (causal)
        h, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=self_attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.ln1(x + self.drop(h))

        # Cross-attn to OD context
        h, _ = self.cross_attn(
            x,
            memory,
            memory,
            need_weights=False,
        )
        x = self.ln2(x + self.drop(h))

        # FFN
        h = self.ff(x)
        x = self.ln3(x + self.drop(h))
        return x


class RegionARModel(nn.Module):
    """
    Region-level autoregressive model:
      p(r_{t+1} | r_{<=t}, O/D region, (start_pos,dest_pos,time,city))

    Key choice (per PI): output is a full-vocab softmax over ALL regions (no candidate mask).
    """

    def __init__(
        self,
        *,
        cfg: RegionARCfg,
        region_city: torch.Tensor,  # (R,)
        region_static: torch.Tensor,  # (R,S)
        region_adj: Optional[torch.Tensor] = None,  # (R,R) bool, optional for diagnostics
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_regions = int(cfg.n_regions)

        self.region_emb = RegionEmbedding(
            n_regions=int(cfg.n_regions),
            d_model=int(cfg.d_model),
            n_route_cities=int(cfg.n_route_cities),
            region_city=region_city,
            region_static=region_static,
            dropout=float(cfg.dropout),
        )
        self.pos_emb = nn.Embedding(int(cfg.max_len), int(cfg.d_model))
        self.cond_enc = ConditionEncoder(
            ConditionEncoderCfg(
                d_model=int(cfg.d_model),
                n_route_cities=int(cfg.n_route_cities),
                coord_scale=float(cfg.coord_scale),
            )
        )
        self.layers = nn.ModuleList(
            [_DecoderLayer(d_model=int(cfg.d_model), n_heads=int(cfg.n_heads), dropout=float(cfg.dropout)) for _ in range(int(cfg.n_layers))]
        )
        self.out = nn.Linear(int(cfg.d_model), int(cfg.n_regions))

        if region_adj is not None:
            self.register_buffer("region_adj", region_adj.to(dtype=torch.bool), persistent=False)
        else:
            self.region_adj = None  # type: ignore[assignment]

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # True = masked (not allowed). Upper triangle strictly above diagonal.
        return torch.triu(torch.ones((int(T), int(T)), device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        *,
        region_seq_in: torch.Tensor,  # (B,T) long, -1 padded
        o_region: torch.Tensor,  # (B,) long
        d_region: torch.Tensor,  # (B,) long
        route_cond: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        device = region_seq_in.device
        B, T = region_seq_in.shape

        # Key padding: True for pads
        key_padding = (region_seq_in < 0)
        # Clamp pads to 0 to allow embedding lookup (masked anyway).
        seq_ids = torch.clamp(region_seq_in, min=0)
        x = self.region_emb(seq_ids)

        # Pos emb
        pos = torch.arange(int(T), device=device, dtype=torch.long).clamp(max=int(self.cfg.max_len) - 1)
        x = x + self.pos_emb(pos)[None, :, :]

        # OD context tokens: [cond, o_region, d_region]
        cond = self.cond_enc(
            start_pos=route_cond["start_pos"],
            dest_pos=route_cond["dest_pos"],
            hour=route_cond["hour"],
            dow=route_cond["dow"],
            route_city=route_cond["route_city"],
        )  # (B,D)
        o = self.region_emb(o_region)  # (B,D)
        d = self.region_emb(d_region)  # (B,D)
        memory = torch.stack([cond, o, d], dim=1)  # (B,3,D)

        attn_mask = self._causal_mask(T, device=device)
        for layer in self.layers:
            x = layer(x, memory=memory, self_attn_mask=attn_mask, key_padding_mask=key_padding)

        return self.out(x)  # (B,T,R)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        seq_pad = batch["region_seq_pad"].to(dtype=torch.long)
        route_cond = batch["route_cond"]
        o_region = batch["o_region"].to(dtype=torch.long)
        d_region = batch["d_region"].to(dtype=torch.long)

        if seq_pad.size(1) <= 1:
            loss = torch.tensor(0.0, device=seq_pad.device)
            return loss, {"loss": 0.0, "acc": 0.0, "invalid_rate": float("nan"), "n_tokens": 0.0}

        x_in = seq_pad[:, :-1]
        tgt = seq_pad[:, 1:]
        mask = tgt >= 0

        logits = self.forward(region_seq_in=x_in, o_region=o_region, d_region=d_region, route_cond=route_cond)  # (B,T-1,R)
        B, T, R = logits.shape

        flat_logits = logits.reshape(B * T, R)
        flat_tgt = torch.clamp(tgt.reshape(B * T), min=0)
        flat_mask = mask.reshape(B * T).to(dtype=torch.float32)

        per = F.cross_entropy(flat_logits, flat_tgt, reduction="none")  # (B*T,)
        loss = (per * flat_mask).sum() / torch.clamp_min(flat_mask.sum(), 1.0)

        with torch.no_grad():
            pred = torch.argmax(flat_logits, dim=-1)  # (B*T,)
            acc = (((pred == flat_tgt).to(dtype=torch.float32) * flat_mask).sum() / torch.clamp_min(flat_mask.sum(), 1.0)).item()

            invalid_rate = float("nan")
            if getattr(self, "region_adj", None) is not None:
                prev = torch.clamp(x_in.reshape(B * T), min=0)
                valid = self.region_adj[prev, pred].to(dtype=torch.float32)
                invalid = (1.0 - valid) * flat_mask
                invalid_rate = float(invalid.sum().item() / float(torch.clamp_min(flat_mask.sum(), 1.0).item()))

        stats = {"loss": float(loss.item()), "acc": float(acc), "invalid_rate": float(invalid_rate), "n_tokens": float(flat_mask.sum().item())}
        return loss, stats

