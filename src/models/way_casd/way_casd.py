from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.casd.perceiver import PerceiverCfg, PerceiverCompressor
from src.models.way_casd.conditions import ConditionEncoderCfg
from src.models.way_casd.way_decoder import WayDecoder, WayDecoderCfg
from src.models.way_casd.way_encoder import WayEncoder, WayFeatureTensors


@dataclass(frozen=True)
class WayCASDAECfg:
    d_model: int = 256
    n_latent: int = 64
    n_heads: int = 8
    dropout: float = 0.1
    max_candidates: int = 32
    max_len: int = 128
    coord_scale: float = 1024.0
    # See WayDecoderCfg.use_dest_dist (for loading older checkpoints).
    decoder_use_dest_dist: bool = True
    # Cross-attention decoder (key improvement for latent utilization)
    decoder_use_cross_attn: bool = True
    decoder_n_cross_heads: int = 4
    # Query enrichment (optional)
    decoder_use_step_emb: bool = False
    decoder_use_dest_query: bool = False
    decoder_use_dir_query: bool = False
    # Candidate-aware cross-attention: each candidate queries z_enc separately.
    decoder_use_cand_query: bool = False
    # Past context: encode past-K path with small Transformer (key fix for exposure bias)
    decoder_use_past_context: bool = False
    decoder_past_k: int = 8
    decoder_past_n_layers: int = 2
    decoder_past_n_heads: int = 4


class WayCASDAutoEncoder(nn.Module):
    """
    Way-CASD Step A: deterministic autoencoder for way sequences.

    Encoder:
      way_ids -> feature-based token embeddings -> Perceiver compressor -> latent tokens
    Decoder:
      AR candidate-set scorer with adjacency mask (provided as candidates)
    """

    def __init__(
        self,
        *,
        cfg: WayCASDAECfg,
        way_features: WayFeatureTensors,
        way_adj_ptr,
        way_adj_idx,
        n_route_cities: int = 4,
        n_highway_types: int = 16,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self.way_enc = WayEncoder(
            features=way_features,
            d_model=int(cfg.d_model),
            n_highway_types=int(n_highway_types),
            coord_scale=float(cfg.coord_scale),
        )
        self.compress = PerceiverCompressor(
            PerceiverCfg(
                d_model=int(cfg.d_model),
                n_latent=int(cfg.n_latent),
                n_heads=int(cfg.n_heads),
                dropout=float(cfg.dropout),
            )
        )
        self.decoder = WayDecoder(
            cfg=WayDecoderCfg(
                d_model=int(cfg.d_model),
                hidden_dim=int(cfg.d_model),
                max_candidates=int(cfg.max_candidates),
                dropout=float(cfg.dropout),
                max_len=int(cfg.max_len),
                use_dest_dist=bool(cfg.decoder_use_dest_dist),
                use_cross_attn=bool(cfg.decoder_use_cross_attn),
                n_cross_heads=int(cfg.decoder_n_cross_heads),
                use_step_emb=bool(cfg.decoder_use_step_emb),
                use_dest_query=bool(cfg.decoder_use_dest_query),
                use_dir_query=bool(cfg.decoder_use_dir_query),
                use_cand_query=bool(cfg.decoder_use_cand_query),
                use_past_context=bool(cfg.decoder_use_past_context),
                past_k=int(cfg.decoder_past_k),
                past_n_layers=int(cfg.decoder_past_n_layers),
                past_n_heads=int(cfg.decoder_past_n_heads),
            ),
            cond_cfg=ConditionEncoderCfg(
                d_model=int(cfg.d_model),
                n_route_cities=int(n_route_cities),
                coord_scale=float(cfg.coord_scale),
            ),
            way_adj_ptr=way_adj_ptr,
            way_adj_idx=way_adj_idx,
        )

    def encode(self, way_seq_pad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tok, mask = self.way_enc(way_seq_pad)
        z = self.compress(tok, mask=mask)
        return z, mask

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        way_seq_pad = batch["way_seq_pad"]
        route_cond = batch["route_cond"]
        trans = batch["trans"]
        target_idx = trans["target_idx"].to(dtype=torch.long)

        z, _mask = self.encode(way_seq_pad)
        logits = self.decoder.score_candidates(way_embedder=self.way_enc, latent_tokens=z, route_cond=route_cond, trans=trans)
        loss = F.cross_entropy(logits, target_idx, reduction="mean")

        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            acc = (pred == target_idx).float().mean() if target_idx.numel() else torch.tensor(0.0, device=loss.device)
            stats = {"loss": float(loss.item()), "acc": float(acc.item()), "n_trans": float(int(target_idx.numel()))}
        return loss, stats
