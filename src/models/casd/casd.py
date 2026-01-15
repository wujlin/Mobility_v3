from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.casd.conditions import ConditionEncoderCfg
from src.models.casd.perceiver import PerceiverCfg, PerceiverCompressor
from src.models.casd.segment_decoder import SegmentDecoder, SegmentDecoderCfg
from src.models.casd.segment_encoder import SegmentEncoder, SegmentFeatureTensors


@dataclass(frozen=True)
class CASDAECfg:
    d_model: int = 256
    n_latent: int = 128
    n_heads: int = 8
    dropout: float = 0.1
    max_candidates: int = 16
    max_len: int = 640
    coord_scale: float = 1024.0


class CASDAutoEncoder(nn.Module):
    """
    CASD Step A: deterministic autoencoder for segment sequences.

    Encoder:
      seg_ids -> feature-based token embeddings -> Perceiver compressor -> latent tokens
    Decoder:
      AR candidate-set scorer with adjacency mask (provided as candidates)
    """

    def __init__(
        self,
        *,
        cfg: CASDAECfg,
        seg_features: SegmentFeatureTensors,
        seg_v,
        seg_succ_ptr,
        seg_succ_idx,
        node_seg_ptr,
        node_seg_idx,
        n_route_cities: int = 4,
        n_corridor_types: int = 4,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self.seg_enc = SegmentEncoder(
            features=seg_features,
            d_model=int(cfg.d_model),
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
        self.decoder = SegmentDecoder(
            cfg=SegmentDecoderCfg(
                d_model=int(cfg.d_model),
                hidden_dim=int(cfg.d_model),
                max_candidates=int(cfg.max_candidates),
                dropout=float(cfg.dropout),
                max_len=int(cfg.max_len),
            ),
            cond_cfg=ConditionEncoderCfg(
                d_model=int(cfg.d_model),
                n_route_cities=int(n_route_cities),
                n_corridor_types=int(n_corridor_types),
                coord_scale=float(cfg.coord_scale),
            ),
            seg_v=seg_v,
            seg_succ_ptr=seg_succ_ptr,
            seg_succ_idx=seg_succ_idx,
            node_seg_ptr=node_seg_ptr,
            node_seg_idx=node_seg_idx,
        )

    def encode(self, seg_seq_pad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tok, mask = self.seg_enc(seg_seq_pad)
        z = self.compress(tok, mask=mask)
        return z, mask

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        seg_seq_pad = batch["seg_seq_pad"]
        route_cond = batch["route_cond"]
        trans = batch["trans"]
        target_idx = trans["target_idx"].to(dtype=torch.long)

        z, _mask = self.encode(seg_seq_pad)
        logits = self.decoder.score_candidates(seg_embedder=self.seg_enc, latent_tokens=z, route_cond=route_cond, trans=trans)
        loss = F.cross_entropy(logits, target_idx, reduction="mean")

        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            acc = (pred == target_idx).float().mean()
            stats = {"loss": float(loss.item()), "acc": float(acc.item()), "n_trans": float(int(target_idx.numel()))}
        return loss, stats

