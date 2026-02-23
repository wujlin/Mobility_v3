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
    # Candidate contrast feature in scorer: include (cand_h - mean_cand_h).
    decoder_use_cand_contrast: bool = False
    # Past context: encode past-K path with small Transformer (key fix for exposure bias)
    decoder_use_past_context: bool = False
    decoder_past_k: int = 8
    decoder_past_n_layers: int = 2
    decoder_past_n_heads: int = 4
    # E8 (optional): multi-scale latent, reserving the last S latent tokens
    # as segment summaries (mean-pooled from encoder tokens).
    segment_size: int = 10
    segment_n_latent: int = 0  # 0=disable; >0=overwrite last S latent tokens with segment tokens
    # SIB: Stochastic Information Bottleneck — force decoder to rely on latent
    latent_noise_std: float = 0.0  # Gaussian noise σ injected into z_enc (0=disable)
    drop_dest_dist_p: float = 0.0  # Prob of zeroing dest_dist bypass per batch (0=disable)
    drop_past_context_p: float = 0.0  # Prob of dropping past_context bypass per batch (0=disable)
    # Optional β-VAE bottleneck on latent tokens.
    # When vae_dim>0, z_tokens -> (mu,logvar) in R^vae_dim -> reparameterize -> project back to tokens.
    vae_dim: int = 0
    vae_beta: float = 0.0


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
        self.segment_size = int(cfg.segment_size)
        self.segment_n_latent = int(cfg.segment_n_latent)
        if self.segment_n_latent > 0:
            if self.segment_size <= 0:
                raise ValueError("segment_n_latent>0 requires cfg.segment_size>0")
            if self.segment_n_latent > int(cfg.n_latent):
                raise ValueError("segment_n_latent must be <= n_latent (we overwrite the last S latent tokens).")
            self.segment_pos_emb = nn.Embedding(int(self.segment_n_latent), int(cfg.d_model))
            self.segment_ln = nn.LayerNorm(int(cfg.d_model))
        else:
            self.segment_pos_emb = None
            self.segment_ln = None

        # Optional β-VAE bottleneck over latent tokens.
        self.vae_dim = int(cfg.vae_dim)
        if self.vae_dim > 0:
            flat_dim = int(cfg.n_latent) * int(cfg.d_model)
            self.vae_to_stats = nn.Linear(flat_dim, int(self.vae_dim) * 2)
            self.vae_from_latent = nn.Linear(int(self.vae_dim), flat_dim)
            self.vae_logvar_clip = 10.0
        else:
            self.vae_to_stats = None
            self.vae_from_latent = None
            self.vae_logvar_clip = 10.0
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
                use_cand_contrast=bool(cfg.decoder_use_cand_contrast),
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

    def _apply_vae_bottleneck(self, z_tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply optional β-VAE bottleneck and return reconstructed latent tokens.
        """
        if int(self.vae_dim) <= 0 or self.vae_to_stats is None or self.vae_from_latent is None:
            return z_tokens, {}
        B, L, D = z_tokens.shape
        flat = z_tokens.reshape(int(B), int(L) * int(D))
        stats = self.vae_to_stats(flat)
        mu, logvar = torch.chunk(stats, 2, dim=-1)
        logvar = torch.clamp(logvar, min=-float(self.vae_logvar_clip), max=float(self.vae_logvar_clip))
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_vec = mu + eps * std
        else:
            z_vec = mu
        flat_rec = self.vae_from_latent(z_vec)
        z_rec = flat_rec.view(int(B), int(L), int(D))
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2.0) - logvar.exp(), dim=-1).mean()
        aux = {
            "mu": mu,
            "logvar": logvar,
            "z_vec": z_vec,
            "kl": kl,
        }
        return z_rec, aux

    def encode_with_stats(self, way_seq_pad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        tok, mask = self.way_enc(way_seq_pad)
        z = self.compress(tok, mask=mask)

        # Optional: overwrite the last S latent tokens with per-segment summaries.
        if self.segment_n_latent > 0 and self.segment_pos_emb is not None and self.segment_ln is not None:
            B, K, D = tok.shape
            S = int(self.segment_n_latent)
            L = int(z.shape[1])
            if S > L:
                raise RuntimeError(f"segment_n_latent={S} > n_latent={L} (invalid cfg).")
            seg = tok.new_zeros((B, S, D))
            pos = torch.arange(int(K), device=tok.device, dtype=torch.long)[None, :]  # (1,K)
            seg_id = (pos // int(self.segment_size)).clamp(max=S - 1)  # (1,K)
            valid = mask.to(dtype=tok.dtype)  # (B,K)
            for s in range(S):
                m = valid * (seg_id == int(s)).to(dtype=tok.dtype)  # (B,K)
                denom = m.sum(dim=1).clamp(min=1.0)  # (B,)
                seg[:, int(s), :] = (tok * m[:, :, None]).sum(dim=1) / denom[:, None]

            seg_pos = torch.arange(S, device=tok.device, dtype=torch.long)
            seg = seg + self.segment_pos_emb(seg_pos)[None, :, :]
            seg = self.segment_ln(seg)
            # Keep length fixed: [global_latents, segment_latents]
            z = torch.cat([z[:, : L - S, :], seg], dim=1)
        z, aux = self._apply_vae_bottleneck(z)
        return z, mask, aux

    def encode(self, way_seq_pad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, mask, _ = self.encode_with_stats(way_seq_pad)
        return z, mask

    def vae_latent_to_tokens(self, z_vec: torch.Tensor) -> torch.Tensor:
        """
        Project VAE latent vector(s) back to token latent shape.
        Accepts z_vec as (B,D) or (B,1,D); returns (B,n_latent,d_model).
        """
        if int(self.vae_dim) <= 0 or self.vae_from_latent is None:
            raise RuntimeError("vae_latent_to_tokens called but cfg.vae_dim<=0.")
        if z_vec.ndim == 3:
            if int(z_vec.shape[1]) != 1:
                raise RuntimeError(f"expected z_vec shape (B,1,D) for 3D input, got {tuple(z_vec.shape)}")
            z_flat_in = z_vec[:, 0, :]
        elif z_vec.ndim == 2:
            z_flat_in = z_vec
        else:
            raise RuntimeError(f"expected z_vec ndim 2 or 3, got ndim={int(z_vec.ndim)}")
        if int(z_flat_in.shape[-1]) != int(self.vae_dim):
            raise RuntimeError(f"vae latent dim mismatch: got {int(z_flat_in.shape[-1])}, expected {int(self.vae_dim)}")
        flat = self.vae_from_latent(z_flat_in)
        B = int(z_flat_in.shape[0])
        return flat.view(B, int(self.cfg.n_latent), int(self.cfg.d_model))

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        current_noise_std: float = 0.0,
        current_vae_beta: float | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        way_seq_pad = batch["way_seq_pad"]
        route_cond = batch["route_cond"]
        trans = batch["trans"]
        target_idx = trans["target_idx"].to(dtype=torch.long)

        z, _mask, aux = self.encode_with_stats(way_seq_pad)

        # SIB: inject Gaussian noise into latent during training
        noise_std = current_noise_std if current_noise_std > 0 else float(self.cfg.latent_noise_std)
        if self.training and noise_std > 0:
            z = z + torch.randn_like(z) * noise_std

        # SIB: stochastic bypass dropout (batch-level, same flag for all transitions in batch)
        drop_dd = False
        drop_pc = False
        if self.training:
            if self.cfg.drop_dest_dist_p > 0 and torch.rand(1).item() < self.cfg.drop_dest_dist_p:
                drop_dd = True
            if self.cfg.drop_past_context_p > 0 and torch.rand(1).item() < self.cfg.drop_past_context_p:
                drop_pc = True

        logits = self.decoder.score_candidates(
            way_embedder=self.way_enc, latent_tokens=z, route_cond=route_cond, trans=trans,
            drop_dest_dist=drop_dd, drop_past_context=drop_pc,
        )
        ce_loss = F.cross_entropy(logits, target_idx, reduction="mean")
        beta = float(self.cfg.vae_beta) if current_vae_beta is None else float(current_vae_beta)
        beta = max(0.0, beta)
        if "kl" in aux and beta > 0.0:
            loss = ce_loss + beta * aux["kl"]
            kl_loss_item = float(aux["kl"].item())
        else:
            loss = ce_loss
            kl_loss_item = float(aux["kl"].item()) if "kl" in aux else 0.0

        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            acc = (pred == target_idx).float().mean() if target_idx.numel() else torch.tensor(0.0, device=loss.device)
            stats = {
                "loss": float(loss.item()),
                "ce_loss": float(ce_loss.item()),
                "kl_loss": float(kl_loss_item),
                "vae_beta": float(beta),
                "acc": float(acc.item()),
                "n_trans": float(int(target_idx.numel())),
            }
        return loss, stats
