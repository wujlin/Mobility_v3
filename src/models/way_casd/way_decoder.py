from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.way_casd.conditions import ConditionEncoder, ConditionEncoderCfg


@dataclass(frozen=True)
class WayDecoderCfg:
    d_model: int = 256
    hidden_dim: int = 256
    max_candidates: int = 32
    dropout: float = 0.1
    max_len: int = 128
    # Candidate contrast feature in scorer: include (cand_h - mean_cand_h) so each candidate
    # can be scored relative to others, not only independently.
    use_cand_contrast: bool = False
    # Cross-attention for querying latent tokens
    use_cross_attn: bool = True
    n_cross_heads: int = 4
    # Backward compatibility:
    use_dest_dist: bool = True
    # Query enrichment (for early-step disambiguation)
    use_step_emb: bool = False
    use_dest_query: bool = False
    # Query-time direction hint (use candidate direction statistics to query z)
    use_dir_query: bool = False
    # Candidate-aware cross-attention: let each candidate query z_enc separately so
    # latent information can directly participate in candidate ranking.
    use_cand_query: bool = False
    # Past context: encode past-K path with small Transformer
    use_past_context: bool = False
    past_k: int = 8  # Number of past steps to include
    past_n_layers: int = 2  # Transformer layers for past encoding
    past_n_heads: int = 4  # Attention heads in past encoder


class PastContextEncoder(nn.Module):
    """
    Encodes past-K way embeddings using a small Transformer.
    
    Input: past_emb (T, K, d_model), past_mask (T, K)
    Output: context (T, d_model) - aggregated past context
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_k: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.max_k = int(max_k)

        # Learnable position embeddings for past positions (relative: -K, ..., -1)
        self.pos_emb = nn.Embedding(int(max_k), int(d_model))

        # Small Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_heads),
            dim_feedforward=int(d_model) * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
        self.out_ln = nn.LayerNorm(int(d_model))

    def forward(
        self,
        past_emb: torch.Tensor,  # (T, K, d_model)
        past_mask: torch.Tensor,  # (T, K) bool, True=valid
    ) -> torch.Tensor:
        """
        Args:
            past_emb: (T, K, d_model) past way embeddings
            past_mask: (T, K) True where past_way is valid
        Returns:
            context: (T, d_model) aggregated past context
        """
        T, K, d = past_emb.shape
        device = past_emb.device

        # Add position embeddings (positions 0..K-1 represent past-K..past-1)
        pos_idx = torch.arange(K, device=device, dtype=torch.long)
        pos_idx = torch.clamp(pos_idx, 0, self.max_k - 1)
        pos = self.pos_emb(pos_idx)  # (K, d_model)
        x = past_emb + pos.unsqueeze(0)  # (T, K, d_model)

        # Transformer expects key_padding_mask: True = ignore
        key_padding_mask = ~past_mask  # (T, K)

        # Handle all-masked case (first step has no history)
        all_masked = key_padding_mask.all(dim=-1)  # (T,)
        if all_masked.any():
            # For all-masked rows, unmask the first position to avoid NaN
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (T, K, d_model)

        # Aggregate: take last valid position (most recent past)
        # For simplicity, use masked mean
        mask_expand = past_mask.unsqueeze(-1).float()  # (T, K, 1)
        x_masked = x * mask_expand
        sum_x = x_masked.sum(dim=1)  # (T, d_model)
        count = mask_expand.sum(dim=1).clamp(min=1.0)  # (T, 1)
        context = self.out_ln(sum_x / count)  # (T, d_model)

        return context


class WayDecoder(nn.Module):
    """
    Constrained AR decoder over way IDs using candidate-set scoring.
    
    Key improvement: Uses cross-attention to query latent_tokens at each step,
    instead of just using mean(latent_tokens). This allows the decoder to
    dynamically extract relevant information based on current position.

    Stop criterion: current_way == dest_way (no EOS token).
    """

    def __init__(
        self,
        *,
        cfg: WayDecoderCfg,
        cond_cfg: ConditionEncoderCfg,
        way_adj_ptr,
        way_adj_idx,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.cond_enc = ConditionEncoder(cond_cfg)
        self.coord_scale = float(getattr(cond_cfg, "coord_scale", 0.0))

        self.use_step_emb = bool(cfg.use_step_emb)
        self.use_dest_query = bool(cfg.use_dest_query)
        self.use_dir_query = bool(cfg.use_dir_query)
        self.use_cand_query = bool(cfg.use_cand_query)
        self.use_cand_contrast = bool(cfg.use_cand_contrast)

        self.register_buffer("way_adj_ptr", torch.as_tensor(way_adj_ptr, dtype=torch.long), persistent=False)
        self.register_buffer("way_adj_idx", torch.as_tensor(way_adj_idx, dtype=torch.long), persistent=False)

        d_model = int(cfg.d_model)
        hidden = int(cfg.hidden_dim)

        if self.use_step_emb:
            # Step index is clamped to [0, max_len], so this is safe across decode lengths.
            self.step_emb = nn.Embedding(int(cfg.max_len) + 1, d_model)
        else:
            self.step_emb = None

        if self.use_dest_query:
            self.dest_proj = nn.Linear(2, d_model)
        else:
            self.dest_proj = None

        # Query-time direction hint: project mean candidate direction (dy, dx) into d_model.
        if self.use_dir_query:
            self.dir_query_proj = nn.Linear(2, d_model)
        else:
            self.dir_query_proj = None

        # Candidate-aware query: project candidate embedding into d_model and add to base query.
        if self.use_cand_query:
            self.cand_query_proj = nn.Linear(d_model, d_model)
        else:
            self.cand_query_proj = None

        # Cross-attention: query=cur_way, key/value=latent_tokens
        self.use_cross_attn = bool(cfg.use_cross_attn)
        if self.use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=int(cfg.n_cross_heads),
                dropout=float(cfg.dropout),
                batch_first=True,
            )
            self.cross_ln = nn.LayerNorm(d_model)
            # Context MLP: cond_emb + cross_attn_output
            self.ctx_mlp = nn.Sequential(
                nn.Linear(d_model * 2, hidden),
                nn.SiLU(),
                nn.Dropout(float(cfg.dropout)),
                nn.Linear(hidden, hidden),
            )
        else:
            # Fallback: original mean-pooling approach
            self.ctx_mlp = nn.Sequential(
                nn.Linear(d_model * 2, hidden),
                nn.SiLU(),
                nn.Dropout(float(cfg.dropout)),
                nn.Linear(hidden, hidden),
            )

        # Past context encoder (Transformer over past-K path)
        self.use_past_context = bool(cfg.use_past_context)
        self.past_k = int(cfg.past_k)
        if self.use_past_context:
            self.past_encoder = PastContextEncoder(
                d_model=int(d_model),
                n_layers=int(cfg.past_n_layers),
                n_heads=int(cfg.past_n_heads),
                dropout=float(cfg.dropout),
                max_k=int(cfg.past_k),
            )
        else:
            self.past_encoder = None
        
        self.cur_proj = nn.Linear(d_model, hidden)
        self.cand_proj = nn.Linear(d_model, hidden)
        in_dim = int(hidden * (4 if self.use_cand_contrast else 3)) + (1 if bool(cfg.use_dest_dist) else 0)
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(hidden, 1),
        )

    def _slice_csr(self, ptr: torch.Tensor, idx: torch.Tensor, i: int) -> torch.Tensor:
        s = int(ptr[i].item())
        e = int(ptr[i + 1].item())
        if e <= s:
            return idx.new_zeros((0,), dtype=torch.long)
        return idx[s:e].to(dtype=torch.long)

    def get_succ_candidates(self, way_id: int) -> torch.Tensor:
        return self._slice_csr(self.way_adj_ptr, self.way_adj_idx, int(way_id))

    def is_dest_reached(self, way_id: int, dest_way: int) -> bool:
        return int(way_id) == int(dest_way)

    def _select_decode_candidates(
        self,
        *,
        way_embedder: nn.Module,
        cand_full: torch.Tensor,  # (C_full,)
        dest_pos: torch.Tensor,  # (1,2) or (2,)
        dest_way: int,
        max_candidates: Optional[int],
        candidate_policy: str,
        include_dest_if_successor: bool,
    ) -> torch.Tensor:
        """
        Select a decode-time candidate subset from full successors.

        max_candidates:
          - None: use self.cfg.max_candidates
          - <=0: use all successors (no truncation)
          - >0: truncate to that many

        candidate_policy:
          - "first": keep the first-K successors (status quo)
          - "destdist": keep the K successors closest to dest_pos (in normalized coord space)

        include_dest_if_successor:
          If dest_way is a direct successor but gets truncated out, force-include it
          (mirror training-time _ensure_target behavior for the last hop).
        """
        if int(cand_full.numel()) == 0:
            return cand_full

        if max_candidates is None:
            k = int(self.cfg.max_candidates)
        else:
            k = int(max_candidates)

        # <=0 means "use all"
        if k <= 0 or int(cand_full.numel()) <= k:
            return cand_full

        policy = str(candidate_policy).lower().strip()
        if policy == "destdist":
            # Compute candidate-to-destination distance in the same normalized coord space
            # as WayEncoder._lookup and score_candidates().
            dest = dest_pos.to(dtype=torch.float32)
            if dest.ndim == 2:
                dest = dest[0]
            coord_scale = float(getattr(way_embedder, "coord_scale", self.coord_scale))
            if coord_scale > 0:
                dest = dest / coord_scale
            try:
                cand_geom, _tier, _hw = way_embedder._lookup(cand_full)
                cand_center = cand_geom[..., :2].to(dtype=torch.float32)
                dist = torch.norm(dest[None, :] - cand_center, dim=-1)
                order = torch.argsort(dist, dim=0)
                cand = cand_full[order[:k]]
            except Exception:
                cand = cand_full[:k]
        else:
            cand = cand_full[:k]

        if bool(include_dest_if_successor):
            dw = int(dest_way)
            if dw >= 0:
                try:
                    in_full = bool((cand_full == dw).any().item())
                    in_sel = bool((cand == dw).any().item())
                except Exception:
                    in_full, in_sel = False, False
                if in_full and (not in_sel):
                    if int(cand.numel()) < k:
                        cand = torch.cat([cand, cand.new_tensor([dw])], dim=0)
                    else:
                        cand = cand.clone()
                        cand[-1] = int(dw)

        return cand

    def _compute_context(
        self,
        *,
        way_embedder: nn.Module,
        latent_tokens: torch.Tensor,  # (B, L, d_model)
        cond_emb: torch.Tensor,  # (B, d_model)
        cur_way: torch.Tensor,  # (T,)
        cand_way: Optional[torch.Tensor] = None,  # (T, C) candidate way IDs (optional)
        cand_mask: Optional[torch.Tensor] = None,  # (T, C) bool mask for candidates (optional)
        cur_emb: Optional[torch.Tensor] = None,  # (T, d_model) optional cache
        cand_emb: Optional[torch.Tensor] = None,  # (T, C, d_model) optional cache
        route_idx: torch.Tensor,  # (T,)
        step: torch.Tensor,  # (T,)
        dest_pos: torch.Tensor,  # (B,2)
        past_way: Optional[torch.Tensor] = None,  # (T, K) past way IDs, -1 for padding
        past_mask: Optional[torch.Tensor] = None,  # (T, K) bool, True=valid
        return_attn_weights: bool = False,
    ) -> torch.Tensor:
        """
        Compute context vector for each transition.
        
        If use_cross_attn: query latent_tokens with cur_way embedding.
        Otherwise: use mean-pooled latent.
        
        If use_past_context: incorporate past_way history via Transformer encoder.
        
        Returns:
            - ctx (T, hidden_dim) when candidate-agnostic
            - ctx (T, C, hidden_dim) when use_cand_query=True (candidate-aware)
        """
        B = int(latent_tokens.shape[0])
        T = int(cur_way.shape[0])
        device = latent_tokens.device
        
        # Get current way embeddings (optional cache)
        if cur_emb is None:
            cur_emb2, _ = way_embedder(cur_way[:, None])  # (T, 1, d_model)
            cur_emb = cur_emb2[:, 0, :]  # (T, d_model)

        # Enrich query (optional)
        query_vec = cur_emb
        if self.use_step_emb and self.step_emb is not None:
            s = step.to(dtype=torch.long)
            # Clamp step to avoid OOB when max_decode_len > cfg.max_len
            s = torch.clamp(s, 0, self.step_emb.num_embeddings - 1)
            query_vec = query_vec + self.step_emb(s)

        if self.use_dest_query and self.dest_proj is not None:
            # Project destination position in the same normalized coord space
            # used by ConditionEncoder / WayEncoder.
            dest = dest_pos.to(dtype=torch.float32)
            coord_scale = float(getattr(way_embedder, "coord_scale", self.coord_scale))
            if coord_scale > 0:
                dest = dest / coord_scale
            dest_t = dest[route_idx]  # (T,2)
            query_vec = query_vec + self.dest_proj(dest_t)

        # Direction hint: summarize candidate directions at this decision step.
        # This is meant to disambiguate "which direction to query" from z_enc when
        # multiple successors share similar local features.
        if self.use_dir_query and self.dir_query_proj is not None and cand_way is not None:
            try:
                cand_geom, _tier, _hw = way_embedder._lookup(cand_way)  # (T,C,5)
                cand_dirs = cand_geom[..., 2:4].to(dtype=torch.float32)  # (T,C,2) (dy,dx)
                if cand_mask is not None:
                    m = cand_mask.to(dtype=torch.float32).unsqueeze(-1)  # (T,C,1)
                    denom = m.sum(dim=1).clamp(min=1.0)  # (T,1)
                    mean_dir = (cand_dirs * m).sum(dim=1) / denom  # (T,2)
                else:
                    mean_dir = cand_dirs.mean(dim=1)  # (T,2)
                query_vec = query_vec + self.dir_query_proj(mean_dir.to(device=device))
            except Exception:
                # Keep decode robust if lookup fails for any reason.
                pass

        # Past context: encode history using Transformer
        if self.use_past_context and self.past_encoder is not None:
            if past_way is not None and past_mask is not None:
                # Embed past ways
                past_emb, _ = way_embedder(past_way)  # (T, K, d_model)
                past_ctx = self.past_encoder(past_emb, past_mask)  # (T, d_model)
                query_vec = query_vec + past_ctx
            # If past_way is None (e.g., step 0), query_vec remains unchanged

        attn_weights_out: Optional[torch.Tensor] = None
        if self.use_cross_attn:
            # Gather latent tokens for each transition: (B, L, d) -> (T, L, d)
            lat_gathered = latent_tokens[route_idx]  # (T, L, d_model)
            cond_t = cond_emb[route_idx]  # (T, d_model)

            # Candidate-aware cross-attention (optional): each candidate queries z_enc separately.
            # This allows z_enc to contribute directly to candidate ranking, not just as a global bias.
            if (
                self.use_cand_query
                and self.cand_query_proj is not None
                and cand_way is not None
                and cand_mask is not None
            ):
                if cand_emb is None:
                    cand_emb2, _ = way_embedder(cand_way)  # (T,C,d_model)
                    cand_emb = cand_emb2
                T2, C = int(cand_way.shape[0]), int(cand_way.shape[1])
                if T2 != T:
                    raise ValueError(f"cand_way T mismatch: T={T} vs cand_way.shape[0]={T2}")

                # Compute candidate-aware cross-attn without exploding memory.
                #
                # NOTE: Avoid flattening valid (t,c) pairs and repeating keys/values per candidate.
                # That pattern scales as O((T*C)*L*d) and can easily OOM on long routes / large batches.
                # Here we treat each transition t as a "batch element" and candidates as the query sequence.
                cand_q = query_vec[:, None, :] + self.cand_query_proj(cand_emb)  # (T,C,d_model)
                valid = cand_mask.to(dtype=torch.bool)
                if not bool(valid.any().item()):
                    ctx0 = torch.zeros((T, C, int(self.cfg.hidden_dim)), dtype=query_vec.dtype, device=device)
                    if bool(return_attn_weights):
                        attn0 = torch.zeros((T, C, int(lat_gathered.shape[1])), dtype=torch.float32, device=device)
                        return ctx0, attn0
                    return ctx0

                # Zero-out invalid candidates to keep outputs stable under padding.
                cand_q = cand_q * valid[:, :, None].to(dtype=cand_q.dtype)

                # Query: (T,C,d), Key/Val: (T,L,d) -> attn_out: (T,C,d)
                attn_out, attn_w = self.cross_attn(
                    cand_q, lat_gathered, lat_gathered, need_weights=bool(return_attn_weights)
                )
                attn_vec = self.cross_ln(attn_out + cand_q)  # (T,C,d), residual
                ctx = self.ctx_mlp(torch.cat([cond_t[:, None, :].expand(T, C, -1), attn_vec], dim=-1))  # (T,C,hidden)
                ctx = ctx * valid[:, :, None].to(dtype=ctx.dtype)

                if bool(return_attn_weights) and isinstance(attn_w, torch.Tensor):
                    w = attn_w.to(dtype=torch.float32)
                    # Default: (T,C,L) when average_attn_weights=True.
                    # If (T,H,C,L), average over heads.
                    if w.ndim == 4:
                        w = w.mean(dim=1)
                    attn_weights_out = w
            else:
                # Candidate-independent context (original): (T,hidden)
                query = query_vec.unsqueeze(1)  # (T,1,d_model)
                attn_out, attn_w = self.cross_attn(
                    query, lat_gathered, lat_gathered, need_weights=bool(return_attn_weights)
                )  # (T,1,d_model)
                attn_out = self.cross_ln(attn_out[:, 0, :] + query_vec)  # (T,d_model), residual
                ctx = self.ctx_mlp(torch.cat([cond_t, attn_out], dim=-1))  # (T,hidden)
                if bool(return_attn_weights) and isinstance(attn_w, torch.Tensor):
                    w = attn_w.to(dtype=torch.float32)
                    if w.ndim == 3:
                        w = w[:, 0, :]  # (T,L)
                    attn_weights_out = w
        else:
            # Fallback: mean-pooled latent
            lat_vec = latent_tokens.mean(dim=1)  # (B, d_model)
            ctx_b = self.ctx_mlp(torch.cat([cond_emb, lat_vec], dim=-1))  # (B, hidden)
            ctx = ctx_b[route_idx]  # (T, hidden)
        
        if bool(return_attn_weights):
            return ctx, attn_weights_out
        return ctx

    def score_candidates(
        self,
        *,
        way_embedder: nn.Module,
        latent_tokens: torch.Tensor,  # (B,L,d_model)
        route_cond: Dict[str, torch.Tensor],
        trans: Dict[str, torch.Tensor],
        cond_emb: Optional[torch.Tensor] = None,  # (B,d_model) optional cache
    ) -> torch.Tensor:
        route_idx = trans["route_idx"].to(dtype=torch.long)
        cur_way = trans["cur_way"].to(dtype=torch.long)
        cand_way = trans["cand_way"].to(dtype=torch.long)
        cand_mask = trans["cand_mask"].to(dtype=torch.bool)
        step = trans.get("step", None)
        if step is None:
            step = torch.zeros_like(cur_way)
        step = step.to(dtype=torch.long)

        # Past context (optional)
        past_way = trans.get("past_way", None)
        past_way_mask = trans.get("past_mask", None)
        if past_way is not None:
            past_way = past_way.to(dtype=torch.long)
        if past_way_mask is not None:
            past_way_mask = past_way_mask.to(dtype=torch.bool)

        B = int(latent_tokens.shape[0])
        if int(route_idx.max().item()) >= B:
            raise ValueError(f"route_idx out of range: max={int(route_idx.max().item())} but B={B}")

        # Condition embedding (optionally cached per-route for decode speed)
        if cond_emb is None:
            cond_emb = self.cond_enc(
                start_pos=route_cond["start_pos"],
                dest_pos=route_cond["dest_pos"],
                hour=route_cond["hour"],
                dow=route_cond["dow"],
                route_city=route_cond["route_city"],
            )
        
        # Precompute embeddings (reused by context + scorer)
        cur_emb, _ = way_embedder(cur_way[:, None])
        cur_emb = cur_emb[:, 0, :]
        cand_emb, _ = way_embedder(cand_way)

        # Compute context with cross-attention (and past context if enabled)
        ctx_out = self._compute_context(
            way_embedder=way_embedder,
            latent_tokens=latent_tokens,
            cond_emb=cond_emb,
            cur_way=cur_way,
            cand_way=cand_way,
            cand_mask=cand_mask,
            cur_emb=cur_emb,
            cand_emb=cand_emb,
            route_idx=route_idx,
            step=step,
            dest_pos=route_cond["dest_pos"],
            past_way=past_way,
            past_mask=past_way_mask,
        )

        # Current way projection
        cur_h = self.cur_proj(cur_emb)

        # Candidate way embeddings
        cand_h = self.cand_proj(cand_emb)

        T, C = cand_way.shape
        if int(ctx_out.ndim) == 2:
            ctx_h = ctx_out[:, None, :].expand(T, C, -1)
        else:
            ctx_h = ctx_out
        cur_h2 = cur_h[:, None, :].expand(T, C, -1)

        diff_from_mean: Optional[torch.Tensor] = None
        if bool(self.use_cand_contrast):
            mask_f = cand_mask[:, :, None].to(dtype=cand_h.dtype)  # (T,C,1)
            denom = mask_f.sum(dim=1).clamp(min=1.0)  # (T,1)
            mean_cand = (cand_h * mask_f).sum(dim=1) / denom  # (T,hidden)
            diff_from_mean = (cand_h - mean_cand[:, None, :]) * mask_f  # (T,C,hidden)
        
        if bool(self.cfg.use_dest_dist):
            # Candidate-to-destination distance
            coord_scale = float(getattr(way_embedder, "coord_scale", self.coord_scale))
            dest = route_cond["dest_pos"][route_idx].to(dtype=torch.float32)
            if coord_scale > 0:
                dest = dest / coord_scale
            try:
                cand_geom, _tier, _hw = way_embedder._lookup(cand_way)
                cand_center = cand_geom[..., :2].to(dtype=torch.float32)
            except Exception:
                cand_center = torch.zeros((T, C, 2), dtype=torch.float32, device=dest.device)
            dist = torch.norm(dest[:, None, :] - cand_center, dim=-1, keepdim=True)
            if diff_from_mean is not None:
                x = torch.cat([ctx_h, cur_h2, cand_h, diff_from_mean, dist], dim=-1)
            else:
                x = torch.cat([ctx_h, cur_h2, cand_h, dist], dim=-1)
        else:
            if diff_from_mean is not None:
                x = torch.cat([ctx_h, cur_h2, cand_h, diff_from_mean], dim=-1)
            else:
                x = torch.cat([ctx_h, cur_h2, cand_h], dim=-1)
        
        logits = self.scorer(x).squeeze(-1)
        logits = logits.masked_fill(~cand_mask, float("-inf"))
        return logits

    @torch.no_grad()
    def beam_search(
        self,
        *,
        way_embedder: nn.Module,
        latent_tokens: torch.Tensor,  # (B,L,d_model)
        route_cond: Dict[str, torch.Tensor],
        start_way: torch.Tensor,  # (B,)
        dest_way: torch.Tensor,  # (B,)
        way_region: Optional[torch.Tensor] = None,  # (n_ways,) long, optional
        region_seq: Optional[List[List[int]]] = None,  # len=B, optional
        region_adj: Optional[torch.Tensor] = None,  # (R,R) bool, optional (required for relaxed mode)
        region_constraint_mode: str = "strict",
        region_constraint_fallback: str = "unconstrained",
        beam_size: int = 5,
        max_len: Optional[int] = None,
        max_candidates: Optional[int] = None,
        candidate_policy: str = "first",
        include_dest_if_successor: bool = False,
        guided_dest_alpha: float = 0.0,
    ) -> List[List[int]]:
        max_len = int(max_len) if max_len is not None else int(self.cfg.max_len)
        beam_size = max(1, int(beam_size))

        B = int(latent_tokens.shape[0])
        out: List[List[int]] = []
        device = latent_tokens.device

        use_region_constraint = (way_region is not None) and (region_seq is not None)
        if use_region_constraint and int(len(region_seq)) != int(B):
            raise ValueError(f"region_seq length mismatch: got {len(region_seq)}, expect {B}")
        mode = str(region_constraint_mode or "").strip().lower()
        if mode and mode not in {"strict", "relaxed"}:
            raise ValueError(f"unsupported region_constraint_mode: {region_constraint_mode!r}")
        fallback = str(region_constraint_fallback or "").strip().lower()
        if fallback and fallback not in {"unconstrained", "stop", "dest_region"}:
            raise ValueError(f"unsupported region_constraint_fallback: {region_constraint_fallback!r}")

        def _compress_consecutive_py(seq: List[int]) -> List[int]:
            out: List[int] = []
            last: Optional[int] = None
            for x in seq:
                xx = int(x)
                if last is None or xx != int(last):
                    out.append(xx)
                    last = xx
            return out

        def _prepare_region_seq(seq: List[int], *, start_region: int, dest_region: int) -> List[int]:
            s = [int(x) for x in seq if int(x) >= 0]
            s = _compress_consecutive_py(s)
            if not s:
                s = [int(start_region), int(dest_region)]
            if int(s[0]) != int(start_region):
                s = [int(start_region)] + s
            if int(s[-1]) != int(dest_region):
                s = s + [int(dest_region)]
            s = _compress_consecutive_py(s)
            return s

        for b in range(B):
            sw = int(start_way[b].item())
            dw = int(dest_way[b].item())
            route_cond_b = {k: v[b : b + 1].to(device=device) for k, v in route_cond.items()}
            cond_emb_b = self.cond_enc(
                start_pos=route_cond_b["start_pos"],
                dest_pos=route_cond_b["dest_pos"],
                hour=route_cond_b["hour"],
                dow=route_cond_b["dow"],
                route_city=route_cond_b["route_city"],
            )
            region_path: Optional[List[int]] = None
            if use_region_constraint and way_region is not None and region_seq is not None:
                sr = int(way_region[int(sw)].item())
                dr = int(way_region[int(dw)].item()) if int(dw) >= 0 else int(sr)
                region_path = _prepare_region_seq(list(region_seq[int(b)]), start_region=sr, dest_region=dr)
            beams: List[Tuple[List[int], float, int]] = [([sw], 0.0, 0)]  # (path, score, region_ptr)

            for _step in range(max_len):
                new_beams: List[Tuple[List[int], float, int]] = []
                all_finished = True
                for path, score, rptr in beams:
                    if path and self.is_dest_reached(path[-1], dw):
                        new_beams.append((path, score, int(rptr)))
                        continue
                    all_finished = False
                    cand_full = self.get_succ_candidates(path[-1])
                    if int(cand_full.numel()) == 0:
                        continue
                    cand = self._select_decode_candidates(
                        way_embedder=way_embedder,
                        cand_full=cand_full.to(device=device),
                        dest_pos=route_cond_b["dest_pos"],
                        dest_way=dw,
                        max_candidates=max_candidates,
                        candidate_policy=candidate_policy,
                        include_dest_if_successor=include_dest_if_successor,
                    )
                    if region_path is not None and way_region is not None:
                        if way_region.device != cand.device:
                            raise ValueError(f"way_region must be on same device as candidates: {way_region.device} vs {cand.device}")
                        if mode == "relaxed":
                            if region_adj is None:
                                raise ValueError("region_adj is required for region_constraint_mode='relaxed'")
                            if region_adj.device != cand.device:
                                raise ValueError(f"region_adj must be on same device as candidates: {region_adj.device} vs {cand.device}")
                        cur_way = int(path[-1])
                        cur_reg = int(way_region[cur_way].item())
                        rr = int(rptr)
                        while rr + 1 < int(len(region_path)) and int(cur_reg) == int(region_path[rr + 1]):
                            rr += 1
                        allow0 = int(region_path[rr])
                        allow1 = int(region_path[rr + 1]) if rr + 1 < int(len(region_path)) else None
                        cand_reg = way_region[cand]
                        m = cand_reg == int(allow0)
                        if allow1 is not None:
                            m = m | (cand_reg == int(allow1))
                        if mode == "relaxed" and region_adj is not None:
                            if int(allow0) >= 0:
                                m = m | region_adj[int(allow0), cand_reg]
                            if allow1 is not None and int(allow1) >= 0:
                                m = m | region_adj[int(allow1), cand_reg]
                        cand_f = cand[m]
                        # keep direct successor-to-dest if present
                        if int(dw) >= 0 and bool((cand == int(dw)).any().item()) and (not bool((cand_f == int(dw)).any().item())):
                            cand_f = torch.cat([cand_f, torch.tensor([int(dw)], dtype=cand.dtype, device=cand.device)], dim=0)
                        if int(cand_f.numel()) > 0:
                            cand = cand_f
                            rptr = rr
                        else:
                            if fallback == "stop":
                                continue
                            if fallback == "dest_region":
                                dest_reg = int(way_region[int(dw)].item()) if int(dw) >= 0 else -1
                                if dest_reg >= 0:
                                    cand_dest = cand[cand_reg == int(dest_reg)]
                                    if int(cand_dest.numel()) > 0:
                                        cand = cand_dest
                    C = int(cand.numel())
                    if C <= 0:
                        continue
                    cand_way = cand.view(1, C)
                    cand_mask = torch.ones((1, C), dtype=torch.bool, device=device)
                    step_idx = max(0, int(len(path) - 1))

                    # Build past_way for past context
                    past_way_tensor: Optional[torch.Tensor] = None
                    past_mask_tensor: Optional[torch.Tensor] = None
                    if self.use_past_context and len(path) > 0:
                        K = self.past_k
                        past_list = path[:-1][-K:] if len(path) > 1 else []
                        past_arr = [-1] * K
                        for i, w in enumerate(past_list):
                            offset = K - len(past_list)
                            past_arr[offset + i] = w
                        past_way_tensor = torch.tensor([past_arr], dtype=torch.long, device=device)
                        past_mask_tensor = (past_way_tensor >= 0)

                    trans = {
                        "route_idx": torch.tensor([0], dtype=torch.long, device=device),
                        "cur_way": torch.tensor([int(path[-1])], dtype=torch.long, device=device),
                        "cand_way": cand_way,
                        "cand_mask": cand_mask,
                        "step": torch.tensor([step_idx], dtype=torch.long, device=device),
                    }
                    if past_way_tensor is not None:
                        trans["past_way"] = past_way_tensor
                        trans["past_mask"] = past_mask_tensor

                    logits = self.score_candidates(
                        way_embedder=way_embedder,
                        latent_tokens=latent_tokens[b : b + 1],
                        route_cond=route_cond_b,
                        trans=trans,
                        cond_emb=cond_emb_b,
                    )[0]
                    alpha = float(guided_dest_alpha)
                    if abs(alpha) > 1e-12:
                        try:
                            coord_scale = float(getattr(way_embedder, "coord_scale", self.coord_scale))
                            dest = route_cond_b["dest_pos"].to(dtype=torch.float32)
                            if coord_scale > 0:
                                dest = dest / coord_scale
                            cand_geom, _tier, _hw = way_embedder._lookup(cand_way)
                            cand_center = cand_geom[..., :2].to(dtype=torch.float32)
                            dist = torch.norm(dest[:, None, :] - cand_center, dim=-1)  # (1,C)
                            logits = logits - alpha * dist[0]
                        except Exception:
                            pass
                    logp = F.log_softmax(logits, dim=-1)
                    topk = min(beam_size, int(logp.numel()))
                    vals, ids = torch.topk(logp, k=topk, dim=-1)
                    for lp, j in zip(vals.tolist(), ids.tolist()):
                        nxt = int(cand[j].item())
                        new_rptr = int(rptr)
                        if region_path is not None and way_region is not None:
                            nxt_reg = int(way_region[int(nxt)].item())
                            while new_rptr + 1 < int(len(region_path)) and int(nxt_reg) == int(region_path[new_rptr + 1]):
                                new_rptr += 1
                        new_beams.append((path + [nxt], score + float(lp), int(new_rptr)))

                if all_finished:
                    break
                new_beams.sort(key=lambda x: -x[1])
                beams = new_beams[:beam_size] if new_beams else beams

            out.append(beams[0][0] if beams else [sw])

        return out

    @torch.no_grad()
    def greedy_decode(
        self,
        *,
        way_embedder: nn.Module,
        latent_tokens: torch.Tensor,  # (B,L,d_model)
        route_cond: Dict[str, torch.Tensor],
        start_way: torch.Tensor,  # (B,)
        dest_way: torch.Tensor,  # (B,)
        way_region: Optional[torch.Tensor] = None,  # (n_ways,) long, optional
        region_seq: Optional[List[List[int]]] = None,  # len=B, optional
        region_adj: Optional[torch.Tensor] = None,  # (R,R) bool, optional (required for relaxed mode)
        region_constraint_mode: str = "strict",
        region_constraint_fallback: str = "unconstrained",
        max_len: Optional[int] = None,
        max_candidates: Optional[int] = None,
        candidate_policy: str = "first",
        include_dest_if_successor: bool = False,
        guided_dest_alpha: float = 0.0,
    ) -> List[List[int]]:
        max_len = int(max_len) if max_len is not None else int(self.cfg.max_len)

        B = int(latent_tokens.shape[0])
        out: List[List[int]] = []
        device = latent_tokens.device

        use_region_constraint = (way_region is not None) and (region_seq is not None)
        if use_region_constraint and int(len(region_seq)) != int(B):
            raise ValueError(f"region_seq length mismatch: got {len(region_seq)}, expect {B}")
        mode = str(region_constraint_mode or "").strip().lower()
        if mode and mode not in {"strict", "relaxed"}:
            raise ValueError(f"unsupported region_constraint_mode: {region_constraint_mode!r}")
        fallback = str(region_constraint_fallback or "").strip().lower()
        if fallback and fallback not in {"unconstrained", "stop", "dest_region"}:
            raise ValueError(f"unsupported region_constraint_fallback: {region_constraint_fallback!r}")

        def _compress_consecutive_py(seq: List[int]) -> List[int]:
            out: List[int] = []
            last: Optional[int] = None
            for x in seq:
                xx = int(x)
                if last is None or xx != int(last):
                    out.append(xx)
                    last = xx
            return out

        def _prepare_region_seq(seq: List[int], *, start_region: int, dest_region: int) -> List[int]:
            s = [int(x) for x in seq if int(x) >= 0]
            s = _compress_consecutive_py(s)
            if not s:
                s = [int(start_region), int(dest_region)]
            if int(s[0]) != int(start_region):
                s = [int(start_region)] + s
            if int(s[-1]) != int(dest_region):
                s = s + [int(dest_region)]
            s = _compress_consecutive_py(s)
            return s

        for b in range(B):
            sw = int(start_way[b].item())
            dw = int(dest_way[b].item())
            route_cond_b = {k: v[b : b + 1].to(device=device) for k, v in route_cond.items()}
            cond_emb_b = self.cond_enc(
                start_pos=route_cond_b["start_pos"],
                dest_pos=route_cond_b["dest_pos"],
                hour=route_cond_b["hour"],
                dow=route_cond_b["dow"],
                route_city=route_cond_b["route_city"],
            )

            region_path: Optional[List[int]] = None
            region_ptr = 0
            if use_region_constraint and way_region is not None and region_seq is not None:
                sr = int(way_region[int(sw)].item())
                dr = int(way_region[int(dw)].item()) if int(dw) >= 0 else int(sr)
                region_path = _prepare_region_seq(list(region_seq[int(b)]), start_region=sr, dest_region=dr)

            path: List[int] = [sw]
            for step_idx in range(max_len):
                if path and self.is_dest_reached(path[-1], dw):
                    break
                cand_full = self.get_succ_candidates(path[-1])
                if int(cand_full.numel()) == 0:
                    break
                cand = self._select_decode_candidates(
                    way_embedder=way_embedder,
                    cand_full=cand_full.to(device=device),
                    dest_pos=route_cond_b["dest_pos"],
                    dest_way=dw,
                    max_candidates=max_candidates,
                    candidate_policy=candidate_policy,
                    include_dest_if_successor=include_dest_if_successor,
                )
                if region_path is not None and way_region is not None:
                    if way_region.device != cand.device:
                        raise ValueError(f"way_region must be on same device as candidates: {way_region.device} vs {cand.device}")
                    if mode == "relaxed":
                        if region_adj is None:
                            raise ValueError("region_adj is required for region_constraint_mode='relaxed'")
                        if region_adj.device != cand.device:
                            raise ValueError(f"region_adj must be on same device as candidates: {region_adj.device} vs {cand.device}")
                    cur_way = int(path[-1])
                    cur_reg = int(way_region[cur_way].item())
                    while region_ptr + 1 < int(len(region_path)) and int(cur_reg) == int(region_path[region_ptr + 1]):
                        region_ptr += 1
                    allow0 = int(region_path[region_ptr])
                    allow1 = int(region_path[region_ptr + 1]) if region_ptr + 1 < int(len(region_path)) else None
                    cand_reg = way_region[cand]
                    m = cand_reg == int(allow0)
                    if allow1 is not None:
                        m = m | (cand_reg == int(allow1))
                    if mode == "relaxed" and region_adj is not None:
                        if int(allow0) >= 0:
                            m = m | region_adj[int(allow0), cand_reg]
                        if allow1 is not None and int(allow1) >= 0:
                            m = m | region_adj[int(allow1), cand_reg]
                    cand_f = cand[m]
                    if int(dw) >= 0 and bool((cand == int(dw)).any().item()) and (not bool((cand_f == int(dw)).any().item())):
                        cand_f = torch.cat([cand_f, torch.tensor([int(dw)], dtype=cand.dtype, device=cand.device)], dim=0)
                    if int(cand_f.numel()) > 0:
                        cand = cand_f
                    else:
                        if fallback == "stop":
                            break
                        if fallback == "dest_region":
                            dest_reg = int(way_region[int(dw)].item()) if int(dw) >= 0 else -1
                            if dest_reg >= 0:
                                cand_dest = cand[cand_reg == int(dest_reg)]
                                if int(cand_dest.numel()) > 0:
                                    cand = cand_dest
                C = int(cand.numel())
                if C <= 0:
                    break
                cand_way = cand.view(1, C)
                cand_mask = torch.ones((1, C), dtype=torch.bool, device=device)

                # Build past_way for past context (last K ways before current)
                past_way_tensor: Optional[torch.Tensor] = None
                past_mask_tensor: Optional[torch.Tensor] = None
                if self.use_past_context and len(path) > 0:
                    K = self.past_k
                    # path includes current position; we want past-K *before* current
                    # At step_idx, path has step_idx+1 elements (including start)
                    # past = path[max(0, len(path)-K):len(path)] but excluding current? 
                    # Actually we want the history leading up to current way.
                    # path[-1] is cur_way, so past is path[:-1][-K:]
                    past_list = path[:-1][-K:] if len(path) > 1 else []
                    past_arr = [-1] * K
                    for i, w in enumerate(past_list):
                        # Right-align: most recent at the end
                        offset = K - len(past_list)
                        past_arr[offset + i] = w
                    past_way_tensor = torch.tensor([past_arr], dtype=torch.long, device=device)  # (1, K)
                    past_mask_tensor = (past_way_tensor >= 0)  # (1, K)

                trans = {
                    "route_idx": torch.tensor([0], dtype=torch.long, device=device),
                    "cur_way": torch.tensor([int(path[-1])], dtype=torch.long, device=device),
                    "cand_way": cand_way,
                    "cand_mask": cand_mask,
                    "step": torch.tensor([int(step_idx)], dtype=torch.long, device=device),
                }
                if past_way_tensor is not None:
                    trans["past_way"] = past_way_tensor
                    trans["past_mask"] = past_mask_tensor

                logits = self.score_candidates(
                    way_embedder=way_embedder,
                    latent_tokens=latent_tokens[b : b + 1],
                    route_cond=route_cond_b,
                    trans=trans,
                    cond_emb=cond_emb_b,
                )[0]
                alpha = float(guided_dest_alpha)
                if abs(alpha) > 1e-12:
                    try:
                        coord_scale = float(getattr(way_embedder, "coord_scale", self.coord_scale))
                        dest = route_cond_b["dest_pos"].to(dtype=torch.float32)
                        if coord_scale > 0:
                            dest = dest / coord_scale
                        cand_geom, _tier, _hw = way_embedder._lookup(cand_way)
                        cand_center = cand_geom[..., :2].to(dtype=torch.float32)
                        dist = torch.norm(dest[:, None, :] - cand_center, dim=-1)  # (1,C)
                        logits = logits - alpha * dist[0]
                    except Exception:
                        pass
                j = int(torch.argmax(logits, dim=-1).item()) if int(logits.numel()) else 0
                path.append(int(cand[j].item()))

            out.append(path)
        return out

    @torch.no_grad()
    def greedy_decode_batched(
        self,
        *,
        way_embedder: nn.Module,
        latent_tokens: torch.Tensor,  # (B,L,d_model)
        route_cond: Dict[str, torch.Tensor],
        start_way: torch.Tensor,  # (B,)
        dest_way: torch.Tensor,  # (B,)
        way_region: Optional[torch.Tensor] = None,  # (n_ways,) long, optional (same device as latent_tokens)
        region_seq: Optional[List[List[int]]] = None,  # len=B, optional
        region_adj: Optional[torch.Tensor] = None,  # (R,R) bool, optional (required for relaxed mode)
        region_constraint_mode: str = "strict",
        region_constraint_fallback: str = "unconstrained",
        max_len: Optional[int] = None,
        max_candidates: Optional[int] = None,
        candidate_policy: str = "first",
        include_dest_if_successor: bool = False,
        guided_dest_alpha: float = 0.0,
        anti_loop_k: int = 0,
        anti_loop_penalty: float = 0.0,
        anti_loop_penalty_k: int = 4,
        value_fn: Optional[nn.Module] = None,
        value_beta: float = 0.0,
    ) -> List[List[int]]:
        """
        Batched greedy decoding for speed.

        Key design:
          - batch all active routes at each step into one score_candidates() forward
          - candidate selection (CSR slicing) remains per-route (cheap vs forward)
        """
        max_len = int(max_len) if max_len is not None else int(self.cfg.max_len)
        B = int(latent_tokens.shape[0])
        device = latent_tokens.device

        use_region_constraint = (way_region is not None) and (region_seq is not None)
        if use_region_constraint and int(len(region_seq)) != int(B):
            raise ValueError(f"region_seq length mismatch: got {len(region_seq)}, expect {B}")
        if use_region_constraint and way_region is not None and way_region.device != device:
            raise ValueError(f"way_region must be on same device as latent_tokens: {way_region.device} vs {device}")
        mode = str(region_constraint_mode or "").strip().lower()
        if mode and mode not in {"strict", "relaxed"}:
            raise ValueError(f"unsupported region_constraint_mode: {region_constraint_mode!r}")
        if use_region_constraint and mode == "relaxed":
            if region_adj is None:
                raise ValueError("region_adj is required for region_constraint_mode='relaxed'")
            if region_adj.device != device:
                raise ValueError(f"region_adj must be on same device as latent_tokens: {region_adj.device} vs {device}")
        fallback = str(region_constraint_fallback or "").strip().lower()
        if fallback and fallback not in {"unconstrained", "stop", "dest_region"}:
            raise ValueError(f"unsupported region_constraint_fallback: {region_constraint_fallback!r}")

        def _compress_consecutive_py(seq: List[int]) -> List[int]:
            out: List[int] = []
            last: Optional[int] = None
            for x in seq:
                xx = int(x)
                if last is None or xx != int(last):
                    out.append(xx)
                    last = xx
            return out

        def _prepare_region_seq(seq: List[int], *, start_region: int, dest_region: int) -> List[int]:
            s = [int(x) for x in seq if int(x) >= 0]
            s = _compress_consecutive_py(s)
            if not s:
                s = [int(start_region), int(dest_region)]
            if int(s[0]) != int(start_region):
                s = [int(start_region)] + s
            if int(s[-1]) != int(dest_region):
                s = s + [int(dest_region)]
            s = _compress_consecutive_py(s)
            return s

        # Precompute route-level conditioning once.
        cond_emb = self.cond_enc(
            start_pos=route_cond["start_pos"],
            dest_pos=route_cond["dest_pos"],
            hour=route_cond["hour"],
            dow=route_cond["dow"],
            route_city=route_cond["route_city"],
        )

        # Output paths stored on CPU as python lists (cheap).
        paths: List[List[int]] = [[int(x)] for x in start_way.to(dtype=torch.long).tolist()]

        cur_way = start_way.to(dtype=torch.long, device=device).clone()
        dw = dest_way.to(dtype=torch.long, device=device).clone()
        active = (cur_way != dw)

        # Region constraint state (per route).
        region_paths: Optional[List[List[int]]] = None
        region_ptr: Optional[List[int]] = None
        if use_region_constraint and way_region is not None and region_seq is not None:
            region_paths = []
            region_ptr = []
            for b in range(B):
                sw = int(cur_way[b].item())
                dw_b = int(dw[b].item())
                sr = int(way_region[int(sw)].item()) if 0 <= sw < int(way_region.numel()) else -1
                dr = int(way_region[int(dw_b)].item()) if 0 <= dw_b < int(way_region.numel()) else int(sr)
                region_paths.append(_prepare_region_seq(list(region_seq[int(b)]), start_region=sr, dest_region=dr))
                region_ptr.append(0)

        # Decode steps.
        for step_idx in range(int(max_len)):
            if not bool(active.any().item()):
                break

            active_ids = torch.nonzero(active, as_tuple=False).reshape(-1)
            # Build candidates per active route (python loop, cheap).
            cand_list: List[torch.Tensor] = []
            keep_ids: List[int] = []
            for bi in active_ids.tolist():
                cur = int(cur_way[int(bi)].item())
                cand_full = self.get_succ_candidates(int(cur))
                if int(cand_full.numel()) == 0:
                    active[int(bi)] = False
                    continue
                cand = self._select_decode_candidates(
                    way_embedder=way_embedder,
                    cand_full=cand_full.to(device=device),
                    dest_pos=route_cond["dest_pos"][int(bi) : int(bi) + 1],
                    dest_way=int(dw[int(bi)].item()),
                    max_candidates=max_candidates,
                    candidate_policy=candidate_policy,
                    include_dest_if_successor=include_dest_if_successor,
                )
                if int(cand.numel()) == 0:
                    active[int(bi)] = False
                    continue
                cand_list.append(cand.to(dtype=torch.long, device=device))
                keep_ids.append(int(bi))

            if not keep_ids:
                break

            keep = torch.tensor(keep_ids, dtype=torch.long, device=device)
            B2 = int(keep.numel())
            Cmax = int(max(int(c.numel()) for c in cand_list))

            cand_way = torch.zeros((B2, Cmax), dtype=torch.long, device=device)
            cand_mask = torch.zeros((B2, Cmax), dtype=torch.bool, device=device)
            for i, c in enumerate(cand_list):
                n = int(c.numel())
                cand_way[i, :n] = c
                cand_mask[i, :n] = True

            # Region constraint -> refine cand_mask (strict: allow current/next region).
            if region_paths is not None and region_ptr is not None and way_region is not None:
                allow0 = torch.full((B2,), -1, dtype=torch.long, device=device)
                allow1 = torch.full((B2,), -1, dtype=torch.long, device=device)
                for i, bi in enumerate(keep_ids):
                    cur = int(cur_way[int(bi)].item())
                    cur_reg = int(way_region[int(cur)].item())
                    rp = int(region_ptr[int(bi)])
                    path = region_paths[int(bi)]
                    while rp + 1 < int(len(path)) and int(cur_reg) == int(path[rp + 1]):
                        rp += 1
                    region_ptr[int(bi)] = int(rp)
                    allow0[i] = int(path[rp]) if rp < int(len(path)) else int(cur_reg)
                    allow1[i] = int(path[rp + 1]) if (rp + 1) < int(len(path)) else -1

                cand_reg = way_region[cand_way]  # (B2,Cmax)
                m = cand_mask & (cand_reg == allow0[:, None])
                has1 = (allow1 >= 0)[:, None]
                m = m | (cand_mask & has1 & (cand_reg == allow1[:, None]))
                if mode == "relaxed" and region_adj is not None:
                    allow0_safe = torch.where(allow0 >= 0, allow0, torch.zeros_like(allow0))
                    allow1_safe = torch.where(allow1 >= 0, allow1, torch.zeros_like(allow1))
                    neigh0 = region_adj[allow0_safe[:, None], cand_reg] & (allow0[:, None] >= 0)
                    neigh1 = region_adj[allow1_safe[:, None], cand_reg] & (allow1[:, None] >= 0)
                    m = m | (cand_mask & neigh0) | (cand_mask & neigh1)
                # Keep direct successor-to-dest if present (match non-batched behavior).
                dest = dw[keep]
                m = m | (cand_mask & (cand_way == dest[:, None]))

                # If empty after masking: fallback.
                row_has = m.any(dim=1)
                if fallback == "stop":
                    bad = torch.nonzero(~row_has, as_tuple=False).reshape(-1)
                    for i in bad.tolist():
                        active[int(keep_ids[int(i)])] = False

                    good = torch.nonzero(row_has, as_tuple=False).reshape(-1)
                    if int(good.numel()) == 0:
                        break

                    # Keep only rows that still have valid candidates.
                    keep_ids = [keep_ids[int(i)] for i in good.tolist()]
                    keep = keep[good]
                    cand_way = cand_way[good]
                    cand_mask = m[good]
                    B2 = int(keep.numel())
                elif fallback == "dest_region":
                    dest_reg = way_region[dw[keep]]  # (B2,)
                    has_dest_reg = (dest_reg >= 0)[:, None]
                    m_dest = cand_mask & has_dest_reg & (cand_reg == dest_reg[:, None])
                    row_has_dest = m_dest.any(dim=1)
                    m2 = torch.where(row_has[:, None], m, m_dest)
                    row_has2 = row_has | row_has_dest
                    cand_mask = torch.where(row_has2[:, None], m2, cand_mask)
                else:
                    cand_mask = torch.where(row_has[:, None], m, cand_mask)

            # Anti-loop (hard mask): exclude candidates that revisit any of the last K visited ways.
            # NOTE: avoid per-element `.item()` on GPU. We build a small CPU tensor of recent IDs then vectorize.
            k_hard = int(anti_loop_k)
            if k_hard > 0:
                before = cand_mask.clone()
                recent_cpu = torch.full((B2, k_hard), -1, dtype=torch.long)
                for i, bi in enumerate(keep_ids):
                    tail = paths[int(bi)][-k_hard:]
                    if tail:
                        recent_cpu[i, : int(len(tail))] = torch.as_tensor(tail, dtype=torch.long)
                recent = recent_cpu.to(device=device)
                hit = (cand_way.unsqueeze(-1) == recent[:, None, :])  # (B2,C,K)
                bad = (hit.any(dim=-1)) & cand_mask  # (B2,C)
                cand_mask = cand_mask & (~bad)
                empty = ~cand_mask.any(dim=1)
                if bool(empty.any().item()):
                    cand_mask[empty] = before[empty]

            # Past context tensors (only if enabled).
            past_way_tensor: Optional[torch.Tensor] = None
            past_mask_tensor: Optional[torch.Tensor] = None
            if bool(self.use_past_context):
                K = int(self.past_k)
                past_way_tensor = torch.full((B2, K), -1, dtype=torch.long, device=device)
                for i, bi in enumerate(keep_ids):
                    path = paths[int(bi)]
                    past_list = path[:-1][-K:] if len(path) > 1 else []
                    off = K - len(past_list)
                    for j, w in enumerate(past_list):
                        past_way_tensor[i, off + j] = int(w)
                past_mask_tensor = (past_way_tensor >= 0)

            trans = {
                "route_idx": torch.arange(B2, device=device, dtype=torch.long),
                "cur_way": cur_way[keep],
                "cand_way": cand_way,
                "cand_mask": cand_mask,
                "step": torch.full((B2,), int(step_idx), dtype=torch.long, device=device),
            }
            if past_way_tensor is not None:
                trans["past_way"] = past_way_tensor
                trans["past_mask"] = past_mask_tensor

            logits = self.score_candidates(
                way_embedder=way_embedder,
                latent_tokens=latent_tokens[keep],
                route_cond={k: v[keep] for k, v in route_cond.items()},
                trans=trans,
                cond_emb=cond_emb[keep],
            )

            alpha = float(guided_dest_alpha)
            if abs(alpha) > 1e-12:
                try:
                    coord_scale = float(getattr(way_embedder, "coord_scale", self.coord_scale))
                    dest = route_cond["dest_pos"][keep].to(dtype=torch.float32)
                    if coord_scale > 0:
                        dest = dest / coord_scale
                    cand_geom, _tier, _hw = way_embedder._lookup(cand_way)
                    cand_center = cand_geom[..., :2].to(dtype=torch.float32)
                    dist = torch.norm(dest[:, None, :] - cand_center, dim=-1)  # (B2,C)
                    logits = logits - alpha * dist
                except Exception:
                    pass

            # Anti-loop (soft penalty): reduce logits for candidates that revisit recent ways.
            # Same vectorization trick as hard mask to avoid `.item()` sync.
            pen = float(anti_loop_penalty)
            k_pen = int(anti_loop_penalty_k)
            if (pen > 0.0) and (k_pen > 0):
                recent_cpu = torch.full((B2, k_pen), -1, dtype=torch.long)
                for i, bi in enumerate(keep_ids):
                    tail = paths[int(bi)][-k_pen:]
                    if tail:
                        recent_cpu[i, : int(len(tail))] = torch.as_tensor(tail, dtype=torch.long)
                recent = recent_cpu.to(device=device)
                hit = (cand_way.unsqueeze(-1) == recent[:, None, :])  # (B2,C,K)
                bad = (hit.any(dim=-1)) & cand_mask  # (B2,C)
                logits = logits - bad.to(dtype=logits.dtype) * pen

            beta = float(value_beta)
            if (value_fn is not None) and (abs(beta) > 1e-12):
                try:
                    # Lookahead value: V(next_way, z, dest)
                    cand_emb, _ = way_embedder(cand_way)  # (B2,C,d_model)
                    z_mean = latent_tokens[keep].mean(dim=1)  # (B2,d_model)
                    z_rep = z_mean[:, None, :].expand_as(cand_emb)
                    cond_rep = cond_emb[keep][:, None, :].expand_as(cand_emb)

                    coord_scale = float(getattr(way_embedder, "coord_scale", self.coord_scale))
                    dest = route_cond["dest_pos"][keep].to(dtype=torch.float32)
                    if coord_scale > 0:
                        dest = dest / coord_scale
                    cand_geom, _tier, _hw = way_embedder._lookup(cand_way)
                    cand_center = cand_geom[..., :2].to(dtype=torch.float32)
                    dist = torch.norm(dest[:, None, :] - cand_center, dim=-1)  # (B2,C)

                    v = value_fn(cur_emb=cand_emb, z_mean=z_rep, cond_emb=cond_rep, dest_dist=dist)  # (B2,C)
                    logits = logits + beta * v.to(dtype=logits.dtype)
                except Exception:
                    pass

            j = torch.argmax(logits, dim=-1)  # (B2,)
            nxt = cand_way[torch.arange(B2, device=device), j].to(dtype=torch.long)

            # Update states.
            for i, bi in enumerate(keep_ids):
                w = int(nxt[int(i)].item())
                paths[int(bi)].append(int(w))
                cur_way[int(bi)] = int(w)
                if int(w) == int(dw[int(bi)].item()):
                    active[int(bi)] = False
                if region_paths is not None and region_ptr is not None and way_region is not None:
                    nr = int(way_region[int(w)].item())
                    rp = int(region_ptr[int(bi)])
                    path = region_paths[int(bi)]
                    while rp + 1 < int(len(path)) and int(nr) == int(path[rp + 1]):
                        rp += 1
                    region_ptr[int(bi)] = int(rp)

        return paths

    def sample_decode_batched(
        self,
        *,
        way_embedder: nn.Module,
        latent_tokens: torch.Tensor,  # (B,L,d_model)
        route_cond: Dict[str, torch.Tensor],
        start_way: torch.Tensor,  # (B,)
        dest_way: torch.Tensor,  # (B,)
        way_region: Optional[torch.Tensor] = None,  # (n_ways,) long, optional (same device as latent_tokens)
        region_seq: Optional[List[List[int]]] = None,  # len=B, optional
        region_adj: Optional[torch.Tensor] = None,  # (R,R) bool, optional (required for relaxed mode)
        region_constraint_mode: str = "strict",
        region_constraint_fallback: str = "unconstrained",
        max_len: Optional[int] = None,
        max_candidates: Optional[int] = None,
        candidate_policy: str = "first",
        include_dest_if_successor: bool = False,
        guided_dest_alpha: float = 0.0,
        temperature: float = 1.0,
        anti_loop_k: int = 0,
        anti_loop_penalty: float = 0.0,
        anti_loop_penalty_k: int = 4,
    ) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
        """
        Batched stochastic decoding (policy sampling) for RL.

        Returns:
          paths: python list of predicted way_id sequences (len=B)
          logp_sum: (B,) sum of log-prob under the sampled actions
          entropy_sum: (B,) sum of per-step categorical entropies (for entropy bonus)
        """
        max_len = int(max_len) if max_len is not None else int(self.cfg.max_len)
        B = int(latent_tokens.shape[0])
        device = latent_tokens.device

        use_region_constraint = (way_region is not None) and (region_seq is not None)
        if use_region_constraint and int(len(region_seq)) != int(B):
            raise ValueError(f"region_seq length mismatch: got {len(region_seq)}, expect {B}")
        if use_region_constraint and way_region is not None and way_region.device != device:
            raise ValueError(f"way_region must be on same device as latent_tokens: {way_region.device} vs {device}")
        mode = str(region_constraint_mode or "").strip().lower()
        if mode and mode not in {"strict", "relaxed"}:
            raise ValueError(f"unsupported region_constraint_mode: {region_constraint_mode!r}")
        if use_region_constraint and mode == "relaxed":
            if region_adj is None:
                raise ValueError("region_adj is required for region_constraint_mode='relaxed'")
            if region_adj.device != device:
                raise ValueError(f"region_adj must be on same device as latent_tokens: {region_adj.device} vs {device}")
        fallback = str(region_constraint_fallback or "").strip().lower()
        if fallback and fallback not in {"unconstrained", "stop", "dest_region"}:
            raise ValueError(f"unsupported region_constraint_fallback: {region_constraint_fallback!r}")

        def _compress_consecutive_py(seq: List[int]) -> List[int]:
            out: List[int] = []
            last: Optional[int] = None
            for x in seq:
                xx = int(x)
                if last is None or xx != int(last):
                    out.append(xx)
                    last = xx
            return out

        def _prepare_region_seq(seq: List[int], *, start_region: int, dest_region: int) -> List[int]:
            s = [int(x) for x in seq if int(x) >= 0]
            s = _compress_consecutive_py(s)
            if not s:
                s = [int(start_region), int(dest_region)]
            if int(s[0]) != int(start_region):
                s = [int(start_region)] + s
            if int(s[-1]) != int(dest_region):
                s = s + [int(dest_region)]
            s = _compress_consecutive_py(s)
            return s

        # Precompute route-level conditioning once.
        cond_emb = self.cond_enc(
            start_pos=route_cond["start_pos"],
            dest_pos=route_cond["dest_pos"],
            hour=route_cond["hour"],
            dow=route_cond["dow"],
            route_city=route_cond["route_city"],
        )

        paths: List[List[int]] = [[int(x)] for x in start_way.to(dtype=torch.long).tolist()]
        cur_way = start_way.to(dtype=torch.long, device=device).clone()
        dw = dest_way.to(dtype=torch.long, device=device).clone()
        active = (cur_way != dw)

        logp_sum = torch.zeros((B,), dtype=torch.float32, device=device)
        entropy_sum = torch.zeros((B,), dtype=torch.float32, device=device)

        # Region constraint state (per route).
        region_paths: Optional[List[List[int]]] = None
        region_ptr: Optional[List[int]] = None
        if use_region_constraint and way_region is not None and region_seq is not None:
            region_paths = []
            region_ptr = []
            for b in range(B):
                sw = int(cur_way[b].item())
                dw_b = int(dw[b].item())
                sr = int(way_region[int(sw)].item()) if 0 <= sw < int(way_region.numel()) else -1
                dr = int(way_region[int(dw_b)].item()) if 0 <= dw_b < int(way_region.numel()) else int(sr)
                region_paths.append(_prepare_region_seq(list(region_seq[int(b)]), start_region=sr, dest_region=dr))
                region_ptr.append(0)

        temp = float(temperature)
        if not (temp > 0.0) or not math.isfinite(temp):
            temp = 1.0

        for step_idx in range(int(max_len)):
            if not bool(active.any().item()):
                break

            active_ids = torch.nonzero(active, as_tuple=False).reshape(-1)
            cand_list: List[torch.Tensor] = []
            keep_ids: List[int] = []
            for bi in active_ids.tolist():
                cur = int(cur_way[int(bi)].item())
                cand_full = self.get_succ_candidates(int(cur))
                if int(cand_full.numel()) == 0:
                    active[int(bi)] = False
                    continue
                cand = self._select_decode_candidates(
                    way_embedder=way_embedder,
                    cand_full=cand_full.to(device=device),
                    dest_pos=route_cond["dest_pos"][int(bi) : int(bi) + 1],
                    dest_way=int(dw[int(bi)].item()),
                    max_candidates=max_candidates,
                    candidate_policy=candidate_policy,
                    include_dest_if_successor=include_dest_if_successor,
                )
                if int(cand.numel()) == 0:
                    active[int(bi)] = False
                    continue
                cand_list.append(cand.to(dtype=torch.long, device=device))
                keep_ids.append(int(bi))

            if not keep_ids:
                break

            keep = torch.tensor(keep_ids, dtype=torch.long, device=device)
            B2 = int(keep.numel())
            Cmax = int(max(int(c.numel()) for c in cand_list))

            cand_way = torch.zeros((B2, Cmax), dtype=torch.long, device=device)
            cand_mask = torch.zeros((B2, Cmax), dtype=torch.bool, device=device)
            for i, c in enumerate(cand_list):
                n = int(c.numel())
                cand_way[i, :n] = c
                cand_mask[i, :n] = True

            # Region constraint mask (same as greedy_decode_batched)
            if region_paths is not None and region_ptr is not None and way_region is not None:
                allow0 = torch.full((B2,), -1, dtype=torch.long, device=device)
                allow1 = torch.full((B2,), -1, dtype=torch.long, device=device)
                for i, bi in enumerate(keep_ids):
                    cur = int(cur_way[int(bi)].item())
                    cur_reg = int(way_region[int(cur)].item())
                    rp = int(region_ptr[int(bi)])
                    path = region_paths[int(bi)]
                    while rp + 1 < int(len(path)) and int(cur_reg) == int(path[rp + 1]):
                        rp += 1
                    region_ptr[int(bi)] = int(rp)
                    allow0[i] = int(path[rp]) if rp < int(len(path)) else int(cur_reg)
                    allow1[i] = int(path[rp + 1]) if (rp + 1) < int(len(path)) else -1

                cand_reg = way_region[cand_way]  # (B2,Cmax)
                m = cand_mask & (cand_reg == allow0[:, None])
                has1 = (allow1 >= 0)[:, None]
                m = m | (cand_mask & has1 & (cand_reg == allow1[:, None]))
                if mode == "relaxed" and region_adj is not None:
                    allow0_safe = torch.where(allow0 >= 0, allow0, torch.zeros_like(allow0))
                    allow1_safe = torch.where(allow1 >= 0, allow1, torch.zeros_like(allow1))
                    neigh0 = region_adj[allow0_safe[:, None], cand_reg] & (allow0[:, None] >= 0)
                    neigh1 = region_adj[allow1_safe[:, None], cand_reg] & (allow1[:, None] >= 0)
                    m = m | (cand_mask & neigh0) | (cand_mask & neigh1)
                dest = dw[keep]
                m = m | (cand_mask & (cand_way == dest[:, None]))

                row_has = m.any(dim=1)
                if fallback == "stop":
                    bad = torch.nonzero(~row_has, as_tuple=False).reshape(-1)
                    for i in bad.tolist():
                        active[int(keep_ids[int(i)])] = False

                    good = torch.nonzero(row_has, as_tuple=False).reshape(-1)
                    if int(good.numel()) == 0:
                        break

                    keep_ids = [keep_ids[int(i)] for i in good.tolist()]
                    keep = keep[good]
                    cand_way = cand_way[good]
                    cand_mask = m[good]
                    B2 = int(keep.numel())
                    Cmax = int(cand_way.shape[1])
                elif fallback == "dest_region":
                    dest_reg = way_region[dw[keep]]  # (B2,)
                    has_dest_reg = (dest_reg >= 0)[:, None]
                    m_dest = cand_mask & has_dest_reg & (cand_reg == dest_reg[:, None])
                    row_has_dest = m_dest.any(dim=1)
                    m2 = torch.where(row_has[:, None], m, m_dest)
                    row_has2 = row_has | row_has_dest
                    cand_mask = torch.where(row_has2[:, None], m2, cand_mask)
                else:
                    cand_mask = torch.where(row_has[:, None], m, cand_mask)

            # Anti-loop hard mask (vectorized)
            k_hard = int(anti_loop_k)
            if k_hard > 0:
                before = cand_mask.clone()
                recent_cpu = torch.full((B2, k_hard), -1, dtype=torch.long)
                for i, bi in enumerate(keep_ids):
                    tail = paths[int(bi)][-k_hard:]
                    if tail:
                        recent_cpu[i, : int(len(tail))] = torch.as_tensor(tail, dtype=torch.long)
                recent = recent_cpu.to(device=device)
                hit = (cand_way.unsqueeze(-1) == recent[:, None, :])
                bad = (hit.any(dim=-1)) & cand_mask
                cand_mask = cand_mask & (~bad)
                empty = ~cand_mask.any(dim=1)
                if bool(empty.any().item()):
                    cand_mask[empty] = before[empty]

            past_way_tensor: Optional[torch.Tensor] = None
            past_mask_tensor: Optional[torch.Tensor] = None
            if bool(self.use_past_context):
                K = int(self.past_k)
                past_way_tensor = torch.full((B2, K), -1, dtype=torch.long, device=device)
                for i, bi in enumerate(keep_ids):
                    path = paths[int(bi)]
                    past_list = path[:-1][-K:] if len(path) > 1 else []
                    off = K - len(past_list)
                    for j, w in enumerate(past_list):
                        past_way_tensor[i, off + j] = int(w)
                past_mask_tensor = (past_way_tensor >= 0)

            trans = {
                "route_idx": torch.arange(B2, device=device, dtype=torch.long),
                "cur_way": cur_way[keep],
                "cand_way": cand_way,
                "cand_mask": cand_mask,
                "step": torch.full((B2,), int(step_idx), dtype=torch.long, device=device),
            }
            if past_way_tensor is not None:
                trans["past_way"] = past_way_tensor
                trans["past_mask"] = past_mask_tensor

            logits = self.score_candidates(
                way_embedder=way_embedder,
                latent_tokens=latent_tokens[keep],
                route_cond={k: v[keep] for k, v in route_cond.items()},
                trans=trans,
                cond_emb=cond_emb[keep],
            )

            alpha = float(guided_dest_alpha)
            if abs(alpha) > 1e-12:
                try:
                    coord_scale = float(getattr(way_embedder, "coord_scale", self.coord_scale))
                    dest = route_cond["dest_pos"][keep].to(dtype=torch.float32)
                    if coord_scale > 0:
                        dest = dest / coord_scale
                    cand_geom, _tier, _hw = way_embedder._lookup(cand_way)
                    cand_center = cand_geom[..., :2].to(dtype=torch.float32)
                    dist = torch.norm(dest[:, None, :] - cand_center, dim=-1)  # (B2,C)
                    logits = logits - alpha * dist
                except Exception:
                    pass

            # Anti-loop soft penalty (vectorized)
            pen = float(anti_loop_penalty)
            k_pen = int(anti_loop_penalty_k)
            if (pen > 0.0) and (k_pen > 0):
                recent_cpu = torch.full((B2, k_pen), -1, dtype=torch.long)
                for i, bi in enumerate(keep_ids):
                    tail = paths[int(bi)][-k_pen:]
                    if tail:
                        recent_cpu[i, : int(len(tail))] = torch.as_tensor(tail, dtype=torch.long)
                recent = recent_cpu.to(device=device)
                hit = (cand_way.unsqueeze(-1) == recent[:, None, :])
                bad = (hit.any(dim=-1)) & cand_mask
                logits = logits - bad.to(dtype=logits.dtype) * pen

            # Temperature sampling.
            logits_t = logits / float(temp)
            dist = torch.distributions.Categorical(logits=logits_t)
            j = dist.sample()  # (B2,)
            logp = F.log_softmax(logits_t, dim=-1)
            lp = logp[torch.arange(B2, device=device), j]
            ent = dist.entropy()

            # Update accumulators.
            logp_sum[keep] = logp_sum[keep] + lp
            entropy_sum[keep] = entropy_sum[keep] + ent

            nxt = cand_way[torch.arange(B2, device=device), j].to(dtype=torch.long)

            for i, bi in enumerate(keep_ids):
                w = int(nxt[int(i)].item())
                paths[int(bi)].append(int(w))
                cur_way[int(bi)] = int(w)
                if int(w) == int(dw[int(bi)].item()):
                    active[int(bi)] = False
                if region_paths is not None and region_ptr is not None and way_region is not None:
                    nr = int(way_region[int(w)].item())
                    rp = int(region_ptr[int(bi)])
                    path = region_paths[int(bi)]
                    while rp + 1 < int(len(path)) and int(nr) == int(path[rp + 1]):
                        rp += 1
                    region_ptr[int(bi)] = int(rp)

        return paths, logp_sum, entropy_sum

    @torch.no_grad()
    def beam_search_batched(
        self,
        *,
        way_embedder: nn.Module,
        latent_tokens: torch.Tensor,  # (B,L,d_model)
        route_cond: Dict[str, torch.Tensor],
        start_way: torch.Tensor,  # (B,)
        dest_way: torch.Tensor,  # (B,)
        way_region: Optional[torch.Tensor] = None,  # (n_ways,) long, optional
        region_seq: Optional[List[List[int]]] = None,  # len=B, optional
        region_adj: Optional[torch.Tensor] = None,  # (R,R) bool, optional (required for relaxed mode)
        region_constraint_mode: str = "strict",
        region_constraint_fallback: str = "unconstrained",
        beam_size: int = 5,
        max_len: Optional[int] = None,
        max_candidates: Optional[int] = None,
        candidate_policy: str = "first",
        include_dest_if_successor: bool = False,
        guided_dest_alpha: float = 0.0,
        anti_loop_k: int = 0,
        anti_loop_penalty: float = 0.0,
        anti_loop_penalty_k: int = 4,
        value_fn: Optional[nn.Module] = None,
        value_beta: float = 0.0,
    ) -> List[List[int]]:
        """
        Batched beam search:
          - one score_candidates() forward per step for all active beam states
        """
        max_len = int(max_len) if max_len is not None else int(self.cfg.max_len)
        beam_size = max(1, int(beam_size))
        B = int(latent_tokens.shape[0])
        device = latent_tokens.device

        use_region_constraint = (way_region is not None) and (region_seq is not None)
        if use_region_constraint and int(len(region_seq)) != int(B):
            raise ValueError(f"region_seq length mismatch: got {len(region_seq)}, expect {B}")
        if use_region_constraint and way_region is not None and way_region.device != device:
            raise ValueError(f"way_region must be on same device as latent_tokens: {way_region.device} vs {device}")
        mode = str(region_constraint_mode or "").strip().lower()
        if mode and mode not in {"strict", "relaxed"}:
            raise ValueError(f"unsupported region_constraint_mode: {region_constraint_mode!r}")
        if use_region_constraint and mode == "relaxed":
            if region_adj is None:
                raise ValueError("region_adj is required for region_constraint_mode='relaxed'")
            if region_adj.device != device:
                raise ValueError(f"region_adj must be on same device as latent_tokens: {region_adj.device} vs {device}")
        fallback = str(region_constraint_fallback or "").strip().lower()
        if fallback and fallback not in {"unconstrained", "stop", "dest_region"}:
            raise ValueError(f"unsupported region_constraint_fallback: {region_constraint_fallback!r}")

        def _compress_consecutive_py(seq: List[int]) -> List[int]:
            out: List[int] = []
            last: Optional[int] = None
            for x in seq:
                xx = int(x)
                if last is None or xx != int(last):
                    out.append(xx)
                    last = xx
            return out

        def _prepare_region_seq(seq: List[int], *, start_region: int, dest_region: int) -> List[int]:
            s = [int(x) for x in seq if int(x) >= 0]
            s = _compress_consecutive_py(s)
            if not s:
                s = [int(start_region), int(dest_region)]
            if int(s[0]) != int(start_region):
                s = [int(start_region)] + s
            if int(s[-1]) != int(dest_region):
                s = s + [int(dest_region)]
            s = _compress_consecutive_py(s)
            return s

        cond_emb = self.cond_enc(
            start_pos=route_cond["start_pos"],
            dest_pos=route_cond["dest_pos"],
            hour=route_cond["hour"],
            dow=route_cond["dow"],
            route_city=route_cond["route_city"],
        )
        z_mean: Optional[torch.Tensor] = None
        beta = float(value_beta)
        if (value_fn is not None) and (abs(beta) > 1e-12):
            z_mean = latent_tokens.mean(dim=1)  # (B,d_model)

        # per-route region paths
        region_paths: Optional[List[List[int]]] = None
        if use_region_constraint and way_region is not None and region_seq is not None:
            region_paths = []
            sw0 = start_way.to(dtype=torch.long).tolist()
            dw0 = dest_way.to(dtype=torch.long).tolist()
            for b in range(B):
                sw = int(sw0[b])
                dwb = int(dw0[b])
                sr = int(way_region[int(sw)].item()) if 0 <= sw < int(way_region.numel()) else -1
                dr = int(way_region[int(dwb)].item()) if 0 <= dwb < int(way_region.numel()) else int(sr)
                region_paths.append(_prepare_region_seq(list(region_seq[int(b)]), start_region=sr, dest_region=dr))

        # beams per route: list[(path, score, region_ptr)]
        beams: List[List[Tuple[List[int], float, int]]] = []
        for b in range(B):
            beams.append([([int(start_way[b].item())], 0.0, 0)])

        for step_idx in range(int(max_len)):
            # flatten active states
            states: List[Tuple[int, List[int], float, int]] = []
            for b in range(B):
                dwb = int(dest_way[b].item())
                for path, score, rptr in beams[b]:
                    if path and self.is_dest_reached(path[-1], dwb):
                        continue
                    states.append((int(b), list(path), float(score), int(rptr)))

            if not states:
                break

            # build candidates per state
            route_ids: List[int] = []
            cur_way_list: List[int] = []
            step_list: List[int] = []
            cand_list: List[torch.Tensor] = []
            score_list: List[float] = []
            rptr_list: List[int] = []
            path_list: List[List[int]] = []

            for b, path, score, rptr in states:
                cur = int(path[-1])
                cand_full = self.get_succ_candidates(int(cur))
                if int(cand_full.numel()) == 0:
                    continue
                cand = self._select_decode_candidates(
                    way_embedder=way_embedder,
                    cand_full=cand_full.to(device=device),
                    dest_pos=route_cond["dest_pos"][int(b) : int(b) + 1],
                    dest_way=int(dest_way[int(b)].item()),
                    max_candidates=max_candidates,
                    candidate_policy=candidate_policy,
                    include_dest_if_successor=include_dest_if_successor,
                )
                if int(cand.numel()) == 0:
                    continue
                route_ids.append(int(b))
                cur_way_list.append(int(cur))
                step_list.append(int(len(path) - 1))
                cand_list.append(cand.to(dtype=torch.long, device=device))
                score_list.append(float(score))
                rptr_list.append(int(rptr))
                path_list.append(path)

            if not cand_list:
                break

            T = int(len(cand_list))
            Cmax = int(max(int(c.numel()) for c in cand_list))
            cand_way = torch.zeros((T, Cmax), dtype=torch.long, device=device)
            cand_mask = torch.zeros((T, Cmax), dtype=torch.bool, device=device)
            for i, c in enumerate(cand_list):
                n = int(c.numel())
                cand_way[i, :n] = c
                cand_mask[i, :n] = True

            # region constraint mask per state
            if region_paths is not None and way_region is not None:
                allow0 = torch.full((T,), -1, dtype=torch.long, device=device)
                allow1 = torch.full((T,), -1, dtype=torch.long, device=device)
                new_rptr_list = list(rptr_list)
                for i in range(T):
                    b = int(route_ids[i])
                    cur_reg = int(way_region[int(cur_way_list[i])].item())
                    rp = int(new_rptr_list[i])
                    path = region_paths[int(b)]
                    while rp + 1 < int(len(path)) and int(cur_reg) == int(path[rp + 1]):
                        rp += 1
                    new_rptr_list[i] = int(rp)
                    allow0[i] = int(path[rp]) if rp < int(len(path)) else int(cur_reg)
                    allow1[i] = int(path[rp + 1]) if (rp + 1) < int(len(path)) else -1
                rptr_list = new_rptr_list

                cand_reg = way_region[cand_way]
                m = cand_mask & (cand_reg == allow0[:, None])
                has1 = (allow1 >= 0)[:, None]
                m = m | (cand_mask & has1 & (cand_reg == allow1[:, None]))
                if mode == "relaxed" and region_adj is not None:
                    allow0_safe = torch.where(allow0 >= 0, allow0, torch.zeros_like(allow0))
                    allow1_safe = torch.where(allow1 >= 0, allow1, torch.zeros_like(allow1))
                    neigh0 = region_adj[allow0_safe[:, None], cand_reg] & (allow0[:, None] >= 0)
                    neigh1 = region_adj[allow1_safe[:, None], cand_reg] & (allow1[:, None] >= 0)
                    m = m | (cand_mask & neigh0) | (cand_mask & neigh1)
                # Keep direct successor-to-dest if present.
                route_ids_t = torch.tensor(route_ids, dtype=torch.long, device=device)
                dest = dest_way[route_ids_t]
                m = m | (cand_mask & (cand_way == dest[:, None]))
                row_has = m.any(dim=1)
                if fallback == "stop":
                    good = torch.nonzero(row_has, as_tuple=False).reshape(-1)
                    if int(good.numel()) == 0:
                        break
                    cand_way = cand_way[good]
                    cand_mask = m[good]
                    route_ids = [route_ids[int(i)] for i in good.tolist()]
                    cur_way_list = [cur_way_list[int(i)] for i in good.tolist()]
                    step_list = [step_list[int(i)] for i in good.tolist()]
                    rptr_list = [rptr_list[int(i)] for i in good.tolist()]
                    path_list = [path_list[int(i)] for i in good.tolist()]
                    score_list = [score_list[int(i)] for i in good.tolist()]
                    T = int(cand_way.shape[0])
                elif fallback == "dest_region":
                    dest_reg = way_region[dest]  # (T,)
                    has_dest_reg = (dest_reg >= 0)[:, None]
                    m_dest = cand_mask & has_dest_reg & (cand_reg == dest_reg[:, None])
                    row_has_dest = m_dest.any(dim=1)
                    m2 = torch.where(row_has[:, None], m, m_dest)
                    row_has2 = row_has | row_has_dest
                    cand_mask = torch.where(row_has2[:, None], m2, cand_mask)
                else:
                    cand_mask = torch.where(row_has[:, None], m, cand_mask)

            # Anti-loop (hard mask): exclude candidates that revisit any of the last K visited ways.
            # NOTE: avoid per-element `.item()` on GPU. We build a small CPU tensor of recent IDs then vectorize.
            k_hard = int(anti_loop_k)
            if k_hard > 0:
                before = cand_mask.clone()
                recent_cpu = torch.full((T, k_hard), -1, dtype=torch.long)
                for i, path in enumerate(path_list):
                    tail = path[-k_hard:]
                    if tail:
                        recent_cpu[i, : int(len(tail))] = torch.as_tensor(tail, dtype=torch.long)
                recent = recent_cpu.to(device=device)
                hit = (cand_way.unsqueeze(-1) == recent[:, None, :])  # (T,C,K)
                bad = (hit.any(dim=-1)) & cand_mask  # (T,C)
                cand_mask = cand_mask & (~bad)
                empty = ~cand_mask.any(dim=1)
                if bool(empty.any().item()):
                    cand_mask[empty] = before[empty]

            # past context per state
            past_way_tensor: Optional[torch.Tensor] = None
            past_mask_tensor: Optional[torch.Tensor] = None
            if bool(self.use_past_context):
                K = int(self.past_k)
                past_way_tensor = torch.full((T, K), -1, dtype=torch.long, device=device)
                for i, path in enumerate(path_list):
                    past_list = path[:-1][-K:] if len(path) > 1 else []
                    off = K - len(past_list)
                    for j, w in enumerate(past_list):
                        past_way_tensor[i, off + j] = int(w)
                past_mask_tensor = (past_way_tensor >= 0)

            trans = {
                "route_idx": torch.tensor(route_ids, dtype=torch.long, device=device),
                "cur_way": torch.tensor(cur_way_list, dtype=torch.long, device=device),
                "cand_way": cand_way,
                "cand_mask": cand_mask,
                "step": torch.tensor(step_list, dtype=torch.long, device=device),
            }
            if past_way_tensor is not None:
                trans["past_way"] = past_way_tensor
                trans["past_mask"] = past_mask_tensor

            logits = self.score_candidates(
                way_embedder=way_embedder,
                latent_tokens=latent_tokens,
                route_cond=route_cond,
                trans=trans,
                cond_emb=cond_emb,
            )  # (T,C)

            alpha = float(guided_dest_alpha)
            if abs(alpha) > 1e-12:
                try:
                    coord_scale = float(getattr(way_embedder, "coord_scale", self.coord_scale))
                    dest = route_cond["dest_pos"][trans["route_idx"]].to(dtype=torch.float32)
                    if coord_scale > 0:
                        dest = dest / coord_scale
                    cand_geom, _tier, _hw = way_embedder._lookup(cand_way)
                    cand_center = cand_geom[..., :2].to(dtype=torch.float32)
                    dist = torch.norm(dest[:, None, :] - cand_center, dim=-1)
                    logits = logits - alpha * dist
                except Exception:
                    pass

            # Anti-loop (soft penalty): reduce logits for candidates that revisit recent ways.
            # Same vectorization trick as hard mask to avoid `.item()` sync.
            pen = float(anti_loop_penalty)
            k_pen = int(anti_loop_penalty_k)
            if (pen > 0.0) and (k_pen > 0):
                recent_cpu = torch.full((T, k_pen), -1, dtype=torch.long)
                for i, path in enumerate(path_list):
                    tail = path[-k_pen:]
                    if tail:
                        recent_cpu[i, : int(len(tail))] = torch.as_tensor(tail, dtype=torch.long)
                recent = recent_cpu.to(device=device)
                hit = (cand_way.unsqueeze(-1) == recent[:, None, :])  # (T,C,K)
                bad = (hit.any(dim=-1)) & cand_mask  # (T,C)
                logits = logits - bad.to(dtype=logits.dtype) * pen

            if (value_fn is not None) and (abs(beta) > 1e-12) and (z_mean is not None):
                try:
                    cand_emb, _ = way_embedder(cand_way)  # (T,C,d_model)
                    ridx = trans["route_idx"].to(dtype=torch.long)  # (T,)
                    z_t = z_mean[ridx]  # (T,d)
                    cond_t = cond_emb[ridx]  # (T,d)
                    z_rep = z_t[:, None, :].expand_as(cand_emb)
                    cond_rep = cond_t[:, None, :].expand_as(cand_emb)

                    coord_scale = float(getattr(way_embedder, "coord_scale", self.coord_scale))
                    dest = route_cond["dest_pos"][ridx].to(dtype=torch.float32)
                    if coord_scale > 0:
                        dest = dest / coord_scale
                    cand_geom, _tier, _hw = way_embedder._lookup(cand_way)
                    cand_center = cand_geom[..., :2].to(dtype=torch.float32)
                    dist = torch.norm(dest[:, None, :] - cand_center, dim=-1)  # (T,C)

                    v = value_fn(cur_emb=cand_emb, z_mean=z_rep, cond_emb=cond_rep, dest_dist=dist)  # (T,C)
                    logits = logits + beta * v.to(dtype=logits.dtype)
                except Exception:
                    pass

            logp = F.log_softmax(logits, dim=-1)
            topk = min(int(beam_size), int(logp.shape[1]))
            vals, ids = torch.topk(logp, k=topk, dim=-1)  # (T,topk)

            # expand beams
            new_beams: List[List[Tuple[List[int], float, int]]] = [[] for _ in range(B)]
            # carry finished
            for b in range(B):
                dwb = int(dest_way[b].item())
                for path, score, rptr in beams[b]:
                    if path and self.is_dest_reached(path[-1], dwb):
                        new_beams[b].append((path, float(score), int(rptr)))

            for i in range(T):
                b = int(route_ids[i])
                base_path = path_list[i]
                base_score = float(score_list[i])
                base_rptr = int(rptr_list[i])
                for lp, j in zip(vals[i].tolist(), ids[i].tolist()):
                    if not math.isfinite(float(lp)):
                        continue
                    if not bool(cand_mask[i, int(j)].item()):
                        continue
                    nxt = int(cand_way[i, int(j)].item())
                    new_rptr = int(base_rptr)
                    if region_paths is not None and way_region is not None:
                        if 0 <= int(nxt) < int(way_region.numel()):
                            nxt_reg = int(way_region[int(nxt)].item())
                            rpath = region_paths[int(b)]
                            while new_rptr + 1 < int(len(rpath)) and int(nxt_reg) == int(rpath[new_rptr + 1]):
                                new_rptr += 1
                    new_beams[int(b)].append((base_path + [int(nxt)], base_score + float(lp), int(new_rptr)))

            # prune beams per route
            for b in range(B):
                if new_beams[b]:
                    new_beams[b].sort(key=lambda x: -x[1])
                    beams[b] = new_beams[b][:beam_size]
                # else: keep previous beams[b] (dead_end / stop)

        # pick best per route
        out: List[List[int]] = []
        for b in range(B):
            if not beams[b]:
                out.append([int(start_way[b].item())])
            else:
                beams[b].sort(key=lambda x: -x[1])
                out.append(beams[b][0][0])
        return out
