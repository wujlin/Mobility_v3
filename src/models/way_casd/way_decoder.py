from __future__ import annotations

from dataclasses import dataclass
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
    # Cross-attention for querying latent tokens
    use_cross_attn: bool = True
    n_cross_heads: int = 4
    # Backward compatibility:
    use_dest_dist: bool = True
    # Query enrichment (for early-step disambiguation)
    use_step_emb: bool = False
    use_dest_query: bool = False
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
        in_dim = int(hidden * 3) + (1 if bool(cfg.use_dest_dist) else 0)
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
        route_idx: torch.Tensor,  # (T,)
        step: torch.Tensor,  # (T,)
        dest_pos: torch.Tensor,  # (B,2)
        past_way: Optional[torch.Tensor] = None,  # (T, K) past way IDs, -1 for padding
        past_mask: Optional[torch.Tensor] = None,  # (T, K) bool, True=valid
    ) -> torch.Tensor:
        """
        Compute context vector for each transition.
        
        If use_cross_attn: query latent_tokens with cur_way embedding.
        Otherwise: use mean-pooled latent.
        
        If use_past_context: incorporate past_way history via Transformer encoder.
        
        Returns: ctx (T, hidden_dim)
        """
        B = int(latent_tokens.shape[0])
        T = int(cur_way.shape[0])
        device = latent_tokens.device
        
        # Get current way embeddings
        cur_emb, _ = way_embedder(cur_way[:, None])  # (T, 1, d_model)
        cur_emb = cur_emb[:, 0, :]  # (T, d_model)

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

        # Past context: encode history using Transformer
        if self.use_past_context and self.past_encoder is not None:
            if past_way is not None and past_mask is not None:
                # Embed past ways
                past_emb, _ = way_embedder(past_way)  # (T, K, d_model)
                past_ctx = self.past_encoder(past_emb, past_mask)  # (T, d_model)
                query_vec = query_vec + past_ctx
            # If past_way is None (e.g., step 0), query_vec remains unchanged

        if self.use_cross_attn:
            # Cross-attention: each cur_way queries its route's latent_tokens
            # We need to gather latent_tokens by route_idx
            # latent_tokens: (B, L, d_model) -> expand to (T, L, d_model) by route_idx
            L = int(latent_tokens.shape[1])
            d = int(latent_tokens.shape[2])
            
            # Gather latent tokens for each transition
            lat_gathered = latent_tokens[route_idx]  # (T, L, d_model)
            
            # Cross-attention: query=query_vec (T,1,d), key/value=lat_gathered (T,L,d)
            query = query_vec.unsqueeze(1)  # (T, 1, d_model)
            attn_out, _ = self.cross_attn(query, lat_gathered, lat_gathered)  # (T, 1, d_model)
            attn_out = self.cross_ln(attn_out[:, 0, :] + query_vec)  # (T, d_model), residual
            
            # Combine with condition embedding
            cond_t = cond_emb[route_idx]  # (T, d_model)
            ctx = self.ctx_mlp(torch.cat([cond_t, attn_out], dim=-1))  # (T, hidden)
        else:
            # Fallback: mean-pooled latent
            lat_vec = latent_tokens.mean(dim=1)  # (B, d_model)
            ctx_b = self.ctx_mlp(torch.cat([cond_emb, lat_vec], dim=-1))  # (B, hidden)
            ctx = ctx_b[route_idx]  # (T, hidden)
        
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
        
        # Compute context with cross-attention (and past context if enabled)
        ctx_t = self._compute_context(
            way_embedder=way_embedder,
            latent_tokens=latent_tokens,
            cond_emb=cond_emb,
            cur_way=cur_way,
            route_idx=route_idx,
            step=step,
            dest_pos=route_cond["dest_pos"],
            past_way=past_way,
            past_mask=past_way_mask,
        )

        # Current way projection
        cur_emb, _ = way_embedder(cur_way[:, None])
        cur_emb = cur_emb[:, 0, :]
        cur_h = self.cur_proj(cur_emb)

        # Candidate way embeddings
        cand_emb, _ = way_embedder(cand_way)
        cand_h = self.cand_proj(cand_emb)

        T, C = cand_way.shape
        ctx_h = ctx_t[:, None, :].expand(T, C, -1)
        cur_h2 = cur_h[:, None, :].expand(T, C, -1)
        
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
            x = torch.cat([ctx_h, cur_h2, cand_h, dist], dim=-1)
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
            beams: List[Tuple[List[int], float]] = [([sw], 0.0)]

            for _step in range(max_len):
                new_beams: List[Tuple[List[int], float]] = []
                all_finished = True
                for path, score in beams:
                    if path and self.is_dest_reached(path[-1], dw):
                        new_beams.append((path, score))
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
                        new_beams.append((path + [int(cand[j].item())], score + float(lp)))

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
