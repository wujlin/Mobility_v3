"""
Decision Point Autoregressive Model (Proposal C).

Key difference from node-level AR:
- Operates on DECISION POINTS, not raw nodes
- Each step: P(next_dp | current_dp, dest_dp, time, context)
- Candidate set: 2-8 successor decision points (variable size)
- Uses Pointer Network / Cross-attention for variable candidate scoring

This is NOT a classifier over fixed vocab - it's a set scorer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DPARConfig:
    """Configuration for Decision Point AR model."""
    # Feature dimensions
    dp_embed_dim: int = 64       # decision point embedding
    time_embed_dim: int = 32     # hour-of-day embedding
    pos_embed_dim: int = 64      # position encoding (x,y coords)
    hidden_dim: int = 128        # MLP hidden dim
    
    # Architecture
    n_heads: int = 4             # attention heads for pointer
    dropout: float = 0.1
    max_candidates: int = 32     # max number of candidate successors
    
    # Training
    label_smoothing: float = 0.0


class PositionalEncoder(nn.Module):
    """Encode (x, y) coordinates into position embeddings."""
    
    def __init__(self, pos_embed_dim: int):
        super().__init__()
        self.pos_embed_dim = pos_embed_dim
        half = pos_embed_dim // 2
        # Fourier features for continuous coordinates
        self.fc_y = nn.Linear(1, half)
        self.fc_x = nn.Linear(1, half)
        self.fc_out = nn.Linear(pos_embed_dim, pos_embed_dim)
        
    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y, x: (...,) coordinates
        Returns:
            (..., pos_embed_dim) position embeddings
        """
        y_emb = torch.sin(self.fc_y(y.unsqueeze(-1)))  # (..., half)
        x_emb = torch.sin(self.fc_x(x.unsqueeze(-1)))  # (..., half)
        pos = torch.cat([y_emb, x_emb], dim=-1)        # (..., pos_embed_dim)
        return self.fc_out(pos)


class TimeEncoder(nn.Module):
    """Encode hour-of-day into embeddings."""
    
    def __init__(self, time_embed_dim: int):
        super().__init__()
        # Cyclic encoding for hour (0-23)
        self.fc = nn.Sequential(
            nn.Linear(2, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
    
    def forward(self, hour: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hour: (...,) hour values (0-23)
        Returns:
            (..., time_embed_dim)
        """
        # Cyclic encoding
        hour_rad = hour.float() * (2 * 3.14159 / 24.0)
        sin_h = torch.sin(hour_rad).unsqueeze(-1)
        cos_h = torch.cos(hour_rad).unsqueeze(-1)
        return self.fc(torch.cat([sin_h, cos_h], dim=-1))


class DecisionPointARModel(nn.Module):
    """
    Decision Point AR Model with Pointer Network.
    
    At each step, given:
        - current_dp: which decision point we're at
        - dest_dp: final destination decision point
        - hour: time of day
        - candidate_set: variable-size set of possible next decision points
        
    Output: probability distribution over candidate_set
    """
    
    def __init__(self, cfg: DPARConfig, n_decision_points: int):
        super().__init__()
        self.cfg = cfg
        self.n_decision_points = n_decision_points
        
        # Embeddings
        self.dp_embed = nn.Embedding(n_decision_points, cfg.dp_embed_dim)
        self.pos_encoder = PositionalEncoder(cfg.pos_embed_dim)
        self.time_encoder = TimeEncoder(cfg.time_embed_dim)
        
        # Context encoder: combines current_dp, dest_dp, time, positions
        context_dim = cfg.dp_embed_dim * 2 + cfg.pos_embed_dim * 2 + cfg.time_embed_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        
        # Candidate encoder: encode each candidate
        cand_input_dim = cfg.dp_embed_dim + cfg.pos_embed_dim
        self.cand_mlp = nn.Sequential(
            nn.Linear(cand_input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        
        # Pointer attention: context queries, candidates are keys
        self.pointer = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        
        # Final scoring layer
        self.score_fc = nn.Linear(cfg.hidden_dim, 1)
        
    def forward(
        self,
        current_dp: torch.Tensor,      # (B,) current decision point indices
        dest_dp: torch.Tensor,         # (B,) destination decision point indices
        hour: torch.Tensor,            # (B,) hour of day
        current_pos: Tuple[torch.Tensor, torch.Tensor],  # (y, x) each (B,)
        dest_pos: Tuple[torch.Tensor, torch.Tensor],     # (y, x) each (B,)
        cand_dp: torch.Tensor,         # (B, max_cand) candidate dp indices (-1 for padding)
        cand_pos: Tuple[torch.Tensor, torch.Tensor],     # (y, x) each (B, max_cand)
        cand_mask: torch.Tensor,       # (B, max_cand) True for valid candidates
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Returns:
            logits: (B, max_cand) scores for each candidate (masked positions = -inf)
        """
        B, max_cand = cand_dp.shape
        device = current_dp.device
        
        # Encode context: current_dp + dest_dp + positions + time
        cur_emb = self.dp_embed(current_dp)                          # (B, dp_embed_dim)
        dest_emb = self.dp_embed(dest_dp)                            # (B, dp_embed_dim)
        cur_pos_emb = self.pos_encoder(current_pos[0], current_pos[1])  # (B, pos_embed_dim)
        dest_pos_emb = self.pos_encoder(dest_pos[0], dest_pos[1])       # (B, pos_embed_dim)
        time_emb = self.time_encoder(hour)                           # (B, time_embed_dim)
        
        context = torch.cat([cur_emb, dest_emb, cur_pos_emb, dest_pos_emb, time_emb], dim=-1)
        context = self.context_mlp(context)  # (B, hidden_dim)
        context = context.unsqueeze(1)       # (B, 1, hidden_dim)
        
        # Encode candidates
        # Replace -1 with 0 for embedding lookup, then mask
        cand_dp_safe = cand_dp.clamp(min=0)
        cand_emb = self.dp_embed(cand_dp_safe)                        # (B, max_cand, dp_embed_dim)
        cand_pos_emb = self.pos_encoder(cand_pos[0], cand_pos[1])     # (B, max_cand, pos_embed_dim)
        
        cand_feat = torch.cat([cand_emb, cand_pos_emb], dim=-1)       # (B, max_cand, cand_input_dim)
        cand_feat = self.cand_mlp(cand_feat)                         # (B, max_cand, hidden_dim)
        
        # Pointer attention: context attends to candidates
        # key_padding_mask: True means IGNORE (padding)
        attn_mask = ~cand_mask  # (B, max_cand) True where padding
        
        attn_out, _ = self.pointer(
            query=context,           # (B, 1, hidden_dim)
            key=cand_feat,           # (B, max_cand, hidden_dim)
            value=cand_feat,         # (B, max_cand, hidden_dim)
            key_padding_mask=attn_mask,
        )  # (B, 1, hidden_dim)
        
        # Score each candidate: (context_attended) dot (candidate)
        # Or simpler: MLP score
        combined = context.expand(-1, max_cand, -1) + cand_feat  # (B, max_cand, hidden_dim)
        logits = self.score_fc(combined).squeeze(-1)              # (B, max_cand)
        
        # Mask invalid candidates
        logits = logits.masked_fill(~cand_mask, float("-inf"))
        
        return logits
    
    def loss(
        self,
        logits: torch.Tensor,       # (B, max_cand)
        target_idx: torch.Tensor,   # (B,) index into cand_dp of the correct next dp
        cand_mask: torch.Tensor,    # (B, max_cand)
    ) -> torch.Tensor:
        """Compute cross-entropy loss over candidates."""
        # target_idx is the position in candidate set, not the dp index
        loss = F.cross_entropy(
            logits, 
            target_idx, 
            label_smoothing=self.cfg.label_smoothing,
            reduction="mean",
        )
        return loss


class DecisionPointARModelSimple(nn.Module):
    """
    Simplified Decision Point AR Model.
    
    Uses MLP scoring without pointer attention.
    May be faster and sufficient for the task.
    """
    
    def __init__(self, cfg: DPARConfig, n_decision_points: int):
        super().__init__()
        self.cfg = cfg
        self.n_decision_points = n_decision_points
        
        # Embeddings
        self.dp_embed = nn.Embedding(n_decision_points, cfg.dp_embed_dim)
        self.pos_encoder = PositionalEncoder(cfg.pos_embed_dim)
        self.time_encoder = TimeEncoder(cfg.time_embed_dim)
        
        # Combined feature dimension for scoring
        # current: dp_embed + pos_embed
        # dest: dp_embed + pos_embed
        # candidate: dp_embed + pos_embed
        # time: time_embed
        input_dim = (cfg.dp_embed_dim + cfg.pos_embed_dim) * 3 + cfg.time_embed_dim
        
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )
        
    def forward(
        self,
        current_dp: torch.Tensor,      # (B,) current decision point indices
        dest_dp: torch.Tensor,         # (B,) destination decision point indices
        hour: torch.Tensor,            # (B,) hour of day
        current_pos: Tuple[torch.Tensor, torch.Tensor],  # (y, x) each (B,)
        dest_pos: Tuple[torch.Tensor, torch.Tensor],     # (y, x) each (B,)
        cand_dp: torch.Tensor,         # (B, max_cand) candidate dp indices
        cand_pos: Tuple[torch.Tensor, torch.Tensor],     # (y, x) each (B, max_cand)
        cand_mask: torch.Tensor,       # (B, max_cand) True for valid
    ) -> torch.Tensor:
        """Forward pass, returns logits (B, max_cand)."""
        B, max_cand = cand_dp.shape
        
        # Encode current and dest
        cur_emb = self.dp_embed(current_dp)                          # (B, dp_embed_dim)
        dest_emb = self.dp_embed(dest_dp)                            # (B, dp_embed_dim)
        cur_pos_emb = self.pos_encoder(current_pos[0], current_pos[1])  # (B, pos_embed_dim)
        dest_pos_emb = self.pos_encoder(dest_pos[0], dest_pos[1])       # (B, pos_embed_dim)
        time_emb = self.time_encoder(hour)                           # (B, time_embed_dim)
        
        cur_feat = torch.cat([cur_emb, cur_pos_emb], dim=-1)         # (B, dp+pos)
        dest_feat = torch.cat([dest_emb, dest_pos_emb], dim=-1)      # (B, dp+pos)
        
        # Expand to match candidates
        cur_feat = cur_feat.unsqueeze(1).expand(-1, max_cand, -1)    # (B, max_cand, dp+pos)
        dest_feat = dest_feat.unsqueeze(1).expand(-1, max_cand, -1)  # (B, max_cand, dp+pos)
        time_emb = time_emb.unsqueeze(1).expand(-1, max_cand, -1)    # (B, max_cand, time)
        
        # Encode candidates
        cand_dp_safe = cand_dp.clamp(min=0)
        cand_emb = self.dp_embed(cand_dp_safe)                       # (B, max_cand, dp_embed_dim)
        cand_pos_emb = self.pos_encoder(cand_pos[0], cand_pos[1])    # (B, max_cand, pos_embed_dim)
        cand_feat = torch.cat([cand_emb, cand_pos_emb], dim=-1)      # (B, max_cand, dp+pos)
        
        # Concatenate all features
        combined = torch.cat([cur_feat, dest_feat, cand_feat, time_emb], dim=-1)
        
        # Score
        logits = self.scorer(combined).squeeze(-1)                   # (B, max_cand)
        logits = logits.masked_fill(~cand_mask, float("-inf"))
        
        return logits
    
    def loss(
        self,
        logits: torch.Tensor,
        target_idx: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss."""
        return F.cross_entropy(
            logits, 
            target_idx, 
            label_smoothing=self.cfg.label_smoothing,
            reduction="mean",
        )
