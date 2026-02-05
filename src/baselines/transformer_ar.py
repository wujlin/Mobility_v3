from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.way_casd.conditions import ConditionEncoder, ConditionEncoderCfg
from src.utils.way_csr import slice_csr


@dataclass(frozen=True)
class WayTransformerARCfg:
    n_ways: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    max_len: int = 160
    max_candidates: int = 32
    n_route_cities: int = 4
    coord_scale: float = 1024.0


def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
    # True = masked (not allowed)
    return torch.triu(torch.ones((int(T), int(T)), device=device, dtype=torch.bool), diagonal=1)


class _CausalSelfAttention(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        d_model = int(d_model)
        n_heads = int(n_heads)
        if d_model <= 0 or n_heads <= 0 or (d_model % n_heads) != 0:
            raise ValueError("d_model must be >0 and divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, *, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Full-sequence causal self-attention.
        Args:
            x: (B,T,D)
            key_padding_mask: (B,T) bool, True=pad (masked as keys)
        Returns:
            y: (B,T,D)
        """
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B,T,3D)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,dh)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = float(self.head_dim) ** -0.5
        att = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,T,T)

        # Causal mask (same for all batches/heads)
        cm = _causal_mask(T, device=x.device)  # (T,T)
        att = att.masked_fill(cm[None, None, :, :], float("-inf"))

        # Key padding mask: mask as keys
        if key_padding_mask is not None:
            km = key_padding_mask.to(dtype=torch.bool).reshape(B, 1, 1, T)
            att = att.masked_fill(km, float("-inf"))

        w = torch.softmax(att, dim=-1)
        w = self.drop(w)
        y = torch.matmul(w, v)  # (B,H,T,dh)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y)

    def step(
        self,
        x_t: torch.Tensor,  # (B,D)
        *,
        k_cache: torch.Tensor,  # (B,H,L,dh)
        v_cache: torch.Tensor,  # (B,H,L,dh)
        t: int,
    ) -> torch.Tensor:
        """
        Incremental step with preallocated KV cache.
        Args:
            x_t: (B,D) token input at position t
            k_cache/v_cache: preallocated caches (B,H,L,dh)
            t: current index (0-based)
        Returns:
            y_t: (B,D) attention output for this position
        """
        B, D = x_t.shape
        qkv = self.qkv(x_t)  # (B,3D)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(B, self.n_heads, self.head_dim)  # (B,H,dh)
        k = k.view(B, self.n_heads, self.head_dim)
        v = v.view(B, self.n_heads, self.head_dim)

        k_cache[:, :, int(t), :] = k
        v_cache[:, :, int(t), :] = v

        kk = k_cache[:, :, : int(t) + 1, :]  # (B,H,t+1,dh)
        vv = v_cache[:, :, : int(t) + 1, :]

        scale = float(self.head_dim) ** -0.5
        att = torch.matmul(q[:, :, None, :], kk.transpose(-2, -1)) * scale  # (B,H,1,t+1)
        w = torch.softmax(att, dim=-1)
        y = torch.matmul(w, vv)[:, :, 0, :]  # (B,H,dh)
        y = y.reshape(B, D)
        return self.out(y)


class _FFN(nn.Module):
    def __init__(self, *, d_model: int, dropout: float) -> None:
        super().__init__()
        d_model = int(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Block(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(int(d_model))
        self.attn = _CausalSelfAttention(d_model=int(d_model), n_heads=int(n_heads), dropout=float(dropout))
        self.ln2 = nn.LayerNorm(int(d_model))
        self.ff = _FFN(d_model=int(d_model), dropout=float(dropout))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, *, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, key_padding_mask=key_padding_mask))
        h = self.ln2(x)
        x = x + self.drop(self.ff(h))
        return x

    def step(
        self,
        x_t: torch.Tensor,  # (B,D)
        *,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        h = self.ln1(x_t)
        x_t = x_t + self.drop(self.attn.step(h, k_cache=k_cache, v_cache=v_cache, t=int(t)))
        h2 = self.ln2(x_t)
        x_t = x_t + self.drop(self.ff(h2))
        return x_t


class WayTransformerAR(nn.Module):
    """
    Way-space autoregressive baseline (Transformer, causal).

    Uses learned way ID embeddings; no way semantic features.
    """

    def __init__(self, *, cfg: WayTransformerARCfg) -> None:
        super().__init__()
        self.cfg = cfg
        d = int(cfg.d_model)
        self.way_emb = nn.Embedding(int(cfg.n_ways), d)
        self.pos_emb = nn.Embedding(int(cfg.max_len) + 1, d)
        self.cond_enc = ConditionEncoder(
            ConditionEncoderCfg(d_model=d, n_route_cities=int(cfg.n_route_cities), coord_scale=float(cfg.coord_scale))
        )
        self.cond_to_tok = nn.Linear(d, d, bias=False)
        self.blocks = nn.ModuleList([_Block(d_model=d, n_heads=int(cfg.n_heads), dropout=float(cfg.dropout)) for _ in range(int(cfg.n_layers))])
        self.out_ln = nn.LayerNorm(d)

        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)

    def to(self, device: torch.device) -> "WayTransformerAR":
        super().to(device)
        return self

    def encode_cond(self, route_cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.cond_enc(
            start_pos=route_cond["start_pos"],
            dest_pos=route_cond["dest_pos"],
            hour=route_cond["hour"],
            dow=route_cond["dow"],
            route_city=route_cond["route_city"],
        )

    def forward_tokens(self, way_in: torch.Tensor, *, route_cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            way_in: (B,T) long, -1 padded
        Returns:
            h: (B,T,D)
        """
        device = way_in.device
        B, T = way_in.shape
        ids = torch.clamp(way_in.to(dtype=torch.long), min=0)
        tok = self.way_emb(ids)

        pos = torch.arange(int(T), device=device, dtype=torch.long).clamp(max=int(self.cfg.max_len))
        tok = tok + self.pos_emb(pos)[None, :, :]

        cond = self.encode_cond(route_cond)  # (B,D)
        tok = tok + self.cond_to_tok(cond)[:, None, :]

        key_pad = (way_in < 0)  # (B,T)
        if bool(key_pad.any()):
            tok = tok.masked_fill(key_pad[:, :, None], 0.0)

        x = tok
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_pad)
        return self.out_ln(x)

    def score_candidates(self, token: torch.Tensor, cand_way: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(token)  # (B,D)
        ids = torch.clamp(cand_way.to(dtype=torch.long), min=0)
        cand = self.way_emb(ids)  # (B,C,D)
        k = self.k_proj(cand)  # (B,C,D)
        logits = (q[:, None, :] * k).sum(dim=-1)
        logits = logits.masked_fill(~cand_mask.to(dtype=torch.bool), float("-inf"))
        return logits

    def _alloc_cache(self, *, B: int, max_len: int, device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        H = int(self.cfg.n_heads)
        dh = int(self.cfg.d_model) // H
        k_list = [torch.zeros((B, H, int(max_len) + 1, dh), dtype=torch.float32, device=device) for _ in self.blocks]
        v_list = [torch.zeros((B, H, int(max_len) + 1, dh), dtype=torch.float32, device=device) for _ in self.blocks]
        return k_list, v_list

    def _step_with_cache(
        self,
        x_t: torch.Tensor,  # (B,D)
        *,
        k_list: List[torch.Tensor],
        v_list: List[torch.Tensor],
        t: int,
    ) -> torch.Tensor:
        h = x_t
        for li, blk in enumerate(self.blocks):
            h = blk.step(h, k_cache=k_list[li], v_cache=v_list[li], t=int(t))
        return self.out_ln(h)

    @torch.no_grad()
    def greedy_decode(
        self,
        *,
        way_adj_ptr: np.ndarray,
        way_adj_idx: np.ndarray,
        start_way: int,
        dest_way: int,
        route_cond: Dict[str, torch.Tensor],
        max_len: Optional[int] = None,
        max_candidates: Optional[int] = None,
    ) -> List[int]:
        self.eval()
        device = next(self.parameters()).device
        max_len = int(max_len) if max_len is not None else int(self.cfg.max_len)
        mc = int(self.cfg.max_candidates) if max_candidates is None else int(max_candidates)

        rc = {
            "start_pos": route_cond["start_pos"].reshape(1, 2).to(device=device),
            "dest_pos": route_cond["dest_pos"].reshape(1, 2).to(device=device),
            "hour": route_cond["hour"].reshape(1).to(device=device),
            "dow": route_cond["dow"].reshape(1).to(device=device),
            "route_city": route_cond["route_city"].reshape(1).to(device=device),
        }
        cond = self.encode_cond(rc)  # (1,D)

        k_list, v_list = self._alloc_cache(B=1, max_len=max_len, device=device)
        cur = int(start_way)
        out: List[int] = [cur]
        for t in range(int(max_len)):
            if int(cur) == int(dest_way):
                break

            # Consume current token at position t.
            x_t = self.way_emb(torch.as_tensor([int(cur)], dtype=torch.long, device=device))
            x_t = x_t + self.pos_emb(torch.as_tensor([int(t)], dtype=torch.long, device=device))
            x_t = x_t + self.cond_to_tok(cond)
            token = self._step_with_cache(x_t, k_list=k_list, v_list=v_list, t=int(t))  # (1,D)

            succ = slice_csr(way_adj_ptr, way_adj_idx, int(cur))
            if succ.size == 0:
                break
            if mc == 0:
                cand_np = succ
            else:
                cand_np = succ[:mc]
            cand = torch.as_tensor(cand_np.reshape(1, -1), dtype=torch.long, device=device)
            cand_mask = torch.ones_like(cand, dtype=torch.bool, device=device)
            logits = self.score_candidates(token, cand, cand_mask)
            j = int(torch.argmax(logits[0]).item())
            cur = int(cand[0, j].item())
            out.append(cur)
        return out

    @torch.no_grad()
    def beam_search(
        self,
        *,
        way_adj_ptr: np.ndarray,
        way_adj_idx: np.ndarray,
        start_way: int,
        dest_way: int,
        route_cond: Dict[str, torch.Tensor],
        beam_size: int = 10,
        max_len: Optional[int] = None,
        max_candidates: Optional[int] = None,
    ) -> List[int]:
        self.eval()
        device = next(self.parameters()).device
        max_len = int(max_len) if max_len is not None else int(self.cfg.max_len)
        beam_size = max(1, int(beam_size))
        mc = int(self.cfg.max_candidates) if max_candidates is None else int(max_candidates)

        rc = {
            "start_pos": route_cond["start_pos"].reshape(1, 2).to(device=device),
            "dest_pos": route_cond["dest_pos"].reshape(1, 2).to(device=device),
            "hour": route_cond["hour"].reshape(1).to(device=device),
            "dow": route_cond["dow"].reshape(1).to(device=device),
            "route_city": route_cond["route_city"].reshape(1).to(device=device),
        }
        cond = self.encode_cond(rc)  # (1,D)

        # Each beam keeps its own KV caches (preallocated) and sequence.
        beams: List[Tuple[float, List[int], List[torch.Tensor], List[torch.Tensor]]] = []
        k0, v0 = self._alloc_cache(B=1, max_len=max_len, device=device)
        beams.append((0.0, [int(start_way)], k0, v0))

        for t in range(int(max_len)):
            new_beams: List[Tuple[float, List[int], List[torch.Tensor], List[torch.Tensor]]] = []
            for score, seq, k_list, v_list in beams:
                cur = int(seq[-1])
                if int(cur) == int(dest_way):
                    new_beams.append((float(score), list(seq), k_list, v_list))
                    continue

                # Consume current token at position t.
                x_t = self.way_emb(torch.as_tensor([int(cur)], dtype=torch.long, device=device))
                x_t = x_t + self.pos_emb(torch.as_tensor([int(t)], dtype=torch.long, device=device))
                x_t = x_t + self.cond_to_tok(cond)
                token = self._step_with_cache(x_t, k_list=k_list, v_list=v_list, t=int(t))  # (1,D)

                succ = slice_csr(way_adj_ptr, way_adj_idx, int(cur))
                if succ.size == 0:
                    new_beams.append((float(score), list(seq), k_list, v_list))
                    continue
                if mc == 0:
                    cand_np = succ
                else:
                    cand_np = succ[:mc]
                cand = torch.as_tensor(cand_np.reshape(1, -1), dtype=torch.long, device=device)
                cand_mask = torch.ones_like(cand, dtype=torch.bool, device=device)
                logits = self.score_candidates(token, cand, cand_mask)
                logp = F.log_softmax(logits[0], dim=-1)

                k = min(int(beam_size), int(logp.numel()))
                topv, topi = torch.topk(logp, k=k, dim=-1)
                for j in range(int(k)):
                    nxt = int(cand[0, int(topi[j])].item())
                    # Clone caches for child beams (they will diverge at next step).
                    kk = [x.clone() for x in k_list]
                    vv = [x.clone() for x in v_list]
                    new_beams.append((float(score + float(topv[j].item())), seq + [nxt], kk, vv))

            new_beams.sort(key=lambda x: float(x[0]), reverse=True)
            beams = new_beams[: int(beam_size)]
            if all(int(b[1][-1]) == int(dest_way) for b in beams):
                break

        def _key(b: Tuple[float, List[int], List[torch.Tensor], List[torch.Tensor]]) -> Tuple[int, float]:
            succ = 0 if int(b[1][-1]) == int(dest_way) else 1
            return (succ, -float(b[0]))

        best = min(beams, key=_key) if beams else (0.0, [int(start_way)], k0, v0)
        return list(best[1])

    def state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}

    def ckpt_payload(self) -> Dict[str, object]:
        return {"cfg": asdict(self.cfg), "model_state_dict": self.state_dict_cpu()}

