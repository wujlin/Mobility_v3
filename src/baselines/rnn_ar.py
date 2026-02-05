from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.way_casd.conditions import ConditionEncoder, ConditionEncoderCfg
from src.utils.way_csr import build_candidate_row, slice_csr


@dataclass(frozen=True)
class WayRNNARCfg:
    n_ways: int
    d_model: int = 256
    n_layers: int = 2
    dropout: float = 0.1
    max_candidates: int = 32
    max_len: int = 160
    n_route_cities: int = 4
    coord_scale: float = 1024.0


class WayRNNAR(nn.Module):
    """
    Way-space autoregressive baseline (RNN).

    Design choices (per experiment protocol):
      - way embedding: nn.Embedding(n_ways, d_model)
      - candidate policy: successors[:max_candidates] ("first")
      - training: teacher forcing with CE over candidate set (target is forced into candidates)
    """

    def __init__(self, *, cfg: WayRNNARCfg) -> None:
        super().__init__()
        self.cfg = cfg

        d = int(cfg.d_model)
        self.way_emb = nn.Embedding(int(cfg.n_ways), d)
        self.cond_enc = ConditionEncoder(
            ConditionEncoderCfg(d_model=d, n_route_cities=int(cfg.n_route_cities), coord_scale=float(cfg.coord_scale))
        )
        self.cond_to_inp = nn.Linear(d, d, bias=False)
        self.cond_to_h0 = nn.Linear(d, int(cfg.n_layers) * d, bias=True)
        self.rnn = nn.GRU(
            input_size=d,
            hidden_size=d,
            num_layers=int(cfg.n_layers),
            batch_first=True,
            dropout=(float(cfg.dropout) if int(cfg.n_layers) > 1 else 0.0),
        )
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)

    def to(self, device: torch.device) -> "WayRNNAR":
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

    def init_state(self, cond_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cond_emb: (B,d_model)
        Returns:
            h0: (n_layers,B,d_model)
        """
        B, d = cond_emb.shape
        h = self.cond_to_h0(cond_emb).reshape(int(self.cfg.n_layers), B, d).contiguous()
        return torch.tanh(h)

    def step(self, cur_way: torch.Tensor, *, cond_emb: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One autoregressive step: consume cur_way, update RNN state, produce token feature.

        Args:
            cur_way: (B,) long
            cond_emb: (B,d_model)
            h: (n_layers,B,d_model)
        Returns:
            token: (B,d_model)
            h_new: (n_layers,B,d_model)
        """
        cur = torch.clamp(cur_way.to(dtype=torch.long), min=0)
        x = self.way_emb(cur) + self.cond_to_inp(cond_emb)
        out, h_new = self.rnn(x[:, None, :], h.contiguous())
        token = out[:, 0, :]
        return token, h_new

    def score_candidates(self, token: torch.Tensor, cand_way: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token: (B,d_model)
            cand_way: (B,C) long, -1 padded
            cand_mask: (B,C) bool
        Returns:
            logits: (B,C) float
        """
        q = self.q_proj(token)  # (B,D)
        ids = torch.clamp(cand_way.to(dtype=torch.long), min=0)
        cand = self.way_emb(ids)  # (B,C,D)
        k = self.k_proj(cand)  # (B,C,D)
        logits = (q[:, None, :] * k).sum(dim=-1)  # (B,C)
        logits = logits.masked_fill(~cand_mask.to(dtype=torch.bool), float("-inf"))
        return logits

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

        # Single-route cond to (1, ...)
        rc = {
            "start_pos": route_cond["start_pos"].reshape(1, 2).to(device=device),
            "dest_pos": route_cond["dest_pos"].reshape(1, 2).to(device=device),
            "hour": route_cond["hour"].reshape(1).to(device=device),
            "dow": route_cond["dow"].reshape(1).to(device=device),
            "route_city": route_cond["route_city"].reshape(1).to(device=device),
        }
        cond = self.encode_cond(rc)
        h = self.init_state(cond)

        cur = int(start_way)
        out: List[int] = [cur]
        for _t in range(int(max_len)):
            if int(cur) == int(dest_way):
                break
            succ = slice_csr(way_adj_ptr, way_adj_idx, int(cur))
            if succ.size == 0:
                break
            if mc == 0:
                cand_np = succ
            else:
                cand_np = succ[:mc]

            cand = torch.as_tensor(cand_np.reshape(1, -1), dtype=torch.long, device=device)
            cand_mask = torch.ones_like(cand, dtype=torch.bool, device=device)
            token, h = self.step(torch.as_tensor([int(cur)], dtype=torch.long, device=device), cond_emb=cond, h=h)
            logits = self.score_candidates(token, cand, cand_mask)
            nxt = int(torch.argmax(logits[0]).item())
            cur = int(cand[0, nxt].item())
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
        cond = self.encode_cond(rc)

        init_h = self.init_state(cond)
        beams: List[Tuple[float, List[int], torch.Tensor]] = [(0.0, [int(start_way)], init_h)]

        for _t in range(int(max_len)):
            new_beams: List[Tuple[float, List[int], torch.Tensor]] = []
            for score, seq, h in beams:
                cur = int(seq[-1])
                if int(cur) == int(dest_way):
                    new_beams.append((float(score), list(seq), h))
                    continue
                succ = slice_csr(way_adj_ptr, way_adj_idx, int(cur))
                if succ.size == 0:
                    new_beams.append((float(score), list(seq), h))
                    continue
                if mc == 0:
                    cand_np = succ
                else:
                    cand_np = succ[:mc]
                cand = torch.as_tensor(cand_np.reshape(1, -1), dtype=torch.long, device=device)
                cand_mask = torch.ones_like(cand, dtype=torch.bool, device=device)

                token, h2 = self.step(torch.as_tensor([int(cur)], dtype=torch.long, device=device), cond_emb=cond, h=h)
                logits = self.score_candidates(token, cand, cand_mask)
                logp = F.log_softmax(logits[0], dim=-1)  # (C,)

                k = min(int(beam_size), int(logp.numel()))
                topv, topi = torch.topk(logp, k=k, dim=-1)
                for j in range(int(k)):
                    nxt = int(cand[0, int(topi[j])].item())
                    new_beams.append((float(score + float(topv[j].item())), seq + [nxt], h2))

            new_beams.sort(key=lambda x: float(x[0]), reverse=True)
            beams = new_beams[: int(beam_size)]
            if all(int(b[1][-1]) == int(dest_way) for b in beams):
                break

        # Prefer success, then best score.
        def _key(b: Tuple[float, List[int], torch.Tensor]) -> Tuple[int, float]:
            succ = 0 if int(b[1][-1]) == int(dest_way) else 1
            return (succ, -float(b[0]))

        best = min(beams, key=_key) if beams else (0.0, [int(start_way)], init_h)
        return list(best[1])

    def state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}

    def ckpt_payload(self) -> Dict[str, object]:
        return {"cfg": asdict(self.cfg), "model_state_dict": self.state_dict_cpu()}

