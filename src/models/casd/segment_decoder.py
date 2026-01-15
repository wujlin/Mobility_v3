from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.casd.conditions import ConditionEncoder, ConditionEncoderCfg


@dataclass(frozen=True)
class SegmentDecoderCfg:
    d_model: int = 256
    hidden_dim: int = 256
    max_candidates: int = 16
    dropout: float = 0.1
    max_len: int = 640


class SegmentDecoder(nn.Module):
    """
    Constrained AR decoder over segment IDs using candidate-set scoring.

    Key properties:
      - No full-vocab softmax (n_segments can be ~1M).
      - Adjacency mask is enforced by providing candidate sets.
      - Stop criterion: seg_v == dest_node (no EOS token).
    """

    def __init__(
        self,
        *,
        cfg: SegmentDecoderCfg,
        cond_cfg: ConditionEncoderCfg,
        seg_v,
        seg_succ_ptr,
        seg_succ_idx,
        node_seg_ptr,
        node_seg_idx,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.cond_enc = ConditionEncoder(cond_cfg)

        self.register_buffer("seg_v", torch.as_tensor(seg_v, dtype=torch.long), persistent=False)
        self.register_buffer("seg_succ_ptr", torch.as_tensor(seg_succ_ptr, dtype=torch.long), persistent=False)
        self.register_buffer("seg_succ_idx", torch.as_tensor(seg_succ_idx, dtype=torch.long), persistent=False)
        self.register_buffer("node_seg_ptr", torch.as_tensor(node_seg_ptr, dtype=torch.long), persistent=False)
        self.register_buffer("node_seg_idx", torch.as_tensor(node_seg_idx, dtype=torch.long), persistent=False)

        d_model = int(cfg.d_model)
        hidden = int(cfg.hidden_dim)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.SiLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(hidden, hidden),
        )
        self.cur_proj = nn.Linear(d_model, hidden)
        self.cand_proj = nn.Linear(d_model, hidden)
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
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

    def get_start_candidates(self, start_node: int) -> torch.Tensor:
        return self._slice_csr(self.node_seg_ptr, self.node_seg_idx, int(start_node))

    def get_succ_candidates(self, seg_id: int) -> torch.Tensor:
        return self._slice_csr(self.seg_succ_ptr, self.seg_succ_idx, int(seg_id))

    def is_dest_reached(self, seg_id: int, dest_node: int) -> bool:
        return int(self.seg_v[int(seg_id)].item()) == int(dest_node)

    def score_candidates(
        self,
        *,
        seg_embedder: nn.Module,
        latent_tokens: torch.Tensor,  # (B,L,d_model)
        route_cond: Dict[str, torch.Tensor],
        trans: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Score candidate next segments for a packed set of transitions.

        route_cond keys: start_pos, dest_pos, hour, route_city, corridor_type
        trans keys: route_idx, cur_seg, cand_seg, cand_mask
        """
        route_idx = trans["route_idx"].to(dtype=torch.long)
        cur_seg = trans["cur_seg"].to(dtype=torch.long)
        cand_seg = trans["cand_seg"].to(dtype=torch.long)
        cand_mask = trans["cand_mask"].to(dtype=torch.bool)

        B = int(latent_tokens.shape[0])
        if int(route_idx.max().item()) >= B:
            raise ValueError(f"route_idx out of range: max={int(route_idx.max().item())} but B={B}")

        cond_emb = self.cond_enc(
            start_pos=route_cond["start_pos"],
            dest_pos=route_cond["dest_pos"],
            hour=route_cond["hour"],
            route_city=route_cond["route_city"],
            corridor_type=route_cond.get("corridor_type", None),
        )
        lat_vec = latent_tokens.mean(dim=1)
        ctx = self.ctx_mlp(torch.cat([cond_emb, lat_vec], dim=-1))
        ctx_t = ctx[route_idx]

        cur_emb, _ = seg_embedder(cur_seg[:, None])
        cur_emb = cur_emb[:, 0, :]
        cur_h = self.cur_proj(cur_emb)

        cand_emb, _ = seg_embedder(cand_seg)
        cand_h = self.cand_proj(cand_emb)

        T, C = cand_seg.shape
        ctx_h = ctx_t[:, None, :].expand(T, C, -1)
        cur_h2 = cur_h[:, None, :].expand(T, C, -1)
        x = torch.cat([ctx_h, cur_h2, cand_h], dim=-1)
        logits = self.scorer(x).squeeze(-1)
        logits = logits.masked_fill(~cand_mask, float("-inf"))
        return logits

    @torch.no_grad()
    def beam_search(
        self,
        *,
        seg_embedder: nn.Module,
        latent_tokens: torch.Tensor,  # (B,L,d_model)
        route_cond: Dict[str, torch.Tensor],
        start_node: torch.Tensor,  # (B,)
        dest_node: torch.Tensor,  # (B,)
        beam_size: int = 5,
        max_len: Optional[int] = None,
    ) -> List[List[int]]:
        max_len = int(max_len) if max_len is not None else int(self.cfg.max_len)
        beam_size = max(1, int(beam_size))

        B = int(latent_tokens.shape[0])
        out: List[List[int]] = []
        device = latent_tokens.device

        for b in range(B):
            sn = int(start_node[b].item())
            dn = int(dest_node[b].item())
            beams: List[Tuple[List[int], float]] = [([], 0.0)]

            for _step in range(max_len):
                new_beams: List[Tuple[List[int], float]] = []
                all_finished = True
                for path, score in beams:
                    if path and self.is_dest_reached(path[-1], dn):
                        new_beams.append((path, score))
                        continue
                    all_finished = False

                    if not path:
                        cand = self.get_start_candidates(sn)
                        cur_seg = -1
                    else:
                        cand = self.get_succ_candidates(path[-1])
                        cur_seg = int(path[-1])
                    if int(cand.numel()) == 0:
                        continue

                    C = min(int(cand.numel()), int(self.cfg.max_candidates))
                    cand = cand[:C].to(device=device)
                    cand_seg = cand.view(1, C)
                    cand_mask = torch.ones((1, C), dtype=torch.bool, device=device)
                    trans = {
                        "route_idx": torch.tensor([b], dtype=torch.long, device=device),
                        "cur_seg": torch.tensor([cur_seg], dtype=torch.long, device=device),
                        "cand_seg": cand_seg,
                        "cand_mask": cand_mask,
                    }
                    logits = self.score_candidates(
                        seg_embedder=seg_embedder,
                        latent_tokens=latent_tokens[b : b + 1],
                        route_cond={k: v[b : b + 1].to(device=device) for k, v in route_cond.items()},
                        trans=trans,
                    )[0]

                    logp = F.log_softmax(logits, dim=-1)
                    topk = min(beam_size, int(logp.numel()))
                    vals, ids = torch.topk(logp, k=topk, dim=-1)
                    for lp, j in zip(vals.tolist(), ids.tolist()):
                        new_beams.append((path + [int(cand[j].item())], score + float(lp)))

                if all_finished:
                    break
                new_beams.sort(key=lambda x: -x[1])
                beams = new_beams[:beam_size] if new_beams else beams

            out.append(beams[0][0] if beams else [])

        return out

