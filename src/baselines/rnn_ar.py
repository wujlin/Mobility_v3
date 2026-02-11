from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.way_casd.conditions import ConditionEncoder, ConditionEncoderCfg
from src.utils.way_csr import slice_csr


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

    def _route_cond_batch_to_device(self, route_cond: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "start_pos": route_cond["start_pos"].to(device=device).reshape(-1, 2),
            "dest_pos": route_cond["dest_pos"].to(device=device).reshape(-1, 2),
            "hour": route_cond["hour"].to(device=device).reshape(-1),
            "dow": route_cond["dow"].to(device=device).reshape(-1),
            "route_city": route_cond["route_city"].to(device=device).reshape(-1),
        }

    def _build_cand_tensors(
        self,
        *,
        way_adj_ptr: np.ndarray,
        way_adj_idx: np.ndarray,
        cur_way: np.ndarray,
        max_candidates: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows: List[np.ndarray] = []
        max_c = 0
        for c in cur_way.tolist():
            succ = slice_csr(way_adj_ptr, way_adj_idx, int(c))
            if max_candidates == 0:
                cand = succ.astype(np.int64, copy=False)
            else:
                cand = succ[: int(max_candidates)].astype(np.int64, copy=False)
            rows.append(cand)
            if int(cand.size) > max_c:
                max_c = int(cand.size)
        if max_c <= 0:
            cand_t = torch.full((len(rows), 1), -1, dtype=torch.long, device=device)
            mask_t = torch.zeros((len(rows), 1), dtype=torch.bool, device=device)
            return cand_t, mask_t
        cand_np = np.full((len(rows), int(max_c)), -1, dtype=np.int64)
        mask_np = np.zeros((len(rows), int(max_c)), dtype=np.bool_)
        for i, cand in enumerate(rows):
            n = int(cand.size)
            if n <= 0:
                continue
            cand_np[i, :n] = cand
            mask_np[i, :n] = True
        cand_t = torch.as_tensor(cand_np, dtype=torch.long, device=device)
        mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        return cand_t, mask_t

    def _build_cand_tensors_from_pad(
        self,
        *,
        succ_pad: torch.Tensor,
        succ_mask: torch.Tensor,
        cur_way: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cur_idx = torch.as_tensor(cur_way.astype(np.int64, copy=False), dtype=torch.long, device=succ_pad.device)
        cur_idx = torch.clamp(cur_idx, min=0)
        cand = succ_pad[cur_idx].clone()
        mask = succ_mask[cur_idx].clone().to(dtype=torch.bool)
        return cand, mask

    @torch.no_grad()
    def greedy_decode_batch(
        self,
        *,
        way_adj_ptr: np.ndarray,
        way_adj_idx: np.ndarray,
        start_way: Sequence[int],
        dest_way: Sequence[int],
        route_cond: Dict[str, torch.Tensor],
        max_len: Optional[int] = None,
        max_candidates: Optional[int] = None,
        succ_pad: Optional[torch.Tensor] = None,
        succ_mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        self.eval()
        device = next(self.parameters()).device
        max_len = int(max_len) if max_len is not None else int(self.cfg.max_len)
        mc = int(self.cfg.max_candidates) if max_candidates is None else int(max_candidates)

        sw = np.asarray(start_way, dtype=np.int64).reshape(-1)
        dw = np.asarray(dest_way, dtype=np.int64).reshape(-1)
        B = int(sw.size)
        if B == 0:
            return []

        rc = self._route_cond_batch_to_device(route_cond, device=device)
        cond = self.encode_cond(rc)  # (B,D)
        h = self.init_state(cond)  # (L,B,D)

        cur = sw.copy()
        out: List[List[int]] = [[int(x)] for x in sw.tolist()]
        active = np.ones((B,), dtype=np.bool_)

        for _ in range(int(max_len)):
            active &= (cur != dw)
            if not bool(np.any(active)):
                break
            idx = np.nonzero(active)[0].astype(np.int64, copy=False)
            cur_act = cur[idx]
            if succ_pad is not None and succ_mask is not None and mc > 0:
                cand_t, mask_t = self._build_cand_tensors_from_pad(succ_pad=succ_pad, succ_mask=succ_mask, cur_way=cur_act)
            else:
                cand_t, mask_t = self._build_cand_tensors(
                    way_adj_ptr=way_adj_ptr,
                    way_adj_idx=way_adj_idx,
                    cur_way=cur_act,
                    max_candidates=mc,
                    device=device,
                )
            has_cand = torch.any(mask_t, dim=1).detach().cpu().numpy().astype(np.bool_, copy=False)
            if not bool(np.any(has_cand)):
                active[idx] = False
                continue

            idx_c = idx[has_cand]
            cur_c = torch.as_tensor(cur[idx_c], dtype=torch.long, device=device)
            cond_c = cond[idx_c]
            h_c = h[:, idx_c, :].contiguous()
            token_c, h_new = self.step(cur_c, cond_emb=cond_c, h=h_c)
            h[:, idx_c, :] = h_new

            cand_c = cand_t[torch.as_tensor(has_cand, dtype=torch.bool, device=device)]
            mask_c = mask_t[torch.as_tensor(has_cand, dtype=torch.bool, device=device)]
            logits = self.score_candidates(token_c, cand_c, mask_c)
            pick = torch.argmax(logits, dim=1)
            nxt = cand_c[torch.arange(int(cand_c.size(0)), device=device), pick].detach().cpu().numpy().astype(np.int64, copy=False)
            for j, rid in enumerate(idx_c.tolist()):
                n = int(nxt[j])
                cur[rid] = n
                out[rid].append(n)

            idx_noc = idx[~has_cand]
            if idx_noc.size > 0:
                active[idx_noc] = False
        return out

    @torch.no_grad()
    def beam_search_batch(
        self,
        *,
        way_adj_ptr: np.ndarray,
        way_adj_idx: np.ndarray,
        start_way: Sequence[int],
        dest_way: Sequence[int],
        route_cond: Dict[str, torch.Tensor],
        beam_size: int = 10,
        max_len: Optional[int] = None,
        max_candidates: Optional[int] = None,
        state_batch_size: int = 4096,
        succ_pad: Optional[torch.Tensor] = None,
        succ_mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        self.eval()
        device = next(self.parameters()).device
        max_len = int(max_len) if max_len is not None else int(self.cfg.max_len)
        beam_size = max(1, int(beam_size))
        mc = int(self.cfg.max_candidates) if max_candidates is None else int(max_candidates)
        state_batch_size = max(64, int(state_batch_size))

        sw = np.asarray(start_way, dtype=np.int64).reshape(-1)
        dw = np.asarray(dest_way, dtype=np.int64).reshape(-1)
        B = int(sw.size)
        if B == 0:
            return []

        rc = self._route_cond_batch_to_device(route_cond, device=device)
        cond_all = self.encode_cond(rc)  # (B,D)
        init_h_all = self.init_state(cond_all)  # (L,B,D)

        beams: List[List[Tuple[float, List[int], torch.Tensor]]] = []
        for i in range(B):
            beams.append([(0.0, [int(sw[i])], init_h_all[:, i : i + 1, :].clone())])

        for _ in range(int(max_len)):
            done_cnt = 0
            new_beams: List[List[Tuple[float, List[int], torch.Tensor]]] = [[] for _ in range(B)]

            expand_items: List[Tuple[int, float, List[int], torch.Tensor, int]] = []
            for rid in range(B):
                route_done = True
                for score, seq, h in beams[rid]:
                    cur = int(seq[-1])
                    if cur == int(dw[rid]):
                        new_beams[rid].append((float(score), list(seq), h))
                        continue
                    route_done = False
                    expand_items.append((rid, float(score), seq, h, cur))
                if route_done:
                    done_cnt += 1

            if done_cnt == B:
                beams = new_beams
                break

            for st in range(0, len(expand_items), state_batch_size):
                chunk = expand_items[st : st + state_batch_size]
                if not chunk:
                    continue
                cur_np = np.asarray([c[4] for c in chunk], dtype=np.int64)
                rid_np = np.asarray([c[0] for c in chunk], dtype=np.int64)
                if succ_pad is not None and succ_mask is not None and mc > 0:
                    cand_t, mask_t = self._build_cand_tensors_from_pad(succ_pad=succ_pad, succ_mask=succ_mask, cur_way=cur_np)
                else:
                    cand_t, mask_t = self._build_cand_tensors(
                        way_adj_ptr=way_adj_ptr,
                        way_adj_idx=way_adj_idx,
                        cur_way=cur_np,
                        max_candidates=mc,
                        device=device,
                    )
                has_cand = torch.any(mask_t, dim=1).detach().cpu().numpy().astype(np.bool_, copy=False)
                if not bool(np.any(has_cand)):
                    continue

                idx_has = np.nonzero(has_cand)[0].astype(np.int64, copy=False)
                rid_has = rid_np[idx_has]
                cur_has = torch.as_tensor(cur_np[idx_has], dtype=torch.long, device=device)
                cond_has = cond_all[torch.as_tensor(rid_has, dtype=torch.long, device=device)]
                h_has = torch.cat([chunk[i][3] for i in idx_has.tolist()], dim=1).contiguous()  # (L,N,D)
                token_has, h2_has = self.step(cur_has, cond_emb=cond_has, h=h_has)
                cand_has = cand_t[torch.as_tensor(has_cand, dtype=torch.bool, device=device)]
                mask_has = mask_t[torch.as_tensor(has_cand, dtype=torch.bool, device=device)]
                logits = self.score_candidates(token_has, cand_has, mask_has)
                logp = F.log_softmax(logits, dim=-1)

                topk = min(int(beam_size), int(logp.size(1)))
                topv, topi = torch.topk(logp, k=topk, dim=-1)
                cand_cpu = cand_has.detach().cpu().numpy()
                topv_cpu = topv.detach().cpu().numpy()
                topi_cpu = topi.detach().cpu().numpy()

                for local_j, src_j in enumerate(idx_has.tolist()):
                    rid = int(rid_has[local_j])
                    base_score, base_seq = float(chunk[src_j][1]), list(chunk[src_j][2])
                    h_child = h2_has[:, local_j : local_j + 1, :]
                    for kk in range(int(topk)):
                        nxt = int(cand_cpu[local_j, int(topi_cpu[local_j, kk])])
                        sc = float(base_score + float(topv_cpu[local_j, kk]))
                        new_beams[rid].append((sc, base_seq + [nxt], h_child.clone()))

            for rid in range(B):
                if len(new_beams[rid]) == 0:
                    new_beams[rid] = beams[rid]
                    continue
                new_beams[rid].sort(key=lambda x: float(x[0]), reverse=True)
                new_beams[rid] = new_beams[rid][: int(beam_size)]
            beams = new_beams

        outs: List[List[int]] = []
        for i in range(B):
            cand = beams[i]
            if not cand:
                outs.append([int(sw[i])])
                continue
            cand.sort(key=lambda b: (0 if int(b[1][-1]) == int(dw[i]) else 1, -float(b[0])))
            outs.append(list(cand[0][1]))
        return outs

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
                    new_beams.append((float(score + float(topv[j].item())), seq + [nxt], h2.clone()))

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
