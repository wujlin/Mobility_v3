from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.way_casd.conditions import ConditionEncoder, ConditionEncoderCfg
from src.utils.way_csr import slice_csr


@dataclass(frozen=True)
class GTGCostNetCfg:
    n_ways: int
    d_model: int = 256
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.1
    max_candidates: int = 32
    n_route_cities: int = 4
    coord_scale: float = 1024.0
    cost_eps: float = 1e-3


class GTGCostNet(nn.Module):
    """
    Simplified GTG reproduction (Phase 1, KISS):
      - Learn a per-edge cost: c(u->v | route_cond)
      - Training: next-hop CE over candidate successors (target forced into candidates)
      - Inference: Dijkstra on directed way graph using learned costs
    """

    def __init__(self, *, cfg: GTGCostNetCfg) -> None:
        super().__init__()
        self.cfg = cfg
        d = int(cfg.d_model)
        h = int(cfg.hidden_dim)

        self.way_emb = nn.Embedding(int(cfg.n_ways), d)
        self.cond_enc = ConditionEncoder(
            ConditionEncoderCfg(d_model=d, n_route_cities=int(cfg.n_route_cities), coord_scale=float(cfg.coord_scale))
        )
        layers = []
        in_dim = 3 * d
        for i in range(int(cfg.n_layers)):
            out_dim = h if i < int(cfg.n_layers) - 1 else 1
            layers.append(nn.Linear(in_dim if i == 0 else h, out_dim))
            if i < int(cfg.n_layers) - 1:
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(float(cfg.dropout)))
        self.edge_mlp = nn.Sequential(*layers)

    def to(self, device: torch.device) -> "GTGCostNet":
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

    def edge_cost(
        self,
        *,
        u: torch.Tensor,  # (N,)
        v: torch.Tensor,  # (N,)
        cond_emb: torch.Tensor,  # (N,D) or (1,D)
    ) -> torch.Tensor:
        uu = torch.clamp(u.to(dtype=torch.long), min=0)
        vv = torch.clamp(v.to(dtype=torch.long), min=0)
        eu = self.way_emb(uu)
        ev = self.way_emb(vv)
        if cond_emb.ndim == 2 and int(cond_emb.shape[0]) == 1 and int(u.shape[0]) > 1:
            cond = cond_emb.expand(int(u.shape[0]), -1)
        else:
            cond = cond_emb
        x = torch.cat([eu, ev, cond], dim=-1)
        raw = self.edge_mlp(x).reshape(-1)
        cost = F.softplus(raw) + float(self.cfg.cost_eps)
        return cost

    def score_candidates(
        self,
        *,
        cur_way: torch.Tensor,  # (B,)
        cand_way: torch.Tensor,  # (B,C)
        cand_mask: torch.Tensor,  # (B,C)
        cond_emb: torch.Tensor,  # (B,D)
    ) -> torch.Tensor:
        B, C = cand_way.shape
        u = cur_way.to(dtype=torch.long)[:, None].expand(B, C).reshape(B * C)
        v = cand_way.to(dtype=torch.long).reshape(B * C)
        m = cand_mask.to(dtype=torch.bool).reshape(B * C)

        cost = torch.zeros((B * C,), dtype=torch.float32, device=cond_emb.device)
        if bool(m.any()):
            cost[m] = self.edge_cost(u=u[m], v=v[m], cond_emb=cond_emb.repeat_interleave(C, dim=0)[m])
        logits = (-cost).reshape(B, C)
        logits = logits.masked_fill(~cand_mask.to(dtype=torch.bool), float("-inf"))
        return logits

    @torch.no_grad()
    def edge_costs_numpy(
        self,
        *,
        u: int,
        v_list: np.ndarray,  # (K,)
        route_cond: Dict[str, torch.Tensor],
    ) -> np.ndarray:
        """
        Compute edge costs for a single source u to multiple v in one forward.
        Returns numpy float64 array (K,).
        """
        self.eval()
        device = next(self.parameters()).device
        vv = np.asarray(v_list, dtype=np.int64).reshape(-1)
        if vv.size == 0:
            return np.zeros((0,), dtype=np.float64)

        rc = {
            "start_pos": route_cond["start_pos"].reshape(1, 2).to(device=device),
            "dest_pos": route_cond["dest_pos"].reshape(1, 2).to(device=device),
            "hour": route_cond["hour"].reshape(1).to(device=device),
            "dow": route_cond["dow"].reshape(1).to(device=device),
            "route_city": route_cond["route_city"].reshape(1).to(device=device),
        }
        cond = self.encode_cond(rc)  # (1,D)
        u_t = torch.full((int(vv.size),), int(u), dtype=torch.long, device=device)
        v_t = torch.as_tensor(vv, dtype=torch.long, device=device)
        cost = self.edge_cost(u=u_t, v=v_t, cond_emb=cond)  # (K,)
        return cost.detach().cpu().numpy().astype(np.float64, copy=False)

    @torch.no_grad()
    def greedy_decode(
        self,
        *,
        way_adj_ptr: np.ndarray,
        way_adj_idx: np.ndarray,
        start_way: int,
        dest_way: int,
        route_cond: Dict[str, torch.Tensor],
        max_len: int = 160,
        max_candidates: Optional[int] = None,
    ) -> list[int]:
        """
        Greedy rollout under learned local costs (not GTG search, provided for debugging).
        """
        self.eval()
        device = next(self.parameters()).device
        mc = int(self.cfg.max_candidates) if max_candidates is None else int(max_candidates)

        rc = {
            "start_pos": route_cond["start_pos"].reshape(1, 2).to(device=device),
            "dest_pos": route_cond["dest_pos"].reshape(1, 2).to(device=device),
            "hour": route_cond["hour"].reshape(1).to(device=device),
            "dow": route_cond["dow"].reshape(1).to(device=device),
            "route_city": route_cond["route_city"].reshape(1).to(device=device),
        }
        cond = self.encode_cond(rc)

        cur = int(start_way)
        out = [cur]
        for _t in range(int(max_len)):
            if int(cur) == int(dest_way):
                break
            succ = slice_csr(way_adj_ptr, way_adj_idx, int(cur))
            if succ.size == 0:
                break
            cand_np = succ if mc == 0 else succ[:mc]
            cand = torch.as_tensor(cand_np.reshape(1, -1), dtype=torch.long, device=device)
            cand_mask = torch.ones_like(cand, dtype=torch.bool, device=device)
            cur_t = torch.as_tensor([int(cur)], dtype=torch.long, device=device)
            logits = self.score_candidates(cur_way=cur_t, cand_way=cand, cand_mask=cand_mask, cond_emb=cond)
            j = int(torch.argmax(logits[0]).item())
            cur = int(cand[0, j].item())
            out.append(cur)
        return out

    def state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}

    def ckpt_payload(self) -> Dict[str, object]:
        return {"cfg": asdict(self.cfg), "model_state_dict": self.state_dict_cpu()}

