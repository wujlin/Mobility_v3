from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.models.way_casd.gps_diffusion import GPSDiffusionExecutionModel, GPSDiffusionCfg
from src.models.way_casd.latent_flow import LatentFlowMatching, LatentFlowCfg
from src.models.way_casd.way_casd import WayCASDAutoEncoder, WayCASDAECfg
from src.models.way_casd.way_encoder import WayFeatureTensors


@dataclass(frozen=True)
class CascadeCfg:
    # Decision stage (latent flow + constrained decode)
    ae: WayCASDAECfg = WayCASDAECfg()
    flow: LatentFlowCfg = LatentFlowCfg()
    # Execution stage (GPS diffusion)
    exec: GPSDiffusionCfg = GPSDiffusionCfg()


class WayCASDCascade(nn.Module):
    """
    Full pipeline (Decision + Execution):

    - Decision: sample skeleton latent tokens z via flow, then decode to way sequence.
    - Execution: sample continuous (y,x) trajectory conditioned on z + route_cond.

    NOTE: This module is inference-oriented. Training should be done per-stage:
      Step A: train WayCASDAutoEncoder
      Step B: train LatentFlowMatching
      Step C: train GPSDiffusionExecutionModel (teacher-forced using GT z from AE.encode)
    """

    def __init__(
        self,
        *,
        cfg: CascadeCfg,
        way_features: WayFeatureTensors,
        way_adj_ptr,
        way_adj_idx,
        n_highway_types: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.ae = WayCASDAutoEncoder(
            cfg=cfg.ae,
            way_features=way_features,
            way_adj_ptr=way_adj_ptr,
            way_adj_idx=way_adj_idx,
            n_highway_types=int(n_highway_types),
        )
        self.flow = LatentFlowMatching(cfg=cfg.flow, cond_cfg=self.ae.decoder.cond_enc.cfg)
        self.exec = GPSDiffusionExecutionModel(cfg=cfg.exec)

    @torch.no_grad()
    def sample(
        self,
        *,
        route_cond: Dict[str, torch.Tensor],
        start_way: torch.Tensor,
        dest_way: torch.Tensor,
        decode: str = "greedy",  # greedy|beam
        beam_size: int = 5,
        max_decode_len: Optional[int] = None,
        traj_len: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Returns:
          {
            "z": (B,L,d),
            "way_seq": List[List[int]],
            "traj_yx_rel": (B,T,2),
          }
        """
        z = self.flow.sample(route_cond=route_cond)
        if str(decode) == "beam":
            way_seq = self.ae.decoder.beam_search(
                way_embedder=self.ae.way_enc,
                latent_tokens=z,
                route_cond=route_cond,
                start_way=start_way,
                dest_way=dest_way,
                beam_size=int(beam_size),
                max_len=max_decode_len,
            )
        else:
            way_seq = self.ae.decoder.greedy_decode(
                way_embedder=self.ae.way_enc,
                latent_tokens=z,
                route_cond=route_cond,
                start_way=start_way,
                dest_way=dest_way,
                max_len=max_decode_len,
            )
        traj = self.exec.sample(route_cond=route_cond, skeleton_latent=z, traj_len=traj_len)
        return {"z": z, "way_seq": way_seq, "traj_yx_rel": traj}

