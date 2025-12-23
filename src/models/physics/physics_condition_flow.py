import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.base_model import BaseTrajectoryModel
from src.models.flow.rectified_flow_model import RectifiedFlowTrajectoryModel
from src.models.physics.cnn_encoder import CNNEncoder


class PhysicsConditionFlow(BaseTrajectoryModel):
    """
    Physics-conditioned Rectified Flow model (NavField patch as a side condition).

    This mirrors PhysicsConditionDiffusion, but the core generator is RectifiedFlowTrajectoryModel.
    """

    def __init__(
        self,
        *,
        obs_dim: int = 4,
        act_dim: int = 2,
        cond_dim: int = 6,
        nav_patch_size: int = 32,
        nav_emb_dim: int = 32,
        nav_emb_scale: float = 1.0,
        nav_emb_dropout: float = 0.0,
        nav_gate: str = "none",
        nav_gate_hidden: int = 32,
        nav_gate_dropout: float = 0.0,
        obs_len: int = 8,
        pred_len: int = 12,
        hidden_dim: int = 128,
        time_scale: float = 1000.0,
        noise_sigma: float = 1.0,
        solver_steps: int = 20,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.cond_dim = int(cond_dim)
        self.obs_len = int(obs_len)

        self.nav_encoder = CNNEncoder(output_dim=int(nav_emb_dim), patch_size=int(nav_patch_size))
        self.nav_emb_scale = float(nav_emb_scale)
        self.nav_emb_dropout = float(nav_emb_dropout)

        nav_gate = str(nav_gate)
        if nav_gate not in ("none", "obscond"):
            raise ValueError(f"Unknown nav_gate: {nav_gate} (expected: none|obscond)")
        self.nav_gate_mode = nav_gate
        self.nav_gate_dropout = float(nav_gate_dropout)
        self.nav_gate: Optional[nn.Module] = None
        if self.nav_gate_mode != "none":
            gate_in_dim = self.obs_len * self.obs_dim + self.cond_dim
            h = int(nav_gate_hidden)
            self.nav_gate = nn.Sequential(
                nn.Linear(gate_in_dim, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )

        self.flow = RectifiedFlowTrajectoryModel(
            obs_dim=int(obs_dim),
            act_dim=int(act_dim),
            cond_dim=int(cond_dim) + int(nav_emb_dim),
            obs_len=int(obs_len),
            pred_len=int(pred_len),
            hidden_dim=int(hidden_dim),
            time_scale=float(time_scale),
            noise_sigma=float(noise_sigma),
            solver_steps=int(solver_steps),
        )

    def set_noise_sigma(self, sigma: float) -> None:
        self.flow.set_noise_sigma(float(sigma))

    def _apply_nav_emb(self, obs: torch.Tensor, cond: torch.Tensor, nav_emb: torch.Tensor) -> torch.Tensor:
        if self.nav_emb_scale != 1.0:
            nav_emb = nav_emb * float(self.nav_emb_scale)
        if self.nav_gate is not None:
            B = obs.shape[0]
            gate_in = torch.cat([obs.reshape(B, -1), cond], dim=-1)
            gate = torch.sigmoid(self.nav_gate(gate_in))
            if self.nav_gate_dropout > 0.0:
                gate = F.dropout(gate, p=float(self.nav_gate_dropout), training=self.training)
            nav_emb = nav_emb * gate
        if self.nav_emb_dropout > 0.0:
            nav_emb = F.dropout(nav_emb, p=float(self.nav_emb_dropout), training=self.training)
        return nav_emb

    def compute_loss(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor,
        target: torch.Tensor,
        *,
        nav_patch: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if nav_patch is None:
            raise ValueError("Nav Patch is required for PhysicsConditionFlow")
        nav_emb = self.nav_encoder(nav_patch)
        nav_emb = self._apply_nav_emb(obs, cond, nav_emb)
        full_cond = torch.cat([cond, nav_emb], dim=-1)
        return self.flow.compute_loss(obs, full_cond, target, sample_weight=sample_weight)

    def forward(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        *,
        nav_patch: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target is None:
            return torch.tensor(0.0, device=obs.device)
        if nav_patch is None:
            raise ValueError("Nav Patch is required for PhysicsConditionFlow")
        return self.compute_loss(obs, cond, target, nav_patch=nav_patch, sample_weight=sample_weight)

    def sample_trajectory(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor,
        horizon: int,
        *,
        nav_patch: Optional[torch.Tensor] = None,
        cond_uncond: Optional[torch.Tensor] = None,
        cfg_scale: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        if nav_patch is None:
            raise ValueError("Nav Patch is required for PhysicsConditionFlow inference")

        nav_emb_base = self.nav_encoder(nav_patch)

        nav_emb = self._apply_nav_emb(obs, cond, nav_emb_base)
        full_cond = torch.cat([cond, nav_emb], dim=-1)

        full_cond_uncond = None
        if cond_uncond is not None:
            nav_emb_u = self._apply_nav_emb(obs, cond_uncond, nav_emb_base)
            full_cond_uncond = torch.cat([cond_uncond, nav_emb_u], dim=-1)

        return self.flow.sample_trajectory(
            obs,
            full_cond,
            int(horizon),
            cond_uncond=full_cond_uncond,
            cfg_scale=float(cfg_scale),
            **kwargs,
        )

    def to(self, device):
        super().to(device)
        self.nav_encoder.to(device)
        self.flow.to(device)
        return self

