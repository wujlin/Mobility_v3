import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from src.models.base_model import BaseTrajectoryModel
from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.models.physics.cnn_encoder import CNNEncoder

class PhysicsConditionDiffusion(BaseTrajectoryModel):
    """
    Physics-Informed Diffusion Model using Condition Learning.
    Input: History + Global Cond + Nav Patch (encoded).
    """
    def __init__(self, 
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
                 hidden_dim: int = 64,
                 diffusion_steps: int = 100):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.cond_dim = int(cond_dim)
        self.obs_len = int(obs_len)

        self.nav_encoder = CNNEncoder(output_dim=nav_emb_dim, patch_size=nav_patch_size)
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
        
        # Instantiate wrapped diffusion model
        # Condition dim increases by nav_emb_dim
        self.diffusion = DiffusionTrajectoryModel(
            obs_dim=obs_dim,
            act_dim=act_dim,
            cond_dim=cond_dim + nav_emb_dim,
            obs_len=obs_len,
            pred_len=pred_len,
            hidden_dim=hidden_dim,
            diffusion_steps=diffusion_steps
        )

    def _apply_nav_emb(self, obs: torch.Tensor, cond: torch.Tensor, nav_emb: torch.Tensor) -> torch.Tensor:
        if self.nav_emb_scale != 1.0:
            nav_emb = nav_emb * self.nav_emb_scale
        if self.nav_gate is not None:
            B = obs.shape[0]
            gate_in = torch.cat([obs.reshape(B, -1), cond], dim=-1)
            gate = torch.sigmoid(self.nav_gate(gate_in))  # (B, 1) in (0, 1)
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
        return_x0_pred: bool = False,
        return_timesteps: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Compute diffusion loss for physics-conditioned model.

        Args:
            obs: (B, H, 4)
            cond: (B, 6)
            target: (B, F, 2)
            nav_patch: (B, 3, K, K)
            sample_weight: optional per-sample weight for diffusion loss (shape: B,)
            return_x0_pred: if True, also return x0_pred (B, act_dim, F)
            return_timesteps: if True, also return timesteps (B,)
        """
        if nav_patch is None:
            raise ValueError("Nav Patch is required for Physics Model")

        nav_emb = self.nav_encoder(nav_patch)  # (B, nav_emb_dim)
        nav_emb = self._apply_nav_emb(obs, cond, nav_emb)
        full_cond = torch.cat([cond, nav_emb], dim=-1)
        return self.diffusion.compute_loss(
            obs,
            full_cond,
            target,
            sample_weight=sample_weight,
            return_x0_pred=return_x0_pred,
            return_timesteps=return_timesteps,
        )

    def forward(self, obs, cond, target=None, nav_patch=None, sample_weight: Optional[torch.Tensor] = None):
        """
        obs: (B, H, 4)
        cond: (B, 6)
        target: (B, F, 2)
        nav_patch: (B, 3, K, K)
        """
        if nav_patch is None:
            raise ValueError("Nav Patch is required for Physics Model")
            
        # Encode Nav
        nav_emb = self.nav_encoder(nav_patch) # (B, nav_emb_dim)
        nav_emb = self._apply_nav_emb(obs, cond, nav_emb)
        
        # Concat Cond
        full_cond = torch.cat([cond, nav_emb], dim=-1)
        
        return self.diffusion.forward(obs, full_cond, target, sample_weight=sample_weight)
        
    def sample_trajectory(self, obs, cond, horizon, nav_patch=None, **kwargs):
        if nav_patch is None:
            raise ValueError("Nav Patch is required for Physics Model inference")
            
        # Encode Nav
        nav_emb = self.nav_encoder(nav_patch)
        nav_emb = self._apply_nav_emb(obs, cond, nav_emb)
        
        # Concat Cond
        full_cond = torch.cat([cond, nav_emb], dim=-1)
        
        return self.diffusion.sample_trajectory(obs, full_cond, horizon, **kwargs)

    def to(self, device):
        super().to(device)
        self.nav_encoder.to(device)
        self.diffusion.to(device)
        return self
