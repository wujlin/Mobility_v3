import torch
import torch.nn as nn
from typing import Tuple, Union
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
                 obs_len: int = 8,
                 pred_len: int = 12,
                 hidden_dim: int = 64,
                 diffusion_steps: int = 100):
        super().__init__()
        
        self.nav_encoder = CNNEncoder(output_dim=nav_emb_dim, patch_size=nav_patch_size)
        
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

    def compute_loss(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor,
        target: torch.Tensor,
        *,
        nav_patch: torch.Tensor,
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
            return_x0_pred: if True, also return x0_pred (B, act_dim, F)
            return_timesteps: if True, also return timesteps (B,)
        """
        if nav_patch is None:
            raise ValueError("Nav Patch is required for Physics Model")

        nav_emb = self.nav_encoder(nav_patch)  # (B, nav_emb_dim)
        full_cond = torch.cat([cond, nav_emb], dim=-1)
        return self.diffusion.compute_loss(
            obs,
            full_cond,
            target,
            return_x0_pred=return_x0_pred,
            return_timesteps=return_timesteps,
        )

    def forward(self, obs, cond, target=None, nav_patch=None):
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
        
        # Concat Cond
        full_cond = torch.cat([cond, nav_emb], dim=-1)
        
        return self.diffusion.forward(obs, full_cond, target)
        
    def sample_trajectory(self, obs, cond, horizon, nav_patch=None, **kwargs):
        if nav_patch is None:
            raise ValueError("Nav Patch is required for Physics Model inference")
            
        # Encode Nav
        nav_emb = self.nav_encoder(nav_patch)
        
        # Concat Cond
        full_cond = torch.cat([cond, nav_emb], dim=-1)
        
        return self.diffusion.sample_trajectory(obs, full_cond, horizon, **kwargs)

    def to(self, device):
        super().to(device)
        self.nav_encoder.to(device)
        self.diffusion.to(device)
        return self
