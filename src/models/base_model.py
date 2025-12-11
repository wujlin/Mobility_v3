import torch
import torch.nn as nn
from typing import Dict, Optional

class BaseTrajectoryModel(nn.Module):
    """Abstract base class for all trajectory generation/prediction models."""
    
    def forward(self, obs: torch.Tensor, cond: torch.Tensor, target: Optional[torch.Tensor] = None):
        """
        Forward pass for training.
        Args:
            obs: (B, H, obs_dim) History [pos, vel, etc]
            cond: (B, cond_dim) Conditional [time, OD]
            target: (B, F, target_dim) Future Ground Truth (for training)
        Returns:
            output (Tensor) or loss (if target provided)
        """
        raise NotImplementedError

    def sample_trajectory(self, 
                          obs: torch.Tensor, 
                          cond: torch.Tensor, 
                          horizon: int,
                          **kwargs) -> torch.Tensor:
        """
        Generate future trajectory.
        Args:
            obs: (B, H, obs_dim)
            cond: (B, cond_dim)
            horizon: Int, length of future F
        Returns:
            generated: (B, horizon, 2) Position or Velocity depending on model type
        """
        raise NotImplementedError
