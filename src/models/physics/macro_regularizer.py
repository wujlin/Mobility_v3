import torch
import torch.nn as nn
from src.evaluation.macro_metrics import compute_msd_curve

class MacroRegularizer(nn.Module):
    """
    Computes macroscopic statistical loss.
    L_macro = || alpha_gen - alpha_target ||^2
    where alpha is the scaling exponent of MSD ~ t^alpha.
    """
    def __init__(self, target_alpha: float = 1.4):
        super().__init__()
        self.target_alpha = target_alpha
        
    def forward(self, generated_traj: torch.Tensor):
        """
        Differentiable estimation of alpha is hard.
        Here we use a simplified proxy: 
        MSD at T should match target MSD assuming power law.
        Or, we compute MSD of generated batch (differentiable?)
        MSD calculation involves (x2 - x1)^2, which IS differentiable.
        
        generated_traj: (B, F, 2)
        """
        # 1. Compute MSD for the batch (keeping gradient)
        # We need MSD at specific lags, e.g. lag=1 and lag=F/2
        B, F, D = generated_traj.shape
        if F < 2:
            return torch.tensor(0.0, device=generated_traj.device)
            
        # Lag 1
        diff1 = generated_traj[:, 1:] - generated_traj[:, :-1]
        msd1 = (diff1**2).sum(dim=-1).mean()
        
        # Lag K (e.g. F-1)
        k = F - 1
        diffK = generated_traj[:, k:] - generated_traj[:, :-k]
        msdK = (diffK**2).sum(dim=-1).mean()
        
        # Alpha approx = log(msdK/msd1) / log(k)
        # Avoid log(0)
        eps = 1e-6
        alpha_gen = torch.log((msdK + eps) / (msd1 + eps)) / torch.log(torch.tensor(k, dtype=torch.float32))
        
        # Loss
        loss = (alpha_gen - self.target_alpha) ** 2
        return loss
