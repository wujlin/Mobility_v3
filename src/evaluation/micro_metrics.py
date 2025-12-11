import torch
import torch.nn.functional as F
from typing import Dict

def compute_ade(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Average Displacement Error (L2 distance mean over steps)."""
    # pred, target: (B, F, 2)
    dist = torch.norm(pred - target, p=2, dim=-1) # (B, F)
    return dist.mean().item()

def compute_fde(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Final Displacement Error (L2 distance at last step)."""
    dist = torch.norm(pred[:, -1] - target[:, -1], p=2, dim=-1) # (B,)
    return dist.mean().item()

def compute_micro_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute standard trajectory prediction metrics.
    Args:
        pred: (B, F, 2) Predicted positions
        target: (B, F, 2) Ground Truth positions
    """
    metrics = {}
    
    # 1. ADE / FDE
    metrics['ADE'] = compute_ade(pred, target)
    metrics['FDE'] = compute_fde(pred, target)
    
    # 2. Step-wise MSE (1, 5, 10, ...)
    horizon = pred.shape[1]
    steps = [1, 5, 10, 20]
    for s in steps:
        if s <= horizon:
            idx = s - 1
            mse = F.mse_loss(pred[:, idx], target[:, idx]).item()
            metrics[f'MSE_{s}'] = mse
            
    return metrics
