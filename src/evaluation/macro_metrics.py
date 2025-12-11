import torch
import numpy as np
from typing import Dict, Any

def compute_msd_curve(trajectories: torch.Tensor) -> np.ndarray:
    """
    Compute MSD vs Time Lag.
    MSD(tau) = < |r(t+tau) - r(t)|^2 >
    Args:
        trajectories: (B, T, 2)
    Returns:
        msd array of shape (T-1,)
    """
    B, T, D = trajectories.shape
    msds = []
    
    # Simple loop for lags (optimized: vectorization usually possible but loop is clear)
    for lag in range(1, T):
        # diff: (B, T-lag, 2)
        diff = trajectories[:, lag:] - trajectories[:, :-lag]
        sq_dist = (diff ** 2).sum(dim=-1) # (B, T-lag)
        msd_lag = sq_dist.mean() # Ensemble and Time average
        msds.append(msd_lag.item())
        
    return np.array(msds)

def compute_radius_of_gyration(trajectories: torch.Tensor) -> float:
    """
    Compute Radius of Gyration.
    Rog = sqrt( 1/T * sum |r_t - r_mean|^2 )
    Args:
        trajectories: (B, T, 2)
    Returns:
        mean Rog over batch
    """
    # Center of mass per trajectory
    # mean_pos: (B, 1, 2)
    mean_pos = trajectories.mean(dim=1, keepdim=True)
    
    # Sq dist from CoM
    # (B, T, 2)
    diff = trajectories - mean_pos
    sq_dist_sum = (diff ** 2).sum(dim=-1).mean(dim=1) # (B,) Mean over T
    
    rog = torch.sqrt(sq_dist_sum).mean() # Mean over Batch
    return rog.item()

def compute_macro_metrics(pred: torch.Tensor) -> Dict[str, Any]:
    """
    Compute macro stats for generated trajectories.
    Args:
        pred: (B, F, 2)
    """
    metrics = {}
    
    # 1. MSD Curve
    msd_curve = compute_msd_curve(pred)
    # Store raw curve or sampled points?
    # Store sampled for summary
    metrics['MSD_1'] = msd_curve[0] if len(msd_curve) > 0 else 0
    metrics['MSD_5'] = msd_curve[4] if len(msd_curve) > 4 else 0
    metrics['MSD_10'] = msd_curve[9] if len(msd_curve) > 9 else 0
    
    metrics['msd_curve'] = msd_curve # Expect caller to handle array in JSON if saving
    
    # 2. Radius of Gyration
    metrics['Rog'] = compute_radius_of_gyration(pred)
    
    return metrics
