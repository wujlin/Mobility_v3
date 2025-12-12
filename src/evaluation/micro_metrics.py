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

import numpy as np
from scipy.spatial.distance import cdist

def compute_frechet(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Discrete Frechet Distance between two trajectories.
    Using iterative dynamic programming.
    """
    # Convert to numpy (B, F, 2)
    P = pred.detach().cpu().numpy()
    Q = target.detach().cpu().numpy()
    
    batch_size = P.shape[0]
    total_dist = 0.0
    
    for i in range(batch_size):
        traj_p = P[i] # (F, 2)
        traj_q = Q[i] # (F, 2)
        
        # Pairwise distances
        # M[i, j] = dist(p_i, q_j)
        dist_matrix = cdist(traj_p, traj_q, metric='euclidean')
        n_p, n_q = dist_matrix.shape
        
        ca = np.ones((n_p, n_q)) * -1.0
        
        # DP Initialization
        ca[0, 0] = dist_matrix[0, 0]
        
        for k in range(1, n_p):
            ca[k, 0] = max(ca[k-1, 0], dist_matrix[k, 0])
            
        for l in range(1, n_q):
            ca[0, l] = max(ca[0, l-1], dist_matrix[0, l])
            
        # DP Fill
        for k in range(1, n_p):
            for l in range(1, n_q):
                ca[k, l] = max(
                    dist_matrix[k, l],
                    min(ca[k-1, l], ca[k, l-1], ca[k-1, l-1])
                )
        
        total_dist += ca[n_p-1, n_q-1]
        
    return total_dist / batch_size

def compute_dtw(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Dynamic Time Warping (DTW) distance.
    """
    P = pred.detach().cpu().numpy()
    Q = target.detach().cpu().numpy()
    
    batch_size = P.shape[0]
    total_dist = 0.0
    
    for i in range(batch_size):
        traj_p = P[i]
        traj_q = Q[i]
        
        dist_matrix = cdist(traj_p, traj_q, metric='euclidean')
        n_p, n_q = dist_matrix.shape
        
        # Cost Accumulation Matrix
        # Initialize with infinity
        D = np.zeros((n_p + 1, n_q + 1))
        D[:] = np.inf
        D[0, 0] = 0
        
        for k in range(1, n_p + 1):
            for l in range(1, n_q + 1):
                cost = dist_matrix[k-1, l-1]
                D[k, l] = cost + min(D[k-1, l], D[k, l-1], D[k-1, l-1])
                
        total_dist += D[n_p, n_q]
        
    return total_dist / batch_size

def compute_micro_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute standard trajectory prediction metrics.
    Args:
        pred: (B, F, 2) Predicted positions
        target: (B, F, 2) Ground Truth positions
    """
    metrics = {}
    
    # 1. ADE / FDE (Vectorized, Fast)
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
            
    # 3. Shape Metrics (Loop-based, Slower)
    # Only compute if batch size is reasonable or requested?
    # For now, always compute.
    metrics['Frechet'] = compute_frechet(pred, target)
    metrics['DTW'] = compute_dtw(pred, target)
            
    return metrics
