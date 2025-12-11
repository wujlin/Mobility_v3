import numpy as np
import torch
from typing import Union, Tuple, Dict
from src.config.settings import NormalizationConfig, NORM

class Normalizer:
    """
    Handles normalization and denormalization of trajectory data.
    Ensures input for neural networks is well-distributed.
    """
    def __init__(self, config: NormalizationConfig = NORM):
        self.config = config
        
        # Pre-compute pytorch tensors for faster operation if needed, 
        # but here we stick to numpy/torch compatible logic
        self.pos_min = np.array(config.pos_min)
        self.pos_max = np.array(config.pos_max)
        self.pos_range = self.pos_max - self.pos_min + 1e-6
        
        self.vel_mean = np.array(config.vel_mean)
        self.vel_std = np.array(config.vel_std) + 1e-6
        
        self.nav_scale = config.nav_scale

    def normalize_pos(self, pos: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Convert [0, H] -> [-1, 1]"""
        # x_norm = 2 * (x - min) / (max - min) - 1
        return 2 * (pos - self.pos_min) / self.pos_range - 1

    def denormalize_pos(self, pos: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Convert [-1, 1] -> [0, H]"""
        # x = (x_norm + 1) / 2 * (max - min) + min
        return (pos + 1) / 2 * self.pos_range + self.pos_min

    def normalize_vel(self, vel: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Z-Score Normalization"""
        return (vel - self.vel_mean) / self.vel_std

    def denormalize_vel(self, vel: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return vel * self.vel_std + self.vel_mean

    def normalize_nav(self, nav: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return nav * self.nav_scale


def check_vector_alignment(nav_field_direction: np.ndarray, 
                           trajectories: list, 
                           sample_size: int = 100,
                           threshold: float = 0.0) -> float:
    """
    Verify that nav field directions align with actual trajectory velocities.
    Used to catch coordinate rotation bugs.
    
    Args:
        nav_field_direction: (2, H, W) unit vectors [y, x]
        trajectories: list of dicts with 'pos' and 'vel'
    
    Returns:
        mean_cosine_similarity
    """
    total_sim = 0.0
    count = 0
    
    indices = np.random.choice(len(trajectories), min(len(trajectories), sample_size), replace=False)
    
    for idx in indices:
        traj = trajectories[idx]
        pos = traj['pos'] # (T, 2)
        vel = traj['vel'] # (T, 2)
        
        # Valid steps (velocity is not zero)
        valid_mask = np.linalg.norm(vel, axis=1) > 0.1
        if not np.any(valid_mask):
            continue
            
        pos_valid = pos[valid_mask]
        vel_valid = vel[valid_mask]
        
        # Lookup nav direction
        # Ensure integer indices and bounds
        y = np.clip(pos_valid[:, 0].astype(int), 0, nav_field_direction.shape[1]-1)
        x = np.clip(pos_valid[:, 1].astype(int), 0, nav_field_direction.shape[2]-1)
        
        nav_vectors = nav_field_direction[:, y, x].T # (N, 2)
        
        # Compute Cosine Similarity
        # v . n / (|v|*|n|)
        # nav_vectors are already unit (mostly), but let's normalize to be safe
        nav_norm = np.linalg.norm(nav_vectors, axis=1, keepdims=True) + 1e-6
        vel_norm = np.linalg.norm(vel_valid, axis=1, keepdims=True) + 1e-6
        
        sim = np.sum((nav_vectors * vel_valid), axis=1) / (nav_norm.squeeze() * vel_norm.squeeze())
        
        total_sim += np.mean(sim)
        count += 1
        
    avg_sim = total_sim / max(count, 1)
    
    if avg_sim < threshold:
        print(f"WARNING: Low vector alignment ({avg_sim:.3f}). Check coordinate rotations!")
        
    return avg_sim
