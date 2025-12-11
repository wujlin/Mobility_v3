import numpy as np
import torch
from pathlib import Path

class NavField:
    """
    Manages the Navigation Field (Direction + Speed).
    Expected data shape in npz:
    - direction: (2, H, W) unit vectors
    - speed: (H, W) scalar
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = np.load(file_path)
        
        # (2, H, W)
        self.direction = self.data['direction'].astype(np.float32)
        # (H, W)
        if 'speed' in self.data:
            self.speed = self.data['speed'].astype(np.float32)
        else:
            self.speed = np.zeros(self.direction.shape[1:], dtype=np.float32)
            
        self.H, self.W = self.direction.shape[1], self.direction.shape[2]
        
    def get_patch(self, center_pos: np.ndarray, patch_size: int = 32) -> np.ndarray:
        """
        Extract a square patch centered at center_pos [y, x].
        Returns: (3, K, K) array [dir_y, dir_x, speed]
        Handles padding if out of bounds.
        """
        y, x = int(center_pos[0]), int(center_pos[1])
        r = patch_size // 2
        
        # Calculate bounds
        y_min, y_max = y - r, y + r
        x_min, x_max = x - r, x + r
        
        # Prepare canvas
        # Channels: 2 for direction + 1 for speed = 3
        patch = np.zeros((3, patch_size, patch_size), dtype=np.float32)
        
        # Intersection with image
        img_y_min = max(0, y_min)
        img_y_max = min(self.H, y_max)
        img_x_min = max(0, x_min)
        img_x_max = min(self.W, x_max)
        
        # Target in patch
        patch_y_min = img_y_min - y_min
        patch_y_max = patch_y_min + (img_y_max - img_y_min)
        patch_x_min = img_x_min - x_min
        patch_x_max = patch_x_min + (img_x_max - img_x_min)
        
        if img_y_max > img_y_min and img_x_max > img_x_min:
            # Copy Direction
            patch[0:2, patch_y_min:patch_y_max, patch_x_min:patch_x_max] = \
                self.direction[:, img_y_min:img_y_max, img_x_min:img_x_max]
            
            # Copy Speed
            patch[2, patch_y_min:patch_y_max, patch_x_min:patch_x_max] = \
                self.speed[img_y_min:img_y_max, img_x_min:img_x_max]
                
        return patch

    def to_tensor(self):
        """Return full field as tensor"""
        dir_t = torch.from_numpy(self.direction)
        spd_t = torch.from_numpy(self.speed).unsqueeze(0)
        return torch.cat([dir_t, spd_t], dim=0) # (3, H, W)
