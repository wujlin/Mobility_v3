from dataclasses import dataclass
from typing import Tuple

@dataclass
class GridConfig:
    """Dimensions for the rasterized grid."""
    H: int = 256  # Example size
    W: int = 256
    
    # Geographic bounds (for checking or projection)
    min_lat: float = 30.0
    max_lat: float = 32.0
    min_lon: float = 120.0
    max_lon: float = 122.0

@dataclass
class NormalizationConfig:
    """
    Statistics for normalization.
    Values should be computed from the training set.
    """
    # Position bounds (usually 0 to H/W)
    pos_min: Tuple[float, float] = (0.0, 0.0) # [y, x]
    pos_max: Tuple[float, float] = (256.0, 256.0)
    
    # Velocity statistics
    vel_mean: Tuple[float, float] = (0.0, 0.0)
    vel_std: Tuple[float, float] = (1.0, 1.0)
    
    # Nav scale
    nav_scale: float = 1.0

# Global instance for easy access, 
# in real usage this might be loaded from a yaml
GRID = GridConfig()
NORM = NormalizationConfig()
