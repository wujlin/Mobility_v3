from dataclasses import dataclass
from typing import Tuple

@dataclass
class GridConfig:
    """Dimensions for the rasterized grid."""
    H: int = 400
    W: int = 800
    
    # Geographic bounds (for checking or projection)
    min_lat: float = 22.45
    max_lat: float = 22.85
    min_lon: float = 113.75
    max_lon: float = 114.65

@dataclass
class NormalizationConfig:
    """
    Statistics for normalization.
    Values should be computed from the training set.
    """
    # Position bounds (usually 0 to H/W)
    pos_min: Tuple[float, float] = (0.0, 0.0) # [y, x]
    pos_max: Tuple[float, float] = (400.0, 800.0)
    
    # Velocity statistics
    # From data_stats.json
    vel_mean: Tuple[float, float] = (-0.08, -0.04)
    vel_std: Tuple[float, float] = (3.23, 4.82)
    
    # Nav scale
    nav_scale: float = 1.0
    nav_max_speed: float = 20.0 # Approximate max speed for normalization

# Global instance for easy access, 
# in real usage this might be loaded from a yaml
GRID = GridConfig()
NORM = NormalizationConfig()
