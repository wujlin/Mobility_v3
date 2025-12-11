from dataclasses import dataclass, field
from typing import Tuple, Dict
from pathlib import Path

# ============================================================
# 深圳市地理范围（从 GPS 数据估计）
# ============================================================
# 经度范围：约 113.75 - 114.65 (东西约 90km)
# 纬度范围：约 22.45 - 22.85 (南北约 45km)

@dataclass
class GridConfig:
    """栅格配置：将经纬度映射到栅格坐标 (y, x)"""
    H: int = 400   # 栅格高度（对应纬度方向）
    W: int = 800   # 栅格宽度（对应经度方向）
    
    # 深圳市地理边界
    min_lat: float = 22.45   # 南边界
    max_lat: float = 22.85   # 北边界
    min_lon: float = 113.75  # 西边界
    max_lon: float = 114.65  # 东边界
    
    @property
    def lat_range(self) -> float:
        return self.max_lat - self.min_lat
    
    @property
    def lon_range(self) -> float:
        return self.max_lon - self.min_lon
    
    @property
    def resolution_lat(self) -> float:
        """每个栅格单元对应的纬度跨度"""
        return self.lat_range / self.H
    
    @property
    def resolution_lon(self) -> float:
        """每个栅格单元对应的经度跨度"""
        return self.lon_range / self.W


@dataclass
class NormalizationConfig:
    """
    归一化配置。
    统计量应从训练集计算得出。
    """
    # 位置边界 [y, x]
    pos_min: Tuple[float, float] = (0.0, 0.0)
    pos_max: Tuple[float, float] = (400.0, 800.0)  # 与 GridConfig 对应
    
    # 速度统计（栅格单位/时间步）
    vel_mean: Tuple[float, float] = (0.0, 0.0)
    vel_std: Tuple[float, float] = (1.0, 1.0)
    
    # 导航场缩放
    nav_scale: float = 1.0


@dataclass
class TripSegmentationConfig:
    """Trip 切分配置"""
    # 最小 trip 长度（点数）
    min_trip_length: int = 10
    
    # 最大时间间隔（秒），超过则切分
    max_time_gap: float = 300.0  # 5分钟
    
    # 最大空间跳跃（米），超过则认为是异常
    max_spatial_jump: float = 1000.0  # 1km
    
    # 最小行程距离（米）
    min_trip_distance: float = 500.0  # 500m
    
    # 使用载客状态切分 trip
    use_status_for_segmentation: bool = True


@dataclass 
class DataPaths:
    """数据路径配置"""
    root: Path = field(default_factory=lambda: Path("data"))
    
    @property
    def raw_gps(self) -> Path:
        return self.root / "raw" / "gps"
    
    @property
    def processed(self) -> Path:
        return self.root / "processed"
    
    @property
    def trajectories(self) -> Path:
        return self.processed / "trajectories"
    
    @property
    def fields(self) -> Path:
        return self.processed / "fields"
    
    @property
    def splits(self) -> Path:
        return self.processed / "splits"


# ============================================================
# 全局配置实例
# ============================================================
GRID = GridConfig()
NORM = NormalizationConfig(
    pos_max=(GRID.H, GRID.W)
)
TRIP_CONFIG = TripSegmentationConfig()
PATHS = DataPaths()
