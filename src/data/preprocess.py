"""
preprocess.py - 轨迹数据预处理模块

主要功能：
1. Trip 切分（基于载客状态或时间间隔）
2. 坐标转换（经纬度 -> 栅格）
3. 异常值过滤
4. 数据归一化
"""

import numpy as np
import pandas as pd
import torch
from typing import Union, Tuple, Dict, List, Optional
from tqdm import tqdm

from src.config.settings import NormalizationConfig, NORM, GRID, TRIP_CONFIG
from src.data.raw_io import latlon_to_grid, filter_by_bounds


# ============================================================
# Trip 切分
# ============================================================

def segment_trips_by_status(df: pd.DataFrame, 
                            min_length: int = None,
                            min_distance: float = None) -> List[pd.DataFrame]:
    """
    基于载客状态（status）切分 trip。
    
    一个完整的载客行程：status 从 0->1（上客）到 1->0（下客）
    
    Args:
        df: 单辆车的 GPS DataFrame，需包含 status 列
        min_length: 最小轨迹长度（点数）
        min_distance: 最小行程距离（米）
    
    Returns:
        Trip DataFrames 列表
    """
    if min_length is None:
        min_length = TRIP_CONFIG.min_trip_length
    if min_distance is None:
        min_distance = TRIP_CONFIG.min_trip_distance
        
    trips = []
    
    # 检测状态变化点
    status_diff = df['status'].diff()
    
    # 上客点：status 从 0 变为 1
    pickup_indices = df.index[status_diff == 1].tolist()
    # 下客点：status 从 1 变为 0
    dropoff_indices = df.index[status_diff == -1].tolist()
    
    # 处理边界情况：如果数据以载客状态开始
    if len(df) > 0 and df.iloc[0]['status'] == 1:
        pickup_indices = [df.index[0]] + pickup_indices
    
    # 匹配上下客点
    for pickup_idx in pickup_indices:
        # 找到下一个下客点
        valid_dropoffs = [d for d in dropoff_indices if d > pickup_idx]
        if not valid_dropoffs:
            continue
        dropoff_idx = valid_dropoffs[0]
        
        # 提取 trip
        trip_df = df.loc[pickup_idx:dropoff_idx].copy()
        
        # 长度过滤
        if len(trip_df) < min_length:
            continue
        
        # 距离过滤（使用经纬度近似计算）
        if min_distance > 0:
            dist = haversine_distance(
                trip_df.iloc[0]['lat'], trip_df.iloc[0]['lon'],
                trip_df.iloc[-1]['lat'], trip_df.iloc[-1]['lon']
            )
            if dist < min_distance:
                continue
        
        trips.append(trip_df)
    
    return trips


def segment_trips_by_time(df: pd.DataFrame,
                          max_gap: float = None,
                          min_length: int = None) -> List[pd.DataFrame]:
    """
    基于时间间隔切分 trip。
    
    当两个连续点之间的时间间隔超过 max_gap 时，切分为新的 trip。
    
    Args:
        df: 单辆车的 GPS DataFrame
        max_gap: 最大时间间隔（秒）
        min_length: 最小轨迹长度
    
    Returns:
        Trip DataFrames 列表
    """
    if max_gap is None:
        max_gap = TRIP_CONFIG.max_time_gap
    if min_length is None:
        min_length = TRIP_CONFIG.min_trip_length
    
    trips = []
    
    # 计算时间差
    time_diff = df['timestamp_unix'].diff()
    
    # 找到切分点
    split_indices = df.index[time_diff > max_gap].tolist()
    
    # 添加首尾
    all_splits = [df.index[0]] + split_indices + [df.index[-1] + 1]
    
    for i in range(len(all_splits) - 1):
        start_idx = all_splits[i]
        end_idx = all_splits[i + 1]
        
        trip_df = df.loc[start_idx:end_idx - 1].copy()
        
        if len(trip_df) >= min_length:
            trips.append(trip_df)
    
    return trips


def extract_all_trips(df: pd.DataFrame,
                      use_status: bool = None,
                      show_progress: bool = True) -> List[Dict]:
    """
    从 GPS 数据中提取所有 trip。
    
    Args:
        df: 合并后的 GPS DataFrame
        use_status: 是否使用载客状态切分
        show_progress: 是否显示进度
    
    Returns:
        Trip 字典列表，每个包含 positions, timestamps, vehicle_id 等
    """
    if use_status is None:
        use_status = TRIP_CONFIG.use_status_for_segmentation
    
    all_trips = []
    vehicle_ids = df['vehicle_id'].unique()
    
    iterator = tqdm(vehicle_ids, desc="Extracting trips") if show_progress else vehicle_ids
    
    for vid in iterator:
        vehicle_df = df[df['vehicle_id'] == vid].sort_values('timestamp_unix')
        
        if use_status and 'status' in vehicle_df.columns:
            trips = segment_trips_by_status(vehicle_df)
        else:
            trips = segment_trips_by_time(vehicle_df)
        
        for trip_df in trips:
            trip_dict = process_trip_dataframe(trip_df, vid)
            if trip_dict is not None:
                all_trips.append(trip_dict)
    
    print(f"Extracted {len(all_trips)} valid trips from {len(vehicle_ids)} vehicles")
    return all_trips


def process_trip_dataframe(trip_df: pd.DataFrame, 
                           vehicle_id: str) -> Optional[Dict]:
    """
    将单个 trip DataFrame 转换为标准字典格式。
    
    Args:
        trip_df: Trip DataFrame
        vehicle_id: 车辆 ID
    
    Returns:
        标准化的 trip 字典，或 None（如果无效）
    """
    # 过滤边界外的点
    trip_df = filter_by_bounds(trip_df)
    
    if len(trip_df) < TRIP_CONFIG.min_trip_length:
        return None
    
    # 坐标转换
    y, x = latlon_to_grid(trip_df['lat'].values, trip_df['lon'].values)
    positions = np.stack([y, x], axis=1).astype(np.float32)  # (T, 2) [y, x]
    
    # 时间戳
    timestamps = trip_df['timestamp_unix'].values.astype(np.int64)
    
    # 检测并移除空间跳跃异常
    positions, timestamps = remove_spatial_jumps(positions, timestamps)
    
    if len(positions) < TRIP_CONFIG.min_trip_length:
        return None
    
    return {
        'positions': positions,      # (T, 2) [y, x] 栅格坐标
        'timestamp': timestamps,     # (T,) Unix 时间戳
        'vehicle_id': hash(vehicle_id) % (2**31),  # 转为 int
        'origin': positions[0],      # 起点
        'destination': positions[-1] # 终点
    }


def remove_spatial_jumps(positions: np.ndarray, 
                         timestamps: np.ndarray,
                         max_jump: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    移除空间跳跃异常点。
    
    Args:
        positions: (T, 2) 位置数组
        timestamps: (T,) 时间戳数组
        max_jump: 最大跳跃距离（栅格单位）
    
    Returns:
        过滤后的 (positions, timestamps)
    """
    if max_jump is None:
        # 将米转换为栅格单位（近似）
        # 深圳：1度约111km，栅格分辨率约 0.001度/像素
        max_jump = TRIP_CONFIG.max_spatial_jump / 111000 * GRID.H / (GRID.max_lat - GRID.min_lat)
    
    if len(positions) < 2:
        return positions, timestamps
    
    # 计算连续点之间的距离
    diffs = np.diff(positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    
    # 找到有效点（第一个点总是有效）
    valid_mask = np.ones(len(positions), dtype=bool)
    valid_mask[1:] = distances < max_jump
    
    return positions[valid_mask], timestamps[valid_mask]


# ============================================================
# 辅助函数
# ============================================================

def haversine_distance(lat1: float, lon1: float, 
                       lat2: float, lon2: float) -> float:
    """
    计算两点之间的 Haversine 距离（米）。
    """
    R = 6371000  # 地球半径（米）
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat / 2) ** 2 + \
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


# ============================================================
# 归一化类
# ============================================================

class Normalizer:
    """
    处理轨迹数据的归一化和反归一化。
    确保神经网络输入分布良好。
    """
    def __init__(self, config: NormalizationConfig = NORM):
        self.config = config
        
        self.pos_min = np.array(config.pos_min)
        self.pos_max = np.array(config.pos_max)
        self.pos_range = self.pos_max - self.pos_min + 1e-6
        
        self.vel_mean = np.array(config.vel_mean)
        self.vel_std = np.array(config.vel_std) + 1e-6
        
        self.nav_scale = config.nav_scale

    def normalize_pos(self, pos: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """位置归一化：[0, H] -> [-1, 1]"""
        return 2 * (pos - self.pos_min) / self.pos_range - 1

    def denormalize_pos(self, pos: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """位置反归一化：[-1, 1] -> [0, H]"""
        return (pos + 1) / 2 * self.pos_range + self.pos_min

    def normalize_vel(self, vel: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """速度 Z-Score 归一化"""
        return (vel - self.vel_mean) / self.vel_std

    def denormalize_vel(self, vel: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """速度反归一化"""
        return vel * self.vel_std + self.vel_mean

    def normalize_nav(self, nav: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """导航场缩放"""
        return nav * self.nav_scale
    
    @classmethod
    def from_data(cls, trips: List[Dict]) -> 'Normalizer':
        """
        从数据中计算归一化统计量。
        
        Args:
            trips: Trip 字典列表
        
        Returns:
            配置好的 Normalizer 实例
        """
        all_pos = np.concatenate([t['positions'] for t in trips], axis=0)
        
        # 位置边界
        pos_min = all_pos.min(axis=0)
        pos_max = all_pos.max(axis=0)
        
        # 计算速度统计
        all_vel = []
        for t in trips:
            if len(t['positions']) > 1:
                vel = np.diff(t['positions'], axis=0)
                all_vel.append(vel)
        
        if all_vel:
            all_vel = np.concatenate(all_vel, axis=0)
            vel_mean = all_vel.mean(axis=0)
            vel_std = all_vel.std(axis=0)
        else:
            vel_mean = np.array([0.0, 0.0])
            vel_std = np.array([1.0, 1.0])
        
        config = NormalizationConfig(
            pos_min=tuple(pos_min),
            pos_max=tuple(pos_max),
            vel_mean=tuple(vel_mean),
            vel_std=tuple(vel_std)
        )
        
        print(f"Normalization stats computed:")
        print(f"  pos_min: {pos_min}, pos_max: {pos_max}")
        print(f"  vel_mean: {vel_mean}, vel_std: {vel_std}")
        
        return cls(config)


# ============================================================
# 向量对齐检查
# ============================================================

def check_vector_alignment(nav_field_direction: np.ndarray, 
                           trajectories: list, 
                           sample_size: int = 100,
                           threshold: float = 0.0) -> float:
    """
    验证导航场方向与实际轨迹速度的对齐程度。
    用于检测坐标旋转 bug。
    
    Args:
        nav_field_direction: (2, H, W) 单位向量 [nav_y, nav_x]
        trajectories: trip 字典列表
        sample_size: 采样数量
        threshold: 警告阈值
    
    Returns:
        平均余弦相似度
    """
    total_sim = 0.0
    count = 0
    
    indices = np.random.choice(len(trajectories), min(len(trajectories), sample_size), replace=False)
    
    for idx in indices:
        traj = trajectories[idx]
        pos = traj['positions']  # (T, 2)
        
        # 计算速度
        if len(pos) < 2:
            continue
        vel = np.diff(pos, axis=0)  # (T-1, 2)
        pos = pos[:-1]  # 对齐
        
        # 有效步（速度不为零）
        valid_mask = np.linalg.norm(vel, axis=1) > 0.1
        if not np.any(valid_mask):
            continue
            
        pos_valid = pos[valid_mask]
        vel_valid = vel[valid_mask]
        
        # 查询导航方向
        y = np.clip(pos_valid[:, 0].astype(int), 0, nav_field_direction.shape[1]-1)
        x = np.clip(pos_valid[:, 1].astype(int), 0, nav_field_direction.shape[2]-1)
        
        nav_vectors = nav_field_direction[:, y, x].T  # (N, 2)
        
        # 计算余弦相似度
        nav_norm = np.linalg.norm(nav_vectors, axis=1, keepdims=True) + 1e-6
        vel_norm = np.linalg.norm(vel_valid, axis=1, keepdims=True) + 1e-6
        
        sim = np.sum((nav_vectors * vel_valid), axis=1) / (nav_norm.squeeze() * vel_norm.squeeze())
        
        total_sim += np.mean(sim)
        count += 1
        
    avg_sim = total_sim / max(count, 1)
    
    if avg_sim < threshold:
        print(f"WARNING: Low vector alignment ({avg_sim:.3f}). Check coordinate rotations!")
    
    return avg_sim
