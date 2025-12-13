"""
轨迹重采样模块

将不规则采样的 GPS 轨迹重采样到固定时间间隔，确保：
1. dt 有明确物理语义（论文版严格评估必需）
2. MSD 标度律计算的 Δt 可解释
3. nav_field 的 speed（步位移模长）可跨时间段/数据集对比

注意：本仓库 v1 的 `vel` 语义为 step displacement（步位移）：
- vel = Δpos（单位 grid_cell/step）
- 需要物理速度时：physical_velocity = vel / dt_fixed（单位 grid_cell/second）

Usage:
    from src.data.resample import resample_trajectory
    
    resampled = resample_trajectory(positions, timestamps, dt=30.0)
"""

from typing import Optional, Tuple

import numpy as np


def resample_trajectory(
    positions: np.ndarray,
    timestamps: np.ndarray,
    dt: float = 30.0,
    min_length: int = 10
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    将不规则采样轨迹重采样到固定时间间隔
    
    Args:
        positions: (T, 2) 原始位置序列 [y, x]
        timestamps: (T,) Unix 时间戳序列（秒，int64）
        dt: 目标采样间隔（秒）
        min_length: 重采样后的最小长度，过短则返回 None
        
    Returns:
        resampled_positions: (T_new, 2) 或 None
        resampled_timestamps: (T_new,) 或 None
    """
    if len(positions) < 2:
        return None, None
    
    ts0 = int(timestamps[0])
    t = (timestamps.astype(np.int64) - ts0).astype(np.int64)
    if np.any(np.diff(t) < 0):
        return None, None

    # 去重：同一秒多点取均值（避免插值要求严格递增）
    uniq_t, inv = np.unique(t, return_inverse=True)
    if uniq_t.size < 2:
        return None, None
    if uniq_t.size != t.size:
        counts = np.bincount(inv).astype(np.float32)
        sums = np.zeros((uniq_t.size, 2), dtype=np.float64)
        np.add.at(sums, inv, positions.astype(np.float64))
        positions = (sums / counts[:, None]).astype(np.float32)
        t = uniq_t.astype(np.int64)
    
    # 计算新的时间序列
    t_max = t[-1]
    if t_max < dt:
        return None, None
    
    dt = float(dt)
    t_new = np.arange(0.0, float(t_max) + 1e-6, dt)
    
    if len(t_new) < min_length:
        return None, None
    
    # 线性插值
    try:
        xp = t.astype(np.float64)
        y = positions[:, 0].astype(np.float64)
        x = positions[:, 1].astype(np.float64)

        new_y = np.interp(t_new, xp, y).astype(np.float32)
        new_x = np.interp(t_new, xp, x).astype(np.float32)
        resampled_positions = np.stack([new_y, new_x], axis=1).astype(np.float32)
        resampled_timestamps = (t_new + float(ts0)).astype(np.int64)

        return resampled_positions, resampled_timestamps
        
    except Exception as e:
        print(f"Interpolation failed: {e}")
        return None, None


def compute_velocity_from_resampled(
    positions: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    从重采样后的位置计算真实速度
    
    Args:
        positions: (T, 2) 重采样后的位置
        dt: 采样间隔（秒）
        
    Returns:
        velocity: (T-1, 2) 速度 [dy/dt, dx/dt]，单位: grid_cells/second
    """
    displacement = np.diff(positions, axis=0)  # (T-1, 2)
    velocity = displacement / dt
    return velocity


def compute_displacement_from_resampled(
    positions: np.ndarray
) -> np.ndarray:
    """
    从重采样后的位置计算步位移（与现有代码兼容）
    
    Args:
        positions: (T, 2) 重采样后的位置
        
    Returns:
        displacement: (T-1, 2) 步位移 [Δy, Δx]，单位: grid_cells/step
    """
    return np.diff(positions, axis=0)


def resample_trajectory_batch(
    trajectories: list,
    dt: float = 30.0,
    min_length: int = 10,
    verbose: bool = True
) -> list:
    """
    批量重采样轨迹
    
    Args:
        trajectories: list of dicts, 每个包含:
            - 'positions': (T, 2)
            - 'timestamps': (T,) 秒级时间戳
            - 其他 metadata
        dt: 目标采样间隔
        min_length: 最小长度阈值
        verbose: 是否打印统计信息
        
    Returns:
        resampled_trajectories: list of dicts
    """
    resampled = []
    skipped = 0
    
    for traj in trajectories:
        pos = traj['positions']
        ts = traj['timestamps']
        
        new_pos, new_ts = resample_trajectory(pos, ts, dt=dt, min_length=min_length)
        
        if new_pos is None:
            skipped += 1
            continue
        
        new_traj = {
            'positions': new_pos,
            'timestamps': new_ts,
            'dt': dt,
            # 保留原始 metadata
            **{k: v for k, v in traj.items() if k not in ['positions', 'timestamps']}
        }
        resampled.append(new_traj)
    
    if verbose:
        print(f"Resampled {len(resampled)} trajectories, skipped {skipped} (too short)")
    
    return resampled


def validate_resampling(
    original: np.ndarray,
    resampled: np.ndarray,
    original_ts: np.ndarray,
    resampled_ts: np.ndarray
) -> dict:
    """
    验证重采样质量
    
    Returns:
        dict with validation metrics
    """
    # 计算原始轨迹的总长度
    orig_length = np.sum(np.linalg.norm(np.diff(original, axis=0), axis=1))
    resamp_length = np.sum(np.linalg.norm(np.diff(resampled, axis=0), axis=1))
    
    # 起终点误差
    start_error = np.linalg.norm(original[0] - resampled[0])
    end_error = np.linalg.norm(original[-1] - resampled[-1])
    
    # 时间范围
    orig_duration = original_ts[-1] - original_ts[0]
    resamp_duration = resampled_ts[-1] - resampled_ts[0]
    
    return {
        'original_length': orig_length,
        'resampled_length': resamp_length,
        'length_ratio': resamp_length / (orig_length + 1e-6),
        'start_error': start_error,
        'end_error': end_error,
        'original_duration': orig_duration,
        'resampled_duration': resamp_duration,
        'original_points': len(original),
        'resampled_points': len(resampled)
    }
