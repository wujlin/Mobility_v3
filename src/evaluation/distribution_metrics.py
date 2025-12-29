from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _compute_velocity(
    pos: np.ndarray,
    *,
    dt_s: float = 1.0,
    meters_per_cell: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Compute velocity vectors from positions.

    Args:
        pos: (B, T, 2) positions in grid space [y, x]
        dt_s: time step in seconds. Use 1.0 to keep "per-step" units (grid/step).
        meters_per_cell: optional (meters_per_cell_y, meters_per_cell_x). If provided,
            converts grid displacement to meters before dividing by dt_s.

    Returns:
        vel: (B, T-1, 2)
    """
    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim != 3 or pos.shape[-1] != 2:
        raise ValueError(f"Expected pos (B,T,2), got {pos.shape}")
    if float(dt_s) <= 0:
        raise ValueError(f"dt_s must be > 0, got {dt_s}")

    disp = pos[:, 1:] - pos[:, :-1]  # (B, T-1, 2) in grid cells
    if meters_per_cell is not None:
        my, mx = float(meters_per_cell[0]), float(meters_per_cell[1])
        disp = disp * np.asarray([my, mx], dtype=np.float32)
    return disp / float(dt_s)


def _compute_speed(pos: np.ndarray, *, dt_s: float = 1.0, meters_per_cell: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Speed magnitude: (B, T-1)."""
    vel = _compute_velocity(pos, dt_s=dt_s, meters_per_cell=meters_per_cell)
    return np.linalg.norm(vel, axis=-1)


def _compute_acceleration(
    pos: np.ndarray,
    *,
    dt_s: float = 1.0,
    meters_per_cell: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Acceleration magnitude: (B, T-2)."""
    vel = _compute_velocity(pos, dt_s=dt_s, meters_per_cell=meters_per_cell)
    if vel.shape[1] < 2:
        return np.zeros((vel.shape[0], 0), dtype=np.float32)
    acc = (vel[:, 1:] - vel[:, :-1]) / float(dt_s)
    return np.linalg.norm(acc, axis=-1)


def _compute_turn_angle(
    pos: np.ndarray,
    *,
    min_speed: float = 1e-3,
) -> np.ndarray:
    """
    Unsigned turn angle between consecutive velocity vectors.

    Args:
        pos: (B, T, 2)
        min_speed: filter threshold in displacement units (grid/step). This is
            applied on raw displacement magnitude to avoid undefined headings
            near zero-speed.

    Returns:
        angles: (B, T-2) in radians, in [0, pi]. Entries that fail the speed
            filter are set to NaN (callers should mask/drop them).
    """
    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim != 3 or pos.shape[-1] != 2:
        raise ValueError(f"Expected pos (B,T,2), got {pos.shape}")

    disp = pos[:, 1:] - pos[:, :-1]  # (B, T-1, 2) displacement (not divided by dt)
    if disp.shape[1] < 2:
        return np.zeros((disp.shape[0], 0), dtype=np.float32)

    v1 = disp[:, :-1]  # (B, T-2, 2)
    v2 = disp[:, 1:]   # (B, T-2, 2)
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    valid = (n1 > float(min_speed)) & (n2 > float(min_speed))

    dot = np.sum(v1 * v2, axis=-1)
    cos = dot / (n1 * n2 + 1e-8)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.arccos(cos).astype(np.float32)
    ang[~valid] = np.nan
    return ang


def _normalize_hist(counts: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts + float(eps)
    s = float(np.sum(counts))
    if s <= 0:
        return np.full_like(counts, 1.0 / max(int(counts.size), 1), dtype=np.float64)
    return counts / s


def jsd_from_hist(p: np.ndarray, q: np.ndarray, *, base: float = 2.0) -> float:
    """
    Jensen–Shannon Divergence between two discrete distributions.

    Notes:
        - Returns JSD (not sqrt-JSD). With base=2.0, the range is [0, 1].
    """
    p = _normalize_hist(p)
    q = _normalize_hist(q)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * (np.log(a) - np.log(b))))

    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    if float(base) != float(np.e):
        jsd = jsd / float(np.log(float(base)))
    return float(jsd)


def compute_jsd_from_samples(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    *,
    bins: int = 50,
    value_range: Optional[Tuple[float, float]] = None,
    range_percentiles: Tuple[float, float] = (0.5, 99.5),
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
) -> float:
    """
    Compute JSD between two empirical 1D sample sets via histogramming.

    Args:
        p_samples/q_samples: 1D arrays
        value_range: optional (min,max). If None, use combined percentiles.
    """
    p_samples = np.asarray(p_samples, dtype=np.float64).reshape(-1)
    q_samples = np.asarray(q_samples, dtype=np.float64).reshape(-1)
    if p_samples.size == 0 or q_samples.size == 0:
        return 0.0

    if value_range is None:
        both = np.concatenate([p_samples, q_samples], axis=0)
        lo, hi = np.percentile(both, [float(range_percentiles[0]), float(range_percentiles[1])]).tolist()
        if clamp_min is not None:
            lo = max(float(lo), float(clamp_min))
        if clamp_max is not None:
            hi = min(float(hi), float(clamp_max))
        if not np.isfinite(lo) or not np.isfinite(hi) or float(hi) <= float(lo):
            lo, hi = float(np.min(both)), float(np.max(both))
        if float(hi) <= float(lo):
            hi = float(lo) + 1e-6
        value_range = (float(lo), float(hi))

    p_hist, _ = np.histogram(p_samples, bins=int(bins), range=value_range, density=False)
    q_hist, _ = np.histogram(q_samples, bins=int(bins), range=value_range, density=False)
    return jsd_from_hist(p_hist, q_hist)


def compute_distribution_metrics(
    pred_pos: np.ndarray,
    gt_pos: np.ndarray,
    *,
    dt_s: float = 1.0,
    meters_per_cell: Optional[Tuple[float, float]] = None,
    speed_bins: int = 120,
    accel_bins: int = 120,
    turn_bins: int = 60,
    turn_min_speed: float = 1e-3,
    range_percentiles: Tuple[float, float] = (0.5, 99.5),
) -> Dict[str, float]:
    """
    Standard physical/statistical consistency metrics via JSD.

    Args:
        pred_pos/gt_pos: (B, T, 2) positions in grid space.
        dt_s/meters_per_cell: optional unit conversion for speed/acceleration.
        turn_min_speed: filter threshold to drop near-static steps for turn angles.

    Returns:
        dict with keys: JSD_Speed, JSD_Accel, JSD_TurnAngle
    """
    pred_speed = _compute_speed(pred_pos, dt_s=dt_s, meters_per_cell=meters_per_cell).reshape(-1)
    gt_speed = _compute_speed(gt_pos, dt_s=dt_s, meters_per_cell=meters_per_cell).reshape(-1)
    jsd_speed = compute_jsd_from_samples(
        pred_speed,
        gt_speed,
        bins=int(speed_bins),
        value_range=None,
        range_percentiles=range_percentiles,
        clamp_min=0.0,
        clamp_max=None,
    )

    pred_acc = _compute_acceleration(pred_pos, dt_s=dt_s, meters_per_cell=meters_per_cell).reshape(-1)
    gt_acc = _compute_acceleration(gt_pos, dt_s=dt_s, meters_per_cell=meters_per_cell).reshape(-1)
    jsd_acc = compute_jsd_from_samples(
        pred_acc,
        gt_acc,
        bins=int(accel_bins),
        value_range=None,
        range_percentiles=range_percentiles,
        clamp_min=0.0,
        clamp_max=None,
    )

    pred_turn = _compute_turn_angle(pred_pos, min_speed=float(turn_min_speed)).reshape(-1)
    gt_turn = _compute_turn_angle(gt_pos, min_speed=float(turn_min_speed)).reshape(-1)
    pred_turn = pred_turn[np.isfinite(pred_turn)]
    gt_turn = gt_turn[np.isfinite(gt_turn)]
    jsd_turn = compute_jsd_from_samples(
        pred_turn,
        gt_turn,
        bins=int(turn_bins),
        value_range=(0.0, float(np.pi)),
        range_percentiles=range_percentiles,
        clamp_min=0.0,
        clamp_max=float(np.pi),
    )

    return {
        "JSD_Speed": float(jsd_speed),
        "JSD_Accel": float(jsd_acc),
        "JSD_TurnAngle": float(jsd_turn),
    }


def compute_violation_metrics(
    pred_pos: np.ndarray,
    gt_pos: np.ndarray,
    *,
    dt_s: float = 1.0,
    meters_per_cell: Optional[Tuple[float, float]] = None,
    speed_pctl: float = 99.5,
    accel_pctl: float = 99.5,
) -> Dict[str, float]:
    """
    Dynamic Constraint Violation (DCV) via data-calibrated thresholds.

    If you do not have absolute physical limits (m/s, m/s^2), calibrate
    thresholds from the GT distribution (common in map-free settings).

    Returns:
        Vio_Speed_Rate, Vio_Accel_Rate, Thresh_Speed, Thresh_Accel, Pctl_Speed, Pctl_Accel
    """
    gt_speed = _compute_speed(gt_pos, dt_s=dt_s, meters_per_cell=meters_per_cell).reshape(-1)
    gt_acc = _compute_acceleration(gt_pos, dt_s=dt_s, meters_per_cell=meters_per_cell).reshape(-1)

    thresh_speed = float(np.percentile(gt_speed, float(speed_pctl))) if gt_speed.size else 0.0
    thresh_acc = float(np.percentile(gt_acc, float(accel_pctl))) if gt_acc.size else 0.0

    pred_speed = _compute_speed(pred_pos, dt_s=dt_s, meters_per_cell=meters_per_cell)
    pred_acc = _compute_acceleration(pred_pos, dt_s=dt_s, meters_per_cell=meters_per_cell)

    speed_vio = float(np.mean(pred_speed > thresh_speed)) if pred_speed.size else 0.0
    acc_vio = float(np.mean(pred_acc > thresh_acc)) if pred_acc.size else 0.0

    return {
        "Vio_Speed_Rate": float(speed_vio),
        "Vio_Accel_Rate": float(acc_vio),
        "Thresh_Speed": float(thresh_speed),
        "Thresh_Accel": float(thresh_acc),
        "Pctl_Speed": float(speed_pctl),
        "Pctl_Accel": float(accel_pctl),
    }









