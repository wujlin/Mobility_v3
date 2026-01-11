"""Temporal feature encoding for route conditioning.

Converts Unix timestamps to cyclical hour-of-day and day-of-week features.
Uses sin/cos encoding to capture periodicity (hour 23 is close to hour 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class TemporalFeatureConfig:
    """Configuration for temporal feature encoding."""
    hour_dim: int = 2  # sin + cos for hour
    dow_dim: int = 2   # sin + cos for day-of-week
    use_hour: bool = True
    use_dow: bool = True

    @property
    def dim(self) -> int:
        d = 0
        if self.use_hour:
            d += self.hour_dim
        if self.use_dow:
            d += self.dow_dim
        return d

    def to_json(self) -> Dict[str, object]:
        return {
            "hour_dim": self.hour_dim,
            "dow_dim": self.dow_dim,
            "use_hour": self.use_hour,
            "use_dow": self.use_dow,
        }

    @staticmethod
    def from_json(d: Dict[str, object]) -> "TemporalFeatureConfig":
        return TemporalFeatureConfig(
            hour_dim=int(d.get("hour_dim", 2)),
            dow_dim=int(d.get("dow_dim", 2)),
            use_hour=bool(d.get("use_hour", True)),
            use_dow=bool(d.get("use_dow", True)),
        )


def encode_temporal_cyclic(
    start_t: np.ndarray,
    *,
    cfg: TemporalFeatureConfig | None = None,
    tz_offset_hours: float = 0.0,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    Encode Unix timestamps into cyclical temporal features.

    Args:
        start_t: (N,) array of Unix timestamps (int64 seconds)
        cfg: temporal feature configuration
        tz_offset_hours: timezone offset from UTC (e.g., -5.0 for EST)

    Returns:
        features: (N, D) float32 array
        keys: tuple of feature names

    Note:
        We use cyclic sin/cos encoding because:
        - Hour 23 should be close to hour 0 (midnight continuity)
        - Sunday should be close to Saturday (weekend continuity)
    """
    if cfg is None:
        cfg = TemporalFeatureConfig()

    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    n = int(start_t.shape[0])

    # Apply timezone offset
    start_t_local = start_t + int(tz_offset_hours * 3600)

    # Extract hour of day (0-23) and day of week (0=Monday, 6=Sunday)
    # We use vectorized computation via datetime arithmetic
    seconds_per_day = 86400
    days_since_epoch = start_t_local // seconds_per_day
    seconds_in_day = start_t_local % seconds_per_day
    
    hour_of_day = (seconds_in_day / 3600.0).astype(np.float32)  # 0.0-24.0
    
    # Day of week: Unix epoch (1970-01-01) was a Thursday (3)
    # So: (days_since_epoch + 3) % 7 gives Monday=0, ..., Sunday=6
    day_of_week = ((days_since_epoch + 3) % 7).astype(np.float32)

    parts = []
    keys = []

    if cfg.use_hour:
        # Hour cyclic encoding: period = 24 hours
        hour_rad = 2.0 * np.pi * hour_of_day / 24.0
        parts.append(np.sin(hour_rad).astype(np.float32))
        parts.append(np.cos(hour_rad).astype(np.float32))
        keys.extend(["hour_sin", "hour_cos"])

    if cfg.use_dow:
        # Day-of-week cyclic encoding: period = 7 days
        dow_rad = 2.0 * np.pi * day_of_week / 7.0
        parts.append(np.sin(dow_rad).astype(np.float32))
        parts.append(np.cos(dow_rad).astype(np.float32))
        keys.extend(["dow_sin", "dow_cos"])

    if not parts:
        return np.zeros((n, 0), dtype=np.float32), ()

    features = np.stack(parts, axis=1)  # (N, D)
    return features.astype(np.float32, copy=False), tuple(keys)


def encode_temporal_onehot(
    start_t: np.ndarray,
    *,
    n_hour_bins: int = 4,   # e.g., night/morning/afternoon/evening
    n_dow_bins: int = 2,    # e.g., weekday/weekend
    tz_offset_hours: float = 0.0,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    Encode timestamps using binned one-hot features.
    Useful for capturing discrete periods (rush hour, weekend vs weekday).

    Args:
        start_t: (N,) Unix timestamps
        n_hour_bins: number of hour bins (4 = 6-hour blocks)
        n_dow_bins: number of day bins (2 = weekday/weekend)
        tz_offset_hours: timezone offset

    Returns:
        features: (N, n_hour_bins + n_dow_bins) float32
        keys: feature names
    """
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    n = int(start_t.shape[0])

    # Apply timezone offset
    start_t_local = start_t + int(tz_offset_hours * 3600)

    seconds_per_day = 86400
    days_since_epoch = start_t_local // seconds_per_day
    seconds_in_day = start_t_local % seconds_per_day

    hour_of_day = (seconds_in_day // 3600).astype(np.int32)  # 0-23
    day_of_week = ((days_since_epoch + 3) % 7).astype(np.int32)  # 0-6

    parts = []
    keys = []

    # Hour bins
    if n_hour_bins > 0:
        hours_per_bin = 24 // n_hour_bins
        hour_bin = (hour_of_day // hours_per_bin).astype(np.int32)
        hour_bin = np.clip(hour_bin, 0, n_hour_bins - 1)
        hour_onehot = np.zeros((n, n_hour_bins), dtype=np.float32)
        hour_onehot[np.arange(n), hour_bin] = 1.0
        parts.append(hour_onehot)
        
        if n_hour_bins == 4:
            keys.extend(["hour_night", "hour_morning", "hour_afternoon", "hour_evening"])
        else:
            keys.extend([f"hour_bin{i}" for i in range(n_hour_bins)])

    # Day-of-week bins
    if n_dow_bins > 0:
        if n_dow_bins == 2:
            # weekday (Mon-Fri: 0-4) vs weekend (Sat-Sun: 5-6)
            is_weekend = (day_of_week >= 5).astype(np.float32)
            parts.append(np.stack([1.0 - is_weekend, is_weekend], axis=1))
            keys.extend(["dow_weekday", "dow_weekend"])
        else:
            days_per_bin = 7 // n_dow_bins
            dow_bin = (day_of_week // days_per_bin).astype(np.int32)
            dow_bin = np.clip(dow_bin, 0, n_dow_bins - 1)
            dow_onehot = np.zeros((n, n_dow_bins), dtype=np.float32)
            dow_onehot[np.arange(n), dow_bin] = 1.0
            parts.append(dow_onehot)
            keys.extend([f"dow_bin{i}" for i in range(n_dow_bins)])

    if not parts:
        return np.zeros((n, 0), dtype=np.float32), ()

    features = np.concatenate(parts, axis=1)
    return features.astype(np.float32, copy=False), tuple(keys)


# For compatibility with existing code that expects (hour, day) as 2D
def encode_temporal_simple(
    start_t: np.ndarray,
    *,
    tz_offset_hours: float = 0.0,
) -> np.ndarray:
    """
    Simple temporal encoding: returns (hour_norm, dow_norm) in [-1, 1].
    
    Args:
        start_t: (N,) Unix timestamps
        tz_offset_hours: timezone offset

    Returns:
        (N, 2) float32: [hour_normalized, dow_normalized]
        where hour_norm = hour/12 - 1 (maps 0-24 to -1..+1)
              dow_norm = dow/3 - 1 (maps 0-6 to roughly -1..+1)
    """
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    n = int(start_t.shape[0])

    start_t_local = start_t + int(tz_offset_hours * 3600)

    seconds_per_day = 86400
    days_since_epoch = start_t_local // seconds_per_day
    seconds_in_day = start_t_local % seconds_per_day

    hour_of_day = (seconds_in_day / 3600.0).astype(np.float32)  # 0-24
    day_of_week = ((days_since_epoch + 3) % 7).astype(np.float32)  # 0-6

    # Normalize to roughly [-1, 1]
    hour_norm = (hour_of_day / 12.0 - 1.0).astype(np.float32)
    dow_norm = (day_of_week / 3.0 - 1.0).astype(np.float32)

    return np.stack([hour_norm, dow_norm], axis=1).astype(np.float32, copy=False)


if __name__ == "__main__":
    # Test
    import sys

    # Some test timestamps (2024-01-15 08:30:00 UTC, 2024-01-20 18:45:00 UTC)
    test_ts = np.array([1705309800, 1705772700], dtype=np.int64)
    
    print("=== Cyclic Encoding ===")
    feat, keys = encode_temporal_cyclic(test_ts, tz_offset_hours=-5.0)  # EST
    print(f"Keys: {keys}")
    print(f"Features:\n{feat}")
    
    print("\n=== One-hot Encoding ===")
    feat, keys = encode_temporal_onehot(test_ts, tz_offset_hours=-5.0)
    print(f"Keys: {keys}")
    print(f"Features:\n{feat}")
    
    print("\n=== Simple Encoding ===")
    feat = encode_temporal_simple(test_ts, tz_offset_hours=-5.0)
    print(f"Features:\n{feat}")
