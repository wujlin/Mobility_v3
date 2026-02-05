from __future__ import annotations

import numpy as np


def hour_from_unix(start_t: np.ndarray, tz_offset_hours: float) -> np.ndarray:
    """
    Args:
        start_t: unix seconds, shape (N,)
        tz_offset_hours: timezone offset (e.g. -5 for US Eastern standard time)
    Returns:
        hour: int64, shape (N,), in [0, 23]
    """
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    sec = ((start_t + tz_sec) % 86400).astype(np.int64, copy=False)
    return (sec // 3600).astype(np.int64, copy=False)


def dow_from_unix(start_t: np.ndarray, tz_offset_hours: float) -> np.ndarray:
    """
    Day-of-week with Monday=0..Sunday=6 (same as datetime.weekday()).

    1970-01-01 is Thursday (weekday=3), so:
      dow = (days_since_epoch + 3) % 7
    """
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    days = ((start_t + tz_sec) // 86400).astype(np.int64, copy=False)
    return ((days + 3) % 7).astype(np.int64, copy=False)

