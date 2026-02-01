from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    p50: float
    p75: float
    p95: float


def summarize(x: Iterable[float]) -> Dict[str, float]:
    a = np.asarray(list(x), dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p75": float("nan"), "p95": float("nan")}
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "p50": float(np.quantile(a, 0.50)),
        "p75": float(np.quantile(a, 0.75)),
        "p95": float(np.quantile(a, 0.95)),
    }


def _cdist_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64 = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    b64 = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    diff = a64[:, None, :] - b64[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dynamic Time Warping (DTW) distance between two 2D polylines.
    a: (Ta,2), b: (Tb,2) in meters.
    """
    aa = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    bb = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    dist = _cdist_euclidean(aa, bb)
    na, nb = dist.shape
    dp = np.full((na + 1, nb + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = dist[i - 1, j - 1]
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[na, nb])


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Discrete Fréchet distance between two 2D polylines.
    a: (Ta,2), b: (Tb,2) in meters.
    """
    aa = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    bb = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    dist = _cdist_euclidean(aa, bb)
    na, nb = dist.shape
    ca = np.full((na, nb), -1.0, dtype=np.float64)
    ca[0, 0] = dist[0, 0]
    for i in range(1, na):
        ca[i, 0] = max(ca[i - 1, 0], dist[i, 0])
    for j in range(1, nb):
        ca[0, j] = max(ca[0, j - 1], dist[0, j])
    for i in range(1, na):
        for j in range(1, nb):
            ca[i, j] = max(dist[i, j], min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]))
    return float(ca[na - 1, nb - 1])


def hausdorff_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Symmetric Hausdorff distance between two 2D point sets.
    a: (Ta,2), b: (Tb,2) in meters.
    """
    aa = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    bb = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    dist = _cdist_euclidean(aa, bb)
    # directed
    h_ab = float(np.max(np.min(dist, axis=1)))
    h_ba = float(np.max(np.min(dist, axis=0)))
    return max(h_ab, h_ba)


def safe_metric(x: Optional[float]) -> float:
    return float(x) if x is not None and np.isfinite(float(x)) else float("nan")
