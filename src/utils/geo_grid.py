from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class BBox:
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float

    def contains(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return (lon >= self.min_lon) & (lon <= self.max_lon) & (lat >= self.min_lat) & (lat <= self.max_lat)


@dataclass(frozen=True)
class GridSpec:
    H: int
    W: int
    bbox: BBox

    def latlon_to_yx(self, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert WGS84 (lat, lon) to grid indices (y, x).

        Convention (must match docs/DATA_CONTRACT.md):
        - y: row index, 0 at north, increases south
        - x: col index, 0 at west, increases east
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)

        x01 = (lon - self.bbox.min_lon) / (self.bbox.max_lon - self.bbox.min_lon)
        y01 = (self.bbox.max_lat - lat) / (self.bbox.max_lat - self.bbox.min_lat)

        x = np.floor(x01 * self.W).astype(np.int64)
        y = np.floor(y01 * self.H).astype(np.int64)
        return y, x

    def yx_to_latlon(self, y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)

        x01 = (x + 0.5) / self.W
        y01 = (y + 0.5) / self.H
        lon = self.bbox.min_lon + x01 * (self.bbox.max_lon - self.bbox.min_lon)
        lat = self.bbox.max_lat - y01 * (self.bbox.max_lat - self.bbox.min_lat)
        return lat, lon

    def in_bounds(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        return (y >= 0) & (y < self.H) & (x >= 0) & (x < self.W)

    def resolution_m(self) -> Tuple[float, float]:
        """
        Approximate meters-per-cell resolution (res_y_m, res_x_m) using haversine distances
        along bbox edges. This is sufficient for Detroit-scale windows.
        """
        res_x = haversine_m(self.bbox.min_lat, self.bbox.min_lon, self.bbox.min_lat, self.bbox.max_lon) / float(self.W)
        res_y = haversine_m(self.bbox.min_lat, self.bbox.min_lon, self.bbox.max_lat, self.bbox.min_lon) / float(self.H)
        return float(res_y), float(res_x)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points on Earth (WGS84), in meters.
    """
    r = 6371008.8  # mean Earth radius (m)
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = phi2 - phi1
    dlmb = np.deg2rad(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(r * c)

