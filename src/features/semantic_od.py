from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class SemanticODNorm:
    keys: Tuple[str, ...]
    mean: np.ndarray  # (D,)
    std: np.ndarray  # (D,)

    def to_json(self) -> Dict[str, object]:
        return {
            "keys": list(self.keys),
            "mean": [float(x) for x in np.asarray(self.mean, dtype=np.float32).reshape(-1).tolist()],
            "std": [float(x) for x in np.asarray(self.std, dtype=np.float32).reshape(-1).tolist()],
        }

    @staticmethod
    def from_json(d: Dict[str, object]) -> "SemanticODNorm":
        keys = tuple(str(x) for x in (d.get("keys") or []))
        mean = np.asarray(d.get("mean") or [], dtype=np.float32).reshape(-1)
        std = np.asarray(d.get("std") or [], dtype=np.float32).reshape(-1)
        if len(keys) != int(mean.size) or int(mean.size) != int(std.size):
            raise ValueError(f"Bad SemanticODNorm json: keys={len(keys)} mean={mean.size} std={std.size}")
        std = np.maximum(std, 1e-3).astype(np.float32, copy=False)
        return SemanticODNorm(keys=keys, mean=mean.astype(np.float32, copy=False), std=std)


def load_poi_total_and_landuse_entropy(semantic_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load minimal spatial semantics rasters from a directory (Detroit core grid).

    Expected files:
      - poi_density_*.npy (multiple categories), summed into poi_total
      - landuse_entropy.npy
    """
    d = Path(semantic_dir)
    poi_paths = sorted(d.glob("poi_density_*.npy"))
    if not poi_paths:
        raise FileNotFoundError(f"No poi_density_*.npy under: {d}")
    poi_total = None
    for p in poi_paths:
        a = np.load(p)
        if a.ndim != 2:
            raise ValueError(f"Bad POI raster shape in {p}: {a.shape} (expected H,W)")
        poi_total = a.astype(np.float32, copy=False) if poi_total is None else (poi_total + a.astype(np.float32, copy=False))
    assert poi_total is not None

    ent_path = d / "landuse_entropy.npy"
    if not ent_path.exists():
        raise FileNotFoundError(f"Missing landuse_entropy.npy under: {d}")
    landuse_entropy = np.load(ent_path).astype(np.float32, copy=False)
    if landuse_entropy.ndim != 2:
        raise ValueError(f"Bad landuse_entropy shape in {ent_path}: {landuse_entropy.shape} (expected H,W)")

    if poi_total.shape != landuse_entropy.shape:
        raise ValueError(f"Raster shape mismatch: poi_total={poi_total.shape} landuse_entropy={landuse_entropy.shape}")
    return poi_total.astype(np.float32, copy=False), landuse_entropy.astype(np.float32, copy=False)


def semantic_od_features(
    *,
    start_ctr: np.ndarray,  # (N,2) [y,x] (typically OD bin centers)
    dest_ctr: np.ndarray,  # (N,2)
    poi_total: np.ndarray,  # (H,W)
    landuse_entropy: np.ndarray,  # (H,W)
    log_poi: bool = True,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    Build a minimal OD semantic vector by reading rasters at (O,D) bin centers.

    Returns:
      feats: (N,4) float32
      keys: semantic feature names (len=4)
    """
    start_ctr = np.asarray(start_ctr, dtype=np.float32)
    dest_ctr = np.asarray(dest_ctr, dtype=np.float32)
    if start_ctr.ndim != 2 or start_ctr.shape[1] != 2:
        raise ValueError(f"Expected start_ctr (N,2), got {start_ctr.shape}")
    if dest_ctr.shape != start_ctr.shape:
        raise ValueError(f"Shape mismatch: start_ctr={start_ctr.shape} dest_ctr={dest_ctr.shape}")
    H, W = poi_total.shape

    y0 = np.clip(np.rint(start_ctr[:, 0]).astype(np.int64), 0, H - 1)
    x0 = np.clip(np.rint(start_ctr[:, 1]).astype(np.int64), 0, W - 1)
    y1 = np.clip(np.rint(dest_ctr[:, 0]).astype(np.int64), 0, H - 1)
    x1 = np.clip(np.rint(dest_ctr[:, 1]).astype(np.int64), 0, W - 1)

    poi_o = poi_total[y0, x0].astype(np.float32, copy=False)
    poi_d = poi_total[y1, x1].astype(np.float32, copy=False)
    if bool(log_poi):
        poi_o = np.log1p(np.maximum(poi_o, 0.0)).astype(np.float32, copy=False)
        poi_d = np.log1p(np.maximum(poi_d, 0.0)).astype(np.float32, copy=False)

    ent_o = landuse_entropy[y0, x0].astype(np.float32, copy=False)
    ent_d = landuse_entropy[y1, x1].astype(np.float32, copy=False)

    feats = np.stack([poi_o, poi_d, ent_o, ent_d], axis=1).astype(np.float32, copy=False)
    keys = ("poi_total_log1p_o", "poi_total_log1p_d", "landuse_entropy_o", "landuse_entropy_d")
    return feats, keys


def fit_semantic_norm(feats: np.ndarray, *, keys: Tuple[str, ...]) -> SemanticODNorm:
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError(f"Expected feats (N,D), got {feats.shape}")
    mean = np.mean(feats, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(feats, axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-3).astype(np.float32, copy=False)
    if len(keys) != int(mean.size):
        raise ValueError(f"keys/mean mismatch: keys={len(keys)} mean={mean.size}")
    return SemanticODNorm(keys=keys, mean=mean, std=std)


def normalize_semantic(feats: np.ndarray, norm: SemanticODNorm) -> np.ndarray:
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[1] != int(norm.mean.size):
        raise ValueError(f"Bad feats shape: {feats.shape}, expected (N,{int(norm.mean.size)})")
    return ((feats - norm.mean[None, :]) / norm.std[None, :]).astype(np.float32, copy=False)


def semantic_norm_json_dumps(norm: SemanticODNorm) -> str:
    return json.dumps(norm.to_json(), ensure_ascii=False, indent=2)

