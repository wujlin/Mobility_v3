from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

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


def load_poi_stack_and_landuse_entropy(semantic_dir: str | Path) -> Tuple[np.ndarray, Tuple[str, ...], np.ndarray]:
    """
    Load POI category rasters as a stack + landuse entropy.

    Expected files:
      - poi_density_*.npy (multiple categories)
      - landuse_entropy.npy

    Returns:
      poi_stack: (C,H,W) float32
      categories: (C,) tuple of category names (derived from filenames)
      landuse_entropy: (H,W) float32
    """
    d = Path(semantic_dir)
    poi_paths = sorted(d.glob("poi_density_*.npy"))
    if not poi_paths:
        raise FileNotFoundError(f"No poi_density_*.npy under: {d}")
    cats: List[str] = []
    stack: List[np.ndarray] = []
    for p in poi_paths:
        cat = p.stem.replace("poi_density_", "", 1)
        a = np.load(p)
        if a.ndim != 2:
            raise ValueError(f"Bad POI raster shape in {p}: {a.shape} (expected H,W)")
        cats.append(str(cat))
        stack.append(a.astype(np.float32, copy=False))
    poi_stack = np.stack(stack, axis=0).astype(np.float32, copy=False)  # (C,H,W)

    ent_path = d / "landuse_entropy.npy"
    if not ent_path.exists():
        raise FileNotFoundError(f"Missing landuse_entropy.npy under: {d}")
    landuse_entropy = np.load(ent_path).astype(np.float32, copy=False)
    if landuse_entropy.ndim != 2:
        raise ValueError(f"Bad landuse_entropy shape in {ent_path}: {landuse_entropy.shape} (expected H,W)")
    if poi_stack.shape[1:] != landuse_entropy.shape:
        raise ValueError(f"Raster shape mismatch: poi_stack={poi_stack.shape} landuse_entropy={landuse_entropy.shape}")
    return poi_stack, tuple(cats), landuse_entropy


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


def semantic_corridor_profile_features(
    *,
    start_ctr: np.ndarray,  # (N,2) [y,x] (OD intent)
    dest_ctr: np.ndarray,  # (N,2)
    poi_stack: np.ndarray,  # (C,H,W)
    categories: Sequence[str],
    landuse_entropy: np.ndarray,  # (H,W)
    num_steps: int = 16,
    offsets: Sequence[float] = (-32.0, 0.0, 32.0),
    log_total: bool = True,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    Build an OD-conditioned *environment semantic profile* by sampling a raster strip along a
    straight OD chord (corridor-agnostic).

    This uses only inference-observable environment features (no GT corridors).

    Returns:
      feats: (N, 2 + C) float32
        [log1p(poi_total_along), mean_entropy_along, poi_frac_cat_0..C-1]
    """
    start_ctr = np.asarray(start_ctr, dtype=np.float32)
    dest_ctr = np.asarray(dest_ctr, dtype=np.float32)
    if start_ctr.ndim != 2 or start_ctr.shape[1] != 2:
        raise ValueError(f"Expected start_ctr (N,2), got {start_ctr.shape}")
    if dest_ctr.shape != start_ctr.shape:
        raise ValueError(f"Shape mismatch: start_ctr={start_ctr.shape} dest_ctr={dest_ctr.shape}")

    poi_stack = np.asarray(poi_stack, dtype=np.float32)
    if poi_stack.ndim != 3:
        raise ValueError(f"Expected poi_stack (C,H,W), got {poi_stack.shape}")
    C, H, W = poi_stack.shape
    if int(len(categories)) != int(C):
        raise ValueError(f"categories length mismatch: categories={len(categories)} poi_stack.C={C}")

    landuse_entropy = np.asarray(landuse_entropy, dtype=np.float32)
    if landuse_entropy.shape != (H, W):
        raise ValueError(f"Expected landuse_entropy {(H, W)}, got {landuse_entropy.shape}")

    M = int(num_steps)
    if M <= 1:
        raise ValueError("--profile_num_steps must be > 1")
    offs = [float(x) for x in offsets]
    if not offs:
        raise ValueError("--profile_offsets must be non-empty")

    t = np.linspace(0.0, 1.0, num=M, dtype=np.float32).reshape(M, 1)  # (M,1)
    eps = 1e-9
    feats = np.zeros((int(start_ctr.shape[0]), 2 + int(C)), dtype=np.float32)

    for i in range(int(start_ctr.shape[0])):
        a = start_ctr[i].astype(np.float32, copy=False)
        b = dest_ctr[i].astype(np.float32, copy=False)
        v = b - a
        L = float(np.linalg.norm(v))
        if not np.isfinite(L) or L <= 1e-6:
            e_perp = np.asarray([0.0, 1.0], dtype=np.float32)
        else:
            e_par = v / float(L)
            e_perp = np.asarray([-e_par[1], e_par[0]], dtype=np.float32)

        base = a[None, :] + t * v[None, :]  # (M,2)
        pts = np.concatenate([base + float(off) * e_perp[None, :] for off in offs], axis=0)  # (M*O,2)
        yy = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, H - 1)
        xx = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, W - 1)

        counts = poi_stack[:, yy, xx].astype(np.float64, copy=False).sum(axis=1)  # (C,)
        total = float(np.sum(counts))
        if total <= 0.0 or not np.isfinite(total):
            frac = np.ones((C,), dtype=np.float64) / float(C)
            total = 0.0
        else:
            frac = counts / (total + eps)

        ent_mean = float(np.mean(landuse_entropy[yy, xx])) if yy.size > 0 else 0.0
        total_feat = float(np.log1p(total)) if bool(log_total) else float(total)

        feats[i, 0] = float(total_feat)
        feats[i, 1] = float(ent_mean)
        feats[i, 2:] = frac.astype(np.float32, copy=False)

    keys: List[str] = ["poi_total_log1p_along", "landuse_entropy_mean_along"]
    keys.extend([f"poi_frac_{str(c)}_along" for c in categories])
    return feats.astype(np.float32, copy=False), tuple(keys)


def semantic_grid_pool_features(
    *,
    start_ctr: np.ndarray,  # (N,2) [y,x]
    dest_ctr: np.ndarray,  # (N,2)
    poi_stack: np.ndarray,  # (C,H,W)
    categories: Sequence[str],
    landuse_entropy: np.ndarray,  # (H,W)
    patch_size: int = 16,
    extent: float = 128.0,
    pool: Literal["quad", "lr"] = "quad",
    log_poi: bool = True,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    Grid-level environment semantics for corridor commitment.

    We crop an OD-aligned semantic patch centered at the OD midpoint, then apply spatial pooling
    to avoid high-dimensional flattening. This keeps coarse spatial structure (e.g., left vs right
    of the OD chord) while remaining corridor-agnostic at inference.

    Pooling modes:
      - 'quad': 4 quadrants (back/front × left/right) => 4*C POI + 4 entropy dims
      - 'lr':   left/right halves => 2*C POI + 2 entropy dims

    Returns:
      feats: (N, D) float32
      keys:  feature names aligned with feats columns
    """
    start_ctr = np.asarray(start_ctr, dtype=np.float32)
    dest_ctr = np.asarray(dest_ctr, dtype=np.float32)
    if start_ctr.ndim != 2 or start_ctr.shape[1] != 2:
        raise ValueError(f"Expected start_ctr (N,2), got {start_ctr.shape}")
    if dest_ctr.shape != start_ctr.shape:
        raise ValueError(f"Shape mismatch: start_ctr={start_ctr.shape} dest_ctr={dest_ctr.shape}")

    poi_stack = np.asarray(poi_stack, dtype=np.float32)
    if poi_stack.ndim != 3:
        raise ValueError(f"Expected poi_stack (C,H,W), got {poi_stack.shape}")
    C, H, W = poi_stack.shape
    if int(len(categories)) != int(C):
        raise ValueError(f"categories length mismatch: categories={len(categories)} poi_stack.C={C}")

    landuse_entropy = np.asarray(landuse_entropy, dtype=np.float32)
    if landuse_entropy.shape != (H, W):
        raise ValueError(f"Expected landuse_entropy {(H, W)}, got {landuse_entropy.shape}")

    S = int(patch_size)
    if S <= 0 or (S % 2) != 0:
        raise ValueError("--grid_patch_size must be a positive even integer.")
    ext = float(extent)
    if not np.isfinite(ext) or ext <= 0.0:
        raise ValueError("--grid_extent must be > 0")
    pool = str(pool)
    if pool not in ("quad", "lr"):
        raise ValueError("--grid_pool must be one of: quad, lr")

    # Patch sampling coordinates in the aligned (par, perp) frame.
    # Use symmetric cell centers; for even S, the seam lies between the two middle columns/rows.
    u = ((np.arange(S, dtype=np.float32) + 0.5) - (float(S) / 2.0)) / (float(S) / 2.0) * ext  # (S,)
    v = u.copy()
    uu, vv = np.meshgrid(u, v, indexing="ij")  # (S,S)
    iu, iv = np.meshgrid(np.arange(S, dtype=np.int64), np.arange(S, dtype=np.int64), indexing="ij")
    iu_f = iu.reshape(-1)
    iv_f = iv.reshape(-1)
    uu_f = uu.reshape(-1).astype(np.float32, copy=False)
    vv_f = vv.reshape(-1).astype(np.float32, copy=False)

    mid_u = S // 2
    if pool == "quad":
        regions = (
            ("bl", (iu_f < mid_u) & (iv_f < mid_u)),
            ("br", (iu_f < mid_u) & (iv_f >= mid_u)),
            ("fl", (iu_f >= mid_u) & (iv_f < mid_u)),
            ("fr", (iu_f >= mid_u) & (iv_f >= mid_u)),
        )
    else:
        regions = (
            ("l", (iv_f < mid_u)),
            ("r", (iv_f >= mid_u)),
        )

    reg_idx = [(name, np.where(mask)[0].astype(np.int64, copy=False)) for name, mask in regions]

    R = int(len(reg_idx))
    feats = np.zeros((int(start_ctr.shape[0]), int(R) * int(C) + int(R)), dtype=np.float32)

    eps = 1e-9
    for i in range(int(start_ctr.shape[0])):
        a = start_ctr[i].astype(np.float32, copy=False)
        b = dest_ctr[i].astype(np.float32, copy=False)
        mid = 0.5 * (a + b)
        d = b - a
        L = float(np.linalg.norm(d))
        if not np.isfinite(L) or L <= 1e-6:
            e_par = np.asarray([1.0, 0.0], dtype=np.float32)
            e_perp = np.asarray([0.0, 1.0], dtype=np.float32)
        else:
            e_par = (d / float(L)).astype(np.float32, copy=False)
            e_perp = np.asarray([-e_par[1], e_par[0]], dtype=np.float32)

        pts = mid[None, :] + uu_f[:, None] * e_par[None, :] + vv_f[:, None] * e_perp[None, :]  # (S*S,2)
        yy = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, H - 1)
        xx = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, W - 1)

        patch_poi = poi_stack[:, yy, xx].astype(np.float64, copy=False)  # (C,P)
        patch_ent = landuse_entropy[yy, xx].astype(np.float64, copy=False)  # (P,)

        col = 0
        for _, idx in reg_idx:
            if idx.size == 0:
                sums = np.zeros((C,), dtype=np.float64)
                ent_mean = 0.0
            else:
                sums = patch_poi[:, idx].sum(axis=1)
                ent_mean = float(np.mean(patch_ent[idx]))
            if bool(log_poi):
                sums = np.log1p(np.maximum(sums, 0.0))
            feats[i, col : col + C] = sums.astype(np.float32, copy=False)
            col += int(C)
            feats[i, col] = float(ent_mean)
            col += 1

        if col != feats.shape[1]:
            raise RuntimeError("Internal feature dimension mismatch.")

    keys: List[str] = []
    for name, _ in reg_idx:
        keys.extend([f"poi_sum_log1p_{str(c)}_{name}" for c in categories])
        keys.append(f"landuse_entropy_mean_{name}")
    return feats.astype(np.float32, copy=False), tuple(keys)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """
    Deterministic 64-bit mixing function (SplitMix64).

    Notes:
      - Pure integer math => stable across Python/Numpy versions.
      - We use it to generate reproducible pseudo-random features keyed by OD intent.
    """
    x = np.asarray(x, dtype=np.uint64)
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = z ^ (z >> np.uint64(31))
    return z.astype(np.uint64, copy=False)


def semantic_rand4_features(
    *,
    start_ctr: np.ndarray,  # (N,2) [y,x]
    dest_ctr: np.ndarray,  # (N,2)
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    A control feature: 4-D pseudo-random vector keyed only by the OD intent (bin centers).

    Purpose:
      - Test whether "weak semantics" gains come from meaningful semantic information,
        or simply from adding extra conditioning dimensions.

    Design:
      - Uses only inference-observable OD intent (no GT).
      - Constant for the same (O_bin, D_bin) pair.
    """
    start_ctr = np.asarray(start_ctr, dtype=np.float32)
    dest_ctr = np.asarray(dest_ctr, dtype=np.float32)
    if start_ctr.ndim != 2 or start_ctr.shape[1] != 2:
        raise ValueError(f"Expected start_ctr (N,2), got {start_ctr.shape}")
    if dest_ctr.shape != start_ctr.shape:
        raise ValueError(f"Shape mismatch: start_ctr={start_ctr.shape} dest_ctr={dest_ctr.shape}")

    y0 = np.rint(start_ctr[:, 0]).astype(np.uint64)
    x0 = np.rint(start_ctr[:, 1]).astype(np.uint64)
    y1 = np.rint(dest_ctr[:, 0]).astype(np.uint64)
    x1 = np.rint(dest_ctr[:, 1]).astype(np.uint64)

    # Pack into a single 64-bit key (safe under pos_max<=1023).
    key = (y0 & np.uint64(0xFFFF)) | ((x0 & np.uint64(0xFFFF)) << np.uint64(16)) | ((y1 & np.uint64(0xFFFF)) << np.uint64(32)) | ((x1 & np.uint64(0xFFFF)) << np.uint64(48))

    z0 = _splitmix64(key)
    z1 = _splitmix64(z0)
    z2 = _splitmix64(z1)
    z3 = _splitmix64(z2)
    z = np.stack([z0, z1, z2, z3], axis=1).astype(np.uint64, copy=False)  # (N,4)

    # Map uint64 -> float32 in [-1, 1].
    u = (z.astype(np.float64) + 0.5) / float(2**64)  # (0,1)
    feats = (2.0 * u - 1.0).astype(np.float32, copy=False)
    keys = ("rand4_0", "rand4_1", "rand4_2", "rand4_3")
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
