from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):  # type: ignore[no-redef]
        return x

from src.features.semantic_od import (
    SemanticGridNorm,
    fit_semantic_norm,
    fit_grid_norm,
    load_osm_road_prob,
    load_poi_stack_and_landuse_entropy,
    load_poi_total_and_landuse_entropy,
    normalize_grid_patch,
    normalize_semantic,
    semantic_corridor_profile_features,
    semantic_grid_patch_tensor,
    semantic_grid_pool_features,
    semantic_od_features,
    semantic_rand4_features,
)
from src.features.temporal import encode_route_temporal_2d
from src.features.waypoints import WaypointConfig, extract_oracle_waypoints_from_future
from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.models.semantic.grid_cnn_encoder import GridCNNEncoder
from src.models.semantic.grid_cross_attention_control import GridCrossAttentionControlMid
from src.models.semantic.semantic_patch_sampler import sample_patch_mean_along_skeleton
from src.models.semantic.waypoint_semantic_posenc import WaypointSemanticPosEnc
from src.training.route_npz_utils import RouteNorm, load_route_windows_npz, make_default_pos_bounds, normalize_pos


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _od_bin_center(pos: np.ndarray, *, bin_size: float) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    b = float(bin_size)
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("--od_bin must be > 0")
    return (np.floor(pos / b) + 0.5) * b


def _waypoints_rel_so(
    *,
    start: np.ndarray,  # (N,2)
    dest: np.ndarray,  # (N,2)
    waypoints: np.ndarray,  # (N,K,2)
    eps: float = 1e-6,
) -> np.ndarray:
    start = np.asarray(start, dtype=np.float32)
    dest = np.asarray(dest, dtype=np.float32)
    wp = np.asarray(waypoints, dtype=np.float32)
    v = dest - start  # (N,2)
    L = np.linalg.norm(v, axis=1).astype(np.float32)
    L = np.maximum(L, float(eps))
    e_par = v / L[:, None]  # (N,2)
    e_perp = np.stack([-e_par[:, 1], e_par[:, 0]], axis=1)  # (N,2)

    d = wp - start[:, None, :]  # (N,K,2)
    s = np.sum(d * e_par[:, None, :], axis=2) / L[:, None]
    o = np.sum(d * e_perp[:, None, :], axis=2) / L[:, None]
    rel = np.stack([s, o], axis=2).astype(np.float32, copy=False)  # (N,K,2)

    # Canonical ordering: sort by s (monotone along chord).
    order = np.argsort(rel[:, :, 0], axis=1)
    rel = np.take_along_axis(rel, order[:, :, None], axis=1)
    return rel.astype(np.float32, copy=False)


def _extract_oracle_waypoints(
    *,
    start_pos: np.ndarray,  # (N,2)
    targets: np.ndarray,  # (N,F,2)
    num_waypoints: int,
    waypoint_mode: str,
    waypoint_turn_alpha: float,
) -> np.ndarray:
    n = int(start_pos.shape[0])
    k = int(num_waypoints)
    cfg = WaypointConfig(mode=str(waypoint_mode), num_waypoints=k, turn_alpha=float(waypoint_turn_alpha))
    out = np.zeros((n, k, 2), dtype=np.float32)
    for i in range(n):
        _, wp = extract_oracle_waypoints_from_future(start_pos=start_pos[i], future_pos=targets[i], cfg=cfg)
        if wp.shape != (k, 2):
            raise RuntimeError(f"Bad oracle waypoint shape: {wp.shape}, expected {(k, 2)}")
        out[i] = wp
    return out.astype(np.float32, copy=False)


def _compute_rel_norm(rel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rel = np.asarray(rel, dtype=np.float32).reshape(-1, 2)
    mean = np.mean(rel, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(rel, axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-3).astype(np.float32, copy=False)
    return mean, std


def _normalize_rel(rel: np.ndarray, *, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    rel = np.asarray(rel, dtype=np.float32)
    return ((rel - mean[None, None, :]) / std[None, None, :]).astype(np.float32, copy=False)


def _parse_float_list(s: str) -> Tuple[float, ...]:
    items = [x.strip() for x in str(s).split(",") if str(x).strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated list.")
    out = []
    for x in items:
        out.append(float(x))
    return tuple(out)


@dataclass(frozen=True)
class TrainConfig:
    train_npz: str
    out_dir: str
    pos_max: int
    max_train_n: Optional[int]
    num_waypoints: int
    waypoint_mode: str
    waypoint_turn_alpha: float
    od_bin: float
    o_clip: float
    temporal_mode: str
    temporal_tz_offset_hours: float
    semantic_dir: Optional[str]
    semantic_mode: str
    semantic_use_bins: bool
    profile_num_steps: int
    profile_offsets: str
    grid_patch_size: int
    grid_extent: float
    grid_pool: str
    grid_channels: str
    grid_emb_dim: int
    semantic_posenc_hidden_dim: int
    semantic_posenc_weight: float
    semantic_posenc_self_correct: bool
    semantic_attn_heads: int
    semantic_attn_weight: float
    semantic_loss_weight: float
    semantic_loss_samples_per_segment: int
    hidden_dim: int
    diff_steps: int
    pred_type: str
    batch_size: int
    epochs: int
    lr: float
    num_workers: int
    max_batches: Optional[int]
    seed: int


class WaypointRelDataset(Dataset):
    def __init__(self, *, obs: np.ndarray, cond: np.ndarray, target_rel_norm: np.ndarray, traj_idx: np.ndarray, start_t: np.ndarray):
        self.obs = np.asarray(obs, dtype=np.float32)
        self.cond = np.asarray(cond, dtype=np.float32)
        self.target = np.asarray(target_rel_norm, dtype=np.float32)
        self.traj_idx = np.asarray(traj_idx, dtype=np.int64)
        self.start_t = np.asarray(start_t, dtype=np.int64)

        if self.obs.ndim != 3 or self.obs.shape[1:] != (1, 4):
            raise ValueError(f"Expected obs (N,1,4), got {self.obs.shape}")
        if self.cond.ndim != 2 or self.cond.shape[1] <= 0:
            raise ValueError(f"Expected cond (N,D), got {self.cond.shape}")
        if self.target.ndim != 3 or self.target.shape[-1] != 2:
            raise ValueError(f"Expected target (N,K,2), got {self.target.shape}")
        if self.obs.shape[0] != self.cond.shape[0] or self.obs.shape[0] != self.target.shape[0]:
            raise ValueError("N mismatch among obs/cond/target")

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, idx: int) -> dict:
        idx = int(idx)
        return {
            "obs": torch.from_numpy(self.obs[idx]).float(),
            "cond": torch.from_numpy(self.cond[idx]).float(),
            "action": torch.from_numpy(self.target[idx]).float(),
            "meta": {"traj_idx": int(self.traj_idx[idx]), "start_t": int(self.start_t[idx])},
        }


class WaypointRelDatasetWithGrid(Dataset):
    def __init__(
        self,
        *,
        obs: np.ndarray,
        start_pos: np.ndarray,
        dest_pos: np.ndarray,
        cond_base: np.ndarray,
        cond_sem: Optional[np.ndarray],
        grid_patch: np.ndarray,
        target_rel_norm: np.ndarray,
        traj_idx: np.ndarray,
        start_t: np.ndarray,
    ) -> None:
        self.obs = np.asarray(obs, dtype=np.float32)
        self.start_pos = np.asarray(start_pos, dtype=np.float32)
        self.dest_pos = np.asarray(dest_pos, dtype=np.float32)
        self.cond_base = np.asarray(cond_base, dtype=np.float32)
        self.cond_sem = (np.asarray(cond_sem, dtype=np.float32) if cond_sem is not None else None)
        self.grid_patch = np.asarray(grid_patch, dtype=np.float32)
        self.target = np.asarray(target_rel_norm, dtype=np.float32)
        self.traj_idx = np.asarray(traj_idx, dtype=np.int64)
        self.start_t = np.asarray(start_t, dtype=np.int64)

        if self.obs.ndim != 3 or self.obs.shape[1:] != (1, 4):
            raise ValueError(f"Expected obs (N,1,4), got {self.obs.shape}")
        if self.start_pos.ndim != 2 or self.start_pos.shape[1] != 2:
            raise ValueError(f"Expected start_pos (N,2), got {self.start_pos.shape}")
        if self.dest_pos.shape != self.start_pos.shape:
            raise ValueError(f"dest_pos shape mismatch: start_pos={self.start_pos.shape} dest_pos={self.dest_pos.shape}")
        if self.cond_base.ndim != 2 or self.cond_base.shape[1] != 6:
            raise ValueError(f"Expected cond_base (N,6), got {self.cond_base.shape}")
        if self.cond_sem is not None and (self.cond_sem.ndim != 2 or self.cond_sem.shape[0] != self.cond_base.shape[0]):
            raise ValueError(f"Bad cond_sem shape: {None if self.cond_sem is None else self.cond_sem.shape}")
        if self.grid_patch.ndim != 4:
            raise ValueError(f"Expected grid_patch (N,C,S,S), got {self.grid_patch.shape}")
        if self.target.ndim != 3 or self.target.shape[-1] != 2:
            raise ValueError(f"Expected target (N,K,2), got {self.target.shape}")
        n = int(self.obs.shape[0])
        if (
            int(self.start_pos.shape[0]) != n
            or int(self.dest_pos.shape[0]) != n
            or int(self.cond_base.shape[0]) != n
            or int(self.grid_patch.shape[0]) != n
            or int(self.target.shape[0]) != n
        ):
            raise ValueError("N mismatch among obs/cond_base/grid_patch/target")

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, idx: int) -> dict:
        idx = int(idx)
        out = {
            "obs": torch.from_numpy(self.obs[idx]).float(),
            "start_pos": torch.from_numpy(self.start_pos[idx]).float(),
            "dest_pos": torch.from_numpy(self.dest_pos[idx]).float(),
            "cond_base": torch.from_numpy(self.cond_base[idx]).float(),
            "grid_patch": torch.from_numpy(self.grid_patch[idx]).float(),
            "action": torch.from_numpy(self.target[idx]).float(),
            "meta": {"traj_idx": int(self.traj_idx[idx]), "start_t": int(self.start_t[idx])},
        }
        if self.cond_sem is not None:
            out["cond_sem"] = torch.from_numpy(self.cond_sem[idx]).float()
        return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a diffusion decision model: p(waypoints | OD-bin) on route windows npz.")
    p.add_argument("--train_npz", type=str, required=True, help="npz with start_pos/targets/dest_pos/traj_idx/start_t")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--pos_max", type=int, default=1023)
    p.add_argument("--max_train_n", type=int, default=None)

    p.add_argument("--num_waypoints", type=int, default=2)
    p.add_argument("--waypoint_mode", type=str, choices=["rdp_dev", "rdp_turn"], default="rdp_turn")
    p.add_argument("--waypoint_turn_alpha", type=float, default=1.0, help="When waypoint_mode=rdp_turn: weight for turn-aware waypoint selection.")
    p.add_argument("--od_bin", type=float, default=128.0, help="Bin size (grid units) for OD intent conditioning.")
    p.add_argument("--o_clip", type=float, default=2.0, help="Clip signed offset o (in chord-normalized units) for stability.")
    p.add_argument("--temporal_mode", type=str, choices=["auto", "simple", "zeros"], default="auto", help="Temporal feature for the (hour,day) slots: auto/simple/zeros.")
    p.add_argument("--temporal_tz_offset_hours", type=float, default=-5.0, help="Timezone offset used when temporal_mode!=zeros (Detroit/Columbus: -5).")
    p.add_argument(
        "--semantic_dir",
        type=str,
        default=None,
        help="Optional directory containing poi_density_*.npy and landuse_entropy.npy (adds environment semantics to cond).",
    )
    p.add_argument(
        "--semantic_mode",
        type=str,
        choices=[
            "od",
            "profile",
            "od_profile",
            "grid",
            "od_grid",
            "gridcnn",
            "od_gridcnn",
            "gridpos",
            "od_gridpos",
            "gridattn",
            "od_gridattn",
            "rand4",
        ],
        default="od",
        help="Semantic feature mode: od (O/D point features), profile (OD-chord strip profile; legacy), grid (OD-aligned grid pooling), gridcnn (OD-aligned patch + CNN encoder), rand4 (OD-keyed random control), or concatenations.",
    )
    p.add_argument(
        "--semantic_use_bins",
        action="store_true",
        help="If set, compute semantic features from OD-bin centers (shared intent) instead of per-window raw O/D.",
    )
    p.add_argument("--profile_num_steps", type=int, default=16, help="When semantic_mode includes 'profile': number of samples along the OD chord.")
    p.add_argument("--profile_offsets", type=str, default="-32,0,32", help="When semantic_mode includes 'profile': comma-separated perpendicular offsets (grid units).")
    p.add_argument("--grid_patch_size", type=int, default=16, help="When semantic_mode includes 'grid': patch size S (even), pooled spatially (no flatten).")
    p.add_argument("--grid_extent", type=float, default=128.0, help="When semantic_mode includes 'grid': patch half-extent (grid units) around OD midpoint.")
    p.add_argument("--grid_pool", type=str, choices=["quad", "lr"], default="quad", help="When semantic_mode includes 'grid': pooling mode (quad=4 quadrants, lr=left/right halves).")
    p.add_argument("--grid_channels", type=str, default="poi,entropy", help="When semantic_mode includes 'gridcnn': comma-separated channels from {poi,entropy,road_prob}.")
    p.add_argument("--grid_emb_dim", type=int, default=64, help="When semantic_mode includes 'gridcnn': CNN output embedding dim.")
    p.add_argument("--semantic_posenc_hidden_dim", type=int, default=256, help="When semantic_mode includes 'gridpos': MLP hidden dim for position-aligned semantic conditioning.")
    p.add_argument("--semantic_posenc_weight", type=float, default=1.0, help="When semantic_mode includes 'gridpos': scale for semantic extra embedding.")
    p.add_argument(
        "--semantic_posenc_self_correct",
        action="store_true",
        help="When semantic_mode includes 'gridpos': use a no-grad x0 estimate to sample semantics (self-correcting guidance).",
    )
    p.add_argument("--semantic_attn_heads", type=int, default=4, help="When semantic_mode includes 'gridattn': number of attention heads.")
    p.add_argument("--semantic_attn_weight", type=float, default=1.0, help="When semantic_mode includes 'gridattn': scale for attention control_mid.")
    p.add_argument("--semantic_loss_weight", type=float, default=0.0, help="Optional (Scheme-C): weight for semantic consistency loss computed along skeleton.")
    p.add_argument("--semantic_loss_samples_per_segment", type=int, default=8, help="When semantic_loss_weight>0: samples per segment for semantic consistency.")

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--diff_steps", type=int, default=50)
    p.add_argument("--pred_type", type=str, choices=["eps", "v"], default="eps")

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=None, help="Limit batches per epoch (smoke runs)")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = TrainConfig(
        train_npz=str(args.train_npz),
        out_dir=str(args.out_dir),
        pos_max=int(args.pos_max),
        max_train_n=(int(args.max_train_n) if args.max_train_n is not None else None),
        num_waypoints=int(args.num_waypoints),
        waypoint_mode=str(args.waypoint_mode),
        waypoint_turn_alpha=float(args.waypoint_turn_alpha),
        od_bin=float(args.od_bin),
        o_clip=float(args.o_clip),
        temporal_mode=str(args.temporal_mode),
        temporal_tz_offset_hours=float(args.temporal_tz_offset_hours),
        semantic_dir=(str(args.semantic_dir) if args.semantic_dir else None),
        semantic_mode=str(args.semantic_mode),
        semantic_use_bins=bool(args.semantic_use_bins),
        profile_num_steps=int(args.profile_num_steps),
        profile_offsets=str(args.profile_offsets),
        grid_patch_size=int(args.grid_patch_size),
        grid_extent=float(args.grid_extent),
        grid_pool=str(args.grid_pool),
        grid_channels=str(args.grid_channels),
        grid_emb_dim=int(args.grid_emb_dim),
        semantic_posenc_hidden_dim=int(args.semantic_posenc_hidden_dim),
        semantic_posenc_weight=float(args.semantic_posenc_weight),
        semantic_posenc_self_correct=bool(args.semantic_posenc_self_correct),
        semantic_attn_heads=int(args.semantic_attn_heads),
        semantic_attn_weight=float(args.semantic_attn_weight),
        semantic_loss_weight=float(args.semantic_loss_weight),
        semantic_loss_samples_per_segment=int(args.semantic_loss_samples_per_segment),
        hidden_dim=int(args.hidden_dim),
        diff_steps=int(args.diff_steps),
        pred_type=str(args.pred_type),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        num_workers=int(args.num_workers),
        max_batches=(int(args.max_batches) if args.max_batches is not None else None),
        seed=int(args.seed),
    )
    _set_seed(int(cfg.seed))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "last.pt"
    summary_path = out_dir / "train_summary.json"

    data = load_route_windows_npz(cfg.train_npz, max_n=cfg.max_train_n, seed=int(cfg.seed))
    start_pos = np.asarray(data["start_pos"], dtype=np.float32)
    targets = np.asarray(data["targets"], dtype=np.float32)
    dest_pos = np.asarray(data["dest_pos"], dtype=np.float32)
    traj_idx = np.asarray(data["traj_idx"], dtype=np.int64)
    start_t = np.asarray(data["start_t"], dtype=np.int64)

    n = int(start_pos.shape[0])
    k = int(cfg.num_waypoints)
    if k <= 0:
        raise ValueError("--num_waypoints must be > 0")

    # OD intent conditioning: bin centers (shared across nearby ODs) -> multi-modality supervision.
    start_ctr = _od_bin_center(start_pos, bin_size=float(cfg.od_bin))
    dest_ctr = _od_bin_center(dest_pos, bin_size=float(cfg.od_bin))

    pos_min, pos_max_arr = make_default_pos_bounds(pos_max=int(cfg.pos_max))
    pos_range = (pos_max_arr - pos_min + 1e-6).astype(np.float32)
    norm = RouteNorm(
        pos_min=pos_min.astype(np.float32, copy=False),
        pos_max=pos_max_arr.astype(np.float32, copy=False),
        pos_range=pos_range.astype(np.float32, copy=False),
        vel_mean=np.zeros((2,), dtype=np.float32),
        vel_std=np.ones((2,), dtype=np.float32),
    )

    start_ctr_norm = normalize_pos(start_ctr, norm)  # (N,2)
    dest_ctr_norm = normalize_pos(dest_ctr, norm)  # (N,2)

    # obs: start-bin only (avoid leaking per-window micro differences).
    obs = np.concatenate([start_ctr_norm, np.zeros((n, 2), dtype=np.float32)], axis=1)[:, None, :]  # (N,1,4)
    temporal, temporal_effective = encode_route_temporal_2d(
        start_t,
        tz_offset_hours=float(cfg.temporal_tz_offset_hours),
        mode=str(cfg.temporal_mode),
    )
    # cond: (hour,day) + (start_bin, dest_bin)
    base_cond = np.concatenate([temporal, start_ctr_norm, dest_ctr_norm], axis=1).astype(np.float32, copy=False)  # (N,6)

    sem_norm = None
    sem_keys = None
    sem_cfg = None
    grid_patch = None
    grid_keys = None
    grid_norm_cfg: Optional[SemanticGridNorm] = None

    sem_mode = str(cfg.semantic_mode)
    uses_semantics = bool(cfg.semantic_dir) or (sem_mode in ("rand4", "gridcnn", "od_gridcnn", "gridpos", "od_gridpos", "gridattn", "od_gridattn"))
    sem_use_bins = (True if sem_mode == "rand4" else bool(cfg.semantic_use_bins))
    if uses_semantics:
        sem_o = start_ctr if bool(sem_use_bins) else start_pos
        sem_d = dest_ctr if bool(sem_use_bins) else dest_pos

        parts = []
        keys_all = []
        if sem_mode == "rand4":
            sem_r, sem_keys_r = semantic_rand4_features(start_ctr=start_ctr, dest_ctr=dest_ctr)
            parts.append(sem_r)
            keys_all.extend(list(sem_keys_r))
        if sem_mode in ("od", "od_profile", "od_grid", "od_gridcnn", "od_gridpos", "od_gridattn"):
            if not cfg.semantic_dir:
                raise ValueError("--semantic_dir is required for semantic_mode including 'od'")
            poi_total, landuse_entropy = load_poi_total_and_landuse_entropy(cfg.semantic_dir)
            sem_od, sem_keys_od = semantic_od_features(
                start_ctr=sem_o,
                dest_ctr=sem_d,
                poi_total=poi_total,
                landuse_entropy=landuse_entropy,
                log_poi=True,
            )
            parts.append(sem_od)
            keys_all.extend(list(sem_keys_od))
        if sem_mode in ("profile", "od_profile"):
            if not cfg.semantic_dir:
                raise ValueError("--semantic_dir is required for semantic_mode including 'profile'")
            poi_stack, categories, landuse_entropy = load_poi_stack_and_landuse_entropy(cfg.semantic_dir)
            offsets = _parse_float_list(cfg.profile_offsets)
            sem_prof, sem_keys_prof = semantic_corridor_profile_features(
                start_ctr=sem_o,
                dest_ctr=sem_d,
                poi_stack=poi_stack,
                categories=categories,
                landuse_entropy=landuse_entropy,
                num_steps=int(cfg.profile_num_steps),
                offsets=offsets,
                log_total=True,
            )
            parts.append(sem_prof)
            keys_all.extend(list(sem_keys_prof))
        if sem_mode in ("grid", "od_grid"):
            if not cfg.semantic_dir:
                raise ValueError("--semantic_dir is required for semantic_mode including 'grid'")
            poi_stack, categories, landuse_entropy = load_poi_stack_and_landuse_entropy(cfg.semantic_dir)
            sem_grid, sem_keys_grid = semantic_grid_pool_features(
                start_ctr=sem_o,
                dest_ctr=sem_d,
                poi_stack=poi_stack,
                categories=categories,
                landuse_entropy=landuse_entropy,
                patch_size=int(cfg.grid_patch_size),
                extent=float(cfg.grid_extent),
                pool=str(cfg.grid_pool),
                log_poi=True,
            )
            parts.append(sem_grid)
            keys_all.extend(list(sem_keys_grid))

        if parts:
            sem_raw = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1).astype(np.float32, copy=False)
            sem_keys = tuple(str(k) for k in keys_all)
            sem_cfg = fit_semantic_norm(sem_raw, keys=sem_keys)
            sem_norm = normalize_semantic(sem_raw, sem_cfg)

        if sem_mode in ("gridcnn", "od_gridcnn", "gridpos", "od_gridpos", "gridattn", "od_gridattn"):
            if not cfg.semantic_dir:
                raise ValueError("--semantic_dir is required for semantic_mode=gridcnn/od_gridcnn/gridpos/od_gridpos/gridattn/od_gridattn")
            chans = {x.strip() for x in str(cfg.grid_channels).split(",") if x.strip()}
            need_poi = ("poi" in chans) or ("entropy" in chans)
            poi_stack = None
            categories = None
            landuse_entropy = None
            osm_road_prob = None
            if need_poi:
                poi_stack, categories, landuse_entropy = load_poi_stack_and_landuse_entropy(cfg.semantic_dir)
            if "road_prob" in chans:
                osm_road_prob = load_osm_road_prob(cfg.semantic_dir)
            # Important: gridpos/grid semantics must align with the waypoint (s,o) frame, which is defined by raw start/dest.
            patch_o = start_pos if sem_mode in ("gridpos", "od_gridpos") else sem_o
            patch_d = dest_pos if sem_mode in ("gridpos", "od_gridpos") else sem_d
            grid_patch_raw, grid_keys = semantic_grid_patch_tensor(
                start_ctr=patch_o,
                dest_ctr=patch_d,
                poi_stack=poi_stack,
                categories=categories,
                landuse_entropy=landuse_entropy,
                osm_road_prob=osm_road_prob,
                patch_size=int(cfg.grid_patch_size),
                extent=float(cfg.grid_extent),
                grid_channels=str(cfg.grid_channels),
                log_poi=True,
            )
            grid_norm_cfg = fit_grid_norm(grid_patch_raw, keys=grid_keys)
            grid_patch = normalize_grid_patch(grid_patch_raw, grid_norm_cfg)

    if grid_patch is not None:
        cond_base = base_cond
        cond_sem = sem_norm
        cond = None
    else:
        cond = np.concatenate([base_cond, sem_norm], axis=1).astype(np.float32, copy=False) if sem_norm is not None else base_cond

    # Oracle targets: waypoints -> chord-relative (s,o).
    wp_abs = _extract_oracle_waypoints(
        start_pos=start_pos,
        targets=targets,
        num_waypoints=int(cfg.num_waypoints),
        waypoint_mode=str(cfg.waypoint_mode),
        waypoint_turn_alpha=float(cfg.waypoint_turn_alpha),
    )  # (N,K,2)
    rel = _waypoints_rel_so(start=start_pos, dest=dest_pos, waypoints=wp_abs)  # (N,K,2)
    rel[:, :, 0] = np.clip(rel[:, :, 0], 0.0, 1.0)
    if float(cfg.o_clip) > 0.0:
        rel[:, :, 1] = np.clip(rel[:, :, 1], -float(cfg.o_clip), float(cfg.o_clip))

    rel_mean, rel_std = _compute_rel_norm(rel)
    target_rel_norm = _normalize_rel(rel, mean=rel_mean, std=rel_std)

    if grid_patch is not None:
        dataset = WaypointRelDatasetWithGrid(
            obs=obs,
            start_pos=start_pos,
            dest_pos=dest_pos,
            cond_base=cond_base,
            cond_sem=cond_sem,
            grid_patch=grid_patch,
            target_rel_norm=target_rel_norm,
            traj_idx=traj_idx,
            start_t=start_t,
        )
    else:
        dataset = WaypointRelDataset(obs=obs, cond=cond, target_rel_norm=target_rel_norm, traj_idx=traj_idx, start_t=start_t)
    g = torch.Generator()
    g.manual_seed(int(cfg.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        generator=g,
        pin_memory=bool(torch.cuda.is_available()),
        persistent_workers=(int(cfg.num_workers) > 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_encoder = None
    posenc = None
    attn_control = None
    if grid_patch is not None:
        assert grid_keys is not None and grid_norm_cfg is not None
        if sem_mode in ("gridcnn", "od_gridcnn"):
            grid_encoder = GridCNNEncoder(in_channels=int(grid_patch.shape[1]), out_dim=int(cfg.grid_emb_dim)).to(device=device)
            cond_dim = 6 + (0 if cond_sem is None else int(cond_sem.shape[1])) + int(cfg.grid_emb_dim)
        else:
            # gridpos: patch is used via position-aligned conditioning (extra embedding), not concatenated into cond.
            cond_dim = 6 + (0 if cond_sem is None else int(cond_sem.shape[1]))
            if sem_mode in ("gridpos", "od_gridpos"):
                posenc = WaypointSemanticPosEnc(
                    in_channels=int(grid_patch.shape[1]),
                    num_waypoints=int(k),
                    extent=float(cfg.grid_extent),
                    rel_mean=torch.from_numpy(rel_mean),
                    rel_std=torch.from_numpy(rel_std),
                    emb_dim=int(cfg.hidden_dim) * 4,
                    diff_steps=int(cfg.diff_steps),
                    mlp_hidden_dim=int(cfg.semantic_posenc_hidden_dim),
                    weight=float(cfg.semantic_posenc_weight),
                ).to(device=device)
            elif sem_mode in ("gridattn", "od_gridattn"):
                attn_control = GridCrossAttentionControlMid(
                    in_channels=int(grid_patch.shape[1]),
                    act_dim=2,
                    model_dim=int(cfg.hidden_dim) * 4,
                    num_heads=int(cfg.semantic_attn_heads),
                    diff_steps=int(cfg.diff_steps),
                    weight=float(cfg.semantic_attn_weight),
                ).to(device=device)
    else:
        cond_dim = int(cond.shape[1])

    if float(cfg.semantic_loss_weight) > 0.0 and grid_patch is None:
        raise ValueError("--semantic_loss_weight requires semantic_mode with grid patch (gridcnn/od_gridcnn/gridpos/od_gridpos).")

    model = DiffusionTrajectoryModel(
        obs_dim=4,
        act_dim=2,
        cond_dim=int(cond_dim),
        obs_len=1,
        pred_len=int(k),
        hidden_dim=int(cfg.hidden_dim),
        diffusion_steps=int(cfg.diff_steps),
        prediction_type=str(cfg.pred_type),
    ).to(device=device)
    params = list(model.parameters())
    if grid_encoder is not None:
        params.extend(list(grid_encoder.parameters()))
    if posenc is not None:
        params.extend(list(posenc.parameters()))
    if attn_control is not None:
        params.extend(list(attn_control.parameters()))
    optimizer = optim.Adam(params, lr=float(cfg.lr))
    rel_mean_t = torch.from_numpy(rel_mean).to(device=device, dtype=torch.float32)
    rel_std_t = torch.from_numpy(rel_std).to(device=device, dtype=torch.float32)

    start_wall = time.time()
    model.train()
    for epoch in range(int(cfg.epochs)):
        epoch_loss = 0.0
        epoch_steps = 0
        total = int(cfg.max_batches) if cfg.max_batches is not None else len(loader)
        pbar = tqdm(enumerate(loader), total=total, desc=f"epoch {epoch+1}/{int(cfg.epochs)}", dynamic_ncols=True)
        for batch_idx, batch in pbar:
            if cfg.max_batches is not None and int(batch_idx) >= int(cfg.max_batches):
                break
            obs_b = batch["obs"].to(device=device, non_blocking=True)
            if grid_patch is not None:
                cond_parts = [batch["cond_base"].to(device=device, non_blocking=True)]
                if "cond_sem" in batch:
                    cond_parts.append(batch["cond_sem"].to(device=device, non_blocking=True))
                if grid_encoder is not None:
                    patch_b = batch["grid_patch"].to(device=device, non_blocking=True)
                    emb_b = grid_encoder(patch_b)
                    cond_parts.append(emb_b)
                cond_b = torch.cat(cond_parts, dim=1)
            else:
                cond_b = batch["cond"].to(device=device, non_blocking=True)
            target_b = batch["action"].to(device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if posenc is None and attn_control is None and float(cfg.semantic_loss_weight) <= 0.0:
                loss = model(obs_b, cond_b, target=target_b)
            else:
                patch_b = batch["grid_patch"].to(device=device, non_blocking=True) if grid_patch is not None else None
                start_b = batch["start_pos"].to(device=device, non_blocking=True) if grid_patch is not None else None
                dest_b = batch["dest_pos"].to(device=device, non_blocking=True) if grid_patch is not None else None

                def _extra(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                    assert posenc is not None and patch_b is not None and start_b is not None and dest_b is not None
                    return posenc(x_t, t, grid_patch=patch_b, start_pos=start_b, dest_pos=dest_b)

                def _unet_kwargs(x_t: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
                    assert attn_control is not None and patch_b is not None
                    ctrl_mid, _ = attn_control(x_t, t, grid_patch=patch_b)
                    return {"control_mid": ctrl_mid}

                need_x0 = bool(float(cfg.semantic_loss_weight) > 0.0)
                if posenc is not None:
                    if bool(cfg.semantic_posenc_self_correct):

                        def _extra_x0(x_t: torch.Tensor, t: torch.Tensor, x0_pred: torch.Tensor) -> torch.Tensor:
                            assert posenc is not None and patch_b is not None and start_b is not None and dest_b is not None
                            return posenc(x0_pred, t, grid_patch=patch_b, start_pos=start_b, dest_pos=dest_b)

                        out = model.compute_loss(
                            obs_b,
                            cond_b,
                            target_b,
                            cond_emb_extra_fn=None,
                            cond_emb_extra_fn_x0=_extra_x0,
                            unet_kwargs_fn=None,
                            return_x0_pred=need_x0,
                        )
                    else:
                        out = model.compute_loss(
                            obs_b,
                            cond_b,
                            target_b,
                            cond_emb_extra_fn=_extra,
                            unet_kwargs_fn=None,
                            return_x0_pred=need_x0,
                        )
                elif attn_control is not None:
                    out = model.compute_loss(
                        obs_b,
                        cond_b,
                        target_b,
                        cond_emb_extra_fn=None,
                        unet_kwargs_fn=_unet_kwargs,
                        return_x0_pred=need_x0,
                    )
                else:
                    out = model.compute_loss(obs_b, cond_b, target_b, return_x0_pred=need_x0)
                if need_x0:
                    diff_loss, x0_pred = out  # type: ignore[misc]
                    assert patch_b is not None and start_b is not None and dest_b is not None
                    rel_pred = x0_pred.permute(0, 2, 1) * rel_std_t[None, None, :] + rel_mean_t[None, None, :]
                    rel_gt = target_b * rel_std_t[None, None, :] + rel_mean_t[None, None, :]
                    sem_pred = sample_patch_mean_along_skeleton(
                        patch=patch_b,
                        start_pos=start_b,
                        dest_pos=dest_b,
                        rel=rel_pred,
                        extent=float(cfg.grid_extent),
                        samples_per_segment=int(cfg.semantic_loss_samples_per_segment),
                    )
                    sem_gt = sample_patch_mean_along_skeleton(
                        patch=patch_b,
                        start_pos=start_b,
                        dest_pos=dest_b,
                        rel=rel_gt,
                        extent=float(cfg.grid_extent),
                        samples_per_segment=int(cfg.semantic_loss_samples_per_segment),
                    ).detach()
                    sem_loss = torch.mean((sem_pred - sem_gt) ** 2)
                    loss = diff_loss + float(cfg.semantic_loss_weight) * sem_loss
                else:
                    loss = out  # type: ignore[assignment]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if grid_encoder is not None:
                torch.nn.utils.clip_grad_norm_(grid_encoder.parameters(), 1.0)
            if posenc is not None:
                torch.nn.utils.clip_grad_norm_(posenc.parameters(), 1.0)
            if attn_control is not None:
                torch.nn.utils.clip_grad_norm_(attn_control.parameters(), 1.0)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())
            epoch_steps += 1
            if epoch_steps > 0 and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=float(loss.detach().cpu().item()), avg=float(epoch_loss / max(epoch_steps, 1)))

        avg_loss = epoch_loss / max(epoch_steps, 1)
        torch.save(
            {
                "epoch": int(epoch),
                "loss": float(avg_loss),
                "model_state_dict": model.state_dict(),
                "grid_encoder_state_dict": (grid_encoder.state_dict() if grid_encoder is not None else None),
                "semantic_posenc_state_dict": (posenc.state_dict() if posenc is not None else None),
                "semantic_attn_state_dict": (attn_control.state_dict() if attn_control is not None else None),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "task": "route_wp_diffusion_rel_npz",
                    "K_waypoints": int(k),
                    "model": {
                        "hidden_dim": int(cfg.hidden_dim),
                        "diff_steps": int(cfg.diff_steps),
                        "pred_type": str(cfg.pred_type),
                        "obs_len": 1,
                        "cond_dim": int(cond_dim),
                    },
                    "od_bin": float(cfg.od_bin),
                    "o_clip": float(cfg.o_clip),
                    "temporal": {
                        "mode": str(cfg.temporal_mode),
                        "tz_offset_hours": float(cfg.temporal_tz_offset_hours),
                        "effective": str(temporal_effective),
                    },
                    "waypoints": {
                        "mode": str(cfg.waypoint_mode),
                        "turn_alpha": float(cfg.waypoint_turn_alpha),
                    },
                    "pos_norm": {
                        "pos_min": [float(x) for x in norm.pos_min.tolist()],
                        "pos_max": [float(x) for x in norm.pos_max.tolist()],
                    },
                    "rel_norm": {"mean": [float(x) for x in rel_mean.tolist()], "std": [float(x) for x in rel_std.tolist()]},
                    "semantic": {
                        "mode": (str(cfg.semantic_mode) if bool(uses_semantics) else None),
                        "use_bins": bool(sem_use_bins),
                        "profile_num_steps": int(cfg.profile_num_steps),
                        "profile_offsets": str(cfg.profile_offsets),
                        "grid_patch_size": int(cfg.grid_patch_size),
                        "grid_extent": float(cfg.grid_extent),
                        "grid_pool": str(cfg.grid_pool),
                        "grid_channels": str(cfg.grid_channels),
                        "grid_emb_dim": int(cfg.grid_emb_dim),
                        "grid_frame": ("raw" if sem_mode in ("gridpos", "od_gridpos") else ("bins" if bool(sem_use_bins) else "raw")),
                        "posenc_hidden_dim": int(cfg.semantic_posenc_hidden_dim),
                        "posenc_weight": float(cfg.semantic_posenc_weight),
                        "posenc_self_correct": bool(cfg.semantic_posenc_self_correct),
                        "attn_heads": int(cfg.semantic_attn_heads),
                        "attn_weight": float(cfg.semantic_attn_weight),
                        "semantic_loss_weight": float(cfg.semantic_loss_weight),
                        "semantic_loss_samples_per_segment": int(cfg.semantic_loss_samples_per_segment),
                    },
                    "semantic_od_norm": (sem_cfg.to_json() if sem_cfg is not None else None),
                    "semantic_grid_norm": (grid_norm_cfg.to_json() if grid_norm_cfg is not None else None),
                },
            },
            ckpt_path,
        )

    elapsed_s = float(time.time() - start_wall)
    result = {
        "inputs": {"train_npz": str(Path(cfg.train_npz).resolve())},
        "config": {
            "pos_max": int(cfg.pos_max),
            "max_train_n": (int(cfg.max_train_n) if cfg.max_train_n is not None else None),
            "num_waypoints": int(cfg.num_waypoints),
            "waypoint_mode": str(cfg.waypoint_mode),
            "waypoint_turn_alpha": float(cfg.waypoint_turn_alpha),
            "od_bin": float(cfg.od_bin),
            "o_clip": float(cfg.o_clip),
            "temporal_mode": str(cfg.temporal_mode),
            "temporal_tz_offset_hours": float(cfg.temporal_tz_offset_hours),
            "temporal_effective": str(temporal_effective),
            "semantic_dir": (str(Path(cfg.semantic_dir).resolve()) if cfg.semantic_dir else None),
            "semantic_mode": (str(cfg.semantic_mode) if bool(uses_semantics) else None),
            "semantic_use_bins": bool(sem_use_bins),
            "profile_num_steps": int(cfg.profile_num_steps),
            "profile_offsets": str(cfg.profile_offsets),
            "grid_patch_size": int(cfg.grid_patch_size),
            "grid_extent": float(cfg.grid_extent),
            "grid_pool": str(cfg.grid_pool),
            "grid_channels": str(cfg.grid_channels),
            "grid_emb_dim": int(cfg.grid_emb_dim),
            "semantic_posenc_hidden_dim": int(cfg.semantic_posenc_hidden_dim),
            "semantic_posenc_weight": float(cfg.semantic_posenc_weight),
            "semantic_attn_heads": int(cfg.semantic_attn_heads),
            "semantic_attn_weight": float(cfg.semantic_attn_weight),
            "semantic_loss_weight": float(cfg.semantic_loss_weight),
            "semantic_loss_samples_per_segment": int(cfg.semantic_loss_samples_per_segment),
            "hidden_dim": int(cfg.hidden_dim),
            "diff_steps": int(cfg.diff_steps),
            "pred_type": str(cfg.pred_type),
            "batch_size": int(cfg.batch_size),
            "epochs": int(cfg.epochs),
            "lr": float(cfg.lr),
            "num_workers": int(cfg.num_workers),
            "max_batches": (int(cfg.max_batches) if cfg.max_batches is not None else None),
            "seed": int(cfg.seed),
        },
        "stats": {"N": int(n), "F": int(targets.shape[1]), "rel_mean": [float(x) for x in rel_mean.tolist()], "rel_std": [float(x) for x in rel_std.tolist()]},
        "outputs": {"checkpoint": str(ckpt_path.resolve())},
        "timing": {"elapsed_s": float(elapsed_s)},
    }
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
