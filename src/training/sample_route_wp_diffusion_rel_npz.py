from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.features.semantic_od import (
    SemanticGridNorm,
    SemanticODNorm,
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
from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.models.semantic.grid_cross_attention_control import GridCrossAttentionControlMid
from src.models.semantic.grid_cnn_encoder import GridCNNEncoder
from src.models.semantic.waypoint_semantic_posenc import WaypointSemanticPosEnc
from src.training.route_npz_utils import RouteNorm, load_route_windows_npz, make_default_pos_bounds, normalize_pos

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):  # type: ignore[no-redef]
        return x


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


def _parse_float_list(s: str) -> Tuple[float, ...]:
    items = [x.strip() for x in str(s).split(",") if str(x).strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated list.")
    out = []
    for x in items:
        out.append(float(x))
    return tuple(out)


def _waypoints_from_rel_so(
    *,
    start: np.ndarray,  # (N,2)
    dest: np.ndarray,  # (N,2)
    rel: np.ndarray,  # (N,K,2) (s,o)
    eps: float = 1e-6,
) -> np.ndarray:
    start = np.asarray(start, dtype=np.float32)
    dest = np.asarray(dest, dtype=np.float32)
    rel = np.asarray(rel, dtype=np.float32)
    v = dest - start
    L = np.linalg.norm(v, axis=1).astype(np.float32)
    L = np.maximum(L, float(eps))
    e_par = v / L[:, None]
    e_perp = np.stack([-e_par[:, 1], e_par[:, 0]], axis=1)
    s = rel[:, :, 0]
    o = rel[:, :, 1]
    wp = (
        start[:, None, :]
        + (s[:, :, None] * L[:, None, None]) * e_par[:, None, :]
        + (o[:, :, None] * L[:, None, None]) * e_perp[:, None, :]
    )
    return wp.astype(np.float32, copy=False)


def _poly_arclength(points: np.ndarray) -> Tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points (M,2), got {points.shape}")
    if points.shape[0] < 2:
        return np.zeros((points.shape[0],), dtype=np.float32), 0.0
    seg = points[1:] - points[:-1]
    seg_len = np.linalg.norm(seg, axis=1).astype(np.float32)
    s = np.concatenate([[0.0], np.cumsum(seg_len)], axis=0).astype(np.float32)
    return s, float(s[-1])


def _resample_by_arclength(points: np.ndarray, *, num: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    s_nodes, total = _poly_arclength(points)
    if int(num) <= 1:
        return points[:1]
    if not np.isfinite(total) or total <= 1e-6:
        return np.repeat(points[:1], repeats=int(num), axis=0)
    s_query = np.linspace(0.0, total, num=int(num), dtype=np.float32)
    y = np.interp(s_query, s_nodes, points[:, 0]).astype(np.float32)
    x = np.interp(s_query, s_nodes, points[:, 1]).astype(np.float32)
    return np.stack([y, x], axis=1).astype(np.float32)


def _load_norm_from_ckpt(cfg: dict, *, pos_max_default: int = 1023) -> RouteNorm:
    pos_norm = cfg.get("pos_norm") if isinstance(cfg, dict) else None
    if not isinstance(pos_norm, dict):
        pos_min, pos_max = make_default_pos_bounds(pos_max=int(pos_max_default))
        pos_range = (pos_max - pos_min + 1e-6).astype(np.float32)
        return RouteNorm(
            pos_min=pos_min,
            pos_max=pos_max,
            pos_range=pos_range,
            vel_mean=np.zeros((2,), dtype=np.float32),
            vel_std=np.ones((2,), dtype=np.float32),
        )
    pos_min = np.asarray(pos_norm.get("pos_min", [0.0, 0.0]), dtype=np.float32).reshape(2)
    pos_max = np.asarray(pos_norm.get("pos_max", [float(pos_max_default), float(pos_max_default)]), dtype=np.float32).reshape(2)
    pos_range = (pos_max - pos_min + 1e-6).astype(np.float32)
    return RouteNorm(pos_min=pos_min, pos_max=pos_max, pos_range=pos_range, vel_mean=np.zeros((2,), dtype=np.float32), vel_std=np.ones((2,), dtype=np.float32))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample a waypoint-diffusion decision model and emit skeleton-only routes (samples.npz).")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--case_npz", type=str, required=True, help="case_XX/gt_case.npz from route_gt_baseline.py")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument(
        "--semantic_dir",
        type=str,
        default=None,
        help="Optional directory containing poi_density_*.npy and landuse_entropy.npy (required if checkpoint uses semantics).",
    )
    p.add_argument("--num_samples_per_condition", type=int, default=20, help="K samples per condition")
    p.add_argument("--max_n", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    k_wp = int(cfg.get("K_waypoints", 0) if isinstance(cfg, dict) else 0)
    if k_wp <= 0:
        raise ValueError("Checkpoint missing K_waypoints")
    od_bin = float(cfg.get("od_bin", 128.0))
    o_clip = float(cfg.get("o_clip", 2.0))
    temporal_meta = cfg.get("temporal", {}) if isinstance(cfg, dict) else {}
    temporal_mode = "zeros"
    temporal_tz = -5.0
    if isinstance(temporal_meta, dict):
        temporal_mode = str(temporal_meta.get("effective") or temporal_meta.get("mode") or "zeros")
        temporal_tz = float(temporal_meta.get("tz_offset_hours", -5.0))

    rel_norm = cfg.get("rel_norm", {}) if isinstance(cfg, dict) else {}
    rel_mean = np.asarray(rel_norm.get("mean", [0.0, 0.0]), dtype=np.float32).reshape(2)
    rel_std = np.asarray(rel_norm.get("std", [1.0, 1.0]), dtype=np.float32).reshape(2)
    rel_std = np.maximum(rel_std, 1e-3).astype(np.float32, copy=False)

    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    diff_steps = int(model_cfg.get("diff_steps", 50))
    pred_type = str(model_cfg.get("pred_type", "eps"))
    cond_dim = int(model_cfg.get("cond_dim", 6))

    norm = _load_norm_from_ckpt(cfg if isinstance(cfg, dict) else {})

    case = load_route_windows_npz(str(args.case_npz), max_n=(int(args.max_n) if args.max_n is not None else None), seed=int(args.seed))
    start_pos = np.asarray(case["start_pos"], dtype=np.float32)
    dest_pos = np.asarray(case["dest_pos"], dtype=np.float32)
    traj_idx = np.asarray(case["traj_idx"], dtype=np.int64)
    start_t = np.asarray(case["start_t"], dtype=np.int64)
    targets = np.asarray(case["targets"], dtype=np.float32)

    n = int(start_pos.shape[0])
    f = int(targets.shape[1])

    # Conditioning uses OD-bin centers (shared intent).
    start_ctr = _od_bin_center(start_pos, bin_size=float(od_bin))
    dest_ctr = _od_bin_center(dest_pos, bin_size=float(od_bin))
    start_ctr_norm = normalize_pos(start_ctr, norm)
    dest_ctr_norm = normalize_pos(dest_ctr, norm)

    obs = np.concatenate([start_ctr_norm, np.zeros((n, 2), dtype=np.float32)], axis=1)[:, None, :]  # (N,1,4)
    temporal, _temporal_eff = encode_route_temporal_2d(start_t, tz_offset_hours=float(temporal_tz), mode=str(temporal_mode))
    base_cond = np.concatenate([temporal, start_ctr_norm, dest_ctr_norm], axis=1).astype(np.float32, copy=False)  # (N,6)

    sem_cfg_raw = cfg.get("semantic_od_norm") if isinstance(cfg, dict) else None
    if sem_cfg_raw is not None and not isinstance(sem_cfg_raw, dict):
        raise TypeError(f"Bad semantic_od_norm in checkpoint config: {type(sem_cfg_raw)}")
    sem_cfg = SemanticODNorm.from_json(sem_cfg_raw) if isinstance(sem_cfg_raw, dict) else None

    sem_meta = cfg.get("semantic") if isinstance(cfg, dict) else None
    if isinstance(sem_meta, dict) and sem_meta.get("mode") is not None:
        sem_mode = str(sem_meta.get("mode"))
        sem_use_bins = bool(sem_meta.get("use_bins", False))
        profile_num_steps = int(sem_meta.get("profile_num_steps", 16))
        profile_offsets_str = str(sem_meta.get("profile_offsets", "-32,0,32"))
        profile_offsets = _parse_float_list(profile_offsets_str)
        grid_patch_size = int(sem_meta.get("grid_patch_size", 16))
        grid_extent = float(sem_meta.get("grid_extent", 128.0))
        grid_pool = str(sem_meta.get("grid_pool", "quad"))
        grid_channels = str(sem_meta.get("grid_channels", "poi,entropy"))
        grid_emb_dim = int(sem_meta.get("grid_emb_dim", 64))
        posenc_hidden_dim = int(sem_meta.get("posenc_hidden_dim", 256))
        posenc_weight = float(sem_meta.get("posenc_weight", 1.0))
        posenc_self_correct = bool(sem_meta.get("posenc_self_correct", False))
        grid_frame = str(sem_meta.get("grid_frame", "raw"))
        attn_heads = int(sem_meta.get("attn_heads", 4))
        attn_weight = float(sem_meta.get("attn_weight", 1.0))
    else:
        sem_mode = None
        sem_use_bins = False
        profile_num_steps = 16
        profile_offsets = (-32.0, 0.0, 32.0)
        grid_patch_size = 16
        grid_extent = 128.0
        grid_pool = "quad"
        grid_channels = "poi,entropy"
        grid_emb_dim = 64
        posenc_hidden_dim = 256
        posenc_weight = 1.0
        posenc_self_correct = False
        grid_frame = "raw"
        attn_heads = 4
        attn_weight = 1.0

    # Backward-compatible fallback for older checkpoints with OD-only semantics.
    if sem_mode is None and sem_cfg is not None:
        sem_mode = "od"

    cond_parts = [base_cond]
    uses_semantics = bool(sem_mode is not None)

    if sem_mode is not None:
        if sem_mode not in ("rand4",) and (
            sem_mode not in ("gridcnn", "od_gridcnn", "gridpos", "od_gridpos", "gridattn", "od_gridattn")
        ) and not args.semantic_dir:
            raise ValueError("--semantic_dir is required because checkpoint includes semantic features")
        if sem_mode in ("gridcnn", "od_gridcnn", "gridpos", "od_gridpos", "gridattn", "od_gridattn") and not args.semantic_dir:
            raise ValueError("--semantic_dir is required because checkpoint includes grid semantics")

        sem_o = start_ctr if sem_use_bins else start_pos
        sem_d = dest_ctr if sem_use_bins else dest_pos

        # Vector semantics (normalized by semantic_od_norm).
        if sem_mode in ("rand4", "od", "od_profile", "od_grid", "profile", "grid", "od_gridcnn", "od_gridpos", "od_gridattn"):
            if sem_cfg is None:
                raise ValueError("Checkpoint semantic mode requires semantic_od_norm, but it is missing.")
            parts = []
            keys_all = []
            if sem_mode == "rand4":
                sem_r, sem_keys_r = semantic_rand4_features(start_ctr=start_ctr, dest_ctr=dest_ctr)
                parts.append(sem_r)
                keys_all.extend(list(sem_keys_r))
            if sem_mode in ("od", "od_profile", "od_grid", "od_gridcnn"):
                poi_total, landuse_entropy = load_poi_total_and_landuse_entropy(args.semantic_dir)
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
                poi_stack, categories, landuse_entropy = load_poi_stack_and_landuse_entropy(args.semantic_dir)
                sem_prof, sem_keys_prof = semantic_corridor_profile_features(
                    start_ctr=sem_o,
                    dest_ctr=sem_d,
                    poi_stack=poi_stack,
                    categories=categories,
                    landuse_entropy=landuse_entropy,
                    num_steps=int(profile_num_steps),
                    offsets=profile_offsets,
                    log_total=True,
                )
                parts.append(sem_prof)
                keys_all.extend(list(sem_keys_prof))
            if sem_mode in ("grid", "od_grid"):
                poi_stack, categories, landuse_entropy = load_poi_stack_and_landuse_entropy(args.semantic_dir)
                sem_grid, sem_keys_grid = semantic_grid_pool_features(
                    start_ctr=sem_o,
                    dest_ctr=sem_d,
                    poi_stack=poi_stack,
                    categories=categories,
                    landuse_entropy=landuse_entropy,
                    patch_size=int(grid_patch_size),
                    extent=float(grid_extent),
                    pool=str(grid_pool),
                    log_poi=True,
                )
                parts.append(sem_grid)
                keys_all.extend(list(sem_keys_grid))
            if not parts:
                raise ValueError(f"Bad semantic mode in checkpoint: {sem_mode}")
            sem_raw = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1).astype(np.float32, copy=False)
            keys = tuple(str(k) for k in keys_all)
            if keys != sem_cfg.keys:
                raise ValueError(f"Semantic keys mismatch: ckpt={sem_cfg.keys} vs computed={keys}")
            sem_norm = normalize_semantic(sem_raw, sem_cfg)
            cond_parts.append(sem_norm.astype(np.float32, copy=False))

        # Grid-CNN semantics.
        if sem_mode in ("gridcnn", "od_gridcnn", "gridpos", "od_gridpos", "gridattn", "od_gridattn"):
            grid_norm_raw = cfg.get("semantic_grid_norm") if isinstance(cfg, dict) else None
            if grid_norm_raw is None or not isinstance(grid_norm_raw, dict):
                raise ValueError("Checkpoint includes grid semantics but missing semantic_grid_norm.")
            grid_norm = SemanticGridNorm.from_json(grid_norm_raw)

            enc_state = ckpt.get("grid_encoder_state_dict")
            posenc_state = ckpt.get("semantic_posenc_state_dict")
            attn_state = ckpt.get("semantic_attn_state_dict")

            chans = {x.strip() for x in str(grid_channels).split(",") if x.strip()}
            need_poi = ("poi" in chans) or ("entropy" in chans)
            poi_stack = None
            categories = None
            landuse_entropy = None
            osm_road_prob = None
            if need_poi:
                poi_stack, categories, landuse_entropy = load_poi_stack_and_landuse_entropy(args.semantic_dir)
            if "road_prob" in chans:
                osm_road_prob = load_osm_road_prob(args.semantic_dir)

            patch_o = start_pos if sem_mode in ("gridpos", "od_gridpos") else sem_o
            patch_d = dest_pos if sem_mode in ("gridpos", "od_gridpos") else sem_d
            grid_patch_raw, grid_keys = semantic_grid_patch_tensor(
                start_ctr=patch_o,
                dest_ctr=patch_d,
                poi_stack=poi_stack,
                categories=categories,
                landuse_entropy=landuse_entropy,
                osm_road_prob=osm_road_prob,
                patch_size=int(grid_patch_size),
                extent=float(grid_extent),
                grid_channels=str(grid_channels),
                log_poi=True,
            )
            if tuple(grid_keys) != grid_norm.keys:
                raise ValueError(f"Grid keys mismatch: ckpt={grid_norm.keys} vs computed={grid_keys}")
            grid_patch = normalize_grid_patch(grid_patch_raw, grid_norm)

            if sem_mode in ("gridcnn", "od_gridcnn"):
                if enc_state is None or not isinstance(enc_state, dict):
                    raise ValueError("Checkpoint includes gridcnn but missing grid_encoder_state_dict.")
                device_enc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                grid_encoder = GridCNNEncoder(in_channels=int(grid_patch.shape[1]), out_dim=int(grid_emb_dim)).to(device=device_enc)
                grid_encoder.load_state_dict(enc_state)
                grid_encoder.eval()
                with torch.no_grad():
                    patch_t = torch.from_numpy(grid_patch).to(device=device_enc, dtype=torch.float32)
                    emb = grid_encoder(patch_t).detach().cpu().numpy().astype(np.float32, copy=False)
                cond_parts.append(emb)
            else:
                if sem_mode in ("gridpos", "od_gridpos"):
                    if posenc_state is None or not isinstance(posenc_state, dict):
                        raise ValueError("Checkpoint includes gridpos but missing semantic_posenc_state_dict.")
                if sem_mode in ("gridattn", "od_gridattn"):
                    if attn_state is None or not isinstance(attn_state, dict):
                        raise ValueError("Checkpoint includes gridattn but missing semantic_attn_state_dict.")

    cond = np.concatenate(cond_parts, axis=1).astype(np.float32, copy=False)

    if int(cond.shape[1]) != int(cond_dim):
        raise ValueError(f"cond_dim mismatch: ckpt={cond_dim} vs built={cond.shape[1]}")

    obs_t = torch.from_numpy(obs).to(dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionTrajectoryModel(
        obs_dim=4,
        act_dim=2,
        cond_dim=int(cond_dim),
        obs_len=1,
        pred_len=int(k_wp),
        hidden_dim=int(hidden_dim),
        diffusion_steps=int(diff_steps),
        prediction_type=str(pred_type),
    ).to(device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    obs_t = obs_t.to(device=device)
    cond_t = torch.from_numpy(cond).to(device=device, dtype=torch.float32)

    k_samples = int(args.num_samples_per_condition)
    preds_k = np.zeros((n, k_samples, f, 2), dtype=np.float32)

    posenc = None
    patch_t = None
    start_t_t = None
    dest_t_t = None
    rel_mean_t = torch.from_numpy(rel_mean).to(device=device, dtype=torch.float32)
    rel_std_t = torch.from_numpy(rel_std).to(device=device, dtype=torch.float32)
    if sem_mode in ("gridpos", "od_gridpos"):
        if grid_patch is None:
            raise ValueError("gridpos requires grid_patch")
        posenc_state = ckpt.get("semantic_posenc_state_dict")
        if posenc_state is None or not isinstance(posenc_state, dict):
            raise ValueError("Checkpoint includes gridpos but missing semantic_posenc_state_dict.")
        posenc = WaypointSemanticPosEnc(
            in_channels=int(grid_patch.shape[1]),
            num_waypoints=int(k_wp),
            extent=float(grid_extent),
            rel_mean=rel_mean_t,
            rel_std=rel_std_t,
            emb_dim=int(hidden_dim) * 4,
            diff_steps=int(diff_steps),
            mlp_hidden_dim=int(posenc_hidden_dim),
            weight=float(posenc_weight),
        ).to(device=device)
        posenc.load_state_dict(posenc_state)
        posenc.eval()
        patch_t = torch.from_numpy(grid_patch).to(device=device, dtype=torch.float32)
        start_t_t = torch.from_numpy(start_pos).to(device=device, dtype=torch.float32)
        dest_t_t = torch.from_numpy(dest_pos).to(device=device, dtype=torch.float32)
    attn_control = None
    if sem_mode in ("gridattn", "od_gridattn"):
        if grid_patch is None:
            raise ValueError("gridattn requires grid_patch")
        attn_state = ckpt.get("semantic_attn_state_dict")
        if attn_state is None or not isinstance(attn_state, dict):
            raise ValueError("Checkpoint includes gridattn but missing semantic_attn_state_dict.")
        attn_control = GridCrossAttentionControlMid(
            in_channels=int(grid_patch.shape[1]),
            act_dim=2,
            model_dim=int(hidden_dim) * 4,
            num_heads=int(attn_heads),
            diff_steps=int(diff_steps),
            weight=float(attn_weight),
        ).to(device=device)
        attn_control.load_state_dict(attn_state)
        attn_control.eval()
        patch_t = torch.from_numpy(grid_patch).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        for kk in tqdm(range(k_samples), desc="sample", dynamic_ncols=True):
            torch.manual_seed(int(args.seed) + 1000 + int(kk))
            if posenc is not None:

                def _extra(x_t: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
                    assert patch_t is not None and start_t_t is not None and dest_t_t is not None
                    return posenc(x_t, ts, grid_patch=patch_t, start_pos=start_t_t, dest_pos=dest_t_t)

                if bool(posenc_self_correct):

                    def _extra_x0(x_t: torch.Tensor, ts: torch.Tensor, x0_pred: torch.Tensor) -> torch.Tensor:
                        assert patch_t is not None and start_t_t is not None and dest_t_t is not None
                        return posenc(x0_pred, ts, grid_patch=patch_t, start_pos=start_t_t, dest_pos=dest_t_t)

                    rel_norm_t = model.sample_trajectory(obs_t, cond_t, horizon=int(k_wp), cond_emb_extra_fn_x0=_extra_x0)  # (N,K_wp,2)
                else:
                    rel_norm_t = model.sample_trajectory(obs_t, cond_t, horizon=int(k_wp), cond_emb_extra_fn=_extra)  # (N,K_wp,2)
            elif attn_control is not None:

                def _unet_kwargs(x_t: torch.Tensor, ts: torch.Tensor) -> dict:
                    assert patch_t is not None
                    ctrl_mid, _ = attn_control(x_t, ts, grid_patch=patch_t)
                    return {"control_mid": ctrl_mid}

                rel_norm_t = model.sample_trajectory(obs_t, cond_t, horizon=int(k_wp), unet_kwargs_fn=_unet_kwargs)  # (N,K_wp,2)
            else:
                rel_norm_t = model.sample_trajectory(obs_t, cond_t, horizon=int(k_wp))  # (N,K_wp,2)
            rel = rel_norm_t.detach().cpu().numpy().astype(np.float32, copy=False)
            rel = rel * rel_std[None, None, :] + rel_mean[None, None, :]

            # Postprocess: s in [0,1], o clipped; sort by s to keep waypoint order.
            rel[:, :, 0] = np.clip(rel[:, :, 0], 0.0, 1.0)
            if float(o_clip) > 0.0:
                rel[:, :, 1] = np.clip(rel[:, :, 1], -float(o_clip), float(o_clip))
            order = np.argsort(rel[:, :, 0], axis=1)
            rel = np.take_along_axis(rel, order[:, :, None], axis=1)

            wp_abs = _waypoints_from_rel_so(start=start_pos, dest=dest_pos, rel=rel)  # (N,K_wp,2)
            for i in range(n):
                vertices = np.concatenate([start_pos[i : i + 1], wp_abs[i], dest_pos[i : i + 1]], axis=0)
                curve = _resample_by_arclength(vertices, num=int(f + 1))
                preds_k[i, kk] = curve[1:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "samples.npz"
    out_json = out_dir / "sample_summary.json"
    np.savez_compressed(
        out_npz,
        start_pos=start_pos.astype(np.float32, copy=False),
        dest_pos=dest_pos.astype(np.float32, copy=False),
        traj_idx=traj_idx.astype(np.int64, copy=False),
        start_t=start_t.astype(np.int64, copy=False),
        preds_k=preds_k.astype(np.float32, copy=False),
    )

    result = {
        "inputs": {"checkpoint": str(ckpt_path.resolve()), "case_npz": str(Path(args.case_npz).resolve())},
        "config": {
            "K_waypoints": int(k_wp),
            "K_samples": int(k_samples),
            "od_bin": float(od_bin),
            "o_clip": float(o_clip),
            "cond_dim": int(cond.shape[1]),
            "uses_semantics": bool(sem_mode is not None),
            "seed": int(args.seed),
        },
        "stats": {"N": int(n), "F": int(f)},
        "outputs": {"samples_npz": str(out_npz.resolve())},
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
