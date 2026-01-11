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
    semantic_grid_patch_tensor,
    semantic_od_features,
)
from src.features.temporal import encode_route_temporal_2d
from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.models.semantic.grid_cross_attention_control import GridCrossAttentionControlMid
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
    p = argparse.ArgumentParser(description="Visualize grid cross-attention weights for a gridattn decision checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--case_npz", type=str, required=True)
    p.add_argument("--semantic_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--index", type=int, default=0, help="Which window index in case_npz to visualize.")
    p.add_argument("--seed", type=int, default=0)
    return p


def _save_heatmap_png(heat: np.ndarray, *, out_png: Path, out_pdf: Optional[Path]) -> None:
    import matplotlib.pyplot as plt

    a = np.asarray(heat, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    im = ax.imshow(a, cmap="magma", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_png, dpi=200)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    sem_meta = cfg.get("semantic") if isinstance(cfg, dict) else None
    if not isinstance(sem_meta, dict) or sem_meta.get("mode") is None:
        raise ValueError("Checkpoint missing semantic.mode (expected gridattn/od_gridattn).")
    sem_mode = str(sem_meta.get("mode"))
    if sem_mode not in ("gridattn", "od_gridattn"):
        raise ValueError(f"Checkpoint semantic.mode must be gridattn/od_gridattn, got {sem_mode}")

    od_bin = float(cfg.get("od_bin", 128.0))
    k_wp = int(cfg.get("K_waypoints", 0))
    if k_wp <= 0:
        raise ValueError("Checkpoint missing K_waypoints")

    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    diff_steps = int(model_cfg.get("diff_steps", 50))
    pred_type = str(model_cfg.get("pred_type", "eps"))
    cond_dim = int(model_cfg.get("cond_dim", 6))

    grid_patch_size = int(sem_meta.get("grid_patch_size", 16))
    grid_extent = float(sem_meta.get("grid_extent", 128.0))
    grid_channels = str(sem_meta.get("grid_channels", "poi,entropy"))
    attn_heads = int(sem_meta.get("attn_heads", 4))
    attn_weight = float(sem_meta.get("attn_weight", 1.0))
    sem_use_bins = bool(sem_meta.get("use_bins", False))
    temporal_meta = cfg.get("temporal", {}) if isinstance(cfg, dict) else {}
    temporal_mode = "zeros"
    temporal_tz = -5.0
    if isinstance(temporal_meta, dict):
        temporal_mode = str(temporal_meta.get("effective") or temporal_meta.get("mode") or "zeros")
        temporal_tz = float(temporal_meta.get("tz_offset_hours", -5.0))

    sem_cfg_raw = cfg.get("semantic_od_norm") if isinstance(cfg, dict) else None
    sem_cfg = SemanticODNorm.from_json(sem_cfg_raw) if isinstance(sem_cfg_raw, dict) else None
    grid_norm_raw = cfg.get("semantic_grid_norm") if isinstance(cfg, dict) else None
    if grid_norm_raw is None or not isinstance(grid_norm_raw, dict):
        raise ValueError("Checkpoint missing semantic_grid_norm.")
    grid_norm = SemanticGridNorm.from_json(grid_norm_raw)

    attn_state = ckpt.get("semantic_attn_state_dict")
    if attn_state is None or not isinstance(attn_state, dict):
        raise ValueError("Checkpoint missing semantic_attn_state_dict.")

    norm = _load_norm_from_ckpt(cfg if isinstance(cfg, dict) else {})

    case = load_route_windows_npz(str(args.case_npz), max_n=None, seed=int(args.seed))
    start_pos = np.asarray(case["start_pos"], dtype=np.float32)
    dest_pos = np.asarray(case["dest_pos"], dtype=np.float32)
    n = int(start_pos.shape[0])
    idx = int(args.index)
    if idx < 0 or idx >= n:
        raise ValueError(f"--index out of range: {idx} (N={n})")

    start_ctr = _od_bin_center(start_pos, bin_size=float(od_bin))
    dest_ctr = _od_bin_center(dest_pos, bin_size=float(od_bin))
    start_ctr_norm = normalize_pos(start_ctr, norm)
    dest_ctr_norm = normalize_pos(dest_ctr, norm)

    obs = np.concatenate([start_ctr_norm, np.zeros((n, 2), dtype=np.float32)], axis=1)[:, None, :]  # (N,1,4)
    start_t_arr = np.asarray(case["start_t"], dtype=np.int64)
    temporal, _temporal_eff = encode_route_temporal_2d(start_t_arr, tz_offset_hours=float(temporal_tz), mode=str(temporal_mode))
    base_cond = np.concatenate([temporal, start_ctr_norm, dest_ctr_norm], axis=1).astype(np.float32, copy=False)  # (N,6)
    cond_parts = [base_cond]

    sem_o = start_ctr if sem_use_bins else start_pos
    sem_d = dest_ctr if sem_use_bins else dest_pos
    if sem_mode == "od_gridattn":
        if sem_cfg is None:
            raise ValueError("od_gridattn requires semantic_od_norm in checkpoint.")
        poi_total, landuse_entropy = load_poi_total_and_landuse_entropy(args.semantic_dir)
        sem_od, sem_keys = semantic_od_features(start_ctr=sem_o, dest_ctr=sem_d, poi_total=poi_total, landuse_entropy=landuse_entropy, log_poi=True)
        if tuple(str(k) for k in sem_keys) != sem_cfg.keys:
            raise ValueError("semantic_od keys mismatch.")
        cond_parts.append(normalize_semantic(sem_od, sem_cfg))

    # Grid patch for the selected idx only.
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
    grid_patch_raw, grid_keys = semantic_grid_patch_tensor(
        start_ctr=sem_o[idx : idx + 1],
        dest_ctr=sem_d[idx : idx + 1],
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
        raise ValueError("semantic_grid keys mismatch.")
    grid_patch = normalize_grid_patch(grid_patch_raw, grid_norm)

    cond = np.concatenate(cond_parts, axis=1).astype(np.float32, copy=False)
    if int(cond.shape[1]) != int(cond_dim):
        raise ValueError(f"cond_dim mismatch: ckpt={cond_dim} vs built={cond.shape[1]}")

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

    attn = GridCrossAttentionControlMid(
        in_channels=int(grid_patch.shape[1]),
        act_dim=2,
        model_dim=int(hidden_dim) * 4,
        num_heads=int(attn_heads),
        diff_steps=int(diff_steps),
        weight=float(attn_weight),
    ).to(device=device)
    attn.load_state_dict(attn_state)
    attn.record_attn = True
    attn.eval()

    obs_t = torch.from_numpy(obs[idx : idx + 1]).to(device=device, dtype=torch.float32)
    cond_t = torch.from_numpy(cond[idx : idx + 1]).to(device=device, dtype=torch.float32)
    patch_t = torch.from_numpy(grid_patch).to(device=device, dtype=torch.float32)

    with torch.no_grad():

        def _unet_kwargs(x_t: torch.Tensor, ts: torch.Tensor) -> dict:
            ctrl_mid, _ = attn(x_t, ts, grid_patch=patch_t)
            return {"control_mid": ctrl_mid}

        _ = model.sample_trajectory(obs_t, cond_t, horizon=int(k_wp), unet_kwargs_fn=_unet_kwargs)

    if attn.last_attn is None:
        raise RuntimeError("No attention weights recorded.")
    w = attn.last_attn.detach().cpu().numpy()
    # (B,H,L,N) -> (S,S) heatmap by averaging over heads and query positions.
    w = w[0]  # (H,L,N)
    heat = np.mean(w, axis=(0, 1))  # (N,) non-negative, should sum to ~1
    heat = np.asarray(heat, dtype=np.float64).reshape(-1)
    heat_sum = float(np.sum(heat))
    if not np.isfinite(heat_sum) or heat_sum <= 0:
        raise RuntimeError(f"Bad attention heatmap sum: {heat_sum}")
    heat = (heat / heat_sum).astype(np.float64, copy=False)
    s = int(grid_patch_size)
    heat2 = heat.reshape(s, s).astype(np.float32, copy=False)

    # Uniformity diagnostics.
    n_tok = int(heat.shape[0])
    uni = np.full((n_tok,), 1.0 / float(n_tok), dtype=np.float64)
    eps = 1e-12
    ent = float(-np.sum(heat * np.log(heat + eps)))
    ent_norm = float(ent / max(np.log(float(n_tok)), eps))
    l1_uni = float(np.sum(np.abs(heat - uni)))
    max_p = float(np.max(heat))
    topk_stats = {}
    for k_top in (1, 4, 16, 32):
        k_top = int(min(k_top, n_tok))
        topk_stats[f"top{k_top}_mass"] = float(np.sum(np.sort(heat)[-k_top:]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_dir / "attn_heatmap.npy"
    out_png = out_dir / "attn_heatmap.png"
    out_pdf = out_dir / "attn_heatmap.pdf"
    np.save(out_npy, heat2)
    _save_heatmap_png(heat2, out_png=out_png, out_pdf=out_pdf)

    report = {
        "inputs": {
            "checkpoint": str(ckpt_path.resolve()),
            "case_npz": str(Path(args.case_npz).resolve()),
            "semantic_dir": str(Path(args.semantic_dir).resolve()),
        },
        "config": {
            "semantic_mode": str(sem_mode),
            "index": int(idx),
            "grid_patch_size": int(grid_patch_size),
            "grid_extent": float(grid_extent),
            "grid_channels": str(grid_channels),
            "attn_heads": int(attn_heads),
            "attn_weight": float(attn_weight),
            "seed": int(args.seed),
        },
        "outputs": {"attn_heatmap_npy": str(out_npy.resolve()), "attn_heatmap_png": str(out_png.resolve()), "attn_heatmap_pdf": str(out_pdf.resolve())},
        "stats": {
            "heat_min": float(np.min(heat2)),
            "heat_max": float(np.max(heat2)),
            "heat_mean": float(np.mean(heat2)),
            "heat_sum": float(np.sum(heat)),
            "entropy_nats": float(ent),
            "entropy_norm_0to1": float(ent_norm),
            "l1_to_uniform": float(l1_uni),
            "max_prob": float(max_p),
            **topk_stats,
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
