from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.datasets_diffusion import DiffusionDataset
from src.models.physics.physics_condition_diffusion import PhysicsConditionDiffusion


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Macro Diffusion: p(z=[wp1,wp2,end_anchor] | obs, trip_od, nav_patch).")
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--nav_file", type=str, required=True, help="nav_field.npz (train-only, used for nav_patch).")
    p.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="train")
    p.add_argument("--splits_dir", type=str, default=None)

    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12, help="micro horizon used to extract oracle waypoints/end_anchor")

    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--nav_patch_channel2", type=str, choices=["count", "speed", "zeros"], default="count")
    p.add_argument("--nav_emb_scale", type=float, default=1.0)
    p.add_argument("--nav_emb_dropout", type=float, default=0.0)
    p.add_argument("--nav_gate", type=str, choices=["none", "obscond"], default="none")
    p.add_argument("--nav_gate_hidden", type=int, default=32)
    p.add_argument("--nav_gate_dropout", type=float, default=0.0)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--diff_steps", type=int, default=20)
    p.add_argument("--pred_type", type=str, choices=["eps", "v"], default="eps")

    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=None, help="limit batches per epoch (for quick iteration)")

    # Training-time monitoring (G1 proxy)
    p.add_argument("--count_thr", type=float, default=1.0, help="Offroad proxy: count < thr treated as non-drivable.")
    p.add_argument("--log_every", type=int, default=100)

    # Differentiable offroad penalty (segment-sampled, approximates G1 collision gate)
    p.add_argument("--offroad_weight", type=float, default=0.0, help="Weight for differentiable offroad penalty (0 disables).")
    p.add_argument("--offroad_samples_per_segment", type=int, default=16, help="Samples per segment for offroad penalty (>=2).")
    return p


def _oob_rates(
    *,
    x0_pred: torch.Tensor,  # (B,2,3) in normalized pos
    pos_min: torch.Tensor,  # (2,)
    pos_range: torch.Tensor,  # (2,)
    nav_count: Optional[np.ndarray],
    count_thr: float,
) -> dict:
    z = x0_pred.permute(0, 2, 1)  # (B,3,2)
    oob_point = (z < -1.0) | (z > 1.0)  # (B,3,2)
    oob_point_rate = float(oob_point.to(torch.float32).mean().item())
    oob_any_rate = float(oob_point.any(dim=-1).any(dim=-1).to(torch.float32).mean().item())

    out = {"oob_point_rate": oob_point_rate, "oob_any_rate": oob_any_rate}
    if nav_count is None:
        return out

    # denormalize to grid, then check nav_count at nearest cell
    pos_grid = (z + 1.0) * 0.5 * pos_range[None, None, :] + pos_min[None, None, :]
    pos_ij = torch.round(pos_grid).to(torch.int64).detach().cpu().numpy()  # (B,3,2)
    H, W = int(nav_count.shape[0]), int(nav_count.shape[1])
    yy = np.clip(pos_ij[:, :, 0], 0, H - 1)
    xx = np.clip(pos_ij[:, :, 1], 0, W - 1)

    bad = (nav_count[yy, xx] < float(count_thr)) | oob_point.any(dim=-1).detach().cpu().numpy()
    bad_point_rate = float(np.mean(bad))
    bad_any_rate = float(np.mean(np.any(bad, axis=1)))
    out.update({"offroad_point_rate": bad_point_rate, "offroad_any_rate": bad_any_rate})
    return out


def _segment_offroad_penalty(
    *,
    start_pos_norm: torch.Tensor,  # (B,2) normalized pos in [-1,1]
    z_norm: torch.Tensor,  # (B,3,2) normalized pos in [-1,1]
    nav_count: torch.Tensor,  # (1,1,H,W) raw count
    pos_min: torch.Tensor,  # (2,)
    pos_range: torch.Tensor,  # (2,)
    count_thr: float,
    samples_per_segment: int,
) -> torch.Tensor:
    """
    Differentiable proxy for G1 collision: sample nav_count along start->wp1->wp2->end polyline
    and penalize count < thr.
    """
    if int(samples_per_segment) < 2:
        raise ValueError("--offroad_samples_per_segment must be >= 2")
    if nav_count.ndim != 4:
        raise ValueError(f"nav_count must be (1,1,H,W), got {tuple(nav_count.shape)}")

    B = int(z_norm.shape[0])
    H = int(nav_count.shape[2])
    W = int(nav_count.shape[3])

    # normalized -> grid coords (y,x)
    start_grid = (start_pos_norm + 1.0) * 0.5 * pos_range[None, :] + pos_min[None, :]
    z_grid = (z_norm + 1.0) * 0.5 * pos_range[None, None, :] + pos_min[None, None, :]

    vertices = torch.cat([start_grid[:, None, :], z_grid], dim=1)  # (B,4,2)
    a = vertices[:, 0:3, :]  # (B,3,2)
    b = vertices[:, 1:4, :]  # (B,3,2)

    t = torch.linspace(0.0, 1.0, steps=int(samples_per_segment), device=z_norm.device, dtype=z_norm.dtype)
    t = t.view(1, 1, -1, 1)  # (1,1,S,1)
    pts = a[:, :, None, :] + t * (b - a)[:, :, None, :]  # (B,3,S,2) [y,x]
    pts = pts.reshape(B, 3 * int(samples_per_segment), 2)  # (B,3S,2)

    # grid_sample expects (x,y) in [-1,1]
    x = pts[:, :, 1]
    y = pts[:, :, 0]
    x_n = (x / max(float(W - 1), 1.0)) * 2.0 - 1.0
    y_n = (y / max(float(H - 1), 1.0)) * 2.0 - 1.0
    grid = torch.stack([x_n, y_n], dim=-1).unsqueeze(2)  # (B,3S,1,2)

    count_in = nav_count.expand(B, -1, -1, -1)
    sampled = F.grid_sample(count_in, grid, mode="bilinear", padding_mode="zeros", align_corners=True)  # (B,1,3S,1)
    sampled = sampled.squeeze(1).squeeze(-1)  # (B,3S)

    viol = F.relu(float(count_thr) - sampled)  # (B,3S)
    return viol.mean(dim=1)  # (B,)


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using seed: {int(args.seed)}")

    traj_ids = None
    if str(args.split) != "all":
        processed_dir = Path(args.data_path).resolve().parents[1]
        splits_dir = Path(args.splits_dir) if args.splits_dir else (processed_dir / "splits")
        split_file = splits_dir / f"{args.split}_ids.npy"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        traj_ids = np.load(split_file).astype(np.int64)
        print(f"Using split={args.split}: {len(traj_ids)} trajectories ({split_file})")

    # Dataset provides:
    # - obs: (H,4)
    # - cond (oracle_wp_end): [hour,day,wp1,wp2,end] in normalized pos
    # - trip_o/trip_d in normalized pos (for condition trip_od)
    # - nav_patch (count channel) at current position
    dataset = DiffusionDataset(
        args.data_path,
        obs_len=int(args.obs_len),
        pred_len=int(args.pred_len),
        nav_field_file=str(args.nav_file),
        nav_patch_size=int(args.patch_size),
        nav_patch_channel2=str(args.nav_patch_channel2),
        traj_ids=traj_ids,
        cond_mode="oracle_wp_end",
        waypoint_mode="rdp_dev",
        num_waypoints=2,
    )

    nav_count = dataset.nav_field.count if dataset.nav_field is not None else None
    if str(args.nav_patch_channel2) == "count" and nav_count is None:
        raise RuntimeError("nav_patch_channel2=count 但 nav_field.npz 缺少 count")

    pos_min = torch.tensor(dataset.normalizer.pos_min, dtype=torch.float32, device=device)
    pos_range = torch.tensor(dataset.normalizer.pos_range, dtype=torch.float32, device=device)
    nav_count_t = None
    if nav_count is not None:
        nav_count_t = torch.from_numpy(np.asarray(nav_count, dtype=np.float32))[None, None, :, :].to(device=device)

    g = torch.Generator()
    g.manual_seed(int(args.seed))
    pin_memory = bool(torch.cuda.is_available())
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        generator=g,
        pin_memory=pin_memory,
        persistent_workers=(int(args.num_workers) > 0),
    )

    # Macro diffusion over z as a 3-step "trajectory" in pos space: (wp1, wp2, end_anchor)
    model = PhysicsConditionDiffusion(
        obs_dim=4,
        act_dim=2,
        cond_dim=6,  # trip_od only
        nav_patch_size=int(args.patch_size),
        nav_emb_scale=float(args.nav_emb_scale),
        nav_emb_dropout=float(args.nav_emb_dropout),
        nav_gate=str(args.nav_gate),
        nav_gate_hidden=int(args.nav_gate_hidden),
        nav_gate_dropout=float(args.nav_gate_dropout),
        obs_len=int(args.obs_len),
        pred_len=3,
        hidden_dim=int(args.hidden_dim),
        diffusion_steps=int(args.diff_steps),
        prediction_type=str(args.pred_type),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))

    save_dir = Path(f"data/experiments/{args.exp_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model_type": "macro_physics_z_diffusion",
        "data_path": str(args.data_path),
        "nav_file": str(args.nav_file),
        "split": str(args.split),
        "splits_dir": (str(args.splits_dir) if args.splits_dir else None),
        "obs_len": int(args.obs_len),
        "pred_len_micro": int(args.pred_len),
        "target_horizon": 3,
        "target_semantics": "z=[wp1,wp2,end_anchor] in normalized pos",
        "cond_semantics": "obs + trip_od + nav_patch(count,current_only)",
        "patch_size": int(args.patch_size),
        "nav_patch_channel2": str(args.nav_patch_channel2),
        "nav_emb_scale": float(args.nav_emb_scale),
        "nav_emb_dropout": float(args.nav_emb_dropout),
        "nav_gate": str(args.nav_gate),
        "nav_gate_hidden": int(args.nav_gate_hidden),
        "nav_gate_dropout": float(args.nav_gate_dropout),
        "hidden_dim": int(args.hidden_dim),
        "diff_steps": int(args.diff_steps),
        "pred_type": str(args.pred_type),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "max_batches": (int(args.max_batches) if args.max_batches is not None else None),
        "count_thr": float(args.count_thr),
        "offroad_weight": float(args.offroad_weight),
        "offroad_samples_per_segment": int(args.offroad_samples_per_segment),
        "offroad_weighting": "alpha2",
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    model.train()
    for epoch in range(int(args.epochs)):
        start = time.time()
        total_diff = 0.0
        total_all = 0.0
        nb = 0

        for bidx, batch in enumerate(loader):
            obs = batch["obs"].to(device)
            cond_oracle = batch["cond"].to(device)  # (B,8)
            nav_patch = batch["nav_patch"].to(device)
            trip_o = batch["trip_o"].to(device)
            trip_d = batch["trip_d"].to(device)

            cond_trip_od = torch.cat([cond_oracle[:, :2], trip_o, trip_d], dim=-1)  # (B,6)
            z = cond_oracle[:, 2:].view(cond_oracle.shape[0], 3, 2)  # (B,3,2)

            optimizer.zero_grad(set_to_none=True)
            diff_loss, x0_pred, _t = model.compute_loss(
                obs=obs,
                cond=cond_trip_od,
                target=z,
                nav_patch=nav_patch,
                return_x0_pred=True,
                return_timesteps=True,
            )
            loss = diff_loss
            offroad_pen = None
            gt_offroad_pen = None
            if float(args.offroad_weight) > 0.0 and nav_count_t is not None:
                z_pred = x0_pred.permute(0, 2, 1)  # (B,3,2)
                z_pred = torch.clamp(z_pred, -1.0, 1.0)
                start_pos_norm = obs[:, -1, :2]  # (B,2)
                per_sample = _segment_offroad_penalty(
                    start_pos_norm=start_pos_norm,
                    z_norm=z_pred,
                    nav_count=nav_count_t,
                    pos_min=pos_min,
                    pos_range=pos_range,
                    count_thr=float(args.count_thr),
                    samples_per_segment=int(args.offroad_samples_per_segment),
                )
                # Sanity: penalty floor on GT z (oracle). If this is not small, your drivable proxy/threshold
                # is inconsistent with GT or the linear polyline is too aggressive.
                with torch.no_grad():
                    gt_per_sample = _segment_offroad_penalty(
                        start_pos_norm=start_pos_norm,
                        z_norm=z,
                        nav_count=nav_count_t,
                        pos_min=pos_min,
                        pos_range=pos_range,
                        count_thr=float(args.count_thr),
                        samples_per_segment=int(args.offroad_samples_per_segment),
                    )
                # Weight by denoising SNR proxy: alpha^2 = alphas_cumprod[t].
                # High-noise timesteps produce very noisy x0_pred; weighting avoids destabilizing training.
                w = model.diffusion.scheduler.alphas_cumprod[_t].to(dtype=per_sample.dtype)  # (B,)
                w = torch.clamp(w, 0.0, 1.0)
                offroad_pen = (per_sample * w).sum() / (w.sum() + 1e-6)
                gt_offroad_pen = (gt_per_sample * w).sum() / (w.sum() + 1e-6)
                loss = loss + float(args.offroad_weight) * offroad_pen

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_diff += float(diff_loss.item())
            total_all += float(loss.item())
            nb += 1

            if int(args.log_every) > 0 and bidx % int(args.log_every) == 0:
                with torch.no_grad():
                    rates = _oob_rates(
                        x0_pred=x0_pred,
                        pos_min=pos_min,
                        pos_range=pos_range,
                        nav_count=nav_count,
                        count_thr=float(args.count_thr),
                    )
                msg = ", ".join(f"{k}={v:.4f}" for k, v in rates.items())
                if offroad_pen is not None:
                    msg = msg + f", offroad_pen_w={float(offroad_pen.item()):.4f}"
                if gt_offroad_pen is not None:
                    msg = msg + f", gt_offroad_pen_w={float(gt_offroad_pen.item()):.4f}"
                msg = msg + f", loss={float(loss.item()):.4f}"
                print(f"Epoch {epoch} | Batch {bidx} | DiffLoss {diff_loss.item():.4f} | {msg}")

            if args.max_batches is not None and int(args.max_batches) > 0 and nb >= int(args.max_batches):
                break

        avg_diff = total_diff / float(max(nb, 1))
        avg_all = total_all / float(max(nb, 1))
        dur = time.time() - start
        print(f"Epoch {epoch} Done. Avg DiffLoss: {avg_diff:.4f}. Avg Loss: {avg_all:.4f}. Time: {dur:.1f}s")

        ckpt = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": float(avg_all),
            "config": run_config,
        }
        torch.save(ckpt, save_dir / "last.pt")


if __name__ == "__main__":
    main()
