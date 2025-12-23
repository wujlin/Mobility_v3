import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.datasets_diffusion import DiffusionDataset
from src.models.flow.rectified_flow_model import RectifiedFlowTrajectoryModel
from src.models.physics.physics_condition_flow import PhysicsConditionFlow
from src.models.seq.seq_baseline import SeqBaseline
from typing import Optional


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_checkpoint(resume_from: str, device: torch.device) -> dict:
    ckpt = torch.load(str(resume_from), map_location=device)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")
    return ckpt


def _load_baseline_prior(prior_checkpoint: str, device: torch.device) -> SeqBaseline:
    """
    Load a frozen SeqBaseline as a deterministic prior for residual flow.

    Supported checkpoint formats:
    - {"model_state_dict": ..., "config": {...}} (preferred)
    - raw state_dict saved via torch.save(model.state_dict())
    """
    ckpt = torch.load(str(prior_checkpoint), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        cfg = ckpt.get("config", {})
        hidden_dim = cfg.get("hidden_dim") if isinstance(cfg, dict) else None
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        hidden_dim = None
    else:
        raise TypeError(f"Unsupported prior checkpoint format: {type(ckpt)}")

    if hidden_dim is None:
        w = state_dict.get("head.weight")
        if hasattr(w, "shape") and len(w.shape) == 2:
            hidden_dim = int(w.shape[1])

    if hidden_dim is None:
        raise ValueError(f"Cannot infer prior hidden_dim from checkpoint: {prior_checkpoint}")

    model = SeqBaseline(obs_dim=4, act_dim=2, cond_dim=6, hidden_dim=int(hidden_dim)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _maybe_estimate_noise_sigma(
    dataset: DiffusionDataset,
    device: torch.device,
    *,
    prior_model: Optional[SeqBaseline],
    obs_len: int,
    pred_len: int,
    max_batches: int,
    batch_size: int,
    num_workers: int,
) -> float:
    """
    Estimate a reasonable RF noise sigma from the residual target distribution.
    KISS: compute std over a few mini-batches of normalized residual velocities.
    """
    g = torch.Generator()
    g.manual_seed(0)
    pin_memory = bool(torch.cuda.is_available())
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        generator=g,
        pin_memory=pin_memory,
        persistent_workers=(int(num_workers) > 0),
    )

    std_sum = 0.0
    std_count = 0
    for i, batch in enumerate(loader):
        if i >= int(max_batches):
            break
        obs = batch["obs"].to(device)
        cond = batch["cond"].to(device)
        gt_vel_norm = batch["action"].to(device)

        if prior_model is not None:
            with torch.no_grad():
                prior_vel_norm = prior_model.sample_trajectory(obs, cond, int(pred_len))
            target = gt_vel_norm - prior_vel_norm
        else:
            target = gt_vel_norm

        std = target.reshape(target.shape[0], -1).std(dim=1)  # per-sample std
        std_sum += float(std.mean().item())
        std_count += 1

    return float(std_sum / max(std_count, 1))


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    _set_seed(int(args.seed))
    print(f"Using seed: {int(args.seed)}")

    # 1) Data
    print("Loading datasets...")
    traj_ids = None
    if str(args.split) != "all":
        processed_dir = Path(args.data_path).resolve().parents[1]
        splits_dir = Path(args.splits_dir) if args.splits_dir else (processed_dir / "splits")
        split_file = splits_dir / f"{args.split}_ids.npy"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        traj_ids = np.load(split_file).astype(np.int64)
        print(f"Using split={args.split}: {len(traj_ids)} trajectories ({split_file})")

    nav_file = None
    if str(args.model_type) == "physics_flow":
        if not args.nav_file:
            raise ValueError("--nav_file is required for --model_type physics_flow")
        nav_file = str(args.nav_file)

    dataset = DiffusionDataset(
        args.data_path,
        obs_len=int(args.obs_len),
        pred_len=int(args.pred_len),
        nav_field_file=nav_file,
        nav_patch_size=int(args.patch_size),
        nav_patch_channel2=str(args.nav_patch_channel2),
        traj_ids=traj_ids,
    )

    g = torch.Generator()
    g.manual_seed(int(args.seed))
    pin_memory = bool(torch.cuda.is_available())
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        generator=g,
        pin_memory=pin_memory,
        persistent_workers=(int(args.num_workers) > 0),
    )

    # torch-friendly denorm constants for displacement-aware weighting (optional)
    vel_mean = torch.tensor(dataset.normalizer.vel_mean, dtype=torch.float32, device=device)
    vel_std = torch.tensor(dataset.normalizer.vel_std, dtype=torch.float32, device=device)

    # 2) Model
    if str(args.model_type) == "flow":
        model = RectifiedFlowTrajectoryModel(
            obs_dim=4,
            act_dim=2,
            cond_dim=6,
            obs_len=int(args.obs_len),
            pred_len=int(args.pred_len),
            hidden_dim=int(args.hidden_dim),
            time_scale=float(args.time_scale),
            noise_sigma=float(args.rf_noise_sigma),
            solver_steps=int(args.solver_steps),
        )
    elif str(args.model_type) == "physics_flow":
        model = PhysicsConditionFlow(
            obs_dim=4,
            act_dim=2,
            cond_dim=6,
            nav_patch_size=int(args.patch_size),
            nav_emb_dim=int(args.nav_emb_dim),
            nav_emb_scale=float(args.nav_emb_scale),
            nav_emb_dropout=float(args.nav_emb_dropout),
            nav_gate=str(args.nav_gate),
            nav_gate_hidden=int(args.nav_gate_hidden),
            nav_gate_dropout=float(args.nav_gate_dropout),
            obs_len=int(args.obs_len),
            pred_len=int(args.pred_len),
            hidden_dim=int(args.hidden_dim),
            time_scale=float(args.time_scale),
            noise_sigma=float(args.rf_noise_sigma),
            solver_steps=int(args.solver_steps),
        )
    else:
        raise ValueError(f"Unknown --model_type: {args.model_type} (expected: flow|physics_flow)")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))

    save_dir = Path(f"data/experiments/{args.exp_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "model_type": str(args.model_type),
        "data_path": str(args.data_path),
        "split": str(args.split),
        "splits_dir": (str(args.splits_dir) if args.splits_dir else None),
        "nav_file": (str(args.nav_file) if args.nav_file else None),
        "patch_size": int(args.patch_size),
        "nav_patch_channel2": str(args.nav_patch_channel2),
        "nav_emb_dim": int(args.nav_emb_dim),
        "nav_emb_scale": float(args.nav_emb_scale),
        "nav_emb_dropout": float(args.nav_emb_dropout),
        "nav_gate": str(args.nav_gate),
        "nav_gate_hidden": int(args.nav_gate_hidden),
        "nav_gate_dropout": float(args.nav_gate_dropout),
        "obs_len": int(args.obs_len),
        "pred_len": int(args.pred_len),
        "hidden_dim": int(args.hidden_dim),
        "time_scale": float(args.time_scale),
        "rf_noise_sigma": float(args.rf_noise_sigma),
        "solver_steps": int(args.solver_steps),
        "prior_checkpoint": (str(args.prior_checkpoint) if args.prior_checkpoint else None),
        "residual_mode": bool(args.prior_checkpoint),
        "disp_weight": str(args.disp_weight),
        "disp_alpha": float(args.disp_alpha),
        "disp_clip_min": float(args.disp_clip_min),
        "disp_clip_max": float(args.disp_clip_max),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "max_batches": (int(args.max_batches) if args.max_batches is not None else None),
    }

    # 2.1) Resume (optional)
    start_epoch = 0
    resume_from = str(args.resume_from) if args.resume_from else None
    if resume_from is not None:
        ckpt = _load_checkpoint(resume_from, device=device)
        if "model_state_dict" not in ckpt:
            raise KeyError(f"Checkpoint missing model_state_dict: {resume_from}")
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and not bool(args.no_resume_optim):
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"[WARN] optimizer_state_dict 加载失败（将继续但不恢复优化器状态）：{e}")
        elif bool(args.no_resume_optim):
            print("[OK] 已跳过 optimizer_state_dict（--no_resume_optim）")
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        ckpt_cfg = ckpt.get("config", {})
        if isinstance(ckpt_cfg, dict):
            if args.prior_checkpoint is None and ckpt_cfg.get("prior_checkpoint"):
                args.prior_checkpoint = str(ckpt_cfg["prior_checkpoint"])
                run_config["prior_checkpoint"] = str(args.prior_checkpoint)
                run_config["residual_mode"] = True
                print(f"[OK] resume: 使用 ckpt 中的 prior_checkpoint={args.prior_checkpoint}")
            for k in ("model_type", "hidden_dim", "obs_len", "pred_len", "solver_steps", "time_scale", "rf_noise_sigma"):
                old = ckpt_cfg.get(k)
                new = run_config.get(k)
                if old is not None and new is not None and str(old) != str(new):
                    print(f"[WARN] resume 配置不一致：{k}: ckpt={old} vs args={new}（可能导致效果异常）")
        print(f"[OK] Resumed from {resume_from} (start_epoch={start_epoch})")

    # 2.2) Prior (optional, residual flow)
    prior_model = None
    if args.prior_checkpoint:
        prior_model = _load_baseline_prior(str(args.prior_checkpoint), device=device)
        print(f"[OK] Residual mode enabled: prior={args.prior_checkpoint}")

    # 2.3) RF noise sigma auto-estimation (optional)
    if bool(args.rf_noise_sigma_auto):
        sigma_est = _maybe_estimate_noise_sigma(
            dataset,
            device,
            prior_model=prior_model,
            obs_len=int(args.obs_len),
            pred_len=int(args.pred_len),
            max_batches=int(args.rf_noise_sigma_auto_batches),
            batch_size=int(args.batch_size),
            num_workers=max(0, min(int(args.num_workers), 4)),
        )
        # Avoid degenerate tiny sigma; clamp to keep training stable.
        sigma_est = float(np.clip(sigma_est, 0.05, 2.0))
        if hasattr(model, "set_noise_sigma"):
            model.set_noise_sigma(float(sigma_est))
        run_config["rf_noise_sigma"] = float(sigma_est)
        print(f"[OK] rf_noise_sigma(auto)={sigma_est:.4f}")

    model.train()
    if start_epoch >= int(args.epochs):
        print(f"[DONE] start_epoch({start_epoch}) >= --epochs({int(args.epochs)}), nothing to train.")
        return

    print(f"Start training {args.exp_name}...")
    for epoch in range(int(start_epoch), int(args.epochs)):
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches is not None and int(args.max_batches) > 0 and batch_idx >= int(args.max_batches):
                break

            obs = batch["obs"].to(device)
            cond = batch["cond"].to(device)
            gt_vel_norm = batch["action"].to(device)  # (B, F, 2)
            nav_patch = batch.get("nav_patch")
            if nav_patch is not None:
                nav_patch = nav_patch.to(device)

            # Residual target (normalized)
            target = gt_vel_norm
            if prior_model is not None:
                with torch.no_grad():
                    prior_vel_norm = prior_model.sample_trajectory(obs, cond, int(args.pred_len))
                target = gt_vel_norm - prior_vel_norm

            # Displacement-aware weighting (recommended for mean-field / low-displacement dominance)
            sample_weight = None
            if str(args.disp_weight) != "none":
                gt_vel_denorm = gt_vel_norm * vel_std + vel_mean
                gt_disp = gt_vel_denorm.sum(dim=1)  # (B, 2)
                disp_norm = torch.linalg.norm(gt_disp, dim=-1)  # (B,)
                if str(args.disp_weight) == "tanh":
                    sample_weight = torch.tanh(float(args.disp_alpha) * disp_norm)
                elif str(args.disp_weight) == "clip":
                    disp_ref = torch.clamp_min(disp_norm.mean(), 1e-6)
                    sample_weight = disp_norm / disp_ref
                    sample_weight = torch.clamp(
                        sample_weight,
                        min=float(args.disp_clip_min),
                        max=float(args.disp_clip_max),
                    )
                else:
                    raise ValueError(f"Unknown --disp_weight: {args.disp_weight}")
                sample_weight = torch.clamp_min(sample_weight, 1e-3)

            optimizer.zero_grad()
            if str(args.model_type) == "physics_flow":
                loss = model(obs, cond, target=target, nav_patch=nav_patch, sample_weight=sample_weight)
            else:
                loss = model(obs, cond, target=target, sample_weight=sample_weight)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")

        denom = max(num_batches, 1)
        avg_loss = total_loss / float(denom)
        duration = time.time() - start_time
        print(f"Epoch {epoch} Done. Loss: {avg_loss:.4f}. Time: {duration:.1f}s")

        torch.save(
            {
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": float(avg_loss),
                "config": run_config,
            },
            save_dir / "last.pt",
        )
        if int(epoch) % 10 == 0:
            torch.save(model.state_dict(), save_dir / f"epoch_{epoch}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["flow", "physics_flow"], default="physics_flow")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--splits_dir", type=str, default=None, help="override splits dir (default: <processed_dir>/splits)")

    parser.add_argument("--nav_file", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--nav_patch_channel2", type=str, choices=["speed", "count", "zeros"], default="speed")
    parser.add_argument("--nav_emb_dim", type=int, default=32)
    parser.add_argument("--nav_emb_scale", type=float, default=1.0)
    parser.add_argument("--nav_emb_dropout", type=float, default=0.0)
    parser.add_argument("--nav_gate", type=str, choices=["none", "obscond"], default="none")
    parser.add_argument("--nav_gate_hidden", type=int, default=32)
    parser.add_argument("--nav_gate_dropout", type=float, default=0.0)

    parser.add_argument("--prior_checkpoint", type=str, default=None, help="enable residual mode with a frozen SeqBaseline prior")

    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=128)

    parser.add_argument("--time_scale", type=float, default=1000.0)
    parser.add_argument("--rf_noise_sigma", type=float, default=1.0)
    parser.add_argument("--rf_noise_sigma_auto", action="store_true", help="estimate noise sigma from residual targets (few batches)")
    parser.add_argument("--rf_noise_sigma_auto_batches", type=int, default=10)
    parser.add_argument("--solver_steps", type=int, default=20, help="ODE solver steps for sampling (Euler), stored in checkpoint")

    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=None, help="limit batches per epoch (for quick iteration)")

    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--no_resume_optim", action="store_true")

    parser.add_argument(
        "--disp_weight",
        type=str,
        choices=["none", "tanh", "clip"],
        default="none",
        help="Displacement-aware weighting (mitigate low-displacement dominance).",
    )
    parser.add_argument("--disp_alpha", type=float, default=0.1)
    parser.add_argument("--disp_clip_min", type=float, default=0.5)
    parser.add_argument("--disp_clip_max", type=float, default=5.0)

    train(parser.parse_args())
