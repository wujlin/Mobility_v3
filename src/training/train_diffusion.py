import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import time
import os
import numpy as np
import random

from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.models.physics.physics_condition_diffusion import PhysicsConditionDiffusion
from src.data.datasets_diffusion import DiffusionDataset
from src.models.seq.seq_baseline import SeqBaseline

def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

def _load_baseline_prior(prior_checkpoint: str, device: torch.device) -> SeqBaseline:
    """
    Load a frozen SeqBaseline as a deterministic prior for residual diffusion.

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

def _rog_per_traj(pos: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Radius of gyration per trajectory.
    Args:
        pos: (B, T, 2)
    Returns:
        rog: (B,)
    """
    mean_pos = pos.mean(dim=1, keepdim=True)
    diff = pos - mean_pos
    sq = (diff ** 2).sum(dim=-1).mean(dim=1)
    return torch.sqrt(sq + float(eps))

def _macro_indices(pred_len: int, points: list[float]) -> list[int]:
    """
    Convert user-provided points to future indices in [0, pred_len-1].

    - If 0 < p <= 1: treat as a fraction of the horizon.
    - If p > 1: treat as an absolute index.
    """
    if pred_len <= 0:
        return []
    out: list[int] = []
    for p in points:
        if p is None:
            continue
        if 0.0 < float(p) <= 1.0:
            idx = int(round(float(p) * float(pred_len - 1)))
        else:
            idx = int(round(float(p)))
        idx = max(0, min(pred_len - 1, idx))
        out.append(idx)
    out = sorted(set(out))
    return out

def _macro_t_weight(
    timesteps: torch.Tensor,
    diff_steps: int,
    mode: str,
    gamma: float,
) -> torch.Tensor:
    """
    Per-sample macro-loss weight as a function of diffusion timestep t.
    Keep timestep sampling uniform; only reweight the macro term.
    """
    if mode == "none":
        return torch.ones_like(timesteps, dtype=torch.float32)
    if mode == "exp":
        denom = max(int(diff_steps) - 1, 1)
        t01 = timesteps.to(torch.float32) / float(denom)
        return torch.exp(-float(gamma) * t01)
    raise ValueError(f"Unknown --macro_t_weight: {mode}")

def _load_checkpoint(resume_from: str, device: torch.device) -> dict:
    ckpt = torch.load(resume_from, map_location=device)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")
    return ckpt

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    _set_seed(int(args.seed))
    print(f"Using seed: {int(args.seed)}")
    
    # 1. Data
    print("Loading datasets...")
    # Conditionally load nav field
    nav_file = args.nav_file if args.model_type == 'physics' else None

    traj_ids = None
    if args.split != 'all':
        processed_dir = Path(args.data_path).resolve().parents[1]
        splits_dir = Path(args.splits_dir) if args.splits_dir else (processed_dir / "splits")
        split_file = splits_dir / f"{args.split}_ids.npy"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        traj_ids = np.load(split_file).astype(np.int64)
        print(f"Using split={args.split}: {len(traj_ids)} trajectories ({split_file})")
    
    dataset = DiffusionDataset(
        args.data_path, 
        obs_len=args.obs_len, 
        pred_len=args.pred_len,
        nav_field_file=nav_file,
        nav_patch_size=args.patch_size,
        nav_patch_channel2=args.nav_patch_channel2,
        traj_ids=traj_ids,
    )
    # torch-friendly normalization constants (avoid numpy<->torch ops in training loop)
    pos_min = torch.tensor(dataset.normalizer.pos_min, dtype=torch.float32, device=device)
    pos_range = torch.tensor(dataset.normalizer.pos_range, dtype=torch.float32, device=device)
    vel_mean = torch.tensor(dataset.normalizer.vel_mean, dtype=torch.float32, device=device)
    vel_std = torch.tensor(dataset.normalizer.vel_std, dtype=torch.float32, device=device)

    g = torch.Generator()
    g.manual_seed(int(args.seed))
    pin_memory = bool(torch.cuda.is_available())
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.num_workers),
        generator=g,
        pin_memory=pin_memory,
        persistent_workers=(int(args.num_workers) > 0),
    )
    
    # 2. Model
    if args.model_type == 'physics':
        print("Initializing PhysicsConditionDiffusion...")
        model = PhysicsConditionDiffusion(
            obs_dim=4, act_dim=2, cond_dim=6,
            nav_patch_size=args.patch_size,
            obs_len=args.obs_len, pred_len=args.pred_len,
            hidden_dim=args.hidden_dim,
            diffusion_steps=args.diff_steps
        )
    else:
        print("Initializing Standard DiffusionTrajectoryModel...")
        model = DiffusionTrajectoryModel(
            obs_dim=4, act_dim=2, cond_dim=6,
            obs_len=args.obs_len, pred_len=args.pred_len,
            hidden_dim=args.hidden_dim,
            diffusion_steps=args.diff_steps
        )
    
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. Setup Logging
    save_dir = Path(f"data/experiments/{args.exp_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Optional resume (for fast iteration without wasting epochs)
    resume_from = args.resume_from
    if args.resume and resume_from is None:
        candidate = save_dir / "last.pt"
        if candidate.exists():
            resume_from = str(candidate)

    run_config = {
        "model_type": args.model_type,
        "data_path": args.data_path,
        "split": args.split,
        "splits_dir": args.splits_dir,
        "nav_file": args.nav_file,
        "patch_size": args.patch_size,
        "nav_patch_channel2": str(args.nav_patch_channel2),
        "obs_len": args.obs_len,
        "pred_len": args.pred_len,
        "hidden_dim": args.hidden_dim,
        "diff_steps": args.diff_steps,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "max_batches": (int(args.max_batches) if args.max_batches is not None else None),
        "lambda_rog": float(args.lambda_rog),
        "macro_metric": str(args.macro_metric),
        "macro_t_threshold": (int(args.macro_t_threshold) if args.macro_t_threshold is not None else None),
        "macro_rel_eps": float(args.macro_rel_eps),
        "rog_loss": str(args.rog_loss),
        "rog_warmup_epochs": int(args.rog_warmup_epochs),
        "rog_eps": float(args.rog_eps),
        "macro_disp_weight": str(args.macro_disp_weight),
        "macro_disp_alpha": float(args.macro_disp_alpha),
        "macro_disp_clip_min": float(args.macro_disp_clip_min),
        "macro_disp_clip_max": float(args.macro_disp_clip_max),
        "macro_t_weight": str(args.macro_t_weight),
        "macro_t_gamma": float(args.macro_t_gamma),
        "macro_points": [float(x) for x in (args.macro_points or [])],
        "prior_checkpoint": (str(args.prior_checkpoint) if args.prior_checkpoint else None),
        "residual_mode": bool(args.prior_checkpoint),
    }
    
    model.train()

    start_epoch = 0
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
            for k in ("model_type", "hidden_dim", "diff_steps", "obs_len", "pred_len", "patch_size"):
                old = ckpt_cfg.get(k)
                new = run_config.get(k)
                if old is not None and new is not None and str(old) != str(new):
                    print(f"[WARN] resume 配置不一致：{k}: ckpt={old} vs args={new}（可能导致加载失败或效果异常）")
            # Prior mismatch is important for residual diffusion.
            old_prior = ckpt_cfg.get("prior_checkpoint")
            new_prior = run_config.get("prior_checkpoint")
            if old_prior is not None and new_prior is not None and str(old_prior) != str(new_prior):
                print(f"[WARN] resume 配置不一致：prior_checkpoint: ckpt={old_prior} vs args={new_prior}（Residual 模式必须一致）")
        print(f"[OK] Resumed from {resume_from} (start_epoch={start_epoch})")

    prior_model = None
    if args.prior_checkpoint:
        prior_model = _load_baseline_prior(str(args.prior_checkpoint), device=device)
        print(f"[OK] Residual mode enabled: prior={args.prior_checkpoint}")

    if start_epoch >= int(args.epochs):
        print(f"[DONE] start_epoch({start_epoch}) >= --epochs({int(args.epochs)}), nothing to train.")
        return
    
    for epoch in range(start_epoch, int(args.epochs)):
        total_loss = 0
        total_diff_loss = 0
        total_macro_loss = 0
        start_time = time.time()
        step_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break
            obs = batch['obs'].to(device)
            cond = batch['cond'].to(device)
            action_full = batch['action'].to(device) # Future Vel (normalized, full)
            
            nav_patch = None
            if args.model_type == 'physics':
                nav_patch = batch['nav_patch'].to(device)

            # Residual target: vel_residual = vel_full - vel_prior
            prior_vel_norm = None
            if prior_model is not None:
                with torch.no_grad():
                    prior_vel_norm = prior_model.sample_trajectory(obs, cond, int(args.pred_len))
                target_action = action_full - prior_vel_norm
            else:
                target_action = action_full
            
            optimizer.zero_grad()
            
            use_macro = float(args.lambda_rog) > 0 and int(epoch) >= int(args.rog_warmup_epochs)

            if use_macro:
                if args.model_type == 'physics':
                    diff_loss, x0_pred, timesteps = model.compute_loss(
                        obs,
                        cond,
                        target_action,
                        nav_patch=nav_patch,
                        return_x0_pred=True,
                        return_timesteps=True,
                    )
                else:
                    diff_loss, x0_pred, timesteps = model.compute_loss(
                        obs,
                        cond,
                        target_action,
                        return_x0_pred=True,
                        return_timesteps=True,
                    )

                start_pos_norm = obs[:, -1, :2]  # (B, 2)
                start_pos = (start_pos_norm + 1.0) / 2.0 * pos_range + pos_min  # (B, 2)

                # x0_pred is the model's prediction of the clean target (residual if prior is enabled).
                pred_vel_norm = x0_pred.permute(0, 2, 1)  # (B, F, 2)
                if prior_vel_norm is not None:
                    pred_vel_norm = pred_vel_norm + prior_vel_norm
                pred_vel = pred_vel_norm * vel_std + vel_mean
                gt_vel = action_full * vel_std + vel_mean

                pred_pos = start_pos[:, None, :] + torch.cumsum(pred_vel, dim=1)
                gt_pos = start_pos[:, None, :] + torch.cumsum(gt_vel, dim=1)

                macro_eps = float(args.rog_eps)
                # displacement magnitude (used for optional displacement-aware weighting)
                gt_disp_end = gt_pos[:, -1, :] - start_pos  # (B, 2)
                gt_disp_norm = torch.linalg.norm(gt_disp_end, dim=-1)  # (B,)

                if args.macro_metric == "rog":
                    macro_pred = _rog_per_traj(pred_pos, eps=macro_eps)  # (B,)
                    macro_gt = _rog_per_traj(gt_pos, eps=macro_eps)      # (B,)
                    if args.rog_loss == "relative":
                        macro_loss_per = ((macro_pred - macro_gt) / (macro_gt + macro_eps)) ** 2
                    elif args.rog_loss == "batch_relative":
                        denom = torch.clamp_min(macro_gt.mean(), float(args.macro_rel_eps))
                        macro_loss_per = ((macro_pred - macro_gt) / denom) ** 2
                    else:
                        macro_loss_per = (macro_pred - macro_gt) ** 2
                elif args.macro_metric == "epe":
                    pred_end = pred_pos[:, -1, :]  # (B, 2)
                    gt_end = gt_pos[:, -1, :]      # (B, 2)
                    diff_vec = pred_end - gt_end
                    macro_loss_per = (diff_vec ** 2).sum(dim=-1)  # (B,)
                    if args.rog_loss == "relative":
                        gt_disp = gt_end - start_pos
                        denom = (gt_disp ** 2).sum(dim=-1)
                        denom = torch.clamp_min(denom, float(args.macro_rel_eps))
                        macro_loss_per = macro_loss_per / denom
                    elif args.rog_loss == "batch_relative":
                        gt_disp = gt_end - start_pos
                        denom = torch.linalg.norm(gt_disp, dim=-1).mean()
                        denom = torch.clamp_min(denom, float(args.macro_rel_eps))
                        macro_loss_per = macro_loss_per / (denom ** 2)
                elif args.macro_metric == "multi_epe":
                    # Multi-point displacement constraint on x0_pred positions (cheap, no unrolling).
                    indices = _macro_indices(int(args.pred_len), list(args.macro_points))
                    if not indices:
                        raise ValueError("--macro_metric multi_epe requires non-empty --macro_points")

                    errs = []
                    for idx in indices:
                        diff_vec = pred_pos[:, idx, :] - gt_pos[:, idx, :]
                        err = (diff_vec ** 2).sum(dim=-1)  # (B,)
                        if args.rog_loss == "relative":
                            gt_disp_k = gt_pos[:, idx, :] - start_pos
                            denom = (gt_disp_k ** 2).sum(dim=-1)
                            denom = torch.clamp_min(denom, float(args.macro_rel_eps))
                            err = err / denom
                        elif args.rog_loss == "batch_relative":
                            denom = torch.clamp_min(gt_disp_norm.mean(), float(args.macro_rel_eps))
                            err = err / (denom ** 2)
                        errs.append(err)
                    macro_loss_per = torch.stack(errs, dim=0).mean(dim=0)  # (B,)
                else:
                    raise ValueError(f"Unknown --macro_metric: {args.macro_metric}")

                # Optional: soft displacement-aware weighting to avoid low-displacement dominance.
                weights = torch.ones_like(macro_loss_per, dtype=torch.float32)
                if args.macro_disp_weight != "none":
                    if args.macro_disp_weight == "tanh":
                        w_disp = torch.tanh(float(args.macro_disp_alpha) * gt_disp_norm)
                    elif args.macro_disp_weight == "clip":
                        w_disp = torch.clamp(gt_disp_norm, float(args.macro_disp_clip_min), float(args.macro_disp_clip_max))
                    else:
                        raise ValueError(f"Unknown --macro_disp_weight: {args.macro_disp_weight}")
                    weights = weights * w_disp.to(torch.float32)

                # Optional: timestep reweighting (keep sampling uniform; only reweight macro term)
                w_t = _macro_t_weight(
                    timesteps=timesteps,
                    diff_steps=int(args.diff_steps),
                    mode=str(args.macro_t_weight),
                    gamma=float(args.macro_t_gamma),
                )
                weights = weights * w_t.to(torch.float32)

                t_thr = int(args.macro_t_threshold) if args.macro_t_threshold is not None else None
                if t_thr is None or t_thr >= int(args.diff_steps):
                    denom = torch.clamp_min(weights.sum(), 1e-6)
                    macro_loss = (macro_loss_per * weights).sum() / denom
                elif t_thr <= 0:
                    macro_loss = torch.zeros((), device=device)
                else:
                    mask = timesteps < t_thr
                    if mask.any():
                        ml = macro_loss_per[mask]
                        w = weights[mask]
                        denom = torch.clamp_min(w.sum(), 1e-6)
                        macro_loss = (ml * w).sum() / denom
                    else:
                        macro_loss = torch.zeros((), device=device)

                loss = diff_loss + float(args.lambda_rog) * macro_loss
            else:
                if args.model_type == 'physics':
                    diff_loss = model(obs, cond, target=target_action, nav_patch=nav_patch)
                else:
                    diff_loss = model(obs, cond, target=target_action)
                macro_loss = None
                loss = diff_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_diff_loss += diff_loss.item()
            if macro_loss is not None:
                total_macro_loss += float(macro_loss.item())
            step_count += 1
            
            if batch_idx % 100 == 0:
                 if macro_loss is None:
                     print(f"Epoch {epoch} | Batch {batch_idx} | Diff Loss {diff_loss.item():.4f}")
                 else:
                     print(
                         f"Epoch {epoch} | Batch {batch_idx} | "
                         f"Diff Loss {diff_loss.item():.4f} | Macro Loss {macro_loss.item():.4f} | "
                         f"lambda_rog={float(args.lambda_rog):g} | metric={args.macro_metric} | "
                         f"t<thr({int(args.macro_t_threshold)})"
                     )
                 
        avg_loss = total_loss / max(step_count, 1)
        duration = time.time() - start_time
        print(f"Epoch {epoch} Done. Loss: {avg_loss:.4f}. Time: {duration:.1f}s")
        
        torch.save(
            {
                "epoch": epoch,
                "loss": float(avg_loss),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": run_config,
            },
            save_dir / "last.pt",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='diff_v1')
    parser.add_argument('--model_type', type=str, choices=['diffusion', 'physics'], default='diffusion')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='train')
    parser.add_argument('--splits_dir', type=str, default=None, help="override splits dir (default: <processed_dir>/splits)")
    # Physics args
    parser.add_argument('--nav_file', type=str, default=None)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument(
        '--nav_patch_channel2',
        type=str,
        choices=['speed', 'count', 'zeros'],
        default='speed',
        help="nav_patch 第3通道：speed(默认)/count/log1p(count)/zeros(置零，仅方向)",
    )
    parser.add_argument('--lambda_macro', type=float, default=0.0, help="(deprecated) use --lambda_rog instead")
    
    # Model args
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--diff_steps', type=int, default=100)
    
    # Train args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_batches', type=int, default=None, help="limit batches per epoch (for smoke runs)")
    parser.add_argument('--resume', action='store_true', help="resume from data/experiments/<exp_name>/last.pt if exists")
    parser.add_argument('--resume_from', type=str, default=None, help="explicit checkpoint path to resume from")
    parser.add_argument('--no_resume_optim', action='store_true', help="when resuming, do NOT load optimizer_state_dict (recommended when changing loss/weights)")
    parser.add_argument('--prior_checkpoint', type=str, default=None, help="Residual diffusion: frozen deterministic prior checkpoint (SeqBaseline last.pt)")

    # Training-time macro regularization (paper-facing; cheap, no sampling)
    parser.add_argument('--lambda_rog', type=float, default=0.0, help="Macro Loss weight (0 disables)")
    parser.add_argument('--macro_metric', type=str, choices=['epe', 'rog', 'multi_epe'], default='epe', help="macro target: epe=endpoint error (pos-space), multi_epe=multi-point EPE, rog=radius of gyration")
    parser.add_argument('--macro_t_threshold', type=int, default=50, help="only apply macro loss when diffusion timestep t < threshold (hard SNR gate); set >=diff_steps to disable")
    parser.add_argument('--rog_loss', type=str, choices=['relative', 'absolute', 'batch_relative'], default='relative', help="macro loss scaling: relative/absolute/batch_relative")
    parser.add_argument('--macro_rel_eps', type=float, default=1.0, help="denominator floor for relative macro loss (prevents blow-up on near-stationary windows)")
    parser.add_argument('--rog_warmup_epochs', type=int, default=0, help="only apply Rog loss after N warmup epochs")
    parser.add_argument('--rog_eps', type=float, default=1e-6)

    # Displacement-aware weighting (to address low-displacement dominance)
    parser.add_argument('--macro_disp_weight', type=str, choices=['none', 'tanh', 'clip'], default='none')
    parser.add_argument('--macro_disp_alpha', type=float, default=0.1, help="tanh(alpha*|gt_disp|) when --macro_disp_weight=tanh")
    parser.add_argument('--macro_disp_clip_min', type=float, default=0.0)
    parser.add_argument('--macro_disp_clip_max', type=float, default=1e9)

    # Macro loss timestep weighting (do NOT bias timestep sampling distribution)
    parser.add_argument('--macro_t_weight', type=str, choices=['none', 'exp'], default='none')
    parser.add_argument('--macro_t_gamma', type=float, default=2.0, help="exp(-gamma*t/T) when --macro_t_weight=exp")

    # Multi-lag points (used when --macro_metric=multi_epe)
    parser.add_argument('--macro_points', type=float, nargs='*', default=[0.25, 0.5, 1.0], help="fractions in (0,1] or indices; e.g., 0.25 0.5 1.0")
    
    args = parser.parse_args()

    # Backward compatibility: allow old flag to drive Rog loss.
    if float(args.lambda_rog) <= 0 and float(args.lambda_macro) > 0:
        args.lambda_rog = float(args.lambda_macro)
        print(f"[WARN] --lambda_macro 已废弃：已自动映射为 --lambda_rog={float(args.lambda_rog):g}")
    
    if args.model_type == 'physics' and args.nav_file is None:
        raise ValueError("Physics model requires --nav_file")
        
    train(args)
