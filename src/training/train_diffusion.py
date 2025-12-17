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

def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

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

    run_config = {
        "model_type": args.model_type,
        "data_path": args.data_path,
        "split": args.split,
        "splits_dir": args.splits_dir,
        "nav_file": args.nav_file,
        "patch_size": args.patch_size,
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
        "rog_loss": str(args.rog_loss),
        "rog_warmup_epochs": int(args.rog_warmup_epochs),
    }
    
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        total_diff_loss = 0
        total_rog_loss = 0
        start_time = time.time()
        step_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break
            obs = batch['obs'].to(device)
            cond = batch['cond'].to(device)
            action = batch['action'].to(device) # Future Vel
            
            nav_patch = None
            if args.model_type == 'physics':
                nav_patch = batch['nav_patch'].to(device)
            
            optimizer.zero_grad()
            
            use_rog = float(args.lambda_rog) > 0 and int(epoch) >= int(args.rog_warmup_epochs)

            if use_rog:
                if args.model_type == 'physics':
                    diff_loss, x0_pred = model.compute_loss(
                        obs,
                        cond,
                        action,
                        nav_patch=nav_patch,
                        return_x0_pred=True,
                    )
                else:
                    diff_loss, x0_pred = model.compute_loss(obs, cond, action, return_x0_pred=True)

                start_pos_norm = obs[:, -1, :2]  # (B, 2)
                start_pos = (start_pos_norm + 1.0) / 2.0 * pos_range + pos_min  # (B, 2)

                pred_vel_norm = x0_pred.permute(0, 2, 1)  # (B, F, 2)
                pred_vel = pred_vel_norm * vel_std + vel_mean
                gt_vel = action * vel_std + vel_mean

                pred_pos = start_pos[:, None, :] + torch.cumsum(pred_vel, dim=1)
                gt_pos = start_pos[:, None, :] + torch.cumsum(gt_vel, dim=1)

                rog_pred = _rog_per_traj(pred_pos, eps=float(args.rog_eps))
                rog_gt = _rog_per_traj(gt_pos, eps=float(args.rog_eps))

                if args.rog_loss == "relative":
                    rog_loss = ((rog_pred - rog_gt) / (rog_gt + float(args.rog_eps))) ** 2
                else:
                    rog_loss = (rog_pred - rog_gt) ** 2
                rog_loss = rog_loss.mean()

                loss = diff_loss + float(args.lambda_rog) * rog_loss
            else:
                if args.model_type == 'physics':
                    diff_loss = model(obs, cond, target=action, nav_patch=nav_patch)
                else:
                    diff_loss = model(obs, cond, target=action)
                rog_loss = None
                loss = diff_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_diff_loss += diff_loss.item()
            if rog_loss is not None:
                total_rog_loss += float(rog_loss.item())
            step_count += 1
            
            if batch_idx % 100 == 0:
                 if rog_loss is None:
                     print(f"Epoch {epoch} | Batch {batch_idx} | Diff Loss {diff_loss.item():.4f}")
                 else:
                     print(
                         f"Epoch {epoch} | Batch {batch_idx} | "
                         f"Diff Loss {diff_loss.item():.4f} | Rog Loss {rog_loss.item():.4f} | "
                         f"lambda_rog={float(args.lambda_rog):g}"
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

    # Training-time macro regularization (paper-facing; cheap, no sampling)
    parser.add_argument('--lambda_rog', type=float, default=0.0, help="Rog Macro Loss weight (0 disables)")
    parser.add_argument('--rog_loss', type=str, choices=['relative', 'absolute'], default='relative')
    parser.add_argument('--rog_warmup_epochs', type=int, default=0, help="only apply Rog loss after N warmup epochs")
    parser.add_argument('--rog_eps', type=float, default=1e-6)
    
    args = parser.parse_args()

    # Backward compatibility: allow old flag to drive Rog loss.
    if float(args.lambda_rog) <= 0 and float(args.lambda_macro) > 0:
        args.lambda_rog = float(args.lambda_macro)
        print(f"[WARN] --lambda_macro 已废弃：已自动映射为 --lambda_rog={float(args.lambda_rog):g}")
    
    if args.model_type == 'physics' and args.nav_file is None:
        raise ValueError("Physics model requires --nav_file")
        
    train(args)
