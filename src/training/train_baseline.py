import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import time
import os
import numpy as np
import random

from src.models.seq.seq_baseline import SeqBaseline
from src.data.datasets_seq import SeqDataset
from src.config.settings import GRID, NORM

def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    _set_seed(int(args.seed))
    print(f"Using seed: {int(args.seed)}")
    
    # 1. Data
    print("Loading datasets...")
    traj_ids = None
    if args.split != 'all':
        processed_dir = Path(args.data_path).resolve().parents[1]
        splits_dir = Path(args.splits_dir) if args.splits_dir else (processed_dir / "splits")
        split_file = splits_dir / f"{args.split}_ids.npy"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        traj_ids = np.load(split_file).astype(np.int64)
        print(f"Using split={args.split}: {len(traj_ids)} trajectories ({split_file})")

    train_dataset = SeqDataset(args.data_path, obs_len=args.obs_len, pred_len=args.pred_len, traj_ids=traj_ids)
    # torch-friendly denorm constants for displacement-aware weighting (optional)
    vel_mean = torch.tensor(train_dataset.normalizer.vel_mean, dtype=torch.float32, device=device)
    vel_std = torch.tensor(train_dataset.normalizer.vel_std, dtype=torch.float32, device=device)
    g = torch.Generator()
    g.manual_seed(int(args.seed))
    pin_memory = bool(torch.cuda.is_available())
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.num_workers),
        generator=g,
        pin_memory=pin_memory,
        persistent_workers=(int(args.num_workers) > 0),
    )
    
    # 2. Model
    model = SeqBaseline(
        obs_dim=4, # [pos, vel]
        act_dim=2, # [vel]
        cond_dim=6,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. Setup Logging
    save_dir = Path(f"data/experiments/{args.exp_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "model_type": "baseline",
        "data_path": args.data_path,
        "split": args.split,
        "splits_dir": args.splits_dir,
        "obs_len": args.obs_len,
        "pred_len": args.pred_len,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "disp_weight": str(args.disp_weight),
        "disp_alpha": float(args.disp_alpha),
        "disp_clip_min": float(args.disp_clip_min),
        "disp_clip_max": float(args.disp_clip_max),
    }
    
    print(f"Start training {args.exp_name}...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            obs = batch['obs'].to(device) # (B, H, 4)
            cond = batch['cond'].to(device) # (B, 6)
            target_vel = batch['target_vel'].to(device) # (B, F, 2)
            
            optimizer.zero_grad()
            
            # Optional displacement-aware weighting to mitigate low-displacement dominance.
            sample_weight = None
            if str(args.disp_weight) != "none":
                # Denormalize velocities to keep weighting in physical grid units.
                target_vel_denorm = target_vel * vel_std + vel_mean  # (B, F, 2)
                gt_disp = target_vel_denorm.sum(dim=1)  # (B, 2)
                disp_norm = torch.linalg.norm(gt_disp, dim=-1)  # (B,)
                if str(args.disp_weight) == "tanh":
                    sample_weight = torch.tanh(float(args.disp_alpha) * disp_norm)
                elif str(args.disp_weight) == "clip":
                    # Multiplicative weighting (additive logic): w = clip(disp / mean_disp, lo, hi)
                    # This boosts large-displacement windows with w>1, instead of only down-weighting small ones.
                    disp_ref = torch.clamp_min(disp_norm.mean(), 1e-6)
                    sample_weight = disp_norm / disp_ref
                    sample_weight = torch.clamp(
                        sample_weight,
                        min=float(args.disp_clip_min),
                        max=float(args.disp_clip_max),
                    )
                else:
                    raise ValueError(f"Unknown --disp_weight: {args.disp_weight}")
                # Avoid exact zeros which can zero-out gradients for near-stationary windows.
                sample_weight = torch.clamp_min(sample_weight, 1e-3)

            # Forward returns Loss directly
            loss = model(obs, cond, target=target_vel, sample_weight=sample_weight)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")

            if args.max_batches is not None and int(args.max_batches) > 0 and num_batches >= int(args.max_batches):
                break
                
        denom = max(num_batches, 1)
        avg_loss = total_loss / float(denom)
        duration = time.time() - start_time
        print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}. Time: {duration:.1f}s")
        
        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': run_config,
        }, save_dir / "last.pt")
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), save_dir / f"epoch_{epoch}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='baseline_v1')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='train')
    parser.add_argument('--splits_dir', type=str, default=None, help="override splits dir (default: <processed_dir>/splits)")
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_batches', type=int, default=None, help="limit batches per epoch (for quick iteration)")
    parser.add_argument(
        '--disp_weight',
        type=str,
        choices=['none', 'tanh', 'clip'],
        default='none',
        help="Displacement-aware weighting for L2 baseline (mitigate low-displacement dominance).",
    )
    parser.add_argument('--disp_alpha', type=float, default=0.1, help="tanh(alpha*|gt_disp|) when --disp_weight=tanh")
    parser.add_argument('--disp_clip_min', type=float, default=0.5, help="when --disp_weight=clip: clip(disp/mean_disp, min, max)")
    parser.add_argument('--disp_clip_max', type=float, default=5.0, help="when --disp_weight=clip: clip(disp/mean_disp, min, max)")
    
    args = parser.parse_args()
    train(args)
