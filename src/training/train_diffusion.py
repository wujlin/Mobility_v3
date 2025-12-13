import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import time
import os
import numpy as np

from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.models.physics.physics_condition_diffusion import PhysicsConditionDiffusion
from src.models.physics.macro_regularizer import MacroRegularizer
from src.data.datasets_diffusion import DiffusionDataset

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
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
    
    # Macro Regularizer (Optional)
    macro_reg = None
    if args.lambda_macro > 0:
        macro_reg = MacroRegularizer(target_alpha=1.4).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. Setup Logging
    save_dir = Path(f"data/experiments/{args.exp_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        total_diff_loss = 0
        total_macro_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            obs = batch['obs'].to(device)
            cond = batch['cond'].to(device)
            action = batch['action'].to(device) # Future Vel
            
            nav_patch = None
            if args.model_type == 'physics':
                nav_patch = batch['nav_patch'].to(device)
            
            optimizer.zero_grad()
            
            # Diffusion Loss
            if args.model_type == 'physics':
                diff_loss = model(obs, cond, target=action, nav_patch=nav_patch)
            else:
                diff_loss = model(obs, cond, target=action)
                
            loss = diff_loss
            
            # Macro Regularization (requires sampling during training - SLOW)
            # Only run every N batches or if explicitly enabled for fine-tuning
            # For this MVP script, we skip it inside the loop unless strictly needed.
            # Development.md CAUTION said this is hard.
            # We implemented the class, but integrating it requires `model.sample_trajectory` inside train loop.
            # This is expensive (multiple denoise steps).
            # Strategy: Skip for now, or apply on predicted noise? No, macro is on trajectory stats.
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_diff_loss += diff_loss.item()
            
            if batch_idx % 100 == 0:
                 print(f"Epoch {epoch} | Batch {batch_idx} | Diff Loss {diff_loss.item():.4f}")
                 
        avg_loss = total_loss / len(dataloader)
        duration = time.time() - start_time
        print(f"Epoch {epoch} Done. Loss: {avg_loss:.4f}. Time: {duration:.1f}s")
        
        torch.save(model.state_dict(), save_dir / "last.pt")

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
    parser.add_argument('--lambda_macro', type=float, default=0.0)
    
    # Model args
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--diff_steps', type=int, default=100)
    
    # Train args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    
    if args.model_type == 'physics' and args.nav_file is None:
        raise ValueError("Physics model requires --nav_file")
        
    train(args)
