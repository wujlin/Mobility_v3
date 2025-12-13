import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import time
import os
import numpy as np

from src.models.seq.seq_baseline import SeqBaseline
from src.data.datasets_seq import SeqDataset
from src.config.settings import GRID, NORM

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
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
    
    print(f"Start training {args.exp_name}...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            obs = batch['obs'].to(device) # (B, H, 4)
            cond = batch['cond'].to(device) # (B, 6)
            target_vel = batch['target_vel'].to(device) # (B, F, 2)
            
            optimizer.zero_grad()
            
            # Forward returns Loss directly
            loss = model(obs, cond, target=target_vel)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        duration = time.time() - start_time
        print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}. Time: {duration:.1f}s")
        
        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
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
    
    args = parser.parse_args()
    train(args)
