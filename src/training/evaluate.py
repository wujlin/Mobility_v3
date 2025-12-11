import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json
import numpy as np
from tqdm import tqdm

from src.models.seq.seq_baseline import SeqBaseline
from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.models.physics.physics_condition_diffusion import PhysicsConditionDiffusion
from src.data.datasets_seq import SeqDataset
from src.data.datasets_diffusion import DiffusionDataset
from src.data.preprocess import Normalizer
from src.config.settings import NORM, GRID
from src.evaluation.micro_metrics import compute_micro_metrics
from src.evaluation.macro_metrics import compute_macro_metrics

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    # Evaluation usually on 'test' split via mode='r'? 
    # Or separate file. Assuming args.data_path points to test file.
    
    norm = Normalizer(NORM)
    
    if args.model_type == 'baseline':
        dataset = SeqDataset(args.data_path, obs_len=args.obs_len, pred_len=args.pred_len)
    else:
        # Physics or Diffusion
        nav_file = args.nav_file if args.model_type == 'physics' else None
        dataset = DiffusionDataset(
            args.data_path, 
            obs_len=args.obs_len, 
            pred_len=args.pred_len,
            nav_field_file=nav_file,
            nav_patch_size=args.patch_size
        )
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 2. Load Model
    print(f"Loading {args.model_type} model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    if args.model_type == 'baseline':
        model = SeqBaseline(
            obs_dim=4, act_dim=2, cond_dim=6,
            hidden_dim=args.hidden_dim
        )
    elif args.model_type == 'diffusion':
        model = DiffusionTrajectoryModel(
            obs_dim=4, act_dim=2, cond_dim=6,
            obs_len=args.obs_len, pred_len=args.pred_len,
            hidden_dim=args.hidden_dim,
            diffusion_steps=args.diff_steps
        )
    elif args.model_type == 'physics':
        model = PhysicsConditionDiffusion(
            obs_dim=4, act_dim=2, cond_dim=6,
            nav_patch_size=args.patch_size,
            obs_len=args.obs_len, pred_len=args.pred_len,
            hidden_dim=args.hidden_dim,
            diffusion_steps=args.diff_steps
        )
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 3. Inference Loop
    all_preds_pos = [] # Denormalized predicted positions
    all_targets_pos = [] # Denormalized GT positions
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            obs = batch['obs'].to(device)
            cond = batch['cond'].to(device)
            
            # Ground Truth Feature
            if args.model_type == 'baseline':
                target_pos_norm = batch['target_pos'].to(device)
            else:
                # Diffusion dataset doesn't return target_pos explicitly in 'action' (it's vel).
                # But it has it in raw storage?
                # Actually DiffusionDataset __getitem__ returns 'action'=vel.
                # We need GT Position for evaluation.
                # We can either:
                # A) Integrate GT Vel from Last Obs Pos.
                # B) Modify Dataset to return GT Pos.
                # Let's use Integration for consistency.
                
                # Get integration start point
                # Obs: (B, H, 4) -> [pos, vel]
                start_pos_norm = obs[:, -1, :2] 
                
                # GT Vel (Action) not used for inference but for GT reconstruction
                gt_vel_norm = batch['action'].to(device)
                
                # Reconstruct GT Pos Norm? No, easier to just use integration logic logic.
                # Wait, integration in normalized space is invalid (as noted before).
                # We need to DENORMALIZE start_pos, DENORMALIZE gt_vel, then integrate.
                pass
                
            # Model Sample
            # Returns: (B, F, 2) - Velocity (for all our models currently)
            # Wait, SeqBaseline predicts Vel. Diffusion predicts Vel.
            nav_patch = batch['nav_patch'].to(device) if args.model_type == 'physics' else None
            
            if args.model_type == 'physics':
                pred_vel_norm = model.sample_trajectory(obs, cond, args.pred_len, nav_patch=nav_patch)
            else:
                pred_vel_norm = model.sample_trajectory(obs, cond, args.pred_len)
                
            # 4. Denormalize & Integrate
            # Need start pos (denormalized)
            start_pos_norm = obs[:, -1, :2]
            start_pos = norm.denormalize_pos(start_pos_norm.cpu().numpy())
            
            pred_vel = norm.denormalize_vel(pred_vel_norm.cpu().numpy())
            
            # Get GT Vel to check against
            if args.model_type == 'baseline':
                gt_vel_norm = batch['target_vel'].cpu().numpy()
            else:
                gt_vel_norm = batch['action'].cpu().numpy()
                
            gt_vel = norm.denormalize_vel(gt_vel_norm)
            
            # Integrate to Position
            # pred_pos[t] = pred_pos[t-1] + pred_vel[t]
            B_size = pred_vel.shape[0]
            curr_pred_pos = start_pos.copy()
            curr_gt_pos = start_pos.copy()
            
            batch_pred_pos = []
            batch_gt_pos = []
            
            for t in range(args.pred_len):
                curr_pred_pos = curr_pred_pos + pred_vel[:, t]
                curr_gt_pos = curr_gt_pos + gt_vel[:, t]
                
                batch_pred_pos.append(curr_pred_pos.copy())
                batch_gt_pos.append(curr_gt_pos.copy())
                
            all_preds_pos.append(np.stack(batch_pred_pos, axis=1)) # (B, F, 2)
            all_targets_pos.append(np.stack(batch_gt_pos, axis=1))
            
    # Concatenate
    preds_pos = np.concatenate(all_preds_pos, axis=0) # (N, F, 2)
    targets_pos = np.concatenate(all_targets_pos, axis=0)
    
    # 5. Compute Metrics
    print("Computing Metrics...")
    # Convert to tensor for metric functions
    preds_t = torch.from_numpy(preds_pos)
    targets_t = torch.from_numpy(targets_pos)
    
    micro = compute_micro_metrics(preds_t, targets_t)
    macro = compute_macro_metrics(preds_t)
    
    # Merge
    results = {**micro, **macro}
    print(json.dumps(results, indent=2))
    
    # 6. Save
    out_dir = Path(f"data/experiments/{args.exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "metrics.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    # Save Samples (Top 100)
    np.savez(out_dir / "samples.npz", 
             preds=preds_pos[:100], 
             targets=targets_pos[:100])
             
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--model_type', type=str, choices=['baseline', 'diffusion', 'physics'], required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    
    # Physics args
    parser.add_argument('--nav_file', type=str, default=None)
    parser.add_argument('--patch_size', type=int, default=32)
    
    # Model args
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=128) # check defaults
    parser.add_argument('--diff_steps', type=int, default=100)
    
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    evaluate(args)
