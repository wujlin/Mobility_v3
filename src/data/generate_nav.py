import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py

from src.data.trajectories import TrajectoryStorage
from src.config.settings import GRID, NORM

def generate_nav_field(args):
    """
    Generate Navigation Field (Direction, Speed) from Training Trajectories.
    """
    traj_path = args.traj_file
    split_path = args.train_ids
    out_path = args.output
    
    print(f"Generating Nav Field from {traj_path}")
    print(f"Using split: {split_path}")
    
    # Load IDs
    train_ids = np.load(split_path)
    
    # Initialize Grids
    # Direction Sum: (2, H, W)
    vel_sum = np.zeros((2, GRID.H, GRID.W), dtype=np.float32)
    count = np.zeros((GRID.H, GRID.W), dtype=np.float32)
    
    # Speed Sum (for mean speed)
    speed_sum = np.zeros((GRID.H, GRID.W), dtype=np.float32)
    
    store = TrajectoryStorage(traj_path, mode='r')
    
    print("Accumulating velocities...")
    # To optimize: read in batches? TrajectoryStorage supports single get.
    # We can just loop.
    
    for idx in tqdm(train_ids):
        traj = store.get_trajectory(idx)
        pos = traj['pos'] # (T, 2)
        vel = traj['vel'] # (T, 2)
        
        # Valid velocities (magnitude > threshold to avoid noise)
        speed = np.linalg.norm(vel, axis=1)
        mask = speed > args.min_speed
        # Check settings: vel_std ~ 3-5. 
        # 0.5 is small enough.
        
        pos_valid = pos[mask]
        vel_valid = vel[mask]
        speed_valid = speed[mask]
        
        if len(pos_valid) == 0:
            continue
            
        # Rasterize
        # Ensure bounds
        ys = np.clip(pos_valid[:, 0].astype(int), 0, GRID.H - 1)
        xs = np.clip(pos_valid[:, 1].astype(int), 0, GRID.W - 1)
        
        # Vectorized accumulation using np.add.at
        # Flatten indices
        flat_indices = ys * GRID.W + xs
        
        np.add.at(vel_sum[0].ravel(), flat_indices, vel_valid[:, 0]) # vy
        np.add.at(vel_sum[1].ravel(), flat_indices, vel_valid[:, 1]) # vx
        np.add.at(speed_sum.ravel(), flat_indices, speed_valid)
        np.add.at(count.ravel(), flat_indices, 1)
        
    store.close()
    
    # Compute Means
    print("Computing fields...")
    mask_count = count > 0
    
    # Initialize with default? 
    # Direction: if no data, 0. Or random? 0.
    avg_vel = np.zeros_like(vel_sum)
    avg_speed = np.zeros_like(speed_sum)
    
    avg_vel[:, mask_count] = vel_sum[:, mask_count] / count[mask_count]
    avg_speed[mask_count] = speed_sum[mask_count] / count[mask_count]
    
    # Direction Unit Vectors
    vel_norm = np.linalg.norm(avg_vel, axis=0)
    # Avoid div by zero
    norm_mask = vel_norm > 1e-6
    
    direction = np.zeros_like(avg_vel)
    direction[:, norm_mask] = avg_vel[:, norm_mask] / vel_norm[norm_mask]
    
    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, direction=direction, speed=avg_speed, count=count)
    print(f"Saved Nav Field to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_file', type=str, default='data/processed/trajectories/shenzhen_trajectories.h5')
    parser.add_argument('--train_ids', type=str, default='data/processed/splits/train_ids.npy')
    parser.add_argument('--output', type=str, default='data/processed/nav_field.npz')
    parser.add_argument('--min_speed', type=float, default=0.5, help="Minimum speed to contribute to nav field (grid units/step)")
    
    args = parser.parse_args()
    generate_nav_field(args)
