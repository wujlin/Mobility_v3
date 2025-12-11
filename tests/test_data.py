import numpy as np
import torch
from pathlib import Path
import os
import shutil
from src.data.trajectories import TrajectoryStorage
from src.data.datasets_seq import SeqDataset
from src.data.datasets_diffusion import DiffusionDataset
from src.features.nav_field import NavField
import tempfile

def test_data_pipeline():
    print("Testing Data Pipeline...")
    
    # Setup temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        traj_file = os.path.join(temp_dir, 'test_traj.h5')
        nav_file = os.path.join(temp_dir, 'test_nav.npz')
        
        # 1. Create Dummy Trajectories
        # Create 10 trajectories, length 50-100
        traj_list = []
        for i in range(10):
            length = np.random.randint(50, 100)
            # Random walk
            pos = np.cumsum(np.random.randn(length, 2), axis=0).astype(np.float32) + 100.0
            # Timestamp: 1 sec interval
            ts = (np.arange(length) * 1 + 1600000000).astype(np.int64)
            
            traj_list.append({
                'positions': pos,
                'vel': np.zeros_like(pos), # Not used for append input usually, calc internally? 
                # Wait, TrajectoryStorage currently DOES compute vel relative to previous pos in read
                # BUT in append it expects 'positions' and 'timestamp'.
                # Let's check TrajectoryStorage.append signature.
                # It takes dict with 'positions', 'timestamp'.
                'timestamp': ts,
                'vehicle_id': i
            })
            
        # Write
        TrajectoryStorage.create(traj_file, overwrite=True)
        with TrajectoryStorage(traj_file, 'r+') as store:
            store.append(traj_list)
            print(f"Written {len(store)} trajectories.")
            
        # 2. Create Dummy Nav Field
        H, W = 256, 256
        direction = np.zeros((2, H, W), dtype=np.float32)
        direction[0] = 1.0 # All pointing Y+
        speed = np.ones((H, W), dtype=np.float32) * 10.0
        
        np.savez(nav_file, direction=direction, speed=speed)
        print("Written Nav Field.")
        
        # 3. Test SeqDataset
        ds_seq = SeqDataset(traj_file, obs_len=8, pred_len=12)
        print(f"SeqDataset size: {len(ds_seq)}")
        
        sample_seq = ds_seq[0]
        # Check shapes
        # Obs: (H, 4) [pos, vel]
        assert sample_seq['obs'].shape == (8, 4)
        assert sample_seq['target_pos'].shape == (12, 2)
        assert sample_seq['cond'].shape == (6,)
        print("SeqDataset Shapes OK.")
        
        # 4. Test DiffusionDataset
        ds_diff = DiffusionDataset(traj_file, obs_len=4, pred_len=16, 
                                   nav_field_file=nav_file, nav_patch_size=32)
        print(f"DiffusionDataset size: {len(ds_diff)}")
        
        sample_diff = ds_diff[0]
        # Obs: (H, 4)
        assert sample_diff['obs'].shape == (4, 4)
        # Action: (F, 2)
        assert sample_diff['action'].shape == (16, 2)
        # Nav Patch: (3, 32, 32)
        assert sample_diff['nav_patch'].shape == (3, 32, 32)
        print("DiffusionDataset Shapes OK.")
        
        # Check normalization
        # input pos was around 100.
        # NORM default is pos_max=256. 
        # So 100 -> (100/256 * 2 - 1) approx -0.2.
        print("Example Obs Pos (Norm):", sample_diff['obs'][0, :2])
        
    finally:
        shutil.rmtree(temp_dir)
        print("Cleaned up.")

if __name__ == "__main__":
    test_data_pipeline()
