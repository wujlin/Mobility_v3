import torch
from torch.utils.data import Dataset
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from src.data.trajectories import TrajectoryStorage
from src.data.preprocess import Normalizer
from src.config.settings import NORM

TZ_SHANGHAI = timezone(timedelta(hours=8))

class SeqDataset(Dataset):
    def __init__(self, 
                 data_file: str, 
                 obs_len: int = 8, 
                 pred_len: int = 12, 
                 step: int = 1,
                 normalizer: Normalizer = None,
                 traj_ids: Optional[np.ndarray] = None,
                 mode: str = 'train'):
        
        self.storage = TrajectoryStorage(data_file, mode='r')
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.window_size = obs_len + pred_len
        self.step = step
        
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            stats_path = Path(data_file).resolve().parents[1] / "data_stats.json"
            self.normalizer = Normalizer.from_stats_json(stats_path) if stats_path.exists() else Normalizer(NORM)

        self.traj_ids = traj_ids.astype(np.int64) if traj_ids is not None else None

        # Build index of valid windows (traj_idx, start_t)
        self.samples = []
        self._build_index()
        
    def _build_index(self):
        """Scan all trajectories to find valid windows."""
        # This might be slow for millions of files, can cache it
        traj_iter = self.traj_ids if self.traj_ids is not None else range(len(self.storage))
        for i in traj_iter:
            i = int(i)
            # We access storage._ptr directly for speed if needed, 
            # but getting length is O(1) in our storage class
            # Wait, we need the length of the trajectory i
            # storage.get_trajectory(i) reads data, which is slow.
            # We need a lightweight way to get length.
            # In TrajectoryStorage for 'r' mode, we have self._ptr loaded.
            start = self.storage._ptr[i]
            end = self.storage._ptr[i+1]
            length = end - start
            
            # Sliding window properties
            # We need indices t such that t + window_size <= length
            for t in range(0, length - self.window_size + 1, self.step):
                self.samples.append((i, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_idx, start_t = self.samples[idx]
        
        # Read the full window
        # Optimization: modify get_trajectory to support partial read?
        # Current implementation reads full traj. If traj is long, this is wasteful.
        # But for now, let's stick to simple API.
        
        full_traj = self.storage.get_trajectory(traj_idx)
        
        obs_end = start_t + self.obs_len
        window_end = obs_end + self.pred_len
        
        # Slice data
        pos_window = full_traj['pos'][start_t:window_end] # (H+F, 2)
        vel_window = full_traj['vel'][start_t:window_end] # (H+F, 2)
        ts_window = full_traj['timestamp'][start_t:window_end]
        
        # Raw -> Normalize
        # Convert to float32 tensors
        pos_norm = self.normalizer.normalize_pos(pos_window)
        vel_norm = self.normalizer.normalize_vel(vel_window)
        
        pos_tensor = torch.from_numpy(pos_norm).float()
        vel_tensor = torch.from_numpy(vel_norm).float()
        
        # Obs: [0 : obs_len]
        obs_pos = pos_tensor[:self.obs_len]
        obs_vel = vel_tensor[:self.obs_len]
        
        # Target: [obs_len : end]
        target_pos = pos_tensor[self.obs_len:]
        target_vel = vel_tensor[self.obs_len:]
        
        # Conditional Features
        # 1. Time
        t0_ts = ts_window[0]
        dt0 = datetime.fromtimestamp(int(t0_ts), tz=TZ_SHANGHAI)
        hour = dt0.hour / 23.0  # Simple norm 0-1
        day = dt0.weekday() / 6.0 # Simple norm 0-1
        
        # 2. OD (normalized pos)
        # Origin = pos[0], Dest = pos[-1] of the TRIP (not just window)
        # But we only have window here...
        # Wait, condition is (o, d) of the *whole trip*?
        # development.md says: o, d, t0.
        # Usually for full trip generation we know D.
        # For sliding window prediction, do we know the global destination?
        # The storage has 'dest_idx' but that's just an index.
        # Ideally we want the coordinate of the destination.
        # Current storage doesn't store dest coord in metadata efficiently.
        # We can take full_traj['pos'][-1] as destination.
        
        trip_o = self.normalizer.normalize_pos(full_traj['pos'][0])
        trip_d = self.normalizer.normalize_pos(full_traj['pos'][-1])
        
        cond_vec = torch.tensor([
            hour, 
            day, 
            trip_o[0], trip_o[1], 
            trip_d[0], trip_d[1]
        ], dtype=torch.float32)
        
        # Construct output
        # Obs usually combines state. [pos, vel]
        obs_combined = torch.cat([obs_pos, obs_vel], dim=-1) # (H, 4)
        
        return {
            "obs": obs_combined,       # (H, 4)
            "target_pos": target_pos,  # (F, 2)
            "target_vel": target_vel,  # (F, 2)
            "cond": cond_vec,          # (6,)
            "meta": {
                "traj_idx": traj_idx,
                "start_t": start_t
            }
        }
