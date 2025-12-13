import torch
from torch.utils.data import Dataset
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from src.data.trajectories import TrajectoryStorage
from src.data.preprocess import Normalizer
from src.features.nav_field import NavField
from src.config.settings import NORM

TZ_SHANGHAI = timezone(timedelta(hours=8))

class DiffusionDataset(Dataset):
    def __init__(self, 
                 data_file: str, 
                 obs_len: int = 4, 
                 pred_len: int = 16, 
                 step: int = 1,
                 nav_field_file: str = None,
                 nav_patch_size: int = 32,
                 normalizer: Normalizer = None,
                 traj_ids: Optional[np.ndarray] = None):
        
        self.storage = TrajectoryStorage(data_file, mode='r')
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.window_size = obs_len + pred_len
        self.step = step
        self.nav_patch_size = nav_patch_size
        
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            stats_path = Path(data_file).resolve().parents[1] / "data_stats.json"
            self.normalizer = Normalizer.from_stats_json(stats_path) if stats_path.exists() else Normalizer(NORM)

        self.traj_ids = traj_ids.astype(np.int64) if traj_ids is not None else None
        
        # Load Nav Field if provided
        self.nav_field = None
        if nav_field_file:
            self.nav_field = NavField(nav_field_file)
            
        self.samples = []
        self._build_index()

    def _build_index(self):
        """Scan valid windows."""
        # Reuse logic from SeqDataset or similar
        traj_iter = self.traj_ids if self.traj_ids is not None else range(len(self.storage))
        for i in traj_iter:
            i = int(i)
            start = self.storage._ptr[i]
            end = self.storage._ptr[i+1]
            length = end - start
            for t in range(0, length - self.window_size + 1, self.step):
                self.samples.append((i, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_idx, start_t = self.samples[idx]
        full_traj = self.storage.get_trajectory(traj_idx)
        
        # Slice
        window_end = start_t + self.window_size
        pos_window = full_traj['pos'][start_t:window_end]
        vel_window = full_traj['vel'][start_t:window_end]
        ts_window = full_traj['timestamp'][start_t:window_end]
        
        # Normalize
        pos_norm = self.normalizer.normalize_pos(pos_window)
        vel_norm = self.normalizer.normalize_vel(vel_window)
        
        pos_tensor = torch.from_numpy(pos_norm).float()
        vel_tensor = torch.from_numpy(vel_norm).float()
        
        # Split Obs / Action
        obs_pos = pos_tensor[:self.obs_len] # (H, 2)
        obs_vel = vel_tensor[:self.obs_len] # (H, 2)
        
        # Action is Future Velocities
        future_vel = vel_tensor[self.obs_len:] # (F, 2)
        
        # Condition (Time + OD)
        t0_ts = ts_window[0]
        dt0 = datetime.fromtimestamp(int(t0_ts), tz=TZ_SHANGHAI)
        hour = dt0.hour / 23.0
        day = dt0.weekday() / 6.0
        
        trip_o = self.normalizer.normalize_pos(full_traj['pos'][0])
        trip_d = self.normalizer.normalize_pos(full_traj['pos'][-1])
        
        cond_vec = torch.tensor([
            hour, day, 
            trip_o[0], trip_o[1], 
            trip_d[0], trip_d[1]
        ], dtype=torch.float32)
        
        # Combined Obs
        obs_combined = torch.cat([obs_pos, obs_vel], dim=-1) # (H, 4)
        
        result = {
            "obs": obs_combined,      # (H, 4)
            "action": future_vel,     # (F, 2)
            "cond": cond_vec,         # (6,)
        }
        
        # Nav Patch (if enabled)
        if self.nav_field:
            # Center is the last observed position
            # Use RAW (denormalized) position for lookup!
            center_pos = pos_window[self.obs_len - 1] 
            
            patch = self.nav_field.get_patch(center_pos, self.nav_patch_size)
            
            # Nav patch is already normalized?
            # NavField returns raw directions (unit) and speed as step displacement magnitude (grid_cell/step).
            # We should normalize speed. Direction is already mostly 0-1 range (well, -1 to 1).
            # Let's normalize speed channel.
            # patch shape (3, K, K). Channel 2 is speed.
            
            patch_tensor = torch.from_numpy(patch).float()
            
            # Normalize speed channel (index 2)
            # Basic scaling or reuse vel_std? Speed is magnitude, vel is vector.
            # Let's say we have nav_max_speed in settings? 
            # Or just divide by a constant like 30.0.
            # NORM.vel_mean is vector mean ~ 0.
            # NORM.vel_std is vector std.
            # Let's use vel_std[0] approx.
            # Simple division for now.
            patch_tensor[2] = patch_tensor[2] / self.normalizer.config.nav_max_speed
            
            result["nav_patch"] = patch_tensor # (3, K, K)
            
        return result
