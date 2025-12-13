import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union

class TrajectoryStorage:
    """
    Handles read/write operations for the unified HDF5 trajectory format.
    Structure: Flat arrays + Pointer array.
    """
    def __init__(self, file_path: Union[str, Path], mode: str = 'r'):
        self.file_path = Path(file_path)
        self.mode = mode
        self.file = None
        
        # Cache for read mode
        self._ptr = None
        self._length = 0
        
        if self.mode == 'r' and self.file_path.exists():
            self.open()

    def open(self):
        if self.file is None:
            self.file = h5py.File(self.file_path, self.mode)
            if self.mode == 'r':
                self._ptr = self.file['traj_ptr'][:]
                self._length = len(self._ptr) - 1

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        if self.mode == 'r':
            return self._length
        elif self.file is not None and 'traj_ptr' in self.file:
             return len(self.file['traj_ptr']) - 1
        return 0

    @staticmethod
    def create(file_path: Union[str, Path], overwrite: bool = False):
        """Initialize a new HDF5 file with empty datasets."""
        path = Path(file_path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists.")
        
        with h5py.File(path, 'w') as f:
            # Resizable datasets
            f.create_dataset('positions', shape=(0, 2), maxshape=(None, 2), dtype='float32') # [y, x]
            f.create_dataset('timestamps', shape=(0,), maxshape=(None,), dtype='int64')
            
            # Pointer: starts with [0]
            f.create_dataset('traj_ptr', data=[0], maxshape=(None,), dtype='int64')
            
            # Metadata arrays (per trajectory)
            f.create_dataset('origin_idx', shape=(0,), maxshape=(None,), dtype='int64')
            f.create_dataset('dest_idx', shape=(0,), maxshape=(None,), dtype='int64')
            
            # Group for other meta
            grp = f.create_group('meta')
            grp.create_dataset('vehicle_id', shape=(0,), maxshape=(None,), dtype='int64')
            grp.create_dataset('start_time', shape=(0,), maxshape=(None,), dtype='int64')
            grp.create_dataset('end_time', shape=(0,), maxshape=(None,), dtype='int64')

    def append(self, trajectories: List[Dict[str, np.ndarray]]):
        """
        Append a batch of trajectories to the file.
        Each traj in trajectories must have:
        - positions: (T, 2)
        - timestamp: (T,)
        - meta: dict with vehicle_id, etc.
        """
        if self.mode != 'r+':
            raise ValueError("File must be opened in 'r+' mode to append.")
            
        if self.file is None:
            self.open()
            
        f = self.file
        n_new = len(trajectories)
        
        # 1. Calculate new sizes
        new_points_count = sum(len(t['positions']) for t in trajectories)
        current_points_count = f['positions'].shape[0]
        current_traj_count = f['traj_ptr'].shape[0] - 1
        
        # 2. Resize datasets
        f['positions'].resize(current_points_count + new_points_count, axis=0)
        f['timestamps'].resize(current_points_count + new_points_count, axis=0)
        f['traj_ptr'].resize(current_traj_count + n_new + 1, axis=0) # +1 for the last ptr
        
        f['origin_idx'].resize(current_traj_count + n_new, axis=0)
        f['dest_idx'].resize(current_traj_count + n_new, axis=0)
        f['meta/vehicle_id'].resize(current_traj_count + n_new, axis=0)
        f['meta/start_time'].resize(current_traj_count + n_new, axis=0)
        f['meta/end_time'].resize(current_traj_count + n_new, axis=0)
        
        # 3. Fill data
        curr_ptr = f['traj_ptr'][-1 if current_traj_count > 0 else 0] # Should match current_points_count
        # If accessing raw dataset [-1] might be tricky if it was just loaded? 
        # Actually f['traj_ptr'][-1] is reliable.
        # But we just resized it. We need the value BEFORE the new entries.
        # The resize fills with 0? No, h5py keeps old data.
        # The last element *was* valid, now we added empty slots.
        # Wait, resize extends.
        
        # Let's read the pointer start
        ptr_start = current_points_count 
        
        new_ptrs = []
        acc_ptr = ptr_start
        
        all_pos = []
        all_time = []
        
        meta_vid = []
        meta_start = []
        meta_end = []
        origin_idx = []
        dest_idx = []
        
        for t in trajectories:
            t_len = len(t['positions'])
            start_idx = acc_ptr
            acc_ptr += t_len
            new_ptrs.append(acc_ptr)
            
            all_pos.append(t['positions'])
            all_time.append(t['timestamp'])
            
            meta_vid.append(t.get('vehicle_id', -1))
            meta_start.append(t['timestamp'][0])
            meta_end.append(t['timestamp'][-1])

            origin_idx.append(start_idx)
            dest_idx.append(acc_ptr - 1)
            
        # Bulk write
        f['positions'][current_points_count:] = np.concatenate(all_pos, axis=0)
        f['timestamps'][current_points_count:] = np.concatenate(all_time, axis=0)
        f['traj_ptr'][current_traj_count+1:] = new_ptrs # Fill the new slots
        
        f['origin_idx'][current_traj_count:] = origin_idx
        f['dest_idx'][current_traj_count:] = dest_idx
        f['meta/vehicle_id'][current_traj_count:] = meta_vid
        f['meta/start_time'][current_traj_count:] = meta_start
        f['meta/end_time'][current_traj_count:] = meta_end

    def get_trajectory(self, idx: int):
        """Return dict with positions, speed, etc. for index idx."""
        if self.file is None:
            self.open()
            
        start = self._ptr[idx]
        end = self._ptr[idx+1]
        
        pos = self.file['positions'][start:end]
        ts = self.file['timestamps'][start:end]
        
        # Online vel: step displacement (backward diff), padded at t=0.
        vel = np.zeros_like(pos)
        if len(pos) > 1:
            vel[1:] = pos[1:] - pos[0:-1]
            vel[0] = vel[1]  # pad
            
        vehicle_id = self.file['meta/vehicle_id'][idx]
        
        return {
            'pos': pos,  # (T, 2) [y, x]
            'vel': vel,  # (T, 2) [vy, vx]
            'timestamp': ts,
            'vehicle_id': vehicle_id
        }
