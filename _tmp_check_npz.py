#!/usr/bin/env python3
import numpy as np

npz = np.load("data/route_windows/detroit/test_1024_route_windows.npz", allow_pickle=True)
print("Keys:", list(npz.files))
for k in npz.files:
    arr = npz[k]
    print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")
    if len(arr.shape) == 1 and arr.shape[0] > 0 and arr.shape[0] < 10000:
        print(f"   first 5: {arr[:5]}")
