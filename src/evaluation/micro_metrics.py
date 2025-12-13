from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def compute_ade(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Average Displacement Error (L2 distance mean over steps)."""
    dist = torch.norm(pred - target, p=2, dim=-1)  # (B, F)
    return dist.mean().item()


def compute_fde(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Final Displacement Error (L2 distance at last step)."""
    dist = torch.norm(pred[:, -1] - target[:, -1], p=2, dim=-1)  # (B,)
    return dist.mean().item()


def _cdist_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise euclidean distances between two point sets."""
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    diff = a64[:, None, :] - b64[None, :, :]
    return np.linalg.norm(diff, axis=-1)

def _frechet_one(traj_p: np.ndarray, traj_q: np.ndarray) -> float:
    dist_matrix = _cdist_euclidean(traj_p, traj_q)
    n_p, n_q = dist_matrix.shape

    ca = np.full((n_p, n_q), -1.0, dtype=np.float64)
    ca[0, 0] = dist_matrix[0, 0]

    for k in range(1, n_p):
        ca[k, 0] = max(ca[k - 1, 0], dist_matrix[k, 0])

    for l in range(1, n_q):
        ca[0, l] = max(ca[0, l - 1], dist_matrix[0, l])

    for k in range(1, n_p):
        for l in range(1, n_q):
            ca[k, l] = max(
                dist_matrix[k, l],
                min(ca[k - 1, l], ca[k, l - 1], ca[k - 1, l - 1]),
            )

    return float(ca[n_p - 1, n_q - 1])


def _dtw_one(traj_p: np.ndarray, traj_q: np.ndarray) -> float:
    dist_matrix = _cdist_euclidean(traj_p, traj_q)
    n_p, n_q = dist_matrix.shape

    d = np.full((n_p + 1, n_q + 1), np.inf, dtype=np.float64)
    d[0, 0] = 0.0

    for k in range(1, n_p + 1):
        for l in range(1, n_q + 1):
            cost = dist_matrix[k - 1, l - 1]
            d[k, l] = cost + min(d[k - 1, l], d[k, l - 1], d[k - 1, l - 1])

    return float(d[n_p, n_q])


def compute_frechet_per_sample(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    pred/target: (B, F, 2) numpy arrays
    returns: (B,) discrete Fréchet distance
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, target={target.shape}")
    b = int(pred.shape[0])
    out = np.zeros((b,), dtype=np.float32)
    for i in range(b):
        out[i] = _frechet_one(pred[i], target[i])
    return out


def compute_dtw_per_sample(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    pred/target: (B, F, 2) numpy arrays
    returns: (B,) DTW distance
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, target={target.shape}")
    b = int(pred.shape[0])
    out = np.zeros((b,), dtype=np.float32)
    for i in range(b):
        out[i] = _dtw_one(pred[i], target[i])
    return out


def compute_frechet(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Discrete Fréchet Distance between two trajectories.
    Using iterative dynamic programming.
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    batch_size = int(pred_np.shape[0])
    total_dist = 0.0

    for i in range(batch_size):
        traj_p = pred_np[i]  # (F, 2)
        traj_q = target_np[i]  # (F, 2)

        total_dist += _frechet_one(traj_p, traj_q)

    return total_dist / max(batch_size, 1)


def compute_dtw(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Dynamic Time Warping (DTW) distance."""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    batch_size = int(pred_np.shape[0])
    total_dist = 0.0

    for i in range(batch_size):
        traj_p = pred_np[i]
        traj_q = target_np[i]

        total_dist += _dtw_one(traj_p, traj_q)

    return total_dist / max(batch_size, 1)


def compute_micro_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute standard trajectory prediction metrics.
    Args:
        pred: (B, F, 2) Predicted positions
        target: (B, F, 2) Ground Truth positions
    """
    metrics: Dict[str, float] = {}

    # 1. ADE / FDE (Vectorized, Fast)
    metrics["ADE"] = compute_ade(pred, target)
    metrics["FDE"] = compute_fde(pred, target)

    # 2. Step-wise MSE (1, 5, 10, ...)
    horizon = int(pred.shape[1])
    steps = [1, 5, 10, 20]
    for step in steps:
        if step <= horizon:
            idx = step - 1
            mse = F.mse_loss(pred[:, idx], target[:, idx]).item()
            metrics[f"MSE_{step}"] = mse

    # 3. Shape Metrics (Loop-based, Slower)
    metrics["Frechet"] = compute_frechet(pred, target)
    metrics["DTW"] = compute_dtw(pred, target)

    return metrics
