"""
Select "interesting" cases from samples.npz for qualitative visualization.

Motivation:
- Random cases often look flat (little branching).
- For journal-quality figures, we want cases where the generative model exhibits
  meaningful multi-modality (e.g., route branching near intersections).

This script ranks conditions by a simple diversity proxy computed from preds_k:
- endpoint_spread: mean distance to the mean endpoint (K samples)

Input:
- samples.npz saved by evaluate.py with --save_all_k, containing preds_k (N,K,F,2).
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path


def endpoint_spread(endpoints: np.ndarray) -> np.ndarray:
    """
    endpoints: (N, K, 2)
    return: (N,) mean radius to mean endpoint
    """
    mean = endpoints.mean(axis=1, keepdims=True)
    d = np.linalg.norm(endpoints - mean, axis=-1)  # (N,K)
    return d.mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, required=True, help="Path to samples.npz (must contain preds_k)")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--metric", type=str, choices=["endpoint_spread"], default="endpoint_spread")
    parser.add_argument("--k_use", type=int, default=None, help="use only first k samples in preds_k")
    args = parser.parse_args()

    data = np.load(Path(args.samples))
    if "preds_k" not in data.files:
        raise ValueError("samples.npz does not contain preds_k. Re-run evaluate with --save_all_k.")
    preds_k = np.asarray(data["preds_k"], dtype=np.float64)  # (N,K,F,2)
    N, K, F, D = preds_k.shape
    k_use = int(args.k_use) if args.k_use is not None else int(K)
    k_use = max(1, min(k_use, int(K)))
    preds_k = preds_k[:, :k_use]

    endpoints = preds_k[:, :, -1, :]  # (N,k,2)
    if args.metric == "endpoint_spread":
        score = endpoint_spread(endpoints)
    else:  # pragma: no cover
        raise ValueError(f"Unknown metric: {args.metric}")

    top_k = max(1, min(int(args.top_k), int(N)))
    idx = np.argsort(-score)[:top_k]

    print("rank,case_idx,score")
    for rank, i in enumerate(idx, start=1):
        print(f"{rank},{int(i)},{float(score[i]):.6f}")


if __name__ == "__main__":
    main()

