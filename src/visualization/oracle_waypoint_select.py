"""
Oracle Waypoint selection (post-hoc, no re-sampling).

Goal (KISS):
  Given a saved samples.npz with preds_k (N,K,F,2) and targets (N,F,2),
  select, for each condition n, the sample k that best matches a GT "waypoint"
  at some step t. This answers:

    "Does the model's support already contain detour-like modes,
     but our usual sampling/plotting misses them?"

This is a cleaner diagnosis than re-conditioning the model with a waypoint,
because it stays in-distribution (same conditioning & sampler).

Outputs:
  - <out_dir>/<stem>.npz: preds (N,F,2) = oracle-selected, plus targets/start_pos
  - <out_dir>/<stem>.json: metrics (JSD+DCV) for baseline vs oracle

No SciPy dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.evaluation.distribution_metrics import compute_distribution_metrics, compute_violation_metrics


def _resolve_wp_idx_fixed(*, pred_len: int, wp_frac: float, wp_idx: int | None) -> int:
    if int(pred_len) < 3:
        raise ValueError(f"pred_len must be >=3, got {pred_len}")
    if wp_idx is not None:
        idx = int(wp_idx)
    else:
        idx = int(round(float(wp_frac) * float(pred_len - 1)))
    # keep away from endpoints (needs a turn angle and meaningful waypoint)
    return max(1, min(idx, int(pred_len) - 2))


def _turn_angle(pos: np.ndarray, *, min_speed: float) -> np.ndarray:
    """
    pos: (N, F, 2)
    returns: (N, F-2) turn angles in [0,pi], NaN where invalid (near-static).
    """
    pos = np.asarray(pos, dtype=np.float32)
    disp = pos[:, 1:] - pos[:, :-1]  # (N,F-1,2)
    if disp.shape[1] < 2:
        return np.zeros((disp.shape[0], 0), dtype=np.float32)
    v1 = disp[:, :-1]
    v2 = disp[:, 1:]
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    valid = (n1 > float(min_speed)) & (n2 > float(min_speed))
    dot = np.sum(v1 * v2, axis=-1)
    cos = dot / (n1 * n2 + 1e-8)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.arccos(cos).astype(np.float32)
    ang[~valid] = np.nan
    return ang


def _line_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Perpendicular distance from points to the infinite line through a->b.
    points: (F,2), a/b: (2,)
    returns: (F,)
    """
    ab = (b - a).astype(np.float64)
    denom = float(np.linalg.norm(ab)) + 1e-12
    ap = (points - a).astype(np.float64)
    cross = np.abs(ab[0] * ap[:, 1] - ab[1] * ap[:, 0])
    return (cross / denom).astype(np.float64)


def _resolve_wp_idx_per_sample(
    targets: np.ndarray,
    start_pos: np.ndarray,
    *,
    mode: str,
    min_speed: float,
) -> np.ndarray:
    """
    Resolve waypoint index per sample (N,) in [1, F-2].
    """
    mode = str(mode)
    N, F, _ = targets.shape
    if F < 3:
        raise ValueError(f"pred_len must be >=3, got F={F}")

    if mode == "fixed":
        raise RuntimeError("internal error: fixed mode should be handled separately")

    if mode == "max_turn":
        ang = _turn_angle(targets, min_speed=float(min_speed))  # (N,F-2)
        ang2 = np.nan_to_num(ang, nan=-1.0)
        idx0 = np.argmax(ang2, axis=1)  # (N,), in [0, F-3]
        # angle at idx0 corresponds to position idx0+1
        wp = idx0 + 1
        return np.clip(wp, 1, F - 2).astype(np.int64)

    if mode == "max_dev":
        wp = np.zeros((N,), dtype=np.int64)
        for i in range(N):
            a = start_pos[i]
            b = targets[i, -1]
            d = _line_distance(targets[i], a, b)  # (F,)
            # exclude endpoints
            d[0] = -1.0
            d[-1] = -1.0
            wp[i] = int(np.argmax(d))
        return np.clip(wp, 1, F - 2).astype(np.int64)

    raise ValueError(f"Unknown --waypoint_mode: {mode} (expected: fixed|max_turn|max_dev)")


def _oracle_select(
    preds_k: np.ndarray,
    targets: np.ndarray,
    wp_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        preds_k: (N,K,F,2)
        targets: (N,F,2)
        wp_idx:  (N,) waypoint position indices (0-based)
    Returns:
        selected: (N,F,2)
        best_k:   (N,) selected sample index
    """
    preds_k = np.asarray(preds_k, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    wp_idx = np.asarray(wp_idx, dtype=np.int64)
    if preds_k.ndim != 4 or preds_k.shape[-1] != 2:
        raise ValueError(f"Expected preds_k (N,K,F,2), got {preds_k.shape}")
    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError(f"Expected targets (N,F,2), got {targets.shape}")
    if wp_idx.ndim != 1 or wp_idx.shape[0] != targets.shape[0]:
        raise ValueError(f"Expected wp_idx (N,), got {wp_idx.shape} with N={targets.shape[0]}")

    N, K, F, _ = preds_k.shape
    if targets.shape[1] != F:
        raise ValueError(f"F mismatch: preds_k F={F} vs targets F={targets.shape[1]}")
    if np.any(wp_idx < 0) or np.any(wp_idx >= F):
        raise ValueError(f"wp_idx out of range [0,{F-1}]")

    n_idx = np.arange(N)[:, None]
    k_idx = np.arange(K)[None, :]
    t_idx = wp_idx[:, None]
    pred_wp = preds_k[n_idx, k_idx, t_idx, :]  # (N,K,2)
    gt_wp = targets[np.arange(N), wp_idx, :][:, None, :]  # (N,1,2)
    dist2 = np.sum((pred_wp - gt_wp) ** 2, axis=-1)  # (N,K)
    best_k = np.argmin(dist2, axis=1).astype(np.int64)
    selected = preds_k[np.arange(N), best_k, :, :].astype(np.float32, copy=False)
    return selected, best_k


def main() -> int:
    parser = argparse.ArgumentParser(description="Oracle waypoint selection from saved preds_k (no resampling)")
    parser.add_argument("--input_npz", type=str, required=True, help="samples.npz with preds_k and targets")
    parser.add_argument("--out_dir", type=str, default=None, help="if set, save selected samples + json")
    parser.add_argument("--stem", type=str, default="samples_oracle_wp")
    parser.add_argument("--k_max", type=int, default=10, help="cap K for selection (<=0 means all)")
    parser.add_argument(
        "--waypoint_mode",
        type=str,
        choices=["fixed", "max_turn", "max_dev"],
        default="fixed",
        help="how to choose GT waypoint index",
    )
    parser.add_argument("--waypoint_frac", type=float, default=0.5, help="fixed mode: fraction of horizon (0..1)")
    parser.add_argument("--waypoint_idx", type=int, default=None, help="fixed mode: explicit idx (0-based, 1..F-2 recommended)")
    parser.add_argument("--turn_min_speed", type=float, default=0.1, help="for max_turn mode: speed filter (grid/step)")
    parser.add_argument("--dcv_speed_pctl", type=float, default=99.5)
    parser.add_argument("--dcv_accel_pctl", type=float, default=99.5)
    args = parser.parse_args()

    path = Path(args.input_npz)
    data = np.load(str(path))
    if "preds_k" not in data.files:
        raise RuntimeError(f"{path} has no preds_k; run evaluate.py with --save_all_k first.")
    preds_k = np.asarray(data["preds_k"], dtype=np.float32)  # (N,K,F,2)
    targets = np.asarray(data["targets"], dtype=np.float32)  # (N,F,2)
    start_pos = np.asarray(data["start_pos"], dtype=np.float32) if "start_pos" in data.files else None

    if start_pos is None:
        # Fallback: use the first GT point as a proxy (only for max_dev). Not ideal but keeps script usable.
        start_pos = targets[:, 0].astype(np.float32, copy=False)

    if int(args.k_max) > 0:
        preds_k = preds_k[:, : int(args.k_max)]

    N, K, F, _ = preds_k.shape
    if args.waypoint_mode == "fixed":
        idx = _resolve_wp_idx_fixed(pred_len=F, wp_frac=float(args.waypoint_frac), wp_idx=args.waypoint_idx)
        wp_idx = np.full((N,), int(idx), dtype=np.int64)
    else:
        wp_idx = _resolve_wp_idx_per_sample(
            targets,
            start_pos,
            mode=str(args.waypoint_mode),
            min_speed=float(args.turn_min_speed),
        )

    selected, best_k = _oracle_select(preds_k, targets, wp_idx)

    # Baseline: use k=0 (backward-compatible `preds` if exists, else preds_k[:,0]).
    if "preds" in data.files:
        baseline = np.asarray(data["preds"], dtype=np.float32)
    else:
        baseline = preds_k[:, 0]

    baseline_jsd = compute_distribution_metrics(
        baseline,
        targets,
        speed_bins=120,
        accel_bins=120,
        turn_bins=60,
        turn_min_speed=float(args.turn_min_speed),
    )
    oracle_jsd = compute_distribution_metrics(
        selected,
        targets,
        speed_bins=120,
        accel_bins=120,
        turn_bins=60,
        turn_min_speed=float(args.turn_min_speed),
    )

    baseline_dcv = compute_violation_metrics(
        baseline,
        targets,
        speed_pctl=float(args.dcv_speed_pctl),
        accel_pctl=float(args.dcv_accel_pctl),
    )
    oracle_dcv = compute_violation_metrics(
        selected,
        targets,
        speed_pctl=float(args.dcv_speed_pctl),
        accel_pctl=float(args.dcv_accel_pctl),
    )

    def _pack(jsd: Dict[str, float], dcv: Dict[str, float]) -> Dict[str, float]:
        out = dict(jsd)
        out.update({k: float(v) for k, v in dcv.items() if k.startswith("Vio_") or k.startswith("Thresh_")})
        return out

    report = {
        "input_npz": str(path),
        "N": int(N),
        "K_used": int(K),
        "waypoint_mode": str(args.waypoint_mode),
        "waypoint_frac": float(args.waypoint_frac),
        "waypoint_idx_fixed": (int(_resolve_wp_idx_fixed(pred_len=F, wp_frac=float(args.waypoint_frac), wp_idx=args.waypoint_idx)) if args.waypoint_mode == "fixed" else None),
        "turn_min_speed": float(args.turn_min_speed),
        "dcv_speed_pctl": float(args.dcv_speed_pctl),
        "dcv_accel_pctl": float(args.dcv_accel_pctl),
        "metrics": {
            "baseline_k0": _pack(baseline_jsd, baseline_dcv),
            "oracle_selected": _pack(oracle_jsd, oracle_dcv),
        },
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npz = out_dir / f"{args.stem}.npz"
        out_json = out_dir / f"{args.stem}.json"
        np.savez(
            out_npz,
            preds=selected.astype(np.float32, copy=False),
            targets=targets.astype(np.float32, copy=False),
            start_pos=start_pos.astype(np.float32, copy=False),
            oracle_best_k=best_k.astype(np.int64, copy=False),
            oracle_wp_idx=wp_idx.astype(np.int64, copy=False),
        )
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] saved {out_npz}")
        print(f"[OK] saved {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

