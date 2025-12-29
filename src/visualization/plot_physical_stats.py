"""
Physical-statistics plots for Phase B.

We use these figures to argue about **valid simulation** beyond pointwise errors:
- Speed distribution (micro texture)
- Turn-angle distribution (path tortuosity)
- MSD curve (macro mobility / directional persistence)

Inputs:
  --inputs "Label:/path/to/samples.npz" (repeatable)
    - samples.npz should contain:
        preds:   (N, F, 2) in grid space
        targets: (N, F, 2) in grid space (optional but recommended for GT)
    - if contains preds_k: (N, K, F, 2), can use --use_all_k to visualize multimodality
  --inputs "Label:/path/to/metrics.json" (repeatable)
    - optional: provides msd_curve / GT_msd_curve

This script is SciPy-free (KISS / portability).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.distribution_metrics import compute_distribution_metrics, compute_violation_metrics
from src.visualization.style_config import get_color, set_style


def compute_speed_and_angle(pos_seq: np.ndarray, *, turn_min_speed: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        pos_seq: (N, F, 2) positions in grid space.
    Returns:
        speeds: (N*(F-1),) step speeds (grid/step)
        angles: (N*(F-2),) unsigned turn angles in radians (0..pi)
    """
    pos_seq = np.asarray(pos_seq, dtype=np.float32)
    if pos_seq.ndim != 3 or pos_seq.shape[-1] != 2:
        raise ValueError(f"Expected pos_seq (N,F,2), got {pos_seq.shape}")

    vel = pos_seq[:, 1:] - pos_seq[:, :-1]  # (N, F-1, 2)
    speeds = np.linalg.norm(vel, axis=-1).reshape(-1)

    if vel.shape[1] < 2:
        return speeds.astype(np.float32, copy=False), np.array([], dtype=np.float32)

    v1 = vel[:, :-1]
    v2 = vel[:, 1:]
    dot = np.sum(v1 * v2, axis=-1)
    norm1 = np.linalg.norm(v1, axis=-1)
    norm2 = np.linalg.norm(v2, axis=-1)
    cos_theta = dot / (norm1 * norm2 + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles = np.arccos(cos_theta)  # (N, F-2)
    thr = float(turn_min_speed)
    if thr > 0:
        # Filter out near-static steps where heading is ill-defined.
        mask = (norm1 > thr) & (norm2 > thr)
        angles = angles[mask]
    angles = angles.reshape(-1)
    return speeds.astype(np.float32, copy=False), angles.astype(np.float32, copy=False)


def compute_msd_curve(pos_seq: np.ndarray) -> np.ndarray:
    """
    MSD(τ) over τ=1..F-1 with small-F loops (F is typically 12/16 in this project).
    Args:
        pos_seq: (N, F, 2)
    Returns:
        msd: (F-1,)
    """
    pos_seq = np.asarray(pos_seq, dtype=np.float32)
    if pos_seq.ndim != 3 or pos_seq.shape[-1] != 2:
        raise ValueError(f"Expected pos_seq (N,F,2), got {pos_seq.shape}")

    N, F, _ = pos_seq.shape
    msd = np.zeros((F - 1,), dtype=np.float64)
    cnt = np.zeros((F - 1,), dtype=np.int64)
    for lag in range(1, F):
        diff = pos_seq[:, lag:] - pos_seq[:, :-lag]
        sq = np.sum(diff * diff, axis=-1)
        msd[lag - 1] += float(np.sum(sq))
        cnt[lag - 1] += int(sq.size)
    return (msd / np.maximum(cnt, 1)).astype(np.float64)


def _gaussian_smooth_1d_fft(y: np.ndarray, sigma_bins: float) -> np.ndarray:
    """
    1D Gaussian smoothing (no SciPy) via frequency-domain multiplication.
    sigma_bins is in histogram-bin units.
    """
    sigma = float(sigma_bins)
    if sigma <= 0:
        return np.asarray(y, dtype=np.float64)

    y = np.asarray(y, dtype=np.float64)
    n = int(y.shape[0])
    f = np.fft.rfftfreq(n)
    kernel_ft = np.exp(-2.0 * (np.pi**2) * (sigma**2) * (f**2))
    out = np.fft.irfft(np.fft.rfft(y) * kernel_ft, n=n)
    return np.clip(out, 0.0, None)


def _parse_inputs(inputs: list[str]) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for raw in inputs:
        if ":" not in raw:
            raise ValueError(f"Invalid --inputs item '{raw}'. Expected 'Label:Path'.")
        label, path = raw.split(":", 1)  # allow Windows drive paths
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid --inputs item '{raw}'. Expected 'Label:Path'.")
        files[label] = path
    return files


def _load_npz_preds(path: Path, *, use_all_k: bool, k_max: int) -> Tuple[np.ndarray, np.ndarray | None]:
    data = np.load(str(path))
    if bool(use_all_k) and "preds_k" in data.files:
        preds_k = np.asarray(data["preds_k"], dtype=np.float32)  # (N,K,F,2)
        if int(k_max) > 0:
            preds_k = preds_k[:, : int(k_max)]
        preds = preds_k.reshape(-1, preds_k.shape[2], preds_k.shape[3])  # (N*K,F,2)
    else:
        preds = np.asarray(data["preds"], dtype=np.float32)  # (N,F,2)
    targets = np.asarray(data["targets"], dtype=np.float32) if "targets" in data.files else None
    return preds, targets


def plot_distributions(
    files_dict: Dict[str, str],
    output_dir: Path,
    *,
    use_all_k: bool,
    k_max: int,
    stride: int,
    speed_bins: int,
    speed_sigma: float,
    accel_bins: int,
    angle_bins: int,
    angle_sigma: float,
    turn_min_speed: float,
    dcv_speed_pctl: float,
    dcv_accel_pctl: float,
    save_metrics: bool,
    stem: str,
) -> None:
    set_style(context="paper")

    fig, (ax_speed, ax_angle, ax_msd) = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    data_store: Dict[str, Dict[str, np.ndarray]] = {}
    msd_store: Dict[str, np.ndarray] = {}
    pos_store: Dict[str, np.ndarray] = {}
    gt_done = False

    for name, filepath in files_dict.items():
        p = Path(filepath)
        if p.suffix == ".npz":
            preds, targets = _load_npz_preds(p, use_all_k=use_all_k, k_max=k_max)
            pos_store[name] = preds
            s_pred, a_pred = compute_speed_and_angle(preds, turn_min_speed=float(turn_min_speed))
            data_store[name] = {"speed": s_pred, "angle": a_pred}
            msd_store[name] = compute_msd_curve(preds)

            if (not gt_done) and targets is not None:
                pos_store["GT"] = targets
                s_gt, a_gt = compute_speed_and_angle(targets, turn_min_speed=float(turn_min_speed))
                data_store["GT"] = {"speed": s_gt, "angle": a_gt}
                msd_store["GT"] = compute_msd_curve(targets)
                gt_done = True

        elif p.name == "metrics.json":
            metrics = json.loads(p.read_text(encoding="utf-8"))
            if "msd_curve" in metrics:
                msd_store[name] = np.asarray(metrics["msd_curve"], dtype=np.float64)
            if "GT_msd_curve" in metrics:
                msd_store.setdefault("GT", np.asarray(metrics["GT_msd_curve"], dtype=np.float64))
        else:
            raise ValueError(f"Unsupported input file: {p} (expected .npz or metrics.json)")

    # ---- Speed distribution (shared x-range) ----
    speed_vals_all = []
    for v in data_store.values():
        vals = v.get("speed")
        if vals is not None and vals.size:
            speed_vals_all.append(vals)
    speed_x_max = float(np.percentile(np.concatenate(speed_vals_all, axis=0), 99)) if speed_vals_all else 1.0

    def _iter_ordered(names: list[str]) -> list[str]:
        out = []
        if "GT" in names:
            out.append("GT")
        for n in files_dict.keys():
            if n in names and n not in out:
                out.append(n)
        for n in names:
            if n not in out:
                out.append(n)
        return out

    for name in _iter_ordered(list(data_store.keys())):
        vals = data_store[name]["speed"]
        if int(stride) > 1:
            vals = vals[:: int(stride)]
        if vals.size == 0:
            continue
        counts, edges = np.histogram(vals, bins=int(speed_bins), range=(0.0, speed_x_max), density=True)
        counts_s = _gaussian_smooth_1d_fft(counts, sigma_bins=float(speed_sigma))
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_speed.plot(centers, counts_s, label=name, color=get_color(name), lw=2)
    ax_speed.set_xlabel("Speed (grid/step)")
    ax_speed.set_ylabel("Density")
    ax_speed.set_title("Speed Distribution")
    ax_speed.legend(frameon=False)

    # ---- Turn-angle distribution (shared bins 0..pi) ----
    for name in _iter_ordered([k for k in data_store.keys() if k != "GT"] + (["GT"] if "GT" in data_store else [])):
        vals = data_store[name]["angle"]
        if int(stride) > 1:
            vals = vals[:: int(stride)]
        if vals.size == 0:
            continue
        counts, edges = np.histogram(vals, bins=int(angle_bins), range=(0.0, float(np.pi)), density=True)
        counts_s = _gaussian_smooth_1d_fft(counts, sigma_bins=float(angle_sigma))
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_angle.plot(centers, counts_s, label=name, color=get_color(name), lw=2)
    ax_angle.set_xlabel("Turn angle (rad)")
    ax_angle.set_ylabel("Density")
    ax_angle.set_title("Turn Angle Distribution")

    # ---- MSD curve (log-log) ----
    names = _iter_ordered([n for n in ["GT", *files_dict.keys()] if n in msd_store])
    for name in names:
        msd = msd_store[name]
        tau = np.arange(1, len(msd) + 1)
        ax_msd.loglog(tau, msd, label=name, color=get_color(name), lw=2, marker=".")
    ax_msd.set_xlabel(r"Time lag $\tau$")
    ax_msd.set_ylabel("MSD")
    ax_msd.set_title("Macroscopic Diffusion (MSD)")
    ax_msd.legend(frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=300)
    print(f"[OK] saved {out_path}")

    # ---- Standard validity metrics (JSD + DCV) ----
    if "GT" not in pos_store:
        print("[WARN] No GT targets found in inputs; skip JSD/DCV metrics.")
        return

    gt_pos = pos_store["GT"]

    def _iter_ordered_for_metrics(names: list[str]) -> list[str]:
        out = []
        for n in files_dict.keys():
            if n in names and n not in out:
                out.append(n)
        for n in names:
            if n not in out:
                out.append(n)
        return out

    validity: Dict[str, Dict[str, float]] = {}
    for name in _iter_ordered_for_metrics([k for k in pos_store.keys() if k != "GT"]):
        pred_pos = pos_store.get(name)
        if pred_pos is None:
            continue
        jsd = compute_distribution_metrics(
            pred_pos,
            gt_pos,
            dt_s=1.0,  # keep per-step units (grid/step) by default
            meters_per_cell=None,
            speed_bins=int(speed_bins),
            accel_bins=int(accel_bins),
            turn_bins=int(angle_bins),
            turn_min_speed=float(turn_min_speed),
        )
        dcv = compute_violation_metrics(
            pred_pos,
            gt_pos,
            dt_s=1.0,
            meters_per_cell=None,
            speed_pctl=float(dcv_speed_pctl),
            accel_pctl=float(dcv_accel_pctl),
        )
        validity[name] = {**jsd, **dcv}

    if validity:
        print("[OK] Validity metrics (vs GT):")
        for name, m in validity.items():
            print(
                f"  - {name}: "
                f"JSD_Turn={m['JSD_TurnAngle']:.4f}, JSD_Speed={m['JSD_Speed']:.4f}, JSD_Accel={m['JSD_Accel']:.4f}; "
                f"DCV_speed={m['Vio_Speed_Rate']:.4%}, DCV_accel={m['Vio_Accel_Rate']:.4%}"
            )

    if bool(save_metrics):
        payload = {
            "config": {
                "use_all_k": bool(use_all_k),
                "k_max": int(k_max),
                "speed_bins": int(speed_bins),
                "accel_bins": int(accel_bins),
                "angle_bins": int(angle_bins),
                "turn_min_speed": float(turn_min_speed),
                "dcv_speed_pctl": float(dcv_speed_pctl),
                "dcv_accel_pctl": float(dcv_accel_pctl),
                "units": "grid/step (dt_s=1.0, meters_per_cell=None)",
            },
            "metrics": validity,
        }
        metrics_path = output_dir / f"{stem}_validity.json"
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[OK] saved {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Label:Path (samples.npz or metrics.json)")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--stem", type=str, default="fig_physical_stats")
    parser.add_argument("--use_all_k", action="store_true", help="Flatten preds_k (N,K,F,2) into (N*K,F,2).")
    parser.add_argument("--k_max", type=int, default=0, help="If use_all_k: cap K (e.g., 10). 0 means all.")
    parser.add_argument("--stride", type=int, default=5, help="Downsample speed/angle arrays for speed.")
    parser.add_argument("--speed_bins", type=int, default=120)
    parser.add_argument("--speed_sigma", type=float, default=1.2, help="Gaussian smooth sigma (bin units).")
    parser.add_argument("--accel_bins", type=int, default=120, help="Bins for acceleration JSD (no accel plot yet).")
    parser.add_argument("--angle_bins", type=int, default=60)
    parser.add_argument("--angle_sigma", type=float, default=1.0, help="Gaussian smooth sigma (bin units).")
    parser.add_argument(
        "--turn_min_speed",
        type=float,
        default=0.1,
        help="Turn-angle speed filter threshold (grid/step) to drop near-static steps.",
    )
    parser.add_argument("--dcv_speed_pctl", type=float, default=99.5, help="DCV speed threshold percentile from GT.")
    parser.add_argument("--dcv_accel_pctl", type=float, default=99.5, help="DCV accel threshold percentile from GT.")
    parser.add_argument("--save_metrics", action="store_true", help="Also save <stem>_validity.json (JSD+DCV).")
    args = parser.parse_args()

    plot_distributions(
        _parse_inputs(list(args.inputs)),
        output_dir=Path(args.output_dir),
        use_all_k=bool(args.use_all_k),
        k_max=int(args.k_max),
        stride=int(args.stride),
        speed_bins=int(args.speed_bins),
        speed_sigma=float(args.speed_sigma),
        accel_bins=int(args.accel_bins),
        angle_bins=int(args.angle_bins),
        angle_sigma=float(args.angle_sigma),
        turn_min_speed=float(args.turn_min_speed),
        dcv_speed_pctl=float(args.dcv_speed_pctl),
        dcv_accel_pctl=float(args.dcv_accel_pctl),
        save_metrics=bool(args.save_metrics),
        stem=str(args.stem),
    )


if __name__ == "__main__":
    main()








