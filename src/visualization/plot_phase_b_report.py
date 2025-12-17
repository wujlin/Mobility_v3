"""
Phase B (paper strict, dt-fixed) paper-ready visualizations.

目标：把 dt-fixed(30s) 的评估产物输出成“子刊级”图件（PDF + PNG），用于论文正文/补充材料。

默认输入（可通过参数覆盖）：
  - data/experiments/baseline_b_dt30_eval_b1/metrics.json (+ samples.npz)
  - data/experiments/diff_b_dt30_eval_b1/metrics.json (+ samples.npz)
  - data/experiments/physics_b_dt30_eval_b1/metrics.json (+ samples.npz)

输出（默认）：
  data/experiments/phase_b_report/figures/

注意：
  - Phase B 为 dt-fixed（默认 30s），MSD 横轴可解释为真实时间：tau = k * dt_fixed。
  - 本脚本只依赖 numpy/matplotlib/seaborn；GT 宏观曲线优先从 metrics.json 的 GT_* 字段读取，
    或者通过 --gt_macro_json 传入预计算结果。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from src.visualization.style_config import PALETTE, get_color, set_style


@dataclass(frozen=True)
class ExpArtifacts:
    name: str
    metrics_path: Path
    samples_path: Path


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(f"[OK] saved {pdf}")
    print(f"[OK] saved {png}")


def _accumulate_msd(pos: np.ndarray, msd_sum: np.ndarray, msd_count: np.ndarray) -> None:
    B, T, _ = pos.shape
    for lag in range(1, T):
        diff = pos[:, lag:] - pos[:, :-lag]
        sq = np.sum(diff * diff, axis=-1)
        msd_sum[lag - 1] += float(np.sum(sq))
        msd_count[lag - 1] += int(sq.size)


def _rog(pos: np.ndarray) -> np.ndarray:
    mean_pos = pos.mean(axis=1, keepdims=True)
    diff = pos - mean_pos
    sq = np.sum(diff * diff, axis=-1).mean(axis=1)
    return np.sqrt(sq)


def compute_gt_macro_from_metrics(metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "GT_msd_curve" not in metrics:
        return None
    return {
        "msd_curve": np.array(metrics["GT_msd_curve"], dtype=np.float64),
        "Rog": float(metrics.get("GT_Rog", 0.0)),
    }


def _fit_alpha(msd_curve: np.ndarray, tau: np.ndarray) -> float:
    # log(MSD) = log(a) + alpha*log(tau)
    tau = tau.astype(np.float64, copy=False)
    msd = msd_curve.astype(np.float64, copy=False)
    valid = (tau > 0) & (msd > 0)
    if np.sum(valid) < 2:
        return float("nan")
    coef = np.polyfit(np.log(tau[valid]), np.log(msd[valid]), 1)
    return float(coef[0])


def plot_micro_metrics(models: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    set_style(context="paper", font_scale=1.2)

    metrics = ["ADE", "FDE", "Frechet", "DTW"]
    model_names = list(models.keys())
    colors = [get_color(n) for n in model_names]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for i, met in enumerate(metrics):
        ax = axes[i]
        means = np.array([models[n][f"{met}_mean"] for n in model_names], dtype=np.float64)
        stds = np.array([models[n].get(f"{met}_std", 0.0) for n in model_names], dtype=np.float64)
        best = np.array([models[n].get(f"{met}_best", models[n][f"{met}_mean"]) for n in model_names], dtype=np.float64)

        x = np.arange(len(model_names))
        ax.bar(x, means, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
        ax.scatter(x, best, marker="v", s=45, color="black", zorder=5, label="best-of-K" if i == 0 else None)

        ax.set_title(met)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha="right")
        ax.grid(True, axis="y", ls="--", alpha=0.35)
        for xi, yi in zip(x, means):
            ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    axes[0].set_ylabel("Error (grid units)")
    axes[0].legend(loc="upper right")
    _save_fig(fig, out_dir, "fig1_micro_metrics")


def plot_msd_curve(
    gt: Optional[Dict[str, Any]],
    models: Dict[str, Dict[str, Any]],
    out_dir: Path,
    dt_fixed_seconds: int,
) -> None:
    set_style(context="paper", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)

    # tau: seconds
    def tau_for(curve: np.ndarray) -> np.ndarray:
        return np.arange(1, len(curve) + 1, dtype=np.float64) * float(dt_fixed_seconds)

    if gt is not None:
        curve = np.array(gt["msd_curve"], dtype=np.float64)
        tau = tau_for(curve)
        alpha = _fit_alpha(curve, tau)
        ax.loglog(
            tau,
            curve,
            label=f"GT ($\\alpha={alpha:.2f}$)",
            color=PALETTE["GT"],
            linewidth=2.6,
            marker="o",
            markersize=4,
        )

    for name, m in models.items():
        curve = np.array(m["msd_curve"], dtype=np.float64)
        tau = tau_for(curve)
        alpha = _fit_alpha(curve, tau)
        ax.loglog(
            tau,
            curve,
            label=f"{name} ($\\alpha={alpha:.2f}$)",
            color=get_color(name),
            linewidth=2.2,
            marker="o",
            markersize=4,
        )

    ax.set_xlabel(r"Time lag $\tau$ (seconds)")
    ax.set_ylabel(r"MSD ($grid\_cell^2$)")
    ax.set_title(f"Macroscopic MSD (Phase B: dt-fixed={dt_fixed_seconds}s)")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend()

    _save_fig(fig, out_dir, "fig2_msd_curve")


def _compute_ade_fde_from_samples(pred: np.ndarray, target: np.ndarray) -> Dict[str, np.ndarray]:
    dist = np.linalg.norm(pred - target, axis=-1)  # (N,F)
    ade = dist.mean(axis=1)
    fde = dist[:, -1]
    return {"ADE": ade, "FDE": fde}


def plot_error_cdf(samples: Dict[str, Dict[str, np.ndarray]], out_dir: Path) -> None:
    set_style(context="paper", font_scale=1.2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for j, met in enumerate(["ADE", "FDE"]):
        ax = axes[j]
        for name, s in samples.items():
            vals = np.sort(s[met].astype(np.float64))
            y = (np.arange(1, len(vals) + 1) / len(vals)).astype(np.float64)
            ax.plot(vals, y, label=name, color=get_color(name), linewidth=2.2)
        ax.set_xlabel(f"{met} (grid units)")
        ax.set_ylabel("Empirical CDF")
        ax.grid(True, ls="--", alpha=0.35)
        ax.set_title(f"{met} CDF (saved samples)")
    axes[0].legend()
    _save_fig(fig, out_dir, "fig4_error_cdf")


def plot_rog_boxplot(
    preds_by_model: Dict[str, np.ndarray],
    target: np.ndarray,
    out_dir: Path,
) -> None:
    set_style(context="paper", font_scale=1.2)

    data = []
    labels = []
    colors = []

    gt_r = _rog(target)
    data.append(gt_r)
    labels.append("GT")
    colors.append(PALETTE["GT"])

    for name, pred in preds_by_model.items():
        data.append(_rog(pred))
        labels.append(name)
        colors.append(get_color(name))

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    for k in ["whiskers", "caps", "medians"]:
        for line in bp[k]:
            line.set_color("black")
            line.set_linewidth(1.0)

    ax.set_ylabel("Rog (grid units)")
    ax.set_title("Radius of gyration (saved samples)")
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    _save_fig(fig, out_dir, "fig5_rog_boxplot")


def plot_traj_overlay(
    target: np.ndarray,
    preds_by_model: Dict[str, np.ndarray],
    out_dir: Path,
    num_plots: int = 9,
    seed: int = 0,
) -> None:
    set_style(context="paper", font_scale=1.05)

    rng = np.random.default_rng(int(seed))
    n = int(target.shape[0])
    k = int(min(n, int(num_plots)))
    idx = rng.choice(n, size=k, replace=False) if k > 0 else np.array([], dtype=np.int64)

    cols = 3
    rows = int(np.ceil(k / cols)) if k > 0 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    handles = None
    labels = None

    for i, si in enumerate(idx):
        ax = axes[i]
        gt = target[si]
        ax.plot(gt[:, 1], gt[:, 0], color=PALETTE["GT"], linewidth=2.6, label="GT")

        for name, pred in preds_by_model.items():
            p = pred[si]
            ax.plot(p[:, 1], p[:, 0], color=get_color(name), linewidth=2.0, linestyle="--", label=name)

        ax.scatter(gt[0, 1], gt[0, 0], color="black", s=55, marker="*", zorder=5)
        ax.scatter(gt[-1, 1], gt[-1, 0], color=PALETTE["GT"], s=30, marker="o", zorder=5)

        ax.set_title(f"Sample #{int(si)}")
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.grid(True, ls="--", alpha=0.25)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    for j in range(k, len(axes)):
        axes[j].axis("off")

    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)

    _save_fig(fig, out_dir, "fig3_traj_overlay")

def _step_speeds_from_pos(pos: np.ndarray, start_pos: Optional[np.ndarray] = None) -> np.ndarray:
    """
    pos: (N, F, 2) future positions
    start_pos: (N, 2) optional last observed position

    returns:
      - if start_pos is provided: (N, F) step speeds including the first step (start_pos -> pos[:,0])
      - else: (N, F-1) step speeds computed from successive diffs (approx)

    NOTE: samples.npz does not include the start position, so this omits the first step (start->pos[0]).
    For relative comparisons across models/GT on the same saved samples, this approximation is acceptable.
    """
    if pos.ndim != 3 or pos.shape[-1] != 2:
        raise ValueError(f"invalid pos shape: {pos.shape}")
    if start_pos is not None:
        if start_pos.ndim != 2 or start_pos.shape[-1] != 2 or start_pos.shape[0] != pos.shape[0]:
            raise ValueError(f"invalid start_pos shape: {start_pos.shape}, expected (N,2) with N={pos.shape[0]}")
        first = pos[:, :1, :] - start_pos[:, None, :]
        rest = pos[:, 1:, :] - pos[:, :-1, :] if pos.shape[1] >= 2 else np.zeros((pos.shape[0], 0, 2), dtype=pos.dtype)
        diff = np.concatenate([first, rest], axis=1)
        return np.linalg.norm(diff, axis=-1)

    if pos.shape[1] < 2:
        return np.zeros((pos.shape[0], 0), dtype=np.float64)
    diff = pos[:, 1:, :] - pos[:, :-1, :]
    return np.linalg.norm(diff, axis=-1)


def plot_amplitude_boxplot(
    preds_by_model: Dict[str, np.ndarray],
    target: np.ndarray,
    start_pos: Optional[np.ndarray],
    out_dir: Path,
) -> None:
    set_style(context="paper", font_scale=1.2)

    labels = ["GT"] + list(preds_by_model.keys())
    colors = [PALETTE["GT"]] + [get_color(n) for n in preds_by_model.keys()]

    gt_speed = _step_speeds_from_pos(target, start_pos=start_pos)
    gt_mean_speed = gt_speed.mean(axis=1) if gt_speed.size else np.zeros((target.shape[0],), dtype=np.float64)
    gt_path_len = gt_speed.sum(axis=1) if gt_speed.size else np.zeros((target.shape[0],), dtype=np.float64)

    speed_data = [gt_mean_speed]
    path_data = [gt_path_len]

    for _, pred in preds_by_model.items():
        sp = _step_speeds_from_pos(pred, start_pos=start_pos)
        mean_sp = sp.mean(axis=1) if sp.size else np.zeros((pred.shape[0],), dtype=np.float64)
        path = sp.sum(axis=1) if sp.size else np.zeros((pred.shape[0],), dtype=np.float64)
        speed_data.append(mean_sp)
        path_data.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for ax, data, title, ylabel in [
        (axes[0], speed_data, "Mean step speed (saved samples)", "grid_cell/step (approx)"),
        (axes[1], path_data, "Path length (saved samples)", "grid_cell (approx)"),
    ]:
        bp = ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            widths=0.55,
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.0)

        for k in ["whiskers", "caps", "medians"]:
            for line in bp[k]:
                line.set_color("black")
                line.set_linewidth(1.0)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", ls="--", alpha=0.35)

    _save_fig(fig, out_dir, "fig6_amplitude_boxplot")


def _read_dt_fixed_from_stats(processed_dir: Path) -> Optional[int]:
    stats = processed_dir / "data_stats.json"
    if not stats.exists():
        return None
    try:
        d = _load_json(stats)
        return int(d.get("time_stats", {}).get("dt_fixed"))
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed_dt30", help="Phase B processed dir (dt-fixed)")
    parser.add_argument("--dt_fixed", type=int, default=None, help="override dt_fixed seconds (default: read from data_stats.json or 30)")
    parser.add_argument("--gt_macro_json", type=str, default=None, help="optional precomputed GT macro json (msd_curve/Rog)")

    parser.add_argument("--out_dir", type=str, default="data/experiments/phase_b_report/figures")

    parser.add_argument("--baseline_dir", type=str, default="data/experiments/baseline_b_dt30_eval_b1")
    parser.add_argument("--diff_dir", type=str, default="data/experiments/diff_b_dt30_eval_b1")
    parser.add_argument("--physics_dir", type=str, default="data/experiments/physics_b_dt30_eval_b1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_traj_plots", type=int, default=9)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    processed_dir = Path(args.processed_dir)

    dt_fixed = int(args.dt_fixed) if args.dt_fixed is not None else (_read_dt_fixed_from_stats(processed_dir) or 30)

    exps = [
        ExpArtifacts(
            name="Baseline",
            metrics_path=Path(args.baseline_dir) / "metrics.json",
            samples_path=Path(args.baseline_dir) / "samples.npz",
        ),
        ExpArtifacts(
            name="Diffusion",
            metrics_path=Path(args.diff_dir) / "metrics.json",
            samples_path=Path(args.diff_dir) / "samples.npz",
        ),
        ExpArtifacts(
            name="Physics",
            metrics_path=Path(args.physics_dir) / "metrics.json",
            samples_path=Path(args.physics_dir) / "samples.npz",
        ),
    ]

    # Load metrics
    models: Dict[str, Dict[str, Any]] = {}
    for exp in exps:
        if not exp.metrics_path.exists():
            raise FileNotFoundError(exp.metrics_path)
        models[exp.name] = _load_json(exp.metrics_path)

    # GT macro (explicit json > metrics)
    gt: Optional[Dict[str, Any]] = None
    if args.gt_macro_json:
        gt_path = Path(args.gt_macro_json)
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        gt_raw = _load_json(gt_path)
        if "msd_curve" not in gt_raw or "Rog" not in gt_raw:
            raise ValueError(f"Invalid gt_macro_json format: {gt_path}")
        gt = {"msd_curve": np.array(gt_raw["msd_curve"], dtype=np.float64), "Rog": float(gt_raw["Rog"])}
    if gt is None:
        for exp in exps:
            gt = compute_gt_macro_from_metrics(models[exp.name])
            if gt is not None:
                break

    # Load sample trajectories (for qualitative + CDF/boxplot)
    preds_by_model: Dict[str, np.ndarray] = {}
    target = None
    start_pos = None
    for exp in exps:
        if not exp.samples_path.exists():
            raise FileNotFoundError(exp.samples_path)
        d = np.load(exp.samples_path)
        preds = d["preds"].astype(np.float64)
        tgt = d["targets"].astype(np.float64)
        sp = d["start_pos"].astype(np.float64) if "start_pos" in d.files else None
        preds_by_model[exp.name] = preds
        target = tgt if target is None else target
        if target is not None:
            if np.max(np.abs(tgt - target)) > 1e-9:
                raise ValueError(f"targets mismatch between samples files: {exp.samples_path}")
        if sp is not None:
            start_pos = sp if start_pos is None else start_pos
            if start_pos is not None and np.max(np.abs(sp - start_pos)) > 1e-9:
                raise ValueError(f"start_pos mismatch between samples files: {exp.samples_path}")

    assert target is not None

    plot_micro_metrics(models=models, out_dir=out_dir)
    plot_msd_curve(gt=gt, models=models, out_dir=out_dir, dt_fixed_seconds=dt_fixed)
    plot_traj_overlay(target=target, preds_by_model=preds_by_model, out_dir=out_dir, num_plots=int(args.num_traj_plots), seed=int(args.seed))

    samples_metrics = {name: _compute_ade_fde_from_samples(pred, target) for name, pred in preds_by_model.items()}
    plot_error_cdf(samples=samples_metrics, out_dir=out_dir)
    plot_rog_boxplot(preds_by_model=preds_by_model, target=target, out_dir=out_dir)
    plot_amplitude_boxplot(preds_by_model=preds_by_model, target=target, start_pos=start_pos, out_dir=out_dir)

    summary = {
        "dt_fixed_seconds": int(dt_fixed),
        "models": {k: {mk: v for mk, v in m.items() if mk != "msd_curve"} for k, m in models.items()},
        "gt_macro": None if gt is None else {"Rog": float(gt.get("Rog", 0.0)), "msd_curve": np.array(gt.get("msd_curve")).tolist()},
        "note": "Phase B is dt-fixed; interpret tau as real seconds. Saved-sample plots use k=0 sample only.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase_b_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved {out_dir / 'phase_b_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
