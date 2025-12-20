"""
Phase A (fast validation) paper-ready visualizations.

目标：用一份脚本把 Phase A 的 quick eval 结果输出成“子刊级”图件（PDF + PNG）。

默认输入（可通过参数覆盖）：
  - data/experiments/baseline_a_full_eval_quick/metrics.json (+ samples.npz)
  - data/experiments/diff_a_full_eval_quick/metrics.json (+ samples.npz)
  - data/experiments/physics_a_full_eval_quick/metrics.json (+ samples.npz)

输出（默认）：
  data/experiments/phase_a_report/figures/

注意：
  - Phase A 为 step-based（dt 不恒定），宏观曲线只应解释为 “lag steps”。
  - GT 的 MSD/Rog 对照：优先使用 metrics.json 中的 GT_* 字段（若已由新版 evaluate.py 生成）；
    否则从 dataset 子集（batch_size/max_batches）在线计算（更慢但更严格）。
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


def _integrate_positions(start_pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    # start_pos: (B,2), vel: (B,F,2) step displacement
    return start_pos[:, None, :] + np.cumsum(vel, axis=1)


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


def compute_gt_macro_from_dataset(
    processed_dir: Path,
    split: str,
    obs_len: int,
    pred_len: int,
    batch_size: int,
    max_batches: int,
    num_workers: int = 0,
) -> Dict[str, Any]:
    try:
        import torch  # type: ignore
        from torch.utils.data import DataLoader  # type: ignore

        from src.data.datasets_diffusion import DiffusionDataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "当前环境缺少 torch/h5py，无法从 dataset 在线计算 GT 宏观指标；"
            "请改用新版 evaluate.py 生成带 GT_* 的 metrics.json，或提供 --gt_macro_json。"
        ) from e

    h5_path = processed_dir / "trajectories" / "shenzhen_trajectories.h5"
    splits_dir = processed_dir / "splits"
    ids = np.load(splits_dir / f"{split}_ids.npy").astype(np.int64)

    ds = DiffusionDataset(
        str(h5_path),
        obs_len=int(obs_len),
        pred_len=int(pred_len),
        traj_ids=ids,
    )
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))

    norm = ds.normalizer
    msd_sum = np.zeros((pred_len - 1,), dtype=np.float64)
    msd_count = np.zeros((pred_len - 1,), dtype=np.int64)
    rog_sum = 0.0
    rog_count = 0
    total_n = 0

    for bi, batch in enumerate(dl):
        if max_batches is not None and bi >= int(max_batches):
            break

        obs = batch["obs"].numpy()
        action = batch["action"].numpy()

        start_pos = norm.denormalize_pos(obs[:, -1, :2])
        vel = norm.denormalize_vel(action)
        pos = _integrate_positions(start_pos, vel)

        _accumulate_msd(pos, msd_sum, msd_count)
        r = _rog(pos)
        rog_sum += float(np.sum(r))
        rog_count += int(r.shape[0])
        total_n += int(pos.shape[0])

    msd_curve = (msd_sum / np.maximum(msd_count, 1)).astype(np.float64)
    gt = {
        "num_conditions": int(total_n),
        "msd_curve": msd_curve,
        "Rog": float(rog_sum / max(rog_count, 1)),
    }
    return gt


def _fit_alpha(msd_curve: np.ndarray) -> float:
    # log(MSD) = log(a) + alpha*log(t)
    t = np.arange(1, len(msd_curve) + 1, dtype=np.float64)
    msd = msd_curve.astype(np.float64)
    valid = msd > 0
    if np.sum(valid) < 2:
        return float("nan")
    coef = np.polyfit(np.log(t[valid]), np.log(msd[valid]), 1)
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

        # annotate mean
        for xi, yi in zip(x, means):
            ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    axes[0].set_ylabel("Error (grid units)")
    axes[0].legend(loc="upper right")

    _save_fig(fig, out_dir, "fig1_micro_metrics")


def plot_msd_curve(
    gt: Optional[Dict[str, Any]],
    models: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> None:
    set_style(context="paper", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)

    if gt is not None:
        curve = np.array(gt["msd_curve"], dtype=np.float64)
        t = np.arange(1, len(curve) + 1)
        alpha = _fit_alpha(curve)
        ax.loglog(
            t,
            curve,
            label=f"GT ($\\alpha={alpha:.2f}$)",
            color=PALETTE["GT"],
            linewidth=2.6,
            marker="o",
            markersize=4,
        )

    for name, m in models.items():
        curve = np.array(m["msd_curve"], dtype=np.float64)
        t = np.arange(1, len(curve) + 1)
        alpha = _fit_alpha(curve)
        ax.loglog(
            t,
            curve,
            label=f"{name} ($\\alpha={alpha:.2f}$)",
            color=get_color(name),
            linewidth=2.2,
            marker="o",
            markersize=4,
        )

    ax.set_xlabel(r"Time lag $\tau$ (steps)")
    ax.set_ylabel(r"MSD ($grid\_cell^2$)")
    ax.set_title("Macroscopic MSD (Phase A: step-based)")
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
    # Use tight_layout with a reserved top margin for the figure-level legend.
    # (constrained_layout does not reliably account for fig.legend and can lead to overlaps.)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), constrained_layout=False)
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

        # start/end markers (GT)
        ax.scatter(gt[0, 1], gt[0, 0], color="black", s=55, marker="*", zorder=5)
        ax.scatter(gt[-1, 1], gt[-1, 0], color=PALETTE["GT"], s=30, marker="o", zorder=5)

        ax.set_title(f"Sample #{int(si)}", pad=2)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.grid(True, ls="--", alpha=0.25)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    # hide unused
    for j in range(k, len(axes)):
        axes[j].axis("off")

    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout()

    _save_fig(fig, out_dir, "fig3_traj_overlay")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Phase A processed dir (for GT macro)")
    parser.add_argument("--gt_macro_json", type=str, default=None, help="optional precomputed GT macro json (msd_curve/Rog)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for GT macro computation")
    parser.add_argument("--out_dir", type=str, default="data/experiments/phase_a_report/figures")

    parser.add_argument("--baseline_dir", type=str, default="data/experiments/baseline_a_full_eval_quick")
    parser.add_argument("--diff_dir", type=str, default="data/experiments/diff_a_full_eval_quick")
    parser.add_argument("--physics_dir", type=str, default="data/experiments/physics_a_full_eval_quick")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_traj_plots", type=int, default=9)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    processed_dir = Path(args.processed_dir)

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

    # GT macro (prefer explicit json > metrics > dataset)
    gt: Optional[Dict[str, Any]] = None
    if args.gt_macro_json:
        gt_path = Path(args.gt_macro_json)
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        gt_raw = _load_json(gt_path)
        if "msd_curve" not in gt_raw or "Rog" not in gt_raw:
            raise ValueError(f"Invalid gt_macro_json format: {gt_path}")
        gt = {"msd_curve": np.array(gt_raw["msd_curve"], dtype=np.float64), "Rog": float(gt_raw["Rog"])}

    for exp in exps:
        if gt is None:
            gt = compute_gt_macro_from_metrics(models[exp.name])
    if gt is None and processed_dir.exists():
        try:
            gt = compute_gt_macro_from_dataset(
                processed_dir=processed_dir,
                split=str(args.split),
                obs_len=int(args.obs_len),
                pred_len=int(args.pred_len),
                batch_size=int(args.batch_size),
                max_batches=int(args.max_batches),
                num_workers=int(args.num_workers),
            )
        except RuntimeError as e:
            print(f"[WARN] 跳过 GT 宏观对照：{e}")

    # Load sample trajectories (for qualitative + CDF/boxplot)
    preds_by_model: Dict[str, np.ndarray] = {}
    target = None
    for exp in exps:
        if not exp.samples_path.exists():
            raise FileNotFoundError(exp.samples_path)
        d = np.load(exp.samples_path)
        preds = d["preds"].astype(np.float64)
        tgt = d["targets"].astype(np.float64)
        preds_by_model[exp.name] = preds
        target = tgt if target is None else target
        if target is not None:
            # sanity: targets must match across methods
            if np.max(np.abs(tgt - target)) > 1e-9:
                raise ValueError(f"targets mismatch between samples files: {exp.samples_path}")

    assert target is not None

    # Micro metrics
    plot_micro_metrics(models=models, out_dir=out_dir)

    # Macro MSD curve
    plot_msd_curve(gt=gt, models=models, out_dir=out_dir)

    # Trajectory overlay (same conditions)
    plot_traj_overlay(
        target=target,
        preds_by_model=preds_by_model,
        out_dir=out_dir,
        num_plots=int(args.num_traj_plots),
        seed=int(args.seed),
    )

    # Error CDF (saved samples)
    samples_metrics = {name: _compute_ade_fde_from_samples(pred, target) for name, pred in preds_by_model.items()}
    plot_error_cdf(samples=samples_metrics, out_dir=out_dir)

    # Rog boxplot (saved samples; GT from target)
    plot_rog_boxplot(preds_by_model=preds_by_model, target=target, out_dir=out_dir)

    # Save a compact summary json for paper writing
    summary = {
        "models": {k: {mk: v for mk, v in m.items() if mk != "msd_curve"} for k, m in models.items()},
        "gt_macro": None if gt is None else {"Rog": float(gt.get("Rog", 0.0)), "msd_curve": np.array(gt.get("msd_curve")).tolist()},
        "note": "Phase A is step-based; interpret lag as steps. Saved-sample plots use k=0 sample only.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase_a_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved {out_dir / 'phase_a_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
