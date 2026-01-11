from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from src.features.semantic_od import (
    load_osm_road_prob,
    load_osm_road_prob_major,
    load_osm_road_prob_minor,
    load_osm_road_prob_service,
)
from src.plot_style import FIGSIZE_FULL, FIGSIZE_HALF, OKABE_ITO, paper_style, save_figure
from src.training.route_npz_utils import load_route_windows_npz


@dataclass(frozen=True)
class Config:
    seed: int
    jacc_cell: float
    road_prob_thr: float
    bin_size: int
    min_count: int
    max_k: int


def _key64(traj_idx: np.ndarray, start_t: np.ndarray) -> np.ndarray:
    traj_idx = np.asarray(traj_idx, dtype=np.int64).reshape(-1)
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    return (traj_idx << np.int64(32)) | (start_t & np.int64(0xFFFFFFFF))


def _occupancy_set(start_pos: np.ndarray, path: np.ndarray, *, cell: float) -> set[int]:
    c = max(float(cell), 1e-6)
    pts = np.concatenate(
        [np.asarray(start_pos, dtype=np.float64).reshape(1, 2), np.asarray(path, dtype=np.float64).reshape(-1, 2)],
        axis=0,
    )
    yy = np.floor(pts[:, 0] / c).astype(np.int64)
    xx = np.floor(pts[:, 1] / c).astype(np.int64)
    h = (yy << np.int64(32)) ^ (xx & np.int64(0xFFFFFFFF))
    return set(int(v) for v in h.tolist())


def _mean_pairwise_jaccard_distance(sets: List[set[int]]) -> float:
    n = int(len(sets))
    if n < 2:
        return 0.0
    s = 0.0
    cnt = 0
    for i in range(n):
        a = sets[i]
        for j in range(i + 1, n):
            b = sets[j]
            inter = len(a & b)
            uni = len(a | b)
            jac = 0.0 if uni <= 0 else float(inter) / float(uni)
            s += 1.0 - jac
            cnt += 1
    return 0.0 if cnt <= 0 else float(s / float(cnt))


def _compute_ade_best(preds_k: np.ndarray, gt: np.ndarray) -> float:
    preds_k = np.asarray(preds_k, dtype=np.float32)  # (K,F,2)
    gt = np.asarray(gt, dtype=np.float32)  # (F,2)
    diff = preds_k - gt[None, :, :]
    dist = np.linalg.norm(diff.astype(np.float64), axis=-1).astype(np.float32)  # (K,F)
    ade_k = dist.mean(axis=1)
    return float(np.min(ade_k))


def _overlay_mask(ax: plt.Axes, *, mask: np.ndarray, extent: Tuple[float, float, float, float], color: str, alpha: float) -> None:
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    r, g, b, _ = to_rgba(color)
    rgba[:, :, 0] = float(r)
    rgba[:, :, 1] = float(g)
    rgba[:, :, 2] = float(b)
    rgba[:, :, 3] = mask.astype(np.float32, copy=False) * float(alpha)
    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run(
    *,
    gt_windows_npz: Path,
    e25_samples_npz: Path,
    e26_samples_npz: Path,
    semantic_dir: Optional[Path],
    out_dir: Path,
    cfg: Config,
) -> Dict[str, object]:
    gt = load_route_windows_npz(str(gt_windows_npz), max_n=None, seed=int(cfg.seed))
    gt_start = np.asarray(gt["start_pos"], dtype=np.float32)
    gt_targets = np.asarray(gt["targets"], dtype=np.float32)
    gt_traj_idx = np.asarray(gt["traj_idx"], dtype=np.int64)
    gt_start_t = np.asarray(gt["start_t"], dtype=np.int64)
    gt_key = _key64(gt_traj_idx, gt_start_t)
    gt_map = {int(k): int(i) for i, k in enumerate(gt_key.tolist())}

    e25 = np.load(str(e25_samples_npz), allow_pickle=True)
    e26 = np.load(str(e26_samples_npz), allow_pickle=True)
    need = {"preds_k", "traj_idx", "start_t", "start_pos"}
    for name, data in (("e25", e25), ("e26", e26)):
        if not need.issubset(set(data.files)):
            raise ValueError(f"{name}_samples_npz missing keys: need={sorted(need)} got={sorted(list(data.files))}")

    def map_from(data: np.lib.npyio.NpzFile) -> Dict[int, int]:
        traj_idx = np.asarray(data["traj_idx"], dtype=np.int64)
        start_t = np.asarray(data["start_t"], dtype=np.int64)
        key = _key64(traj_idx, start_t)
        return {int(k): int(i) for i, k in enumerate(key.tolist())}

    e25_map = map_from(e25)
    e26_map = map_from(e26)
    keys_common = [k for k in gt_map.keys() if k in e25_map and k in e26_map]
    if not keys_common:
        raise RuntimeError("No matched windows across GT/E25/E26 (traj_idx/start_t mismatch).")

    gt_idx = np.asarray([gt_map[int(k)] for k in keys_common], dtype=np.int64)
    e25_idx = np.asarray([e25_map[int(k)] for k in keys_common], dtype=np.int64)
    e26_idx = np.asarray([e26_map[int(k)] for k in keys_common], dtype=np.int64)

    gt_start_m = gt_start[gt_idx]
    gt_targets_m = gt_targets[gt_idx]
    preds25 = np.asarray(e25["preds_k"], dtype=np.float32)[e25_idx]
    preds26 = np.asarray(e26["preds_k"], dtype=np.float32)[e26_idx]

    N = int(gt_targets_m.shape[0])
    K = int(min(int(cfg.max_k), int(preds25.shape[1]), int(preds26.shape[1])))

    ade25 = np.zeros((N,), dtype=np.float32)
    ade26 = np.zeros((N,), dtype=np.float32)
    jac25 = np.zeros((N,), dtype=np.float32)
    jac26 = np.zeros((N,), dtype=np.float32)

    for i in range(N):
        ade25[i] = _compute_ade_best(preds25[i, :K], gt_targets_m[i])
        ade26[i] = _compute_ade_best(preds26[i, :K], gt_targets_m[i])
        sets = [_occupancy_set(gt_start_m[i], preds25[i, kk], cell=float(cfg.jacc_cell)) for kk in range(K)]
        jac25[i] = float(_mean_pairwise_jaccard_distance(sets))
        sets = [_occupancy_set(gt_start_m[i], preds26[i, kk], cell=float(cfg.jacc_cell)) for kk in range(K)]
        jac26[i] = float(_mean_pairwise_jaccard_distance(sets))

    # Heatmap binning: assign window to the mean GT position (route "center of mass").
    gt_center = np.mean(np.concatenate([gt_start_m[:, None, :], gt_targets_m], axis=1), axis=1)  # (N,2)
    if semantic_dir is not None:
        rp = load_osm_road_prob(semantic_dir)
        H, W = int(rp.shape[0]), int(rp.shape[1])
    else:
        H = 1024
        W = 1024

    b = int(cfg.bin_size)
    ny = int(math.ceil(float(H) / float(b)))
    nx = int(math.ceil(float(W) / float(b)))

    def agg(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sums = np.zeros((ny, nx), dtype=np.float64)
        cnt = np.zeros((ny, nx), dtype=np.int64)
        yy = np.clip((gt_center[:, 0] / float(b)).astype(np.int64), 0, ny - 1)
        xx = np.clip((gt_center[:, 1] / float(b)).astype(np.int64), 0, nx - 1)
        for i in range(N):
            sums[int(yy[i]), int(xx[i])] += float(values[i])
            cnt[int(yy[i]), int(xx[i])] += 1
        mean = np.zeros((ny, nx), dtype=np.float32)
        mask = cnt >= int(cfg.min_count)
        mean[mask] = (sums[mask] / np.maximum(cnt[mask], 1)).astype(np.float32, copy=False)
        mean[~mask] = np.nan
        return mean, cnt

    mean26, cnt = agg(ade26)
    mean25, _ = agg(ade25)
    diff = mean25 - mean26

    # Background rasters (optional).
    road_prob = road_major = road_minor = road_service = None
    if semantic_dir is not None:
        road_prob = load_osm_road_prob(semantic_dir)
        for name, fn in (
            ("major", load_osm_road_prob_major),
            ("minor", load_osm_road_prob_minor),
            ("service", load_osm_road_prob_service),
        ):
            try:
                if name == "major":
                    road_major = fn(semantic_dir)
                elif name == "minor":
                    road_minor = fn(semantic_dir)
                else:
                    road_service = fn(semantic_dir)
            except FileNotFoundError:
                pass

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure: ADE heatmaps ---
    with paper_style():
        fig, axes = plt.subplots(1, 3, figsize=(FIGSIZE_FULL[0] * 1.25, FIGSIZE_FULL[1]))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.88, wspace=0.06)
        extent = (0.0, float(W), 0.0, float(H))

        def draw_bg(ax: plt.Axes) -> None:
            if road_prob is not None:
                ax.imshow(road_prob, origin="lower", extent=extent, cmap="Greys", vmin=0.0, vmax=1.0, alpha=0.18, interpolation="nearest")
            if road_service is not None:
                m = (road_service >= float(cfg.road_prob_thr)).astype(np.float32, copy=False)
                _overlay_mask(ax, mask=m, extent=extent, color="#BBBBBB", alpha=0.10)
            if road_minor is not None:
                m = (road_minor >= float(cfg.road_prob_thr)).astype(np.float32, copy=False)
                _overlay_mask(ax, mask=m, extent=extent, color="#666666", alpha=0.12)
            if road_major is not None:
                m = (road_major >= float(cfg.road_prob_thr)).astype(np.float32, copy=False)
                _overlay_mask(ax, mask=m, extent=extent, color="#000000", alpha=0.16)

        ims = []
        for ax, data, title in (
            (axes[0], mean26, "E26 ADE_best (mean/bin)"),
            (axes[1], mean25, "E25 ADE_best (mean/bin)"),
            (axes[2], diff, "E25 - E26 ADE_best (mean/bin)"),
        ):
            draw_bg(ax)
            if title.startswith("E25 - E26"):
                vmax = float(np.nanpercentile(np.abs(diff), 95)) if np.isfinite(np.nanmax(np.abs(diff))) else 1.0
                im = ax.imshow(data, origin="lower", extent=extent, cmap="coolwarm", vmin=-vmax, vmax=vmax, alpha=0.75, interpolation="nearest")
            else:
                vmax = float(np.nanpercentile(np.concatenate([mean25[~np.isnan(mean25)], mean26[~np.isnan(mean26)]]), 95)) if np.isfinite(np.nanmax(mean25)) else 1.0
                im = ax.imshow(data, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=vmax, alpha=0.75, interpolation="nearest")
            ims.append(im)
            ax.set_title(title, pad=2.0)
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])

        cbar = fig.colorbar(ims[0], ax=axes[:2].ravel().tolist(), fraction=0.025, pad=0.01)
        cbar.set_label("ADE_best")
        cbar2 = fig.colorbar(ims[2], ax=[axes[2]], fraction=0.045, pad=0.01)
        cbar2.set_label("Δ ADE_best")

        heat_pdf = out_dir / "ade_heatmaps.pdf"
        heat_png = out_dir / "ade_heatmaps.png"
        save_figure(fig, heat_pdf)
        save_figure(fig, heat_png, dpi=150)
        plt.close(fig)

    # --- Figure: scatter ADE vs Jaccard ---
    with paper_style():
        fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.92)
        ax.scatter(jac26, ade26, s=10, c=OKABE_ITO["blue"], alpha=0.25, edgecolors="none", label="E26 (OD-only)")
        ax.scatter(jac25, ade25, s=10, c=OKABE_ITO["vermillion"], alpha=0.25, edgecolors="none", label="E25 (tier-road)")
        ax.set_xlabel("Jaccard diversity (mean pairwise distance)")
        ax.set_ylabel("ADE_best")
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        ax.grid(False)
        scat_pdf = out_dir / "ade_vs_jaccard.pdf"
        scat_png = out_dir / "ade_vs_jaccard.png"
        save_figure(fig, scat_pdf)
        save_figure(fig, scat_png, dpi=150)
        plt.close(fig)

    report: Dict[str, object] = {
        "gate": "E28_e25_e26_error_diagnostics",
        "inputs": {
            "gt_windows_npz": str(gt_windows_npz),
            "e25_samples_npz": str(e25_samples_npz),
            "e26_samples_npz": str(e26_samples_npz),
            "semantic_dir": (str(semantic_dir) if semantic_dir is not None else None),
        },
        "config": {
            "seed": int(cfg.seed),
            "K_used": int(K),
            "jacc_cell": float(cfg.jacc_cell),
            "road_prob_thr": float(cfg.road_prob_thr),
            "bin_size": int(cfg.bin_size),
            "min_count": int(cfg.min_count),
        },
        "stats": {
            "N_matched": int(N),
            "heatmap_bins": {"ny": int(ny), "nx": int(nx)},
            "heatmap_counts": {"min": int(np.min(cnt)), "p50": int(np.median(cnt)), "max": int(np.max(cnt))},
        },
        "summary": {
            "E25": {"ADE_best_mean": float(np.mean(ade25)), "jaccard_mean": float(np.mean(jac25)), "corr(ADE,J)": float(_corr(ade25, jac25))},
            "E26": {"ADE_best_mean": float(np.mean(ade26)), "jaccard_mean": float(np.mean(jac26)), "corr(ADE,J)": float(_corr(ade26, jac26))},
        },
        "outputs": {
            "ade_heatmaps_pdf": str((out_dir / "ade_heatmaps.pdf").resolve()),
            "ade_heatmaps_png": str((out_dir / "ade_heatmaps.png").resolve()),
            "ade_vs_jaccard_pdf": str((out_dir / "ade_vs_jaccard.pdf").resolve()),
            "ade_vs_jaccard_png": str((out_dir / "ade_vs_jaccard.png").resolve()),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="E28: Error diagnostics for E25 vs E26 (ADE heatmaps + ADE vs Jaccard scatter).")
    p.add_argument("--gt_windows_npz", type=str, required=True)
    p.add_argument("--e25_samples_npz", type=str, required=True)
    p.add_argument("--e26_samples_npz", type=str, required=True)
    p.add_argument("--semantic_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jacc_cell", type=float, default=8.0)
    p.add_argument("--road_prob_thr", type=float, default=0.5)
    p.add_argument("--bin_size", type=int, default=16)
    p.add_argument("--min_count", type=int, default=8)
    p.add_argument("--max_k", type=int, default=20)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config(
        seed=int(args.seed),
        jacc_cell=float(args.jacc_cell),
        road_prob_thr=float(args.road_prob_thr),
        bin_size=int(args.bin_size),
        min_count=int(args.min_count),
        max_k=int(args.max_k),
    )
    report = run(
        gt_windows_npz=Path(args.gt_windows_npz),
        e25_samples_npz=Path(args.e25_samples_npz),
        e26_samples_npz=Path(args.e26_samples_npz),
        semantic_dir=(Path(args.semantic_dir) if args.semantic_dir else None),
        out_dir=Path(args.out_dir),
        cfg=cfg,
    )
    compact = {
        "gate": report["gate"],
        "stats": report["stats"],
        "summary": report["summary"],
        "out_dir": str(Path(args.out_dir).resolve()),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
