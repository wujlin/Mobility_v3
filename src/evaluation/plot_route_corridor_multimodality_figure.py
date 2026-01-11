from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.features.semantic_od import load_osm_road_prob
from src.plot_style import FIGSIZE_FULL, OKABE_ITO, add_panel_label, paper_style, save_figure


@dataclass(frozen=True)
class Config:
    max_gt: int
    k_pred: int
    road_mask_thr: float
    road_alpha: float
    seed: int
    pad_frac: float
    png_dpi: int
    bbox_from: str


def _key64(traj_idx: np.ndarray, start_t: np.ndarray) -> np.ndarray:
    traj_idx = np.asarray(traj_idx, dtype=np.int64).reshape(-1)
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    return (traj_idx << np.int64(32)) | (start_t & np.int64(0xFFFFFFFF))


def _kmeans2(x: np.ndarray, *, seed: int, iters: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n < 2:
        return np.zeros((n,), dtype=np.int64), np.zeros((2, d), dtype=np.float64)

    i0 = int(np.argmin(x[:, 0]))
    i1 = int(np.argmax(x[:, 0]))
    if i0 == i1:
        rng = np.random.default_rng(int(seed))
        i1 = int(rng.integers(0, n))
    c = np.stack([x[i0], x[i1]], axis=0)

    labels = np.zeros((n,), dtype=np.int64)
    for _ in range(int(iters)):
        d0 = np.sum((x - c[0]) ** 2, axis=1)
        d1 = np.sum((x - c[1]) ** 2, axis=1)
        new_labels = (d1 < d0).astype(np.int64)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for k in (0, 1):
            mask = labels == k
            if not np.any(mask):
                continue
            c[k] = np.mean(x[mask], axis=0)
    return labels.astype(np.int64, copy=False), c.astype(np.float64, copy=False)


def _polyline_features_to_dest_single(start_pos: np.ndarray, path: np.ndarray, dest_pos: np.ndarray) -> np.ndarray:
    start_pos = np.asarray(start_pos, dtype=np.float64).reshape(2)
    path = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    dest_pos = np.asarray(dest_pos, dtype=np.float64).reshape(2)

    poly = np.concatenate([start_pos[None, :], path], axis=0)
    a = start_pos
    b = dest_pos
    ab = b - a
    chord = float(np.linalg.norm(ab)) + 1e-12

    ap = poly - a[None, :]
    cross = ab[0] * ap[:, 1] - ab[1] * ap[:, 0]
    dist_signed = cross / chord
    dist_signed[0] = 0.0
    idx = int(np.argmax(np.abs(dist_signed)))
    dev_signed = float(dist_signed[idx])
    signed_dev_ratio = float(dev_signed / chord)

    end_seg = poly[-1]
    proj = float(np.sum((end_seg - a) * ab) / (chord * chord))

    seg = poly[1:] - poly[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    path_len = float(np.sum(seg_len))
    len_ratio = float(path_len / chord)

    return np.asarray([signed_dev_ratio, proj, len_ratio], dtype=np.float64)


def _fit_two_corridors(feats: np.ndarray, *, seed: int) -> Dict[str, np.ndarray]:
    feats = np.asarray(feats, dtype=np.float64)
    mu = np.mean(feats, axis=0)
    sig = np.std(feats, axis=0) + 1e-6
    x = (feats - mu) / sig
    labels, centers = _kmeans2(x, seed=int(seed))
    return {
        "mu": mu.astype(np.float64, copy=False),
        "sig": sig.astype(np.float64, copy=False),
        "centers": centers.astype(np.float64, copy=False),
        "labels": labels.astype(np.int64, copy=False),
    }


def _assign_cluster(feat: np.ndarray, *, mu: np.ndarray, sig: np.ndarray, centers: np.ndarray) -> int:
    z = (np.asarray(feat, dtype=np.float64) - mu) / sig
    d0 = float(np.sum((z - centers[0]) ** 2))
    d1 = float(np.sum((z - centers[1]) ** 2))
    return 1 if d1 < d0 else 0


def _remap_corridor_labels_by_signed_dev(feats: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    feats = np.asarray(feats, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if feats.ndim != 2 or feats.shape[1] < 1:
        return labels, {"mu0": float("nan"), "mu1": float("nan"), "swap": 0.0}
    d = feats[:, 0]
    mu0 = float(np.mean(d[labels == 0])) if np.any(labels == 0) else float("nan")
    mu1 = float(np.mean(d[labels == 1])) if np.any(labels == 1) else float("nan")
    swap = bool(np.isfinite(mu0) and np.isfinite(mu1) and (mu0 > mu1))
    if swap:
        labels = (1 - labels).astype(np.int64, copy=False)
    return labels, {"mu0": float(mu0), "mu1": float(mu1), "swap": float(1.0 if swap else 0.0)}


def _sample_indices_stratified(labels: np.ndarray, *, max_n: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = int(labels.size)
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    half = int(max_n) // 2

    take0 = min(int(idx0.size), half)
    take1 = min(int(idx1.size), half)
    rem = int(max_n) - int(take0) - int(take1)
    if rem > 0:
        if int(idx0.size) - take0 >= int(idx1.size) - take1:
            take0 = min(int(idx0.size), take0 + rem)
        else:
            take1 = min(int(idx1.size), take1 + rem)

    pick0 = rng.choice(idx0, size=int(take0), replace=False) if take0 > 0 else np.zeros((0,), dtype=np.int64)
    pick1 = rng.choice(idx1, size=int(take1), replace=False) if take1 > 0 else np.zeros((0,), dtype=np.int64)
    pick = np.concatenate([pick0, pick1], axis=0)
    pick = np.sort(pick.astype(np.int64, copy=False))
    if int(pick.size) > int(max_n):
        pick = pick[: int(max_n)]
    return pick


def _poly_points(start_pos: np.ndarray, path: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(start_pos, dtype=np.float64).reshape(1, 2), np.asarray(path, dtype=np.float64).reshape(-1, 2)], axis=0)


def _compute_bbox(polys: np.ndarray, *, pad_frac: float) -> Tuple[int, int, int, int]:
    polys = np.asarray(polys, dtype=np.float64).reshape(-1, 2)
    y0 = float(np.min(polys[:, 0]))
    y1 = float(np.max(polys[:, 0]))
    x0 = float(np.min(polys[:, 1]))
    x1 = float(np.max(polys[:, 1]))
    dy = float(y1 - y0)
    dx = float(x1 - x0)
    pad = float(max(dx, dy, 1.0)) * float(max(0.0, pad_frac))
    y0i = int(np.floor(y0 - pad))
    y1i = int(np.ceil(y1 + pad))
    x0i = int(np.floor(x0 - pad))
    x1i = int(np.ceil(x1 + pad))
    return y0i, y1i, x0i, x1i


def _draw_road_basemap(ax: plt.Axes, road_prob: np.ndarray, *, y0: int, y1: int, x0: int, x1: int, thr: float, alpha: float) -> None:
    road_prob = np.asarray(road_prob, dtype=np.float32)
    H, W = int(road_prob.shape[0]), int(road_prob.shape[1])
    y0c = max(0, min(H - 1, int(y0)))
    y1c = max(0, min(H - 1, int(y1)))
    x0c = max(0, min(W - 1, int(x0)))
    x1c = max(0, min(W - 1, int(x1)))
    if y1c <= y0c:
        y1c = min(H - 1, y0c + 1)
    if x1c <= x0c:
        x1c = min(W - 1, x0c + 1)

    crop = road_prob[y0c : y1c + 1, x0c : x1c + 1]
    mask = (crop >= float(thr)).astype(np.int8, copy=False)
    extent = (float(x0c) - 0.5, float(x1c) + 0.5, float(y1c) + 0.5, float(y0c) - 0.5)
    cmap = ListedColormap(["#FFFFFF", "#CCCCCC"])
    ax.imshow(mask, cmap=cmap, vmin=0, vmax=1, alpha=float(alpha), origin="upper", extent=extent, interpolation="nearest")


def _find_index_by_key(ms: np.lib.npyio.NpzFile, *, key: int) -> int:
    if "traj_idx" not in ms.files or "start_t" not in ms.files:
        raise ValueError("model samples npz must contain traj_idx and start_t for alignment")
    ms_key = _key64(np.asarray(ms["traj_idx"]), np.asarray(ms["start_t"]))
    hit = np.where(ms_key == np.int64(key))[0]
    if hit.size <= 0:
        raise RuntimeError(f"Cannot find window key={key} in model samples (traj_idx/start_t mismatch).")
    return int(hit[0])


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Figure 1 hero: corridor-level multi-modality (GT vs L2 vs diffusion vs CascadeTraj).")
    p.add_argument("--gt_case_npz", type=str, required=True)
    p.add_argument("--l2_samples_npz", type=str, required=True)
    p.add_argument("--diffusion_samples_npz", type=str, required=True)
    p.add_argument("--ours_samples_npz", type=str, required=True)
    p.add_argument("--semantic_dir", type=str, default=None, help="Optional directory containing osm_road_prob.npy for light basemap.")
    p.add_argument("--out_pdf", type=str, required=True)
    p.add_argument("--out_png", type=str, default=None)
    p.add_argument("--out_json", type=str, default=None)

    p.add_argument("--max_gt", type=int, default=50)
    p.add_argument("--k_pred", type=int, default=20)
    p.add_argument("--road_mask_thr", type=float, default=0.5)
    p.add_argument("--road_alpha", type=float, default=0.3)
    p.add_argument("--pad_frac", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--png_dpi", type=int, default=300)
    p.add_argument(
        "--bbox_from",
        type=str,
        default="gt_l2_ours",
        choices=["gt_l2_ours", "all"],
        help="Which polylines define the shared axis limits. Use 'gt_l2_ours' to avoid huge blank areas when E2E diverges.",
    )
    p.add_argument("--rep_traj_idx", type=int, default=None, help="Optional: force representative window traj_idx instead of the first in gt_case.")
    p.add_argument("--rep_start_t", type=int, default=None, help="Optional: force representative window start_t (used with --rep_traj_idx).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config(
        max_gt=int(args.max_gt),
        k_pred=int(args.k_pred),
        road_mask_thr=float(args.road_mask_thr),
        road_alpha=float(args.road_alpha),
        seed=int(args.seed),
        pad_frac=float(args.pad_frac),
        png_dpi=int(args.png_dpi),
        bbox_from=str(args.bbox_from),
    )

    gt = np.load(str(Path(args.gt_case_npz)), allow_pickle=True)
    need_gt = {"start_pos", "targets", "dest_pos", "traj_idx", "start_t"}
    if not need_gt.issubset(set(gt.files)):
        raise ValueError(f"gt_case_npz must contain {sorted(need_gt)}, got {sorted(list(gt.files))}")
    gt_start = np.asarray(gt["start_pos"], dtype=np.float32)
    gt_targets = np.asarray(gt["targets"], dtype=np.float32)
    gt_dest = np.asarray(gt["dest_pos"], dtype=np.float32)
    gt_traj_idx = np.asarray(gt["traj_idx"], dtype=np.int64)
    gt_start_t = np.asarray(gt["start_t"], dtype=np.int64)

    N, F = int(gt_targets.shape[0]), int(gt_targets.shape[1])
    if gt_targets.ndim != 3 or gt_targets.shape[-1] != 2 or gt_start.shape != (N, 2) or gt_dest.shape != (N, 2):
        raise ValueError(f"Bad GT shapes: start={gt_start.shape} dest={gt_dest.shape} targets={gt_targets.shape}")

    # Fit corridor clustering on GT (case-level).
    feats_gt = np.stack([_polyline_features_to_dest_single(gt_start[i], gt_targets[i], gt_dest[i]) for i in range(N)], axis=0)
    cl = _fit_two_corridors(feats_gt, seed=int(cfg.seed))
    labels_gt = np.asarray(cl["labels"], dtype=np.int64)
    labels_gt, label_map_info = _remap_corridor_labels_by_signed_dev(feats_gt, labels_gt)

    # Representative window: default to the first one, or override via (traj_idx,start_t).
    if (args.rep_traj_idx is None) ^ (args.rep_start_t is None):
        raise ValueError("rep_traj_idx and rep_start_t must be both set or both None.")
    if args.rep_traj_idx is not None and args.rep_start_t is not None:
        rep_key = int(_key64(np.asarray([args.rep_traj_idx], dtype=np.int64), np.asarray([args.rep_start_t], dtype=np.int64))[0])
    else:
        rep_key = int(_key64(gt_traj_idx[:1], gt_start_t[:1])[0])
    gt_keys = _key64(gt_traj_idx, gt_start_t)
    hit = np.where(gt_keys == np.int64(rep_key))[0]
    if hit.size <= 0:
        raise RuntimeError(f"Cannot find rep window key={rep_key} in gt_case_npz.")
    rep_i = int(hit[0])
    rep_start = gt_start[rep_i]
    rep_dest = gt_dest[rep_i]

    # Load model samples and locate rep window.
    l2 = np.load(str(Path(args.l2_samples_npz)), allow_pickle=True)
    diff = np.load(str(Path(args.diffusion_samples_npz)), allow_pickle=True)
    ours = np.load(str(Path(args.ours_samples_npz)), allow_pickle=True)
    for ms, name in ((l2, "l2"), (diff, "diffusion"), (ours, "ours")):
        if "preds_k" not in ms.files:
            raise ValueError(f"{name}_samples_npz missing preds_k: {sorted(list(ms.files))}")

    idx_l2 = _find_index_by_key(l2, key=rep_key)
    idx_diff = _find_index_by_key(diff, key=rep_key)
    idx_ours = _find_index_by_key(ours, key=rep_key)

    k_pred = int(max(1, cfg.k_pred))
    l2_paths = np.asarray(l2["preds_k"], dtype=np.float32)[idx_l2, :k_pred]  # (K,F,2)
    diff_paths = np.asarray(diff["preds_k"], dtype=np.float32)[idx_diff, :k_pred]
    ours_paths = np.asarray(ours["preds_k"], dtype=np.float32)[idx_ours, :k_pred]

    # Assign corridor labels for ours via GT clustering.
    mu = np.asarray(cl["mu"], dtype=np.float64)
    sig = np.asarray(cl["sig"], dtype=np.float64)
    centers = np.asarray(cl["centers"], dtype=np.float64)
    ours_labels = []
    for i in range(int(ours_paths.shape[0])):
        feat = _polyline_features_to_dest_single(rep_start, ours_paths[i], rep_dest)
        ours_labels.append(_assign_cluster(feat, mu=mu, sig=sig, centers=centers))
    ours_labels = np.asarray(ours_labels, dtype=np.int64)
    # Apply the same remap as GT (if swap happened).
    if float(label_map_info.get("swap", 0.0)) > 0.5:
        ours_labels = (1 - ours_labels).astype(np.int64, copy=False)

    # Subsample GT routes for clarity.
    gt_pick = _sample_indices_stratified(labels_gt, max_n=int(cfg.max_gt), seed=int(cfg.seed))
    gt_pick = gt_pick.astype(np.int64, copy=False)

    # BBox for shared extent across panels.
    # Important: if E2E samples diverge far away, including them in the bbox can make other panels unreadable.
    polys = []
    polys.append(_poly_points(rep_start, rep_dest[None, :]))  # O/D
    for i in gt_pick.tolist():
        polys.append(_poly_points(gt_start[i], gt_targets[i]))
    for p in l2_paths:
        polys.append(_poly_points(rep_start, p))
    for p in ours_paths:
        polys.append(_poly_points(rep_start, p))
    if str(cfg.bbox_from) == "all":
        for p in diff_paths:
            polys.append(_poly_points(rep_start, p))
    all_pts = np.concatenate(polys, axis=0)
    y0, y1, x0, x1 = _compute_bbox(all_pts, pad_frac=float(cfg.pad_frac))

    road_prob = None
    if args.semantic_dir:
        road_prob = load_osm_road_prob(args.semantic_dir)

    out_pdf = Path(args.out_pdf)
    out_png = Path(args.out_png) if args.out_png else out_pdf.with_suffix(".png")
    out_json = Path(args.out_json) if args.out_json else out_pdf.with_suffix(".json")

    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_FULL, squeeze=False)
        fig.subplots_adjust(left=0.045, right=0.995, bottom=0.07, top=0.92, wspace=0.06, hspace=0.16)

        ax_gt, ax_l2 = axes[0, 0], axes[0, 1]
        ax_diff, ax_ours = axes[1, 0], axes[1, 1]

        panels = [
            (ax_gt, "Ground Truth (N=%d)" % int(gt_pick.size), "a"),
            (ax_l2, "L2 Regression (Collapse)", "b"),
            (ax_diff, "End-to-end Diffusion (Divergence)", "c"),
            (ax_ours, "CascadeTraj (Ours)", "d"),
        ]

        for ax, title, lab in panels:
            if road_prob is not None:
                _draw_road_basemap(
                    ax,
                    road_prob,
                    y0=int(y0),
                    y1=int(y1),
                    x0=int(x0),
                    x1=int(x1),
                    thr=float(cfg.road_mask_thr),
                    alpha=float(cfg.road_alpha),
                )
            ax.set_title(str(title), pad=2.0)
            ax.set_xlim(float(x0), float(x1))
            ax.set_ylim(float(y1), float(y0))  # y down
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(1.2)
            add_panel_label(ax, lab)

            ax.scatter(
                [float(rep_start[1])],
                [float(rep_start[0])],
                c="black",
                s=80,
                zorder=10,
                edgecolors="white",
                linewidths=1.5,
                marker="o",
            )
            ax.scatter(
                [float(rep_dest[1])],
                [float(rep_dest[0])],
                c="black",
                s=80,
                zorder=10,
                edgecolors="white",
                linewidths=1.5,
                marker="s",
            )

        # (a) GT routes: color by corridor cluster.
        for i in gt_pick.tolist():
            lab = int(labels_gt[int(i)])
            c = OKABE_ITO["blue"] if lab == 0 else OKABE_ITO["vermillion"]
            poly = _poly_points(gt_start[int(i)], gt_targets[int(i)])
            ax_gt.plot(poly[:, 1], poly[:, 0], color=c, alpha=0.60, linewidth=1.5)

        # (b) L2: collapsed gray.
        for p in l2_paths:
            poly = _poly_points(rep_start, p)
            ax_l2.plot(poly[:, 1], poly[:, 0], color=OKABE_ITO["gray"], alpha=0.80, linewidth=2.0)

        # (c) E2E diffusion: blur (purple).
        purple = "#8B5CF6"
        for p in diff_paths:
            poly = _poly_points(rep_start, p)
            ax_diff.plot(poly[:, 1], poly[:, 0], color=purple, alpha=0.50, linewidth=1.5)

        # (d) Ours: corridor-colored.
        for i, p in enumerate(ours_paths):
            lab = int(ours_labels[int(i)])
            c = OKABE_ITO["blue"] if lab == 0 else OKABE_ITO["vermillion"]
            poly = _poly_points(rep_start, p)
            ax_ours.plot(poly[:, 1], poly[:, 0], color=c, alpha=0.70, linewidth=1.8)

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_pdf)
        save_figure(fig, out_png, dpi=int(cfg.png_dpi))
        plt.close(fig)

    report: Dict[str, object] = {
        "gate": "E17 (Figure1 hero: corridor multi-modality)",
        "inputs": {
            "gt_case_npz": str(Path(args.gt_case_npz).resolve()),
            "l2_samples_npz": str(Path(args.l2_samples_npz).resolve()),
            "diffusion_samples_npz": str(Path(args.diffusion_samples_npz).resolve()),
            "ours_samples_npz": str(Path(args.ours_samples_npz).resolve()),
            "semantic_dir": (str(Path(args.semantic_dir).resolve()) if args.semantic_dir else None),
        },
        "config": {
            "max_gt": int(cfg.max_gt),
            "k_pred": int(cfg.k_pred),
            "road_mask_thr": float(cfg.road_mask_thr),
            "road_alpha": float(cfg.road_alpha),
            "pad_frac": float(cfg.pad_frac),
            "seed": int(cfg.seed),
        },
        "stats": {
            "N_gt_total": int(N),
            "N_gt_plotted": int(gt_pick.size),
            "F": int(F),
            "K_plotted": int(k_pred),
        },
        "rep_window": {"traj_idx": int(gt_traj_idx[0]), "start_t": int(gt_start_t[0])},
        "corridors": {
            "label_remap": label_map_info,
            "gt_counts_total": {"c0": int(np.sum(labels_gt == 0)), "c1": int(np.sum(labels_gt == 1))},
            "gt_counts_plotted": {"c0": int(np.sum(labels_gt[gt_pick] == 0)), "c1": int(np.sum(labels_gt[gt_pick] == 1))},
            "ours_counts_plotted": {"c0": int(np.sum(ours_labels == 0)), "c1": int(np.sum(ours_labels == 1))},
        },
        "bbox": {"y0": int(y0), "y1": int(y1), "x0": int(x0), "x1": int(x1)},
        "outputs": {"figure_pdf": str(out_pdf.resolve()), "figure_png": str(out_png.resolve())},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
