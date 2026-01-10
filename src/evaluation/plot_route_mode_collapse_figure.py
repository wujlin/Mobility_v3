from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt

from src.plot_style import FIGSIZE_FULL, OKABE_ITO, add_panel_label, paper_style, save_figure


@dataclass(frozen=True)
class Config:
    max_k: int
    max_gt: int
    max_paths: int
    seed: int
    jacc_cell: float
    ncols: int
    panel_dx: float
    panel_dy: float
    png_dpi: int


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
    return labels.astype(np.int64, copy=False), c


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


def _polyline_features_segment_end_single(start_pos: np.ndarray, path: np.ndarray) -> np.ndarray:
    start_pos = np.asarray(start_pos, dtype=np.float64).reshape(2)
    path = np.asarray(path, dtype=np.float64).reshape(-1, 2)

    poly = np.concatenate([start_pos[None, :], path], axis=0)
    a = poly[0]
    b = poly[-1]
    ab = b - a
    chord = float(np.linalg.norm(ab)) + 1e-12

    ap = poly - a[None, :]
    cross = ab[0] * ap[:, 1] - ab[1] * ap[:, 0]
    dist_signed = cross / chord
    dist_signed[0] = 0.0
    dist_signed[-1] = 0.0
    idx = int(np.argmax(np.abs(dist_signed)))
    dev_signed = float(dist_signed[idx])
    signed_dev_ratio = float(dev_signed / chord)

    seg = poly[1:] - poly[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(s[-1]) + 1e-12
    s_frac = float(s[idx] / total)

    path_len = float(np.sum(seg_len))
    len_ratio = float(path_len / chord)
    return np.asarray([signed_dev_ratio, s_frac, len_ratio], dtype=np.float64)


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


def _parse_models(items: Optional[List[str]]) -> List[Tuple[str, Path]]:
    if not items:
        raise ValueError("At least one --model is required, e.g. --model 'Baseline=/path/samples.npz'")
    out: List[Tuple[str, Path]] = []
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Bad --model '{raw}', expected 'Label=/path/to/samples.npz'")
        # Split by the last '=' so labels like 'res=0.1' are supported.
        label, path = raw.rsplit("=", 1)
        label = label.strip()
        path = path.strip()
        if not label:
            raise ValueError(f"Bad --model '{raw}': empty label")
        if not path:
            raise ValueError(f"Bad --model '{raw}': empty path")
        out.append((label, Path(path)))
    return out


def _stack_polyline(start_pos: np.ndarray, path: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(start_pos, dtype=np.float64).reshape(1, 2), np.asarray(path, dtype=np.float64).reshape(-1, 2)], axis=0)


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

    # Fill remainder from the larger cluster.
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


def _compute_bbox(polys: List[np.ndarray]) -> Tuple[float, float, float, float]:
    if not polys:
        return 0.0, 1.0, 0.0, 1.0
    xs = []
    ys = []
    for p in polys:
        p = np.asarray(p, dtype=np.float64)
        ys.append(p[:, 0])
        xs.append(p[:, 1])
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    x0 = float(np.min(x))
    x1 = float(np.max(x))
    y0 = float(np.min(y))
    y1 = float(np.max(y))
    dx = x1 - x0
    dy = y1 - y0
    pad = 0.05 * float(max(dx, dy, 1.0))
    return x0 - pad, x1 + pad, y0 - pad, y1 + pad


def run_figure(
    *,
    gt_case_npz: Path,
    models: List[Tuple[str, Path]],
    out_pdf: Path,
    out_png: Optional[Path],
    cfg: Config,
    out_json: Optional[Path],
) -> Dict[str, object]:
    gt = np.load(str(gt_case_npz), allow_pickle=True)
    need_gt = {"start_pos", "targets"}
    if not need_gt.issubset(set(gt.files)):
        raise ValueError(f"gt_case_npz must contain {sorted(need_gt)}, got {sorted(list(gt.files))}")

    gt_start = np.asarray(gt["start_pos"], dtype=np.float32)
    gt_targets = np.asarray(gt["targets"], dtype=np.float32)
    gt_dest = np.asarray(gt["dest_pos"], dtype=np.float32) if "dest_pos" in gt.files else None
    gt_traj_idx = np.asarray(gt["traj_idx"], dtype=np.int64) if "traj_idx" in gt.files else None
    gt_start_t = np.asarray(gt["start_t"], dtype=np.int64) if "start_t" in gt.files else None

    N = int(gt_targets.shape[0])
    F = int(gt_targets.shape[1])
    if gt_start.shape[0] != N or gt_start.shape[1] != 2:
        raise ValueError(f"Bad gt start_pos shape: {gt_start.shape} (expected N,2 with N={N})")
    if gt_targets.ndim != 3 or gt_targets.shape[-1] != 2:
        raise ValueError(f"Bad gt targets shape: {gt_targets.shape} (expected N,F,2)")
    if gt_dest is not None and (gt_dest.shape[0] != N or gt_dest.shape[1] != 2):
        raise ValueError(f"Bad gt dest_pos shape: {gt_dest.shape} (expected N,2)")

    # Fit corridor clustering on GT.
    feats = []
    if gt_dest is not None:
        for i in range(N):
            feats.append(_polyline_features_to_dest_single(gt_start[i], gt_targets[i], gt_dest[i]))
        od_end = "dest_pos"
    else:
        for i in range(N):
            feats.append(_polyline_features_segment_end_single(gt_start[i], gt_targets[i]))
        od_end = "segment_end"
    feats_arr = np.stack(feats, axis=0)
    cl = _fit_two_corridors(feats_arr, seed=int(cfg.seed))

    # Preload GT polylines (subsample for background).
    rng = np.random.default_rng(int(cfg.seed))
    max_gt = int(cfg.max_gt)
    if max_gt <= 0:
        gt_pick = np.zeros((0,), dtype=np.int64)
    else:
        gt_pick = np.arange(N, dtype=np.int64)
        if N > max_gt:
            gt_pick = rng.choice(gt_pick, size=max_gt, replace=False).astype(np.int64)
            gt_pick = np.sort(gt_pick)
    gt_polys = [_stack_polyline(gt_start[i], gt_targets[i]) for i in gt_pick.tolist()]
    gt_labels = cl["labels"][gt_pick] if cl.get("labels") is not None else np.zeros((gt_pick.size,), dtype=np.int64)

    # Load model samples, align to GT windows when possible, and collect polylines.
    model_payloads: List[Dict[str, object]] = []
    model_plot_data: List[Dict[str, object]] = []

    gt_key = None
    gt_map = None
    if gt_traj_idx is not None and gt_start_t is not None:
        gt_key = _key64(gt_traj_idx, gt_start_t)
        gt_map = {int(k): int(i) for i, k in enumerate(gt_key.tolist())}

    bbox_polys: List[np.ndarray] = []
    bbox_polys.extend(gt_polys)

    for mi, (label, path) in enumerate(models):
        data = np.load(str(path), allow_pickle=True)
        if "preds_k" in data.files:
            preds_k = np.asarray(data["preds_k"], dtype=np.float32)
        elif "preds" in data.files:
            preds_k = np.asarray(data["preds"], dtype=np.float32)[:, None, :, :]
        else:
            raise ValueError(f"{path} missing preds/preds_k, got keys={list(data.files)}")

        if preds_k.ndim != 4 or preds_k.shape[-1] != 2:
            raise ValueError(f"{path} bad preds_k shape: {preds_k.shape} (expected N,K,F,2)")
        n_ms, k_ms, f_ms, _ = preds_k.shape
        if int(f_ms) != int(F):
            raise ValueError(f"{path} pred_len mismatch: {f_ms} vs gt F={F}")

        ms_start = np.asarray(data["start_pos"], dtype=np.float32) if "start_pos" in data.files else None
        ms_dest = np.asarray(data["dest_pos"], dtype=np.float32) if "dest_pos" in data.files else None
        ms_traj_idx = np.asarray(data["traj_idx"], dtype=np.int64) if "traj_idx" in data.files else None
        ms_start_t = np.asarray(data["start_t"], dtype=np.int64) if "start_t" in data.files else None

        aligned = False
        if gt_map is not None and ms_traj_idx is not None and ms_start_t is not None:
            ms_key = _key64(ms_traj_idx, ms_start_t)
            ms_map = {int(k): int(i) for i, k in enumerate(ms_key.tolist())}
            idx = [ms_map.get(int(k)) for k in gt_key.tolist()]  # type: ignore[union-attr]
            idx = [i for i in idx if i is not None]
            if len(idx) > 0:
                aligned = True
                use_idx = np.asarray(idx, dtype=np.int64)
            else:
                use_idx = np.arange(int(n_ms), dtype=np.int64)
        else:
            use_idx = np.arange(int(n_ms), dtype=np.int64)

        k_use = int(min(int(cfg.max_k), int(k_ms)))
        if k_use <= 0:
            k_use = int(k_ms)

        paths = []
        starts = []
        dests = []
        for ii in use_idx.tolist():
            if ms_start is None:
                raise ValueError(f"{path} missing start_pos (required for plotting)")
            s = ms_start[int(ii)]
            d = None
            if od_end == "dest_pos":
                if ms_dest is not None:
                    d = ms_dest[int(ii)]
                elif gt_dest is not None and gt_map is not None and ms_traj_idx is not None and ms_start_t is not None:
                    # Fallback: map to gt dest if possible.
                    k0 = int(_key64(ms_traj_idx[int(ii)], ms_start_t[int(ii)])[0])
                    gi = gt_map.get(k0)
                    d = gt_dest[int(gi)] if gi is not None else None
                if d is None and gt_dest is not None:
                    d = gt_dest[0]

            for kk in range(k_use):
                p = preds_k[int(ii), int(kk)]
                paths.append(p.astype(np.float32, copy=False))
                starts.append(s.astype(np.float32, copy=False))
                if d is not None:
                    dests.append(np.asarray(d, dtype=np.float32))

        if not paths:
            raise RuntimeError(f"{path}: no paths collected for plotting")

        paths_arr = np.stack(paths, axis=0)  # (M,F,2)
        starts_arr = np.stack(starts, axis=0)  # (M,2)
        dests_arr = np.stack(dests, axis=0) if dests else None  # (M,2) optional

        # Assign corridor clusters for stratified sampling / coloring.
        labels = np.zeros((paths_arr.shape[0],), dtype=np.int64)
        for i in range(int(paths_arr.shape[0])):
            if od_end == "dest_pos" and dests_arr is not None:
                feat = _polyline_features_to_dest_single(starts_arr[i], paths_arr[i], dests_arr[i])
            else:
                feat = _polyline_features_segment_end_single(starts_arr[i], paths_arr[i])
            labels[i] = int(_assign_cluster(feat, mu=cl["mu"], sig=cl["sig"], centers=cl["centers"]))

        pick = _sample_indices_stratified(labels, max_n=int(cfg.max_paths), seed=int(cfg.seed) + 31 * int(mi))
        pick = pick.astype(np.int64, copy=False)

        plot_paths = paths_arr[pick]
        plot_starts = starts_arr[pick]
        plot_labels = labels[pick]
        plot_dests = dests_arr[pick] if dests_arr is not None else None

        polys = [_stack_polyline(plot_starts[i], plot_paths[i]) for i in range(int(plot_paths.shape[0]))]
        bbox_polys.extend(polys)

        model_plot_data.append(
            {
                "label": str(label),
                "aligned_to_gt": bool(aligned),
                "N_windows_in_file": int(n_ms),
                "K_in_file": int(k_ms),
                "K_used": int(k_use),
                "paths_total": int(paths_arr.shape[0]),
                "paths_plotted": int(plot_paths.shape[0]),
                "cluster_counts_plotted": {"c0": int(np.sum(plot_labels == 0)), "c1": int(np.sum(plot_labels == 1))},
                "paths": plot_paths,
                "starts": plot_starts,
                "labels": plot_labels,
                "dests": plot_dests,
            }
        )
        model_payloads.append(
            {
                "label": str(label),
                "samples_npz": str(path),
                "aligned_to_gt": bool(aligned),
                "N_windows_in_file": int(n_ms),
                "K_in_file": int(k_ms),
                "K_used": int(k_use),
                "paths_total": int(paths_arr.shape[0]),
                "paths_plotted": int(plot_paths.shape[0]),
                "cluster_counts_plotted": {"c0": int(np.sum(plot_labels == 0)), "c1": int(np.sum(plot_labels == 1))},
            }
        )

    x0, x1, y0, y1 = _compute_bbox(bbox_polys)

    # Plot.
    n_models = int(len(model_plot_data))
    ncols = int(cfg.ncols)
    if ncols <= 0:
        ncols = 2 if n_models >= 4 else n_models
    ncols = max(1, min(ncols, n_models))
    nrows = int(math.ceil(float(n_models) / float(ncols)))

    fig_w = float(FIGSIZE_FULL[0])
    fig_h = 2.45 * float(nrows)  # compact, match FIGSIZE_HALF height per row
    with paper_style():
        fig, axes = plt.subplots(
            nrows=int(nrows),
            ncols=int(ncols),
            figsize=(fig_w, fig_h),
            squeeze=False,
        )
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.04, top=0.90, wspace=0.05, hspace=0.10)
        colors = [OKABE_ITO["blue"], OKABE_ITO["vermillion"]]
        panels = "abcdefghijklmnopqrstuvwxyz"

        flat = axes.reshape(-1)
        for j, md in enumerate(model_plot_data):
            ax = flat[j]
            ax.set_title(str(md["label"]), pad=2.0)

            # Background GT (optional).
            for i in range(len(gt_polys)):
                c = OKABE_ITO["gray"]
                ax.plot(gt_polys[i][:, 1], gt_polys[i][:, 0], color=c, alpha=0.08, linewidth=1.0)

            p = np.asarray(md["paths"], dtype=np.float32)
            s = np.asarray(md["starts"], dtype=np.float32)
            lab = np.asarray(md["labels"], dtype=np.int64)
            for i in range(int(p.shape[0])):
                k = int(lab[i])
                poly = _stack_polyline(s[i], p[i])
                ax.plot(poly[:, 1], poly[:, 0], color=colors[k], alpha=0.12, linewidth=1.2)

            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            add_panel_label(ax, panels[j], dx=float(cfg.panel_dx), dy=float(cfg.panel_dy))

        for k in range(int(n_models), int(flat.size)):
            flat[k].axis("off")

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_pdf)
        if out_png is None and str(out_pdf).lower().endswith(".pdf"):
            out_png = out_pdf.with_suffix(".png")
        if out_png is not None:
            save_figure(fig, out_png, dpi=int(cfg.png_dpi))
        plt.close(fig)

    out_outputs: Dict[str, str] = {"figure_pdf": str(out_pdf)}
    if out_png is not None:
        out_outputs["figure_png"] = str(out_png)

    result: Dict[str, object] = {
        "inputs": {
            "gt_case_npz": str(gt_case_npz),
            "models": [{"label": str(l), "samples_npz": str(p)} for l, p in models],
        },
        "config": {
            "od_end": str(od_end),
            "max_k": int(cfg.max_k),
            "max_gt": int(cfg.max_gt),
            "max_paths": int(cfg.max_paths),
            "seed": int(cfg.seed),
            "jacc_cell": float(cfg.jacc_cell),
        },
        "gt": {
            "N_windows": int(N),
            "F": int(F),
            "gt_plotted": int(len(gt_polys)),
            "gt_cluster_counts_plotted": {"c0": int(np.sum(gt_labels == 0)), "c1": int(np.sum(gt_labels == 1))},
        },
        "models": model_payloads,
        "outputs": out_outputs,
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot Fig1-style mode collapse comparison for one GT case bucket.")
    p.add_argument("--gt_case_npz", type=str, required=True, help="From route_gt_baseline.py --save_case_npz (case_XX/gt_case.npz)")
    p.add_argument("--model", action="append", default=None, help="Repeatable: 'Label=/path/to/samples.npz' (expects preds_k or preds).")
    p.add_argument("--out_pdf", type=str, required=True)
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--out_png", type=str, default=None, help="Optional preview output (PNG). If omitted, writes alongside --out_pdf.")

    p.add_argument("--max_k", type=int, default=20, help="Max K per window to plot/use (<=K in file).")
    p.add_argument("--max_gt", type=int, default=80, help="Max GT trajectories to plot as gray background (0 to disable).")
    p.add_argument("--max_paths", type=int, default=400, help="Max predicted polylines to plot per model (stratified by corridor cluster).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jacc_cell", type=float, default=8.0)
    p.add_argument("--ncols", type=int, default=0, help="Layout columns (0 => auto; >=4 models defaults to 2).")
    p.add_argument("--panel_dx", type=float, default=-28.0, help="Panel label x-offset in points (negative => left of axis).")
    p.add_argument("--panel_dy", type=float, default=4.0, help="Panel label y-offset in points.")
    p.add_argument("--png_dpi", type=int, default=150, help="DPI for preview PNG.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config(
        max_k=int(args.max_k),
        max_gt=int(args.max_gt),
        max_paths=int(args.max_paths),
        seed=int(args.seed),
        jacc_cell=float(args.jacc_cell),
        ncols=int(args.ncols),
        panel_dx=float(args.panel_dx),
        panel_dy=float(args.panel_dy),
        png_dpi=int(args.png_dpi),
    )
    report = run_figure(
        gt_case_npz=Path(args.gt_case_npz),
        models=_parse_models(args.model),
        out_pdf=Path(args.out_pdf),
        out_png=(Path(args.out_png) if args.out_png else None),
        out_json=(Path(args.out_json) if args.out_json else None),
        cfg=cfg,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
