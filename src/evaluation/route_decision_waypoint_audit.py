from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from src.features.semantic_od import load_osm_road_prob
from src.features.waypoints import WaypointConfig, extract_oracle_waypoints_from_future
from src.plot_style import OKABE_ITO, paper_style, save_figure


def _key64(traj_idx: np.ndarray, start_t: np.ndarray) -> np.ndarray:
    traj_idx = np.asarray(traj_idx, dtype=np.int64).reshape(-1)
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    return (traj_idx << np.int64(32)) | (start_t & np.int64(0xFFFFFFFF))


def _q(x: np.ndarray, p: float) -> Optional[float]:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return None
    return float(np.quantile(x, float(p)))


def _point_to_polyline_min_dist(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] <= 0 or pts.shape[0] <= 0:
        return np.zeros((0,), dtype=np.float32)
    d = np.linalg.norm(poly[None, :, :] - pts[:, None, :], axis=2)  # (P,T)
    return np.min(d, axis=1).astype(np.float32, copy=False)


def _bbox_from_points(pts: np.ndarray, *, pad_ratio: float = 0.08) -> Tuple[int, int, int, int]:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    y0 = float(np.min(pts[:, 0]))
    y1 = float(np.max(pts[:, 0]))
    x0 = float(np.min(pts[:, 1]))
    x1 = float(np.max(pts[:, 1]))
    pad = max((x1 - x0), (y1 - y0), 1.0) * float(pad_ratio)
    y0i, y1i = int(np.floor(y0 - pad)), int(np.ceil(y1 + pad))
    x0i, x1i = int(np.floor(x0 - pad)), int(np.ceil(x1 + pad))
    return y0i, y1i, x0i, x1i


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit decision-stage waypoint realism: compare predicted waypoints vs GT waypoints on a fixed case.")
    p.add_argument("--gt_case_npz", type=str, required=True, help="E0/E0s case npz with start_pos/targets/traj_idx/start_t.")
    p.add_argument("--samples_npz", type=str, required=True, help="Sampling npz that includes wp_abs_k (and traj_idx/start_t).")
    p.add_argument("--semantic_dir", type=str, required=True, help="Dir containing osm_road_prob.npy for a light road-mask background.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--case_name", type=str, default="case", help="Only used for plot title / report.")
    p.add_argument("--road_thr", type=float, default=0.5, help="Threshold for road mask visualization.")
    p.add_argument("--wp_mode", type=str, choices=["rdp_dev", "rdp_turn"], default="rdp_turn")
    p.add_argument("--num_waypoints", type=int, default=2)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = np.load(str(Path(args.gt_case_npz)), allow_pickle=True)
    ms = np.load(str(Path(args.samples_npz)), allow_pickle=True)

    need_gt = {"start_pos", "targets", "traj_idx", "start_t"}
    need_ms = {"traj_idx", "start_t", "wp_abs_k"}
    if not need_gt.issubset(set(gt.files)):
        raise ValueError(f"gt_case_npz missing keys: {sorted(list(need_gt - set(gt.files)))}")
    if not need_ms.issubset(set(ms.files)):
        raise ValueError(f"samples_npz missing keys: {sorted(list(need_ms - set(ms.files)))}")

    keys_gt = _key64(gt["traj_idx"], gt["start_t"])
    keys_ms = _key64(ms["traj_idx"], ms["start_t"])
    ms_map: Dict[int, int] = {int(k): int(i) for i, k in enumerate(keys_ms.tolist())}

    cfg = WaypointConfig(mode=str(args.wp_mode), num_waypoints=int(args.num_waypoints), turn_alpha=1.0)

    polys_all = []
    gt_wp_all = []
    pred_wp_all = []
    dist_poly_all = []

    for i, k in enumerate(keys_gt.tolist()):
        j = ms_map.get(int(k))
        if j is None:
            continue

        start = np.asarray(gt["start_pos"][i], dtype=np.float32).reshape(2)
        future = np.asarray(gt["targets"][i], dtype=np.float32).reshape(-1, 2)
        poly = np.concatenate([start[None, :], future], axis=0)  # (F+1,2)
        polys_all.append(poly)

        _, wp_gt = extract_oracle_waypoints_from_future(start_pos=start, future_pos=future, cfg=cfg)
        gt_wp_all.append(np.asarray(wp_gt, dtype=np.float32).reshape(-1, 2))

        wp_pred_k = np.asarray(ms["wp_abs_k"][j], dtype=np.float32)  # (K, num_wp, 2) or (K,?,2)
        wp_pred = wp_pred_k.reshape(-1, 2)
        pred_wp_all.append(wp_pred)

        dist_poly_all.append(_point_to_polyline_min_dist(poly, wp_pred))

    gt_wp = np.concatenate(gt_wp_all, axis=0) if gt_wp_all else np.zeros((0, 2), dtype=np.float32)
    pred_wp = np.concatenate(pred_wp_all, axis=0) if pred_wp_all else np.zeros((0, 2), dtype=np.float32)
    dist_poly = np.concatenate(dist_poly_all, axis=0) if dist_poly_all else np.zeros((0,), dtype=np.float32)

    road = load_osm_road_prob(args.semantic_dir)
    H, W = int(road.shape[0]), int(road.shape[1])

    pts_for_bbox = []
    for poly in polys_all:
        pts_for_bbox.append(poly)
    if gt_wp.size:
        pts_for_bbox.append(gt_wp)
    if pred_wp.size:
        pts_for_bbox.append(pred_wp)
    pts = np.concatenate(pts_for_bbox, axis=0) if pts_for_bbox else np.zeros((0, 2), dtype=np.float32)
    y0i, y1i, x0i, x1i = _bbox_from_points(pts) if pts.size else (0, H - 1, 0, W - 1)
    y0i, y1i = max(0, y0i), min(H - 1, y1i)
    x0i, x1i = max(0, x0i), min(W - 1, x1i)

    # ---- Plot (PNG + PDF) ----
    with paper_style():
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        mask = (road[y0i : y1i + 1, x0i : x1i + 1] >= float(args.road_thr)).astype(np.int32)
        ax.imshow(
            mask,
            cmap=ListedColormap(["#FFFFFF", "#CCCCCC"]),
            extent=(x0i - 0.5, x1i + 0.5, y1i + 0.5, y0i - 0.5),
            alpha=0.3,
            origin="upper",
        )

        for poly in polys_all:
            ax.plot(poly[:, 1], poly[:, 0], color="#888888", alpha=0.15, lw=1.0)

        if pred_wp.size:
            ax.scatter(pred_wp[:, 1], pred_wp[:, 0], s=8, c=OKABE_ITO["sky_blue"], alpha=0.25, lw=0, label="Pred WP")
        if gt_wp.size:
            ax.scatter(gt_wp[:, 1], gt_wp[:, 0], s=18, c="black", alpha=0.85, edgecolors="white", lw=0.5, zorder=10, label="GT WP")

        ax.set_title(f"Decision WP vs GT ({args.case_name})")
        ax.axis("off")

        save_figure(fig, out_dir / "wp_audit.png", dpi=300)
        save_figure(fig, out_dir / "wp_audit.pdf")
        plt.close(fig)

    report = {
        "ok": True,
        "case": str(args.case_name),
        "inputs": {"gt_case_npz": str(Path(args.gt_case_npz)), "samples_npz": str(Path(args.samples_npz)), "semantic_dir": str(Path(args.semantic_dir))},
        "matched": {"n_gt": int(keys_gt.shape[0]), "n_samples": int(keys_ms.shape[0]), "n_matched": int(len(polys_all))},
        "wp_dist_to_gt_poly": {
            "mean": (float(np.mean(dist_poly)) if dist_poly.size else None),
            "p50": _q(dist_poly, 0.5),
            "p90": _q(dist_poly, 0.9),
        },
        "outputs": {"plot_png": str((out_dir / "wp_audit.png").resolve()), "plot_pdf": str((out_dir / "wp_audit.pdf").resolve())},
    }

    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

