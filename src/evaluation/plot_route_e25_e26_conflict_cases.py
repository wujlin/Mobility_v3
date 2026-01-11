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
from src.features.waypoints import WaypointConfig, extract_oracle_waypoints_from_future
from src.plot_style import FIGSIZE_FULL, OKABE_ITO, paper_style, save_figure
from src.training.route_npz_utils import load_route_windows_npz


@dataclass(frozen=True)
class Config:
    num_cases: int
    seed: int
    jacc_cell: float
    road_prob_thr: float
    max_k_plot: int
    margin: float
    waypoint_mode: str
    waypoint_turn_alpha: float
    num_waypoints: int


def _key64(traj_idx: np.ndarray, start_t: np.ndarray) -> np.ndarray:
    traj_idx = np.asarray(traj_idx, dtype=np.int64).reshape(-1)
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    return (traj_idx << np.int64(32)) | (start_t & np.int64(0xFFFFFFFF))


def _stack_polyline(start_pos: np.ndarray, path: np.ndarray) -> np.ndarray:
    start_pos = np.asarray(start_pos, dtype=np.float32).reshape(1, 2)
    path = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    return np.concatenate([start_pos, path], axis=0).astype(np.float32, copy=False)


def _poly_len(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=np.float32)
    if poly.shape[0] < 2:
        return 0.0
    seg = poly[1:] - poly[:-1]
    return float(np.sum(np.linalg.norm(seg.astype(np.float64), axis=1)))


def _turn_score_mean(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=np.float32)
    if poly.shape[0] < 3:
        return 0.0
    v1 = poly[1:-1] - poly[:-2]
    v2 = poly[2:] - poly[1:-1]
    n1 = np.linalg.norm(v1.astype(np.float64), axis=1)
    n2 = np.linalg.norm(v2.astype(np.float64), axis=1)
    valid = (n1 > 1e-6) & (n2 > 1e-6)
    if not bool(np.any(valid)):
        return 0.0
    dot = np.sum(v1[valid].astype(np.float64) * v2[valid].astype(np.float64), axis=1)
    cos = dot / (n1[valid] * n2[valid] + 1e-12)
    cos = np.clip(cos, -1.0, 1.0)
    # 1 - cos(theta) in [0,2]
    return float(np.mean(1.0 - cos))


def _occupancy_set(start_pos: np.ndarray, path: np.ndarray, *, cell: float) -> set[int]:
    c = max(float(cell), 1e-6)
    pts = _stack_polyline(start_pos, path).astype(np.float64, copy=False)
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


def _compute_ade_best(preds_k: np.ndarray, gt: np.ndarray) -> Tuple[float, float, int]:
    preds_k = np.asarray(preds_k, dtype=np.float32)  # (K,F,2)
    gt = np.asarray(gt, dtype=np.float32)  # (F,2)
    diff = preds_k - gt[None, :, :]
    dist = np.linalg.norm(diff.astype(np.float64), axis=-1).astype(np.float32)  # (K,F)
    ade_k = dist.mean(axis=1)
    best_k = int(np.argmin(ade_k))
    return float(ade_k[best_k]), float(np.mean(ade_k)), best_k


def _compute_crop_bbox(polys: List[np.ndarray], *, margin: float, H: int, W: int) -> Tuple[int, int, int, int]:
    if not polys:
        return 0, H, 0, W
    xs = []
    ys = []
    for p in polys:
        p = np.asarray(p, dtype=np.float32)
        ys.append(p[:, 0])
        xs.append(p[:, 1])
    y0 = float(np.min(np.concatenate(ys))) - float(margin)
    y1 = float(np.max(np.concatenate(ys))) + float(margin)
    x0 = float(np.min(np.concatenate(xs))) - float(margin)
    x1 = float(np.max(np.concatenate(xs))) + float(margin)
    y0i = int(np.floor(y0))
    y1i = int(np.ceil(y1))
    x0i = int(np.floor(x0))
    x1i = int(np.ceil(x1))
    y0i = max(0, min(H - 1, y0i))
    y1i = max(y0i + 1, min(H, y1i))
    x0i = max(0, min(W - 1, x0i))
    x1i = max(x0i + 1, min(W, x1i))
    return y0i, y1i, x0i, x1i


def _overlay_mask(
    ax: plt.Axes,
    *,
    mask: np.ndarray,
    extent: Tuple[float, float, float, float],
    color: str,
    alpha: float,
) -> None:
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    r, g, b, _ = to_rgba(color)
    rgba[:, :, 0] = float(r)
    rgba[:, :, 1] = float(g)
    rgba[:, :, 2] = float(b)
    rgba[:, :, 3] = mask.astype(np.float32, copy=False) * float(alpha)
    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")


def _plot_case_trajs(
    *,
    out_pdf: Path,
    out_png: Path,
    start_pos: np.ndarray,
    dest_pos: np.ndarray,
    gt_targets: np.ndarray,
    preds_e25: np.ndarray,  # (K,F,2)
    preds_e26: np.ndarray,  # (K,F,2)
    best_k_e25: int,
    best_k_e26: int,
    ade_e25: float,
    jac_e25: float,
    ade_e26: float,
    jac_e26: float,
    road_prob: Optional[np.ndarray],
    road_major: Optional[np.ndarray],
    road_minor: Optional[np.ndarray],
    road_service: Optional[np.ndarray],
    cfg: Config,
) -> None:
    gt_poly = _stack_polyline(start_pos, gt_targets)
    polys = [gt_poly]
    for arr in (preds_e25, preds_e26):
        for k in range(int(min(arr.shape[0], cfg.max_k_plot))):
            polys.append(_stack_polyline(start_pos, arr[k]))

    H = int(road_prob.shape[0]) if road_prob is not None else 1024
    W = int(road_prob.shape[1]) if road_prob is not None else 1024
    y0, y1, x0, x1 = _compute_crop_bbox(polys, margin=float(cfg.margin), H=H, W=W)
    extent = (float(x0), float(x1), float(y0), float(y1))

    with paper_style():
        fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)

        if road_prob is not None:
            crop = road_prob[y0:y1, x0:x1]
            ax.imshow(crop, origin="lower", extent=extent, cmap="Greys", vmin=0.0, vmax=1.0, alpha=0.22, interpolation="nearest")
        if road_service is not None:
            m = (road_service[y0:y1, x0:x1] >= float(cfg.road_prob_thr)).astype(np.float32, copy=False)
            _overlay_mask(ax, mask=m, extent=extent, color="#BBBBBB", alpha=0.12)
        if road_minor is not None:
            m = (road_minor[y0:y1, x0:x1] >= float(cfg.road_prob_thr)).astype(np.float32, copy=False)
            _overlay_mask(ax, mask=m, extent=extent, color="#666666", alpha=0.16)
        if road_major is not None:
            m = (road_major[y0:y1, x0:x1] >= float(cfg.road_prob_thr)).astype(np.float32, copy=False)
            _overlay_mask(ax, mask=m, extent=extent, color="#000000", alpha=0.22)

        # GT
        ax.plot(gt_poly[:, 1], gt_poly[:, 0], color=OKABE_ITO["black"], alpha=0.9, linewidth=2.2, label="GT")

        # E26 (blue)
        k_plot = int(min(preds_e26.shape[0], cfg.max_k_plot))
        for k in range(k_plot):
            poly = _stack_polyline(start_pos, preds_e26[k])
            ax.plot(poly[:, 1], poly[:, 0], color=OKABE_ITO["blue"], alpha=0.18, linewidth=1.2, linestyle="--")
        poly_b = _stack_polyline(start_pos, preds_e26[int(best_k_e26)])
        ax.plot(poly_b[:, 1], poly_b[:, 0], color=OKABE_ITO["blue"], alpha=0.95, linewidth=2.2, linestyle="--", label="E26 (OD-only)")

        # E25 (red)
        k_plot = int(min(preds_e25.shape[0], cfg.max_k_plot))
        for k in range(k_plot):
            poly = _stack_polyline(start_pos, preds_e25[k])
            ax.plot(poly[:, 1], poly[:, 0], color=OKABE_ITO["vermillion"], alpha=0.18, linewidth=1.2, linestyle="--")
        poly_r = _stack_polyline(start_pos, preds_e25[int(best_k_e25)])
        ax.plot(poly_r[:, 1], poly_r[:, 0], color=OKABE_ITO["vermillion"], alpha=0.95, linewidth=2.2, linestyle="--", label="E25 (tier-road)")

        # O/D markers
        ax.scatter([start_pos[1]], [start_pos[0]], s=90, c="black", edgecolors="white", linewidths=1.4, zorder=5)
        ax.scatter([dest_pos[1]], [dest_pos[0]], s=90, c="black", edgecolors="white", linewidths=1.4, marker="s", zorder=5)

        ax.set_xlim(float(x0), float(x1))
        ax.set_ylim(float(y0), float(y1))
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower right", frameon=True, framealpha=0.9)

        txt = (
            f"E26: ADE_best={ade_e26:.2f}, J={jac_e26:.3f}\n"
            f"E25: ADE_best={ade_e25:.2f}, J={jac_e25:.3f}"
        )
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.0,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
        )

        save_figure(fig, out_pdf)
        save_figure(fig, out_png, dpi=150)
        plt.close(fig)


def _plot_case_waypoints(
    *,
    out_pdf: Path,
    out_png: Path,
    start_pos: np.ndarray,
    dest_pos: np.ndarray,
    gt_targets: np.ndarray,
    wp_e25: np.ndarray,  # (K,2,2)
    wp_e26: np.ndarray,  # (K,2,2)
    road_prob: Optional[np.ndarray],
    road_major: Optional[np.ndarray],
    road_minor: Optional[np.ndarray],
    road_service: Optional[np.ndarray],
    cfg: Config,
) -> None:
    gt_poly = _stack_polyline(start_pos, gt_targets)
    polys = [gt_poly, wp_e25.reshape(-1, 2), wp_e26.reshape(-1, 2)]
    H = int(road_prob.shape[0]) if road_prob is not None else 1024
    W = int(road_prob.shape[1]) if road_prob is not None else 1024
    y0, y1, x0, x1 = _compute_crop_bbox(polys, margin=float(cfg.margin), H=H, W=W)
    extent = (float(x0), float(x1), float(y0), float(y1))

    wcfg = WaypointConfig(mode=str(cfg.waypoint_mode), num_waypoints=int(cfg.num_waypoints), turn_alpha=float(cfg.waypoint_turn_alpha))
    _, gt_wp = extract_oracle_waypoints_from_future(start_pos=start_pos, future_pos=gt_targets, cfg=wcfg)

    with paper_style():
        fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)

        if road_prob is not None:
            crop = road_prob[y0:y1, x0:x1]
            ax.imshow(crop, origin="lower", extent=extent, cmap="Greys", vmin=0.0, vmax=1.0, alpha=0.22, interpolation="nearest")
        if road_service is not None:
            m = (road_service[y0:y1, x0:x1] >= float(cfg.road_prob_thr)).astype(np.float32, copy=False)
            _overlay_mask(ax, mask=m, extent=extent, color="#BBBBBB", alpha=0.12)
        if road_minor is not None:
            m = (road_minor[y0:y1, x0:x1] >= float(cfg.road_prob_thr)).astype(np.float32, copy=False)
            _overlay_mask(ax, mask=m, extent=extent, color="#666666", alpha=0.16)
        if road_major is not None:
            m = (road_major[y0:y1, x0:x1] >= float(cfg.road_prob_thr)).astype(np.float32, copy=False)
            _overlay_mask(ax, mask=m, extent=extent, color="#000000", alpha=0.22)

        # GT polyline light
        ax.plot(gt_poly[:, 1], gt_poly[:, 0], color=OKABE_ITO["black"], alpha=0.25, linewidth=1.6)

        # Model waypoints
        k_plot = int(min(wp_e26.shape[0], cfg.max_k_plot))
        pts26 = wp_e26[:k_plot].reshape(-1, 2)
        pts25 = wp_e25[:k_plot].reshape(-1, 2)
        ax.scatter(pts26[:, 1], pts26[:, 0], s=22, c=OKABE_ITO["blue"], alpha=0.35, edgecolors="none", label="E26 WP")
        ax.scatter(pts25[:, 1], pts25[:, 0], s=22, c=OKABE_ITO["vermillion"], alpha=0.35, edgecolors="none", label="E25 WP")

        # GT waypoints
        if gt_wp.size:
            ax.scatter(gt_wp[:, 1], gt_wp[:, 0], s=60, c="black", alpha=0.95, edgecolors="white", linewidths=1.2, label="GT WP")

        # O/D markers
        ax.scatter([start_pos[1]], [start_pos[0]], s=90, c="black", edgecolors="white", linewidths=1.4, zorder=5)
        ax.scatter([dest_pos[1]], [dest_pos[0]], s=90, c="black", edgecolors="white", linewidths=1.4, marker="s", zorder=5)

        ax.set_xlim(float(x0), float(x1))
        ax.set_ylim(float(y0), float(y1))
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower right", frameon=True, framealpha=0.9)

        save_figure(fig, out_pdf)
        save_figure(fig, out_png, dpi=150)
        plt.close(fig)


def _select_cases(
    *,
    chord: np.ndarray,
    detour: np.ndarray,
    turn: np.ndarray,
    ade25: np.ndarray,
    jac25: np.ndarray,
    ade26: np.ndarray,
    jac26: np.ndarray,
    cfg: Config,
) -> List[int]:
    n = int(chord.size)
    if n <= 0:
        return []
    rng = np.random.default_rng(int(cfg.seed))

    p33 = float(np.percentile(chord, 33))
    p66 = float(np.percentile(chord, 66))
    bins = {
        "short": np.where(chord <= p33)[0],
        "mid": np.where((chord > p33) & (chord <= p66))[0],
        "long": np.where(chord > p66)[0],
    }

    ade_diff = ade25 - ade26
    jac_diff = jac25 - jac26
    # Prefer "conflict" windows: E25 worse ADE but higher diversity.
    mask_conf = (ade_diff > 1.0) & (jac_diff > 0.03)
    score = np.maximum(ade_diff, 0.0) * np.maximum(jac_diff, 0.0)

    picks: List[int] = []

    def pick_from(cand: np.ndarray, k: int) -> List[int]:
        if cand.size == 0:
            return []
        c = cand.copy()
        rng.shuffle(c)
        return [int(i) for i in c[: int(k)].tolist()]

    for name in ("short", "mid", "long"):
        cand = bins[name]
        cand_c = cand[mask_conf[cand]] if cand.size else cand
        if cand_c.size >= 2:
            top = cand_c[np.argsort(score[cand_c])[::-1]]
            picks.extend([int(x) for x in top[:2].tolist()])
        else:
            # Relax: maximize ADE diff within the bin.
            top = cand[np.argsort(ade_diff[cand])[::-1]] if cand.size else cand
            picks.extend([int(x) for x in top[:2].tolist()])

    # Two "choice-point" windows: high turn score, prefer conflict if possible.
    idx_turn = np.argsort(turn)[::-1]
    for i in idx_turn.tolist():
        i = int(i)
        if i in picks:
            continue
        if len(picks) >= 8:
            break
        if bool(mask_conf[i]) or len(picks) >= 6:
            picks.append(i)

    # Ensure unique and trim to num_cases.
    uniq = []
    for i in picks:
        if i not in uniq:
            uniq.append(int(i))
    if len(uniq) > int(cfg.num_cases):
        uniq = uniq[: int(cfg.num_cases)]
    # If still not enough, fill random.
    if len(uniq) < int(cfg.num_cases):
        rest = [i for i in range(n) if i not in uniq]
        rng.shuffle(rest)
        uniq.extend(rest[: int(cfg.num_cases) - len(uniq)])
    return uniq


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
    gt_dest = np.asarray(gt["dest_pos"], dtype=np.float32)
    gt_traj_idx = np.asarray(gt["traj_idx"], dtype=np.int64)
    gt_start_t = np.asarray(gt["start_t"], dtype=np.int64)
    gt_key = _key64(gt_traj_idx, gt_start_t)
    gt_map = {int(k): int(i) for i, k in enumerate(gt_key.tolist())}

    e25 = np.load(str(e25_samples_npz), allow_pickle=True)
    e26 = np.load(str(e26_samples_npz), allow_pickle=True)
    need = {"preds_k", "wp_abs_k", "traj_idx", "start_t", "start_pos", "dest_pos"}
    for name, data in (("e25", e25), ("e26", e26)):
        if not need.issubset(set(data.files)):
            raise ValueError(f"{name}_samples_npz missing keys: need={sorted(need)} got={sorted(list(data.files))}")

    def map_from(data: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, Dict[int, int]]:
        traj_idx = np.asarray(data["traj_idx"], dtype=np.int64)
        start_t = np.asarray(data["start_t"], dtype=np.int64)
        key = _key64(traj_idx, start_t)
        return key, {int(k): int(i) for i, k in enumerate(key.tolist())}

    e25_key, e25_map = map_from(e25)
    e26_key, e26_map = map_from(e26)

    keys_common = [k for k in gt_map.keys() if k in e25_map and k in e26_map]
    if not keys_common:
        raise RuntimeError("No matched windows across GT/E25/E26 (traj_idx/start_t mismatch).")

    gt_idx = np.asarray([gt_map[int(k)] for k in keys_common], dtype=np.int64)
    e25_idx = np.asarray([e25_map[int(k)] for k in keys_common], dtype=np.int64)
    e26_idx = np.asarray([e26_map[int(k)] for k in keys_common], dtype=np.int64)

    gt_start_m = gt_start[gt_idx]
    gt_dest_m = gt_dest[gt_idx]
    gt_targets_m = gt_targets[gt_idx]

    preds25 = np.asarray(e25["preds_k"], dtype=np.float32)[e25_idx]  # (N,K,F,2)
    preds26 = np.asarray(e26["preds_k"], dtype=np.float32)[e26_idx]
    wp25 = np.asarray(e25["wp_abs_k"], dtype=np.float32)[e25_idx]  # (N,K,2,2)
    wp26 = np.asarray(e26["wp_abs_k"], dtype=np.float32)[e26_idx]

    N = int(gt_targets_m.shape[0])
    K = int(preds25.shape[1])
    F = int(gt_targets_m.shape[1])

    # Per-window metrics for selection.
    ade25 = np.zeros((N,), dtype=np.float32)
    ade26 = np.zeros((N,), dtype=np.float32)
    jac25 = np.zeros((N,), dtype=np.float32)
    jac26 = np.zeros((N,), dtype=np.float32)
    chord = np.zeros((N,), dtype=np.float32)
    detour = np.zeros((N,), dtype=np.float32)
    turn = np.zeros((N,), dtype=np.float32)
    best_k25 = np.zeros((N,), dtype=np.int64)
    best_k26 = np.zeros((N,), dtype=np.int64)

    for i in range(N):
        ade_b, _ade_m, k_b = _compute_ade_best(preds25[i], gt_targets_m[i])
        ade25[i] = float(ade_b)
        best_k25[i] = int(k_b)
        ade_b, _ade_m, k_b = _compute_ade_best(preds26[i], gt_targets_m[i])
        ade26[i] = float(ade_b)
        best_k26[i] = int(k_b)

        sets = [_occupancy_set(gt_start_m[i], preds25[i, kk], cell=float(cfg.jacc_cell)) for kk in range(K)]
        jac25[i] = float(_mean_pairwise_jaccard_distance(sets))
        sets = [_occupancy_set(gt_start_m[i], preds26[i, kk], cell=float(cfg.jacc_cell)) for kk in range(K)]
        jac26[i] = float(_mean_pairwise_jaccard_distance(sets))

        chord_len = float(np.linalg.norm((gt_dest_m[i] - gt_start_m[i]).astype(np.float64))) + 1e-6
        chord[i] = float(chord_len)
        gt_poly = _stack_polyline(gt_start_m[i], gt_targets_m[i])
        plen = _poly_len(gt_poly)
        detour[i] = float(plen / chord_len)
        turn[i] = float(_turn_score_mean(gt_poly))

    pick = _select_cases(
        chord=chord,
        detour=detour,
        turn=turn,
        ade25=ade25,
        jac25=jac25,
        ade26=ade26,
        jac26=jac26,
        cfg=cfg,
    )

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
    cases_out = []
    for j, i in enumerate(pick, start=1):
        i = int(i)
        case_dir = out_dir / f"case_{j:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        traj_pdf = case_dir / "traj_compare.pdf"
        traj_png = case_dir / "traj_compare.png"
        wp_pdf = case_dir / "waypoints_compare.pdf"
        wp_png = case_dir / "waypoints_compare.png"

        _plot_case_trajs(
            out_pdf=traj_pdf,
            out_png=traj_png,
            start_pos=gt_start_m[i],
            dest_pos=gt_dest_m[i],
            gt_targets=gt_targets_m[i],
            preds_e25=preds25[i],
            preds_e26=preds26[i],
            best_k_e25=int(best_k25[i]),
            best_k_e26=int(best_k26[i]),
            ade_e25=float(ade25[i]),
            jac_e25=float(jac25[i]),
            ade_e26=float(ade26[i]),
            jac_e26=float(jac26[i]),
            road_prob=road_prob,
            road_major=road_major,
            road_minor=road_minor,
            road_service=road_service,
            cfg=cfg,
        )
        _plot_case_waypoints(
            out_pdf=wp_pdf,
            out_png=wp_png,
            start_pos=gt_start_m[i],
            dest_pos=gt_dest_m[i],
            gt_targets=gt_targets_m[i],
            wp_e25=wp25[i],
            wp_e26=wp26[i],
            road_prob=road_prob,
            road_major=road_major,
            road_minor=road_minor,
            road_service=road_service,
            cfg=cfg,
        )

        k = int(keys_common[i])
        case_payload = {
            "case_id": int(j),
            "window": {
                "traj_idx": int(gt_traj_idx[gt_idx[i]]),
                "start_t": int(gt_start_t[gt_idx[i]]),
                "key64": int(k),
            },
            "gt": {
                "chord_len": float(chord[i]),
                "detour_ratio": float(detour[i]),
                "turn_score_mean": float(turn[i]),
            },
            "E25": {"ADE_best": float(ade25[i]), "jaccard": float(jac25[i]), "best_k": int(best_k25[i])},
            "E26": {"ADE_best": float(ade26[i]), "jaccard": float(jac26[i]), "best_k": int(best_k26[i])},
            "diff": {
                "ADE_best_E25_minus_E26": float(ade25[i] - ade26[i]),
                "jaccard_E25_minus_E26": float(jac25[i] - jac26[i]),
            },
            "outputs": {
                "traj_pdf": str(traj_pdf),
                "traj_png": str(traj_png),
                "wp_pdf": str(wp_pdf),
                "wp_png": str(wp_png),
            },
        }
        (case_dir / "report.json").write_text(json.dumps(case_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        cases_out.append(case_payload)

    summary = {
        "gate": "E27_viz_e25_vs_e26_metric_conflict",
        "inputs": {
            "gt_windows_npz": str(gt_windows_npz),
            "e25_samples_npz": str(e25_samples_npz),
            "e26_samples_npz": str(e26_samples_npz),
            "semantic_dir": (str(semantic_dir) if semantic_dir is not None else None),
        },
        "stats": {"N_matched": int(N), "K": int(K), "F": int(F), "num_cases": int(len(cases_out))},
        "selection": {
            "strategy": "auto_by_distance_and_conflict",
            "notes": {
                "short_mid_long": "2 windows per chord-length tercile",
                "choice_points": "2 windows with high GT turn score (prefer conflict if possible)",
                "conflict_pref": "prefer windows where E25 has higher Jaccard but worse ADE (heuristic thresholds)",
            },
        },
        "cases": cases_out,
    }
    out_json = out_dir / "report.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="E27: Visual diagnostics for E25 vs E26 metric conflict (per-window GT vs samples + waypoint comparison).")
    p.add_argument("--gt_windows_npz", type=str, required=True)
    p.add_argument("--e25_samples_npz", type=str, required=True, help="Typically refined_dist_step1p0_it10.npz from E25")
    p.add_argument("--e26_samples_npz", type=str, required=True, help="Typically refined_dist_step1p0_it10.npz from E26")
    p.add_argument("--semantic_dir", type=str, default=None, help="If set, overlays OSM road_prob + tier-road rasters as background.")
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--num_cases", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jacc_cell", type=float, default=8.0)
    p.add_argument("--road_prob_thr", type=float, default=0.5)
    p.add_argument("--max_k_plot", type=int, default=20)
    p.add_argument("--margin", type=float, default=32.0)
    p.add_argument("--waypoint_mode", type=str, choices=["rdp_dev", "rdp_turn"], default="rdp_turn")
    p.add_argument("--waypoint_turn_alpha", type=float, default=1.0)
    p.add_argument("--num_waypoints", type=int, default=2)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config(
        num_cases=int(args.num_cases),
        seed=int(args.seed),
        jacc_cell=float(args.jacc_cell),
        road_prob_thr=float(args.road_prob_thr),
        max_k_plot=int(args.max_k_plot),
        margin=float(args.margin),
        waypoint_mode=str(args.waypoint_mode),
        waypoint_turn_alpha=float(args.waypoint_turn_alpha),
        num_waypoints=int(args.num_waypoints),
    )
    report = run(
        gt_windows_npz=Path(args.gt_windows_npz),
        e25_samples_npz=Path(args.e25_samples_npz),
        e26_samples_npz=Path(args.e26_samples_npz),
        semantic_dir=(Path(args.semantic_dir) if args.semantic_dir else None),
        out_dir=Path(args.out_dir),
        cfg=cfg,
    )
    # Print compact summary (avoid long arrays / per-point prints).
    compact = {
        "gate": report["gate"],
        "stats": report["stats"],
        "cases": [
            {
                "case_id": c["case_id"],
                "traj_idx": c["window"]["traj_idx"],
                "start_t": c["window"]["start_t"],
                "ADE_best_E25": c["E25"]["ADE_best"],
                "ADE_best_E26": c["E26"]["ADE_best"],
                "jaccard_E25": c["E25"]["jaccard"],
                "jaccard_E26": c["E26"]["jaccard"],
            }
            for c in report["cases"]
        ],
        "out_dir": str(Path(args.out_dir).resolve()),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

