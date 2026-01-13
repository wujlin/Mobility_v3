from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
except ModuleNotFoundError as e:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    cm = None  # type: ignore[assignment]
    _MPL_ERR = e

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception as e:  # pragma: no cover
    cKDTree = None  # type: ignore[assignment]
    _KD_ERR = e

from src.data.road_graph.gate_candidate_paths_from_routes_npz import (  # noqa: F401
    _dilate_cells,
    _jaccard,
    _path_cells,
    _poly_cells,
    _load_graph_npz,
    k_shortest_paths_yen,
)


TZ_SHANGHAI = timezone(timedelta(hours=8))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Diagnose candidate coverage: visualize GT vs K graph candidates and quantify mismatch."
    )
    p.add_argument("--routes_npz", type=str, required=True)
    p.add_argument("--road_graph_npz", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_cases", type=int, default=10)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--dilate_r", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe_factor", type=int, default=20, help="Probe up to num_cases*probe_factor trajectories to pick low/med/high cases.")
    p.add_argument("--max_probe", type=int, default=200, help="Upper bound on probe trajectories for picking cases.")
    p.add_argument("--margin", type=int, default=32, help="Plot margin (grid cells) around the GT trajectory bbox.")
    return p


def _q(x: np.ndarray, p: float) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return float(np.quantile(x, float(p)))


def _make_road_bg(node_y: np.ndarray, node_x: np.ndarray, *, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    h = int(y1 - y0)
    w = int(x1 - x0)
    img = np.zeros((h, w), dtype=np.uint8)
    yy = node_y.astype(np.int32, copy=False)
    xx = node_x.astype(np.int32, copy=False)
    m = (yy >= int(y0)) & (yy < int(y1)) & (xx >= int(x0)) & (xx < int(x1))
    if not np.any(m):
        return img
    yy = yy[m] - int(y0)
    xx = xx[m] - int(x0)
    img[yy, xx] = 1
    return img


def _plot_case(
    *,
    out_png: Path,
    g,
    start: np.ndarray,
    targets: np.ndarray,
    dest: np.ndarray,
    cand_paths: List[List[int]],
    cand_j: np.ndarray,
    best_j: float,
    dilate_r: int,
    margin: int,
) -> None:
    if plt is None:  # pragma: no cover
        raise SystemExit(f"Missing dependency: matplotlib. Error: {_MPL_ERR}")

    H, W = int(g.grid.H), int(g.grid.W)
    poly = np.concatenate([start.reshape(1, 2), targets.reshape(-1, 2), dest.reshape(1, 2)], axis=0)
    y_min = int(max(0, np.floor(np.min(poly[:, 0]) - float(margin))))
    y_max = int(min(H, np.ceil(np.max(poly[:, 0]) + float(margin))))
    x_min = int(max(0, np.floor(np.min(poly[:, 1]) - float(margin))))
    x_max = int(min(W, np.ceil(np.max(poly[:, 1]) + float(margin))))

    bg = _make_road_bg(g.node_y, g.node_x, y0=y_min, y1=y_max, x0=x_min, x1=x_max)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(bg, cmap="Greys", vmin=0, vmax=1, alpha=0.35, origin="upper", extent=[x_min, x_max, y_max, y_min])

    # Candidates: blue colormap, low jaccard = light, high = dark.
    jj = np.asarray(cand_j, dtype=np.float32).reshape(-1)
    vmax = float(np.max(jj)) if jj.size else 1.0
    vmin = float(np.min(jj)) if jj.size else 0.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = lambda v: float((float(v) - vmin) / (vmax - vmin))
    cmap = cm.get_cmap("Blues")

    for p, j in zip(cand_paths, jj.tolist()):
        if len(p) < 2:
            continue
        nodes_yx = np.stack([g.node_y[np.asarray(p, dtype=np.int32)], g.node_x[np.asarray(p, dtype=np.int32)]], axis=1)
        col = cmap(norm(j))
        ax.plot(nodes_yx[:, 1], nodes_yx[:, 0], color=col, linewidth=1.0, alpha=0.7, zorder=3)

    # GT: red thick.
    ax.plot(poly[:, 1], poly[:, 0], color="#D62728", linewidth=3.0, alpha=0.9, zorder=5)

    ax.set_title(f"GT vs {len(cand_paths)} candidates (best_j={best_j:.3f}, dilate_r={int(dilate_r)})", fontsize=11)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_max, y_min])
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout(pad=0.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    args = build_argparser().parse_args()
    if cKDTree is None:  # pragma: no cover
        raise SystemExit(f"Missing scipy.spatial.cKDTree (scipy). Error: {_KD_ERR}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g = _load_graph_npz(Path(args.road_graph_npz))
    node_xy = np.stack([g.node_y, g.node_x], axis=1).astype(np.float64, copy=False)
    tree = cKDTree(node_xy)

    data = np.load(str(Path(args.routes_npz)), allow_pickle=True)
    need = {"start_pos", "targets", "dest_pos", "traj_idx", "start_t"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"routes_npz missing keys: {missing}")
    start_pos = np.asarray(data["start_pos"], dtype=np.float32).reshape(-1, 2)
    dest_pos = np.asarray(data["dest_pos"], dtype=np.float32).reshape(-1, 2)
    targets = np.asarray(data["targets"], dtype=np.float32)
    traj_idx = np.asarray(data["traj_idx"], dtype=np.int64).reshape(-1)
    start_t = np.asarray(data["start_t"], dtype=np.int64).reshape(-1)
    n = int(start_pos.shape[0])

    # Global GT point-to-road distance stats (diagnose A vs B).
    pts_all = np.concatenate([start_pos, targets.reshape(-1, 2), dest_pos], axis=0).astype(np.float64, copy=False)
    gt_dist_all, _ = tree.query(pts_all, k=1)
    gt_dist_all = np.asarray(gt_dist_all, dtype=np.float64).reshape(-1)

    # Probe trajectories to pick representative low/med/high cases.
    rng = np.random.default_rng(int(args.seed))
    num_cases = max(1, int(args.num_cases))
    probe_n = int(min(n, max(int(num_cases) * int(args.probe_factor), 10), int(args.max_probe)))
    probe_idx = rng.choice(n, size=probe_n, replace=False).astype(np.int64, copy=False)

    # Snap OD endpoints to nearest nodes for probing.
    s_dist, s_idx = tree.query(start_pos[probe_idx].astype(np.float64, copy=False), k=1)
    t_dist, t_idx = tree.query(dest_pos[probe_idx].astype(np.float64, copy=False), k=1)
    s_idx = np.asarray(s_idx, dtype=np.int32).reshape(-1)
    t_idx = np.asarray(t_idx, dtype=np.int32).reshape(-1)

    # Compute best_j for each probe trajectory (cache candidates per OD).
    od_cache: Dict[Tuple[int, int], List[List[int]]] = {}
    best_j_probe = np.zeros((probe_n,), dtype=np.float32)
    K = int(args.K)
    dilate_r = int(args.dilate_r)
    H, W = int(g.grid.H), int(g.grid.W)

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        def tqdm(x, *a, **k):  # type: ignore
            return x

    for j, i in enumerate(tqdm(range(probe_n), desc="probe", dynamic_ncols=True)):
        si = int(s_idx[i])
        ti = int(t_idx[i])
        od = (si, ti)
        paths = od_cache.get(od)
        if paths is None:
            paths = k_shortest_paths_yen(g, start=si, goal=ti, K=K)
            od_cache[od] = paths

        gi = int(probe_idx[i])
        gt_poly = np.concatenate([start_pos[gi : gi + 1], targets[gi], dest_pos[gi : gi + 1]], axis=0)
        gt_cells = _poly_cells(gt_poly, H=H, W=W)
        if dilate_r > 0:
            gt_cells = _dilate_cells(gt_cells, H=H, W=W, r=dilate_r)
        bj = 0.0
        for p in paths:
            if len(p) < 2:
                continue
            nodes_yx = np.stack([g.node_y[np.asarray(p, dtype=np.int32)], g.node_x[np.asarray(p, dtype=np.int32)]], axis=1)
            cand_cells = _path_cells(nodes_yx, H=H, W=W)
            if dilate_r > 0:
                cand_cells = _dilate_cells(cand_cells, H=H, W=W, r=dilate_r)
            bj = max(bj, float(_jaccard(gt_cells, cand_cells)))
        best_j_probe[j] = float(bj)

    # Pick cases across the probe distribution (low -> high).
    order = np.argsort(best_j_probe.astype(np.float64), kind="stable")
    picks = np.linspace(0, max(0, order.size - 1), num=num_cases)
    pick_pos = np.unique(np.round(picks).astype(np.int64))
    pick_pos = pick_pos[:num_cases]
    case_indices = probe_idx[order[pick_pos]].astype(np.int64, copy=False).tolist()

    cases_out: List[Dict[str, object]] = []
    for ci, gi in enumerate(tqdm(case_indices, desc="cases", dynamic_ncols=True)):
        # Snap OD for this case (use KDTree on nodes).
        sd, si = tree.query(start_pos[gi].astype(np.float64, copy=False), k=1)
        td, ti = tree.query(dest_pos[gi].astype(np.float64, copy=False), k=1)
        od = (int(si), int(ti))
        paths = od_cache.get(od)
        if paths is None:
            paths = k_shortest_paths_yen(g, start=int(od[0]), goal=int(od[1]), K=K)
            od_cache[od] = paths

        gt_poly = np.concatenate([start_pos[gi : gi + 1], targets[gi], dest_pos[gi : gi + 1]], axis=0)
        gt_cells0 = _poly_cells(gt_poly, H=H, W=W)
        gt_cells = _dilate_cells(gt_cells0, H=H, W=W, r=dilate_r) if dilate_r > 0 else gt_cells0

        cand_j = []
        for p in paths:
            if len(p) < 2:
                cand_j.append(0.0)
                continue
            nodes_yx = np.stack([g.node_y[np.asarray(p, dtype=np.int32)], g.node_x[np.asarray(p, dtype=np.int32)]], axis=1)
            cand_cells0 = _path_cells(nodes_yx, H=H, W=W)
            cand_cells = _dilate_cells(cand_cells0, H=H, W=W, r=dilate_r) if dilate_r > 0 else cand_cells0
            cand_j.append(float(_jaccard(gt_cells, cand_cells)))
        cand_j_arr = np.asarray(cand_j, dtype=np.float32)
        best_k = int(np.argmax(cand_j_arr).item()) if cand_j_arr.size else -1
        best_j = float(np.max(cand_j_arr).item()) if cand_j_arr.size else 0.0

        out_png = out_dir / f"case_{ci:02d}_traj{int(traj_idx[gi])}.png"
        _plot_case(
            out_png=out_png,
            g=g,
            start=start_pos[gi],
            targets=targets[gi],
            dest=dest_pos[gi],
            cand_paths=paths,
            cand_j=cand_j_arr,
            best_j=best_j,
            dilate_r=dilate_r,
            margin=int(args.margin),
        )

        cases_out.append(
            {
                "case_id": int(ci),
                "index_in_npz": int(gi),
                "traj_idx": int(traj_idx[gi]),
                "start_pos": [float(start_pos[gi, 0]), float(start_pos[gi, 1])],
                "dest_pos": [float(dest_pos[gi, 0]), float(dest_pos[gi, 1])],
                "start_t": int(start_t[gi]),
                "snap_dist_grid": {"start": float(sd), "dest": float(td)},
                "best_jaccard": float(best_j),
                "best_k": int(best_k),
                "cand_jaccard": cand_j_arr.tolist(),
                "out_png": str(out_png.resolve()),
            }
        )

    report = {
        "ok": True,
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "tool": "diagnose_candidate_coverage",
        "inputs": {"routes_npz": str(Path(args.routes_npz)), "road_graph_npz": str(Path(args.road_graph_npz))},
        "config": {
            "num_cases": int(num_cases),
            "K": int(K),
            "dilate_r": int(dilate_r),
            "seed": int(args.seed),
            "probe_n": int(probe_n),
            "margin": int(args.margin),
        },
        "global_stats": {
            "N": int(n),
            "graph": {
                "n_nodes": int(g.node_y.shape[0]),
                "n_edges_directed": int(len(g.edge_cost)),
                "node_y_minmax": [float(np.min(g.node_y)), float(np.max(g.node_y))] if g.node_y.size else None,
                "node_x_minmax": [float(np.min(g.node_x)), float(np.max(g.node_x))] if g.node_x.size else None,
            },
            "gt_point_dist_to_road_grid": {
                "num_points": int(gt_dist_all.size),
                "p50": _q(gt_dist_all, 0.5),
                "p90": _q(gt_dist_all, 0.9),
                "frac_le_1": float(np.mean((gt_dist_all <= 1.0).astype(np.float32))),
                "frac_le_2": float(np.mean((gt_dist_all <= 2.0).astype(np.float32))),
            },
            "probe_best_jaccard": {
                "mean": float(np.mean(best_j_probe)),
                "p50": _q(best_j_probe, 0.5),
                "p90": _q(best_j_probe, 0.9),
            },
        },
        "cases": cases_out,
        "outputs": {"diagnose_report_json": str((out_dir / "diagnose_report.json").resolve())},
    }

    (out_dir / "diagnose_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Compact stdout.
    print(
        json.dumps(
            {
                "ok": True,
                "out_dir": str(out_dir.resolve()),
                "report": report["outputs"]["diagnose_report_json"],
                "gt_point_p90": report["global_stats"]["gt_point_dist_to_road_grid"]["p90"],
                "probe_best_j_p50": report["global_stats"]["probe_best_jaccard"]["p50"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

