from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore
except Exception as e:  # pragma: no cover
    linkage = None  # type: ignore[assignment]
    fcluster = None  # type: ignore[assignment]
    _HCLUST_ERR = e

from src.data.road_graph.gate_candidate_paths_from_routes_npz import _load_graph_npz
from src.features.semantic_od import (
    load_osm_road_prob_major,
    load_osm_road_prob_minor,
    load_osm_road_prob_service,
)
from src.plot_style import OKABE_ITO, paper_style, save_figure


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class VizCfg:
    od_bin: int
    cluster_dist_thr: float
    min_cluster_size: int
    tz_offset_hours: float
    max_groups: int
    pad_ratio: float
    road_thr: float


def _time_hour(start_t: np.ndarray, *, tz_offset_hours: float) -> np.ndarray:
    t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    t = (t + int(round(float(tz_offset_hours) * 3600.0))).astype(np.int64, copy=False)
    sec = np.mod(t, 86400).astype(np.float64, copy=False)
    hour = sec / 3600.0
    return hour.astype(np.float32, copy=False)


def _od_bin_key(start_pos: np.ndarray, dest_pos: np.ndarray, *, od_bin: int) -> np.ndarray:
    s = np.asarray(start_pos, dtype=np.float32).reshape(-1, 2)
    d = np.asarray(dest_pos, dtype=np.float32).reshape(-1, 2)
    b = int(max(1, od_bin))
    s_bin = np.floor(s / float(b)).astype(np.int32)
    d_bin = np.floor(d / float(b)).astype(np.int32)
    return np.concatenate([s_bin, d_bin], axis=1).astype(np.int32, copy=False)


def _iter_groups(keys: np.ndarray) -> Dict[Tuple[int, ...], np.ndarray]:
    keys = np.asarray(keys, dtype=np.int32)
    view = keys.view([("", keys.dtype)] * keys.shape[1])
    order = np.argsort(view.reshape(-1), kind="mergesort")
    keys_sorted = keys[order]
    idx_sorted = order
    out: Dict[Tuple[int, ...], np.ndarray] = {}
    i = 0
    n = int(keys_sorted.shape[0])
    while i < n:
        j = i + 1
        while j < n and np.array_equal(keys_sorted[j], keys_sorted[i]):
            j += 1
        out[tuple(int(x) for x in keys_sorted[i].tolist())] = idx_sorted[i:j].astype(np.int64, copy=False)
        i = j
    return out


def _seq_from_pad(node_seq_pad: np.ndarray, node_seq_len: np.ndarray, i: int) -> List[int]:
    L = int(node_seq_len[i])
    if L <= 0:
        return []
    seq = node_seq_pad[i, :L].astype(np.int64, copy=False).tolist()
    return [int(x) for x in seq if int(x) >= 0]


def _edge_set(seq: Sequence[int]) -> set[Tuple[int, int]]:
    out: set[Tuple[int, int]] = set()
    for a, b in zip(seq[:-1], seq[1:]):
        aa = int(a)
        bb = int(b)
        if aa >= 0 and bb >= 0 and aa != bb:
            out.add((aa, bb))
    return out


def _jaccard_edges(a: set[Tuple[int, int]], b: set[Tuple[int, int]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    denom = len(a) + len(b) - inter
    if denom <= 0:
        return 0.0
    return float(inter) / float(denom)


def _condensed_from_dist(dist: np.ndarray) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float64)
    m = int(dist.shape[0])
    out = []
    for i in range(m):
        for j in range(i + 1, m):
            out.append(float(dist[i, j]))
    return np.asarray(out, dtype=np.float64)


def _cluster_labels(dist: np.ndarray, *, thr: float) -> np.ndarray:
    if linkage is None or fcluster is None:  # pragma: no cover
        raise SystemExit(f"Missing scipy.cluster.hierarchy (scipy). Error: {_HCLUST_ERR}")
    cd = _condensed_from_dist(dist)
    if cd.size == 0:
        return np.ones((dist.shape[0],), dtype=np.int32)
    Z = linkage(cd, method="average")
    lab = fcluster(Z, t=float(thr), criterion="distance").astype(np.int32, copy=False)
    return lab


def _pick_top2(labels: np.ndarray, *, min_cluster_size: int) -> Optional[Tuple[int, int]]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    uniq, cnt = np.unique(labels, return_counts=True)
    order = np.argsort(-cnt)
    uniq = uniq[order]
    cnt = cnt[order]
    if uniq.size < 2:
        return None
    if int(cnt[0]) < int(min_cluster_size) or int(cnt[1]) < int(min_cluster_size):
        return None
    return int(uniq[0]), int(uniq[1])


def _bbox_from_points(pts_yx: np.ndarray, *, pad_ratio: float) -> Tuple[int, int, int, int]:
    pts = np.asarray(pts_yx, dtype=np.float32).reshape(-1, 2)
    y0 = float(np.min(pts[:, 0]))
    y1 = float(np.max(pts[:, 0]))
    x0 = float(np.min(pts[:, 1]))
    x1 = float(np.max(pts[:, 1]))
    pad = max((x1 - x0), (y1 - y0), 1.0) * float(pad_ratio)
    y0i, y1i = int(np.floor(y0 - pad)), int(np.ceil(y1 + pad))
    x0i, x1i = int(np.floor(x0 - pad)), int(np.ceil(x1 + pad))
    return y0i, y1i, x0i, x1i


def _tier_rgb_crop(
    major: np.ndarray,
    minor: np.ndarray,
    service: np.ndarray,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    crop_ma = major[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32, copy=False)
    crop_mi = minor[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32, copy=False)
    crop_sv = service[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32, copy=False)
    # RGB = (service, minor, major).
    rgb = np.stack([crop_sv, crop_mi, crop_ma], axis=2)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def run(
    *,
    paths_graph_npz: Path,
    road_graph_npz: Path,
    semantic_dir: Path,
    gate_dir: Path,
    out_dir: Path,
    cfg: VizCfg,
) -> Dict[str, object]:
    g = _load_graph_npz(Path(road_graph_npz))
    data = np.load(str(paths_graph_npz), allow_pickle=True)
    need = {"start_t", "start_pos", "dest_pos", "node_seq_pad", "node_seq_len"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"paths_graph.npz missing keys: {missing}")
    start_t = np.asarray(data["start_t"], dtype=np.int64).reshape(-1)
    start_pos = np.asarray(data["start_pos"], dtype=np.float32).reshape(-1, 2)
    dest_pos = np.asarray(data["dest_pos"], dtype=np.float32).reshape(-1, 2)
    node_seq_pad = np.asarray(data["node_seq_pad"], dtype=np.int32)
    node_seq_len = np.asarray(data["node_seq_len"], dtype=np.int32).reshape(-1)

    major = load_osm_road_prob_major(semantic_dir)
    minor = load_osm_road_prob_minor(semantic_dir)
    service = load_osm_road_prob_service(semantic_dir)

    gate_report = json.loads((Path(gate_dir) / "report.json").read_text(encoding="utf-8"))
    events_path = Path(gate_dir) / "events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events.jsonl under gate_dir: {gate_dir}")
    events = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))

    od_bin = int(cfg.od_bin)
    od_keys = _od_bin_key(start_pos, dest_pos, od_bin=od_bin)
    groups_map = _iter_groups(od_keys)

    hour = _time_hour(start_t, tz_offset_hours=float(cfg.tz_offset_hours))

    out_dir.mkdir(parents=True, exist_ok=True)
    plotted = []

    for ev in events[: int(cfg.max_groups) if int(cfg.max_groups) > 0 else len(events)]:
        od_key = tuple(int(x) for x in ev["od_key"])
        idx = groups_map.get(od_key)
        if idx is None:
            continue
        rows = idx.tolist()
        seqs = [_seq_from_pad(node_seq_pad, node_seq_len, int(i)) for i in rows]
        edge_sets = [_edge_set(s) for s in seqs]
        m = int(len(edge_sets))
        dist = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(i + 1, m):
                sim = _jaccard_edges(edge_sets[i], edge_sets[j])
                d = 1.0 - float(sim)
                dist[i, j] = d
                dist[j, i] = d
        lab = _cluster_labels(dist, thr=float(cfg.cluster_dist_thr))
        top2 = _pick_top2(lab, min_cluster_size=int(cfg.min_cluster_size))
        if top2 is None:
            continue
        c0, c1 = top2

        # Build polylines in grid coords.
        polys = []
        pts_all = [start_pos[np.asarray(rows, dtype=np.int64)], dest_pos[np.asarray(rows, dtype=np.int64)]]
        for s in seqs:
            if not s:
                continue
            nodes = np.asarray(s, dtype=np.int64)
            poly = np.stack([g.node_y[nodes], g.node_x[nodes]], axis=1).astype(np.float32, copy=False)
            polys.append(poly)
            pts_all.append(poly)
        pts = np.concatenate(pts_all, axis=0)
        H, W = major.shape
        y0i, y1i, x0i, x1i = _bbox_from_points(pts, pad_ratio=float(cfg.pad_ratio))
        y0i, y1i = max(0, y0i), min(H - 1, y1i)
        x0i, x1i = max(0, x0i), min(W - 1, x1i)

        # Background masks.
        road_any = np.maximum.reduce([major, minor, service])
        mask = (road_any[y0i : y1i + 1, x0i : x1i + 1] >= float(cfg.road_thr)).astype(np.int32)
        rgb = _tier_rgb_crop(major, minor, service, y0=y0i, y1=y1i, x0=x0i, x1=x1i)
        extent = (x0i - 0.5, x1i + 0.5, y1i + 0.5, y0i - 0.5)

        # Plot.
        with paper_style():
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap

            fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), constrained_layout=True)

            # Panel 1: clusters.
            ax = axes[0]
            ax.imshow(mask, cmap=ListedColormap(["#FFFFFF", "#DDDDDD"]), extent=extent, alpha=0.25, origin="upper")
            cmap = plt.get_cmap("tab20")
            for poly, lab_i in zip(polys, lab.tolist()):
                c = cmap((int(lab_i) - 1) % 20)
                ax.plot(poly[:, 1], poly[:, 0], color=c, alpha=0.55, lw=1.0)
            ax.scatter(start_pos[rows[0], 1], start_pos[rows[0], 0], s=38, c="black", edgecolors="white", linewidths=0.8, zorder=10)
            ax.scatter(dest_pos[rows[0], 1], dest_pos[rows[0], 0], s=42, c="black", marker="s", edgecolors="white", linewidths=0.8, zorder=10)
            ax.set_title(f"Traj Clusters (n={int(m)}, k={int(np.unique(lab).size)})")
            ax.axis("off")

            # Panel 2: time vs top2 cluster (binary).
            ax = axes[1]
            sel = (lab == int(c0)) | (lab == int(c1))
            hh = hour[np.asarray(rows, dtype=np.int64)][sel]
            yy = (lab[sel] == int(c1)).astype(np.int32, copy=False)
            jitter = (np.arange(int(yy.size), dtype=np.float32) % 7) * 0.02 - 0.06
            ax.scatter(hh, yy.astype(np.float32) + jitter, s=16, c=[OKABE_ITO["blue"] if v == 0 else OKABE_ITO["vermillion"] for v in yy.tolist()], alpha=0.75, edgecolors="none")
            ax.set_yticks([0, 1])
            ax.set_yticklabels([f"c{c0}", f"c{c1}"])
            ax.set_xlabel("Hour-of-day")
            ax.set_title(f"Time vs Top2 (AUC_time={ev['auc']['time_only']:.2f})")
            ax.grid(alpha=0.2)

            # Panel 3: tier-road raster + top2 trajectories.
            ax = axes[2]
            ax.imshow(rgb, extent=extent, alpha=0.35, origin="upper", interpolation="nearest")
            for poly, lab_i in zip(polys, lab.tolist()):
                if int(lab_i) not in (int(c0), int(c1)):
                    continue
                col = OKABE_ITO["blue"] if int(lab_i) == int(c0) else OKABE_ITO["vermillion"]
                ax.plot(poly[:, 1], poly[:, 0], color=col, alpha=0.65, lw=1.2)
            ax.scatter(start_pos[rows[0], 1], start_pos[rows[0], 0], s=38, c="black", edgecolors="white", linewidths=0.8, zorder=10)
            ax.scatter(dest_pos[rows[0], 1], dest_pos[rows[0], 0], s=42, c="black", marker="s", edgecolors="white", linewidths=0.8, zorder=10)
            ax.set_title(f"Tier-road + Top2 (AUC_tier={ev['auc']['tier_od']:.2f}, AUC_tt={ev['auc']['time_tier']:.2f})")
            ax.axis("off")

            name = "od_" + "_".join(str(int(x)) for x in od_key)
            out_png = out_dir / f"{name}.png"
            out_pdf = out_dir / f"{name}.pdf"
            save_figure(fig, out_png, dpi=250)
            save_figure(fig, out_pdf)
            plt.close(fig)

        plotted.append(
            {
                "od_key": [int(x) for x in od_key],
                "n_total": int(m),
                "n_clusters": int(np.unique(lab).size),
                "top2_frac": float(np.mean(sel.astype(np.float32))),
                "auc": ev["auc"],
                "plot_png": str(out_png),
                "plot_pdf": str(out_pdf),
            }
        )

    report = {
        "ok": True,
        "tool": "viz_cluster_gate",
        "inputs": {
            "paths_graph_npz": str(paths_graph_npz),
            "road_graph_npz": str(road_graph_npz),
            "semantic_dir": str(semantic_dir),
            "gate_dir": str(gate_dir),
        },
        "config": {
            "od_bin": int(cfg.od_bin),
            "cluster_dist_thr": float(cfg.cluster_dist_thr),
            "min_cluster_size": int(cfg.min_cluster_size),
            "tz_offset_hours": float(cfg.tz_offset_hours),
            "max_groups": int(cfg.max_groups),
            "pad_ratio": float(cfg.pad_ratio),
            "road_thr": float(cfg.road_thr),
        },
        "gate_summary": {"decision": gate_report.get("decision"), "auc": gate_report.get("stats", {}).get("auc")},
        "outputs": {"out_dir": str(out_dir), "num_plotted": int(len(plotted)), "plots": plotted},
        "meta": {"created_at": datetime.now(tz=TZ_SHANGHAI).isoformat()},
    }
    (out_dir / "viz_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize cluster-gate results: (1) traj-by-cluster, (2) time vs top2 cluster, (3) tier-road raster overlay.")
    p.add_argument("--paths_graph_npz", type=Path, required=True)
    p.add_argument("--road_graph_npz", type=Path, required=True)
    p.add_argument("--semantic_dir", type=Path, required=True)
    p.add_argument("--gate_dir", type=Path, required=True, help="Gate output dir containing report.json + events.jsonl")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--od_bin", type=int, default=128)
    p.add_argument("--cluster_dist_thr", type=float, default=0.5)
    p.add_argument("--min_cluster_size", type=int, default=3)
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--max_groups", type=int, default=10)
    p.add_argument("--pad_ratio", type=float, default=0.10)
    p.add_argument("--road_thr", type=float, default=0.05)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = VizCfg(
        od_bin=int(args.od_bin),
        cluster_dist_thr=float(args.cluster_dist_thr),
        min_cluster_size=int(args.min_cluster_size),
        tz_offset_hours=float(args.tz_offset_hours),
        max_groups=int(args.max_groups),
        pad_ratio=float(args.pad_ratio),
        road_thr=float(args.road_thr),
    )
    report = run(
        paths_graph_npz=Path(args.paths_graph_npz),
        road_graph_npz=Path(args.road_graph_npz),
        semantic_dir=Path(args.semantic_dir),
        gate_dir=Path(args.gate_dir),
        out_dir=Path(args.out_dir),
        cfg=cfg,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "out_dir": report["outputs"]["out_dir"],
                "num_plotted": report["outputs"]["num_plotted"],
                "viz_report_json": str((Path(report["outputs"]["out_dir"]) / "viz_report.json").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

