from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(x, *args, **kwargs):  # type: ignore[no-redef]
        return x

from src.data.road_graph.gate_candidate_paths_from_routes_npz import _astar, _load_graph_npz, k_shortest_paths_yen
from src.features.waypoints import pick_waypoint_indices_rdp_turn_fixed_k
from src.models.road_graph import ARGraphWaypointBin, WaypointBinARConfig
from src.training.train_graph_ar_waypoint_bins import _time_features
from src.plot_style import OKABE_ITO, paper_style, save_figure


TZ_SHANGHAI = timezone(timedelta(hours=8))


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _seq_from_pad(pad: np.ndarray, lens: np.ndarray, i: int) -> List[int]:
    L = int(lens[i])
    if L <= 0:
        return []
    s = pad[i, :L].astype(np.int64, copy=False).tolist()
    return [int(x) for x in s if int(x) >= 0]


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
    return float(inter) / float(max(1, denom))


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros((0,), dtype=np.float64)
    x = x - float(np.max(x))
    ex = np.exp(np.clip(x, -60.0, 60.0))
    s = float(np.sum(ex))
    if not np.isfinite(s) or s <= 0:
        return np.full_like(ex, 1.0 / float(max(1, ex.size)))
    return (ex / s).astype(np.float64, copy=False)


def _build_node_tier_min(n_nodes: int, edge_u: np.ndarray, edge_tier: np.ndarray) -> np.ndarray:
    out = np.full((int(n_nodes),), 3, dtype=np.int64)
    u = np.asarray(edge_u, dtype=np.int64).reshape(-1)
    t = np.asarray(edge_tier, dtype=np.int64).reshape(-1)
    for uu, tt in zip(u.tolist(), t.tolist()):
        if 0 <= int(uu) < int(n_nodes):
            out[int(uu)] = min(int(out[int(uu)]), int(np.clip(int(tt), 0, 3)))
    return out


def _bin_id_from_yx(y: float, x: float, *, wp_bin: int, H: int, W: int) -> Tuple[int, int, int]:
    wp_bin = int(wp_bin)
    n_by = int((int(H) + wp_bin - 1) // wp_bin)
    n_bx = int((int(W) + wp_bin - 1) // wp_bin)
    by = int(np.clip(math.floor(float(y) / float(wp_bin)), 0, n_by - 1))
    bx = int(np.clip(math.floor(float(x) / float(wp_bin)), 0, n_bx - 1))
    cls = by * int(n_bx) + bx
    return cls, n_by, n_bx


def _build_bin_to_nodes(*, node_y: np.ndarray, node_x: np.ndarray, wp_bin: int, H: int, W: int) -> Tuple[List[np.ndarray], int, int]:
    wp_bin = int(wp_bin)
    n_by = int((int(H) + wp_bin - 1) // wp_bin)
    n_bx = int((int(W) + wp_bin - 1) // wp_bin)
    bins: List[List[int]] = [[] for _ in range(int(n_by * n_bx))]
    for nid, (yy, xx) in enumerate(zip(node_y.tolist(), node_x.tolist())):
        cls, _, _ = _bin_id_from_yx(float(yy), float(xx), wp_bin=wp_bin, H=H, W=W)
        bins[int(cls)].append(int(nid))
    out = [np.asarray(v, dtype=np.int64) if v else np.zeros((0,), dtype=np.int64) for v in bins]
    return out, int(n_by), int(n_bx)


def _pick_node_in_bin(
    *,
    bin_nodes: np.ndarray,
    node_y: np.ndarray,
    node_x: np.ndarray,
    by: int,
    bx: int,
    wp_bin: int,
    rng: np.random.Generator,
) -> Optional[int]:
    if bin_nodes.size == 0:
        return None
    # Prefer nodes near bin center to avoid sampling tiny dead-ends.
    cy = float(by * int(wp_bin) + int(wp_bin) * 0.5)
    cx = float(bx * int(wp_bin) + int(wp_bin) * 0.5)
    yy = node_y[bin_nodes].astype(np.float64, copy=False)
    xx = node_x[bin_nodes].astype(np.float64, copy=False)
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    order = np.argsort(d2, kind="mergesort")
    top = bin_nodes[order[: min(int(order.size), 64)]]
    return int(rng.choice(top))


@dataclass(frozen=True)
class SampleCfg:
    K: int
    temperature: float
    max_steps: int


def _plot_case(
    *,
    out_png: Path,
    out_pdf: Path,
    node_y: np.ndarray,
    node_x: np.ndarray,
    gt_seq: Sequence[int],
    pred_paths: Sequence[Sequence[int]],
    gt_wp: Optional[Sequence[int]],
    pred_wps: Sequence[Sequence[int]],
    title: str,
) -> None:
    gt = np.asarray(gt_seq, dtype=np.int64)
    with paper_style():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.2))
        ax.plot(node_x[gt], node_y[gt], color="black", lw=2.0, alpha=0.9, label="GT")
        for p in pred_paths:
            pp = np.asarray(p, dtype=np.int64)
            ax.plot(node_x[pp], node_y[pp], color=OKABE_ITO["blue"], lw=1.0, alpha=0.22)
        # GT waypoints (optional)
        if gt_wp is not None and len(gt_wp) >= 2:
            w = np.asarray(gt_wp, dtype=np.int64)
            ax.scatter(node_x[w], node_y[w], s=20, c="black", edgecolors="white", linewidths=0.5, zorder=10, label="GT-WP")
        # Pred waypoint clouds
        for wps in pred_wps:
            if not wps:
                continue
            w = np.asarray(wps, dtype=np.int64)
            ax.scatter(node_x[w], node_y[w], s=8, c=OKABE_ITO["sky_blue"], alpha=0.25, linewidths=0.0)

        ax.scatter([node_x[gt[0]]], [node_y[gt[0]]], s=60, c="black", edgecolors="white", linewidths=1.0, zorder=10)
        ax.scatter([node_x[gt[-1]]], [node_y[gt[-1]]], s=60, c="black", marker="s", edgecolors="white", linewidths=1.0, zorder=10)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
        save_figure(fig, out_png, dpi=250)
        save_figure(fig, out_pdf)
        plt.close(fig)


def run(
    *,
    checkpoint: Path,
    road_graph_npz: Path,
    paths_graph_npz: Path,
    waypoints_npz: Optional[Path],
    out_dir: Path,
    K: int,
    temperature: float,
    num_routes: int,
    baseline_k: int,
    tz_offset_hours: float,
    seed: int,
    viz_cases: int,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "report.json"

    # Graph (A* + optional Yen baseline)
    g = _load_graph_npz(road_graph_npz)
    H = int(g.grid.H)
    W = int(g.grid.W)

    raw = np.load(str(road_graph_npz), allow_pickle=True)
    node_y = np.asarray(raw["node_y"], dtype=np.float32).reshape(-1)
    node_x = np.asarray(raw["node_x"], dtype=np.float32).reshape(-1)
    edge_u = np.asarray(raw["edge_u"], dtype=np.int32).reshape(-1)
    edge_tier = np.asarray(raw["edge_tier"], dtype=np.uint8).reshape(-1)
    n_nodes = int(node_y.size)

    node_tier_min = _build_node_tier_min(n_nodes, edge_u=edge_u, edge_tier=edge_tier)

    p = np.load(str(paths_graph_npz), allow_pickle=True)
    node_seq_pad = np.asarray(p["node_seq_pad"], dtype=np.int32)
    node_seq_len = np.asarray(p["node_seq_len"], dtype=np.int32).reshape(-1)
    start_t = np.asarray(p["start_t"], dtype=np.int64).reshape(-1)
    start_node = np.asarray(p["start_node"], dtype=np.int32).reshape(-1)
    dest_node = np.asarray(p["dest_node"], dtype=np.int32).reshape(-1)
    route_city = np.asarray(p["route_city"], dtype=np.int8).reshape(-1) if "route_city" in p.files else np.zeros_like(start_node, dtype=np.int8)
    n_routes_total = int(start_node.size)

    gt_wp_seq = None
    if waypoints_npz is not None and waypoints_npz.exists():
        w = np.load(str(waypoints_npz), allow_pickle=True)
        if "wp_seq" in w.files:
            gt_wp_seq = np.asarray(w["wp_seq"], dtype=np.int32)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    model_cfg = ckpt.get("model_cfg") or {}
    num_steps = int(model_cfg.get("num_steps", 4))
    n_classes = int(model_cfg.get("n_classes", 1024))
    wp_bin = int(model_cfg.get("wp_bin", 32))
    num_cities = int(model_cfg.get("num_cities", 2))

    model = ARGraphWaypointBin(cfg=WaypointBinARConfig(hidden_dim=int(model_cfg.get("hidden_dim", 256)), num_cities=int(num_cities)), n_classes=n_classes, num_steps=num_steps).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    node_yx = torch.from_numpy(np.stack([node_y, node_x], axis=1).astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)
    node_tier_min_t = torch.from_numpy(node_tier_min.astype(np.int64, copy=False)).to(device=device, dtype=torch.long)
    tf = _time_features(start_t, tz_offset_hours=float(tz_offset_hours))

    # Bin index (for sampling bins -> nodes)
    bin_to_nodes, n_by, n_bx = _build_bin_to_nodes(node_y=node_y, node_x=node_x, wp_bin=wp_bin, H=H, W=W)
    if int(n_by * n_bx) != int(n_classes):
        # Keep going but report mismatch.
        pass

    rng = np.random.default_rng(int(seed))
    pick = rng.choice(n_routes_total, size=int(min(int(num_routes), n_routes_total)), replace=False)
    pick = np.sort(pick.astype(np.int64))

    od_cache: Dict[Tuple[int, int, int], List[List[int]]] = {}

    rows = []
    best_j_list = []
    succ_rate_list = []
    base_best_list = []

    cfg_s = SampleCfg(K=int(K), temperature=float(temperature), max_steps=4096)

    viz = 0
    for rid in tqdm(pick.tolist(), desc="sample_wp_astar", dynamic_ncols=True):
        gt_seq = _seq_from_pad(node_seq_pad, node_seq_len, int(rid))
        if len(gt_seq) < 2:
            continue
        gt_es = _edge_set(gt_seq)
        s = int(start_node[int(rid)])
        d = int(dest_node[int(rid)])
        city = int(route_city[int(rid)])
        time_feat = torch.from_numpy(tf[int(rid) : int(rid) + 1].astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)

        pred_paths: List[List[int]] = []
        pred_wps: List[List[int]] = []
        succ = 0
        best_j = 0.0

        for k in range(int(cfg_s.K)):
            cur = int(s)
            wps: List[int] = []
            ok = True
            # sample waypoint bins
            for step in range(int(num_steps)):
                with torch.no_grad():
                    logits, _ = model(
                        node_yx=node_yx,
                        node_tier_min=node_tier_min_t,
                        cur=torch.tensor([cur], device=device, dtype=torch.long),
                        dest=torch.tensor([d], device=device, dtype=torch.long),
                        time_feat=time_feat,
                        route_city=torch.tensor([city], device=device, dtype=torch.long),
                        step_idx=torch.tensor([step], device=device, dtype=torch.long),
                    )
                    logits_np = logits.detach().cpu().numpy().reshape(-1)

                temp = float(cfg_s.temperature)
                if temp <= 0:
                    cls = int(np.argmax(logits_np))
                else:
                    prob = _softmax_np(logits_np / max(1e-6, temp))
                    cls = int(rng.choice(int(prob.size), p=prob))

                by = int(cls // int(n_bx))
                bx = int(cls % int(n_bx))
                cand_nodes = bin_to_nodes[int(cls)]
                nxt = _pick_node_in_bin(bin_nodes=cand_nodes, node_y=node_y, node_x=node_x, by=by, bx=bx, wp_bin=wp_bin, rng=rng)
                if nxt is None:
                    ok = False
                    break
                wps.append(int(nxt))
                cur = int(nxt)

            pred_wps.append(wps)

            if not ok:
                pred_paths.append([s])
                continue

            # Connect with A*
            full: List[int] = [int(s)]
            prev = int(s)
            for nxt in wps + [int(d)]:
                _, seg = _astar(g, start=int(prev), goal=int(nxt))
                if not seg:
                    ok = False
                    break
                # avoid duplicating prev
                full.extend([int(x) for x in seg[1:]])
                prev = int(nxt)
            if ok and full and full[-1] == int(d):
                succ += 1
            pred_paths.append(full)
            j = _jaccard_edges(_edge_set(full), gt_es)
            best_j = float(max(best_j, j))

        succ_rate = float(succ) / float(max(1, int(cfg_s.K)))

        base_best = None
        if int(baseline_k) > 0:
            key = (s, d, int(baseline_k))
            if key not in od_cache:
                od_cache[key] = k_shortest_paths_yen(g, start=s, goal=d, K=int(baseline_k))
            bj = 0.0
            for path in od_cache[key]:
                bj = max(bj, _jaccard_edges(_edge_set(path), gt_es))
            base_best = float(bj)

        rows.append(
            {
                "route_id": int(rid),
                "city": int(city),
                "start": int(s),
                "dest": int(d),
                "gt_len": int(len(gt_seq)),
                "best_jaccard": float(best_j),
                "success_rate": float(succ_rate),
                "baseline_best_jaccard_kshortest": (float(base_best) if base_best is not None else None),
            }
        )
        best_j_list.append(float(best_j))
        succ_rate_list.append(float(succ_rate))
        if base_best is not None:
            base_best_list.append(float(base_best))

        if int(viz_cases) > 0 and viz < int(viz_cases):
            name = f"case_{int(viz):02d}_rid{int(rid)}"
            gt_wp = None
            if gt_wp_seq is not None and int(rid) < int(gt_wp_seq.shape[0]):
                gt_wp = [int(x) for x in gt_wp_seq[int(rid)].astype(np.int64, copy=False).tolist() if int(x) >= 0]
            else:
                # fallback: compute GT waypoints for reference (same K as model)
                pts = np.stack([node_y[np.asarray(gt_seq, dtype=np.int64)], node_x[np.asarray(gt_seq, dtype=np.int64)]], axis=1).astype(np.float32, copy=False)
                idx = pick_waypoint_indices_rdp_turn_fixed_k(pts, k=int(num_steps), turn_alpha=1.0)
                gt_wp = [int(gt_seq[0])] + [int(gt_seq[int(j)]) for j in idx.tolist()] + [int(gt_seq[-1])]

            _plot_case(
                out_png=out_dir / f"{name}.png",
                out_pdf=out_dir / f"{name}.pdf",
                node_y=node_y,
                node_x=node_x,
                gt_seq=gt_seq,
                pred_paths=pred_paths,
                gt_wp=gt_wp,
                pred_wps=pred_wps,
                title=f"rid={int(rid)} bestJ={best_j:.3f} succ={succ_rate:.2f}",
            )
            viz += 1

    def _q(a: List[float], p: float) -> Optional[float]:
        if not a:
            return None
        return float(np.percentile(np.asarray(a, dtype=np.float64), p))

    report = {
        "ok": True,
        "task": "sample_graph_ar_waypoints_astar",
        "inputs": {
            "checkpoint": str(checkpoint),
            "road_graph_npz": str(road_graph_npz),
            "paths_graph_npz": str(paths_graph_npz),
            "waypoints_npz": (str(waypoints_npz) if waypoints_npz is not None else None),
        },
        "config": {
            "K": int(K),
            "temperature": float(temperature),
            "wp_bin": int(wp_bin),
            "num_waypoints": int(num_steps),
            "num_routes": int(num_routes),
            "baseline_k": int(baseline_k),
            "seed": int(seed),
            "tz_offset_hours": float(tz_offset_hours),
        },
        "stats": {
            "num_routes_sampled": int(len(rows)),
            "success_rate": {"mean": float(np.mean(np.asarray(succ_rate_list, dtype=np.float64))) if succ_rate_list else None, "p50": _q(succ_rate_list, 50), "p90": _q(succ_rate_list, 90)},
            "best_jaccard": {"mean": float(np.mean(np.asarray(best_j_list, dtype=np.float64))) if best_j_list else None, "p50": _q(best_j_list, 50), "p90": _q(best_j_list, 90)},
            "baseline_best_jaccard_kshortest": {
                "mean": float(np.mean(np.asarray(base_best_list, dtype=np.float64))) if base_best_list else None,
                "p50": _q(base_best_list, 50) if base_best_list else None,
                "p90": _q(base_best_list, 90) if base_best_list else None,
                "n": int(len(base_best_list)),
            },
        },
        "rows": rows[:200],
        "outputs": {"report_json": str(report_json), "out_dir": str(out_dir)},
        "meta": {"created_at": datetime.now(tz=TZ_SHANGHAI).isoformat()},
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample waypoint-AR (bin) model, then connect waypoints with A* to form full graph corridors.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--road_graph_npz", type=Path, required=True)
    p.add_argument("--paths_graph_npz", type=Path, required=True)
    p.add_argument("--waypoints_npz", type=Path, default=None, help="Optional GT waypoint npz for visualization.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--num_routes", type=int, default=200)
    p.add_argument("--baseline_k", type=int, default=0)
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--viz_cases", type=int, default=10)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))
    report = run(
        checkpoint=Path(args.checkpoint),
        road_graph_npz=Path(args.road_graph_npz),
        paths_graph_npz=Path(args.paths_graph_npz),
        waypoints_npz=(Path(args.waypoints_npz) if args.waypoints_npz is not None else None),
        out_dir=Path(args.out_dir),
        K=int(args.K),
        temperature=float(args.temperature),
        num_routes=int(args.num_routes),
        baseline_k=int(args.baseline_k),
        tz_offset_hours=float(args.tz_offset_hours),
        seed=int(args.seed),
        viz_cases=int(args.viz_cases),
    )
    compact = {
        "ok": True,
        "out_dir": report["outputs"]["out_dir"],
        "num_routes_sampled": report["stats"]["num_routes_sampled"],
        "best_jaccard_mean": report["stats"]["best_jaccard"]["mean"],
        "success_rate_mean": report["stats"]["success_rate"]["mean"],
        "report_json": report["outputs"]["report_json"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
