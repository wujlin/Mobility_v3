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

from src.data.road_graph.gate_candidate_paths_from_routes_npz import _astar, k_shortest_paths_yen, _load_graph_npz
from src.models.road_graph import ARDecisionConfig, ARGraphDecisionMarkov
from src.training.train_graph_ar_decision import _time_features
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
    if denom <= 0:
        return 0.0
    return float(inter) / float(denom)


@dataclass(frozen=True)
class SampleCfg:
    K: int
    temperature: float
    max_steps: int
    avoid_backtrack: bool
    avoid_cycles: bool


def _neighbors(ptr: np.ndarray, idx: np.ndarray, tier: np.ndarray, u: int) -> Tuple[np.ndarray, np.ndarray]:
    s = int(ptr[int(u)])
    e = int(ptr[int(u) + 1])
    if e <= s:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    vv = idx[s:e].astype(np.int64, copy=False)
    tt = tier[s:e].astype(np.int64, copy=False)
    return vv, tt


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


def sample_one(
    *,
    model: ARGraphDecisionMarkov,
    node_yx: torch.Tensor,
    ptr: np.ndarray,
    idx: np.ndarray,
    tier: np.ndarray,
    time_feat: torch.Tensor,  # (1,5)
    start: int,
    dest: int,
    cfg: SampleCfg,
    seed: int,
) -> Tuple[List[int], bool]:
    rng = np.random.default_rng(int(seed))
    cur = int(start)
    dest = int(dest)
    seq = [cur]
    prev: Optional[int] = None
    visited = {cur}
    for _ in range(int(cfg.max_steps)):
        if cur == dest:
            return seq, True
        vv, tt = _neighbors(ptr, idx, tier, cur)
        if vv.size == 0:
            return seq, False
        mask = np.ones((vv.size,), dtype=np.uint8)
        if cfg.avoid_backtrack and prev is not None:
            mask &= (vv != int(prev)).astype(np.uint8)
        if cfg.avoid_cycles:
            mask &= np.asarray([0 if int(x) in visited else 1 for x in vv.tolist()], dtype=np.uint8)
        keep = np.nonzero(mask)[0].astype(np.int64)
        if keep.size == 0:
            # fallback: allow backtrack/cycle if stuck
            keep = np.arange(vv.size, dtype=np.int64)
        vv2 = vv[keep]
        tt2 = tt[keep]

        with torch.no_grad():
            logits, _ = model.score_neighbors(
                node_yx=node_yx,
                cur=torch.tensor([cur], device=node_yx.device, dtype=torch.long),
                dest=torch.tensor([dest], device=node_yx.device, dtype=torch.long),
                neigh=torch.from_numpy(vv2.reshape(1, -1)).to(device=node_yx.device, dtype=torch.long),
                neigh_tier=torch.from_numpy(tt2.reshape(1, -1)).to(device=node_yx.device, dtype=torch.long),
                time_feat=time_feat,
            )
            logits_np = logits.detach().cpu().numpy().reshape(-1)

        temp = float(cfg.temperature)
        if temp <= 0:
            pick = int(np.argmax(logits_np))
        else:
            prob = _softmax_np(logits_np / max(1e-6, temp))
            pick = int(rng.choice(int(prob.size), p=prob))
        nxt = int(vv2[pick])
        prev = cur
        cur = nxt
        seq.append(cur)
        visited.add(cur)
    return seq, (cur == dest)


def _plot_case(
    *,
    out_png: Path,
    out_pdf: Path,
    node_y: np.ndarray,
    node_x: np.ndarray,
    gt_seq: Sequence[int],
    samples: Sequence[Sequence[int]],
    title: str,
) -> None:
    gt = np.asarray(gt_seq, dtype=np.int64)
    with paper_style():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.2))
        ax.plot(node_x[gt], node_y[gt], color="black", lw=2.0, alpha=0.9, label="GT")
        for s in samples:
            ss = np.asarray(s, dtype=np.int64)
            ax.plot(node_x[ss], node_y[ss], color=OKABE_ITO["blue"], lw=1.0, alpha=0.25)
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
    out_dir: Path,
    K: int,
    temperature: float,
    max_steps: int,
    num_routes: int,
    baseline_k: int,
    tz_offset_hours: float,
    seed: int,
    viz_cases: int,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "report.json"

    # Graph loader for baseline k-shortest (uses edge_w_m + A* heuristic).
    g_base = _load_graph_npz(road_graph_npz)

    raw = np.load(str(road_graph_npz), allow_pickle=True)
    node_y = np.asarray(raw["node_y"], dtype=np.float32).reshape(-1)
    node_x = np.asarray(raw["node_x"], dtype=np.float32).reshape(-1)
    edge_u = np.asarray(raw["edge_u"], dtype=np.int32).reshape(-1)
    edge_v = np.asarray(raw["edge_v"], dtype=np.int32).reshape(-1)
    edge_tier = np.asarray(raw["edge_tier"], dtype=np.uint8).reshape(-1)
    n_nodes = int(node_y.size)

    # CSR neighbors for fast sampling.
    order = np.argsort(edge_u.astype(np.int64), kind="mergesort")
    u = edge_u[order].astype(np.int64, copy=False)
    v = edge_v[order].astype(np.int32, copy=False)
    t = edge_tier[order].astype(np.uint8, copy=False)
    cnt = np.bincount(u.astype(np.int64, copy=False), minlength=n_nodes).astype(np.int64, copy=False)
    ptr = np.zeros((n_nodes + 1,), dtype=np.int64)
    ptr[1:] = np.cumsum(cnt)

    p = np.load(str(paths_graph_npz), allow_pickle=True)
    node_seq_pad = np.asarray(p["node_seq_pad"], dtype=np.int32)
    node_seq_len = np.asarray(p["node_seq_len"], dtype=np.int32).reshape(-1)
    start_t = np.asarray(p["start_t"], dtype=np.int64).reshape(-1)
    start_node = np.asarray(p["start_node"], dtype=np.int32).reshape(-1)
    dest_node = np.asarray(p["dest_node"], dtype=np.int32).reshape(-1)
    n_routes_total = int(start_node.size)

    rng = np.random.default_rng(int(seed))
    pick = rng.choice(n_routes_total, size=int(min(int(num_routes), n_routes_total)), replace=False)
    pick = np.sort(pick.astype(np.int64))

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    cfg_d = ckpt.get("cfg") or {}
    model = ARGraphDecisionMarkov(
        cfg=ARDecisionConfig(hidden_dim=int(cfg_d.get("hidden_dim", 256)), edge_tier_emb_dim=int(cfg_d.get("edge_tier_emb_dim", 8)))
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    node_yx = torch.from_numpy(np.stack([node_y, node_x], axis=1).astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)
    tf = _time_features(start_t, tz_offset_hours=float(tz_offset_hours))
    tf_t = torch.from_numpy(tf).to(device=device, dtype=torch.float32)

    # Baseline cache: OD -> K paths
    od_cache: Dict[Tuple[int, int, int], List[List[int]]] = {}

    rows = []
    best_j_list = []
    succ_rate_list = []
    base_best_list = []

    cfg_s = SampleCfg(K=int(K), temperature=float(temperature), max_steps=int(max_steps), avoid_backtrack=True, avoid_cycles=True)

    viz = 0
    for rid in tqdm(pick.tolist(), desc="sample", dynamic_ncols=True):
        gt_seq = _seq_from_pad(node_seq_pad, node_seq_len, int(rid))
        if len(gt_seq) < 2:
            continue
        gt_es = _edge_set(gt_seq)
        s = int(start_node[int(rid)])
        d = int(dest_node[int(rid)])
        time_feat = tf_t[int(rid) : int(rid) + 1]

        samples = []
        succ = 0
        best_j = 0.0
        for k in range(int(K)):
            seq, ok = sample_one(
                model=model,
                node_yx=node_yx,
                ptr=ptr,
                idx=v,
                tier=t,
                time_feat=time_feat,
                start=s,
                dest=d,
                cfg=cfg_s,
                seed=int(seed) + int(rid) * 1000 + int(k),
            )
            samples.append(seq)
            succ += int(ok)
            j = _jaccard_edges(_edge_set(seq), gt_es)
            best_j = float(max(best_j, j))

        succ_rate = float(succ) / float(max(1, int(K)))

        # Baseline best-jaccard with k-shortest (optional)
        base_best = None
        if int(baseline_k) > 0:
            key = (s, d, int(baseline_k))
            if key not in od_cache:
                od_cache[key] = k_shortest_paths_yen(g_base, start=s, goal=d, K=int(baseline_k))
            bj = 0.0
            for path in od_cache[key]:
                bj = max(bj, _jaccard_edges(_edge_set(path), gt_es))
            base_best = float(bj)

        rows.append(
            {
                "route_id": int(rid),
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
            _plot_case(
                out_png=out_dir / f"{name}.png",
                out_pdf=out_dir / f"{name}.pdf",
                node_y=node_y,
                node_x=node_x,
                gt_seq=gt_seq,
                samples=samples,
                title=f"rid={int(rid)} bestJ={best_j:.3f} succ={succ_rate:.2f}",
            )
            viz += 1

    def _q(a: List[float], p: float) -> Optional[float]:
        if not a:
            return None
        return float(np.percentile(np.asarray(a, dtype=np.float64), p))

    report = {
        "ok": True,
        "task": "sample_graph_ar_decision",
        "inputs": {"checkpoint": str(checkpoint), "road_graph_npz": str(road_graph_npz), "paths_graph_npz": str(paths_graph_npz)},
        "config": {
            "K": int(K),
            "temperature": float(temperature),
            "max_steps": int(max_steps),
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
    p = argparse.ArgumentParser(description="T3: Sample AR-on-graph decision model and evaluate corridor coverage (edge Jaccard).")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--road_graph_npz", type=Path, required=True)
    p.add_argument("--paths_graph_npz", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_steps", type=int, default=2048)
    p.add_argument("--num_routes", type=int, default=200)
    p.add_argument("--baseline_k", type=int, default=0, help="Optional k-shortest baseline K for best-jaccard comparison (0=skip).")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--viz_cases", type=int, default=10, help="Number of cases to render GT vs samples (png+pdf).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))
    report = run(
        checkpoint=Path(args.checkpoint),
        road_graph_npz=Path(args.road_graph_npz),
        paths_graph_npz=Path(args.paths_graph_npz),
        out_dir=Path(args.out_dir),
        K=int(args.K),
        temperature=float(args.temperature),
        max_steps=int(args.max_steps),
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

