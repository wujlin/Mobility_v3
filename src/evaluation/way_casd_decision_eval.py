from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.models.way_casd.latent_flow import LatentFlowCfg, LatentFlowMatching
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz, make_way_feature_tensors

TZ_SHANGHAI = timezone(timedelta(hours=8))

LatentSource = Literal["gt", "flow"]
DecodeMethod = Literal["greedy", "beam"]


@dataclass(frozen=True)
class EvalCfg:
    seed: int
    device: str
    tz_offset_hours: float

    n_routes: int  # per city
    n_samples_per_route: int
    max_way_len: int
    max_decode_len: int
    beam_size: int
    solver_steps: Optional[int]
    decode_max_candidates: int  # -1=use model cfg; 0=all successors; >0=override
    decode_candidate_policy: str  # "first" | "destdist"
    decode_include_dest_if_successor: bool


def _infer_decoder_config_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """
    从 checkpoint 的 state_dict 推断 decoder 配置（尤其是 past-context 相关）。

    目的：避免把“带 past-context 的新模型”当成“旧模型”来评估，导致到达率被系统性低估。
    """
    cfg: Dict[str, object] = {}

    # use_dest_dist：通过 scorer 第一层的输入维度推断
    w = state.get("decoder.scorer.0.weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        hidden = int(w.shape[0])
        in_dim = int(w.shape[1])
        # in_dim == 3*hidden (+ optional scalar features)
        cfg["decoder_use_dest_dist"] = (int(in_dim - hidden * 3) != 0)
    else:
        cfg["decoder_use_dest_dist"] = True

    cfg["decoder_use_cross_attn"] = any(str(k).startswith("decoder.cross_attn.") for k in state.keys())
    cfg["decoder_use_step_emb"] = any(str(k).startswith("decoder.step_emb.") for k in state.keys())
    cfg["decoder_use_dest_query"] = any(str(k).startswith("decoder.dest_proj.") for k in state.keys())
    cfg["decoder_use_dir_query"] = any(str(k).startswith("decoder.dir_query_proj.") for k in state.keys())
    cfg["decoder_use_cand_query"] = any(str(k).startswith("decoder.cand_query_proj.") for k in state.keys())
    cfg["decoder_use_past_context"] = any(str(k).startswith("decoder.past_encoder.") for k in state.keys())

    pe = state.get("decoder.past_encoder.pos_emb.weight", None)
    if isinstance(pe, torch.Tensor) and pe.ndim == 2:
        cfg["decoder_past_k"] = int(pe.shape[0])
    else:
        cfg["decoder_past_k"] = 8

    return cfg


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _hour_from_unix(start_t: np.ndarray, tz_offset_hours: float) -> np.ndarray:
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    sec = ((start_t + tz_sec) % 86400).astype(np.int64, copy=False)
    return (sec // 3600).astype(np.int64, copy=False)


def _dow_from_unix(start_t: np.ndarray, tz_offset_hours: float) -> np.ndarray:
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    days = ((start_t + tz_sec) // 86400).astype(np.int64, copy=False)
    return ((days + 3) % 7).astype(np.int64, copy=False)


def _quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, float(q)))


def _summary_stats(x: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(list(x), dtype=np.float64)
    return {
        "mean": float(np.mean(a)) if a.size else float("nan"),
        "p50": _quantile(a, 0.50),
        "p90": _quantile(a, 0.90),
        "p99": _quantile(a, 0.99),
    }


def _jaccard(a: List[int], b: List[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb)) / float(len(sa | sb))


def _len_buckets() -> List[Tuple[str, int, int]]:
    # inclusive ranges [lo, hi]
    return [
        ("L<=15", 0, 15),
        ("16-30", 16, 30),
        ("31-60", 31, 60),
        ("61-128", 61, 128),
    ]


def _bucket_key(L: int) -> str:
    for name, lo, hi in _len_buckets():
        if int(lo) <= int(L) <= int(hi):
            return str(name)
    return "other"


def _decode_one(
    *,
    ae: WayCASDAutoEncoder,
    z: torch.Tensor,  # (B,L,d)
    route_cond: Dict[str, torch.Tensor],  # (B,*)
    start_way: torch.Tensor,  # (B,)
    dest_way: torch.Tensor,  # (B,)
    method: DecodeMethod,
    beam_size: int,
    max_decode_len: int,
    decode_max_candidates: int,
    decode_candidate_policy: str,
    decode_include_dest_if_successor: bool,
) -> List[List[int]]:
    max_candidates = None if int(decode_max_candidates) < 0 else int(decode_max_candidates)
    if str(method) == "beam":
        return ae.decoder.beam_search(
            way_embedder=ae.way_enc,
            latent_tokens=z,
            route_cond=route_cond,
            start_way=start_way,
            dest_way=dest_way,
            beam_size=int(beam_size),
            max_len=int(max_decode_len),
            max_candidates=max_candidates,
            candidate_policy=str(decode_candidate_policy),
            include_dest_if_successor=bool(decode_include_dest_if_successor),
        )
    return ae.decoder.greedy_decode(
        way_embedder=ae.way_enc,
        latent_tokens=z,
        route_cond=route_cond,
        start_way=start_way,
        dest_way=dest_way,
        max_len=int(max_decode_len),
        max_candidates=max_candidates,
        candidate_policy=str(decode_candidate_policy),
        include_dest_if_successor=bool(decode_include_dest_if_successor),
    )


def _is_reachable_bfs(ptr: np.ndarray, idx: np.ndarray, start: int, dest: int, *, max_visits: int = 200000) -> bool:
    """
    Reachability check on a CSR way-graph. Intended for small graphs (~10^4 ways).
    """
    ptr = np.asarray(ptr, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    n = int(ptr.size) - 1
    if start < 0 or dest < 0 or start >= n or dest >= n:
        return False
    if start == dest:
        return True

    visited = np.zeros((n,), dtype=np.bool_)
    q: List[int] = [int(start)]
    visited[int(start)] = True
    seen = 0
    while q:
        u = q.pop()
        s = int(ptr[u])
        e = int(ptr[u + 1])
        for v in idx[s:e].tolist():
            vv = int(v)
            if not visited[vv]:
                if vv == int(dest):
                    return True
                visited[vv] = True
                q.append(vv)
        seen += 1
        if seen >= int(max_visits):
            break
    return bool(visited[int(dest)])


@torch.no_grad()
def _eval_city(
    *,
    city: int,
    cfg: EvalCfg,
    routes_npz: Path,
    ae: WayCASDAutoEncoder,
    flow: Optional[LatentFlowMatching],
    latent_sources: Sequence[LatentSource],
    decode_methods: Sequence[DecodeMethod],
    way_adj_ptr: np.ndarray,
    way_adj_idx: np.ndarray,
    way_center_y: np.ndarray,
    way_center_x: np.ndarray,
    device: torch.device,
    log_every: int = 50,
) -> Dict[str, object]:
    routes = load_way_routes_npz(Path(routes_npz))
    N = int(routes.way_seq_len.shape[0])
    keep = (routes.route_city.astype(np.int64) == int(city)) & (routes.way_seq_len > 1) & (routes.way_seq_len <= int(cfg.max_way_len))
    ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
    if ids.size == 0:
        return {"city": int(city), "n_candidates": 0, "results": {}}

    rng = np.random.default_rng(int(cfg.seed) + 101 * int(city))
    rng.shuffle(ids)
    pick = ids[: min(int(cfg.n_routes), int(ids.size))]

    # Preload per-route meta
    start_t = routes.start_t[pick].astype(np.int64, copy=False)
    hour = _hour_from_unix(start_t, float(cfg.tz_offset_hours))
    dow = _dow_from_unix(start_t, float(cfg.tz_offset_hours))
    start_pos = routes.start_pos[pick].astype(np.float32, copy=False)
    dest_pos = routes.dest_pos[pick].astype(np.float32, copy=False)
    start_way = routes.start_way[pick].astype(np.int64, copy=False)
    dest_way = routes.dest_way[pick].astype(np.int64, copy=False)
    gt_len = routes.way_seq_len[pick].astype(np.int64, copy=False)

    # Preload gt sequences
    gt_seqs: List[List[int]] = []
    for rid, L in zip(pick.tolist(), gt_len.tolist()):
        s = int(routes.way_seq_ptr[int(rid)])
        e = s + int(L)
        gt = routes.way_seq_idx[s:e].astype(np.int64, copy=False).tolist()
        gt_seqs.append([int(x) for x in gt])

    # Batched tensors for cond
    base_route_cond = {
        "start_pos": torch.as_tensor(start_pos, dtype=torch.float32, device=device),
        "dest_pos": torch.as_tensor(dest_pos, dtype=torch.float32, device=device),
        "hour": torch.as_tensor(hour, dtype=torch.long, device=device),
        "dow": torch.as_tensor(dow, dtype=torch.long, device=device),
        "route_city": torch.full((int(pick.size),), int(city), dtype=torch.long, device=device),
    }
    start_way_t = torch.as_tensor(start_way, dtype=torch.long, device=device)
    dest_way_t = torch.as_tensor(dest_way, dtype=torch.long, device=device)

    # ---------------- Diagnostics: GT consistency / reachability / candidate coverage ----------------
    B = int(pick.size)
    start_match = 0
    dest_match = 0
    reachable = 0
    for sw, dw, gt in zip(start_way.tolist(), dest_way.tolist(), gt_seqs):
        if gt:
            start_match += int(int(sw) == int(gt[0]))
            dest_match += int(int(dw) == int(gt[-1]))
        reachable += int(_is_reachable_bfs(way_adj_ptr, way_adj_idx, int(sw), int(dw)))

    k_eff = int(cfg.decode_max_candidates)
    if k_eff < 0:
        k_eff = int(ae.cfg.max_candidates)

    way_center_y = np.asarray(way_center_y, dtype=np.float64).reshape(-1)
    way_center_x = np.asarray(way_center_x, dtype=np.float64).reshape(-1)

    def _select_cands(prev: int, dest_yx: np.ndarray) -> np.ndarray:
        s = int(way_adj_ptr[int(prev)])
        e = int(way_adj_ptr[int(prev) + 1])
        succ = np.asarray(way_adj_idx[s:e], dtype=np.int64)
        if k_eff <= 0 or succ.size <= k_eff:
            return succ
        if str(cfg.decode_candidate_policy).lower().strip() == "destdist":
            cy = way_center_y[succ]
            cx = way_center_x[succ]
            dy = cy - float(dest_yx[0])
            dx = cx - float(dest_yx[1])
            dist = dy * dy + dx * dx
            order = np.argsort(dist)
            return succ[order[:k_eff]]
        return succ[:k_eff]

    n_trans_total = 0
    n_trans_in = 0
    n_final_total = 0
    n_final_in = 0
    prev_degs: List[int] = []
    for gt, dpos in zip(gt_seqs, dest_pos):
        if len(gt) <= 1:
            continue
        dpos_yx = np.asarray(dpos, dtype=np.float64).reshape(2)
        for j in range(1, len(gt)):
            prev = int(gt[j - 1])
            tgt = int(gt[j])
            s = int(way_adj_ptr[prev])
            e = int(way_adj_ptr[prev + 1])
            prev_degs.append(int(e - s))
            cands = _select_cands(prev, dpos_yx)
            n_trans_total += 1
            n_trans_in += int(int(tgt) in set(cands.tolist()))
        prev = int(gt[-2])
        tgt = int(gt[-1])
        cands = _select_cands(prev, dpos_yx)
        n_final_total += 1
        n_final_in += int(int(tgt) in set(cands.tolist()))

    def _deg_q(q: float) -> float:
        if not prev_degs:
            return float("nan")
        a = np.asarray(prev_degs, dtype=np.float64)
        return float(np.quantile(a, float(q)))

    diag = {
        "gt_start_match_rate": float(start_match) / float(max(1, B)),
        "gt_dest_match_rate": float(dest_match) / float(max(1, B)),
        "dest_reachable_rate": float(reachable) / float(max(1, B)),
        "decode_k_eff": int(k_eff),
        "gt_next_in_decode_cands_rate": float(n_trans_in) / float(max(1, n_trans_total)),
        "gt_final_in_decode_cands_rate": float(n_final_in) / float(max(1, n_final_total)),
        "gt_prev_out_deg": {"p50": _deg_q(0.50), "p90": _deg_q(0.90), "max": float(max(prev_degs)) if prev_degs else float("nan")},
    }

    # Pad GT for encoder
    maxL = int(gt_len.max())
    pad = np.full((B, maxL), -1, dtype=np.int64)
    for i, seq in enumerate(gt_seqs):
        Li = min(int(len(seq)), int(maxL))
        pad[i, :Li] = np.asarray(seq[:Li], dtype=np.int64)
    way_pad = torch.as_tensor(pad, dtype=torch.long, device=device)

    # z_enc: (B,L_lat,d)
    z_enc, _ = ae.encode(way_pad)

    results: Dict[str, object] = {}
    for latent_src in latent_sources:
        if str(latent_src) == "flow" and flow is None:
            continue
        for dec in decode_methods:
            key = f"{latent_src}:{dec}"
            K = int(cfg.n_samples_per_route) if str(latent_src) == "flow" else 1

            route_success = []
            route_jacc_best = []
            route_len_ratio_best = []

            sample_success = []
            sample_hit_wall = []
            sample_dead_end = []
            sample_jacc = []
            sample_len_ratio = []

            bucket_stats: Dict[str, Dict[str, List[float]]] = {}
            for name, _lo, _hi in _len_buckets():
                bucket_stats[name] = {"succ": [], "jacc_best": [], "len_ratio_best": []}

            for i0 in range(0, B, 64):
                i1 = min(B, i0 + 64)
                # Slice batch routes
                z_b = z_enc[i0:i1]
                rc_b = {k: v[i0:i1] for k, v in base_route_cond.items()}
                sw_b = start_way_t[i0:i1]
                dw_b = dest_way_t[i0:i1]
                gt_b = gt_seqs[i0:i1]
                gtlen_b = gt_len[i0:i1]

                if str(latent_src) == "flow":
                    assert flow is not None
                    # Sample K latents per route (B*K, L, d)
                    rc_rep = {k: v.repeat_interleave(K, dim=0) for k, v in rc_b.items()}
                    z_flow = flow.sample(route_cond=rc_rep, solver_steps=cfg.solver_steps)
                    z_flow = z_flow.view(int(i1 - i0), int(K), int(z_flow.shape[1]), int(z_flow.shape[2]))
                else:
                    z_flow = None

                # For each sample, decode as a batch (B routes)
                pred_per_k: List[List[List[int]]] = []
                for k in range(K):
                    if z_flow is not None:
                        z_use = z_flow[:, k]
                    else:
                        z_use = z_b
                    preds = _decode_one(
                        ae=ae,
                        z=z_use,
                        route_cond=rc_b,
                        start_way=sw_b,
                        dest_way=dw_b,
                        method=dec,
                        beam_size=int(cfg.beam_size),
                        max_decode_len=int(cfg.max_decode_len),
                        decode_max_candidates=int(cfg.decode_max_candidates),
                        decode_candidate_policy=str(cfg.decode_candidate_policy),
                        decode_include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
                    )
                    pred_per_k.append(preds)

                # Aggregate per-route
                for bi in range(int(i1 - i0)):
                    gt_seq = gt_b[bi]
                    gtL = int(gtlen_b[bi])
                    max_len = int(cfg.max_decode_len) + 1
                    per_route_succ = False
                    best_j = -1.0
                    best_lr = float("nan")

                    for k in range(K):
                        pred_seq = pred_per_k[k][bi]
                        succ = bool(int(pred_seq[-1]) == int(dw_b[bi].item()))
                        j = _jaccard(gt_seq, pred_seq)
                        lr = float(len(pred_seq)) / float(max(1, gtL))

                        sample_success.append(float(succ))
                        sample_jacc.append(float(j))
                        sample_len_ratio.append(float(lr))

                        if (not succ) and (len(pred_seq) >= max_len):
                            sample_hit_wall.append(1.0)
                        else:
                            sample_hit_wall.append(0.0)
                        if (not succ) and (len(pred_seq) < max_len):
                            # greedy/beam stopped early -> likely dead-end (no succ candidates)
                            sample_dead_end.append(1.0)
                        else:
                            sample_dead_end.append(0.0)

                        if succ:
                            per_route_succ = True
                        if j > best_j:
                            best_j = float(j)
                            best_lr = float(lr)

                    route_success.append(float(per_route_succ))
                    route_jacc_best.append(float(best_j))
                    route_len_ratio_best.append(float(best_lr))

                    bk = _bucket_key(gtL)
                    if bk in bucket_stats:
                        bucket_stats[bk]["succ"].append(float(per_route_succ))
                        bucket_stats[bk]["jacc_best"].append(float(best_j))
                        bucket_stats[bk]["len_ratio_best"].append(float(best_lr))

                done = int(i1)
                if log_every > 0 and (done % int(log_every) == 0 or done == B):
                    print(f"[city{int(city)}] {key} done={done}/{B}")

            # Summaries
            out = {
                "n_routes": int(B),
                "n_samples_per_route": int(K),
                "route_success_rate": float(np.mean(np.asarray(route_success, dtype=np.float64))) if route_success else float("nan"),
                "sample_success_rate": float(np.mean(np.asarray(sample_success, dtype=np.float64))) if sample_success else float("nan"),
                "sample_hit_wall_rate": float(np.mean(np.asarray(sample_hit_wall, dtype=np.float64))) if sample_hit_wall else float("nan"),
                "sample_dead_end_rate": float(np.mean(np.asarray(sample_dead_end, dtype=np.float64))) if sample_dead_end else float("nan"),
                "jaccard_best": _summary_stats(route_jacc_best),
                "len_ratio_best": _summary_stats(route_len_ratio_best),
                "jaccard_per_sample": _summary_stats(sample_jacc),
                "len_ratio_per_sample": _summary_stats(sample_len_ratio),
                "by_gt_len": {
                    k: {
                        "n": int(len(v["succ"])),
                        "route_success_rate": float(np.mean(np.asarray(v["succ"], dtype=np.float64))) if v["succ"] else float("nan"),
                        "jaccard_best": _summary_stats(v["jacc_best"]),
                        "len_ratio_best": _summary_stats(v["len_ratio_best"]),
                    }
                    for k, v in bucket_stats.items()
                },
            }
            results[key] = out

    return {
        "city": int(city),
        "n_candidates": int(ids.size),
        "n_eval": int(pick.size),
        "picked_route_ids": pick.tolist(),
        "diag": diag,
        "results": results,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quantitative evaluation for Way-CASD Decision stage (Flow/Decoder).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--flow_ckpt", type=Path, default=None, help="Required if latent_sources includes 'flow'.")
    p.add_argument("--out_json", type=Path, required=True)

    p.add_argument("--latent_sources", type=str, nargs="+", default=["gt", "flow"], choices=["gt", "flow"])
    p.add_argument("--decode_methods", type=str, nargs="+", default=["greedy", "beam"], choices=["greedy", "beam"])
    p.add_argument("--beam_size", type=int, default=10)
    p.add_argument("--solver_steps", type=int, default=0, help="Override flow solver steps (0=use ckpt/default).")
    p.add_argument(
        "--decode_max_candidates",
        type=int,
        default=-1,
        help="Decode candidate cap: -1=use model cfg; 0=all successors; >0=override.",
    )
    p.add_argument("--decode_candidate_policy", type=str, default="first", choices=["first", "destdist"])
    p.add_argument(
        "--decode_include_dest_if_successor",
        action="store_true",
        help="If dest_way is a direct successor but truncated out, force-include it (decode-time only).",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--n_routes", type=int, default=500, help="Per city.")
    p.add_argument("--n_samples_per_route", type=int, default=4, help="Only used for latent_source=flow.")
    p.add_argument("--max_way_len", type=int, default=128)
    p.add_argument("--max_decode_len", type=int, default=160)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cfg = EvalCfg(
        seed=int(args.seed),
        device=str(args.device),
        tz_offset_hours=float(args.tz_offset_hours),
        n_routes=int(args.n_routes),
        n_samples_per_route=int(args.n_samples_per_route),
        max_way_len=int(args.max_way_len),
        max_decode_len=int(args.max_decode_len),
        beam_size=int(args.beam_size),
        solver_steps=(int(args.solver_steps) if int(args.solver_steps) > 0 else None),
        decode_max_candidates=int(args.decode_max_candidates),
        decode_candidate_policy=str(args.decode_candidate_policy),
        decode_include_dest_if_successor=bool(args.decode_include_dest_if_successor),
    )
    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # Load way graph + features
    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    way_features = load_way_features_from_npz(Path(args.way_features_npz), device=device)
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1

    # Load AE
    ckpt_ae = torch.load(str(args.ae_ckpt), map_location=device)
    ae_state = ckpt_ae["model_state_dict"] if isinstance(ckpt_ae, dict) and "model_state_dict" in ckpt_ae else ckpt_ae
    ae_cfg_dict = ckpt_ae.get("config", {}) if isinstance(ckpt_ae, dict) else {}
    inferred = _infer_decoder_config_from_state(ae_state) if isinstance(ae_state, dict) else {}
    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(ae_cfg_dict.get("d_model", 256)),
            n_latent=int(ae_cfg_dict.get("n_latent", 64)),
            n_heads=int(ae_cfg_dict.get("n_heads", 8)),
            dropout=float(ae_cfg_dict.get("dropout", 0.1)),
            max_candidates=int(ae_cfg_dict.get("max_candidates", 32)),
            max_len=int(ae_cfg_dict.get("max_len", cfg.max_way_len)),
            coord_scale=float(ae_cfg_dict.get("coord_scale", 1024.0)),
            decoder_use_dest_dist=bool(inferred.get("decoder_use_dest_dist", ae_cfg_dict.get("decoder_use_dest_dist", True))),
            decoder_use_cross_attn=bool(inferred.get("decoder_use_cross_attn", ae_cfg_dict.get("decoder_use_cross_attn", True))),
            decoder_n_cross_heads=int(ae_cfg_dict.get("decoder_n_cross_heads", 4)),
            decoder_use_step_emb=bool(inferred.get("decoder_use_step_emb", ae_cfg_dict.get("decoder_use_step_emb", False))),
            decoder_use_dest_query=bool(inferred.get("decoder_use_dest_query", ae_cfg_dict.get("decoder_use_dest_query", False))),
            decoder_use_dir_query=bool(inferred.get("decoder_use_dir_query", ae_cfg_dict.get("decoder_use_dir_query", False))),
            decoder_use_cand_query=bool(inferred.get("decoder_use_cand_query", ae_cfg_dict.get("decoder_use_cand_query", False))),
            decoder_use_past_context=bool(inferred.get("decoder_use_past_context", ae_cfg_dict.get("decoder_use_past_context", False))),
            decoder_past_k=int(inferred.get("decoder_past_k", ae_cfg_dict.get("decoder_past_k", 8))),
            decoder_past_n_layers=int(ae_cfg_dict.get("decoder_past_n_layers", 2)),
            decoder_past_n_heads=int(ae_cfg_dict.get("decoder_past_n_heads", 4)),
        ),
        way_features=way_features,
        way_adj_ptr=wg["way_adj_ptr"],
        way_adj_idx=wg["way_adj_idx"],
        n_highway_types=int(max(4, n_highway_types)),
    ).to(device)
    ckpt_strict_load_ok = True
    try:
        ae.load_state_dict(ae_state, strict=True)
    except Exception as e:
        ckpt_strict_load_ok = False
        print(f"[WARN] strict load_state_dict failed, fallback to strict=False: {e}")
        ae.load_state_dict(ae_state, strict=False)
    ae.eval()

    # Load Flow (optional)
    flow: Optional[LatentFlowMatching] = None
    latent_sources = [str(x) for x in args.latent_sources]
    if "flow" in latent_sources:
        if args.flow_ckpt is None:
            raise SystemExit("--flow_ckpt is required when latent_sources includes 'flow'")
        ckpt_f = torch.load(str(args.flow_ckpt), map_location=device)
        f_state = ckpt_f["model_state_dict"] if isinstance(ckpt_f, dict) and "model_state_dict" in ckpt_f else ckpt_f
        f_cfg = ckpt_f.get("config", {}) if isinstance(ckpt_f, dict) else {}
        flow = LatentFlowMatching(
            cfg=LatentFlowCfg(
                d_model=int(f_cfg.get("d_model", ae.cfg.d_model)),
                n_latent=int(f_cfg.get("n_latent", ae.cfg.n_latent)),
                n_layers=int(f_cfg.get("n_layers", 6)),
                n_heads=int(f_cfg.get("n_heads", 8)),
                dropout=float(f_cfg.get("dropout", 0.1)),
                noise_sigma=float(f_cfg.get("noise_sigma", 1.0)),
                solver_steps=int(f_cfg.get("solver_steps", 20)),
            ),
            cond_cfg=ae.decoder.cond_enc.cfg,  # reuse same cond cfg (OD/time/city)
        ).to(device)
        flow.load_state_dict(f_state, strict=False)
        flow.eval()

    rep = {
        "ok": True,
        "task": "way_casd_decision_eval",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
            "flow_ckpt": str(args.flow_ckpt) if args.flow_ckpt is not None else None,
        },
        "ckpt_strict_load_ok": bool(ckpt_strict_load_ok),
        "ckpt_decoder_cfg_inferred": dict(inferred),
        "per_city": [],
    }

    per_city = []
    for city in (0, 1):
        per_city.append(
            _eval_city(
                city=int(city),
                cfg=cfg,
                routes_npz=Path(args.way_routes_npz),
                ae=ae,
                flow=flow,
                latent_sources=[str(x) for x in args.latent_sources],  # type: ignore[arg-type]
                decode_methods=[str(x) for x in args.decode_methods],  # type: ignore[arg-type]
                way_adj_ptr=np.asarray(wg["way_adj_ptr"], dtype=np.int64),
                way_adj_idx=np.asarray(wg["way_adj_idx"], dtype=np.int64),
                way_center_y=np.asarray(wf["way_center_y"], dtype=np.float32),
                way_center_x=np.asarray(wf["way_center_x"], dtype=np.float32),
                device=device,
                log_every=50,
            )
        )
    rep["per_city"] = per_city
    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
