from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    seed: int
    device: str
    tz_offset_hours: float

    n_routes: int  # per city
    min_hops: int
    max_way_len: int
    max_decode_len: int

    decode_max_candidates: int  # -1=use model cfg; 0=all successors; >0=override
    decode_candidate_policy: str  # first | destdist
    decode_include_dest_if_successor: bool
    decode_guided_dest_alpha: float

    focus_early_fail_le_k: int
    focus_max_examples: int
    focus_trace_radius: int

    progress_every: int
    compute_hop: bool

    dump_attn: bool
    attn_topk: int  # 0=store full attn weights; >0=store top-k per candidate


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _hour_from_unix(start_t: int, tz_offset_hours: float) -> int:
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    sec = int((int(start_t) + tz_sec) % 86400)
    return int(sec // 3600)


def _dow_from_unix(start_t: int, tz_offset_hours: float) -> int:
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    days = int((int(start_t) + tz_sec) // 86400)
    return int((days + 3) % 7)


def _jaccard_set(a: List[int], b: List[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter) / float(union) if union > 0 else 0.0


def _first_mismatch_index(gt: List[int], pred: List[int]) -> Optional[int]:
    n = min(len(gt), len(pred))
    for i in range(n):
        if int(gt[i]) != int(pred[i]):
            return int(i)
    if len(gt) != len(pred):
        return int(n)
    return None


def _quantiles_float(values: List[float], qs: Tuple[int, ...] = (0, 50, 90, 95, 99, 100)) -> Dict[str, Optional[float]]:
    if not values:
        return {f"p{q:02d}": None for q in qs}
    arr = np.asarray(values, dtype=np.float64)
    out: Dict[str, Optional[float]] = {}
    for q in qs:
        out[f"p{q:02d}"] = float(np.percentile(arr, float(q)))
    return out


def _quantiles_int(values: List[int], qs: Tuple[int, ...] = (0, 50, 90, 95, 99, 100)) -> Dict[str, Optional[int]]:
    if not values:
        return {f"p{q:02d}": None for q in qs}
    arr = np.asarray(values, dtype=np.float64)
    out: Dict[str, Optional[int]] = {}
    for q in qs:
        out[f"p{q:02d}"] = int(np.percentile(arr, float(q)))
    return out


def _build_reverse_csr(*, ptr: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    n = int(ptr.size) - 1
    indeg = np.zeros((n,), dtype=np.int64)
    for u in range(n):
        s = int(ptr[u])
        e = int(ptr[u + 1])
        if e > s:
            vs = idx[s:e]
            indeg[vs] += 1
    rptr = np.zeros((n + 1,), dtype=np.int64)
    np.cumsum(indeg, out=rptr[1:])
    ridx = np.empty((int(rptr[-1]),), dtype=np.int64)
    cursor = rptr[:-1].copy()
    for u in range(n):
        s = int(ptr[u])
        e = int(ptr[u + 1])
        for v in idx[s:e].tolist():
            vv = int(v)
            p = int(cursor[vv])
            ridx[p] = int(u)
            cursor[vv] += 1
    return rptr, ridx


def _reverse_bfs_dist(*, rev_ptr: np.ndarray, rev_idx: np.ndarray, dest: int) -> np.ndarray:
    rev_ptr = np.asarray(rev_ptr, dtype=np.int64).reshape(-1)
    rev_idx = np.asarray(rev_idx, dtype=np.int64).reshape(-1)
    n = int(rev_ptr.size) - 1
    dist = np.full((n,), -1, dtype=np.int32)
    if dest < 0 or dest >= n:
        return dist
    from collections import deque

    q: deque[int] = deque()
    dist[int(dest)] = 0
    q.append(int(dest))
    while q:
        u = int(q.popleft())
        du = int(dist[u])
        s = int(rev_ptr[u])
        e = int(rev_ptr[u + 1])
        for v in rev_idx[s:e].tolist():
            vv = int(v)
            if dist[vv] < 0:
                dist[vv] = du + 1
                q.append(vv)
    return dist


def _infer_decoder_use_dest_dist_from_state(state: Dict[str, torch.Tensor]) -> bool:
    # Backward-compatible inference based on scorer input dim.
    # Old: in_dim = 3*hidden (+1 if dest_dist)
    # New: in_dim = 4*hidden (+1 if dest_dist) when cand_contrast enabled.
    w = state.get("decoder.scorer.0.weight", None)
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        return True
    hidden = int(w.shape[0])
    in_dim = int(w.shape[1])
    d4 = int(in_dim - hidden * 4)
    if d4 in (0, 1):
        return bool(d4 == 1)
    d3 = int(in_dim - hidden * 3)
    if d3 in (0, 1):
        return bool(d3 == 1)
    return True


def _infer_decoder_use_cand_contrast_from_state(state: Dict[str, torch.Tensor]) -> bool:
    w = state.get("decoder.scorer.0.weight", None)
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        return False
    hidden = int(w.shape[0])
    in_dim = int(w.shape[1])
    d4 = int(in_dim - hidden * 4)
    if d4 in (0, 1):
        return True
    d3 = int(in_dim - hidden * 3)
    if d3 in (0, 1):
        return False
    return False


def _infer_flag(state: Dict[str, torch.Tensor], prefix: str) -> bool:
    return any(str(k).startswith(prefix) for k in state.keys())

def _safe_hop(hop: Optional[np.ndarray], way_id: int) -> int:
    if hop is None:
        return -1
    if 0 <= int(way_id) < int(hop.size):
        return int(hop[int(way_id)])
    return -1


@torch.no_grad()
def run(
    cfg: Cfg,
    *,
    way_routes_npz: Path,
    way_graph_npz: Path,
    way_features_npz: Path,
    ae_ckpt: Path,
    out_json: Path,
) -> Dict[str, Any]:
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    _set_seed(cfg.seed)

    routes = load_way_routes_npz(Path(way_routes_npz))
    wg = np.load(str(way_graph_npz), allow_pickle=True)
    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64)
    idx = np.asarray(wg["way_adj_idx"], dtype=np.int64)
    rev_ptr, rev_idx = _build_reverse_csr(ptr=ptr, idx=idx)

    wf = np.load(str(way_features_npz), allow_pickle=True)
    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1

    # Build AE
    way_features = load_way_features_from_npz(Path(way_features_npz), device=device)
    ckpt = torch.load(str(ae_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ae_cfg_dict = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    inferred = {
        "decoder_use_dest_dist": _infer_decoder_use_dest_dist_from_state(state) if isinstance(state, dict) else True,
        "decoder_use_cand_contrast": _infer_decoder_use_cand_contrast_from_state(state) if isinstance(state, dict) else False,
        "decoder_use_cross_attn": _infer_flag(state, "decoder.cross_attn.") if isinstance(state, dict) else True,
        "decoder_use_step_emb": _infer_flag(state, "decoder.step_emb.") if isinstance(state, dict) else False,
        "decoder_use_dest_query": _infer_flag(state, "decoder.dest_proj.") if isinstance(state, dict) else False,
        "decoder_use_dir_query": _infer_flag(state, "decoder.dir_query_proj.") if isinstance(state, dict) else False,
        "decoder_use_cand_query": _infer_flag(state, "decoder.cand_query_proj.") if isinstance(state, dict) else False,
        "decoder_use_past_context": _infer_flag(state, "decoder.past_encoder.") if isinstance(state, dict) else False,
        "decoder_past_k": 8,
    }
    pe = state.get("decoder.past_encoder.pos_emb.weight", None) if isinstance(state, dict) else None
    if isinstance(pe, torch.Tensor) and pe.ndim == 2 and int(pe.shape[0]) > 0:
        inferred["decoder_past_k"] = int(pe.shape[0])

    dump_attn_effective = bool(cfg.dump_attn) and bool(inferred.get("decoder_use_cand_query", False))
    if bool(cfg.dump_attn) and (not dump_attn_effective):
        print("[WARN] --dump_attn requested but decoder_use_cand_query=False; skip candidate-level attn dump (no per-candidate attn).")

    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(ae_cfg_dict.get("d_model", 256)),
            n_latent=int(ae_cfg_dict.get("n_latent", 64)),
            n_heads=int(ae_cfg_dict.get("n_heads", 8)),
            dropout=float(ae_cfg_dict.get("dropout", 0.1)),
            max_candidates=int(ae_cfg_dict.get("max_candidates", 32)),
            max_len=int(ae_cfg_dict.get("max_len", cfg.max_way_len)),
            coord_scale=float(ae_cfg_dict.get("coord_scale", 1024.0)),
            decoder_use_dest_dist=bool(inferred["decoder_use_dest_dist"]),
            decoder_use_cand_contrast=bool(inferred["decoder_use_cand_contrast"]),
            decoder_use_cross_attn=bool(inferred["decoder_use_cross_attn"]),
            decoder_n_cross_heads=int(ae_cfg_dict.get("decoder_n_cross_heads", 4)),
            decoder_use_step_emb=bool(inferred["decoder_use_step_emb"]),
            decoder_use_dest_query=bool(inferred["decoder_use_dest_query"]),
            decoder_use_dir_query=bool(inferred.get("decoder_use_dir_query", False)),
            decoder_use_cand_query=bool(inferred.get("decoder_use_cand_query", False)),
            decoder_use_past_context=bool(inferred["decoder_use_past_context"]),
            decoder_past_k=int(inferred["decoder_past_k"]),
            decoder_past_n_layers=int(ae_cfg_dict.get("decoder_past_n_layers", 2)),
            decoder_past_n_heads=int(ae_cfg_dict.get("decoder_past_n_heads", 4)),
        ),
        way_features=way_features,
        way_adj_ptr=ptr,
        way_adj_idx=idx,
        n_highway_types=int(max(4, n_highway_types)),
    ).to(device)
    ckpt_strict_load_ok = True
    try:
        ae.load_state_dict(state, strict=True)
    except Exception as e:
        ckpt_strict_load_ok = False
        print(f"[WARN] strict load_state_dict failed, fallback strict=False: {e}")
        ae.load_state_dict(state, strict=False)
    ae.eval()

    # Route sampling per city
    def _pick_city(city: int) -> np.ndarray:
        keep = (
            (routes.route_city.astype(np.int64) == int(city))
            & (routes.way_seq_len >= (int(cfg.min_hops) + 1))
            & (routes.way_seq_len <= int(cfg.max_way_len))
        )
        ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
        rng = np.random.default_rng(int(cfg.seed) + 101 * int(city))
        rng.shuffle(ids)
        return ids[: min(int(cfg.n_routes), int(ids.size))]

    picks = {0: _pick_city(0), 1: _pick_city(1)}
    cities = sorted(int(c) for c in picks.keys())
    ways_used_by_city: Dict[int, set[int]] = {int(c): set() for c in cities}

    max_candidates = int(cfg.decode_max_candidates)
    if max_candidates < 0:
        max_candidates = int(ae.cfg.max_candidates)

    # Aggregation containers
    per_route: List[Dict[str, Any]] = []
    focus_traces: List[Dict[str, Any]] = []

    hop_cache: Dict[int, np.ndarray] = {}
    hop_cache_hits = 0
    hop_cache_misses = 0

    # Q1/Q2 containers
    div_outdeg: List[int] = []
    div_succ_full_gt: List[int] = []
    div_succ_sel_gt: List[int] = []
    div_margin: List[float] = []
    div_gt_rank: List[int] = []
    div_gt_gap: List[float] = []
    div_is_close: List[bool] = []  # top1-top2 margin small
    div_dist_pred_closer: List[bool] = []  # dest_dist shortcut hypothesis
    div_dist_diff: List[float] = []  # dist_pred - dist_gt (negative means pred is closer)

    # Q3 containers (diverged but succeeded)
    recov_rejoin: List[bool] = []
    recov_rejoin_step: List[int] = []
    recov_min_hop_after_div: List[int] = []
    recov_hop_drop: List[int] = []

    t0 = time.time()
    done = 0

    for city in cities:
        for rid in picks[int(city)].tolist():
            rid = int(rid)
            done += 1
            L = int(routes.way_seq_len[rid])
            s = int(routes.way_seq_ptr[rid])
            gt = routes.way_seq_idx[s : s + L].astype(np.int64).tolist()
            gt = [int(x) for x in gt]
            ways_used_by_city[int(city)].update(int(x) for x in gt)

            start_way = int(routes.start_way[rid])
            dest_way = int(routes.dest_way[rid])
            start_pos = routes.start_pos[rid].astype(np.float64).reshape(2)
            dest_pos = routes.dest_pos[rid].astype(np.float64).reshape(2)
            start_t = int(routes.start_t[rid])
            hour = int(_hour_from_unix(start_t, float(cfg.tz_offset_hours)))
            dow = int(_dow_from_unix(start_t, float(cfg.tz_offset_hours)))

            # Encode GT -> z_enc
            way_pad = np.full((1, L), -1, dtype=np.int64)
            way_pad[0, :L] = np.asarray(gt, dtype=np.int64)
            way_pad_t = torch.as_tensor(way_pad, dtype=torch.long, device=device)
            z_enc, _ = ae.encode(way_pad_t)

            # Pre-compute hop distance for this dest
            hop: Optional[np.ndarray] = None
            if bool(cfg.compute_hop):
                key = int(dest_way)
                cached = hop_cache.get(key, None)
                if cached is None:
                    hop_cache_misses += 1
                    cached = _reverse_bfs_dist(rev_ptr=rev_ptr, rev_idx=rev_idx, dest=int(dest_way))
                    hop_cache[key] = cached
                else:
                    hop_cache_hits += 1
                hop = cached

            route_cond = {
                "start_pos": torch.as_tensor(start_pos[None, :], dtype=torch.float32, device=device),
                "dest_pos": torch.as_tensor(dest_pos[None, :], dtype=torch.float32, device=device),
                "hour": torch.as_tensor(np.asarray([hour], dtype=np.int64), dtype=torch.long, device=device),
                "dow": torch.as_tensor(np.asarray([dow], dtype=np.int64), dtype=torch.long, device=device),
                "route_city": torch.as_tensor(np.asarray([int(city)], dtype=np.int64), dtype=torch.long, device=device),
            }
            cond_emb = ae.decoder.cond_enc(
                start_pos=route_cond["start_pos"],
                dest_pos=route_cond["dest_pos"],
                hour=route_cond["hour"],
                dow=route_cond["dow"],
                route_city=route_cond["route_city"],
            )

            path: List[int] = [int(start_way)]
            gt_prefix_ok = bool(gt) and int(gt[0]) == int(start_way)
            first_div_transition: Optional[Dict[str, Any]] = None

            # Save a compact step log for focused cases only
            step_logs: List[Dict[str, Any]] = []

            for step_idx in range(int(cfg.max_decode_len)):
                cur = int(path[-1])
                if cur == int(dest_way):
                    break
                # successors
                cand_full = ae.decoder.get_succ_candidates(cur)
                succ_full_n = int(cand_full.numel())
                if succ_full_n <= 0:
                    break

                # select candidates (same as decode)
                cand_sel = ae.decoder._select_decode_candidates(
                    way_embedder=ae.way_enc,
                    cand_full=cand_full.to(device=device),
                    dest_pos=route_cond["dest_pos"],
                    dest_way=int(dest_way),
                    max_candidates=(None if int(max_candidates) <= 0 else int(max_candidates)),
                    candidate_policy=str(cfg.decode_candidate_policy),
                    include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
                )
                succ_sel_n = int(cand_sel.numel())
                if succ_sel_n <= 0:
                    break

                cand_way = cand_sel.view(1, succ_sel_n).to(device=device)
                cand_mask = torch.ones((1, succ_sel_n), dtype=torch.bool, device=device)

                # past context (if enabled)
                trans: Dict[str, torch.Tensor] = {
                    "route_idx": torch.tensor([0], dtype=torch.long, device=device),
                    "cur_way": torch.tensor([cur], dtype=torch.long, device=device),
                    "cand_way": cand_way,
                    "cand_mask": cand_mask,
                    "step": torch.tensor([int(step_idx)], dtype=torch.long, device=device),
                }
                if bool(ae.decoder.use_past_context) and len(path) > 0:
                    K = int(ae.decoder.past_k)
                    past_list = path[:-1][-K:] if len(path) > 1 else []
                    past_arr = [-1] * K
                    for i, w in enumerate(past_list):
                        offset = K - len(past_list)
                        past_arr[offset + i] = int(w)
                    past_way = torch.tensor([past_arr], dtype=torch.long, device=device)
                    past_mask = (past_way >= 0)
                    trans["past_way"] = past_way
                    trans["past_mask"] = past_mask

                logits = ae.decoder.score_candidates(
                    way_embedder=ae.way_enc,
                    latent_tokens=z_enc,
                    route_cond=route_cond,
                    trans=trans,
                    cond_emb=cond_emb,
                )[0]

                # Optional guided heuristic (same as decoder)
                alpha = float(cfg.decode_guided_dest_alpha)
                if abs(alpha) > 1e-12:
                    try:
                        coord_scale = float(getattr(ae.way_enc, "coord_scale", ae.decoder.coord_scale))
                        dest = route_cond["dest_pos"].to(dtype=torch.float32)
                        if coord_scale > 0:
                            dest = dest / coord_scale
                        cand_geom, _tier, _hw = ae.way_enc._lookup(cand_way)
                        cand_center = cand_geom[..., :2].to(dtype=torch.float32)
                        dist = torch.norm(dest[:, None, :] - cand_center, dim=-1)  # (1,C)
                        logits = logits - alpha * dist[0]
                    except Exception:
                        pass

                # stats on logits
                logits_np = logits.detach().to("cpu").to(dtype=torch.float32).numpy().reshape(-1)
                top2 = np.sort(logits_np)[-2:] if logits_np.size >= 2 else np.asarray([logits_np.max(), logits_np.max()], dtype=np.float32)
                margin = float(top2[-1] - top2[-2]) if top2.size >= 2 else 0.0
                close_call = bool(margin < 0.2)

                j = int(torch.argmax(logits, dim=-1).item()) if int(logits.numel()) else 0
                pred_next = int(cand_sel[j].item())

                # GT next (only meaningful if we are still on GT prefix and aligned)
                gt_next: Optional[int] = None
                if gt_prefix_ok and step_idx < len(gt) - 1 and int(gt[step_idx]) == cur:
                    gt_next = int(gt[step_idx + 1])
                else:
                    gt_prefix_ok = False

                gt_in_full = None
                gt_in_sel = None
                gt_rank = None
                gt_gap = None
                logit_gap_pred_gt = None
                if gt_next is not None:
                    full_list = set(int(x) for x in cand_full.detach().to("cpu").numpy().reshape(-1).tolist())
                    sel_list = [int(x) for x in cand_sel.detach().to("cpu").numpy().reshape(-1).tolist()]
                    gt_in_full = bool(int(gt_next) in full_list)
                    gt_in_sel = bool(int(gt_next) in set(sel_list))
                    if gt_in_sel:
                        # rank among selected
                        gt_pos = sel_list.index(int(gt_next))
                        order = np.argsort(-logits_np)
                        gt_rank = int(np.where(order == gt_pos)[0][0]) + 1  # 1-based
                        gt_gap = float(np.max(logits_np) - float(logits_np[gt_pos]))
                        logit_gap_pred_gt = float(float(logits_np[j]) - float(logits_np[gt_pos]))

                # First divergence transition (while on GT prefix)
                if first_div_transition is None and gt_next is not None and pred_next != int(gt_next):
                    # Compute dest_dist for pred_next and gt_next (Euclidean to dest_pos)
                    dist_pred = None
                    dist_gt = None
                    dist_pred_closer = None
                    dist_pred_minus_gt = None
                    if 0 <= pred_next < len(way_center_y) and 0 <= int(gt_next) < len(way_center_y):
                        pred_cy, pred_cx = float(way_center_y[pred_next]), float(way_center_x[pred_next])
                        gt_cy, gt_cx = float(way_center_y[int(gt_next)]), float(way_center_x[int(gt_next)])
                        dy, dx = float(dest_pos[0]), float(dest_pos[1])
                        dist_pred = float(np.sqrt((pred_cy - dy) ** 2 + (pred_cx - dx) ** 2))
                        dist_gt = float(np.sqrt((gt_cy - dy) ** 2 + (gt_cx - dx) ** 2))
                        dist_pred_closer = bool(dist_pred < dist_gt)
                        dist_pred_minus_gt = float(dist_pred - dist_gt)

                    # Extra diagnostics (candidate/context differences)
                    ctx_norm = None
                    ctx_pred_norm = None
                    ctx_gt_norm = None
                    ctx_diff_norm = None
                    cand_h_diff = None
                    cand_attn_weights = None
                    cand_attn_topk_idx = None
                    cand_attn_topk_val = None
                    gt_attn_weight = None
                    pred_attn_weight = None
                    attn_cos_pred_gt = None
                    gt_attn_topk_idx = None
                    gt_attn_topk_val = None
                    pred_attn_topk_idx = None
                    pred_attn_topk_val = None
                    try:
                        # Candidate embedding diff
                        cand_emb_dbg, _ = ae.way_enc(cand_way)  # (1,C,d)
                        cand_h_dbg = ae.decoder.cand_proj(cand_emb_dbg)[0]  # (C,H)
                        if gt_rank is not None and gt_in_sel:
                            sel_list = [int(x) for x in cand_sel.detach().to("cpu").numpy().reshape(-1).tolist()]
                            gt_pos2 = int(sel_list.index(int(gt_next)))
                            cand_h_diff = float(torch.norm(cand_h_dbg[int(j)] - cand_h_dbg[int(gt_pos2)]).item())

                        # Context norm(s)
                        attn_dbg = None
                        if bool(dump_attn_effective):
                            ctx_out_dbg, attn_dbg = ae.decoder._compute_context(
                                way_embedder=ae.way_enc,
                                latent_tokens=z_enc,
                                cond_emb=cond_emb,
                                cur_way=trans["cur_way"],
                                cand_way=trans["cand_way"],
                                cand_mask=trans["cand_mask"],
                                cur_emb=None,
                                cand_emb=None,
                                route_idx=trans["route_idx"],
                                step=trans["step"],
                                dest_pos=route_cond["dest_pos"],
                                past_way=trans.get("past_way", None),
                                past_mask=trans.get("past_mask", None),
                                return_attn_weights=True,
                            )
                        else:
                            ctx_out_dbg = ae.decoder._compute_context(
                                way_embedder=ae.way_enc,
                                latent_tokens=z_enc,
                                cond_emb=cond_emb,
                                cur_way=trans["cur_way"],
                                cand_way=trans["cand_way"],
                                cand_mask=trans["cand_mask"],
                                cur_emb=None,
                                cand_emb=None,
                                route_idx=trans["route_idx"],
                                step=trans["step"],
                                dest_pos=route_cond["dest_pos"],
                                past_way=trans.get("past_way", None),
                                past_mask=trans.get("past_mask", None),
                            )
                        if ctx_out_dbg.ndim == 2:
                            v = ctx_out_dbg[0]
                            ctx_norm = float(torch.norm(v).item())
                            ctx_pred_norm = ctx_norm
                            ctx_gt_norm = ctx_norm
                            ctx_diff_norm = 0.0
                        else:
                            v_pred = ctx_out_dbg[0, int(j)]
                            ctx_pred_norm = float(torch.norm(v_pred).item())
                            ctx_norm = ctx_pred_norm
                            if gt_in_sel:
                                sel_list = [int(x) for x in cand_sel.detach().to("cpu").numpy().reshape(-1).tolist()]
                                gt_pos2 = int(sel_list.index(int(gt_next)))
                                v_gt = ctx_out_dbg[0, gt_pos2]
                                ctx_gt_norm = float(torch.norm(v_gt).item())
                                ctx_diff_norm = float(torch.norm(v_pred - v_gt).item())

                        # Candidate attention weights (only meaningful for candidate-aware ctx: (T,C,hidden))
                        if bool(dump_attn_effective) and isinstance(attn_dbg, torch.Tensor) and attn_dbg.ndim == 3:
                            # attn_dbg: (T,C,L); here T=1
                            w = attn_dbg[0].detach().to("cpu", dtype=torch.float32)  # (C,L)
                            if gt_rank is not None and gt_in_sel:
                                sel_list = [int(x) for x in cand_sel.detach().to("cpu").numpy().reshape(-1).tolist()]
                                gt_pos2 = int(sel_list.index(int(gt_next)))
                                attn_cos_pred_gt = float(
                                    F.cosine_similarity(w[int(j)].unsqueeze(0), w[int(gt_pos2)].unsqueeze(0), dim=-1).item()
                                )
                            if int(cfg.attn_topk) > 0:
                                k = min(int(cfg.attn_topk), int(w.shape[1]))
                                vals, idxs = torch.topk(w, k=k, dim=-1)  # (C,k)
                                cand_attn_topk_idx = idxs.tolist()
                                cand_attn_topk_val = vals.tolist()
                                if gt_rank is not None and gt_in_sel:
                                    sel_list = [int(x) for x in cand_sel.detach().to("cpu").numpy().reshape(-1).tolist()]
                                    gt_pos2 = int(sel_list.index(int(gt_next)))
                                    gt_attn_topk_idx = cand_attn_topk_idx[int(gt_pos2)]
                                    gt_attn_topk_val = cand_attn_topk_val[int(gt_pos2)]
                                pred_attn_topk_idx = cand_attn_topk_idx[int(j)]
                                pred_attn_topk_val = cand_attn_topk_val[int(j)]
                            else:
                                cand_attn_weights = w.tolist()
                                pred_attn_weight = cand_attn_weights[int(j)]
                                if gt_rank is not None and gt_in_sel:
                                    sel_list = [int(x) for x in cand_sel.detach().to("cpu").numpy().reshape(-1).tolist()]
                                    gt_pos2 = int(sel_list.index(int(gt_next)))
                                    gt_attn_weight = cand_attn_weights[int(gt_pos2)]
                    except Exception:
                        pass
                    
                    first_div_transition = {
                        "step_idx": int(step_idx),
                        "cur_way": int(cur),
                        "gt_next": int(gt_next),
                        "pred_next": int(pred_next),
                        "succ_full_n": int(succ_full_n),
                        "succ_sel_n": int(succ_sel_n),
                        "gt_in_full": gt_in_full,
                        "gt_in_sel": gt_in_sel,
                        "gt_rank": gt_rank,
                        "logit_margin": float(margin),
                        "gt_gap": gt_gap,
                        "logit_gap_pred_gt": logit_gap_pred_gt,
                        "close_call": bool(close_call),
                        "hop_cur": _safe_hop(hop, int(cur)),
                        "hop_pred_next": _safe_hop(hop, int(pred_next)),
                        "dist_pred_to_dest": dist_pred,
                        "dist_gt_to_dest": dist_gt,
                        "dist_pred_minus_gt": dist_pred_minus_gt,
                        "dist_pred_closer": dist_pred_closer,
                        "ctx_norm": ctx_norm,
                        "ctx_pred_norm": ctx_pred_norm,
                        "ctx_gt_norm": ctx_gt_norm,
                        "ctx_diff_norm": ctx_diff_norm,
                        "cand_h_diff": cand_h_diff,
                        "attn_cos_pred_gt": attn_cos_pred_gt,
                        "attn_topk": int(cfg.attn_topk) if bool(dump_attn_effective) else None,
                        "cand_attn_weights": cand_attn_weights,
                        "gt_attn_weight": gt_attn_weight,
                        "pred_attn_weight": pred_attn_weight,
                        "cand_attn_topk_idx": cand_attn_topk_idx,
                        "cand_attn_topk_val": cand_attn_topk_val,
                        "gt_attn_topk_idx": gt_attn_topk_idx,
                        "gt_attn_topk_val": gt_attn_topk_val,
                        "pred_attn_topk_idx": pred_attn_topk_idx,
                        "pred_attn_topk_val": pred_attn_topk_val,
                    }
                    div_outdeg.append(int(succ_full_n))
                    if gt_in_full is not None:
                        div_succ_full_gt.append(1 if bool(gt_in_full) else 0)
                    if gt_in_sel is not None:
                        div_succ_sel_gt.append(1 if bool(gt_in_sel) else 0)
                    if gt_rank is not None:
                        div_gt_rank.append(int(gt_rank))
                    if gt_gap is not None:
                        div_gt_gap.append(float(gt_gap))
                    div_margin.append(float(margin))
                    div_is_close.append(bool(close_call))
                    if dist_pred_closer is not None:
                        div_dist_pred_closer.append(bool(dist_pred_closer))
                    if dist_pred is not None and dist_gt is not None:
                        div_dist_diff.append(float(dist_pred) - float(dist_gt))

                # Minimal step log (for possible focus trace)
                step_logs.append(
                    {
                        "step": int(step_idx),
                        "cur_way": int(cur),
                        "pred_next": int(pred_next),
                        "succ_full_n": int(succ_full_n),
                        "succ_sel_n": int(succ_sel_n),
                        "gt_next": int(gt_next) if gt_next is not None else None,
                        "gt_in_sel": gt_in_sel,
                        "gt_rank": gt_rank,
                        "logit_margin": float(margin),
                        "close_call": bool(close_call),
                        "hop_cur": _safe_hop(hop, int(cur)),
                        "hop_pred_next": _safe_hop(hop, int(pred_next)),
                    }
                )

                path.append(int(pred_next))
                if gt_prefix_ok and gt_next is not None and int(pred_next) != int(gt_next):
                    gt_prefix_ok = False

            success = bool(path and int(path[-1]) == int(dest_way))
            div_idx = _first_mismatch_index(gt, path)
            seq_exact = bool(div_idx is None)
            jac = float(_jaccard_set(gt, path))

            # Recovery analysis (only for success & diverged)
            rejoin = False
            rejoin_step = None
            hop_drop = None
            min_hop_after = None
            if success and (div_idx is not None):
                gt_suffix = set(int(x) for x in gt[div_idx:])
                for k in range(div_idx, len(path)):
                    if int(path[k]) in gt_suffix:
                        rejoin = True
                        rejoin_step = int(k)
                        break
                # hop trend after divergence (optional; hop may be disabled for speed)
                if hop is not None:
                    hop_seq = [_safe_hop(hop, int(w)) for w in path]
                    hop_after = [h for h in hop_seq[div_idx:] if int(h) >= 0]
                    if hop_after:
                        min_hop_after = int(min(hop_after))
                        hop_drop = int(hop_after[0] - min_hop_after)

                recov_rejoin.append(bool(rejoin))
                if rejoin_step is not None:
                    recov_rejoin_step.append(int(rejoin_step))
                if min_hop_after is not None:
                    recov_min_hop_after_div.append(int(min_hop_after))
                if hop_drop is not None:
                    recov_hop_drop.append(int(hop_drop))

            # Focus traces selection (store compact window around divergence)
            is_fail_early = (not success) and (div_idx is not None) and (int(div_idx) <= int(cfg.focus_early_fail_le_k))
            is_succ_div = success and (div_idx is not None)
            if (is_fail_early or is_succ_div) and len(focus_traces) < int(cfg.focus_max_examples):
                center = int(div_idx) if div_idx is not None else 0
                rad = int(cfg.focus_trace_radius)
                a = max(0, center - rad)
                b = min(len(step_logs), center + rad + 1)
                focus_traces.append(
                    {
                        "route_id": int(rid),
                        "city": int(city),
                        "gt_len": int(len(gt)),
                        "pred_len": int(len(path)),
                        "success": bool(success),
                        "jaccard": float(jac),
                        "diverge_idx": int(div_idx) if div_idx is not None else None,
                        "first_div_transition": first_div_transition,
                        "trace_window": {"a": int(a), "b": int(b), "center": int(center)},
                        "steps": step_logs[a:b],
                    }
                )

            per_route.append(
                {
                    "route_id": int(rid),
                    "city": int(city),
                    "gt_len": int(len(gt)),
                    "pred_len": int(len(path)),
                    "success": bool(success),
                    "seq_exact": bool(seq_exact),
                    "jaccard": float(jac),
                    "diverge_idx": int(div_idx) if div_idx is not None else None,
                    "first_div_transition": first_div_transition,
                }
            )

            if int(cfg.progress_every) > 0 and (done % int(cfg.progress_every) == 0):
                dt = max(1e-6, time.time() - t0)
                rps = float(done) / float(dt)
                succ_rate = float(np.mean([1.0 if r["success"] else 0.0 for r in per_route])) if per_route else 0.0
                print(f"[progress] done={done} rps={rps:.2f} succ={succ_rate:.2%}")

    # --- Aggregate Q1/Q2/Q3 summaries ---
    n_eval = int(len(per_route))
    succ_rows = [r for r in per_route if bool(r["success"])]
    fail_rows = [r for r in per_route if not bool(r["success"])]
    diverged_succ = [r for r in succ_rows if r.get("diverge_idx") is not None]
    exact_succ = [r for r in succ_rows if bool(r.get("seq_exact", False))]

    div_succ_full_frac = float(np.mean(div_succ_full_gt)) if div_succ_full_gt else None
    div_succ_sel_frac = float(np.mean(div_succ_sel_gt)) if div_succ_sel_gt else None

    # --- Per-city diagnosis (Detroit vs Columbus) ---
    way_sem = wf.get("way_semantic", None)
    if way_sem is not None:
        way_sem = np.asarray(way_sem, dtype=np.float32)

    by_city: Dict[str, Dict[str, Any]] = {}
    for c in cities:
        rows_c = [r for r in per_route if int(r.get("city", -1)) == int(c)]
        fail_c = [r for r in rows_c if not bool(r.get("success", False))]

        outdeg_fail: List[float] = []
        cos_fail: List[float] = []
        for r in fail_c:
            fd = r.get("first_div_transition", None)
            if not isinstance(fd, dict):
                continue
            if fd.get("succ_full_n", None) is not None:
                outdeg_fail.append(float(fd["succ_full_n"]))
            if fd.get("attn_cos_pred_gt", None) is not None:
                cos_fail.append(float(fd["attn_cos_pred_gt"]))

        sem_var = None
        if way_sem is not None:
            ids = np.asarray(list(ways_used_by_city.get(int(c), set())), dtype=np.int64).reshape(-1)
            ids = ids[(ids >= 0) & (ids < int(way_sem.shape[0]))]
            if ids.size > 0:
                v = np.var(way_sem[ids], axis=0)
                sem_var = float(np.mean(v))

        by_city[str(int(c))] = {
            "city": int(c),
            "n_eval": int(len(rows_c)),
            "n_fail": int(len(fail_c)),
            "first_div_outdeg_mean": float(np.mean(outdeg_fail)) if outdeg_fail else None,
            "semantic_variance": sem_var,
            "fail_cos_attn_mean": float(np.mean(cos_fail)) if cos_fail else None,
        }

    out = {
        "ok": True,
        "task": "way_casd_oracle_step_diagnose",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "way_routes_npz": str(way_routes_npz),
            "way_graph_npz": str(way_graph_npz),
            "way_features_npz": str(way_features_npz),
            "ae_ckpt": str(ae_ckpt),
        },
        "ckpt_strict_load_ok": ckpt_strict_load_ok,
        "ckpt_decoder_cfg_inferred": inferred,
        "perf": {
            "dump_attn_effective": bool(dump_attn_effective),
            "compute_hop": bool(cfg.compute_hop),
            "hop_cache_size": int(len(hop_cache)) if bool(cfg.compute_hop) else 0,
            "hop_cache_hits": int(hop_cache_hits) if bool(cfg.compute_hop) else 0,
            "hop_cache_misses": int(hop_cache_misses) if bool(cfg.compute_hop) else 0,
        },
        "n_eval": n_eval,
        "summary": {
            "success_rate": float(len(succ_rows)) / float(max(1, n_eval)),
            "success_exact_rate": float(len(exact_succ)) / float(max(1, len(succ_rows))) if succ_rows else 0.0,
            "success_diverged_rate": float(len(diverged_succ)) / float(max(1, len(succ_rows))) if succ_rows else 0.0,
        },
        "q1_branching": {
            "first_div_outdeg_quantiles": _quantiles_int(div_outdeg),
            "first_div_outdeg_gt32_frac": float(np.mean([1.0 if int(x) > 32 else 0.0 for x in div_outdeg])) if div_outdeg else None,
        },
        "q2_logits": {
            "first_div_gt_in_full_frac": div_succ_full_frac,
            "first_div_gt_in_sel_frac": div_succ_sel_frac,
            "first_div_logit_margin_quantiles": _quantiles_float(div_margin),
            "first_div_close_call_frac": float(np.mean([1.0 if bool(x) else 0.0 for x in div_is_close])) if div_is_close else None,
            "first_div_gt_rank_quantiles": _quantiles_int(div_gt_rank),
            "first_div_gt_gap_quantiles": _quantiles_float(div_gt_gap),
        },
        "q4_dest_dist_shortcut": {
            "first_div_pred_closer_to_dest_frac": float(np.mean([1.0 if bool(x) else 0.0 for x in div_dist_pred_closer])) if div_dist_pred_closer else None,
            "first_div_dist_diff_quantiles": _quantiles_float(div_dist_diff),
            "interpretation": (
                "pred_closer_frac > 0.7 suggests dest_dist shortcut; ~0.5 suggests other cause"
                if div_dist_pred_closer else "no data"
            ),
        },
        "q3_recovery": {
            "diverged_success_n": int(len(diverged_succ)),
            "diverged_success_rejoin_frac": float(np.mean([1.0 if bool(x) else 0.0 for x in recov_rejoin])) if recov_rejoin else None,
            "diverged_success_rejoin_step_quantiles": _quantiles_int(recov_rejoin_step),
            "diverged_success_min_hop_after_div_quantiles": _quantiles_int(recov_min_hop_after_div),
            "diverged_success_hop_drop_quantiles": _quantiles_int(recov_hop_drop),
        },
        "city_diag": by_city,
        "per_route": per_route,
        "focus_traces": focus_traces,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_json), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_json}")
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Way-CASD oracle step-by-step diagnose (Q1/Q2/Q3)")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)

    p.add_argument("--n_routes", type=int, default=200, help="Per-city sample size")
    p.add_argument("--min_hops", type=int, default=1, help="Filter routes with fewer than this many way transitions (hops).")
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--max_decode_len", type=int, default=160)

    p.add_argument("--decode_max_candidates", type=int, default=-1)
    p.add_argument("--decode_candidate_policy", type=str, default="first", choices=["first", "destdist"])
    p.add_argument("--decode_include_dest_if_successor", action="store_true")
    p.add_argument("--decode_guided_dest_alpha", type=float, default=0.0)

    p.add_argument("--focus_early_fail_le_k", type=int, default=5)
    p.add_argument("--focus_max_examples", type=int, default=40)
    p.add_argument("--focus_trace_radius", type=int, default=12)

    p.add_argument("--progress_every", type=int, default=50)
    p.add_argument(
        "--compute_hop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute hop-to-dest distances via reverse BFS (slow; set --no-compute_hop to speed up).",
    )

    p.add_argument("--dump_attn", action="store_true", help="Dump candidate attention weights at the first divergence step (focus traces only).")
    p.add_argument(
        "--attn_topk",
        type=int,
        default=0,
        help="If >0, store top-k latent indices/weights per candidate instead of full (C,L) attention matrix.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Cfg(
        seed=int(args.seed),
        device=str(args.device),
        tz_offset_hours=float(args.tz_offset_hours),
        n_routes=int(args.n_routes),
        min_hops=int(args.min_hops),
        max_way_len=int(args.max_way_len),
        max_decode_len=int(args.max_decode_len),
        decode_max_candidates=int(args.decode_max_candidates),
        decode_candidate_policy=str(args.decode_candidate_policy),
        decode_include_dest_if_successor=bool(args.decode_include_dest_if_successor),
        decode_guided_dest_alpha=float(args.decode_guided_dest_alpha),
        focus_early_fail_le_k=int(args.focus_early_fail_le_k),
        focus_max_examples=int(args.focus_max_examples),
        focus_trace_radius=int(args.focus_trace_radius),
        progress_every=int(args.progress_every),
        compute_hop=bool(args.compute_hop),
        dump_attn=bool(args.dump_attn),
        attn_topk=int(args.attn_topk),
    )
    run(
        cfg,
        way_routes_npz=Path(args.way_routes_npz),
        way_graph_npz=Path(args.way_graph_npz),
        way_features_npz=Path(args.way_features_npz),
        ae_ckpt=Path(args.ae_ckpt),
        out_json=Path(args.out_json),
    )


if __name__ == "__main__":
    main()
