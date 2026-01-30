from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz, make_way_feature_tensors

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    seed: int
    device: str
    tz_offset_hours: float

    n_routes: int  # per city
    max_way_len: int
    max_decode_len: int
    decode: str  # greedy | beam
    beam_size: int

    decode_max_candidates: int  # -1=use model cfg; 0=all successors; >0=override
    decode_candidate_policy: str  # first | destdist
    decode_include_dest_if_successor: bool
    decode_guided_dest_alpha: float

    last_k_steps: int
    reachability_max_visits: int
    progress_every: int


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


def _jaccard(a: List[int], b: List[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb)) / float(len(sa | sb))


def _build_reverse_csr(*, ptr: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build reverse adjacency CSR for a directed graph given forward CSR.
    Reverse graph stores incoming neighbors for each node.
    """
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


def _reverse_reachable(
    *,
    rev_ptr: np.ndarray,
    rev_idx: np.ndarray,
    dest: int,
    max_visits: int,
) -> np.ndarray:
    """
    Reverse BFS from dest on reverse graph => nodes that can reach dest on forward graph.
    """
    rev_ptr = np.asarray(rev_ptr, dtype=np.int64).reshape(-1)
    rev_idx = np.asarray(rev_idx, dtype=np.int64).reshape(-1)
    n = int(rev_ptr.size) - 1
    reachable = np.zeros((n,), dtype=np.bool_)
    if dest < 0 or dest >= n:
        return reachable
    stack: List[int] = [int(dest)]
    reachable[int(dest)] = True
    seen = 0
    while stack:
        u = int(stack.pop())
        s = int(rev_ptr[u])
        e = int(rev_ptr[u + 1])
        for v in rev_idx[s:e].tolist():
            vv = int(v)
            if not reachable[vv]:
                reachable[vv] = True
                stack.append(vv)
        seen += 1
        if seen >= int(max_visits):
            break
    return reachable


def _slice_csr(ptr: np.ndarray, idx: np.ndarray, u: int) -> np.ndarray:
    s = int(ptr[u])
    e = int(ptr[u + 1])
    if e <= s:
        return np.asarray([], dtype=np.int64)
    return np.asarray(idx[s:e], dtype=np.int64)

def _infer_decoder_use_dest_dist_from_state(state: Dict[str, torch.Tensor]) -> bool:
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


def _infer_decoder_use_cross_attn_from_state(state: Dict[str, torch.Tensor]) -> bool:
    return any(str(k).startswith("decoder.cross_attn.") for k in state.keys())


def _infer_decoder_use_step_emb_from_state(state: Dict[str, torch.Tensor]) -> bool:
    return any(str(k).startswith("decoder.step_emb.") for k in state.keys())


def _infer_decoder_use_dest_query_from_state(state: Dict[str, torch.Tensor]) -> bool:
    return any(str(k).startswith("decoder.dest_proj.") for k in state.keys())


def _infer_decoder_use_dir_query_from_state(state: Dict[str, torch.Tensor]) -> bool:
    return any(str(k).startswith("decoder.dir_query_proj.") for k in state.keys())


def _infer_decoder_use_cand_query_from_state(state: Dict[str, torch.Tensor]) -> bool:
    return any(str(k).startswith("decoder.cand_query_proj.") for k in state.keys())


def _infer_decoder_use_past_context_from_state(state: Dict[str, torch.Tensor]) -> bool:
    return any(str(k).startswith("decoder.past_encoder.") for k in state.keys())


def _infer_n_route_cities_from_state(state: Dict[str, torch.Tensor]) -> Optional[int]:
    w = state.get("decoder.cond_enc.route_city_embed.weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2 and int(w.shape[0]) > 0:
        return int(w.shape[0])
    return None


def _select_decode_succ(
    *,
    succ_full: np.ndarray,
    dest_pos_yx: np.ndarray,
    way_center_y: np.ndarray,
    way_center_x: np.ndarray,
    max_candidates: int,
    candidate_policy: str,
    include_dest_if_successor: bool,
    dest_way: int,
) -> np.ndarray:
    succ_full = np.asarray(succ_full, dtype=np.int64).reshape(-1)
    if succ_full.size == 0:
        return succ_full
    k = int(max_candidates)
    if k <= 0 or succ_full.size <= k:
        return succ_full
    policy = str(candidate_policy).lower().strip()
    if policy == "destdist":
        cy = way_center_y[succ_full]
        cx = way_center_x[succ_full]
        dy = cy - float(dest_pos_yx[0])
        dx = cx - float(dest_pos_yx[1])
        dist = dy * dy + dx * dx
        order = np.argsort(dist)
        sel = succ_full[order[:k]]
    else:
        sel = succ_full[:k]
    if bool(include_dest_if_successor) and int(dest_way) >= 0:
        dw = int(dest_way)
        if dw in set(succ_full.tolist()) and dw not in set(sel.tolist()):
            sel = sel.copy()
            sel[-1] = int(dw)
    return sel


def _prefix_match_len(gt: List[int], pred: List[int]) -> int:
    n = min(len(gt), len(pred))
    k = 0
    for i in range(n):
        if int(gt[i]) != int(pred[i]):
            break
        k += 1
    return int(k)


@torch.no_grad()
def run(cfg: Cfg, *, way_routes_npz: Path, way_graph_npz: Path, way_features_npz: Path, ae_ckpt: Path, out_json: Path) -> Dict[str, object]:
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    routes = load_way_routes_npz(Path(way_routes_npz))
    wg = np.load(str(way_graph_npz), allow_pickle=True)
    wf = np.load(str(way_features_npz), allow_pickle=True)
    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64)
    idx = np.asarray(wg["way_adj_idx"], dtype=np.int64)

    rev_ptr, rev_idx = _build_reverse_csr(ptr=ptr, idx=idx)

    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)

    # Build AE
    way_features = load_way_features_from_npz(Path(way_features_npz), device=device)
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1

    ckpt = torch.load(str(ae_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ae_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    use_dest_dist = _infer_decoder_use_dest_dist_from_state(state) if isinstance(state, dict) else True
    use_cand_contrast = (_infer_decoder_use_cand_contrast_from_state(state) if isinstance(state, dict) else False) or bool(
        ae_cfg.get("decoder_use_cand_contrast", False)
    )
    use_cross_attn = (_infer_decoder_use_cross_attn_from_state(state) if isinstance(state, dict) else False) or bool(ae_cfg.get("decoder_use_cross_attn", True))
    use_step_emb = (_infer_decoder_use_step_emb_from_state(state) if isinstance(state, dict) else False) or bool(ae_cfg.get("decoder_use_step_emb", False))
    use_dest_query = (_infer_decoder_use_dest_query_from_state(state) if isinstance(state, dict) else False) or bool(ae_cfg.get("decoder_use_dest_query", False))
    use_dir_query = (_infer_decoder_use_dir_query_from_state(state) if isinstance(state, dict) else False) or bool(ae_cfg.get("decoder_use_dir_query", False))
    use_cand_query = (_infer_decoder_use_cand_query_from_state(state) if isinstance(state, dict) else False) or bool(ae_cfg.get("decoder_use_cand_query", False))
    use_past_context = (_infer_decoder_use_past_context_from_state(state) if isinstance(state, dict) else False) or bool(ae_cfg.get("decoder_use_past_context", False))
    past_k = int(ae_cfg.get("decoder_past_k", 8))
    if use_past_context and isinstance(state, dict):
        pe = state.get("decoder.past_encoder.pos_emb.weight", None)
        if isinstance(pe, torch.Tensor) and pe.ndim == 2 and int(pe.shape[0]) > 0:
            past_k = int(pe.shape[0])
    past_n_layers = int(ae_cfg.get("decoder_past_n_layers", 2))
    past_n_heads = int(ae_cfg.get("decoder_past_n_heads", 4))
    n_route_cities = _infer_n_route_cities_from_state(state) if isinstance(state, dict) else None
    if n_route_cities is None:
        n_route_cities = int(ae_cfg.get("n_route_cities", 4))
    n_city_obs = int(np.max(routes.route_city.astype(np.int64))) + 1
    n_route_cities = max(int(n_route_cities), int(n_city_obs))
    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(ae_cfg.get("d_model", 256)),
            n_latent=int(ae_cfg.get("n_latent", 64)),
            n_heads=int(ae_cfg.get("n_heads", 8)),
            dropout=float(ae_cfg.get("dropout", 0.1)),
            max_candidates=int(ae_cfg.get("max_candidates", 32)),
            max_len=int(ae_cfg.get("max_len", cfg.max_way_len)),
            coord_scale=float(ae_cfg.get("coord_scale", 1024.0)),
            decoder_use_dest_dist=bool(use_dest_dist),
            decoder_use_cross_attn=bool(use_cross_attn),
            decoder_n_cross_heads=int(ae_cfg.get("decoder_n_cross_heads", 4)),
            decoder_use_step_emb=bool(use_step_emb),
            decoder_use_dest_query=bool(use_dest_query),
            decoder_use_dir_query=bool(use_dir_query),
            decoder_use_cand_query=bool(use_cand_query),
            decoder_use_cand_contrast=bool(use_cand_contrast),
            decoder_use_past_context=bool(use_past_context),
            decoder_past_k=int(past_k),
            decoder_past_n_layers=int(past_n_layers),
            decoder_past_n_heads=int(past_n_heads),
        ),
        way_features=way_features,
        way_adj_ptr=ptr,
        way_adj_idx=idx,
        n_highway_types=int(max(4, n_highway_types)),
        n_route_cities=int(n_route_cities),
    ).to(device)
    strict_ok = True
    try:
        ae.load_state_dict(state, strict=True)
    except Exception:
        strict_ok = False
        ae.load_state_dict(state, strict=False)
    ae.eval()

    # Route sampling per city
    def _pick_city(city: int) -> np.ndarray:
        keep = (routes.route_city.astype(np.int64) == int(city)) & (routes.way_seq_len > 1) & (routes.way_seq_len <= int(cfg.max_way_len))
        ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
        rng = np.random.default_rng(int(cfg.seed) + 101 * int(city))
        rng.shuffle(ids)
        return ids[: min(int(cfg.n_routes), int(ids.size))]

    picks = {0: _pick_city(0), 1: _pick_city(1)}

    max_candidates = int(cfg.decode_max_candidates)
    if max_candidates < 0:
        max_candidates = int(ae.cfg.max_candidates)

    dest_reach_cache: Dict[int, np.ndarray] = {}

    def _get_reach(dest: int) -> np.ndarray:
        dd = int(dest)
        hit = dest_reach_cache.get(dd, None)
        if hit is not None:
            return hit
        reach = _reverse_reachable(rev_ptr=rev_ptr, rev_idx=rev_idx, dest=dd, max_visits=int(cfg.reachability_max_visits))
        dest_reach_cache[dd] = reach
        return reach

    out: Dict[str, object] = {
        "ok": True,
        "task": "way_casd_oracle_decode_diagnose",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "ckpt_strict_load_ok": bool(strict_ok),
        "ae_cfg_inferred": {
            "decoder_use_dest_dist": bool(use_dest_dist),
            "decoder_use_cross_attn": bool(use_cross_attn),
            "decoder_use_step_emb": bool(use_step_emb),
            "decoder_use_dest_query": bool(use_dest_query),
            "decoder_use_dir_query": bool(use_dir_query),
            "decoder_use_cand_query": bool(use_cand_query),
            "decoder_use_past_context": bool(use_past_context),
            "decoder_past_k": int(past_k),
            "decoder_past_n_layers": int(past_n_layers),
            "decoder_past_n_heads": int(past_n_heads),
            "n_route_cities": int(n_route_cities),
        },
        "inputs": {
            "way_routes_npz": str(way_routes_npz),
            "way_graph_npz": str(way_graph_npz),
            "way_features_npz": str(way_features_npz),
            "ae_ckpt": str(ae_ckpt),
        },
        "per_city": [],
    }

    for city in (0, 1):
        pick = picks[int(city)]
        failures: List[Dict[str, object]] = []
        successes: List[int] = []

        t0 = time.time()
        n = int(pick.size)
        for ii, rid in enumerate(pick.tolist(), start=1):
            L = int(routes.way_seq_len[int(rid)])
            s = int(routes.way_seq_ptr[int(rid)])
            gt = routes.way_seq_idx[s : s + L].astype(np.int64, copy=False).tolist()
            gt = [int(x) for x in gt]

            start_way = int(routes.start_way[int(rid)])
            dest_way = int(routes.dest_way[int(rid)])
            start_pos = routes.start_pos[int(rid)].astype(np.float64, copy=False).reshape(2)
            dest_pos = routes.dest_pos[int(rid)].astype(np.float64, copy=False).reshape(2)
            start_t = int(routes.start_t[int(rid)])
            hour = int(_hour_from_unix(start_t, float(cfg.tz_offset_hours)))
            dow = int(_dow_from_unix(start_t, float(cfg.tz_offset_hours)))

            # Encode GT -> z_enc
            way_pad = np.full((1, L), -1, dtype=np.int64)
            way_pad[0, : len(gt)] = np.asarray(gt, dtype=np.int64)
            way_pad_t = torch.as_tensor(way_pad, dtype=torch.long, device=device)
            z_enc, _ = ae.encode(way_pad_t)

            route_cond = {
                "start_pos": torch.as_tensor(start_pos[None, :], dtype=torch.float32, device=device),
                "dest_pos": torch.as_tensor(dest_pos[None, :], dtype=torch.float32, device=device),
                "hour": torch.as_tensor(np.asarray([hour], dtype=np.int64), dtype=torch.long, device=device),
                "dow": torch.as_tensor(np.asarray([dow], dtype=np.int64), dtype=torch.long, device=device),
                "route_city": torch.as_tensor(np.asarray([int(city)], dtype=np.int64), dtype=torch.long, device=device),
            }
            sw_t = torch.as_tensor(np.asarray([start_way], dtype=np.int64), dtype=torch.long, device=device)
            dw_t = torch.as_tensor(np.asarray([dest_way], dtype=np.int64), dtype=torch.long, device=device)

            if str(cfg.decode) == "beam":
                pred = ae.decoder.beam_search(
                    way_embedder=ae.way_enc,
                    latent_tokens=z_enc,
                    route_cond=route_cond,
                    start_way=sw_t,
                    dest_way=dw_t,
                    beam_size=int(cfg.beam_size),
                    max_len=int(cfg.max_decode_len),
                    max_candidates=(None if int(cfg.decode_max_candidates) < 0 else int(cfg.decode_max_candidates)),
                    candidate_policy=str(cfg.decode_candidate_policy),
                    include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
                    guided_dest_alpha=float(cfg.decode_guided_dest_alpha),
                )[0]
            else:
                pred = ae.decoder.greedy_decode(
                    way_embedder=ae.way_enc,
                    latent_tokens=z_enc,
                    route_cond=route_cond,
                    start_way=sw_t,
                    dest_way=dw_t,
                    max_len=int(cfg.max_decode_len),
                    max_candidates=(None if int(cfg.decode_max_candidates) < 0 else int(cfg.decode_max_candidates)),
                    candidate_policy=str(cfg.decode_candidate_policy),
                    include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
                    guided_dest_alpha=float(cfg.decode_guided_dest_alpha),
                )[0]

            pred = [int(x) for x in pred]
            success = bool(pred and int(pred[-1]) == int(dest_way))
            max_len_hit = int(cfg.max_decode_len) + 1
            hit_wall = bool((not success) and (len(pred) >= max_len_hit))
            outdeg_last = int(ptr[int(pred[-1]) + 1] - ptr[int(pred[-1])]) if pred and 0 <= int(pred[-1]) + 1 < int(ptr.size) else 0
            dead_end = bool((not success) and (not hit_wall) and (outdeg_last == 0))

            if success:
                successes.append(int(rid))
            else:
                # Reachability (start->dest) via reverse BFS cache
                reach = _get_reach(dest_way)
                start_reach = bool(reach[int(start_way)]) if 0 <= int(start_way) < int(reach.size) else False

                # Divergence (prefix match)
                pre_k = _prefix_match_len(gt, pred)
                div_step = int(pre_k)
                div_pred = int(pred[div_step]) if div_step < len(pred) else None
                div_gt = int(gt[div_step]) if div_step < len(gt) else None
                div_can_reach = bool(reach[int(div_pred)]) if (div_pred is not None and 0 <= int(div_pred) < int(reach.size)) else False

                # Rejoin heuristic: does pred hit any remaining GT nodes after divergence?
                gt_suffix = set(int(x) for x in gt[div_step:]) if div_step < len(gt) else set()
                rejoin = False
                if gt_suffix and div_step < len(pred):
                    for x in pred[div_step:]:
                        if int(x) in gt_suffix:
                            rejoin = True
                            break

                # Candidate analysis for last K transitions
                last_k = max(1, int(cfg.last_k_steps))
                trans_lo = max(0, (len(pred) - 1) - last_k)
                dest_preds = set(_slice_csr(rev_ptr, rev_idx, int(dest_way)).tolist()) if 0 <= int(dest_way) + 1 < int(rev_ptr.size) else set()
                last_steps: List[Dict[str, object]] = []
                any_dest_in_sel = False
                any_dest_in_full = False
                any_pred_of_dest_in_sel = False
                any_pred_of_dest_in_full = False
                any_missed_direct = False
                for t in range(trans_lo, max(0, len(pred) - 1)):
                    cur = int(pred[t])
                    nxt = int(pred[t + 1])
                    if cur < 0 or cur + 1 >= int(ptr.size):
                        continue
                    succ_full = _slice_csr(ptr, idx, int(cur))
                    succ_sel = _select_decode_succ(
                        succ_full=succ_full,
                        dest_pos_yx=dest_pos,
                        way_center_y=way_center_y,
                        way_center_x=way_center_x,
                        max_candidates=int(max_candidates),
                        candidate_policy=str(cfg.decode_candidate_policy),
                        include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
                        dest_way=int(dest_way),
                    )
                    dest_in_full = bool(int(dest_way) in set(succ_full.tolist()))
                    dest_in_sel = bool(int(dest_way) in set(succ_sel.tolist()))
                    pred_of_dest_in_full = bool(any(int(x) in dest_preds for x in succ_full.tolist())) if dest_preds else False
                    pred_of_dest_in_sel = bool(any(int(x) in dest_preds for x in succ_sel.tolist())) if dest_preds else False
                    any_dest_in_full = any_dest_in_full or dest_in_full
                    any_dest_in_sel = any_dest_in_sel or dest_in_sel
                    any_pred_of_dest_in_full = any_pred_of_dest_in_full or pred_of_dest_in_full
                    any_pred_of_dest_in_sel = any_pred_of_dest_in_sel or pred_of_dest_in_sel
                    if dest_in_sel and (int(nxt) != int(dest_way)):
                        any_missed_direct = True
                    last_steps.append(
                        {
                            "t": int(t),
                            "cur": int(cur),
                            "next": int(nxt),
                            "succ_full_n": int(succ_full.size),
                            "succ_sel_n": int(succ_sel.size),
                            "dest_in_full": bool(dest_in_full),
                            "dest_in_sel": bool(dest_in_sel),
                            "pred_of_dest_in_full": bool(pred_of_dest_in_full),
                            "pred_of_dest_in_sel": bool(pred_of_dest_in_sel),
                        }
                    )

                # GT final hop candidate coverage (decode policy)
                gt_final_dest_in_sel = None
                gt_final_dest_in_full = None
                if len(gt) >= 2:
                    u = int(gt[-2])
                    v = int(gt[-1])
                    if 0 <= u + 1 < int(ptr.size):
                        succ_full = _slice_csr(ptr, idx, int(u))
                        succ_sel = _select_decode_succ(
                            succ_full=succ_full,
                            dest_pos_yx=dest_pos,
                            way_center_y=way_center_y,
                            way_center_x=way_center_x,
                            max_candidates=int(max_candidates),
                            candidate_policy=str(cfg.decode_candidate_policy),
                            include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
                            dest_way=int(dest_way),
                        )
                        gt_final_dest_in_full = bool(int(v) in set(succ_full.tolist()))
                        gt_final_dest_in_sel = bool(int(v) in set(succ_sel.tolist()))

                failures.append(
                    {
                        "route_id": int(rid),
                        "route_city": int(city),
                        "hour": int(hour),
                        "dow": int(dow),
                        "gt_len": int(len(gt)),
                        "pred_len": int(len(pred)),
                        "success": bool(success),
                        "hit_wall": bool(hit_wall),
                        "dead_end": bool(dead_end),
                        "jaccard": float(_jaccard(gt, pred)),
                        "start_way": int(start_way),
                        "dest_way": int(dest_way),
                        "gt_start_way": int(gt[0]) if gt else None,
                        "gt_last_way": int(gt[-1]) if gt else None,
                        "start_match": bool(gt and int(start_way) == int(gt[0])),
                        "dest_match": bool(gt and int(dest_way) == int(gt[-1])),
                        "start_can_reach_dest": bool(start_reach),
                        "prefix_match_len": int(pre_k),
                        "diverge_step": int(div_step),
                        "diverge_pred_way": div_pred,
                        "diverge_gt_way": div_gt,
                        "diverge_pred_can_reach_dest": bool(div_can_reach),
                        "rejoin_gt_after_diverge": bool(rejoin),
                        "last_k": {
                            "k": int(last_k),
                            "any_dest_in_full": bool(any_dest_in_full),
                            "any_dest_in_sel": bool(any_dest_in_sel),
                            "any_pred_of_dest_in_full": bool(any_pred_of_dest_in_full),
                            "any_pred_of_dest_in_sel": bool(any_pred_of_dest_in_sel),
                            "any_missed_direct_dest_when_available": bool(any_missed_direct),
                            "steps": last_steps,
                        },
                        "gt_final_hop": {"gt_final_in_full": gt_final_dest_in_full, "gt_final_in_sel": gt_final_dest_in_sel},
                    }
                )

            if int(cfg.progress_every) > 0 and (ii % int(cfg.progress_every) == 0 or ii == n):
                dt = max(1e-6, time.time() - t0)
                rps = float(ii) / float(dt)
                print(f"[city{int(city)}] done={ii}/{n} rps={rps:.2f} succ={len(successes)}/{ii}")

        # Summaries
        n_fail = int(len(failures))
        n_succ = int(len(successes))
        n_eval = int(n_fail + n_succ)
        hit_wall_rate = float(np.mean([1.0 if bool(f["hit_wall"]) else 0.0 for f in failures])) if failures else 0.0
        dead_end_rate = float(np.mean([1.0 if bool(f["dead_end"]) else 0.0 for f in failures])) if failures else 0.0
        start_match_rate = float(np.mean([1.0 if bool(f["start_match"]) else 0.0 for f in failures])) if failures else float("nan")
        dest_match_rate = float(np.mean([1.0 if bool(f["dest_match"]) else 0.0 for f in failures])) if failures else float("nan")
        start_reach_rate = float(np.mean([1.0 if bool(f["start_can_reach_dest"]) else 0.0 for f in failures])) if failures else float("nan")

        out["per_city"].append(
            {
                "city": int(city),
                "n_eval": int(n_eval),
                "success_rate": float(n_succ) / float(max(1, n_eval)),
                "hit_wall_rate": float(hit_wall_rate),
                "dead_end_rate": float(dead_end_rate),
                "failures_n": int(n_fail),
                "successes_n": int(n_succ),
                "failures": failures,
                "success_route_ids": successes,
                "notes": {
                    "start_match_rate_among_failures": float(start_match_rate),
                    "dest_match_rate_among_failures": float(dest_match_rate),
                    "start_can_reach_dest_rate_among_failures": float(start_reach_rate),
                },
            }
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diagnose oracle (z_enc) decode failures for Way-CASD decision decoder.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)

    p.add_argument("--n_routes", type=int, default=500, help="Per city (0 and 1).")
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--max_decode_len", type=int, default=160)
    p.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    p.add_argument("--beam_size", type=int, default=10)

    p.add_argument("--decode_max_candidates", type=int, default=-1, help="-1=use model cfg; 0=all successors; >0=override.")
    p.add_argument("--decode_candidate_policy", choices=["first", "destdist"], default="first")
    p.add_argument("--decode_include_dest_if_successor", action="store_true")
    p.add_argument(
        "--decode_guided_dest_alpha",
        type=float,
        default=0.0,
        help="Decode-time heuristic: logits <- logits - alpha * dist_to_dest (in normalized coord space).",
    )

    p.add_argument("--last_k_steps", type=int, default=5, help="Inspect last K transitions for dest/pred-of-dest availability.")
    p.add_argument("--reachability_max_visits", type=int, default=200000, help="Safety cap for reverse BFS visits.")
    p.add_argument("--progress_every", type=int, default=25)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Cfg(
        seed=int(args.seed),
        device=str(args.device),
        tz_offset_hours=float(args.tz_offset_hours),
        n_routes=int(args.n_routes),
        max_way_len=int(args.max_way_len),
        max_decode_len=int(args.max_decode_len),
        decode=str(args.decode),
        beam_size=int(args.beam_size),
        decode_max_candidates=int(args.decode_max_candidates),
        decode_candidate_policy=str(args.decode_candidate_policy),
        decode_include_dest_if_successor=bool(args.decode_include_dest_if_successor),
        decode_guided_dest_alpha=float(args.decode_guided_dest_alpha),
        last_k_steps=int(args.last_k_steps),
        reachability_max_visits=int(args.reachability_max_visits),
        progress_every=int(args.progress_every),
    )
    _set_seed(cfg.seed)
    rep = run(
        cfg,
        way_routes_npz=Path(args.way_routes_npz),
        way_graph_npz=Path(args.way_graph_npz),
        way_features_npz=Path(args.way_features_npz),
        ae_ckpt=Path(args.ae_ckpt),
        out_json=Path(args.out_json),
    )
    print(f"[saved] {args.out_json} cities={len(rep.get('per_city', []))}")


if __name__ == "__main__":
    main()
