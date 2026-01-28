"""
Way-CASD Hit-Wall Trace (PI 诊断3)

目标：在 "GT latent + 全后继可见 + cross-attention" 下，
挑选若干 hit_wall 案例，打印/保存最后 K 步的打分行为：
- 每步 top5 logits (way_id, logit)
- dest 是否在候选中；若在，dest logit 排名
- 模型选中的候选 vs（若仍在 GT 前缀上）GT 下一步的 logit 差

说明：
- 只做 greedy decode（hit_wall 主要来自 greedy 的误差累积/振荡）。
- 本脚本会从 ckpt state_dict 推断 past-context/step-emb 等 decoder 配置，避免口径不一致。
"""

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

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    seed: int
    device: str
    tz_offset_hours: float
    n_routes: int  # per city to scan
    max_way_len: int
    max_decode_len: int
    decode_max_candidates: int
    decode_candidate_policy: str
    decode_include_dest_if_successor: bool
    last_k_steps: int
    topk_logits: int
    n_cases: int


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


def _infer_decoder_cfg_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    w = state.get("decoder.scorer.0.weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        hidden = int(w.shape[0])
        in_dim = int(w.shape[1])
        delta = int(in_dim - hidden * 3)
        cfg["decoder_use_dest_dist"] = (delta != 0)
    else:
        cfg["decoder_use_dest_dist"] = True

    cfg["decoder_use_cross_attn"] = any(str(k).startswith("decoder.cross_attn.") for k in state.keys())
    cfg["decoder_use_step_emb"] = any(str(k).startswith("decoder.step_emb.") for k in state.keys())
    cfg["decoder_use_dest_query"] = any(str(k).startswith("decoder.dest_proj.") for k in state.keys())
    cfg["decoder_use_dir_query"] = any(str(k).startswith("decoder.dir_query_proj.") for k in state.keys())
    cfg["decoder_use_cand_query"] = any(str(k).startswith("decoder.cand_query_proj.") for k in state.keys())
    cfg["decoder_use_past_context"] = any(str(k).startswith("decoder.past_encoder.") for k in state.keys())

    pe = state.get("decoder.past_encoder.pos_emb.weight", None)
    cfg["decoder_past_k"] = int(pe.shape[0]) if isinstance(pe, torch.Tensor) and pe.ndim == 2 else 8
    return cfg


def _pick_routes_per_city(routes, *, city: int, n_routes: int, max_way_len: int, seed: int) -> np.ndarray:
    keep = (routes.route_city.astype(np.int64) == int(city)) & (routes.way_seq_len > 1) & (routes.way_seq_len <= int(max_way_len))
    ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
    rng = np.random.default_rng(int(seed) + 101 * int(city))
    rng.shuffle(ids)
    return ids[: min(int(n_routes), int(ids.size))]


def _topk_with_ids(logits: torch.Tensor, cand: torch.Tensor, k: int) -> List[Dict[str, Any]]:
    # logits: (C,), cand: (C,)
    k = max(1, min(int(k), int(logits.numel())))
    v, ix = torch.topk(logits, k=k, largest=True, sorted=True)
    out: List[Dict[str, Any]] = []
    for vv, ii in zip(v.tolist(), ix.tolist()):
        out.append({"way": int(cand[int(ii)].item()), "logit": float(vv)})
    return out


def _rank_of_way(logits: torch.Tensor, cand: torch.Tensor, way: int) -> Optional[int]:
    # 1 = highest
    mask = cand == int(way)
    if not bool(mask.any().item()):
        return None
    pos = int(torch.nonzero(mask, as_tuple=False)[0].item())
    # rank by descending logit
    order = torch.argsort(logits, descending=True)
    rank = int(torch.nonzero(order == pos, as_tuple=False)[0].item()) + 1
    return int(rank)


@torch.no_grad()
def _trace_one_hit_wall(
    *,
    ae: WayCASDAutoEncoder,
    routes,
    rid: int,
    cfg: Cfg,
    device: torch.device,
) -> Dict[str, Any]:
    L = int(routes.way_seq_len[rid])
    s = int(routes.way_seq_ptr[rid])
    gt = routes.way_seq_idx[s : s + L].astype(np.int64).tolist()
    gt = [int(x) for x in gt]

    city = int(routes.route_city[rid])
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
    z_enc, _ = ae.encode(way_pad_t)  # (1, n_latent, d)

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

    max_candidates = int(cfg.decode_max_candidates)
    if max_candidates < 0:
        max_candidates = int(ae.cfg.max_candidates)
    max_cand_opt = None if int(max_candidates) <= 0 else int(max_candidates)

    path: List[int] = [int(start_way)]
    gt_prefix_ok = bool(gt) and int(gt[0]) == int(start_way)

    # Only keep logs for last K steps
    last_logs: List[Dict[str, Any]] = []
    last_k = int(cfg.last_k_steps)
    topk = int(cfg.topk_logits)

    for step_idx in range(int(cfg.max_decode_len)):
        cur = int(path[-1])
        if cur == int(dest_way):
            break

        cand_full = ae.decoder.get_succ_candidates(cur)
        if int(cand_full.numel()) == 0:
            break

        cand_sel = ae.decoder._select_decode_candidates(
            way_embedder=ae.way_enc,
            cand_full=cand_full.to(device=device),
            dest_pos=route_cond["dest_pos"],
            dest_way=int(dest_way),
            max_candidates=max_cand_opt,
            candidate_policy=str(cfg.decode_candidate_policy),
            include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
        )
        C = int(cand_sel.numel())
        cand_way = cand_sel.view(1, C).to(device=device)
        cand_mask = torch.ones((1, C), dtype=torch.bool, device=device)

        # past context (optional)
        trans: Dict[str, torch.Tensor] = {
            "route_idx": torch.as_tensor([0], dtype=torch.long, device=device),
            "cur_way": torch.as_tensor([cur], dtype=torch.long, device=device),
            "cand_way": cand_way,
            "cand_mask": cand_mask,
            "step": torch.as_tensor([step_idx], dtype=torch.long, device=device),
        }
        if bool(ae.cfg.decoder_use_past_context):
            pk = int(ae.cfg.decoder_past_k)
            past_seq = path[:-1]
            past_len = min(len(past_seq), pk)
            past_row = np.full((pk,), -1, dtype=np.int64)
            if past_len > 0:
                past_row[pk - past_len :] = np.asarray(past_seq[-past_len:], dtype=np.int64)
            past_mask = (past_row >= 0)
            trans["past_way"] = torch.as_tensor(past_row[None, :], dtype=torch.long, device=device)
            trans["past_mask"] = torch.as_tensor(past_mask[None, :], dtype=torch.bool, device=device)

        logits = ae.decoder.score_candidates(
            way_embedder=ae.way_enc,
            latent_tokens=z_enc,
            route_cond=route_cond,
            trans=trans,
            cond_emb=cond_emb,
        )[0]  # (C,)

        # Choose next (greedy)
        j = int(torch.argmax(logits, dim=-1).item())
        pred_next = int(cand_sel[j].item())

        # Save last-k logs (steps near the end)
        if step_idx >= int(cfg.max_decode_len) - last_k:
            dest_rank = _rank_of_way(logits, cand_sel, int(dest_way))
            gt_next = None
            gt_next_logit = None
            gt_next_rank = None
            chosen_logit = float(logits[j].item())
            gt_pos = None

            if gt_prefix_ok:
                # If we are still exactly on GT prefix, then GT next is well-defined.
                if int(step_idx + 1) < len(gt):
                    gt_next = int(gt[step_idx + 1])
                    gt_next_rank = _rank_of_way(logits, cand_sel, int(gt_next))
                    if gt_next_rank is not None:
                        # direct fetch
                        gt_pos = int(torch.nonzero(cand_sel == int(gt_next), as_tuple=False)[0].item())
                        gt_next_logit = float(logits[gt_pos].item())

            # Extra diagnostics (context/candidate/dist diffs)
            ctx_norm = None
            ctx_pred_norm = None
            ctx_gt_norm = None
            ctx_diff_norm = None
            cand_h_diff = None
            dist_pred_to_dest = None
            dist_gt_to_dest = None
            dist_pred_minus_gt = None
            try:
                # Candidate embedding diff
                cand_emb_dbg, _ = ae.way_enc(cand_way)  # (1,C,d)
                cand_h_dbg = ae.decoder.cand_proj(cand_emb_dbg)[0]  # (C,H)
                if gt_pos is not None:
                    cand_h_diff = float(torch.norm(cand_h_dbg[int(j)] - cand_h_dbg[int(gt_pos)]).item())

                # Context norms
                ctx_out_dbg = ae.decoder._compute_context(
                    way_embedder=ae.way_enc,
                    latent_tokens=z_enc,
                    cond_emb=cond_emb,
                    cur_way=trans["cur_way"],
                    cand_way=trans["cand_way"],
                    cand_mask=trans["cand_mask"],
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
                    if gt_pos is not None:
                        v_gt = ctx_out_dbg[0, int(gt_pos)]
                        ctx_gt_norm = float(torch.norm(v_gt).item())
                        ctx_diff_norm = float(torch.norm(v_pred - v_gt).item())

                # Distance-to-dest (normalized coord space)
                coord_scale = float(getattr(ae.way_enc, "coord_scale", ae.decoder.coord_scale))
                dest = route_cond["dest_pos"].to(dtype=torch.float32)
                if coord_scale > 0:
                    dest = dest / coord_scale
                cand_geom, _tier, _hw = ae.way_enc._lookup(cand_way)
                cand_center = cand_geom[..., :2].to(dtype=torch.float32)  # (1,C,2)
                dists = torch.norm(dest[:, None, :] - cand_center, dim=-1)[0]  # (C,)
                dist_pred_to_dest = float(dists[int(j)].item())
                if gt_pos is not None:
                    dist_gt_to_dest = float(dists[int(gt_pos)].item())
                    dist_pred_minus_gt = float(dist_pred_to_dest - dist_gt_to_dest)
            except Exception:
                pass

            last_logs.append(
                {
                    "step": int(step_idx),
                    "cur_way": int(cur),
                    "chosen_next": int(pred_next),
                    "chosen_logit": float(chosen_logit),
                    "topk": _topk_with_ids(logits, cand_sel, k=topk),
                    "dest_in_cands": bool(dest_rank is not None),
                    "dest_rank": int(dest_rank) if dest_rank is not None else None,
                    "gt_prefix_ok": bool(gt_prefix_ok),
                    "gt_next": int(gt_next) if gt_next is not None else None,
                    "gt_next_in_cands": bool(gt_next_rank is not None),
                    "gt_next_rank": int(gt_next_rank) if gt_next_rank is not None else None,
                    "gt_next_logit": float(gt_next_logit) if gt_next_logit is not None else None,
                    "chosen_minus_gt_next": float(chosen_logit - float(gt_next_logit)) if gt_next_logit is not None else None,
                    "ctx_norm": ctx_norm,
                    "ctx_pred_norm": ctx_pred_norm,
                    "ctx_gt_norm": ctx_gt_norm,
                    "ctx_diff_norm": ctx_diff_norm,
                    "cand_h_diff": cand_h_diff,
                    "dist_pred_to_dest": dist_pred_to_dest,
                    "dist_gt_to_dest": dist_gt_to_dest,
                    "dist_pred_minus_gt": dist_pred_minus_gt,
                }
            )

        path.append(int(pred_next))
        if gt_prefix_ok and int(step_idx + 1) < len(gt) and int(pred_next) != int(gt[step_idx + 1]):
            gt_prefix_ok = False

    success = bool(path and int(path[-1]) == int(dest_way))
    hit_wall = (not success) and (len(path) >= int(cfg.max_decode_len) + 1)

    return {
        "route_id": int(rid),
        "city": int(city),
        "gt_len": int(len(gt)),
        "pred_len": int(len(path)),
        "success": bool(success),
        "hit_wall": bool(hit_wall),
        "last_k_steps": int(cfg.last_k_steps),
        "steps": last_logs,
    }


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

    way_features = load_way_features_from_npz(Path(way_features_npz), device=device)
    wf = np.load(str(way_features_npz), allow_pickle=True)
    n_highway_types_data = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1
    n_cities_data = int(np.max(routes.route_city.astype(np.int64))) + 1

    ckpt = torch.load(str(ae_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ae_cfg_dict = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    inferred = _infer_decoder_cfg_from_state(state) if isinstance(state, dict) else {}

    # Infer embedding sizes from checkpoint to avoid shape mismatch (e.g., route_city_embed).
    n_route_cities_ckpt = None
    n_highway_types_ckpt = None
    if isinstance(state, dict):
        w_city = state.get("decoder.cond_enc.route_city_embed.weight", None)
        if isinstance(w_city, torch.Tensor) and w_city.ndim == 2:
            n_route_cities_ckpt = int(w_city.shape[0])
        w_hw = state.get("way_enc.highway_embed.weight", None)
        if isinstance(w_hw, torch.Tensor) and w_hw.ndim == 2:
            n_highway_types_ckpt = int(w_hw.shape[0])

    n_route_cities = int(ae_cfg_dict.get("n_route_cities", n_route_cities_ckpt or n_cities_data))
    n_route_cities = max(int(n_route_cities), int(n_cities_data))  # must cover observed city ids
    n_highway_types = int(ae_cfg_dict.get("n_highway_types", n_highway_types_ckpt or n_highway_types_data))
    n_highway_types = max(int(n_highway_types), int(n_highway_types_data))

    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(ae_cfg_dict.get("d_model", 256)),
            n_latent=int(ae_cfg_dict.get("n_latent", 64)),
            n_heads=int(ae_cfg_dict.get("n_heads", 8)),
            dropout=float(ae_cfg_dict.get("dropout", 0.1)),
            max_candidates=int(ae_cfg_dict.get("max_candidates", 32)),
            max_len=int(ae_cfg_dict.get("max_len", cfg.max_way_len)),
            coord_scale=float(ae_cfg_dict.get("coord_scale", 1024.0)),
            decoder_use_dest_dist=bool(inferred.get("decoder_use_dest_dist", True)),
            decoder_use_cross_attn=bool(inferred.get("decoder_use_cross_attn", True)),
            decoder_n_cross_heads=int(ae_cfg_dict.get("decoder_n_cross_heads", 4)),
            decoder_use_step_emb=bool(inferred.get("decoder_use_step_emb", False)),
            decoder_use_dest_query=bool(inferred.get("decoder_use_dest_query", False)),
            decoder_use_dir_query=bool(inferred.get("decoder_use_dir_query", False)),
            decoder_use_cand_query=bool(inferred.get("decoder_use_cand_query", False)),
            decoder_use_past_context=bool(inferred.get("decoder_use_past_context", False)),
            decoder_past_k=int(inferred.get("decoder_past_k", 8)),
            decoder_past_n_layers=int(ae_cfg_dict.get("decoder_past_n_layers", 2)),
            decoder_past_n_heads=int(ae_cfg_dict.get("decoder_past_n_heads", 4)),
        ),
        way_features=way_features,
        way_adj_ptr=ptr,
        way_adj_idx=idx,
        n_route_cities=int(n_route_cities),
        n_highway_types=int(n_highway_types),
    ).to(device=device)
    ae.load_state_dict(state, strict=False)
    ae.eval()

    # Scan routes and collect hit-wall examples.
    picks: Dict[int, np.ndarray] = {}
    for c in range(n_cities_data):
        picks[c] = _pick_routes_per_city(routes, city=int(c), n_routes=int(cfg.n_routes), max_way_len=int(cfg.max_way_len), seed=int(cfg.seed))

    scan = []
    hit_wall_cases: List[Dict[str, Any]] = []
    t0 = time.time()
    n_scanned = 0
    n_hit_wall = 0
    n_success = 0

    for c in range(n_cities_data):
        for rid in picks[c].tolist():
            rid = int(rid)
            n_scanned += 1
            rep = _trace_one_hit_wall(ae=ae, routes=routes, rid=rid, cfg=cfg, device=device)
            scan.append({"route_id": int(rid), "city": int(c), "success": bool(rep["success"]), "hit_wall": bool(rep["hit_wall"])})
            if bool(rep["success"]):
                n_success += 1
            if bool(rep["hit_wall"]):
                n_hit_wall += 1
                if len(hit_wall_cases) < int(cfg.n_cases):
                    hit_wall_cases.append(rep)
            if len(hit_wall_cases) >= int(cfg.n_cases) and n_scanned >= int(cfg.n_routes) * int(n_cities_data):
                # already have enough cases; still finish scan for stats? (KISS: stop early)
                pass

    dt = max(1e-6, time.time() - t0)
    out: Dict[str, Any] = {
        "ok": True,
        "task": "way_casd_hit_wall_trace",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "way_routes_npz": str(way_routes_npz),
            "way_graph_npz": str(way_graph_npz),
            "way_features_npz": str(way_features_npz),
            "ae_ckpt": str(ae_ckpt),
        },
        "ckpt_decoder_cfg_inferred": inferred,
        "scan_stats": {
            "n_cities": int(n_cities_data),
            "n_scanned": int(n_scanned),
            "n_success": int(n_success),
            "n_hit_wall": int(n_hit_wall),
            "success_rate": float(n_success) / float(max(1, n_scanned)),
            "hit_wall_rate": float(n_hit_wall) / float(max(1, n_scanned)),
            "elapsed_s": float(dt),
        },
        "hit_wall_cases": hit_wall_cases,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(str(out_json))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="PI diag3: dump last-K-step scorer behavior for hit-wall cases.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--n_routes", type=int, default=200, help="How many routes to scan per city.")
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--max_decode_len", type=int, default=160)
    p.add_argument("--decode_max_candidates", type=int, default=0, help="0=all successors; -1=model default; >0 cap.")
    p.add_argument("--decode_candidate_policy", type=str, default="first", choices=["first", "destdist"])
    p.add_argument("--decode_include_dest_if_successor", action="store_true")
    p.add_argument("--last_k_steps", type=int, default=5)
    p.add_argument("--topk_logits", type=int, default=5)
    p.add_argument("--n_cases", type=int, default=3)
    args = p.parse_args()

    cfg = Cfg(
        seed=int(args.seed),
        device=str(args.device),
        tz_offset_hours=float(args.tz_offset_hours),
        n_routes=int(args.n_routes),
        max_way_len=int(args.max_way_len),
        max_decode_len=int(args.max_decode_len),
        decode_max_candidates=int(args.decode_max_candidates),
        decode_candidate_policy=str(args.decode_candidate_policy),
        decode_include_dest_if_successor=bool(args.decode_include_dest_if_successor),
        last_k_steps=int(args.last_k_steps),
        topk_logits=int(args.topk_logits),
        n_cases=int(args.n_cases),
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
