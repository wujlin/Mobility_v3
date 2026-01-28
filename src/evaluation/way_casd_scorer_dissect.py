"""
诊断脚本：分解 WayDecoder 的 scorer 在首次分歧点的各维度贡献。

核心问题：为什么 gt_rank=2 如此稳定（93% 的失败样本）？

分解目标：
- scorer 输入 = [ctx_h, cur_h, cand_h, dist]
- 对于 pred_next 和 gt_next，分别计算各部分的差异

关键指标：
1. cand_h 差异：pred 和 gt 的 cand_h cosine similarity
2. ctx_h · cand_h：ctx 与 pred/gt 的 dot product 差
3. cur_h · cand_h：cur 与 pred/gt 的 dot product 差
4. dist 差异：pred 和 gt 到终点的距离差
5. scorer 中间层输出差异
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz


@dataclass(frozen=True)
class Cfg:
    seed: int
    device: str
    tz_offset_hours: float
    n_routes: int
    max_way_len: int
    max_decode_len: int


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _hour_from_unix(start_t: int, tz_offset_hours: float) -> int:
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    sec = int((int(start_t) + tz_sec) % 86400)
    return int(sec // 3600)


def _dow_from_unix(start_t: int, tz_offset_hours: float) -> int:
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    days = int((int(start_t) + tz_sec) // 86400)
    return int((days + 3) % 7)


def _infer_decoder_use_dest_dist_from_state(state: Dict[str, torch.Tensor]) -> bool:
    w = state.get("decoder.scorer.0.weight", None)
    if w is not None and isinstance(w, torch.Tensor) and w.ndim == 2:
        hidden = int(w.shape[0])
        in_dim = int(w.shape[1])
        delta = int(in_dim - hidden * 3)
        return bool(delta != 0)
    return True


def _infer_flag(state: Dict[str, torch.Tensor], prefix: str) -> bool:
    return any(str(k).startswith(prefix) for k in state.keys())


def _infer_n_route_cities_from_state(state: Dict[str, torch.Tensor]) -> Optional[int]:
    w = state.get("decoder.cond_enc.route_city_embed.weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2 and int(w.shape[0]) > 0:
        return int(w.shape[0])
    return None


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

    wf = np.load(str(way_features_npz), allow_pickle=True)
    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1

    way_features = load_way_features_from_npz(Path(way_features_npz), device=device)
    ckpt = torch.load(str(ae_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    inferred = {
        "decoder_use_dest_dist": _infer_decoder_use_dest_dist_from_state(state) if isinstance(state, dict) else bool(ckpt_cfg.get("decoder_use_dest_dist", True)),
        "decoder_use_cross_attn": _infer_flag(state, "decoder.cross_attn.") if isinstance(state, dict) else bool(ckpt_cfg.get("decoder_use_cross_attn", True)),
        "decoder_use_past_context": _infer_flag(state, "decoder.past_encoder.") if isinstance(state, dict) else bool(ckpt_cfg.get("decoder_use_past_context", False)),
        "decoder_use_step_emb": _infer_flag(state, "decoder.step_emb.") if isinstance(state, dict) else bool(ckpt_cfg.get("decoder_use_step_emb", False)),
        "decoder_use_dest_query": _infer_flag(state, "decoder.dest_proj.") if isinstance(state, dict) else bool(ckpt_cfg.get("decoder_use_dest_query", False)),
        "decoder_use_dir_query": _infer_flag(state, "decoder.dir_query_proj.") if isinstance(state, dict) else bool(ckpt_cfg.get("decoder_use_dir_query", False)),
        "decoder_use_cand_query": _infer_flag(state, "decoder.cand_query_proj.") if isinstance(state, dict) else bool(ckpt_cfg.get("decoder_use_cand_query", False)),
    }

    ae_cfg = WayCASDAECfg(
        d_model=int(ckpt_cfg.get("d_model", 256)),
        n_latent=int(ckpt_cfg.get("n_latent", 64)),
        n_heads=int(ckpt_cfg.get("n_heads", 8)),
        dropout=float(ckpt_cfg.get("dropout", 0.1)),
        max_candidates=int(ckpt_cfg.get("max_candidates", 32)),
        max_len=int(ckpt_cfg.get("max_len", cfg.max_way_len)),
        coord_scale=float(ckpt_cfg.get("coord_scale", 1024.0)),
        decoder_use_dest_dist=bool(inferred["decoder_use_dest_dist"]),
        decoder_use_cross_attn=bool(inferred["decoder_use_cross_attn"]),
        decoder_n_cross_heads=int(ckpt_cfg.get("decoder_n_cross_heads", 4)),
        decoder_use_step_emb=bool(inferred["decoder_use_step_emb"]),
        decoder_use_dest_query=bool(inferred["decoder_use_dest_query"]),
        decoder_use_dir_query=bool(inferred["decoder_use_dir_query"]),
        decoder_use_cand_query=bool(inferred["decoder_use_cand_query"]),
        decoder_use_past_context=bool(inferred["decoder_use_past_context"]),
        decoder_past_k=int(ckpt_cfg.get("decoder_past_k", 8)),
        decoder_past_n_layers=int(ckpt_cfg.get("decoder_past_n_layers", 2)),
        decoder_past_n_heads=int(ckpt_cfg.get("decoder_past_n_heads", 4)),
    )

    n_route_cities = _infer_n_route_cities_from_state(state) if isinstance(state, dict) else None
    if n_route_cities is None:
        if routes.route_city is not None:
            n_route_cities = int(np.max(np.asarray(routes.route_city, dtype=np.int64))) + 1
        else:
            n_route_cities = 1

    ae = WayCASDAutoEncoder(
        cfg=ae_cfg,
        way_adj_ptr=ptr,
        way_adj_idx=idx,
        way_features=way_features,
        n_route_cities=int(n_route_cities),
        n_highway_types=int(max(4, n_highway_types)),
    ).to(device)
    strict_ok = True
    try:
        ae.load_state_dict(state, strict=True)
    except Exception:
        strict_ok = False
        ae.load_state_dict(state, strict=False)
    ae.eval()

    # Sample routes
    keep = (routes.way_seq_len > 1) & (routes.way_seq_len <= int(cfg.max_way_len))
    all_idx = np.nonzero(keep)[0].astype(np.int64, copy=False)
    rng = np.random.default_rng(int(cfg.seed))
    rng.shuffle(all_idx)
    sample_idx = all_idx[: min(int(cfg.n_routes), int(all_idx.size))]

    # Accumulate dissection stats
    dissections: List[Dict[str, Any]] = []

    for ri in sample_idx:
        rid = int(ri)
        L = int(routes.way_seq_len[rid])
        if L < 2 or L > int(cfg.max_way_len):
            continue
        s = int(routes.way_seq_ptr[rid])
        e = s + L
        gt = [int(x) for x in routes.way_seq_idx[s:e].tolist()]
        if len(gt) < 2:
            continue

        start_pos = np.asarray(routes.start_pos[rid], dtype=np.float64).reshape(2)
        dest_pos = np.asarray(routes.dest_pos[rid], dtype=np.float64).reshape(2)
        start_t = int(routes.start_t[rid])
        route_city = int(routes.route_city[rid])
        start_way = int(gt[0])
        dest_way = int(gt[-1])

        hour = _hour_from_unix(start_t, cfg.tz_offset_hours)
        dow = _dow_from_unix(start_t, cfg.tz_offset_hours)

        route_cond = {
            "start_pos": torch.tensor([start_pos], dtype=torch.float32, device=device),
            "dest_pos": torch.tensor([dest_pos], dtype=torch.float32, device=device),
            "hour": torch.tensor([hour], dtype=torch.long, device=device),
            "dow": torch.tensor([dow], dtype=torch.long, device=device),
            "route_city": torch.tensor([route_city], dtype=torch.long, device=device),
        }

        # Encode GT
        pad_n = int(cfg.max_way_len) - int(len(gt))
        way_seq_pad = torch.tensor([gt + ([-1] * max(0, pad_n))], dtype=torch.long, device=device)
        z_enc, _mask = ae.encode(way_seq_pad)

        cond_emb = ae.decoder.cond_enc(
            start_pos=route_cond["start_pos"],
            dest_pos=route_cond["dest_pos"],
            hour=route_cond["hour"],
            dow=route_cond["dow"],
            route_city=route_cond["route_city"],
        )

        # Decode step by step
        path = [start_way]
        cur = start_way
        gt_prefix_ok = True
        step_idx = 0
        
        while step_idx < cfg.max_decode_len and cur != dest_way:
            # Get successors
            s_ptr = int(ptr[cur])
            e_ptr = int(ptr[cur + 1])
            cand_full = idx[s_ptr:e_ptr] if e_ptr > s_ptr else np.array([], dtype=np.int64)
            if cand_full.size == 0:
                break
            
            cand_sel = torch.tensor(cand_full, dtype=torch.long, device=device)
            C = cand_sel.shape[0]
            
            # Get GT next
            gt_next = None
            if gt_prefix_ok and step_idx < len(gt) - 1 and gt[step_idx] == cur:
                gt_next = gt[step_idx + 1]
            
            # Build trans dict for scorer
            trans = {
                "route_idx": torch.tensor([0], dtype=torch.long, device=device),
                "cur_way": torch.tensor([cur], dtype=torch.long, device=device),
                "cand_way": cand_sel.unsqueeze(0),  # (1, C)
                "cand_mask": torch.ones((1, C), dtype=torch.bool, device=device),
                "step": torch.tensor([step_idx], dtype=torch.long, device=device),
            }
            
            # Build past context
            if ae.decoder.use_past_context:
                past_k = ae.decoder.past_k
                past_seq = path[:-1] if len(path) > 1 else []
                past_len = min(len(past_seq), past_k)
                past_row = [-1] * past_k
                if past_len > 0:
                    offset = past_k - past_len
                    past_row[offset:] = past_seq[-past_len:]
                trans["past_way"] = torch.tensor([past_row], dtype=torch.long, device=device)
                trans["past_mask"] = torch.tensor([[x >= 0 for x in past_row]], dtype=torch.bool, device=device)
            
            # Get scorer internals
            logits = ae.decoder.score_candidates(
                way_embedder=ae.way_enc,
                latent_tokens=z_enc,
                route_cond=route_cond,
                trans=trans,
                cond_emb=cond_emb,
            )
            
            pred_idx = int(torch.argmax(logits, dim=-1).item())
            pred_next = int(cand_sel[pred_idx].item())
            
            # Check for divergence
            if gt_next is not None and pred_next != gt_next:
                # Found first divergence, now dissect
                if gt_next in cand_full.tolist():
                    gt_idx = cand_full.tolist().index(gt_next)
                    
                    # Get intermediate representations
                    # Re-compute to get internal values
                    cur_emb, _ = ae.way_enc(trans["cur_way"][:, None])  # (1, 1, d_model)
                    cur_emb = cur_emb[:, 0, :]  # (1, d_model)
                    
                    cand_emb, _ = ae.way_enc(trans["cand_way"])  # (1, C, d_model)
                    
                    # ctx via cross-attention
                    ctx_t = ae.decoder._compute_context(
                        way_embedder=ae.way_enc,
                        latent_tokens=z_enc,
                        cond_emb=cond_emb,
                        cur_way=trans["cur_way"],
                        route_idx=trans["route_idx"],
                        step=trans["step"],
                        dest_pos=route_cond["dest_pos"],
                        past_way=trans.get("past_way"),
                        past_mask=trans.get("past_mask"),
                    )  # (1, hidden)
                    
                    cur_h = ae.decoder.cur_proj(cur_emb)  # (1, hidden)
                    cand_h = ae.decoder.cand_proj(cand_emb)  # (1, C, hidden)
                    
                    # Extract pred and gt candidates
                    cand_h_pred = cand_h[0, pred_idx, :]  # (hidden,)
                    cand_h_gt = cand_h[0, gt_idx, :]  # (hidden,)
                    ctx_h = ctx_t[0, :]  # (hidden,)
                    cur_h_vec = cur_h[0, :]  # (hidden,)
                    
                    # Compute dissection metrics
                    # 1. Cosine similarity between pred and gt cand_h
                    cos_sim = F.cosine_similarity(cand_h_pred.unsqueeze(0), cand_h_gt.unsqueeze(0)).item()
                    
                    # 2. Dot product with ctx
                    dot_ctx_pred = torch.dot(ctx_h, cand_h_pred).item()
                    dot_ctx_gt = torch.dot(ctx_h, cand_h_gt).item()
                    ctx_diff = dot_ctx_pred - dot_ctx_gt
                    
                    # 3. Dot product with cur
                    dot_cur_pred = torch.dot(cur_h_vec, cand_h_pred).item()
                    dot_cur_gt = torch.dot(cur_h_vec, cand_h_gt).item()
                    cur_diff = dot_cur_pred - dot_cur_gt
                    
                    # 4. L2 norm differences
                    norm_pred = torch.norm(cand_h_pred).item()
                    norm_gt = torch.norm(cand_h_gt).item()
                    norm_ctx = torch.norm(ctx_h).item()
                    norm_cur = torch.norm(cur_h_vec).item()
                    
                    # 5. Dest dist difference
                    dist_pred = None
                    dist_gt = None
                    if 0 <= pred_next < len(way_center_y) and 0 <= gt_next < len(way_center_y):
                        pred_cy, pred_cx = float(way_center_y[pred_next]), float(way_center_x[pred_next])
                        gt_cy, gt_cx = float(way_center_y[gt_next]), float(way_center_x[gt_next])
                        dy, dx = float(dest_pos[0]), float(dest_pos[1])
                        dist_pred = float(np.sqrt((pred_cy - dy) ** 2 + (pred_cx - dx) ** 2))
                        dist_gt = float(np.sqrt((gt_cy - dy) ** 2 + (gt_cx - dx) ** 2))
                    
                    # 6. Logit difference
                    logit_pred = logits[0, pred_idx].item()
                    logit_gt = logits[0, gt_idx].item()
                    logit_diff = logit_pred - logit_gt
                    
                    # 7. Get scorer first layer weights to understand contribution
                    # scorer: Linear(in_dim, hidden) -> SiLU -> Dropout -> Linear(hidden, 1)
                    w0 = ae.decoder.scorer[0].weight  # (hidden, in_dim)
                    b0 = ae.decoder.scorer[0].bias    # (hidden,)
                    
                    hidden_dim = int(ctx_h.shape[0])
                    
                    # Construct scorer input for pred and gt
                    if ae.decoder.cfg.use_dest_dist:
                        coord_scale = float(getattr(ae.way_enc, "coord_scale", 1024.0))
                        dest_norm = route_cond["dest_pos"][0].to(dtype=torch.float32) / coord_scale
                        
                        cand_geom_pred, _, _ = ae.way_enc._lookup(torch.tensor([[pred_next]], dtype=torch.long, device=device))
                        dist_pred_t = torch.norm(dest_norm - cand_geom_pred[0, 0, :2]).unsqueeze(0)
                        
                        cand_geom_gt, _, _ = ae.way_enc._lookup(torch.tensor([[gt_next]], dtype=torch.long, device=device))
                        dist_gt_t = torch.norm(dest_norm - cand_geom_gt[0, 0, :2]).unsqueeze(0)
                        
                        x_pred = torch.cat([ctx_h, cur_h_vec, cand_h_pred, dist_pred_t])
                        x_gt = torch.cat([ctx_h, cur_h_vec, cand_h_gt, dist_gt_t])
                    else:
                        x_pred = torch.cat([ctx_h, cur_h_vec, cand_h_pred])
                        x_gt = torch.cat([ctx_h, cur_h_vec, cand_h_gt])
                    
                    # Contribution analysis: which part of x contributes most to the difference?
                    # x = [ctx_h, cur_h, cand_h, (dist)]
                    # logit ≈ W2 @ SiLU(W1 @ x + b1) + b2
                    # First layer output difference
                    z1_pred = F.silu(w0 @ x_pred + b0)
                    z1_gt = F.silu(w0 @ x_gt + b0)
                    z1_diff = z1_pred - z1_gt  # (hidden,)
                    
                    # Get contribution from each input segment
                    # ctx: [0:hidden], cur: [hidden:2*hidden], cand: [2*hidden:3*hidden], dist: [3*hidden:]
                    w0_ctx = w0[:, :hidden_dim]
                    w0_cur = w0[:, hidden_dim:2*hidden_dim]
                    w0_cand = w0[:, 2*hidden_dim:3*hidden_dim]
                    
                    # Linear contribution (before SiLU)
                    lin_ctx = w0_ctx @ ctx_h  # same for pred and gt
                    lin_cur = w0_cur @ cur_h_vec  # same for pred and gt
                    lin_cand_pred = w0_cand @ cand_h_pred
                    lin_cand_gt = w0_cand @ cand_h_gt
                    lin_cand_diff = lin_cand_pred - lin_cand_gt  # (hidden,)
                    
                    if ae.decoder.cfg.use_dest_dist:
                        w0_dist = w0[:, 3*hidden_dim:]
                        lin_dist_pred = w0_dist @ dist_pred_t
                        lin_dist_gt = w0_dist @ dist_gt_t
                        lin_dist_diff = lin_dist_pred - lin_dist_gt
                    else:
                        lin_dist_diff = torch.zeros((hidden_dim,), device=device)
                    
                    dissections.append({
                        "route_id": int(ri),
                        "step_idx": int(step_idx),
                        "n_cand": int(C),
                        "pred_next": int(pred_next),
                        "gt_next": int(gt_next),
                        "logit_diff": float(logit_diff),
                        "cos_sim_pred_gt": float(cos_sim),
                        "dot_ctx_pred": float(dot_ctx_pred),
                        "dot_ctx_gt": float(dot_ctx_gt),
                        "ctx_diff": float(ctx_diff),
                        "dot_cur_pred": float(dot_cur_pred),
                        "dot_cur_gt": float(dot_cur_gt),
                        "cur_diff": float(cur_diff),
                        "norm_pred": float(norm_pred),
                        "norm_gt": float(norm_gt),
                        "norm_ctx": float(norm_ctx),
                        "norm_cur": float(norm_cur),
                        "dist_pred": dist_pred,
                        "dist_gt": dist_gt,
                        "dist_diff": float(dist_pred - dist_gt) if dist_pred is not None and dist_gt is not None else None,
                        "lin_cand_diff_mean": float(lin_cand_diff.mean().item()),
                        "lin_cand_diff_std": float(lin_cand_diff.std().item()),
                        "lin_dist_diff_mean": float(lin_dist_diff.mean().item()) if ae.decoder.cfg.use_dest_dist else None,
                        "z1_diff_mean": float(z1_diff.mean().item()),
                        "z1_diff_std": float(z1_diff.std().item()),
                    })
                
                break  # Only collect first divergence
            
            path.append(pred_next)
            cur = pred_next
            step_idx += 1
            
            if gt_next is not None and pred_next != gt_next:
                gt_prefix_ok = False

    # Aggregate statistics
    if dissections:
        agg = {
            "n_dissections": len(dissections),
            "logit_diff_mean": float(np.mean([d["logit_diff"] for d in dissections])),
            "logit_diff_std": float(np.std([d["logit_diff"] for d in dissections])),
            "cos_sim_mean": float(np.mean([d["cos_sim_pred_gt"] for d in dissections])),
            "ctx_diff_mean": float(np.mean([d["ctx_diff"] for d in dissections])),
            "ctx_diff_std": float(np.std([d["ctx_diff"] for d in dissections])),
            "cur_diff_mean": float(np.mean([d["cur_diff"] for d in dissections])),
            "cur_diff_std": float(np.std([d["cur_diff"] for d in dissections])),
            "lin_cand_diff_mean_mean": float(np.mean([d["lin_cand_diff_mean"] for d in dissections])),
            "z1_diff_mean_mean": float(np.mean([d["z1_diff_mean"] for d in dissections])),
            
            # Contribution signs (how often does each component favor pred?)
            "ctx_diff_positive_frac": float(np.mean([1.0 if d["ctx_diff"] > 0 else 0.0 for d in dissections])),
            "cur_diff_positive_frac": float(np.mean([1.0 if d["cur_diff"] > 0 else 0.0 for d in dissections])),
            "lin_cand_diff_positive_frac": float(np.mean([1.0 if d["lin_cand_diff_mean"] > 0 else 0.0 for d in dissections])),
            
            # Distance
            "dist_diff_mean": float(np.mean([d["dist_diff"] for d in dissections if d["dist_diff"] is not None])) if any(d["dist_diff"] is not None for d in dissections) else None,
            "dist_pred_closer_frac": float(np.mean([1.0 if d["dist_diff"] is not None and d["dist_diff"] < 0 else 0.0 for d in dissections if d["dist_diff"] is not None])) if any(d["dist_diff"] is not None for d in dissections) else None,
        }
    else:
        agg = {"n_dissections": 0}

    result = {
        "config": asdict(cfg),
        "aggregate": agg,
        "dissections": dissections[:50],  # Save first 50 for inspection
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {out_json}")
    print(f"\n=== Aggregate ===")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--way_routes_npz", type=str, required=True)
    parser.add_argument("--way_graph_npz", type=str, required=True)
    parser.add_argument("--way_features_npz", type=str, required=True)
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--n_routes", type=int, default=500)
    parser.add_argument("--max_way_len", type=int, default=128)
    parser.add_argument("--max_decode_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tz_offset_hours", type=float, default=-5.0)
    args = parser.parse_args()

    cfg = Cfg(
        seed=args.seed,
        device=args.device,
        tz_offset_hours=args.tz_offset_hours,
        n_routes=args.n_routes,
        max_way_len=args.max_way_len,
        max_decode_len=args.max_decode_len,
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
