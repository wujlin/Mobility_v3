"""
z_enc Informativeness Diagnostic

核心问题：z_enc 到底有没有提供可用的路径信息？

实验设计（三组对照）：
1. true: 用本 route 的 z_enc
2. shuffle: 同 batch/同城随机打乱 z_enc（同分布，破坏对应关系）
3. zero: 全零 z_enc（强干预，测下限）

观察指标：
- oracle_success: 是否到达终点
- jaccard: 预测路径与 GT 的相似度

解读：
- 若 true ≈ shuffle：说明 decoder 基本没在用 z_enc
- 若 true 明显好于 shuffle：说明 z_enc 携带可用路径信息，问题在解码策略
"""

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
from src.models.way_casd.way_encoder import load_way_features_from_npz

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    seed: int
    device: str
    tz_offset_hours: float
    n_routes: int  # per city
    max_way_len: int
    max_decode_len: int
    decode_max_candidates: int
    decode_candidate_policy: str
    decode_include_dest_if_successor: bool
    decode_guided_dest_alpha: float
    decode_batch_size: int


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
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter) / float(union) if union > 0 else 0.0


def _first_diverge_step(gt: List[int], pred: List[int]) -> Optional[int]:
    """
    Return the first step index where pred diverges from gt (0-based).
    If sequences are identical, return None.
    If one is a strict prefix of the other, divergence is at the first missing index.
    """
    n = min(len(gt), len(pred))
    for i in range(n):
        if int(gt[i]) != int(pred[i]):
            return int(i)
    if len(gt) != len(pred):
        return int(n)
    return None


def _quantiles_int(values: List[int], qs: Tuple[int, ...] = (0, 50, 90, 95, 99, 100)) -> Dict[str, Optional[int]]:
    if not values:
        return {f"p{q:02d}": None for q in qs}
    arr = np.asarray(values, dtype=np.float64)
    out: Dict[str, Optional[int]] = {}
    for q in qs:
        out[f"p{q:02d}"] = int(np.percentile(arr, float(q)))
    return out


def _infer_decoder_config_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """Infer decoder config from checkpoint state dict."""
    cfg = {}
    
    # scorer features: use_dest_dist + use_cand_contrast
    w = state.get("decoder.scorer.0.weight", None)
    if w is not None and isinstance(w, torch.Tensor) and w.ndim == 2:
        hidden = int(w.shape[0])
        in_dim = int(w.shape[1])
        # Old: in_dim = 3*hidden (+1 if dest_dist)
        # New: in_dim = 4*hidden (+1 if dest_dist) when cand_contrast enabled.
        d4 = int(in_dim - hidden * 4)
        d3 = int(in_dim - hidden * 3)
        if d4 in (0, 1):
            cfg["decoder_use_cand_contrast"] = True
            cfg["decoder_use_dest_dist"] = bool(d4 == 1)
        elif d3 in (0, 1):
            cfg["decoder_use_cand_contrast"] = False
            cfg["decoder_use_dest_dist"] = bool(d3 == 1)
        else:
            cfg["decoder_use_cand_contrast"] = False
            cfg["decoder_use_dest_dist"] = True
    else:
        cfg["decoder_use_dest_dist"] = True
        cfg["decoder_use_cand_contrast"] = False
    
    cfg["decoder_use_cross_attn"] = any(str(k).startswith("decoder.cross_attn.") for k in state.keys())
    cfg["decoder_use_step_emb"] = any(str(k).startswith("decoder.step_emb.") for k in state.keys())
    cfg["decoder_use_dest_query"] = any(str(k).startswith("decoder.dest_proj.") for k in state.keys())
    cfg["decoder_use_dir_query"] = any(str(k).startswith("decoder.dir_query_proj.") for k in state.keys())
    cfg["decoder_use_cand_query"] = any(str(k).startswith("decoder.cand_query_proj.") for k in state.keys())
    cfg["decoder_use_past_context"] = any(str(k).startswith("decoder.past_encoder.") for k in state.keys())
    
    # past_k from pos_emb
    pe = state.get("decoder.past_encoder.pos_emb.weight", None)
    if isinstance(pe, torch.Tensor) and pe.ndim == 2:
        cfg["decoder_past_k"] = int(pe.shape[0])
    else:
        cfg["decoder_past_k"] = 8
    
    return cfg


@torch.no_grad()
def run(
    cfg: Cfg,
    *,
    way_routes_npz: Path,
    way_graph_npz: Path,
    way_features_npz: Path,
    ae_ckpt: Path,
    out_json: Path,
) -> Dict[str, object]:
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    _set_seed(cfg.seed)

    # Load data
    routes = load_way_routes_npz(Path(way_routes_npz))
    wg = np.load(str(way_graph_npz), allow_pickle=True)
    wf = np.load(str(way_features_npz), allow_pickle=True)
    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64)
    idx = np.asarray(wg["way_adj_idx"], dtype=np.int64)

    # Build AE
    way_features = load_way_features_from_npz(Path(way_features_npz), device=device)
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1

    ckpt = torch.load(str(ae_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ae_cfg_dict = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    inferred = _infer_decoder_config_from_state(state) if isinstance(state, dict) else {}

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
            decoder_use_cand_contrast=bool(inferred.get("decoder_use_cand_contrast", False)),
            decoder_use_past_context=bool(inferred.get("decoder_use_past_context", False)),
            decoder_past_k=int(inferred.get("decoder_past_k", 8)),
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
        print(f"[WARN] strict load_state_dict failed, fallback to strict=False: {e}")
        ae.load_state_dict(state, strict=False)
    ae.eval()

    # Sample routes per city
    def _pick_city(city: int) -> np.ndarray:
        keep = (routes.route_city.astype(np.int64) == int(city)) & \
               (routes.way_seq_len > 1) & \
               (routes.way_seq_len <= int(cfg.max_way_len))
        ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
        rng = np.random.default_rng(int(cfg.seed) + 101 * int(city))
        rng.shuffle(ids)
        return ids[: min(int(cfg.n_routes), int(ids.size))]

    picks = {0: _pick_city(0), 1: _pick_city(1)}
    
    max_candidates = int(cfg.decode_max_candidates)
    if max_candidates < 0:
        max_candidates = int(ae.cfg.max_candidates)

    # Pre-encode all selected routes
    all_rids: List[int] = []
    all_z_enc: List[torch.Tensor] = []
    all_gt: List[List[int]] = []
    all_meta: List[Dict] = []

    for city in (0, 1):
        for rid in picks[int(city)].tolist():
            rid = int(rid)
            L = int(routes.way_seq_len[rid])
            s = int(routes.way_seq_ptr[rid])
            gt = routes.way_seq_idx[s : s + L].astype(np.int64).tolist()
            gt = [int(x) for x in gt]

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

            all_rids.append(rid)
            all_z_enc.append(z_enc)
            all_gt.append(gt)
            all_meta.append({
                "city": int(city),
                "start_way": start_way,
                "dest_way": dest_way,
                "start_pos": start_pos,
                "dest_pos": dest_pos,
                "hour": hour,
                "dow": dow,
            })

    N = len(all_rids)
    print(f"Encoded {N} routes")
    if N <= 0:
        raise RuntimeError("No routes selected for diagnosis")

    # Cache route-level tensors for batched decoding.
    start_pos_all = torch.as_tensor(
        np.stack([m["start_pos"] for m in all_meta], axis=0),
        dtype=torch.float32,
        device=device,
    )
    dest_pos_all = torch.as_tensor(
        np.stack([m["dest_pos"] for m in all_meta], axis=0),
        dtype=torch.float32,
        device=device,
    )
    hour_all = torch.as_tensor([int(m["hour"]) for m in all_meta], dtype=torch.long, device=device)
    dow_all = torch.as_tensor([int(m["dow"]) for m in all_meta], dtype=torch.long, device=device)
    city_all = torch.as_tensor([int(m["city"]) for m in all_meta], dtype=torch.long, device=device)
    start_way_all = torch.as_tensor([int(m["start_way"]) for m in all_meta], dtype=torch.long, device=device)
    dest_way_all = torch.as_tensor([int(m["dest_way"]) for m in all_meta], dtype=torch.long, device=device)

    # Create shuffle permutation (within each city to stay in-distribution)
    rng = np.random.default_rng(cfg.seed + 999)
    city0_idx = [i for i in range(N) if all_meta[i]["city"] == 0]
    city1_idx = [i for i in range(N) if all_meta[i]["city"] == 1]
    
    shuffle_map = list(range(N))
    if len(city0_idx) > 1:
        perm0 = rng.permutation(len(city0_idx)).tolist()
        for i, j in enumerate(city0_idx):
            shuffle_map[j] = city0_idx[perm0[i]]
    if len(city1_idx) > 1:
        perm1 = rng.permutation(len(city1_idx)).tolist()
        for i, j in enumerate(city1_idx):
            shuffle_map[j] = city1_idx[perm1[i]]

    # Create zero z_enc
    z_shape = all_z_enc[0].shape
    z_zero = torch.zeros(z_shape, dtype=all_z_enc[0].dtype, device=device)
    decode_batch_size = max(1, int(cfg.decode_batch_size))

    def _decode_paths_batched(idxs: List[int], z_batch: torch.Tensor) -> List[List[int]]:
        if len(idxs) <= 0:
            return []
        t_idx = torch.as_tensor(idxs, dtype=torch.long, device=device)
        route_cond = {
            "start_pos": start_pos_all[t_idx],
            "dest_pos": dest_pos_all[t_idx],
            "hour": hour_all[t_idx],
            "dow": dow_all[t_idx],
            "route_city": city_all[t_idx],
        }
        sw_t = start_way_all[t_idx]
        dw_t = dest_way_all[t_idx]
        paths = ae.decoder.greedy_decode_batched(
            way_embedder=ae.way_enc,
            latent_tokens=z_batch,
            route_cond=route_cond,
            start_way=sw_t,
            dest_way=dw_t,
            max_len=int(cfg.max_decode_len),
            max_candidates=max_candidates,
            candidate_policy=str(cfg.decode_candidate_policy),
            include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
            guided_dest_alpha=float(cfg.decode_guided_dest_alpha),
        )
        return [[int(x) for x in p] for p in paths]

    def _run_condition(cond: str) -> List[Dict[str, object]]:
        out_rows: List[Dict[str, object]] = []
        for s in range(0, N, decode_batch_size):
            idxs = list(range(s, min(N, s + decode_batch_size)))
            if cond == "true":
                z_batch = torch.cat([all_z_enc[i] for i in idxs], dim=0)
            elif cond == "shuffle":
                z_batch = torch.cat([all_z_enc[shuffle_map[i]] for i in idxs], dim=0)
            elif cond == "zero":
                z_batch = z_zero.expand(len(idxs), -1, -1)
            else:
                raise ValueError(f"unsupported condition: {cond}")

            preds = _decode_paths_batched(idxs, z_batch)
            for j, ridx in enumerate(idxs):
                pred = preds[j]
                gt = all_gt[ridx]
                succ = bool(pred and int(pred[-1]) == int(all_meta[ridx]["dest_way"]))
                jac = _jaccard(gt, pred)
                out_rows.append(
                    {
                        "success": bool(succ),
                        "jaccard": float(jac),
                        "city": int(all_meta[ridx]["city"]),
                        "route_id": int(all_rids[ridx]),
                        "_pred": pred,  # internal use for true analysis
                        "_gt": gt,      # internal use for true analysis
                    }
                )
        return out_rows

    # Run three conditions
    results = {"true": [], "shuffle": [], "zero": []}
    true_per_route: List[Dict[str, object]] = []

    print("Running true z_enc (batched)...")
    true_rows = _run_condition("true")
    for r in true_rows:
        pred = r["_pred"]
        gt = r["_gt"]
        succ = bool(r["success"])
        jac = float(r["jaccard"])
        div = _first_diverge_step(gt, pred)
        seq_exact = (len(gt) == len(pred)) and all(int(a) == int(b) for a, b in zip(gt, pred))
        jac_1 = bool(abs(float(jac) - 1.0) < 1e-12)
        results["true"].append(
            {
                "success": succ,
                "jaccard": jac,
                "city": int(r["city"]),
                "route_id": int(r["route_id"]),
            }
        )
        true_per_route.append(
            {
                "route_id": int(r["route_id"]),
                "city": int(r["city"]),
                "gt_len": int(len(gt)),
                "pred_len": int(len(pred)),
                "success": bool(succ),
                "jaccard": float(jac),
                "jaccard_eq_1": bool(jac_1),
                "seq_exact": bool(seq_exact),
                "diverge_step": (int(div) if div is not None else None),
            }
        )

    print("Running shuffle z_enc (batched)...")
    for r in _run_condition("shuffle"):
        results["shuffle"].append(
            {
                "success": bool(r["success"]),
                "jaccard": float(r["jaccard"]),
                "city": int(r["city"]),
                "route_id": int(r["route_id"]),
            }
        )

    print("Running zero z_enc (batched)...")
    for r in _run_condition("zero"):
        results["zero"].append(
            {
                "success": bool(r["success"]),
                "jaccard": float(r["jaccard"]),
                "city": int(r["city"]),
                "route_id": int(r["route_id"]),
            }
        )

    # Aggregate
    def agg(lst: List[Dict]) -> Dict:
        succs = [bool(r.get("success", False)) for r in lst]
        jacs = [float(r.get("jaccard", 0.0)) for r in lst]
        return {
            "n": len(lst),
            "success_rate": sum(succs) / len(succs) if succs else 0.0,
            "jaccard_mean": sum(jacs) / len(jacs) if jacs else 0.0,
        }

    summary = {
        "true": agg(results["true"]),
        "shuffle": agg(results["shuffle"]),
        "zero": agg(results["zero"]),
    }

    summary_by_city: Dict[str, Dict[str, Dict[str, float]]] = {}
    for city in (0, 1):
        key = f"city{city}"
        summary_by_city[key] = {}
        for cond in ("true", "shuffle", "zero"):
            lst = [r for r in results[cond] if int(r.get("city", -1)) == int(city)]
            summary_by_city[key][cond] = agg(lst)

    # --- Additional diagnosis on true condition ---
    def _analyze_true(rows: List[Dict[str, object]]) -> Dict[str, object]:
        succ_rows = [r for r in rows if bool(r.get("success", False))]
        fail_rows = [r for r in rows if not bool(r.get("success", False))]

        def _frac(mask: List[bool]) -> float:
            return float(sum(bool(x) for x in mask)) / float(len(mask)) if mask else 0.0

        fail_div_steps = [int(r["diverge_step"]) for r in fail_rows if r.get("diverge_step") is not None]
        succ_div_steps = [int(r["diverge_step"]) for r in succ_rows if r.get("diverge_step") is not None]

        out: Dict[str, object] = {
            "n": int(len(rows)),
            "n_success": int(len(succ_rows)),
            "n_fail": int(len(fail_rows)),
            "success_rate": float(len(succ_rows)) / float(len(rows)) if rows else 0.0,
            "success_seq_exact_rate": _frac([bool(r.get("seq_exact", False)) for r in succ_rows]),
            "success_jaccard_eq_1_rate": _frac([bool(r.get("jaccard_eq_1", False)) for r in succ_rows]),
            "success_diverged_rate": _frac([r.get("diverge_step") is not None and not bool(r.get("seq_exact", False)) for r in succ_rows]),
            "fail_diverge_step_quantiles": _quantiles_int(fail_div_steps),
            "fail_diverge_le3_frac": _frac([r.get("diverge_step") is not None and int(r["diverge_step"]) <= 3 for r in fail_rows if r.get("diverge_step") is not None]),
            "fail_diverge_le5_frac": _frac([r.get("diverge_step") is not None and int(r["diverge_step"]) <= 5 for r in fail_rows if r.get("diverge_step") is not None]),
            "success_diverge_step_quantiles": _quantiles_int(succ_div_steps),
        }

        # A few examples to inspect quickly
        succ_non_exact = [r for r in succ_rows if not bool(r.get("seq_exact", False))]
        succ_non_exact.sort(key=lambda r: float(r.get("jaccard", 0.0)))
        fail_early = [r for r in fail_rows if r.get("diverge_step") is not None]
        fail_early.sort(key=lambda r: int(r.get("diverge_step", 10**9)))
        out["examples_success_non_exact"] = [int(r["route_id"]) for r in succ_non_exact[:10]]
        out["examples_fail_early_diverge"] = [int(r["route_id"]) for r in fail_early[:10]]
        return out

    true_analysis = _analyze_true(true_per_route)
    true_analysis_by_city: Dict[str, Dict[str, object]] = {}
    for city in (0, 1):
        rows_city = [r for r in true_per_route if int(r.get("city", -1)) == int(city)]
        true_analysis_by_city[f"city{city}"] = _analyze_true(rows_city)

    # Interpretation
    true_succ = summary["true"]["success_rate"]
    shuffle_succ = summary["shuffle"]["success_rate"]
    zero_succ = summary["zero"]["success_rate"]
    
    interpretation = []
    if abs(true_succ - shuffle_succ) < 0.05:
        interpretation.append("true ≈ shuffle: Decoder 基本没在用 z_enc，或 z_enc 无可用信息")
    elif true_succ > shuffle_succ + 0.05:
        interpretation.append(f"true > shuffle (+{true_succ - shuffle_succ:.2%}): z_enc 携带可用路径信息，问题在解码策略")
    
    if shuffle_succ > zero_succ + 0.05:
        interpretation.append("shuffle > zero: 同分布的 z_enc 比全零好，说明 z_enc 结构本身有用")
    
    out = {
        "ok": True,
        "task": "way_casd_zenc_informativeness",
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
        "n_routes_total": N,
        "summary": summary,
        "summary_by_city": summary_by_city,
        "true_analysis": true_analysis,
        "true_analysis_by_city": true_analysis_by_city,
        "true_per_route": true_per_route,
        "interpretation": interpretation,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_json), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_json}")

    # Print summary
    print("\n" + "=" * 60)
    print("z_enc Informativeness Diagnostic Results")
    print("=" * 60)
    for cond in ["true", "shuffle", "zero"]:
        s = summary[cond]
        print(f"  {cond:8s}: success={s['success_rate']:.2%}, jaccard={s['jaccard_mean']:.4f}")
    print("-" * 60)
    for city_key in ("city0", "city1"):
        if city_key in summary_by_city:
            s0 = summary_by_city[city_key]
            print(
                f"  [{city_key}] "
                + " | ".join(
                    f"{cond}:{s0[cond]['success_rate']:.2%}/{s0[cond]['jaccard_mean']:.3f}"
                    for cond in ("true", "shuffle", "zero")
                )
            )
    print("-" * 60)
    for line in interpretation:
        print(f"  → {line}")
    print("=" * 60)

    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="z_enc informativeness diagnostic")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--n_routes", type=int, default=100)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--max_decode_len", type=int, default=160)
    p.add_argument("--decode_max_candidates", type=int, default=-1)
    p.add_argument("--decode_candidate_policy", type=str, default="first", choices=["first", "destdist"])
    p.add_argument("--decode_include_dest_if_successor", action="store_true")
    p.add_argument("--decode_guided_dest_alpha", type=float, default=0.0)
    p.add_argument("--decode_batch_size", type=int, default=256)
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
        decode_max_candidates=int(args.decode_max_candidates),
        decode_candidate_policy=str(args.decode_candidate_policy),
        decode_include_dest_if_successor=bool(args.decode_include_dest_if_successor),
        decode_guided_dest_alpha=float(args.decode_guided_dest_alpha),
        decode_batch_size=int(args.decode_batch_size),
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
