from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _infer_use_dest_dist(state: Dict[str, torch.Tensor]) -> bool:
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


def _infer_use_cand_contrast(state: Dict[str, torch.Tensor]) -> bool:
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


def _infer_bool_by_prefix(state: Dict[str, torch.Tensor], prefix: str) -> bool:
    return any(str(k).startswith(prefix) for k in state.keys())


def _infer_n_route_cities(state: Dict[str, torch.Tensor]) -> Optional[int]:
    w = state.get("decoder.cond_enc.route_city_embed.weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2 and int(w.shape[0]) > 0:
        return int(w.shape[0])
    return None


def _seq_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    den = len(sa | sb)
    if den == 0:
        return 1.0
    return float(len(sa & sb) / float(den))


def _pairwise_diversity(seqs: List[List[int]]) -> float:
    m = len(seqs)
    if m < 2:
        return float("nan")
    vals: List[float] = []
    for i in range(m):
        for j in range(i + 1, m):
            vals.append(1.0 - _seq_jaccard(seqs[i], seqs[j]))
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def _pct(xs: List[float], q: float) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


@dataclass(frozen=True)
class Cfg:
    seed: int
    device: str
    tz_offset_hours: float
    n_routes: int
    min_hops: int
    max_way_len: int
    max_decode_len: int
    n_samples_per_route: int
    latent_noise_std: float
    decode_stochastic: bool
    temperature: float
    decode_max_candidates: int
    decode_candidate_policy: str
    decode_include_dest_if_successor: bool
    split_json: Optional[str]
    split_part: Optional[str]
    jaccard_threshold: float
    min_routes_per_od: int
    k_per_od: int
    dump_samples: bool
    progress_every: int


def _build_ae(
    *,
    ae_ckpt: Path,
    way_graph_npz: Path,
    way_features_npz: Path,
    device: torch.device,
) -> Tuple[WayCASDAutoEncoder, bool]:
    wg = np.load(str(way_graph_npz), allow_pickle=True)
    wf = np.load(str(way_features_npz), allow_pickle=True)
    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64)
    idx = np.asarray(wg["way_adj_idx"], dtype=np.int64)
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1

    way_features = load_way_features_from_npz(Path(way_features_npz), device=device)
    ckpt = torch.load(str(ae_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ae_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(state, dict):
        raise SystemExit("[FATAL] unexpected ckpt format (state_dict missing)")

    use_dest_dist = _infer_use_dest_dist(state)
    use_cand_contrast = _infer_use_cand_contrast(state) or bool(ae_cfg.get("decoder_use_cand_contrast", False))
    use_cross_attn = _infer_bool_by_prefix(state, "decoder.cross_attn.") or bool(ae_cfg.get("decoder_use_cross_attn", True))
    use_step_emb = _infer_bool_by_prefix(state, "decoder.step_emb.") or bool(ae_cfg.get("decoder_use_step_emb", False))
    use_dest_query = _infer_bool_by_prefix(state, "decoder.dest_proj.") or bool(ae_cfg.get("decoder_use_dest_query", False))
    use_dir_query = _infer_bool_by_prefix(state, "decoder.dir_query_proj.") or bool(ae_cfg.get("decoder_use_dir_query", False))
    use_cand_query = _infer_bool_by_prefix(state, "decoder.cand_query_proj.") or bool(ae_cfg.get("decoder_use_cand_query", False))
    use_past_context = _infer_bool_by_prefix(state, "decoder.past_encoder.") or bool(ae_cfg.get("decoder_use_past_context", False))
    past_k = int(ae_cfg.get("decoder_past_k", 8))
    if use_past_context:
        pe = state.get("decoder.past_encoder.pos_emb.weight", None)
        if isinstance(pe, torch.Tensor) and pe.ndim == 2 and int(pe.shape[0]) > 0:
            past_k = int(pe.shape[0])
    past_n_layers = int(ae_cfg.get("decoder_past_n_layers", 2))
    past_n_heads = int(ae_cfg.get("decoder_past_n_heads", 4))
    n_route_cities = _infer_n_route_cities(state)
    if n_route_cities is None:
        n_route_cities = int(ae_cfg.get("n_route_cities", 4))

    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(ae_cfg.get("d_model", 256)),
            n_latent=int(ae_cfg.get("n_latent", 64)),
            n_heads=int(ae_cfg.get("n_heads", 8)),
            dropout=float(ae_cfg.get("dropout", 0.1)),
            max_candidates=int(ae_cfg.get("max_candidates", 32)),
            max_len=int(ae_cfg.get("max_len", 160)),
            coord_scale=float(ae_cfg.get("coord_scale", 1024.0)),
            segment_size=int(ae_cfg.get("segment_size", 10)),
            segment_n_latent=int(ae_cfg.get("segment_n_latent", 0)),
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
    return ae, strict_ok


def _pick_route_ids(routes: Any, *, cfg: Cfg) -> Dict[int, np.ndarray]:
    split_ids: Optional[np.ndarray] = None
    if cfg.split_json is not None:
        if cfg.split_part is None:
            raise SystemExit("[FATAL] --split_part is required when --split_json is set.")
        split_path = Path(str(cfg.split_json))
        if not split_path.exists():
            raise SystemExit(f"[FATAL] file not found: {split_path}")
        split_obj = _read_json(split_path)
        splits = split_obj.get("splits", split_obj)
        ids_raw = splits.get(str(cfg.split_part), None) if isinstance(splits, dict) else None
        if ids_raw is None:
            raise SystemExit(f"[FATAL] split_json missing part={cfg.split_part!r}")
        split_ids = np.asarray([int(x) for x in list(ids_raw)], dtype=np.int64).reshape(-1)
        if int(split_ids.size) == 0:
            raise SystemExit(f"[FATAL] split {cfg.split_part!r} is empty")

    cities = sorted(set(int(x) for x in routes.route_city.astype(np.int64).tolist()))
    picks: Dict[int, np.ndarray] = {}
    for city in cities:
        keep = (
            (routes.route_city.astype(np.int64) == int(city))
            & (routes.way_seq_len >= (int(cfg.min_hops) + 1))
            & (routes.way_seq_len <= int(cfg.max_way_len))
        )
        ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
        if split_ids is not None:
            ids = ids[np.isin(ids, split_ids, assume_unique=False)]
        rng = np.random.default_rng(int(cfg.seed) + 101 * int(city))
        rng.shuffle(ids)
        picks[int(city)] = ids[: min(int(cfg.n_routes), int(ids.size))]
    return picks


@torch.no_grad()
def _teacher_forcing_reconstruct_k(
    *,
    ae: WayCASDAutoEncoder,
    gt_way_ids: List[int],
    route_cond_1: Dict[str, torch.Tensor],
    city: int,
    max_decode_len: int,
    n_samples: int,
    latent_noise_std: float,
    decode_stochastic: bool,
    temperature: float,
    decode_max_candidates: int,
    decode_candidate_policy: str,
    decode_include_dest_if_successor: bool,
    device: torch.device,
) -> List[List[int]]:
    if not gt_way_ids:
        return [[] for _ in range(int(n_samples))]

    L = int(len(gt_way_ids))
    K = int(max(1, n_samples))
    start_way = int(gt_way_ids[0])
    dest_way = int(gt_way_ids[-1])

    way_pad = torch.as_tensor(np.asarray(gt_way_ids, dtype=np.int64)[None, :], dtype=torch.long, device=device)
    z_gt, _ = ae.encode(way_pad)  # (1,n_latent,d)

    if float(latent_noise_std) > 0.0:
        z = z_gt.repeat(K, 1, 1) + float(latent_noise_std) * torch.randn((K, z_gt.shape[1], z_gt.shape[2]), device=device)
    else:
        z = z_gt.repeat(K, 1, 1)

    # Repeat route_cond to K samples.
    route_cond_k = {
        "start_pos": route_cond_1["start_pos"].repeat(K, 1),
        "dest_pos": route_cond_1["dest_pos"].repeat(K, 1),
        "hour": route_cond_1["hour"].repeat(K),
        "dow": route_cond_1["dow"].repeat(K),
        "route_city": torch.as_tensor(np.full((K,), int(city), dtype=np.int64), dtype=torch.long, device=device),
    }
    cond_emb_k = ae.decoder.cond_enc(
        start_pos=route_cond_k["start_pos"],
        dest_pos=route_cond_k["dest_pos"],
        hour=route_cond_k["hour"],
        dow=route_cond_k["dow"],
        route_city=route_cond_k["route_city"],
    )

    paths: List[List[int]] = [[int(start_way)] for _ in range(K)]
    steps = min(int(max_decode_len), max(0, int(L) - 1))

    for step_idx in range(steps):
        cur = int(gt_way_ids[step_idx])  # teacher forcing: current token comes from GT prefix
        cand_full = ae.decoder.get_succ_candidates(int(cur))
        if int(cand_full.numel()) <= 0:
            break
        cand = ae.decoder._select_decode_candidates(  # pylint: disable=protected-access
            way_embedder=ae.way_enc,
            cand_full=cand_full.to(device=device),
            dest_pos=route_cond_1["dest_pos"],
            dest_way=int(dest_way),
            max_candidates=(None if int(decode_max_candidates) < 0 else int(decode_max_candidates)),
            candidate_policy=str(decode_candidate_policy),
            include_dest_if_successor=bool(decode_include_dest_if_successor),
        )
        C = int(cand.numel())
        if C <= 0:
            break

        cand_way = cand.view(1, C).repeat(K, 1)
        cand_mask = torch.ones((K, C), dtype=torch.bool, device=device)

        trans: Dict[str, torch.Tensor] = {
            "route_idx": torch.arange(K, dtype=torch.long, device=device),
            "cur_way": torch.full((K,), int(cur), dtype=torch.long, device=device),
            "cand_way": cand_way,
            "cand_mask": cand_mask,
            "step": torch.full((K,), int(step_idx), dtype=torch.long, device=device),
        }
        if bool(ae.decoder.use_past_context):
            Kpast = int(ae.decoder.past_k)
            past_seq = gt_way_ids[:step_idx] if step_idx > 0 else []
            past_len = min(int(len(past_seq)), Kpast)
            past_row = np.full((Kpast,), -1, dtype=np.int64)
            if past_len > 0:
                off = Kpast - past_len
                past_row[off:] = np.asarray(past_seq[-past_len:], dtype=np.int64)
            past_way = torch.as_tensor(past_row[None, :].repeat(K, axis=0), dtype=torch.long, device=device)
            trans["past_way"] = past_way
            trans["past_mask"] = (past_way >= 0)

        logits = ae.decoder.score_candidates(
            way_embedder=ae.way_enc,
            latent_tokens=z,
            route_cond=route_cond_k,
            trans=trans,
            cond_emb=cond_emb_k,
        )
        if bool(decode_stochastic):
            temp = max(1e-6, float(temperature))
            probs = torch.softmax(logits / temp, dim=-1)
            pick = torch.multinomial(probs, num_samples=1).reshape(-1)
        else:
            pick = torch.argmax(logits, dim=-1)
        nxt = cand_way[torch.arange(K, device=device), pick].detach().cpu().numpy().astype(np.int64, copy=False).tolist()
        for i in range(K):
            paths[i].append(int(nxt[i]))
    return paths


def _analyze_per_od(
    *,
    gt_by_od: Dict[Tuple[int, int], List[List[int]]],
    pred_success_by_od: Dict[Tuple[int, int], List[List[int]]],
    min_routes_per_od: int,
    jaccard_threshold: float,
    k_per_od: int,
) -> Dict[str, Any]:
    kept_keys = [k for k, v in gt_by_od.items() if int(len(v)) >= int(min_routes_per_od)]
    per_od: List[Dict[str, Any]] = []
    coverage_vals: List[float] = []
    diversity_vals: List[float] = []
    n_no_success = 0
    n_div_valid = 0

    for od in kept_keys:
        gt_list = gt_by_od[od]
        pred_list_full = pred_success_by_od.get(od, [])
        if int(k_per_od) > 0:
            pred_list = pred_list_full[: int(k_per_od)]
        else:
            pred_list = pred_list_full
        if len(pred_list) == 0:
            n_no_success += 1

        matched = 0
        for gt in gt_list:
            ok = any(_seq_jaccard(gt, pr) >= float(jaccard_threshold) for pr in pred_list)
            matched += 1 if ok else 0
        cov = float(matched / max(1, len(gt_list)))
        coverage_vals.append(cov)

        div = _pairwise_diversity(pred_list)
        if math.isfinite(div):
            diversity_vals.append(div)
            n_div_valid += 1

        per_od.append(
            {
                "start_way": int(od[0]),
                "dest_way": int(od[1]),
                "n_gt_routes": int(len(gt_list)),
                "n_pred_success_used": int(len(pred_list)),
                "gt_coverage_at_k": float(cov),
                "self_diversity_at_k": (float(div) if math.isfinite(div) else None),
            }
        )

    cov_arr = np.asarray(coverage_vals, dtype=np.float64)
    div_arr = np.asarray(diversity_vals, dtype=np.float64)
    return {
        "n_od_groups_all": int(len(gt_by_od)),
        "n_od_groups_kept": int(len(kept_keys)),
        "n_od_groups_no_success": int(n_no_success),
        "gt_coverage_at_k": {
            "mean": float(np.mean(cov_arr)) if cov_arr.size else float("nan"),
            "p25": _pct(coverage_vals, 25),
            "p50": _pct(coverage_vals, 50),
            "p75": _pct(coverage_vals, 75),
            "n": int(cov_arr.size),
        },
        "self_diversity_at_k": {
            "mean": float(np.mean(div_arr)) if div_arr.size else float("nan"),
            "p25": _pct(diversity_vals, 25),
            "p50": _pct(diversity_vals, 50),
            "p75": _pct(diversity_vals, 75),
            "n": int(n_div_valid),
        },
        "per_od": per_od,
    }


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(
        description="Diagnostic: teacher-forcing reconstruction coverage from AE z_gt (corridor information probe)."
    )
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_per_route_json", type=Path, default=None)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--n_routes", type=int, default=5000, help="Per city.")
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--max_decode_len", type=int, default=160)

    p.add_argument("--n_samples_per_route", type=int, default=16)
    p.add_argument("--latent_noise_std", type=float, default=0.0, help="Gaussian std added to z_gt for each reconstruction sample.")
    p.add_argument("--decode_stochastic", action="store_true", help="Sample next-way from softmax logits (instead of argmax).")
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument("--decode_max_candidates", type=int, default=0, help="-1 use ckpt cfg; 0 all successors; >0 truncate.")
    p.add_argument("--decode_candidate_policy", choices=["first", "destdist"], default="first")
    p.add_argument("--decode_include_dest_if_successor", action="store_true")

    p.add_argument("--split_json", type=Path, default=None)
    p.add_argument("--split_part", choices=["train", "val", "test"], default=None)

    p.add_argument("--jaccard_threshold", type=float, default=0.5)
    p.add_argument("--min_routes_per_od", type=int, default=3)
    p.add_argument("--k_per_od", type=int, default=0, help="Cap successful predictions per OD when computing coverage/diversity (0=all).")
    p.add_argument("--dump_samples", action="store_true", help="Include all per-sample pred_way_ids in per-route output.")
    p.add_argument("--progress_every", type=int, default=100)
    args = p.parse_args()

    cfg = Cfg(
        seed=int(args.seed),
        device=str(args.device),
        tz_offset_hours=float(args.tz_offset_hours),
        n_routes=int(args.n_routes),
        min_hops=int(args.min_hops),
        max_way_len=int(args.max_way_len),
        max_decode_len=int(args.max_decode_len),
        n_samples_per_route=max(1, int(args.n_samples_per_route)),
        latent_noise_std=float(args.latent_noise_std),
        decode_stochastic=bool(args.decode_stochastic),
        temperature=float(args.temperature),
        decode_max_candidates=int(args.decode_max_candidates),
        decode_candidate_policy=str(args.decode_candidate_policy),
        decode_include_dest_if_successor=bool(args.decode_include_dest_if_successor),
        split_json=(str(args.split_json) if args.split_json is not None else None),
        split_part=(str(args.split_part) if args.split_part is not None else ("test" if args.split_json is not None else None)),
        jaccard_threshold=float(args.jaccard_threshold),
        min_routes_per_od=int(args.min_routes_per_od),
        k_per_od=int(args.k_per_od),
        dump_samples=bool(args.dump_samples),
        progress_every=max(1, int(args.progress_every)),
    )

    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    _set_seed(cfg.seed)

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    ae, strict_ok = _build_ae(
        ae_ckpt=Path(args.ae_ckpt),
        way_graph_npz=Path(args.way_graph_npz),
        way_features_npz=Path(args.way_features_npz),
        device=device,
    )

    picks = _pick_route_ids(routes, cfg=cfg)

    per_route: List[Dict[str, Any]] = []
    per_route_samples: List[List[List[int]]] = []
    gt_by_od: Dict[Tuple[int, int], List[List[int]]] = {}
    pred_success_by_od: Dict[Tuple[int, int], List[List[int]]] = {}
    n_total_samples = 0
    n_success_samples = 0
    n_route_any_success = 0

    for city, pick in picks.items():
        n_city = int(pick.size)
        if n_city <= 0:
            continue
        print(f"[city{int(city)}] start n_routes={n_city} K={int(cfg.n_samples_per_route)}", flush=True)
        for ii, rid in enumerate(pick.tolist(), start=1):
            rid_i = int(rid)
            L = int(routes.way_seq_len[rid_i])
            s = int(routes.way_seq_ptr[rid_i])
            gt = routes.way_seq_idx[s : s + L].astype(np.int64, copy=False).tolist()
            gt_ids = [int(x) for x in gt]
            if len(gt_ids) <= 1:
                continue

            sw = int(routes.start_way[rid_i])
            dw = int(routes.dest_way[rid_i])
            st = int(routes.start_t[rid_i])
            start_pos = routes.start_pos[rid_i].astype(np.float32, copy=False).reshape(2)
            dest_pos = routes.dest_pos[rid_i].astype(np.float32, copy=False).reshape(2)
            hr = int(_hour_from_unix(np.asarray([st], dtype=np.int64), tz_offset_hours=float(cfg.tz_offset_hours))[0])
            dow = int(_dow_from_unix(np.asarray([st], dtype=np.int64), tz_offset_hours=float(cfg.tz_offset_hours))[0])
            route_cond_1 = {
                "start_pos": torch.as_tensor(start_pos[None, :], dtype=torch.float32, device=device),
                "dest_pos": torch.as_tensor(dest_pos[None, :], dtype=torch.float32, device=device),
                "hour": torch.as_tensor(np.asarray([hr], dtype=np.int64), dtype=torch.long, device=device),
                "dow": torch.as_tensor(np.asarray([dow], dtype=np.int64), dtype=torch.long, device=device),
                "route_city": torch.as_tensor(np.asarray([int(city)], dtype=np.int64), dtype=torch.long, device=device),
            }

            pred_samples = _teacher_forcing_reconstruct_k(
                ae=ae,
                gt_way_ids=gt_ids,
                route_cond_1=route_cond_1,
                city=int(city),
                max_decode_len=int(cfg.max_decode_len),
                n_samples=int(cfg.n_samples_per_route),
                latent_noise_std=float(cfg.latent_noise_std),
                decode_stochastic=bool(cfg.decode_stochastic),
                temperature=float(cfg.temperature),
                decode_max_candidates=int(cfg.decode_max_candidates),
                decode_candidate_policy=str(cfg.decode_candidate_policy),
                decode_include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
                device=device,
            )
            sample_success = [bool(ps and int(ps[-1]) == int(dw)) for ps in pred_samples]
            succ_rate = float(np.mean(np.asarray(sample_success, dtype=np.float64))) if pred_samples else 0.0
            any_succ = bool(any(sample_success))
            n_total_samples += int(len(pred_samples))
            n_success_samples += int(sum(1 for x in sample_success if x))
            if any_succ:
                n_route_any_success += 1

            od = (int(sw), int(dw))
            gt_by_od.setdefault(od, []).append(gt_ids)
            succ_preds = [pred_samples[i] for i, ok in enumerate(sample_success) if bool(ok)]
            if succ_preds:
                pred_success_by_od.setdefault(od, []).extend(succ_preds)

            rec: Dict[str, Any] = {
                "route_id": int(rid_i),
                "city": int(city),
                "start_way": int(sw),
                "dest_way": int(dw),
                "gt_hops": int(len(gt_ids) - 1),
                "gt_way_ids": gt_ids,
                "teacher_forcing": {
                    "n_samples": int(len(pred_samples)),
                    "sample_success_rate": float(succ_rate),
                    "route_any_success": bool(any_succ),
                },
            }
            if bool(cfg.dump_samples):
                rec["teacher_forcing"]["sample_pred_way_ids"] = [[int(x) for x in seq] for seq in pred_samples]
            per_route.append(rec)
            if args.out_per_route_json is not None:
                per_route_samples.append([[int(x) for x in seq] for seq in pred_samples])

            if int(cfg.progress_every) > 0 and (ii % int(cfg.progress_every) == 0 or ii == n_city):
                print(
                    f"[city{int(city)}] done={ii}/{n_city} "
                    f"sample_sr={(float(n_success_samples)/max(1,n_total_samples)):.4f}",
                    flush=True,
                )

    od_stats = _analyze_per_od(
        gt_by_od=gt_by_od,
        pred_success_by_od=pred_success_by_od,
        min_routes_per_od=int(cfg.min_routes_per_od),
        jaccard_threshold=float(cfg.jaccard_threshold),
        k_per_od=int(cfg.k_per_od),
    )

    out = {
        "ok": True,
        "task": "way_casd_teacher_forcing_coverage",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": {
            "seed": int(cfg.seed),
            "device": str(cfg.device),
            "tz_offset_hours": float(cfg.tz_offset_hours),
            "n_routes": int(cfg.n_routes),
            "min_hops": int(cfg.min_hops),
            "max_way_len": int(cfg.max_way_len),
            "max_decode_len": int(cfg.max_decode_len),
            "n_samples_per_route": int(cfg.n_samples_per_route),
            "latent_noise_std": float(cfg.latent_noise_std),
            "decode_stochastic": bool(cfg.decode_stochastic),
            "temperature": float(cfg.temperature),
            "decode_max_candidates": int(cfg.decode_max_candidates),
            "decode_candidate_policy": str(cfg.decode_candidate_policy),
            "decode_include_dest_if_successor": bool(cfg.decode_include_dest_if_successor),
            "split_json": (str(args.split_json) if args.split_json is not None else None),
            "split_part": str(cfg.split_part) if cfg.split_part is not None else None,
            "jaccard_threshold": float(cfg.jaccard_threshold),
            "min_routes_per_od": int(cfg.min_routes_per_od),
            "k_per_od": int(cfg.k_per_od),
            "dump_samples": bool(cfg.dump_samples),
        },
        "inputs": {
            "way_routes_npz": str(Path(args.way_routes_npz)),
            "way_graph_npz": str(Path(args.way_graph_npz)),
            "way_features_npz": str(Path(args.way_features_npz)),
            "ae_ckpt": str(Path(args.ae_ckpt)),
        },
        "ckpt_strict_load_ok": bool(strict_ok),
        "summary": {
            "n_routes_eval": int(len(per_route)),
            "n_samples_total": int(n_total_samples),
            "n_samples_success": int(n_success_samples),
            "sample_arrival_rate": float(n_success_samples / max(1, n_total_samples)),
            "route_any_success_rate": float(n_route_any_success / max(1, len(per_route))),
            **od_stats,
        },
    }

    if args.out_per_route_json is not None:
        # Flatten to od_coverage_diversity compatible schema (one sample = one route record).
        flat_per_route: List[Dict[str, Any]] = []
        fid = 0
        for rec, samples in zip(per_route, per_route_samples):
            gt_ids = [int(x) for x in rec["gt_way_ids"]]
            for seq in samples:
                pred_ids = [int(x) for x in seq]
                ok = bool(pred_ids and int(pred_ids[-1]) == int(rec["dest_way"]))
                flat_per_route.append(
                    {
                        "route_id": int(fid),
                        "city": int(rec["city"]),
                        "start_way": int(rec["start_way"]),
                        "dest_way": int(rec["dest_way"]),
                        "gt_way_ids": gt_ids,
                        "greedy": {
                            "success": bool(ok),
                            "pred_way_ids": pred_ids,
                        },
                    }
                )
                fid += 1
        out2 = {
            "ok": True,
            "task": "way_casd_teacher_forcing_coverage_per_route_flat",
            "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
            "cfg": out["cfg"],
            "per_route": flat_per_route,
        }
        op2 = Path(args.out_per_route_json)
        op2.parent.mkdir(parents=True, exist_ok=True)
        op2.write_text(json.dumps(out2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] saved flat per_route: {op2}")

    op = Path(args.out_json)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] saved: {op}")
    s = out["summary"]
    print(
        "Teacher-Forcing Probe | "
        f"sample_arrival={float(s['sample_arrival_rate']):.4f} | "
        f"route_any_success={float(s['route_any_success_rate']):.4f} | "
        f"coverage_mean={float(s['gt_coverage_at_k']['mean']):.4f} | "
        f"div_mean={float(s['self_diversity_at_k']['mean']):.4f} | "
        f"n_od={int(s['n_od_groups_kept'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
