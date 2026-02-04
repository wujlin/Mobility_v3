from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.way_graph.way_sequence_dataset import WayRouteDataset, load_way_routes_npz, make_way_casd_collate_fn
from src.models.way_casd.latent_flow import LatentFlowCfg, LatentFlowMatching
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz

TZ_SHANGHAI = timezone(timedelta(hours=8))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainCfg:
    batch_size: int
    num_workers: int
    n_epochs: int
    lr: float
    weight_decay: float
    val_ratio: float
    seed: int
    device: str
    tz_offset_hours: float
    min_hops: int
    max_way_len: int
    max_routes: Optional[int]

    latent_source: str  # {"gt","flow"}
    flow_solver_steps: Optional[int]
    max_decode_len: int
    decode_max_candidates: int
    decode_candidate_policy: str
    decode_include_dest_if_successor: bool
    guided_dest_alpha: float
    temperature: float

    # Region constraint (optional)
    region_constraint: str  # {"none","gt"}
    region_constraint_mode: str  # {"strict","relaxed"}
    region_constraint_fallback: str  # {"unconstrained","stop","dest_region"}

    # Anti-loop during sampling (optional)
    anti_loop_k: int
    anti_loop_penalty: float
    anti_loop_penalty_k: int

    # RL objective
    reward_success: float
    reward_dist: float
    penalty_len: float
    penalty_loop: float
    entropy_coef: float
    baseline: str  # {"mean","ema"}
    baseline_ema_beta: float
    ce_weight: float
    max_grad_norm: float


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _split_dataset(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n))
    n_val = int(round(float(val_ratio) * float(n)))
    n_val = max(1, min(n_val, n - 1))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx.astype(np.int64, copy=False), val_idx.astype(np.int64, copy=False)


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


def _infer_decoder_past_k_from_state(state: Dict[str, torch.Tensor]) -> Optional[int]:
    w = state.get("decoder.past_encoder.pos_emb.weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[0])
    return None


def _infer_decoder_past_n_layers_from_state(state: Dict[str, torch.Tensor]) -> Optional[int]:
    max_i = -1
    prefix = "decoder.past_encoder.transformer.layers."
    for k in state.keys():
        ks = str(k)
        if not ks.startswith(prefix):
            continue
        rest = ks[len(prefix) :]
        i_str = rest.split(".", 1)[0]
        try:
            i = int(i_str)
        except Exception:
            continue
        max_i = max(max_i, i)
    if max_i >= 0:
        return int(max_i + 1)
    return None


def _load_ae(
    *,
    ae_ckpt: Path,
    way_graph_npz: Path,
    way_features_npz: Path,
    device: torch.device,
) -> WayCASDAutoEncoder:
    wg = np.load(str(way_graph_npz), allow_pickle=True)
    wf = np.load(str(way_features_npz), allow_pickle=True)
    way_features = load_way_features_from_npz(Path(way_features_npz))
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1

    ckpt = torch.load(str(ae_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cfg_dict: Dict[str, object] = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected AE state format: {type(state)}")

    use_dest_dist = _infer_decoder_use_dest_dist_from_state(state)
    use_cand_contrast = (_infer_decoder_use_cand_contrast_from_state(state) if isinstance(state, dict) else False) or bool(
        cfg_dict.get("decoder_use_cand_contrast", False)
    )
    use_cross_attn = _infer_decoder_use_cross_attn_from_state(state)
    use_step_emb = _infer_decoder_use_step_emb_from_state(state) or bool(cfg_dict.get("decoder_use_step_emb", False))
    use_dest_query = _infer_decoder_use_dest_query_from_state(state) or bool(cfg_dict.get("decoder_use_dest_query", False))
    use_dir_query = _infer_decoder_use_dir_query_from_state(state) or bool(cfg_dict.get("decoder_use_dir_query", False))
    use_cand_query = _infer_decoder_use_cand_query_from_state(state) or bool(cfg_dict.get("decoder_use_cand_query", False))
    use_past_ctx = _infer_decoder_use_past_context_from_state(state) or bool(cfg_dict.get("decoder_use_past_context", False))
    past_k = cfg_dict.get("decoder_past_k", None)
    if past_k is None:
        past_k = _infer_decoder_past_k_from_state(state)
    past_n_layers = cfg_dict.get("decoder_past_n_layers", None)
    if past_n_layers is None:
        past_n_layers = _infer_decoder_past_n_layers_from_state(state)
    past_n_heads = cfg_dict.get("decoder_past_n_heads", None)

    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(cfg_dict.get("d_model", 256)),
            n_latent=int(cfg_dict.get("n_latent", 64)),
            n_heads=int(cfg_dict.get("n_heads", 8)),
            dropout=float(cfg_dict.get("dropout", 0.1)),
            max_candidates=int(cfg_dict.get("max_candidates", 32)),
            max_len=int(cfg_dict.get("max_len", 160)),
            coord_scale=float(cfg_dict.get("coord_scale", 1024.0)),
            segment_size=int(cfg_dict.get("segment_size", 10)),
            segment_n_latent=int(cfg_dict.get("segment_n_latent", 0)),
            decoder_use_dest_dist=bool(use_dest_dist),
            decoder_use_cross_attn=bool(use_cross_attn),
            decoder_use_step_emb=bool(use_step_emb),
            decoder_use_dest_query=bool(use_dest_query),
            decoder_use_dir_query=bool(use_dir_query),
            decoder_use_cand_query=bool(use_cand_query),
            decoder_use_cand_contrast=bool(use_cand_contrast),
            decoder_use_past_context=bool(use_past_ctx),
            decoder_past_k=int(past_k) if past_k is not None else 8,
            decoder_past_n_layers=int(past_n_layers) if past_n_layers is not None else 2,
            decoder_past_n_heads=int(past_n_heads) if past_n_heads is not None else 4,
        ),
        way_features=way_features,
        way_adj_ptr=wg["way_adj_ptr"],
        way_adj_idx=wg["way_adj_idx"],
        n_highway_types=int(max(4, n_highway_types)),
    ).to(device)

    try:
        ae.load_state_dict(state, strict=True)
    except RuntimeError as e:
        log.warning(f"AE strict load failed; falling back to strict=False. err={e}")
        missing, unexpected = ae.load_state_dict(state, strict=False)
        missing_critical = [k for k in missing if str(k).startswith(("way_enc.", "compress."))]
        if missing_critical:
            raise RuntimeError(f"AE load missing critical encoder keys (example={missing_critical[:3]})")
        if missing:
            log.warning(f"AE non-strict load: missing={len(missing)} (example={missing[:3]})")
        if unexpected:
            log.warning(f"AE non-strict load: unexpected={len(unexpected)} (example={unexpected[:3]})")

    ae.eval()
    return ae


def _load_flow(*, flow_ckpt: Path, ae: WayCASDAutoEncoder, device: torch.device) -> LatentFlowMatching:
    ckpt = torch.load(str(flow_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cfg_dict = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(state, dict):
        raise SystemExit("[FATAL] unexpected flow ckpt format (state_dict missing).")

    flow_cfg = LatentFlowCfg(
        d_model=int(cfg_dict.get("d_model", ae.cfg.d_model)),
        n_latent=int(cfg_dict.get("n_latent", ae.cfg.n_latent)),
        n_layers=int(cfg_dict.get("n_layers", 6)),
        n_heads=int(cfg_dict.get("n_heads", 8)),
        dropout=float(cfg_dict.get("dropout", 0.1)),
        noise_sigma=float(cfg_dict.get("noise_sigma", 1.0)),
        solver_steps=int(cfg_dict.get("solver_steps", 20)),
        cond_inject=str(cfg_dict.get("cond_inject", "add")),
        use_region_seq=bool(cfg_dict.get("use_region_seq", False)),
        n_regions=int(cfg_dict.get("n_regions", 154)),
        region_max_len=int(cfg_dict.get("region_max_len", 16)),
    )
    if int(flow_cfg.d_model) != int(ae.cfg.d_model) or int(flow_cfg.n_latent) != int(ae.cfg.n_latent):
        raise SystemExit(
            f"[FATAL] AE/Flow mismatch: AE(d_model={int(ae.cfg.d_model)}, n_latent={int(ae.cfg.n_latent)}) "
            f"vs Flow(d_model={int(flow_cfg.d_model)}, n_latent={int(flow_cfg.n_latent)})."
        )
    flow = LatentFlowMatching(cfg=flow_cfg, cond_cfg=ae.decoder.cond_enc.cfg).to(device)
    flow.load_state_dict(state, strict=False)
    flow.eval()
    return flow


def _load_region_adj(*, way_regions_npz: Path) -> np.ndarray:
    wr = np.load(str(way_regions_npz), allow_pickle=True)
    need = {"region_adj_ptr", "region_adj_idx"}
    missing = sorted(list(need - set(wr.files)))
    if missing:
        raise SystemExit(f"[FATAL] way_regions_npz missing keys: {missing}")
    ptr = np.asarray(wr["region_adj_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(wr["region_adj_idx"], dtype=np.int64).reshape(-1)
    R = int(ptr.size) - 1
    adj = np.zeros((R, R), dtype=bool)
    np.fill_diagonal(adj, True)
    for r in range(R):
        s = int(ptr[r])
        e = int(ptr[r + 1])
        if e <= s:
            continue
        nn = idx[s:e].astype(np.int64, copy=False)
        nn = nn[(nn >= 0) & (nn < R)]
        adj[r, nn] = True
    return adj


def _compress_consecutive(seq: List[int]) -> List[int]:
    out: List[int] = []
    last: Optional[int] = None
    for x in seq:
        xx = int(x)
        if last is None or xx != int(last):
            out.append(xx)
            last = xx
    return out


def _region_seq_from_way_seq(way_seq: List[int], way_region_np: np.ndarray) -> List[int]:
    rr: List[int] = []
    for w in way_seq:
        wi = int(w)
        if wi < 0 or wi >= int(way_region_np.size):
            continue
        rr.append(int(way_region_np[wi]))
    rr = [int(x) for x in rr if int(x) >= 0]
    return _compress_consecutive(rr)


def _has_loop(path: List[int]) -> bool:
    # Exclude the start node only? Keep it simple: any repeated way means loop.
    return len(set(path)) < len(path)


def _summarize_reward(x: torch.Tensor) -> Dict[str, float]:
    a = x.detach().float().cpu().numpy().reshape(-1)
    if a.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {"mean": float(np.mean(a)), "p50": float(np.quantile(a, 0.5)), "p95": float(np.quantile(a, 0.95))}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="E7: RL fine-tuning for Way-CASD decoder (REINFORCE, batched sampling).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True, help="Base AE ckpt to fine-tune (decoder only).")
    p.add_argument("--flow_ckpt", type=Path, default=None, help="Optional: when --latent_source=flow.")
    p.add_argument("--way_regions_npz", type=Path, default=None, help="Required when --region_constraint != none.")
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--n_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--max_routes", type=int, default=None, help="Debug: cap number of routes (after filtering).")

    p.add_argument("--latent_source", type=str, default="gt", choices=["gt", "flow"])
    p.add_argument("--flow_solver_steps", type=int, default=-1, help="-1=use ckpt cfg; >0=override.")

    p.add_argument("--max_decode_len", type=int, default=160)
    p.add_argument("--decode_max_candidates", type=int, default=0, help="0=all successors; >0=truncate; -1=use AE cfg.")
    p.add_argument("--decode_candidate_policy", type=str, default="first", choices=["first", "destdist"])
    p.add_argument("--decode_include_dest_if_successor", action="store_true")
    p.add_argument("--decode_guided_dest_alpha", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument("--region_constraint", type=str, default="none", choices=["none", "gt"])
    p.add_argument("--region_constraint_mode", type=str, default="relaxed", choices=["strict", "relaxed"])
    p.add_argument("--region_constraint_fallback", type=str, default="dest_region", choices=["unconstrained", "stop", "dest_region"])

    p.add_argument("--anti_loop_k", type=int, default=0)
    p.add_argument("--anti_loop_penalty", type=float, default=0.0)
    p.add_argument("--anti_loop_penalty_k", type=int, default=4)

    p.add_argument("--reward_success", type=float, default=1.0)
    p.add_argument("--reward_dist", type=float, default=1.0, help="Penalty weight for final dest distance in normalized coord.")
    p.add_argument("--penalty_len", type=float, default=0.002, help="Penalty per hop.")
    p.add_argument("--penalty_loop", type=float, default=0.0, help="Extra penalty if predicted path has a loop.")
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--baseline", type=str, default="mean", choices=["mean", "ema"])
    p.add_argument("--baseline_ema_beta", type=float, default=0.98)
    p.add_argument("--ce_weight", type=float, default=0.0, help="Optional: mix teacher-forcing CE loss (0=disable).")
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=1)
    return p


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    way_seq_pad = batch["way_seq_pad"].to(device)
    route_cond = {k: v.to(device) for k, v in batch["route_cond"].items()}
    trans = {k: v.to(device) for k, v in batch["trans"].items()}
    route_id = batch["route_id"].to(device)
    way_seq_len = batch["way_seq_len"].to(device)
    return {"way_seq_pad": way_seq_pad, "way_seq_len": way_seq_len, "route_cond": route_cond, "trans": trans, "route_id": route_id}


def _decode_and_reward(
    *,
    ae: WayCASDAutoEncoder,
    flow: Optional[LatentFlowMatching],
    batch: Dict[str, object],
    cfg: TrainCfg,
    way_region_t: Optional[torch.Tensor],
    way_region_np: Optional[np.ndarray],
    region_adj_t: Optional[torch.Tensor],
    device: torch.device,
    baseline_ema: float,
    train: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], float]:
    b = _to_device(batch, device)
    route_cond = b["route_cond"]
    start_way = route_cond["start_way"]
    dest_way = route_cond["dest_way"]
    B = int(start_way.shape[0])

    # Build region_seq from GT (if needed by constraint or by Flow conditioning).
    region_seq: Optional[List[List[int]]] = None
    if str(cfg.region_constraint) != "none":
        if way_region_np is None:
            raise RuntimeError("region_constraint requires way_region_np (need --way_regions_npz).")
        way_seq_pad = b["way_seq_pad"].detach().cpu().numpy().astype(np.int64, copy=False)
        way_seq_len = b["way_seq_len"].detach().cpu().numpy().astype(np.int64, copy=False)
        region_seq = []
        for i in range(int(B)):
            L = int(way_seq_len[i])
            ws = [int(x) for x in way_seq_pad[i, :L].tolist() if int(x) >= 0]
            region_seq.append(_region_seq_from_way_seq(ws, way_region_np))

    # Latent source.
    with torch.no_grad():
        if str(cfg.latent_source) == "gt":
            z, _ = ae.encode(b["way_seq_pad"])
        else:
            if flow is None:
                raise RuntimeError("latent_source=flow requires --flow_ckpt.")
            steps = cfg.flow_solver_steps
            steps_use = int(steps) if steps is not None else None
            if bool(flow.cfg.use_region_seq):
                if region_seq is None:
                    if way_region_np is None:
                        raise RuntimeError("Flow requires region_seq conditioning, but way_region_np is missing.")
                    way_seq_pad = b["way_seq_pad"].detach().cpu().numpy().astype(np.int64, copy=False)
                    way_seq_len = b["way_seq_len"].detach().cpu().numpy().astype(np.int64, copy=False)
                    region_seq = []
                    for i in range(int(B)):
                        L = int(way_seq_len[i])
                        ws = [int(x) for x in way_seq_pad[i, :L].tolist() if int(x) >= 0]
                        region_seq.append(_region_seq_from_way_seq(ws, way_region_np))
                maxS = max(1, max(len(x) for x in region_seq))
                pad = torch.full((B, maxS), -1, dtype=torch.long, device=device)
                for i, rs in enumerate(region_seq):
                    if rs:
                        pad[i, : len(rs)] = torch.as_tensor(rs, dtype=torch.long, device=device)
                route_cond = dict(route_cond)
                route_cond["region_seq_pad"] = pad
            z = flow.sample(route_cond=route_cond, solver_steps=steps_use)

    # Sampling decode (policy).
    paths, logp_sum, entropy_sum = ae.decoder.sample_decode_batched(
        way_embedder=ae.way_enc,
        latent_tokens=z,
        route_cond=route_cond,
        start_way=start_way,
        dest_way=dest_way,
        way_region=way_region_t,
        region_seq=region_seq,
        region_adj=region_adj_t,
        region_constraint_mode=str(cfg.region_constraint_mode),
        region_constraint_fallback=str(cfg.region_constraint_fallback),
        max_len=int(cfg.max_decode_len),
        max_candidates=(None if int(cfg.decode_max_candidates) < 0 else int(cfg.decode_max_candidates)),
        candidate_policy=str(cfg.decode_candidate_policy),
        include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
        guided_dest_alpha=float(cfg.guided_dest_alpha),
        temperature=float(cfg.temperature),
        anti_loop_k=int(cfg.anti_loop_k),
        anti_loop_penalty=float(cfg.anti_loop_penalty),
        anti_loop_penalty_k=int(cfg.anti_loop_penalty_k),
    )

    # Compute rewards (torch on device for convenience; reward itself is treated as constant in REINFORCE).
    last_way = torch.as_tensor([int(p[-1]) if p else int(start_way[i].item()) for i, p in enumerate(paths)], device=device, dtype=torch.long)
    success = (last_way == dest_way.to(dtype=torch.long)).to(dtype=torch.float32)

    # Final distance to destination way center (normalized coord).
    with torch.no_grad():
        coord_scale = float(getattr(ae.way_enc, "coord_scale", ae.cfg.coord_scale))
        pred_geom, _tier, _hw = ae.way_enc._lookup(last_way)
        dest_geom, _tier2, _hw2 = ae.way_enc._lookup(dest_way.to(dtype=torch.long))
        pred_center = pred_geom[..., :2].to(dtype=torch.float32) / coord_scale
        dest_center = dest_geom[..., :2].to(dtype=torch.float32) / coord_scale
        dist = torch.norm(pred_center - dest_center, dim=-1)  # (B,)

    hops = torch.as_tensor([max(0, len(p) - 1) for p in paths], device=device, dtype=torch.float32)
    has_loop = torch.as_tensor([1.0 if _has_loop(p) else 0.0 for p in paths], device=device, dtype=torch.float32)

    reward = float(cfg.reward_success) * success - float(cfg.reward_dist) * dist - float(cfg.penalty_len) * hops - float(cfg.penalty_loop) * has_loop

    # Baseline for variance reduction.
    if str(cfg.baseline) == "ema":
        mean_r = float(reward.detach().mean().item()) if reward.numel() else 0.0
        beta = float(cfg.baseline_ema_beta)
        baseline_ema = beta * float(baseline_ema) + (1.0 - beta) * mean_r
        baseline = torch.full_like(reward, float(baseline_ema))
    else:
        baseline = reward.detach().mean() if reward.numel() else torch.tensor(0.0, device=device)
        baseline = baseline.expand_as(reward)

    adv = (reward - baseline).detach()

    # Optional: CE loss (teacher forcing) to stabilize.
    ce_loss = torch.tensor(0.0, device=device)
    if float(cfg.ce_weight) > 0.0:
        tgt = b["trans"]["target_idx"].to(dtype=torch.long)
        if int(tgt.numel()) > 0:
            logits = ae.decoder.score_candidates(
                way_embedder=ae.way_enc,
                latent_tokens=z.detach(),
                route_cond=route_cond,
                trans=b["trans"],
            )
            ce_loss = F.cross_entropy(logits, tgt, reduction="mean")

    loss_rl = -(adv * logp_sum).mean() - float(cfg.entropy_coef) * entropy_sum.mean()
    loss = loss_rl + float(cfg.ce_weight) * ce_loss

    stats = {
        "loss": float(loss.detach().item()),
        "loss_rl": float(loss_rl.detach().item()),
        "loss_ce": float(ce_loss.detach().item()) if float(cfg.ce_weight) > 0.0 else 0.0,
        "reward_mean": float(reward.detach().mean().item()) if reward.numel() else float("nan"),
        "success_rate": float(success.detach().mean().item()) if success.numel() else float("nan"),
        "dist_mean": float(dist.detach().mean().item()) if dist.numel() else float("nan"),
        "hops_mean": float(hops.detach().mean().item()) if hops.numel() else float("nan"),
        "loop_rate": float(has_loop.detach().mean().item()) if has_loop.numel() else float("nan"),
        "reward": _summarize_reward(reward),
    }
    return loss, reward.detach(), success.detach(), entropy_sum.detach(), stats, float(baseline_ema)


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainCfg(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        n_epochs=int(args.n_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        device=str(args.device),
        tz_offset_hours=float(args.tz_offset_hours),
        min_hops=int(args.min_hops),
        max_way_len=int(args.max_way_len),
        max_routes=(int(args.max_routes) if args.max_routes is not None else None),
        latent_source=str(args.latent_source),
        flow_solver_steps=(None if int(args.flow_solver_steps) <= 0 else int(args.flow_solver_steps)),
        max_decode_len=int(args.max_decode_len),
        decode_max_candidates=int(args.decode_max_candidates),
        decode_candidate_policy=str(args.decode_candidate_policy),
        decode_include_dest_if_successor=bool(args.decode_include_dest_if_successor),
        guided_dest_alpha=float(args.decode_guided_dest_alpha),
        temperature=float(args.temperature),
        region_constraint=str(args.region_constraint),
        region_constraint_mode=str(args.region_constraint_mode),
        region_constraint_fallback=str(args.region_constraint_fallback),
        anti_loop_k=max(0, int(args.anti_loop_k)),
        anti_loop_penalty=max(0.0, float(args.anti_loop_penalty)),
        anti_loop_penalty_k=max(0, int(args.anti_loop_penalty_k)),
        reward_success=float(args.reward_success),
        reward_dist=float(args.reward_dist),
        penalty_len=float(args.penalty_len),
        penalty_loop=float(args.penalty_loop),
        entropy_coef=float(args.entropy_coef),
        baseline=str(args.baseline),
        baseline_ema_beta=float(args.baseline_ema_beta),
        ce_weight=float(args.ce_weight),
        max_grad_norm=float(args.max_grad_norm),
    )

    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    log.info(f"device={device}")

    # Dataset
    routes = load_way_routes_npz(Path(args.way_routes_npz))
    dataset = WayRouteDataset(routes, max_routes=cfg.max_routes, max_way_len=int(cfg.max_way_len), min_hops=int(cfg.min_hops))
    tr_idx, va_idx = _split_dataset(len(dataset), cfg.val_ratio, cfg.seed)
    train_set = Subset(dataset, tr_idx.tolist())
    val_set = Subset(dataset, va_idx.tolist())
    log.info(f"routes: total={len(dataset)} train={len(train_set)} val={len(val_set)} min_hops={cfg.min_hops} max_way_len={cfg.max_way_len}")

    # Load AE
    ae = _load_ae(
        ae_ckpt=Path(args.ae_ckpt),
        way_graph_npz=Path(args.way_graph_npz),
        way_features_npz=Path(args.way_features_npz),
        device=device,
    )

    # Freeze everything except decoder.
    for p in ae.parameters():
        p.requires_grad_(False)
    for name, p in ae.named_parameters():
        if str(name).startswith("decoder."):
            p.requires_grad_(True)

    n_trainable = sum(int(p.numel()) for p in ae.parameters() if p.requires_grad)
    log.info(f"trainable_params(decoder)={n_trainable}")

    # Optional: Flow
    flow: Optional[LatentFlowMatching] = None
    if str(cfg.latent_source) == "flow":
        if args.flow_ckpt is None:
            raise SystemExit("[FATAL] --flow_ckpt is required when --latent_source=flow")
        flow = _load_flow(flow_ckpt=Path(args.flow_ckpt), ae=ae, device=device)
        for p in flow.parameters():
            p.requires_grad_(False)
        if cfg.flow_solver_steps is None:
            cfg = TrainCfg(**{**asdict(cfg), "flow_solver_steps": int(flow.cfg.solver_steps)})  # type: ignore[arg-type]
        log.info(f"loaded flow: cond_inject={flow.cfg.cond_inject} use_region_seq={flow.cfg.use_region_seq} solver_steps={flow.cfg.solver_steps}")

    # Region constraint inputs
    way_region_np: Optional[np.ndarray] = None
    way_region_t: Optional[torch.Tensor] = None
    region_adj_t: Optional[torch.Tensor] = None
    if str(cfg.region_constraint) != "none":
        if args.way_regions_npz is None:
            raise SystemExit("[FATAL] --way_regions_npz is required when --region_constraint != none")
        wr = np.load(str(Path(args.way_regions_npz)), allow_pickle=True)
        if "way_region" not in wr.files:
            raise SystemExit("[FATAL] way_regions_npz missing key: way_region")
        way_region_np = np.asarray(wr["way_region"], dtype=np.int64).reshape(-1)
        way_region_t = torch.as_tensor(way_region_np, dtype=torch.long, device=device)
        if str(cfg.region_constraint_mode) == "relaxed":
            adj = _load_region_adj(way_regions_npz=Path(args.way_regions_npz))
            region_adj_t = torch.as_tensor(adj, dtype=torch.bool, device=device)

    # Collate for CE (optional). Keep it consistent with AE decoder_past_k.
    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    collate_fn = make_way_casd_collate_fn(
        way_adj_ptr=wg["way_adj_ptr"],
        way_adj_idx=wg["way_adj_idx"],
        max_candidates=int(ae.cfg.max_candidates),
        tz_offset_hours=float(cfg.tz_offset_hours),
        past_k=int(ae.cfg.decoder_past_k),
    )

    pin = bool(device.type == "cuda")
    num_workers = max(0, int(cfg.num_workers))
    prefetch_factor = 4 if num_workers > 0 else None
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
    )

    opt = torch.optim.AdamW([p for p in ae.parameters() if p.requires_grad], lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    best = -float("inf")
    best_epoch = 0
    best_path = out_dir / "ckpt_best.pt"
    last_path = out_dir / "ckpt_last.pt"
    progress_path = out_dir / "progress.json"
    hist_path = out_dir / "history.jsonl"

    baseline_ema = 0.0
    save_every = max(1, int(args.save_every))
    log_every = max(1, int(args.log_every))

    for epoch in range(1, int(cfg.n_epochs) + 1):
        ae.train()
        total_loss = 0.0
        total_reward = 0.0
        total_succ = 0.0
        total_batches = 0

        for step, batch in enumerate(train_loader, start=1):
            opt.zero_grad(set_to_none=True)
            loss, reward, success, entropy_sum, stats, baseline_ema = _decode_and_reward(
                ae=ae,
                flow=flow,
                batch=batch,
                cfg=cfg,
                way_region_t=way_region_t,
                way_region_np=way_region_np,
                region_adj_t=region_adj_t,
                device=device,
                baseline_ema=baseline_ema,
                train=True,
            )
            loss.backward()
            if float(cfg.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_([p for p in ae.parameters() if p.requires_grad], max_norm=float(cfg.max_grad_norm))
            opt.step()

            total_loss += float(stats["loss"])
            total_reward += float(stats["reward_mean"]) if np.isfinite(float(stats["reward_mean"])) else 0.0
            total_succ += float(stats["success_rate"]) if np.isfinite(float(stats["success_rate"])) else 0.0
            total_batches += 1

            if (step % log_every) == 0:
                log.info(
                    f"epoch={epoch} step={step} "
                    f"loss={stats['loss']:.4f} r={stats['reward_mean']:.4f} succ={stats['success_rate']:.3f} "
                    f"dist={stats['dist_mean']:.3f} loop={stats['loop_rate']:.3f} ent={float(entropy_sum.mean().item()):.3f}"
                )

        denom = max(1, int(total_batches))
        tr_report = {"loss": float(total_loss / denom), "reward_mean": float(total_reward / denom), "success_rate": float(total_succ / denom)}

        # Val (sampling, no grad)
        ae.eval()
        with torch.no_grad():
            v_loss = 0.0
            v_reward = 0.0
            v_succ = 0.0
            v_batches = 0
            for batch in val_loader:
                loss, reward, success, _ent, stats, _ = _decode_and_reward(
                    ae=ae,
                    flow=flow,
                    batch=batch,
                    cfg=cfg,
                    way_region_t=way_region_t,
                    way_region_np=way_region_np,
                    region_adj_t=region_adj_t,
                    device=device,
                    baseline_ema=baseline_ema,
                    train=False,
                )
                v_loss += float(stats["loss"])
                v_reward += float(stats["reward_mean"]) if np.isfinite(float(stats["reward_mean"])) else 0.0
                v_succ += float(stats["success_rate"]) if np.isfinite(float(stats["success_rate"])) else 0.0
                v_batches += 1
            denom_v = max(1, int(v_batches))
            va_report = {"loss": float(v_loss / denom_v), "reward_mean": float(v_reward / denom_v), "success_rate": float(v_succ / denom_v)}

        log.info(
            f"epoch={epoch} train(loss={tr_report['loss']:.4f}, r={tr_report['reward_mean']:.4f}, succ={tr_report['success_rate']:.3f}) "
            f"val(loss={va_report['loss']:.4f}, r={va_report['reward_mean']:.4f}, succ={va_report['success_rate']:.3f})"
        )

        with hist_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": int(epoch),
                        "train": tr_report,
                        "val": va_report,
                        "baseline_ema": float(baseline_ema),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        score = float(va_report["reward_mean"])
        if float(score) > float(best):
            best = float(score)
            best_epoch = int(epoch)
            torch.save(
                {
                    "model_state_dict": ae.state_dict(),
                    "config": asdict(cfg),
                    "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
                    "epoch": int(epoch),
                    "best_score": float(best),
                    "best_epoch": int(best_epoch),
                    "baseline_ema": float(baseline_ema),
                },
                str(best_path),
            )

        if (int(epoch) % save_every) == 0:
            torch.save(
                {
                    "model_state_dict": ae.state_dict(),
                    "opt_state_dict": opt.state_dict(),
                    "config": asdict(cfg),
                    "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
                    "epoch": int(epoch),
                    "best_score": float(best),
                    "best_epoch": int(best_epoch),
                    "baseline_ema": float(baseline_ema),
                },
                str(last_path),
            )

        progress = {
            "ok": True,
            "task": "train_way_casd_decoder_rl",
            "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
            "epoch": int(epoch),
            "train": tr_report,
            "val": va_report,
            "best_score": float(best),
            "best_epoch": int(best_epoch),
            "baseline_ema": float(baseline_ema),
        }
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

    torch.save(
        {
            "model_state_dict": ae.state_dict(),
            "opt_state_dict": opt.state_dict(),
            "config": asdict(cfg),
            "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
            "epoch": int(cfg.n_epochs),
            "best_score": float(best),
            "best_epoch": int(best_epoch),
            "baseline_ema": float(baseline_ema),
        },
        str(last_path),
    )

    report = {
        "ok": True,
        "task": "train_way_casd_decoder_rl",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
            "flow_ckpt": (str(args.flow_ckpt) if args.flow_ckpt is not None else None),
            "way_regions_npz": (str(args.way_regions_npz) if args.way_regions_npz is not None else None),
        },
        "out_dir": str(out_dir),
        "best_score": float(best),
        "best_ckpt": str(best_path),
        "last_ckpt": str(last_path),
        "best_epoch": int(best_epoch),
        "cfg": asdict(cfg),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"saved: {out_dir/'report.json'}")


if __name__ == "__main__":
    main()
