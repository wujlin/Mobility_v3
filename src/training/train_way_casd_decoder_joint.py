"""
E2: Flow-Decoder Joint Fine-tuning (Teacher Forcing)

Goal:
  Fine-tune ONLY the decoder parameters so that scoring is compatible with Flow-sampled latents.

Key idea:
  Use z_flow = Flow.sample(route_cond) as latent tokens, but keep teacher-forcing CE loss on GT transitions.

Outputs (out_dir):
  - ckpt_best.pt / ckpt_last.pt
  - progress.json (last epoch summary)
  - report.json (artifact index)
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.way_graph.way_sequence_dataset import WayRouteDataset, load_way_routes_npz, make_way_casd_collate_fn
from src.evaluation.shortest_path_baseline import dijkstra_way_path
from src.models.way_casd.latent_flow import LatentFlowCfg, LatentFlowMatching
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz

TZ_SHANGHAI = timezone(timedelta(hours=8))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# Shortest-path oracle worker globals (for multiprocessing pool).
_SP_PTR: Optional[np.ndarray] = None
_SP_IDX: Optional[np.ndarray] = None
_SP_WAY_LEN_M: Optional[np.ndarray] = None
_SP_MAX_VISITS: int = 5000


def _init_sp_worker(ptr: np.ndarray, idx: np.ndarray, way_len_m: np.ndarray, max_visits: int) -> None:
    global _SP_PTR, _SP_IDX, _SP_WAY_LEN_M, _SP_MAX_VISITS
    _SP_PTR = ptr
    _SP_IDX = idx
    _SP_WAY_LEN_M = way_len_m
    _SP_MAX_VISITS = int(max_visits)


def _sp_next_for_pair(pair: Tuple[int, int]) -> Tuple[int, int, int]:
    global _SP_PTR, _SP_IDX, _SP_WAY_LEN_M, _SP_MAX_VISITS
    if _SP_PTR is None or _SP_IDX is None or _SP_WAY_LEN_M is None:
        raise RuntimeError("shortest-path worker is not initialized")
    cur_node, dest_node = int(pair[0]), int(pair[1])
    sp = dijkstra_way_path(
        ptr=_SP_PTR,
        idx=_SP_IDX,
        way_len_m=_SP_WAY_LEN_M,
        start=cur_node,
        dest=dest_node,
        max_visits=int(_SP_MAX_VISITS),
    )
    sp_next = int(sp[1]) if len(sp) >= 2 else -1
    return (cur_node, dest_node, sp_next)


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
    flow_solver_steps: Optional[int]
    save_every: int
    early_stop_patience: int
    max_grad_norm: float
    scheduled_sampling_max_p: float
    scheduled_sampling_warmup_epochs: int
    scheduled_sampling_expert: str
    drop_dest_dist_p: float = 0.0
    drop_past_context_p: float = 0.0
    train_decoder_scope: str = "all"
    z_contrast_lambda: float = 0.0
    z_contrast_margin: float = 0.5
    focus_train_route_ids_json: Optional[str] = None
    split_json: Optional[str] = None
    max_train_batches: int = 0
    max_val_batches: int = 0
    log_every_batches: int = 200
    cache_flow_latents: bool = False
    cache_latent_dtype: str = "fp16"
    cache_batch_size: int = 0
    cache_log_every_batches: int = 50
    cache_resample_every_epochs: int = 0
    scheduled_sampling_sp_max_visits: int = 5000
    scheduled_sampling_sp_cache_size: int = 2000000
    scheduled_sampling_sp_workers: int = 0
    scheduled_sampling_sp_pool_chunksize: int = 64
    scheduled_sampling_sp_min_parallel: int = 32
    scheduled_sampling_sp_start_method: str = "auto"


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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _subset_indices_from_route_ids(dataset: WayRouteDataset, route_ids: np.ndarray) -> np.ndarray:
    route_ids = np.asarray(route_ids, dtype=np.int64).reshape(-1)
    if route_ids.size == 0:
        return np.zeros((0,), dtype=np.int64)
    mask = np.isin(dataset.route_ids.astype(np.int64, copy=False), route_ids, assume_unique=False)
    return np.nonzero(mask)[0].astype(np.int64, copy=False)


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    way_seq_pad = batch["way_seq_pad"].to(device)
    way_seq_len = batch["way_seq_len"].to(device)
    route_cond = {k: v.to(device) for k, v in batch["route_cond"].items()}
    trans = {k: v.to(device) for k, v in batch["trans"].items()}
    route_id = batch["route_id"].to(device)
    return {"way_seq_pad": way_seq_pad, "way_seq_len": way_seq_len, "route_cond": route_cond, "trans": trans, "route_id": route_id}


def _build_flow_route_cond(
    *,
    batch: Dict[str, object],
    flow: LatentFlowMatching,
    way_region: Optional[np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    route_cond = batch["route_cond"]
    route_cond_use = {
        "start_pos": route_cond["start_pos"],
        "dest_pos": route_cond["dest_pos"],
        "hour": route_cond["hour"],
        "dow": route_cond["dow"],
        "route_city": route_cond["route_city"],
    }
    if bool(flow.cfg.use_region_seq):
        if way_region is None:
            raise RuntimeError("Flow requires region_seq conditioning, but way_region is missing.")
        pad = batch["way_seq_pad"].detach().cpu().numpy()
        lens = batch["way_seq_len"].detach().cpu().numpy()
        seqs: list[list[int]] = []
        for i in range(int(pad.shape[0])):
            L = int(lens[i])
            seq = pad[i, :L].astype(np.int64, copy=False)
            seqs.append(_region_seq_from_way_seq(seq, way_region))
        route_cond_use["region_seq_pad"] = _pad_region_seqs(seqs, device=device)
    return route_cond_use


def _cache_store_dtype(name: str) -> torch.dtype:
    n = str(name).strip().lower()
    if n == "fp16":
        return torch.float16
    if n == "fp32":
        return torch.float32
    raise ValueError(f"unsupported cache dtype: {name!r} (expect fp16/fp32)")


def _build_flow_latent_cache(
    *,
    flow: LatentFlowMatching,
    loader: DataLoader,
    way_region: Optional[np.ndarray],
    solver_steps: Optional[int],
    device: torch.device,
    cache_dtype: str,
    log_every_batches: int,
    tag: str,
) -> Dict[str, object]:
    flow.eval()
    dt_store = _cache_store_dtype(cache_dtype)
    rid_chunks: list[np.ndarray] = []
    z_chunks: list[torch.Tensor] = []
    n_batches = len(loader)
    t0 = time.perf_counter()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            b = _to_device(batch, device)
            route_cond_use = _build_flow_route_cond(batch=b, flow=flow, way_region=way_region, device=device)
            z = flow.sample(route_cond=route_cond_use, solver_steps=solver_steps)
            z_chunks.append(z.detach().to(device="cpu", dtype=dt_store))
            rid_chunks.append(b["route_id"].detach().cpu().numpy().astype(np.int64, copy=False))
            if int(log_every_batches) > 0 and (((bi + 1) % int(log_every_batches)) == 0 or (bi + 1) == int(n_batches)):
                elapsed = max(1e-6, float(time.perf_counter() - t0))
                it_s = float((bi + 1) / elapsed)
                eta = float((int(n_batches) - int(bi + 1)) / max(1e-6, it_s))
                log.info(
                    f"[cache:{tag}] batch={bi+1}/{int(n_batches)} it/s={it_s:.2f} eta={eta/60.0:.1f}m"
                )

    if len(rid_chunks) <= 0 or len(z_chunks) <= 0:
        raise RuntimeError(f"empty latent cache for {tag}")

    route_ids = np.concatenate(rid_chunks, axis=0).astype(np.int64, copy=False)
    z_cpu = torch.cat(z_chunks, dim=0).contiguous()
    if int(route_ids.size) != int(z_cpu.shape[0]):
        raise RuntimeError(
            f"latent cache shape mismatch for {tag}: route_ids={int(route_ids.size)} vs z_rows={int(z_cpu.shape[0])}"
        )

    rid_max = int(route_ids.max()) if route_ids.size > 0 else -1
    rid_to_row = np.full((rid_max + 1,), -1, dtype=np.int64)
    rid_to_row[route_ids] = np.arange(route_ids.size, dtype=np.int64)
    n_unique = int(np.unique(route_ids).size)
    if n_unique != int(route_ids.size):
        log.warning(f"[cache:{tag}] duplicated route_id found: unique={n_unique} total={int(route_ids.size)}")

    elapsed = max(1e-6, float(time.perf_counter() - t0))
    size_mb = float(z_cpu.numel() * z_cpu.element_size()) / (1024.0 * 1024.0)
    log.info(
        f"[cache:{tag}] built routes={int(route_ids.size)} unique={n_unique} "
        f"shape={tuple(int(x) for x in z_cpu.shape)} dtype={str(z_cpu.dtype)} "
        f"size={size_mb:.1f}MB elapsed={elapsed/60.0:.1f}m"
    )
    return {
        "route_ids": route_ids,
        "rid_to_row": rid_to_row,
        "z_cpu": z_cpu,
    }


def _gather_cached_latents(
    *,
    route_id: torch.Tensor,
    cache: Dict[str, object],
    device: torch.device,
) -> torch.Tensor:
    rid = route_id.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    rid_to_row = cache["rid_to_row"]
    if not isinstance(rid_to_row, np.ndarray):
        raise RuntimeError("invalid cache: rid_to_row")
    if rid.size > 0 and int(np.max(rid)) >= int(rid_to_row.size):
        raise RuntimeError("route_id out of cache range")
    rows = rid_to_row[rid]
    if int(np.any(rows < 0)):
        miss = int(rid[int(np.where(rows < 0)[0][0])])
        raise RuntimeError(f"cached z missing for route_id={miss}")
    rows_t = torch.as_tensor(rows, dtype=torch.long, device="cpu")
    z_cpu = cache["z_cpu"]
    if not isinstance(z_cpu, torch.Tensor):
        raise RuntimeError("invalid cache: z_cpu")
    return z_cpu.index_select(0, rows_t).to(device=device, dtype=torch.float32, non_blocking=True)


def _compress_consecutive_int(seq) -> list[int]:
    out: list[int] = []
    last = None
    for x in seq:
        xx = int(x)
        if last is None or xx != int(last):
            out.append(xx)
            last = xx
    return out


def _region_seq_from_way_seq(way_seq: np.ndarray, way_region: np.ndarray) -> list[int]:
    reg = []
    for w in way_seq.tolist():
        wi = int(w)
        if 0 <= wi < int(way_region.size):
            rr = int(way_region[wi])
            if rr >= 0:
                reg.append(int(rr))
    return _compress_consecutive_int(reg)


def _pad_region_seqs(seqs: list[list[int]], device: torch.device) -> torch.Tensor:
    B = int(len(seqs))
    if B == 0:
        return torch.zeros((0, 1), dtype=torch.long, device=device)
    maxL = max(1, max(len(s) for s in seqs))
    pad = torch.full((B, maxL), -1, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        if not s:
            continue
        pad[i, : len(s)] = torch.as_tensor(s, dtype=torch.long, device=device)
    return pad


def _read_json_ids(path: Path) -> np.ndarray:
    """
    Read a json file containing route_ids.

    Supported formats:
      - [1,2,3]
      - {"route_ids":[...]}
      - {"routes":[...]}
      - {"ids":[...]}
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        ids = obj
    elif isinstance(obj, dict):
        ids = None
        for k in ("route_ids", "routes", "ids"):
            if k in obj and isinstance(obj[k], list):
                ids = obj[k]
                break
        if ids is None:
            raise ValueError(f"Unsupported id json format: keys={sorted(list(obj.keys()))}")
    else:
        raise ValueError(f"Unsupported id json format: type={type(obj)}")
    return np.asarray([int(x) for x in ids], dtype=np.int64).reshape(-1)


def _ss_p(epoch: int, *, max_p: float, warmup_epochs: int) -> float:
    """
    Linear scheduled-sampling probability warmup: 0 -> max_p over warmup_epochs.
    """
    mp = float(max_p)
    if not (mp > 0.0):
        return 0.0
    w = int(warmup_epochs)
    if w <= 1:
        return float(mp)
    t = float(max(0, min(int(epoch), int(w)))) / float(w)
    return float(mp) * t


def _infer_decoder_use_dest_dist_from_state(state: Dict[str, torch.Tensor]) -> bool:
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


def _infer_decoder_past_k_from_state(state: Dict[str, torch.Tensor]) -> int:
    pe = state.get("decoder.past_encoder.pos_emb.weight", None)
    if not isinstance(pe, torch.Tensor) or pe.ndim != 2:
        return 8
    return int(pe.shape[0])


def _load_ae(
    *,
    ae_ckpt: Path,
    way_graph_npz: Path,
    way_features_npz: Path,
    device: torch.device,
    force_decoder_use_step_emb: Optional[bool] = None,
    force_decoder_past_k: Optional[int] = None,
) -> WayCASDAutoEncoder:
    wg = np.load(str(way_graph_npz), allow_pickle=True)
    wf = np.load(str(way_features_npz), allow_pickle=True)
    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64)
    idx = np.asarray(wg["way_adj_idx"], dtype=np.int64)
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1
    way_features = load_way_features_from_npz(Path(way_features_npz), device=device)

    ckpt = torch.load(str(ae_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cfg_dict: Dict[str, object] = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(state, dict):
        raise SystemExit("[FATAL] unexpected AE ckpt format (state_dict missing).")

    use_dest_dist = _infer_decoder_use_dest_dist_from_state(state)
    use_cand_contrast = bool(cfg_dict.get("decoder_use_cand_contrast", False)) or _infer_decoder_use_cand_contrast_from_state(state)
    use_cross_attn = bool(cfg_dict.get("decoder_use_cross_attn", True)) or _infer_decoder_use_cross_attn_from_state(state)
    use_step_emb = bool(cfg_dict.get("decoder_use_step_emb", False)) or _infer_decoder_use_step_emb_from_state(state)
    use_dest_query = bool(cfg_dict.get("decoder_use_dest_query", False)) or _infer_decoder_use_dest_query_from_state(state)
    use_dir_query = bool(cfg_dict.get("decoder_use_dir_query", False)) or _infer_decoder_use_dir_query_from_state(state)
    use_cand_query = bool(cfg_dict.get("decoder_use_cand_query", False)) or _infer_decoder_use_cand_query_from_state(state)
    use_past_ctx = bool(cfg_dict.get("decoder_use_past_context", False)) or _infer_decoder_use_past_context_from_state(state)
    past_k = int(cfg_dict.get("decoder_past_k", _infer_decoder_past_k_from_state(state)))
    if force_decoder_use_step_emb is not None:
        use_step_emb = bool(force_decoder_use_step_emb)
    if force_decoder_past_k is not None and int(force_decoder_past_k) > 0:
        past_k = int(force_decoder_past_k)

    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(cfg_dict.get("d_model", 256)),
            n_latent=int(cfg_dict.get("n_latent", 64)),
            n_heads=int(cfg_dict.get("n_heads", 8)),
            dropout=float(cfg_dict.get("dropout", 0.1)),
            max_candidates=int(cfg_dict.get("max_candidates", 32)),
            max_len=int(cfg_dict.get("max_len", 160)),
            coord_scale=float(cfg_dict.get("coord_scale", 1024.0)),
            decoder_use_dest_dist=bool(use_dest_dist),
            decoder_use_cross_attn=bool(use_cross_attn),
            decoder_n_cross_heads=int(cfg_dict.get("decoder_n_cross_heads", 4)),
            decoder_use_step_emb=bool(use_step_emb),
            decoder_use_dest_query=bool(use_dest_query),
            decoder_use_dir_query=bool(use_dir_query),
            decoder_use_cand_query=bool(use_cand_query),
            decoder_use_cand_contrast=bool(use_cand_contrast),
            decoder_use_past_context=bool(use_past_ctx),
            decoder_past_k=int(past_k),
            decoder_past_n_layers=int(cfg_dict.get("decoder_past_n_layers", 2)),
            decoder_past_n_heads=int(cfg_dict.get("decoder_past_n_heads", 4)),
            segment_size=int(cfg_dict.get("segment_size", 10)),
            segment_n_latent=int(cfg_dict.get("segment_n_latent", 0)),
        ),
        way_features=way_features,
        way_adj_ptr=ptr,
        way_adj_idx=idx,
        n_highway_types=int(max(4, n_highway_types)),
    ).to(device)
    ae.load_state_dict(state, strict=False)
    log.info(
        "AE decoder cfg inferred/override: step_emb=%s past_k=%d cross_attn=%s cand_query=%s past_ctx=%s dest_query=%s dir_query=%s",
        bool(ae.cfg.decoder_use_step_emb),
        int(ae.cfg.decoder_past_k),
        bool(ae.cfg.decoder_use_cross_attn),
        bool(ae.cfg.decoder_use_cand_query),
        bool(ae.cfg.decoder_use_past_context),
        bool(ae.cfg.decoder_use_dest_query),
        bool(ae.cfg.decoder_use_dir_query),
    )
    return ae


def _load_flow(*, flow_ckpt: Path, ae: WayCASDAutoEncoder, device: torch.device) -> LatentFlowMatching:
    ckpt = torch.load(str(flow_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cfg_dict: Dict[str, object] = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(state, dict):
        raise SystemExit("[FATAL] unexpected Flow ckpt format (state_dict missing).")

    cfg = LatentFlowCfg(
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
    if int(cfg.d_model) != int(ae.cfg.d_model) or int(cfg.n_latent) != int(ae.cfg.n_latent):
        raise SystemExit(
            f"[FATAL] AE/Flow mismatch: AE(d_model={int(ae.cfg.d_model)}, n_latent={int(ae.cfg.n_latent)}) "
            f"vs Flow(d_model={int(cfg.d_model)}, n_latent={int(cfg.n_latent)})."
        )
    flow = LatentFlowMatching(cfg=cfg, cond_cfg=ae.decoder.cond_enc.cfg).to(device)
    flow.load_state_dict(state, strict=False)
    return flow


def _succ_slice(ptr: np.ndarray, idx: np.ndarray, way: int) -> np.ndarray:
    s = int(ptr[int(way)])
    e = int(ptr[int(way) + 1])
    if e <= s:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(idx[s:e], dtype=np.int64)


def _build_candidates_ss(
    *,
    ptr: np.ndarray,
    idx: np.ndarray,
    cur_way: int,
    gt_next_way: int,
    max_candidates: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build a candidate row from successors of cur_way (truncate to max_candidates).

    Returns:
      cand_row: (Cmax,) int64, padded with -1
      cand_mask: (Cmax,) bool
      gt_pos: int (>=0 iff gt_next_way is in full successors; injected if truncated, but NOT injected if off-graph)
    """
    succ = _succ_slice(ptr, idx, int(cur_way))
    gt = int(gt_next_way)
    succ_set = set(int(x) for x in succ.tolist())
    gt_is_succ = bool(gt in succ_set)

    c = succ[: int(max_candidates)].astype(np.int64, copy=True)
    gt_pos = -1
    if gt_is_succ:
        if c.size == 0:
            c = np.asarray([gt], dtype=np.int64)
        elif int(gt) not in set(int(x) for x in c.tolist()):
            if int(c.size) < int(max_candidates):
                c = np.concatenate([c, np.asarray([gt], dtype=np.int64)], axis=0)
            else:
                c[-1] = int(gt)
        where = np.nonzero(c == int(gt))[0]
        gt_pos = int(where[0]) if int(where.size) > 0 else -1

    C = min(int(c.size), int(max_candidates))
    row = np.full((int(max_candidates),), -1, dtype=np.int64)
    mask = np.zeros((int(max_candidates),), dtype=bool)
    if C > 0:
        row[:C] = c[:C]
        mask[:C] = True
    return row, mask, int(gt_pos)


def _train_batch_scheduled_sampling(
    *,
    ae: WayCASDAutoEncoder,
    flow: LatentFlowMatching,
    batch: Dict[str, object],
    ptr: np.ndarray,
    idx: np.ndarray,
    way_region: Optional[np.ndarray],
    solver_steps: Optional[int],
    p_ss: float,
    expert_policy: str,
    opt: torch.optim.Optimizer,
    params: List[torch.nn.Parameter],
    max_grad_norm: float,
    device: torch.device,
    drop_dest_dist_p: float = 0.0,
    drop_past_context_p: float = 0.0,
    z_override: Optional[torch.Tensor] = None,
    way_len_m: Optional[np.ndarray] = None,
    sp_next_cache: Optional[Dict[Tuple[int, int], int]] = None,
    sp_max_visits: int = 5000,
    sp_cache_size: int = 0,
    sp_pool: Optional[object] = None,
    sp_pool_chunksize: int = 64,
    sp_min_parallel: int = 32,
) -> Dict[str, float]:
    """
    Scheduled sampling / DAgger for decoder fine-tuning on Flow latents.

    We roll out step-by-step and, with prob p_ss, feed the model's argmax back as next input.
    When the GT next way is NOT a successor under the self-play state, we use an
    expert fallback to keep training on valid graph transitions:
      - expert_policy="destdist": choose successor with min dest distance (geometric heuristic).
      - expert_policy="shortest_path": Dijkstra shortest path oracle from current node to dest.
      - expert_policy="skip": skip this step (no gradient).
    """
    way_seq_pad = batch["way_seq_pad"]  # (B,K) long
    way_seq_len = batch["way_seq_len"]  # (B,) long
    route_cond = batch["route_cond"]

    B = int(way_seq_pad.shape[0])
    coord_scale = float(getattr(ae.way_enc, "coord_scale", ae.cfg.coord_scale))

    route_cond_use = _build_flow_route_cond(batch=batch, flow=flow, way_region=way_region, device=device)
    if z_override is None:
        with torch.no_grad():
            z = flow.sample(route_cond=route_cond_use, solver_steps=solver_steps)
    else:
        z = z_override

    cur_way = way_seq_pad[:, 0].detach().clone().to(dtype=torch.long)  # (B,)
    dest_way = route_cond.get("dest_way", None)
    dest_way = (dest_way.to(device=device, dtype=torch.long) if isinstance(dest_way, torch.Tensor) else None)
    done = torch.zeros((B,), dtype=torch.bool, device=device)
    if dest_way is not None:
        done = done | (cur_way == dest_way)

    paths: List[List[int]] = [[int(w)] for w in cur_way.detach().cpu().tolist()]

    max_steps = int(way_seq_len.max().item()) - 1 if B > 0 else 0
    max_steps = max(0, int(max_steps))

    opt.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_acc = 0.0
    total_steps = 0
    n_use_pred = 0
    n_off_gt = 0
    n_skipped = 0
    n_sp_queries = 0
    n_sp_cache_hit = 0

    Cmax = int(ae.cfg.max_candidates)
    Kpast = int(ae.cfg.decoder_past_k)
    use_past = bool(ae.decoder.use_past_context)
    # SIB-style bypass dropout (sample once per batch, apply to all steps in this batch)
    drop_dd = bool(float(drop_dest_dist_p) > 0.0 and torch.rand(1).item() < float(drop_dest_dist_p))
    drop_pc = bool(float(drop_past_context_p) > 0.0 and torch.rand(1).item() < float(drop_past_context_p))

    for step_idx in range(int(max_steps)):
        has_next = (way_seq_len > int(step_idx) + 1)
        active = (~done) & has_next
        if not bool(active.any().item()):
            break
        route_idx = torch.nonzero(active, as_tuple=False).reshape(-1)
        T = int(route_idx.numel())
        if T <= 0:
            break

        cur_t = cur_way[route_idx].detach().cpu().numpy().astype(np.int64, copy=False)
        gt_next_t = way_seq_pad[route_idx, int(step_idx) + 1].detach().cpu().numpy().astype(np.int64, copy=False)

        cand_rows = np.full((T, Cmax), -1, dtype=np.int64)
        cand_masks = np.zeros((T, Cmax), dtype=bool)
        gt_pos = np.full((T,), -1, dtype=np.int64)
        for i in range(T):
            row, m, pos = _build_candidates_ss(
                ptr=ptr,
                idx=idx,
                cur_way=int(cur_t[i]),
                gt_next_way=int(gt_next_t[i]),
                max_candidates=int(Cmax),
            )
            cand_rows[i] = row
            cand_masks[i] = m
            gt_pos[i] = int(pos)

        cand_way = torch.as_tensor(cand_rows, dtype=torch.long, device=device)
        cand_mask = torch.as_tensor(cand_masks, dtype=torch.bool, device=device)

        expert_idx = torch.full((T,), -1, dtype=torch.long, device=device)
        ok_gt = torch.as_tensor(gt_pos >= 0, dtype=torch.bool, device=device)
        if bool(ok_gt.any().item()):
            expert_idx[ok_gt] = torch.as_tensor(gt_pos[gt_pos >= 0], dtype=torch.long, device=device)

        off = ~ok_gt
        if bool(off.any().item()):
            n_off_gt += int(off.long().sum().item())
            if str(expert_policy) == "skip":
                n_skipped += int(off.long().sum().item())
            elif str(expert_policy) == "destdist":
                dest = route_cond_use["dest_pos"][route_idx].to(dtype=torch.float32)
                if coord_scale > 0:
                    dest = dest / float(coord_scale)
                cand_geom, _tier, _hw = ae.way_enc._lookup(cand_way)
                cand_center = cand_geom[..., :2].to(dtype=torch.float32)
                dist = torch.norm(dest[:, None, :] - cand_center, dim=-1)
                dist = dist.masked_fill(~cand_mask, float("inf"))
                best = torch.argmin(dist, dim=-1)
                expert_idx[off] = best[off]
            elif str(expert_policy) == "shortest_path":
                # DAgger oracle: Dijkstra shortest path from current node to dest.
                if way_len_m is None:
                    raise ValueError("shortest_path expert requires way_len_m (load from way_features_npz)")
                off_indices = torch.nonzero(off, as_tuple=False).reshape(-1).cpu().tolist()
                dw_cpu = dest_way[route_idx].cpu().tolist() if dest_way is not None else [0] * T

                # Build unique miss keys per step to avoid repeated Dijkstra calls.
                miss_keys: List[Tuple[int, int]] = []
                miss_map: Dict[Tuple[int, int], List[int]] = {}
                key_for_ii: Dict[int, Tuple[int, int]] = {}
                for ii in off_indices:
                    n_sp_queries += 1
                    cur_node = int(cur_t[ii])
                    dest_node = int(dw_cpu[ii])
                    cache_key = (cur_node, dest_node)
                    key_for_ii[int(ii)] = cache_key
                    if sp_next_cache is not None and cache_key in sp_next_cache:
                        n_sp_cache_hit += 1
                        continue
                    if cache_key not in miss_map:
                        miss_map[cache_key] = [int(ii)]
                        miss_keys.append(cache_key)
                    else:
                        miss_map[cache_key].append(int(ii))

                # Solve misses (parallel if pool is available).
                if miss_keys:
                    solved: Dict[Tuple[int, int], int] = {}
                    use_parallel = (sp_pool is not None and len(miss_keys) >= int(sp_min_parallel))
                    if use_parallel:
                        try:
                            it = sp_pool.imap_unordered(_sp_next_for_pair, miss_keys, chunksize=max(1, int(sp_pool_chunksize)))
                            for cur_node, dest_node, sp_next in it:
                                solved[(int(cur_node), int(dest_node))] = int(sp_next)
                        except Exception:
                            # Fallback to local solve if pool fails.
                            for cache_key in miss_keys:
                                cur_node, dest_node = int(cache_key[0]), int(cache_key[1])
                                sp = dijkstra_way_path(
                                    ptr=ptr,
                                    idx=idx,
                                    way_len_m=way_len_m,
                                    start=cur_node,
                                    dest=dest_node,
                                    max_visits=int(sp_max_visits),
                                )
                                solved[cache_key] = int(sp[1]) if len(sp) >= 2 else -1
                    else:
                        for cache_key in miss_keys:
                            cur_node, dest_node = int(cache_key[0]), int(cache_key[1])
                            sp = dijkstra_way_path(
                                ptr=ptr,
                                idx=idx,
                                way_len_m=way_len_m,
                                start=cur_node,
                                dest=dest_node,
                                max_visits=int(sp_max_visits),
                            )
                            solved[cache_key] = int(sp[1]) if len(sp) >= 2 else -1

                    if sp_next_cache is not None:
                        for cache_key, sp_next in solved.items():
                            if int(sp_cache_size) <= 0 or len(sp_next_cache) < int(sp_cache_size):
                                sp_next_cache[cache_key] = int(sp_next)

                for ii in off_indices:
                    cache_key = key_for_ii[int(ii)]
                    sp_next = int(sp_next_cache.get(cache_key, -1)) if sp_next_cache is not None else -1
                    if sp_next >= 0:
                        cand_row_i = cand_rows[ii]
                        found = -1
                        for ci in range(Cmax):
                            if int(cand_row_i[ci]) == sp_next:
                                found = ci
                                break
                        if found >= 0:
                            expert_idx[ii] = found
                        else:
                            n_skipped += 1
                    else:
                        n_skipped += 1
            else:
                raise ValueError(f"unsupported scheduled_sampling_expert: {expert_policy!r}")

        past_way_tensor: Optional[torch.Tensor] = None
        past_mask_tensor: Optional[torch.Tensor] = None
        if use_past:
            past_way_tensor = torch.full((T, Kpast), -1, dtype=torch.long, device=device)
            for i, ri in enumerate(route_idx.detach().cpu().tolist()):
                hist = paths[int(ri)]
                past_list = hist[:-1][-Kpast:] if len(hist) > 1 else []
                off0 = Kpast - len(past_list)
                for j, w in enumerate(past_list):
                    past_way_tensor[i, off0 + j] = int(w)
            past_mask_tensor = (past_way_tensor >= 0)

        trans = {
            "route_idx": route_idx,
            "cur_way": cur_way[route_idx],
            "cand_way": cand_way,
            "cand_mask": cand_mask,
            "step": torch.full((T,), int(step_idx), dtype=torch.long, device=device),
        }
        if past_way_tensor is not None and past_mask_tensor is not None:
            trans["past_way"] = past_way_tensor
            trans["past_mask"] = past_mask_tensor

        # IMPORTANT: do NOT reuse a single cond_emb tensor across multiple backward() calls.
        # Scheduled sampling runs step-by-step and calls backward per step; caching cond_emb would
        # cause "Trying to backward through the graph a second time" because cond_emb's graph
        # would be shared across steps and freed after the first backward.
        cond_emb = ae.decoder.cond_enc(
            start_pos=route_cond_use["start_pos"],
            dest_pos=route_cond_use["dest_pos"],
            hour=route_cond_use["hour"],
            dow=route_cond_use["dow"],
            route_city=route_cond_use["route_city"],
        )  # (B,d)

        logits = ae.decoder.score_candidates(
            way_embedder=ae.way_enc,
            latent_tokens=z,
            route_cond=route_cond_use,
            trans=trans,
            cond_emb=cond_emb,
            drop_dest_dist=drop_dd,
            drop_past_context=drop_pc,
        )

        with torch.no_grad():
            pred_idx = torch.argmax(logits, dim=-1)
            pred_next = cand_way[torch.arange(T, device=device), pred_idx]
            # For invalid expert_idx (e.g., expert_policy="skip"), fall back to model prediction.
            expert_next = torch.where(
                (expert_idx >= 0),
                cand_way[torch.arange(T, device=device), torch.clamp(expert_idx, min=0)],
                pred_next,
            )

        valid = expert_idx >= 0
        if bool(valid.any().item()):
            tgt = expert_idx[valid]
            loss = F.cross_entropy(logits[valid], tgt, reduction="mean")
            loss.backward()
            n_valid = int(valid.long().sum().item())
            total_loss += float(loss.detach().item()) * float(n_valid)
            total_steps += int(n_valid)
            total_acc += float((pred_idx[valid] == tgt).float().sum().item())

        with torch.no_grad():
            if float(p_ss) > 0.0:
                use_pred = (torch.rand((T,), device=device) < float(p_ss))
            else:
                use_pred = torch.zeros((T,), dtype=torch.bool, device=device)
            n_use_pred += int(use_pred.long().sum().item())
            nxt = torch.where(use_pred, pred_next, expert_next)

            for i, ri in enumerate(route_idx.detach().cpu().tolist()):
                w = int(nxt[int(i)].item())
                cur_way[int(ri)] = int(w)
                paths[int(ri)].append(int(w))
                if dest_way is not None and int(w) == int(dest_way[int(ri)].item()):
                    done[int(ri)] = True

    torch.nn.utils.clip_grad_norm_(params, max_norm=float(max_grad_norm))
    opt.step()

    denom = max(1, int(total_steps))
    return {
        "loss": float(total_loss / float(denom)),
        "acc": float(total_acc / float(denom)),
        "n_steps": float(int(total_steps)),
        "p_ss": float(p_ss),
        "use_pred_frac": float(n_use_pred / float(max(1, total_steps))),
        "off_gt_frac": float(n_off_gt / float(max(1, total_steps + n_skipped))),
        "skipped_frac": float(n_skipped / float(max(1, total_steps + n_skipped))),
        "sp_queries": float(int(n_sp_queries)),
        "sp_cache_hit_frac": float(n_sp_cache_hit / float(max(1, n_sp_queries))),
    }


@torch.no_grad()
def _eval_epoch(
    *,
    ae: WayCASDAutoEncoder,
    flow: LatentFlowMatching,
    loader: DataLoader,
    device: torch.device,
    way_region: Optional[np.ndarray],
    solver_steps: Optional[int],
    max_batches: int,
    latent_cache: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    ae.eval()
    flow.eval()
    losses: list[float] = []
    n_items = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if int(max_batches) > 0 and (bi + 1) > int(max_batches):
                break
            b = _to_device(batch, device)
            route_cond_use = _build_flow_route_cond(batch=b, flow=flow, way_region=way_region, device=device)
            if latent_cache is None:
                z = flow.sample(route_cond=route_cond_use, solver_steps=solver_steps)
            else:
                z = _gather_cached_latents(route_id=b["route_id"], cache=latent_cache, device=device)
            logits = ae.decoder.score_candidates(way_embedder=ae.way_enc, latent_tokens=z, route_cond=route_cond_use, trans=b["trans"])
            tgt = b["trans"]["target_idx"].to(dtype=torch.long)
            loss = F.cross_entropy(logits, tgt, reduction="mean")
            losses.append(float(loss.detach().item()))
            n_items += 1
    return {"loss": float(np.mean(losses)) if n_items > 0 else float("nan")}


def main() -> None:
    p = argparse.ArgumentParser(description="E2: Joint fine-tune Way decoder on Flow latents (teacher forcing).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--flow_ckpt", type=Path, required=True)
    p.add_argument("--way_regions_npz", type=Path, default=None, help="Required when Flow uses region_seq.")
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--max_routes", type=int, default=None)
    p.add_argument(
        "--split_json",
        type=Path,
        default=None,
        help="Optional OD-disjoint split json (expects splits.train/val/test route_ids). Overrides val_ratio.",
    )
    p.add_argument("--flow_solver_steps", type=int, default=0, help="Override flow solver steps (0=use ckpt).")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_train_batches", type=int, default=0, help="Limit batches per epoch for faster iteration (0=all).")
    p.add_argument("--max_val_batches", type=int, default=0, help="Limit val batches per epoch for faster iteration (0=all).")
    p.add_argument("--log_every_batches", type=int, default=200, help="Log training progress every N batches (0=disable).")
    p.add_argument("--cache_flow_latents", action="store_true", help="Cache one z_flow per route and reuse during E2 training (deterministic latent per route).")
    p.add_argument("--cache_latent_dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="CPU dtype for cached z_flow tensors.")
    p.add_argument("--cache_batch_size", type=int, default=0, help="Batch size for cache precompute (0=use train batch_size).")
    p.add_argument("--cache_log_every_batches", type=int, default=50, help="Log cache precompute progress every N batches.")
    p.add_argument("--cache_resample_every_epochs", type=int, default=0, help="If >0, rebuild cache every N epochs (0=build once).")
    p.add_argument("--scheduled_sampling_sp_max_visits", type=int, default=5000, help="Dijkstra max_visits for shortest_path expert.")
    p.add_argument("--scheduled_sampling_sp_cache_size", type=int, default=2000000, help="Max cached (cur,dest)->next entries for shortest_path expert (0=unlimited).")
    p.add_argument("--scheduled_sampling_sp_workers", type=int, default=0, help="Process workers for shortest_path expert (0=disable multiprocessing).")
    p.add_argument("--scheduled_sampling_sp_pool_chunksize", type=int, default=64, help="Pool imap chunksize for shortest_path expert.")
    p.add_argument("--scheduled_sampling_sp_min_parallel", type=int, default=32, help="Minimum unique shortest-path queries in a step to use multiprocessing.")
    p.add_argument(
        "--scheduled_sampling_sp_start_method",
        type=str,
        default="auto",
        choices=["auto", "fork", "spawn", "forkserver"],
        help="Multiprocessing start method for shortest_path expert pool.",
    )
    p.add_argument("--force_decoder_use_step_emb", action="store_true", help="Force-enable decoder step embedding even if AE ckpt did not use it.")
    p.add_argument("--force_decoder_past_k", type=int, default=0, help="Override decoder past_k (0=use ckpt inferred).")
    p.add_argument("--scheduled_sampling_max_p", type=float, default=0.0, help="Scheduled sampling max p (0=disable).")
    p.add_argument("--scheduled_sampling_warmup_epochs", type=int, default=20, help="Warmup epochs to reach max_p.")
    p.add_argument(
        "--scheduled_sampling_expert",
        type=str,
        default="destdist",
        choices=["destdist", "skip", "shortest_path"],
        help="Expert fallback when GT next is not a successor under self-play state. "
             "shortest_path uses Dijkstra oracle on the way graph (DAgger).",
    )
    p.add_argument(
        "--focus_train_route_ids_json",
        type=Path,
        default=None,
        help="Optional: further restrict training set to these route_ids (json list or {route_ids:[...]}).",
    )
    p.add_argument("--drop_dest_dist_p", type=float, default=0.0, help="SIB-style bypass dropout prob for dest_dist during E2 training (0=disable).")
    p.add_argument("--drop_past_context_p", type=float, default=0.0, help="SIB-style bypass dropout prob for past_context during E2 training (0=disable).")
    p.add_argument(
        "--train_decoder_scope",
        type=str,
        default="all",
        choices=["all", "scorer"],
        help="Decoder params to train: all=decoder.* (default), scorer=decoder.scorer.* only.",
    )
    p.add_argument("--z_contrast_lambda", type=float, default=0.0, help="z-contrastive regularization weight (0=disable). Forces decoder to distinguish real vs shuffled z.")
    p.add_argument("--z_contrast_margin", type=float, default=0.5, help="Minimum gap between shuffled-z loss and real-z loss.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (0.0 <= float(args.drop_dest_dist_p) <= 1.0):
        raise SystemExit("[FATAL] --drop_dest_dist_p must be in [0, 1].")
    if not (0.0 <= float(args.drop_past_context_p) <= 1.0):
        raise SystemExit("[FATAL] --drop_past_context_p must be in [0, 1].")

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
        flow_solver_steps=(int(args.flow_solver_steps) if int(args.flow_solver_steps) > 0 else None),
        save_every=int(args.save_every),
        early_stop_patience=int(args.early_stop_patience),
        max_grad_norm=float(args.max_grad_norm),
        scheduled_sampling_max_p=float(args.scheduled_sampling_max_p),
        scheduled_sampling_warmup_epochs=int(args.scheduled_sampling_warmup_epochs),
        scheduled_sampling_expert=str(args.scheduled_sampling_expert),
        drop_dest_dist_p=float(args.drop_dest_dist_p),
        drop_past_context_p=float(args.drop_past_context_p),
        train_decoder_scope=str(args.train_decoder_scope),
        z_contrast_lambda=float(args.z_contrast_lambda),
        z_contrast_margin=float(args.z_contrast_margin),
        focus_train_route_ids_json=(str(args.focus_train_route_ids_json) if args.focus_train_route_ids_json is not None else None),
        split_json=(str(args.split_json) if args.split_json is not None else None),
        max_train_batches=int(args.max_train_batches),
        max_val_batches=int(args.max_val_batches),
        log_every_batches=int(args.log_every_batches),
        cache_flow_latents=bool(args.cache_flow_latents),
        cache_latent_dtype=str(args.cache_latent_dtype),
        cache_batch_size=int(args.cache_batch_size),
        cache_log_every_batches=int(args.cache_log_every_batches),
        cache_resample_every_epochs=int(args.cache_resample_every_epochs),
        scheduled_sampling_sp_max_visits=int(args.scheduled_sampling_sp_max_visits),
        scheduled_sampling_sp_cache_size=int(args.scheduled_sampling_sp_cache_size),
        scheduled_sampling_sp_workers=int(args.scheduled_sampling_sp_workers),
        scheduled_sampling_sp_pool_chunksize=int(args.scheduled_sampling_sp_pool_chunksize),
        scheduled_sampling_sp_min_parallel=int(args.scheduled_sampling_sp_min_parallel),
        scheduled_sampling_sp_start_method=str(args.scheduled_sampling_sp_start_method),
    )

    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    log.info(f"device={device}")

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    if args.split_json is not None and args.max_routes is not None:
        log.warning("--split_json is set, ignoring --max_routes to avoid inconsistent splits.")
    dataset = WayRouteDataset(
        routes,
        max_routes=(None if args.split_json is not None else cfg.max_routes),
        max_way_len=int(cfg.max_way_len),
        min_hops=int(cfg.min_hops),
    )
    if args.split_json is None:
        tr_idx, va_idx = _split_dataset(len(dataset), cfg.val_ratio, cfg.seed)
    else:
        split = _read_json(Path(args.split_json))
        splits = split.get("splits", split)
        tr_rids = np.asarray(splits.get("train", []), dtype=np.int64).reshape(-1)
        va_rids = np.asarray(splits.get("val", []), dtype=np.int64).reshape(-1)
        tr_idx = _subset_indices_from_route_ids(dataset, tr_rids)
        va_idx = _subset_indices_from_route_ids(dataset, va_rids)
        if int(tr_idx.size) == 0 or int(va_idx.size) == 0:
            raise SystemExit(
                f"[FATAL] split_json produced empty subsets: train_idx={int(tr_idx.size)} val_idx={int(va_idx.size)}. "
                "Check min_hops/max_way_len match split generation."
            )
        log.info(
            f"split_json={args.split_json} train_routes={int(tr_rids.size)} val_routes={int(va_rids.size)} "
            f"=> train_idx={int(tr_idx.size)} val_idx={int(va_idx.size)}"
        )
    train_set = Subset(dataset, tr_idx.tolist())
    val_set = Subset(dataset, va_idx.tolist())
    if args.focus_train_route_ids_json is not None:
        focus_rids = _read_json_ids(Path(args.focus_train_route_ids_json))
        focus_idx = _subset_indices_from_route_ids(dataset, focus_rids)
        if int(focus_idx.size) == 0:
            raise SystemExit(f"[FATAL] focus_train_route_ids_json produced empty subset: {args.focus_train_route_ids_json}")
        tr_mask = np.isin(tr_idx.astype(np.int64, copy=False), focus_idx.astype(np.int64, copy=False), assume_unique=False)
        tr_idx2 = tr_idx[tr_mask]
        if int(tr_idx2.size) == 0:
            raise SystemExit("[FATAL] focus_train_route_ids_json removed all training samples (intersection empty).")
        train_set = Subset(dataset, tr_idx2.tolist())
        log.info(f"focus_train_route_ids_json={args.focus_train_route_ids_json} => train={len(train_set)} (was {int(tr_idx.size)})")
    log.info(f"routes: total={len(dataset)} train={len(train_set)} val={len(val_set)} min_hops={cfg.min_hops} max_way_len={cfg.max_way_len}")

    # Load models
    ae = _load_ae(
        ae_ckpt=Path(args.ae_ckpt),
        way_graph_npz=Path(args.way_graph_npz),
        way_features_npz=Path(args.way_features_npz),
        device=device,
        force_decoder_use_step_emb=(True if bool(args.force_decoder_use_step_emb) else None),
        force_decoder_past_k=(int(args.force_decoder_past_k) if int(args.force_decoder_past_k) > 0 else None),
    )
    flow = _load_flow(flow_ckpt=Path(args.flow_ckpt), ae=ae, device=device)

    # Freeze everything except selected decoder scope.
    for p0 in ae.parameters():
        p0.requires_grad_(False)
    scope = str(cfg.train_decoder_scope).strip().lower()
    train_prefix = "decoder.scorer." if scope == "scorer" else "decoder."
    for name, p0 in ae.named_parameters():
        if str(name).startswith(train_prefix):
            p0.requires_grad_(True)
    for p0 in flow.parameters():
        p0.requires_grad_(False)
    ae.train()
    flow.eval()

    way_region: Optional[np.ndarray] = None
    if bool(flow.cfg.use_region_seq):
        if args.way_regions_npz is None:
            raise SystemExit("[FATAL] Flow requires region_seq conditioning, so --way_regions_npz is required.")
        wr = np.load(str(Path(args.way_regions_npz)), allow_pickle=True)
        if "way_region" not in wr.files:
            raise SystemExit("[FATAL] way_regions_npz missing key: way_region")
        way_region = np.asarray(wr["way_region"], dtype=np.int64).reshape(-1)

    # Collate (for candidate sets and past context).
    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64)
    adj = np.asarray(wg["way_adj_idx"], dtype=np.int64)

    # Load way_len_m for Dijkstra oracle (DAgger shortest_path expert)
    _way_len_m: Optional[np.ndarray] = None
    if str(cfg.scheduled_sampling_expert) == "shortest_path":
        wf = np.load(str(args.way_features_npz), allow_pickle=True)
        _way_len_m = np.asarray(wf["way_len_m"], dtype=np.float64).reshape(-1)
        log.info(f"Loaded way_len_m ({_way_len_m.shape[0]} ways) for shortest_path expert (DAgger)")

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

    sp_pool = None
    if str(cfg.scheduled_sampling_expert) == "shortest_path" and int(cfg.scheduled_sampling_sp_workers) > 0:
        if _way_len_m is None:
            raise SystemExit("[FATAL] shortest_path expert requested but way_len_m is not loaded.")
        method = str(cfg.scheduled_sampling_sp_start_method).strip().lower()
        if method == "auto":
            method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        try:
            ctx = mp.get_context(method)
            sp_pool = ctx.Pool(
                processes=int(cfg.scheduled_sampling_sp_workers),
                initializer=_init_sp_worker,
                initargs=(ptr, adj, _way_len_m, int(cfg.scheduled_sampling_sp_max_visits)),
            )
            log.info(
                f"shortest_path pool enabled: workers={int(cfg.scheduled_sampling_sp_workers)} "
                f"start_method={method} chunksize={int(cfg.scheduled_sampling_sp_pool_chunksize)} "
                f"min_parallel={int(cfg.scheduled_sampling_sp_min_parallel)}"
            )
        except Exception as e:
            sp_pool = None
            log.warning(f"failed to create shortest_path pool (fallback to local): {e}")

    cache_batch_size = int(cfg.cache_batch_size) if int(cfg.cache_batch_size) > 0 else int(cfg.batch_size)
    cache_train_loader = DataLoader(
        train_set,
        batch_size=int(cache_batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
    )
    cache_val_loader = DataLoader(
        val_set,
        batch_size=int(cache_batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
    )

    params = [p0 for p0 in ae.parameters() if p0.requires_grad]
    n_train = int(sum(int(p0.numel()) for p0 in params))
    n_all = int(sum(int(p0.numel()) for p0 in ae.parameters()))
    log.info(f"train_decoder_scope={scope} trainable_params={n_train}/{n_all} ({(100.0 * n_train / max(1, n_all)):.2f}%)")
    if len(params) == 0:
        raise SystemExit(f"[FATAL] no trainable params selected for scope={scope!r}.")
    opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    train_latent_cache: Optional[Dict[str, object]] = None
    val_latent_cache: Optional[Dict[str, object]] = None
    sp_next_cache: Optional[Dict[Tuple[int, int], int]] = (
        {} if str(cfg.scheduled_sampling_expert) == "shortest_path" else None
    )

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    start_epoch = 1

    def _save_ckpt(path: Path, *, epoch: int, val_loss: float) -> None:
        ckpt = {
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "model_state_dict": ae.state_dict(),
            "config": asdict(ae.cfg),
        }
        torch.save(ckpt, str(path))

    for epoch in range(start_epoch, int(cfg.n_epochs) + 1):
        ae.train()
        losses: list[float] = []
        ss_stats: list[Dict[str, float]] = []
        t_epoch = time.perf_counter()
        if bool(cfg.cache_flow_latents):
            need_rebuild = (
                train_latent_cache is None
                or val_latent_cache is None
                or (int(cfg.cache_resample_every_epochs) > 0 and ((int(epoch) - int(start_epoch)) % int(cfg.cache_resample_every_epochs) == 0))
            )
            if need_rebuild:
                log.info(
                    f"[cache] rebuilding z_flow cache at epoch={int(epoch)} "
                    f"dtype={cfg.cache_latent_dtype} batch_size={int(cache_batch_size)}"
                )
                train_latent_cache = _build_flow_latent_cache(
                    flow=flow,
                    loader=cache_train_loader,
                    way_region=way_region,
                    solver_steps=cfg.flow_solver_steps,
                    device=device,
                    cache_dtype=str(cfg.cache_latent_dtype),
                    log_every_batches=int(cfg.cache_log_every_batches),
                    tag=f"train/e{int(epoch)}",
                )
                val_latent_cache = _build_flow_latent_cache(
                    flow=flow,
                    loader=cache_val_loader,
                    way_region=way_region,
                    solver_steps=cfg.flow_solver_steps,
                    device=device,
                    cache_dtype=str(cfg.cache_latent_dtype),
                    log_every_batches=int(cfg.cache_log_every_batches),
                    tag=f"val/e{int(epoch)}",
                )
        n_total = len(train_loader)
        n_cap = int(cfg.max_train_batches) if int(cfg.max_train_batches) > 0 else int(n_total)
        n_cap = min(int(n_cap), int(n_total))
        for bi, batch in enumerate(train_loader):
            if int(cfg.max_train_batches) > 0 and (bi + 1) > int(cfg.max_train_batches):
                break
            b = _to_device(batch, device)

            p_ss = _ss_p(epoch, max_p=float(cfg.scheduled_sampling_max_p), warmup_epochs=int(cfg.scheduled_sampling_warmup_epochs))
            if float(p_ss) > 0.0:
                z_override = None
                if train_latent_cache is not None:
                    z_override = _gather_cached_latents(route_id=b["route_id"], cache=train_latent_cache, device=device)
                st = _train_batch_scheduled_sampling(
                    ae=ae,
                    flow=flow,
                    batch=b,
                    ptr=ptr,
                    idx=adj,
                    way_region=way_region,
                    solver_steps=cfg.flow_solver_steps,
                    p_ss=float(p_ss),
                    expert_policy=str(cfg.scheduled_sampling_expert),
                    opt=opt,
                    params=params,
                    max_grad_norm=float(cfg.max_grad_norm),
                    device=device,
                    drop_dest_dist_p=float(cfg.drop_dest_dist_p),
                    drop_past_context_p=float(cfg.drop_past_context_p),
                    z_override=z_override,
                    way_len_m=_way_len_m,
                    sp_next_cache=sp_next_cache,
                    sp_max_visits=int(cfg.scheduled_sampling_sp_max_visits),
                    sp_cache_size=int(cfg.scheduled_sampling_sp_cache_size),
                    sp_pool=sp_pool,
                    sp_pool_chunksize=int(cfg.scheduled_sampling_sp_pool_chunksize),
                    sp_min_parallel=int(cfg.scheduled_sampling_sp_min_parallel),
                )
                losses.append(float(st["loss"]))
                ss_stats.append(st)
                if int(cfg.log_every_batches) > 0 and ((bi + 1) % int(cfg.log_every_batches) == 0):
                    dt = max(1e-6, float(time.perf_counter() - t_epoch))
                    steps = int(bi + 1)
                    it_s = float(steps / dt)
                    eta_s = float((max(0, int(n_cap) - steps)) / max(1e-6, it_s))
                    log.info(
                        f"epoch={epoch} batch={steps}/{int(n_cap)} "
                        f"train_loss={float(np.mean(losses)) if losses else float('nan'):.4f} "
                        f"ss(p={float(st.get('p_ss', 0.0)):.3f}, acc={float(st.get('acc', float('nan'))):.3f}, "
                        f"off_gt={float(st.get('off_gt_frac', 0.0)):.3f}, skip={float(st.get('skipped_frac', 0.0)):.3f}, "
                        f"sp_q={int(st.get('sp_queries', 0.0))}, sp_hit={float(st.get('sp_cache_hit_frac', 0.0)):.3f}) "
                        f"it/s={it_s:.2f} eta={eta_s/60.0:.1f}m"
                    )
                continue

            # Teacher forcing (default)
            route_cond_use = _build_flow_route_cond(batch=b, flow=flow, way_region=way_region, device=device)
            if train_latent_cache is None:
                with torch.no_grad():
                    z = flow.sample(route_cond=route_cond_use, solver_steps=cfg.flow_solver_steps)
            else:
                z = _gather_cached_latents(route_id=b["route_id"], cache=train_latent_cache, device=device)

            # SIB-style bypass dropout (sample once per training batch)
            drop_dd = bool(float(cfg.drop_dest_dist_p) > 0.0 and torch.rand(1).item() < float(cfg.drop_dest_dist_p))
            drop_pc = bool(float(cfg.drop_past_context_p) > 0.0 and torch.rand(1).item() < float(cfg.drop_past_context_p))
            logits = ae.decoder.score_candidates(
                way_embedder=ae.way_enc,
                latent_tokens=z,
                route_cond=route_cond_use,
                trans=b["trans"],
                drop_dest_dist=drop_dd,
                drop_past_context=drop_pc,
            )
            tgt = b["trans"]["target_idx"].to(dtype=torch.long)
            loss_real = F.cross_entropy(logits, tgt, reduction="mean")

            # z-Contrastive Regularization: force decoder to distinguish real vs shuffled z
            z_contrast_lam = float(cfg.z_contrast_lambda)
            if z_contrast_lam > 0.0:
                B_z = z.size(0)
                perm = torch.randperm(B_z, device=z.device)
                z_shuf = z[perm]
                logits_shuf = ae.decoder.score_candidates(
                    way_embedder=ae.way_enc,
                    latent_tokens=z_shuf,
                    route_cond=route_cond_use,
                    trans=b["trans"],
                    drop_dest_dist=drop_dd,
                    drop_past_context=drop_pc,
                )
                loss_shuf = F.cross_entropy(logits_shuf, tgt, reduction="mean")
                z_gap = loss_shuf - loss_real
                z_reg = F.relu(float(cfg.z_contrast_margin) - z_gap)
                loss = loss_real + z_contrast_lam * z_reg
            else:
                loss = loss_real

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(cfg.max_grad_norm))
            opt.step()
            losses.append(float(loss_real.detach().item()))

            if int(cfg.log_every_batches) > 0 and ((bi + 1) % int(cfg.log_every_batches) == 0):
                dt = max(1e-6, float(time.perf_counter() - t_epoch))
                steps = int(bi + 1)
                it_s = float(steps / dt)
                eta_s = float((max(0, int(n_cap) - steps)) / max(1e-6, it_s))
                log.info(
                    f"epoch={epoch} batch={steps}/{int(n_cap)} "
                    f"train_loss={float(np.mean(losses)) if losses else float('nan'):.4f} "
                    f"it/s={it_s:.2f} eta={eta_s/60.0:.1f}m"
                )

        tr_loss = float(np.mean(losses)) if losses else float("nan")
        train_ss: Dict[str, float] = {}
        if ss_stats:
            keys = ["p_ss", "acc", "use_pred_frac", "off_gt_frac", "skipped_frac", "n_steps", "sp_queries", "sp_cache_hit_frac"]
            train_ss = {k: float(np.mean([float(s.get(k, float("nan"))) for s in ss_stats])) for k in keys}
        va = _eval_epoch(
            ae=ae,
            flow=flow,
            loader=val_loader,
            device=device,
            way_region=way_region,
            solver_steps=cfg.flow_solver_steps,
            max_batches=int(cfg.max_val_batches),
            latent_cache=val_latent_cache,
        )
        va_loss = float(va["loss"])
        if train_ss:
            log.info(
                f"epoch={epoch} train_loss={tr_loss:.6f} val_loss={va_loss:.6f} best={best_val:.6f}@{best_epoch} "
                f"ss(p={train_ss.get('p_ss', 0.0):.3f}, acc={train_ss.get('acc', float('nan')):.3f}, "
                f"use_pred={train_ss.get('use_pred_frac', 0.0):.3f}, off_gt={train_ss.get('off_gt_frac', 0.0):.3f}, "
                f"skip={train_ss.get('skipped_frac', 0.0):.3f}, steps={int(train_ss.get('n_steps', 0.0))}, "
                f"sp_q={int(train_ss.get('sp_queries', 0.0))}, sp_hit={train_ss.get('sp_cache_hit_frac', 0.0):.3f})"
            )
        else:
            log.info(f"epoch={epoch} train_loss={tr_loss:.6f} val_loss={va_loss:.6f} best={best_val:.6f}@{best_epoch}")

        # Save progress snapshot (single-file, easy to sync).
        progress = {
            "ok": True,
            "task": "train_way_casd_decoder_joint",
            "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
            "epoch": int(epoch),
            "train": {"loss": float(tr_loss)},
            "train_ss": train_ss,
            "val": {"loss": float(va_loss)},
            "best_val_loss": float(best_val),
            "best_epoch": int(best_epoch),
        }
        (out_dir / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

        if va_loss < best_val:
            best_val = float(va_loss)
            best_epoch = int(epoch)
            bad_epochs = 0
            _save_ckpt(out_dir / "ckpt_best.pt", epoch=epoch, val_loss=best_val)
        else:
            bad_epochs += 1
        if int(cfg.save_every) > 0 and (int(epoch) % int(cfg.save_every) == 0):
            _save_ckpt(out_dir / "ckpt_last.pt", epoch=epoch, val_loss=va_loss)
        if int(cfg.early_stop_patience) > 0 and bad_epochs >= int(cfg.early_stop_patience):
            log.info(f"early stop: bad_epochs={bad_epochs} >= patience={int(cfg.early_stop_patience)}")
            break

    if sp_pool is not None:
        sp_pool.close()
        sp_pool.join()

    report = {
        "ok": True,
        "task": "train_way_casd_decoder_joint",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
            "flow_ckpt": str(args.flow_ckpt),
            "way_regions_npz": (str(args.way_regions_npz) if args.way_regions_npz is not None else None),
            "split_json": (str(args.split_json) if args.split_json is not None else None),
        },
        "out_dir": str(out_dir),
        "best_val_loss": float(best_val),
        "best_ckpt": str(out_dir / "ckpt_best.pt"),
        "last_ckpt": str(out_dir / "ckpt_last.pt"),
        "best_epoch": int(best_epoch),
        "cfg": asdict(cfg),
        "decoder_overrides": {
            "force_decoder_use_step_emb": bool(args.force_decoder_use_step_emb),
            "force_decoder_past_k": int(args.force_decoder_past_k),
        },
        "flow_cfg": asdict(flow.cfg),
        "ae_cfg": asdict(ae.cfg),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"saved: {out_dir/'report.json'}")


if __name__ == "__main__":
    main()
