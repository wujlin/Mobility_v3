from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.way_graph.way_sequence_dataset import (
    WayRouteDataset,
    load_way_routes_npz,
    make_way_casd_collate_fn,
)
from src.models.way_casd.latent_flow import LatentFlowCfg, LatentFlowMatching
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz, make_way_feature_tensors
from src.models.way_casd.conditions import ConditionEncoderCfg

TZ_SHANGHAI = timezone(timedelta(hours=8))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
_DEFAULT_D_MODEL = 256
_DEFAULT_N_LAYERS = 6


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
    max_candidates: int
    max_routes: Optional[int]

    d_model: int
    n_latent: int
    n_layers: int
    n_heads: int
    dropout: float
    noise_sigma: float
    solver_steps: int
    cond_dropout_p: float
    cond_inject: str
    use_region_seq: bool
    n_regions: int
    region_max_len: int
    region_seq_npz: Optional[str]
    way_regions_npz: Optional[str]
    split_json: Optional[str] = None


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
    for k in state.keys():
        if str(k).startswith("decoder.cross_attn."):
            return True
    return False


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
    # Keys look like: decoder.past_encoder.transformer.layers.{i}.*
    max_i = -1
    prefix = "decoder.past_encoder.transformer.layers."
    for k in state.keys():
        ks = str(k)
        if not ks.startswith(prefix):
            continue
        rest = ks[len(prefix) :]
        # rest begins with "{i}."
        i_str = rest.split(".", 1)[0]
        try:
            i = int(i_str)
        except Exception:
            continue
        max_i = max(max_i, i)
    if max_i >= 0:
        return int(max_i + 1)
    return None


class RegionSeqLookup:
    """
    Lightweight CSR-backed lookup for region_seq by route_id.

    Expects region_seq_npz produced by src.data.way_graph.extract_region_seq_stats (--out_npz).
    Keys:
      - route_id: (N,) int64
      - region_seq_ptr: (N+1,) int64
      - region_seq_idx: (total,) int32/64
      - region_seq_len: (N,) int32
    """

    def __init__(self, *, region_seq_npz: Path) -> None:
        data = np.load(str(region_seq_npz), allow_pickle=True)
        need = {"route_id", "region_seq_ptr", "region_seq_idx", "region_seq_len"}
        missing = sorted(list(need - set(data.files)))
        if missing:
            raise ValueError(f"region_seq_npz missing keys: {missing}")
        self.route_id = np.asarray(data["route_id"], dtype=np.int64).reshape(-1)
        self.ptr = np.asarray(data["region_seq_ptr"], dtype=np.int64).reshape(-1)
        self.idx = np.asarray(data["region_seq_idx"], dtype=np.int64).reshape(-1)
        self.len = np.asarray(data["region_seq_len"], dtype=np.int64).reshape(-1)

        rid_max = int(np.max(self.route_id)) if self.route_id.size else -1
        self.rid_to_row = np.full((rid_max + 1,), -1, dtype=np.int64)
        for row, rid in enumerate(self.route_id.tolist()):
            rr = int(rid)
            if 0 <= rr < self.rid_to_row.size:
                self.rid_to_row[rr] = np.int64(row)

        meta = data["meta"] if "meta" in data.files else None
        self.meta = meta.item() if meta is not None else None

    def padded(self, *, route_id: torch.Tensor) -> torch.Tensor:
        rid = route_id.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        if rid.size == 0:
            return torch.zeros((0, 1), dtype=torch.long)
        if int(np.max(rid)) >= int(self.rid_to_row.size):
            raise KeyError("region_seq lookup route_id out of range (region_seq_npz was built from a different routes set).")
        row = self.rid_to_row[rid]
        if int(np.any(row < 0)):
            bad = int(rid[int(np.where(row < 0)[0][0])])
            raise KeyError(f"region_seq missing for route_id={bad} (filter mismatch?)")

        lens = self.len[row].astype(np.int64, copy=False)
        maxL = int(np.max(lens)) if lens.size else 1
        maxL = max(1, int(maxL))
        pad = np.full((int(rid.size), int(maxL)), -1, dtype=np.int64)
        for i, r in enumerate(row.tolist()):
            L = int(lens[i])
            s = int(self.ptr[int(r)])
            pad[int(i), :L] = self.idx[s : s + L]
        return torch.as_tensor(pad, dtype=torch.long)


def _to_device(batch: Dict[str, object], device: torch.device, *, region_seq: Optional[RegionSeqLookup] = None) -> Dict[str, object]:
    way_seq_pad = batch["way_seq_pad"].to(device)
    route_cond = {k: v.to(device) for k, v in batch["route_cond"].items()}
    if region_seq is not None:
        rpad = region_seq.padded(route_id=batch["route_id"])  # (B,S) long, -1 padded
        route_cond["region_seq_pad"] = rpad.to(device)
    # Flow training only needs encoder latents + route_cond; moving transitions to GPU is wasted.
    return {"way_seq_pad": way_seq_pad, "route_cond": route_cond}


def train_epoch(
    *,
    encoder: WayCASDAutoEncoder,
    flow: LatentFlowMatching,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    region_seq: Optional[RegionSeqLookup] = None,
) -> Dict[str, float]:
    flow.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        b = _to_device(batch, device, region_seq=region_seq)
        with torch.no_grad():
            z1, _ = encoder.encode(b["way_seq_pad"])
        opt.zero_grad(set_to_none=True)
        loss, _stats = flow.compute_loss(z1=z1, route_cond=b["route_cond"])
        loss.backward()
        opt.step()
        total_loss += float(loss.item())
        total_batches += 1

    denom = max(1, int(total_batches))
    return {"loss": float(total_loss / denom)}


@torch.no_grad()
def eval_epoch(
    *,
    encoder: WayCASDAutoEncoder,
    flow: LatentFlowMatching,
    loader: DataLoader,
    device: torch.device,
    region_seq: Optional[RegionSeqLookup] = None,
) -> Dict[str, float]:
    flow.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        b = _to_device(batch, device, region_seq=region_seq)
        z1, _ = encoder.encode(b["way_seq_pad"])
        loss, _stats = flow.compute_loss(z1=z1, route_cond=b["route_cond"])
        total_loss += float(loss.item())
        total_batches += 1
    denom = max(1, int(total_batches))
    return {"loss": float(total_loss / denom)}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Way-CASD latent flow (Step B).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--min_hops", type=int, default=1, help="Filter routes with fewer than this many way transitions (hops).")
    p.add_argument("--max_way_len", type=int, default=128)
    p.add_argument("--max_candidates", type=int, default=32)
    p.add_argument("--max_routes", type=int, default=None, help="Debug: cap number of routes (after filtering).")
    p.add_argument(
        "--split_json",
        type=Path,
        default=None,
        help="Optional OD-disjoint split json (expects splits.train/val/test route_ids). Overrides val_ratio.",
    )
    p.add_argument(
        "--split_part",
        choices=["train", "val", "test"],
        default=None,
        help="Compatibility arg (ignored in training). train_way_casd_flow always uses split train+val.",
    )

    p.add_argument("--d_model", type=int, default=_DEFAULT_D_MODEL)
    p.add_argument("--n_latent", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=_DEFAULT_N_LAYERS)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--noise_sigma", type=float, default=1.0)
    p.add_argument("--solver_steps", type=int, default=20)
    p.add_argument("--cond_dropout_p", type=float, default=0.0, help="CFG training: probability to drop conditions (0=disable).")
    p.add_argument("--cond_inject", type=str, default="add", choices=["add", "xattn"], help="How to inject conditions into latent tokens.")
    p.add_argument("--use_region_seq", action="store_true", help="If set, condition Flow on coarse region_seq (from --region_seq_npz).")
    p.add_argument("--region_seq_npz", type=Path, default=None, help="region_seq_min*.npz from extract_region_seq_stats.py (required when --use_region_seq).")
    p.add_argument("--way_regions_npz", type=Path, default=None, help="way_regions_louvain_per_city*.npz (to infer n_regions; recommended).")
    p.add_argument("--region_max_len", type=int, default=16, help="Max positional embedding length for region_seq tokens.")

    # Backward-compat aliases (some wrappers still use these names).
    p.add_argument("--flow_n_layers", type=int, default=None, help="Deprecated alias for --n_layers.")
    p.add_argument("--flow_n_hidden", type=int, default=None, help="Deprecated alias for --d_model.")

    # Long-run training ergonomics
    p.add_argument("--resume_ckpt", type=Path, default=None, help="Optional: resume from ckpt_last.pt/ckpt_best.pt.")
    p.add_argument("--resume_epoch", type=int, default=None, help="Optional: override resume epoch (when ckpt has no epoch).")
    p.add_argument("--save_every", type=int, default=20, help="Save ckpt_last.pt every N epochs (best ckpt saved on improve).")
    p.add_argument("--early_stop_patience", type=int, default=0, help="Optional early stop (0=disable).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize deprecated alias args.
    if args.flow_n_layers is not None:
        if int(args.n_layers) != int(_DEFAULT_N_LAYERS) and int(args.n_layers) != int(args.flow_n_layers):
            raise SystemExit("[FATAL] Both --n_layers and --flow_n_layers are set with different values.")
        log.warning("DEPRECATED arg: --flow_n_layers. Please use --n_layers instead.")
        args.n_layers = int(args.flow_n_layers)
    if args.flow_n_hidden is not None:
        if int(args.d_model) != int(_DEFAULT_D_MODEL) and int(args.d_model) != int(args.flow_n_hidden):
            raise SystemExit("[FATAL] Both --d_model and --flow_n_hidden are set with different values.")
        log.warning("DEPRECATED arg: --flow_n_hidden. Please use --d_model instead.")
        args.d_model = int(args.flow_n_hidden)

    use_region_seq = bool(args.use_region_seq)
    region_seq_npz = Path(args.region_seq_npz) if args.region_seq_npz is not None else None
    way_regions_npz = Path(args.way_regions_npz) if args.way_regions_npz is not None else None
    cond_inject = str(args.cond_inject)
    region_max_len = int(args.region_max_len)

    n_regions = 154
    if use_region_seq:
        if region_seq_npz is None:
            raise SystemExit("[FATAL] --use_region_seq requires --region_seq_npz.")
        if way_regions_npz is not None:
            reg = np.load(str(way_regions_npz), allow_pickle=True)
            meta = reg["meta"] if "meta" in reg.files else None
            meta_dict = None
            if meta is not None:
                try:
                    meta_dict = meta.item()
                except Exception:
                    meta_dict = None
            if isinstance(meta_dict, dict) and "n_regions" in meta_dict:
                n_regions = int(meta_dict["n_regions"])
            elif "way_region" in reg.files:
                wr = np.asarray(reg["way_region"], dtype=np.int64).reshape(-1)
                n_regions = int(np.max(wr)) + 1 if wr.size else 1
        else:
            log.warning("use_region_seq enabled but --way_regions_npz is missing; inferring n_regions from region_seq_npz (may undercount).")
            rs = np.load(str(region_seq_npz), allow_pickle=True)
            if "region_seq_idx" not in rs.files:
                raise SystemExit("[FATAL] region_seq_npz missing key: region_seq_idx")
            ridx = np.asarray(rs["region_seq_idx"], dtype=np.int64).reshape(-1)
            ridx = ridx[ridx >= 0]
            n_regions = int(np.max(ridx)) + 1 if ridx.size else 1

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
        max_candidates=int(args.max_candidates),
        max_routes=(int(args.max_routes) if args.max_routes is not None else None),
        d_model=int(args.d_model),
        n_latent=int(args.n_latent),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        dropout=float(args.dropout),
        noise_sigma=float(args.noise_sigma),
        solver_steps=int(args.solver_steps),
        cond_dropout_p=float(args.cond_dropout_p),
        cond_inject=str(cond_inject),
        use_region_seq=bool(use_region_seq),
        n_regions=int(n_regions),
        region_max_len=int(region_max_len),
        region_seq_npz=(str(region_seq_npz) if region_seq_npz is not None else None),
        way_regions_npz=(str(way_regions_npz) if way_regions_npz is not None else None),
        split_json=(str(args.split_json) if args.split_json is not None else None),
    )

    if args.split_part is not None:
        log.warning("--split_part=%s is ignored by train_way_casd_flow (train uses split train+val).", str(args.split_part))

    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    log.info(f"device={device}")

    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    way_features = load_way_features_from_npz(Path(args.way_features_npz))
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1
    n_semantic = way_features.way_semantic.shape[-1] if way_features.way_semantic is not None else 0
    if n_semantic > 0:
        log.info(f"loaded way_semantic: n_channels={n_semantic}")

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
        train_ids, val_ids = _split_dataset(len(dataset), cfg.val_ratio, cfg.seed)
    else:
        split = _read_json(Path(args.split_json))
        splits = split.get("splits", split)
        tr_rids = np.asarray(splits.get("train", []), dtype=np.int64).reshape(-1)
        va_rids = np.asarray(splits.get("val", []), dtype=np.int64).reshape(-1)
        train_ids = _subset_indices_from_route_ids(dataset, tr_rids)
        val_ids = _subset_indices_from_route_ids(dataset, va_rids)
        if int(train_ids.size) == 0 or int(val_ids.size) == 0:
            raise SystemExit(
                f"[FATAL] split_json produced empty subsets: train_idx={int(train_ids.size)} val_idx={int(val_ids.size)}. "
                "Check min_hops/max_way_len match split generation."
            )
        log.info(
            f"split_json={args.split_json} train_routes={int(tr_rids.size)} val_routes={int(va_rids.size)} "
            f"=> train_idx={int(train_ids.size)} val_idx={int(val_ids.size)}"
        )
    train_set = Subset(dataset, train_ids.tolist())
    val_set = Subset(dataset, val_ids.tolist())
    log.info(
        f"routes: total={len(dataset)} train={len(train_set)} val={len(val_set)} min_hops={cfg.min_hops} max_way_len={cfg.max_way_len}"
    )

    collate_fn = make_way_casd_collate_fn(
        way_adj_ptr=wg["way_adj_ptr"],
        way_adj_idx=wg["way_adj_idx"],
        max_candidates=int(cfg.max_candidates),
        tz_offset_hours=float(cfg.tz_offset_hours),
    )

    region_seq_lookup: Optional[RegionSeqLookup] = None
    if bool(cfg.use_region_seq):
        if region_seq_npz is None:
            raise SystemExit("[FATAL] use_region_seq enabled but region_seq_npz is missing.")
        region_seq_lookup = RegionSeqLookup(region_seq_npz=Path(region_seq_npz))
        if isinstance(region_seq_lookup.meta, dict) and isinstance(region_seq_lookup.meta.get("cfg", None), dict):
            mc = region_seq_lookup.meta["cfg"]
            if int(mc.get("min_hops", cfg.min_hops)) != int(cfg.min_hops) or int(mc.get("max_way_len", cfg.max_way_len)) != int(cfg.max_way_len):
                log.warning(
                    f"region_seq_npz filter mismatch: meta(min_hops={mc.get('min_hops')}, max_way_len={mc.get('max_way_len')}) "
                    f"vs train(min_hops={cfg.min_hops}, max_way_len={cfg.max_way_len})"
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

    ckpt = torch.load(str(args.ae_ckpt), map_location=device)
    ae_state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ae_cfg_dict: Dict[str, object] = {}
    if isinstance(ckpt, dict) and isinstance(ckpt.get("config", None), dict):
        ae_cfg_dict = ckpt["config"]  # type: ignore[assignment]

    if not isinstance(ae_state, dict):
        raise TypeError(f"Unexpected AE state format: {type(ae_state)}")

    use_dest_dist = _infer_decoder_use_dest_dist_from_state(ae_state)
    use_cand_contrast = (_infer_decoder_use_cand_contrast_from_state(ae_state) if isinstance(ae_state, dict) else False) or bool(
        ae_cfg_dict.get("decoder_use_cand_contrast", False)
    )
    use_cross_attn = _infer_decoder_use_cross_attn_from_state(ae_state)
    use_step_emb = _infer_decoder_use_step_emb_from_state(ae_state) or bool(ae_cfg_dict.get("decoder_use_step_emb", False))
    use_dest_query = _infer_decoder_use_dest_query_from_state(ae_state) or bool(ae_cfg_dict.get("decoder_use_dest_query", False))
    use_dir_query = _infer_decoder_use_dir_query_from_state(ae_state) or bool(ae_cfg_dict.get("decoder_use_dir_query", False))
    use_cand_query = _infer_decoder_use_cand_query_from_state(ae_state) or bool(ae_cfg_dict.get("decoder_use_cand_query", False))
    use_past_ctx = _infer_decoder_use_past_context_from_state(ae_state) or bool(ae_cfg_dict.get("decoder_use_past_context", False))
    past_k = ae_cfg_dict.get("decoder_past_k", None)
    if past_k is None:
        past_k = _infer_decoder_past_k_from_state(ae_state)
    past_n_layers = ae_cfg_dict.get("decoder_past_n_layers", None)
    if past_n_layers is None:
        past_n_layers = _infer_decoder_past_n_layers_from_state(ae_state)
    past_n_heads = ae_cfg_dict.get("decoder_past_n_heads", None)

    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(ae_cfg_dict.get("d_model", cfg.d_model)),
            n_latent=int(ae_cfg_dict.get("n_latent", cfg.n_latent)),
            n_heads=int(ae_cfg_dict.get("n_heads", cfg.n_heads)),
            dropout=float(ae_cfg_dict.get("dropout", cfg.dropout)),
            max_candidates=int(ae_cfg_dict.get("max_candidates", cfg.max_candidates)),
            max_len=int(ae_cfg_dict.get("max_len", cfg.max_way_len)),
            coord_scale=float(ae_cfg_dict.get("coord_scale", 1024.0)),
            segment_size=int(ae_cfg_dict.get("segment_size", 10)),
            segment_n_latent=int(ae_cfg_dict.get("segment_n_latent", 0)),
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
        ae.load_state_dict(ae_state, strict=True)
    except RuntimeError as e:
        # Most common reason: training checkpoint contains decoder-only keys (e.g., past_context encoder)
        # while Flow training only needs the encoder. Fall back to a non-strict load to unblock training.
        log.warning(f"AE strict load failed; falling back to strict=False. err={e}")
        missing, unexpected = ae.load_state_dict(ae_state, strict=False)
        # Flow training depends on the encoder (way_enc + perceiver compressor). If those are missing,
        # training would silently proceed with wrong weights, which is unacceptable.
        missing_critical = [k for k in missing if str(k).startswith(("way_enc.", "compress."))]
        if missing_critical:
            raise RuntimeError(f"AE load missing critical encoder keys (example={missing_critical[:3]})")
        if missing:
            log.warning(f"AE non-strict load: missing={len(missing)} (example={missing[:3]})")
        if unexpected:
            log.warning(f"AE non-strict load: unexpected={len(unexpected)} (example={unexpected[:3]})")
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    flow = LatentFlowMatching(
        cfg=LatentFlowCfg(
            d_model=int(cfg.d_model),
            n_latent=int(cfg.n_latent),
            n_layers=int(cfg.n_layers),
            n_heads=int(cfg.n_heads),
            dropout=float(cfg.dropout),
            noise_sigma=float(cfg.noise_sigma),
            solver_steps=int(cfg.solver_steps),
            cond_dropout_p=float(cfg.cond_dropout_p),
            cond_inject=str(cfg.cond_inject),
            use_region_seq=bool(cfg.use_region_seq),
            n_regions=int(cfg.n_regions),
            region_max_len=int(cfg.region_max_len),
        ),
        cond_cfg=ConditionEncoderCfg(d_model=int(cfg.d_model), coord_scale=1024.0),
    ).to(device)

    opt = torch.optim.AdamW(flow.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    best = float("inf")
    best_epoch = 0
    best_path = out_dir / "ckpt_best.pt"
    last_path = out_dir / "ckpt_last.pt"
    progress_path = out_dir / "progress.json"
    hist_path = out_dir / "history.jsonl"

    start_epoch = 1
    history = []
    patience = 0

    if args.resume_ckpt is not None:
        ckpt = torch.load(str(args.resume_ckpt), map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            missing, unexpected = flow.load_state_dict(ckpt["model_state_dict"], strict=False)
            if missing or unexpected:
                log.warning(f"resume: flow state mismatch: missing={len(missing)} unexpected={len(unexpected)}")
            if "opt_state_dict" in ckpt:
                try:
                    opt.load_state_dict(ckpt["opt_state_dict"])
                except Exception as e:  # pragma: no cover
                    log.warning(f"resume: failed to load optimizer state (ignored): {e}")
            if "best_val_loss" in ckpt:
                try:
                    best = float(ckpt["best_val_loss"])
                except Exception:
                    best = float("inf")
            if "best_epoch" in ckpt:
                try:
                    best_epoch = int(ckpt["best_epoch"])
                except Exception:
                    best_epoch = 0
            if "epoch" in ckpt:
                try:
                    start_epoch = int(ckpt["epoch"]) + 1
                except Exception:
                    start_epoch = 1
        elif isinstance(ckpt, dict):
            missing, unexpected = flow.load_state_dict(ckpt, strict=False)
            if missing or unexpected:
                log.warning(f"resume: flow state mismatch: missing={len(missing)} unexpected={len(unexpected)}")

        if args.resume_epoch is not None:
            start_epoch = int(args.resume_epoch) + 1

        log.info(f"resume_ckpt={args.resume_ckpt} start_epoch={start_epoch} best_val_loss={best} best_epoch={best_epoch}")

    if not np.isfinite(best) or best == float("inf"):
        va0 = eval_epoch(encoder=ae, flow=flow, loader=val_loader, device=device, region_seq=region_seq_lookup)
        best = float(va0["loss"])
        best_epoch = int(start_epoch - 1)
        log.info(f"init best_val_loss={best:.6f} from current weights (epoch={best_epoch})")
        # Ensure ckpt_best exists even if training is interrupted or val never improves.
        torch.save(
            {
                "model_state_dict": flow.state_dict(),
                "config": asdict(cfg),
                "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
                "epoch": int(best_epoch),
                "best_val_loss": float(best),
                "best_epoch": int(best_epoch),
            },
            str(best_path),
        )

    save_every = max(1, int(args.save_every))
    early_stop_patience = max(0, int(args.early_stop_patience))

    for epoch in range(int(start_epoch), int(cfg.n_epochs) + 1):
        tr = train_epoch(encoder=ae, flow=flow, loader=train_loader, opt=opt, device=device, region_seq=region_seq_lookup)
        va = eval_epoch(encoder=ae, flow=flow, loader=val_loader, device=device, region_seq=region_seq_lookup)
        history.append({"epoch": int(epoch), "train": tr, "val": va})
        log.info(f"epoch={epoch} train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f}")

        with hist_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": int(epoch), "train_loss": float(tr["loss"]), "val_loss": float(va["loss"])}) + "\n")

        if float(va["loss"]) < float(best):
            best = float(va["loss"])
            best_epoch = int(epoch)
            patience = 0
            torch.save(
                {
                    "model_state_dict": flow.state_dict(),
                    "config": asdict(cfg),
                    "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
                    "epoch": int(epoch),
                    "best_val_loss": float(best),
                    "best_epoch": int(best_epoch),
                },
                str(best_path),
            )
        else:
            patience += 1

        if (int(epoch) % save_every) == 0:
            torch.save(
                {
                    "model_state_dict": flow.state_dict(),
                    "opt_state_dict": opt.state_dict(),
                    "config": asdict(cfg),
                    "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
                    "epoch": int(epoch),
                    "best_val_loss": float(best),
                    "best_epoch": int(best_epoch),
                },
                str(last_path),
            )

        progress = {
            "ok": True,
            "task": "train_way_casd_flow",
            "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
            "epoch": int(epoch),
            "train_loss": float(tr["loss"]),
            "val_loss": float(va["loss"]),
            "best_val_loss": float(best),
            "best_epoch": int(best_epoch),
            "save_every": int(save_every),
            "early_stop_patience": int(early_stop_patience),
        }
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

        if early_stop_patience > 0 and patience >= early_stop_patience:
            log.info(f"early_stop: patience={patience} reached (best_epoch={best_epoch} best_val_loss={best:.6f})")
            break

    torch.save(
        {
            "model_state_dict": flow.state_dict(),
            "opt_state_dict": opt.state_dict(),
            "config": asdict(cfg),
            "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
            "epoch": int(history[-1]["epoch"]) if history else int(start_epoch - 1),
            "best_val_loss": float(best),
            "best_epoch": int(best_epoch),
        },
        str(last_path),
    )
    if not best_path.exists() and last_path.exists():
        # Safety: some runs might never improve over init; keep evaluation scripts unblocked.
        best_path.write_bytes(last_path.read_bytes())

    report = {
        "ok": True,
        "task": "train_way_casd_flow",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
            "region_seq_npz": (str(args.region_seq_npz) if args.region_seq_npz is not None else None),
            "way_regions_npz": (str(args.way_regions_npz) if args.way_regions_npz is not None else None),
            "split_json": (str(args.split_json) if args.split_json is not None else None),
        },
        "out_dir": str(out_dir),
        "best_val_loss": float(best),
        "best_ckpt": str(best_path),
        "last_ckpt": str(last_path),
        "best_epoch": int(best_epoch),
        "start_epoch": int(start_epoch),
        "early_stop_patience": int(early_stop_patience),
        "save_every": int(save_every),
        "history": history,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"saved: {out_dir/'report.json'}")


if __name__ == "__main__":
    main()
