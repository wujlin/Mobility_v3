from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Make `src.*` imports work when running as a script:
#   python tools/flow_z_alignment_probe.py
import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.way_graph.way_sequence_dataset import WayRouteDataset, load_way_routes_npz, make_way_casd_collate_fn
from src.models.way_casd.conditions import ConditionEncoderCfg
from src.models.way_casd.latent_flow import LatentFlowCfg, LatentFlowMatching
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz
from src.training.train_way_casd_flow import (
    RegionSeqLookup,
    _infer_decoder_past_k_from_state,
    _infer_decoder_past_n_layers_from_state,
    _infer_decoder_use_cand_contrast_from_state,
    _infer_decoder_use_cand_query_from_state,
    _infer_decoder_use_cross_attn_from_state,
    _infer_decoder_use_dest_dist_from_state,
    _infer_decoder_use_dest_query_from_state,
    _infer_decoder_use_dir_query_from_state,
    _infer_decoder_use_past_context_from_state,
    _infer_decoder_use_step_emb_from_state,
    _set_seed,
    _subset_indices_from_route_ids,
    _to_device,
)

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _qstats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "n": 0.0}
    return {
        "mean": float(np.mean(x)),
        "p25": float(np.quantile(x, 0.25)),
        "p50": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "n": float(int(x.size)),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Probe per-route alignment between Flow z and GT z_enc.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--flow_ckpt", type=Path, required=True)
    p.add_argument("--split_json", type=Path, required=True)
    p.add_argument("--split_part", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--n_routes", type=int, default=5000)
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--solver_steps", type=int, default=None)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--region_seq_npz", type=Path, default=None)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p


def _build_ae(*, ae_ckpt: Path, way_features_npz: Path, way_graph_npz: Path, device: torch.device) -> WayCASDAutoEncoder:
    ckpt = torch.load(str(ae_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cfg_in = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected AE state format: {type(state)}")
    if not isinstance(cfg_in, dict):
        cfg_in = {}

    way_features = load_way_features_from_npz(way_features_npz)
    wf = np.load(str(way_features_npz), allow_pickle=True)
    wg = np.load(str(way_graph_npz), allow_pickle=True)
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1

    use_dest_dist = _infer_decoder_use_dest_dist_from_state(state)
    use_cand_contrast = (_infer_decoder_use_cand_contrast_from_state(state) if isinstance(state, dict) else False) or bool(
        cfg_in.get("decoder_use_cand_contrast", False)
    )
    use_cross_attn = _infer_decoder_use_cross_attn_from_state(state)
    use_step_emb = _infer_decoder_use_step_emb_from_state(state) or bool(cfg_in.get("decoder_use_step_emb", False))
    use_dest_query = _infer_decoder_use_dest_query_from_state(state) or bool(cfg_in.get("decoder_use_dest_query", False))
    use_dir_query = _infer_decoder_use_dir_query_from_state(state) or bool(cfg_in.get("decoder_use_dir_query", False))
    use_cand_query = _infer_decoder_use_cand_query_from_state(state) or bool(cfg_in.get("decoder_use_cand_query", False))
    use_past_ctx = _infer_decoder_use_past_context_from_state(state) or bool(cfg_in.get("decoder_use_past_context", False))
    past_k = cfg_in.get("decoder_past_k", None)
    if past_k is None:
        past_k = _infer_decoder_past_k_from_state(state)
    past_n_layers = cfg_in.get("decoder_past_n_layers", None)
    if past_n_layers is None:
        past_n_layers = _infer_decoder_past_n_layers_from_state(state)
    past_n_heads = cfg_in.get("decoder_past_n_heads", None)

    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(cfg_in.get("d_model", 256)),
            n_latent=int(cfg_in.get("n_latent", 64)),
            n_heads=int(cfg_in.get("n_heads", 8)),
            dropout=float(cfg_in.get("dropout", 0.1)),
            max_candidates=int(cfg_in.get("max_candidates", 32)),
            max_len=int(cfg_in.get("max_len", 160)),
            coord_scale=float(cfg_in.get("coord_scale", 1024.0)),
            segment_size=int(cfg_in.get("segment_size", 10)),
            segment_n_latent=int(cfg_in.get("segment_n_latent", 0)),
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
    missing, unexpected = ae.load_state_dict(state, strict=False)
    missing_critical = [k for k in missing if str(k).startswith(("way_enc.", "compress."))]
    if missing_critical:
        raise RuntimeError(f"AE load missing critical encoder keys (example={missing_critical[:3]})")
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def _build_flow(*, flow_ckpt: Path, device: torch.device) -> tuple[LatentFlowMatching, dict]:
    ckpt = torch.load(str(flow_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cfg_in = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected Flow state format: {type(state)}")
    if not isinstance(cfg_in, dict):
        cfg_in = {}
    flow = LatentFlowMatching(
        cfg=LatentFlowCfg(
            d_model=int(cfg_in.get("d_model", 256)),
            n_latent=int(cfg_in.get("n_latent", 64)),
            n_layers=int(cfg_in.get("n_layers", 6)),
            n_heads=int(cfg_in.get("n_heads", 8)),
            dropout=float(cfg_in.get("dropout", 0.1)),
            noise_sigma=float(cfg_in.get("noise_sigma", 1.0)),
            solver_steps=int(cfg_in.get("solver_steps", 20)),
            cond_dropout_p=float(cfg_in.get("cond_dropout_p", 0.0)),
            cond_inject=str(cfg_in.get("cond_inject", "add")),
            use_region_seq=bool(cfg_in.get("use_region_seq", False)),
            n_regions=int(cfg_in.get("n_regions", 154)),
            region_max_len=int(cfg_in.get("region_max_len", 16)),
        ),
        cond_cfg=ConditionEncoderCfg(
            d_model=int(cfg_in.get("d_model", 256)),
            coord_scale=1024.0,
            use_time=not bool(cfg_in.get("flow_disable_time_cond", False)),
        ),
    ).to(device)
    missing, unexpected = flow.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Flow ckpt mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    flow.eval()
    return flow, cfg_in


@torch.no_grad()
def main() -> None:
    args = build_argparser().parse_args()
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    _set_seed(int(args.seed))
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    dataset = WayRouteDataset(routes, max_routes=None, max_way_len=int(args.max_way_len), min_hops=int(args.min_hops))
    split = _read_json(Path(args.split_json))
    splits = split.get("splits", split)
    route_ids = np.asarray(splits.get(str(args.split_part), []), dtype=np.int64).reshape(-1)
    sub_idx = _subset_indices_from_route_ids(dataset, route_ids)
    if int(sub_idx.size) == 0:
        raise SystemExit(f"[FATAL] split_part={args.split_part} yielded empty subset")
    if int(args.n_routes) > 0:
        sub_idx = sub_idx[: int(args.n_routes)]
    subset = Subset(dataset, sub_idx.tolist())

    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    collate_fn = make_way_casd_collate_fn(
        way_adj_ptr=wg["way_adj_ptr"],
        way_adj_idx=wg["way_adj_idx"],
        max_candidates=32,
        tz_offset_hours=float(args.tz_offset_hours),
    )
    loader = DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=bool(device.type == "cuda"),
        persistent_workers=bool(int(args.num_workers) > 0),
        prefetch_factor=(4 if int(args.num_workers) > 0 else None),
        collate_fn=collate_fn,
    )

    ae = _build_ae(
        ae_ckpt=Path(args.ae_ckpt),
        way_features_npz=Path(args.way_features_npz),
        way_graph_npz=Path(args.way_graph_npz),
        device=device,
    )
    flow, flow_cfg = _build_flow(flow_ckpt=Path(args.flow_ckpt), device=device)

    region_seq_lookup: Optional[RegionSeqLookup] = None
    if bool(flow_cfg.get("use_region_seq", False)):
        rnpz = Path(args.region_seq_npz) if args.region_seq_npz is not None else Path(str(flow_cfg.get("region_seq_npz", "")))
        if not str(rnpz):
            raise SystemExit("[FATAL] Flow requires region_seq but --region_seq_npz is missing and ckpt config has no path.")
        region_seq_lookup = RegionSeqLookup(region_seq_npz=rnpz)

    cos_vals: list[np.ndarray] = []
    l2_vals: list[np.ndarray] = []
    l2_per_dim_vals: list[np.ndarray] = []
    n_done = 0
    for batch in loader:
        b = _to_device(batch, device, region_seq=region_seq_lookup, need_trans=False)
        z_gt, _ = ae.encode(b["way_seq_pad"])
        z_flow = flow.sample(
            route_cond=b["route_cond"],
            solver_steps=(int(args.solver_steps) if args.solver_steps is not None else None),
            cfg_scale=float(args.cfg_scale),
        )

        flat_gt = z_gt.reshape(int(z_gt.shape[0]), -1).to(dtype=torch.float32)
        flat_flow = z_flow.reshape(int(z_flow.shape[0]), -1).to(dtype=torch.float32)
        cos = F.cosine_similarity(flat_flow, flat_gt, dim=-1)
        diff = flat_flow - flat_gt
        l2 = torch.norm(diff, dim=-1)
        l2_per_dim = torch.sqrt((diff * diff).mean(dim=-1))

        cos_vals.append(cos.detach().cpu().numpy().astype(np.float64, copy=False))
        l2_vals.append(l2.detach().cpu().numpy().astype(np.float64, copy=False))
        l2_per_dim_vals.append(l2_per_dim.detach().cpu().numpy().astype(np.float64, copy=False))
        n_done += int(flat_gt.shape[0])
        if (n_done % 1000) == 0:
            print(f"[probe] done={n_done}/{len(subset)}")

    cos_np = np.concatenate(cos_vals, axis=0) if cos_vals else np.zeros((0,), dtype=np.float64)
    l2_np = np.concatenate(l2_vals, axis=0) if l2_vals else np.zeros((0,), dtype=np.float64)
    l2pd_np = np.concatenate(l2_per_dim_vals, axis=0) if l2_per_dim_vals else np.zeros((0,), dtype=np.float64)

    report = {
        "ok": True,
        "task": "flow_z_alignment_probe",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
            "flow_ckpt": str(args.flow_ckpt),
            "split_json": str(args.split_json),
            "split_part": str(args.split_part),
            "region_seq_npz": (str(args.region_seq_npz) if args.region_seq_npz is not None else None),
        },
        "cfg": {
            "n_routes": int(args.n_routes),
            "batch_size": int(args.batch_size),
            "solver_steps": (int(args.solver_steps) if args.solver_steps is not None else None),
            "cfg_scale": float(args.cfg_scale),
            "flow_cfg": asdict(flow.cfg),
        },
        "n_routes_eval": int(cos_np.size),
        "cosine": _qstats(cos_np),
        "l2": _qstats(l2_np),
        "l2_per_dim": _qstats(l2pd_np),
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved: {out_json}")
    print(
        "Flow-Z Alignment | "
        f"cos_mean={report['cosine']['mean']:.4f} p50={report['cosine']['p50']:.4f} | "
        f"l2_mean={report['l2']['mean']:.4f} l2pd_mean={report['l2_per_dim']['mean']:.4f}"
    )


if __name__ == "__main__":
    main()
