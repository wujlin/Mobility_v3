from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Allow running as:
#   python tools/flow_z_multimodality_probe.py ...
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.way_graph.way_sequence_dataset import (  # noqa: E402
    WayRouteDataset,
    load_way_routes_npz,
    make_way_casd_collate_fn,
)
from src.models.way_casd.conditions import ConditionEncoderCfg  # noqa: E402
from src.models.way_casd.latent_flow import LatentFlowCfg, LatentFlowMatching  # noqa: E402
from src.training.train_way_casd_flow import RegionSeqLookup, _read_json, _set_seed, _subset_indices_from_route_ids, _to_device  # noqa: E402

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _qstats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0.0, "mean": float("nan"), "p25": float("nan"), "p50": float("nan"), "p75": float("nan"), "p90": float("nan")}
    return {
        "n": float(int(x.size)),
        "mean": float(np.mean(x)),
        "p25": float(np.quantile(x, 0.25)),
        "p50": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
    }


def _build_flow(*, flow_ckpt: Path, device: torch.device) -> Tuple[LatentFlowMatching, Dict[str, object]]:
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
        cond_cfg=ConditionEncoderCfg(d_model=int(cfg_in.get("d_model", 256)), coord_scale=1024.0),
    ).to(device)
    missing, unexpected = flow.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Flow ckpt mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    flow.eval()
    return flow, cfg_in


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Probe Flow z multimodality by pairwise cosine among K samples per route.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--flow_ckpt", type=Path, required=True)
    p.add_argument("--split_json", type=Path, required=True)
    p.add_argument("--split_part", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--n_routes", type=int, default=5000)
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--solver_steps", type=int, default=None)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--n_samples_per_route", type=int, default=16)
    p.add_argument("--region_seq_npz", type=Path, default=None)
    p.add_argument("--log_every_batches", type=int, default=5)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_per_route_json", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p


@torch.no_grad()
def main() -> None:
    args = build_argparser().parse_args()
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    _set_seed(int(args.seed))
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    K = max(2, int(args.n_samples_per_route))

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

    flow, flow_cfg = _build_flow(flow_ckpt=Path(args.flow_ckpt), device=device)
    region_seq_lookup: Optional[RegionSeqLookup] = None
    if bool(flow_cfg.get("use_region_seq", False)):
        rnpz = Path(args.region_seq_npz) if args.region_seq_npz is not None else Path(str(flow_cfg.get("region_seq_npz", "")))
        if not str(rnpz):
            raise SystemExit("[FATAL] Flow requires region_seq but --region_seq_npz is missing and ckpt config has no path.")
        region_seq_lookup = RegionSeqLookup(region_seq_npz=rnpz)

    tri = torch.triu_indices(K, K, offset=1, device=device)
    all_pair_vals: List[np.ndarray] = []
    all_route_mean: List[float] = []
    per_route_rows: List[Dict[str, object]] = []
    od_bucket: Dict[Tuple[int, int, int], List[float]] = {}
    n_done = 0
    total = int(len(subset))
    n_batches = int((total + int(args.batch_size) - 1) // int(args.batch_size))

    for bi, batch in enumerate(loader, start=1):
        b = _to_device(batch, device, region_seq=region_seq_lookup, need_trans=False)
        cond = b["route_cond"]
        B = int(cond["start_pos"].shape[0])

        cond_rep: Dict[str, torch.Tensor] = {}
        for k, v in cond.items():
            cond_rep[k] = v.repeat_interleave(K, dim=0)

        z = flow.sample(
            route_cond=cond_rep,
            solver_steps=(int(args.solver_steps) if args.solver_steps is not None else None),
            cfg_scale=float(args.cfg_scale),
        )  # (B*K, L, D)
        z = z.reshape(B, K, -1).to(dtype=torch.float32)  # (B, K, LD)
        z = F.normalize(z, p=2, dim=-1, eps=1e-8)
        sim = torch.bmm(z, z.transpose(1, 2))  # (B, K, K)
        vals = sim[:, tri[0], tri[1]]  # (B, K*(K-1)/2)
        vals_np = vals.detach().cpu().numpy().astype(np.float32, copy=False)
        mean_np = np.mean(vals_np, axis=1, dtype=np.float64)
        p50_np = np.quantile(vals_np, 0.5, axis=1)
        p90_np = np.quantile(vals_np, 0.9, axis=1)
        min_np = np.min(vals_np, axis=1)
        max_np = np.max(vals_np, axis=1)

        route_ids_b = batch["route_id"].detach().cpu().numpy().astype(np.int64, copy=False)
        city_b = batch["route_cond"]["route_city"].detach().cpu().numpy().astype(np.int64, copy=False)
        sw_b = batch["route_cond"]["start_way"].detach().cpu().numpy().astype(np.int64, copy=False)
        dw_b = batch["route_cond"]["dest_way"].detach().cpu().numpy().astype(np.int64, copy=False)

        all_pair_vals.append(vals_np.reshape(-1).astype(np.float64, copy=False))
        all_route_mean.extend(mean_np.tolist())

        for i in range(B):
            key = (int(city_b[i]), int(sw_b[i]), int(dw_b[i]))
            od_bucket.setdefault(key, []).append(float(mean_np[i]))
            row = {
                "route_id": int(route_ids_b[i]),
                "city": int(city_b[i]),
                "start_way": int(sw_b[i]),
                "dest_way": int(dw_b[i]),
                "pairwise_cos_mean": float(mean_np[i]),
                "pairwise_cos_p50": float(p50_np[i]),
                "pairwise_cos_p90": float(p90_np[i]),
                "pairwise_cos_min": float(min_np[i]),
                "pairwise_cos_max": float(max_np[i]),
            }
            per_route_rows.append(row)

        n_done += B
        if int(args.log_every_batches) > 0 and (bi % int(args.log_every_batches) == 0 or bi == n_batches):
            global_mean = float(np.mean(np.asarray(all_route_mean, dtype=np.float64))) if all_route_mean else float("nan")
            collapse_rate_095 = float(
                np.mean((np.asarray(all_route_mean, dtype=np.float64) > 0.95).astype(np.float32))
            ) if all_route_mean else float("nan")
            print(
                f"[probe] batch {bi}/{n_batches} routes {n_done}/{total} "
                f"route_mean_cos={global_mean:.4f} collapse@0.95={collapse_rate_095:.4f}",
                flush=True,
            )

    all_pair = np.concatenate(all_pair_vals, axis=0) if all_pair_vals else np.zeros((0,), dtype=np.float64)
    route_mean_np = np.asarray(all_route_mean, dtype=np.float64)
    od_means = np.asarray([float(np.mean(v)) for v in od_bucket.values() if len(v) > 0], dtype=np.float64)

    rep: Dict[str, object] = {
        "ok": True,
        "task": "flow_z_multimodality_probe",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": {
            "seed": int(args.seed),
            "device": str(device),
            "split_part": str(args.split_part),
            "n_routes": int(total),
            "min_hops": int(args.min_hops),
            "max_way_len": int(args.max_way_len),
            "batch_size": int(args.batch_size),
            "n_samples_per_route": int(K),
            "solver_steps": (int(args.solver_steps) if args.solver_steps is not None else None),
            "cfg_scale": float(args.cfg_scale),
        },
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "flow_ckpt": str(args.flow_ckpt),
            "split_json": str(args.split_json),
            "region_seq_npz": (str(args.region_seq_npz) if args.region_seq_npz is not None else None),
        },
        "summary": {
            "pairwise_cos_all_samples": _qstats(all_pair),
            "route_pairwise_cos_mean": _qstats(route_mean_np),
            "od_pairwise_cos_mean": _qstats(od_means),
            "route_collapse_rate_cos_gt_0p95": float(np.mean((route_mean_np > 0.95).astype(np.float32))) if route_mean_np.size else float("nan"),
            "route_collapse_rate_cos_gt_0p90": float(np.mean((route_mean_np > 0.90).astype(np.float32))) if route_mean_np.size else float("nan"),
            "n_od": int(len(od_bucket)),
        },
    }

    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved: {out_json}", flush=True)

    if args.out_per_route_json is not None:
        out_pr = Path(args.out_per_route_json)
        out_pr.parent.mkdir(parents=True, exist_ok=True)
        out_pr.write_text(
            json.dumps(
                {
                    "ok": True,
                    "task": "flow_z_multimodality_probe_per_route",
                    "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
                    "cfg": rep["cfg"],
                    "rows": per_route_rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[OK] saved per-route: {out_pr}", flush=True)

    s = rep["summary"]
    route_mean = s["route_pairwise_cos_mean"]
    print(
        "Flow-z multimodality | "
        f"route_mean_cos={float(route_mean['mean']):.4f} "
        f"(p50={float(route_mean['p50']):.4f}, p90={float(route_mean['p90']):.4f}) | "
        f"collapse@0.95={float(s['route_collapse_rate_cos_gt_0p95']):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
