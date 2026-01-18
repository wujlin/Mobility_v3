from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.models.way_casd.latent_flow import LatentFlowCfg, LatentFlowMatching
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import make_way_feature_tensors

TZ_SHANGHAI = timezone(timedelta(hours=8))

LatentSource = Literal["gt", "flow"]
DecodeMethod = Literal["greedy", "beam"]


@dataclass(frozen=True)
class EvalCfg:
    seed: int
    device: str
    tz_offset_hours: float

    n_routes: int  # per city
    n_samples_per_route: int
    max_way_len: int
    max_decode_len: int
    beam_size: int
    solver_steps: Optional[int]


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def _quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, float(q)))


def _summary_stats(x: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(list(x), dtype=np.float64)
    return {
        "mean": float(np.mean(a)) if a.size else float("nan"),
        "p50": _quantile(a, 0.50),
        "p90": _quantile(a, 0.90),
        "p99": _quantile(a, 0.99),
    }


def _jaccard(a: List[int], b: List[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb)) / float(len(sa | sb))


def _len_buckets() -> List[Tuple[str, int, int]]:
    # inclusive ranges [lo, hi]
    return [
        ("L<=15", 0, 15),
        ("16-30", 16, 30),
        ("31-60", 31, 60),
        ("61-128", 61, 128),
    ]


def _bucket_key(L: int) -> str:
    for name, lo, hi in _len_buckets():
        if int(lo) <= int(L) <= int(hi):
            return str(name)
    return "other"


def _decode_one(
    *,
    ae: WayCASDAutoEncoder,
    z: torch.Tensor,  # (B,L,d)
    route_cond: Dict[str, torch.Tensor],  # (B,*)
    start_way: torch.Tensor,  # (B,)
    dest_way: torch.Tensor,  # (B,)
    method: DecodeMethod,
    beam_size: int,
    max_decode_len: int,
) -> List[List[int]]:
    if str(method) == "beam":
        return ae.decoder.beam_search(
            way_embedder=ae.way_enc,
            latent_tokens=z,
            route_cond=route_cond,
            start_way=start_way,
            dest_way=dest_way,
            beam_size=int(beam_size),
            max_len=int(max_decode_len),
        )
    return ae.decoder.greedy_decode(
        way_embedder=ae.way_enc,
        latent_tokens=z,
        route_cond=route_cond,
        start_way=start_way,
        dest_way=dest_way,
        max_len=int(max_decode_len),
    )


@torch.no_grad()
def _eval_city(
    *,
    city: int,
    cfg: EvalCfg,
    routes_npz: Path,
    ae: WayCASDAutoEncoder,
    flow: Optional[LatentFlowMatching],
    latent_sources: Sequence[LatentSource],
    decode_methods: Sequence[DecodeMethod],
    device: torch.device,
    log_every: int = 50,
) -> Dict[str, object]:
    routes = load_way_routes_npz(Path(routes_npz))
    N = int(routes.way_seq_len.shape[0])
    keep = (routes.route_city.astype(np.int64) == int(city)) & (routes.way_seq_len > 1) & (routes.way_seq_len <= int(cfg.max_way_len))
    ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
    if ids.size == 0:
        return {"city": int(city), "n_candidates": 0, "results": {}}

    rng = np.random.default_rng(int(cfg.seed) + 101 * int(city))
    rng.shuffle(ids)
    pick = ids[: min(int(cfg.n_routes), int(ids.size))]

    # Preload per-route meta
    start_t = routes.start_t[pick].astype(np.int64, copy=False)
    hour = _hour_from_unix(start_t, float(cfg.tz_offset_hours))
    dow = _dow_from_unix(start_t, float(cfg.tz_offset_hours))
    start_pos = routes.start_pos[pick].astype(np.float32, copy=False)
    dest_pos = routes.dest_pos[pick].astype(np.float32, copy=False)
    start_way = routes.start_way[pick].astype(np.int64, copy=False)
    dest_way = routes.dest_way[pick].astype(np.int64, copy=False)
    gt_len = routes.way_seq_len[pick].astype(np.int64, copy=False)

    # Preload gt sequences
    gt_seqs: List[List[int]] = []
    for rid, L in zip(pick.tolist(), gt_len.tolist()):
        s = int(routes.way_seq_ptr[int(rid)])
        e = s + int(L)
        gt = routes.way_seq_idx[s:e].astype(np.int64, copy=False).tolist()
        gt_seqs.append([int(x) for x in gt])

    # Batched tensors for cond
    base_route_cond = {
        "start_pos": torch.as_tensor(start_pos, dtype=torch.float32, device=device),
        "dest_pos": torch.as_tensor(dest_pos, dtype=torch.float32, device=device),
        "hour": torch.as_tensor(hour, dtype=torch.long, device=device),
        "dow": torch.as_tensor(dow, dtype=torch.long, device=device),
        "route_city": torch.full((int(pick.size),), int(city), dtype=torch.long, device=device),
    }
    start_way_t = torch.as_tensor(start_way, dtype=torch.long, device=device)
    dest_way_t = torch.as_tensor(dest_way, dtype=torch.long, device=device)

    # Pad GT for encoder
    B = int(pick.size)
    maxL = int(gt_len.max())
    pad = np.full((B, maxL), -1, dtype=np.int64)
    for i, seq in enumerate(gt_seqs):
        Li = min(int(len(seq)), int(maxL))
        pad[i, :Li] = np.asarray(seq[:Li], dtype=np.int64)
    way_pad = torch.as_tensor(pad, dtype=torch.long, device=device)

    # z_enc: (B,L_lat,d)
    z_enc, _ = ae.encode(way_pad)

    results: Dict[str, object] = {}
    for latent_src in latent_sources:
        if str(latent_src) == "flow" and flow is None:
            continue
        for dec in decode_methods:
            key = f"{latent_src}:{dec}"
            K = int(cfg.n_samples_per_route) if str(latent_src) == "flow" else 1

            route_success = []
            route_jacc_best = []
            route_len_ratio_best = []

            sample_success = []
            sample_hit_wall = []
            sample_dead_end = []
            sample_jacc = []
            sample_len_ratio = []

            bucket_stats: Dict[str, Dict[str, List[float]]] = {}
            for name, _lo, _hi in _len_buckets():
                bucket_stats[name] = {"succ": [], "jacc_best": [], "len_ratio_best": []}

            for i0 in range(0, B, 64):
                i1 = min(B, i0 + 64)
                # Slice batch routes
                z_b = z_enc[i0:i1]
                rc_b = {k: v[i0:i1] for k, v in base_route_cond.items()}
                sw_b = start_way_t[i0:i1]
                dw_b = dest_way_t[i0:i1]
                gt_b = gt_seqs[i0:i1]
                gtlen_b = gt_len[i0:i1]

                if str(latent_src) == "flow":
                    assert flow is not None
                    # Sample K latents per route (B*K, L, d)
                    rc_rep = {k: v.repeat_interleave(K, dim=0) for k, v in rc_b.items()}
                    z_flow = flow.sample(route_cond=rc_rep, solver_steps=cfg.solver_steps)
                    z_flow = z_flow.view(int(i1 - i0), int(K), int(z_flow.shape[1]), int(z_flow.shape[2]))
                else:
                    z_flow = None

                # For each sample, decode as a batch (B routes)
                pred_per_k: List[List[List[int]]] = []
                for k in range(K):
                    if z_flow is not None:
                        z_use = z_flow[:, k]
                    else:
                        z_use = z_b
                    preds = _decode_one(
                        ae=ae,
                        z=z_use,
                        route_cond=rc_b,
                        start_way=sw_b,
                        dest_way=dw_b,
                        method=dec,
                        beam_size=int(cfg.beam_size),
                        max_decode_len=int(cfg.max_decode_len),
                    )
                    pred_per_k.append(preds)

                # Aggregate per-route
                for bi in range(int(i1 - i0)):
                    gt_seq = gt_b[bi]
                    gtL = int(gtlen_b[bi])
                    max_len = int(cfg.max_decode_len) + 1
                    per_route_succ = False
                    best_j = -1.0
                    best_lr = float("nan")

                    for k in range(K):
                        pred_seq = pred_per_k[k][bi]
                        succ = bool(int(pred_seq[-1]) == int(dw_b[bi].item()))
                        j = _jaccard(gt_seq, pred_seq)
                        lr = float(len(pred_seq)) / float(max(1, gtL))

                        sample_success.append(float(succ))
                        sample_jacc.append(float(j))
                        sample_len_ratio.append(float(lr))

                        if (not succ) and (len(pred_seq) >= max_len):
                            sample_hit_wall.append(1.0)
                        else:
                            sample_hit_wall.append(0.0)
                        if (not succ) and (len(pred_seq) < max_len):
                            # greedy/beam stopped early -> likely dead-end (no succ candidates)
                            sample_dead_end.append(1.0)
                        else:
                            sample_dead_end.append(0.0)

                        if succ:
                            per_route_succ = True
                        if j > best_j:
                            best_j = float(j)
                            best_lr = float(lr)

                    route_success.append(float(per_route_succ))
                    route_jacc_best.append(float(best_j))
                    route_len_ratio_best.append(float(best_lr))

                    bk = _bucket_key(gtL)
                    if bk in bucket_stats:
                        bucket_stats[bk]["succ"].append(float(per_route_succ))
                        bucket_stats[bk]["jacc_best"].append(float(best_j))
                        bucket_stats[bk]["len_ratio_best"].append(float(best_lr))

                done = int(i1)
                if log_every > 0 and (done % int(log_every) == 0 or done == B):
                    print(f"[city{int(city)}] {key} done={done}/{B}")

            # Summaries
            out = {
                "n_routes": int(B),
                "n_samples_per_route": int(K),
                "route_success_rate": float(np.mean(np.asarray(route_success, dtype=np.float64))) if route_success else float("nan"),
                "sample_success_rate": float(np.mean(np.asarray(sample_success, dtype=np.float64))) if sample_success else float("nan"),
                "sample_hit_wall_rate": float(np.mean(np.asarray(sample_hit_wall, dtype=np.float64))) if sample_hit_wall else float("nan"),
                "sample_dead_end_rate": float(np.mean(np.asarray(sample_dead_end, dtype=np.float64))) if sample_dead_end else float("nan"),
                "jaccard_best": _summary_stats(route_jacc_best),
                "len_ratio_best": _summary_stats(route_len_ratio_best),
                "jaccard_per_sample": _summary_stats(sample_jacc),
                "len_ratio_per_sample": _summary_stats(sample_len_ratio),
                "by_gt_len": {
                    k: {
                        "n": int(len(v["succ"])),
                        "route_success_rate": float(np.mean(np.asarray(v["succ"], dtype=np.float64))) if v["succ"] else float("nan"),
                        "jaccard_best": _summary_stats(v["jacc_best"]),
                        "len_ratio_best": _summary_stats(v["len_ratio_best"]),
                    }
                    for k, v in bucket_stats.items()
                },
            }
            results[key] = out

    return {"city": int(city), "n_candidates": int(ids.size), "n_eval": int(pick.size), "picked_route_ids": pick.tolist(), "results": results}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quantitative evaluation for Way-CASD Decision stage (Flow/Decoder).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--flow_ckpt", type=Path, default=None, help="Required if latent_sources includes 'flow'.")
    p.add_argument("--out_json", type=Path, required=True)

    p.add_argument("--latent_sources", type=str, nargs="+", default=["gt", "flow"], choices=["gt", "flow"])
    p.add_argument("--decode_methods", type=str, nargs="+", default=["greedy", "beam"], choices=["greedy", "beam"])
    p.add_argument("--beam_size", type=int, default=10)
    p.add_argument("--solver_steps", type=int, default=0, help="Override flow solver steps (0=use ckpt/default).")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--n_routes", type=int, default=500, help="Per city.")
    p.add_argument("--n_samples_per_route", type=int, default=4, help="Only used for latent_source=flow.")
    p.add_argument("--max_way_len", type=int, default=128)
    p.add_argument("--max_decode_len", type=int, default=160)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cfg = EvalCfg(
        seed=int(args.seed),
        device=str(args.device),
        tz_offset_hours=float(args.tz_offset_hours),
        n_routes=int(args.n_routes),
        n_samples_per_route=int(args.n_samples_per_route),
        max_way_len=int(args.max_way_len),
        max_decode_len=int(args.max_decode_len),
        beam_size=int(args.beam_size),
        solver_steps=(int(args.solver_steps) if int(args.solver_steps) > 0 else None),
    )
    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # Load way graph + features
    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    way_features = make_way_feature_tensors(
        way_center_y=wf["way_center_y"],
        way_center_x=wf["way_center_x"],
        way_dir_y=wf["way_dir_y"],
        way_dir_x=wf["way_dir_x"],
        way_len_m=wf["way_len_m"],
        way_tier=wf["way_tier"],
        way_highway_code=wf["way_highway_code"],
        device=device,
    )
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1

    # Load AE
    ckpt_ae = torch.load(str(args.ae_ckpt), map_location=device)
    ae_state = ckpt_ae["model_state_dict"] if isinstance(ckpt_ae, dict) and "model_state_dict" in ckpt_ae else ckpt_ae
    ae_cfg_dict = ckpt_ae.get("config", {}) if isinstance(ckpt_ae, dict) else {}
    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(ae_cfg_dict.get("d_model", 256)),
            n_latent=int(ae_cfg_dict.get("n_latent", 64)),
            n_heads=int(ae_cfg_dict.get("n_heads", 8)),
            dropout=float(ae_cfg_dict.get("dropout", 0.1)),
            max_candidates=int(ae_cfg_dict.get("max_candidates", 32)),
            max_len=int(ae_cfg_dict.get("max_len", cfg.max_way_len)),
            coord_scale=float(ae_cfg_dict.get("coord_scale", 1024.0)),
            decoder_use_dest_dist=bool(ae_cfg_dict.get("decoder_use_dest_dist", True)),
            decoder_use_cross_attn=bool(ae_cfg_dict.get("decoder_use_cross_attn", True)),
            decoder_n_cross_heads=int(ae_cfg_dict.get("decoder_n_cross_heads", 4)),
        ),
        way_features=way_features,
        way_adj_ptr=wg["way_adj_ptr"],
        way_adj_idx=wg["way_adj_idx"],
        n_highway_types=int(max(4, n_highway_types)),
    ).to(device)
    ae.load_state_dict(ae_state, strict=False)
    ae.eval()

    # Load Flow (optional)
    flow: Optional[LatentFlowMatching] = None
    latent_sources = [str(x) for x in args.latent_sources]
    if "flow" in latent_sources:
        if args.flow_ckpt is None:
            raise SystemExit("--flow_ckpt is required when latent_sources includes 'flow'")
        ckpt_f = torch.load(str(args.flow_ckpt), map_location=device)
        f_state = ckpt_f["model_state_dict"] if isinstance(ckpt_f, dict) and "model_state_dict" in ckpt_f else ckpt_f
        f_cfg = ckpt_f.get("config", {}) if isinstance(ckpt_f, dict) else {}
        flow = LatentFlowMatching(
            cfg=LatentFlowCfg(
                d_model=int(f_cfg.get("d_model", ae.cfg.d_model)),
                n_latent=int(f_cfg.get("n_latent", ae.cfg.n_latent)),
                n_layers=int(f_cfg.get("n_layers", 6)),
                n_heads=int(f_cfg.get("n_heads", 8)),
                dropout=float(f_cfg.get("dropout", 0.1)),
                noise_sigma=float(f_cfg.get("noise_sigma", 1.0)),
                solver_steps=int(f_cfg.get("solver_steps", 20)),
            ),
            cond_cfg=ae.decoder.cond_enc.cfg,  # reuse same cond cfg (OD/time/city)
        ).to(device)
        flow.load_state_dict(f_state, strict=False)
        flow.eval()

    rep = {
        "ok": True,
        "task": "way_casd_decision_eval",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
            "flow_ckpt": str(args.flow_ckpt) if args.flow_ckpt is not None else None,
        },
        "per_city": [],
    }

    per_city = []
    for city in (0, 1):
        per_city.append(
            _eval_city(
                city=int(city),
                cfg=cfg,
                routes_npz=Path(args.way_routes_npz),
                ae=ae,
                flow=flow,
                latent_sources=[str(x) for x in args.latent_sources],  # type: ignore[arg-type]
                decode_methods=[str(x) for x in args.decode_methods],  # type: ignore[arg-type]
                device=device,
                log_every=50,
            )
        )
    rep["per_city"] = per_city
    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()

