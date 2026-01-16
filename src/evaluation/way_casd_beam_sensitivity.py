from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.models.way_casd.latent_flow import LatentFlowCfg, LatentFlowMatching
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import make_way_feature_tensors

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    seed: int
    device: str
    tz_offset_hours: float
    latent_source: str  # "gt" or "flow"
    n_samples_per_route: int
    max_way_len: int
    max_decode_len: int
    beam_sizes: List[int]
    cfg_scale: float
    solver_steps: Optional[int]
    corridor_type_override: Optional[int]
    n_per_bucket: int


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


def _load_ckpt_state_and_cfg(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
        return state, cfg
    if isinstance(ckpt, dict):
        return ckpt, {}
    raise TypeError(f"Unexpected checkpoint format: {type(ckpt)}")


def _infer_decoder_use_dest_dist_from_state(state: Dict[str, torch.Tensor]) -> bool:
    w = state.get("decoder.scorer.0.weight", None)
    if w is None:
        # Default to current behavior.
        return True
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        return True
    hidden = int(w.shape[0])
    in_dim = int(w.shape[1])
    delta = int(in_dim - hidden * 3)
    if delta == 0:
        return False
    if delta == 1:
        return True
    raise SystemExit(f"Unexpected decoder.scorer.0.weight shape: {tuple(w.shape)} (cannot infer use_dest_dist).")


def _slice_csr(ptr: np.ndarray, idx: np.ndarray, i: int) -> np.ndarray:
    s = int(ptr[i])
    e = int(ptr[i + 1])
    if e <= s:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(idx[s:e], dtype=np.int64)


def _is_valid_path(seq: List[int], ptr: np.ndarray, idx: np.ndarray) -> bool:
    if len(seq) <= 1:
        return False
    ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    for a, b in zip(seq[:-1], seq[1:]):
        nbr = _slice_csr(ptr, idx, int(a))
        if int(b) not in set(int(x) for x in nbr.tolist()):
            return False
    return True


def _jaccard(a: List[int], b: List[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb)) / float(len(sa | sb))


def _bucket_name(L: int) -> str:
    # Exactly match the PI bucket convention:
    # <15 / 15-30 / 30-60 / >60
    if int(L) < 15:
        return "lt15"
    if 15 <= int(L) <= 30:
        return "15_30"
    if 31 <= int(L) <= 60:
        return "31_60"
    return "gt60"


def _pick_routes_by_buckets(*, lens: np.ndarray, max_way_len: int, n_per_bucket: int, seed: int) -> np.ndarray:
    lens = np.asarray(lens, dtype=np.int64).reshape(-1)
    keep = (lens > 1) & (lens <= int(max_way_len))
    ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
    if ids.size == 0:
        return np.zeros((0,), dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    out: List[int] = []
    for name in ("lt15", "15_30", "31_60", "gt60"):
        if name == "lt15":
            pool = ids[lens[ids] < 15]
        elif name == "15_30":
            pool = ids[(lens[ids] >= 15) & (lens[ids] <= 30)]
        elif name == "31_60":
            pool = ids[(lens[ids] >= 31) & (lens[ids] <= 60)]
        else:
            pool = ids[lens[ids] > 60]
        if pool.size == 0:
            continue
        k = min(int(n_per_bucket), int(pool.size))
        out.extend(rng.choice(pool, size=k, replace=False).astype(np.int64).tolist())

    pick = np.asarray(sorted(set(out)), dtype=np.int64)
    return pick


def _print_one_line(summary: Dict[str, object]) -> None:
    beam = int(summary["beam_size"])
    overall = float(summary["overall"]["success_rate"])
    b = summary["by_bucket"]
    parts = []
    for k in ("lt15", "15_30", "31_60", "gt60"):
        parts.append(f"{k}:{float(b[k]['success_rate']):.3f}(n={int(b[k]['n_routes'])})")
    print(f"[beam={beam}] overall_success={overall:.3f} | " + " ".join(parts))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diagnose Way-CASD decode success vs beam_size, bucketed by GT route length.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--flow_ckpt", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--beam_sizes", type=int, nargs="+", default=[10, 50], help="E.g., --beam_sizes 10 50 100")
    p.add_argument("--latent_source", choices=["gt", "flow"], default="gt")
    p.add_argument("--n_samples_per_route", type=int, default=4, help="Only meaningful for latent_source=flow.")
    p.add_argument("--cfg_scale", type=float, default=1.5)
    p.add_argument("--solver_steps", type=int, default=0, help="Override solver steps (0=use ckpt/default).")
    p.add_argument("--corridor_type_override", type=int, default=-1, help="Force corridor_type in {0,1,2,3}. -1 uses GT.")

    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--max_decode_len", type=int, default=160)
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--n_per_bucket", type=int, default=50, help="Routes per length bucket (before de-dup).")

    p.add_argument("--route_ids", type=int, nargs="*", default=None, help="Optional explicit route indices.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Cfg(
        seed=int(args.seed),
        device=str(args.device),
        tz_offset_hours=float(args.tz_offset_hours),
        latent_source=str(args.latent_source),
        n_samples_per_route=int(args.n_samples_per_route),
        max_way_len=int(args.max_way_len),
        max_decode_len=int(args.max_decode_len),
        beam_sizes=[int(x) for x in args.beam_sizes],
        cfg_scale=float(args.cfg_scale),
        solver_steps=(int(args.solver_steps) if int(args.solver_steps) > 0 else None),
        corridor_type_override=(None if int(args.corridor_type_override) < 0 else int(args.corridor_type_override)),
        n_per_bucket=int(args.n_per_bucket),
    )
    _set_seed(cfg.seed)

    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    N = int(routes.way_seq_len.shape[0])

    if args.route_ids:
        pick = np.asarray([int(x) for x in args.route_ids], dtype=np.int64)
        pick = pick[(pick >= 0) & (pick < N)]
    else:
        pick = _pick_routes_by_buckets(lens=routes.way_seq_len, max_way_len=int(cfg.max_way_len), n_per_bucket=int(cfg.n_per_bucket), seed=int(cfg.seed))
    if pick.size == 0:
        raise SystemExit("No routes selected. Try increasing --max_way_len or lowering filters.")

    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    way_adj_ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64)
    way_adj_idx = np.asarray(wg["way_adj_idx"], dtype=np.int64)

    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1
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

    ae_state, ae_cfg_dict = _load_ckpt_state_and_cfg(Path(args.ae_ckpt))
    use_dest_dist = _infer_decoder_use_dest_dist_from_state(ae_state)
    d_model = int(ae_cfg_dict.get("d_model", 256))
    n_latent = int(ae_cfg_dict.get("n_latent", 32))
    n_heads = int(ae_cfg_dict.get("n_heads", 8))
    dropout = float(ae_cfg_dict.get("dropout", 0.1))
    max_candidates = int(ae_cfg_dict.get("max_candidates", 32))
    max_len = int(ae_cfg_dict.get("max_len", 128))
    coord_scale = float(ae_cfg_dict.get("coord_scale", 1024.0))

    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(d_model),
            n_latent=int(n_latent),
            n_heads=int(n_heads),
            dropout=float(dropout),
            max_candidates=int(max_candidates),
            max_len=int(max_len),
            coord_scale=float(coord_scale),
            decoder_use_dest_dist=bool(use_dest_dist),
        ),
        way_features=way_features,
        way_adj_ptr=way_adj_ptr,
        way_adj_idx=way_adj_idx,
        n_highway_types=int(max(4, n_highway_types)),
    ).to(device)
    ae.load_state_dict(ae_state, strict=True)
    ae.eval()

    flow = None
    flow_cfg_dict: Dict[str, object] = {}
    if cfg.latent_source == "flow":
        if args.flow_ckpt is None:
            raise SystemExit("--flow_ckpt is required when --latent_source=flow")
        flow_state, flow_cfg_dict = _load_ckpt_state_and_cfg(Path(args.flow_ckpt))
        fd = int(flow_cfg_dict.get("d_model", d_model))
        fl = int(flow_cfg_dict.get("n_latent", n_latent))
        if fd != d_model or fl != n_latent:
            raise SystemExit(f"AE/Flow mismatch: AE(d_model={d_model},n_latent={n_latent}) vs Flow(d_model={fd},n_latent={fl}). Retrain Flow for this AE.")
        flow = LatentFlowMatching(
            cfg=LatentFlowCfg(
                d_model=int(fd),
                n_latent=int(fl),
                n_layers=int(flow_cfg_dict.get("n_layers", 6)),
                n_heads=int(flow_cfg_dict.get("n_heads", 8)),
                dropout=float(flow_cfg_dict.get("dropout", 0.1)),
                noise_sigma=float(flow_cfg_dict.get("noise_sigma", 1.0)),
                solver_steps=int(flow_cfg_dict.get("solver_steps", 20)),
                cfg_drop_prob=float(flow_cfg_dict.get("cfg_drop_prob", 0.1)),
            ),
            cond_cfg=ae.decoder.cond_enc.cfg,
        ).to(device)
        flow.load_state_dict(flow_state, strict=True)
        flow.eval()

    # Precompute route buckets and GT validity.
    gt_meta: Dict[int, Dict[str, object]] = {}
    for rid in pick.tolist():
        L = int(routes.way_seq_len[int(rid)])
        s = int(routes.way_seq_ptr[int(rid)])
        e = s + L
        gt = np.asarray(routes.way_seq_idx[s:e], dtype=np.int64).tolist()
        gt_valid = _is_valid_path(gt, way_adj_ptr, way_adj_idx)
        gt_meta[int(rid)] = {"len": int(L), "bucket": _bucket_name(int(L)), "gt_valid": bool(gt_valid), "gt": gt}

    def _empty_bucket() -> Dict[str, object]:
        return {"n_routes": 0, "gt_valid_rate": float("nan"), "success_rate": float("nan")}

    summaries = []
    for beam in cfg.beam_sizes:
        beam = max(1, int(beam))
        per_route = []

        for rid in pick.tolist():
            meta = gt_meta[int(rid)]
            L = int(meta["len"])
            gt = meta["gt"]
            hour = int(_hour_from_unix(np.asarray([routes.start_t[int(rid)]], dtype=np.int64), cfg.tz_offset_hours)[0])
            dow = int(_dow_from_unix(np.asarray([routes.start_t[int(rid)]], dtype=np.int64), cfg.tz_offset_hours)[0])
            city = int(routes.route_city[int(rid)])
            gt_corr = int(routes.corridor_type[int(rid)])
            corr = int(cfg.corridor_type_override) if cfg.corridor_type_override is not None else int(gt_corr)

            if cfg.latent_source == "gt":
                K = 1
            else:
                K = int(cfg.n_samples_per_route)
            route_cond = {
                "start_pos": torch.as_tensor(np.repeat(routes.start_pos[int(rid)][None, :], K, axis=0), dtype=torch.float32, device=device),
                "dest_pos": torch.as_tensor(np.repeat(routes.dest_pos[int(rid)][None, :], K, axis=0), dtype=torch.float32, device=device),
                "hour": torch.as_tensor(np.full((K,), hour, dtype=np.int64), dtype=torch.long, device=device),
                "dow": torch.as_tensor(np.full((K,), dow, dtype=np.int64), dtype=torch.long, device=device),
                "route_city": torch.as_tensor(np.full((K,), city, dtype=np.int64), dtype=torch.long, device=device),
                "corridor_type": torch.as_tensor(np.full((K,), corr, dtype=np.int64), dtype=torch.long, device=device),
            }
            start_way = torch.as_tensor(np.full((K,), int(routes.start_way[int(rid)]), dtype=np.int64), dtype=torch.long, device=device)
            dest_way = torch.as_tensor(np.full((K,), int(routes.dest_way[int(rid)]), dtype=np.int64), dtype=torch.long, device=device)

            if cfg.latent_source == "gt":
                # Encode GT to latent (single row) to isolate decoder behavior.
                way_seq_pad = np.full((1, L), -1, dtype=np.int64)
                way_seq_pad[0, : len(gt)] = np.asarray(gt, dtype=np.int64)
                z1, _ = ae.encode(torch.as_tensor(way_seq_pad, dtype=torch.long, device=device))
                z = z1.repeat(K, 1, 1)
            else:
                assert flow is not None
                z = flow.sample(route_cond=route_cond, cfg_scale=float(cfg.cfg_scale), solver_steps=cfg.solver_steps)

            pred = ae.decoder.beam_search(
                way_embedder=ae.way_enc,
                latent_tokens=z,
                route_cond=route_cond,
                start_way=start_way,
                dest_way=dest_way,
                beam_size=int(beam),
                max_len=int(cfg.max_decode_len),
            )
            succ = [bool(p and int(p[-1]) == int(routes.dest_way[int(rid)])) for p in pred]
            jac = [float(_jaccard(gt, p)) for p in pred]
            per_route.append(
                {
                    "route_id": int(rid),
                    "gt_len": int(L),
                    "bucket": str(meta["bucket"]),
                    "gt_valid": bool(meta["gt_valid"]),
                    "n_samples": int(K),
                    "success_rate": float(np.mean(succ) if succ else float("nan")),
                    "jaccard_mean": float(np.mean(jac) if jac else float("nan")),
                }
            )

        # Aggregate.
        by_bucket: Dict[str, Dict[str, object]] = {k: _empty_bucket() for k in ("lt15", "15_30", "31_60", "gt60")}
        overall_succ: List[float] = []
        overall_valid: List[float] = []
        for b in by_bucket.keys():
            rows = [r for r in per_route if r["bucket"] == b]
            if not rows:
                continue
            by_bucket[b]["n_routes"] = int(len(rows))
            by_bucket[b]["gt_valid_rate"] = float(np.mean([1.0 if rr["gt_valid"] else 0.0 for rr in rows]))
            by_bucket[b]["success_rate"] = float(np.mean([float(rr["success_rate"]) for rr in rows]))
        overall_succ = [float(r["success_rate"]) for r in per_route]
        overall_valid = [1.0 if bool(r["gt_valid"]) else 0.0 for r in per_route]

        summary = {
            "beam_size": int(beam),
            "overall": {
                "n_routes": int(len(per_route)),
                "gt_valid_rate": float(np.mean(overall_valid) if overall_valid else float("nan")),
                "success_rate": float(np.mean(overall_succ) if overall_succ else float("nan")),
            },
            "by_bucket": by_bucket,
        }
        _print_one_line(summary)
        summaries.append({"summary": summary, "per_route": per_route})

    report = {
        "ok": True,
        "task": "way_casd_beam_sensitivity",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
            "flow_ckpt": (str(args.flow_ckpt) if args.flow_ckpt is not None else None),
        },
        "picked_routes": pick.astype(np.int64).tolist(),
        "ae_ckpt_cfg": ae_cfg_dict,
        "flow_ckpt_cfg": flow_cfg_dict,
        "results": summaries,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_dir/'report.json'}")


if __name__ == "__main__":
    main()
