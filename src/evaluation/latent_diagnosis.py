"""
Latent Distribution Diagnosis (PI Phase E1)

Goal:
  Quantify mismatch between z_gt (AE.encode on GT route) and z_flow (Flow.sample on conditions),
  and stratify by gt_hops bins and city.

Outputs (out_dir):
  - latent_stats.json: summary stats (overall + per-city + per-bin)
  - latent_pairs.npz: per-sample arrays (route_id/city/gt_hops/mse/cos/norm_gt/norm_flow)
  - latent_pca.png: optional PCA scatter (gt vs flow)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.models.way_casd.latent_flow import LatentFlowCfg, LatentFlowMatching
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    seed: int
    device: str
    tz_offset_hours: float
    n_routes: int  # per city
    min_hops: int
    max_way_len: int
    batch_size: int
    flow_solver_steps: Optional[int]


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


def _compress_consecutive_int(seq: List[int]) -> List[int]:
    out: List[int] = []
    last: Optional[int] = None
    for x in seq:
        xx = int(x)
        if last is None or xx != int(last):
            out.append(xx)
            last = xx
    return out


def _region_seq_from_way_seq(way_seq: List[int], way_region: np.ndarray) -> List[int]:
    reg: List[int] = []
    for w in way_seq:
        wi = int(w)
        if 0 <= wi < int(way_region.size):
            rr = int(way_region[wi])
            if rr >= 0:
                reg.append(int(rr))
    return _compress_consecutive_int(reg)


def _pad_int_seqs(*, seqs: List[List[int]], device: torch.device) -> torch.Tensor:
    B = int(len(seqs))
    if B == 0:
        return torch.zeros((0, 1), dtype=torch.long, device=device)
    maxL = max(1, max(int(len(s)) for s in seqs))
    pad = torch.full((B, maxL), -1, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        if not s:
            continue
        pad[i, : int(len(s))] = torch.as_tensor(s, dtype=torch.long, device=device)
    return pad


def _parse_city_name_kv(spec: str) -> Tuple[int, str]:
    s = str(spec or "").strip()
    if "=" in s:
        k, v = s.split("=", 1)
    elif ":" in s:
        k, v = s.split(":", 1)
    else:
        raise ValueError(f"Bad spec (expect CITY=NAME): {spec!r}")
    return int(str(k).strip()), str(v).strip()


def _read_split_ids(*, split_json: Path, split_part: str) -> np.ndarray:
    obj = json.loads(Path(split_json).read_text(encoding="utf-8"))
    splits = obj.get("splits", obj)
    if not isinstance(splits, dict):
        raise ValueError(f"bad split_json: {split_json} (expect dict)")
    ids_raw = splits.get(str(split_part), None)
    if ids_raw is None:
        raise ValueError(f"split_json missing part={split_part!r} (expects splits.train/val/test).")
    ids = np.asarray([int(x) for x in list(ids_raw)], dtype=np.int64).reshape(-1)
    if ids.size == 0:
        raise ValueError(f"split {split_part!r} is empty in {split_json}")
    return ids


def _hops_bins() -> List[Tuple[int, Optional[int], str]]:
    # Match PI decision: [5,10), [10,20), [20,30), [30,40), [40,60), [60,+)
    return [
        (5, 10, "[5,10)"),
        (10, 20, "[10,20)"),
        (20, 30, "[20,30)"),
        (30, 40, "[30,40)"),
        (40, 60, "[40,60)"),
        (60, None, "[60,+)"),
    ]


def _bin_label(hops: int) -> str:
    h = int(hops)
    for lo, hi, name in _hops_bins():
        if h >= int(lo) and (hi is None or h < int(hi)):
            return str(name)
    return "[other)"


def _summ(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0.0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {
        "n": float(v.size),
        "mean": float(np.mean(v)),
        "p50": float(np.quantile(v, 0.50)),
        "p90": float(np.quantile(v, 0.90)),
        "p95": float(np.quantile(v, 0.95)),
    }

def _safe_div(num: np.ndarray, den: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    a = np.asarray(num, dtype=np.float64).reshape(-1)
    b = np.asarray(den, dtype=np.float64).reshape(-1)
    return a / (b + float(eps))


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


def _pick_routes_per_city(routes, *, city: int, n_routes: int, min_hops: int, max_way_len: int, seed: int) -> np.ndarray:
    keep = (
        (routes.route_city.astype(np.int64) == int(city))
        & (routes.way_seq_len >= (int(min_hops) + 1))
        & (routes.way_seq_len <= int(max_way_len))
    )
    ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
    rng = np.random.default_rng(int(seed) + 101 * int(city))
    rng.shuffle(ids)
    return ids[: min(int(n_routes), int(ids.size))]


def _plot_pca(
    *,
    out_png: Path,
    z_gt: np.ndarray,  # (N,D)
    z_flow: np.ndarray,  # (N,D)
    city: np.ndarray,  # (N,)
    city_names: Dict[int, str],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    N = int(z_gt.shape[0])
    if N <= 1:
        return
    X = np.concatenate([z_gt, z_flow], axis=0).astype(np.float64, copy=False)
    X = X - np.mean(X, axis=0, keepdims=True)
    # PCA via Gram matrix (2N x 2N), stable for D>>N.
    G = (X @ X.T) / float(max(1, X.shape[1]))
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    w2 = np.maximum(w[:2], 0.0)
    coords = V[:, :2] * np.sqrt(w2[None, :])
    c_gt = coords[:N]
    c_fl = coords[N:]

    city = np.asarray(city, dtype=np.int64).reshape(-1)
    cities = sorted(set(int(x) for x in np.unique(city).astype(np.int64).tolist()))
    if not cities:
        return
    fig, axes = plt.subplots(1, len(cities), figsize=(4.6 * float(len(cities)), 4.2), dpi=160, squeeze=False)
    for ax, c in zip(list(axes[0]), cities):
        mask = city == int(c)
        ax.scatter(c_gt[mask, 0], c_gt[mask, 1], s=10, alpha=0.7, label="z_gt", c="#1f77b4")
        ax.scatter(c_fl[mask, 0], c_fl[mask, 1], s=10, alpha=0.7, label="z_flow", c="#ff7f0e")
        ax.set_title(city_names.get(int(c), f"city{int(c)}"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    axes[0][0].legend(frameon=False, loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="PI Phase E1: diagnose z_flow vs z_gt mismatch.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--flow_ckpt", type=Path, required=True)
    p.add_argument("--way_regions_npz", type=Path, default=None, help="Required when Flow uses region_seq conditioning.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--n_routes", type=int, default=200, help="Per city (0 and 1).")
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--flow_solver_steps", type=int, default=0, help="Override flow solver steps (0=use ckpt).")
    p.add_argument("--split_json", type=Path, default=None, help="Optional OD-disjoint split json (expects splits.train/val/test route_ids).")
    p.add_argument("--split_part", choices=["train", "val", "test"], default=None, help="Only used when --split_json is set.")
    p.add_argument("--city_name", action="append", default=[], help="Optional: CITY=NAME for plots/labels (repeatable).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Cfg(
        seed=int(args.seed),
        device=str(args.device),
        tz_offset_hours=float(args.tz_offset_hours),
        n_routes=int(args.n_routes),
        min_hops=int(args.min_hops),
        max_way_len=int(args.max_way_len),
        batch_size=int(args.batch_size),
        flow_solver_steps=(int(args.flow_solver_steps) if int(args.flow_solver_steps) > 0 else None),
    )

    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    _set_seed(int(cfg.seed))

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64)
    idx = np.asarray(wg["way_adj_idx"], dtype=np.int64)
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1
    way_features = load_way_features_from_npz(Path(args.way_features_npz), device=device)

    # Load AE
    ckpt = torch.load(str(args.ae_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ae_cfg_dict: Dict[str, object] = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(state, dict):
        raise SystemExit("[FATAL] unexpected AE ckpt format (state_dict missing).")

    use_dest_dist = _infer_decoder_use_dest_dist_from_state(state)
    use_cand_contrast = bool(ae_cfg_dict.get("decoder_use_cand_contrast", False)) or _infer_decoder_use_cand_contrast_from_state(state)
    use_cross_attn = bool(ae_cfg_dict.get("decoder_use_cross_attn", True)) or _infer_decoder_use_cross_attn_from_state(state)
    use_step_emb = bool(ae_cfg_dict.get("decoder_use_step_emb", False)) or _infer_decoder_use_step_emb_from_state(state)
    use_dest_query = bool(ae_cfg_dict.get("decoder_use_dest_query", False)) or _infer_decoder_use_dest_query_from_state(state)
    use_dir_query = bool(ae_cfg_dict.get("decoder_use_dir_query", False)) or _infer_decoder_use_dir_query_from_state(state)
    use_cand_query = bool(ae_cfg_dict.get("decoder_use_cand_query", False)) or _infer_decoder_use_cand_query_from_state(state)
    use_past_ctx = bool(ae_cfg_dict.get("decoder_use_past_context", False)) or _infer_decoder_use_past_context_from_state(state)
    past_k = int(ae_cfg_dict.get("decoder_past_k", _infer_decoder_past_k_from_state(state)))

    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(ae_cfg_dict.get("d_model", 256)),
            n_latent=int(ae_cfg_dict.get("n_latent", 64)),
            n_heads=int(ae_cfg_dict.get("n_heads", 8)),
            dropout=float(ae_cfg_dict.get("dropout", 0.1)),
            max_candidates=int(ae_cfg_dict.get("max_candidates", 32)),
            max_len=int(ae_cfg_dict.get("max_len", int(cfg.max_way_len))),
            coord_scale=float(ae_cfg_dict.get("coord_scale", 1024.0)),
            decoder_use_dest_dist=bool(use_dest_dist),
            decoder_use_cross_attn=bool(use_cross_attn),
            decoder_n_cross_heads=int(ae_cfg_dict.get("decoder_n_cross_heads", 4)),
            decoder_use_step_emb=bool(use_step_emb),
            decoder_use_dest_query=bool(use_dest_query),
            decoder_use_dir_query=bool(use_dir_query),
            decoder_use_cand_query=bool(use_cand_query),
            decoder_use_cand_contrast=bool(use_cand_contrast),
            decoder_use_past_context=bool(use_past_ctx),
            decoder_past_k=int(past_k),
            decoder_past_n_layers=int(ae_cfg_dict.get("decoder_past_n_layers", 2)),
            decoder_past_n_heads=int(ae_cfg_dict.get("decoder_past_n_heads", 4)),
            segment_size=int(ae_cfg_dict.get("segment_size", 10)),
            segment_n_latent=int(ae_cfg_dict.get("segment_n_latent", 0)),
        ),
        way_features=way_features,
        way_adj_ptr=ptr,
        way_adj_idx=idx,
        n_highway_types=int(max(4, n_highway_types)),
    ).to(device)
    try:
        ae.load_state_dict(state, strict=True)
        ckpt_strict_ok = True
    except Exception as e:
        print(f"[WARN] strict load_state_dict failed, fallback strict=False: {e}")
        ae.load_state_dict(state, strict=False)
        ckpt_strict_ok = False
    ae.eval()

    # Load Flow
    ckpt_f = torch.load(str(args.flow_ckpt), map_location=device)
    f_state = ckpt_f["model_state_dict"] if isinstance(ckpt_f, dict) and "model_state_dict" in ckpt_f else ckpt_f
    flow_cfg_dict = ckpt_f.get("config", {}) if isinstance(ckpt_f, dict) else {}
    if not isinstance(f_state, dict):
        raise SystemExit("[FATAL] unexpected Flow ckpt format (state_dict missing).")

    flow_cfg = LatentFlowCfg(
        d_model=int(flow_cfg_dict.get("d_model", ae.cfg.d_model)),
        n_latent=int(flow_cfg_dict.get("n_latent", ae.cfg.n_latent)),
        n_layers=int(flow_cfg_dict.get("n_layers", 6)),
        n_heads=int(flow_cfg_dict.get("n_heads", 8)),
        dropout=float(flow_cfg_dict.get("dropout", 0.1)),
        noise_sigma=float(flow_cfg_dict.get("noise_sigma", 1.0)),
        solver_steps=int(flow_cfg_dict.get("solver_steps", 20)),
        cond_inject=str(flow_cfg_dict.get("cond_inject", "add")),
        use_region_seq=bool(flow_cfg_dict.get("use_region_seq", False)),
        n_regions=int(flow_cfg_dict.get("n_regions", 154)),
        region_max_len=int(flow_cfg_dict.get("region_max_len", 16)),
    )
    if int(flow_cfg.d_model) != int(ae.cfg.d_model) or int(flow_cfg.n_latent) != int(ae.cfg.n_latent):
        raise SystemExit(
            f"[FATAL] AE/Flow mismatch: AE(d_model={int(ae.cfg.d_model)}, n_latent={int(ae.cfg.n_latent)}) "
            f"vs Flow(d_model={int(flow_cfg.d_model)}, n_latent={int(flow_cfg.n_latent)})."
        )
    flow = LatentFlowMatching(cfg=flow_cfg, cond_cfg=ae.decoder.cond_enc.cfg).to(device)
    flow.load_state_dict(f_state, strict=False)
    flow.eval()

    way_region: Optional[np.ndarray] = None
    if bool(flow.cfg.use_region_seq):
        if args.way_regions_npz is None:
            raise SystemExit("[FATAL] Flow requires region_seq, so --way_regions_npz is required.")
        wr = np.load(str(Path(args.way_regions_npz)), allow_pickle=True)
        if "way_region" not in wr.files:
            raise SystemExit("[FATAL] way_regions_npz missing key: way_region")
        way_region = np.asarray(wr["way_region"], dtype=np.int64).reshape(-1)

    # City name map (for PCA plot labels only).
    city_names: Dict[int, str] = {}
    for spec in list(args.city_name or []):
        c, name = _parse_city_name_kv(str(spec))
        if name:
            city_names[int(c)] = str(name)

    # Pick routes (per observed city). Prefer OD-disjoint split_json if provided.
    cities_obs = sorted(set(int(x) for x in np.unique(routes.route_city.astype(np.int64)).tolist()))
    pick_ids_pool: Optional[np.ndarray] = None
    split_json_src: Optional[str] = None
    split_part = None
    if args.split_json is not None:
        split_part = str(args.split_part) if args.split_part is not None else "test"
        pick_ids_pool = _read_split_ids(split_json=Path(args.split_json), split_part=str(split_part))
        split_json_src = str(Path(args.split_json))

    picks: Dict[int, np.ndarray] = {}
    for city in cities_obs:
        if pick_ids_pool is None:
            picks[int(city)] = _pick_routes_per_city(
                routes,
                city=int(city),
                n_routes=int(cfg.n_routes),
                min_hops=int(cfg.min_hops),
                max_way_len=int(cfg.max_way_len),
                seed=int(cfg.seed),
            )
        else:
            ids = pick_ids_pool
            keep = routes.route_city[ids] == int(city)
            keep &= routes.way_seq_len[ids] >= (int(cfg.min_hops) + 1)
            keep &= routes.way_seq_len[ids] <= int(cfg.max_way_len)
            ids2 = ids[np.nonzero(keep)[0]].astype(np.int64, copy=False)
            if ids2.size == 0:
                picks[int(city)] = np.zeros((0,), dtype=np.int64)
                continue
            rng = np.random.default_rng(int(cfg.seed) + int(city) * 10007)
            rng.shuffle(ids2)
            picks[int(city)] = ids2[: int(min(int(cfg.n_routes), int(ids2.size)))]

    # Accumulate per-sample metrics + for PCA.
    route_ids: List[int] = []
    cities: List[int] = []
    gt_hops_list: List[int] = []
    mse_list: List[float] = []
    cos_list: List[float] = []
    norm_gt_list: List[float] = []
    norm_flow_list: List[float] = []
    zgt_flat: List[np.ndarray] = []
    zfl_flat: List[np.ndarray] = []

    for city in cities_obs:
        pick = picks[int(city)]
        if pick.size == 0:
            continue

        start_way = routes.start_way[pick].astype(np.int64, copy=False)
        dest_way = routes.dest_way[pick].astype(np.int64, copy=False)
        start_pos = routes.start_pos[pick].astype(np.float32, copy=False).reshape(-1, 2)
        dest_pos = routes.dest_pos[pick].astype(np.float32, copy=False).reshape(-1, 2)
        start_t = routes.start_t[pick].astype(np.int64, copy=False)
        hour = _hour_from_unix(start_t, float(cfg.tz_offset_hours))
        dow = _dow_from_unix(start_t, float(cfg.tz_offset_hours))

        gt_seqs: List[List[int]] = []
        gt_len = routes.way_seq_len[pick].astype(np.int64, copy=False)
        for rid, L in zip(pick.tolist(), gt_len.tolist()):
            s = int(routes.way_seq_ptr[int(rid)])
            seq = routes.way_seq_idx[s : s + int(L)].astype(np.int64, copy=False).tolist()
            gt_seqs.append([int(x) for x in seq])

        B = int(pick.size)
        bs = max(1, int(cfg.batch_size))
        for i0 in range(0, B, bs):
            i1 = min(B, i0 + bs)
            gt_b = gt_seqs[i0:i1]
            rid_b = pick[i0:i1].astype(np.int64, copy=False)
            spos_b = start_pos[i0:i1]
            dpos_b = dest_pos[i0:i1]
            hour_b = hour[i0:i1]
            dow_b = dow[i0:i1]

            maxL = int(max(len(x) for x in gt_b))
            pad = np.full((int(i1 - i0), maxL), -1, dtype=np.int64)
            for j, seq in enumerate(gt_b):
                pad[j, : len(seq)] = np.asarray(seq, dtype=np.int64)
            way_pad_t = torch.as_tensor(pad, dtype=torch.long, device=device)
            z_gt, _ = ae.encode(way_pad_t)  # (B,L,D)

            route_cond: Dict[str, torch.Tensor] = {
                "start_pos": torch.as_tensor(spos_b, dtype=torch.float32, device=device),
                "dest_pos": torch.as_tensor(dpos_b, dtype=torch.float32, device=device),
                "hour": torch.as_tensor(hour_b, dtype=torch.long, device=device),
                "dow": torch.as_tensor(dow_b, dtype=torch.long, device=device),
                "route_city": torch.as_tensor(np.full((int(i1 - i0),), int(city), dtype=np.int64), dtype=torch.long, device=device),
            }
            if bool(flow.cfg.use_region_seq):
                assert way_region is not None
                rseq = [_region_seq_from_way_seq(seq, way_region) for seq in gt_b]
                route_cond["region_seq_pad"] = _pad_int_seqs(seqs=rseq, device=device)

            z_flow = flow.sample(route_cond=route_cond, solver_steps=cfg.flow_solver_steps)

            # Per-sample metrics.
            diff = (z_flow - z_gt).to(dtype=torch.float32)
            mse = torch.mean(diff * diff, dim=(1, 2)).detach().cpu().numpy().astype(np.float64, copy=False)
            # mean cosine over latent tokens.
            dot = torch.sum(z_flow.to(dtype=torch.float32) * z_gt.to(dtype=torch.float32), dim=-1)
            nz = torch.norm(z_flow.to(dtype=torch.float32), dim=-1)
            ng = torch.norm(z_gt.to(dtype=torch.float32), dim=-1)
            cos_tok = dot / (nz * ng + 1e-8)
            cos = torch.mean(cos_tok, dim=1).detach().cpu().numpy().astype(np.float64, copy=False)

            # L2 norm stats (flattened latent tokens). Useful for diagnosing scale mismatch.
            # shape: (B,)
            norm_gt = torch.norm(z_gt.to(dtype=torch.float32).reshape(int(i1 - i0), -1), dim=-1)
            norm_flow = torch.norm(z_flow.to(dtype=torch.float32).reshape(int(i1 - i0), -1), dim=-1)
            norm_gt = norm_gt.detach().cpu().numpy().astype(np.float64, copy=False)
            norm_flow = norm_flow.detach().cpu().numpy().astype(np.float64, copy=False)

            for k in range(int(i1 - i0)):
                rid = int(rid_b[k])
                L = int(routes.way_seq_len[rid])
                hops = max(0, L - 1)
                route_ids.append(rid)
                cities.append(int(city))
                gt_hops_list.append(int(hops))
                mse_list.append(float(mse[k]))
                cos_list.append(float(cos[k]))
                norm_gt_list.append(float(norm_gt[k]))
                norm_flow_list.append(float(norm_flow[k]))

            # Flatten for PCA (float32 to reduce memory).
            zgt_flat.append(z_gt.detach().cpu().numpy().reshape(int(i1 - i0), -1).astype(np.float32, copy=False))
            zfl_flat.append(z_flow.detach().cpu().numpy().reshape(int(i1 - i0), -1).astype(np.float32, copy=False))

    route_id_np = np.asarray(route_ids, dtype=np.int64)
    city_np = np.asarray(cities, dtype=np.int64)
    gt_hops_np = np.asarray(gt_hops_list, dtype=np.int64)
    mse_np = np.asarray(mse_list, dtype=np.float64)
    cos_np = np.asarray(cos_list, dtype=np.float64)
    norm_gt_np = np.asarray(norm_gt_list, dtype=np.float64)
    norm_flow_np = np.asarray(norm_flow_list, dtype=np.float64)
    zgt = np.concatenate(zgt_flat, axis=0) if zgt_flat else np.zeros((0, int(flow_cfg.n_latent) * int(flow_cfg.d_model)), dtype=np.float32)
    zfl = np.concatenate(zfl_flat, axis=0) if zfl_flat else np.zeros_like(zgt)

    # Recommended global rescaling factor to roughly match z_flow magnitude to z_gt.
    # Use p50 (median) for robustness.
    scale_p50 = float("nan")
    if np.isfinite(norm_gt_np).any() and np.isfinite(norm_flow_np).any():
        ng_p50 = float(np.quantile(norm_gt_np[np.isfinite(norm_gt_np)], 0.50))
        nf_p50 = float(np.quantile(norm_flow_np[np.isfinite(norm_flow_np)], 0.50))
        if nf_p50 > 0:
            scale_p50 = float(ng_p50 / nf_p50)

    # Summaries.
    rep: Dict[str, Any] = {
        "ok": True,
        "task": "latent_diagnosis",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
            "flow_ckpt": str(args.flow_ckpt),
            "way_regions_npz": (str(args.way_regions_npz) if args.way_regions_npz is not None else None),
        },
        "ckpt_strict_load_ok": bool(ckpt_strict_ok),
        "flow_cfg": asdict(flow_cfg),
        "cities_obs": [int(c) for c in cities_obs],
        "split_json": split_json_src,
        "split_part": (str(split_part) if split_part is not None else None),
        "overall": {
            "n": int(route_id_np.size),
            "mse": _summ(mse_np),
            "cos": _summ(cos_np),
            "norm_gt": _summ(norm_gt_np),
            "norm_flow": _summ(norm_flow_np),
            "norm_ratio_gt_over_flow": _summ(_safe_div(norm_gt_np, norm_flow_np)),
            "recommended": {
                "flow_latent_scale_p50": scale_p50,
            },
        },
        "per_city": [],
    }

    for city in cities_obs:
        m_city = city_np == int(city)
        mse_c = mse_np[m_city]
        cos_c = cos_np[m_city]
        ng_c = norm_gt_np[m_city]
        nf_c = norm_flow_np[m_city]
        hops_c = gt_hops_np[m_city]
        by_bin: Dict[str, Any] = {}
        for _lo, _hi, name in _hops_bins():
            lab = str(name)
            sel = np.asarray([_bin_label(int(h)) == lab for h in hops_c], dtype=bool)
            by_bin[lab] = {
                "n": int(np.sum(sel)),
                "mse": _summ(mse_c[sel]),
                "cos": _summ(cos_c[sel]),
                "norm_gt": _summ(ng_c[sel]),
                "norm_flow": _summ(nf_c[sel]),
                "norm_ratio_gt_over_flow": _summ(_safe_div(ng_c[sel], nf_c[sel])),
            }
        rep["per_city"].append(
            {
                "city": int(city),
                "city_name": str(city_names.get(int(city), f"city{int(city)}")),
                "n": int(np.sum(m_city)),
                "mse": _summ(mse_c),
                "cos": _summ(cos_c),
                "norm_gt": _summ(ng_c),
                "norm_flow": _summ(nf_c),
                "norm_ratio_gt_over_flow": _summ(_safe_div(ng_c, nf_c)),
                "by_bin": by_bin,
            }
        )

    # Save files.
    (out_dir / "latent_stats.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "latent_pairs.npz",
        route_id=route_id_np,
        city=city_np,
        gt_hops=gt_hops_np,
        mse=mse_np.astype(np.float32, copy=False),
        cos=cos_np.astype(np.float32, copy=False),
        norm_gt=norm_gt_np.astype(np.float32, copy=False),
        norm_flow=norm_flow_np.astype(np.float32, copy=False),
    )
    if not city_names:
        city_names = {int(c): f"city{int(c)}" for c in cities_obs}
    _plot_pca(
        out_png=out_dir / "latent_pca.png",
        z_gt=zgt,
        z_flow=zfl,
        city=city_np,
        city_names=city_names,
    )

    print(str(out_dir / "latent_stats.json"))


if __name__ == "__main__":
    main()
