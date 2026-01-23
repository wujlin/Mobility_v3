from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.models.way_casd.conditions import ConditionEncoderCfg
from src.models.way_casd.latent_flow import LatentFlowCfg, LatentFlowMatching
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import make_way_feature_tensors

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    seed: int
    device: str
    tz_offset_hours: float
    n_routes: int
    n_samples_per_route: int
    max_way_len: int
    decode: str  # "greedy" or "beam"
    beam_size: int  # only used when decode="beam"
    max_decode_len: int
    solver_steps: Optional[int]
    decode_max_candidates: int  # -1=use model cfg; 0=all successors; >0=override
    decode_candidate_policy: str  # "first" | "destdist"
    decode_include_dest_if_successor: bool
    plot_all_ways: bool


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
    return True


def _infer_decoder_use_cross_attn_from_state(state: Dict[str, torch.Tensor]) -> bool:
    # New decoder has keys like "decoder.cross_attn.in_proj_weight".
    for k in state.keys():
        if str(k).startswith("decoder.cross_attn."):
            return True
    return False


def _infer_decoder_use_step_emb_from_state(state: Dict[str, torch.Tensor]) -> bool:
    return any(str(k).startswith("decoder.step_emb.") for k in state.keys())


def _infer_decoder_use_dest_query_from_state(state: Dict[str, torch.Tensor]) -> bool:
    return any(str(k).startswith("decoder.dest_proj.") for k in state.keys())


def _jaccard(a: List[int], b: List[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb)) / float(len(sa | sb))


def _is_valid_path(seq: List[int], ptr: np.ndarray, idx: np.ndarray) -> bool:
    ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    for u, v in zip(seq[:-1], seq[1:]):
        u = int(u)
        v = int(v)
        if u < 0 or u + 1 >= int(ptr.size):
            return False
        s = int(ptr[u])
        e = int(ptr[u + 1])
        if e <= s:
            return False
        if v not in idx[s:e]:
            return False
    return True


def _decode(
    *,
    ae: WayCASDAutoEncoder,
    z: torch.Tensor,
    route_cond: Dict[str, torch.Tensor],
    start_way: torch.Tensor,
    dest_way: torch.Tensor,
    decode: str,
    beam_size: int,
    max_decode_len: int,
    decode_max_candidates: int,
    decode_candidate_policy: str,
    decode_include_dest_if_successor: bool,
) -> List[List[int]]:
    max_candidates = None if int(decode_max_candidates) < 0 else int(decode_max_candidates)
    if str(decode) == "beam":
        return ae.decoder.beam_search(
            way_embedder=ae.way_enc,
            latent_tokens=z,
            route_cond=route_cond,
            start_way=start_way,
            dest_way=dest_way,
            beam_size=int(beam_size),
            max_len=int(max_decode_len),
            max_candidates=max_candidates,
            candidate_policy=str(decode_candidate_policy),
            include_dest_if_successor=bool(decode_include_dest_if_successor),
        )
    return ae.decoder.greedy_decode(
        way_embedder=ae.way_enc,
        latent_tokens=z,
        route_cond=route_cond,
        start_way=start_way,
        dest_way=dest_way,
        max_len=int(max_decode_len),
        max_candidates=max_candidates,
        candidate_policy=str(decode_candidate_policy),
        include_dest_if_successor=bool(decode_include_dest_if_successor),
    )


def _seq_to_xy(seq: List[int], *, way_center_x: np.ndarray, way_center_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cx = np.asarray(way_center_x, dtype=np.float64).reshape(-1)
    cy = np.asarray(way_center_y, dtype=np.float64).reshape(-1)
    s = np.asarray(seq, dtype=np.int64)
    s = np.clip(s, 0, cx.shape[0] - 1)
    x = cx[s]
    y = cy[s]
    return x, y


def _load_city_road_prob(semantic_dir: Path) -> Optional[np.ndarray]:
    p = Path(semantic_dir) / "osm_road_prob.npy"
    if not p.exists():
        return None
    a = np.load(str(p))
    if a.ndim != 2:
        return None
    return np.asarray(a, dtype=np.float32)


def _load_city_poi_total(semantic_dir: Path) -> Optional[np.ndarray]:
    meta_path = Path(semantic_dir) / "poi_raster_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cats = meta.get("categories", None)
    if not isinstance(cats, list) or not cats:
        return None

    total = None
    for cat in cats:
        p = Path(semantic_dir) / f"poi_density_{cat}.npy"
        if not p.exists():
            continue
        a = np.load(str(p))
        if a.ndim != 2:
            continue
        a = np.asarray(a, dtype=np.float32)
        total = a if total is None else (total + a)
    return total


def _imshow_background(ax, arr: np.ndarray, *, cmap: str, alpha: float, vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    H, W = map(int, arr.shape)
    ax.imshow(
        arr,
        cmap=cmap,
        origin="lower",
        extent=(0, W, 0, H),
        alpha=float(alpha),
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )


def _plot_one(
    *,
    out_png: Path,
    route_id: int,
    city: int,
    hour: int,
    dow: int,
    gt_seq: List[int],
    pred_seqs: List[List[int]],
    pred_success: List[bool],
    pred_jaccard: List[float],
    way_center_x: np.ndarray,
    way_center_y: np.ndarray,
    all_way_x: Optional[np.ndarray],
    all_way_y: Optional[np.ndarray],
    road_prob: Optional[np.ndarray],
    poi_total: Optional[np.ndarray],
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("Missing dependency: matplotlib (needed for plotting).") from e

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.0))
    ax_route, ax_corr, ax_poi = axes.tolist()

    # Background: prefer road_prob (dense road mapping), else fall back to way-center scatter.
    if road_prob is not None:
        rp = np.asarray(road_prob, dtype=np.float32)
        rp = np.clip(rp, 0.0, 1.0)
        for ax in (ax_route, ax_corr):
            _imshow_background(ax, rp, cmap="Greys", alpha=0.35, vmin=0.0, vmax=1.0)
    elif all_way_x is not None and all_way_y is not None:
        for ax in (ax_route, ax_corr):
            ax.scatter(all_way_x, all_way_y, s=1, c="#d0d0d0", alpha=0.12, linewidths=0)

    # Common geometry.
    gx, gy = _seq_to_xy(gt_seq, way_center_x=way_center_x, way_center_y=way_center_y)
    gt_set = set(int(x) for x in gt_seq)

    # Pick "best" sample for route-level comparison.
    best_i = int(np.argmax(np.asarray(pred_jaccard, dtype=np.float64))) if pred_jaccard else 0
    best_seq = pred_seqs[best_i] if pred_seqs else []
    best_set = set(int(x) for x in best_seq)
    best_overlap = sorted(list(gt_set & best_set))

    # Panel 1: Route comparison (GT vs best pred).
    ax_route.plot(gx, gy, color="black", linewidth=2.0, alpha=0.90, label="GT")
    if best_seq:
        bx, by = _seq_to_xy(best_seq, way_center_x=way_center_x, way_center_y=way_center_y)
        ax_route.plot(bx, by, color="#4c72b0", linewidth=2.0, alpha=0.85, label="Pred(best)")
    if best_overlap:
        ox, oy = _seq_to_xy(best_overlap, way_center_x=way_center_x, way_center_y=way_center_y)
        ax_route.scatter(ox, oy, s=18, c="#55a868", alpha=0.8, linewidths=0, label="Overlap")

    # Panel 2: Corridor comparison (pred samples as corridor footprint).
    ax_corr.plot(gx, gy, color="black", linewidth=2.0, alpha=0.65, label="GT")
    union: set[int] = set()
    for seq in pred_seqs:
        union |= set(int(x) for x in seq)
        x, y = _seq_to_xy(seq, way_center_x=way_center_x, way_center_y=way_center_y)
        ax_corr.plot(x, y, color="#4c72b0", linewidth=1.4, alpha=0.25)
    overlap_union = sorted(list(union & gt_set))
    if overlap_union:
        ox, oy = _seq_to_xy(overlap_union, way_center_x=way_center_x, way_center_y=way_center_y)
        ax_corr.scatter(ox, oy, s=16, c="#55a868", alpha=0.75, linewidths=0, label="Overlap(union)")

    # Panel 3: POI heatmap (optional) + route overlay.
    if poi_total is not None:
        poi = np.asarray(poi_total, dtype=np.float32)
        poi = np.log1p(np.clip(poi, 0.0, None))
        vmax = float(np.percentile(poi[np.isfinite(poi)], 99.0)) if np.any(np.isfinite(poi)) else None
        _imshow_background(ax_poi, poi, cmap="magma", alpha=0.85, vmin=0.0, vmax=vmax)
    elif road_prob is not None:
        rp = np.asarray(road_prob, dtype=np.float32)
        rp = np.clip(rp, 0.0, 1.0)
        _imshow_background(ax_poi, rp, cmap="Greys", alpha=0.25, vmin=0.0, vmax=1.0)
        ax_poi.text(0.02, 0.98, "POI raster missing", transform=ax_poi.transAxes, va="top", ha="left", fontsize=10)
    else:
        ax_poi.text(0.5, 0.5, "POI raster missing", transform=ax_poi.transAxes, va="center", ha="center", fontsize=12)
    ax_poi.plot(gx, gy, color="black", linewidth=2.0, alpha=0.85, label="GT")
    if best_seq:
        bx, by = _seq_to_xy(best_seq, way_center_x=way_center_x, way_center_y=way_center_y)
        ax_poi.plot(bx, by, color="#4c72b0", linewidth=2.0, alpha=0.85, label="Pred(best)")
    if best_overlap:
        ox, oy = _seq_to_xy(best_overlap, way_center_x=way_center_x, way_center_y=way_center_y)
        ax_poi.scatter(ox, oy, s=18, c="#55a868", alpha=0.8, linewidths=0, label="Overlap")

    # Start/dest markers (shared).
    if gt_seq:
        sx, sy = gx[0], gy[0]
        dx, dy = gx[-1], gy[-1]
        for ax in (ax_route, ax_corr, ax_poi):
            ax.scatter([sx], [sy], s=80, c="white", edgecolors="black", linewidths=2.0, zorder=5)
            ax.scatter([dx], [dy], s=80, c="black", marker="s", edgecolors="white", linewidths=1.5, zorder=5)

    succ_n = int(np.sum(np.asarray(pred_success, dtype=bool)))
    jac_mean = float(np.mean(pred_jaccard)) if pred_jaccard else 0.0
    jac_best = float(pred_jaccard[best_i]) if pred_jaccard else 0.0
    corr_jac = float(len(gt_set & union)) / float(len(gt_set | union)) if (gt_set or union) else 1.0
    max_len = max([len(s) for s in pred_seqs], default=0)
    hit_wall = int(sum((not su) and (len(s) == max_len) for s, su in zip(pred_seqs, pred_success)))

    ax_route.set_title(f"Route: GT vs best (J={jac_best:.2f})")
    ax_corr.set_title(f"Corridor: union vs GT (J={corr_jac:.2f})")
    ax_poi.set_title("POI heatmap (log1p density)")
    fig.suptitle(
        f"route={route_id} city={city} hour={hour} dow={dow} succ={succ_n}/{len(pred_seqs)} hit_wall={hit_wall} J(mean)={jac_mean:.2f}",
        fontsize=12,
    )

    # Zoom to a robust bbox around GT + predictions (ignore extreme outliers).
    try:
        xs = [gx]
        ys = [gy]
        for seq in pred_seqs:
            x, y = _seq_to_xy(seq, way_center_x=way_center_x, way_center_y=way_center_y)
            xs.append(x)
            ys.append(y)
        x_all = np.concatenate(xs) if xs else np.asarray([], dtype=np.float64)
        y_all = np.concatenate(ys) if ys else np.asarray([], dtype=np.float64)
        if x_all.size > 0 and y_all.size > 0:
            qlo, qhi = (0.02, 0.98) if x_all.size >= 50 else (0.0, 1.0)
            xmin = float(np.quantile(x_all, qlo))
            xmax = float(np.quantile(x_all, qhi))
            ymin = float(np.quantile(y_all, qlo))
            ymax = float(np.quantile(y_all, qhi))
            pad = max(12.0, 0.08 * max(xmax - xmin, ymax - ymin))
            for ax in (ax_route, ax_corr, ax_poi):
                ax.set_xlim(xmin - pad, xmax + pad)
                ax.set_ylim(ymin - pad, ymax + pad)
    except Exception:
        pass

    for ax in (ax_route, ax_corr, ax_poi):
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower left", frameon=False, fontsize=9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample Way-CASD (Flow→latent→AR decode) and visualize.")
    p.add_argument("--way_routes_npz", type=Path, required=True, help="W5_way_routes_labeled/way_routes_labeled.npz")
    p.add_argument("--way_graph_npz", type=Path, required=True, help="W3_way_graph/way_graph.npz")
    p.add_argument("--way_features_npz", type=Path, required=True, help="W4_way_features/way_features.npz")
    p.add_argument("--ae_ckpt", type=Path, required=True, help="W6_train_ae/ckpt_best.pt")
    p.add_argument("--flow_ckpt", type=Path, default=None, help="W7_train_flow/ckpt_best.pt (required if latent_source=flow)")
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--n_routes", type=int, default=8, help="Number of GT routes to visualize.")
    p.add_argument("--n_samples_per_route", type=int, default=4, help="Number of samples per route (different noise).")
    p.add_argument("--route_ids", type=int, nargs="*", default=None, help="Explicit route indices in routes_npz (0-based).")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--max_way_len", type=int, default=160)

    p.add_argument(
        "--semantic_dirs",
        type=Path,
        nargs="*",
        default=None,
        help="Optional: per-city semantic dirs (contain osm_road_prob.npy and poi_raster_meta.json). "
        "Index by route_city (e.g., city0_dir city1_dir). If omitted, falls back to scatter background.",
    )
    p.add_argument("--decode", choices=["greedy", "beam"], default="greedy", help="Decode strategy (default: greedy).")
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--max_decode_len", type=int, default=160)
    p.add_argument(
        "--decode_max_candidates",
        type=int,
        default=-1,
        help="Decode candidate cap: -1=use model cfg; 0=all successors; >0=override.",
    )
    p.add_argument("--decode_candidate_policy", type=str, default="first", choices=["first", "destdist"])
    p.add_argument(
        "--decode_include_dest_if_successor",
        action="store_true",
        help="If dest_way is a direct successor but truncated out, force-include it (decode-time only).",
    )
    p.add_argument("--solver_steps", type=int, default=0, help="Override solver steps (0=use ckpt/default).")
    p.add_argument(
        "--latent_source",
        choices=["flow", "gt"],
        default="flow",
        help="Where to get latent tokens: flow (default) or gt (encode GT and decode; used to isolate decoder).",
    )
    p.add_argument("--plot_all_ways", action="store_true", help="Scatter all way centers as grey background.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Cfg(
        seed=int(args.seed),
        device=str(args.device),
        tz_offset_hours=float(args.tz_offset_hours),
        n_routes=int(args.n_routes),
        n_samples_per_route=int(args.n_samples_per_route),
        max_way_len=int(args.max_way_len),
        decode=str(args.decode),
        beam_size=int(args.beam_size),
        max_decode_len=int(args.max_decode_len),
        solver_steps=(int(args.solver_steps) if int(args.solver_steps) > 0 else None),
        decode_max_candidates=int(args.decode_max_candidates),
        decode_candidate_policy=str(args.decode_candidate_policy),
        decode_include_dest_if_successor=bool(args.decode_include_dest_if_successor),
        plot_all_ways=bool(args.plot_all_ways),
    )
    _set_seed(cfg.seed)

    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    N = int(routes.way_seq_len.shape[0])
    keep = (routes.way_seq_len > 1) & (routes.way_seq_len <= int(cfg.max_way_len))
    keep_ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
    if keep_ids.size == 0:
        raise SystemExit(f"No routes left after filtering max_way_len={cfg.max_way_len}.")

    if args.route_ids:
        pick = np.asarray([int(x) for x in args.route_ids], dtype=np.int64)
        pick = pick[(pick >= 0) & (pick < N)]
        pick = pick[keep[pick]]
    else:
        rng = np.random.default_rng(int(cfg.seed))
        n_pick = min(int(cfg.n_routes), int(keep_ids.size))
        pick = rng.choice(keep_ids, size=n_pick, replace=False)
    pick = pick.astype(np.int64, copy=False)
    pick.sort()

    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64)
    idx = np.asarray(wg["way_adj_idx"], dtype=np.int64)

    way_features = make_way_feature_tensors(
        way_center_y=wf["way_center_y"],
        way_center_x=wf["way_center_x"],
        way_dir_y=wf["way_dir_y"],
        way_dir_x=wf["way_dir_x"],
        way_len_m=wf["way_len_m"],
        way_tier=wf["way_tier"],
        way_highway_code=wf["way_highway_code"],
    )
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1

    # ===== Load AE =====
    ae_state, ae_cfg_dict = _load_ckpt_state_and_cfg(Path(args.ae_ckpt))
    use_dest_dist = _infer_decoder_use_dest_dist_from_state(ae_state)
    use_cross_attn = _infer_decoder_use_cross_attn_from_state(ae_state) or bool(ae_cfg_dict.get("decoder_use_cross_attn", True))
    use_step_emb = _infer_decoder_use_step_emb_from_state(ae_state) or bool(ae_cfg_dict.get("decoder_use_step_emb", False))
    use_dest_query = _infer_decoder_use_dest_query_from_state(ae_state) or bool(ae_cfg_dict.get("decoder_use_dest_query", False))
    ae_cfg = WayCASDAECfg(
        d_model=int(ae_cfg_dict.get("d_model", 256)),
        n_latent=int(ae_cfg_dict.get("n_latent", 32)),
        n_heads=int(ae_cfg_dict.get("n_heads", 8)),
        dropout=float(ae_cfg_dict.get("dropout", 0.1)),
        max_candidates=int(ae_cfg_dict.get("max_candidates", 32)),
        max_len=int(ae_cfg_dict.get("max_len", 160)),
        coord_scale=float(ae_cfg_dict.get("coord_scale", 1024.0)),
        decoder_use_dest_dist=bool(use_dest_dist),
        decoder_use_cross_attn=bool(use_cross_attn),
        decoder_n_cross_heads=int(ae_cfg_dict.get("decoder_n_cross_heads", 4)),
        decoder_use_step_emb=bool(use_step_emb),
        decoder_use_dest_query=bool(use_dest_query),
    )
    ae = WayCASDAutoEncoder(
        cfg=ae_cfg,
        way_features=way_features,
        way_adj_ptr=ptr,
        way_adj_idx=idx,
        n_highway_types=int(max(4, n_highway_types)),
    ).to(device)
    ae.load_state_dict(ae_state, strict=True)
    ae.eval()

    # ===== Load Flow =====
    flow = None
    flow_cfg_dict: Dict[str, object] = {}
    if str(args.latent_source) == "flow":
        if args.flow_ckpt is None:
            raise SystemExit("--flow_ckpt is required when --latent_source=flow")
        flow_state, flow_cfg_dict = _load_ckpt_state_and_cfg(Path(args.flow_ckpt))
        flow_cfg = LatentFlowCfg(
            d_model=int(flow_cfg_dict.get("d_model", ae_cfg.d_model)),
            n_latent=int(flow_cfg_dict.get("n_latent", ae_cfg.n_latent)),
            n_layers=int(flow_cfg_dict.get("n_layers", 6)),
            n_heads=int(flow_cfg_dict.get("n_heads", ae_cfg.n_heads)),
            dropout=float(flow_cfg_dict.get("dropout", ae_cfg.dropout)),
            noise_sigma=float(flow_cfg_dict.get("noise_sigma", 1.0)),
            solver_steps=int(flow_cfg_dict.get("solver_steps", 20)),
        )
        flow = LatentFlowMatching(cfg=flow_cfg, cond_cfg=ConditionEncoderCfg(d_model=int(flow_cfg.d_model), coord_scale=1024.0)).to(device)
        flow.load_state_dict(flow_state, strict=True)
        flow.eval()

    # ===== Optional background scatter =====
    all_way_x = None
    all_way_y = None
    if cfg.plot_all_ways:
        all_way_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)
        all_way_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)

    # ===== Optional semantic rasters (per-city) =====
    semantic_dirs: List[Optional[Path]] = []
    if args.semantic_dirs:
        semantic_dirs = [Path(p) for p in args.semantic_dirs]
    else:
        # Convenience: infer from env RAW_ROOT if present.
        raw_root = os.environ.get("RAW_ROOT", "")
        if raw_root:
            d0 = Path(raw_root) / "worldtrace" / "detroit_core_v1"
            d1 = Path(raw_root) / "worldtrace" / "columbus_core_v1"
            if d0.exists() or d1.exists():
                semantic_dirs = [d0 if d0.exists() else None, d1 if d1.exists() else None]

    city_cache: Dict[int, Dict[str, Optional[np.ndarray]]] = {}

    # ===== Run sampling per route =====
    per_route = []
    for rid in pick.tolist():
        L = int(routes.way_seq_len[rid])
        s = int(routes.way_seq_ptr[rid])
        gt = routes.way_seq_idx[s : s + L].astype(np.int64, copy=False).tolist()
        city = int(routes.route_city[rid])
        start_t = int(routes.start_t[rid])
        hour = int(_hour_from_unix(np.asarray([start_t], dtype=np.int64), cfg.tz_offset_hours)[0])
        dow = int(_dow_from_unix(np.asarray([start_t], dtype=np.int64), cfg.tz_offset_hours)[0])

        K = int(cfg.n_samples_per_route)
        route_cond = {
            "start_pos": torch.as_tensor(np.repeat(routes.start_pos[rid][None, :], K, axis=0), dtype=torch.float32, device=device),
            "dest_pos": torch.as_tensor(np.repeat(routes.dest_pos[rid][None, :], K, axis=0), dtype=torch.float32, device=device),
            "hour": torch.as_tensor(np.full((K,), hour, dtype=np.int64), dtype=torch.long, device=device),
            "dow": torch.as_tensor(np.full((K,), dow, dtype=np.int64), dtype=torch.long, device=device),
            "route_city": torch.as_tensor(np.full((K,), city, dtype=np.int64), dtype=torch.long, device=device),
        }
        start_way = torch.as_tensor(np.full((K,), int(routes.start_way[rid]), dtype=np.int64), dtype=torch.long, device=device)
        dest_way = torch.as_tensor(np.full((K,), int(routes.dest_way[rid]), dtype=np.int64), dtype=torch.long, device=device)

        if str(args.latent_source) == "gt":
            # Encode GT route to latent (replicated K times) to isolate decoder behavior.
            way_seq_pad = np.full((K, L), -1, dtype=np.int64)
            gt_arr = np.asarray(gt, dtype=np.int64)
            way_seq_pad[:, : gt_arr.size] = gt_arr[None, :]
            way_seq_pad_t = torch.as_tensor(way_seq_pad, dtype=torch.long, device=device)
            z, _ = ae.encode(way_seq_pad_t)
        else:
            assert flow is not None
            z = flow.sample(route_cond=route_cond, solver_steps=cfg.solver_steps)
        pred = _decode(
            ae=ae,
            z=z,
            route_cond=route_cond,
            start_way=start_way,
            dest_way=dest_way,
            decode=str(cfg.decode),
            beam_size=int(cfg.beam_size),
            max_decode_len=int(cfg.max_decode_len),
            decode_max_candidates=int(cfg.decode_max_candidates),
            decode_candidate_policy=str(cfg.decode_candidate_policy),
            decode_include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
        )

        pred_success = [bool(p and int(p[-1]) == int(routes.dest_way[rid])) for p in pred]
        pred_valid = [_is_valid_path(p, ptr, idx) for p in pred]
        pred_jac = [_jaccard(gt, p) for p in pred]

        sem = semantic_dirs[city] if (semantic_dirs and city < len(semantic_dirs)) else None
        if city not in city_cache:
            city_cache[city] = {"road_prob": None, "poi_total": None}
            if sem is not None:
                city_cache[city]["road_prob"] = _load_city_road_prob(sem)
                city_cache[city]["poi_total"] = _load_city_poi_total(sem)

        out_png = out_dir / f"case_route{rid:05d}.png"
        _plot_one(
            out_png=out_png,
            route_id=int(rid),
            city=int(city),
            hour=int(hour),
            dow=int(dow),
            gt_seq=gt,
            pred_seqs=pred,
            pred_success=pred_success,
            pred_jaccard=pred_jac,
            way_center_x=wf["way_center_x"],
            way_center_y=wf["way_center_y"],
            all_way_x=all_way_x,
            all_way_y=all_way_y,
            road_prob=city_cache[city].get("road_prob"),
            poi_total=city_cache[city].get("poi_total"),
        )

        per_route.append(
            {
                "route_id": int(rid),
                "route_city": int(city),
                "hour": int(hour),
                "dow": int(dow),
                "gt": {"len": int(len(gt))},
                "pred": [
                    {
                        "len": int(len(p)),
                        "success": bool(su),
                        "valid": bool(vv),
                        "jaccard": float(jj),
                    }
                    for p, su, vv, jj in zip(pred, pred_success, pred_valid, pred_jac)
                ],
            }
        )

    report = {
        "ok": True,
        "task": "way_casd_sample_viz",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
            "flow_ckpt": (str(args.flow_ckpt) if args.flow_ckpt is not None else None),
            "latent_source": str(args.latent_source),
        },
        "picked_routes": pick.astype(np.int64).tolist(),
        "per_route": per_route,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_dir/'report.json'}")
    print(f"[saved] figures: {out_dir}")


if __name__ == "__main__":
    main()
