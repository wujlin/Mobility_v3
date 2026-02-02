from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.evaluation.shape_metrics import dtw_distance, frechet_distance, summarize
from src.models.way_casd.latent_flow import LatentFlowCfg, LatentFlowMatching
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz
from src.models.way_casd.region_ar import RegionARCfg, RegionARModel

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_city_kv(spec: str) -> Tuple[int, Path]:
    s = str(spec or "").strip()
    if "=" in s:
        k, v = s.split("=", 1)
    elif ":" in s:
        k, v = s.split(":", 1)
    else:
        raise ValueError(f"Bad spec (expect CITY=PATH): {spec!r}")
    city = int(str(k).strip())
    path = Path(str(v).strip()).expanduser()
    return city, path


def _decode_meta(meta_obj: object) -> Optional[dict]:
    if meta_obj is None:
        return None
    if isinstance(meta_obj, np.ndarray):
        if meta_obj.size != 1:
            return None
        meta_obj = meta_obj.item()
    return meta_obj if isinstance(meta_obj, dict) else None


def _grid_bbox_from_meta(meta: dict) -> Optional[Tuple[int, int, float, float, float, float]]:
    grid = meta.get("grid", {}) if isinstance(meta, dict) else {}
    if not isinstance(grid, dict):
        return None
    H = grid.get("H", None)
    W = grid.get("W", None)
    bbox = grid.get("bbox", None)
    if not isinstance(bbox, dict):
        return None
    try:
        H_i = int(H)
        W_i = int(W)
        min_lon = float(bbox["min_lon"])
        min_lat = float(bbox["min_lat"])
        max_lon = float(bbox["max_lon"])
        max_lat = float(bbox["max_lat"])
    except Exception:
        return None
    if H_i <= 0 or W_i <= 0:
        return None
    return (H_i, W_i, min_lon, min_lat, max_lon, max_lat)


def _meta_from_city_grid_meta(path: Path) -> dict:
    """
    Load a per-city grid meta from:
      - osm_road_prob_meta.json (recommended), or
      - a single-city way_features.npz (meta.grid.* must exist).
    """
    if str(path).endswith(".npz"):
        wf = np.load(str(path), allow_pickle=True)
        meta = _decode_meta(wf.get("meta", None))
        if meta is None:
            raise ValueError(f"{path} missing meta (need meta.grid.H/W/bbox).")
    else:
        meta = _read_json(path)

    # Normalize: allow meta with bbox/H/W at root by wrapping into meta['grid'].
    if _grid_bbox_from_meta(meta) is None:
        if isinstance(meta, dict) and ("H" in meta) and ("W" in meta) and ("bbox" in meta):
            meta = {"grid": {"H": meta["H"], "W": meta["W"], "bbox": meta["bbox"]}}
    if _grid_bbox_from_meta(meta) is None:
        raise ValueError(f"{path} missing grid meta (need grid.H/grid.W/grid.bbox).")
    return meta


def _grid_yx_to_xy_m(y: np.ndarray, x: np.ndarray, *, meta: dict) -> np.ndarray:
    """
    Convert grid y/x to local planar meters via bbox (equirectangular).
    Output: (N,2) [x_m, y_m]
    """
    bb = _grid_bbox_from_meta(meta)
    if bb is None:
        raise ValueError("meta missing grid bbox")
    H, W, min_lon, min_lat, max_lon, max_lat = bb
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    lon = min_lon + (x / float(W)) * (max_lon - min_lon)
    lat = max_lat - (y / float(H)) * (max_lat - min_lat)

    lat0 = 0.5 * (min_lat + max_lat)
    lon0 = 0.5 * (min_lon + max_lon)
    r = 6371000.0
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    lat0_r = math.radians(float(lat0))
    lon0_r = math.radians(float(lon0))
    x_m = (lon_r - lon0_r) * math.cos(lat0_r) * r
    y_m = (lat_r - lat0_r) * r
    return np.stack([x_m, y_m], axis=1).astype(np.float64, copy=False)


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


def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb)) / float(len(sa | sb))


def _has_loop(seq: Sequence[int]) -> bool:
    seen: set[int] = set()
    for x in seq:
        xx = int(x)
        if xx in seen:
            return True
        seen.add(xx)
    return False


def _sum_way_len_m(way_len_m: np.ndarray, seq: Sequence[int]) -> float:
    if not seq:
        return float("nan")
    ids = np.asarray([int(x) for x in seq], dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < int(way_len_m.size))]
    if ids.size == 0:
        return float("nan")
    return float(np.sum(way_len_m[ids].astype(np.float64, copy=False)))


def _hops_bins() -> List[Tuple[int, Optional[int], str]]:
    # [lo, hi) except last is [lo, +inf)
    return [
        (5, 10, "[5,10)"),
        (10, 20, "[10,20)"),
        (20, 30, "[20,30)"),
        (30, 40, "[30,40)"),
        (40, 60, "[40,60)"),
        (60, None, "[60,+)"),
    ]


def _bin_label(hops: int) -> str:
    hh = int(hops)
    for lo, hi, name in _hops_bins():
        if hh < int(lo):
            continue
        if hi is None or hh < int(hi):
            return str(name)
    return str(_hops_bins()[-1][2])


@dataclass(frozen=True)
class Cfg:
    seed: int
    device: str
    tz_offset_hours: float

    latent_source: str  # "gt" | "flow"
    n_samples_per_route: int  # only used when latent_source=flow
    sample_select: str  # "first" | "best"
    flow_solver_steps: Optional[int]  # None=use ckpt/default

    n_routes: int  # per city
    min_hops: int
    max_way_len: int
    max_decode_len: int

    decode_max_candidates: int
    decode_candidate_policy: str
    decode_include_dest_if_successor: bool
    guided_dest_alpha: float

    beam_size: int
    compare_beam: bool

    region_constraint: str  # "none" | "gt" | "ar"
    region_constraint_mode: str  # only "strict" for now
    region_constraint_fallback: str  # "unconstrained" | "dest_region" | "stop"

    region_ar_max_len: int  # only used when region_constraint=ar


def _compress_consecutive_int(seq: List[int]) -> List[int]:
    out: List[int] = []
    last: Optional[int] = None
    for x in seq:
        xx = int(x)
        if last is None or xx != int(last):
            out.append(xx)
            last = xx
    return out


def _region_seq_from_way_seq(way_seq: List[int], way_region_np: np.ndarray) -> List[int]:
    reg = []
    for w in way_seq:
        wi = int(w)
        if 0 <= wi < int(way_region_np.size):
            rr = int(way_region_np[wi])
            if rr >= 0:
                reg.append(int(rr))
    return _compress_consecutive_int(reg)


def _load_region_ar_meta(*, way_regions_npz: Path, way_features_npz: Path, coord_scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """
    Build Region AR meta tensors:
      - region_city (R,)
      - region_static (R,4)
      - region_adj (R,R) bool
    """
    wr = np.load(str(way_regions_npz), allow_pickle=True)
    need = {"region_way_ptr", "region_way_idx", "region_adj_ptr", "region_adj_idx", "meta"}
    missing = sorted(list(need - set(wr.files)))
    if missing:
        raise SystemExit(f"[FATAL] way_regions_npz missing keys: {missing}")
    region_way_ptr = np.asarray(wr["region_way_ptr"], dtype=np.int64).reshape(-1)
    region_way_idx = np.asarray(wr["region_way_idx"], dtype=np.int64).reshape(-1)
    region_adj_ptr = np.asarray(wr["region_adj_ptr"], dtype=np.int64).reshape(-1)
    region_adj_idx = np.asarray(wr["region_adj_idx"], dtype=np.int64).reshape(-1)

    meta_obj = wr["meta"]
    if isinstance(meta_obj, np.ndarray) and meta_obj.size == 1:
        meta_obj = meta_obj.item()
    meta = meta_obj if isinstance(meta_obj, dict) else None
    if meta is None:
        raise SystemExit("[FATAL] way_regions_npz missing meta (need per_city region offsets).")
    per_city = meta.get("per_city", {})
    if not isinstance(per_city, dict) or not per_city:
        raise SystemExit("[FATAL] way_regions_npz meta missing per_city.")

    n_regions = int(region_way_ptr.size) - 1
    region_city = np.full((n_regions,), -1, dtype=np.int64)
    n_cities = 0
    for k, v in per_city.items():
        try:
            city = int(k)
            off = int(v.get("region_id_offset", 0))
            nr = int(v.get("n_regions", 0))
        except Exception:
            continue
        if nr <= 0:
            continue
        region_city[off : off + nr] = int(city)
        n_cities = max(n_cities, city + 1)
    if int(np.sum(region_city < 0)) > 0:
        raise SystemExit(f"[FATAL] region_city has unassigned entries: {int(np.sum(region_city < 0))}/{n_regions}")

    wf = np.load(str(way_features_npz), allow_pickle=True)
    need = {"way_center_y", "way_center_x"}
    missing = sorted(list(need - set(wf.files)))
    if missing:
        raise SystemExit(f"[FATAL] way_features_npz missing keys: {missing}")
    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)

    coord_scale = float(coord_scale)
    cent_y = np.zeros((n_regions,), dtype=np.float64)
    cent_x = np.zeros((n_regions,), dtype=np.float64)
    n_ways = np.zeros((n_regions,), dtype=np.float64)
    for r in range(n_regions):
        s = int(region_way_ptr[r])
        e = int(region_way_ptr[r + 1])
        ways = region_way_idx[s:e]
        n = int(ways.size)
        n_ways[r] = float(n)
        if n <= 0:
            continue
        cent_y[r] = float(np.mean(way_center_y[ways]))
        cent_x[r] = float(np.mean(way_center_x[ways]))

    deg = (region_adj_ptr[1:] - region_adj_ptr[:-1]).astype(np.int64, copy=False)
    deg_f = deg.astype(np.float64, copy=False)
    static = np.stack([cent_y / coord_scale, cent_x / coord_scale, np.log1p(n_ways), np.log1p(deg_f)], axis=1).astype(np.float32, copy=False)

    adj = np.zeros((n_regions, n_regions), dtype=bool)
    np.fill_diagonal(adj, True)
    for r in range(n_regions):
        s = int(region_adj_ptr[r])
        e = int(region_adj_ptr[r + 1])
        for nb in region_adj_idx[s:e].tolist():
            b = int(nb)
            if 0 <= b < n_regions:
                adj[r, b] = True

    rep = {
        "n_regions": int(n_regions),
        "n_cities": int(n_cities),
        "coord_scale": float(coord_scale),
    }
    return (
        torch.as_tensor(region_city, dtype=torch.long),
        torch.as_tensor(static, dtype=torch.float32),
        torch.as_tensor(adj, dtype=torch.bool),
        rep,
    )


@torch.no_grad()
def _decode_region_seq_greedy(
    *,
    model: RegionARModel,
    region_adj: torch.Tensor,
    route_cond: Dict[str, torch.Tensor],
    o_region: int,
    d_region: int,
    max_len: int,
) -> List[int]:
    seq: List[int] = [int(o_region)]
    for _ in range(max(1, int(max_len)) - 1):
        cur = int(seq[-1])
        if cur == int(d_region):
            break
        x = torch.as_tensor(np.asarray(seq, dtype=np.int64)[None, :], dtype=torch.long, device=route_cond["route_city"].device)
        logits = model(
            region_seq_in=x,
            o_region=torch.as_tensor([int(o_region)], dtype=torch.long, device=x.device),
            d_region=torch.as_tensor([int(d_region)], dtype=torch.long, device=x.device),
            route_cond=route_cond,
        )
        next_logits = logits[0, -1]  # (R,)
        if bool(model.cfg.use_candidate_mask):
            allowed = region_adj[int(cur)].clone()
            if 0 <= int(cur) < int(allowed.numel()):
                allowed[int(cur)] = False
            if bool(allowed.sum().item() == 0):
                allowed[int(cur)] = True
            next_logits = next_logits.masked_fill(~allowed, -1e9)
        nxt = int(torch.argmax(next_logits).item())
        seq.append(int(nxt))
    # ensure compressed
    return _compress_consecutive_int(seq)


def _require_city_meta(city_meta: Dict[int, dict], cities: Iterable[int]) -> None:
    missing = [int(c) for c in cities if int(c) not in city_meta]
    if missing:
        raise SystemExit(f"[FATAL] missing --city_grid_meta for cities={missing} (PI: meters is mandatory).")


def _nanmean(x: Sequence[float]) -> float:
    a = np.asarray(list(x), dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float("nan")


def _best_sample_index(samples: List[Dict[str, object]]) -> int:
    if not samples:
        return 0

    def _key(m: Dict[str, object]) -> Tuple[int, float, float]:
        succ = 0 if bool(m.get("success", False)) else 1
        dtw = float(m.get("dtw_m", float("nan")))
        fre = float(m.get("frechet_m", float("nan")))
        dtw = dtw if math.isfinite(dtw) else float("inf")
        fre = fre if math.isfinite(fre) else float("inf")
        return (succ, dtw, fre)

    best_i = 0
    best_k = _key(samples[0])
    for i in range(1, len(samples)):
        k = _key(samples[i])
        if k < best_k:
            best_k = k
            best_i = int(i)
    return int(best_i)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Way-CASD binned evaluation (meters): success + DTW/Fréchet + length ratio, stratified by gt_hops.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)

    p.add_argument("--latent_source", choices=["gt", "flow"], default="gt", help="gt=oracle (GT->AE.encode); flow=generation (Flow.sample).")
    p.add_argument("--flow_ckpt", type=Path, default=None, help="Required when --latent_source=flow.")
    p.add_argument("--n_samples_per_route", type=int, default=1, help="Only used when --latent_source=flow.")
    p.add_argument("--sample_select", choices=["first", "best"], default="first", help="When n_samples_per_route>1, which sample to report as route-level metrics.")
    p.add_argument("--flow_solver_steps", type=int, default=0, help="Override flow solver steps (0=use ckpt/default).")

    p.add_argument("--n_routes", type=int, default=200, help="Per city (0 and 1).")
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--max_decode_len", type=int, default=160)

    p.add_argument("--decode_max_candidates", type=int, default=-1, help="-1=use ckpt cfg; 0=all successors; >0=override.")
    p.add_argument("--decode_candidate_policy", choices=["first", "destdist"], default="first")
    p.add_argument("--decode_include_dest_if_successor", action="store_true")
    p.add_argument("--decode_guided_dest_alpha", type=float, default=0.0)

    p.add_argument("--beam_size", type=int, default=10)
    p.add_argument("--no_compare_beam", action="store_true", help="If set, only evaluate greedy (skip beam).")

    # Region-constrained decoding (hierarchical P0).
    p.add_argument(
        "--region_constraint",
        choices=["none", "gt", "ar"],
        default="none",
        help="none=baseline; gt=use GT-derived region_seq; ar=use Region AR predicted region_seq.",
    )
    p.add_argument("--way_regions_npz", type=Path, default=None, help="Required when --region_constraint != none.")
    p.add_argument("--region_ar_ckpt", type=Path, default=None, help="Required when --region_constraint=ar.")
    p.add_argument("--region_ar_max_len", type=int, default=16, help="Max region_seq length for Region AR greedy rollout.")
    p.add_argument("--region_constraint_mode", choices=["strict", "relaxed"], default="strict")
    p.add_argument("--region_constraint_fallback", choices=["unconstrained", "dest_region", "stop"], default="unconstrained")
    p.add_argument("--out_per_route_json", type=Path, default=None, help="Optional: dump per-route records for diff analysis.")

    p.add_argument(
        "--city_grid_meta",
        type=str,
        action="append",
        default=[],
        help="Per-city grid meta for meters conversion, format CITY=PATH (osm_road_prob_meta.json or single-city way_features.npz).",
    )
    args = p.parse_args()

    cfg = Cfg(
        seed=int(args.seed),
        device=str(args.device),
        tz_offset_hours=float(args.tz_offset_hours),
        latent_source=str(args.latent_source),
        n_samples_per_route=max(1, int(args.n_samples_per_route)),
        sample_select=str(args.sample_select),
        flow_solver_steps=(int(args.flow_solver_steps) if int(args.flow_solver_steps) > 0 else None),
        n_routes=int(args.n_routes),
        min_hops=int(args.min_hops),
        max_way_len=int(args.max_way_len),
        max_decode_len=int(args.max_decode_len),
        decode_max_candidates=int(args.decode_max_candidates),
        decode_candidate_policy=str(args.decode_candidate_policy),
        decode_include_dest_if_successor=bool(args.decode_include_dest_if_successor),
        guided_dest_alpha=float(args.decode_guided_dest_alpha),
        beam_size=int(args.beam_size),
        compare_beam=(not bool(args.no_compare_beam)),
        region_constraint=str(args.region_constraint),
        region_constraint_mode=str(args.region_constraint_mode),
        region_constraint_fallback=str(args.region_constraint_fallback),
        region_ar_max_len=int(args.region_ar_max_len),
    )

    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64)
    idx = np.asarray(wg["way_adj_idx"], dtype=np.int64)
    way_len_m = np.asarray(wf["way_len_m"], dtype=np.float64).reshape(-1)
    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)

    # Optional: region constraint inputs (P0 hierarchical experiment).
    way_region_np: Optional[np.ndarray] = None
    way_region_t: Optional[torch.Tensor] = None
    region_ar_model: Optional[RegionARModel] = None
    region_ar_adj: Optional[torch.Tensor] = None
    region_ar_meta: Optional[dict] = None
    region_adj_t: Optional[torch.Tensor] = None

    if str(cfg.region_constraint) != "none":
        if args.way_regions_npz is None:
            raise SystemExit("[FATAL] --way_regions_npz is required when --region_constraint != none")
        if not Path(args.way_regions_npz).exists():
            raise SystemExit(f"[FATAL] file not found: {args.way_regions_npz}")
        wr = np.load(str(Path(args.way_regions_npz)), allow_pickle=True)
        if "way_region" not in wr.files:
            raise SystemExit("[FATAL] way_regions_npz missing key: way_region")
        way_region_np = np.asarray(wr["way_region"], dtype=np.int64).reshape(-1)
        way_region_t = torch.as_tensor(way_region_np, dtype=torch.long, device=device)

        if str(cfg.region_constraint_mode) == "relaxed" and str(cfg.region_constraint) != "ar":
            # For relaxed mode under GT-derived region_seq, we still need region adjacency.
            _rc, _rs, radj, _rrep = _load_region_ar_meta(
                way_regions_npz=Path(args.way_regions_npz),
                way_features_npz=Path(args.way_features_npz),
                coord_scale=1024.0,
            )
            region_adj_t = radj.to(device=device)

        if str(cfg.region_constraint) == "ar":
            if args.region_ar_ckpt is None:
                raise SystemExit("[FATAL] --region_ar_ckpt is required when --region_constraint=ar")
            if not Path(args.region_ar_ckpt).exists():
                raise SystemExit(f"[FATAL] file not found: {args.region_ar_ckpt}")
            ckpt = torch.load(str(Path(args.region_ar_ckpt)), map_location=device)
            if not isinstance(ckpt, dict) or "model" not in ckpt:
                raise SystemExit("[FATAL] unexpected region_ar_ckpt format (need dict with key 'model').")
            cfg_ar = ckpt.get("cfg", {})
            if not isinstance(cfg_ar, dict):
                raise SystemExit("[FATAL] region_ar_ckpt missing cfg dict.")
            coord_scale_ar = float(cfg_ar.get("coord_scale", 1024.0))
            rc, rs, radj, rrep = _load_region_ar_meta(
                way_regions_npz=Path(args.way_regions_npz),
                way_features_npz=Path(args.way_features_npz),
                coord_scale=coord_scale_ar,
            )
            region_ar_meta = rrep
            region_ar_adj = radj.to(device=device)
            region_adj_t = region_ar_adj
            region_ar_model = RegionARModel(
                cfg=RegionARCfg(
                    d_model=int(cfg_ar.get("d_model", 256)),
                    n_heads=int(cfg_ar.get("n_heads", 8)),
                    n_layers=int(cfg_ar.get("n_layers", 4)),
                    dropout=float(cfg_ar.get("dropout", 0.1)),
                    max_len=int(cfg_ar.get("max_len", int(cfg.region_ar_max_len))),
                    n_regions=int(rc.numel()),
                    n_route_cities=int(cfg_ar.get("n_route_cities", 2)),
                    coord_scale=float(coord_scale_ar),
                    use_candidate_mask=bool(cfg_ar.get("use_candidate_mask", True)),
                ),
                region_city=rc.to(device=device),
                region_static=rs.to(device=device),
                region_adj=region_ar_adj,
            ).to(device)
            region_ar_model.load_state_dict(ckpt["model"], strict=True)
            region_ar_model.eval()

    # Load per-city meta (mandatory for meters).
    city_meta: Dict[int, dict] = {}
    city_meta_src: Dict[int, str] = {}
    for spec in list(args.city_grid_meta or []):
        c, path = _parse_city_kv(str(spec))
        if not path.exists():
            raise SystemExit(f"[FATAL] file not found: {path}")
        city_meta[int(c)] = _meta_from_city_grid_meta(path)
        city_meta_src[int(c)] = str(path)
    cities_obs = sorted(set(int(x) for x in routes.route_city.astype(np.int64).tolist()))
    _require_city_meta(city_meta, cities_obs)

    # Precompute way center coords (meters) per city meta.
    way_xy_m: Dict[int, np.ndarray] = {}
    for c in cities_obs:
        way_xy_m[int(c)] = _grid_yx_to_xy_m(way_center_y, way_center_x, meta=city_meta[int(c)])

    # Build AE (infer cfg from ckpt).
    way_features = load_way_features_from_npz(Path(args.way_features_npz), device=device)
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1
    ckpt = torch.load(str(args.ae_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ae_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(state, dict):
        raise SystemExit("[FATAL] unexpected ckpt format (state_dict missing).")
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
    n_route_cities = max(int(n_route_cities), int(max(cities_obs) + 1))

    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(ae_cfg.get("d_model", 256)),
            n_latent=int(ae_cfg.get("n_latent", 64)),
            n_heads=int(ae_cfg.get("n_heads", 8)),
            dropout=float(ae_cfg.get("dropout", 0.1)),
            max_candidates=int(ae_cfg.get("max_candidates", 32)),
            max_len=int(ae_cfg.get("max_len", cfg.max_way_len)),
            coord_scale=float(ae_cfg.get("coord_scale", 1024.0)),
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

    flow: Optional[LatentFlowMatching] = None
    flow_cfg_dict: Dict[str, object] = {}
    if str(cfg.latent_source) == "flow":
        if args.flow_ckpt is None:
            raise SystemExit("[FATAL] --flow_ckpt is required when --latent_source=flow")
        ckpt_f = torch.load(str(args.flow_ckpt), map_location=device)
        f_state = ckpt_f["model_state_dict"] if isinstance(ckpt_f, dict) and "model_state_dict" in ckpt_f else ckpt_f
        flow_cfg_dict = ckpt_f.get("config", {}) if isinstance(ckpt_f, dict) else {}
        if not isinstance(f_state, dict):
            raise SystemExit("[FATAL] unexpected flow ckpt format (state_dict missing).")

        flow_cfg = LatentFlowCfg(
            d_model=int(flow_cfg_dict.get("d_model", ae.cfg.d_model)),
            n_latent=int(flow_cfg_dict.get("n_latent", ae.cfg.n_latent)),
            n_layers=int(flow_cfg_dict.get("n_layers", 6)),
            n_heads=int(flow_cfg_dict.get("n_heads", 8)),
            dropout=float(flow_cfg_dict.get("dropout", 0.1)),
            noise_sigma=float(flow_cfg_dict.get("noise_sigma", 1.0)),
            solver_steps=int(flow_cfg_dict.get("solver_steps", 20)),
        )
        if int(flow_cfg.d_model) != int(ae.cfg.d_model) or int(flow_cfg.n_latent) != int(ae.cfg.n_latent):
            raise SystemExit(
                f"[FATAL] AE/Flow mismatch: AE(d_model={int(ae.cfg.d_model)}, n_latent={int(ae.cfg.n_latent)}) "
                f"vs Flow(d_model={int(flow_cfg.d_model)}, n_latent={int(flow_cfg.n_latent)})."
            )
        flow = LatentFlowMatching(cfg=flow_cfg, cond_cfg=ae.decoder.cond_enc.cfg).to(device)
        flow.load_state_dict(f_state, strict=False)
        flow.eval()

    max_candidates = int(cfg.decode_max_candidates)
    if max_candidates < 0:
        max_candidates = int(ae.cfg.max_candidates)

    # Route sampling per city (fixed seed).
    picks: Dict[int, np.ndarray] = {}
    for city in cities_obs:
        keep = (
            (routes.route_city.astype(np.int64) == int(city))
            & (routes.way_seq_len >= (int(cfg.min_hops) + 1))
            & (routes.way_seq_len <= int(cfg.max_way_len))
        )
        ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
        rng = np.random.default_rng(int(cfg.seed) + 101 * int(city))
        rng.shuffle(ids)
        picks[int(city)] = ids[: min(int(cfg.n_routes), int(ids.size))]

    # Evaluate.
    per_route: List[Dict[str, Any]] = []
    for city in cities_obs:
        pick = picks[int(city)]
        if pick.size == 0:
            continue

        print(
            f"[city{int(city)}] start n_routes={int(pick.size)} "
            f"latent_source={str(cfg.latent_source)} region_constraint={str(cfg.region_constraint)} "
            f"compare_beam={bool(cfg.compare_beam)}",
            flush=True,
        )

        # Preload per-route arrays.
        start_way = routes.start_way[pick].astype(np.int64, copy=False)
        dest_way = routes.dest_way[pick].astype(np.int64, copy=False)
        start_pos = routes.start_pos[pick].astype(np.float64, copy=False).reshape(-1, 2)
        dest_pos = routes.dest_pos[pick].astype(np.float64, copy=False).reshape(-1, 2)
        start_t = routes.start_t[pick].astype(np.int64, copy=False)
        hour = _hour_from_unix(start_t, cfg.tz_offset_hours)
        dow = _dow_from_unix(start_t, cfg.tz_offset_hours)

        # Load GT sequences (list of lists).
        gt_seqs: List[List[int]] = []
        gt_len = routes.way_seq_len[pick].astype(np.int64, copy=False)
        for rid, L in zip(pick.tolist(), gt_len.tolist()):
            s = int(routes.way_seq_ptr[int(rid)])
            seq = routes.way_seq_idx[s : s + int(L)].astype(np.int64, copy=False).tolist()
            gt_seqs.append([int(x) for x in seq])

        # Encode GT -> z_enc in batches.
        B = int(pick.size)
        for i0 in range(0, B, 64):
            i1 = min(B, i0 + 64)
            print(f"[city{int(city)}] batch {int(i0)}:{int(i1)}/{int(B)}", flush=True)
            gt_b = gt_seqs[i0:i1]
            rid_b = pick[i0:i1].astype(np.int64, copy=False)
            sw_b = start_way[i0:i1]
            dw_b = dest_way[i0:i1]
            spos_b = start_pos[i0:i1]
            dpos_b = dest_pos[i0:i1]
            hour_b = hour[i0:i1]
            dow_b = dow[i0:i1]

            route_cond = {
                "start_pos": torch.as_tensor(spos_b, dtype=torch.float32, device=device),
                "dest_pos": torch.as_tensor(dpos_b, dtype=torch.float32, device=device),
                "hour": torch.as_tensor(hour_b, dtype=torch.long, device=device),
                "dow": torch.as_tensor(dow_b, dtype=torch.long, device=device),
                "route_city": torch.as_tensor(np.full((int(i1 - i0),), int(city), dtype=np.int64), dtype=torch.long, device=device),
            }
            sw_t = torch.as_tensor(sw_b, dtype=torch.long, device=device)
            dw_t = torch.as_tensor(dw_b, dtype=torch.long, device=device)

            K = 1
            z_use: torch.Tensor
            route_cond_use: Dict[str, torch.Tensor]
            sw_use: torch.Tensor
            dw_use: torch.Tensor
            if str(cfg.latent_source) == "gt":
                maxL = int(max(len(x) for x in gt_b))
                pad = np.full((int(i1 - i0), maxL), -1, dtype=np.int64)
                for j, seq in enumerate(gt_b):
                    pad[j, : len(seq)] = np.asarray(seq, dtype=np.int64)
                way_pad_t = torch.as_tensor(pad, dtype=torch.long, device=device)
                z_use, _ = ae.encode(way_pad_t)
                route_cond_use = route_cond
                sw_use = sw_t
                dw_use = dw_t
            else:
                if flow is None:
                    raise SystemExit("[FATAL] latent_source=flow but Flow is not loaded (missing --flow_ckpt?)")
                K = int(max(1, int(cfg.n_samples_per_route)))
                route_cond_use = {k: v.repeat_interleave(K, dim=0) for k, v in route_cond.items()}
                sw_use = sw_t.repeat_interleave(K, dim=0)
                dw_use = dw_t.repeat_interleave(K, dim=0)
                z_use = flow.sample(route_cond=route_cond_use, solver_steps=cfg.flow_solver_steps)

            region_seq_use: Optional[List[List[int]]] = None
            region_routes: Optional[List[List[int]]] = None
            if str(cfg.region_constraint) != "none":
                if way_region_np is None or way_region_t is None:
                    raise RuntimeError("region_constraint enabled but way_region is missing")
                region_routes = []
                if str(cfg.region_constraint) == "gt":
                    for seq in gt_b:
                        region_routes.append(_region_seq_from_way_seq(seq, way_region_np))
                elif str(cfg.region_constraint) == "ar":
                    if region_ar_model is None or region_ar_adj is None:
                        raise RuntimeError("region_constraint=ar but region_ar_model is not loaded")
                    for bi in range(int(i1 - i0)):
                        sr = int(way_region_np[int(sw_b[bi])]) if 0 <= int(sw_b[bi]) < int(way_region_np.size) else -1
                        dr = int(way_region_np[int(dw_b[bi])]) if 0 <= int(dw_b[bi]) < int(way_region_np.size) else -1
                        route_cond_1 = {
                            "start_pos": route_cond["start_pos"][bi : bi + 1],
                            "dest_pos": route_cond["dest_pos"][bi : bi + 1],
                            "hour": route_cond["hour"][bi : bi + 1],
                            "dow": route_cond["dow"][bi : bi + 1],
                            "route_city": route_cond["route_city"][bi : bi + 1],
                        }
                        region_routes.append(
                            _decode_region_seq_greedy(
                                model=region_ar_model,
                                region_adj=region_ar_adj,
                                route_cond=route_cond_1,
                                o_region=int(sr),
                                d_region=int(dr),
                                max_len=int(cfg.region_ar_max_len),
                            )
                        )
                else:
                    raise RuntimeError(f"unknown region_constraint: {cfg.region_constraint!r}")

                # Replicate per route for flow sampling (K per route).
                region_seq_use = []
                for rs in region_routes:
                    for _k in range(int(K)):
                        region_seq_use.append(list(rs))

            greedy = ae.decoder.greedy_decode_batched(
                way_embedder=ae.way_enc,
                latent_tokens=z_use,
                route_cond=route_cond_use,
                start_way=sw_use,
                dest_way=dw_use,
                way_region=way_region_t,
                region_seq=region_seq_use,
                region_adj=region_adj_t,
                region_constraint_mode=str(cfg.region_constraint_mode),
                region_constraint_fallback=str(cfg.region_constraint_fallback),
                max_len=int(cfg.max_decode_len),
                max_candidates=(None if int(cfg.decode_max_candidates) < 0 else int(max_candidates)),
                candidate_policy=str(cfg.decode_candidate_policy),
                include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
                guided_dest_alpha=float(cfg.guided_dest_alpha),
            )

            beam: Optional[List[List[int]]] = None
            if bool(cfg.compare_beam):
                beam = ae.decoder.beam_search_batched(
                    way_embedder=ae.way_enc,
                    latent_tokens=z_use,
                    route_cond=route_cond_use,
                    start_way=sw_use,
                    dest_way=dw_use,
                    way_region=way_region_t,
                    region_seq=region_seq_use,
                    region_adj=region_adj_t,
                    region_constraint_mode=str(cfg.region_constraint_mode),
                    region_constraint_fallback=str(cfg.region_constraint_fallback),
                    beam_size=int(cfg.beam_size),
                    max_len=int(cfg.max_decode_len),
                    max_candidates=(None if int(cfg.decode_max_candidates) < 0 else int(max_candidates)),
                    candidate_policy=str(cfg.decode_candidate_policy),
                    include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
                    guided_dest_alpha=float(cfg.guided_dest_alpha),
                )

            # Metrics per route.
            xy_way = way_xy_m[int(city)]
            dpos_xy_m = _grid_yx_to_xy_m(dpos_b[:, 0], dpos_b[:, 1], meta=city_meta[int(city)])
            for bi in range(int(i1 - i0)):
                rid = int(rid_b[bi])
                gt = [int(x) for x in gt_b[bi]]
                g_samples = [[int(x) for x in greedy[int(bi) * int(K) + int(k)]] for k in range(int(K))]
                b_samples = (
                    [[int(x) for x in beam[int(bi) * int(K) + int(k)]] for k in range(int(K))] if beam is not None else None
                )

                gt_hops = int(max(0, len(gt) - 1))
                gt_len_m = _sum_way_len_m(way_len_m, gt)
                gt_xy = xy_way[np.asarray(gt, dtype=np.int64)]

                def _eval_pred(pred: List[int]) -> Dict[str, object]:
                    if not pred:
                        return {
                            "success": False,
                            "hit_wall": True,
                            "dead_end": False,
                            "has_loop": False,
                            "hops": 0,
                            "jaccard": float("nan"),
                            "len_m": float("nan"),
                            "len_ratio": float("nan"),
                            "dtw_m": float("nan"),
                            "frechet_m": float("nan"),
                            "final_error_m": float("nan"),
                        }
                    success = bool(int(pred[-1]) == int(dw_b[bi]))
                    max_len_hit = int(cfg.max_decode_len) + 1
                    hit_wall = bool((not success) and (len(pred) >= max_len_hit))
                    outdeg_last = (
                        int(ptr[int(pred[-1]) + 1] - ptr[int(pred[-1])]) if 0 <= int(pred[-1]) + 1 < int(ptr.size) else 0
                    )
                    dead_end = bool((not success) and (not hit_wall) and (outdeg_last == 0))
                    pred_len_m = _sum_way_len_m(way_len_m, pred)
                    pred_xy = xy_way[np.asarray(pred, dtype=np.int64)]
                    dtw_m = dtw_distance(pred_xy, gt_xy)
                    fre_m = frechet_distance(pred_xy, gt_xy)
                    # final error to destination position (meters, in same local frame)
                    last_xy = pred_xy[-1].astype(np.float64, copy=False)
                    err_m = float(np.linalg.norm(last_xy - dpos_xy_m[bi].astype(np.float64, copy=False)))
                    return {
                        "success": bool(success),
                        "hit_wall": bool(hit_wall),
                        "dead_end": bool(dead_end),
                        "has_loop": bool(_has_loop(pred)),
                        "hops": int(max(0, len(pred) - 1)),
                        "jaccard": float(_jaccard(gt, pred)),
                        "len_m": float(pred_len_m),
                        "len_ratio": float(pred_len_m / gt_len_m) if (math.isfinite(gt_len_m) and gt_len_m > 0 and math.isfinite(pred_len_m)) else float("nan"),
                        "dtw_m": float(dtw_m),
                        "frechet_m": float(fre_m),
                        "final_error_m": float(err_m),
                    }

                mg_list = [_eval_pred(p) for p in g_samples]
                if int(K) > 1:
                    k_sel = 0 if str(cfg.sample_select) == "first" else _best_sample_index(mg_list)
                    mg = dict(mg_list[int(k_sel)])
                    mg.update(
                        {
                            "n_samples": int(K),
                            "selected_k": int(k_sel),
                            "route_any_success": bool(any(bool(m.get("success", False)) for m in mg_list)),
                            "sample_success_rate": float(np.mean([1.0 if bool(m.get("success", False)) else 0.0 for m in mg_list])),
                            "sample_hit_wall_rate": float(np.mean([1.0 if bool(m.get("hit_wall", False)) else 0.0 for m in mg_list])),
                            "sample_dead_end_rate": float(np.mean([1.0 if bool(m.get("dead_end", False)) else 0.0 for m in mg_list])),
                            "sample_loop_rate": float(np.mean([1.0 if bool(m.get("has_loop", False)) else 0.0 for m in mg_list])),
                            "sample_dtw_m_mean": _nanmean([float(m.get("dtw_m", float("nan"))) for m in mg_list]),
                            "sample_frechet_m_mean": _nanmean([float(m.get("frechet_m", float("nan"))) for m in mg_list]),
                            "sample_len_ratio_mean": _nanmean([float(m.get("len_ratio", float("nan"))) for m in mg_list]),
                            "sample_final_error_m_mean": _nanmean([float(m.get("final_error_m", float("nan"))) for m in mg_list]),
                        }
                    )
                else:
                    mg = mg_list[0]

                mb: Optional[Dict[str, object]] = None
                if b_samples is not None:
                    mb_list = [_eval_pred(p) for p in b_samples]
                    if int(K) > 1:
                        k_sel = 0 if str(cfg.sample_select) == "first" else _best_sample_index(mb_list)
                        mb = dict(mb_list[int(k_sel)])
                        mb.update(
                            {
                                "n_samples": int(K),
                                "selected_k": int(k_sel),
                                "route_any_success": bool(any(bool(m.get("success", False)) for m in mb_list)),
                                "sample_success_rate": float(np.mean([1.0 if bool(m.get("success", False)) else 0.0 for m in mb_list])),
                                "sample_hit_wall_rate": float(np.mean([1.0 if bool(m.get("hit_wall", False)) else 0.0 for m in mb_list])),
                                "sample_dead_end_rate": float(np.mean([1.0 if bool(m.get("dead_end", False)) else 0.0 for m in mb_list])),
                                "sample_loop_rate": float(np.mean([1.0 if bool(m.get("has_loop", False)) else 0.0 for m in mb_list])),
                                "sample_dtw_m_mean": _nanmean([float(m.get("dtw_m", float("nan"))) for m in mb_list]),
                                "sample_frechet_m_mean": _nanmean([float(m.get("frechet_m", float("nan"))) for m in mb_list]),
                                "sample_len_ratio_mean": _nanmean([float(m.get("len_ratio", float("nan"))) for m in mb_list]),
                                "sample_final_error_m_mean": _nanmean([float(m.get("final_error_m", float("nan"))) for m in mb_list]),
                            }
                        )
                    else:
                        mb = mb_list[0]

                rec: Dict[str, Any] = {
                    "route_id": int(rid),
                    "city": int(city),
                    "gt_hops": int(gt_hops),
                    "gt_len_m": float(gt_len_m),
                    "gt_avg_way_len_m": float(gt_len_m / float(max(1, len(gt)))) if math.isfinite(gt_len_m) else float("nan"),
                    "dest_way_len_m": float(way_len_m[int(dw_b[bi])]) if 0 <= int(dw_b[bi]) < int(way_len_m.size) else float("nan"),
                    "greedy": mg,
                }
                if region_routes is not None:
                    rec["region_seq"] = [int(x) for x in region_routes[int(bi)]]
                if mb is not None:
                    rec["beam"] = mb
                per_route.append(rec)

        print(f"[city{int(city)}] done={int(pick.size)} routes")

    # Bin aggregation.
    def _agg(records: List[Dict[str, Any]], *, key: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for _lo, _hi, name in _hops_bins():
            out[str(name)] = {"n": 0, "success": [], "dtw_m": [], "frechet_m": [], "len_ratio": [], "final_error_m": [], "hit_wall": [], "dead_end": [], "has_loop": []}
        for r in records:
            hops = int(r.get("gt_hops", 0))
            lab = _bin_label(hops)
            cell = out[lab]
            cell["n"] += 1
            m = r.get(key, {}) if isinstance(r.get(key, {}), dict) else {}
            cell["success"].append(1.0 if bool(m.get("success", False)) else 0.0)
            cell["dtw_m"].append(float(m.get("dtw_m", float("nan"))))
            cell["frechet_m"].append(float(m.get("frechet_m", float("nan"))))
            cell["len_ratio"].append(float(m.get("len_ratio", float("nan"))))
            cell["final_error_m"].append(float(m.get("final_error_m", float("nan"))))
            cell["hit_wall"].append(1.0 if bool(m.get("hit_wall", False)) else 0.0)
            cell["dead_end"].append(1.0 if bool(m.get("dead_end", False)) else 0.0)
            cell["has_loop"].append(1.0 if bool(m.get("has_loop", False)) else 0.0)

        # summarize
        rep: Dict[str, Any] = {"bins": [b[2] for b in _hops_bins()], "cells": {}}
        for lab, cell in out.items():
            n = int(cell["n"])
            rep["cells"][lab] = {
                "n": int(n),
                "success_rate": float(np.mean(np.asarray(cell["success"], dtype=np.float64))) if n else float("nan"),
                "dtw_m": summarize(cell["dtw_m"]),
                "frechet_m": summarize(cell["frechet_m"]),
                "len_ratio": summarize(cell["len_ratio"]),
                "final_error_m": summarize(cell["final_error_m"]),
                "hit_wall_rate": float(np.mean(np.asarray(cell["hit_wall"], dtype=np.float64))) if n else float("nan"),
                "dead_end_rate": float(np.mean(np.asarray(cell["dead_end"], dtype=np.float64))) if n else float("nan"),
                "loop_rate": float(np.mean(np.asarray(cell["has_loop"], dtype=np.float64))) if n else float("nan"),
            }
        return rep

    # per-city split
    per_city: List[Dict[str, Any]] = []
    for city in cities_obs:
        recs = [r for r in per_route if int(r.get("city", -1)) == int(city)]
        city_out: Dict[str, Any] = {"city": int(city), "n": int(len(recs)), "greedy": _agg(recs, key="greedy")}
        if bool(cfg.compare_beam):
            city_out["beam"] = _agg(recs, key="beam")
            # Δsuccess per bin
            delta: Dict[str, Any] = {"bins": city_out["greedy"]["bins"], "cells": {}}
            for lab in city_out["greedy"]["cells"]:
                g = city_out["greedy"]["cells"][lab]
                b = city_out["beam"]["cells"][lab]
                delta["cells"][lab] = {"delta_success_rate": float(b["success_rate"]) - float(g["success_rate"])}
            city_out["beam_gain"] = delta
        per_city.append(city_out)

    overall: Dict[str, Any] = {"n": int(len(per_route)), "greedy": _agg(per_route, key="greedy")}
    if bool(cfg.compare_beam):
        overall["beam"] = _agg(per_route, key="beam")
        delta = {"bins": overall["greedy"]["bins"], "cells": {}}
        for lab in overall["greedy"]["cells"]:
            g = overall["greedy"]["cells"][lab]
            b = overall["beam"]["cells"][lab]
            delta["cells"][lab] = {"delta_success_rate": float(b["success_rate"]) - float(g["success_rate"])}
        overall["beam_gain"] = delta

    out = {
        "ok": True,
        "task": "way_casd_binned_eval",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
            "flow_ckpt": (str(args.flow_ckpt) if args.flow_ckpt is not None else None),
            "way_regions_npz": (str(args.way_regions_npz) if args.way_regions_npz is not None else None),
            "region_ar_ckpt": (str(args.region_ar_ckpt) if args.region_ar_ckpt is not None else None),
            "city_grid_meta": {str(int(k)): str(v) for k, v in sorted(city_meta_src.items(), key=lambda kv: int(kv[0]))},
        },
        "ckpt_strict_load_ok": bool(strict_ok),
        "ae_cfg_inferred": {
            "decoder_use_dest_dist": bool(use_dest_dist),
            "decoder_use_cross_attn": bool(use_cross_attn),
            "decoder_use_step_emb": bool(use_step_emb),
            "decoder_use_dest_query": bool(use_dest_query),
            "decoder_use_dir_query": bool(use_dir_query),
            "decoder_use_cand_query": bool(use_cand_query),
            "decoder_use_cand_contrast": bool(use_cand_contrast),
            "decoder_use_past_context": bool(use_past_context),
            "decoder_past_k": int(past_k),
            "decoder_past_n_layers": int(past_n_layers),
            "decoder_past_n_heads": int(past_n_heads),
            "n_route_cities": int(n_route_cities),
        },
        "flow_cfg_inferred": (
            {
                "d_model": int(flow_cfg_dict.get("d_model", ae.cfg.d_model)) if isinstance(flow_cfg_dict, dict) else int(ae.cfg.d_model),
                "n_latent": int(flow_cfg_dict.get("n_latent", ae.cfg.n_latent)) if isinstance(flow_cfg_dict, dict) else int(ae.cfg.n_latent),
                "n_layers": int(flow_cfg_dict.get("n_layers", 6)) if isinstance(flow_cfg_dict, dict) else 6,
                "n_heads": int(flow_cfg_dict.get("n_heads", 8)) if isinstance(flow_cfg_dict, dict) else 8,
                "dropout": float(flow_cfg_dict.get("dropout", 0.1)) if isinstance(flow_cfg_dict, dict) else 0.1,
                "noise_sigma": float(flow_cfg_dict.get("noise_sigma", 1.0)) if isinstance(flow_cfg_dict, dict) else 1.0,
                "solver_steps": int(flow_cfg_dict.get("solver_steps", 20)) if isinstance(flow_cfg_dict, dict) else 20,
            }
            if flow is not None
            else None
        ),
        "per_city": per_city,
        "overall": overall,
        "notes": {
            "shape_metric": "DTW/Fréchet on way-center sequences (meters, equirectangular projection from osm_road_prob_meta.json bbox).",
            "bins": [b[2] for b in _hops_bins()],
            "latent_source": "gt=oracle (GT->AE.encode->Decoder); flow=Flow.sample->Decoder (generation).",
            "sample_select": "When n_samples_per_route>1: first=use sample0; best=prefer success then min(DTW, Fréchet).",
            "region_constraint": "If enabled: use Region seq to filter way candidates by target region (modes: strict/relaxed; fallback: unconstrained/dest_region/stop).",
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {out_json}")

    if args.out_per_route_json is not None:
        out_rec = Path(args.out_per_route_json)
        out_rec.parent.mkdir(parents=True, exist_ok=True)
        out_rec.write_text(
            json.dumps(
                {
                    "ok": True,
                    "task": str(out.get("task", "")),
                    "created_at": str(out.get("created_at", "")),
                    "cfg": dict(out.get("cfg", {})),
                    "inputs": dict(out.get("inputs", {})),
                    "per_route": per_route,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[OK] saved: {out_rec}")


if __name__ == "__main__":
    main()
