#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Allow running as a file: `python tools/xxx.py ...` (so that `import src.*` works).
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz
from src.plot_style import OKABE_ITO, add_panel_label, paper_style, save_figure

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[FATAL] file not found: {path}")
    if not path.is_file():
        raise SystemExit(f"[FATAL] not a file: {path}")


def _hour_from_unix(start_t: int, tz_offset_hours: float) -> int:
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    sec = int((int(start_t) + tz_sec) % 86400)
    return int(sec // 3600)


def _dow_from_unix(start_t: int, tz_offset_hours: float) -> int:
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    days = int((int(start_t) + tz_sec) // 86400)
    return int((days + 3) % 7)


def _first_repeat_step(seq: List[int]) -> Optional[int]:
    seen: set[int] = set()
    for i, x in enumerate(seq):
        xx = int(x)
        if xx in seen:
            return int(i)
        seen.add(xx)
    return None


def _slice_csr(ptr: np.ndarray, idx: np.ndarray, u: int) -> np.ndarray:
    s = int(ptr[u])
    e = int(ptr[u + 1])
    if e <= s:
        return np.asarray([], dtype=np.int64)
    return np.asarray(idx[s:e], dtype=np.int64)


def _shortest_hops_bfs(ptr: np.ndarray, idx: np.ndarray, start: int, dest: int, *, max_visits: int = 200000) -> Optional[int]:
    n = int(ptr.size) - 1
    s = int(start)
    d = int(dest)
    if s < 0 or s >= n or d < 0 or d >= n:
        return None
    if s == d:
        return 0

    # KISS: plain BFS (n small per query); enough for 6 cases.
    from collections import deque

    q = deque([(s, 0)])
    visited = set([s])
    seen = 0
    while q:
        u, dist = q.popleft()
        for v in _slice_csr(ptr, idx, int(u)).tolist():
            vv = int(v)
            if vv == d:
                return int(dist + 1)
            if vv not in visited:
                visited.add(vv)
                q.append((vv, int(dist + 1)))
        seen += 1
        if seen >= int(max_visits):
            return None
    return None


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


@dataclass(frozen=True)
class Case:
    city: int
    category: str  # easy | recovered | hard
    route_id: int

    gt_way_ids: List[int]
    greedy_way_ids: List[int]
    beam_way_ids: List[int]

    greedy_success: bool
    beam_success: bool
    greedy_hit_wall: bool
    beam_hit_wall: bool

    gt_hops: int
    shortest_hops: Optional[int]
    gt_detour_over_shortest: Optional[float]

    greedy_has_loop: bool
    beam_has_loop: bool

    start_pos_yx: Tuple[float, float]
    dest_pos_yx: Tuple[float, float]


def _build_city_index(rep: dict) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for c in rep.get("per_city", []) or []:
        if not isinstance(c, dict):
            continue
        city = int(c.get("city", -1))
        succ = [int(x) for x in (c.get("success_route_ids") or [])]
        fails = [f for f in (c.get("failures") or []) if isinstance(f, dict) and f.get("route_id") is not None]
        fail_ids = [int(f.get("route_id")) for f in fails]
        fail_by_id = {int(f["route_id"]): f for f in fails}
        out[city] = {"success_ids": succ, "fail_ids": fail_ids, "fail_by_id": fail_by_id}
    return out


@torch.no_grad()
def _decode_one_oracle(
    *,
    ae: WayCASDAutoEncoder,
    routes,
    ptr_np: np.ndarray,
    idx_np: np.ndarray,
    route_id: int,
    device: torch.device,
    tz_offset_hours: float,
    decode_max_candidates: int,
    decode_candidate_policy: str,
    decode_include_dest_if_successor: bool,
    guided_dest_alpha: float,
    max_decode_len: int,
    beam_size: int,
) -> Case:
    rid = int(route_id)
    city = int(routes.route_city[rid])
    L = int(routes.way_seq_len[rid])
    s = int(routes.way_seq_ptr[rid])
    gt = routes.way_seq_idx[s : s + L].astype(np.int64, copy=False).tolist()
    gt = [int(x) for x in gt]

    start_way = int(routes.start_way[rid])
    dest_way = int(routes.dest_way[rid])
    start_pos = routes.start_pos[rid].astype(np.float64, copy=False).reshape(2)
    dest_pos = routes.dest_pos[rid].astype(np.float64, copy=False).reshape(2)
    start_t = int(routes.start_t[rid])
    hour = int(_hour_from_unix(start_t, float(tz_offset_hours)))
    dow = int(_dow_from_unix(start_t, float(tz_offset_hours)))

    way_pad = np.full((1, L), -1, dtype=np.int64)
    way_pad[0, :L] = np.asarray(gt, dtype=np.int64)
    way_pad_t = torch.as_tensor(way_pad, dtype=torch.long, device=device)
    z_enc, _ = ae.encode(way_pad_t)

    route_cond = {
        "start_pos": torch.as_tensor(start_pos[None, :], dtype=torch.float32, device=device),
        "dest_pos": torch.as_tensor(dest_pos[None, :], dtype=torch.float32, device=device),
        "hour": torch.as_tensor(np.asarray([hour], dtype=np.int64), dtype=torch.long, device=device),
        "dow": torch.as_tensor(np.asarray([dow], dtype=np.int64), dtype=torch.long, device=device),
        "route_city": torch.as_tensor(np.asarray([int(city)], dtype=np.int64), dtype=torch.long, device=device),
    }
    sw_t = torch.as_tensor(np.asarray([start_way], dtype=np.int64), dtype=torch.long, device=device)
    dw_t = torch.as_tensor(np.asarray([dest_way], dtype=np.int64), dtype=torch.long, device=device)

    max_cand = None if int(decode_max_candidates) < 0 else int(decode_max_candidates)

    greedy = ae.decoder.greedy_decode(
        way_embedder=ae.way_enc,
        latent_tokens=z_enc,
        route_cond=route_cond,
        start_way=sw_t,
        dest_way=dw_t,
        max_len=int(max_decode_len),
        max_candidates=max_cand,
        candidate_policy=str(decode_candidate_policy),
        include_dest_if_successor=bool(decode_include_dest_if_successor),
        guided_dest_alpha=float(guided_dest_alpha),
    )[0]
    beam = ae.decoder.beam_search(
        way_embedder=ae.way_enc,
        latent_tokens=z_enc,
        route_cond=route_cond,
        start_way=sw_t,
        dest_way=dw_t,
        beam_size=int(beam_size),
        max_len=int(max_decode_len),
        max_candidates=max_cand,
        candidate_policy=str(decode_candidate_policy),
        include_dest_if_successor=bool(decode_include_dest_if_successor),
        guided_dest_alpha=float(guided_dest_alpha),
    )[0]

    greedy_ids = [int(x) for x in greedy]
    beam_ids = [int(x) for x in beam]
    greedy_success = bool(greedy_ids and int(greedy_ids[-1]) == int(dest_way))
    beam_success = bool(beam_ids and int(beam_ids[-1]) == int(dest_way))
    greedy_hit_wall = bool((not greedy_success) and (len(greedy_ids) >= int(max_decode_len) + 1))
    beam_hit_wall = bool((not beam_success) and (len(beam_ids) >= int(max_decode_len) + 1))

    gt_hops = max(0, int(len(gt)) - 1)
    shortest_hops = _shortest_hops_bfs(ptr_np, idx_np, int(start_way), int(dest_way))
    detour = None
    if shortest_hops is not None and int(shortest_hops) > 0:
        detour = float(gt_hops) / float(shortest_hops)

    return Case(
        city=int(city),
        category="",
        route_id=int(rid),
        gt_way_ids=gt,
        greedy_way_ids=greedy_ids,
        beam_way_ids=beam_ids,
        greedy_success=bool(greedy_success),
        beam_success=bool(beam_success),
        greedy_hit_wall=bool(greedy_hit_wall),
        beam_hit_wall=bool(beam_hit_wall),
        gt_hops=int(gt_hops),
        shortest_hops=(int(shortest_hops) if shortest_hops is not None else None),
        gt_detour_over_shortest=(float(detour) if detour is not None else None),
        greedy_has_loop=bool(_first_repeat_step(greedy_ids) is not None),
        beam_has_loop=bool(_first_repeat_step(beam_ids) is not None),
        start_pos_yx=(float(start_pos[0]), float(start_pos[1])),
        dest_pos_yx=(float(dest_pos[0]), float(dest_pos[1])),
    )


def _seq_to_xy(seq: List[int], *, way_center_x: np.ndarray, way_center_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cx = np.asarray(way_center_x, dtype=np.float64).reshape(-1)
    cy = np.asarray(way_center_y, dtype=np.float64).reshape(-1)
    s = np.asarray([int(x) for x in seq], dtype=np.int64)
    if s.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    s = np.clip(s, 0, cx.shape[0] - 1)
    return cx[s], cy[s]


def _compute_bbox(xs: List[np.ndarray], ys: List[np.ndarray], *, pad_frac: float = 0.2) -> Tuple[float, float, float, float]:
    x = np.concatenate([a.reshape(-1) for a in xs if a.size > 0], axis=0)
    y = np.concatenate([a.reshape(-1) for a in ys if a.size > 0], axis=0)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)
    px = float(pad_frac) * dx
    py = float(pad_frac) * dy
    return xmin - px, xmax + px, ymin - py, ymax + py


def _pick_cases(
    *,
    routes,
    ptr_np: np.ndarray,
    idx_np: np.ndarray,
    ae: WayCASDAutoEncoder,
    device: torch.device,
    greedy_rep: dict,
    beam10_rep: dict,
    city_names: Dict[int, str],
    beam_size: int,
    out_dir: Path,
) -> List[Case]:
    g = _build_city_index(greedy_rep)
    b = _build_city_index(beam10_rep)

    cfg = greedy_rep.get("cfg") or {}
    tz_offset_hours = float(cfg.get("tz_offset_hours", -5.0))
    decode_max_candidates = int(cfg.get("decode_max_candidates", 0))
    decode_candidate_policy = str(cfg.get("decode_candidate_policy", "first"))
    decode_include_dest_if_successor = bool(cfg.get("decode_include_dest_if_successor", False))
    guided_dest_alpha = float(cfg.get("decode_guided_dest_alpha", 0.0))
    max_decode_len = int(cfg.get("max_decode_len", 160))

    out_cases: List[Case] = []
    meta: Dict[str, Any] = {
        "ok": True,
        "task": "waycasd_plot_city_micro_case_study",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "eval_cfg": cfg,
            "beam_size": int(beam_size),
        },
        "picked": [],
    }

    def _try_pick(route_ids: List[int], *, want: str, city: int) -> Case:
        for rid in route_ids:
            case = _decode_one_oracle(
                ae=ae,
                routes=routes,
                ptr_np=ptr_np,
                idx_np=idx_np,
                route_id=int(rid),
                device=device,
                tz_offset_hours=float(tz_offset_hours),
                decode_max_candidates=int(decode_max_candidates),
                decode_candidate_policy=str(decode_candidate_policy),
                decode_include_dest_if_successor=bool(decode_include_dest_if_successor),
                guided_dest_alpha=float(guided_dest_alpha),
                max_decode_len=int(max_decode_len),
                beam_size=int(beam_size),
            )
            if int(case.city) != int(city):
                continue
            if want == "easy" and bool(case.greedy_success):
                return case
            if want == "recovered" and (not bool(case.greedy_success)) and bool(case.beam_success):
                return case
            if want == "hard" and (not bool(case.beam_success)):
                return case
        raise SystemExit(f"[FATAL] cannot find a valid '{want}' case for city={city}.")

    for city in (0, 1):
        city_name = city_names.get(int(city), f"city{int(city)}")

        # Easy: greedy success & short (<30 hops), else shortest among successes.
        succ = [int(x) for x in g.get(int(city), {}).get("success_ids", [])]
        succ = [rid for rid in succ if int(routes.way_seq_len[int(rid)]) > 1]
        succ_short = [rid for rid in succ if int(routes.way_seq_len[int(rid)]) - 1 < 30]
        succ_short.sort(key=lambda rid: int(routes.way_seq_len[int(rid)]))
        succ.sort(key=lambda rid: int(routes.way_seq_len[int(rid)]))
        easy_candidates = succ_short if succ_short else succ
        easy = _try_pick(easy_candidates[:50], want="easy", city=int(city))
        easy = Case(**{**asdict(easy), "category": "easy"})
        out_cases.append(easy)

        # Recovered: greedy fail + beam10 success. Prefer greedy hit_wall + early diverge + low jaccard.
        g_fail_by_id = g.get(int(city), {}).get("fail_by_id", {})
        g_fail_ids = [int(x) for x in g.get(int(city), {}).get("fail_ids", [])]
        b_succ_set = set(int(x) for x in b.get(int(city), {}).get("success_ids", []))
        rec = [rid for rid in g_fail_ids if int(rid) in b_succ_set]

        def _rec_key(rid: int) -> Tuple[int, int, float]:
            f = g_fail_by_id.get(int(rid), {})
            hit_wall = bool(f.get("hit_wall", False))
            div = int(f.get("diverge_step", 10**9))
            jac = float(f.get("jaccard", 1.0))
            return (0 if hit_wall else 1, div, jac)

        rec.sort(key=_rec_key)
        recovered = _try_pick(rec[:80], want="recovered", city=int(city))
        recovered = Case(**{**asdict(recovered), "category": "recovered"})
        out_cases.append(recovered)

        # Hard: beam10 fail. Prefer larger gt_len.
        b_fail_by_id = b.get(int(city), {}).get("fail_by_id", {})
        b_fail_ids = [int(x) for x in b.get(int(city), {}).get("fail_ids", [])]

        def _hard_key(rid: int) -> Tuple[int, int]:
            f = b_fail_by_id.get(int(rid), {})
            gt_len = int(f.get("gt_len", 0))
            pred_len = int(f.get("pred_len", 0))
            return (-gt_len, -pred_len)

        b_fail_ids.sort(key=_hard_key)
        hard = _try_pick(b_fail_ids[:50], want="hard", city=int(city))
        hard = Case(**{**asdict(hard), "category": "hard"})
        out_cases.append(hard)

        meta["picked"].append(
            {
                "city": int(city),
                "city_name": str(city_name),
                "easy_route_id": int(easy.route_id),
                "recovered_route_id": int(recovered.route_id),
                "hard_route_id": int(hard.route_id),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "micro_case_study_selected_routes.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_cases


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WayCASD micro case study (GT vs Greedy vs Beam10) in city space.")
    p.add_argument("--eval_dir", type=Path, required=True, help="Strong-ckpt eval dir (contains oracle_decode_greedy/beam10 json).")
    p.add_argument("--out_dir", type=Path, default=Path("_sync/wsa/paper_figures/waycasd_v1/micro"))
    p.add_argument("--style", type=str, choices=["paper"], default="paper")
    p.add_argument("--cases_per_city", type=int, default=3, help="PI default: 3 (easy/recovered/hard).")

    p.add_argument("--greedy_json", type=Path, default=None)
    p.add_argument("--beam10_json", type=Path, default=None)

    p.add_argument("--way_routes_npz", type=Path, default=None)
    p.add_argument("--way_graph_npz", type=Path, default=None)
    p.add_argument("--way_features_npz", type=Path, default=None)
    p.add_argument("--ae_ckpt", type=Path, default=None)

    p.add_argument("--beam_size", type=int, default=10)
    p.add_argument("--city_name", type=str, nargs="*", default=["0:Detroit", "1:Columbus"])

    p.add_argument("--pad_frac", type=float, default=0.2, help="Plot bbox padding for per-case view.")
    p.add_argument("--bg_alpha", type=float, default=0.4)
    p.add_argument("--bg_s", type=float, default=0.8, help="Way-center background marker size.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if int(args.cases_per_city) != 3:
        raise SystemExit("[FATAL] This script currently supports cases_per_city=3 only (easy/recovered/hard).")
    eval_dir = Path(args.eval_dir)

    greedy_json = Path(args.greedy_json) if args.greedy_json is not None else (eval_dir / "oracle_decode_greedy_n200.json")
    beam10_json = Path(args.beam10_json) if args.beam10_json is not None else (eval_dir / "oracle_decode_beam10_n200.json")
    _require_file(greedy_json)
    _require_file(beam10_json)

    greedy_rep = _read_json(greedy_json)
    beam10_rep = _read_json(beam10_json)

    # Inputs: default from greedy json, can override explicitly.
    inputs = greedy_rep.get("inputs") or {}
    way_routes_npz = Path(args.way_routes_npz) if args.way_routes_npz is not None else Path(inputs["way_routes_npz"])
    way_graph_npz = Path(args.way_graph_npz) if args.way_graph_npz is not None else Path(inputs["way_graph_npz"])
    way_features_npz = Path(args.way_features_npz) if args.way_features_npz is not None else Path(inputs["way_features_npz"])
    ae_ckpt = Path(args.ae_ckpt) if args.ae_ckpt is not None else Path(inputs["ae_ckpt"])
    for p in (way_routes_npz, way_graph_npz, way_features_npz, ae_ckpt):
        _require_file(p)

    city_names: Dict[int, str] = {}
    for raw in args.city_name or []:
        if ":" not in str(raw):
            continue
        k, v = str(raw).split(":", 1)
        try:
            city_names[int(k)] = v.strip()
        except Exception:
            continue
    city_names.setdefault(0, "Detroit")
    city_names.setdefault(1, "Columbus")

    device = torch.device("cuda" if (torch.cuda.is_available() and str(greedy_rep.get("cfg", {}).get("device", "cuda")) == "cuda") else "cpu")

    # Load data
    routes = load_way_routes_npz(Path(way_routes_npz))
    wg = np.load(str(way_graph_npz), allow_pickle=True)
    wf = np.load(str(way_features_npz), allow_pickle=True)
    ptr_np = np.asarray(wg["way_adj_ptr"], dtype=np.int64)
    idx_np = np.asarray(wg["way_adj_idx"], dtype=np.int64)
    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)

    # Build AE (infer decoder cfg from state dict; keep consistent with eval scripts).
    way_features = load_way_features_from_npz(Path(way_features_npz), device=device)
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1

    ckpt = torch.load(str(ae_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ae_cfg_dict = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    use_dest_dist = _infer_use_dest_dist(state) if isinstance(state, dict) else True
    use_cand_contrast = (_infer_use_cand_contrast(state) if isinstance(state, dict) else False) or bool(ae_cfg_dict.get("decoder_use_cand_contrast", False))
    use_cross_attn = (_infer_bool_by_prefix(state, "decoder.cross_attn.") if isinstance(state, dict) else False) or bool(ae_cfg_dict.get("decoder_use_cross_attn", True))
    use_step_emb = (_infer_bool_by_prefix(state, "decoder.step_emb.") if isinstance(state, dict) else False) or bool(ae_cfg_dict.get("decoder_use_step_emb", False))
    use_dest_query = (_infer_bool_by_prefix(state, "decoder.dest_proj.") if isinstance(state, dict) else False) or bool(ae_cfg_dict.get("decoder_use_dest_query", False))
    use_dir_query = (_infer_bool_by_prefix(state, "decoder.dir_query_proj.") if isinstance(state, dict) else False) or bool(ae_cfg_dict.get("decoder_use_dir_query", False))
    use_cand_query = (_infer_bool_by_prefix(state, "decoder.cand_query_proj.") if isinstance(state, dict) else False) or bool(ae_cfg_dict.get("decoder_use_cand_query", False))
    use_past_context = (_infer_bool_by_prefix(state, "decoder.past_encoder.") if isinstance(state, dict) else False) or bool(ae_cfg_dict.get("decoder_use_past_context", False))
    past_k = int(ae_cfg_dict.get("decoder_past_k", 8))
    if use_past_context and isinstance(state, dict):
        pe = state.get("decoder.past_encoder.pos_emb.weight", None)
        if isinstance(pe, torch.Tensor) and pe.ndim == 2 and int(pe.shape[0]) > 0:
            past_k = int(pe.shape[0])
    past_n_layers = int(ae_cfg_dict.get("decoder_past_n_layers", 2))
    past_n_heads = int(ae_cfg_dict.get("decoder_past_n_heads", 4))

    n_route_cities = _infer_n_route_cities(state) if isinstance(state, dict) else None
    if n_route_cities is None:
        n_route_cities = int(ae_cfg_dict.get("n_route_cities", 4))
    n_city_obs = int(np.max(routes.route_city.astype(np.int64))) + 1
    n_route_cities = max(int(n_route_cities), int(n_city_obs))

    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(ae_cfg_dict.get("d_model", 256)),
            n_latent=int(ae_cfg_dict.get("n_latent", 64)),
            n_heads=int(ae_cfg_dict.get("n_heads", 8)),
            dropout=float(ae_cfg_dict.get("dropout", 0.1)),
            max_candidates=int(ae_cfg_dict.get("max_candidates", 32)),
            max_len=int(ae_cfg_dict.get("max_len", greedy_rep.get("cfg", {}).get("max_way_len", 160))),
            coord_scale=float(ae_cfg_dict.get("coord_scale", 1024.0)),
            decoder_use_dest_dist=bool(use_dest_dist),
            decoder_use_cross_attn=bool(use_cross_attn),
            decoder_n_cross_heads=int(ae_cfg_dict.get("decoder_n_cross_heads", 4)),
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
        way_adj_ptr=ptr_np,
        way_adj_idx=idx_np,
        n_highway_types=int(max(4, n_highway_types)),
        n_route_cities=int(n_route_cities),
    ).to(device)
    ae.eval()
    try:
        ae.load_state_dict(state, strict=True)
    except Exception:
        ae.load_state_dict(state, strict=False)

    out_dir = Path(args.out_dir)
    cases = _pick_cases(
        routes=routes,
        ptr_np=ptr_np,
        idx_np=idx_np,
        ae=ae,
        device=device,
        greedy_rep=greedy_rep,
        beam10_rep=beam10_rep,
        city_names=city_names,
        beam_size=int(args.beam_size),
        out_dir=out_dir,
    )

    # Plot: 2 cities x 3 categories.
    city_order = [0, 1]
    cat_order = ["easy", "recovered", "hard"]
    case_by_key = {(int(c.city), str(c.category)): c for c in cases}

    with paper_style():
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12.6, 7.6), constrained_layout=True)
        panel = 0
        legend_handles = None

        for r, city in enumerate(city_order):
            for cc, cat in enumerate(cat_order):
                ax = axes[r, cc]
                c = case_by_key[(int(city), str(cat))]

                gt_x, gt_y = _seq_to_xy(c.gt_way_ids, way_center_x=way_center_x, way_center_y=way_center_y)
                g_x, g_y = _seq_to_xy(c.greedy_way_ids, way_center_x=way_center_x, way_center_y=way_center_y)
                b_x, b_y = _seq_to_xy(c.beam_way_ids, way_center_x=way_center_x, way_center_y=way_center_y)

                sx, sy = float(c.start_pos_yx[1]), float(c.start_pos_yx[0])
                dx, dy = float(c.dest_pos_yx[1]), float(c.dest_pos_yx[0])

                xmin, xmax, ymin, ymax = _compute_bbox([gt_x, g_x, b_x, np.asarray([sx, dx])], [gt_y, g_y, b_y, np.asarray([sy, dy])], pad_frac=float(args.pad_frac))
                mask = (way_center_x >= xmin) & (way_center_x <= xmax) & (way_center_y >= ymin) & (way_center_y <= ymax)
                ax.scatter(
                    way_center_x[mask],
                    way_center_y[mask],
                    s=float(args.bg_s),
                    c="#DDDDDD",
                    alpha=float(args.bg_alpha),
                    linewidths=0,
                    zorder=1,
                )

                # Trajectories
                ln_gt = ax.plot(gt_x, gt_y, color=OKABE_ITO["blue"], lw=2.4, ls="-", label="GT", zorder=3)[0]
                ln_g = ax.plot(g_x, g_y, color=OKABE_ITO["vermillion"], lw=2.0, ls="--", label="Greedy", zorder=4)[0]
                ln_b = ax.plot(b_x, b_y, color=OKABE_ITO["bluish_green"], lw=2.0, ls="-.", label="Beam-10", zorder=5)[0]

                # O/D markers
                mk_o = ax.scatter([sx], [sy], s=80, c="#000000", marker="o", edgecolors="white", linewidths=0.8, zorder=6, label="O")
                mk_d = ax.scatter([dx], [dy], s=90, c="#000000", marker="*", edgecolors="white", linewidths=0.8, zorder=6, label="D")

                if legend_handles is None:
                    legend_handles = [ln_gt, ln_g, ln_b, mk_o, mk_d]

                city_name = city_names.get(int(city), f"city{int(city)}")
                cat_title = {"easy": "Easy", "recovered": "Recovered", "hard": "Hard"}[str(cat)]
                ax.set_title(f"{city_name} · {cat_title} (rid={c.route_id})")

                # Hard-case annotation (only).
                if str(cat) == "hard":
                    det = c.gt_detour_over_shortest
                    det_s = f"{det:.2f}×" if det is not None else "n/a"
                    loop_s = "Yes" if bool(c.beam_has_loop) else "No"
                    ax.text(
                        0.02,
                        0.02,
                        f"GT detour: {det_s} shortest\nBeam loop: {loop_s}",
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=9,
                        color="#222222",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
                        zorder=10,
                    )

                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_aspect("equal", adjustable="box")
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])

                add_panel_label(ax, chr(ord("a") + panel))
                panel += 1

        if legend_handles is not None:
            fig.legend(
                handles=legend_handles,
                labels=["GT", "Greedy", "Beam-10", "O", "D"],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=5,
                frameon=False,
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = out_dir / "waycasd_city_micro_case_study.pdf"
        out_png = out_dir / "waycasd_city_micro_case_study.png"
        save_figure(fig, out_pdf)
        save_figure(fig, out_png, dpi=300)
        plt.close(fig)

    # Save resolved case data (includes sequences for reproducibility).
    meta2 = {
        "ok": True,
        "task": "waycasd_city_micro_case_study_cases",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "eval_dir": str(eval_dir),
        "greedy_json": str(greedy_json),
        "beam10_json": str(beam10_json),
        "inputs": {
            "way_routes_npz": str(way_routes_npz),
            "way_graph_npz": str(way_graph_npz),
            "way_features_npz": str(way_features_npz),
            "ae_ckpt": str(ae_ckpt),
        },
        "cases": [asdict(c) for c in cases],
    }
    (out_dir / "waycasd_city_micro_case_study_cases.json").write_text(
        json.dumps(meta2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[OK] saved {out_pdf}")
    print(f"[OK] saved {out_png}")


if __name__ == "__main__":
    main()
