from __future__ import annotations

"""
Train a lightweight value function V(cur_way, z, dest) from existing decode trajectories.

This is meant for decode-time lookahead scoring to reduce hit_wall / loop:

  score(next_way) = logp_decoder(next_way) + beta * V(next_way, z, dest)

Training supervision:
  For each route, we take the model-predicted way sequence (dump_way_seqs output),
  and label ALL visited states by the route-level success (reached dest).
"""

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.models.way_casd.latent_flow import LatentFlowCfg, LatentFlowMatching
from src.models.way_casd.value_fn import WayValueFn, WayValueFnCfg
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz
from src.utils.time_unix import dow_from_unix, hour_from_unix

TZ_SHANGHAI = timezone(timedelta(hours=8))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _infer_bool_by_prefix(state: Dict[str, torch.Tensor], prefix: str) -> bool:
    return any(str(k).startswith(prefix) for k in state.keys())


def _infer_decoder_past_k_from_state(state: Dict[str, torch.Tensor]) -> int:
    pe = state.get("decoder.past_encoder.pos_emb.weight", None)
    if not isinstance(pe, torch.Tensor) or pe.ndim != 2:
        return 8
    return int(pe.shape[0])


def _load_ae(*, ae_ckpt: Path, way_graph_npz: Path, way_features_npz: Path, device: torch.device) -> WayCASDAutoEncoder:
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
    use_cross_attn = bool(cfg_dict.get("decoder_use_cross_attn", True)) or _infer_bool_by_prefix(state, "decoder.cross_attn.")
    use_step_emb = bool(cfg_dict.get("decoder_use_step_emb", False)) or _infer_bool_by_prefix(state, "decoder.step_emb.")
    use_dest_query = bool(cfg_dict.get("decoder_use_dest_query", False)) or _infer_bool_by_prefix(state, "decoder.dest_proj.")
    use_dir_query = bool(cfg_dict.get("decoder_use_dir_query", False)) or _infer_bool_by_prefix(state, "decoder.dir_query_proj.")
    use_cand_query = bool(cfg_dict.get("decoder_use_cand_query", False)) or _infer_bool_by_prefix(state, "decoder.cand_query_proj.")
    use_cand_contrast = bool(cfg_dict.get("decoder_use_cand_contrast", False))
    use_past_ctx = bool(cfg_dict.get("decoder_use_past_context", False)) or _infer_bool_by_prefix(state, "decoder.past_encoder.")
    past_k = int(cfg_dict.get("decoder_past_k", _infer_decoder_past_k_from_state(state)))

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
        if s:
            pad[i, : len(s)] = torch.as_tensor(s, dtype=torch.long, device=device)
    return pad


class _StateDataset(Dataset):
    def __init__(self, *, route_idx: np.ndarray, cur_way: np.ndarray, y: np.ndarray) -> None:
        self.route_idx = np.asarray(route_idx, dtype=np.int64).reshape(-1)
        self.cur_way = np.asarray(cur_way, dtype=np.int64).reshape(-1)
        self.y = np.asarray(y, dtype=np.float32).reshape(-1)
        if int(self.route_idx.size) != int(self.cur_way.size) or int(self.y.size) != int(self.cur_way.size):
            raise ValueError("StateDataset: size mismatch")

    def __len__(self) -> int:
        return int(self.cur_way.size)

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        return {"route_idx": np.asarray(int(self.route_idx[int(i)]), dtype=np.int64), "cur_way": np.asarray(int(self.cur_way[int(i)]), dtype=np.int64), "y": np.asarray(float(self.y[int(i)]), dtype=np.float32)}


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
    latent_source: str
    flow_solver_steps: Optional[int]
    max_states_per_route: int
    save_every: int


def main() -> None:
    p = argparse.ArgumentParser(description="Train WayValueFn from per_route decode trajectories.")
    p.add_argument("--per_route_json", type=Path, required=True, help="Output json with per_route + pred_way_ids (dump_way_seqs).")
    p.add_argument("--pred_key", type=str, default="beam", choices=["greedy", "beam"], help="Which decode to use as trajectories.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--flow_ckpt", type=Path, default=None, help="Required when --latent_source=flow.")
    p.add_argument("--way_regions_npz", type=Path, default=None, help="Required when Flow uses region_seq.")
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--latent_source", type=str, default="flow", choices=["gt", "flow"])
    p.add_argument("--flow_solver_steps", type=int, default=0, help="Override solver steps (0=use ckpt).")
    p.add_argument("--max_states_per_route", type=int, default=64, help="Subsample visited states per route (0=all).")
    p.add_argument("--save_every", type=int, default=1)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        latent_source=str(args.latent_source),
        flow_solver_steps=(int(args.flow_solver_steps) if int(args.flow_solver_steps) > 0 else None),
        max_states_per_route=int(args.max_states_per_route),
        save_every=int(args.save_every),
    )

    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    log.info(f"device={device}")

    rep = _read_json(Path(args.per_route_json))
    per_route = rep.get("per_route", rep.get("routes", None))
    if not isinstance(per_route, list):
        raise SystemExit("[FATAL] per_route_json missing list field: per_route")

    # Extract trajectories (route_id -> pred seq) and route label.
    route_ids: list[int] = []
    preds: list[list[int]] = []
    y_route: list[float] = []
    key = str(args.pred_key)
    for r in per_route:
        if not isinstance(r, dict):
            continue
        rid = r.get("route_id", None)
        if rid is None:
            continue
        block = r.get(key, None)
        if not isinstance(block, dict):
            # fallback: some scripts use only "greedy"
            block = r.get("greedy", None)
        if not isinstance(block, dict):
            continue
        seq = block.get("pred_way_ids", None)
        if not isinstance(seq, list) or not seq:
            continue
        succ = bool(block.get("success", False))
        route_ids.append(int(rid))
        preds.append([int(x) for x in seq])
        y_route.append(1.0 if succ else 0.0)

    if not route_ids:
        raise SystemExit("[FATAL] no trajectories found (need dump_way_seqs with pred_way_ids).")
    log.info(f"routes_in_json={len(route_ids)} pred_key={key}")

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    n_routes_total = int(routes.way_seq_len.size)
    if any((rid < 0) or (rid >= n_routes_total) for rid in route_ids):
        raise SystemExit("[FATAL] per_route_json contains out-of-range route_id.")

    # Build per-route conditioning.
    rid_arr = np.asarray(route_ids, dtype=np.int64)
    start_pos = routes.start_pos[rid_arr].astype(np.float32, copy=False)
    dest_pos = routes.dest_pos[rid_arr].astype(np.float32, copy=False)
    start_t = routes.start_t[rid_arr].astype(np.int64, copy=False)
    route_city = routes.route_city[rid_arr].astype(np.int64, copy=False)
    hour = hour_from_unix(start_t, tz_offset_hours=float(cfg.tz_offset_hours))
    dow = dow_from_unix(start_t, tz_offset_hours=float(cfg.tz_offset_hours))

    # Load models for embeddings and (optional) latent summaries.
    ae = _load_ae(ae_ckpt=Path(args.ae_ckpt), way_graph_npz=Path(args.way_graph_npz), way_features_npz=Path(args.way_features_npz), device=device)
    ae.eval()
    flow: Optional[LatentFlowMatching] = None
    if str(cfg.latent_source) == "flow":
        if args.flow_ckpt is None:
            raise SystemExit("[FATAL] latent_source=flow requires --flow_ckpt.")
        flow = _load_flow(flow_ckpt=Path(args.flow_ckpt), ae=ae, device=device)
        flow.eval()

    way_region: Optional[np.ndarray] = None
    if flow is not None and bool(flow.cfg.use_region_seq):
        if args.way_regions_npz is None:
            raise SystemExit("[FATAL] Flow requires region_seq conditioning, so --way_regions_npz is required.")
        wr = np.load(str(Path(args.way_regions_npz)), allow_pickle=True)
        if "way_region" not in wr.files:
            raise SystemExit("[FATAL] way_regions_npz missing key: way_region")
        way_region = np.asarray(wr["way_region"], dtype=np.int64).reshape(-1)

    # Compute route-level z_mean and cond_emb once.
    with torch.no_grad():
        route_cond_t = {
            "start_pos": torch.as_tensor(start_pos, dtype=torch.float32, device=device),
            "dest_pos": torch.as_tensor(dest_pos, dtype=torch.float32, device=device),
            "hour": torch.as_tensor(hour, dtype=torch.long, device=device),
            "dow": torch.as_tensor(dow, dtype=torch.long, device=device),
            "route_city": torch.as_tensor(route_city, dtype=torch.long, device=device),
        }
        if flow is not None and bool(flow.cfg.use_region_seq):
            if way_region is None:
                raise RuntimeError("Flow requires region_seq conditioning, but way_region is missing.")
            seqs: list[list[int]] = []
            for rid in rid_arr.tolist():
                L = int(routes.way_seq_len[int(rid)])
                s = int(routes.way_seq_ptr[int(rid)])
                ws = routes.way_seq_idx[s : s + L].astype(np.int64, copy=False)
                seqs.append(_region_seq_from_way_seq(ws, way_region))
            route_cond_t["region_seq_pad"] = _pad_region_seqs(seqs, device=device)

        cond_emb = ae.decoder.cond_enc(
            start_pos=route_cond_t["start_pos"],
            dest_pos=route_cond_t["dest_pos"],
            hour=route_cond_t["hour"],
            dow=route_cond_t["dow"],
            route_city=route_cond_t["route_city"],
        )  # (N,d)

        if str(cfg.latent_source) == "gt":
            # Encode GT way sequences.
            gt_seqs: list[np.ndarray] = []
            maxL = 1
            for rid in rid_arr.tolist():
                L = int(routes.way_seq_len[int(rid)])
                s = int(routes.way_seq_ptr[int(rid)])
                ws = routes.way_seq_idx[s : s + L].astype(np.int64, copy=False)
                gt_seqs.append(ws)
                maxL = max(maxL, int(ws.size))
            pad = np.full((int(rid_arr.size), int(maxL)), -1, dtype=np.int64)
            for i, ws in enumerate(gt_seqs):
                pad[i, : int(ws.size)] = ws
            z, _ = ae.encode(torch.as_tensor(pad, dtype=torch.long, device=device))
        else:
            assert flow is not None
            z = flow.sample(route_cond=route_cond_t, solver_steps=cfg.flow_solver_steps)

        z_mean = z.mean(dim=1)  # (N,d)

    # Build state samples.
    max_states = int(cfg.max_states_per_route)
    state_route_idx: list[int] = []
    state_cur_way: list[int] = []
    state_y: list[float] = []
    for i, seq in enumerate(preds):
        if not seq:
            continue
        if max_states > 0 and len(seq) > max_states:
            pick = np.linspace(0, int(len(seq) - 1), int(max_states), dtype=np.int64)
            ways = [int(seq[int(j)]) for j in pick.tolist()]
        else:
            ways = [int(x) for x in seq]
        for w in ways:
            state_route_idx.append(int(i))
            state_cur_way.append(int(w))
            state_y.append(float(y_route[int(i)]))

    if not state_cur_way:
        raise SystemExit("[FATAL] no states extracted from trajectories.")
    log.info(f"states={len(state_cur_way)} (max_states_per_route={max_states})")

    # Split by route_idx to avoid leakage across states.
    rng = np.random.default_rng(int(cfg.seed))
    nR = int(len(route_ids))
    perm = rng.permutation(nR)
    n_val = int(round(float(cfg.val_ratio) * float(nR)))
    n_val = max(1, min(n_val, nR - 1))
    val_routes = set(int(x) for x in perm[:n_val].tolist())
    is_val = np.asarray([int(ri) in val_routes for ri in state_route_idx], dtype=bool)

    def _sub(x, m):
        a = np.asarray(x)
        return a[m]

    tr_ds = _StateDataset(route_idx=_sub(state_route_idx, ~is_val), cur_way=_sub(state_cur_way, ~is_val), y=_sub(state_y, ~is_val))
    va_ds = _StateDataset(route_idx=_sub(state_route_idx, is_val), cur_way=_sub(state_cur_way, is_val), y=_sub(state_y, is_val))
    log.info(f"train_states={len(tr_ds)} val_states={len(va_ds)} val_routes={n_val}/{nR}")

    def _collate(rows: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        return {
            "route_idx": torch.as_tensor(np.asarray([int(r["route_idx"]) for r in rows], dtype=np.int64), dtype=torch.long),
            "cur_way": torch.as_tensor(np.asarray([int(r["cur_way"]) for r in rows], dtype=np.int64), dtype=torch.long),
            "y": torch.as_tensor(np.asarray([float(r["y"]) for r in rows], dtype=np.float32), dtype=torch.float32),
        }

    pin = bool(device.type == "cuda")
    nw = max(0, int(cfg.num_workers))
    tr_loader = DataLoader(tr_ds, batch_size=int(cfg.batch_size), shuffle=True, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0), collate_fn=_collate)
    va_loader = DataLoader(va_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0), collate_fn=_collate)

    vf = WayValueFn(cfg=WayValueFnCfg(d_model=int(ae.cfg.d_model), hidden_dim=int(ae.cfg.d_model), dropout=0.1)).to(device)
    opt = torch.optim.AdamW(vf.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    best_val = float("inf")
    best_epoch = 0

    def _save(path: Path, *, epoch: int, val_loss: float) -> None:
        ckpt = {"epoch": int(epoch), "val_loss": float(val_loss), "model_state_dict": vf.state_dict(), "config": asdict(vf.cfg)}
        torch.save(ckpt, str(path))

    @torch.no_grad()
    def _eval(loader: DataLoader) -> float:
        vf.eval()
        losses: list[float] = []
        for batch in loader:
            ridx = batch["route_idx"].to(device=device, dtype=torch.long)
            cur = batch["cur_way"].to(device=device, dtype=torch.long)
            y = batch["y"].to(device=device, dtype=torch.float32)
            cur_emb, _m = ae.way_enc(cur[:, None])
            cur_emb = cur_emb[:, 0, :]
            z_b = z_mean[ridx]
            c_b = cond_emb[ridx]
            geom, _tier, _hw = ae.way_enc._lookup(cur)
            center = geom[:, :2].to(dtype=torch.float32)
            dest = route_cond_t["dest_pos"][ridx].to(dtype=torch.float32)
            if coord_scale := float(getattr(ae.way_enc, "coord_scale", 0.0)):
                dest = dest / float(coord_scale)
            dist = torch.norm(dest - center, dim=-1)  # (B,)
            logit = vf(cur_emb=cur_emb, z_mean=z_b, cond_emb=c_b, dest_dist=dist)
            loss = F.binary_cross_entropy_with_logits(logit, y, reduction="mean")
            losses.append(float(loss.detach().item()))
        return float(np.mean(losses)) if losses else float("nan")

    for epoch in range(1, int(cfg.n_epochs) + 1):
        vf.train()
        losses: list[float] = []
        for batch in tr_loader:
            ridx = batch["route_idx"].to(device=device, dtype=torch.long)
            cur = batch["cur_way"].to(device=device, dtype=torch.long)
            y = batch["y"].to(device=device, dtype=torch.float32)
            cur_emb, _m = ae.way_enc(cur[:, None])
            cur_emb = cur_emb[:, 0, :]
            z_b = z_mean[ridx]
            c_b = cond_emb[ridx]
            geom, _tier, _hw = ae.way_enc._lookup(cur)
            center = geom[:, :2].to(dtype=torch.float32)
            dest = route_cond_t["dest_pos"][ridx].to(dtype=torch.float32)
            coord_scale = float(getattr(ae.way_enc, "coord_scale", 0.0))
            if coord_scale > 0:
                dest = dest / float(coord_scale)
            dist = torch.norm(dest - center, dim=-1)  # (B,)

            logit = vf(cur_emb=cur_emb, z_mean=z_b, cond_emb=c_b, dest_dist=dist)
            loss = F.binary_cross_entropy_with_logits(logit, y, reduction="mean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().item()))

        tr_loss = float(np.mean(losses)) if losses else float("nan")
        va_loss = _eval(va_loader)
        log.info(f"epoch={epoch} train_loss={tr_loss:.6f} val_loss={va_loss:.6f} best={best_val:.6f}@{best_epoch}")

        progress = {
            "ok": True,
            "task": "train_way_casd_value_fn",
            "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
            "epoch": int(epoch),
            "train": {"loss": float(tr_loss)},
            "val": {"loss": float(va_loss)},
            "best_val_loss": float(best_val),
            "best_epoch": int(best_epoch),
        }
        (out_dir / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

        if va_loss < best_val:
            best_val = float(va_loss)
            best_epoch = int(epoch)
            _save(out_dir / "ckpt_best.pt", epoch=epoch, val_loss=best_val)
        if int(cfg.save_every) > 0 and (int(epoch) % int(cfg.save_every) == 0):
            _save(out_dir / "ckpt_last.pt", epoch=epoch, val_loss=va_loss)

    report = {
        "ok": True,
        "task": "train_way_casd_value_fn",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "per_route_json": str(args.per_route_json),
            "pred_key": str(args.pred_key),
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
            "flow_ckpt": (str(args.flow_ckpt) if args.flow_ckpt is not None else None),
            "way_regions_npz": (str(args.way_regions_npz) if args.way_regions_npz is not None else None),
        },
        "out_dir": str(out_dir),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "cfg": asdict(cfg),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"saved: {out_dir/'report.json'}")


if __name__ == "__main__":
    main()
