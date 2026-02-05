"""
E2: Flow-Decoder Joint Fine-tuning (Teacher Forcing)

Goal:
  Fine-tune ONLY the decoder parameters so that scoring is compatible with Flow-sampled latents.

Key idea:
  Use z_flow = Flow.sample(route_cond) as latent tokens, but keep teacher-forcing CE loss on GT transitions.

Outputs (out_dir):
  - ckpt_best.pt / ckpt_last.pt
  - progress.json (last epoch summary)
  - report.json (artifact index)
"""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.way_graph.way_sequence_dataset import WayRouteDataset, load_way_routes_npz, make_way_casd_collate_fn
from src.models.way_casd.latent_flow import LatentFlowCfg, LatentFlowMatching
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz

TZ_SHANGHAI = timezone(timedelta(hours=8))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


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
    max_routes: Optional[int]
    flow_solver_steps: Optional[int]
    save_every: int
    early_stop_patience: int
    max_grad_norm: float


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


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    way_seq_pad = batch["way_seq_pad"].to(device)
    way_seq_len = batch["way_seq_len"].to(device)
    route_cond = {k: v.to(device) for k, v in batch["route_cond"].items()}
    trans = {k: v.to(device) for k, v in batch["trans"].items()}
    route_id = batch["route_id"].to(device)
    return {"way_seq_pad": way_seq_pad, "way_seq_len": way_seq_len, "route_cond": route_cond, "trans": trans, "route_id": route_id}


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
        if not s:
            continue
        pad[i, : len(s)] = torch.as_tensor(s, dtype=torch.long, device=device)
    return pad


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
    use_cand_contrast = bool(cfg_dict.get("decoder_use_cand_contrast", False)) or _infer_decoder_use_cand_contrast_from_state(state)
    use_cross_attn = bool(cfg_dict.get("decoder_use_cross_attn", True)) or _infer_decoder_use_cross_attn_from_state(state)
    use_step_emb = bool(cfg_dict.get("decoder_use_step_emb", False)) or _infer_decoder_use_step_emb_from_state(state)
    use_dest_query = bool(cfg_dict.get("decoder_use_dest_query", False)) or _infer_decoder_use_dest_query_from_state(state)
    use_dir_query = bool(cfg_dict.get("decoder_use_dir_query", False)) or _infer_decoder_use_dir_query_from_state(state)
    use_cand_query = bool(cfg_dict.get("decoder_use_cand_query", False)) or _infer_decoder_use_cand_query_from_state(state)
    use_past_ctx = bool(cfg_dict.get("decoder_use_past_context", False)) or _infer_decoder_use_past_context_from_state(state)
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


@torch.no_grad()
def _eval_epoch(
    *,
    ae: WayCASDAutoEncoder,
    flow: LatentFlowMatching,
    loader: DataLoader,
    device: torch.device,
    way_region: Optional[np.ndarray],
    solver_steps: Optional[int],
) -> Dict[str, float]:
    ae.eval()
    flow.eval()
    losses: list[float] = []
    n_items = 0
    for batch in loader:
        b = _to_device(batch, device)
        route_cond = b["route_cond"]
        route_cond_use = {
            "start_pos": route_cond["start_pos"],
            "dest_pos": route_cond["dest_pos"],
            "hour": route_cond["hour"],
            "dow": route_cond["dow"],
            "route_city": route_cond["route_city"],
        }
        if bool(flow.cfg.use_region_seq):
            if way_region is None:
                raise RuntimeError("Flow requires region_seq conditioning, but way_region is missing.")
            pad = b["way_seq_pad"].detach().cpu().numpy()
            lens = b["way_seq_len"].detach().cpu().numpy()
            seqs: list[list[int]] = []
            for i in range(int(pad.shape[0])):
                L = int(lens[i])
                seq = pad[i, :L].astype(np.int64, copy=False)
                seqs.append(_region_seq_from_way_seq(seq, way_region))
            route_cond_use["region_seq_pad"] = _pad_region_seqs(seqs, device=device)
        z = flow.sample(route_cond=route_cond_use, solver_steps=solver_steps)
        logits = ae.decoder.score_candidates(way_embedder=ae.way_enc, latent_tokens=z, route_cond=route_cond_use, trans=b["trans"])
        tgt = b["trans"]["target_idx"].to(dtype=torch.long)
        loss = F.cross_entropy(logits, tgt, reduction="mean")
        losses.append(float(loss.detach().item()))
        n_items += 1
    return {"loss": float(np.mean(losses)) if n_items > 0 else float("nan")}


def main() -> None:
    p = argparse.ArgumentParser(description="E2: Joint fine-tune Way decoder on Flow latents (teacher forcing).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--flow_ckpt", type=Path, required=True)
    p.add_argument("--way_regions_npz", type=Path, default=None, help="Required when Flow uses region_seq.")
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--max_routes", type=int, default=None)
    p.add_argument("--flow_solver_steps", type=int, default=0, help="Override flow solver steps (0=use ckpt).")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
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
        min_hops=int(args.min_hops),
        max_way_len=int(args.max_way_len),
        max_routes=(int(args.max_routes) if args.max_routes is not None else None),
        flow_solver_steps=(int(args.flow_solver_steps) if int(args.flow_solver_steps) > 0 else None),
        save_every=int(args.save_every),
        early_stop_patience=int(args.early_stop_patience),
        max_grad_norm=float(args.max_grad_norm),
    )

    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    log.info(f"device={device}")

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    dataset = WayRouteDataset(routes, max_routes=cfg.max_routes, max_way_len=int(cfg.max_way_len), min_hops=int(cfg.min_hops))
    tr_idx, va_idx = _split_dataset(len(dataset), cfg.val_ratio, cfg.seed)
    train_set = Subset(dataset, tr_idx.tolist())
    val_set = Subset(dataset, va_idx.tolist())
    log.info(f"routes: total={len(dataset)} train={len(train_set)} val={len(val_set)} min_hops={cfg.min_hops} max_way_len={cfg.max_way_len}")

    # Load models
    ae = _load_ae(ae_ckpt=Path(args.ae_ckpt), way_graph_npz=Path(args.way_graph_npz), way_features_npz=Path(args.way_features_npz), device=device)
    flow = _load_flow(flow_ckpt=Path(args.flow_ckpt), ae=ae, device=device)

    # Freeze everything except decoder.*
    for p0 in ae.parameters():
        p0.requires_grad_(False)
    for name, p0 in ae.named_parameters():
        if str(name).startswith("decoder."):
            p0.requires_grad_(True)
    for p0 in flow.parameters():
        p0.requires_grad_(False)
    ae.train()
    flow.eval()

    way_region: Optional[np.ndarray] = None
    if bool(flow.cfg.use_region_seq):
        if args.way_regions_npz is None:
            raise SystemExit("[FATAL] Flow requires region_seq conditioning, so --way_regions_npz is required.")
        wr = np.load(str(Path(args.way_regions_npz)), allow_pickle=True)
        if "way_region" not in wr.files:
            raise SystemExit("[FATAL] way_regions_npz missing key: way_region")
        way_region = np.asarray(wr["way_region"], dtype=np.int64).reshape(-1)

    # Collate (for candidate sets and past context).
    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    collate_fn = make_way_casd_collate_fn(
        way_adj_ptr=wg["way_adj_ptr"],
        way_adj_idx=wg["way_adj_idx"],
        max_candidates=int(ae.cfg.max_candidates),
        tz_offset_hours=float(cfg.tz_offset_hours),
        past_k=int(ae.cfg.decoder_past_k),
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

    params = [p0 for p0 in ae.parameters() if p0.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    start_epoch = 1

    def _save_ckpt(path: Path, *, epoch: int, val_loss: float) -> None:
        ckpt = {
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "model_state_dict": ae.state_dict(),
            "config": asdict(ae.cfg),
        }
        torch.save(ckpt, str(path))

    for epoch in range(start_epoch, int(cfg.n_epochs) + 1):
        ae.train()
        losses: list[float] = []
        for batch in train_loader:
            b = _to_device(batch, device)
            route_cond = b["route_cond"]
            route_cond_use = {
                "start_pos": route_cond["start_pos"],
                "dest_pos": route_cond["dest_pos"],
                "hour": route_cond["hour"],
                "dow": route_cond["dow"],
                "route_city": route_cond["route_city"],
            }
            if bool(flow.cfg.use_region_seq):
                if way_region is None:
                    raise RuntimeError("Flow requires region_seq conditioning, but way_region is missing.")
                pad = b["way_seq_pad"].detach().cpu().numpy()
                lens = b["way_seq_len"].detach().cpu().numpy()
                seqs: list[list[int]] = []
                for i in range(int(pad.shape[0])):
                    L = int(lens[i])
                    seq = pad[i, :L].astype(np.int64, copy=False)
                    seqs.append(_region_seq_from_way_seq(seq, way_region))
                route_cond_use["region_seq_pad"] = _pad_region_seqs(seqs, device=device)

            with torch.no_grad():
                z = flow.sample(route_cond=route_cond_use, solver_steps=cfg.flow_solver_steps)

            logits = ae.decoder.score_candidates(way_embedder=ae.way_enc, latent_tokens=z, route_cond=route_cond_use, trans=b["trans"])
            tgt = b["trans"]["target_idx"].to(dtype=torch.long)
            loss = F.cross_entropy(logits, tgt, reduction="mean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(cfg.max_grad_norm))
            opt.step()
            losses.append(float(loss.detach().item()))

        tr_loss = float(np.mean(losses)) if losses else float("nan")
        va = _eval_epoch(ae=ae, flow=flow, loader=val_loader, device=device, way_region=way_region, solver_steps=cfg.flow_solver_steps)
        va_loss = float(va["loss"])
        log.info(f"epoch={epoch} train_loss={tr_loss:.6f} val_loss={va_loss:.6f} best={best_val:.6f}@{best_epoch}")

        # Save progress snapshot (single-file, easy to sync).
        progress = {
            "ok": True,
            "task": "train_way_casd_decoder_joint",
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
            bad_epochs = 0
            _save_ckpt(out_dir / "ckpt_best.pt", epoch=epoch, val_loss=best_val)
        else:
            bad_epochs += 1
        if int(cfg.save_every) > 0 and (int(epoch) % int(cfg.save_every) == 0):
            _save_ckpt(out_dir / "ckpt_last.pt", epoch=epoch, val_loss=va_loss)
        if int(cfg.early_stop_patience) > 0 and bad_epochs >= int(cfg.early_stop_patience):
            log.info(f"early stop: bad_epochs={bad_epochs} >= patience={int(cfg.early_stop_patience)}")
            break

    report = {
        "ok": True,
        "task": "train_way_casd_decoder_joint",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
            "flow_ckpt": str(args.flow_ckpt),
            "way_regions_npz": (str(args.way_regions_npz) if args.way_regions_npz is not None else None),
        },
        "out_dir": str(out_dir),
        "best_val_loss": float(best_val),
        "best_ckpt": str(out_dir / "ckpt_best.pt"),
        "last_ckpt": str(out_dir / "ckpt_last.pt"),
        "best_epoch": int(best_epoch),
        "cfg": asdict(cfg),
        "flow_cfg": asdict(flow.cfg),
        "ae_cfg": asdict(ae.cfg),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"saved: {out_dir/'report.json'}")


if __name__ == "__main__":
    main()

