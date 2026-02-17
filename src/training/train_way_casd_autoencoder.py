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
from torch.utils.data import DataLoader, Subset

from src.data.way_graph.way_sequence_dataset import (
    WayRouteDataset,
    load_way_routes_npz,
    make_way_casd_collate_fn,
)
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import load_way_features_from_npz, make_way_feature_tensors

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
    max_candidates: int
    max_routes: Optional[int]

    d_model: int
    n_latent: int
    n_heads: int
    dropout: float
    max_len: int
    decoder_use_dest_dist: bool
    decoder_use_step_emb: bool
    decoder_use_dest_query: bool
    decoder_use_dir_query: bool
    decoder_use_cand_query: bool
    decoder_use_cand_contrast: bool
    # Cross-attention
    decoder_use_cross_attn: bool
    # Past context
    decoder_use_past_context: bool
    decoder_past_k: int
    decoder_past_n_layers: int
    decoder_past_n_heads: int
    # E8 (optional): multi-scale latent (segment tokens)
    segment_size: int
    segment_n_latent: int
    # SIB: Stochastic Information Bottleneck
    latent_noise_std: float = 0.0
    drop_dest_dist_p: float = 0.0
    drop_past_context_p: float = 0.0
    noise_warmup_epochs: int = 0  # Linear ramp noise_std from 0 to latent_noise_std over this many epochs
    split_json: Optional[str] = None


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    way_seq_pad = batch["way_seq_pad"].to(device)
    route_cond = {k: v.to(device) for k, v in batch["route_cond"].items()}
    trans = {k: v.to(device) for k, v in batch["trans"].items()}
    return {"way_seq_pad": way_seq_pad, "route_cond": route_cond, "trans": trans}


def _split_dataset(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n))
    n_val = int(round(float(val_ratio) * float(n)))
    n_val = max(1, min(n_val, n - 1))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx.astype(np.int64, copy=False), val_idx.astype(np.int64, copy=False)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _subset_indices_from_route_ids(dataset: WayRouteDataset, route_ids: np.ndarray) -> np.ndarray:
    route_ids = np.asarray(route_ids, dtype=np.int64).reshape(-1)
    if route_ids.size == 0:
        return np.zeros((0,), dtype=np.int64)
    mask = np.isin(dataset.route_ids.astype(np.int64, copy=False), route_ids, assume_unique=False)
    return np.nonzero(mask)[0].astype(np.int64, copy=False)


def train_epoch(model: WayCASDAutoEncoder, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device, *, current_noise_std: float = 0.0) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_trans = 0

    for batch in loader:
        b = _to_device(batch, device)
        opt.zero_grad(set_to_none=True)
        loss, stats = model.compute_loss(b, current_noise_std=current_noise_std)
        loss.backward()
        opt.step()

        n_trans = int(stats["n_trans"])
        total_loss += float(stats["loss"]) * n_trans
        total_acc += float(stats["acc"]) * n_trans
        total_trans += n_trans

    denom = max(1, int(total_trans))
    return {"loss": float(total_loss / denom), "acc": float(total_acc / denom), "n_trans": float(total_trans)}


@torch.no_grad()
def eval_epoch(model: WayCASDAutoEncoder, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_trans = 0
    for batch in loader:
        b = _to_device(batch, device)
        loss, stats = model.compute_loss(b)
        n_trans = int(stats["n_trans"])
        total_loss += float(stats["loss"]) * n_trans
        total_acc += float(stats["acc"]) * n_trans
        total_trans += n_trans
    denom = max(1, int(total_trans))
    return {"loss": float(total_loss / denom), "acc": float(total_acc / denom), "n_trans": float(total_trans)}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Way-CASD autoencoder (Step A).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--min_hops", type=int, default=1, help="Filter routes with fewer than this many way transitions (hops).")
    p.add_argument("--max_way_len", type=int, default=128)
    p.add_argument("--max_candidates", type=int, default=32)
    p.add_argument("--max_routes", type=int, default=None, help="Debug: cap number of routes (after filtering).")
    p.add_argument(
        "--split_json",
        type=Path,
        default=None,
        help="Optional OD-disjoint split json (expects splits.train/val/test route_ids). Overrides val_ratio.",
    )

    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_latent", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument(
        "--decoder_use_dest_dist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include candidate-to-destination distance as an extra scalar feature in the decoder scorer.",
    )
    p.add_argument("--decoder_use_step_emb", action="store_true", help="Add step embedding into cross-attn query (decoder).")
    p.add_argument("--decoder_use_dest_query", action="store_true", help="Add dest_pos projection into cross-attn query (decoder).")
    p.add_argument("--decoder_use_dir_query", action="store_true", help="Add candidate-direction hint into cross-attn query (decoder).")
    p.add_argument(
        "--decoder_use_cand_query",
        action="store_true",
        help="Candidate-aware cross-attn: let each candidate query z_enc separately (decoder).",
    )
    p.add_argument(
        "--decoder_use_cand_contrast",
        action="store_true",
        help="Contrastive candidate feature in scorer: include (cand_h - mean_cand_h) to score candidates relatively.",
    )
    # Cross-attention ablation
    p.add_argument(
        "--decoder_use_cross_attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cross-attention to query latent tokens (default=True). Set --no-decoder_use_cross_attn to ablate.",
    )
    # Past context: encode past-K path with small Transformer
    p.add_argument("--decoder_use_past_context", action="store_true", help="Add past-K path context via Transformer encoder.")
    p.add_argument("--decoder_past_k", type=int, default=8, help="Number of past steps to include.")
    p.add_argument("--decoder_past_n_layers", type=int, default=2, help="Transformer layers for past encoder.")
    p.add_argument("--decoder_past_n_heads", type=int, default=4, help="Attention heads in past encoder.")
    # E8: multi-scale latent (reserve last S latent tokens as segment summaries)
    p.add_argument("--segment_size", type=int, default=10, help="Segment size for multi-scale latent (only used when --segment_n_latent>0).")
    p.add_argument(
        "--segment_n_latent",
        type=int,
        default=0,
        help="E8: number of segment latent tokens to overwrite at the tail of z_enc (0=disable).",
    )

    # SIB: Stochastic Information Bottleneck (force decoder to rely on latent)
    p.add_argument("--latent_noise_std", type=float, default=0.0, help="Max Gaussian noise σ injected into z_enc during AE training (0=disable).")
    p.add_argument("--drop_dest_dist_p", type=float, default=0.0, help="Probability of zeroing dest_dist bypass per batch (0=disable).")
    p.add_argument("--drop_past_context_p", type=float, default=0.0, help="Probability of dropping past_context bypass per batch (0=disable).")
    p.add_argument("--noise_warmup_epochs", type=int, default=0, help="Linear ramp noise_std from 0 to latent_noise_std over N epochs (0=no warmup).")

    # Long-run training ergonomics
    p.add_argument("--resume_ckpt", type=Path, default=None, help="Optional: resume from ckpt_last.pt/ckpt_best.pt.")
    p.add_argument("--resume_epoch", type=int, default=None, help="Optional: override resume epoch (when ckpt has no epoch).")
    p.add_argument("--save_every", type=int, default=20, help="Save ckpt_last.pt every N epochs (best ckpt saved on improve).")
    p.add_argument("--early_stop_patience", type=int, default=0, help="Optional early stop (0=disable).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parameter sanity checks (fail fast to avoid silent misconfiguration).
    if float(args.latent_noise_std) < 0.0:
        raise SystemExit("[FATAL] --latent_noise_std must be >= 0.")
    if not (0.0 <= float(args.drop_dest_dist_p) <= 1.0):
        raise SystemExit("[FATAL] --drop_dest_dist_p must be in [0, 1].")
    if not (0.0 <= float(args.drop_past_context_p) <= 1.0):
        raise SystemExit("[FATAL] --drop_past_context_p must be in [0, 1].")
    if int(args.noise_warmup_epochs) < 0:
        raise SystemExit("[FATAL] --noise_warmup_epochs must be >= 0.")

    if bool(args.decoder_use_cand_contrast) and (not bool(args.decoder_use_cand_query)):
        log.warning(
            "decoder_use_cand_contrast=True but decoder_use_cand_query=False. "
            "In this setting, ctx_h is candidate-agnostic; contrast features may hurt unless this is an intended ablation. "
            "If you want the strong baseline, also enable --decoder_use_cand_query and (recommended) --decoder_use_past_context."
        )

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
        max_candidates=int(args.max_candidates),
        max_routes=(int(args.max_routes) if args.max_routes is not None else None),
        d_model=int(args.d_model),
        n_latent=int(args.n_latent),
        n_heads=int(args.n_heads),
        dropout=float(args.dropout),
        max_len=int(args.max_len),
        decoder_use_dest_dist=bool(args.decoder_use_dest_dist),
        decoder_use_step_emb=bool(args.decoder_use_step_emb),
        decoder_use_dest_query=bool(args.decoder_use_dest_query),
        decoder_use_dir_query=bool(args.decoder_use_dir_query),
        decoder_use_cand_query=bool(args.decoder_use_cand_query),
        decoder_use_cand_contrast=bool(args.decoder_use_cand_contrast),
        decoder_use_cross_attn=bool(args.decoder_use_cross_attn),
        decoder_use_past_context=bool(args.decoder_use_past_context),
        decoder_past_k=int(args.decoder_past_k),
        decoder_past_n_layers=int(args.decoder_past_n_layers),
        decoder_past_n_heads=int(args.decoder_past_n_heads),
        segment_size=int(args.segment_size),
        segment_n_latent=int(args.segment_n_latent),
        latent_noise_std=float(args.latent_noise_std),
        drop_dest_dist_p=float(args.drop_dest_dist_p),
        drop_past_context_p=float(args.drop_past_context_p),
        noise_warmup_epochs=int(args.noise_warmup_epochs),
        split_json=(str(args.split_json) if args.split_json is not None else None),
    )

    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    log.info(f"device={device}")

    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    way_features = load_way_features_from_npz(Path(args.way_features_npz))
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1
    n_semantic = way_features.way_semantic.shape[-1] if way_features.way_semantic is not None else 0
    if n_semantic > 0:
        log.info(f"loaded way_semantic: n_channels={n_semantic}")

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    if args.split_json is not None and args.max_routes is not None:
        log.warning("--split_json is set, ignoring --max_routes to avoid inconsistent splits.")
    dataset = WayRouteDataset(
        routes,
        max_routes=(None if args.split_json is not None else cfg.max_routes),
        max_way_len=int(cfg.max_way_len),
        min_hops=int(cfg.min_hops),
    )
    if args.split_json is None:
        train_ids, val_ids = _split_dataset(len(dataset), cfg.val_ratio, cfg.seed)
    else:
        split = _read_json(Path(args.split_json))
        splits = split.get("splits", split)
        tr_rids = np.asarray(splits.get("train", []), dtype=np.int64).reshape(-1)
        va_rids = np.asarray(splits.get("val", []), dtype=np.int64).reshape(-1)
        train_ids = _subset_indices_from_route_ids(dataset, tr_rids)
        val_ids = _subset_indices_from_route_ids(dataset, va_rids)
        if int(train_ids.size) == 0 or int(val_ids.size) == 0:
            raise SystemExit(
                f"[FATAL] split_json produced empty subsets: train_idx={int(train_ids.size)} val_idx={int(val_ids.size)}. "
                "Check min_hops/max_way_len match split generation."
            )
        log.info(
            f"split_json={args.split_json} train_routes={int(tr_rids.size)} val_routes={int(va_rids.size)} "
            f"=> train_idx={int(train_ids.size)} val_idx={int(val_ids.size)}"
        )
    train_set = Subset(dataset, train_ids.tolist())
    val_set = Subset(dataset, val_ids.tolist())
    log.info(
        f"routes: total={len(dataset)} train={len(train_set)} val={len(val_set)} min_hops={cfg.min_hops} max_way_len={cfg.max_way_len}"
    )

    collate_fn = make_way_casd_collate_fn(
        way_adj_ptr=wg["way_adj_ptr"],
        way_adj_idx=wg["way_adj_idx"],
        max_candidates=int(cfg.max_candidates),
        tz_offset_hours=float(cfg.tz_offset_hours),
        past_k=int(cfg.decoder_past_k),
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

    model = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(cfg.d_model),
            n_latent=int(cfg.n_latent),
            n_heads=int(cfg.n_heads),
            dropout=float(cfg.dropout),
            max_candidates=int(cfg.max_candidates),
            max_len=int(cfg.max_len),
            segment_size=int(cfg.segment_size),
            segment_n_latent=int(cfg.segment_n_latent),
            decoder_use_dest_dist=bool(cfg.decoder_use_dest_dist),
            decoder_use_cross_attn=bool(cfg.decoder_use_cross_attn),
            decoder_use_step_emb=bool(cfg.decoder_use_step_emb),
            decoder_use_dest_query=bool(cfg.decoder_use_dest_query),
            decoder_use_dir_query=bool(cfg.decoder_use_dir_query),
            decoder_use_cand_query=bool(cfg.decoder_use_cand_query),
            decoder_use_cand_contrast=bool(cfg.decoder_use_cand_contrast),
            decoder_use_past_context=bool(cfg.decoder_use_past_context),
            decoder_past_k=int(cfg.decoder_past_k),
            decoder_past_n_layers=int(cfg.decoder_past_n_layers),
            decoder_past_n_heads=int(cfg.decoder_past_n_heads),
            latent_noise_std=float(cfg.latent_noise_std),
            drop_dest_dist_p=float(cfg.drop_dest_dist_p),
            drop_past_context_p=float(cfg.drop_past_context_p),
        ),
        way_features=way_features,
        way_adj_ptr=wg["way_adj_ptr"],
        way_adj_idx=wg["way_adj_idx"],
        n_highway_types=int(max(4, n_highway_types)),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    best = float("inf")
    best_epoch = 0
    best_path = out_dir / "ckpt_best.pt"
    last_path = out_dir / "ckpt_last.pt"
    progress_path = out_dir / "progress.json"
    hist_path = out_dir / "history.jsonl"

    start_epoch = 1
    history = []
    patience = 0

    if args.resume_ckpt is not None:
        ckpt = torch.load(str(args.resume_ckpt), map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if missing or unexpected:
                log.warning(f"resume: model state mismatch: missing={len(missing)} unexpected={len(unexpected)}")
            if "opt_state_dict" in ckpt:
                try:
                    opt.load_state_dict(ckpt["opt_state_dict"])
                except Exception as e:  # pragma: no cover
                    log.warning(f"resume: failed to load optimizer state (ignored): {e}")
            if "best_val_loss" in ckpt:
                try:
                    best = float(ckpt["best_val_loss"])
                except Exception:
                    best = float("inf")
            if "best_epoch" in ckpt:
                try:
                    best_epoch = int(ckpt["best_epoch"])
                except Exception:
                    best_epoch = 0
            if "epoch" in ckpt:
                try:
                    start_epoch = int(ckpt["epoch"]) + 1
                except Exception:
                    start_epoch = 1
        elif isinstance(ckpt, dict):
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            if missing or unexpected:
                log.warning(f"resume: model state mismatch: missing={len(missing)} unexpected={len(unexpected)}")

        if args.resume_epoch is not None:
            start_epoch = int(args.resume_epoch) + 1

        log.info(f"resume_ckpt={args.resume_ckpt} start_epoch={start_epoch} best_val_loss={best} best_epoch={best_epoch}")

    if not np.isfinite(best) or best == float("inf"):
        va0 = eval_epoch(model, val_loader, device)
        best = float(va0["loss"])
        best_epoch = int(start_epoch - 1)
        log.info(f"init best_val_loss={best:.6f} from current weights (epoch={best_epoch})")

    save_every = max(1, int(args.save_every))
    early_stop_patience = max(0, int(args.early_stop_patience))

    for epoch in range(int(start_epoch), int(cfg.n_epochs) + 1):
        # SIB: compute current noise_std with optional warmup schedule
        if cfg.latent_noise_std > 0 and cfg.noise_warmup_epochs > 0:
            # Ramp starts at 0 on the first epoch and reaches sigma at/after warmup end.
            if int(cfg.noise_warmup_epochs) <= 1:
                ramp = 1.0
            else:
                ramp = min(
                    1.0,
                    max(0.0, float(int(epoch) - 1) / float(int(cfg.noise_warmup_epochs) - 1)),
                )
            current_noise_std = float(cfg.latent_noise_std) * ramp
        else:
            current_noise_std = float(cfg.latent_noise_std)
        tr = train_epoch(model, train_loader, opt, device, current_noise_std=current_noise_std)
        va = eval_epoch(model, val_loader, device)
        history.append({"epoch": int(epoch), "train": tr, "val": va})
        log.info(f"epoch={epoch} train_loss={tr['loss']:.4f} train_acc={tr['acc']:.3f} val_loss={va['loss']:.4f} val_acc={va['acc']:.3f}")

        with hist_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"epoch": int(epoch), "train_loss": float(tr["loss"]), "train_acc": float(tr["acc"]), "val_loss": float(va["loss"]), "val_acc": float(va["acc"])}
                )
                + "\n"
            )

        if float(va["loss"]) < float(best):
            best = float(va["loss"])
            best_epoch = int(epoch)
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
                    "epoch": int(epoch),
                    "best_val_loss": float(best),
                    "best_epoch": int(best_epoch),
                },
                str(best_path),
            )
        else:
            patience += 1

        if (int(epoch) % save_every) == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "opt_state_dict": opt.state_dict(),
                    "config": asdict(cfg),
                    "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
                    "epoch": int(epoch),
                    "best_val_loss": float(best),
                    "best_epoch": int(best_epoch),
                },
                str(last_path),
            )

        progress = {
            "ok": True,
            "task": "train_way_casd_autoencoder",
            "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
            "epoch": int(epoch),
            "train_loss": float(tr["loss"]),
            "train_acc": float(tr["acc"]),
            "val_loss": float(va["loss"]),
            "val_acc": float(va["acc"]),
            "best_val_loss": float(best),
            "best_epoch": int(best_epoch),
            "save_every": int(save_every),
            "early_stop_patience": int(early_stop_patience),
        }
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

        if early_stop_patience > 0 and patience >= early_stop_patience:
            log.info(f"early_stop: patience={patience} reached (best_epoch={best_epoch} best_val_loss={best:.6f})")
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "opt_state_dict": opt.state_dict(),
            "config": asdict(cfg),
            "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
            "epoch": int(history[-1]["epoch"]) if history else int(start_epoch - 1),
            "best_val_loss": float(best),
            "best_epoch": int(best_epoch),
        },
        str(last_path),
    )

    report = {
        "ok": True,
        "task": "train_way_casd_autoencoder",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "split_json": (str(args.split_json) if args.split_json is not None else None),
        },
        "out_dir": str(out_dir),
        "best_val_loss": float(best),
        "best_ckpt": str(best_path),
        "last_ckpt": str(last_path),
        "best_epoch": int(best_epoch),
        "start_epoch": int(start_epoch),
        "early_stop_patience": int(early_stop_patience),
        "save_every": int(save_every),
        "history": history,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"saved: {out_dir/'report.json'}")


if __name__ == "__main__":
    main()
