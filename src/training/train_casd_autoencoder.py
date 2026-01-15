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
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.road_graph.segment_sequence_dataset import (
    SegmentRouteDataset,
    load_segment_routes_npz,
    make_casd_collate_fn,
)
from src.models.casd.casd import CASDAECfg, CASDAutoEncoder
from src.models.casd.segment_encoder import make_segment_feature_tensors

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
    max_seg_len: int
    max_candidates: int
    max_routes: Optional[int]

    d_model: int
    n_latent: int
    n_heads: int
    dropout: float


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    seg_seq_pad = batch["seg_seq_pad"].to(device)
    route_cond = {k: v.to(device) for k, v in batch["route_cond"].items()}
    trans = {k: v.to(device) for k, v in batch["trans"].items()}
    return {"seg_seq_pad": seg_seq_pad, "route_cond": route_cond, "trans": trans}


def _split_dataset(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n))
    n_val = int(round(float(val_ratio) * float(n)))
    n_val = max(1, min(n_val, n - 1))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx.astype(np.int64, copy=False), val_idx.astype(np.int64, copy=False)


def train_epoch(model: CASDAutoEncoder, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_trans = 0

    for batch in loader:
        b = _to_device(batch, device)
        opt.zero_grad(set_to_none=True)
        loss, stats = model.compute_loss(b)
        loss.backward()
        opt.step()

        n_trans = int(stats["n_trans"])
        total_loss += float(stats["loss"]) * n_trans
        total_acc += float(stats["acc"]) * n_trans
        total_trans += n_trans

    denom = max(1, int(total_trans))
    return {"loss": float(total_loss / denom), "acc": float(total_acc / denom), "n_trans": float(total_trans)}


@torch.no_grad()
def eval_epoch(model: CASDAutoEncoder, loader: DataLoader, device: torch.device) -> Dict[str, float]:
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
    p = argparse.ArgumentParser(description="Train CASD autoencoder (Step A).")
    p.add_argument("--segment_graph_npz", type=Path, required=True)
    p.add_argument("--routes_npz", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--max_seg_len", type=int, default=640)
    p.add_argument("--max_candidates", type=int, default=16)
    p.add_argument("--max_routes", type=int, default=None, help="Debug: cap number of routes (after filtering).")

    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_latent", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    return p


def main() -> None:
    args = build_argparser().parse_args()
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
        max_seg_len=int(args.max_seg_len),
        max_candidates=int(args.max_candidates),
        max_routes=(int(args.max_routes) if args.max_routes is not None else None),
        d_model=int(args.d_model),
        n_latent=int(args.n_latent),
        n_heads=int(args.n_heads),
        dropout=float(args.dropout),
    )

    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    log.info(f"device={device}")

    sg = np.load(str(args.segment_graph_npz), allow_pickle=True)
    seg_features = make_segment_feature_tensors(
        seg_center_y=sg["seg_center_y"],
        seg_center_x=sg["seg_center_x"],
        seg_dir_y=sg["seg_dir_y"],
        seg_dir_x=sg["seg_dir_x"],
        seg_len_m=sg["seg_len_m"],
        seg_tier=sg["seg_tier"],
        seg_city=sg["seg_city"],
    )

    routes = load_segment_routes_npz(Path(args.routes_npz))
    dataset = SegmentRouteDataset(routes, max_routes=cfg.max_routes, max_seg_len=int(cfg.max_seg_len))
    train_ids, val_ids = _split_dataset(len(dataset), cfg.val_ratio, cfg.seed)
    train_set = Subset(dataset, train_ids.tolist())
    val_set = Subset(dataset, val_ids.tolist())
    log.info(f"routes: total={len(dataset)} train={len(train_set)} val={len(val_set)} max_seg_len={cfg.max_seg_len}")

    collate_fn = make_casd_collate_fn(
        node_seg_ptr=sg["node_seg_ptr"],
        node_seg_idx=sg["node_seg_idx"],
        seg_succ_ptr=sg["seg_succ_ptr"],
        seg_succ_idx=sg["seg_succ_idx"],
        max_candidates=int(cfg.max_candidates),
        tz_offset_hours=float(cfg.tz_offset_hours),
    )

    pin = bool(device.type == "cuda")
    num_workers = max(0, int(cfg.num_workers))
    prefetch_factor = 2 if num_workers > 0 else None
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

    model = CASDAutoEncoder(
        cfg=CASDAECfg(
            d_model=int(cfg.d_model),
            n_latent=int(cfg.n_latent),
            n_heads=int(cfg.n_heads),
            dropout=float(cfg.dropout),
            max_candidates=int(cfg.max_candidates),
            max_len=int(cfg.max_seg_len),
        ),
        seg_features=seg_features,
        seg_v=sg["seg_v"],
        seg_succ_ptr=sg["seg_succ_ptr"],
        seg_succ_idx=sg["seg_succ_idx"],
        node_seg_ptr=sg["node_seg_ptr"],
        node_seg_idx=sg["node_seg_idx"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    best = float("inf")
    best_path = out_dir / "ckpt_best.pt"
    history = []

    for epoch in range(1, int(cfg.n_epochs) + 1):
        tr = train_epoch(model, train_loader, opt, device)
        va = eval_epoch(model, val_loader, device)
        history.append({"epoch": int(epoch), "train": tr, "val": va})
        log.info(f"epoch={epoch} train_loss={tr['loss']:.4f} train_acc={tr['acc']:.3f} val_loss={va['loss']:.4f} val_acc={va['acc']:.3f}")
        if float(va["loss"]) < best:
            best = float(va["loss"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
                },
                str(best_path),
            )

    report = {
        "ok": True,
        "task": "train_casd_autoencoder",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {"segment_graph_npz": str(args.segment_graph_npz), "routes_npz": str(args.routes_npz)},
        "out_dir": str(out_dir),
        "best_val_loss": float(best),
        "best_ckpt": str(best_path),
        "history": history,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"saved: {out_dir/'report.json'}")


if __name__ == "__main__":
    main()
