"""
Train Decision Point AR Model (Proposal C).

Training data:
- From decision_point_graph.npz: dp_seq_pad, dp_seq_len, start_t, etc.
- Each training sample: (current_dp, dest_dp, hour, candidate_set, target_idx)

Key difference from node-level AR training:
- Training samples are (dp_i -> dp_{i+1}) transitions, NOT (node -> node)
- Candidate set is the observed successors of current_dp, NOT neighbors
- Sequence length is ~5-10 steps (not 500+)
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.road_graph.ar_decision_point import (
    DecisionPointARModelSimple,
    DPARConfig,
)

TZ_SHANGHAI = timezone(timedelta(hours=8))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


@dataclass
class TrainCfg:
    batch_size: int
    n_epochs: int
    lr: float
    weight_decay: float
    max_candidates: int
    val_ratio: float
    seed: int
    device: str


class DPARDataset(Dataset):
    """
    Dataset for Decision Point AR training.
    
    Each sample is a transition: (current_dp, next_dp) from a DP sequence.
    We need to construct candidate sets from the decision point graph.
    """
    
    def __init__(
        self,
        dp_seq_pad: np.ndarray,    # (N, max_len) dp index sequences
        dp_seq_len: np.ndarray,    # (N,) lengths
        start_t: np.ndarray,       # (N,) start timestamps (unix)
        dp_y: np.ndarray,          # (D,) y coords of decision points
        dp_x: np.ndarray,          # (D,) x coords
        dp_succ_ptr: np.ndarray,   # (D+1,) CSR pointer
        dp_succ_idx: np.ndarray,   # (E,) successor indices
        max_candidates: int,
    ):
        self.dp_seq_pad = dp_seq_pad
        self.dp_seq_len = dp_seq_len
        self.start_t = start_t
        self.dp_y = dp_y
        self.dp_x = dp_x
        self.dp_succ_ptr = dp_succ_ptr
        self.dp_succ_idx = dp_succ_idx
        self.max_candidates = max_candidates
        
        # Build transition samples: (route_id, step_idx)
        self.samples: List[Tuple[int, int]] = []
        n_routes = dp_seq_len.shape[0]
        for rid in range(n_routes):
            L = int(dp_seq_len[rid])
            if L < 2:
                continue
            # Each step except the last is a training sample
            for step in range(L - 1):
                self.samples.append((rid, step))
        
        log.info(f"DPARDataset: {len(self.samples)} transition samples from {n_routes} routes")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _get_successors(self, dp_idx: int) -> np.ndarray:
        """Get successors of a decision point from CSR graph."""
        start = int(self.dp_succ_ptr[dp_idx])
        end = int(self.dp_succ_ptr[dp_idx + 1])
        return self.dp_succ_idx[start:end].copy()
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rid, step = self.samples[idx]
        
        seq = self.dp_seq_pad[rid]
        seq_len = int(self.dp_seq_len[rid])
        
        current_dp = int(seq[step])
        next_dp = int(seq[step + 1])
        dest_dp = int(seq[seq_len - 1])  # Last dp in sequence is destination
        
        # Hour from timestamp
        ts = int(self.start_t[rid])
        dt = datetime.fromtimestamp(ts, tz=TZ_SHANGHAI)
        hour = dt.hour
        
        # Build candidate set: successors of current_dp
        successors = self._get_successors(current_dp)
        
        # Make sure next_dp is in candidates (it should be by construction)
        if next_dp not in successors:
            # This shouldn't happen, but handle gracefully
            successors = np.append(successors, next_dp)
        
        n_cand = min(len(successors), self.max_candidates)
        
        # Pad candidates
        cand_dp = np.full((self.max_candidates,), -1, dtype=np.int64)
        cand_dp[:n_cand] = successors[:n_cand]
        
        cand_mask = np.zeros((self.max_candidates,), dtype=bool)
        cand_mask[:n_cand] = True
        
        # Find target index (position of next_dp in candidate set)
        target_idx = int(np.where(cand_dp == next_dp)[0][0])
        
        # Positions
        current_y = float(self.dp_y[current_dp])
        current_x = float(self.dp_x[current_dp])
        dest_y = float(self.dp_y[dest_dp])
        dest_x = float(self.dp_x[dest_dp])
        
        cand_y = np.zeros((self.max_candidates,), dtype=np.float32)
        cand_x = np.zeros((self.max_candidates,), dtype=np.float32)
        for i in range(n_cand):
            c = int(cand_dp[i])
            cand_y[i] = float(self.dp_y[c])
            cand_x[i] = float(self.dp_x[c])
        
        return {
            "current_dp": torch.tensor(current_dp, dtype=torch.long),
            "dest_dp": torch.tensor(dest_dp, dtype=torch.long),
            "hour": torch.tensor(hour, dtype=torch.long),
            "current_y": torch.tensor(current_y, dtype=torch.float32),
            "current_x": torch.tensor(current_x, dtype=torch.float32),
            "dest_y": torch.tensor(dest_y, dtype=torch.float32),
            "dest_x": torch.tensor(dest_x, dtype=torch.float32),
            "cand_dp": torch.tensor(cand_dp, dtype=torch.long),
            "cand_y": torch.tensor(cand_y, dtype=torch.float32),
            "cand_x": torch.tensor(cand_x, dtype=torch.float32),
            "cand_mask": torch.tensor(cand_mask, dtype=torch.bool),
            "target_idx": torch.tensor(target_idx, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate batch of samples."""
    return {
        "current_dp": torch.stack([b["current_dp"] for b in batch]),
        "dest_dp": torch.stack([b["dest_dp"] for b in batch]),
        "hour": torch.stack([b["hour"] for b in batch]),
        "current_y": torch.stack([b["current_y"] for b in batch]),
        "current_x": torch.stack([b["current_x"] for b in batch]),
        "dest_y": torch.stack([b["dest_y"] for b in batch]),
        "dest_x": torch.stack([b["dest_x"] for b in batch]),
        "cand_dp": torch.stack([b["cand_dp"] for b in batch]),
        "cand_y": torch.stack([b["cand_y"] for b in batch]),
        "cand_x": torch.stack([b["cand_x"] for b in batch]),
        "cand_mask": torch.stack([b["cand_mask"] for b in batch]),
        "target_idx": torch.stack([b["target_idx"] for b in batch]),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in loader:
        # Move to device
        current_dp = batch["current_dp"].to(device)
        dest_dp = batch["dest_dp"].to(device)
        hour = batch["hour"].to(device)
        current_pos = (batch["current_y"].to(device), batch["current_x"].to(device))
        dest_pos = (batch["dest_y"].to(device), batch["dest_x"].to(device))
        cand_dp = batch["cand_dp"].to(device)
        cand_pos = (batch["cand_y"].to(device), batch["cand_x"].to(device))
        cand_mask = batch["cand_mask"].to(device)
        target_idx = batch["target_idx"].to(device)
        
        optimizer.zero_grad()
        
        logits = model(
            current_dp=current_dp,
            dest_dp=dest_dp,
            hour=hour,
            current_pos=current_pos,
            dest_pos=dest_pos,
            cand_dp=cand_dp,
            cand_pos=cand_pos,
            cand_mask=cand_mask,
        )
        
        loss = model.loss(logits, target_idx, cand_mask)
        loss.backward()
        optimizer.step()
        
        # Accuracy
        preds = logits.argmax(dim=-1)
        correct = (preds == target_idx).sum().item()
        
        total_loss += loss.item() * current_dp.size(0)
        total_correct += correct
        total_samples += current_dp.size(0)
    
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in loader:
        current_dp = batch["current_dp"].to(device)
        dest_dp = batch["dest_dp"].to(device)
        hour = batch["hour"].to(device)
        current_pos = (batch["current_y"].to(device), batch["current_x"].to(device))
        dest_pos = (batch["dest_y"].to(device), batch["dest_x"].to(device))
        cand_dp = batch["cand_dp"].to(device)
        cand_pos = (batch["cand_y"].to(device), batch["cand_x"].to(device))
        cand_mask = batch["cand_mask"].to(device)
        target_idx = batch["target_idx"].to(device)
        
        logits = model(
            current_dp=current_dp,
            dest_dp=dest_dp,
            hour=hour,
            current_pos=current_pos,
            dest_pos=dest_pos,
            cand_dp=cand_dp,
            cand_pos=cand_pos,
            cand_mask=cand_mask,
        )
        
        loss = model.loss(logits, target_idx, cand_mask)
        preds = logits.argmax(dim=-1)
        correct = (preds == target_idx).sum().item()
        
        total_loss += loss.item() * current_dp.size(0)
        total_correct += correct
        total_samples += current_dp.size(0)
    
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def run(
    *,
    dp_graph_npz: Path,
    out_dir: Path,
    cfg: TrainCfg,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pt"
    report_json = out_dir / "report.json"
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")
    
    # Load decision point graph
    log.info(f"Loading {dp_graph_npz}")
    data = np.load(str(dp_graph_npz), allow_pickle=True)
    
    dp_seq_pad = np.asarray(data["dp_seq_pad"], dtype=np.int64)
    dp_seq_len = np.asarray(data["dp_seq_len"], dtype=np.int32)
    start_t = np.asarray(data["start_t"], dtype=np.int64)
    dp_y = np.asarray(data["dp_y"], dtype=np.float32)
    dp_x = np.asarray(data["dp_x"], dtype=np.float32)
    dp_succ_ptr = np.asarray(data["dp_succ_ptr"], dtype=np.int64)
    dp_succ_idx = np.asarray(data["dp_succ_idx"], dtype=np.int32)
    decision_points = np.asarray(data["decision_points"], dtype=np.int64)
    
    n_dp = len(decision_points)
    log.info(f"Loaded {n_dp} decision points, {dp_seq_len.shape[0]} routes")
    
    # Create dataset
    dataset = DPARDataset(
        dp_seq_pad=dp_seq_pad,
        dp_seq_len=dp_seq_len,
        start_t=start_t,
        dp_y=dp_y,
        dp_x=dp_x,
        dp_succ_ptr=dp_succ_ptr,
        dp_succ_idx=dp_succ_idx,
        max_candidates=cfg.max_candidates,
    )
    
    # Split train/val
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    n_val = int(n_samples * cfg.val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_subset = torch.utils.data.Subset(dataset, train_indices.tolist())
    val_subset = torch.utils.data.Subset(dataset, val_indices.tolist())
    
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    log.info(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")
    
    # Create model
    model_cfg = DPARConfig(max_candidates=cfg.max_candidates)
    model = DecisionPointARModelSimple(model_cfg, n_decision_points=n_dp)
    model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    history: List[Dict[str, float]] = []
    
    for epoch in range(cfg.n_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = eval_epoch(model, val_loader, device)
        
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
        })
        
        log.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
        )
        
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_cfg": model_cfg.__dict__,
                "n_decision_points": n_dp,
                "decision_points": decision_points,
                "epoch": epoch,
                "val_acc": best_val_acc,
            }, str(model_path))
    
    # Report
    report = {
        "ok": True,
        "task": "train_graph_ar_decision_point",
        "inputs": {
            "dp_graph_npz": str(dp_graph_npz),
        },
        "config": {
            "batch_size": cfg.batch_size,
            "n_epochs": cfg.n_epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "max_candidates": cfg.max_candidates,
            "val_ratio": cfg.val_ratio,
            "seed": cfg.seed,
            "device": cfg.device,
        },
        "stats": {
            "n_decision_points": n_dp,
            "n_train_samples": len(train_subset),
            "n_val_samples": len(val_subset),
            "n_params": n_params,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
        },
        "outputs": {
            "model_path": str(model_path),
        },
        "history": history[-5:],  # Last 5 epochs
        "meta": {"created_at": datetime.now(tz=TZ_SHANGHAI).isoformat()},
    }
    
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Training complete. Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    log.info(f"Model saved to {model_path}")
    
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Train Decision Point AR Model")
    p.add_argument("--dp_graph_npz", type=Path, required=True, help="Path to decision_point_graph.npz")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_candidates", type=int, default=32)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    
    cfg = TrainCfg(
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_candidates=args.max_candidates,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
    )
    
    run(
        dp_graph_npz=args.dp_graph_npz,
        out_dir=args.out_dir,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
