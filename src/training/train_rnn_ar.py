from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.baselines.rnn_ar import WayRNNAR, WayRNNARCfg
from src.data.way_graph.way_sequence_dataset import WayRouteDataset, load_way_routes_npz
from src.utils.time_unix import dow_from_unix, hour_from_unix
from src.utils.way_csr import build_candidate_row, infer_n_ways_from_ptr, slice_csr


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _split_dataset(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(n)
    if n <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_val = int(round(float(val_ratio) * float(n)))
    n_val = max(1, min(n - 1, n_val)) if n >= 2 else 0
    val = idx[:n_val]
    tr = idx[n_val:]
    return tr, val


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _subset_indices_from_route_ids(dataset: WayRouteDataset, route_ids: np.ndarray) -> np.ndarray:
    route_ids = np.asarray(route_ids, dtype=np.int64).reshape(-1)
    if route_ids.size == 0:
        return np.zeros((0,), dtype=np.int64)
    mask = np.isin(dataset.route_ids.astype(np.int64, copy=False), route_ids, assume_unique=False)
    return np.nonzero(mask)[0].astype(np.int64, copy=False)


def _collate_way_routes(batch: List[Dict[str, np.ndarray]], *, tz_offset_hours: float) -> Dict[str, torch.Tensor]:
    B = int(len(batch))
    way_lens = np.asarray([int(b["way_len"]) for b in batch], dtype=np.int64)
    Kmax = int(way_lens.max()) if B > 0 else 1
    way_pad = np.full((B, Kmax), -1, dtype=np.int64)
    for i, b in enumerate(batch):
        L = int(b["way_len"])
        way_pad[i, :L] = np.asarray(b["way_seq"], dtype=np.int64)[:L]

    start_t = np.asarray([int(b["start_t"]) for b in batch], dtype=np.int64)
    hour = hour_from_unix(start_t, tz_offset_hours=float(tz_offset_hours))
    dow = dow_from_unix(start_t, tz_offset_hours=float(tz_offset_hours))

    route_cond = {
        "start_pos": torch.as_tensor(np.stack([b["start_pos"] for b in batch], axis=0), dtype=torch.float32),
        "dest_pos": torch.as_tensor(np.stack([b["dest_pos"] for b in batch], axis=0), dtype=torch.float32),
        "hour": torch.as_tensor(hour, dtype=torch.long),
        "dow": torch.as_tensor(dow, dtype=torch.long),
        "route_city": torch.as_tensor(np.asarray([int(b["route_city"]) for b in batch], dtype=np.int64), dtype=torch.long),
    }
    return {
        "route_id": torch.as_tensor(np.asarray([int(b["route_id"]) for b in batch], dtype=np.int64), dtype=torch.long),
        "way_seq_pad": torch.as_tensor(way_pad, dtype=torch.long),
        "way_seq_len": torch.as_tensor(way_lens, dtype=torch.long),
        "route_cond": route_cond,
        "start_way": torch.as_tensor(np.asarray([int(b["start_way"]) for b in batch], dtype=np.int64), dtype=torch.long),
        "dest_way": torch.as_tensor(np.asarray([int(b["dest_way"]) for b in batch], dtype=np.int64), dtype=torch.long),
    }


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device=device) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _build_candidates_t(
    *,
    ptr: np.ndarray,
    idx: np.ndarray,
    cur_way: np.ndarray,  # (B,)
    next_way: np.ndarray,  # (B,)
    valid: np.ndarray,  # (B,) bool
    max_candidates: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = int(cur_way.size)
    C = int(max_candidates)
    cand = np.full((B, C), -1, dtype=np.int64)
    mask = np.zeros((B, C), dtype=bool)
    tgt_idx = np.zeros((B,), dtype=np.int64)
    tgt_mask = np.asarray(valid, dtype=bool).reshape(-1)

    for i in range(B):
        if not bool(tgt_mask[i]):
            continue
        u = int(cur_way[i])
        v = int(next_way[i])
        succ = slice_csr(ptr, idx, u)
        row = build_candidate_row(succ, max_candidates=C, target=v)
        cand[i] = row.cand
        mask[i] = row.mask
        tgt_idx[i] = int(row.target_idx if row.target_idx is not None else 0)

    cand_t = torch.as_tensor(cand, dtype=torch.long)
    mask_t = torch.as_tensor(mask, dtype=torch.bool)
    tgt_t = torch.as_tensor(tgt_idx, dtype=torch.long)
    tgt_mask_t = torch.as_tensor(tgt_mask, dtype=torch.bool)
    return cand_t, mask_t, tgt_t, tgt_mask_t


def _run_epoch(
    *,
    model: WayRNNAR,
    loader: DataLoader,
    device: torch.device,
    ptr: np.ndarray,
    idx: np.ndarray,
    max_candidates: int,
    train: bool,
    opt: Optional[torch.optim.Optimizer],
    max_batches: Optional[int],
) -> Dict[str, float]:
    if train:
        if opt is None:
            raise ValueError("train=True requires an optimizer")
        model.train()
    else:
        model.eval()

    losses: List[float] = []
    accs: List[float] = []
    n_steps_total = 0

    for bi, batch in enumerate(loader):
        b = _to_device(batch, device)
        way = b["way_seq_pad"]  # (B,K)
        lens = b["way_seq_len"]  # (B,)
        rc = b["route_cond"]

        B, K = way.shape
        cond_emb = model.encode_cond(rc)  # (B,D)
        h = model.init_state(cond_emb)  # (L,B,D)

        total = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        n_tok = torch.tensor(0.0, device=device)

        K_eff = int(K) - 1
        for t in range(int(K_eff)):
            cur = way[:, t]
            nxt = way[:, t + 1]
            valid = (t + 1 < lens).detach().cpu().numpy().astype(bool, copy=False)

            token, h_new = model.step(cur, cond_emb=cond_emb, h=h)
            upd = (cur >= 0).to(dtype=torch.bool).view(1, B, 1)
            h = torch.where(upd, h_new, h)

            cand_t, mask_t, tgt_t, tgt_mask_t = _build_candidates_t(
                ptr=ptr,
                idx=idx,
                cur_way=cur.detach().cpu().numpy(),
                next_way=nxt.detach().cpu().numpy(),
                valid=valid,
                max_candidates=int(max_candidates),
            )
            cand_t = cand_t.to(device=device)
            mask_t = mask_t.to(device=device)
            tgt_t = tgt_t.to(device=device)
            tgt_mask_t = tgt_mask_t.to(device=device)

            if not bool(tgt_mask_t.any()):
                continue

            logits = model.score_candidates(token, cand_t, mask_t)
            flat_logits = logits[tgt_mask_t]
            flat_tgt = tgt_t[tgt_mask_t]
            loss = F.cross_entropy(flat_logits, flat_tgt, reduction="mean")
            n_step = int(flat_tgt.numel())
            total = total + loss * float(n_step)

            with torch.no_grad():
                pred = torch.argmax(flat_logits, dim=-1)
                correct = correct + (pred == flat_tgt).float().sum()
                n_tok = n_tok + torch.as_tensor(int(n_step), device=device, dtype=torch.float32)

        denom = torch.clamp_min(n_tok, 1.0)
        loss_avg = total / denom

        if train and opt is not None:
            opt.zero_grad(set_to_none=True)
            loss_avg.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        losses.append(float(loss_avg.detach().cpu().item()))
        accs.append(float((correct / denom).detach().cpu().item()))
        n_steps_total += int(n_tok.detach().cpu().item())

        if max_batches is not None and int(max_batches) > 0 and (bi + 1) >= int(max_batches):
            break

    return {
        "loss": float(np.mean(np.asarray(losses, dtype=np.float64))) if losses else float("nan"),
        "acc": float(np.mean(np.asarray(accs, dtype=np.float64))) if accs else float("nan"),
        "n_tokens": float(n_steps_total),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train B2: RNN AR baseline (way-space, candidate CE).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument(
        "--split_json",
        type=Path,
        default=None,
        help="Optional OD-disjoint split json (expects splits.train/val/test route_ids). Overrides val_ratio.",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)

    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--max_routes", type=int, default=None)

    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_candidates", type=int, default=32)
    p.add_argument("--n_route_cities", type=int, default=4)
    p.add_argument("--coord_scale", type=float, default=1024.0)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--max_batches", type=int, default=None)
    args = p.parse_args()

    _set_seed(int(args.seed))
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}")

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    if args.split_json is not None and args.max_routes is not None:
        print("[WARN] --split_json is set, ignoring --max_routes to avoid inconsistent splits.", flush=True)
    dataset = WayRouteDataset(
        routes,
        max_routes=(None if args.split_json is not None else (int(args.max_routes) if args.max_routes is not None else None)),
        max_way_len=int(args.max_way_len),
        min_hops=int(args.min_hops),
    )
    if args.split_json is None:
        tr_idx, va_idx = _split_dataset(len(dataset), float(args.val_ratio), int(args.seed))
    else:
        split = _read_json(Path(args.split_json))
        splits = split.get("splits", split)
        tr_rids = np.asarray(splits.get("train", []), dtype=np.int64).reshape(-1)
        va_rids = np.asarray(splits.get("val", []), dtype=np.int64).reshape(-1)
        tr_idx = _subset_indices_from_route_ids(dataset, tr_rids)
        va_idx = _subset_indices_from_route_ids(dataset, va_rids)
        if int(tr_idx.size) == 0 or int(va_idx.size) == 0:
            raise SystemExit(
                f"[FATAL] split_json produced empty subsets: train_idx={int(tr_idx.size)} val_idx={int(va_idx.size)}. "
                "Check min_hops/max_way_len match split generation."
            )
        print(
            f"[split] {args.split_json} train_routes={int(tr_rids.size)} val_routes={int(va_rids.size)} "
            f"=> train_idx={int(tr_idx.size)} val_idx={int(va_idx.size)}",
            flush=True,
        )
    train_set = Subset(dataset, tr_idx.tolist())
    val_set = Subset(dataset, va_idx.tolist())
    print(f"[data] total={len(dataset)} train={len(train_set)} val={len(val_set)}", flush=True)

    wg = np.load(str(Path(args.way_graph_npz)), allow_pickle=True)
    ptr = np.asarray(wg["way_adj_ptr"], dtype=np.int64)
    idx = np.asarray(wg["way_adj_idx"], dtype=np.int64)
    n_ways = infer_n_ways_from_ptr(ptr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = WayRNNARCfg(
        n_ways=int(n_ways),
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
        max_candidates=int(args.max_candidates),
        max_len=int(args.max_way_len),
        n_route_cities=int(args.n_route_cities),
        coord_scale=float(args.coord_scale),
    )
    model = WayRNNAR(cfg=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    collate = lambda b: _collate_way_routes(b, tz_offset_hours=float(args.tz_offset_hours))
    pin = bool(device.type == "cuda")
    num_workers = max(0, int(args.num_workers))
    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        collate_fn=collate,
    )

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    best = float("inf")
    for epoch in range(1, int(args.n_epochs) + 1):
        t0 = time.time()
        tr = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            ptr=ptr,
            idx=idx,
            max_candidates=int(args.max_candidates),
            train=True,
            opt=opt,
            max_batches=args.max_batches,
        )
        va = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            ptr=ptr,
            idx=idx,
            max_candidates=int(args.max_candidates),
            train=False,
            opt=None,
            max_batches=args.max_batches,
        )
        dt = time.time() - t0
        print(
            f"[epoch {epoch:03d}] "
            f"train loss={tr['loss']:.4f} acc={tr['acc']:.3f} | "
            f"val loss={va['loss']:.4f} acc={va['acc']:.3f} | "
            f"time={dt:.1f}s"
        )

        ckpt = {
            "epoch": int(epoch),
            "cfg": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "train": tr,
            "val": va,
        }
        torch.save(ckpt, str(out_dir / "last.pt"))
        if float(va["loss"]) < float(best):
            best = float(va["loss"])
            torch.save(ckpt, str(out_dir / "ckpt_best.pt"))

        if int(args.save_every) > 0 and (epoch % int(args.save_every) == 0):
            torch.save(ckpt, str(out_dir / f"epoch_{epoch:03d}.pt"))

    print(f"[OK] saved: {out_dir}")


if __name__ == "__main__":
    main()
