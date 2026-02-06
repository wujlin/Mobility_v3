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
from torch.utils.data import DataLoader, Subset

from src.data.way_graph.way_sequence_dataset import WayRouteDataset, load_way_routes_npz
from src.sota.difftraj import DiffTrajCfg, DiffTrajModel
from src.utils.time_unix import dow_from_unix, hour_from_unix


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


def _resample_yx(yx: np.ndarray, T: int) -> np.ndarray:
    yx = np.asarray(yx, dtype=np.float32).reshape(-1, 2)
    L = int(yx.shape[0])
    if L <= 0:
        return np.zeros((int(T), 2), dtype=np.float32)
    if L == 1:
        return np.repeat(yx, repeats=int(T), axis=0)
    t_src = np.linspace(0.0, float(L - 1), num=L, dtype=np.float32)
    t_tgt = np.linspace(0.0, float(L - 1), num=int(T), dtype=np.float32)
    y_t = np.interp(t_tgt, t_src, yx[:, 0]).astype(np.float32, copy=False)
    x_t = np.interp(t_tgt, t_src, yx[:, 1]).astype(np.float32, copy=False)
    return np.stack([y_t, x_t], axis=1).astype(np.float32, copy=False)


def _collate_difftraj(
    batch: List[Dict[str, np.ndarray]],
    *,
    tz_offset_hours: float,
    way_center_y: np.ndarray,
    way_center_x: np.ndarray,
    traj_len: int,
    coord_scale: float,
) -> Dict[str, object]:
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

    start_pos = np.stack([b["start_pos"] for b in batch], axis=0).astype(np.float32, copy=False)  # (B,2) yx
    dest_pos = np.stack([b["dest_pos"] for b in batch], axis=0).astype(np.float32, copy=False)

    traj = np.zeros((B, int(traj_len), 2), dtype=np.float32)
    for i in range(B):
        L = int(way_lens[i])
        seq = way_pad[i, :L]
        seq = seq[(seq >= 0)]
        if seq.size == 0:
            yx = np.zeros((1, 2), dtype=np.float32)
        else:
            y = way_center_y[seq.astype(np.int64, copy=False)]
            x = way_center_x[seq.astype(np.int64, copy=False)]
            yx = np.stack([y, x], axis=1).astype(np.float32, copy=False)
        yx_rs = _resample_yx(yx, int(traj_len))  # (T,2) abs yx
        yx_rel = yx_rs - start_pos[i : i + 1, :]
        if float(coord_scale) > 0:
            yx_rel = yx_rel / float(coord_scale)
        traj[i] = yx_rel.astype(np.float32, copy=False)

    route_cond = {
        "start_pos": torch.as_tensor(start_pos, dtype=torch.float32),
        "dest_pos": torch.as_tensor(dest_pos, dtype=torch.float32),
        "hour": torch.as_tensor(hour, dtype=torch.long),
        "dow": torch.as_tensor(dow, dtype=torch.long),
        "route_city": torch.as_tensor(np.asarray([int(b["route_city"]) for b in batch], dtype=np.int64), dtype=torch.long),
    }
    return {
        "route_id": torch.as_tensor(np.asarray([int(b["route_id"]) for b in batch], dtype=np.int64), dtype=torch.long),
        "traj_yx_rel": torch.as_tensor(traj, dtype=torch.float32),  # (B,T,2)
        "route_cond": route_cond,
        "way_seq_len": torch.as_tensor(way_lens, dtype=torch.long),
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


def _run_epoch(
    *,
    model: DiffTrajModel,
    loader: DataLoader,
    device: torch.device,
    train: bool,
    opt: Optional[torch.optim.Optimizer],
    max_batches: Optional[int],
) -> float:
    if train:
        if opt is None:
            raise ValueError("train=True requires an optimizer")
        model.train()
    else:
        model.eval()

    losses: List[float] = []
    for bi, batch in enumerate(loader):
        b = _to_device(batch, device)
        traj = b["traj_yx_rel"]
        rc = b["route_cond"]

        loss = model.compute_loss(traj_yx_rel=traj, route_cond=rc)
        if train and opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        losses.append(float(loss.detach().cpu().item()))
        if max_batches is not None and int(max_batches) > 0 and (bi + 1) >= int(max_batches):
            break
    return float(np.mean(np.asarray(losses, dtype=np.float64))) if losses else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="Train S2 (simplified): DiffTraj diffusion on way-center GPS sequences.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
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

    p.add_argument("--traj_len", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--emb_dim", type=int, default=512)
    p.add_argument("--diffusion_steps", type=int, default=100)
    p.add_argument("--d_model", type=int, default=256)
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

    wf = np.load(str(Path(args.way_features_npz)), allow_pickle=True)
    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float32).reshape(-1)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float32).reshape(-1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DiffTrajCfg(
        traj_len=int(args.traj_len),
        hidden_dim=int(args.hidden_dim),
        emb_dim=int(args.emb_dim),
        diffusion_steps=int(args.diffusion_steps),
        d_model=int(args.d_model),
        n_route_cities=int(args.n_route_cities),
        coord_scale=float(args.coord_scale),
    )
    model = DiffTrajModel(cfg=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    collate = lambda b: _collate_difftraj(
        b,
        tz_offset_hours=float(args.tz_offset_hours),
        way_center_y=way_center_y,
        way_center_x=way_center_x,
        traj_len=int(args.traj_len),
        coord_scale=float(args.coord_scale),
    )
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
        tr_loss = _run_epoch(model=model, loader=train_loader, device=device, train=True, opt=opt, max_batches=args.max_batches)
        va_loss = _run_epoch(model=model, loader=val_loader, device=device, train=False, opt=None, max_batches=args.max_batches)
        dt = time.time() - t0
        print(f"[epoch {epoch:03d}] train loss={tr_loss:.4f} | val loss={va_loss:.4f} | time={dt:.1f}s")

        ckpt = {"epoch": int(epoch), "cfg": asdict(cfg), "model_state_dict": model.state_dict(), "train_loss": tr_loss, "val_loss": va_loss}
        torch.save(ckpt, str(out_dir / "last.pt"))
        if float(va_loss) < float(best):
            best = float(va_loss)
            torch.save(ckpt, str(out_dir / "ckpt_best.pt"))
        if int(args.save_every) > 0 and (epoch % int(args.save_every) == 0):
            torch.save(ckpt, str(out_dir / f"epoch_{epoch:03d}.pt"))

    print(f"[OK] saved: {out_dir}")


if __name__ == "__main__":
    main()
