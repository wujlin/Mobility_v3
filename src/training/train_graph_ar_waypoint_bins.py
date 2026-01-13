from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(x, *args, **kwargs):  # type: ignore[no-redef]
        return x

from src.models.road_graph import ARGraphWaypointBin, WaypointBinARConfig


TZ_SHANGHAI = timezone(timedelta(hours=8))


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _time_features(start_t: np.ndarray, *, tz_offset_hours: float) -> np.ndarray:
    """
    Output (N,5): [sin(hour), cos(hour), sin(dow), cos(dow), is_weekend]
    """
    t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    t = (t + int(round(float(tz_offset_hours) * 3600.0))).astype(np.int64, copy=False)
    sec = np.mod(t, 86400).astype(np.float64, copy=False)
    hour = sec / 86400.0 * (2.0 * np.pi)
    day = (t // 86400).astype(np.int64, copy=False)
    dow = np.mod(day, 7).astype(np.float64, copy=False) / 7.0 * (2.0 * np.pi)
    is_weekend = (np.mod(day, 7) >= 5).astype(np.float64, copy=False)
    return np.stack([np.sin(hour), np.cos(hour), np.sin(dow), np.cos(dow), is_weekend], axis=1).astype(np.float32, copy=False)


def _load_graph_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    need = {"node_y", "node_x", "edge_u", "edge_tier", "meta"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"road_graph.npz missing keys: {missing}")
    meta = data["meta"]
    if isinstance(meta, np.ndarray) and meta.shape == ():
        meta = meta.item()
    if not isinstance(meta, dict):
        raise ValueError("road_graph.npz meta must be a dict.")
    return {
        "node_y": np.asarray(data["node_y"], dtype=np.float32).reshape(-1),
        "node_x": np.asarray(data["node_x"], dtype=np.float32).reshape(-1),
        "edge_u": np.asarray(data["edge_u"], dtype=np.int32).reshape(-1),
        "edge_tier": np.asarray(data["edge_tier"], dtype=np.uint8).reshape(-1),
        "meta": meta,
    }


def _load_waypoints_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    need = {"wp_seq", "start_t", "traj_idx"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"waypoints_graph.npz missing keys: {missing}")
    meta = data["meta"] if "meta" in data.files else None
    if isinstance(meta, np.ndarray) and meta.shape == ():
        meta = meta.item()
    return {
        "wp_seq": np.asarray(data["wp_seq"], dtype=np.int32),
        "start_t": np.asarray(data["start_t"], dtype=np.int64).reshape(-1),
        "traj_idx": np.asarray(data["traj_idx"], dtype=np.int64).reshape(-1),
        "route_city": np.asarray(data["route_city"], dtype=np.int8).reshape(-1) if "route_city" in data.files and data["route_city"] is not None else None,
        "meta": meta if isinstance(meta, dict) else None,
    }


def _node_tier_min(*, n_nodes: int, edge_u: np.ndarray, edge_tier: np.ndarray) -> np.ndarray:
    n = int(n_nodes)
    out = np.full((n,), 3, dtype=np.int64)
    u = np.asarray(edge_u, dtype=np.int64).reshape(-1)
    t = np.asarray(edge_tier, dtype=np.int64).reshape(-1)
    if u.size != t.size:
        raise ValueError("edge_u/edge_tier length mismatch")
    for uu, tt in zip(u.tolist(), t.tolist()):
        if 0 <= int(uu) < n:
            out[int(uu)] = min(int(out[int(uu)]), int(np.clip(int(tt), 0, 3)))
    return out


def _bin_class_from_nodes(*, node_y: np.ndarray, node_x: np.ndarray, nodes: np.ndarray, wp_bin: int, H: int, W: int) -> Tuple[np.ndarray, int, int]:
    wp_bin = int(wp_bin)
    if wp_bin <= 0:
        raise ValueError("--wp_bin must be > 0")
    n_by = int((int(H) + wp_bin - 1) // wp_bin)
    n_bx = int((int(W) + wp_bin - 1) // wp_bin)
    nn = np.asarray(nodes, dtype=np.int64).reshape(-1)
    y = node_y[nn].astype(np.float64, copy=False)
    x = node_x[nn].astype(np.float64, copy=False)
    by = np.floor(y / float(wp_bin)).astype(np.int64, copy=False)
    bx = np.floor(x / float(wp_bin)).astype(np.int64, copy=False)
    by = np.clip(by, 0, n_by - 1)
    bx = np.clip(bx, 0, n_bx - 1)
    cls = (by * int(n_bx) + bx).astype(np.int64, copy=False)
    return cls.astype(np.int64, copy=False), int(n_by), int(n_bx)


class WaypointDataset(Dataset):
    def __init__(
        self,
        *,
        cur: np.ndarray,
        dest: np.ndarray,
        time_feat: np.ndarray,
        route_city: np.ndarray,
        step_idx: np.ndarray,
        target_cls: np.ndarray,
    ) -> None:
        self.cur = np.asarray(cur, dtype=np.int64).reshape(-1)
        self.dest = np.asarray(dest, dtype=np.int64).reshape(-1)
        self.time_feat = np.asarray(time_feat, dtype=np.float32).reshape(-1, 5)
        self.route_city = np.asarray(route_city, dtype=np.int64).reshape(-1)
        self.step_idx = np.asarray(step_idx, dtype=np.int64).reshape(-1)
        self.target_cls = np.asarray(target_cls, dtype=np.int64).reshape(-1)
        n = int(self.cur.size)
        if not (
            int(self.dest.size) == n
            and int(self.route_city.size) == n
            and int(self.step_idx.size) == n
            and int(self.target_cls.size) == n
            and int(self.time_feat.shape[0]) == n
        ):
            raise ValueError("WaypointDataset length mismatch")

    def __len__(self) -> int:
        return int(self.cur.size)

    def __getitem__(self, idx: int) -> dict:
        i = int(idx)
        return {
            "cur": int(self.cur[i]),
            "dest": int(self.dest[i]),
            "time_feat": self.time_feat[i].astype(np.float32, copy=False),
            "route_city": int(self.route_city[i]),
            "step_idx": int(self.step_idx[i]),
            "target_cls": int(self.target_cls[i]),
        }


@dataclass(frozen=True)
class TrainCfg:
    waypoints_npz: str
    road_graph_npz: str
    out_dir: str
    wp_bin: int
    tz_offset_hours: float
    val_frac: float
    hidden_dim: int
    batch_size: int
    epochs: int
    lr: float
    l2: float
    num_workers: int
    seed: int


def run_train(*, cfg: TrainCfg) -> Dict[str, object]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_log = out_dir / "train.log"
    train_summary_json = out_dir / "train_summary.json"
    checkpoint = out_dir / "last.pt"

    g = _load_graph_npz(Path(cfg.road_graph_npz))
    node_y = g["node_y"]
    node_x = g["node_x"]
    meta_g = g["meta"]
    H = int(meta_g.get("grid", {}).get("H", 1024))
    W = int(meta_g.get("grid", {}).get("W", 1024))
    n_nodes = int(node_y.size)

    w = _load_waypoints_npz(Path(cfg.waypoints_npz))
    wp_seq = w["wp_seq"].astype(np.int64, copy=False)
    start_t = w["start_t"]
    route_city_raw = w["route_city"]
    meta_w = w["meta"]

    N, L = wp_seq.shape
    num_steps = int(L - 2)  # internal waypoints count
    if num_steps <= 0:
        raise ValueError("wp_seq must include at least [start,dest]")

    # Flatten per-step samples.
    # current = wp_seq[:, i], target = wp_seq[:, i+1] for i in [0..num_steps-1]
    cur = wp_seq[:, :num_steps].reshape(-1).astype(np.int64, copy=False)
    target_nodes = wp_seq[:, 1 : num_steps + 1].reshape(-1).astype(np.int64, copy=False)
    dest = np.repeat(wp_seq[:, -1].astype(np.int64, copy=False), repeats=num_steps, axis=0)
    step_idx = np.tile(np.arange(num_steps, dtype=np.int64), reps=int(N))

    tf = _time_features(start_t, tz_offset_hours=float(cfg.tz_offset_hours))
    tf_rep = np.repeat(tf, repeats=num_steps, axis=0)

    if route_city_raw is None:
        route_city = np.zeros((int(N),), dtype=np.int64)
        num_cities = 1
    else:
        route_city = route_city_raw.astype(np.int64, copy=False)
        num_cities = int(np.max(route_city) + 1)
        num_cities = max(1, num_cities)
    route_city_rep = np.repeat(route_city, repeats=num_steps, axis=0)

    target_cls, n_by, n_bx = _bin_class_from_nodes(node_y=node_y, node_x=node_x, nodes=target_nodes, wp_bin=int(cfg.wp_bin), H=H, W=W)
    n_classes = int(n_by * n_bx)

    # Split by route id to avoid leakage across steps.
    rng = np.random.default_rng(int(cfg.seed))
    route_ids = np.arange(int(N), dtype=np.int64)
    rng.shuffle(route_ids)
    n_val = int(round(float(cfg.val_frac) * float(N)))
    n_val = int(np.clip(n_val, 1, int(N) - 1))
    val_routes = np.sort(route_ids[:n_val])
    train_routes = np.sort(route_ids[n_val:])

    rid_rep = np.repeat(np.arange(int(N), dtype=np.int64), repeats=num_steps, axis=0)
    m_train = np.isin(rid_rep, train_routes)
    m_val = np.isin(rid_rep, val_routes)

    ds_train = WaypointDataset(
        cur=cur[m_train],
        dest=dest[m_train],
        time_feat=tf_rep[m_train],
        route_city=route_city_rep[m_train],
        step_idx=step_idx[m_train],
        target_cls=target_cls[m_train],
    )
    ds_val = WaypointDataset(
        cur=cur[m_val],
        dest=dest[m_val],
        time_feat=tf_rep[m_val],
        route_city=route_city_rep[m_val],
        step_idx=step_idx[m_val],
        target_cls=target_cls[m_val],
    )

    dl_train = DataLoader(ds_train, batch_size=int(cfg.batch_size), shuffle=True, num_workers=int(cfg.num_workers), pin_memory=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=int(cfg.batch_size), shuffle=False, num_workers=int(cfg.num_workers), pin_memory=True, drop_last=False)

    node_tier_min = _node_tier_min(n_nodes=n_nodes, edge_u=g["edge_u"], edge_tier=g["edge_tier"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_yx = torch.from_numpy(np.stack([node_y, node_x], axis=1).astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)
    node_tier_min_t = torch.from_numpy(node_tier_min.astype(np.int64, copy=False)).to(device=device, dtype=torch.long)

    model = ARGraphWaypointBin(
        cfg=WaypointBinARConfig(hidden_dim=int(cfg.hidden_dim), num_cities=int(num_cities)),
        n_classes=int(n_classes),
        num_steps=int(num_steps),
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.l2))
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = None
    best_epoch = None
    t0 = time.time()

    with train_log.open("w", encoding="utf-8") as f:
        for epoch in range(1, int(cfg.epochs) + 1):
            model.train()
            tr_loss = 0.0
            tr_ok = 0
            tr_n = 0
            for batch in tqdm(dl_train, desc=f"train {epoch}/{int(cfg.epochs)}", dynamic_ncols=True, leave=False):
                cur_b = torch.as_tensor(batch["cur"], device=device, dtype=torch.long)
                dest_b = torch.as_tensor(batch["dest"], device=device, dtype=torch.long)
                tf_b = torch.as_tensor(batch["time_feat"], device=device, dtype=torch.float32)
                city_b = torch.as_tensor(batch["route_city"], device=device, dtype=torch.long)
                step_b = torch.as_tensor(batch["step_idx"], device=device, dtype=torch.long)
                y_b = torch.as_tensor(batch["target_cls"], device=device, dtype=torch.long)

                logits, _ = model(
                    node_yx=node_yx,
                    node_tier_min=node_tier_min_t,
                    cur=cur_b,
                    dest=dest_b,
                    time_feat=tf_b,
                    route_city=city_b,
                    step_idx=step_b,
                )
                loss = loss_fn(logits, y_b)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                tr_loss += float(loss.detach().cpu().item()) * int(y_b.shape[0])
                pred = torch.argmax(logits.detach(), dim=1)
                tr_ok += int(torch.sum(pred == y_b).detach().cpu().item())
                tr_n += int(y_b.shape[0])

            model.eval()
            va_loss = 0.0
            va_ok = 0
            va_n = 0
            with torch.no_grad():
                for batch in dl_val:
                    cur_b = torch.as_tensor(batch["cur"], device=device, dtype=torch.long)
                    dest_b = torch.as_tensor(batch["dest"], device=device, dtype=torch.long)
                    tf_b = torch.as_tensor(batch["time_feat"], device=device, dtype=torch.float32)
                    city_b = torch.as_tensor(batch["route_city"], device=device, dtype=torch.long)
                    step_b = torch.as_tensor(batch["step_idx"], device=device, dtype=torch.long)
                    y_b = torch.as_tensor(batch["target_cls"], device=device, dtype=torch.long)
                    logits, _ = model(
                        node_yx=node_yx,
                        node_tier_min=node_tier_min_t,
                        cur=cur_b,
                        dest=dest_b,
                        time_feat=tf_b,
                        route_city=city_b,
                        step_idx=step_b,
                    )
                    loss = loss_fn(logits, y_b)
                    va_loss += float(loss.detach().cpu().item()) * int(y_b.shape[0])
                    pred = torch.argmax(logits.detach(), dim=1)
                    va_ok += int(torch.sum(pred == y_b).detach().cpu().item())
                    va_n += int(y_b.shape[0])

            tr_loss_m = tr_loss / float(max(1, tr_n))
            va_loss_m = va_loss / float(max(1, va_n))
            tr_acc = float(tr_ok) / float(max(1, tr_n))
            va_acc = float(va_ok) / float(max(1, va_n))

            if best_val_acc is None or va_acc > float(best_val_acc):
                best_val_acc = float(va_acc)
                best_epoch = int(epoch)

            row = {"epoch": int(epoch), "train_loss": tr_loss_m, "train_acc": tr_acc, "val_loss": va_loss_m, "val_acc": va_acc}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    elapsed = float(time.time() - t0)

    ckpt = {
        "cfg": asdict(cfg),
        "model_cfg": {"hidden_dim": int(cfg.hidden_dim), "num_cities": int(num_cities), "num_steps": int(num_steps), "n_classes": int(n_classes), "wp_bin": int(cfg.wp_bin)},
        "classes": {"n_by": int(n_by), "n_bx": int(n_bx)},
        "model": model.state_dict(),
        "meta": {"created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(), "graph_meta": meta_g, "waypoints_meta": meta_w},
    }
    torch.save(ckpt, str(checkpoint))

    summary = {
        "ok": True,
        "task": "train_graph_ar_waypoint_bins",
        "config": asdict(cfg),
        "stats": {
            "n_nodes": int(n_nodes),
            "n_routes": int(N),
            "num_steps": int(num_steps),
            "n_classes": int(n_classes),
            "n_bins_y": int(n_by),
            "n_bins_x": int(n_bx),
            "train_samples": int(len(ds_train)),
            "val_samples": int(len(ds_val)),
            "best_val_acc": float(best_val_acc) if best_val_acc is not None else None,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "elapsed_s": elapsed,
        },
        "outputs": {"checkpoint": str(checkpoint), "train_log": str(train_log), "train_summary_json": str(train_summary_json)},
        "meta": {"created_at": ckpt["meta"]["created_at"]},
    }
    train_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train waypoint-level AR (bin classification) from GT waypoint sequences.")
    p.add_argument("--waypoints_npz", type=str, required=True)
    p.add_argument("--road_graph_npz", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--wp_bin", type=int, default=32, help="Bin size in grid units (e.g., 32 => 32x32 bins => 1024 classes).")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))
    cfg = TrainCfg(
        waypoints_npz=str(args.waypoints_npz),
        road_graph_npz=str(args.road_graph_npz),
        out_dir=str(args.out_dir),
        wp_bin=int(args.wp_bin),
        tz_offset_hours=float(args.tz_offset_hours),
        val_frac=float(args.val_frac),
        hidden_dim=int(args.hidden_dim),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    report = run_train(cfg=cfg)
    compact = {"ok": True, "best_val_acc": report["stats"]["best_val_acc"], "checkpoint": report["outputs"]["checkpoint"], "train_summary_json": report["outputs"]["train_summary_json"]}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

