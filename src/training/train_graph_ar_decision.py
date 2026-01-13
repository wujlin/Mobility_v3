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

from src.models.road_graph import ARDecisionConfig, ARGraphDecisionMarkov


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


def _build_neighbors_csr(edge_u: np.ndarray, edge_v: np.ndarray, edge_tier: np.ndarray, *, n_nodes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      ptr: (n_nodes+1,) int64
      idx: (E,) int32 (neighbors)
      tier:(E,) uint8
    """
    u = np.asarray(edge_u, dtype=np.int64).reshape(-1)
    v = np.asarray(edge_v, dtype=np.int64).reshape(-1)
    t = np.asarray(edge_tier, dtype=np.uint8).reshape(-1)
    if not (u.size == v.size == t.size):
        raise ValueError("edge_u/edge_v/edge_tier length mismatch")
    order = np.argsort(u, kind="mergesort")
    u = u[order]
    v = v[order].astype(np.int32, copy=False)
    t = t[order].astype(np.uint8, copy=False)
    if u.size == 0:
        ptr = np.zeros((int(n_nodes) + 1,), dtype=np.int64)
        return ptr, v, t
    if int(np.min(u)) < 0 or int(np.max(u)) >= int(n_nodes):
        raise ValueError(f"edge_u out of range: min={int(np.min(u))} max={int(np.max(u))} n_nodes={int(n_nodes)}")
    cnt = np.bincount(u.astype(np.int64, copy=False), minlength=int(n_nodes)).astype(np.int64, copy=False)
    ptr = np.zeros((int(n_nodes) + 1,), dtype=np.int64)
    ptr[1:] = np.cumsum(cnt)
    return ptr, v, t


def _load_graph_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    need = {"node_y", "node_x", "edge_u", "edge_v", "edge_tier", "meta"}
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
        "edge_v": np.asarray(data["edge_v"], dtype=np.int32).reshape(-1),
        "edge_tier": np.asarray(data["edge_tier"], dtype=np.uint8).reshape(-1),
        "meta": meta,
    }


def _load_paths_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    need = {"start_t", "start_node", "dest_node", "node_seq_pad", "node_seq_len", "traj_idx"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"paths_graph.npz missing keys: {missing}")
    meta = data["meta"] if "meta" in data.files else None
    if isinstance(meta, np.ndarray) and meta.shape == ():
        meta = meta.item()
    return {
        "start_t": np.asarray(data["start_t"], dtype=np.int64).reshape(-1),
        "start_node": np.asarray(data["start_node"], dtype=np.int32).reshape(-1),
        "dest_node": np.asarray(data["dest_node"], dtype=np.int32).reshape(-1),
        "node_seq_pad": np.asarray(data["node_seq_pad"], dtype=np.int32),
        "node_seq_len": np.asarray(data["node_seq_len"], dtype=np.int32).reshape(-1),
        "traj_idx": np.asarray(data["traj_idx"], dtype=np.int64).reshape(-1),
        "route_city": np.asarray(data["route_city"], dtype=np.int8).reshape(-1) if "route_city" in data.files else None,
        "meta": meta if isinstance(meta, dict) else None,
    }


def _build_transitions(
    *,
    node_seq_pad: np.ndarray,
    node_seq_len: np.ndarray,
    start_t: np.ndarray,
    route_ids: np.ndarray,
    max_steps_per_route: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    route_ids = np.asarray(route_ids, dtype=np.int64).reshape(-1)
    u_list = []
    v_list = []
    d_list = []
    rid_list = []
    for rid in route_ids.tolist():
        L = int(node_seq_len[int(rid)])
        if L < 2:
            continue
        seq = node_seq_pad[int(rid), :L].astype(np.int64, copy=False)
        dest = int(seq[-1])
        n_steps = int(L - 1)
        if max_steps_per_route > 0 and n_steps > int(max_steps_per_route):
            pick = rng.choice(n_steps, size=int(max_steps_per_route), replace=False)
            pick = np.sort(pick.astype(np.int64))
        else:
            pick = np.arange(n_steps, dtype=np.int64)
        for j in pick.tolist():
            uu = int(seq[int(j)])
            vv = int(seq[int(j) + 1])
            if uu < 0 or vv < 0:
                continue
            u_list.append(uu)
            v_list.append(vv)
            d_list.append(dest)
            rid_list.append(int(rid))
    if not u_list:
        raise RuntimeError("No transitions built (check node_seq_len / max_steps_per_route).")
    return {
        "u": np.asarray(u_list, dtype=np.int64),
        "v_next": np.asarray(v_list, dtype=np.int64),
        "dest": np.asarray(d_list, dtype=np.int64),
        "route_id": np.asarray(rid_list, dtype=np.int64),
    }


class TransitionDataset(Dataset):
    def __init__(self, u: np.ndarray, v_next: np.ndarray, dest: np.ndarray, route_id: np.ndarray) -> None:
        self.u = np.asarray(u, dtype=np.int64).reshape(-1)
        self.v_next = np.asarray(v_next, dtype=np.int64).reshape(-1)
        self.dest = np.asarray(dest, dtype=np.int64).reshape(-1)
        self.route_id = np.asarray(route_id, dtype=np.int64).reshape(-1)
        n = int(self.u.size)
        if not (int(self.v_next.size) == n and int(self.dest.size) == n and int(self.route_id.size) == n):
            raise ValueError("TransitionDataset length mismatch")

    def __len__(self) -> int:
        return int(self.u.size)

    def __getitem__(self, idx: int) -> dict:
        i = int(idx)
        return {
            "u": int(self.u[i]),
            "v_next": int(self.v_next[i]),
            "dest": int(self.dest[i]),
            "route_id": int(self.route_id[i]),
        }


@dataclass(frozen=True)
class TrainCfg:
    paths_graph_npz: str
    road_graph_npz: str
    out_dir: str
    tz_offset_hours: float
    max_steps_per_route: int
    val_frac: float
    hidden_dim: int
    edge_tier_emb_dim: int
    batch_size: int
    epochs: int
    lr: float
    l2: float
    num_workers: int
    seed: int


def _neighbors_batch(
    *,
    ptr: np.ndarray,
    idx: np.ndarray,
    tier: np.ndarray,
    u: np.ndarray,
    v_next: np.ndarray,
    max_deg_cap: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build padded neighbor matrices for a batch.
    Returns:
      neigh: (B,M) int64 (-1 pad)
      neigh_tier: (B,M) int64
      tgt_pos: (B,) int64 (-1 invalid)
      valid: (B,) bool
    """
    u = np.asarray(u, dtype=np.int64).reshape(-1)
    v_next = np.asarray(v_next, dtype=np.int64).reshape(-1)
    B = int(u.size)
    deg = (ptr[u + 1] - ptr[u]).astype(np.int64, copy=False)
    M = int(np.max(deg).item()) if B > 0 else 0
    if max_deg_cap > 0:
        M = int(min(int(M), int(max_deg_cap)))
    M = int(max(1, M))
    neigh = np.full((B, M), -1, dtype=np.int64)
    neigh_tier = np.zeros((B, M), dtype=np.int64)
    tgt_pos = np.full((B,), -1, dtype=np.int64)
    valid = np.zeros((B,), dtype=np.uint8)
    for i in range(B):
        uu = int(u[i])
        s = int(ptr[uu])
        e = int(ptr[uu + 1])
        if e <= s:
            continue
        vv = idx[s:e].astype(np.int64, copy=False)
        tt = tier[s:e].astype(np.int64, copy=False)
        if vv.size == 0:
            continue
        if vv.size > M:
            vv = vv[:M]
            tt = tt[:M]
        neigh[i, : vv.size] = vv
        neigh_tier[i, : tt.size] = tt
        # find target in vv
        hit = np.nonzero(vv == int(v_next[i]))[0]
        if hit.size > 0:
            tgt_pos[i] = int(hit[0])
            valid[i] = 1
    return neigh, neigh_tier, tgt_pos, valid.astype(bool)


def train(cfg: TrainCfg) -> Dict[str, object]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "last.pt"
    report_json = out_dir / "train_summary.json"
    train_log = out_dir / "train.log"

    g = _load_graph_npz(Path(cfg.road_graph_npz))
    n_nodes = int(g["node_y"].shape[0])
    ptr, n_idx, n_tier = _build_neighbors_csr(g["edge_u"], g["edge_v"], g["edge_tier"], n_nodes=n_nodes)

    p = _load_paths_npz(Path(cfg.paths_graph_npz))
    n_routes = int(p["start_t"].shape[0])
    tf_routes = _time_features(p["start_t"], tz_offset_hours=float(cfg.tz_offset_hours))

    # Split by route (avoid leakage).
    rng = np.random.default_rng(int(cfg.seed))
    order = rng.permutation(n_routes).astype(np.int64, copy=False)
    n_val = int(round(float(cfg.val_frac) * float(n_routes)))
    n_val = int(max(1, min(n_val, n_routes - 1)))
    val_routes = order[:n_val]
    train_routes = order[n_val:]

    tr = _build_transitions(
        node_seq_pad=p["node_seq_pad"],
        node_seq_len=p["node_seq_len"],
        start_t=p["start_t"],
        route_ids=train_routes,
        max_steps_per_route=int(cfg.max_steps_per_route),
        seed=int(cfg.seed),
    )
    va = _build_transitions(
        node_seq_pad=p["node_seq_pad"],
        node_seq_len=p["node_seq_len"],
        start_t=p["start_t"],
        route_ids=val_routes,
        max_steps_per_route=int(cfg.max_steps_per_route),
        seed=int(cfg.seed) + 7,
    )

    tr_ds = TransitionDataset(tr["u"], tr["v_next"], tr["dest"], tr["route_id"])
    va_ds = TransitionDataset(va["u"], va["v_next"], va["dest"], va["route_id"])

    dl_tr = DataLoader(tr_ds, batch_size=int(cfg.batch_size), shuffle=True, num_workers=int(cfg.num_workers), pin_memory=torch.cuda.is_available())
    dl_va = DataLoader(va_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=int(cfg.num_workers), pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ARGraphDecisionMarkov(cfg=ARDecisionConfig(hidden_dim=int(cfg.hidden_dim), edge_tier_emb_dim=int(cfg.edge_tier_emb_dim))).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.l2))
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    node_yx = torch.from_numpy(np.stack([g["node_y"], g["node_x"]], axis=1).astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)
    tf_routes_t = torch.from_numpy(tf_routes).to(device=device, dtype=torch.float32)

    best_val = -1.0
    best_epoch = -1
    t0 = time.time()

    with train_log.open("w", encoding="utf-8") as flog:
        for epoch in range(1, int(cfg.epochs) + 1):
            model.train()
            loss_sum = 0.0
            acc_sum = 0.0
            n_sum = 0

            it = tqdm(dl_tr, desc=f"epoch {epoch}/{int(cfg.epochs)}", dynamic_ncols=True)
            for batch in it:
                u = np.asarray(batch["u"], dtype=np.int64)
                v_next = np.asarray(batch["v_next"], dtype=np.int64)
                dest = np.asarray(batch["dest"], dtype=np.int64)
                rid = np.asarray(batch["route_id"], dtype=np.int64)

                neigh, neigh_tier, tgt_pos, valid = _neighbors_batch(ptr=ptr, idx=n_idx, tier=n_tier, u=u, v_next=v_next, max_deg_cap=0)
                if not np.any(valid):
                    continue
                u_t = torch.from_numpy(u[valid]).to(device=device, dtype=torch.long)
                d_t = torch.from_numpy(dest[valid]).to(device=device, dtype=torch.long)
                neigh_t = torch.from_numpy(neigh[valid]).to(device=device, dtype=torch.long)
                tier_t = torch.from_numpy(neigh_tier[valid]).to(device=device, dtype=torch.long)
                tgt_t = torch.from_numpy(tgt_pos[valid]).to(device=device, dtype=torch.long)
                rid_t = torch.from_numpy(rid[valid]).to(device=device, dtype=torch.long)
                time_feat = tf_routes_t[rid_t]

                opt.zero_grad(set_to_none=True)
                logits, _ = model.score_neighbors(node_yx=node_yx, cur=u_t, dest=d_t, neigh=neigh_t, neigh_tier=tier_t, time_feat=time_feat)
                loss = loss_fn(logits, tgt_t).mean()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    acc = float((pred == tgt_t).float().mean().item())
                loss_sum += float(loss.item()) * int(tgt_t.shape[0])
                acc_sum += float(acc) * int(tgt_t.shape[0])
                n_sum += int(tgt_t.shape[0])
                it.set_postfix(loss=float(loss.item()), acc=float(acc))

            train_loss = float(loss_sum / max(1, n_sum))
            train_acc = float(acc_sum / max(1, n_sum))

            # Validation
            model.eval()
            v_loss_sum = 0.0
            v_acc_sum = 0.0
            v_n = 0
            with torch.no_grad():
                for batch in dl_va:
                    u = np.asarray(batch["u"], dtype=np.int64)
                    v_next = np.asarray(batch["v_next"], dtype=np.int64)
                    dest = np.asarray(batch["dest"], dtype=np.int64)
                    rid = np.asarray(batch["route_id"], dtype=np.int64)

                    neigh, neigh_tier, tgt_pos, valid = _neighbors_batch(ptr=ptr, idx=n_idx, tier=n_tier, u=u, v_next=v_next, max_deg_cap=0)
                    if not np.any(valid):
                        continue
                    u_t = torch.from_numpy(u[valid]).to(device=device, dtype=torch.long)
                    d_t = torch.from_numpy(dest[valid]).to(device=device, dtype=torch.long)
                    neigh_t = torch.from_numpy(neigh[valid]).to(device=device, dtype=torch.long)
                    tier_t = torch.from_numpy(neigh_tier[valid]).to(device=device, dtype=torch.long)
                    tgt_t = torch.from_numpy(tgt_pos[valid]).to(device=device, dtype=torch.long)
                    rid_t = torch.from_numpy(rid[valid]).to(device=device, dtype=torch.long)
                    time_feat = tf_routes_t[rid_t]

                    logits, _ = model.score_neighbors(node_yx=node_yx, cur=u_t, dest=d_t, neigh=neigh_t, neigh_tier=tier_t, time_feat=time_feat)
                    loss = loss_fn(logits, tgt_t).mean()
                    pred = torch.argmax(logits, dim=1)
                    acc = float((pred == tgt_t).float().mean().item())
                    v_loss_sum += float(loss.item()) * int(tgt_t.shape[0])
                    v_acc_sum += float(acc) * int(tgt_t.shape[0])
                    v_n += int(tgt_t.shape[0])

            val_loss = float(v_loss_sum / max(1, v_n))
            val_acc = float(v_acc_sum / max(1, v_n))

            flog.write(json.dumps({"epoch": int(epoch), "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}, ensure_ascii=False) + "\n")
            flog.flush()

            if val_acc > best_val:
                best_val = float(val_acc)
                best_epoch = int(epoch)
                torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "meta": {"best_epoch": best_epoch, "best_val_acc": best_val}}, ckpt_path)

    elapsed = float(time.time() - t0)
    summary = {
        "ok": True,
        "task": "train_graph_ar_decision",
        "config": asdict(cfg),
        "stats": {
            "n_nodes": int(n_nodes),
            "n_routes": int(n_routes),
            "train_routes": int(train_routes.size),
            "val_routes": int(val_routes.size),
            "train_transitions": int(len(tr_ds)),
            "val_transitions": int(len(va_ds)),
            "best_val_acc": float(best_val),
            "best_epoch": int(best_epoch),
            "elapsed_s": float(elapsed),
        },
        "outputs": {"checkpoint": str(ckpt_path), "train_log": str(train_log), "train_summary_json": str(report_json)},
        "meta": {"created_at": datetime.now(tz=TZ_SHANGHAI).isoformat()},
    }
    report_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="T3: Train an autoregressive (Markov) neighbor scorer on graph paths (teacher forcing).")
    p.add_argument("--paths_graph_npz", type=str, required=True)
    p.add_argument("--road_graph_npz", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--max_steps_per_route", type=int, default=128, help="Cap transitions sampled per route (0=all).")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--edge_tier_emb_dim", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = TrainCfg(
        paths_graph_npz=str(args.paths_graph_npz),
        road_graph_npz=str(args.road_graph_npz),
        out_dir=str(args.out_dir),
        tz_offset_hours=float(args.tz_offset_hours),
        max_steps_per_route=int(args.max_steps_per_route),
        val_frac=float(args.val_frac),
        hidden_dim=int(args.hidden_dim),
        edge_tier_emb_dim=int(args.edge_tier_emb_dim),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    _set_seed(int(cfg.seed))
    report = train(cfg)
    compact = {
        "ok": True,
        "checkpoint": report["outputs"]["checkpoint"],
        "best_val_acc": report["stats"]["best_val_acc"],
        "train_transitions": report["stats"]["train_transitions"],
        "val_transitions": report["stats"]["val_transitions"],
        "train_summary_json": report["outputs"]["train_summary_json"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

