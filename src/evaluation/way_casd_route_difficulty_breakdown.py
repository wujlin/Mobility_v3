from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.way_graph.way_sequence_dataset import WayRouteDataset, load_way_routes_npz, make_way_casd_collate_fn
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import make_way_feature_tensors

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(x, *args, **kwargs):  # type: ignore[no-redef]
        return x


TZ_SHANGHAI = timezone(timedelta(hours=8))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cfg:
    seed: int
    device: str
    batch_size: int
    num_workers: int
    n_routes_per_city: int
    tz_offset_hours: float
    val_ratio: float
    max_way_len: int
    max_candidates: int
    top_k_hard: int


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
    route_id = batch["route_id"].to(device)
    way_seq_pad = batch["way_seq_pad"].to(device)
    route_cond = {k: v.to(device) for k, v in batch["route_cond"].items()}
    trans = {k: v.to(device) for k, v in batch["trans"].items()}
    return {"route_id": route_id, "way_seq_pad": way_seq_pad, "route_cond": route_cond, "trans": trans}


def _quantiles(x: np.ndarray, qs: List[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{int(q*100):02d}": float("nan") for q in qs}
    out = {}
    for q in qs:
        out[f"p{int(q*100):02d}"] = float(np.quantile(x, float(q)))
    return out


def _summary(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), **_quantiles(x, [0.1, 0.5, 0.9, 0.99])}
    return {
        "mean": float(np.mean(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        **_quantiles(x, [0.1, 0.5, 0.9, 0.99]),
    }


def _route_graph_features(ptr: np.ndarray, idx: np.ndarray, seq: np.ndarray) -> Dict[str, float]:
    ptr = np.asarray(ptr, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    seq = np.asarray(seq, dtype=np.int64).reshape(-1)
    L = int(seq.size)
    if L <= 1:
        return {"len": float(L), "n_steps": float(max(0, L - 1)), "branch_points": 0.0, "avg_out_deg": 0.0, "max_out_deg": 0.0}
    prev = seq[:-1]
    out_deg = (ptr[prev + 1] - ptr[prev]).astype(np.int64, copy=False)
    return {
        "len": float(L),
        "n_steps": float(L - 1),
        "branch_points": float(int(np.sum(out_deg >= 2))),
        "avg_out_deg": float(np.mean(out_deg)) if out_deg.size else 0.0,
        "max_out_deg": float(np.max(out_deg)) if out_deg.size else 0.0,
    }


@torch.no_grad()
def _eval_subset(
    *,
    ae: WayCASDAutoEncoder,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, Dict[str, float]]]:
    """
    Returns:
      acc_counts[rid] = (correct, total)
      feat[rid] = {len, branch_points, avg_out_deg, max_out_deg}
    """
    acc_counts: Dict[int, Tuple[int, int]] = {}
    feat: Dict[int, Dict[str, float]] = {}

    # Graph pointers for route feature computation (CPU np arrays)
    ptr = np.asarray(ae.decoder.way_adj_ptr, dtype=np.int64)
    idx = np.asarray(ae.decoder.way_adj_idx, dtype=np.int64)

    for batch in tqdm(loader, desc="eval", dynamic_ncols=True):
        b = _to_device(batch, device)
        route_id = b["route_id"].detach().cpu().numpy().astype(np.int64, copy=False)
        way_seq_pad = b["way_seq_pad"].detach().cpu().numpy().astype(np.int64, copy=False)
        way_seq_len = batch["way_seq_len"].detach().cpu().numpy().astype(np.int64, copy=False)

        # Route-level features (once per route)
        for rid, L, row in zip(route_id.tolist(), way_seq_len.tolist(), way_seq_pad, strict=False):
            rr = int(rid)
            if rr not in feat:
                seq = row[: int(L)]
                feat[rr] = _route_graph_features(ptr, idx, seq)

        trans = b["trans"]
        if int(trans["target_idx"].numel()) == 0:
            continue

        z, _mask = ae.encode(b["way_seq_pad"])
        logits = ae.decoder.score_candidates(way_embedder=ae.way_enc, latent_tokens=z, route_cond=b["route_cond"], trans=trans)
        pred = torch.argmax(logits, dim=-1)
        tgt = trans["target_idx"].to(dtype=torch.long)
        ok = (pred == tgt).to(dtype=torch.int64)

        # Map each transition to global route_id via route_idx
        rix = trans["route_idx"].detach().cpu().numpy().astype(np.int64, copy=False)
        ok_np = ok.detach().cpu().numpy().astype(np.int64, copy=False)
        for bi, is_ok in zip(rix.tolist(), ok_np.tolist(), strict=False):
            rr = int(route_id[int(bi)])
            c, t = acc_counts.get(rr, (0, 0))
            acc_counts[rr] = (int(c) + int(is_ok), int(t) + 1)

    return acc_counts, feat


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Per-route teacher-forcing accuracy distribution (difficulty) for Way-CASD AE.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--n_routes_per_city", type=int, default=200)
    p.add_argument("--top_k_hard", type=int, default=20)

    p.add_argument("--tz_offset_hours", type=float, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--max_way_len", type=int, default=None)
    p.add_argument("--max_candidates", type=int, default=None)
    p.add_argument("--splits", type=str, default="train,val", help="Comma-separated: train,val")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    _set_seed(int(args.seed))
    log.info(f"device={device}")

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    wf = np.load(str(args.way_features_npz), allow_pickle=True)

    ckpt = torch.load(str(args.ae_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ck_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    cfg = Cfg(
        seed=int(args.seed),
        device=str(args.device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        n_routes_per_city=int(args.n_routes_per_city),
        tz_offset_hours=float(args.tz_offset_hours) if args.tz_offset_hours is not None else float(ck_cfg.get("tz_offset_hours", -5.0)),
        val_ratio=float(args.val_ratio) if args.val_ratio is not None else float(ck_cfg.get("val_ratio", 0.1)),
        max_way_len=int(args.max_way_len) if args.max_way_len is not None else int(ck_cfg.get("max_way_len", 128)),
        max_candidates=int(args.max_candidates) if args.max_candidates is not None else int(ck_cfg.get("max_candidates", 32)),
        top_k_hard=int(args.top_k_hard),
    )

    way_features = make_way_feature_tensors(
        way_center_y=wf["way_center_y"],
        way_center_x=wf["way_center_x"],
        way_dir_y=wf["way_dir_y"],
        way_dir_x=wf["way_dir_x"],
        way_len_m=wf["way_len_m"],
        way_tier=wf["way_tier"],
        way_highway_code=wf["way_highway_code"],
        device=device,
    )
    n_highway_types = int(np.max(np.asarray(wf["way_highway_code"], dtype=np.int64))) + 1

    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(ck_cfg.get("d_model", 256)),
            n_latent=int(ck_cfg.get("n_latent", 64)),
            n_heads=int(ck_cfg.get("n_heads", 8)),
            dropout=float(ck_cfg.get("dropout", 0.1)),
            max_candidates=int(ck_cfg.get("max_candidates", cfg.max_candidates)),
            max_len=int(ck_cfg.get("max_len", cfg.max_way_len)),
            coord_scale=float(ck_cfg.get("coord_scale", 1024.0)),
            decoder_use_dest_dist=bool(ck_cfg.get("decoder_use_dest_dist", True)),
            decoder_use_cross_attn=bool(ck_cfg.get("decoder_use_cross_attn", True)),
            decoder_n_cross_heads=int(ck_cfg.get("decoder_n_cross_heads", 4)),
            decoder_use_step_emb=bool(ck_cfg.get("decoder_use_step_emb", False)),
            decoder_use_dest_query=bool(ck_cfg.get("decoder_use_dest_query", False)),
        ),
        way_features=way_features,
        way_adj_ptr=np.asarray(wg["way_adj_ptr"], dtype=np.int64),
        way_adj_idx=np.asarray(wg["way_adj_idx"], dtype=np.int64),
        n_highway_types=int(max(4, n_highway_types)),
    ).to(device)
    ae.load_state_dict(state, strict=False)
    ae.eval()

    max_routes = ck_cfg.get("max_routes", None)
    dataset = WayRouteDataset(routes, max_routes=(int(max_routes) if max_routes is not None else None), max_way_len=int(cfg.max_way_len))
    train_idx, val_idx = _split_dataset(len(dataset), float(cfg.val_ratio), int(cfg.seed))

    route_ids = dataset.route_ids.astype(np.int64, copy=False)
    city_by_ds = routes.route_city[route_ids].astype(np.int64, copy=False)
    cities = sorted(set(int(x) for x in np.unique(city_by_ds).tolist()))

    collate_fn = make_way_casd_collate_fn(
        way_adj_ptr=wg["way_adj_ptr"],
        way_adj_idx=wg["way_adj_idx"],
        max_candidates=int(cfg.max_candidates),
        tz_offset_hours=float(cfg.tz_offset_hours),
    )

    pin = bool(device.type == "cuda")
    num_workers = max(0, int(cfg.num_workers))
    prefetch_factor = 4 if num_workers > 0 else None

    want_splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    split_map = {"train": train_idx, "val": val_idx}

    out: Dict[str, object] = {
        "ok": True,
        "task": "way_casd_route_difficulty_breakdown",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "way_routes_npz": str(Path(args.way_routes_npz)),
            "way_graph_npz": str(Path(args.way_graph_npz)),
            "way_features_npz": str(Path(args.way_features_npz)),
            "ae_ckpt": str(Path(args.ae_ckpt)),
        },
        "config": asdict(cfg),
        "ckpt_meta": {
            "epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
            "best_epoch": int(ckpt.get("best_epoch", -1)) if isinstance(ckpt, dict) else -1,
            "best_val_loss": float(ckpt.get("best_val_loss", float("nan"))) if isinstance(ckpt, dict) else float("nan"),
        },
        "splits": {},
    }

    for split_name in want_splits:
        if split_name not in split_map:
            raise ValueError(f"unknown split: {split_name} (supported: train,val)")
        base_idx = split_map[split_name]
        rng = np.random.default_rng(int(cfg.seed) + (0 if split_name == "train" else 99991))

        split_res: Dict[str, object] = {"by_city": {}, "overall": {}, "cities": cities}

        all_route_rows: List[Dict[str, float]] = []
        for city in cities:
            mask = city_by_ds[base_idx] == int(city)
            idx_city = base_idx[mask]
            if idx_city.size == 0:
                split_res["by_city"][str(int(city))] = {"n_routes": 0, "route_acc": {}, "hard_routes": []}
                continue
            idx_city = idx_city.copy()
            rng.shuffle(idx_city)
            pick_n = min(int(cfg.n_routes_per_city), int(idx_city.size))
            pick = idx_city[:pick_n]

            subset = Subset(dataset, pick.tolist())
            loader = DataLoader(
                subset,
                batch_size=int(cfg.batch_size),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin,
                persistent_workers=(num_workers > 0),
                prefetch_factor=prefetch_factor,
                collate_fn=collate_fn,
            )
            acc_counts, feat = _eval_subset(ae=ae, loader=loader, device=device)

            rows: List[Dict[str, float]] = []
            for rid, (c, t) in acc_counts.items():
                f = feat.get(int(rid), {})
                rows.append(
                    {
                        "route_id": float(int(rid)),
                        "route_acc": float(c / t) if t > 0 else float("nan"),
                        "n_trans": float(int(t)),
                        "len": float(f.get("len", float("nan"))),
                        "branch_points": float(f.get("branch_points", float("nan"))),
                        "avg_out_deg": float(f.get("avg_out_deg", float("nan"))),
                        "max_out_deg": float(f.get("max_out_deg", float("nan"))),
                        "city": float(int(city)),
                    }
                )
            all_route_rows.extend(rows)

            acc_arr = np.asarray([r["route_acc"] for r in rows], dtype=np.float64)
            split_res["by_city"][str(int(city))] = {
                "n_routes": int(pick_n),
                "n_routes_scored": int(acc_arr.size),
                "route_acc": _summary(acc_arr),
                "hard_routes": sorted(rows, key=lambda x: float(x["route_acc"]))[: int(cfg.top_k_hard)],
            }

        # overall across sampled cities
        all_acc = np.asarray([r["route_acc"] for r in all_route_rows], dtype=np.float64)
        split_res["overall"] = {
            "n_routes": int(sum(int(split_res["by_city"][str(int(c))]["n_routes"]) for c in cities)),
            "n_routes_scored": int(all_acc.size),
            "route_acc": _summary(all_acc),
            "hard_routes": sorted(all_route_rows, key=lambda x: float(x["route_acc"]))[: int(cfg.top_k_hard)],
        }

        out["splits"][str(split_name)] = split_res

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()

