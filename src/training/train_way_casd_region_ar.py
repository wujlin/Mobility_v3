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

from src.data.way_graph.region_sequence_dataset import load_region_ar_dataset, make_region_ar_collate_fn
from src.models.way_casd.region_ar import RegionARCfg, RegionARModel

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

    d_model: int
    n_heads: int
    n_layers: int
    dropout: float
    max_len: int
    coord_scale: float
    n_route_cities: int
    max_routes: Optional[int]
    use_candidate_mask: bool
    split_json: Optional[str] = None


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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _subset_indices_from_route_ids(ds: object, route_ids: np.ndarray) -> np.ndarray:
    """
    RegionRouteDataset is indexed by row (region_seq CSR row), not by route_id.
    This helper maps a list of route_ids -> dataset indices.
    """
    route_ids = np.asarray(route_ids, dtype=np.int64).reshape(-1)
    if route_ids.size == 0:
        return np.zeros((0,), dtype=np.int64)
    # ds.row_ids are CSR rows; region_seq.route_id[row] is the corresponding route_id.
    rid_per_idx = np.asarray(ds.region_seq.route_id[ds.row_ids], dtype=np.int64).reshape(-1)
    mask = np.isin(rid_per_idx, route_ids, assume_unique=False)
    return np.nonzero(mask)[0].astype(np.int64, copy=False)


def _load_region_meta(*, way_regions_npz: Path, way_features_npz: Path, coord_scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """
    Build:
      - region_city: (R,) long
      - region_static: (R,4) float = [centroid_y_norm, centroid_x_norm, log1p(n_ways), log1p(deg)]
      - region_adj: (R,R) bool (diag=True)
    """
    wr = np.load(str(way_regions_npz), allow_pickle=True)
    need = {"way_region", "region_way_ptr", "region_way_idx", "region_adj_ptr", "region_adj_idx"}
    missing = sorted(list(need - set(wr.files)))
    if missing:
        raise SystemExit(f"[FATAL] way_regions_npz missing keys: {missing}")

    way_region = np.asarray(wr["way_region"], dtype=np.int64).reshape(-1)
    region_way_ptr = np.asarray(wr["region_way_ptr"], dtype=np.int64).reshape(-1)
    region_way_idx = np.asarray(wr["region_way_idx"], dtype=np.int64).reshape(-1)
    region_adj_ptr = np.asarray(wr["region_adj_ptr"], dtype=np.int64).reshape(-1)
    region_adj_idx = np.asarray(wr["region_adj_idx"], dtype=np.int64).reshape(-1)

    n_regions = int(region_way_ptr.size) - 1
    meta = None
    if "meta" in wr.files:
        meta_obj = wr["meta"]
        if isinstance(meta_obj, np.ndarray) and meta_obj.size == 1:
            meta_obj = meta_obj.item()
        meta = meta_obj if isinstance(meta_obj, dict) else None
    per_city = (meta or {}).get("per_city", {})
    if not isinstance(per_city, dict) or not per_city:
        # Backward-compatible fallback for single-city region files.
        log.warning("way_regions_npz meta/per_city missing; fallback to single-city mapping (all regions -> city 0).")
        region_city = np.zeros((n_regions,), dtype=np.int64)
        n_cities = 1
    else:
        region_city = np.full((n_regions,), -1, dtype=np.int64)
        n_cities = 0
        for k, v in per_city.items():
            try:
                city = int(k)
                off = int(v.get("region_id_offset", 0))
                nr = int(v.get("n_regions", 0))
            except Exception:
                continue
            if nr <= 0:
                continue
            region_city[off : off + nr] = int(city)
            n_cities = max(n_cities, city + 1)

        if int(np.sum(region_city < 0)) > 0:
            raise SystemExit(f"[FATAL] region_city has unassigned entries: {int(np.sum(region_city < 0))}/{n_regions}")

    wf = np.load(str(way_features_npz), allow_pickle=True)
    need = {"way_center_y", "way_center_x"}
    missing = sorted(list(need - set(wf.files)))
    if missing:
        raise SystemExit(f"[FATAL] way_features_npz missing keys: {missing}")
    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)

    # region static features
    coord_scale = float(coord_scale)
    cent_y = np.zeros((n_regions,), dtype=np.float64)
    cent_x = np.zeros((n_regions,), dtype=np.float64)
    n_ways = np.zeros((n_regions,), dtype=np.float64)
    for r in range(n_regions):
        s = int(region_way_ptr[r])
        e = int(region_way_ptr[r + 1])
        ways = region_way_idx[s:e]
        n = int(ways.size)
        n_ways[r] = float(n)
        if n <= 0:
            continue
        yy = way_center_y[ways]
        xx = way_center_x[ways]
        cent_y[r] = float(np.mean(yy))
        cent_x[r] = float(np.mean(xx))

    deg = (region_adj_ptr[1:] - region_adj_ptr[:-1]).astype(np.int64, copy=False)
    deg_f = deg.astype(np.float64, copy=False)

    static = np.stack(
        [
            cent_y / coord_scale,
            cent_x / coord_scale,
            np.log1p(n_ways),
            np.log1p(deg_f),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    # region adjacency (for diagnostics only)
    adj = np.zeros((n_regions, n_regions), dtype=bool)
    np.fill_diagonal(adj, True)
    for r in range(n_regions):
        s = int(region_adj_ptr[r])
        e = int(region_adj_ptr[r + 1])
        for nb in region_adj_idx[s:e].tolist():
            b = int(nb)
            if 0 <= b < n_regions:
                adj[r, b] = True

    report = {
        "n_regions": int(n_regions),
        "n_cities": int(n_cities),
        "region_city_counts": {str(i): int(np.sum(region_city == i)) for i in range(int(n_cities))},
        "static_dim": int(static.shape[1]),
        "deg": {"p50": float(np.percentile(deg_f, 50)), "p90": float(np.percentile(deg_f, 90)), "max": int(deg.max()) if deg.size else 0},
        "n_ways": {"p50": float(np.percentile(n_ways, 50)), "p90": float(np.percentile(n_ways, 90)), "max": float(np.max(n_ways)) if n_ways.size else 0.0},
        "coord_scale": float(coord_scale),
    }

    return (
        torch.as_tensor(region_city, dtype=torch.long),
        torch.as_tensor(static, dtype=torch.float32),
        torch.as_tensor(adj, dtype=torch.bool),
        report,
    )


def train_epoch(model: RegionARModel, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_invalid = 0.0
    total_tgt_not_neighbor = 0.0
    total_tokens = 0.0

    for batch in loader:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        b["route_cond"] = {k: v.to(device) for k, v in b["route_cond"].items()}

        opt.zero_grad(set_to_none=True)
        loss, stats = model.compute_loss(b)
        loss.backward()
        opt.step()

        n_tok = float(stats["n_tokens"])
        total_loss += float(stats["loss"]) * n_tok
        total_acc += float(stats["acc"]) * n_tok
        if np.isfinite(float(stats["invalid_rate"])):
            total_invalid += float(stats["invalid_rate"]) * n_tok
        if np.isfinite(float(stats.get("tgt_not_neighbor_rate", float("nan")))):
            total_tgt_not_neighbor += float(stats["tgt_not_neighbor_rate"]) * n_tok
        total_tokens += n_tok

    denom = max(1.0, float(total_tokens))
    return {
        "loss": float(total_loss / denom),
        "acc": float(total_acc / denom),
        "invalid_rate": float(total_invalid / denom) if total_tokens > 0 else float("nan"),
        "tgt_not_neighbor_rate": float(total_tgt_not_neighbor / denom) if total_tokens > 0 else float("nan"),
        "n_tokens": float(total_tokens),
    }


@torch.no_grad()
def eval_epoch(model: RegionARModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_invalid = 0.0
    total_tgt_not_neighbor = 0.0
    total_tokens = 0.0

    for batch in loader:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        b["route_cond"] = {k: v.to(device) for k, v in b["route_cond"].items()}
        loss, stats = model.compute_loss(b)

        n_tok = float(stats["n_tokens"])
        total_loss += float(stats["loss"]) * n_tok
        total_acc += float(stats["acc"]) * n_tok
        if np.isfinite(float(stats["invalid_rate"])):
            total_invalid += float(stats["invalid_rate"]) * n_tok
        if np.isfinite(float(stats.get("tgt_not_neighbor_rate", float("nan")))):
            total_tgt_not_neighbor += float(stats["tgt_not_neighbor_rate"]) * n_tok
        total_tokens += n_tok

    denom = max(1.0, float(total_tokens))
    return {
        "loss": float(total_loss / denom),
        "acc": float(total_acc / denom),
        "invalid_rate": float(total_invalid / denom) if total_tokens > 0 else float("nan"),
        "tgt_not_neighbor_rate": float(total_tgt_not_neighbor / denom) if total_tokens > 0 else float("nan"),
        "n_tokens": float(total_tokens),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Region AR (hierarchical coarse layer) for Way-CASD.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_regions_npz", type=Path, required=True)
    p.add_argument("--region_seq_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--n_epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--max_routes", type=int, default=None, help="Debug: cap number of routes (after filtering).")
    p.add_argument(
        "--split_json",
        type=Path,
        default=None,
        help="Optional OD-disjoint split json (expects splits.train/val/test route_ids). Overrides val_ratio.",
    )

    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=16, help="Max region_seq length used (truncate longer).")
    p.add_argument("--coord_scale", type=float, default=1024.0, help="Normalize (y,x) coords by this constant (must match data).")
    p.add_argument("--n_route_cities", type=int, default=2)
    p.add_argument("--use_candidate_mask", action="store_true", help="Score only neighbor regions (inductive bias).")
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
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
        max_len=int(args.max_len),
        coord_scale=float(args.coord_scale),
        n_route_cities=int(args.n_route_cities),
        max_routes=(int(args.max_routes) if args.max_routes is not None else None),
        use_candidate_mask=bool(args.use_candidate_mask),
        split_json=(str(args.split_json) if args.split_json is not None else None),
    )

    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    log.info(f"device={device}")

    region_city, region_static, region_adj, region_meta_report = _load_region_meta(
        way_regions_npz=Path(args.way_regions_npz),
        way_features_npz=Path(args.way_features_npz),
        coord_scale=float(cfg.coord_scale),
    )
    log.info(f"region_meta: {region_meta_report}")

    if args.split_json is not None and args.max_routes is not None:
        log.warning("--split_json is set, ignoring --max_routes to avoid inconsistent splits.")
    ds = load_region_ar_dataset(
        way_routes_npz=Path(args.way_routes_npz),
        region_seq_npz=Path(args.region_seq_npz),
        way_regions_npz=Path(args.way_regions_npz),
        tz_offset_hours=float(cfg.tz_offset_hours),
        max_routes=(None if args.split_json is not None else cfg.max_routes),
    )
    if args.split_json is None:
        train_idx, val_idx = _split_dataset(len(ds), val_ratio=float(cfg.val_ratio), seed=int(cfg.seed))
    else:
        split = _read_json(Path(args.split_json))
        splits = split.get("splits", split)
        tr_rids = np.asarray(splits.get("train", []), dtype=np.int64).reshape(-1)
        va_rids = np.asarray(splits.get("val", []), dtype=np.int64).reshape(-1)
        train_idx = _subset_indices_from_route_ids(ds, tr_rids)
        val_idx = _subset_indices_from_route_ids(ds, va_rids)
        if int(train_idx.size) == 0 or int(val_idx.size) == 0:
            raise SystemExit(
                f"[FATAL] split_json produced empty subsets: train_idx={int(train_idx.size)} val_idx={int(val_idx.size)}. "
                "Check min_hops/max_way_len match split generation and region_seq_npz."
            )
        log.info(
            f"split_json={args.split_json} train_routes={int(tr_rids.size)} val_routes={int(va_rids.size)} "
            f"=> train_idx={int(train_idx.size)} val_idx={int(val_idx.size)}"
        )
    train_ds = Subset(ds, train_idx.tolist())
    val_ds = Subset(ds, val_idx.tolist())

    collate = make_region_ar_collate_fn(max_len=int(cfg.max_len))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )

    model = RegionARModel(
        cfg=RegionARCfg(
            d_model=int(cfg.d_model),
            n_heads=int(cfg.n_heads),
            n_layers=int(cfg.n_layers),
            dropout=float(cfg.dropout),
            max_len=int(cfg.max_len),
            n_regions=int(region_city.numel()),
            n_route_cities=int(cfg.n_route_cities),
            coord_scale=float(cfg.coord_scale),
            use_candidate_mask=bool(cfg.use_candidate_mask),
        ),
        region_city=region_city,
        region_static=region_static,
        region_adj=region_adj,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    best_by_loss = {"val_loss": float("inf"), "epoch": -1, "val_acc": 0.0}
    best_by_acc = {"val_acc": float("-inf"), "epoch": -1, "val_loss": float("inf")}
    for epoch in range(int(cfg.n_epochs)):
        tr = train_epoch(model, train_loader, opt, device)
        va = eval_epoch(model, val_loader, device)
        log.info(f"[epoch {epoch:03d}] train={tr} val={va}")

        improved_loss = float(va["loss"]) < float(best_by_loss["val_loss"])
        if improved_loss:
            best_by_loss = {"val_loss": float(va["loss"]), "epoch": int(epoch), "val_acc": float(va["acc"])}
            ckpt = {
                "model": model.state_dict(),
                "cfg": asdict(cfg),
                "region_meta": region_meta_report,
                "best_by_loss": best_by_loss,
                "best_by_acc": best_by_acc,
            }
            torch.save(ckpt, out_dir / "ckpt_best.pt")

        improved_acc = float(va["acc"]) > float(best_by_acc["val_acc"])
        if improved_acc:
            best_by_acc = {"val_acc": float(va["acc"]), "epoch": int(epoch), "val_loss": float(va["loss"])}
            ckpt = {
                "model": model.state_dict(),
                "cfg": asdict(cfg),
                "region_meta": region_meta_report,
                "best_by_loss": best_by_loss,
                "best_by_acc": best_by_acc,
            }
            torch.save(ckpt, out_dir / "ckpt_best_acc.pt")

        if (epoch + 1) % 10 == 0:
            ckpt = {
                "model": model.state_dict(),
                "cfg": asdict(cfg),
                "region_meta": region_meta_report,
                "best_by_loss": best_by_loss,
                "best_by_acc": best_by_acc,
                "epoch": int(epoch),
            }
            torch.save(ckpt, out_dir / "ckpt_last.pt")

    report = {
        "ok": True,
        "task": "train_way_casd_region_ar",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_regions_npz": str(args.way_regions_npz),
            "region_seq_npz": str(args.region_seq_npz),
            "way_features_npz": str(args.way_features_npz),
            "split_json": (str(args.split_json) if args.split_json is not None else None),
        },
        "region_meta": region_meta_report,
        "best_by_loss": best_by_loss,
        "best_by_acc": best_by_acc,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info(f"saved: {out_dir/'report.json'}")
    log.info(f"ckpt_best: {out_dir/'ckpt_best.pt'}")
    log.info(f"ckpt_best_acc: {out_dir/'ckpt_best_acc.pt'}")


if __name__ == "__main__":
    main()
