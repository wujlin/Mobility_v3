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
from torch.utils.data import DataLoader, Dataset, Subset

from src.models.way_casd.gps_diffusion import GPSDiffusionCfg, GPSDiffusionExecutionModel, SkeletonCrossAttnCfg
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import make_way_feature_tensors

try:
    import pyarrow.parquet as pq  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pq = None


TZ_SHANGHAI = timezone(timedelta(hours=8))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainCfg:
    seed: int
    device: str
    batch_size: int
    num_workers: int
    n_epochs: int
    lr: float
    weight_decay: float
    val_ratio: float

    # data
    tz_offset_hours: float
    traj_len: int
    max_way_len: int
    min_way_len: int
    prefer_matched: bool
    limit_rows: int

    # model
    d_model: int
    n_latent: int
    hidden_dim: int
    emb_dim: int
    diffusion_steps: int
    prediction_type: str
    skel_attn_heads: int
    skel_noise_sigma: float
    coord_scale: float


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_ckpt_state_and_cfg(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
        return state, cfg
    if isinstance(ckpt, dict):
        return ckpt, {}
    raise TypeError(f"Unexpected checkpoint format: {type(ckpt)}")


def _infer_decoder_use_dest_dist_from_state(state: Dict[str, torch.Tensor]) -> bool:
    w = state.get("decoder.scorer.0.weight", None)
    if w is None:
        return True
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        return True
    hidden = int(w.shape[0])
    in_dim = int(w.shape[1])
    delta = int(in_dim - hidden * 3)
    if delta == 0:
        return False
    if delta == 1:
        return True
    return True


def _infer_decoder_use_cross_attn_from_state(state: Dict[str, torch.Tensor]) -> bool:
    # New decoder has keys like "decoder.cross_attn.in_proj_weight".
    for k in state.keys():
        if str(k).startswith("decoder.cross_attn."):
            return True
    return False


def _split_dataset(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n))
    n_val = int(round(float(val_ratio) * float(n)))
    n_val = max(1, min(n_val, n - 1))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx.astype(np.int64, copy=False), val_idx.astype(np.int64, copy=False)


def _hour_from_unix(start_t: np.ndarray, tz_offset_hours: float) -> np.ndarray:
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    sec = ((start_t + tz_sec) % 86400).astype(np.int64, copy=False)
    return (sec // 3600).astype(np.int64, copy=False)


def _dow_from_unix(start_t: np.ndarray, tz_offset_hours: float) -> np.ndarray:
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    tz_sec = int(round(float(tz_offset_hours) * 3600.0))
    days = ((start_t + tz_sec) // 86400).astype(np.int64, copy=False)
    return ((days + 3) % 7).astype(np.int64, copy=False)


def _dedup_consecutive(seq: List[int]) -> List[int]:
    out: List[int] = []
    last: Optional[int] = None
    for x in seq:
        xi = int(x)
        if last is None or xi != int(last):
            out.append(xi)
            last = xi
    return out


def _resample_polyline(yx: np.ndarray, *, n: int) -> np.ndarray:
    """
    Resample a polyline to fixed length n using arc-length parameterization.
    yx: (P,2) float32
    returns: (n,2) float32
    """
    yx = np.asarray(yx, dtype=np.float32)
    if yx.ndim != 2 or yx.shape[1] != 2:
        raise ValueError(f"yx must be (P,2), got {yx.shape}")
    P = int(yx.shape[0])
    if P <= 0:
        return np.zeros((int(n), 2), dtype=np.float32)
    if P == 1 or int(n) <= 1:
        return np.repeat(yx[:1], repeats=int(n), axis=0).astype(np.float32, copy=False)

    d = np.sqrt(np.sum((yx[1:] - yx[:-1]) ** 2, axis=1)).astype(np.float32, copy=False)  # (P-1,)
    s = np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(d, dtype=np.float32)], axis=0)  # (P,)
    total = float(s[-1])
    if not np.isfinite(total) or total <= 1e-6:
        # Fall back to index-based interpolation.
        t = np.linspace(0.0, float(P - 1), int(n), dtype=np.float32)
        xp = np.arange(P, dtype=np.float32)
        y = np.interp(t, xp, yx[:, 0]).astype(np.float32, copy=False)
        x = np.interp(t, xp, yx[:, 1]).astype(np.float32, copy=False)
        return np.stack([y, x], axis=1)

    u = np.linspace(0.0, total, int(n), dtype=np.float32)
    y = np.interp(u, s, yx[:, 0]).astype(np.float32, copy=False)
    x = np.interp(u, s, yx[:, 1]).astype(np.float32, copy=False)
    return np.stack([y, x], axis=1)


class WorldTraceExecutionDataset(Dataset):
    def __init__(
        self,
        *,
        segments_parquet: List[Path],
        route_city: List[int],
        way_osm_id_vocab: np.ndarray,
        cfg: TrainCfg,
    ) -> None:
        if pq is None:
            raise ModuleNotFoundError("pyarrow is required (pip/conda install pyarrow).")
        if len(segments_parquet) != len(route_city):
            raise ValueError("--segments_parquet and --route_city must have the same length")
        self.cfg = cfg

        way_osm_id_vocab = np.asarray(way_osm_id_vocab, dtype=np.int64).reshape(-1)
        self.way_to_idx = {int(w): int(i) for i, w in enumerate(way_osm_id_vocab.tolist())}

        samples: List[Dict[str, object]] = []

        for sp, city in zip(segments_parquet, route_city):
            cols = ["osm_way_id", "t", "y", "x"]
            if cfg.prefer_matched:
                cols.append("is_matched")
            table = pq.read_table(str(sp), columns=cols)
            way_col = table.column("osm_way_id").to_pylist()
            t_col = table.column("t").to_pylist()
            y_col = table.column("y").to_pylist()
            x_col = table.column("x").to_pylist()
            m_col = table.column("is_matched").to_pylist() if (cfg.prefer_matched and "is_matched" in table.column_names) else None

            n_rows = int(len(way_col))
            if int(cfg.limit_rows) > 0:
                n_rows = min(n_rows, int(cfg.limit_rows))

            for i in range(n_rows):
                ways0 = way_col[i] or []
                ys0 = y_col[i] or []
                xs0 = x_col[i] or []
                ts0 = t_col[i] or []
                if not (ways0 and ys0 and xs0 and ts0):
                    continue

                if m_col is not None:
                    mm = m_col[i] or []
                    if len(mm) == len(ys0):
                        keep = [int(v) != 0 for v in mm]
                        ys = [int(y) for y, k in zip(ys0, keep) if k]
                        xs = [int(x) for x, k in zip(xs0, keep) if k]
                        ts = [int(t) for t, k in zip(ts0, keep) if k]
                        ways = [int(w) for w, k in zip(ways0, keep) if k]
                    else:
                        ys = [int(y) for y in ys0]
                        xs = [int(x) for x in xs0]
                        ts = [int(t) for t in ts0]
                        ways = [int(w) for w in ways0]
                else:
                    ys = [int(y) for y in ys0]
                    xs = [int(x) for x in xs0]
                    ts = [int(t) for t in ts0]
                    ways = [int(w) for w in ways0]

                ways = [int(w) for w in ways if int(w) > 0]
                if not ways:
                    continue
                ways = _dedup_consecutive(ways)
                enc = [self.way_to_idx[int(w)] for w in ways if int(w) in self.way_to_idx]
                if len(enc) < int(cfg.min_way_len):
                    continue
                if len(enc) > int(cfg.max_way_len):
                    enc = enc[: int(cfg.max_way_len)]

                yx = np.stack([np.asarray(ys, dtype=np.float32), np.asarray(xs, dtype=np.float32)], axis=1)
                traj = _resample_polyline(yx, n=int(cfg.traj_len))  # (T,2) absolute
                start_pos = traj[0].astype(np.float32, copy=False)
                dest_pos = traj[-1].astype(np.float32, copy=False)
                traj_rel = traj - start_pos[None, :]
                if float(cfg.coord_scale) > 0:
                    traj_rel = traj_rel / float(cfg.coord_scale)

                start_t = int(ts[0])
                hour = int(_hour_from_unix(np.asarray([start_t], dtype=np.int64), float(cfg.tz_offset_hours))[0])
                dow = int(_dow_from_unix(np.asarray([start_t], dtype=np.int64), float(cfg.tz_offset_hours))[0])

                samples.append(
                    {
                        "way_seq": np.asarray(enc, dtype=np.int64),
                        "way_len": np.asarray(int(len(enc)), dtype=np.int64),
                        "traj_rel": traj_rel.astype(np.float32, copy=False),
                        "start_pos": start_pos.astype(np.float32, copy=False),
                        "dest_pos": dest_pos.astype(np.float32, copy=False),
                        "hour": np.asarray(hour, dtype=np.int64),
                        "dow": np.asarray(dow, dtype=np.int64),
                        "route_city": np.asarray(int(city), dtype=np.int64),
                    }
                )

        self.samples = samples

    def __len__(self) -> int:
        return int(len(self.samples))

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        s = self.samples[int(idx)]
        return {
            "way_seq": s["way_seq"],
            "way_len": s["way_len"],
            "traj_rel": s["traj_rel"],
            "start_pos": s["start_pos"],
            "dest_pos": s["dest_pos"],
            "hour": s["hour"],
            "dow": s["dow"],
            "route_city": s["route_city"],
        }


def _collate(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    lens = np.asarray([int(b["way_len"]) for b in batch], dtype=np.int64)
    Lmax = int(lens.max()) if B > 0 else 1
    way_pad = np.full((B, Lmax), -1, dtype=np.int64)
    for i, b in enumerate(batch):
        L = int(b["way_len"])
        way_pad[i, :L] = np.asarray(b["way_seq"], dtype=np.int64)[:L]

    traj = np.stack([np.asarray(b["traj_rel"], dtype=np.float32) for b in batch], axis=0)  # (B,T,2)
    route_cond = {
        "start_pos": torch.as_tensor(np.stack([b["start_pos"] for b in batch], axis=0), dtype=torch.float32),
        "dest_pos": torch.as_tensor(np.stack([b["dest_pos"] for b in batch], axis=0), dtype=torch.float32),
        "hour": torch.as_tensor(np.asarray([int(b["hour"]) for b in batch], dtype=np.int64), dtype=torch.long),
        "dow": torch.as_tensor(np.asarray([int(b["dow"]) for b in batch], dtype=np.int64), dtype=torch.long),
        "route_city": torch.as_tensor(np.asarray([int(b["route_city"]) for b in batch], dtype=np.int64), dtype=torch.long),
    }
    return {
        "way_seq_pad": torch.as_tensor(way_pad, dtype=torch.long),
        "way_seq_len": torch.as_tensor(lens, dtype=torch.long),
        "traj_rel": torch.as_tensor(traj, dtype=torch.float32),
        "route_cond": route_cond,
    }


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        "way_seq_pad": batch["way_seq_pad"].to(device),
        "traj_rel": batch["traj_rel"].to(device),
        "route_cond": {k: v.to(device) for k, v in batch["route_cond"].items()},
    }


def train_epoch(
    *,
    ae: WayCASDAutoEncoder,
    model: GPSDiffusionExecutionModel,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        b = _to_device(batch, device)
        with torch.no_grad():
            z, _ = ae.encode(b["way_seq_pad"])
        opt.zero_grad(set_to_none=True)
        loss = model.compute_loss(traj_yx_rel=b["traj_rel"], route_cond=b["route_cond"], skeleton_latent=z)
        loss.backward()
        opt.step()
        total_loss += float(loss.item())
        total_batches += 1

    denom = max(1, int(total_batches))
    return {"loss": float(total_loss / denom)}


@torch.no_grad()
def eval_epoch(*, ae: WayCASDAutoEncoder, model: GPSDiffusionExecutionModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        b = _to_device(batch, device)
        z, _ = ae.encode(b["way_seq_pad"])
        loss = model.compute_loss(traj_yx_rel=b["traj_rel"], route_cond=b["route_cond"], skeleton_latent=z)
        total_loss += float(loss.item())
        total_batches += 1
    denom = max(1, int(total_batches))
    return {"loss": float(total_loss / denom)}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Way-CASD Execution Stage: GPS-level conditional diffusion.")
    p.add_argument("--segments_parquet", type=Path, nargs="+", required=True, help="One or more segments_with_wayid.parquet")
    p.add_argument("--route_city", type=int, nargs="+", required=True, help="route_city id per segments_parquet")
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True, help="Decision AE checkpoint (provides skeleton_latent).")
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)

    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--traj_len", type=int, default=256)
    p.add_argument("--max_way_len", type=int, default=128)
    p.add_argument("--min_way_len", type=int, default=2)
    p.add_argument("--prefer_matched", action="store_true", help="Use matched points only when is_matched exists.")
    p.add_argument("--limit_rows", type=int, default=0, help="Debug: cap rows per parquet (0=no limit).")

    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_latent", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--emb_dim", type=int, default=512)
    p.add_argument("--diffusion_steps", type=int, default=100)
    p.add_argument("--prediction_type", choices=["eps", "v"], default="eps")
    p.add_argument("--skel_attn_heads", type=int, default=4)
    p.add_argument("--skel_noise_sigma", type=float, default=0.1)
    p.add_argument("--coord_scale", type=float, default=1024.0)

    # Long-run training ergonomics
    p.add_argument("--resume_ckpt", type=Path, default=None, help="Optional: resume from a previous execution ckpt (.pt).")
    p.add_argument("--resume_epoch", type=int, default=None, help="Optional: override resume epoch (when ckpt has no epoch).")
    p.add_argument("--save_every", type=int, default=20, help="Save ckpt_last.pt every N epochs (best ckpt still saved on improve).")
    p.add_argument("--early_stop_patience", type=int, default=0, help="Optional early stop (0=disable).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainCfg(
        seed=int(args.seed),
        device=str(args.device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        n_epochs=int(args.n_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_ratio=float(args.val_ratio),
        tz_offset_hours=float(args.tz_offset_hours),
        traj_len=int(args.traj_len),
        max_way_len=int(args.max_way_len),
        min_way_len=int(args.min_way_len),
        prefer_matched=bool(args.prefer_matched),
        limit_rows=int(args.limit_rows),
        d_model=int(args.d_model),
        n_latent=int(args.n_latent),
        hidden_dim=int(args.hidden_dim),
        emb_dim=int(args.emb_dim),
        diffusion_steps=int(args.diffusion_steps),
        prediction_type=str(args.prediction_type),
        skel_attn_heads=int(args.skel_attn_heads),
        skel_noise_sigma=float(args.skel_noise_sigma),
        coord_scale=float(args.coord_scale),
    )

    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    log.info(f"device={device}")

    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
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

    # Load AE (frozen) to provide skeleton_latent.
    ae_state, ae_cfg_dict = _load_ckpt_state_and_cfg(Path(args.ae_ckpt))
    use_dest_dist = _infer_decoder_use_dest_dist_from_state(ae_state)
    use_cross_attn = _infer_decoder_use_cross_attn_from_state(ae_state)
    ae_cfg = WayCASDAECfg(
        d_model=int(ae_cfg_dict.get("d_model", cfg.d_model)),
        n_latent=int(ae_cfg_dict.get("n_latent", cfg.n_latent)),
        n_heads=int(ae_cfg_dict.get("n_heads", 8)),
        dropout=float(ae_cfg_dict.get("dropout", 0.1)),
        max_candidates=int(ae_cfg_dict.get("max_candidates", 32)),
        max_len=int(ae_cfg_dict.get("max_len", cfg.max_way_len)),
        coord_scale=float(ae_cfg_dict.get("coord_scale", cfg.coord_scale)),
        decoder_use_dest_dist=bool(use_dest_dist),
        decoder_use_cross_attn=bool(use_cross_attn),
    )
    ae = WayCASDAutoEncoder(
        cfg=ae_cfg,
        way_features=way_features,
        way_adj_ptr=wg["way_adj_ptr"],
        way_adj_idx=wg["way_adj_idx"],
        n_highway_types=int(max(4, n_highway_types)),
    ).to(device)
    ae.load_state_dict(ae_state, strict=True)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    # Align training cfg with AE (avoid accidental mismatch).
    cfg = TrainCfg(
        **{
            **asdict(cfg),
            "d_model": int(ae_cfg.d_model),
            "n_latent": int(ae_cfg.n_latent),
            "coord_scale": float(ae_cfg.coord_scale),
        }
    )

    # Build dataset from segments parquet (osm_way_id) using the unified way_osm_id vocab.
    dataset = WorldTraceExecutionDataset(
        segments_parquet=[Path(p) for p in args.segments_parquet],
        route_city=[int(x) for x in args.route_city],
        way_osm_id_vocab=np.asarray(wf["way_osm_id"], dtype=np.int64),
        cfg=cfg,
    )
    if len(dataset) < 10:
        raise SystemExit(f"Too few routes for execution training: N={len(dataset)}. Check inputs/filters.")

    train_ids, val_ids = _split_dataset(len(dataset), cfg.val_ratio, cfg.seed)
    train_set = Subset(dataset, train_ids.tolist())
    val_set = Subset(dataset, val_ids.tolist())
    log.info(f"routes: total={len(dataset)} train={len(train_set)} val={len(val_set)} traj_len={cfg.traj_len} max_way_len={cfg.max_way_len}")

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
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        collate_fn=_collate,
    )

    model = GPSDiffusionExecutionModel(
        cfg=GPSDiffusionCfg(
            traj_len=int(cfg.traj_len),
            act_dim=2,
            hidden_dim=int(cfg.hidden_dim),
            emb_dim=int(cfg.emb_dim),
            diffusion_steps=int(cfg.diffusion_steps),
            prediction_type=str(cfg.prediction_type),
            d_model=int(cfg.d_model),
            n_route_cities=4,
            coord_scale=float(cfg.coord_scale),
            skel_attn=SkeletonCrossAttnCfg(
                d_skel=int(cfg.d_model),
                act_dim=2,
                model_dim=int(cfg.hidden_dim),
                num_heads=int(cfg.skel_attn_heads),
                diff_steps=int(cfg.diffusion_steps),
                weight=1.0,
            ),
            skel_noise_sigma=float(cfg.skel_noise_sigma),
        )
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
            if "history" in ckpt and isinstance(ckpt["history"], list):
                history = ckpt["history"]
            if "epoch" in ckpt:
                try:
                    start_epoch = int(ckpt["epoch"]) + 1
                except Exception:
                    start_epoch = 1
        elif isinstance(ckpt, dict):
            # weights-only checkpoint
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            if missing or unexpected:
                log.warning(f"resume: model state mismatch: missing={len(missing)} unexpected={len(unexpected)}")

        if args.resume_epoch is not None:
            start_epoch = int(args.resume_epoch) + 1

        log.info(
            f"resume_ckpt={args.resume_ckpt} start_epoch={start_epoch} best_val_loss={best} best_epoch={best_epoch} history_len={len(history)}"
        )

    # If best is unknown (e.g., weights-only resume), initialize from current val.
    if not np.isfinite(best) or best == float("inf"):
        va0 = eval_epoch(ae=ae, model=model, loader=val_loader, device=device)
        best = float(va0["loss"])
        best_epoch = int(start_epoch - 1)
        log.info(f"init best_val_loss={best:.6f} from current weights (epoch={best_epoch})")

    save_every = max(1, int(args.save_every))
    early_stop_patience = max(0, int(args.early_stop_patience))

    for epoch in range(int(start_epoch), int(cfg.n_epochs) + 1):
        tr = train_epoch(ae=ae, model=model, loader=train_loader, opt=opt, device=device)
        va = eval_epoch(ae=ae, model=model, loader=val_loader, device=device)
        history.append({"epoch": int(epoch), "train": tr, "val": va})
        log.info(f"epoch={epoch} train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f}")

        # Append lightweight history for long runs.
        with hist_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": int(epoch), "train_loss": float(tr["loss"]), "val_loss": float(va["loss"])}) + "\n")

        # Best checkpoint.
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

        # Periodic last checkpoint for resuming.
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

        # Progress snapshot (small JSON, cheap to update).
        progress = {
            "ok": True,
            "task": "train_way_casd_gps_diffusion",
            "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
            "epoch": int(epoch),
            "train_loss": float(tr["loss"]),
            "val_loss": float(va["loss"]),
            "best_val_loss": float(best),
            "best_epoch": int(best_epoch),
            "save_every": int(save_every),
            "early_stop_patience": int(early_stop_patience),
        }
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

        if early_stop_patience > 0 and patience >= early_stop_patience:
            log.info(f"early_stop: patience={patience} reached (best_epoch={best_epoch} best_val_loss={best:.6f})")
            break

    # Always save a final last checkpoint.
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
        "task": "train_way_casd_gps_diffusion",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "segments_parquet": [str(p) for p in args.segments_parquet],
            "route_city": [int(x) for x in args.route_city],
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
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
