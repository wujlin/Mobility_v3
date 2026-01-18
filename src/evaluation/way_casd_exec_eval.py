from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.evaluation.distribution_metrics import compute_distribution_metrics, compute_jsd_from_samples, jsd_from_hist
from src.evaluation.micro_metrics import compute_frechet_per_sample
from src.models.way_casd.gps_diffusion import GPSDiffusionCfg, GPSDiffusionExecutionModel
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import make_way_feature_tensors

try:
    import pyarrow.parquet as pq  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pq = None


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class EvalCfg:
    seed: int
    device: str
    tz_offset_hours: float
    traj_len: int
    n_routes: int
    n_samples_per_route: int
    max_way_len: int
    min_way_len: int
    prefer_matched: bool
    fix_ends: bool
    coord_scale: float

    batch_routes: int
    frechet_points: int

    onroad_prob_thr: float
    spatial_stride: int
    speed_bins: int
    accel_bins: int
    turn_bins: int
    turn_min_speed: float


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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
    yx = np.asarray(yx, dtype=np.float32)
    P = int(yx.shape[0])
    if P <= 0:
        return np.zeros((int(n), 2), dtype=np.float32)
    if P == 1 or int(n) <= 1:
        return np.repeat(yx[:1], repeats=int(n), axis=0).astype(np.float32, copy=False)

    d = np.sqrt(np.sum((yx[1:] - yx[:-1]) ** 2, axis=1)).astype(np.float32, copy=False)
    s = np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(d, dtype=np.float32)], axis=0)
    total = float(s[-1])
    if not np.isfinite(total) or total <= 1e-6:
        t = np.linspace(0.0, float(P - 1), int(n), dtype=np.float32)
        xp = np.arange(P, dtype=np.float32)
        y = np.interp(t, xp, yx[:, 0]).astype(np.float32, copy=False)
        x = np.interp(t, xp, yx[:, 1]).astype(np.float32, copy=False)
        return np.stack([y, x], axis=1)

    u = np.linspace(0.0, total, int(n), dtype=np.float32)
    y = np.interp(u, s, yx[:, 0]).astype(np.float32, copy=False)
    x = np.interp(u, s, yx[:, 1]).astype(np.float32, copy=False)
    return np.stack([y, x], axis=1)


def _load_road_prob(*, semantic_dir: Optional[Path], route_city: int) -> Optional[np.ndarray]:
    if semantic_dir is not None:
        p = Path(semantic_dir) / "osm_road_prob.npy"
        if p.exists():
            rp = np.load(str(p))
            return np.asarray(rp, dtype=np.float32) if rp.ndim == 2 else None

    raw_root = os.environ.get("RAW_ROOT", "")
    if not raw_root:
        return None
    base = Path(raw_root) / "worldtrace"
    sem = base / ("detroit_core_v1" if int(route_city) == 0 else "columbus_core_v1")
    p = sem / "osm_road_prob.npy"
    if p.exists():
        rp = np.load(str(p))
        return np.asarray(rp, dtype=np.float32) if rp.ndim == 2 else None
    return None


def _road_stats_for_abs_yx(*, abs_yx: np.ndarray, road_prob: np.ndarray, prob_thr: float) -> Tuple[float, float]:
    rp = np.asarray(road_prob, dtype=np.float32)
    H, W = map(int, rp.shape)
    y = np.rint(abs_yx[..., 0]).astype(np.int64, copy=False)
    x = np.rint(abs_yx[..., 1]).astype(np.int64, copy=False)
    y = np.clip(y, 0, H - 1)
    x = np.clip(x, 0, W - 1)
    p = rp[y, x].astype(np.float32, copy=False)
    return float(np.mean(p)), float(np.mean(p >= float(prob_thr)))


def _visit_hist(*, abs_yx: np.ndarray, H: int, W: int, stride: int) -> np.ndarray:
    stride = max(1, int(stride))
    yx = abs_yx[:, ::stride, :] if abs_yx.ndim == 3 else abs_yx[::stride, :]
    y = np.rint(yx[..., 0]).astype(np.int64, copy=False)
    x = np.rint(yx[..., 1]).astype(np.int64, copy=False)
    y = np.clip(y, 0, int(H) - 1)
    x = np.clip(x, 0, int(W) - 1)
    idx = (y * int(W) + x).reshape(-1)
    return np.bincount(idx, minlength=int(H) * int(W)).astype(np.float64, copy=False)


def _path_len(pos_yx: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos_yx, dtype=np.float32)
    disp = pos[:, 1:] - pos[:, :-1]
    step = np.linalg.norm(disp, axis=-1)
    return np.sum(step, axis=1)

def _linspace_indices(*, T: int, n: int) -> np.ndarray:
    T = int(T)
    n = int(n)
    if n <= 0 or T <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n >= T:
        return np.arange(T, dtype=np.int64)
    idx = np.rint(np.linspace(0, float(T - 1), n, dtype=np.float64)).astype(np.int64, copy=False)
    idx[0] = 0
    idx[-1] = T - 1
    return np.clip(idx, 0, T - 1).astype(np.int64, copy=False)


@torch.no_grad()
def run_eval(
    *,
    segments_parquet: Path,
    route_city: int,
    semantic_dir: Optional[Path],
    cfg: EvalCfg,
    way_to_idx: Dict[int, int],
    ae: WayCASDAutoEncoder,
    exec_model: GPSDiffusionExecutionModel,
    device: torch.device,
) -> Dict[str, object]:
    if pq is None:
        raise SystemExit("pyarrow is required. Install: conda/pip install pyarrow")

    road_prob = _load_road_prob(semantic_dir=semantic_dir, route_city=int(route_city))
    H = W = None
    if road_prob is not None:
        H, W = map(int, road_prob.shape)

    cols = ["osm_way_id", "t", "y", "x"]
    if bool(cfg.prefer_matched):
        cols.append("is_matched")
    table = pq.read_table(str(segments_parquet), columns=cols)
    way_col = table.column("osm_way_id").to_pylist()
    t_col = table.column("t").to_pylist()
    y_col = table.column("y").to_pylist()
    x_col = table.column("x").to_pylist()
    m_col = table.column("is_matched").to_pylist() if (bool(cfg.prefer_matched) and "is_matched" in table.column_names) else None

    ids = np.arange(len(way_col), dtype=np.int64)
    rng = np.random.default_rng(int(cfg.seed) + int(route_city) * 101)
    rng.shuffle(ids)

    picked: List[int] = []
    for rid in ids.tolist():
        ways0 = way_col[int(rid)] or []
        ys0 = y_col[int(rid)] or []
        xs0 = x_col[int(rid)] or []
        ts0 = t_col[int(rid)] or []
        if not (ways0 and ys0 and xs0 and ts0):
            continue
        ways = [int(w) for w in ways0 if int(w) > 0]
        if not ways:
            continue
        ways = _dedup_consecutive(ways)
        enc = [way_to_idx[int(w)] for w in ways if int(w) in way_to_idx]
        if len(enc) < int(cfg.min_way_len):
            continue
        if len(enc) > int(cfg.max_way_len):
            continue
        picked.append(int(rid))
        if len(picked) >= int(cfg.n_routes):
            break

    if not picked:
        raise SystemExit("No valid routes found after filters.")

    ade_best = []
    ade_mean = []
    fde_best = []
    fde_mean = []
    fre_best = []
    fre_mean = []
    road_prob_mean = []
    onroad_rate = []
    length_gt = []
    length_pred = []

    gt_abs_list = []
    pred_abs_best_list = []

    K = int(cfg.n_samples_per_route)
    batch_routes = max(1, int(cfg.batch_routes))
    fre_n = max(0, int(cfg.frechet_points))
    fre_idx = _linspace_indices(T=int(cfg.traj_len), n=int(fre_n)) if fre_n > 0 else None

    for b0 in range(0, len(picked), int(batch_routes)):
        rids = picked[b0 : b0 + int(batch_routes)]
        enc_list: List[List[int]] = []
        lens: List[int] = []
        gt_abs_batch: List[np.ndarray] = []
        start_pos_batch: List[np.ndarray] = []
        dest_pos_batch: List[np.ndarray] = []
        hour_batch: List[int] = []
        dow_batch: List[int] = []

        for rid in rids:
            ways0 = way_col[int(rid)] or []
            ys0 = y_col[int(rid)] or []
            xs0 = x_col[int(rid)] or []
            ts0 = t_col[int(rid)] or []
            if m_col is not None:
                mm = m_col[int(rid)] or []
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
            ways = _dedup_consecutive(ways)
            enc = [way_to_idx[int(w)] for w in ways if int(w) in way_to_idx]
            enc = enc[: int(cfg.max_way_len)]
            L = int(len(enc))
            if L < int(cfg.min_way_len):
                continue

            yx = np.stack([np.asarray(ys, dtype=np.float32), np.asarray(xs, dtype=np.float32)], axis=1)
            gt_abs = _resample_polyline(yx, n=int(cfg.traj_len))
            start_pos = gt_abs[0].astype(np.float32, copy=False)
            dest_pos = gt_abs[-1].astype(np.float32, copy=False)

            start_t = int(ts[0])
            hour = int(_hour_from_unix(np.asarray([start_t], dtype=np.int64), float(cfg.tz_offset_hours))[0])
            dow = int(_dow_from_unix(np.asarray([start_t], dtype=np.int64), float(cfg.tz_offset_hours))[0])

            enc_list.append(enc)
            lens.append(L)
            gt_abs_batch.append(gt_abs.astype(np.float32, copy=False))
            start_pos_batch.append(start_pos.astype(np.float32, copy=False))
            dest_pos_batch.append(dest_pos.astype(np.float32, copy=False))
            hour_batch.append(int(hour))
            dow_batch.append(int(dow))

        B = int(len(enc_list))
        if B <= 0:
            continue

        maxL = int(max(lens))
        pad = np.full((B, maxL), -1, dtype=np.int64)
        for i, enc in enumerate(enc_list):
            Li = int(len(enc))
            pad[i, :Li] = np.asarray(enc, dtype=np.int64)

        z, _ = ae.encode(torch.as_tensor(pad, dtype=torch.long, device=device))

        start_pos_np = np.stack(start_pos_batch, axis=0).astype(np.float32, copy=False)  # (B,2)
        dest_pos_np = np.stack(dest_pos_batch, axis=0).astype(np.float32, copy=False)  # (B,2)
        rc = {
            "start_pos": torch.as_tensor(start_pos_np, dtype=torch.float32, device=device),
            "dest_pos": torch.as_tensor(dest_pos_np, dtype=torch.float32, device=device),
            "hour": torch.as_tensor(np.asarray(hour_batch, dtype=np.int64), dtype=torch.long, device=device),
            "dow": torch.as_tensor(np.asarray(dow_batch, dtype=np.int64), dtype=torch.long, device=device),
            "route_city": torch.full((B,), int(route_city), dtype=torch.long, device=device),
        }

        # Sample K trajectories per route in one batched call (GPU-efficient).
        z_rep = z.repeat_interleave(int(K), dim=0)
        rc_rep = {k: v.repeat_interleave(int(K), dim=0) for k, v in rc.items()}
        pr_rel = exec_model.sample(route_cond=rc_rep, skeleton_latent=z_rep, traj_len=int(cfg.traj_len), fix_ends=bool(cfg.fix_ends))
        pr_rel_np = pr_rel.detach().cpu().numpy().astype(np.float32, copy=False)  # (B*K,T,2)

        start_rep = np.repeat(start_pos_np, repeats=int(K), axis=0).astype(np.float32, copy=False)  # (B*K,2)
        preds_abs_rep = start_rep[:, None, :] + pr_rel_np * float(cfg.coord_scale)  # (B*K,T,2)
        preds_abs = preds_abs_rep.reshape(B, int(K), int(cfg.traj_len), 2)

        gt_abs_arr = np.stack(gt_abs_batch, axis=0).astype(np.float32, copy=False)  # (B,T,2)

        diff = preds_abs - gt_abs_arr[:, None, :, :]
        ade_k = np.linalg.norm(diff, axis=-1).mean(axis=-1)  # (B,K)
        fde_k = np.linalg.norm(preds_abs[:, :, -1, :] - gt_abs_arr[:, None, -1, :], axis=-1)  # (B,K)

        if fre_idx is not None and fre_idx.size > 0:
            pred_f = preds_abs[:, :, fre_idx, :].reshape(B * int(K), int(fre_idx.size), 2)
            gt_f = np.repeat(gt_abs_arr[:, None, fre_idx, :], repeats=int(K), axis=1).reshape(B * int(K), int(fre_idx.size), 2)
            fre_flat = compute_frechet_per_sample(pred_f, gt_f).reshape(B, int(K))
        else:
            fre_flat = np.full((B, int(K)), np.nan, dtype=np.float32)

        for i in range(B):
            ade_best.append(float(np.min(ade_k[i])))
            ade_mean.append(float(np.mean(ade_k[i])))
            fde_best.append(float(np.min(fde_k[i])))
            fde_mean.append(float(np.mean(fde_k[i])))
            fre_best.append(float(np.min(fre_flat[i])))
            fre_mean.append(float(np.mean(fre_flat[i])))

            best_idx = int(np.argmin(ade_k[i]))
            gt_abs_list.append(gt_abs_arr[i].astype(np.float32, copy=False))
            pred_abs_best_list.append(preds_abs[i, best_idx].astype(np.float32, copy=False))

            if road_prob is not None:
                p_mean, r_on = _road_stats_for_abs_yx(
                    abs_yx=preds_abs[i, best_idx], road_prob=road_prob, prob_thr=float(cfg.onroad_prob_thr)
                )
                road_prob_mean.append(float(p_mean))
                onroad_rate.append(float(r_on))

            length_gt.append(float(_path_len(gt_abs_arr[i : i + 1])[0]))
            length_pred.append(float(_path_len(preds_abs[i, best_idx : best_idx + 1])[0]))

    # aggregate
    out: Dict[str, object] = {
        "inputs": {"segments_parquet": str(segments_parquet), "route_city": int(route_city), "semantic_dir": str(semantic_dir) if semantic_dir else None},
        "n_used": int(len(gt_abs_list)),
        "micro": {
            "ADE_best": float(np.mean(ade_best)) if ade_best else float("nan"),
            "ADE_mean": float(np.mean(ade_mean)) if ade_mean else float("nan"),
            "FDE_best": float(np.mean(fde_best)) if fde_best else float("nan"),
            "FDE_mean": float(np.mean(fde_mean)) if fde_mean else float("nan"),
            "Frechet_best": float(np.mean(fre_best)) if fre_best else float("nan"),
            "Frechet_mean": float(np.mean(fre_mean)) if fre_mean else float("nan"),
            "frechet_points": int(fre_n),
        },
        "onroad": (
            {
                "road_prob_mean": float(np.mean(road_prob_mean)) if road_prob_mean else float("nan"),
                "onroad_rate": float(np.mean(onroad_rate)) if onroad_rate else float("nan"),
                "prob_thr": float(cfg.onroad_prob_thr),
            }
            if road_prob is not None
            else None
        ),
    }

    if gt_abs_list and pred_abs_best_list:
        gt_abs_arr = np.stack(gt_abs_list, axis=0)
        pred_abs_arr = np.stack(pred_abs_best_list, axis=0)
        out["distribution"] = compute_distribution_metrics(
            pred_abs_arr,
            gt_abs_arr,
            dt_s=1.0,
            meters_per_cell=None,
            speed_bins=int(cfg.speed_bins),
            accel_bins=int(cfg.accel_bins),
            turn_bins=int(cfg.turn_bins),
            turn_min_speed=float(cfg.turn_min_speed),
        )
        out["length_jsd"] = float(
            compute_jsd_from_samples(
                np.asarray(length_pred, dtype=np.float32),
                np.asarray(length_gt, dtype=np.float32),
                bins=80,
                value_range=None,
                range_percentiles=(0.5, 99.5),
                clamp_min=0.0,
                clamp_max=None,
            )
        )

        if road_prob is not None and H is not None and W is not None:
            gt_hist = _visit_hist(abs_yx=gt_abs_arr, H=int(H), W=int(W), stride=int(cfg.spatial_stride))
            pr_hist = _visit_hist(abs_yx=pred_abs_arr, H=int(H), W=int(W), stride=int(cfg.spatial_stride))
            out["spatial_jsd"] = float(jsd_from_hist(pr_hist, gt_hist, base=2.0))
            out["spatial_stride"] = int(cfg.spatial_stride)

    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Way-CASD Execution (GPS diffusion): micro + distribution + on-road.")
    p.add_argument("--segments_parquet", type=Path, nargs="+", required=True)
    p.add_argument("--route_city", type=int, nargs="+", required=True)
    p.add_argument("--semantic_dir", type=Path, nargs="*", default=None, help="Optional per-city semantic dirs (contain osm_road_prob.npy).")

    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--exec_ckpt", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--traj_len", type=int, default=256)
    p.add_argument("--n_routes", type=int, default=512, help="Per city (after filtering).")
    p.add_argument("--n_samples_per_route", type=int, default=4)
    p.add_argument("--max_way_len", type=int, default=128)
    p.add_argument("--min_way_len", type=int, default=2)
    p.add_argument("--prefer_matched", action="store_true")
    p.add_argument("--no_fix_ends", action="store_true")
    p.add_argument("--coord_scale", type=float, default=1024.0)

    p.add_argument("--batch_routes", type=int, default=16, help="GPU batching: number of routes per sampling batch (each expands by K).")
    p.add_argument("--frechet_points", type=int, default=64, help="Downsample points for discrete Frechet (0=skip).")

    p.add_argument("--onroad_prob_thr", type=float, default=0.5)
    p.add_argument("--spatial_stride", type=int, default=4)
    p.add_argument("--speed_bins", type=int, default=120)
    p.add_argument("--accel_bins", type=int, default=120)
    p.add_argument("--turn_bins", type=int, default=60)
    p.add_argument("--turn_min_speed", type=float, default=1e-3)
    return p


def main() -> None:
    if pq is None:
        raise SystemExit("pyarrow is required. Install: conda/pip install pyarrow")

    args = build_argparser().parse_args()
    if len(args.segments_parquet) != len(args.route_city):
        raise SystemExit("--segments_parquet and --route_city must have the same length")

    sem_dirs = None
    if args.semantic_dir is not None and len(args.semantic_dir) > 0:
        sem_dirs = [Path(p) for p in args.semantic_dir]
        if len(sem_dirs) != len(args.route_city):
            raise SystemExit("--semantic_dir (if provided) must match --route_city length")

    cfg = EvalCfg(
        seed=int(args.seed),
        device=str(args.device),
        tz_offset_hours=float(args.tz_offset_hours),
        traj_len=int(args.traj_len),
        n_routes=int(args.n_routes),
        n_samples_per_route=int(args.n_samples_per_route),
        max_way_len=int(args.max_way_len),
        min_way_len=int(args.min_way_len),
        prefer_matched=bool(args.prefer_matched),
        fix_ends=(not bool(args.no_fix_ends)),
        coord_scale=float(args.coord_scale),
        batch_routes=int(args.batch_routes),
        frechet_points=int(args.frechet_points),
        onroad_prob_thr=float(args.onroad_prob_thr),
        spatial_stride=int(args.spatial_stride),
        speed_bins=int(args.speed_bins),
        accel_bins=int(args.accel_bins),
        turn_bins=int(args.turn_bins),
        turn_min_speed=float(args.turn_min_speed),
    )
    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    way_osm_id = np.asarray(wf["way_osm_id"], dtype=np.int64).reshape(-1)
    way_to_idx = {int(w): int(i) for i, w in enumerate(way_osm_id.tolist())}

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

    # Load AE (encoder only).
    ckpt_ae = torch.load(str(args.ae_ckpt), map_location=device)
    ae_state = ckpt_ae["model_state_dict"] if isinstance(ckpt_ae, dict) and "model_state_dict" in ckpt_ae else ckpt_ae
    ae_cfg_dict = ckpt_ae.get("config", {}) if isinstance(ckpt_ae, dict) else {}
    ae = WayCASDAutoEncoder(
        cfg=WayCASDAECfg(
            d_model=int(ae_cfg_dict.get("d_model", 256)),
            n_latent=int(ae_cfg_dict.get("n_latent", 64)),
            n_heads=int(ae_cfg_dict.get("n_heads", 8)),
            dropout=float(ae_cfg_dict.get("dropout", 0.1)),
            max_candidates=int(ae_cfg_dict.get("max_candidates", 32)),
            max_len=int(ae_cfg_dict.get("max_len", cfg.max_way_len)),
            coord_scale=float(ae_cfg_dict.get("coord_scale", cfg.coord_scale)),
            decoder_use_dest_dist=bool(ae_cfg_dict.get("decoder_use_dest_dist", True)),
            decoder_use_cross_attn=bool(ae_cfg_dict.get("decoder_use_cross_attn", True)),
            decoder_n_cross_heads=int(ae_cfg_dict.get("decoder_n_cross_heads", 4)),
        ),
        way_features=way_features,
        way_adj_ptr=wg["way_adj_ptr"],
        way_adj_idx=wg["way_adj_idx"],
        n_highway_types=int(max(4, n_highway_types)),
    ).to(device)
    ae.load_state_dict(ae_state, strict=False)
    ae.eval()

    # Load execution diffusion.
    ckpt_ex = torch.load(str(args.exec_ckpt), map_location=device)
    ex_state = ckpt_ex["model_state_dict"] if isinstance(ckpt_ex, dict) and "model_state_dict" in ckpt_ex else ckpt_ex
    ex_cfg = ckpt_ex.get("config", {}) if isinstance(ckpt_ex, dict) else {}
    exec_model = GPSDiffusionExecutionModel(
        cfg=GPSDiffusionCfg(
            traj_len=int(ex_cfg.get("traj_len", cfg.traj_len)),
            hidden_dim=int(ex_cfg.get("hidden_dim", 128)),
            emb_dim=int(ex_cfg.get("emb_dim", 512)),
            diffusion_steps=int(ex_cfg.get("diffusion_steps", 100)),
            prediction_type=str(ex_cfg.get("prediction_type", "eps")),
            d_model=int(ex_cfg.get("d_model", 256)),
            n_route_cities=int(ex_cfg.get("n_route_cities", 4)),
            coord_scale=float(ex_cfg.get("coord_scale", cfg.coord_scale)),
            skel_noise_sigma=float(ex_cfg.get("skel_noise_sigma", 0.1)),
        )
    ).to(device)
    exec_model.load_state_dict(ex_state, strict=False)
    exec_model.eval()

    per_city = []
    for i, (sp, cid) in enumerate(zip(args.segments_parquet, args.route_city)):
        sem = sem_dirs[i] if sem_dirs is not None else None
        rep = run_eval(
            segments_parquet=Path(sp),
            route_city=int(cid),
            semantic_dir=sem,
            cfg=cfg,
            way_to_idx=way_to_idx,
            ae=ae,
            exec_model=exec_model,
            device=device,
        )
        per_city.append(rep)

    out = {
        "ok": True,
        "task": "way_casd_exec_eval",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "segments_parquet": [str(p) for p in args.segments_parquet],
            "route_city": [int(x) for x in args.route_city],
            "semantic_dir": [str(p) for p in sem_dirs] if sem_dirs is not None else None,
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
            "exec_ckpt": str(args.exec_ckpt),
        },
        "per_city": per_city,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
