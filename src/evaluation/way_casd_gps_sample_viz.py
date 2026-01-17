from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models.way_casd.gps_diffusion import GPSDiffusionCfg, GPSDiffusionExecutionModel
from src.models.way_casd.way_casd import WayCASDAECfg, WayCASDAutoEncoder
from src.models.way_casd.way_encoder import make_way_feature_tensors

try:
    import pyarrow.parquet as pq  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pq = None

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
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


def _imshow_background(ax, arr: np.ndarray, *, cmap: str, alpha: float, vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    H, W = map(int, arr.shape)
    ax.imshow(arr, cmap=cmap, origin="lower", extent=(0, W, 0, H), alpha=float(alpha), vmin=vmin, vmax=vmax, interpolation="nearest")


def _plot_one(
    *,
    out_png: Path,
    rid: int,
    gt_abs: np.ndarray,  # (T,2) yx
    pred_abs_list: List[np.ndarray],  # list of (T,2)
    road_prob: Optional[np.ndarray],
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("Missing dependency: matplotlib (needed for plotting).") from e

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.0))
    ax_main, ax_spread = axes.tolist()

    if road_prob is not None:
        rp = np.clip(np.asarray(road_prob, dtype=np.float32), 0.0, 1.0)
        for ax in (ax_main, ax_spread):
            _imshow_background(ax, rp, cmap="Greys", alpha=0.35, vmin=0.0, vmax=1.0)

    # Panel 1: GT vs best pred (by endpoint error).
    gt = np.asarray(gt_abs, dtype=np.float64)
    best_i = 0
    if pred_abs_list:
        end = gt[-1]
        errs = [float(np.linalg.norm(p[-1] - end)) for p in pred_abs_list]
        best_i = int(np.argmin(np.asarray(errs, dtype=np.float64)))
    ax_main.plot(gt[:, 1], gt[:, 0], color="black", linewidth=2.0, alpha=0.9, label="GT")
    if pred_abs_list:
        p = np.asarray(pred_abs_list[best_i], dtype=np.float64)
        ax_main.plot(p[:, 1], p[:, 0], color="#4c72b0", linewidth=2.0, alpha=0.9, label="Pred(best)")

    # Panel 2: sample spread
    ax_spread.plot(gt[:, 1], gt[:, 0], color="black", linewidth=2.0, alpha=0.7, label="GT")
    for p in pred_abs_list:
        p = np.asarray(p, dtype=np.float64)
        ax_spread.plot(p[:, 1], p[:, 0], color="#4c72b0", linewidth=1.2, alpha=0.25)

    for ax in (ax_main, ax_spread):
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower left", frameon=False, fontsize=9)

    ax_main.set_title("GT vs best pred")
    ax_spread.set_title("Pred sample spread")
    fig.suptitle(f"route={rid} (Execution diffusion)", fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample Way-CASD Execution Stage (GPS diffusion) and visualize.")
    p.add_argument("--segments_parquet", type=Path, required=True)
    p.add_argument("--route_city", type=int, required=True)
    p.add_argument("--semantic_dir", type=Path, default=None, help="Optional: contains osm_road_prob.npy for background.")

    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--exec_ckpt", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--traj_len", type=int, default=256)
    p.add_argument("--n_routes", type=int, default=8)
    p.add_argument("--n_samples_per_route", type=int, default=4)
    p.add_argument("--max_way_len", type=int, default=128)
    p.add_argument("--min_way_len", type=int, default=2)
    p.add_argument("--prefer_matched", action="store_true")
    p.add_argument("--no_fix_ends", action="store_true")
    return p


def main() -> None:
    if pq is None:
        raise SystemExit("pyarrow is required. Install: conda/pip install pyarrow")

    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Cfg(
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
    )
    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    way_osm_id = np.asarray(wf["way_osm_id"], dtype=np.int64).reshape(-1)
    way_to_idx = {int(w): int(i) for i, w in enumerate(way_osm_id.tolist())}
    coord_scale = float(1024.0)

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
        )
    ).to(device)
    exec_model.load_state_dict(ex_state, strict=False)
    exec_model.eval()

    # Background: road_prob raster (optional)
    road_prob = None
    sem = Path(args.semantic_dir) if args.semantic_dir is not None else None
    if sem is None:
        raw_root = os.environ.get("RAW_ROOT", "")
        if raw_root:
            sem0 = Path(raw_root) / "worldtrace" / "detroit_core_v1"
            if int(args.route_city) == 1:
                sem0 = Path(raw_root) / "worldtrace" / "columbus_core_v1"
            sem = sem0 if sem0.exists() else None
    if sem is not None:
        p = Path(sem) / "osm_road_prob.npy"
        if p.exists():
            rp = np.load(str(p))
            if rp.ndim == 2:
                road_prob = np.asarray(rp, dtype=np.float32)

    # Load segments parquet rows.
    cols = ["osm_way_id", "t", "y", "x"]
    if cfg.prefer_matched:
        cols.append("is_matched")
    table = pq.read_table(str(args.segments_parquet), columns=cols)
    way_col = table.column("osm_way_id").to_pylist()
    t_col = table.column("t").to_pylist()
    y_col = table.column("y").to_pylist()
    x_col = table.column("x").to_pylist()
    m_col = table.column("is_matched").to_pylist() if (cfg.prefer_matched and "is_matched" in table.column_names) else None

    ids = np.arange(len(way_col), dtype=np.int64)
    rng = np.random.default_rng(int(cfg.seed))
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
        picked.append(int(rid))
        if len(picked) >= int(cfg.n_routes):
            break

    if not picked:
        raise SystemExit("No valid routes found in segments_parquet after filters.")

    per_route = []
    for rid in picked:
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

        yx = np.stack([np.asarray(ys, dtype=np.float32), np.asarray(xs, dtype=np.float32)], axis=1)
        gt_abs = _resample_polyline(yx, n=int(cfg.traj_len))
        start_pos = gt_abs[0].astype(np.float32, copy=False)
        dest_pos = gt_abs[-1].astype(np.float32, copy=False)
        gt_rel = (gt_abs - start_pos[None, :]) / coord_scale

        start_t = int(ts[0])
        hour = int(_hour_from_unix(np.asarray([start_t], dtype=np.int64), cfg.tz_offset_hours)[0])
        dow = int(_dow_from_unix(np.asarray([start_t], dtype=np.int64), cfg.tz_offset_hours)[0])

        # Encode skeleton latent from GT way seq.
        pad = np.full((1, L), -1, dtype=np.int64)
        if L > 0:
            pad[0, :L] = np.asarray(enc, dtype=np.int64)
        z, _ = ae.encode(torch.as_tensor(pad, dtype=torch.long, device=device))

        K = int(cfg.n_samples_per_route)
        route_cond = {
            "start_pos": torch.as_tensor(np.repeat(start_pos[None, :], K, axis=0), dtype=torch.float32, device=device),
            "dest_pos": torch.as_tensor(np.repeat(dest_pos[None, :], K, axis=0), dtype=torch.float32, device=device),
            "hour": torch.as_tensor(np.full((K,), hour, dtype=np.int64), dtype=torch.long, device=device),
            "dow": torch.as_tensor(np.full((K,), dow, dtype=np.int64), dtype=torch.long, device=device),
            "route_city": torch.as_tensor(np.full((K,), int(args.route_city), dtype=np.int64), dtype=torch.long, device=device),
        }
        z_rep = z.repeat(K, 1, 1)
        pred_rel = exec_model.sample(route_cond=route_cond, skeleton_latent=z_rep, traj_len=int(cfg.traj_len), fix_ends=bool(cfg.fix_ends))
        pred_rel = pred_rel.detach().cpu().numpy().astype(np.float32, copy=False)  # (K,T,2)
        pred_abs_list = [(p * coord_scale + start_pos[None, :]).astype(np.float32, copy=False) for p in pred_rel]

        out_png = out_dir / f"case_exec_{rid:05d}.png"
        _plot_one(out_png=out_png, rid=int(rid), gt_abs=gt_abs, pred_abs_list=pred_abs_list, road_prob=road_prob)

        per_route.append({"row_id": int(rid), "way_len": int(L), "hour": int(hour), "dow": int(dow)})

    report = {
        "ok": True,
        "task": "way_casd_gps_sample_viz",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": asdict(cfg),
        "inputs": {
            "segments_parquet": str(args.segments_parquet),
            "route_city": int(args.route_city),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
            "exec_ckpt": str(args.exec_ckpt),
        },
        "picked_rows": picked,
        "per_route": per_route,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_dir/'report.json'}")
    print(f"[saved] figures: {out_dir}")


if __name__ == "__main__":
    main()
