from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.way_graph.way_sequence_dataset import WayRouteDataset, load_way_routes_npz, make_way_casd_collate_fn
from src.evaluation.way_casd_teacher_forcing_coverage import _build_ae, _set_seed
from src.models.way_casd.conditions import ConditionEncoderCfg
from src.models.way_casd.latent_flow import LatentFlowCfg, LatentFlowMatching
from src.training.train_way_casd_flow import RegionSeqLookup, _to_device

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _qstats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
        }
    return {
        "n": float(int(arr.size)),
        "mean": float(np.mean(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def _jaccard_dist(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    if union <= 0:
        return 0.0
    return float(1.0 - (float(inter) / float(union)))


def _cluster_by_jaccard_threshold(*, way_sets: List[set[int]], dist_thr: float) -> np.ndarray:
    m = int(len(way_sets))
    if m <= 0:
        return np.zeros((0,), dtype=np.int32)
    if m == 1:
        return np.asarray([0], dtype=np.int32)

    adj: List[List[int]] = [[] for _ in range(m)]
    thr = float(dist_thr)
    for i in range(m):
        si = way_sets[i]
        for j in range(i + 1, m):
            if _jaccard_dist(si, way_sets[j]) <= thr:
                adj[i].append(j)
                adj[j].append(i)

    labels = np.full((m,), -1, dtype=np.int32)
    cur = 0
    for s in range(m):
        if int(labels[s]) >= 0:
            continue
        stack = [int(s)]
        labels[s] = int(cur)
        while stack:
            u = int(stack.pop())
            for v in adj[u]:
                if int(labels[v]) >= 0:
                    continue
                labels[v] = int(cur)
                stack.append(int(v))
        cur += 1
    return labels


def _subset_indices_from_route_ids(dataset: WayRouteDataset, route_ids: np.ndarray) -> np.ndarray:
    route_ids = np.asarray(route_ids, dtype=np.int64).reshape(-1)
    if route_ids.size == 0:
        return np.zeros((0,), dtype=np.int64)
    mask = np.isin(dataset.route_ids.astype(np.int64, copy=False), route_ids, assume_unique=False)
    return np.nonzero(mask)[0].astype(np.int64, copy=False)


def _build_flow(*, flow_ckpt: Path, device: torch.device) -> tuple[LatentFlowMatching, dict]:
    ckpt = torch.load(str(flow_ckpt), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cfg_in = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected Flow state format: {type(state)}")
    if not isinstance(cfg_in, dict):
        cfg_in = {}

    flow = LatentFlowMatching(
        cfg=LatentFlowCfg(
            d_model=int(cfg_in.get("d_model", 256)),
            n_latent=int(cfg_in.get("n_latent", 64)),
            n_layers=int(cfg_in.get("n_layers", 6)),
            n_heads=int(cfg_in.get("n_heads", 8)),
            dropout=float(cfg_in.get("dropout", 0.1)),
            noise_sigma=float(cfg_in.get("noise_sigma", 1.0)),
            solver_steps=int(cfg_in.get("solver_steps", 20)),
            cond_dropout_p=float(cfg_in.get("cond_dropout_p", 0.0)),
            cond_inject=str(cfg_in.get("cond_inject", "add")),
            use_region_seq=bool(cfg_in.get("use_region_seq", False)),
            n_regions=int(cfg_in.get("n_regions", 154)),
            region_max_len=int(cfg_in.get("region_max_len", 16)),
        ),
        cond_cfg=ConditionEncoderCfg(d_model=int(cfg_in.get("d_model", 256)), coord_scale=1024.0),
    ).to(device)
    missing, unexpected = flow.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Flow ckpt mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    flow.eval()
    return flow, cfg_in


def _pick_route_ids(
    routes: Any,
    *,
    seed: int,
    n_routes: int,
    min_hops: int,
    max_way_len: int,
    split_json: Path,
    split_part: str,
) -> np.ndarray:
    sp = _read_json(split_json)
    splits = sp.get("splits", sp)
    ids_raw = splits.get(str(split_part), None) if isinstance(splits, dict) else None
    if ids_raw is None:
        raise SystemExit(f"[FATAL] split_json missing split_part={split_part}")
    split_ids = np.asarray([int(x) for x in list(ids_raw)], dtype=np.int64)
    if split_ids.size <= 0:
        raise SystemExit(f"[FATAL] split_part={split_part} is empty")

    keep = (
        np.isin(np.arange(routes.way_seq_len.shape[0], dtype=np.int64), split_ids, assume_unique=False)
        & (routes.way_seq_len >= (int(min_hops) + 1))
        & (routes.way_seq_len <= int(max_way_len))
    )
    ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(ids)
    return ids[: min(int(n_routes), int(ids.size))]


@dataclass(frozen=True)
class Cfg:
    seed: int
    device: str
    n_routes: int
    min_hops: int
    max_way_len: int
    split_part: str
    jaccard_dist_thr: float
    min_routes_per_od: int
    min_corridors_per_od: int
    min_routes_per_corridor: int
    encode_batch_size: int
    batch_size: int
    num_workers: int
    n_samples_per_route: int
    solver_steps: Optional[int]
    cfg_scale: float
    log_every_batches: int
    route_level_threshold: float


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(
        description="Final diagnostic: nearest corridor-centroid cosine for Flow z samples."
    )
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--flow_ckpt", type=Path, required=True)
    p.add_argument("--split_json", type=Path, required=True)
    p.add_argument("--split_part", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--region_seq_npz", type=Path, default=None)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_per_route_json", type=Path, default=None)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_routes", type=int, default=5000)
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)

    p.add_argument("--jaccard_dist_thr", type=float, default=0.3)
    p.add_argument("--min_routes_per_od", type=int, default=3)
    p.add_argument("--min_corridors_per_od", type=int, default=2)
    p.add_argument("--min_routes_per_corridor", type=int, default=2)

    p.add_argument("--encode_batch_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--n_samples_per_route", type=int, default=16)
    p.add_argument("--solver_steps", type=int, default=10)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--log_every_batches", type=int, default=5)
    p.add_argument("--route_level_threshold", type=float, default=0.9)
    args = p.parse_args()

    cfg = Cfg(
        seed=int(args.seed),
        device=str(args.device),
        n_routes=int(args.n_routes),
        min_hops=int(args.min_hops),
        max_way_len=int(args.max_way_len),
        split_part=str(args.split_part),
        jaccard_dist_thr=float(args.jaccard_dist_thr),
        min_routes_per_od=max(2, int(args.min_routes_per_od)),
        min_corridors_per_od=max(1, int(args.min_corridors_per_od)),
        min_routes_per_corridor=max(1, int(args.min_routes_per_corridor)),
        encode_batch_size=max(1, int(args.encode_batch_size)),
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(args.num_workers)),
        n_samples_per_route=max(2, int(args.n_samples_per_route)),
        solver_steps=(None if int(args.solver_steps) <= 0 else int(args.solver_steps)),
        cfg_scale=float(args.cfg_scale),
        log_every_batches=max(1, int(args.log_every_batches)),
        route_level_threshold=float(args.route_level_threshold),
    )

    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    all_ids = _pick_route_ids(
        routes,
        seed=cfg.seed,
        n_routes=cfg.n_routes,
        min_hops=cfg.min_hops,
        max_way_len=cfg.max_way_len,
        split_json=Path(args.split_json),
        split_part=cfg.split_part,
    )
    if all_ids.size <= 0:
        raise SystemExit("[FATAL] no routes selected.")

    # Gather per-route GT seq and OD keys.
    rid_list = all_ids.tolist()
    seqs: List[List[int]] = []
    way_sets: List[set[int]] = []
    od_keys: List[Tuple[int, int, int]] = []  # city, start_way, dest_way
    for rid in rid_list:
        rid_i = int(rid)
        l = int(routes.way_seq_len[rid_i])
        s = int(routes.way_seq_ptr[rid_i])
        gt = routes.way_seq_idx[s : s + l].astype(np.int64, copy=False).tolist()
        gt_ids = [int(x) for x in gt]
        if len(gt_ids) <= 1:
            continue
        city = int(routes.route_city[rid_i])
        sw = int(routes.start_way[rid_i])
        dw = int(routes.dest_way[rid_i])
        seqs.append(gt_ids)
        way_sets.append(set(gt_ids))
        od_keys.append((city, sw, dw))

    if len(seqs) <= 0:
        raise SystemExit("[FATAL] no valid routes after sequence extraction.")

    # Re-build route list (drop invalids if any).
    rid_kept = np.asarray([int(rid_list[i]) for i in range(len(seqs))], dtype=np.int64)
    n_kept = int(rid_kept.size)
    print(f"[probe] selected routes={n_kept}", flush=True)

    # Build AE and encode z_enc for clustering/centroids.
    ae, strict_ok = _build_ae(
        ae_ckpt=Path(args.ae_ckpt),
        way_graph_npz=Path(args.way_graph_npz),
        way_features_npz=Path(args.way_features_npz),
        device=device,
    )

    z_flat = np.zeros((n_kept, int(ae.cfg.n_latent) * int(ae.cfg.d_model)), dtype=np.float32)
    bs_enc = int(cfg.encode_batch_size)
    n_enc_batches = int((n_kept + bs_enc - 1) // bs_enc)
    for bi in range(n_enc_batches):
        i0 = int(bi * bs_enc)
        i1 = int(min(n_kept, (bi + 1) * bs_enc))
        batch_seqs = seqs[i0:i1]
        max_l = int(max(len(x) for x in batch_seqs))
        pad = np.full((i1 - i0, max_l), -1, dtype=np.int64)
        for j, seq in enumerate(batch_seqs):
            pad[j, : len(seq)] = np.asarray(seq, dtype=np.int64)
        way_pad = torch.as_tensor(pad, dtype=torch.long, device=device)
        z, _ = ae.encode(way_pad)
        z_flat[i0:i1] = z.reshape(i1 - i0, -1).detach().cpu().numpy().astype(np.float32, copy=False)
        if (bi + 1) % 5 == 0 or (bi + 1) == n_enc_batches:
            print(f"[encode] batch {bi+1}/{n_enc_batches} routes {i1}/{n_kept}", flush=True)

    # Build OD -> local indices
    od_to_idx: Dict[Tuple[int, int, int], List[int]] = {}
    for i, od in enumerate(od_keys):
        od_to_idx.setdefault(od, []).append(int(i))

    # Build corridor centroids per OD
    od_centroids_np: Dict[Tuple[int, int, int], np.ndarray] = {}
    corridors_per_od: List[int] = []
    for od, idxs in od_to_idx.items():
        if len(idxs) < int(cfg.min_routes_per_od):
            continue
        local_sets = [way_sets[i] for i in idxs]
        labels = _cluster_by_jaccard_threshold(way_sets=local_sets, dist_thr=float(cfg.jaccard_dist_thr))
        cents: List[np.ndarray] = []
        for cid in np.unique(labels).tolist():
            cid_i = int(cid)
            mem_local = np.nonzero(labels == cid_i)[0].astype(np.int64, copy=False)
            if int(mem_local.size) < int(cfg.min_routes_per_corridor):
                continue
            mem_global = np.asarray([idxs[int(k)] for k in mem_local.tolist()], dtype=np.int64)
            zc = np.mean(z_flat[mem_global], axis=0, dtype=np.float64).astype(np.float32, copy=False)
            nz = np.linalg.norm(zc)
            if np.isfinite(nz) and nz > 0:
                zc = (zc / nz).astype(np.float32, copy=False)
                cents.append(zc)
        if len(cents) >= int(cfg.min_corridors_per_od):
            od_centroids_np[od] = np.stack(cents, axis=0).astype(np.float32, copy=False)
            corridors_per_od.append(int(len(cents)))

    if len(od_centroids_np) <= 0:
        raise SystemExit("[FATAL] no OD kept after corridor centroid filtering.")

    # Eval subset: only routes whose OD is kept
    eval_route_ids = np.asarray(
        [int(rid_kept[i]) for i, od in enumerate(od_keys) if od in od_centroids_np],
        dtype=np.int64,
    )
    eval_set = set(int(x) for x in eval_route_ids.tolist())
    print(
        f"[probe] routes_total={n_kept} routes_eval={int(eval_route_ids.size)} "
        f"od_all={len(od_to_idx)} od_kept={len(od_centroids_np)}",
        flush=True,
    )

    # Build Flow + dataloader for eval routes.
    flow, flow_cfg = _build_flow(flow_ckpt=Path(args.flow_ckpt), device=device)
    region_seq_lookup: Optional[RegionSeqLookup] = None
    if bool(flow_cfg.get("use_region_seq", False)):
        rnpz = Path(args.region_seq_npz) if args.region_seq_npz is not None else Path(str(flow_cfg.get("region_seq_npz", "")))
        if not str(rnpz):
            raise SystemExit("[FATAL] Flow requires region_seq but --region_seq_npz is missing and ckpt config has no path.")
        region_seq_lookup = RegionSeqLookup(region_seq_npz=rnpz)

    dataset = WayRouteDataset(routes, max_routes=None, max_way_len=int(cfg.max_way_len), min_hops=int(cfg.min_hops))
    subset_idx = _subset_indices_from_route_ids(dataset, eval_route_ids)
    subset = Subset(dataset, subset_idx.tolist())
    wg = np.load(str(args.way_graph_npz), allow_pickle=True)
    collate_fn = make_way_casd_collate_fn(
        way_adj_ptr=wg["way_adj_ptr"],
        way_adj_idx=wg["way_adj_idx"],
        max_candidates=32,
        tz_offset_hours=-5.0,
    )
    loader = DataLoader(
        subset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(device.type == "cuda"),
        persistent_workers=bool(int(cfg.num_workers) > 0),
        prefetch_factor=(4 if int(cfg.num_workers) > 0 else None),
        collate_fn=collate_fn,
    )

    K = int(cfg.n_samples_per_route)
    nearest_all: List[float] = []
    route_mean_all: List[float] = []
    od_mean_map: Dict[Tuple[int, int, int], List[float]] = {}
    per_route_rows: List[Dict[str, Any]] = []
    n_done = 0
    n_total = int(eval_route_ids.size)
    n_batches = int((n_total + int(cfg.batch_size) - 1) // int(cfg.batch_size))

    for bi, batch in enumerate(loader, start=1):
        b = _to_device(batch, device, region_seq=region_seq_lookup, need_trans=False)
        cond = b["route_cond"]
        B = int(cond["start_pos"].shape[0])
        cond_rep = {k: v.repeat_interleave(K, dim=0) for k, v in cond.items()}
        zf = flow.sample(
            route_cond=cond_rep,
            solver_steps=cfg.solver_steps,
            cfg_scale=float(cfg.cfg_scale),
        )  # (B*K, L, D)
        zf = zf.reshape(B, K, -1).to(dtype=torch.float32)
        zf = F.normalize(zf, p=2, dim=-1, eps=1e-8)

        route_ids_b = batch["route_id"].detach().cpu().numpy().astype(np.int64, copy=False)
        city_b = batch["route_cond"]["route_city"].detach().cpu().numpy().astype(np.int64, copy=False)
        sw_b = batch["route_cond"]["start_way"].detach().cpu().numpy().astype(np.int64, copy=False)
        dw_b = batch["route_cond"]["dest_way"].detach().cpu().numpy().astype(np.int64, copy=False)

        for i in range(B):
            rid = int(route_ids_b[i])
            if rid not in eval_set:
                continue
            od = (int(city_b[i]), int(sw_b[i]), int(dw_b[i]))
            cents_np = od_centroids_np.get(od, None)
            if cents_np is None or cents_np.shape[0] <= 0:
                continue
            cents = torch.as_tensor(cents_np, dtype=torch.float32, device=device)  # (C,D)
            sim = torch.matmul(zf[i], cents.t())  # (K,C)
            nearest = torch.max(sim, dim=1).values  # (K,)
            nearest_np = nearest.detach().cpu().numpy().astype(np.float32, copy=False)
            mean_i = float(np.mean(nearest_np, dtype=np.float64))
            p50_i = float(np.quantile(nearest_np, 0.5))
            p90_i = float(np.quantile(nearest_np, 0.9))

            nearest_all.extend(nearest_np.astype(np.float64, copy=False).tolist())
            route_mean_all.append(mean_i)
            od_mean_map.setdefault(od, []).append(mean_i)
            per_route_rows.append(
                {
                    "route_id": rid,
                    "city": int(od[0]),
                    "start_way": int(od[1]),
                    "dest_way": int(od[2]),
                    "n_corridors": int(cents_np.shape[0]),
                    "nearest_corridor_cos_mean": mean_i,
                    "nearest_corridor_cos_p50": p50_i,
                    "nearest_corridor_cos_p90": p90_i,
                }
            )

        n_done += B
        if bi % int(cfg.log_every_batches) == 0 or bi == n_batches:
            rm = np.asarray(route_mean_all, dtype=np.float64)
            m = float(np.mean(rm)) if rm.size else float("nan")
            rate = float(np.mean((rm >= float(cfg.route_level_threshold)).astype(np.float32))) if rm.size else float("nan")
            print(
                f"[probe] batch {bi}/{n_batches} routes {min(n_done, n_total)}/{n_total} "
                f"route_nearest_cos_mean={m:.4f} route_rate@{cfg.route_level_threshold:.2f}={rate:.4f}",
                flush=True,
            )

    od_means = np.asarray([float(np.mean(v)) for v in od_mean_map.values() if len(v) > 0], dtype=np.float64)
    route_mean_np = np.asarray(route_mean_all, dtype=np.float64)
    nearest_np = np.asarray(nearest_all, dtype=np.float64)
    thr = float(cfg.route_level_threshold)

    rep = {
        "ok": True,
        "task": "way_casd_flow_corridor_residual_probe",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": {
            "seed": int(cfg.seed),
            "device": str(device),
            "split_part": str(cfg.split_part),
            "n_routes": int(cfg.n_routes),
            "min_hops": int(cfg.min_hops),
            "max_way_len": int(cfg.max_way_len),
            "jaccard_dist_thr": float(cfg.jaccard_dist_thr),
            "min_routes_per_od": int(cfg.min_routes_per_od),
            "min_corridors_per_od": int(cfg.min_corridors_per_od),
            "min_routes_per_corridor": int(cfg.min_routes_per_corridor),
            "n_samples_per_route": int(cfg.n_samples_per_route),
            "solver_steps": (None if cfg.solver_steps is None else int(cfg.solver_steps)),
            "cfg_scale": float(cfg.cfg_scale),
            "route_level_threshold": float(cfg.route_level_threshold),
        },
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
            "flow_ckpt": str(args.flow_ckpt),
            "split_json": str(args.split_json),
            "region_seq_npz": (str(args.region_seq_npz) if args.region_seq_npz is not None else None),
        },
        "summary": {
            "n_routes_total_selected": int(n_kept),
            "n_routes_eval": int(eval_route_ids.size),
            "n_od_all": int(len(od_to_idx)),
            "n_od_kept": int(len(od_centroids_np)),
            "corridors_per_od": _qstats(corridors_per_od),
            "nearest_corridor_cos_all_samples": _qstats(nearest_np),
            "route_mean_nearest_corridor_cos": _qstats(route_mean_np),
            "od_mean_nearest_corridor_cos": _qstats(od_means),
            "route_rate_mean_ge_threshold": float(np.mean((route_mean_np >= thr).astype(np.float32))) if route_mean_np.size else float("nan"),
            "sample_rate_ge_0p90": float(np.mean((nearest_np >= 0.90).astype(np.float32))) if nearest_np.size else float("nan"),
            "sample_rate_ge_0p85": float(np.mean((nearest_np >= 0.85).astype(np.float32))) if nearest_np.size else float("nan"),
        },
        "ae_ckpt_strict_load_ok": bool(strict_ok),
    }

    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved: {out_json}", flush=True)
    if args.out_per_route_json is not None:
        out_per = Path(args.out_per_route_json)
        out_per.parent.mkdir(parents=True, exist_ok=True)
        out_per.write_text(
            json.dumps(
                {
                    "ok": True,
                    "task": "way_casd_flow_corridor_residual_probe_per_route",
                    "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
                    "rows": per_route_rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[OK] saved per-route: {out_per}", flush=True)

    s = rep["summary"]
    print(
        "Flow->Nearest-Corridor residual | "
        f"route_mean_cos={float(s['route_mean_nearest_corridor_cos']['mean']):.4f} "
        f"(p50={float(s['route_mean_nearest_corridor_cos']['p50']):.4f}, "
        f"p90={float(s['route_mean_nearest_corridor_cos']['p90']):.4f}) | "
        f"sample>=0.90={float(s['sample_rate_ge_0p90']):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
