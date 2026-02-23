from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.evaluation.way_casd_teacher_forcing_coverage import _build_ae, _set_seed

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_stats(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size <= 0:
        return {
            "mean": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "n": 0,
        }
    return {
        "mean": float(np.mean(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "n": int(arr.size),
    }


def _pick_route_ids(
    routes: Any,
    *,
    seed: int,
    n_routes: int,
    min_hops: int,
    max_way_len: int,
    split_json: Optional[Path],
    split_part: Optional[str],
) -> Dict[int, np.ndarray]:
    split_ids: Optional[np.ndarray] = None
    if split_json is not None:
        if split_part is None:
            raise SystemExit("[FATAL] --split_part is required when --split_json is set.")
        if not split_json.exists():
            raise SystemExit(f"[FATAL] split_json not found: {split_json}")
        sp = _read_json(split_json)
        splits = sp.get("splits", sp)
        ids_raw = splits.get(str(split_part), None) if isinstance(splits, dict) else None
        if ids_raw is None:
            raise SystemExit(f"[FATAL] split_json missing split_part={split_part}")
        split_ids = np.asarray([int(x) for x in list(ids_raw)], dtype=np.int64)
        if split_ids.size <= 0:
            raise SystemExit(f"[FATAL] split_part={split_part} is empty")

    cities = sorted(set(int(x) for x in routes.route_city.astype(np.int64).tolist()))
    out: Dict[int, np.ndarray] = {}
    for city in cities:
        keep = (
            (routes.route_city.astype(np.int64) == int(city))
            & (routes.way_seq_len >= (int(min_hops) + 1))
            & (routes.way_seq_len <= int(max_way_len))
        )
        ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
        if split_ids is not None:
            ids = ids[np.isin(ids, split_ids, assume_unique=False)]
        rng = np.random.default_rng(int(seed) + 101 * int(city))
        rng.shuffle(ids)
        out[int(city)] = ids[: min(int(n_routes), int(ids.size))]
    return out


def _jaccard_dist(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    if union <= 0:
        return 0.0
    return float(1.0 - (float(inter) / float(union)))


def _cluster_by_jaccard_threshold(
    *,
    way_sets: List[set[int]],
    dist_thr: float,
) -> np.ndarray:
    """
    Single-link style clustering by thresholded Jaccard distance.
    Routes i,j are connected if dist(i,j) <= dist_thr. Clusters are connected components.
    """
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
            d = _jaccard_dist(si, way_sets[j])
            if d <= thr:
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


@dataclass(frozen=True)
class Cfg:
    seed: int
    device: str
    n_routes: int
    min_hops: int
    max_way_len: int
    split_json: Optional[str]
    split_part: Optional[str]
    min_routes_per_od: int
    min_routes_per_corridor: int
    jaccard_dist_thr: float
    encode_batch_size: int
    max_cross_pairs: int
    progress_every: int
    save_per_od: bool


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(
        description="Probe AE corridor separability with OD-internal Jaccard clustering + z_enc cosine."
    )
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_routes", type=int, default=5000, help="Per city.")
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--split_json", type=Path, default=None)
    p.add_argument("--split_part", choices=["train", "val", "test"], default=None)

    p.add_argument("--min_routes_per_od", type=int, default=3)
    p.add_argument("--min_routes_per_corridor", type=int, default=2)
    p.add_argument("--jaccard_dist_thr", type=float, default=0.3, help="Within-OD clustering threshold on Jaccard distance.")
    p.add_argument("--encode_batch_size", type=int, default=512)
    p.add_argument("--max_cross_pairs", type=int, default=200000)
    p.add_argument("--progress_every", type=int, default=20)
    p.add_argument("--save_per_od", action="store_true")
    args = p.parse_args()

    cfg = Cfg(
        seed=int(args.seed),
        device=str(args.device),
        n_routes=int(args.n_routes),
        min_hops=int(args.min_hops),
        max_way_len=int(args.max_way_len),
        split_json=(str(args.split_json) if args.split_json is not None else None),
        split_part=(str(args.split_part) if args.split_part is not None else ("test" if args.split_json is not None else None)),
        min_routes_per_od=int(args.min_routes_per_od),
        min_routes_per_corridor=int(args.min_routes_per_corridor),
        jaccard_dist_thr=float(args.jaccard_dist_thr),
        encode_batch_size=max(1, int(args.encode_batch_size)),
        max_cross_pairs=max(1, int(args.max_cross_pairs)),
        progress_every=max(1, int(args.progress_every)),
        save_per_od=bool(args.save_per_od),
    )
    if float(cfg.jaccard_dist_thr) < 0.0 or float(cfg.jaccard_dist_thr) > 1.0:
        raise SystemExit("[FATAL] --jaccard_dist_thr must be in [0,1].")

    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    ae, strict_ok = _build_ae(
        ae_ckpt=Path(args.ae_ckpt),
        way_graph_npz=Path(args.way_graph_npz),
        way_features_npz=Path(args.way_features_npz),
        device=device,
    )

    picks = _pick_route_ids(
        routes,
        seed=int(cfg.seed),
        n_routes=int(cfg.n_routes),
        min_hops=int(cfg.min_hops),
        max_way_len=int(cfg.max_way_len),
        split_json=(Path(cfg.split_json) if cfg.split_json is not None else None),
        split_part=(str(cfg.split_part) if cfg.split_part is not None else None),
    )

    all_rids: List[int] = []
    all_city: List[int] = []
    all_od: List[Tuple[int, int]] = []
    all_seq: List[List[int]] = []
    all_way_sets: List[set[int]] = []
    for city, ids in picks.items():
        for rid in ids.tolist():
            rid_i = int(rid)
            l = int(routes.way_seq_len[rid_i])
            s = int(routes.way_seq_ptr[rid_i])
            gt = routes.way_seq_idx[s : s + l].astype(np.int64, copy=False).tolist()
            gt_ids = [int(x) for x in gt]
            if len(gt_ids) <= 1:
                continue
            sw = int(routes.start_way[rid_i])
            dw = int(routes.dest_way[rid_i])
            all_rids.append(int(rid_i))
            all_city.append(int(city))
            all_od.append((int(sw), int(dw)))
            all_seq.append(gt_ids)
            all_way_sets.append(set(int(x) for x in gt_ids))

    n = int(len(all_rids))
    if n <= 0:
        raise SystemExit("[FATAL] no routes selected after filtering.")
    print(f"[encode] selected routes={n}", flush=True)

    # Encode z_enc and L2-normalize for cosine similarity.
    emb_list: List[np.ndarray] = []
    bs = int(cfg.encode_batch_size)
    n_batches = int((n + bs - 1) // bs)
    for b in range(n_batches):
        i0 = int(b * bs)
        i1 = int(min(n, (b + 1) * bs))
        batch_seqs = all_seq[i0:i1]
        max_l = int(max(len(x) for x in batch_seqs))
        way_pad = np.full((len(batch_seqs), max_l), -1, dtype=np.int64)
        for i, seq in enumerate(batch_seqs):
            way_pad[i, : len(seq)] = np.asarray(seq, dtype=np.int64)
        way_pad_t = torch.as_tensor(way_pad, dtype=torch.long, device=device)
        z, _ = ae.encode(way_pad_t)  # (B,n_lat,d)
        zflat = z.reshape(z.shape[0], -1).float()
        zflat = zflat / (torch.linalg.norm(zflat, dim=-1, keepdim=True) + 1e-8)
        emb_list.append(zflat.detach().cpu().numpy().astype(np.float32, copy=False))
        if (b + 1) % int(cfg.progress_every) == 0 or (b + 1) == n_batches:
            print(f"[encode] batch {b+1}/{n_batches} routes {i1}/{n}", flush=True)

    emb = np.concatenate(emb_list, axis=0)  # (N,D), normalized
    od_arr = np.asarray(all_od, dtype=np.int64).reshape(n, 2)

    # Group by OD.
    od_to_indices: Dict[Tuple[int, int], List[int]] = {}
    for i, od in enumerate(all_od):
        od_to_indices.setdefault((int(od[0]), int(od[1])), []).append(int(i))
    od_keys_all = list(od_to_indices.keys())
    od_keys_kept = [k for k in od_keys_all if len(od_to_indices[k]) >= int(cfg.min_routes_per_od)]

    within_od_vals: List[float] = []
    within_corr_vals: List[float] = []
    per_od_rows: List[Dict[str, Any]] = []
    n_corridors_total = 0
    n_corridors_kept = 0

    for k in od_keys_kept:
        idxs = od_to_indices[k]
        m = int(len(idxs))
        if m < 2:
            continue
        e = emb[np.asarray(idxs, dtype=np.int64)]  # (m,d), normalized
        sim_all = np.clip(e @ e.T, -1.0, 1.0)
        iu = np.triu_indices(m, k=1)
        vals_all = sim_all[iu].astype(np.float64, copy=False)
        if vals_all.size > 0:
            within_od_vals.extend(vals_all.tolist())

        local_sets = [all_way_sets[i] for i in idxs]
        labels = _cluster_by_jaccard_threshold(way_sets=local_sets, dist_thr=float(cfg.jaccard_dist_thr))
        uniq, cnt = np.unique(labels, return_counts=True)
        n_corridors_total += int(uniq.size)

        row: Dict[str, Any] = {
            "start_way": int(k[0]),
            "dest_way": int(k[1]),
            "n_routes": int(m),
            "n_pairs_within_od": int(vals_all.size),
            "within_od_cos_mean": float(np.mean(vals_all)) if vals_all.size > 0 else float("nan"),
            "n_corridors": int(uniq.size),
            "corridors": [],
        }

        for lab, sz in zip(uniq.tolist(), cnt.tolist()):
            cid = int(lab)
            csz = int(sz)
            idx_loc = [i for i, x in enumerate(labels.tolist()) if int(x) == cid]
            if csz < int(cfg.min_routes_per_corridor):
                if bool(cfg.save_per_od):
                    row["corridors"].append(
                        {
                            "corridor_id": cid,
                            "n_routes": csz,
                            "n_pairs": 0,
                            "cos_mean": float("nan"),
                            "kept": False,
                        }
                    )
                continue
            n_corridors_kept += 1
            e_c = e[np.asarray(idx_loc, dtype=np.int64)]
            sim_c = np.clip(e_c @ e_c.T, -1.0, 1.0)
            iu_c = np.triu_indices(csz, k=1)
            vals_c = sim_c[iu_c].astype(np.float64, copy=False)
            if vals_c.size > 0:
                within_corr_vals.extend(vals_c.tolist())
            if bool(cfg.save_per_od):
                row["corridors"].append(
                    {
                        "corridor_id": cid,
                        "n_routes": csz,
                        "n_pairs": int(vals_c.size),
                        "cos_mean": float(np.mean(vals_c)) if vals_c.size > 0 else float("nan"),
                        "cos_p50": float(np.percentile(vals_c, 50)) if vals_c.size > 0 else float("nan"),
                        "kept": True,
                    }
                )

        if bool(cfg.save_per_od):
            per_od_rows.append(row)

    # Cross-OD baseline (same sampling count as within-corridor pairs when possible).
    n_within_corr_pairs = int(len(within_corr_vals))
    target_cross = int(min(int(cfg.max_cross_pairs), max(1, n_within_corr_pairs)))
    rng = np.random.default_rng(int(cfg.seed) + 991)
    cross_vals: List[float] = []
    trials = 0
    max_trials = int(max(20000, target_cross * 40))
    while len(cross_vals) < target_cross and trials < max_trials:
        need = int(target_cross - len(cross_vals))
        chunk = int(min(max(need * 3, 2048), 65536))
        i = rng.integers(0, n, size=chunk, endpoint=False)
        j = rng.integers(0, n, size=chunk, endpoint=False)
        trials += int(chunk)
        valid = (i != j) & ((od_arr[i, 0] != od_arr[j, 0]) | (od_arr[i, 1] != od_arr[j, 1]))
        if not np.any(valid):
            continue
        iv = i[valid]
        jv = j[valid]
        if iv.size > need:
            iv = iv[:need]
            jv = jv[:need]
        vals = np.sum(emb[iv] * emb[jv], axis=1).astype(np.float64, copy=False)
        cross_vals.extend(vals.tolist())

    within_od_stats = _summary_stats(within_od_vals)
    within_corr_stats = _summary_stats(within_corr_vals)
    cross_stats = _summary_stats(cross_vals)

    out: Dict[str, Any] = {
        "ok": True,
        "task": "way_casd_corridor_cluster_z_similarity",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": {
            "seed": int(cfg.seed),
            "device": str(cfg.device),
            "n_routes": int(cfg.n_routes),
            "min_hops": int(cfg.min_hops),
            "max_way_len": int(cfg.max_way_len),
            "split_json": cfg.split_json,
            "split_part": cfg.split_part,
            "min_routes_per_od": int(cfg.min_routes_per_od),
            "min_routes_per_corridor": int(cfg.min_routes_per_corridor),
            "jaccard_dist_thr": float(cfg.jaccard_dist_thr),
            "encode_batch_size": int(cfg.encode_batch_size),
            "max_cross_pairs": int(cfg.max_cross_pairs),
            "save_per_od": bool(cfg.save_per_od),
        },
        "inputs": {
            "way_routes_npz": str(Path(args.way_routes_npz)),
            "way_graph_npz": str(Path(args.way_graph_npz)),
            "way_features_npz": str(Path(args.way_features_npz)),
            "ae_ckpt": str(Path(args.ae_ckpt)),
        },
        "ckpt_strict_load_ok": bool(strict_ok),
        "summary": {
            "n_routes_eval": int(n),
            "n_od_groups_all": int(len(od_keys_all)),
            "n_od_groups_kept": int(len(od_keys_kept)),
            "n_corridors_total": int(n_corridors_total),
            "n_corridors_kept": int(n_corridors_kept),
            "within_od_cos": within_od_stats,
            "within_corridor_cos": within_corr_stats,
            "cross_od_cos": cross_stats,
            "within_corridor_minus_within_od_mean": (
                float(within_corr_stats["mean"] - within_od_stats["mean"])
                if np.isfinite(within_corr_stats["mean"]) and np.isfinite(within_od_stats["mean"])
                else float("nan")
            ),
            "within_corridor_minus_cross_mean": (
                float(within_corr_stats["mean"] - cross_stats["mean"])
                if np.isfinite(within_corr_stats["mean"]) and np.isfinite(cross_stats["mean"])
                else float("nan")
            ),
            "cross_sampling": {
                "target_pairs": int(target_cross),
                "obtained_pairs": int(len(cross_vals)),
                "trials": int(trials),
            },
        },
    }
    if bool(cfg.save_per_od):
        out["per_od"] = per_od_rows

    op = Path(args.out_json)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {op}", flush=True)
    s = out["summary"]
    print(
        "Corridor-cluster z-sim | "
        f"within_corr_mean={float(s['within_corridor_cos']['mean']):.4f} "
        f"(p50={float(s['within_corridor_cos']['p50']):.4f}, n={int(s['within_corridor_cos']['n'])}) | "
        f"within_od_mean={float(s['within_od_cos']['mean']):.4f} "
        f"(p50={float(s['within_od_cos']['p50']):.4f}, n={int(s['within_od_cos']['n'])}) | "
        f"cross_mean={float(s['cross_od_cos']['mean']):.4f} "
        f"(p50={float(s['cross_od_cos']['p50']):.4f}, n={int(s['cross_od_cos']['n'])})",
        flush=True,
    )


if __name__ == "__main__":
    main()
