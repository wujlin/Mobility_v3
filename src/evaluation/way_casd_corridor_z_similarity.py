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


def _pct(values: Sequence[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size <= 0:
        return float("nan")
    return float(np.percentile(arr, q))


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
    encode_batch_size: int
    max_cross_pairs: int
    progress_every: int
    save_per_od: bool


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(
        description="Probe AE corridor separability: compare z_enc cosine similarity within same OD vs across OD."
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
        encode_batch_size=max(1, int(args.encode_batch_size)),
        max_cross_pairs=max(1, int(args.max_cross_pairs)),
        progress_every=max(1, int(args.progress_every)),
        save_per_od=bool(args.save_per_od),
    )

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
    for city, ids in picks.items():
        for rid in ids.tolist():
            rid_i = int(rid)
            L = int(routes.way_seq_len[rid_i])
            s = int(routes.way_seq_ptr[rid_i])
            gt = routes.way_seq_idx[s : s + L].astype(np.int64, copy=False).tolist()
            gt_ids = [int(x) for x in gt]
            if len(gt_ids) <= 1:
                continue
            sw = int(routes.start_way[rid_i])
            dw = int(routes.dest_way[rid_i])
            all_rids.append(int(rid_i))
            all_city.append(int(city))
            all_od.append((int(sw), int(dw)))
            all_seq.append(gt_ids)

    N = int(len(all_rids))
    if N <= 0:
        raise SystemExit("[FATAL] no routes selected after filtering.")
    print(f"[encode] selected routes={N}", flush=True)

    # Encode in batches, flatten z_enc -> vector and L2 normalize for cosine.
    emb_list: List[np.ndarray] = []
    bs = int(cfg.encode_batch_size)
    n_batches = int((N + bs - 1) // bs)
    for b in range(n_batches):
        i0 = int(b * bs)
        i1 = int(min(N, (b + 1) * bs))
        batch_seqs = all_seq[i0:i1]
        maxL = int(max(len(x) for x in batch_seqs))
        way_pad = np.full((len(batch_seqs), maxL), -1, dtype=np.int64)
        for i, seq in enumerate(batch_seqs):
            way_pad[i, : len(seq)] = np.asarray(seq, dtype=np.int64)
        way_pad_t = torch.as_tensor(way_pad, dtype=torch.long, device=device)
        z, _ = ae.encode(way_pad_t)  # (B,n_lat,d)
        zflat = z.reshape(z.shape[0], -1).float()
        zflat = zflat / (torch.linalg.norm(zflat, dim=-1, keepdim=True) + 1e-8)
        emb_list.append(zflat.detach().cpu().numpy().astype(np.float32, copy=False))
        if (b + 1) % int(cfg.progress_every) == 0 or (b + 1) == n_batches:
            print(f"[encode] batch {b+1}/{n_batches} routes {i1}/{N}", flush=True)

    emb = np.concatenate(emb_list, axis=0)  # (N, D), normalized
    od_arr = np.asarray(all_od, dtype=np.int64).reshape(N, 2)

    # Group by OD
    od_to_indices: Dict[Tuple[int, int], List[int]] = {}
    for i, od in enumerate(all_od):
        od_to_indices.setdefault((int(od[0]), int(od[1])), []).append(int(i))
    od_keys_all = list(od_to_indices.keys())
    od_keys_kept = [k for k in od_keys_all if len(od_to_indices[k]) >= int(cfg.min_routes_per_od)]

    within_vals: List[float] = []
    per_od_rows: List[Dict[str, Any]] = []
    for k in od_keys_kept:
        idxs = od_to_indices[k]
        m = int(len(idxs))
        if m < 2:
            continue
        e = emb[np.asarray(idxs, dtype=np.int64)]  # (m,d), normalized
        sim = np.clip(e @ e.T, -1.0, 1.0)
        iu = np.triu_indices(m, k=1)
        vals = sim[iu].astype(np.float64, copy=False)
        if vals.size <= 0:
            continue
        within_vals.extend(vals.tolist())
        if bool(cfg.save_per_od):
            per_od_rows.append(
                {
                    "start_way": int(k[0]),
                    "dest_way": int(k[1]),
                    "n_routes": int(m),
                    "n_pairs": int(vals.size),
                    "cos_mean": float(np.mean(vals)),
                    "cos_p25": _pct(vals, 25),
                    "cos_p50": _pct(vals, 50),
                    "cos_p75": _pct(vals, 75),
                }
            )

    n_within_pairs = int(len(within_vals))
    target_cross = int(min(int(cfg.max_cross_pairs), max(1, n_within_pairs)))
    rng = np.random.default_rng(int(cfg.seed) + 991)
    cross_vals: List[float] = []
    trials = 0
    max_trials = int(max(20000, target_cross * 40))
    while len(cross_vals) < target_cross and trials < max_trials:
        need = int(target_cross - len(cross_vals))
        chunk = int(min(max(need * 3, 2048), 65536))
        i = rng.integers(0, N, size=chunk, endpoint=False)
        j = rng.integers(0, N, size=chunk, endpoint=False)
        trials += int(chunk)
        valid = (i != j) & (
            (od_arr[i, 0] != od_arr[j, 0]) | (od_arr[i, 1] != od_arr[j, 1])
        )
        if not np.any(valid):
            continue
        iv = i[valid]
        jv = j[valid]
        if iv.size > need:
            iv = iv[:need]
            jv = jv[:need]
        vals = np.sum(emb[iv] * emb[jv], axis=1).astype(np.float64, copy=False)
        cross_vals.extend(vals.tolist())

    within_stats = _summary_stats(within_vals)
    cross_stats = _summary_stats(cross_vals)

    out: Dict[str, Any] = {
        "ok": True,
        "task": "way_casd_corridor_z_similarity",
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
            "n_routes_eval": int(N),
            "n_od_groups_all": int(len(od_keys_all)),
            "n_od_groups_kept": int(len(od_keys_kept)),
            "within_od_cos": within_stats,
            "cross_od_cos": cross_stats,
            "within_minus_cross_mean": (
                float(within_stats["mean"] - cross_stats["mean"])
                if np.isfinite(within_stats["mean"]) and np.isfinite(cross_stats["mean"])
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
    print(f"[OK] saved: {op}")
    s = out["summary"]
    print(
        "Corridor z-sim | "
        f"within_mean={float(s['within_od_cos']['mean']):.4f} "
        f"(p50={float(s['within_od_cos']['p50']):.4f}, n={int(s['within_od_cos']['n'])}) | "
        f"cross_mean={float(s['cross_od_cos']['mean']):.4f} "
        f"(p50={float(s['cross_od_cos']['p50']):.4f}, n={int(s['cross_od_cos']['n'])}) | "
        f"delta={float(s['within_minus_cross_mean']):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

