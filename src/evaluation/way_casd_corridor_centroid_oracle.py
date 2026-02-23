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


def _seq_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    den = len(sa | sb)
    if den == 0:
        return 1.0
    return float(len(sa & sb) / float(den))


def _route_len_m(seq: Sequence[int], way_len_m: np.ndarray) -> float:
    if len(seq) <= 0:
        return 0.0
    idx = np.asarray([int(x) for x in seq], dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < int(way_len_m.shape[0]))]
    if idx.size <= 0:
        return 0.0
    return float(np.sum(way_len_m[idx], dtype=np.float64))


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


def _summary_stats(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size <= 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "n": int(arr.size),
    }


@dataclass(frozen=True)
class Cfg:
    seed: int
    device: str
    tz_offset_hours: float
    n_routes: int
    min_hops: int
    max_way_len: int
    max_decode_len: int
    split_json: Optional[str]
    split_part: Optional[str]
    decode_batch_size: int
    decode_max_candidates: int
    decode_candidate_policy: str
    decode_include_dest_if_successor: bool
    jaccard_dist_thr: float
    min_routes_per_od: int
    save_per_route: bool


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


def _agg_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = int(len(rows))
    succ = [bool(r.get("success", False)) for r in rows]
    jacs = [float(r.get("jaccard", 0.0)) for r in rows]
    lenr = [float(r.get("len_ratio", float("nan"))) for r in rows]
    succ_lenr = [float(r.get("len_ratio", float("nan"))) for r in rows if bool(r.get("success", False))]
    return {
        "n": n,
        "success_rate": float(np.mean(np.asarray(succ, dtype=np.float64))) if n > 0 else 0.0,
        "jaccard_mean": float(np.mean(np.asarray(jacs, dtype=np.float64))) if n > 0 else 0.0,
        "len_ratio": _summary_stats([x for x in lenr if np.isfinite(x)]),
        "success_only_len_ratio": _summary_stats([x for x in succ_lenr if np.isfinite(x)]),
    }


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Oracle probe: decode with corridor centroid latent (K=1).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_per_route_json", type=Path, default=None)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--n_routes", type=int, default=5000)
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--max_decode_len", type=int, default=160)
    p.add_argument("--split_json", type=Path, default=None)
    p.add_argument("--split_part", choices=["train", "val", "test"], default=None)

    p.add_argument("--decode_batch_size", type=int, default=256)
    p.add_argument("--decode_max_candidates", type=int, default=32)
    p.add_argument("--decode_candidate_policy", type=str, default="first", choices=["first", "destdist"])
    p.add_argument("--decode_include_dest_if_successor", action="store_true")

    p.add_argument("--jaccard_dist_thr", type=float, default=0.3)
    p.add_argument("--min_routes_per_od", type=int, default=3)
    p.add_argument("--save_per_route", action="store_true")
    args = p.parse_args()

    cfg = Cfg(
        seed=int(args.seed),
        device=str(args.device),
        tz_offset_hours=float(args.tz_offset_hours),
        n_routes=int(args.n_routes),
        min_hops=int(args.min_hops),
        max_way_len=int(args.max_way_len),
        max_decode_len=int(args.max_decode_len),
        split_json=(str(args.split_json) if args.split_json is not None else None),
        split_part=(str(args.split_part) if args.split_part is not None else ("test" if args.split_json is not None else None)),
        decode_batch_size=max(1, int(args.decode_batch_size)),
        decode_max_candidates=int(args.decode_max_candidates),
        decode_candidate_policy=str(args.decode_candidate_policy),
        decode_include_dest_if_successor=bool(args.decode_include_dest_if_successor),
        jaccard_dist_thr=float(args.jaccard_dist_thr),
        min_routes_per_od=max(2, int(args.min_routes_per_od)),
        save_per_route=bool(args.save_per_route),
    )

    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    way_len_m = np.asarray(wf["way_len_m"], dtype=np.float64)

    ae, strict_ok = _build_ae(
        ae_ckpt=Path(args.ae_ckpt),
        way_graph_npz=Path(args.way_graph_npz),
        way_features_npz=Path(args.way_features_npz),
        device=device,
    )
    max_candidates = int(cfg.decode_max_candidates)
    if max_candidates < 0:
        max_candidates = int(ae.cfg.max_candidates)

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
    all_start_pos: List[np.ndarray] = []
    all_dest_pos: List[np.ndarray] = []
    all_hour: List[int] = []
    all_dow: List[int] = []
    all_sw: List[int] = []
    all_dw: List[int] = []

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
            st = int(routes.start_t[rid_i])
            hour = int(_hour_from_unix(np.asarray([st], dtype=np.int64), cfg.tz_offset_hours)[0])
            dow = int(_dow_from_unix(np.asarray([st], dtype=np.int64), cfg.tz_offset_hours)[0])

            all_rids.append(int(rid_i))
            all_city.append(int(city))
            all_od.append((int(sw), int(dw)))
            all_seq.append(gt_ids)
            all_way_sets.append(set(int(x) for x in gt_ids))
            all_start_pos.append(routes.start_pos[rid_i].astype(np.float32).reshape(2))
            all_dest_pos.append(routes.dest_pos[rid_i].astype(np.float32).reshape(2))
            all_hour.append(hour)
            all_dow.append(dow)
            all_sw.append(int(sw))
            all_dw.append(int(dw))

    n = int(len(all_rids))
    if n <= 0:
        raise SystemExit("[FATAL] no routes selected after filtering.")

    print(f"[probe] selected routes={n}", flush=True)

    # Encode all GT z_enc.
    z_all: List[torch.Tensor] = []
    for i, seq in enumerate(all_seq):
        L = int(len(seq))
        way_pad = np.full((1, L), -1, dtype=np.int64)
        way_pad[0, :L] = np.asarray(seq, dtype=np.int64)
        way_pad_t = torch.as_tensor(way_pad, dtype=torch.long, device=device)
        z, _ = ae.encode(way_pad_t)
        z_all.append(z.detach())
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"[encode] {i+1}/{n}", flush=True)

    z_true = torch.cat(z_all, dim=0)  # (N,n_lat,d)

    # Build OD groups and corridor assignment.
    od_to_indices: Dict[Tuple[int, int], List[int]] = {}
    for i, od in enumerate(all_od):
        od_to_indices.setdefault((int(od[0]), int(od[1])), []).append(int(i))

    corr_label_global = np.full((n,), -1, dtype=np.int64)
    od_kept = 0
    n_corridors_total = 0
    n_corridors_kept = 0

    z_centroid = torch.empty_like(z_true)
    z_centroid_loo = torch.empty_like(z_true)

    for od, idxs in od_to_indices.items():
        m = int(len(idxs))
        if m < int(cfg.min_routes_per_od):
            # fallback: use self latent
            for gi in idxs:
                z_centroid[gi] = z_true[gi]
                z_centroid_loo[gi] = z_true[gi]
            continue

        od_kept += 1
        local_sets = [all_way_sets[i] for i in idxs]
        labels = _cluster_by_jaccard_threshold(way_sets=local_sets, dist_thr=float(cfg.jaccard_dist_thr))
        uniq, _ = np.unique(labels, return_counts=True)
        n_corridors_total += int(uniq.size)

        for lab in uniq.tolist():
            lab_i = int(lab)
            members_local = [j for j, x in enumerate(labels.tolist()) if int(x) == lab_i]
            members_global = [idxs[j] for j in members_local]
            c = int(len(members_global))
            if c <= 0:
                continue
            n_corridors_kept += 1

            z_stack = torch.stack([z_true[g] for g in members_global], dim=0)  # (c,n_lat,d)
            z_sum = torch.sum(z_stack, dim=0)
            z_mean = z_sum / float(c)
            for g in members_global:
                corr_label_global[g] = lab_i
                z_centroid[g] = z_mean
                if c >= 2:
                    z_centroid_loo[g] = (z_sum - z_true[g]) / float(c - 1)
                else:
                    z_centroid_loo[g] = z_true[g]

    # Any route not assigned (OD too small) fallback self.
    miss = np.nonzero(corr_label_global < 0)[0].tolist()
    for gi in miss:
        z_centroid[gi] = z_true[gi]
        z_centroid_loo[gi] = z_true[gi]

    start_pos_t = torch.as_tensor(np.stack(all_start_pos, axis=0), dtype=torch.float32, device=device)
    dest_pos_t = torch.as_tensor(np.stack(all_dest_pos, axis=0), dtype=torch.float32, device=device)
    hour_t = torch.as_tensor(np.asarray(all_hour, dtype=np.int64), dtype=torch.long, device=device)
    dow_t = torch.as_tensor(np.asarray(all_dow, dtype=np.int64), dtype=torch.long, device=device)
    city_t = torch.as_tensor(np.asarray(all_city, dtype=np.int64), dtype=torch.long, device=device)
    sw_t = torch.as_tensor(np.asarray(all_sw, dtype=np.int64), dtype=torch.long, device=device)
    dw_t = torch.as_tensor(np.asarray(all_dw, dtype=np.int64), dtype=torch.long, device=device)

    def _decode_with_z(name: str, z_tok: torch.Tensor) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        bs = int(cfg.decode_batch_size)
        n_batches = int((n + bs - 1) // bs)
        for b in range(n_batches):
            i0 = int(b * bs)
            i1 = int(min(n, (b + 1) * bs))
            tidx = torch.arange(i0, i1, device=device, dtype=torch.long)
            route_cond = {
                "start_pos": start_pos_t[tidx],
                "dest_pos": dest_pos_t[tidx],
                "hour": hour_t[tidx],
                "dow": dow_t[tidx],
                "route_city": city_t[tidx],
            }
            preds = ae.decoder.greedy_decode_batched(
                way_embedder=ae.way_enc,
                latent_tokens=z_tok[tidx],
                route_cond=route_cond,
                start_way=sw_t[tidx],
                dest_way=dw_t[tidx],
                max_len=int(cfg.max_decode_len),
                max_candidates=int(max_candidates),
                candidate_policy=str(cfg.decode_candidate_policy),
                include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
            )
            for j, pred in enumerate(preds):
                gi = int(i0 + j)
                pred_ids = [int(x) for x in pred]
                gt_ids = all_seq[gi]
                succ = bool(len(pred_ids) > 0 and int(pred_ids[-1]) == int(all_dw[gi]))
                jac = float(_seq_jaccard(gt_ids, pred_ids))
                gt_len_m = _route_len_m(gt_ids, way_len_m)
                pred_len_m = _route_len_m(pred_ids, way_len_m)
                len_ratio = float(pred_len_m / gt_len_m) if gt_len_m > 1e-6 else float("nan")
                rows.append(
                    {
                        "condition": str(name),
                        "route_id": int(all_rids[gi]),
                        "city": int(all_city[gi]),
                        "start_way": int(all_sw[gi]),
                        "dest_way": int(all_dw[gi]),
                        "corridor_id": int(corr_label_global[gi]),
                        "success": bool(succ),
                        "jaccard": float(jac),
                        "gt_len_m": float(gt_len_m),
                        "pred_len_m": float(pred_len_m),
                        "len_ratio": float(len_ratio),
                        "gt_hops": int(max(0, len(gt_ids) - 1)),
                        "pred_hops": int(max(0, len(pred_ids) - 1)),
                    }
                )
            if (b + 1) % 5 == 0 or (b + 1) == n_batches:
                print(f"[{name}] batch {b+1}/{n_batches} routes {i1}/{n}", flush=True)
        return rows

    rows_true = _decode_with_z("true", z_true)
    rows_cent = _decode_with_z("corridor_centroid", z_centroid)
    rows_cent_loo = _decode_with_z("corridor_centroid_loo", z_centroid_loo)

    summary = {
        "true": _agg_rows(rows_true),
        "corridor_centroid": _agg_rows(rows_cent),
        "corridor_centroid_loo": _agg_rows(rows_cent_loo),
    }

    out = {
        "ok": True,
        "task": "way_casd_corridor_centroid_oracle",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": {
            "seed": int(cfg.seed),
            "device": str(cfg.device),
            "tz_offset_hours": float(cfg.tz_offset_hours),
            "n_routes": int(cfg.n_routes),
            "min_hops": int(cfg.min_hops),
            "max_way_len": int(cfg.max_way_len),
            "max_decode_len": int(cfg.max_decode_len),
            "split_json": cfg.split_json,
            "split_part": cfg.split_part,
            "decode_batch_size": int(cfg.decode_batch_size),
            "decode_max_candidates": int(cfg.decode_max_candidates),
            "decode_candidate_policy": str(cfg.decode_candidate_policy),
            "decode_include_dest_if_successor": bool(cfg.decode_include_dest_if_successor),
            "jaccard_dist_thr": float(cfg.jaccard_dist_thr),
            "min_routes_per_od": int(cfg.min_routes_per_od),
            "save_per_route": bool(cfg.save_per_route),
        },
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
        },
        "ckpt_strict_load_ok": bool(strict_ok),
        "corridor_stats": {
            "n_routes_eval": int(n),
            "n_od_groups_all": int(len(od_to_indices)),
            "n_od_groups_kept": int(od_kept),
            "n_corridors_total": int(n_corridors_total),
            "n_corridors_kept": int(n_corridors_kept),
            "n_unassigned_routes": int(len(miss)),
        },
        "summary": summary,
        "delta": {
            "centroid_minus_true_success_rate": float(summary["corridor_centroid"]["success_rate"] - summary["true"]["success_rate"]),
            "centroid_loo_minus_true_success_rate": float(summary["corridor_centroid_loo"]["success_rate"] - summary["true"]["success_rate"]),
            "centroid_minus_true_succ_lenr_p50": float(summary["corridor_centroid"]["success_only_len_ratio"]["p50"] - summary["true"]["success_only_len_ratio"]["p50"]) if np.isfinite(summary["corridor_centroid"]["success_only_len_ratio"]["p50"]) and np.isfinite(summary["true"]["success_only_len_ratio"]["p50"]) else float("nan"),
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {out_path}", flush=True)

    if bool(cfg.save_per_route) or (args.out_per_route_json is not None):
        rows_all = rows_true + rows_cent + rows_cent_loo
        pth = Path(args.out_per_route_json) if args.out_per_route_json is not None else (out_path.parent / "corridor_centroid_oracle_per_route.json")
        pth.write_text(json.dumps(rows_all, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] saved per-route: {pth}", flush=True)

    print(
        "Corridor-centroid oracle | "
        f"true={summary['true']['success_rate']:.4f} | "
        f"cent={summary['corridor_centroid']['success_rate']:.4f} | "
        f"cent_loo={summary['corridor_centroid_loo']['success_rate']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
