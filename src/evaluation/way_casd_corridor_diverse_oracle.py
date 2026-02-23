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
from src.evaluation.way_casd_teacher_forcing_coverage import _analyze_per_od, _build_ae, _set_seed

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


def _alloc_counts(k: int, n_corr: int) -> List[int]:
    if n_corr <= 0 or k <= 0:
        return []
    base = int(k // n_corr)
    rem = int(k % n_corr)
    out = [base + (1 if i < rem else 0) for i in range(n_corr)]
    # If K < n_corr, first K corridors get 1, others 0.
    return out


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
    decode_max_candidates: int
    decode_candidate_policy: str
    decode_include_dest_if_successor: bool
    jaccard_dist_thr: float
    min_routes_per_od: int
    min_corridors_per_od: int
    n_samples_per_route: int
    centroid_noise_std: float
    jaccard_threshold: float
    k_per_od: int
    progress_every: int
    dump_samples: bool


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Corridor-diverse oracle ceiling: decode with OD corridor centroids (+noise).")
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

    p.add_argument("--decode_max_candidates", type=int, default=32)
    p.add_argument("--decode_candidate_policy", type=str, default="first", choices=["first", "destdist"])
    p.add_argument("--decode_include_dest_if_successor", action="store_true")

    p.add_argument("--jaccard_dist_thr", type=float, default=0.3)
    p.add_argument("--min_routes_per_od", type=int, default=3)
    p.add_argument("--min_corridors_per_od", type=int, default=2)
    p.add_argument("--n_samples_per_route", type=int, default=16)
    p.add_argument("--centroid_noise_std", type=float, default=0.05)

    p.add_argument("--jaccard_threshold", type=float, default=0.5)
    p.add_argument("--k_per_od", type=int, default=16)
    p.add_argument("--progress_every", type=int, default=100)
    p.add_argument("--dump_samples", action="store_true")
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
        decode_max_candidates=int(args.decode_max_candidates),
        decode_candidate_policy=str(args.decode_candidate_policy),
        decode_include_dest_if_successor=bool(args.decode_include_dest_if_successor),
        jaccard_dist_thr=float(args.jaccard_dist_thr),
        min_routes_per_od=max(2, int(args.min_routes_per_od)),
        min_corridors_per_od=max(1, int(args.min_corridors_per_od)),
        n_samples_per_route=max(1, int(args.n_samples_per_route)),
        centroid_noise_std=max(0.0, float(args.centroid_noise_std)),
        jaccard_threshold=float(args.jaccard_threshold),
        k_per_od=int(args.k_per_od),
        progress_every=max(1, int(args.progress_every)),
        dump_samples=bool(args.dump_samples),
    )
    keep_samples_for_export = bool(cfg.dump_samples) or (args.out_per_route_json is not None)

    _set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    routes = load_way_routes_npz(Path(args.way_routes_npz))
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

    # Collect routes.
    all_rids: List[int] = []
    all_city: List[int] = []
    all_od: List[Tuple[int, int]] = []
    all_seq: List[List[int]] = []
    all_way_sets: List[set[int]] = []
    all_sw: List[int] = []
    all_dw: List[int] = []
    all_start_pos: List[np.ndarray] = []
    all_dest_pos: List[np.ndarray] = []
    all_hour: List[int] = []
    all_dow: List[int] = []

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

            all_rids.append(rid_i)
            all_city.append(int(city))
            all_od.append((sw, dw))
            all_seq.append(gt_ids)
            all_way_sets.append(set(gt_ids))
            all_sw.append(sw)
            all_dw.append(dw)
            all_start_pos.append(routes.start_pos[rid_i].astype(np.float32).reshape(2))
            all_dest_pos.append(routes.dest_pos[rid_i].astype(np.float32).reshape(2))
            all_hour.append(hour)
            all_dow.append(dow)

    n_all = int(len(all_rids))
    if n_all <= 0:
        raise SystemExit("[FATAL] no routes selected after filtering.")

    # Encode all z_true once.
    z_true_list: List[torch.Tensor] = []
    for i, seq in enumerate(all_seq):
        L = int(len(seq))
        way_pad = np.full((1, L), -1, dtype=np.int64)
        way_pad[0, :L] = np.asarray(seq, dtype=np.int64)
        way_pad_t = torch.as_tensor(way_pad, dtype=torch.long, device=device)
        z, _ = ae.encode(way_pad_t)
        z_true_list.append(z.detach())
        if (i + 1) % 500 == 0 or (i + 1) == n_all:
            print(f"[encode] {i+1}/{n_all}", flush=True)
    z_true = torch.cat(z_true_list, dim=0)

    # Build OD corridors and centroid bank.
    od_to_idxs: Dict[Tuple[int, int], List[int]] = {}
    for i, od in enumerate(all_od):
        od_to_idxs.setdefault((int(od[0]), int(od[1])), []).append(i)

    od_centroids: Dict[Tuple[int, int], List[torch.Tensor]] = {}
    od_corr_sizes: Dict[Tuple[int, int], List[int]] = {}
    n_od_kept = 0
    for od, idxs in od_to_idxs.items():
        if len(idxs) < int(cfg.min_routes_per_od):
            continue
        labels = _cluster_by_jaccard_threshold(
            way_sets=[all_way_sets[i] for i in idxs],
            dist_thr=float(cfg.jaccard_dist_thr),
        )
        uniq = np.unique(labels)
        centroids: List[torch.Tensor] = []
        csizes: List[int] = []
        for lab in uniq.tolist():
            mids = [idxs[j] for j, lb in enumerate(labels.tolist()) if int(lb) == int(lab)]
            if len(mids) <= 0:
                continue
            zc = torch.mean(torch.stack([z_true[m] for m in mids], dim=0), dim=0)
            centroids.append(zc)
            csizes.append(int(len(mids)))
        if len(centroids) >= int(cfg.min_corridors_per_od):
            od_centroids[od] = centroids
            od_corr_sizes[od] = csizes
            n_od_kept += 1

    eval_indices = [i for i, od in enumerate(all_od) if (int(od[0]), int(od[1])) in od_centroids]
    if len(eval_indices) <= 0:
        raise SystemExit("[FATAL] no routes left after OD/corridor filtering.")

    print(
        f"[probe] routes_total={n_all} routes_eval={len(eval_indices)} od_all={len(od_to_idxs)} od_kept={n_od_kept}",
        flush=True,
    )

    pred_success_by_od: Dict[Tuple[int, int], List[List[int]]] = {}
    gt_by_od: Dict[Tuple[int, int], List[List[int]]] = {}

    n_total_samples = 0
    n_success_samples = 0
    n_route_any_success = 0
    per_route: List[Dict[str, Any]] = []

    for rr, gi in enumerate(eval_indices, start=1):
        od = (int(all_sw[gi]), int(all_dw[gi]))
        gt_ids = all_seq[gi]
        gt_by_od.setdefault(od, []).append(gt_ids)

        cents = od_centroids[od]
        C = int(len(cents))
        counts = _alloc_counts(int(cfg.n_samples_per_route), C)
        z_rows: List[torch.Tensor] = []
        cid_rows: List[int] = []
        for cid, (zc, nk) in enumerate(zip(cents, counts)):
            for _ in range(int(nk)):
                z = zc.clone()
                if float(cfg.centroid_noise_std) > 0.0:
                    z = z + (float(cfg.centroid_noise_std) * torch.randn_like(z))
                z_rows.append(z)
                cid_rows.append(int(cid))

        if len(z_rows) <= 0:
            continue

        z_tok = torch.stack(z_rows, dim=0)
        K = int(z_tok.shape[0])

        route_cond = {
            "start_pos": torch.as_tensor(np.repeat(all_start_pos[gi][None, :], K, axis=0), dtype=torch.float32, device=device),
            "dest_pos": torch.as_tensor(np.repeat(all_dest_pos[gi][None, :], K, axis=0), dtype=torch.float32, device=device),
            "hour": torch.as_tensor(np.repeat(np.asarray([all_hour[gi]], dtype=np.int64), K, axis=0), dtype=torch.long, device=device),
            "dow": torch.as_tensor(np.repeat(np.asarray([all_dow[gi]], dtype=np.int64), K, axis=0), dtype=torch.long, device=device),
            "route_city": torch.as_tensor(np.repeat(np.asarray([all_city[gi]], dtype=np.int64), K, axis=0), dtype=torch.long, device=device),
        }
        sw_t = torch.as_tensor(np.repeat(np.asarray([all_sw[gi]], dtype=np.int64), K, axis=0), dtype=torch.long, device=device)
        dw_t = torch.as_tensor(np.repeat(np.asarray([all_dw[gi]], dtype=np.int64), K, axis=0), dtype=torch.long, device=device)

        preds = ae.decoder.greedy_decode_batched(
            way_embedder=ae.way_enc,
            latent_tokens=z_tok,
            route_cond=route_cond,
            start_way=sw_t,
            dest_way=dw_t,
            max_len=int(cfg.max_decode_len),
            max_candidates=int(max_candidates),
            candidate_policy=str(cfg.decode_candidate_policy),
            include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
        )

        sample_success = []
        sample_rows: List[Dict[str, Any]] = []
        succ_preds: List[List[int]] = []
        for si, pred in enumerate(preds):
            pred_ids = [int(x) for x in pred]
            ok = bool(len(pred_ids) > 0 and int(pred_ids[-1]) == int(all_dw[gi]))
            sample_success.append(ok)
            if ok:
                succ_preds.append(pred_ids)
            sample_rows.append(
                {
                    "sample_idx": int(si),
                    "corridor_id": int(cid_rows[si]),
                    "success": bool(ok),
                    "pred_way_ids": pred_ids,
                    "jaccard": float(_seq_jaccard(gt_ids, pred_ids)),
                }
            )

        if len(succ_preds) > 0:
            pred_success_by_od.setdefault(od, []).extend(succ_preds)

        n_total_samples += int(len(sample_rows))
        n_success_samples += int(sum(1 for x in sample_success if x))
        any_succ = bool(any(sample_success))
        if any_succ:
            n_route_any_success += 1

        rec: Dict[str, Any] = {
            "route_id": int(all_rids[gi]),
            "city": int(all_city[gi]),
            "start_way": int(all_sw[gi]),
            "dest_way": int(all_dw[gi]),
            "gt_way_ids": [int(x) for x in gt_ids],
            "n_corridors": int(C),
            "sample_success_rate": float(np.mean(np.asarray(sample_success, dtype=np.float64))) if sample_success else 0.0,
            "route_any_success": bool(any_succ),
        }
        if bool(keep_samples_for_export):
            rec["samples"] = sample_rows
        per_route.append(rec)

        if rr % int(cfg.progress_every) == 0 or rr == len(eval_indices):
            print(
                f"[decode] {rr}/{len(eval_indices)} routes | sample_sr={(float(n_success_samples)/max(1,n_total_samples)):.4f}",
                flush=True,
            )

    od_stats = _analyze_per_od(
        gt_by_od=gt_by_od,
        pred_success_by_od=pred_success_by_od,
        min_routes_per_od=int(cfg.min_routes_per_od),
        jaccard_threshold=float(cfg.jaccard_threshold),
        k_per_od=int(cfg.k_per_od),
    )

    out = {
        "ok": True,
        "task": "way_casd_corridor_diverse_oracle",
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
            "decode_max_candidates": int(cfg.decode_max_candidates),
            "decode_candidate_policy": str(cfg.decode_candidate_policy),
            "decode_include_dest_if_successor": bool(cfg.decode_include_dest_if_successor),
            "jaccard_dist_thr": float(cfg.jaccard_dist_thr),
            "min_routes_per_od": int(cfg.min_routes_per_od),
            "min_corridors_per_od": int(cfg.min_corridors_per_od),
            "n_samples_per_route": int(cfg.n_samples_per_route),
            "centroid_noise_std": float(cfg.centroid_noise_std),
            "jaccard_threshold": float(cfg.jaccard_threshold),
            "k_per_od": int(cfg.k_per_od),
            "dump_samples": bool(cfg.dump_samples),
        },
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_graph_npz": str(args.way_graph_npz),
            "way_features_npz": str(args.way_features_npz),
            "ae_ckpt": str(args.ae_ckpt),
        },
        "ckpt_strict_load_ok": bool(strict_ok),
        "summary": {
            "n_routes_selected_total": int(n_all),
            "n_routes_eval": int(len(eval_indices)),
            "n_od_groups_all": int(len(od_to_idxs)),
            "n_od_groups_eval": int(n_od_kept),
            "n_samples_total": int(n_total_samples),
            "n_samples_success": int(n_success_samples),
            "sample_arrival_rate": float(n_success_samples / max(1, n_total_samples)),
            "route_any_success_rate": float(n_route_any_success / max(1, len(eval_indices))),
            **od_stats,
            "corridors_per_od": _summary_stats([len(v) for v in od_centroids.values()]),
        },
    }

    if args.out_per_route_json is not None:
        # Flat format compatible with od_coverage_diversity_eval (decode=greedy)
        flat: List[Dict[str, Any]] = []
        fid = 0
        for rec in per_route:
            gt_ids = [int(x) for x in rec["gt_way_ids"]]
            samples = rec.get("samples", []) if isinstance(rec.get("samples", None), list) else []
            for s in samples:
                pred_ids = [int(x) for x in s.get("pred_way_ids", [])]
                ok = bool(s.get("success", False))
                flat.append(
                    {
                        "route_id": int(fid),
                        "city": int(rec["city"]),
                        "start_way": int(rec["start_way"]),
                        "dest_way": int(rec["dest_way"]),
                        "gt_way_ids": gt_ids,
                        "greedy": {
                            "success": bool(ok),
                            "pred_way_ids": pred_ids,
                        },
                    }
                )
                fid += 1
        Path(args.out_per_route_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_per_route_json).write_text(json.dumps({"per_route": flat}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {out_path}", flush=True)

    if args.out_per_route_json is not None:
        print(f"[OK] saved flat per-route: {args.out_per_route_json}", flush=True)

    s = out["summary"]
    print(
        "Corridor-diverse oracle | "
        f"sample_arrival={float(s['sample_arrival_rate']):.4f} | "
        f"route_any_success={float(s['route_any_success_rate']):.4f} | "
        f"coverage_mean={float(s['gt_coverage_at_k']['mean']):.4f} | "
        f"div_mean={float(s['self_diversity_at_k']['mean']):.4f} | "
        f"n_od={int(s['n_od_groups_kept'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
