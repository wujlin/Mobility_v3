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


def _seq_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    den = len(sa | sb)
    if den == 0:
        return 1.0
    return float(len(sa & sb) / float(den))


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

    n_lim = int(n_routes)
    if n_lim <= 0:
        n_lim = 10**12

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
        out[int(city)] = ids[: min(int(n_lim), int(ids.size))]
    return out


def _flatten_records_from_picks(routes: Any, picks: Dict[int, np.ndarray], *, tz_offset_hours: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for city, ids in picks.items():
        for rid in ids.tolist():
            rid_i = int(rid)
            l = int(routes.way_seq_len[rid_i])
            s = int(routes.way_seq_ptr[rid_i])
            seq = routes.way_seq_idx[s : s + l].astype(np.int64, copy=False).tolist()
            seq_ids = [int(x) for x in seq]
            if len(seq_ids) <= 1:
                continue
            sw = int(routes.start_way[rid_i])
            dw = int(routes.dest_way[rid_i])
            st = int(routes.start_t[rid_i])
            hour = int(_hour_from_unix(np.asarray([st], dtype=np.int64), float(tz_offset_hours))[0])
            dow = int(_dow_from_unix(np.asarray([st], dtype=np.int64), float(tz_offset_hours))[0])
            sp = routes.start_pos[rid_i].astype(np.float32, copy=False).reshape(2)
            dp = routes.dest_pos[rid_i].astype(np.float32, copy=False).reshape(2)
            rows.append(
                {
                    "route_id": int(rid_i),
                    "city": int(city),
                    "start_way": int(sw),
                    "dest_way": int(dw),
                    "start_pos": sp,
                    "dest_pos": dp,
                    "hour": int(hour),
                    "dow": int(dow),
                    "seq": seq_ids,
                    "way_set": set(seq_ids),
                }
            )
    return rows


@torch.no_grad()
def _encode_z_batched(
    *,
    ae: Any,
    records: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    progress_every: int,
    tag: str,
) -> torch.Tensor:
    n = int(len(records))
    if n <= 0:
        return torch.empty((0, int(ae.cfg.n_latent), int(ae.cfg.d_model)), dtype=torch.float16, device="cpu")
    bs = max(1, int(batch_size))
    n_batches = (n + bs - 1) // bs
    out: List[torch.Tensor] = []
    for b in range(n_batches):
        i0 = int(b * bs)
        i1 = int(min(n, (b + 1) * bs))
        seg = records[i0:i1]
        max_l = max(len(r["seq"]) for r in seg)
        way_pad = np.full((len(seg), max_l), -1, dtype=np.int64)
        for i, r in enumerate(seg):
            s = r["seq"]
            way_pad[i, : len(s)] = np.asarray(s, dtype=np.int64)
        way_pad_t = torch.as_tensor(way_pad, dtype=torch.long, device=device)
        z, _ = ae.encode(way_pad_t)
        out.append(z.detach().to(dtype=torch.float16, device="cpu"))
        if (b + 1) % max(1, int(progress_every)) == 0 or (b + 1) == n_batches:
            print(f"[encode:{tag}] batch {b+1}/{n_batches} routes {i1}/{n}", flush=True)
    return torch.cat(out, dim=0)


def _build_train_corridor_bank(
    *,
    train_records: List[Dict[str, Any]],
    z_train: torch.Tensor,
    min_routes_per_od: int,
    min_corridors_per_od: int,
    jaccard_dist_thr: float,
) -> Dict[str, Any]:
    od_to_idxs: Dict[Tuple[int, int], List[int]] = {}
    for i, r in enumerate(train_records):
        od = (int(r["start_way"]), int(r["dest_way"]))
        od_to_idxs.setdefault(od, []).append(i)

    od_keys: List[Tuple[int, int]] = []
    od_vecs: List[np.ndarray] = []
    od_centroids: List[List[torch.Tensor]] = []
    od_corr_sizes: List[List[int]] = []
    n_routes_kept = 0
    n_corr_total = 0

    for od, idxs in od_to_idxs.items():
        if len(idxs) < int(min_routes_per_od):
            continue
        local_sets = [train_records[i]["way_set"] for i in idxs]
        labels = _cluster_by_jaccard_threshold(way_sets=local_sets, dist_thr=float(jaccard_dist_thr))
        uniq = np.unique(labels)
        cents: List[torch.Tensor] = []
        csz: List[int] = []
        for lab in uniq.tolist():
            mids = [idxs[j] for j, lb in enumerate(labels.tolist()) if int(lb) == int(lab)]
            if len(mids) <= 0:
                continue
            zc = torch.mean(z_train[torch.as_tensor(mids, dtype=torch.long)], dim=0)
            cents.append(zc.to(dtype=torch.float16, device="cpu"))
            csz.append(int(len(mids)))
        if len(cents) < int(min_corridors_per_od):
            continue

        sps = np.stack([train_records[i]["start_pos"] for i in idxs], axis=0).astype(np.float32, copy=False)
        dps = np.stack([train_records[i]["dest_pos"] for i in idxs], axis=0).astype(np.float32, copy=False)
        od_vec = np.concatenate([np.mean(sps, axis=0), np.mean(dps, axis=0)], axis=0).astype(np.float32, copy=False)

        od_keys.append((int(od[0]), int(od[1])))
        od_vecs.append(od_vec)
        od_centroids.append(cents)
        od_corr_sizes.append(csz)
        n_routes_kept += int(len(idxs))
        n_corr_total += int(len(cents))

    if len(od_keys) <= 0:
        raise SystemExit("[FATAL] no train OD kept for corridor bank.")

    return {
        "od_keys": od_keys,
        "od_vecs": np.stack(od_vecs, axis=0).astype(np.float32, copy=False),  # (Nod,4)
        "od_centroids": od_centroids,  # list[list[tensor(n_lat,d)]]
        "od_corr_sizes": od_corr_sizes,
        "n_train_od_all": int(len(od_to_idxs)),
        "n_train_od_kept": int(len(od_keys)),
        "n_train_routes_kept": int(n_routes_kept),
        "n_corridors_total": int(n_corr_total),
    }


def _retrieve_topm_od(*, q_vec: np.ndarray, bank_vec: np.ndarray, top_m: int) -> np.ndarray:
    # q_vec: (4,), bank_vec: (N,4)
    diff = bank_vec - q_vec.reshape(1, 4)
    d2 = np.sum(diff * diff, axis=1)
    m = min(int(top_m), int(bank_vec.shape[0]))
    if m <= 0:
        return np.zeros((0,), dtype=np.int64)
    idx = np.argpartition(d2, kth=m - 1)[:m]
    # sort for deterministic output
    idx = idx[np.argsort(d2[idx])]
    return idx.astype(np.int64, copy=False)


def _select_centroids_diverse(*, pool: List[torch.Tensor], k: int, rng: np.random.Generator) -> List[torch.Tensor]:
    n = int(len(pool))
    if n <= 0 or k <= 0:
        return []
    if n <= k:
        out = [pool[i] for i in range(n)]
        while len(out) < k:
            out.append(pool[int(rng.integers(0, n))])
        return out

    # Greedy farthest-point sampling in cosine space on flattened, normalized vectors.
    x = torch.stack(pool, dim=0).reshape(n, -1).float()
    x = x / (torch.linalg.norm(x, dim=1, keepdim=True) + 1e-8)
    sel: List[int] = [int(rng.integers(0, n))]
    # max cosine diversity == min cosine similarity
    sim_to_sel = (x @ x[sel[0] : sel[0] + 1].T).squeeze(1)  # (n,)
    for _ in range(1, int(k)):
        # pick point minimizing max similarity to selected set
        cand = int(torch.argmin(sim_to_sel).item())
        sel.append(cand)
        sim_new = (x @ x[cand : cand + 1].T).squeeze(1)
        sim_to_sel = torch.maximum(sim_to_sel, sim_new)
    return [pool[i] for i in sel[: int(k)]]


@dataclass(frozen=True)
class Cfg:
    seed: int
    device: str
    tz_offset_hours: float
    train_n_routes: int
    test_n_routes: int
    min_hops: int
    max_way_len: int
    max_decode_len: int
    split_json: str
    train_split_part: str
    test_split_part: str
    encode_batch_size: int
    encode_log_every: int
    decode_route_batch: int
    decode_max_candidates: int
    decode_candidate_policy: str
    decode_include_dest_if_successor: bool
    min_routes_per_od: int
    min_corridors_per_od: int
    jaccard_dist_thr: float
    retrieval_top_m_od: int
    n_samples_per_route: int
    centroid_noise_std: float
    jaccard_threshold: float
    k_per_od: int
    progress_every: int
    dump_samples: bool


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Corridor retrieval decode: train OD corridor centroid bank + test OD retrieval.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_graph_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--ae_ckpt", type=Path, required=True)
    p.add_argument("--split_json", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_per_route_json", type=Path, default=None)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)

    p.add_argument("--train_split_part", choices=["train", "val", "test"], default="train")
    p.add_argument("--test_split_part", choices=["train", "val", "test"], default="test")
    p.add_argument("--train_n_routes", type=int, default=0, help="Per city. <=0 means all available.")
    p.add_argument("--test_n_routes", type=int, default=5000, help="Per city.")
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--max_decode_len", type=int, default=160)

    p.add_argument("--encode_batch_size", type=int, default=512)
    p.add_argument("--encode_log_every", type=int, default=20)
    p.add_argument("--decode_route_batch", type=int, default=16, help="Number of test routes decoded together (each with K samples).")
    p.add_argument("--decode_max_candidates", type=int, default=32)
    p.add_argument("--decode_candidate_policy", choices=["first", "destdist"], default="first")
    p.add_argument("--decode_include_dest_if_successor", action="store_true")

    p.add_argument("--min_routes_per_od", type=int, default=3)
    p.add_argument("--min_corridors_per_od", type=int, default=2)
    p.add_argument("--jaccard_dist_thr", type=float, default=0.3)
    p.add_argument("--retrieval_top_m_od", type=int, default=10)
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
        train_n_routes=int(args.train_n_routes),
        test_n_routes=int(args.test_n_routes),
        min_hops=int(args.min_hops),
        max_way_len=int(args.max_way_len),
        max_decode_len=int(args.max_decode_len),
        split_json=str(args.split_json),
        train_split_part=str(args.train_split_part),
        test_split_part=str(args.test_split_part),
        encode_batch_size=max(1, int(args.encode_batch_size)),
        encode_log_every=max(1, int(args.encode_log_every)),
        decode_route_batch=max(1, int(args.decode_route_batch)),
        decode_max_candidates=int(args.decode_max_candidates),
        decode_candidate_policy=str(args.decode_candidate_policy),
        decode_include_dest_if_successor=bool(args.decode_include_dest_if_successor),
        min_routes_per_od=max(2, int(args.min_routes_per_od)),
        min_corridors_per_od=max(1, int(args.min_corridors_per_od)),
        jaccard_dist_thr=float(args.jaccard_dist_thr),
        retrieval_top_m_od=max(1, int(args.retrieval_top_m_od)),
        n_samples_per_route=max(1, int(args.n_samples_per_route)),
        centroid_noise_std=max(0.0, float(args.centroid_noise_std)),
        jaccard_threshold=float(args.jaccard_threshold),
        k_per_od=int(args.k_per_od),
        progress_every=max(1, int(args.progress_every)),
        dump_samples=bool(args.dump_samples),
    )

    keep_samples_for_export = bool(cfg.dump_samples) or (args.out_per_route_json is not None)
    _set_seed(cfg.seed)
    rng = np.random.default_rng(int(cfg.seed) + 20260223)
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

    # Build train corridor bank.
    train_picks = _pick_route_ids(
        routes,
        seed=int(cfg.seed),
        n_routes=int(cfg.train_n_routes),
        min_hops=int(cfg.min_hops),
        max_way_len=int(cfg.max_way_len),
        split_json=Path(cfg.split_json),
        split_part=str(cfg.train_split_part),
    )
    train_records = _flatten_records_from_picks(routes, train_picks, tz_offset_hours=float(cfg.tz_offset_hours))
    print(f"[train] selected routes={len(train_records)}", flush=True)
    z_train = _encode_z_batched(
        ae=ae,
        records=train_records,
        device=device,
        batch_size=int(cfg.encode_batch_size),
        progress_every=int(cfg.encode_log_every),
        tag="train",
    )
    bank = _build_train_corridor_bank(
        train_records=train_records,
        z_train=z_train,
        min_routes_per_od=int(cfg.min_routes_per_od),
        min_corridors_per_od=int(cfg.min_corridors_per_od),
        jaccard_dist_thr=float(cfg.jaccard_dist_thr),
    )
    print(
        f"[bank] train_od_kept={bank['n_train_od_kept']} train_routes_kept={bank['n_train_routes_kept']} "
        f"corridors={bank['n_corridors_total']}",
        flush=True,
    )

    # Build test records.
    test_picks = _pick_route_ids(
        routes,
        seed=int(cfg.seed),
        n_routes=int(cfg.test_n_routes),
        min_hops=int(cfg.min_hops),
        max_way_len=int(cfg.max_way_len),
        split_json=Path(cfg.split_json),
        split_part=str(cfg.test_split_part),
    )
    test_records = _flatten_records_from_picks(routes, test_picks, tz_offset_hours=float(cfg.tz_offset_hours))
    print(f"[test] selected routes={len(test_records)}", flush=True)
    if len(test_records) <= 0:
        raise SystemExit("[FATAL] no test routes selected.")

    od_vec_bank = np.asarray(bank["od_vecs"], dtype=np.float32)
    od_centroids = bank["od_centroids"]

    n_total_samples = 0
    n_success_samples = 0
    n_route_any_success = 0
    gt_by_od: Dict[Tuple[int, int], List[List[int]]] = {}
    pred_success_by_od: Dict[Tuple[int, int], List[List[int]]] = {}
    per_route: List[Dict[str, Any]] = []
    pool_size_list: List[int] = []
    topm_list: List[int] = []

    Rb = int(cfg.decode_route_batch)
    K = int(cfg.n_samples_per_route)
    n_test = int(len(test_records))
    n_chunks = (n_test + Rb - 1) // Rb

    for cb in range(n_chunks):
        i0 = int(cb * Rb)
        i1 = int(min(n_test, (cb + 1) * Rb))
        chunk = test_records[i0:i1]

        all_z_rows: List[torch.Tensor] = []
        all_sw: List[int] = []
        all_dw: List[int] = []
        all_city: List[int] = []
        all_hour: List[int] = []
        all_dow: List[int] = []
        all_sp: List[np.ndarray] = []
        all_dp: List[np.ndarray] = []
        route_meta: List[Dict[str, Any]] = []
        route_sample_cids: List[List[int]] = []

        for r in chunk:
            q_vec = np.concatenate([r["start_pos"], r["dest_pos"]], axis=0).astype(np.float32, copy=False)
            topm = _retrieve_topm_od(q_vec=q_vec, bank_vec=od_vec_bank, top_m=int(cfg.retrieval_top_m_od))
            topm_list.append(int(topm.size))

            pool: List[torch.Tensor] = []
            cid_tags: List[int] = []
            for oi in topm.tolist():
                cents = od_centroids[int(oi)]
                for cj, zc in enumerate(cents):
                    pool.append(zc)
                    # tag as packed (od_index*1000 + local_corr)
                    cid_tags.append(int(oi) * 1000 + int(cj))

            pool_size_list.append(int(len(pool)))
            if len(pool) <= 0:
                # fallback: skip route (keeps semantics clean)
                route_meta.append({"skip": True, "record": r, "sample_cids": []})
                route_sample_cids.append([])
                continue

            sel = _select_centroids_diverse(pool=pool, k=int(K), rng=rng)
            # Need CID mapping for selected centroids (best-effort by pointer equality fallback to first match).
            sel_cids: List[int] = []
            for zc in sel:
                found = -1
                for pi, zp in enumerate(pool):
                    if zc.data_ptr() == zp.data_ptr():
                        found = int(cid_tags[pi])
                        break
                if found < 0:
                    found = int(cid_tags[0])
                sel_cids.append(found)

            z_rows = []
            for zc in sel:
                z = zc.clone()
                if float(cfg.centroid_noise_std) > 0.0:
                    z = z + float(cfg.centroid_noise_std) * torch.randn_like(z)
                z_rows.append(z)
            z_tok = torch.stack(z_rows, dim=0).to(dtype=torch.float16, device=device)
            all_z_rows.append(z_tok)
            route_sample_cids.append(sel_cids)

            for _ in range(K):
                all_sw.append(int(r["start_way"]))
                all_dw.append(int(r["dest_way"]))
                all_city.append(int(r["city"]))
                all_hour.append(int(r["hour"]))
                all_dow.append(int(r["dow"]))
                all_sp.append(r["start_pos"])
                all_dp.append(r["dest_pos"])

            route_meta.append(
                {
                    "skip": False,
                    "record": r,
                    "sample_cids": sel_cids,
                    "topm_od": int(topm.size),
                    "pool_size": int(len(pool)),
                }
            )

        if len(all_z_rows) <= 0:
            continue

        z_batch = torch.cat(all_z_rows, dim=0)  # (B_eff*K,n_lat,d)
        route_cond = {
            "start_pos": torch.as_tensor(np.stack(all_sp, axis=0), dtype=torch.float32, device=device),
            "dest_pos": torch.as_tensor(np.stack(all_dp, axis=0), dtype=torch.float32, device=device),
            "hour": torch.as_tensor(np.asarray(all_hour, dtype=np.int64), dtype=torch.long, device=device),
            "dow": torch.as_tensor(np.asarray(all_dow, dtype=np.int64), dtype=torch.long, device=device),
            "route_city": torch.as_tensor(np.asarray(all_city, dtype=np.int64), dtype=torch.long, device=device),
        }
        sw_t = torch.as_tensor(np.asarray(all_sw, dtype=np.int64), dtype=torch.long, device=device)
        dw_t = torch.as_tensor(np.asarray(all_dw, dtype=np.int64), dtype=torch.long, device=device)

        preds = ae.decoder.greedy_decode_batched(
            way_embedder=ae.way_enc,
            latent_tokens=z_batch,
            route_cond=route_cond,
            start_way=sw_t,
            dest_way=dw_t,
            max_len=int(cfg.max_decode_len),
            max_candidates=int(max_candidates),
            candidate_policy=str(cfg.decode_candidate_policy),
            include_dest_if_successor=bool(cfg.decode_include_dest_if_successor),
        )

        # Unpack back to route-level.
        ptr = 0
        for rm in route_meta:
            r = rm["record"]
            if bool(rm["skip"]):
                continue
            od = (int(r["start_way"]), int(r["dest_way"]))
            gt_ids = [int(x) for x in r["seq"]]
            gt_by_od.setdefault(od, []).append(gt_ids)

            sample_rows: List[Dict[str, Any]] = []
            succ_preds: List[List[int]] = []
            sample_success: List[bool] = []
            for kk in range(K):
                pred_ids = [int(x) for x in preds[ptr]]
                ptr += 1
                ok = bool(len(pred_ids) > 0 and int(pred_ids[-1]) == int(r["dest_way"]))
                sample_success.append(ok)
                if ok:
                    succ_preds.append(pred_ids)
                sample_rows.append(
                    {
                        "sample_idx": int(kk),
                        "retrieved_corridor_tag": int(rm["sample_cids"][kk]),
                        "success": bool(ok),
                        "pred_way_ids": pred_ids,
                        "jaccard": float(_seq_jaccard(gt_ids, pred_ids)),
                    }
                )

            if len(succ_preds) > 0:
                pred_success_by_od.setdefault(od, []).extend(succ_preds)

            n_total_samples += int(K)
            n_success_samples += int(sum(1 for x in sample_success if x))
            any_succ = bool(any(sample_success))
            if any_succ:
                n_route_any_success += 1

            rec: Dict[str, Any] = {
                "route_id": int(r["route_id"]),
                "city": int(r["city"]),
                "start_way": int(r["start_way"]),
                "dest_way": int(r["dest_way"]),
                "gt_way_ids": gt_ids,
                "sample_success_rate": float(np.mean(np.asarray(sample_success, dtype=np.float64))),
                "route_any_success": bool(any_succ),
                "retrieval_topm_od": int(rm["topm_od"]),
                "retrieval_pool_size": int(rm["pool_size"]),
            }
            if bool(keep_samples_for_export):
                rec["samples"] = sample_rows
            per_route.append(rec)

        if (cb + 1) % max(1, int(cfg.progress_every)) == 0 or (cb + 1) == n_chunks:
            done = int(i1)
            print(
                f"[decode] chunk {cb+1}/{n_chunks} routes {done}/{n_test} "
                f"| sample_sr={(float(n_success_samples)/max(1,n_total_samples)):.4f}",
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
        "task": "way_casd_corridor_retrieval_decode",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": {
            "seed": int(cfg.seed),
            "device": str(cfg.device),
            "tz_offset_hours": float(cfg.tz_offset_hours),
            "train_split_part": str(cfg.train_split_part),
            "test_split_part": str(cfg.test_split_part),
            "train_n_routes": int(cfg.train_n_routes),
            "test_n_routes": int(cfg.test_n_routes),
            "min_hops": int(cfg.min_hops),
            "max_way_len": int(cfg.max_way_len),
            "max_decode_len": int(cfg.max_decode_len),
            "encode_batch_size": int(cfg.encode_batch_size),
            "decode_route_batch": int(cfg.decode_route_batch),
            "decode_max_candidates": int(cfg.decode_max_candidates),
            "decode_candidate_policy": str(cfg.decode_candidate_policy),
            "decode_include_dest_if_successor": bool(cfg.decode_include_dest_if_successor),
            "min_routes_per_od": int(cfg.min_routes_per_od),
            "min_corridors_per_od": int(cfg.min_corridors_per_od),
            "jaccard_dist_thr": float(cfg.jaccard_dist_thr),
            "retrieval_top_m_od": int(cfg.retrieval_top_m_od),
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
            "split_json": str(args.split_json),
        },
        "ckpt_strict_load_ok": bool(strict_ok),
        "bank_stats": {
            "n_train_od_all": int(bank["n_train_od_all"]),
            "n_train_od_kept": int(bank["n_train_od_kept"]),
            "n_train_routes_kept": int(bank["n_train_routes_kept"]),
            "n_corridors_total": int(bank["n_corridors_total"]),
        },
        "summary": {
            "n_test_routes_selected": int(len(test_records)),
            "n_test_routes_evaluated": int(len(per_route)),
            "n_samples_total": int(n_total_samples),
            "n_samples_success": int(n_success_samples),
            "sample_arrival_rate": float(n_success_samples / max(1, n_total_samples)),
            "route_any_success_rate": float(n_route_any_success / max(1, len(per_route))),
            "retrieval_topm_od": _summary_stats(topm_list),
            "retrieval_pool_size": _summary_stats(pool_size_list),
            **od_stats,
        },
    }

    if args.out_per_route_json is not None:
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
        op = Path(args.out_per_route_json)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps({"per_route": flat}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] saved flat per-route: {op}", flush=True)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {out_path}", flush=True)

    s = out["summary"]
    print(
        "Corridor-retrieval decode | "
        f"sample_arrival={float(s['sample_arrival_rate']):.4f} | "
        f"route_any_success={float(s['route_any_success_rate']):.4f} | "
        f"coverage_mean={float(s['gt_coverage_at_k']['mean']):.4f} | "
        f"div_mean={float(s['self_diversity_at_k']['mean']):.4f} | "
        f"n_od={int(s['n_od_groups_kept'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
