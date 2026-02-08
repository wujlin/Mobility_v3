"""
Porto OD diversity scan: 在 way_routes_labeled.npz 上扫描同一 OD 区域的路径多样性。

动机：
  Detroit/Columbus 的一个根因是 "GT≈最短路" → route generation 被 trivialize。
  Porto 如果同一 OD 区域存在多条结构性不同的 corridor，则更适合验证 latent diversity。

方法（轻量、可解释、可复现）：
  1) 对 (start_pos, dest_pos) 做粗粒度 OD bin（在 grid 坐标系中）
  2) 在每个 bin 内抽样多条 route 的 way 序列
  3) 用 LCS distance 作为“走廊结构差异”的 proxy：
       dist = 1 - LCS(seq_i, seq_j) / max(len_i, len_j)
  4) 若 max_pairwise_dist >= thr，则认为该 bin 存在 multimodal corridor diversity

注意：
  - 该脚本只用于诊断，不追求最强聚类算法；优先 KISS + 可解释。
  - way_routes 的 CSR 字段在仓库中为 way_seq_ptr + way_seq_idx（旧脚本常写成 way_seq_val）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _decode_meta(meta_obj: object) -> Optional[dict]:
    if meta_obj is None:
        return None
    if isinstance(meta_obj, np.ndarray):
        if meta_obj.size != 1:
            return None
        meta_obj = meta_obj.item()
    return meta_obj if isinstance(meta_obj, dict) else None


def _grid_hw_bbox_from_meta(meta: dict) -> Optional[Tuple[int, int, float, float, float, float]]:
    grid = meta.get("grid", {}) if isinstance(meta, dict) else {}
    if not isinstance(grid, dict):
        return None
    bbox = grid.get("bbox", None)
    if not isinstance(bbox, dict):
        return None
    try:
        H = int(grid.get("H", 0))
        W = int(grid.get("W", 0))
        min_lon = float(bbox["min_lon"])
        min_lat = float(bbox["min_lat"])
        max_lon = float(bbox["max_lon"])
        max_lat = float(bbox["max_lat"])
    except Exception:
        return None
    if H <= 0 or W <= 0:
        return None
    return H, W, min_lon, min_lat, max_lon, max_lat


def load_npz(npz_path: Path) -> dict:
    d = np.load(str(npz_path), allow_pickle=True)
    return {k: np.asarray(d[k]) for k in d.files}


def lcs_len(a: List[int], b: List[int]) -> int:
    """Compute LCS length between two sequences (O(n*m) DP, capped)."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    # Cap to avoid huge DP tables
    MAX = 200
    if n > MAX:
        a = a[:MAX]
        n = MAX
    if m > MAX:
        b = b[:MAX]
        m = MAX
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def lcs_dist(a: List[int], b: List[int]) -> float:
    """LCS distance: 1 - LCS_len / max(len_a, len_b). 0=identical, 1=disjoint."""
    mx = max(len(a), len(b))
    if mx <= 0:
        return 0.0
    return 1.0 - float(lcs_len(a, b)) / float(mx)


def _compress_consecutive_int(seq: List[int]) -> List[int]:
    out: List[int] = []
    last: Optional[int] = None
    for x in seq:
        xx = int(x)
        if last is None or xx != int(last):
            out.append(xx)
            last = xx
    return out


def extract_way_seq(data: dict, route_id: int) -> List[int]:
    ptr = data["way_seq_ptr"]
    vals = data.get("way_seq_val", None)
    if vals is None:
        vals = data.get("way_seq_idx", None)
    if vals is None:
        raise KeyError("way_routes_npz missing way_seq_val/way_seq_idx (need CSR values).")
    s = int(ptr[int(route_id)])
    e = int(ptr[int(route_id) + 1])
    seq = [int(x) for x in vals[s:e].tolist()]
    return _compress_consecutive_int(seq)


def _infer_grid_bin(*, wf: dict, od_bin_deg: float) -> Optional[float]:
    meta = _decode_meta(wf.get("meta", None))
    if meta is None:
        return None
    pack = _grid_hw_bbox_from_meta(meta)
    if pack is None:
        return None
    H, W, min_lon, min_lat, max_lon, max_lat = pack
    if not (float(od_bin_deg) > 0):
        return None
    lon_span = float(max_lon - min_lon)
    lat_span = float(max_lat - min_lat)
    if lon_span <= 0 or lat_span <= 0:
        return None
    # Convert degrees -> grid units (rough, but fine for coarse binning).
    bin_x = float(od_bin_deg) * float(W) / lon_span
    bin_y = float(od_bin_deg) * float(H) / lat_span
    if not (bin_x > 0 and bin_y > 0):
        return None
    return float(0.5 * (bin_x + bin_y))


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan OD corridor diversity in way_routes_labeled.npz (Porto).")
    ap.add_argument("--way_routes_npz", type=Path, required=True)
    ap.add_argument("--way_features_npz", type=Path, required=True, help="Used to read meta.grid.* for bin sizing.")
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument(
        "--od_bin_deg",
        type=float,
        default=0.01,
        help="Coarse OD binning size in degrees; converted to grid units via meta.grid.* (ignored if --grid_bin>0).",
    )
    ap.add_argument("--grid_bin", type=float, default=0.0, help="OD bin size in grid units (y/x). If >0, overrides od_bin_deg.")
    ap.add_argument("--min_routes_per_bin", type=int, default=5)
    ap.add_argument("--min_hops", type=int, default=5)
    ap.add_argument("--max_way_len", type=int, default=160)
    ap.add_argument("--max_sigs_per_bin", type=int, default=32, help="Cap signatures per bin (random sample).")
    ap.add_argument("--lcs_sep_thr", type=float, default=0.50, help="max LCS dist >= thr => multimodal bin.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_bins", type=int, default=0, help="If >0, randomly sample this many bins for speed.")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.RandomState(int(args.seed))

    print(f"[od_diversity] Loading {args.way_routes_npz} ...", file=sys.stderr)
    data = load_npz(Path(args.way_routes_npz))
    n_routes = int(np.asarray(data["way_seq_len"]).shape[0])

    wf = load_npz(Path(args.way_features_npz))

    start_pos = np.asarray(data["start_pos"], dtype=np.float64)  # (N,2) = (y, x)
    dest_pos = np.asarray(data["dest_pos"], dtype=np.float64)
    way_seq_len = np.asarray(data["way_seq_len"], dtype=np.int64).reshape(-1)

    keep = (way_seq_len >= int(args.min_hops) + 1) & (way_seq_len <= int(args.max_way_len))
    keep_ids = np.where(keep)[0].astype(np.int64, copy=False)
    print(
        f"[od_diversity] {n_routes} total routes, {len(keep_ids)} after min_hops={int(args.min_hops)} max_len={int(args.max_way_len)}",
        file=sys.stderr,
    )

    grid_bin = float(args.grid_bin)
    if not (grid_bin > 0):
        inferred = _infer_grid_bin(wf=wf, od_bin_deg=float(args.od_bin_deg))
        grid_bin = float(inferred) if inferred is not None else 50.0
    print(f"[od_diversity] Binning with grid_bin={grid_bin:.2f} (grid units)", file=sys.stderr)

    od_bins: Dict[Tuple[int, int, int, int], List[int]] = defaultdict(list)
    for rid in keep_ids.tolist():
        sy, sx = start_pos[int(rid)]
        dy, dx = dest_pos[int(rid)]
        key = (int(sy // grid_bin), int(sx // grid_bin), int(dy // grid_bin), int(dx // grid_bin))
        od_bins[key].append(int(rid))

    valid_bins = {k: v for k, v in od_bins.items() if int(len(v)) >= int(args.min_routes_per_bin)}
    print(f"[od_diversity] {len(od_bins)} unique OD bins, {len(valid_bins)} with >= {int(args.min_routes_per_bin)} routes", file=sys.stderr)

    bin_sizes = [len(v) for v in valid_bins.values()]
    if bin_sizes:
        bs = np.asarray(bin_sizes, dtype=np.int64)
        print(
            f"[od_diversity] Routes per bin: p50={np.median(bs):.0f} p90={np.percentile(bs,90):.0f} "
            f"p99={np.percentile(bs,99):.0f} max={int(bs.max())}",
            file=sys.stderr,
        )

    bin_keys = list(valid_bins.keys())
    if int(args.sample_bins) > 0 and len(bin_keys) > int(args.sample_bins):
        rng.shuffle(bin_keys)
        bin_keys = bin_keys[: int(args.sample_bins)]
        print(f"[od_diversity] Sampled {len(bin_keys)} bins for analysis", file=sys.stderr)

    n_multimodal = 0
    n_analyzed = 0
    diversity_scores: List[float] = []
    multimodal_examples: List[dict] = []

    print(f"[od_diversity] Analyzing {len(bin_keys)} bins ...", file=sys.stderr)
    for bi, key in enumerate(bin_keys):
        if (bi + 1) % 500 == 0:
            print(f"  [{bi+1}/{len(bin_keys)}] {(time.time()-t0):.1f}s ...", file=sys.stderr)

        rids = list(valid_bins[key])
        if len(rids) > int(args.max_sigs_per_bin):
            sel = rng.choice(len(rids), size=int(args.max_sigs_per_bin), replace=False)
            rids = [rids[int(i)] for i in sel]

        seqs = [extract_way_seq(data, int(r)) for r in rids]
        n_seqs = int(len(seqs))
        if n_seqs < 2:
            continue

        dists: List[float] = []
        for i in range(n_seqs):
            for j in range(i + 1, n_seqs):
                dists.append(lcs_dist(seqs[i], seqs[j]))
        if not dists:
            continue

        n_analyzed += 1
        darr = np.asarray(dists, dtype=np.float64)
        mean_dist = float(darr.mean())
        max_dist = float(darr.max())
        diversity_scores.append(mean_dist)

        if max_dist >= float(args.lcs_sep_thr):
            n_multimodal += 1
            if len(multimodal_examples) < 20:
                idx = int(darr.argmax())
                ii = jj = 0
                cnt = 0
                for i in range(n_seqs):
                    for j in range(i + 1, n_seqs):
                        if cnt == idx:
                            ii, jj = i, j
                            break
                        cnt += 1
                    else:
                        continue
                    break
                multimodal_examples.append(
                    {
                        "od_bin": list(key),
                        "n_routes": int(len(valid_bins[key])),
                        "n_sampled": int(n_seqs),
                        "max_lcs_dist": round(float(max_dist), 3),
                        "mean_lcs_dist": round(float(mean_dist), 3),
                        "pair_hops": [int(len(seqs[ii])), int(len(seqs[jj]))],
                        "pair_route_ids": [int(rids[ii]), int(rids[jj])],
                    }
                )

    elapsed = float(time.time() - t0)
    diversity_arr = np.asarray(diversity_scores, dtype=np.float64)

    def _pct(x: np.ndarray, q: float) -> Optional[float]:
        if x.size == 0:
            return None
        return float(np.percentile(x, q))

    result = {
        "ok": True,
        "task": "porto_od_diversity_scan",
        "elapsed_s": round(elapsed, 1),
        "inputs": {"way_routes_npz": str(args.way_routes_npz), "way_features_npz": str(args.way_features_npz)},
        "cfg": {
            "od_bin_deg": float(args.od_bin_deg),
            "grid_bin": float(grid_bin),
            "min_routes_per_bin": int(args.min_routes_per_bin),
            "min_hops": int(args.min_hops),
            "max_way_len": int(args.max_way_len),
            "max_sigs_per_bin": int(args.max_sigs_per_bin),
            "lcs_sep_thr": float(args.lcs_sep_thr),
            "seed": int(args.seed),
            "sample_bins": int(args.sample_bins),
        },
        "summary": {
            "n_routes_total": int(n_routes),
            "n_routes_filtered": int(keep_ids.size),
            "n_od_bins_total": int(len(od_bins)),
            "n_od_bins_valid": int(len(valid_bins)),
            "n_od_bins_analyzed": int(n_analyzed),
            "n_multimodal": int(n_multimodal),
            "multimodal_frac": round(float(n_multimodal) / float(max(1, n_analyzed)), 4),
            "routes_per_bin": {
                "p50": int(np.median(np.asarray(bin_sizes))) if bin_sizes else None,
                "p90": int(np.percentile(np.asarray(bin_sizes), 90)) if bin_sizes else None,
                "p99": int(np.percentile(np.asarray(bin_sizes), 99)) if bin_sizes else None,
                "max": int(np.max(np.asarray(bin_sizes))) if bin_sizes else None,
            },
            "diversity_mean_lcs_dist": {
                "mean": round(float(diversity_arr.mean()), 4) if diversity_arr.size else None,
                "p25": round(float(_pct(diversity_arr, 25)), 4) if diversity_arr.size else None,
                "p50": round(float(_pct(diversity_arr, 50)), 4) if diversity_arr.size else None,
                "p75": round(float(_pct(diversity_arr, 75)), 4) if diversity_arr.size else None,
                "p90": round(float(_pct(diversity_arr, 90)), 4) if diversity_arr.size else None,
            },
        },
        "multimodal_examples": multimodal_examples[:20],
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[OK] saved: {args.out_json}", file=sys.stderr)
    print(
        f"[od_diversity] multimodal={n_multimodal}/{max(1,n_analyzed)} "
        f"({100.0*float(n_multimodal)/float(max(1,n_analyzed)):.1f}%)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

