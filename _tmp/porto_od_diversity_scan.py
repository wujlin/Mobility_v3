"""
Porto OD diversity scan: 在 way_routes_labeled.npz 上直接扫描同一 OD 区域的路径多样性。

核心问题：Porto 的高 detour 是否体现为"同一 OD 有多条结构性不同的路径"？
如果是 → 正是 Way-CASD latent diversity 的理想训练数据。
如果否 → 高 detour 只是噪声（随机巡游），不可学。

方法：
1. 按 OD 坐标 bin (od_bin_deg ≈ 0.5km) 分组
2. 同组内用 way_id 序列的 LCS 距离衡量路径差异
3. 聚类并计算 multimodal OD 比例
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_way_routes(npz_path: Path) -> dict:
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
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def lcs_dist(a: List[int], b: List[int]) -> float:
    """LCS distance: 1 - LCS_len / max(len_a, len_b). 0=identical, 1=disjoint."""
    mx = max(len(a), len(b))
    if mx == 0:
        return 0.0
    return 1.0 - lcs_len(a, b) / mx


def extract_way_seq(data: dict, route_id: int) -> List[int]:
    ptr = data["way_seq_ptr"]
    vals = data["way_seq_val"]
    s = int(ptr[route_id])
    e = int(ptr[route_id + 1])
    return [int(x) for x in vals[s:e]]


def main():
    ap = argparse.ArgumentParser(description="Scan OD diversity in way_routes_labeled.npz")
    ap.add_argument("--way_routes_npz", type=Path, required=True)
    ap.add_argument("--way_features_npz", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--od_bin_deg", type=float, default=0.01,
                    help="OD binning in degrees (~1km at 0.01 for Porto latitude)")
    ap.add_argument("--min_routes_per_bin", type=int, default=5)
    ap.add_argument("--min_hops", type=int, default=5)
    ap.add_argument("--max_way_len", type=int, default=160)
    ap.add_argument("--max_sigs_per_bin", type=int, default=32,
                    help="Cap signatures per bin (random sample)")
    ap.add_argument("--lcs_sep_thr", type=float, default=0.50,
                    help="LCS distance >= this means 'structurally different corridor'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_bins", type=int, default=0,
                    help="If >0, randomly sample this many bins for speed")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.RandomState(args.seed)

    print(f"[od_diversity] Loading {args.way_routes_npz} ...", file=sys.stderr)
    data = load_way_routes(args.way_routes_npz)
    n_routes = int(data["way_seq_len"].shape[0])

    # Load way features for coordinate conversion
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)

    # Need to convert grid coords to lat/lon for binning
    # start_pos / dest_pos are in grid coords; we need the bbox meta to convert
    # For simplicity, bin on grid coords directly (relative binning still works)
    start_pos = data["start_pos"].astype(np.float64)  # (N, 2) = (y, x)
    dest_pos = data["dest_pos"].astype(np.float64)
    way_seq_len = data["way_seq_len"].astype(np.int64)

    # Filter by hops
    keep = (way_seq_len >= args.min_hops + 1) & (way_seq_len <= args.max_way_len)
    keep_ids = np.where(keep)[0]
    print(f"[od_diversity] {n_routes} total routes, {len(keep_ids)} after min_hops={args.min_hops} max_len={args.max_way_len}", file=sys.stderr)

    # Bin OD pairs: use grid coordinates, bin_size in grid units
    # Grid coords are typically in [0, H] x [0, W] range (1024x1024 for Porto)
    # We need a bin size in grid units. od_bin_deg is conceptual; let's compute grid bin size.
    # Porto bbox: (-8.72, 41.08, -8.50, 41.25) → W=0.22 deg lon, H=0.17 deg lat, grid=1024x1024
    # So 0.01 deg ≈ 1024 * 0.01 / 0.22 ≈ 46.5 grid units (lon) or 1024*0.01/0.17 ≈ 60.2 (lat)
    # Use a fixed grid bin size for simplicity
    grid_bin = 50.0  # ~1km in Porto's grid

    print(f"[od_diversity] Binning with grid_bin={grid_bin:.0f} ...", file=sys.stderr)
    od_bins: Dict[Tuple, List[int]] = defaultdict(list)
    for rid in keep_ids:
        sy, sx = start_pos[rid]
        dy, dx = dest_pos[rid]
        key = (int(sy // grid_bin), int(sx // grid_bin),
               int(dy // grid_bin), int(dx // grid_bin))
        od_bins[key].append(int(rid))

    # Filter bins with enough routes
    valid_bins = {k: v for k, v in od_bins.items() if len(v) >= args.min_routes_per_bin}
    print(f"[od_diversity] {len(od_bins)} unique OD bins, {len(valid_bins)} with >= {args.min_routes_per_bin} routes", file=sys.stderr)

    # Route count distribution
    bin_sizes = [len(v) for v in valid_bins.values()]
    if bin_sizes:
        bs = np.array(bin_sizes)
        print(f"[od_diversity] Routes per bin: p50={np.median(bs):.0f} p90={np.percentile(bs,90):.0f} "
              f"p99={np.percentile(bs,99):.0f} max={bs.max()}", file=sys.stderr)

    # Optionally sample bins for speed
    bin_keys = list(valid_bins.keys())
    if args.sample_bins > 0 and len(bin_keys) > args.sample_bins:
        rng.shuffle(bin_keys)
        bin_keys = bin_keys[:args.sample_bins]
        print(f"[od_diversity] Sampled {len(bin_keys)} bins for analysis", file=sys.stderr)

    # For each bin, extract way_id signatures and compute pairwise LCS distances
    n_multimodal = 0
    n_analyzed = 0
    diversity_scores = []
    multimodal_examples = []

    print(f"[od_diversity] Analyzing {len(bin_keys)} bins ...", file=sys.stderr)
    for bi, key in enumerate(bin_keys):
        if (bi + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{bi+1}/{len(bin_keys)}] {elapsed:.1f}s ...", file=sys.stderr)

        rids = valid_bins[key]
        # Sample if too many
        if len(rids) > args.max_sigs_per_bin:
            sel = rng.choice(len(rids), size=args.max_sigs_per_bin, replace=False)
            rids = [rids[i] for i in sel]

        # Extract way sequences
        seqs = [extract_way_seq(data, r) for r in rids]
        n_seqs = len(seqs)

        if n_seqs < 2:
            continue

        # Compute pairwise LCS distances (upper triangle)
        dists = []
        for i in range(n_seqs):
            for j in range(i + 1, n_seqs):
                dists.append(lcs_dist(seqs[i], seqs[j]))

        if not dists:
            continue

        n_analyzed += 1
        darr = np.array(dists)
        mean_dist = float(darr.mean())
        max_dist = float(darr.max())
        diversity_scores.append(mean_dist)

        is_mm = max_dist >= args.lcs_sep_thr
        if is_mm:
            n_multimodal += 1
            if len(multimodal_examples) < 20:
                # Find the most diverse pair
                idx = int(darr.argmax())
                # Convert flat index to (i,j)
                ii, jj = 0, 0
                cnt = 0
                for i in range(n_seqs):
                    for j in range(i + 1, n_seqs):
                        if cnt == idx:
                            ii, jj = i, j
                        cnt += 1
                multimodal_examples.append({
                    "od_bin": list(key),
                    "n_routes": len(valid_bins[key]),
                    "n_sampled": n_seqs,
                    "max_lcs_dist": round(max_dist, 3),
                    "mean_lcs_dist": round(mean_dist, 3),
                    "pair_hops": [len(seqs[ii]), len(seqs[jj])],
                    "pair_route_ids": [rids[ii], rids[jj]],
                })

    diversity_arr = np.array(diversity_scores) if diversity_scores else np.array([])

    elapsed = time.time() - t0
    result = {
        "ok": True,
        "task": "porto_od_diversity_scan",
        "elapsed_s": round(elapsed, 1),
        "cfg": {
            "grid_bin": grid_bin,
            "min_routes_per_bin": args.min_routes_per_bin,
            "min_hops": args.min_hops,
            "max_way_len": args.max_way_len,
            "max_sigs_per_bin": args.max_sigs_per_bin,
            "lcs_sep_thr": args.lcs_sep_thr,
            "seed": args.seed,
            "sample_bins": args.sample_bins,
        },
        "summary": {
            "n_routes_total": n_routes,
            "n_routes_filtered": int(len(keep_ids)),
            "n_od_bins_total": len(od_bins),
            "n_od_bins_valid": len(valid_bins),
            "n_od_bins_analyzed": n_analyzed,
            "n_multimodal": n_multimodal,
            "multimodal_frac": round(n_multimodal / max(n_analyzed, 1), 4),
            "diversity_score": {
                "mean": round(float(diversity_arr.mean()), 4) if len(diversity_arr) else None,
                "p25": round(float(np.percentile(diversity_arr, 25)), 4) if len(diversity_arr) else None,
                "p50": round(float(np.percentile(diversity_arr, 50)), 4) if len(diversity_arr) else None,
                "p75": round(float(np.percentile(diversity_arr, 75)), 4) if len(diversity_arr) else None,
                "p90": round(float(np.percentile(diversity_arr, 90)), 4) if len(diversity_arr) else None,
            },
        },
        "multimodal_examples": multimodal_examples[:20],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"OD Diversity Scan Results", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    print(f"  Routes: {n_routes} total → {len(keep_ids)} filtered", file=sys.stderr)
    print(f"  OD bins: {len(od_bins)} total → {len(valid_bins)} with >={args.min_routes_per_bin} routes", file=sys.stderr)
    print(f"  Analyzed: {n_analyzed} bins", file=sys.stderr)
    print(f"  Multimodal (LCS dist >= {args.lcs_sep_thr}): {n_multimodal}/{n_analyzed} "
          f"({n_multimodal/max(n_analyzed,1)*100:.1f}%)", file=sys.stderr)
    if len(diversity_arr):
        print(f"  Mean LCS distance: p50={np.median(diversity_arr):.3f}  "
              f"p75={np.percentile(diversity_arr,75):.3f}  "
              f"p90={np.percentile(diversity_arr,90):.3f}", file=sys.stderr)
    print(f"  Output: {args.out_json}", file=sys.stderr)
    print(f"  Elapsed: {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
