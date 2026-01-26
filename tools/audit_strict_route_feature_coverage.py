#!/usr/bin/env python3
"""
Audit: 检查 strict routes 中的 way token 是否落在 missing way 上。

核心问题：way_features.npz 有 13.09% 的 way 缺失几何特征（way_len_m <= 0），
这些 missing way 是否出现在 strict routes 的序列中？

注意：仓库当前主流的 way_routes 格式键为：
  - way_seq_ptr / way_seq_idx / way_seq_len
而不是早期实验里可能出现的 way_seq_flat。
本脚本同时兼容两种键名，避免因为格式差异导致误判。

输出：
  - route-level: 有多少 route 包含至少一个 missing way token
  - token-level: 所有 token 中有多少是 missing way
  - 分位数: 每条 route 的 missing token 占比分布
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _load_routes_npz(path: Path):
    data = np.load(str(path), allow_pickle=True)

    # New (current) format: CSR + explicit lengths.
    if {"way_seq_ptr", "way_seq_idx"}.issubset(set(data.files)):
        way_seq = np.asarray(data["way_seq_idx"], dtype=np.int64).reshape(-1)
        way_ptr = np.asarray(data["way_seq_ptr"], dtype=np.int64).reshape(-1)
        way_len = np.asarray(data["way_seq_len"], dtype=np.int64).reshape(-1) if "way_seq_len" in data.files else None
    # Legacy format (rare): flat + ptr.
    elif {"way_seq_ptr", "way_seq_flat"}.issubset(set(data.files)):
        way_seq = np.asarray(data["way_seq_flat"], dtype=np.int64).reshape(-1)
        way_ptr = np.asarray(data["way_seq_ptr"], dtype=np.int64).reshape(-1)
        way_len = None
    else:
        raise KeyError(f"Unsupported routes npz keys: {sorted(data.files)}")

    route_city = np.asarray(data["route_city"], dtype=np.int64).reshape(-1) if "route_city" in data.files else None
    return way_seq, way_ptr, way_len, route_city


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--way_routes_npz", type=Path, required=True, help="strict routes")
    p.add_argument("--way_features_npz", type=Path, required=True, help="way features with semantic")
    p.add_argument("--out_json", type=Path, default=None)
    args = p.parse_args()

    # Load routes
    way_seq_flat, way_seq_ptr, way_seq_len, route_city = _load_routes_npz(args.way_routes_npz)
    if way_seq_len is not None:
        N = int(way_seq_len.size)
    else:
        N = int(way_seq_ptr.size) - 1
    print(f"Loaded {N} routes, {int(way_seq_flat.size)} total tokens")

    # Load features
    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    way_len_m = np.asarray(wf["way_len_m"], dtype=np.float32).reshape(-1)
    M = int(way_len_m.size)
    print(f"Loaded {M} way features")

    # Define missing: way_len_m <= 0 (same criterion as partner's audit)
    is_missing = (way_len_m <= 0)
    n_missing_global = int(np.sum(is_missing))
    print(f"Missing ways (global): {n_missing_global}/{M} = {100.0 * n_missing_global / M:.2f}%")

    # Check semantic presence
    if "way_semantic" in wf.files:
        ws = np.asarray(wf["way_semantic"], dtype=np.float32)
        print(f"way_semantic: shape={ws.shape}")
    else:
        print("way_semantic: NOT FOUND")

    # Token-level: which tokens in strict routes are missing?
    # Clip to valid range (in case of index overflow)
    way_seq_clipped = np.clip(way_seq_flat, 0, M - 1)
    token_is_missing = is_missing[way_seq_clipped]
    n_missing_tokens = int(np.sum(token_is_missing))
    n_total_tokens = int(way_seq_flat.size)
    print(f"\n=== Token-level pollution ===")
    print(f"Missing tokens: {n_missing_tokens}/{n_total_tokens} = {100.0 * n_missing_tokens / n_total_tokens:.4f}%")

    # Route-level: how many routes contain at least one missing token?
    route_has_missing = []
    route_missing_frac = []
    for i in range(N):
        lo = int(way_seq_ptr[i])
        if way_seq_len is not None:
            hi = lo + int(way_seq_len[i])
        else:
            hi = int(way_seq_ptr[i + 1])
        seq = way_seq_clipped[lo:hi]
        n_miss = int(np.sum(is_missing[seq]))
        L = int(hi - lo)
        route_has_missing.append(n_miss > 0)
        route_missing_frac.append(float(n_miss / L) if L > 0 else 0.0)

    route_has_missing = np.asarray(route_has_missing, dtype=bool)
    route_missing_frac = np.asarray(route_missing_frac, dtype=np.float64)

    n_routes_with_missing = int(np.sum(route_has_missing))
    print(f"\n=== Route-level pollution ===")
    print(f"Routes with >=1 missing token: {n_routes_with_missing}/{N} = {100.0 * n_routes_with_missing / N:.2f}%")

    print(f"\n=== Per-route missing fraction distribution ===")
    for q in [0, 0.5, 0.9, 0.95, 0.99, 1.0]:
        v = np.percentile(route_missing_frac, 100 * q)
        print(f"  p{int(100*q):02d}: {100.0 * v:.2f}%")

    # List worst routes
    worst_idx = np.argsort(-route_missing_frac)[:10]
    print(f"\n=== Top 10 worst routes ===")
    for rank, idx in enumerate(worst_idx):
        lo = int(way_seq_ptr[idx])
        if way_seq_len is not None:
            hi = lo + int(way_seq_len[idx])
        else:
            hi = int(way_seq_ptr[idx + 1])
        L = int(hi - lo)
        mf = route_missing_frac[idx]
        print(f"  #{rank+1}: route_id={idx}, len={L}, missing_frac={100.0*mf:.1f}%")

    if route_city is not None:
        print(f"\n=== Breakdown by city ===")
        for c in sorted(set(route_city.tolist())):
            m = route_city == int(c)
            n = int(np.sum(m))
            bad = int(np.sum(route_has_missing[m]))
            print(f"  city={int(c)}: routes={n}, bad_routes={bad}, bad_rate={100.0 * bad / max(1, n):.2f}%")

    # Summary
    result = {
        "n_routes": int(N),
        "n_tokens": int(n_total_tokens),
        "n_ways_global": int(M),
        "n_missing_ways_global": int(n_missing_global),
        "missing_way_frac_global": float(n_missing_global / M),
        "n_missing_tokens": int(n_missing_tokens),
        "token_pollution_rate": float(n_missing_tokens / n_total_tokens),
        "n_routes_with_missing": int(n_routes_with_missing),
        "route_pollution_rate": float(n_routes_with_missing / N),
        "has_route_city": bool(route_city is not None),
        "route_missing_frac_quantiles": {
            f"p{int(100*q):02d}": float(np.percentile(route_missing_frac, 100 * q))
            for q in [0, 0.5, 0.9, 0.95, 0.99, 1.0]
        },
    }

    print(f"\n=== Summary JSON ===")
    print(json.dumps(result, indent=2))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
