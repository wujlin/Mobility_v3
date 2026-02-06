#!/usr/bin/env python3
"""
分析训练/测试集的 OD 重叠率和 transition 覆盖率。
验证 RNN-AR 是否在"记忆"而非"泛化"。

Usage:
  python -m tools.analyze_od_overlap --way_routes_npz <path> --val_ratio 0.1 --seed 0
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np


def load_way_routes(path: Path) -> Dict[str, np.ndarray]:
    """加载 way routes 数据"""
    return dict(np.load(path, allow_pickle=True))


def split_dataset(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """复现训练代码的划分逻辑"""
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_val = int(round(float(val_ratio) * float(n)))
    n_val = max(1, min(n - 1, n_val)) if n >= 2 else 0
    val = idx[:n_val]
    tr = idx[n_val:]
    return tr, val


def extract_od_pairs(data: Dict[str, np.ndarray], indices: np.ndarray) -> Set[Tuple[int, int]]:
    """提取指定索引的 (start_way, dest_way) 对"""
    way_seq = data["way_seq"]  # object array of arrays
    od_pairs = set()
    for i in indices:
        seq = way_seq[i]
        if len(seq) >= 2:
            od_pairs.add((int(seq[0]), int(seq[-1])))
    return od_pairs


def extract_transitions(data: Dict[str, np.ndarray], indices: np.ndarray) -> Set[Tuple[int, int]]:
    """提取指定索引的所有 (cur_way, next_way) transition"""
    way_seq = data["way_seq"]
    transitions = set()
    for i in indices:
        seq = way_seq[i]
        for j in range(len(seq) - 1):
            transitions.add((int(seq[j]), int(seq[j + 1])))
    return transitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--way_routes_npz", type=Path, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_hops", type=int, default=5, help="和评估一致的过滤条件")
    args = parser.parse_args()
    
    print("=" * 70)
    print("OD 重叠率和 Transition 覆盖率分析")
    print("=" * 70)
    
    # 加载数据
    data = load_way_routes(args.way_routes_npz)
    way_seq = data["way_seq"]
    n_total = len(way_seq)
    print(f"\n总路径数: {n_total}")
    
    # 过滤 min_hops
    valid_mask = np.array([len(seq) - 1 >= args.min_hops for seq in way_seq])
    valid_indices = np.where(valid_mask)[0]
    print(f"min_hops >= {args.min_hops} 过滤后: {len(valid_indices)} ({len(valid_indices)/n_total*100:.1f}%)")
    
    # 模拟训练代码的划分（注意：训练代码是在所有数据上划分，不是过滤后的数据）
    # 但评估时只用 min_hops >= 5 的数据
    # 这里为了分析，我们按训练代码的逻辑来
    
    # 方案 A：在所有数据上划分，然后过滤
    print("\n" + "-" * 70)
    print("方案 A：先划分后过滤（当前训练代码逻辑）")
    print("-" * 70)
    
    train_idx_all, val_idx_all = split_dataset(n_total, args.val_ratio, args.seed)
    
    # 过滤
    train_idx = np.array([i for i in train_idx_all if valid_mask[i]])
    val_idx = np.array([i for i in val_idx_all if valid_mask[i]])
    
    print(f"训练集: {len(train_idx)}, 验证集: {len(val_idx)}")
    
    # 提取 OD 对
    train_od = extract_od_pairs(data, train_idx)
    val_od = extract_od_pairs(data, val_idx)
    
    overlap_od = train_od & val_od
    print(f"\n训练集 OD 对数: {len(train_od)}")
    print(f"验证集 OD 对数: {len(val_od)}")
    print(f"重叠 OD 对数: {len(overlap_od)}")
    print(f"验证集 OD 重叠率: {len(overlap_od) / len(val_od) * 100:.1f}%")
    
    # 提取 transition
    train_trans = extract_transitions(data, train_idx)
    val_trans = extract_transitions(data, val_idx)
    
    overlap_trans = train_trans & val_trans
    print(f"\n训练集 transition 数: {len(train_trans)}")
    print(f"验证集 transition 数: {len(val_trans)}")
    print(f"重叠 transition 数: {len(overlap_trans)}")
    print(f"验证集 transition 覆盖率: {len(overlap_trans) / len(val_trans) * 100:.1f}%")
    
    # 更细致的分析：每条路径的 transition 覆盖情况
    print("\n" + "-" * 70)
    print("每条验证路径的 transition 覆盖情况")
    print("-" * 70)
    
    coverage_per_route = []
    for i in val_idx:
        seq = way_seq[i]
        route_trans = set()
        for j in range(len(seq) - 1):
            route_trans.add((int(seq[j]), int(seq[j + 1])))
        if route_trans:
            covered = len(route_trans & train_trans)
            coverage_per_route.append(covered / len(route_trans))
    
    coverage_arr = np.array(coverage_per_route)
    print(f"路径级 transition 覆盖率:")
    print(f"  mean: {coverage_arr.mean() * 100:.1f}%")
    print(f"  p50:  {np.percentile(coverage_arr, 50) * 100:.1f}%")
    print(f"  p25:  {np.percentile(coverage_arr, 25) * 100:.1f}%")
    print(f"  p10:  {np.percentile(coverage_arr, 10) * 100:.1f}%")
    print(f"  完全覆盖 (100%): {(coverage_arr == 1.0).sum()} ({(coverage_arr == 1.0).mean() * 100:.1f}%)")
    print(f"  高覆盖 (>=90%): {(coverage_arr >= 0.9).sum()} ({(coverage_arr >= 0.9).mean() * 100:.1f}%)")
    print(f"  低覆盖 (<50%): {(coverage_arr < 0.5).sum()} ({(coverage_arr < 0.5).mean() * 100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    od_overlap_rate = len(overlap_od) / len(val_od) * 100
    trans_coverage_rate = len(overlap_trans) / len(val_trans) * 100
    
    if od_overlap_rate > 50:
        print(f"⚠️  OD 重叠率 {od_overlap_rate:.1f}% 较高")
        print("    → 模型可能在"记忆"特定 OD 对的路径")
    
    if trans_coverage_rate > 80:
        print(f"⚠️  Transition 覆盖率 {trans_coverage_rate:.1f}% 很高")
        print("    → 模型几乎可以通过记忆 transition 表来预测")
    
    if (coverage_arr == 1.0).mean() > 0.3:
        print(f"⚠️  {(coverage_arr == 1.0).mean() * 100:.0f}% 的验证路径 transition 完全被训练集覆盖")
        print("    → 这些路径上 RNN-AR 可以直接"背诵"答案")


if __name__ == "__main__":
    main()
