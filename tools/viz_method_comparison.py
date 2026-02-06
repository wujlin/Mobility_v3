#!/usr/bin/env python3
"""
可视化对比不同方法在相同 OD 对上的轨迹。

Usage:
  python -m tools.viz_method_comparison \
    --way_graph_npz <WAY_GRAPH_NPZ> \
    --way_features_npz <WAY_FEATS_NPZ> \
    --result_jsons "SP=path1.json" "RNN=path2.json" "WayCasd=path3.json" \
    --out_dir viz_compare \
    --n_samples 10

Requirements:
  - 各 result_json 需要用 --dump_way_seqs 生成，包含 pred_way_ids 和 gt_way_ids
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class VizCfg:
    way_graph_npz: str
    way_features_npz: str
    result_jsons: Dict[str, str]  # name -> path
    out_dir: str
    n_samples: int = 10
    seed: int = 0
    city: int = 0  # 只看一个 city


def load_way_graph(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (ptr, idx) CSR 格式的邻接表"""
    d = np.load(path)
    return d["way_adj_ptr"], d["way_adj_idx"]


def load_way_xy(path: str) -> np.ndarray:
    """返回 way center xy 坐标 (N, 2)"""
    d = np.load(path)
    return d["way_center_yx"][:, ::-1]  # yx -> xy


def load_results_with_way_seqs(path: str) -> List[Dict[str, Any]]:
    """加载结果 JSON，返回包含 way_ids 的记录"""
    with open(path) as f:
        data = json.load(f)
    
    # 尝试多种格式
    if "_raw_metrics" in data:
        return data["_raw_metrics"]
    if "per_route" in data:
        return data["per_route"]
    if isinstance(data, list):
        return data
    
    raise ValueError(f"Cannot find per-route records in {path}")


def plot_routes(
    xy: np.ndarray,
    gt_ids: List[int],
    pred_ids_dict: Dict[str, List[int]],  # method_name -> way_ids
    out_path: str,
    title: str = "",
):
    """绘制 GT 和多个方法的预测轨迹"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # 颜色方案
    colors = {
        "GT": "black",
        "SP": "gray",
        "GTG": "purple",
        "RNN-AR": "blue",
        "Tr-AR": "cyan",
        "DiffTraj": "orange",
        "WayCasd": "red",
        "E2": "red",
        "B3": "darkred",
    }
    
    # 绘制 GT
    if gt_ids:
        gt_xy = xy[np.array(gt_ids)]
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'o-', color=colors.get("GT", "black"), 
                linewidth=3, markersize=8, label="GT", alpha=0.6, zorder=10)
        # 标记起点和终点
        ax.scatter([gt_xy[0, 0]], [gt_xy[0, 1]], s=200, c='green', marker='s', zorder=20, label='Start')
        ax.scatter([gt_xy[-1, 0]], [gt_xy[-1, 1]], s=200, c='red', marker='*', zorder=20, label='Dest')
    
    # 绘制各方法的预测
    for method, pred_ids in pred_ids_dict.items():
        if not pred_ids:
            continue
        pred_xy = xy[np.array(pred_ids)]
        color = colors.get(method, "gray")
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], 'o-', color=color,
                linewidth=2, markersize=4, label=method, alpha=0.7)
    
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--way_graph_npz", required=True)
    parser.add_argument("--way_features_npz", required=True)
    parser.add_argument("--result_jsons", nargs="+", required=True, 
                        help="格式: NAME=path.json, 例如 'RNN=rnn_results.json'")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--city", type=int, default=0)
    args = parser.parse_args()
    
    # 解析 result_jsons
    result_jsons = {}
    for item in args.result_jsons:
        name, path = item.split("=", 1)
        result_jsons[name] = path
    
    cfg = VizCfg(
        way_graph_npz=args.way_graph_npz,
        way_features_npz=args.way_features_npz,
        result_jsons=result_jsons,
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        city=args.city,
    )
    
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    # 加载 way 坐标
    xy = load_way_xy(cfg.way_features_npz)
    print(f"[INFO] Loaded way_xy: {xy.shape}")
    
    # 加载各方法结果
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    for name, path in cfg.result_jsons.items():
        try:
            all_results[name] = load_results_with_way_seqs(path)
            print(f"[INFO] Loaded {name}: {len(all_results[name])} routes")
        except Exception as e:
            print(f"[WARN] Failed to load {name}: {e}")
    
    if not all_results:
        print("[ERROR] No valid results loaded!")
        return
    
    # 找到所有方法共有的路径（按 route_idx 或 start/dest 匹配）
    # 简单起见，假设所有方法的 per_route 顺序一致
    first_method = list(all_results.keys())[0]
    n_routes = len(all_results[first_method])
    
    # 过滤指定 city
    valid_indices = []
    for i, rec in enumerate(all_results[first_method]):
        if int(rec.get("city", rec.get("route_city", 0))) == cfg.city:
            valid_indices.append(i)
    
    print(f"[INFO] City {cfg.city} has {len(valid_indices)} routes")
    
    # 随机采样
    rng = np.random.default_rng(cfg.seed)
    if len(valid_indices) > cfg.n_samples:
        sample_indices = rng.choice(valid_indices, size=cfg.n_samples, replace=False)
    else:
        sample_indices = valid_indices[:cfg.n_samples]
    
    # 绘制
    for i, idx in enumerate(sample_indices):
        # 获取 GT
        rec0 = all_results[first_method][idx]
        gt_ids = rec0.get("gt_way_ids", rec0.get("beam", {}).get("gt_way_ids", []))
        if not gt_ids:
            # 尝试从 greedy 获取
            gt_ids = rec0.get("greedy", {}).get("gt_way_ids", [])
        
        # 获取各方法预测
        pred_dict = {}
        success_dict = {}
        for name, results in all_results.items():
            if idx >= len(results):
                continue
            rec = results[idx]
            # 尝试多种字段名
            pred = rec.get("pred_way_ids", 
                          rec.get("beam", {}).get("pred_way_ids",
                          rec.get("greedy", {}).get("pred_way_ids", [])))
            if pred:
                pred_dict[name] = pred
                # 检查是否成功
                succ = rec.get("beam", {}).get("success", rec.get("greedy", {}).get("success", False))
                success_dict[name] = succ
        
        if not gt_ids and not pred_dict:
            print(f"[WARN] Route {idx}: no way_ids found, skipping")
            continue
        
        # 构建标题
        gt_hops = len(gt_ids) - 1 if gt_ids else 0
        succ_str = ", ".join(f"{k}:{'✓' if v else '✗'}" for k, v in success_dict.items())
        title = f"Route {idx}, GT hops={gt_hops}\n{succ_str}"
        
        out_path = os.path.join(cfg.out_dir, f"route_{idx:04d}.png")
        plot_routes(xy, gt_ids, pred_dict, out_path, title=title)
        print(f"[OK] Saved {out_path}")
    
    print(f"\n[DONE] Saved {len(sample_indices)} visualizations to {cfg.out_dir}")


if __name__ == "__main__":
    main()
