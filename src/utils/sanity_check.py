"""
Sanity Check Script
验证 processed 数据产物的完整性、一致性与“无泄漏”要求。

Usage:
    python -m src.utils.sanity_check --data_path data/processed
    python -m src.utils.sanity_check --data_path data/processed --strict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_nav_field(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[str, Any]]]:
    data = np.load(path, allow_pickle=True)

    if "direction" in data:
        direction = data["direction"]
    elif "nav_y" in data and "nav_x" in data:
        direction = np.stack([data["nav_y"], data["nav_x"]], axis=0)
    else:
        raise KeyError("nav_field.npz must contain 'direction' or ('nav_y','nav_x').")

    direction = direction.astype(np.float32)
    if direction.ndim != 3:
        raise ValueError(f"Invalid direction shape: {direction.shape}")
    if direction.shape[0] == 2:
        direction_2hw = direction
    elif direction.shape[-1] == 2:
        direction_2hw = np.transpose(direction, (2, 0, 1))
    else:
        raise ValueError(f"Invalid direction shape: {direction.shape}")

    count = data["count"].astype(np.float32) if "count" in data else None

    metadata = None
    if "metadata" in data:
        meta = data["metadata"]
        metadata = meta.item() if hasattr(meta, "item") else meta

    return direction_2hw, count, metadata


def check_splits_no_overlap(data_path: Path) -> bool:
    print("\n[1] 检查 split 是否重叠...")
    splits_dir = data_path / "splits"

    train_ids = set(np.load(splits_dir / "train_ids.npy").tolist())
    val_ids = set(np.load(splits_dir / "val_ids.npy").tolist())
    test_ids = set(np.load(splits_dir / "test_ids.npy").tolist())

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("  ✗ FAIL: split 有重叠")
        print(f"    train∩val: {len(train_val_overlap)}")
        print(f"    train∩test: {len(train_test_overlap)}")
        print(f"    val∩test: {len(val_test_overlap)}")
        return False

    print("  ✓ PASS: split 无重叠")
    print(f"    train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    return True


def check_data_stats_source(data_path: Path, strict: bool) -> bool:
    print("\n[2] 检查 data_stats.json 的 source...")
    stats_file = data_path / "data_stats.json"
    if not stats_file.exists():
        print("  ✗ FAIL: 缺少 data_stats.json")
        return False

    stats = _load_json(stats_file)
    if "source" not in stats:
        print("  ⚠ WARN: data_stats.json 缺少 source（legacy 格式）")
        return not strict

    source = stats["source"] or {}
    print("  ✓ PASS: source 已记录")
    print(f"    split: {source.get('split', 'N/A')}")
    print(f"    trajectory_ids_file: {source.get('trajectory_ids_file', 'N/A')}")
    print(f"    trajectory_ids_sha256: {source.get('trajectory_ids_sha256', 'N/A')}")
    return True


def check_nav_field_source(data_path: Path, strict: bool) -> bool:
    print("\n[3] 检查 nav_field.npz 的结构与 metadata...")
    nav_file = data_path / "nav_field.npz"
    if not nav_file.exists():
        print("  ⚠ SKIP: 缺少 nav_field.npz（尚未生成）")
        return True

    try:
        direction, count, metadata = _load_nav_field(nav_file)
    except Exception as e:
        print(f"  ✗ FAIL: nav_field.npz 读取失败: {e}")
        return False

    print("  ✓ PASS: nav_field 结构 OK")
    print(f"    direction shape: {tuple(direction.shape)}")
    if count is not None:
        print(f"    count shape: {tuple(count.shape)}")

    if metadata is None:
        print("  ⚠ WARN: nav_field.npz 缺少 metadata（建议 strict 版本必须补齐）")
        return not strict

    print(f"    metadata.source_split: {metadata.get('source_split', 'N/A')}")
    return True


def _compute_dt_summary(
    timestamps: np.ndarray,
    traj_ptr: np.ndarray,
    sample_traj: int,
    seed: int = 0,
) -> Dict[str, Any]:
    n_traj = int(len(traj_ptr) - 1)
    if n_traj <= 0:
        return {"count": 0}

    rng = np.random.default_rng(seed)
    sample_traj = int(min(sample_traj, n_traj))
    traj_ids = rng.choice(n_traj, size=sample_traj, replace=False) if sample_traj > 0 else np.array([], dtype=np.int64)

    dts = []
    for tid in traj_ids:
        start = int(traj_ptr[tid])
        end = int(traj_ptr[tid + 1])
        if end - start < 2:
            continue
        dt = np.diff(timestamps[start:end].astype(np.int64))
        dts.append(dt)

    if not dts:
        return {"count": 0}

    dts = np.concatenate(dts)
    dts = dts[dts >= 0]
    if dts.size == 0:
        return {"count": 0}

    vals, counts = np.unique(dts, return_counts=True)
    top_idx = np.argsort(counts)[::-1][:10]
    top = [(int(vals[i]), int(counts[i])) for i in top_idx]

    return {
        "count": int(dts.size),
        "min": int(dts.min()),
        "max": int(dts.max()),
        "p50": float(np.percentile(dts, 50)),
        "p95": float(np.percentile(dts, 95)),
        "p99": float(np.percentile(dts, 99)),
        "top_dt": top,
    }


def check_trajectory_dt(data_path: Path, expected_dt: int, sample_traj: int, strict: bool) -> bool:
    print("\n[4] 检查 timestamps 的 dt 分布...")
    h5_file = data_path / "trajectories" / "shenzhen_trajectories.h5"
    if not h5_file.exists():
        print(f"  ✗ FAIL: 缺少 {h5_file}")
        return False

    with h5py.File(h5_file, "r") as f:
        if "timestamps" not in f or "traj_ptr" not in f:
            print("  ✗ FAIL: HDF5 缺少 timestamps/traj_ptr（结构不符合约定）")
            return False
        timestamps = f["timestamps"][:].astype(np.int64)
        traj_ptr = f["traj_ptr"][:].astype(np.int64)

    summary = _compute_dt_summary(timestamps, traj_ptr, sample_traj=sample_traj)
    if summary.get("count", 0) == 0:
        print("  ⚠ WARN: 无有效 dt 样本（轨迹太短或数据为空）")
        return not strict

    print("  ✓ dt 统计（抽样）:")
    print(f"    count={summary['count']}, min={summary['min']}, p50={summary['p50']:.1f}, p95={summary['p95']:.1f}, max={summary['max']}")
    print(f"    top_dt={summary['top_dt']}")

    if expected_dt > 0:
        # 粗略判断：最频繁的 dt 是否等于 expected_dt
        top0 = summary["top_dt"][0][0] if summary["top_dt"] else None
        if top0 != int(expected_dt):
            print(f"  ⚠ WARN: 最频繁 dt={top0}，不等于 expected_dt={expected_dt}")
            return not strict

    return True


def check_coordinate_range(data_path: Path, strict: bool) -> bool:
    print("\n[5] 检查 positions 的坐标范围...")
    h5_file = data_path / "trajectories" / "shenzhen_trajectories.h5"
    if not h5_file.exists():
        print("  ⚠ SKIP: 缺少 trajectories 文件")
        return True

    stats_file = data_path / "data_stats.json"
    stats = _load_json(stats_file) if stats_file.exists() else {}
    grid = stats.get("grid_config") or {}
    H = grid.get("H")
    W = grid.get("W")

    with h5py.File(h5_file, "r") as f:
        if "positions" not in f:
            print("  ✗ FAIL: HDF5 缺少 positions")
            return False
        positions = f["positions"]
        n = int(positions.shape[0])
        k = min(10000, n)
        sample = positions[:k].astype(np.float32) if k > 0 else np.zeros((0, 2), dtype=np.float32)

    if sample.size == 0:
        print("  ⚠ WARN: positions 为空")
        return not strict

    pos_min = sample.min(axis=0)
    pos_max = sample.max(axis=0)
    print(f"  ✓ 抽样范围: y∈[{pos_min[0]:.2f},{pos_max[0]:.2f}], x∈[{pos_min[1]:.2f},{pos_max[1]:.2f}]")

    if H is not None and W is not None:
        bad = (pos_min[0] < -1) or (pos_min[1] < -1) or (pos_max[0] > float(H) + 1) or (pos_max[1] > float(W) + 1)
        if bad:
            print(f"  ⚠ WARN: 坐标疑似超出 grid_config(H={H}, W={W})")
            return not strict

    norm = stats.get("normalization") or {}
    if norm:
        print(f"    stats.pos_min={norm.get('pos_min', 'N/A')}, stats.pos_max={norm.get('pos_max', 'N/A')}")

    return True


def _alignment_stats(
    direction_2hw: np.ndarray,
    positions: np.ndarray,
    traj_ptr: np.ndarray,
    count_hw: Optional[np.ndarray],
    sample_traj: int,
    min_count: int,
    max_samples: int,
    seed: int = 0,
) -> Dict[str, Any]:
    H, W = int(direction_2hw.shape[1]), int(direction_2hw.shape[2])

    n_traj = int(len(traj_ptr) - 1)
    rng = np.random.default_rng(seed)
    sample_traj = int(min(sample_traj, n_traj))
    traj_ids = rng.choice(n_traj, size=sample_traj, replace=False) if sample_traj > 0 else np.array([], dtype=np.int64)

    cos_all = []
    for tid in traj_ids:
        start = int(traj_ptr[tid])
        end = int(traj_ptr[tid + 1])
        if end - start < 2:
            continue

        p0 = positions[start : end - 1]
        dv = positions[start + 1 : end] - positions[start : end - 1]

        dv_norm = np.linalg.norm(dv, axis=1)
        valid = dv_norm > 1e-6
        if not np.any(valid):
            continue

        p0 = p0[valid]
        dv = dv[valid]
        dv_norm = dv_norm[valid]

        yi = np.clip(p0[:, 0].astype(np.int64), 0, H - 1)
        xi = np.clip(p0[:, 1].astype(np.int64), 0, W - 1)

        if count_hw is not None and min_count > 0:
            valid2 = count_hw[yi, xi] >= float(min_count)
            if not np.any(valid2):
                continue
            yi = yi[valid2]
            xi = xi[valid2]
            dv = dv[valid2]
            dv_norm = dv_norm[valid2]

        nav = direction_2hw[:, yi, xi].T  # (N,2)
        nav_norm = np.linalg.norm(nav, axis=1)
        valid3 = nav_norm > 1e-6
        if not np.any(valid3):
            continue
        nav = nav[valid3]
        dv = dv[valid3]
        dv_norm = dv_norm[valid3]
        nav_norm = nav_norm[valid3]

        cos = np.sum(nav * dv, axis=1) / (nav_norm * dv_norm)
        cos_all.append(cos.astype(np.float32))

        if sum(len(x) for x in cos_all) >= max_samples:
            break

    if not cos_all:
        return {"count": 0}

    cos_all = np.concatenate(cos_all)
    if cos_all.size > max_samples:
        cos_all = cos_all[:max_samples]

    return {
        "count": int(cos_all.size),
        "mean_cos": float(np.mean(cos_all)),
        "mean_abs_cos": float(np.mean(np.abs(cos_all))),
        "neg_ratio": float(np.mean(cos_all < 0)),
        "p10": float(np.percentile(cos_all, 10)),
        "p50": float(np.percentile(cos_all, 50)),
        "p90": float(np.percentile(cos_all, 90)),
    }


def check_nav_field_alignment(
    data_path: Path,
    sample_traj: int,
    min_count: int,
    abs_threshold: float,
    strict: bool,
) -> bool:
    print("\n[6] 检查 nav_field 与轨迹方向的一致性...")

    nav_file = data_path / "nav_field.npz"
    h5_file = data_path / "trajectories" / "shenzhen_trajectories.h5"
    if not nav_file.exists() or not h5_file.exists():
        print("  ⚠ SKIP: 缺少 nav_field 或 trajectories")
        return True

    try:
        direction, count_hw, _ = _load_nav_field(nav_file)
    except Exception as e:
        print(f"  ✗ FAIL: nav_field 读取失败: {e}")
        return False

    with h5py.File(h5_file, "r") as f:
        if "positions" not in f or "traj_ptr" not in f:
            print("  ✗ FAIL: HDF5 缺少 positions/traj_ptr")
            return False
        positions = f["positions"][:].astype(np.float32)
        traj_ptr = f["traj_ptr"][:].astype(np.int64)

    stats = _alignment_stats(
        direction_2hw=direction,
        positions=positions,
        traj_ptr=traj_ptr,
        count_hw=count_hw,
        sample_traj=sample_traj,
        min_count=min_count,
        max_samples=200000,
    )

    if stats.get("count", 0) == 0:
        print("  ⚠ WARN: 无有效 alignment 样本（可能 count 太低或方向场全 0）")
        return not strict

    print(f"  ✓ 样本数: {stats['count']} (min_count={min_count})")
    print(f"    mean_cos={stats['mean_cos']:.3f}, mean|cos|={stats['mean_abs_cos']:.3f}, neg_ratio={stats['neg_ratio']:.3f}")
    print(f"    p10={stats['p10']:.3f}, p50={stats['p50']:.3f}, p90={stats['p90']:.3f}")

    if stats["mean_abs_cos"] < float(abs_threshold):
        print(f"  ⚠ WARN: mean|cos|={stats['mean_abs_cos']:.3f} < 阈值 {abs_threshold}")
        return not strict

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity check for data products")
    parser.add_argument("--data_path", type=str, default="data/processed", help="processed data directory")
    parser.add_argument("--strict", action="store_true", help="将 WARN 视为 FAIL")
    parser.add_argument("--expected_dt", type=int, default=30, help="期望最频繁的 dt（秒），<=0 表示不检查")
    parser.add_argument("--dt_sample_traj", type=int, default=500, help="dt 统计抽样轨迹数")

    parser.add_argument("--align_sample_traj", type=int, default=200, help="alignment 抽样轨迹数")
    parser.add_argument("--align_min_count", type=int, default=10, help="alignment 仅使用 count>=min_count 的格子（无 count 则忽略）")
    parser.add_argument("--align_abs_threshold", type=float, default=0.6, help="alignment 的 mean|cos| 阈值（仅用于告警/严格模式）")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    strict = bool(args.strict)

    print("=" * 60)
    print("SANITY CHECK")
    print("=" * 60)
    print(f"data_path: {data_path}")
    print(f"strict:    {strict}")

    results = []
    results.append(("Split overlap", check_splits_no_overlap(data_path)))
    results.append(("Data stats source", check_data_stats_source(data_path, strict=strict)))
    results.append(("Nav field source", check_nav_field_source(data_path, strict=strict)))
    results.append(("Trajectory dt", check_trajectory_dt(data_path, expected_dt=args.expected_dt, sample_traj=args.dt_sample_traj, strict=strict)))
    results.append(("Coordinate range", check_coordinate_range(data_path, strict=strict)))
    results.append((
        "Nav field alignment",
        check_nav_field_alignment(
            data_path,
            sample_traj=args.align_sample_traj,
            min_count=args.align_min_count,
            abs_threshold=args.align_abs_threshold,
            strict=strict,
        ),
    ))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All checks passed!")
        return 0

    print("\n✗ Some checks failed. Please review above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
