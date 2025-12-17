"""
calibrate_vel_scale.py

目的：为“运动幅度偏小/收缩”问题提供一个**可复现**的 vel_scale 校准工具。

核心原则（与论文叙事一致）：
- Temperature/噪声强度：主要影响多样性与抖动，不应拿来“撑大”位移幅度。
- vel_scale：对 future step displacement 做系统性缩放，直接控制物理幅度（path_len / speed）。

推荐做法：
1) 在 val split 上跑一次 evaluate（vel_scale=1.0），得到 metrics.json
2) 用本脚本计算推荐的 vel_scale（优先使用 speed/path_len 的比值）
3) 固定 vel_scale，在 test split 上复现评估（避免泄漏）
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_ratio(numer: float, denom: float) -> Optional[float]:
    if denom == 0:
        return None
    return float(numer) / float(denom)


def _safe_sqrt_ratio(numer: float, denom: float) -> Optional[float]:
    if denom <= 0 or numer <= 0:
        return None
    return math.sqrt(float(numer) / float(denom))


def _get_float(d: Dict[str, Any], key: str) -> Optional[float]:
    v = d.get(key, None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _summarize(name: str, vals: List[float]) -> str:
    if not vals:
        return f"{name}: n=0"
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    med = statistics.median(vals)
    return f"{name}: n={len(vals)} mean={mean:.4f} std={sd:.4f} median={med:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate vel_scale from evaluate metrics.json files (val split recommended).")
    parser.add_argument("metrics", nargs="+", help="paths to metrics.json (can pass multiple seeds)")
    parser.add_argument(
        "--prefer",
        type=str,
        default="speed",
        choices=["speed", "path_len", "rog", "msd10"],
        help="which estimator to recommend as vel_scale (default: speed)",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.metrics]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    scale_speed = []
    scale_path = []
    scale_rog = []
    scale_msd10 = []

    for p in paths:
        m = _load_json(p)

        pred_speed = _get_float(m, "pred_speed_mean")
        gt_speed = _get_float(m, "gt_speed_mean")
        if pred_speed is not None and gt_speed is not None:
            r = _safe_ratio(gt_speed, pred_speed)
            if r is not None:
                scale_speed.append(r)

        pred_path = _get_float(m, "pred_path_len_mean")
        gt_path = _get_float(m, "gt_path_len_mean")
        if pred_path is not None and gt_path is not None:
            r = _safe_ratio(gt_path, pred_path)
            if r is not None:
                scale_path.append(r)

        rog = _get_float(m, "Rog")
        gt_rog = _get_float(m, "GT_Rog")
        if rog is not None and gt_rog is not None:
            r = _safe_ratio(gt_rog, rog)
            if r is not None:
                scale_rog.append(r)

        msd10 = _get_float(m, "MSD_10")
        gt_msd10 = _get_float(m, "GT_MSD_10")
        if msd10 is not None and gt_msd10 is not None:
            r = _safe_sqrt_ratio(gt_msd10, msd10)
            if r is not None:
                scale_msd10.append(r)

    print(_summarize("vel_scale_speed = gt_speed_mean / pred_speed_mean", scale_speed))
    print(_summarize("vel_scale_path_len = gt_path_len_mean / pred_path_len_mean", scale_path))
    print(_summarize("vel_scale_rog = GT_Rog / Rog", scale_rog))
    print(_summarize("vel_scale_msd10 = sqrt(GT_MSD_10 / MSD_10)", scale_msd10))

    prefer_map = {
        "speed": scale_speed,
        "path_len": scale_path,
        "rog": scale_rog,
        "msd10": scale_msd10,
    }
    chosen = prefer_map.get(str(args.prefer), [])
    if chosen:
        rec = statistics.median(chosen)
        print(f"[RECOMMEND] vel_scale ({args.prefer}) = {rec:.4f}")
    else:
        print(f"[RECOMMEND] 无法基于 prefer={args.prefer} 计算（缺少对应字段）。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

