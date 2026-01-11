#!/usr/bin/env python3
"""
Audit POI raster data distribution under a semantic_dir (city-agnostic).
检查 POI 栅格数据的空间分布，快速判断“稀疏/集中/覆盖”。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _as_number_list(x: Any) -> List[float] | None:
    if not isinstance(x, list) or not x:
        return None
    out: List[float] = []
    for v in x:
        if isinstance(v, bool):
            return None
        if not isinstance(v, (int, float)):
            return None
        if not np.isfinite(float(v)):
            return None
        out.append(float(v))
    return out


def _format_topk_hour_peaks(vals: List[float], *, k: int = 4) -> str:
    a = np.asarray(vals, dtype=np.float64).reshape(-1)
    idx = np.argsort(a)[::-1][: int(k)]

    def _fmt(v: float) -> str:
        if abs(v - round(v)) < 1e-6:
            return str(int(round(v)))
        return f"{v:.3f}"

    parts = [f"{int(h):02d}:{_fmt(float(a[int(h)]))}" for h in idx.tolist()]
    return ", ".join(parts)


def _print_meta_summary(meta: Dict[str, Any]) -> None:
    print("=== Meta Info (summary) ===")
    # 尽量按“读者友好”的方式打印：长数组不再逐行展开。
    for k in sorted(meta.keys()):
        v = meta[k]
        num_list = _as_number_list(v)
        if num_list is not None and len(num_list) == 24:
            total = float(np.sum(np.asarray(num_list, dtype=np.float64)))
            peaks = _format_topk_hour_peaks(num_list, k=4)
            print(f"{k}: sum={total:.0f}, top4={peaks}")
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            print(f"{k}: {v}")
            continue
        if isinstance(v, dict):
            # 只展开一层
            small = {kk: vv for kk, vv in v.items() if isinstance(vv, (str, int, float, bool)) or vv is None}
            if small:
                print(f"{k}: {json.dumps(small, ensure_ascii=False)}")
            else:
                print(f"{k}: <dict>")
            continue
        if num_list is not None:
            if len(num_list) <= 12:
                joined = ",".join(str(int(round(x))) if abs(x - round(x)) < 1e-6 else f"{x:.3f}" for x in num_list)
                print(f"{k}: [{joined}]")
            else:
                arr = np.asarray(num_list, dtype=np.float64)
                print(f"{k}: len={len(num_list)}, min={float(arr.min()):.3f}, mean={float(arr.mean()):.3f}, max={float(arr.max()):.3f}")
            continue
        print(f"{k}: <{type(v).__name__}>")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit POI rasters under a semantic_dir (poi_density_*.npy + optional poi_raster_meta.json).")
    p.add_argument("--semantic_dir", type=str, default="data/worldtrace/detroit_core_v1", help="Directory containing poi_density_*.npy and landuse_entropy.npy.")
    p.add_argument("--meta", type=str, default="summary", choices=["summary", "pretty", "compact", "none"], help="How to print poi_raster_meta.json.")
    p.add_argument("--region_size", type=int, default=500, help="Region size for coarse coverage check.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    d = Path(args.semantic_dir)
    if not d.exists():
        raise SystemExit(f"ERROR: semantic_dir not found: {d}")

    poi_files = sorted(d.glob("poi_density_*.npy"))
    if not poi_files:
        raise SystemExit(f"ERROR: No poi_density_*.npy files found in {d}")

    print(f"=== POI Raster Audit: {d} ===\n")

    poi_stack = []
    for p in poi_files:
        cat = p.stem.replace("poi_density_", "")
        arr = np.load(p)
        poi_stack.append(arr)

        nonzero = int(np.sum(arr > 0))
        total = float(np.sum(arr))
        max_val = float(np.max(arr))
        mean_nonzero = float(np.mean(arr[arr > 0])) if nonzero > 0 else 0.0

        print(f"[{cat}] shape={tuple(arr.shape)} nonzero={nonzero:,}/{arr.size:,} ({100*nonzero/arr.size:.2f}%) total={total:,.0f} max={max_val:.0f} mean_nz={mean_nonzero:.2f}")

    poi_total = np.sum(poi_stack, axis=0)
    nonzero_total = int(np.sum(poi_total > 0))
    total_total = float(np.sum(poi_total))
    max_total = float(np.max(poi_total))
    mean_nz_total = float(np.mean(poi_total[poi_total > 0])) if nonzero_total > 0 else 0.0

    print("\n=== TOTAL (all categories) ===")
    print(
        f"shape={tuple(poi_total.shape)} nonzero={nonzero_total:,}/{poi_total.size:,} ({100*nonzero_total/poi_total.size:.2f}%) "
        f"total={total_total:,.0f} max={max_total:.0f} mean_nz={mean_nz_total:.2f}"
    )

    print("\n=== Distribution (poi_total) ===")
    for t in [0, 1, 5, 10, 50, 100]:
        count = int(np.sum(poi_total > t))
        print(f"> {t:3d}: {count:,} cells ({100*count/poi_total.size:.2f}%)")

    H, W = int(poi_total.shape[0]), int(poi_total.shape[1])
    region_size = int(args.region_size)
    stride = max(region_size // 2, 1)
    regions_with_poi = 0
    total_regions = 0
    for y in range(0, max(H - region_size + 1, 1), stride):
        for x in range(0, max(W - region_size + 1, 1), stride):
            region = poi_total[y : y + region_size, x : x + region_size]
            total_regions += 1
            if float(np.sum(region)) > 0.0:
                regions_with_poi += 1

    print(f"\n=== Spatial Coverage ({region_size}x{region_size}, stride={stride}) ===")
    if total_regions > 0:
        print(f"regions_with_poi={regions_with_poi}/{total_regions} ({100*regions_with_poi/total_regions:.1f}%)")
    else:
        print("regions_with_poi=0/0")

    entropy_path = d / "landuse_entropy.npy"
    if entropy_path.exists():
        entropy = np.load(entropy_path)
        nz = int(np.sum(entropy > 0))
        print("\n=== Landuse Entropy ===")
        print(
            f"shape={tuple(entropy.shape)} nonzero={nz:,}/{entropy.size:,} ({100*nz/entropy.size:.2f}%) "
            f"mean_nz={float(np.mean(entropy[entropy > 0])) if nz > 0 else 0.0:.3f} max={float(np.max(entropy)):.3f} std_nz={float(np.std(entropy[entropy > 0])) if nz > 0 else 0.0:.3f}"
        )

    meta_path = d / "poi_raster_meta.json"
    if meta_path.exists() and str(args.meta) != "none":
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print()
        if str(args.meta) == "pretty":
            print(json.dumps(meta, ensure_ascii=False, indent=2))
        elif str(args.meta) == "compact":
            print(json.dumps(meta, ensure_ascii=False, separators=(",", ":")))
        else:
            if not isinstance(meta, dict):
                print(f"poi_raster_meta.json: <{type(meta).__name__}>")
            else:
                _print_meta_summary(meta)


if __name__ == "__main__":
    main()
