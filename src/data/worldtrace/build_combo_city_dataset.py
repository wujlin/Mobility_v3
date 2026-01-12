from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from src.data.worldtrace.merge_route_npz import MergeConfig, merge_two
from src.data.worldtrace.stack_semantic_dirs import StackConfig, stack_two_dirs


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class LayoutConfig:
    layout: str
    gap: int
    offset_y: Optional[int]
    offset_x: Optional[int]


def _parse_files(s: str) -> Tuple[str, ...]:
    items = [x.strip() for x in str(s).split(",") if str(x).strip()]
    if not items:
        raise ValueError("--files must be a non-empty comma-separated list.")
    return tuple(items)


def _infer_offsets(a_dir: Path, files: Tuple[str, ...], cfg: LayoutConfig) -> Tuple[int, int]:
    if (cfg.offset_y is None) ^ (cfg.offset_x is None):
        raise ValueError("--b_offset_y/--b_offset_x must be both set or both None.")
    if cfg.offset_y is not None and cfg.offset_x is not None:
        return int(cfg.offset_y), int(cfg.offset_x)

    # Infer from layout + gap using the first raster shape.
    first = a_dir / files[0]
    if not first.exists():
        raise FileNotFoundError(f"Cannot infer offsets: missing {first}")
    a = np.load(first).astype(np.float32, copy=False)
    if a.ndim != 2:
        raise ValueError(f"Bad raster shape in {first}: {a.shape} (expected H,W)")
    H, W = int(a.shape[0]), int(a.shape[1])
    gap = int(cfg.gap)
    if cfg.layout == "vertical":
        return H + gap, 0
    if cfg.layout == "horizontal":
        return 0, W + gap
    raise ValueError(f"Unknown layout: {cfg.layout}")


def build_combo(
    *,
    a_route_npz: Path,
    b_route_npz: Path,
    a_semantic_dir: Path,
    b_semantic_dir: Path,
    out_dir: Path,
    city_a: Optional[str],
    city_b: Optional[str],
    files: Tuple[str, ...],
    layout_cfg: LayoutConfig,
    traj_idx_offset: int,
    merge_op: str,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    offset_y, offset_x = _infer_offsets(a_semantic_dir, files, layout_cfg)
    sem_out = out_dir / "semantic"
    route_out = out_dir / "routes.npz"
    route_report = out_dir / "routes.report.json"

    stack_report = stack_two_dirs(
        a_dir=a_semantic_dir,
        b_dir=b_semantic_dir,
        out_dir=sem_out,
        files=files,
        cfg=StackConfig(offset_y=int(offset_y), offset_x=int(offset_x), fill_value=0.0, merge_op=str(merge_op)),
        city_a=city_a,
        city_b=city_b,
    )

    merge_report = merge_two(
        a_npz=a_route_npz,
        b_npz=b_route_npz,
        out_npz=route_out,
        cfg=MergeConfig(offset_y=float(offset_y), offset_x=float(offset_x), traj_idx_offset=int(traj_idx_offset)),
        city_a=city_a,
        city_b=city_b,
    )
    route_report.write_text(json.dumps(merge_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Recommended pos bounds for training (y,x).
    # If semantic rasters are stacked, use their shape.
    # We assume all stacked files share the same shape (enforced in stack_two_dirs output).
    meta = json.loads((sem_out / "stack_meta.json").read_text())
    H = int(meta["stats"]["out_H"])
    W = int(meta["stats"]["out_W"])
    pos_max_y = int(H - 1)
    pos_max_x = int(W - 1)

    report = {
        "ok": True,
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "build_combo_city_dataset",
        "inputs": {
            "a_route_npz": str(a_route_npz),
            "b_route_npz": str(b_route_npz),
            "a_semantic_dir": str(a_semantic_dir),
            "b_semantic_dir": str(b_semantic_dir),
            "city_a": (str(city_a) if city_a else None),
            "city_b": (str(city_b) if city_b else None),
        },
        "layout": {"layout": str(layout_cfg.layout), "gap": int(layout_cfg.gap), "b_offset_y": int(offset_y), "b_offset_x": int(offset_x)},
        "outputs": {
            "out_dir": str(out_dir),
            "routes_npz": str(route_out),
            "routes_report_json": str(route_report),
            "semantic_dir": str(sem_out),
            "semantic_stack_meta_json": str(sem_out / "stack_meta.json"),
        },
        "recommend": {"pos_max_y": int(pos_max_y), "pos_max_x": int(pos_max_x)},
        "summary": {"routes": {"N": int(merge_report["N"]), "F": int(merge_report["F"])}, "semantic": {"H": int(H), "W": int(W)}},
        "reports": {"stack": stack_report, "merge": merge_report},
    }
    (out_dir / "combo_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a combo dataset for cross-city training via (1) stacking semantic rasters and (2) shifting+merging route npz.")
    p.add_argument("--city_a", type=str, default=None)
    p.add_argument("--city_b", type=str, default=None)
    p.add_argument("--a_route_npz", type=str, required=True)
    p.add_argument("--b_route_npz", type=str, required=True)
    p.add_argument("--a_semantic_dir", type=str, required=True)
    p.add_argument("--b_semantic_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--layout", type=str, choices=["vertical", "horizontal"], default="vertical")
    p.add_argument("--gap", type=int, default=256, help="Gap between city canvases in grid units (only used when offsets are not explicitly set).")
    p.add_argument("--b_offset_y", type=int, default=None)
    p.add_argument("--b_offset_x", type=int, default=None)
    p.add_argument("--b_traj_idx_offset", type=int, default=1_000_000_000)
    p.add_argument(
        "--files",
        type=str,
        default="osm_road_prob.npy,osm_road_prob_major.npy,osm_road_prob_minor.npy,osm_road_prob_service.npy",
        help="Comma-separated 2D .npy filenames to stack into the combo semantic_dir.",
    )
    p.add_argument("--merge_op", type=str, choices=["max", "overwrite"], default="max")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    files = _parse_files(args.files)
    report = build_combo(
        a_route_npz=Path(args.a_route_npz),
        b_route_npz=Path(args.b_route_npz),
        a_semantic_dir=Path(args.a_semantic_dir),
        b_semantic_dir=Path(args.b_semantic_dir),
        out_dir=Path(args.out_dir),
        city_a=(str(args.city_a) if args.city_a else None),
        city_b=(str(args.city_b) if args.city_b else None),
        files=files,
        layout_cfg=LayoutConfig(layout=str(args.layout), gap=int(args.gap), offset_y=(int(args.b_offset_y) if args.b_offset_y is not None else None), offset_x=(int(args.b_offset_x) if args.b_offset_x is not None else None)),
        traj_idx_offset=int(args.b_traj_idx_offset),
        merge_op=str(args.merge_op),
    )
    compact = {
        "ok": True,
        "out_dir": report["outputs"]["out_dir"],
        "routes_npz": report["outputs"]["routes_npz"],
        "semantic_dir": report["outputs"]["semantic_dir"],
        "pos_max_y": report["recommend"]["pos_max_y"],
        "pos_max_x": report["recommend"]["pos_max_x"],
        "combo_report_json": str(Path(report["outputs"]["out_dir"]) / "combo_report.json"),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

