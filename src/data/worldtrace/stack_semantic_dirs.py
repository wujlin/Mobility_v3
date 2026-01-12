from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class StackConfig:
    offset_y: int
    offset_x: int
    fill_value: float
    merge_op: str


def _parse_files(s: str) -> Tuple[str, ...]:
    items = [x.strip() for x in str(s).split(",") if str(x).strip()]
    if not items:
        raise ValueError("--files must be a non-empty comma-separated list.")
    return tuple(items)


def _load_2d(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing raster: {path}")
    a = np.load(path).astype(np.float32, copy=False)
    if a.ndim != 2:
        raise ValueError(f"Bad raster shape in {path}: {a.shape} (expected H,W)")
    return a


def _write_raster(path: Path, a: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(a, dtype=np.float32))


def _place(out: np.ndarray, patch: np.ndarray, *, y0: int, x0: int, op: str) -> None:
    h, w = int(patch.shape[0]), int(patch.shape[1])
    view = out[y0 : y0 + h, x0 : x0 + w]
    if op == "max":
        np.maximum(view, patch, out=view)
    elif op == "overwrite":
        view[...] = patch
    else:
        raise ValueError(f"Unknown merge_op: {op}")


def stack_two_dirs(
    *,
    a_dir: Path,
    b_dir: Path,
    out_dir: Path,
    files: Tuple[str, ...],
    cfg: StackConfig,
    city_a: Optional[str],
    city_b: Optional[str],
) -> Dict[str, object]:
    rasters_a = {name: _load_2d(a_dir / name) for name in files}
    rasters_b = {name: _load_2d(b_dir / name) for name in files}

    # Compute output shape.
    H = 0
    W = 0
    for name in files:
        ha, wa = rasters_a[name].shape
        hb, wb = rasters_b[name].shape
        H = max(H, int(ha), int(cfg.offset_y) + int(hb))
        W = max(W, int(wa), int(cfg.offset_x) + int(wb))
    if H <= 0 or W <= 0:
        raise RuntimeError("Invalid output shape computed for stacked rasters.")

    out_dir.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Dict[str, object]] = {}

    for name in files:
        a = rasters_a[name]
        b = rasters_b[name]
        out = np.full((H, W), float(cfg.fill_value), dtype=np.float32)
        _place(out, a, y0=0, x0=0, op=str(cfg.merge_op))
        _place(out, b, y0=int(cfg.offset_y), x0=int(cfg.offset_x), op=str(cfg.merge_op))
        _write_raster(out_dir / name, out)
        stats[name] = {
            "a_shape": [int(a.shape[0]), int(a.shape[1])],
            "b_shape": [int(b.shape[0]), int(b.shape[1])],
            "out_shape": [int(out.shape[0]), int(out.shape[1])],
            "out_min": float(np.min(out)),
            "out_max": float(np.max(out)),
        }

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "stack_semantic_dirs",
        "inputs": {"a_dir": str(a_dir), "b_dir": str(b_dir), "city_a": (str(city_a) if city_a else None), "city_b": (str(city_b) if city_b else None)},
        "config": {"b_offset_y": int(cfg.offset_y), "b_offset_x": int(cfg.offset_x), "fill_value": float(cfg.fill_value), "merge_op": str(cfg.merge_op), "files": list(files)},
        "stats": {"out_H": int(H), "out_W": int(W), "files": stats},
    }
    (out_dir / "stack_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "out_dir": str(out_dir), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stack two city semantic dirs into one canvas by shifting B in (y,x).")
    p.add_argument("--a_dir", type=str, required=True)
    p.add_argument("--b_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--city_a", type=str, default=None)
    p.add_argument("--city_b", type=str, default=None)
    p.add_argument("--b_offset_y", type=int, required=True)
    p.add_argument("--b_offset_x", type=int, default=0)
    p.add_argument(
        "--files",
        type=str,
        default="osm_road_prob.npy,osm_road_prob_major.npy,osm_road_prob_minor.npy,osm_road_prob_service.npy",
        help="Comma-separated 2D .npy filenames to stack.",
    )
    p.add_argument("--fill_value", type=float, default=0.0, help="Fill value for empty canvas regions.")
    p.add_argument("--merge_op", type=str, choices=["max", "overwrite"], default="max")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    files = _parse_files(args.files)
    cfg = StackConfig(offset_y=int(args.b_offset_y), offset_x=int(args.b_offset_x), fill_value=float(args.fill_value), merge_op=str(args.merge_op))
    report = stack_two_dirs(
        a_dir=Path(args.a_dir),
        b_dir=Path(args.b_dir),
        out_dir=Path(args.out_dir),
        files=files,
        cfg=cfg,
        city_a=(str(args.city_a) if args.city_a else None),
        city_b=(str(args.city_b) if args.city_b else None),
    )
    compact = {
        "ok": True,
        "out_dir": report["out_dir"],
        "b_offset_y": int(args.b_offset_y),
        "b_offset_x": int(args.b_offset_x),
        "files": list(files),
        "stack_meta_json": str(Path(args.out_dir) / "stack_meta.json"),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

