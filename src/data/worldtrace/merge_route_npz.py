from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class MergeConfig:
    offset_y: float
    offset_x: float
    traj_idx_offset: int


def _load_route_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    need = {"start_pos", "targets"}
    if not need.issubset(set(data.files)):
        raise ValueError(f"npz missing keys: {sorted(list(need - set(data.files)))}. got={sorted(list(data.files))}")

    start_pos = np.asarray(data["start_pos"], dtype=np.float32).reshape(-1, 2)
    targets = np.asarray(data["targets"], dtype=np.float32)
    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError(f"Bad targets shape in {path}: {targets.shape} (expected N,F,2)")
    n = int(start_pos.shape[0])
    if targets.shape[0] != n:
        raise ValueError(f"Bad npz {path}: start_pos N={n} != targets N={targets.shape[0]}")

    if "dest_pos" in data.files:
        dest_pos = np.asarray(data["dest_pos"], dtype=np.float32).reshape(-1, 2)
        if dest_pos.shape[0] != n:
            raise ValueError(f"Bad npz {path}: dest_pos N={dest_pos.shape[0]} != {n}")
    else:
        dest_pos = targets[:, -1, :].astype(np.float32, copy=False)

    traj_idx = np.asarray(data["traj_idx"], dtype=np.int64).reshape(-1) if "traj_idx" in data.files else np.arange(n, dtype=np.int64)
    start_t = np.asarray(data["start_t"], dtype=np.int64).reshape(-1) if "start_t" in data.files else np.zeros((n,), dtype=np.int64)
    if traj_idx.shape[0] != n or start_t.shape[0] != n:
        raise ValueError(f"Bad npz {path}: traj_idx/start_t N mismatch (traj_idx={traj_idx.shape[0]}, start_t={start_t.shape[0]}, expected={n})")

    meta = data["meta"].item() if ("meta" in data.files and isinstance(data["meta"], np.ndarray) and data["meta"].shape == ()) else None
    return {
        "start_pos": start_pos.astype(np.float32, copy=False),
        "targets": targets.astype(np.float32, copy=False),
        "dest_pos": dest_pos.astype(np.float32, copy=False),
        "traj_idx": traj_idx.astype(np.int64, copy=False),
        "start_t": start_t.astype(np.int64, copy=False),
        "meta": meta,
    }


def _apply_offset_xy(a: np.ndarray, *, dy: float, dx: float) -> np.ndarray:
    out = np.asarray(a, dtype=np.float32, copy=True)
    out[..., 0] += float(dy)
    out[..., 1] += float(dx)
    return out.astype(np.float32, copy=False)


def _range_stats(pos: np.ndarray) -> Dict[str, float]:
    pos = np.asarray(pos, dtype=np.float32).reshape(-1, 2)
    return {
        "y_min": float(np.min(pos[:, 0])),
        "y_p50": float(np.percentile(pos[:, 0], 50)),
        "y_max": float(np.max(pos[:, 0])),
        "x_min": float(np.min(pos[:, 1])),
        "x_p50": float(np.percentile(pos[:, 1], 50)),
        "x_max": float(np.max(pos[:, 1])),
    }


def merge_two(
    *,
    a_npz: Path,
    b_npz: Path,
    out_npz: Path,
    cfg: MergeConfig,
    city_a: Optional[str],
    city_b: Optional[str],
) -> Dict[str, object]:
    a = _load_route_npz(a_npz)
    b = _load_route_npz(b_npz)
    fa = int(a["targets"].shape[1])
    fb = int(b["targets"].shape[1])
    if fa != fb:
        raise ValueError(f"F mismatch: a.F={fa} b.F={fb}")

    b_start = _apply_offset_xy(b["start_pos"], dy=float(cfg.offset_y), dx=float(cfg.offset_x))
    b_dest = _apply_offset_xy(b["dest_pos"], dy=float(cfg.offset_y), dx=float(cfg.offset_x))
    b_tgt = _apply_offset_xy(b["targets"], dy=float(cfg.offset_y), dx=float(cfg.offset_x))

    b_traj = np.asarray(b["traj_idx"], dtype=np.int64) + np.int64(cfg.traj_idx_offset)

    start_pos = np.concatenate([a["start_pos"], b_start], axis=0).astype(np.float32, copy=False)
    dest_pos = np.concatenate([a["dest_pos"], b_dest], axis=0).astype(np.float32, copy=False)
    targets = np.concatenate([a["targets"], b_tgt], axis=0).astype(np.float32, copy=False)
    traj_idx = np.concatenate([a["traj_idx"], b_traj], axis=0).astype(np.int64, copy=False)
    start_t = np.concatenate([a["start_t"], b["start_t"]], axis=0).astype(np.int64, copy=False)

    # Collision check (cheap at this scale).
    inter = np.intersect1d(np.unique(a["traj_idx"]), np.unique(b_traj))
    if int(inter.size) > 0:
        raise ValueError(f"traj_idx collision after offset: {int(inter.size)} overlaps; increase --b_traj_idx_offset")

    poly_a = np.concatenate([a["start_pos"][:, None, :], a["targets"]], axis=1)
    poly_b = np.concatenate([b_start[:, None, :], b_tgt], axis=1)
    poly_all = np.concatenate([poly_a.reshape(-1, 2), poly_b.reshape(-1, 2)], axis=0)

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "merge_route_npz",
        "inputs": {
            "a_npz": str(a_npz),
            "b_npz": str(b_npz),
            "city_a": (str(city_a) if city_a else None),
            "city_b": (str(city_b) if city_b else None),
        },
        "config": {"b_offset_y": float(cfg.offset_y), "b_offset_x": float(cfg.offset_x), "b_traj_idx_offset": int(cfg.traj_idx_offset)},
        "stats": {
            "F": int(fa),
            "N_a": int(a["start_pos"].shape[0]),
            "N_b": int(b_start.shape[0]),
            "N_total": int(start_pos.shape[0]),
            "poly_a": _range_stats(poly_a),
            "poly_b_shifted": _range_stats(poly_b),
            "poly_total": _range_stats(poly_all),
        },
        "sources_meta": {"a_meta": a.get("meta"), "b_meta": b.get("meta")},
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        start_pos=start_pos,
        targets=targets,
        dest_pos=dest_pos,
        traj_idx=traj_idx,
        start_t=start_t,
        meta=meta,
    )

    return {"ok": True, "out_npz": str(out_npz), "N": int(start_pos.shape[0]), "F": int(fa), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge two route npz files by shifting B in (y,x) and offsetting traj_idx.")
    p.add_argument("--a_npz", type=str, required=True)
    p.add_argument("--b_npz", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--city_a", type=str, default=None)
    p.add_argument("--city_b", type=str, default=None)
    p.add_argument("--b_offset_y", type=float, default=0.0)
    p.add_argument("--b_offset_x", type=float, default=0.0)
    p.add_argument("--b_traj_idx_offset", type=int, default=1_000_000_000)
    p.add_argument("--out_report_json", type=str, default=None, help="Optional report.json path (default: out_npz + .report.json)")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_npz = Path(args.out_npz)
    out_report = Path(args.out_report_json) if args.out_report_json else Path(str(out_npz) + ".report.json")
    cfg = MergeConfig(offset_y=float(args.b_offset_y), offset_x=float(args.b_offset_x), traj_idx_offset=int(args.b_traj_idx_offset))
    report = merge_two(
        a_npz=Path(args.a_npz),
        b_npz=Path(args.b_npz),
        out_npz=out_npz,
        cfg=cfg,
        city_a=(str(args.city_a) if args.city_a else None),
        city_b=(str(args.city_b) if args.city_b else None),
    )
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    compact = {
        "ok": True,
        "out_npz": report["out_npz"],
        "N": report["N"],
        "F": report["F"],
        "b_offset_y": float(args.b_offset_y),
        "b_offset_x": float(args.b_offset_x),
        "b_traj_idx_offset": int(args.b_traj_idx_offset),
        "report_json": str(out_report),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

