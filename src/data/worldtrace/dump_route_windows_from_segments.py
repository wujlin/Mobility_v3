from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover
    pq = None


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class DumpConfig:
    pred_len: int
    num_samples: int
    seed: int
    mode: str
    use_epoch_start_t: bool
    chunk_n: int


def _load_segments_columns(segments_parquet: Path) -> Tuple[list[list[int]], list[list[int]], list[list[int]]]:
    if pq is None:
        raise ModuleNotFoundError("pyarrow is required to read segments.parquet (please install pyarrow).")
    table = pq.read_table(str(segments_parquet), columns=["y", "x", "t"])
    y_col = table.column("y").to_pylist()
    x_col = table.column("x").to_pylist()
    t_col = table.column("t").to_pylist()
    if not (len(y_col) == len(x_col) == len(t_col)):
        raise RuntimeError("segments.parquet columns length mismatch")
    return y_col, x_col, t_col


def _build_ragged_buffers(
    *,
    y_list: list[list[int]],
    x_list: list[list[int]],
    t_list: list[list[int]],
    pred_len: int,
) -> Dict[str, object]:
    window_len = int(pred_len) + 1
    n_seg_total = int(len(y_list))
    seg_len = np.asarray([len(v) for v in y_list], dtype=np.int64)
    if seg_len.shape[0] != n_seg_total:
        raise RuntimeError("seg_len mismatch")
    win = np.maximum(seg_len - int(window_len) + 1, 0).astype(np.int64)
    keep = win > 0
    if not np.any(keep):
        raise RuntimeError("No segments can provide a full window (pred_len too long).")

    seg_orig = np.nonzero(keep)[0].astype(np.int64)
    y_keep = [y_list[int(i)] for i in seg_orig.tolist()]
    x_keep = [x_list[int(i)] for i in seg_orig.tolist()]
    t_keep = [t_list[int(i)] for i in seg_orig.tolist()]
    seg_len_keep = seg_len[keep].astype(np.int64, copy=False)
    win_keep = win[keep].astype(np.int64, copy=False)

    total_points = int(np.sum(seg_len_keep))
    pos = np.empty((total_points, 2), dtype=np.int32)
    ts = np.empty((total_points,), dtype=np.int64)
    ptr = np.zeros((int(len(seg_len_keep)) + 1,), dtype=np.int64)
    ptr[1:] = np.cumsum(seg_len_keep, dtype=np.int64)

    for i in range(int(len(seg_len_keep))):
        i0 = int(ptr[i])
        i1 = int(ptr[i + 1])
        yi = np.asarray(y_keep[i], dtype=np.int32)
        xi = np.asarray(x_keep[i], dtype=np.int32)
        ti = np.asarray(t_keep[i], dtype=np.int64)
        if yi.size != xi.size or yi.size != ti.size:
            raise RuntimeError(f"Segment length mismatch at idx={i}: y={yi.size} x={xi.size} t={ti.size}")
        pos[i0:i1, 0] = yi
        pos[i0:i1, 1] = xi
        ts[i0:i1] = ti

    return {
        "pos": pos,
        "ts": ts,
        "ptr": ptr,
        "seg_orig": seg_orig,
        "seg_len": seg_len_keep,
        "win": win_keep,
        "window_len": window_len,
        "n_seg_total": n_seg_total,
        "n_seg_used": int(len(seg_len_keep)),
        "total_windows_available": int(np.sum(win_keep)),
        "total_points_used": int(total_points),
    }


def _sample_windows_weighted(
    *,
    ptr: np.ndarray,  # (S+1,)
    win: np.ndarray,  # (S,)
    num_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    win = np.asarray(win, dtype=np.int64).reshape(-1)
    if not np.any(win > 0):
        raise RuntimeError("No valid segments (win==0).")
    w = win.astype(np.float64)
    w = w / max(float(np.sum(w)), 1.0)
    rng = np.random.default_rng(int(seed))
    seg_idx = rng.choice(win.shape[0], size=int(num_samples), replace=True, p=w).astype(np.int64, copy=False)
    # per-sample offset in [0, win[seg)-1]
    u = rng.random(int(num_samples), dtype=np.float64)
    off = np.floor(u * win[seg_idx].astype(np.float64)).astype(np.int64)
    start = ptr[seg_idx] + off
    return seg_idx, start


def run_dump(
    *,
    segments_parquet: Path,
    out_npz: Path,
    cfg: DumpConfig,
) -> Dict[str, object]:
    if int(cfg.pred_len) <= 0:
        raise ValueError("--pred_len must be > 0")
    if int(cfg.num_samples) <= 0:
        raise ValueError("--num_samples must be > 0")
    if int(cfg.chunk_n) <= 0:
        raise ValueError("--chunk_n must be > 0")
    if str(cfg.mode) != "weighted":
        raise ValueError("--mode currently only supports: weighted")

    y_list, x_list, t_list = _load_segments_columns(segments_parquet)
    rag = _build_ragged_buffers(y_list=y_list, x_list=x_list, t_list=t_list, pred_len=int(cfg.pred_len))
    pos = np.asarray(rag["pos"], dtype=np.int32)
    ts = np.asarray(rag["ts"], dtype=np.int64)
    ptr = np.asarray(rag["ptr"], dtype=np.int64)
    seg_orig = np.asarray(rag["seg_orig"], dtype=np.int64)
    win = np.asarray(rag["win"], dtype=np.int64)
    window_len = int(rag["window_len"])

    seg_idx, start_global = _sample_windows_weighted(ptr=ptr, win=win, num_samples=int(cfg.num_samples), seed=int(cfg.seed))
    traj_idx = seg_orig[seg_idx].astype(np.int64, copy=False)

    if bool(cfg.use_epoch_start_t):
        start_t = ts[start_global].astype(np.int64, copy=False)
    else:
        start_t = (start_global - ptr[seg_idx]).astype(np.int64, copy=False)

    n = int(cfg.num_samples)
    f = int(cfg.pred_len)
    start_pos = np.empty((n, 2), dtype=np.float32)
    dest_pos = np.empty((n, 2), dtype=np.float32)
    targets = np.empty((n, f, 2), dtype=np.float32)

    step = np.arange(window_len, dtype=np.int64)[None, :]
    chunk_n = int(cfg.chunk_n)
    for i0 in range(0, n, chunk_n):
        i1 = min(n, i0 + chunk_n)
        base = start_global[i0:i1].astype(np.int64, copy=False)[:, None]
        idx = base + step  # (B, window_len)
        wpos = pos[idx]  # (B, window_len, 2) int32
        wpos_f32 = wpos.astype(np.float32, copy=False)
        start_pos[i0:i1] = wpos_f32[:, 0, :]
        targets[i0:i1] = wpos_f32[:, 1:, :]
        dest_pos[i0:i1] = wpos_f32[:, -1, :]

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "segments_parquet": str(segments_parquet),
        "config": {
            "pred_len": int(cfg.pred_len),
            "num_samples": int(cfg.num_samples),
            "seed": int(cfg.seed),
            "mode": str(cfg.mode),
            "use_epoch_start_t": bool(cfg.use_epoch_start_t),
            "chunk_n": int(cfg.chunk_n),
        },
        "stats": {
            "n_seg_total": int(rag["n_seg_total"]),
            "n_seg_used": int(rag["n_seg_used"]),
            "total_points_used": int(rag["total_points_used"]),
            "window_len": int(window_len),
            "total_windows_available": int(rag["total_windows_available"]),
        },
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        start_pos=start_pos.astype(np.float32, copy=False),
        targets=targets.astype(np.float32, copy=False),
        dest_pos=dest_pos.astype(np.float32, copy=False),
        traj_idx=traj_idx.astype(np.int64, copy=False),
        start_t=start_t.astype(np.int64, copy=False),
        meta=meta,
    )
    return {"ok": True, "out_npz": str(out_npz), "N": int(n), "F": int(f), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump route-windows npz from WorldTrace segments.parquet (CPU-only).")
    p.add_argument("--segments_parquet", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--pred_len", type=int, default=256, help="F (number of future positions). Window length is F+1.")
    p.add_argument("--num_samples", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", type=str, default="weighted", choices=["weighted"], help="Sample windows weighted by available windows per segment.")
    p.add_argument("--use_epoch_start_t", action="store_true", help="Store epoch seconds as start_t (from segments.t). Default: store window offset.")
    p.add_argument("--chunk_n", type=int, default=4096, help="Batch windows extraction to control peak memory.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = DumpConfig(
        pred_len=int(args.pred_len),
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        mode=str(args.mode),
        use_epoch_start_t=bool(args.use_epoch_start_t),
        chunk_n=int(args.chunk_n),
    )
    report = run_dump(
        segments_parquet=Path(args.segments_parquet),
        out_npz=Path(args.out_npz),
        cfg=cfg,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

