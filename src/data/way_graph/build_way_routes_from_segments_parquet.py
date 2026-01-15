from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import pyarrow.parquet as pq  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pq = None


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Config:
    route_city: int
    min_seq_len: int
    limit_rows: int


def _p(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def _dedup_consecutive_int(seq: List[int]) -> List[int]:
    out: List[int] = []
    last: Optional[int] = None
    for x in seq:
        xi = int(x)
        if last is None or xi != int(last):
            out.append(xi)
            last = xi
    return out


def build_routes(*, segments_parquet: Path, out_npz: Path, cfg: Config) -> Dict[str, object]:
    if pq is None:
        raise ModuleNotFoundError("pyarrow is required (pip/conda install pyarrow).")
    need = {"osm_way_id", "t", "y", "x"}
    table = pq.read_table(str(segments_parquet))
    missing = sorted(list(need - set(table.column_names)))
    if missing:
        raise SystemExit(f"{segments_parquet} 缺少列: {missing}")

    cols = ["osm_way_id", "t", "y", "x"]
    if "way_id_missing_ratio" in table.column_names:
        cols.append("way_id_missing_ratio")
    table = pq.read_table(str(segments_parquet), columns=cols)

    way_col = table.column("osm_way_id").to_pylist()
    t_col = table.column("t").to_pylist()
    y_col = table.column("y").to_pylist()
    x_col = table.column("x").to_pylist()
    miss_ratio = None
    if "way_id_missing_ratio" in table.column_names:
        miss_ratio = np.asarray(table.column("way_id_missing_ratio").to_numpy(), dtype=np.float64).reshape(-1)

    n_rows = int(len(way_col))
    if int(cfg.limit_rows) > 0:
        n_rows = min(n_rows, int(cfg.limit_rows))
        way_col = way_col[:n_rows]
        t_col = t_col[:n_rows]
        y_col = y_col[:n_rows]
        x_col = x_col[:n_rows]
        if miss_ratio is not None:
            miss_ratio = miss_ratio[:n_rows]

    # First pass: extract per-route raw way_id sequences (dedup consecutive), and collect vocab.
    seqs: List[List[int]] = []
    start_t: List[int] = []
    start_pos: List[List[float]] = []
    dest_pos: List[List[float]] = []
    way_vocab: set[int] = set()

    dropped_empty = 0
    dropped_short = 0

    min_seq_len = int(cfg.min_seq_len)
    for i in range(n_rows):
        ways0 = way_col[i] or []
        # Keep only valid way ids.
        ways = [int(w) for w in ways0 if int(w) > 0]
        if not ways:
            dropped_empty += 1
            continue
        dedup = _dedup_consecutive_int(ways)
        if len(dedup) < min_seq_len:
            dropped_short += 1
            continue

        ts = t_col[i] or []
        ys = y_col[i] or []
        xs = x_col[i] or []
        if not (ts and ys and xs):
            dropped_empty += 1
            continue
        seqs.append(dedup)
        start_t.append(int(ts[0]))
        start_pos.append([float(ys[0]), float(xs[0])])
        dest_pos.append([float(ys[-1]), float(xs[-1])])
        way_vocab.update(dedup)

    way_osm_id = np.asarray(sorted(list(way_vocab)), dtype=np.int64).reshape(-1)
    way_to_idx = {int(w): int(i) for i, w in enumerate(way_osm_id.tolist())}

    # Second pass: encode to CSR.
    N = int(len(seqs))
    ptr = np.zeros((N + 1,), dtype=np.int64)
    lens = np.zeros((N,), dtype=np.int32)
    start_way = np.zeros((N,), dtype=np.int32)
    dest_way = np.zeros((N,), dtype=np.int32)

    flat: List[int] = []
    for i, s in enumerate(seqs):
        enc = [way_to_idx[int(w)] for w in s]
        L = int(len(enc))
        lens[i] = np.int32(L)
        start_way[i] = np.int32(enc[0])
        dest_way[i] = np.int32(enc[-1])
        flat.extend(enc)
        ptr[i + 1] = np.int64(len(flat))

    way_seq_idx = np.asarray(flat, dtype=np.int32)
    start_t_arr = np.asarray(start_t, dtype=np.int64).reshape(-1)
    start_pos_arr = np.asarray(start_pos, dtype=np.float32).reshape(-1, 2)
    dest_pos_arr = np.asarray(dest_pos, dtype=np.float32).reshape(-1, 2)
    route_city_arr = np.full((N,), int(cfg.route_city), dtype=np.int8)
    corridor_type = np.full((N,), -1, dtype=np.int8)  # filled later (optional)

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "build_way_routes_from_segments_parquet",
        "inputs": {"segments_parquet": str(segments_parquet)},
        "config": {"route_city": int(cfg.route_city), "min_seq_len": int(cfg.min_seq_len), "limit_rows": int(cfg.limit_rows)},
        "stats": {
            "n_rows_in": int(n_rows),
            "n_routes": int(N),
            "n_way_vocab": int(way_osm_id.size),
            "dropped_empty": int(dropped_empty),
            "dropped_short": int(dropped_short),
            "way_seq_len": {"p50": _p(lens, 50), "p90": _p(lens, 90), "max": int(np.max(lens) if lens.size else 0)},
            "way_id_missing_ratio": (
                {"p50": _p(miss_ratio, 50), "p90": _p(miss_ratio, 90)} if miss_ratio is not None and miss_ratio.size else None
            ),
        },
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        way_osm_id=way_osm_id,
        way_seq_ptr=ptr,
        way_seq_idx=way_seq_idx,
        way_seq_len=lens,
        start_way=start_way,
        dest_way=dest_way,
        start_t=start_t_arr,
        route_city=route_city_arr,
        corridor_type=corridor_type,
        start_pos=start_pos_arr,
        dest_pos=dest_pos_arr,
        meta=meta,
    )
    return {"ok": True, "out_npz": str(out_npz), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build way-level route sequences from WorldTrace segments parquet (osm_way_id).")
    p.add_argument("--segments_parquet", type=Path, required=True)
    p.add_argument("--out_npz", type=Path, required=True)
    p.add_argument("--route_city", type=int, default=0)
    p.add_argument("--min_seq_len", type=int, default=2, help="Drop routes whose deduped way sequence length < this.")
    p.add_argument("--limit_rows", type=int, default=0, help="Debug: limit number of rows (0=no limit).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = build_routes(
        segments_parquet=Path(args.segments_parquet),
        out_npz=Path(args.out_npz),
        cfg=Config(route_city=int(args.route_city), min_seq_len=int(args.min_seq_len), limit_rows=int(args.limit_rows)),
    )
    meta = report["meta"]
    st = meta["stats"]
    wl = st["way_seq_len"]
    compact = {
        "ok": True,
        "out_npz": report["out_npz"],
        "n_routes": int(st["n_routes"]),
        "n_way_vocab": int(st["n_way_vocab"]),
        "way_seq_len_p50": float(wl["p50"]),
        "way_seq_len_p90": float(wl["p90"]),
        "way_seq_len_max": int(wl["max"]),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

