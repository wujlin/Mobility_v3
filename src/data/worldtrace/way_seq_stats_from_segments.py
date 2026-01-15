from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover
    pq = None


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Config:
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


def run_stats(*, segments_parquet: Path, out_json: Optional[Path], cfg: Config) -> Dict[str, object]:
    if pq is None:
        raise ModuleNotFoundError("pyarrow is required (pip/conda install pyarrow).")
    # Optional: older files may not have this column.
    table = pq.read_table(str(segments_parquet))
    if "osm_way_id" not in table.column_names:
        raise SystemExit(
            "segments.parquet 缺少 `osm_way_id` 列。\n"
            "请用更新后的脚本重新导出（src.data.worldtrace.build_detroit_segments 已支持写入 osm_way_id），例如：\n"
            "python -m src.data.worldtrace.build_detroit_segments --trajectory_zip ... --out_parquet ... --require_way_id"
        )
    cols = ["osm_way_id"]
    if "way_id_missing_ratio" in table.column_names:
        cols.append("way_id_missing_ratio")
    table = pq.read_table(str(segments_parquet), columns=cols)
    way_col = table.column("osm_way_id").to_pylist()
    miss_ratio = None
    if "way_id_missing_ratio" in table.column_names:
        miss_ratio = np.asarray(table.column("way_id_missing_ratio").to_numpy(), dtype=np.float64).reshape(-1)

    n = int(len(way_col))
    if int(cfg.limit_rows) > 0:
        n = min(n, int(cfg.limit_rows))
        way_col = way_col[:n]
        if miss_ratio is not None:
            miss_ratio = miss_ratio[:n]

    uniq_len_valid = np.zeros((n,), dtype=np.int32)
    uniq_len_all = np.zeros((n,), dtype=np.int32)
    any_valid = np.zeros((n,), dtype=np.uint8)

    for i in range(n):
        ways = way_col[i] or []
        ways_i = [int(x) for x in ways]
        dedup_all = _dedup_consecutive_int(ways_i)
        uniq_len_all[i] = int(len(dedup_all))
        valid = [w for w in ways_i if int(w) > 0]
        if valid:
            any_valid[i] = 1
            dedup_valid = _dedup_consecutive_int(valid)
            uniq_len_valid[i] = int(len(dedup_valid))
        else:
            uniq_len_valid[i] = 0

    report: Dict[str, object] = {
        "inputs": {"segments_parquet": str(segments_parquet)},
        "config": {"limit_rows": int(cfg.limit_rows)},
        "stats": {
            "n_segments": int(n),
            "n_segments_with_any_way": int(np.sum(any_valid.astype(np.int64))),
            "uniq_way_seq_len_valid": {
                "p50": _p(uniq_len_valid, 50),
                "p90": _p(uniq_len_valid, 90),
                "max": int(np.max(uniq_len_valid)) if n else 0,
            },
            "uniq_way_seq_len_all": {
                "p50": _p(uniq_len_all, 50),
                "p90": _p(uniq_len_all, 90),
                "max": int(np.max(uniq_len_all)) if n else 0,
            },
            "way_id_missing_ratio": (
                {
                    "p50": _p(miss_ratio, 50),
                    "p90": _p(miss_ratio, 90),
                }
                if miss_ratio is not None and miss_ratio.size
                else None
            ),
        },
        "meta": {"created_at": datetime.now(tz=TZ_SHANGHAI).isoformat()},
    }

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit osm_way_id sequences from WorldTrace segments.parquet (dedup consecutive).")
    p.add_argument("--segments_parquet", type=Path, required=True)
    p.add_argument("--out_json", type=Path, default=None)
    p.add_argument("--limit_rows", type=int, default=0, help="Debug: limit number of segments (0=no limit).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = run_stats(
        segments_parquet=Path(args.segments_parquet),
        out_json=(Path(args.out_json) if args.out_json is not None else None),
        cfg=Config(limit_rows=int(args.limit_rows)),
    )
    st = report["stats"]
    v = st["uniq_way_seq_len_valid"]
    a = st["uniq_way_seq_len_all"]
    print(f"[segments] {report['inputs']['segments_parquet']}")
    print(f"[N] {int(st['n_segments'])} with_any_way={int(st['n_segments_with_any_way'])}")
    print(f"[uniq_way_seq_len_valid] p50={float(v['p50']):.1f} p90={float(v['p90']):.1f} max={int(v['max'])}")
    print(f"[uniq_way_seq_len_all] p50={float(a['p50']):.1f} p90={float(a['p90']):.1f} max={int(a['max'])}")
    if st.get("way_id_missing_ratio") is not None:
        mr = st["way_id_missing_ratio"] or {}
        print(f"[way_id_missing_ratio] p50={float(mr.get('p50', float('nan'))):.3f} p90={float(mr.get('p90', float('nan'))):.3f}")
    if args.out_json is not None:
        print(f"[saved] {Path(args.out_json)}")


if __name__ == "__main__":
    main()
