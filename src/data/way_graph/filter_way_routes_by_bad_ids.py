from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.data.way_graph.way_sequence_dataset import WayRoutes, load_way_routes_npz


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class FilterCfg:
    min_way_len: int = 3
    max_way_len: int = 160
    add_orig_route_id: bool = True


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_bad_ids(path: Path) -> np.ndarray:
    obj = _read_json(path)
    bad = obj.get("bad_route_ids", None)
    if bad is None:
        raise KeyError(f"{path} missing key: bad_route_ids")
    return np.asarray([int(x) for x in list(bad)], dtype=np.int64).reshape(-1)


def _compress_runs(ids: np.ndarray) -> np.ndarray:
    """
    ids: sorted int64 array
    returns: (R,2) array of [start_idx_in_ids, end_idx_in_ids] inclusive
    """
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if ids.size == 1:
        return np.asarray([[0, 0]], dtype=np.int64)
    diff = np.diff(ids)
    cut = np.nonzero(diff != 1)[0].astype(np.int64, copy=False) + 1
    starts = np.concatenate([np.asarray([0], dtype=np.int64), cut], axis=0)
    ends = np.concatenate([cut - 1, np.asarray([ids.size - 1], dtype=np.int64)], axis=0)
    return np.stack([starts, ends], axis=1).astype(np.int64, copy=False)


def filter_routes(
    routes: WayRoutes,
    *,
    bad_ids: np.ndarray,
    cfg: FilterCfg,
) -> Dict[str, object]:
    bad_ids = np.asarray(bad_ids, dtype=np.int64).reshape(-1)
    N = int(routes.way_seq_len.size)

    keep_len = (routes.way_seq_len >= int(cfg.min_way_len)) & (routes.way_seq_len <= int(cfg.max_way_len))
    is_bad = np.zeros((N,), dtype=bool)
    if bad_ids.size:
        bad_ids = bad_ids[(bad_ids >= 0) & (bad_ids < N)]
        is_bad[bad_ids] = True

    keep = keep_len & (~is_bad)
    kept = np.nonzero(keep)[0].astype(np.int64, copy=False)
    if kept.size == 0:
        return {"ok": False, "error": "no routes kept after filtering", "kept": kept}

    # Slice per-route fields (vectorized).
    out_len = routes.way_seq_len[kept].astype(np.int32, copy=False)
    out_ptr = np.zeros((int(kept.size) + 1,), dtype=np.int64)
    out_ptr[1:] = np.cumsum(out_len.astype(np.int64), axis=0)
    total = int(out_ptr[-1])
    out_idx = np.zeros((total,), dtype=np.int32)

    # Copy CSR segments in blocks of consecutive route ids to reduce Python overhead.
    runs = _compress_runs(kept)
    in_ptr = routes.way_seq_ptr.astype(np.int64, copy=False)
    in_len = routes.way_seq_len.astype(np.int64, copy=False)
    in_idx = routes.way_seq_idx.astype(np.int32, copy=False)

    for rs, re in runs.tolist():
        r0 = int(kept[int(rs)])
        r1 = int(kept[int(re)])
        src_s = int(in_ptr[r0])
        src_e = int(in_ptr[r1]) + int(in_len[r1])
        dst_s = int(out_ptr[int(rs)])
        dst_e = dst_s + int(src_e - src_s)
        exp_e = int(out_ptr[int(re) + 1])
        if dst_e != exp_e:
            raise RuntimeError(f"run copy mismatch: dst_e={dst_e} exp_e={exp_e} (rs={rs} re={re} r0={r0} r1={r1})")
        out_idx[dst_s:dst_e] = in_idx[src_s:src_e]

    payload: Dict[str, object] = {
        "ok": True,
        "kept_route_ids": kept,
        "out": {
            "way_osm_id": routes.way_osm_id.astype(np.int64, copy=False),
            "way_seq_ptr": out_ptr,
            "way_seq_idx": out_idx,
            "way_seq_len": out_len,
            "corridor_type": routes.corridor_type[kept].astype(np.int8, copy=False),
            "start_way": routes.start_way[kept].astype(np.int32, copy=False),
            "dest_way": routes.dest_way[kept].astype(np.int32, copy=False),
            "start_t": routes.start_t[kept].astype(np.int64, copy=False),
            "route_city": routes.route_city[kept].astype(np.int8, copy=False),
            "start_pos": routes.start_pos[kept].astype(np.float32, copy=False),
            "dest_pos": routes.dest_pos[kept].astype(np.float32, copy=False),
        },
    }
    if bool(cfg.add_orig_route_id):
        payload["out"]["orig_route_id"] = kept.astype(np.int64, copy=False)
    return payload


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filter way_routes.npz by a bad-route-id list (from audit_way_routes_quality --out_bad_json).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--bad_routes_json", type=Path, required=True, help="JSON containing bad_route_ids (see audit_way_routes_quality).")
    p.add_argument("--out_npz", type=Path, required=True)
    p.add_argument("--out_report_json", type=Path, default=None)
    p.add_argument("--min_way_len", type=int, default=None, help="Override min_way_len (default: read from bad_routes_json if present).")
    p.add_argument("--max_way_len", type=int, default=None, help="Override max_way_len (default: read from bad_routes_json if present).")
    p.add_argument("--no_orig_route_id", action="store_true", help="Do not store orig_route_id mapping in output npz.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    bad_obj = _read_json(Path(args.bad_routes_json))
    cfg = FilterCfg(
        min_way_len=int(args.min_way_len) if args.min_way_len is not None else int(bad_obj.get("min_way_len", 3)),
        max_way_len=int(args.max_way_len) if args.max_way_len is not None else int(bad_obj.get("max_way_len", 160)),
        add_orig_route_id=(not bool(args.no_orig_route_id)),
    )

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    bad_ids = _load_bad_ids(Path(args.bad_routes_json))
    rep = filter_routes(routes, bad_ids=bad_ids, cfg=cfg)
    if not bool(rep.get("ok", False)):
        raise SystemExit(f"[FATAL] {rep.get('error', 'unknown error')}")

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_payload = rep["out"]
    np.savez_compressed(out_npz, **out_payload)

    # Optional report (small).
    if args.out_report_json is not None:
        kept = rep["kept_route_ids"]
        out_report = {
            "ok": True,
            "task": "filter_way_routes_by_bad_ids",
            "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
            "inputs": {"way_routes_npz": str(args.way_routes_npz), "bad_routes_json": str(args.bad_routes_json)},
            "cfg": asdict(cfg),
            "thresholds": bad_obj.get("thresholds", None),
            "counts": {
                "n_in": int(routes.way_seq_len.size),
                "n_bad": int(bad_ids.size),
                "n_out": int(len(kept)),
            },
            "output": {"out_npz": str(out_npz)},
        }
        out_path = Path(args.out_report_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[in]  {args.way_routes_npz}")
    print(f"[bad] {args.bad_routes_json} (n_bad={int(bad_ids.size)})")
    print(f"[out] {out_npz}")


if __name__ == "__main__":
    main()

