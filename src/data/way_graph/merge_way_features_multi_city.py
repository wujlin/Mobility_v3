"""
Merge way_features.npz from multiple cities (processed with different OSM pbf files).

The key insight: way_osm_id is globally unique, so we can match features by osm_id.
For ways that appear in multiple cities' pbf (unlikely but possible), we take the
first non-zero value.

Usage:
    python -m src.data.way_graph.merge_way_features_multi_city \
        --inputs city0/way_features.npz city1/way_features.npz \
        --way_routes_npz merged/way_routes.npz \
        --out_npz merged/way_features.npz
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np


TZ_SHANGHAI = timezone(timedelta(hours=8))


def _p(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def merge_features(inputs: List[Path], way_routes_npz: Path, out_npz: Path) -> Dict[str, object]:
    """
    Merge way_features from multiple cities.

    Each city's way_features.npz was built with the unified way_routes.npz vocab,
    but only filled features for ways that appear in that city's OSM pbf bbox.
    We merge by taking non-zero values from each city.
    """
    routes = np.load(str(way_routes_npz), allow_pickle=True)
    unified_way_osm_id = np.asarray(routes["way_osm_id"], dtype=np.int64).reshape(-1)
    M = int(unified_way_osm_id.size)

    # Initialize output arrays.
    out_len_m = np.zeros((M,), dtype=np.float32)
    out_center_y = np.zeros((M,), dtype=np.float32)
    out_center_x = np.zeros((M,), dtype=np.float32)
    out_dir_y = np.zeros((M,), dtype=np.float32)
    out_dir_x = np.zeros((M,), dtype=np.float32)
    out_tier = np.full((M,), 3, dtype=np.int64)  # default: unknown tier
    out_hw_code = np.zeros((M,), dtype=np.int64)
    filled = np.zeros((M,), dtype=bool)

    hw_vocab = None
    sem_keys = None
    out_semantic = None

    for inp in inputs:
        data = np.load(str(inp), allow_pickle=True)

        # Each city file should have same way_osm_id ordering (from unified routes).
        city_way_osm_id = np.asarray(data["way_osm_id"], dtype=np.int64).reshape(-1)
        if city_way_osm_id.size != M:
            raise ValueError(f"way_osm_id size mismatch: {inp} has {city_way_osm_id.size}, expected {M}")

        city_len = np.asarray(data["way_len_m"], dtype=np.float32).reshape(-1)
        city_center_y = np.asarray(data["way_center_y"], dtype=np.float32).reshape(-1)
        city_center_x = np.asarray(data["way_center_x"], dtype=np.float32).reshape(-1)
        city_dir_y = np.asarray(data["way_dir_y"], dtype=np.float32).reshape(-1)
        city_dir_x = np.asarray(data["way_dir_x"], dtype=np.float32).reshape(-1)
        city_tier = np.asarray(data["way_tier"], dtype=np.int64).reshape(-1)
        city_hw = np.asarray(data["way_highway_code"], dtype=np.int64).reshape(-1)

        # Get highway vocab from meta (should be same across cities).
        if "meta" in data.files:
            meta = data["meta"].item() if hasattr(data["meta"], "item") else data["meta"]
            if isinstance(meta, dict) and "vocab" in meta and "highway" in meta["vocab"]:
                if hw_vocab is None:
                    hw_vocab = meta["vocab"]["highway"]
            if isinstance(meta, dict) and "semantic" in meta and isinstance(meta["semantic"], dict):
                keys = meta["semantic"].get("keys")
                if isinstance(keys, (list, tuple)) and keys:
                    if sem_keys is None:
                        sem_keys = list(keys)
                    else:
                        if list(keys) != list(sem_keys):
                            raise ValueError(f"semantic keys mismatch across inputs: {inp} has {list(keys)} expected {list(sem_keys)}")

        city_sem = None
        if "way_semantic" in data.files:
            city_sem = np.asarray(data["way_semantic"], dtype=np.float32)
            if city_sem.ndim != 2 or city_sem.shape[0] != M:
                raise ValueError(f"Bad way_semantic shape in {inp}: {city_sem.shape} (expected {(M, 'C')})")
            if out_semantic is None:
                out_semantic = np.zeros((M, int(city_sem.shape[1])), dtype=np.float32)
            elif out_semantic.shape[1] != int(city_sem.shape[1]):
                raise ValueError(f"way_semantic dim mismatch: {inp} has C={city_sem.shape[1]} expected {out_semantic.shape[1]}")

        # Fill unfilled positions with this city's data.
        city_has_data = city_len > 0
        to_fill = city_has_data & ~filled

        out_len_m[to_fill] = city_len[to_fill]
        out_center_y[to_fill] = city_center_y[to_fill]
        out_center_x[to_fill] = city_center_x[to_fill]
        out_dir_y[to_fill] = city_dir_y[to_fill]
        out_dir_x[to_fill] = city_dir_x[to_fill]
        out_tier[to_fill] = city_tier[to_fill]
        out_hw_code[to_fill] = city_hw[to_fill]
        if out_semantic is not None:
            if city_sem is None:
                raise ValueError(f"Missing way_semantic in {inp} but previous inputs had it (need consistent features).")
            out_semantic[to_fill] = city_sem[to_fill]
        filled[to_fill] = True

    n_filled = int(np.sum(filled))
    n_missing = M - n_filled

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "merge_way_features_multi_city",
        "inputs": [str(p) for p in inputs],
        "vocab": {"highway": hw_vocab} if hw_vocab else {},
        "stats": {
            "n_way_vocab": int(M),
            "n_filled": int(n_filled),
            "n_missing": int(n_missing),
            "missing_frac": float(n_missing / max(1, M)),
            "way_len_m": {
                "p50": float(_p(out_len_m[filled], 50)) if n_filled else float("nan"),
                "p90": float(_p(out_len_m[filled], 90)) if n_filled else float("nan"),
            },
        },
    }
    if out_semantic is not None:
        meta["semantic"] = {"keys": list(sem_keys) if sem_keys else [], "dim": int(out_semantic.shape[1])}

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_kwargs = dict(
        way_osm_id=unified_way_osm_id,
        way_len_m=out_len_m,
        way_center_y=out_center_y,
        way_center_x=out_center_x,
        way_dir_y=out_dir_y,
        way_dir_x=out_dir_x,
        way_tier=out_tier,
        way_highway_code=out_hw_code,
        meta=meta,
    )
    if out_semantic is not None:
        out_kwargs["way_semantic"] = out_semantic
    np.savez_compressed(out_npz, **out_kwargs)
    return {"ok": True, "out_npz": str(out_npz), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge way_features.npz from multiple cities.")
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="List of way_features.npz to merge.")
    p.add_argument("--way_routes_npz", type=Path, required=True, help="Unified way_routes.npz (for way vocab).")
    p.add_argument("--out_npz", type=Path, required=True, help="Output merged way_features.npz.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = merge_features(
        inputs=[Path(p) for p in args.inputs],
        way_routes_npz=Path(args.way_routes_npz),
        out_npz=Path(args.out_npz),
    )
    meta = report["meta"]
    st = meta["stats"]
    compact = {
        "ok": True,
        "out_npz": report["out_npz"],
        "n_way_vocab": int(st["n_way_vocab"]),
        "n_filled": int(st["n_filled"]),
        "missing_frac": float(st["missing_frac"]),
        "way_len_p50_m": float(st["way_len_m"]["p50"]),
        "way_len_p90_m": float(st["way_len_m"]["p90"]),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
