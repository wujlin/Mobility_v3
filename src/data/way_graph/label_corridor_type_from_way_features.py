from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import numpy as np


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Config:
    dominant_thr: float


def label(*, way_routes_npz: Path, way_features_npz: Path, out_npz: Path, cfg: Config) -> Dict[str, object]:
    routes = np.load(str(way_routes_npz), allow_pickle=True)
    feats = np.load(str(way_features_npz), allow_pickle=True)
    need_r = {"way_seq_ptr", "way_seq_idx", "way_seq_len", "way_osm_id"}
    need_f = {"way_osm_id", "way_tier"}
    miss_r = sorted(list(need_r - set(routes.files)))
    miss_f = sorted(list(need_f - set(feats.files)))
    if miss_r:
        raise ValueError(f"way_routes_npz missing keys: {miss_r}")
    if miss_f:
        raise ValueError(f"way_features_npz missing keys: {miss_f}")

    way_osm_id_r = np.asarray(routes["way_osm_id"], dtype=np.int64).reshape(-1)
    way_osm_id_f = np.asarray(feats["way_osm_id"], dtype=np.int64).reshape(-1)
    if way_osm_id_r.shape[0] != way_osm_id_f.shape[0] or not np.all(way_osm_id_r == way_osm_id_f):
        raise ValueError("way_routes_npz and way_features_npz vocab mismatch (way_osm_id). Build features from the same routes.")

    ptr = np.asarray(routes["way_seq_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(routes["way_seq_idx"], dtype=np.int64).reshape(-1)
    lens = np.asarray(routes["way_seq_len"], dtype=np.int64).reshape(-1)
    way_tier = np.asarray(feats["way_tier"], dtype=np.int64).reshape(-1)

    N = int(lens.size)
    corridor = np.full((N,), 3, dtype=np.int8)
    thr = float(cfg.dominant_thr)
    thr = max(0.0, min(thr, 1.0))

    for r in range(N):
        L = int(lens[r])
        if L <= 0:
            corridor[r] = 3
            continue
        s = int(ptr[r])
        e = s + L
        seq = idx[s:e]
        tiers = way_tier[np.clip(seq.astype(np.int64), 0, way_tier.shape[0] - 1)]
        c0 = int(np.sum(tiers == 0))
        c1 = int(np.sum(tiers == 1))
        c2 = int(np.sum(tiers == 2))
        tot = max(1, int(L))
        if (c0 / tot) >= thr:
            corridor[r] = 0
        elif (c1 / tot) >= thr:
            corridor[r] = 1
        elif (c2 / tot) >= thr:
            corridor[r] = 2
        else:
            corridor[r] = 3

    # Copy all arrays, override corridor_type.
    payload = {k: routes[k] for k in routes.files if k != "corridor_type"}
    payload["corridor_type"] = corridor
    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "label_corridor_type_from_way_features",
        "inputs": {"way_routes_npz": str(way_routes_npz), "way_features_npz": str(way_features_npz)},
        "config": {"dominant_thr": float(thr)},
        "stats": {
            "N": int(N),
            "corridor_type_counts": np.bincount(corridor.astype(np.int64), minlength=4).astype(np.int64).tolist(),
        },
    }
    payload["meta_corridor"] = meta

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)
    return {"ok": True, "out_npz": str(out_npz), "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Label corridor_type for way_routes using way_tier features.")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--out_npz", type=Path, required=True)
    p.add_argument("--dominant_thr", type=float, default=0.5, help="Dominant tier fraction threshold (default 0.5).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = label(
        way_routes_npz=Path(args.way_routes_npz),
        way_features_npz=Path(args.way_features_npz),
        out_npz=Path(args.out_npz),
        cfg=Config(dominant_thr=float(args.dominant_thr)),
    )
    c = report["meta"]["stats"]["corridor_type_counts"]
    compact = {"ok": True, "out_npz": report["out_npz"], "corridor_type_counts": c}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

