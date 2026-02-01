#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Allow running as a file: `python tools/xxx.py ...` (so that `import src.*` works).
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz


TZ_SHANGHAI = timezone(timedelta(hours=8))


def _quantiles_int(x: np.ndarray, qs=(0.25, 0.5, 0.75, 0.95)) -> Dict[str, Optional[int]]:
    x = np.asarray(x, dtype=np.int64).reshape(-1)
    if x.size == 0:
        return {f"p{int(q*100):02d}": None for q in qs}
    return {f"p{int(q*100):02d}": int(np.quantile(x, float(q))) for q in qs}


def main() -> None:
    p = argparse.ArgumentParser(description="WayCASD route filtering stats (min_hops / max_way_len).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--min_hops", type=int, default=5)
    p.add_argument("--max_way_len", type=int, default=160)
    p.add_argument("--out_json", type=Path, default=None, help="Optional: save stats json.")
    args = p.parse_args()

    routes = load_way_routes_npz(Path(args.way_routes_npz))
    hops = routes.way_seq_len.astype(np.int64) - 1

    min_hops = int(args.min_hops)
    max_way_len = int(args.max_way_len)
    keep = (routes.way_seq_len >= (min_hops + 1)) & (routes.way_seq_len <= max_way_len)

    out = {
        "ok": True,
        "task": "waycasd_filter_routes_stats",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {"way_routes_npz": str(args.way_routes_npz), "min_hops": int(min_hops), "max_way_len": int(max_way_len)},
        "overall": {
            "n_total": int(hops.size),
            "n_keep": int(np.sum(keep)),
            "keep_frac": float(np.mean(keep)),
            "hops_quantiles_all": _quantiles_int(hops),
            "hops_quantiles_keep": _quantiles_int(hops[keep]),
        },
        "by_city": {},
    }

    for city in sorted(set(int(x) for x in routes.route_city.astype(np.int64).tolist())):
        m = routes.route_city.astype(np.int64) == int(city)
        if not bool(np.any(m)):
            continue
        out["by_city"][str(int(city))] = {
            "n_total": int(np.sum(m)),
            "n_keep": int(np.sum(m & keep)),
            "keep_frac": float(np.mean(keep[m])) if int(np.sum(m)) > 0 else float("nan"),
            "hops_quantiles_all": _quantiles_int(hops[m]),
            "hops_quantiles_keep": _quantiles_int(hops[m & keep]),
        }

    # Console summary (PI: 5min blocking).
    ov = out["overall"]
    print(
        f"[Filter] min_hops={min_hops} max_way_len={max_way_len} "
        f"keep={ov['n_keep']}/{ov['n_total']} ({ov['keep_frac']:.1%})"
    )
    print(f"[Hops all] {ov['hops_quantiles_all']}")
    print(f"[Hops keep] {ov['hops_quantiles_keep']}")
    for city, s in out["by_city"].items():
        print(f"[City {city}] keep={s['n_keep']}/{s['n_total']} ({s['keep_frac']:.1%})")
    if args.out_json is not None:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] saved: {out_json}")


if __name__ == "__main__":
    main()
