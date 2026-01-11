from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.utils.geo_grid import BBox, GridSpec


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class DetroitSpec:
    H: int = 1024
    W: int = 1024
    min_lon: float = -83.25
    max_lon: float = -82.95
    min_lat: float = 42.25
    max_lat: float = 42.50

    def grid(self) -> GridSpec:
        return GridSpec(
            H=int(self.H),
            W=int(self.W),
            bbox=BBox(min_lon=float(self.min_lon), max_lon=float(self.max_lon), min_lat=float(self.min_lat), max_lat=float(self.max_lat)),
        )


def _summ(a: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(a)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def _hist(a: np.ndarray, *, bins: int) -> Dict[str, object]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"bins": [], "counts": []}
    b = int(bins)
    b = max(10, min(200, b))
    counts, edges = np.histogram(a, bins=b)
    return {"bins": [float(x) for x in edges.tolist()], "counts": [int(x) for x in counts.tolist()]}


def run_stats(
    *,
    npz_path: Path,
    out_json: Path,
    chunk_n: int,
    hist_bins: int,
    chord_small_m: float,
    city: str,
) -> Dict[str, object]:
    data = np.load(str(npz_path), allow_pickle=True)
    for k in ("start_pos", "targets", "dest_pos"):
        if k not in data:
            raise KeyError(f"npz missing required key: {k}")

    start = np.asarray(data["start_pos"], dtype=np.float32)
    targets = np.asarray(data["targets"], dtype=np.float32)
    dest = np.asarray(data["dest_pos"], dtype=np.float32)
    n = int(start.shape[0])
    f = int(targets.shape[1])

    # City grid spec for meters conversion.
    if str(city).lower() != "detroit":
        raise ValueError("Only city=detroit is supported for now (extend with bbox args when needed).")
    grid = DetroitSpec().grid()
    res_y_m, res_x_m = grid.resolution_m()

    chord_m = np.empty((n,), dtype=np.float32)
    total_m = np.empty((n,), dtype=np.float32)
    detour = np.empty((n,), dtype=np.float32)
    step_mean_m = np.empty((n,), dtype=np.float32)

    chunk = int(chunk_n)
    if chunk <= 0:
        raise ValueError("--chunk_n must be > 0")

    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        s = start[i0:i1].astype(np.float64, copy=False)
        d = dest[i0:i1].astype(np.float64, copy=False)
        chord = np.hypot((d[:, 1] - s[:, 1]) * float(res_x_m), (d[:, 0] - s[:, 0]) * float(res_y_m))
        chord = np.maximum(chord, 1e-6)
        chord_m[i0:i1] = chord.astype(np.float32, copy=False)

        # Reconstruct positions for path length in a streaming way.
        pts = np.concatenate([s[:, None, :], targets[i0:i1].astype(np.float64, copy=False)], axis=1)  # (B, F+1, 2)
        dy = (pts[:, 1:, 0] - pts[:, :-1, 0]) * float(res_y_m)
        dx = (pts[:, 1:, 1] - pts[:, :-1, 1]) * float(res_x_m)
        step = np.hypot(dx, dy)
        path = np.sum(step, axis=1)
        total_m[i0:i1] = path.astype(np.float32, copy=False)
        detour[i0:i1] = (path / chord).astype(np.float32, copy=False)
        step_mean_m[i0:i1] = (path / float(max(f, 1))).astype(np.float32, copy=False)

    small_thr = float(chord_small_m)
    frac_small = float(np.mean(chord_m < small_thr)) if n > 0 else 0.0

    meta = None
    if "meta" in data:
        try:
            meta = data["meta"].item()  # type: ignore[attr-defined]
        except Exception:
            meta = None

    report: Dict[str, object] = {
        "inputs": {"npz": str(npz_path)},
        "config": {"chunk_n": int(chunk_n), "hist_bins": int(hist_bins), "chord_small_m": float(chord_small_m), "city": str(city)},
        "stats": {"N": int(n), "F": int(f), "chord_small_frac": float(frac_small)},
        "grid": {"H": int(grid.H), "W": int(grid.W), "bbox": grid.bbox.__dict__, "res_m": {"x": float(res_x_m), "y": float(res_y_m)}},
        "window": {
            "chord_m": _summ(chord_m),
            "total_m": _summ(total_m),
            "detour_ratio": _summ(detour),
            "step_mean_m": _summ(step_mean_m),
            "hist": {
                "chord_m": _hist(chord_m, bins=int(hist_bins)),
                "detour_ratio": _hist(detour, bins=int(hist_bins)),
            },
        },
        "meta": meta,
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P0: window-level geometry stats from route windows npz (diagnose short/stop artifacts).")
    p.add_argument("--npz", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--city", type=str, default="detroit", help="Currently supports: detroit")
    p.add_argument("--chunk_n", type=int, default=8192)
    p.add_argument("--hist_bins", type=int, default=60)
    p.add_argument("--chord_small_m", type=float, default=200.0, help="Threshold to count 'short/near-stationary' windows (meters).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = run_stats(
        npz_path=Path(args.npz),
        out_json=Path(args.out_json),
        chunk_n=int(args.chunk_n),
        hist_bins=int(args.hist_bins),
        chord_small_m=float(args.chord_small_m),
        city=str(args.city),
    )
    compact = {
        "ok": True,
        "npz": report["inputs"]["npz"],
        "N": report["stats"]["N"],
        "F": report["stats"]["F"],
        "chord_m_p50": report["window"]["chord_m"]["p50"],
        "detour_p50": report["window"]["detour_ratio"]["p50"],
        "chord_small_frac": report["stats"]["chord_small_frac"],
        "out_json": str(Path(args.out_json).resolve()),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

