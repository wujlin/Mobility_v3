from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.features.temporal import encode_route_temporal_2d


def _hour_and_dow_from_epoch(start_t: np.ndarray, *, tz_offset_hours: float) -> Tuple[np.ndarray, np.ndarray]:
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    start_t_local = start_t + int(float(tz_offset_hours) * 3600)
    seconds_per_day = 86400
    days_since_epoch = start_t_local // seconds_per_day
    seconds_in_day = start_t_local % seconds_per_day
    hour = (seconds_in_day // 3600).astype(np.int64)
    dow = ((days_since_epoch + 3) % 7).astype(np.int64)  # 0=Mon,...,6=Sun
    return hour, dow


def run_audit(*, npz_path: Path, temporal_mode: str, tz_offset_hours: float) -> Dict[str, object]:
    data = np.load(str(npz_path), allow_pickle=True)
    if "start_t" not in data:
        raise KeyError("npz missing required key: start_t")
    start_t = np.asarray(data["start_t"], dtype=np.int64).reshape(-1)
    n = int(start_t.shape[0])

    temporal, effective = encode_route_temporal_2d(start_t, tz_offset_hours=float(tz_offset_hours), mode=str(temporal_mode))

    start_t_min = int(np.min(start_t)) if n > 0 else 0
    start_t_max = int(np.max(start_t)) if n > 0 else 0
    start_t_zero_frac = float(np.mean(start_t == 0)) if n > 0 else 0.0

    # Only meaningful when start_t is epoch seconds.
    hour_hist = None
    dow_hist = None
    commute = None
    if str(effective) == "simple":
        hour, dow = _hour_and_dow_from_epoch(start_t, tz_offset_hours=float(tz_offset_hours))
        hour_hist = np.bincount(hour, minlength=24).astype(np.int64).tolist()
        dow_hist = np.bincount(dow, minlength=7).astype(np.int64).tolist()
        is_weekday = (dow <= 4)
        is_commute_hour = (hour >= 7) & (hour <= 9) | (hour >= 16) & (hour <= 18)
        commute_mask = is_weekday & is_commute_hour
        commute = {
            "weekday_commute_n": int(np.sum(commute_mask)),
            "weekday_commute_frac": float(np.mean(commute_mask)) if n > 0 else 0.0,
            "weekday_n": int(np.sum(is_weekday)),
        }

    meta = None
    if "meta" in data:
        try:
            meta = data["meta"].item()  # type: ignore[attr-defined]
        except Exception:
            meta = None

    return {
        "ok": True,
        "npz": str(npz_path),
        "N": int(n),
        "start_t": {
            "min": start_t_min,
            "max": start_t_max,
            "zero_frac": start_t_zero_frac,
        },
        "temporal": {
            "mode": str(temporal_mode),
            "tz_offset_hours": float(tz_offset_hours),
            "effective": str(effective),
            "sample_5": temporal[:5].astype(np.float32).tolist(),
        },
        "hour_hist_24": hour_hist,
        "dow_hist_7": dow_hist,
        "commute": commute,
        "meta": meta,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit whether route NPZ start_t enables temporal_mode=auto (epoch seconds).")
    p.add_argument("--npz", type=str, required=True, help="npz containing start_t (int64)")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--temporal_mode", type=str, choices=["auto", "simple", "zeros"], default="auto")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_audit(npz_path=Path(args.npz), temporal_mode=str(args.temporal_mode), tz_offset_hours=float(args.tz_offset_hours))
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Compact stdout for PI review.
    compact = {
        "ok": True,
        "npz": report["npz"],
        "N": report["N"],
        "temporal_effective": report["temporal"]["effective"],
        "start_t_min": report["start_t"]["min"],
        "start_t_max": report["start_t"]["max"],
        "start_t_zero_frac": report["start_t"]["zero_frac"],
    }
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()

