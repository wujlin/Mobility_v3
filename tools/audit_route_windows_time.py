#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _tz(tz_offset_hours: float) -> timezone:
    return timezone(timedelta(seconds=int(round(float(tz_offset_hours) * 3600.0))))


def _iso_local(ts: int, *, tz_offset_hours: float) -> str:
    return datetime.fromtimestamp(int(ts), tz=_tz(float(tz_offset_hours))).isoformat()


def _topk_pairs(counts: np.ndarray, *, k: int) -> List[Tuple[int, int]]:
    c = np.asarray(counts, dtype=np.int64).reshape(-1)
    idx = np.argsort(c)[::-1][: int(k)]
    return [(int(i), int(c[int(i)])) for i in idx.tolist()]


def _format_hour_counts(counts: np.ndarray) -> str:
    c = np.asarray(counts, dtype=np.int64).reshape(24)
    return " ".join([f"{h:02d}:{int(c[h])}" for h in range(24)])


def _format_dow_counts(counts: np.ndarray) -> str:
    c = np.asarray(counts, dtype=np.int64).reshape(7)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return " ".join([f"{names[i]}:{int(c[i])}" for i in range(7)])


def _start_t_to_local_hour_dow(start_t: np.ndarray, *, tz_offset_hours: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match the exact convention used in src/features/temporal.py:
      - local time = start_t + tz_offset_hours
      - hour = (sec_in_day / 3600) in [0..23]
      - dow = Monday=0..Sunday=6, via (days_since_epoch + 3) % 7
    """
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    tz_s = int(round(float(tz_offset_hours) * 3600.0))
    start_t_local = start_t + np.int64(tz_s)
    sec_in_day = start_t_local % np.int64(86400)
    hour = (sec_in_day // np.int64(3600)).astype(np.int64, copy=False)
    days_since_epoch = start_t_local // np.int64(86400)
    dow = ((days_since_epoch + np.int64(3)) % np.int64(7)).astype(np.int64, copy=False)
    return hour, dow


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit start_t distribution in route-windows npz (hour-of-day / day-of-week).")
    p.add_argument("--npz", type=str, required=True, help="Route windows npz containing start_t (epoch seconds).")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0, help="Fixed timezone offset used by temporal encoding (Detroit/Columbus: -5).")
    p.add_argument("--max_n", type=int, default=None, help="Optional subsample for quick scan (keeps distribution roughly).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed when --max_n is set.")
    p.add_argument("--allow_non_unix", action="store_true", help="Allow start_t that does not look like Unix seconds (otherwise exit with a hint).")
    p.add_argument("--out_json", type=str, default=None, help="Optional output JSON path for the summary report.")
    p.add_argument("--out_png", type=str, default=None, help="Optional output PNG path (requires matplotlib).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    npz_path = Path(args.npz)
    data = np.load(str(npz_path), allow_pickle=True)
    if "start_t" not in data.files:
        raise SystemExit(f"Missing start_t in npz: {npz_path} (has {sorted(list(data.files))})")
    start_t = np.asarray(data["start_t"], dtype=np.int64).reshape(-1)
    if start_t.size <= 0:
        raise SystemExit(f"Empty start_t: {npz_path}")

    mx0 = int(np.max(start_t))
    if mx0 < 1_000_000_000 and not bool(args.allow_non_unix):
        raise SystemExit(
            f"start_t does not look like Unix epoch seconds (max={mx0}). "
            f"Re-dump gt_windows with `--use_epoch_start_t` (see src/data/worldtrace/dump_route_windows_from_segments.py). "
            f"Or re-run with --allow_non_unix for a debug-only scan."
        )

    if args.max_n is not None and int(args.max_n) > 0 and int(start_t.size) > int(args.max_n):
        rng = np.random.default_rng(int(args.seed))
        pick = rng.choice(int(start_t.size), size=int(args.max_n), replace=False)
        start_t = start_t[np.sort(pick)]

    hour, dow = _start_t_to_local_hour_dow(start_t, tz_offset_hours=float(args.tz_offset_hours))

    hour_counts = np.bincount(hour.astype(np.int64, copy=False), minlength=24).astype(np.int64, copy=False)
    dow_counts = np.bincount(dow.astype(np.int64, copy=False), minlength=7).astype(np.int64, copy=False)

    weekday_mask = dow < 5
    weekend_mask = ~weekday_mask
    hour_counts_weekday = np.bincount(hour[weekday_mask].astype(np.int64, copy=False), minlength=24).astype(np.int64, copy=False)
    hour_counts_weekend = np.bincount(hour[weekend_mask].astype(np.int64, copy=False), minlength=24).astype(np.int64, copy=False)

    heat = np.zeros((7, 24), dtype=np.int64)
    np.add.at(heat, (dow.astype(np.int64, copy=False), hour.astype(np.int64, copy=False)), 1)

    st_min = int(np.min(start_t))
    st_p50 = int(np.percentile(start_t.astype(np.float64), 50))
    st_p90 = int(np.percentile(start_t.astype(np.float64), 90))
    st_max = int(np.max(start_t))

    report: Dict[str, Any] = {
        "inputs": {"npz": str(npz_path), "max_n": (int(args.max_n) if args.max_n is not None else None), "seed": int(args.seed)},
        "config": {"tz_offset_hours": float(args.tz_offset_hours)},
        "stats": {
            "N": int(start_t.size),
            "start_t_min": st_min,
            "start_t_p50": st_p50,
            "start_t_p90": st_p90,
            "start_t_max": st_max,
            "start_t_min_local_iso": _iso_local(st_min, tz_offset_hours=float(args.tz_offset_hours)),
            "start_t_max_local_iso": _iso_local(st_max, tz_offset_hours=float(args.tz_offset_hours)),
        },
        "hour": {
            "counts": [int(x) for x in hour_counts.tolist()],
            "top4": [{"hour": int(h), "count": int(c)} for h, c in _topk_pairs(hour_counts, k=4)],
            "weekday_counts": [int(x) for x in hour_counts_weekday.tolist()],
            "weekend_counts": [int(x) for x in hour_counts_weekend.tolist()],
            "weekday_top4": [{"hour": int(h), "count": int(c)} for h, c in _topk_pairs(hour_counts_weekday, k=4)],
            "weekend_top4": [{"hour": int(h), "count": int(c)} for h, c in _topk_pairs(hour_counts_weekend, k=4)],
        },
        "dow": {
            "counts": [int(x) for x in dow_counts.tolist()],
            "top3": [{"dow": int(d), "count": int(c)} for d, c in _topk_pairs(dow_counts, k=3)],
        },
        "heat_7x24": heat.tolist(),
    }

    print(f"[npz] {npz_path}")
    print(f"[N] {int(start_t.size)} tz_offset_hours={float(args.tz_offset_hours):.2f} (fixed)")
    print(
        "[start_t] "
        f"min={st_min} ({report['stats']['start_t_min_local_iso']}), "
        f"p50={st_p50}, p90={st_p90}, "
        f"max={st_max} ({report['stats']['start_t_max_local_iso']})"
    )
    print(f"[hour_counts] {_format_hour_counts(hour_counts)}")
    print(f"[dow_counts] {_format_dow_counts(dow_counts)}")
    top_all = ", ".join([f"{h:02d}:{c}" for h, c in _topk_pairs(hour_counts, k=4)])
    top_wd = ", ".join([f"{h:02d}:{c}" for h, c in _topk_pairs(hour_counts_weekday, k=4)])
    top_we = ", ".join([f"{h:02d}:{c}" for h, c in _topk_pairs(hour_counts_weekend, k=4)])
    print(f"[hour_top4_all] {top_all}")
    print(f"[hour_top4_weekday] {top_wd}")
    print(f"[hour_top4_weekend] {top_we}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {out_json}")

    if args.out_png:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover
            print(f"[warn] matplotlib not available, skip --out_png ({type(e).__name__}: {e})")
        else:
            out_png = Path(args.out_png)
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig = plt.figure(figsize=(10, 8), dpi=150)
            gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.8, 1.2], hspace=0.35)

            ax0 = fig.add_subplot(gs[0, 0])
            ax0.bar(np.arange(24), hour_counts.astype(np.float64))
            ax0.set_title("start_t hour-of-day (local, fixed tz offset)")
            ax0.set_xlabel("hour")
            ax0.set_ylabel("count")
            ax0.set_xticks(np.arange(24))

            ax1 = fig.add_subplot(gs[1, 0])
            ax1.bar(np.arange(7), dow_counts.astype(np.float64))
            ax1.set_title("start_t day-of-week (Mon=0)")
            ax1.set_xlabel("dow")
            ax1.set_ylabel("count")
            ax1.set_xticks(np.arange(7))
            ax1.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

            ax2 = fig.add_subplot(gs[2, 0])
            im = ax2.imshow(heat.astype(np.float64), aspect="auto", origin="upper")
            ax2.set_title("heatmap: dow x hour (counts)")
            ax2.set_xlabel("hour")
            ax2.set_ylabel("dow (Mon..Sun)")
            ax2.set_xticks(np.arange(0, 24, 2))
            ax2.set_yticks(np.arange(7))
            ax2.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.02)

            fig.savefig(out_png)
            plt.close(fig)
            print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
