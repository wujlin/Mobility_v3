#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Allow running as a file: `python tools/xxx.py ...` (so that `import src.*` works).
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.plot_style import FIGSIZE_HALF, OKABE_ITO, add_panel_label, paper_style, save_figure


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[FATAL] file not found: {path}")
    if not path.is_file():
        raise SystemExit(f"[FATAL] not a file: {path}")


def _weighted_success_rate(cells: Dict[str, Any]) -> float:
    tot = 0.0
    ok = 0.0
    for v in cells.values():
        if not isinstance(v, dict):
            continue
        n = float(v.get("n", 0.0))
        s = float(v.get("success_rate", float("nan")))
        if not np.isfinite(n) or n <= 0 or not np.isfinite(s):
            continue
        tot += n
        ok += n * s
    return float(ok / tot) if tot > 0 else float("nan")


@dataclass(frozen=True)
class Point:
    k: int
    succ: float
    path: str


def _load_points(paths: List[Path], *, key: str) -> List[Point]:
    out: List[Point] = []
    for p in paths:
        _require_file(p)
        obj = _read_json(p)
        cfg = obj.get("cfg", {})
        K = int(cfg.get("n_samples_per_route", 1))
        overall = obj.get("overall", {})
        part = overall.get(str(key), {})
        cells = part.get("cells", {}) if isinstance(part, dict) else {}
        succ = _weighted_success_rate(cells if isinstance(cells, dict) else {})
        out.append(Point(k=K, succ=succ, path=str(p)))
    out.sort(key=lambda x: int(x.k))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Plot best-of-K curve from way_casd_binned_eval.json outputs.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--title", type=str, default="best-of-K curve")
    p.add_argument("--key", choices=["beam", "greedy"], default="beam")
    p.add_argument("--json", type=Path, action="append", default=[], help="One or more binned_eval json paths.")
    args = p.parse_args()

    if not args.json:
        raise SystemExit("[FATAL] need at least one --json")

    pts = _load_points(list(args.json), key=str(args.key))
    if not pts:
        raise SystemExit("[FATAL] no points loaded")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "bestofk_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "success_rate", "json_path"])
        for pt in pts:
            w.writerow([int(pt.k), float(pt.succ), str(pt.path)])

    # Plot
    ks = np.asarray([pt.k for pt in pts], dtype=np.int64)
    ys = np.asarray([pt.succ * 100.0 for pt in pts], dtype=np.float64)

    with paper_style():
        fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
        ax.plot(ks, ys, marker="o", color=OKABE_ITO["blue"], linewidth=2.2)
        for k, y in zip(ks.tolist(), ys.tolist()):
            if np.isfinite(y):
                ax.text(float(k), float(y) + 0.8, f"{y:.1f}", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("K (samples per route)")
        ax.set_ylabel(f"{args.key} success rate (%)")
        ax.set_title(str(args.title))
        ax.set_xticks(ks.tolist())
        ax.set_ylim(0.0, 100.0)
        add_panel_label(ax, "a")

        save_figure(fig, out_dir / "bestofk_curve.png")
        plt.close(fig)

    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {out_dir / 'bestofk_curve.png'}")


if __name__ == "__main__":
    main()

