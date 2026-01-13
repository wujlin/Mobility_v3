from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pyarrow.parquet as pq  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pq = None


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class CityInfo:
    city: str
    dir: str
    segments_parquet: Optional[str]
    num_segments: Optional[int]
    has_osm_meta: bool
    has_road_prob: bool
    has_tier_major: bool
    has_tier_minor: bool
    has_tier_service: bool

    def to_json(self) -> Dict[str, object]:
        return {
            "city": self.city,
            "dir": self.dir,
            "segments_parquet": self.segments_parquet,
            "num_segments": self.num_segments,
            "has_osm_meta": self.has_osm_meta,
            "has_road_prob": self.has_road_prob,
            "has_tier_major": self.has_tier_major,
            "has_tier_minor": self.has_tier_minor,
            "has_tier_service": self.has_tier_service,
            "tier_ready": bool(self.has_tier_major and self.has_tier_minor and self.has_tier_service),
        }


def _num_rows_parquet(path: Path) -> Optional[int]:
    if pq is None:  # pragma: no cover
        return None
    try:
        meta = pq.read_metadata(str(path))
    except Exception:
        return None
    return int(meta.num_rows)


def scan_root(worldtrace_root: Path) -> Dict[str, object]:
    root = Path(worldtrace_root)
    if not root.exists():
        raise FileNotFoundError(f"worldtrace_root not found: {root}")
    if not root.is_dir():
        raise ValueError(f"worldtrace_root is not a dir: {root}")

    cities: List[CityInfo] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        seg = d / "segments.parquet"
        meta = d / "osm_road_prob_meta.json"
        road = d / "osm_road_prob.npy"
        maj = d / "osm_road_prob_major.npy"
        mi = d / "osm_road_prob_minor.npy"
        sv = d / "osm_road_prob_service.npy"
        n = _num_rows_parquet(seg) if seg.exists() else None
        cities.append(
            CityInfo(
                city=str(d.name),
                dir=str(d),
                segments_parquet=(str(seg) if seg.exists() else None),
                num_segments=(int(n) if n is not None else None),
                has_osm_meta=bool(meta.exists()),
                has_road_prob=bool(road.exists()),
                has_tier_major=bool(maj.exists()),
                has_tier_minor=bool(mi.exists()),
                has_tier_service=bool(sv.exists()),
            )
        )

    with_seg = [c for c in cities if c.segments_parquet is not None and c.num_segments is not None]
    tier_ready = [c for c in with_seg if bool(c.has_tier_major and c.has_tier_minor and c.has_tier_service)]

    # Sort by num_segments desc (unknown -> last).
    with_seg_sorted = sorted(with_seg, key=lambda x: int(x.num_segments or -1), reverse=True)
    top = with_seg_sorted[:20]

    report = {
        "ok": True,
        "worldtrace_root": str(root),
        "stats": {
            "num_dirs_scanned": int(len(cities)),
            "num_with_segments": int(len(with_seg)),
            "num_tier_ready": int(len(tier_ready)),
            "sum_segments_tier_ready": int(sum(int(c.num_segments or 0) for c in tier_ready)),
        },
        "top_by_segments": [c.to_json() for c in top],
        "cities": [c.to_json() for c in with_seg_sorted],
        "meta": {"created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(), "requires_pyarrow": bool(pq is not None)},
    }
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scan $RAW_ROOT/worldtrace/* for available cities and segment counts.")
    p.add_argument("--worldtrace_root", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = scan_root(Path(args.worldtrace_root))
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    compact = {
        "ok": True,
        "worldtrace_root": report["worldtrace_root"],
        "num_with_segments": report["stats"]["num_with_segments"],
        "num_tier_ready": report["stats"]["num_tier_ready"],
        "sum_segments_tier_ready": report["stats"]["sum_segments_tier_ready"],
        "out_json": str(out),
        "top_by_segments": [
            {"city": c["city"], "num_segments": c["num_segments"], "tier_ready": c["tier_ready"]}
            for c in report["top_by_segments"]
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

