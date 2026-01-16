from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Task:
    od_index: int  # index into scan_report_json["multimodal_od_bins"]
    cluster_id: int  # 0/1/...
    member: str  # zip member path


def _safe_float(v: object) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pick_latlon(row: Dict[str, str], *, prefer_matched: bool) -> Tuple[Optional[float], Optional[float]]:
    if prefer_matched:
        lat_m = _safe_float(row.get("matched_latitude", "")) or _safe_float(row.get("matched_lat", ""))
        lon_m = _safe_float(row.get("matched_longitude", "")) or _safe_float(row.get("matched_lon", ""))
        if lat_m is not None and lon_m is not None:
            return float(lat_m), float(lon_m)
    lat = _safe_float(row.get("latitude", "")) or _safe_float(row.get("lat", ""))
    lon = _safe_float(row.get("longitude", "")) or _safe_float(row.get("lon", ""))
    if lat is None or lon is None:
        return None, None
    return float(lat), float(lon)


def _iter_csv_rows_from_zip(zf: zipfile.ZipFile, member: str) -> Iterator[Dict[str, str]]:
    with zf.open(member, "r") as f:
        text = io.TextIOWrapper(f, encoding="utf-8", errors="ignore", newline="")
        reader = csv.DictReader(text)
        for row in reader:
            yield row


def _read_traj_from_zip(
    zf: zipfile.ZipFile,
    member: str,
    *,
    prefer_matched: bool,
    downsample_step: int,
) -> np.ndarray:
    step = max(1, int(downsample_step))
    pts: List[Tuple[float, float]] = []
    for i, row in enumerate(_iter_csv_rows_from_zip(zf, member)):
        if (i % step) != 0:
            continue
        lat, lon = _pick_latlon(row, prefer_matched=bool(prefer_matched))
        if lat is None or lon is None:
            continue
        pts.append((float(lat), float(lon)))
    if len(pts) < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def _process_task_chunk(
    zip_path: str,
    tasks: List[Task],
    *,
    prefer_matched: bool,
    downsample_step: int,
) -> List[Tuple[int, int, str, np.ndarray]]:
    out: List[Tuple[int, int, str, np.ndarray]] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for t in tasks:
            try:
                zf.getinfo(t.member)
            except KeyError:
                continue
            pts = _read_traj_from_zip(zf, t.member, prefer_matched=bool(prefer_matched), downsample_step=int(downsample_step))
            if pts.shape[0] >= 2:
                out.append((int(t.od_index), int(t.cluster_id), str(t.member), pts))
    return out


def _chunks(xs: Sequence[Task], n: int) -> List[List[Task]]:
    n = max(1, int(n))
    return [list(xs[i : i + n]) for i in range(0, len(xs), n)]


def dump_cache(
    *,
    scan_report_json: Path,
    trajectory_zip: Path,
    out_npz: Path,
    od_indices: Optional[List[int]],
    top_k: int,
    clusters_keep: int,
    max_files_per_cluster: int,
    prefer_matched: bool,
    downsample_step: int,
    num_workers: int,
    mp_start: str,
    chunk_size: int,
) -> Dict[str, object]:
    rep = json.loads(Path(scan_report_json).read_text(encoding="utf-8"))
    mm = rep.get("multimodal_od_bins", [])
    if not isinstance(mm, list) or not mm:
        raise SystemExit(f"No multimodal_od_bins found in: {scan_report_json}")

    if od_indices:
        pick = [int(i) for i in od_indices if 0 <= int(i) < len(mm)]
    else:
        pick = list(range(min(int(top_k), len(mm))))

    tasks: List[Task] = []
    od_meta: List[Dict[str, object]] = []
    for od_i in pick:
        ent = mm[int(od_i)]
        if not isinstance(ent, dict):
            continue
        rep_files = ent.get("cluster_rep_files", [])
        if not isinstance(rep_files, list) or not rep_files:
            continue
        k_clusters = min(int(clusters_keep), int(len(rep_files)))
        for ci in range(k_clusters):
            files = rep_files[ci]
            if not isinstance(files, list):
                continue
            for member in files[: int(max_files_per_cluster)]:
                if isinstance(member, str):
                    tasks.append(Task(od_index=int(od_i), cluster_id=int(ci), member=str(member)))
        od_meta.append(
            {
                "od_index": int(od_i),
                "od_bin": ent.get("od_bin", None),
                "n_routes": int(ent.get("n_routes", 0)),
                "cluster_sizes": ent.get("cluster_sizes", None),
                "top2_lcs_dist": ent.get("top2_lcs_dist", None),
                "top2_jaccard_dist": ent.get("top2_jaccard_dist", None),
            }
        )

    if not tasks:
        raise SystemExit("No representative tasks to dump (check scan report / max_files_per_cluster).")

    zip_path = str(trajectory_zip)
    if not Path(zip_path).exists():
        raise SystemExit(f"Missing trajectory_zip: {trajectory_zip}")

    num_workers = int(num_workers)
    if num_workers <= 0:
        num_workers = os.cpu_count() or 1
    chunk_size = max(1, int(chunk_size))
    task_chunks = _chunks(tasks, int(chunk_size))

    mp_ctx = None
    if str(mp_start) in {"fork", "spawn"}:
        import multiprocessing as mp

        mp_ctx = mp.get_context(str(mp_start))

    t0 = time.time()
    results: List[Tuple[int, int, str, np.ndarray]] = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as ex:
        futs = [
            ex.submit(
                _process_task_chunk,
                zip_path,
                chunk,
                prefer_matched=bool(prefer_matched),
                downsample_step=int(downsample_step),
            )
            for chunk in task_chunks
        ]
        for fut in as_completed(futs):
            results.extend(fut.result())
    elapsed = float(time.time() - t0)

    if not results:
        raise SystemExit("Dumped 0 trajectories (members missing or too short after downsample).")

    # Build compact arrays.
    traj_od_index: List[int] = []
    traj_cluster: List[int] = []
    traj_member: List[str] = []
    ptr = [0]
    flat: List[np.ndarray] = []
    for od_i, ci, member, pts in results:
        traj_od_index.append(int(od_i))
        traj_cluster.append(int(ci))
        traj_member.append(str(member))
        flat.append(np.asarray(pts, dtype=np.float32))
        ptr.append(ptr[-1] + int(pts.shape[0]))

    latlon = np.concatenate(flat, axis=0) if flat else np.zeros((0, 2), dtype=np.float32)
    ptr_arr = np.asarray(ptr, dtype=np.int64)

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "dump_multimodal_viz_cache",
        "inputs": {"scan_report_json": str(scan_report_json), "trajectory_zip": str(trajectory_zip)},
        "config": {
            "top_k": int(top_k),
            "od_indices": (pick if od_indices else None),
            "clusters_keep": int(clusters_keep),
            "max_files_per_cluster": int(max_files_per_cluster),
            "prefer_matched": bool(prefer_matched),
            "downsample_step": int(downsample_step),
            "num_workers": int(num_workers),
            "chunk_size": int(chunk_size),
            "mp_start": str(mp_start),
        },
        "stats": {
            "n_od": int(len(pick)),
            "n_tasks": int(len(tasks)),
            "n_traj": int(len(results)),
            "total_points": int(latlon.shape[0]),
            "elapsed_s": float(elapsed),
        },
        "od_meta": od_meta,
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        traj_ptr=ptr_arr,
        traj_latlon=latlon,
        traj_od_index=np.asarray(traj_od_index, dtype=np.int32),
        traj_cluster=np.asarray(traj_cluster, dtype=np.int8),
        traj_member=np.asarray(traj_member, dtype=object),
        meta=meta,
    )
    report = {"ok": True, "out_npz": str(out_npz), "meta": meta}
    (out_npz.parent / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump a small trajectory cache (rep files) for multimodal OD visualization.")
    p.add_argument("--scan_report_json", type=Path, required=True)
    p.add_argument("--trajectory_zip", type=Path, required=True)
    p.add_argument("--out_npz", type=Path, required=True)

    p.add_argument("--top_k", type=int, default=200, help="Cache top-K multimodal OD bins (in report order).")
    p.add_argument("--od_indices", type=int, nargs="*", default=None, help="Optional: explicit OD indices in multimodal_od_bins.")
    p.add_argument("--clusters_keep", type=int, default=2, help="How many clusters' rep files to cache per OD (KISS: 2).")
    p.add_argument("--max_files_per_cluster", type=int, default=3)
    p.add_argument("--prefer_matched", action="store_true")
    p.add_argument("--downsample_step", type=int, default=10)

    p.add_argument("--num_workers", type=int, default=48)
    p.add_argument("--mp_start", choices=["fork", "spawn"], default="fork")
    p.add_argument("--chunk_size", type=int, default=256, help="Tasks per worker chunk (not zip members).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = dump_cache(
        scan_report_json=Path(args.scan_report_json),
        trajectory_zip=Path(args.trajectory_zip),
        out_npz=Path(args.out_npz),
        od_indices=([int(x) for x in args.od_indices] if args.od_indices else None),
        top_k=int(args.top_k),
        clusters_keep=int(args.clusters_keep),
        max_files_per_cluster=int(args.max_files_per_cluster),
        prefer_matched=bool(args.prefer_matched),
        downsample_step=int(args.downsample_step),
        num_workers=int(args.num_workers),
        mp_start=str(args.mp_start),
        chunk_size=int(args.chunk_size),
    )
    st = report["meta"]["stats"]
    print(
        json.dumps(
            {
                "ok": True,
                "out_npz": report["out_npz"],
                "n_traj": int(st["n_traj"]),
                "total_points": int(st["total_points"]),
                "elapsed_s": float(st["elapsed_s"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

