from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # optional
    pa = None
    pq = None


def _parse_lonlat(v: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    WorldTrace Meta 常见坐标字段是 GeoJSON 风格 [lon, lat]。
    这里按 [lon, lat] 解析；若格式不符合则返回 (None, None)。
    """
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return None, None
    try:
        lon = float(v[0])
        lat = float(v[1])
    except (TypeError, ValueError):
        return None, None
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        return None, None
    return lon, lat


def iter_meta_json_from_zip(meta_zip: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    with zipfile.ZipFile(meta_zip, "r") as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".json"):
                continue
            with zf.open(info, "r") as f:
                try:
                    obj = json.load(f)
                except json.JSONDecodeError:
                    continue
            yield info.filename, obj


def iter_meta_json_from_dir(meta_dir: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    for p in meta_dir.rglob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        yield str(p.relative_to(meta_dir)), obj


def meta_to_row(obj: Dict[str, Any]) -> Dict[str, Any]:
    filename = obj.get("Filename")
    points = obj.get("Points")
    uploaded = obj.get("Uploaded")
    dist = obj.get("Distance")
    duration = obj.get("Time")

    start_lon, start_lat = _parse_lonlat(obj.get("Start coordinate"))
    end_lon, end_lat = _parse_lonlat(obj.get("End coordinate"))

    return {
        "traj_filename": filename,
        "points": points,
        "uploaded": uploaded,
        "distance": dist,
        "duration": duration,
        "start_lat": start_lat,
        "start_lon": start_lon,
        "end_lat": end_lat,
        "end_lon": end_lon,
    }


def _open_out_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffixes[-2:] == [".csv", ".gz"] or path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def _open_parquet_writer(path: Path) -> tuple[pq.ParquetWriter, pa.Schema]:
    if pa is None or pq is None:
        raise SystemExit("pyarrow is required for parquet output. Install: conda/pip install pyarrow")
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("traj_filename", pa.string()),
            ("points", pa.int64()),
            ("uploaded", pa.string()),
            ("distance", pa.float64()),
            ("duration", pa.float64()),
            ("start_lat", pa.float64()),
            ("start_lon", pa.float64()),
            ("end_lat", pa.float64()),
            ("end_lon", pa.float64()),
        ]
    )
    return pq.ParquetWriter(str(path), schema=schema, compression="zstd"), schema


def main() -> None:
    ap = argparse.ArgumentParser(description="Build WorldTrace Meta manifest (streaming; CSV or Parquet).")
    ap.add_argument("--meta_zip", type=Path, default=None, help="Path to Meta.zip")
    ap.add_argument("--meta_dir", type=Path, default=None, help="Path to extracted Meta/ directory")
    ap.add_argument(
        "--out_manifest",
        type=Path,
        required=True,
        help="Output manifest path (.parquet or .csv/.csv.gz)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for debugging (0 = no limit)")
    args = ap.parse_args()

    if (args.meta_zip is None) == (args.meta_dir is None):
        raise SystemExit("Provide exactly one of --meta_zip or --meta_dir")

    if args.meta_zip is not None:
        it = iter_meta_json_from_zip(args.meta_zip)
    else:
        it = iter_meta_json_from_dir(args.meta_dir)

    out_path: Path = args.out_manifest
    n = 0

    if out_path.suffix == ".parquet":
        writer, schema = _open_parquet_writer(out_path)
        try:
            batch_rows: Dict[str, list] = {
                "traj_filename": [],
                "points": [],
                "uploaded": [],
                "distance": [],
                "duration": [],
                "start_lat": [],
                "start_lon": [],
                "end_lat": [],
                "end_lon": [],
            }

            def _flush_batch():
                nonlocal batch_rows
                if not batch_rows["traj_filename"]:
                    return
                # 必须显式使用 schema；否则当某一批次里某列全为 None 时，Arrow 会推断为 null 类型，
                # 导致与 ParquetWriter 的 float64 schema 不一致（例如 end_lat/end_lon）。
                table = pa.Table.from_pydict(batch_rows, schema=schema)
                writer.write_table(table)  # type: ignore[arg-type]
                batch_rows = {k: [] for k in batch_rows}

            for _, obj in it:
                row = meta_to_row(obj)
                for k in batch_rows:
                    batch_rows[k].append(row.get(k))
                n += 1
                if len(batch_rows["traj_filename"]) >= 10000:
                    _flush_batch()
                if args.limit and n >= args.limit:
                    break
                if n % 200000 == 0:
                    print(f"[INFO] wrote {n} rows...", file=sys.stderr)

            _flush_batch()
        finally:
            writer.close()
    else:
        out_f = _open_out_csv(out_path)
        with out_f as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "traj_filename",
                    "points",
                    "uploaded",
                    "distance",
                    "duration",
                    "start_lat",
                    "start_lon",
                    "end_lat",
                    "end_lon",
                ],
            )
            writer.writeheader()
            for _, obj in it:
                writer.writerow(meta_to_row(obj))
                n += 1
                if args.limit and n >= args.limit:
                    break
                if n % 200000 == 0:
                    print(f"[INFO] wrote {n} rows...", file=sys.stderr)

    print(json.dumps({"out_manifest": str(out_path), "rows": n}, indent=2))


if __name__ == "__main__":
    main()
