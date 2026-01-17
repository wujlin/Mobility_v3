from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # optional
    pa = None
    pq = None

from src.data.worldtrace.audit_owner_from_meta_and_segments import _extract_owner, _meta_member_candidates, _sha1_8

TZ_SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Cfg:
    od_bin_deg: float
    min_od_dist_km: float
    max_way_seq_len: int
    merge_dist_thr: float
    top2_sep_thr: float
    top_k_od: int
    tz_offset_hours: float
    owner: str  # empty means top-1 owner


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dlat = p2 - p1
    dlon = math.radians(float(lon2) - float(lon1))
    a = math.sin(dlat / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return float(r * c)


def _bin_int(x: float, deg: float) -> int:
    return int(math.floor(float(x) / float(deg)))


def _endpoint_idx(osm_way_id: np.ndarray) -> Optional[Tuple[int, int]]:
    w = np.asarray(osm_way_id, dtype=np.int64).reshape(-1)
    good = np.nonzero(w > 0)[0]
    if good.size == 0:
        return None
    return int(good[0]), int(good[-1])


def _dedup_way_seq(osm_way_id: np.ndarray, *, max_len: int) -> Tuple[int, ...]:
    w = np.asarray(osm_way_id, dtype=np.int64).reshape(-1)
    out: List[int] = []
    last = None
    for v in w.tolist():
        iv = int(v)
        if iv <= 0:
            continue
        if last is None or iv != last:
            out.append(iv)
            last = iv
            if len(out) >= int(max_len):
                break
    return tuple(out)


def _lcs_length(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    if m < n:
        a, b = b, a
        m, n = n, m
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev, curr = curr, prev
    return int(prev[n])


def _lcs_dist(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    if not a and not b:
        return 0.0
    ml = max(len(a), len(b))
    if ml <= 0:
        return 0.0
    l = _lcs_length(a, b)
    return 1.0 - (float(l) / float(ml))


@dataclass
class SigEntry:
    count: int
    reps: List[int]  # route indices for plotting


def _cluster_signatures_with_map(
    sig_items: List[Tuple[Tuple[int, ...], SigEntry]],
    *,
    merge_dist_thr: float,
) -> Tuple[List[Dict[str, object]], Dict[Tuple[int, ...], int]]:
    """
    Greedy clustering by LCS distance (same spirit as scan_multimodal_od_region).
    Returns clusters + signature->cluster_id map.
    """
    clusters: List[Dict[str, object]] = []
    sig2c: Dict[Tuple[int, ...], int] = {}
    for sig, ent in sig_items:
        placed = False
        for ci, c in enumerate(clusters):
            if _lcs_dist(sig, c["rep_sig"]) < float(merge_dist_thr):
                c["count"] += int(ent.count)
                c["rep_route_ids"].extend(list(ent.reps))
                sig2c[sig] = int(ci)
                placed = True
                break
        if not placed:
            ci = int(len(clusters))
            clusters.append({"rep_sig": sig, "count": int(ent.count), "rep_route_ids": list(ent.reps)})
            sig2c[sig] = int(ci)
    clusters.sort(key=lambda x: -int(x["count"]))
    # Re-index after sorting
    rep_sig_to_new = {c["rep_sig"]: i for i, c in enumerate(clusters)}
    # Build a robust mapping: assign signature to the first cluster whose rep_sig matches its assigned old cluster.
    # Since we sorted clusters, we rebuild sig2c by nearest rep (same threshold) to keep consistency.
    sig2c_new: Dict[Tuple[int, ...], int] = {}
    for sig, _ent in sig_items:
        # find closest cluster rep among sorted clusters
        best_i = 0
        best_d = float("inf")
        for i, c in enumerate(clusters):
            d = _lcs_dist(sig, c["rep_sig"])
            if d < best_d:
                best_d = d
                best_i = i
        sig2c_new[sig] = int(best_i)
    return clusters, sig2c_new


def _entropy_from_counts(counts: List[int]) -> float:
    tot = float(sum(int(x) for x in counts))
    if tot <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        p = float(c) / tot
        if p > 0:
            h -= p * math.log(p + 1e-12)
    return float(h)


def _owner_by_traj_key(meta_zip: Path, keys: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with zipfile.ZipFile(meta_zip, "r") as zf:
        for k in keys:
            for cand in _meta_member_candidates(k):
                try:
                    with zf.open(cand, "r") as f:
                        obj = json.load(f)
                except KeyError:
                    continue
                except json.JSONDecodeError:
                    continue
                owner = _extract_owner(obj)
                if owner:
                    out[str(k)] = str(owner)
                break
    return out


def _load_road_prob(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    a = np.load(str(p))
    if a.ndim != 2:
        return None
    return np.asarray(a, dtype=np.float32)


def _plot_top_od_bins(
    *,
    out_dir: Path,
    od_bins: List[Dict[str, object]],
    road_prob: Optional[np.ndarray],
    routes_xy: List[Tuple[np.ndarray, np.ndarray]],
    palette: List[str],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("Missing dependency: matplotlib (needed for plotting).") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    for k, ent in enumerate(od_bins, start=1):
        rid_list = ent["route_ids"]
        cid_list = ent["cluster_ids"]
        if not rid_list:
            continue
        fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=200)
        if road_prob is not None:
            rp = np.clip(np.asarray(road_prob, dtype=np.float32), 0.0, 1.0)
            ax.imshow(rp, cmap="Greys", origin="upper", alpha=0.35, vmin=0.0, vmax=1.0)

        # Plot all trajectories, colored by corridor id.
        for rid, cid in zip(rid_list, cid_list):
            x, y = routes_xy[int(rid)]
            if x.size < 2:
                continue
            color = palette[int(cid) % len(palette)]
            ax.plot(x, y, color=color, alpha=0.25, linewidth=1.2)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"OD#{k} n={ent['n_routes']} K={ent['n_clusters']} sizes={ent['cluster_sizes']}")
        fig.tight_layout()
        fig.savefig(out_dir / f"top_od_{k:02d}.png")
        plt.close(fig)


def run(
    *,
    segments_parquet: Path,
    meta_zip: Path,
    out_parquet: Path,
    out_json: Path,
    out_viz_dir: Optional[Path],
    road_prob_npy: Optional[Path],
    cfg: Cfg,
) -> Dict[str, object]:
    if pq is None or pa is None:
        raise SystemExit("pyarrow is required. Install: pip/conda install pyarrow")

    cols = ["traj_csv", "t", "lat", "lon", "y", "x", "osm_way_id"]
    pf = pq.ParquetFile(str(segments_parquet))

    traj_key: List[str] = []
    start_t: List[int] = []
    od_key: List[Tuple[int, int, int, int]] = []
    od_km: List[float] = []
    way_sig: List[Tuple[int, ...]] = []
    routes_xy: List[Tuple[np.ndarray, np.ndarray]] = []

    scanned = 0
    kept = 0

    for batch in pf.iter_batches(batch_size=128, columns=cols):
        d = batch.to_pydict()
        n_rows = len(d["traj_csv"])
        for i in range(n_rows):
            scanned += 1
            key = Path(str(d["traj_csv"][i])).stem
            osm = np.asarray(d["osm_way_id"][i], dtype=np.int64)
            end_idx = _endpoint_idx(osm)
            if end_idx is None:
                continue
            i0, i1 = end_idx

            lat = np.asarray(d["lat"][i], dtype=np.float64)
            lon = np.asarray(d["lon"][i], dtype=np.float64)
            if lat.size <= max(i0, i1) or lon.size <= max(i0, i1):
                continue
            dist_km = _haversine_km(float(lat[i0]), float(lon[i0]), float(lat[i1]), float(lon[i1]))
            if float(dist_km) < float(cfg.min_od_dist_km):
                continue

            ok = (_bin_int(float(lon[i0]), cfg.od_bin_deg), _bin_int(float(lat[i0]), cfg.od_bin_deg),
                  _bin_int(float(lon[i1]), cfg.od_bin_deg), _bin_int(float(lat[i1]), cfg.od_bin_deg))

            t = np.asarray(d["t"][i], dtype=np.int64)
            if t.size < 1:
                continue

            sig = _dedup_way_seq(osm, max_len=int(cfg.max_way_seq_len))
            if len(sig) < 2:
                continue

            y = np.asarray(d["y"][i], dtype=np.float32)
            x = np.asarray(d["x"][i], dtype=np.float32)
            routes_xy.append((x, y))
            traj_key.append(str(key))
            start_t.append(int(t[0]))
            od_key.append(ok)
            od_km.append(float(dist_km))
            way_sig.append(sig)
            kept += 1

    if kept <= 0:
        raise SystemExit("No trips kept after filters.")

    owner_map = _owner_by_traj_key(Path(meta_zip), traj_key)
    owners = [owner_map.get(k, "") for k in traj_key]
    counts: Dict[str, int] = {}
    for o in owners:
        if o:
            counts[o] = int(counts.get(o, 0)) + 1
    if not counts:
        raise SystemExit("Owner join failed (no owners).")

    if str(cfg.owner).strip():
        owner_target = str(cfg.owner).strip()
        if owner_target not in counts:
            raise SystemExit(f"--owner not found: {owner_target}")
    else:
        owner_target = max(counts.items(), key=lambda kv: kv[1])[0]

    mask_owner = np.asarray([o == owner_target for o in owners], dtype=bool)
    idx_owner = np.nonzero(mask_owner)[0].astype(np.int64)

    # Group by OD bins within this owner.
    by_od: Dict[Tuple[int, int, int, int], List[int]] = {}
    for rid in idx_owner.tolist():
        by_od.setdefault(od_key[int(rid)], []).append(int(rid))

    # Build corridor stats for each OD (n>=2).
    rows_out: List[Dict[str, object]] = []
    top_od_entries: List[Tuple[Tuple[int, int, int, int], int]] = []
    for k, rids in by_od.items():
        if len(rids) < 2:
            continue
        top_od_entries.append((k, int(len(rids))))
    top_od_entries.sort(key=lambda t: -t[1])

    def _corridor_for_od(rids: List[int]) -> Tuple[List[Dict[str, object]], Dict[Tuple[int, ...], int], Dict[Tuple[int, ...], SigEntry]]:
        sig_table: Dict[Tuple[int, ...], SigEntry] = {}
        for rid in rids:
            sig = way_sig[int(rid)]
            ent = sig_table.get(sig)
            if ent is None:
                sig_table[sig] = SigEntry(count=1, reps=[int(rid)])
            else:
                ent.count += 1
                if len(ent.reps) < 3:
                    ent.reps.append(int(rid))
        sig_items = list(sig_table.items())
        sig_items.sort(key=lambda kv: -int(kv[1].count))
        clusters, sig2c = _cluster_signatures_with_map(sig_items, merge_dist_thr=float(cfg.merge_dist_thr))
        return clusters, sig2c, sig_table

    # Compute metrics per OD
    for od_k, rids in top_od_entries:
        clusters, sig2c, sig_table = _corridor_for_od(rids)
        cluster_sizes = [int(c["count"]) for c in clusters]
        n_routes = int(len(rids))
        n_clusters = int(len(clusters))
        h = _entropy_from_counts(cluster_sizes)
        eff = float(math.exp(h)) if h > 0 else 1.0
        top2 = 0.0
        if n_clusters >= 2:
            top2 = float(_lcs_dist(clusters[0]["rep_sig"], clusters[1]["rep_sig"]))
        rows_out.append(
            {
                "owner_hash": _sha1_8(owner_target),
                "od_o_lon_bin": int(od_k[0]),
                "od_o_lat_bin": int(od_k[1]),
                "od_d_lon_bin": int(od_k[2]),
                "od_d_lat_bin": int(od_k[3]),
                "n_routes": int(n_routes),
                "n_clusters": int(n_clusters),
                "entropy": float(h),
                "effective_k": float(eff),
                "top2_lcs_dist": float(top2),
                "cluster_sizes": cluster_sizes,
            }
        )

    # Write parquet
    table = pa.Table.from_pylist(rows_out) if rows_out else pa.Table.from_pylist([])
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out_parquet), compression="zstd")

    # Top-k OD visualizations
    road_prob = _load_road_prob(road_prob_npy)
    palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd"]
    top_bins_for_viz: List[Dict[str, object]] = []
    for od_k, n in top_od_entries[: int(cfg.top_k_od)]:
        rids = by_od[od_k]
        clusters, sig2c, _sig_table = _corridor_for_od(rids)
        # Assign cluster id per route
        cid_list = []
        for rid in rids:
            cid_list.append(int(sig2c.get(way_sig[int(rid)], 0)))
        top_bins_for_viz.append(
            {
                "od_bin": [int(x) for x in od_k],
                "n_routes": int(len(rids)),
                "n_clusters": int(len(clusters)),
                "cluster_sizes": [int(c["count"]) for c in clusters],
                "route_ids": [int(r) for r in rids],
                "cluster_ids": cid_list,
            }
        )

    if out_viz_dir is not None:
        _plot_top_od_bins(out_dir=Path(out_viz_dir), od_bins=top_bins_for_viz, road_prob=road_prob, routes_xy=routes_xy, palette=palette)

    report: Dict[str, object] = {
        "ok": True,
        "task": "within_owner_corridor_diversity",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {"segments_parquet": str(segments_parquet), "meta_zip": str(meta_zip), "road_prob_npy": (str(road_prob_npy) if road_prob_npy else None)},
        "cfg": asdict(cfg),
        "owner": {"owner_hash": _sha1_8(owner_target), "n_trips": int(np.sum(mask_owner)), "unique_owner_count": int(len(counts))},
        "stats": {
            "scanned": int(scanned),
            "kept": int(kept),
            "n_od_bins_owner": int(len(by_od)),
            "n_od_bins_owner_ge2trips": int(sum(1 for v in by_od.values() if len(v) >= 2)),
        },
        "outputs": {
            "out_parquet": str(out_parquet),
            "out_viz_dir": (str(out_viz_dir) if out_viz_dir is not None else None),
        },
        "top_od_bins": top_bins_for_viz,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Within-owner corridor diversity (LCS clustering) from segments_with_wayid.parquet.")
    p.add_argument("--segments_parquet", type=Path, required=True)
    p.add_argument("--meta_zip", type=Path, required=True)
    p.add_argument("--out_parquet", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_viz_dir", type=Path, default=None, help="Optional: save Top-K OD-bin corridor plots.")
    p.add_argument("--road_prob_npy", type=Path, default=None, help="Optional: osm_road_prob.npy for grey road background.")

    p.add_argument("--od_bin_deg", type=float, default=0.02)
    p.add_argument("--min_od_dist_km", type=float, default=1.0)
    p.add_argument("--max_way_seq_len", type=int, default=128)
    p.add_argument("--merge_dist_thr", type=float, default=0.15)
    p.add_argument("--top2_sep_thr", type=float, default=0.0, help="Reserved (not used yet); keep for future gating.")
    p.add_argument("--top_k_od", type=int, default=10)
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--owner", type=str, default="", help="Exact owner string; empty means top-1 owner.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Cfg(
        od_bin_deg=float(args.od_bin_deg),
        min_od_dist_km=float(args.min_od_dist_km),
        max_way_seq_len=int(args.max_way_seq_len),
        merge_dist_thr=float(args.merge_dist_thr),
        top2_sep_thr=float(args.top2_sep_thr),
        top_k_od=int(args.top_k_od),
        tz_offset_hours=float(args.tz_offset_hours),
        owner=str(args.owner),
    )
    rep = run(
        segments_parquet=Path(args.segments_parquet),
        meta_zip=Path(args.meta_zip),
        out_parquet=Path(args.out_parquet),
        out_json=Path(args.out_json),
        out_viz_dir=(Path(args.out_viz_dir) if args.out_viz_dir is not None else None),
        road_prob_npy=(Path(args.road_prob_npy) if args.road_prob_npy is not None else None),
        cfg=cfg,
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

