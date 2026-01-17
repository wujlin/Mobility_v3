from __future__ import annotations

import argparse
import json
import math
import re
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
    corridor_method: str  # lcs|decision_points (only affects Top-K viz coloring)
    min_choice_count: int  # only used when corridor_method=decision_points
    dp_tier_keep: str  # empty => disable tier filtering for decision-points
    dp_next_min_keep: int  # only used when dp_tier_keep non-empty
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


def _dedup_way_seq_with_xy(
    osm_way_id: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    max_len: int,
) -> Tuple[Tuple[int, ...], np.ndarray]:
    """
    Deduplicate consecutive osm_way_id (filter <=0), and also return a representative (x,y)
    per deduplicated way token (the coordinate at the first occurrence of that token).
    """
    w = np.asarray(osm_way_id, dtype=np.int64).reshape(-1)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    n = int(min(w.size, x.size, y.size))
    out: List[int] = []
    xy: List[Tuple[float, float]] = []
    last = None
    for i in range(n):
        iv = int(w[i])
        if iv <= 0:
            continue
        if last is None or iv != last:
            out.append(iv)
            xy.append((float(x[i]), float(y[i])))
            last = iv
            if len(out) >= int(max_len):
                break
    if not out:
        return tuple(), np.zeros((0, 2), dtype=np.float32)
    return tuple(out), np.asarray(xy, dtype=np.float32)


def _parse_int_set(spec: str) -> set[int]:
    s = str(spec or "").strip()
    if not s:
        return set()
    out: set[int] = set()
    for tok in re.split(r"[,\s]+", s):
        t = tok.strip()
        if not t:
            continue
        out.add(int(t))
    return out


def _load_way_tier_map(way_features_npz: Path) -> Dict[int, int]:
    data = np.load(str(way_features_npz), allow_pickle=True)
    if "way_osm_id" not in data.files or "way_tier" not in data.files:
        raise ValueError(f"way_features_npz missing way_osm_id/way_tier: {way_features_npz}")
    way_osm_id = np.asarray(data["way_osm_id"], dtype=np.int64).reshape(-1)
    way_tier = np.asarray(data["way_tier"], dtype=np.int64).reshape(-1)
    if way_osm_id.size != way_tier.size:
        raise ValueError(f"way_features_npz shape mismatch: way_osm_id={way_osm_id.size} way_tier={way_tier.size}")
    return {int(w): int(t) for w, t in zip(way_osm_id.tolist(), way_tier.tolist())}


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


def _cluster_decision_points_for_od(
    rids: List[int],
    *,
    way_sig: List[Tuple[int, ...]],
    way_xy_sig: List[np.ndarray],
    min_choice_count: int,
    way_tier: Optional[Dict[int, int]] = None,
    dp_tier_keep: Optional[set[int]] = None,
    dp_next_min_keep: int = 2,
) -> Dict[str, object]:
    """
    Way-level data-driven decision points within one OD-bin group.

    Definition:
      A way u is a decision point if there exist >=2 next ways v with count(u->v) >= min_choice_count,
      where counts are computed from deduplicated way sequences (per-route unique transitions).

    Corridor signature:
      For each route, extract the sequence of (u,v) decisions encountered along the route,
      skipping (u->v) pairs where v is not in the valid-next set of u.
    """
    min_choice_count = int(max(1, min_choice_count))

    # Transition counts (per-route unique transitions to reduce loop inflation).
    trans: Dict[int, Dict[int, int]] = {}
    for rid in rids:
        seq = way_sig[int(rid)]
        if len(seq) < 2:
            continue
        seen = set()
        for a, b in zip(seq[:-1], seq[1:]):
            aa = int(a)
            bb = int(b)
            if aa <= 0 or bb <= 0 or aa == bb:
                continue
            seen.add((aa, bb))
        for aa, bb in seen:
            m = trans.get(aa)
            if m is None:
                m = {}
                trans[aa] = m
            m[bb] = int(m.get(bb, 0)) + 1

    valid_next_set: Dict[int, set[int]] = {}
    for u, nxt in trans.items():
        keep = {int(v) for v, c in nxt.items() if int(c) >= min_choice_count}
        if len(keep) >= 2:
            valid_next_set[int(u)] = keep

    decision_points_all = sorted(valid_next_set.keys())

    # Optional: tier-filter decision points (keep only corridor-defining branches on major roads).
    tier_keep = dp_tier_keep or set()
    dp_next_min_keep = int(max(1, dp_next_min_keep))
    dp_tier_all: Dict[int, int] = {}
    dp_tier_kept: Dict[int, int] = {}
    if tier_keep and way_tier is not None:
        filtered: Dict[int, set[int]] = {}
        for u, vs in valid_next_set.items():
            ut = int(way_tier.get(int(u), 3))
            dp_tier_all[int(u)] = ut
            if ut not in tier_keep:
                continue
            keep_vs: set[int] = set()
            for v in vs:
                vt = int(way_tier.get(int(v), 3))
                if vt in tier_keep:
                    keep_vs.add(int(v))
            required = int(max(2, dp_next_min_keep))
            if len(keep_vs) < required:
                continue
            filtered[int(u)] = keep_vs
            dp_tier_kept[int(u)] = ut
        valid_next_set = filtered

    decision_points = sorted(valid_next_set.keys())

    # Route -> decision signature
    sig_to_rids: Dict[Tuple[int, ...], List[int]] = {}
    rid_sig: Dict[int, Tuple[int, ...]] = {}
    for rid in rids:
        seq = way_sig[int(rid)]
        ds: List[int] = []
        if len(seq) >= 2 and valid_next_set:
            for a, b in zip(seq[:-1], seq[1:]):
                aa = int(a)
                bb = int(b)
                vs = valid_next_set.get(aa)
                if vs is None or bb not in vs:
                    continue
                ds.append(aa)
                ds.append(bb)
        key = tuple(ds)
        rid_sig[int(rid)] = key
        sig_to_rids.setdefault(key, []).append(int(rid))

    # Sort clusters by size desc (stable by signature for tie).
    clusters_raw = sorted(sig_to_rids.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    clusters = [{"rep_sig": sig, "count": int(len(rr)), "rep_route_ids": list(rr[:3])} for sig, rr in clusters_raw]

    rid2c: Dict[int, int] = {}
    for ci, (_sig, rr) in enumerate(clusters_raw):
        for rrid in rr:
            rid2c[int(rrid)] = int(ci)

    # Decision point positions (median token coord across routes)
    dp_xy: Dict[int, Tuple[float, float]] = {}
    if decision_points:
        for dp in decision_points:
            xs: List[float] = []
            ys: List[float] = []
            for rid in rids:
                seq = way_sig[int(rid)]
                if not seq:
                    continue
                try:
                    j = seq.index(int(dp))
                except ValueError:
                    continue
                xy = way_xy_sig[int(rid)]
                if xy.shape[0] <= j:
                    continue
                xs.append(float(xy[j, 0]))
                ys.append(float(xy[j, 1]))
            if xs:
                dp_xy[int(dp)] = (float(np.median(np.asarray(xs, dtype=np.float64))), float(np.median(np.asarray(ys, dtype=np.float64))))

    return {
        "decision_points": decision_points,
        "decision_points_all": decision_points_all,
        "valid_next": {int(u): sorted(list(vs)) for u, vs in valid_next_set.items()},
        "clusters": clusters,
        "rid2c": rid2c,
        "rid_sig": rid_sig,
        "decision_points_xy": dp_xy,
        "decision_points_tier": dp_tier_kept,
        "decision_points_tier_all": dp_tier_all,
    }


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


def _bbox_from_xy(points: np.ndarray, *, pad_ratio: float = 0.08) -> Optional[Tuple[float, float, float, float]]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return None
    x0 = float(np.min(pts[:, 0]))
    x1 = float(np.max(pts[:, 0]))
    y0 = float(np.min(pts[:, 1]))
    y1 = float(np.max(pts[:, 1]))
    span = max(x1 - x0, y1 - y0, 1.0)
    pad = span * float(pad_ratio)
    return x0 - pad, x1 + pad, y0 - pad, y1 + pad


def _clip_xy_for_plot(x: np.ndarray, y: np.ndarray, *, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if x.size == 0 or y.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    m = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < float(W)) & (y >= 0) & (y < float(H))
    if not np.any(m):
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return x[m].astype(np.float32, copy=False), y[m].astype(np.float32, copy=False)


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
        H = int(road_prob.shape[0]) if road_prob is not None else 1024
        W = int(road_prob.shape[1]) if road_prob is not None else 1024
        if road_prob is not None:
            rp = np.clip(np.asarray(road_prob, dtype=np.float32), 0.0, 1.0)
            ax.imshow(rp, cmap="Greys", origin="upper", alpha=0.35, vmin=0.0, vmax=1.0)

        # Plot all trajectories, colored by corridor id.
        all_pts: List[np.ndarray] = []
        for rid, cid in zip(rid_list, cid_list):
            x, y = routes_xy[int(rid)]
            x2, y2 = _clip_xy_for_plot(x, y, H=H, W=W)
            if x2.size < 2:
                continue
            color = palette[int(cid) % len(palette)]
            ax.plot(x2, y2, color=color, alpha=0.25, linewidth=1.2)
            all_pts.append(np.stack([x2, y2], axis=1))

        # Decision points (optional)
        dps = ent.get("decision_points_xy", None)
        if isinstance(dps, dict) and dps:
            xs: List[float] = []
            ys: List[float] = []
            for _dp, xy in dps.items():
                try:
                    xx, yy = float(xy[0]), float(xy[1])
                except Exception:
                    continue
                xs.append(xx)
                ys.append(yy)
            if xs:
                ax.scatter(xs, ys, s=18, c="black", alpha=0.7, linewidths=0.0, zorder=10)

        # Auto-zoom to the OD's data extent
        if all_pts:
            bb = _bbox_from_xy(np.concatenate(all_pts, axis=0))
            if bb is not None:
                x0, x1, y0, y1 = bb
                ax.set_xlim(x0, x1)
                ax.set_ylim(y1, y0)  # origin="upper"

        ax.set_xticks([])
        ax.set_yticks([])
        title = f"OD#{k} n={ent['n_routes']} K={ent['n_clusters']} sizes={ent['cluster_sizes']}"
        extra = ent.get("title_extra", "")
        if isinstance(extra, str) and extra.strip():
            title = f"{title} {extra.strip()}"
        ax.set_title(title)
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
    way_features_npz: Optional[Path],
    cfg: Cfg,
) -> Dict[str, object]:
    if pq is None or pa is None:
        raise SystemExit("pyarrow is required. Install: pip/conda install pyarrow")

    cols = ["traj_csv", "t", "lat", "lon", "y", "x", "osm_way_id"]
    pf = pq.ParquetFile(str(segments_parquet))

    tier_keep = _parse_int_set(cfg.dp_tier_keep)
    way_tier = None
    if tier_keep:
        if way_features_npz is None:
            raise SystemExit("--dp_tier_keep requires --way_features_npz (need way_osm_id->way_tier mapping).")
        way_tier = _load_way_tier_map(Path(way_features_npz))

    traj_key: List[str] = []
    start_t: List[int] = []
    od_key: List[Tuple[int, int, int, int]] = []
    od_km: List[float] = []
    way_sig: List[Tuple[int, ...]] = []
    way_xy_sig: List[np.ndarray] = []
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

            x = np.asarray(d["x"][i], dtype=np.float32)
            y = np.asarray(d["y"][i], dtype=np.float32)
            sig, sig_xy = _dedup_way_seq_with_xy(osm, x=x, y=y, max_len=int(cfg.max_way_seq_len))
            if len(sig) < 2:
                continue

            routes_xy.append((x, y))
            traj_key.append(str(key))
            start_t.append(int(t[0]))
            od_key.append(ok)
            od_km.append(float(dist_km))
            way_sig.append(sig)
            way_xy_sig.append(sig_xy)
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

    def _corridor_for_od_lcs(rids: List[int]) -> Tuple[List[Dict[str, object]], Dict[Tuple[int, ...], int], Dict[Tuple[int, ...], SigEntry]]:
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
    for od_k, _n in top_od_entries:
        rids = by_od[od_k]
        clusters_lcs, sig2c_lcs, _sig_table = _corridor_for_od_lcs(rids)
        cluster_sizes_lcs = [int(c["count"]) for c in clusters_lcs]
        n_routes = int(len(rids))
        n_clusters = int(len(clusters_lcs))
        h = _entropy_from_counts(cluster_sizes_lcs)
        eff = float(math.exp(h)) if h > 0 else 1.0
        top2 = 0.0
        if n_clusters >= 2:
            top2 = float(_lcs_dist(clusters_lcs[0]["rep_sig"], clusters_lcs[1]["rep_sig"]))

        dp = _cluster_decision_points_for_od(
            rids,
            way_sig=way_sig,
            way_xy_sig=way_xy_sig,
            min_choice_count=int(cfg.min_choice_count),
            way_tier=way_tier,
            dp_tier_keep=tier_keep,
            dp_next_min_keep=int(cfg.dp_next_min_keep),
        )
        clusters_dp = dp["clusters"]
        cluster_sizes_dp = [int(c["count"]) for c in clusters_dp]
        n_clusters_dp = int(len(clusters_dp))
        h_dp = _entropy_from_counts(cluster_sizes_dp)
        eff_dp = float(math.exp(h_dp)) if h_dp > 0 else 1.0
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
                "cluster_sizes": cluster_sizes_lcs,
                "n_decision_points": int(len(dp.get("decision_points", []))),
                "decision_points": [int(x) for x in dp.get("decision_points", [])],
                "n_clusters_dp": int(n_clusters_dp),
                "entropy_dp": float(h_dp),
                "effective_k_dp": float(eff_dp),
                "cluster_sizes_dp": cluster_sizes_dp,
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
        clusters_lcs, sig2c_lcs, _sig_table = _corridor_for_od_lcs(rids)
        dp = _cluster_decision_points_for_od(
            rids,
            way_sig=way_sig,
            way_xy_sig=way_xy_sig,
            min_choice_count=int(cfg.min_choice_count),
            way_tier=way_tier,
            dp_tier_keep=tier_keep,
            dp_next_min_keep=int(cfg.dp_next_min_keep),
        )
        clusters_dp = dp["clusters"]
        rid2c_dp = dp["rid2c"]

        method = str(cfg.corridor_method)
        if method == "decision_points":
            cid_list = [int(rid2c_dp.get(int(rid), 0)) for rid in rids]
            clusters_viz = clusters_dp
            title_extra = f"(DP:K={len(clusters_dp)}|LCS:K={len(clusters_lcs)})"
        else:
            cid_list = [int(sig2c_lcs.get(way_sig[int(rid)], 0)) for rid in rids]
            clusters_viz = clusters_lcs
            title_extra = f"(LCS:K={len(clusters_lcs)}|DP:K={len(clusters_dp)})"
        top_bins_for_viz.append(
            {
                "od_bin": [int(x) for x in od_k],
                "n_routes": int(len(rids)),
                "n_clusters": int(len(clusters_viz)),
                "cluster_sizes": [int(c["count"]) for c in clusters_viz],
                "route_ids": [int(r) for r in rids],
                "cluster_ids": cid_list,
                "decision_points": [int(x) for x in dp.get("decision_points", [])],
                "decision_points_all": [int(x) for x in dp.get("decision_points_all", [])],
                "dp_tier_keep": sorted(list(tier_keep)),
                "dp_next_min_keep": int(cfg.dp_next_min_keep),
                "decision_points_tier": {str(k): int(v) for k, v in dp.get("decision_points_tier", {}).items()},
                "decision_points_tier_all": {str(k): int(v) for k, v in dp.get("decision_points_tier_all", {}).items()},
                "decision_points_xy": {str(k): [float(v[0]), float(v[1])] for k, v in dp.get("decision_points_xy", {}).items()},
                "title_extra": str(title_extra),
            }
        )

    if out_viz_dir is not None:
        _plot_top_od_bins(out_dir=Path(out_viz_dir), od_bins=top_bins_for_viz, road_prob=road_prob, routes_xy=routes_xy, palette=palette)

    report: Dict[str, object] = {
        "ok": True,
        "task": "within_owner_corridor_diversity",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "segments_parquet": str(segments_parquet),
            "meta_zip": str(meta_zip),
            "road_prob_npy": (str(road_prob_npy) if road_prob_npy else None),
            "way_features_npz": (str(way_features_npz) if way_features_npz else None),
        },
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
    p.add_argument(
        "--way_features_npz",
        type=Path,
        default=None,
        help="Optional: way_features.npz (must contain way_osm_id + way_tier). Required when using --dp_tier_keep.",
    )

    p.add_argument("--od_bin_deg", type=float, default=0.02)
    p.add_argument("--min_od_dist_km", type=float, default=1.0)
    p.add_argument("--max_way_seq_len", type=int, default=128)
    p.add_argument("--merge_dist_thr", type=float, default=0.15)
    p.add_argument(
        "--corridor_method",
        type=str,
        default="decision_points",
        choices=["lcs", "decision_points"],
        help="Which corridor definition to use for Top-K visualization coloring (parquet always contains both LCS and decision-point metrics).",
    )
    p.add_argument(
        "--min_choice_count",
        type=int,
        default=2,
        help="Only used when corridor_method=decision_points. A valid branch option must appear in >= this many routes (per OD-bin group).",
    )
    p.add_argument(
        "--dp_tier_keep",
        type=str,
        default="",
        help="Optional: comma/space-separated way_tier ids to keep for decision points (e.g., '0' for major roads). Requires --way_features_npz. Tier ids follow build_way_features_from_osm_pbf.py: 0=major,1=minor,2=service,3=other.",
    )
    p.add_argument(
        "--dp_next_min_keep",
        type=int,
        default=2,
        help="Only used when dp_tier_keep is set. Keep a decision point u only if it has >= this many branch options v whose way_tier is also in the keep set (helps filter corridor-internal micro-branches).",
    )
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
        corridor_method=str(args.corridor_method),
        min_choice_count=int(args.min_choice_count),
        dp_tier_keep=str(args.dp_tier_keep),
        dp_next_min_keep=int(args.dp_next_min_keep),
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
        way_features_npz=(Path(args.way_features_npz) if args.way_features_npz is not None else None),
        cfg=cfg,
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
