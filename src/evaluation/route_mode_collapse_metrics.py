from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Config:
    jacc_cell: float
    collapse_ratio: float
    seed: int


def _key64(traj_idx: np.ndarray, start_t: np.ndarray) -> np.ndarray:
    traj_idx = np.asarray(traj_idx, dtype=np.int64).reshape(-1)
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    return (traj_idx << np.int64(32)) | (start_t & np.int64(0xFFFFFFFF))


def _polyline_features_to_dest_single(
    start_pos: np.ndarray,
    targets: np.ndarray,
    dest_pos: np.ndarray,
) -> np.ndarray:
    """
    Features with global destination as endpoint:
      - signed_dev_ratio (max signed perpendicular deviation / chord_len)
      - progress_ratio (endpoint progress along chord)
      - len_ratio (path_len / chord_len)
    """
    start_pos = np.asarray(start_pos, dtype=np.float64).reshape(2)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    dest_pos = np.asarray(dest_pos, dtype=np.float64).reshape(2)

    poly = np.concatenate([start_pos[None, :], targets], axis=0)  # (T,2)
    a = start_pos
    b = dest_pos
    ab = b - a
    chord = float(np.linalg.norm(ab)) + 1e-12

    ap = poly - a[None, :]
    cross = ab[0] * ap[:, 1] - ab[1] * ap[:, 0]
    dist_signed = cross / chord
    dist_signed[0] = 0.0
    idx = int(np.argmax(np.abs(dist_signed)))
    dev_signed = float(dist_signed[idx])
    signed_dev_ratio = float(dev_signed / chord)

    end_seg = poly[-1]
    proj = float(np.sum((end_seg - a) * ab) / (chord * chord))

    seg = poly[1:] - poly[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    path_len = float(np.sum(seg_len))
    len_ratio = float(path_len / chord)

    return np.asarray([signed_dev_ratio, proj, len_ratio], dtype=np.float64)


def _polyline_features_segment_end_single(start_pos: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Features with segment end as endpoint:
      - signed_dev_ratio
      - s_frac (arc-length fraction of max deviation)
      - len_ratio
    """
    start_pos = np.asarray(start_pos, dtype=np.float64).reshape(2)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    poly = np.concatenate([start_pos[None, :], targets], axis=0)

    a = poly[0]
    b = poly[-1]
    ab = b - a
    chord = float(np.linalg.norm(ab)) + 1e-12

    ap = poly - a[None, :]
    cross = ab[0] * ap[:, 1] - ab[1] * ap[:, 0]
    dist_signed = cross / chord
    dist_signed[0] = 0.0
    dist_signed[-1] = 0.0
    idx = int(np.argmax(np.abs(dist_signed)))
    dev_signed = float(dist_signed[idx])
    signed_dev_ratio = float(dev_signed / chord)

    seg = poly[1:] - poly[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(s[-1]) + 1e-12
    s_frac = float(s[idx] / total)

    path_len = float(np.sum(seg_len))
    len_ratio = float(path_len / chord)
    return np.asarray([signed_dev_ratio, s_frac, len_ratio], dtype=np.float64)


def _kmeans2(x: np.ndarray, *, seed: int, iters: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n < 2:
        return np.zeros((n,), dtype=np.int64), np.zeros((2, d), dtype=np.float64)

    i0 = int(np.argmin(x[:, 0]))
    i1 = int(np.argmax(x[:, 0]))
    if i0 == i1:
        rng = np.random.default_rng(int(seed))
        i1 = int(rng.integers(0, n))
    c = np.stack([x[i0], x[i1]], axis=0)

    labels = np.zeros((n,), dtype=np.int64)
    for _ in range(int(iters)):
        d0 = np.sum((x - c[0]) ** 2, axis=1)
        d1 = np.sum((x - c[1]) ** 2, axis=1)
        new_labels = (d1 < d0).astype(np.int64)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for k in (0, 1):
            mask = labels == k
            if not np.any(mask):
                continue
            c[k] = np.mean(x[mask], axis=0)
    return labels.astype(np.int64, copy=False), c


def _fit_two_corridors(feats: np.ndarray, *, seed: int) -> Dict[str, np.ndarray]:
    feats = np.asarray(feats, dtype=np.float64)
    mu = np.mean(feats, axis=0)
    sig = np.std(feats, axis=0) + 1e-6
    x = (feats - mu) / sig
    labels, centers = _kmeans2(x, seed=int(seed))
    return {
        "mu": mu.astype(np.float64, copy=False),
        "sig": sig.astype(np.float64, copy=False),
        "centers": centers.astype(np.float64, copy=False),
        "labels": labels.astype(np.int64, copy=False),
    }


def _assign_cluster(feat: np.ndarray, *, mu: np.ndarray, sig: np.ndarray, centers: np.ndarray) -> int:
    z = (np.asarray(feat, dtype=np.float64) - mu) / sig
    d0 = float(np.sum((z - centers[0]) ** 2))
    d1 = float(np.sum((z - centers[1]) ** 2))
    return 1 if d1 < d0 else 0


def _occupancy_set(start_pos: np.ndarray, path: np.ndarray, *, cell: float) -> set[int]:
    c = max(float(cell), 1e-6)
    pts = np.concatenate([np.asarray(start_pos, dtype=np.float64).reshape(1, 2), np.asarray(path, dtype=np.float64).reshape(-1, 2)], axis=0)
    yy = np.floor(pts[:, 0] / c).astype(np.int64)
    xx = np.floor(pts[:, 1] / c).astype(np.int64)
    h = (yy << np.int64(32)) ^ (xx & np.int64(0xFFFFFFFF))
    return set(int(v) for v in h.tolist())


def _mean_pairwise_jaccard_distance(sets: List[set[int]]) -> float:
    n = int(len(sets))
    if n < 2:
        return 0.0
    s = 0.0
    cnt = 0
    for i in range(n):
        a = sets[i]
        for j in range(i + 1, n):
            b = sets[j]
            inter = len(a & b)
            uni = len(a | b)
            jac = 0.0 if uni <= 0 else float(inter) / float(uni)
            s += 1.0 - jac
            cnt += 1
    return 0.0 if cnt <= 0 else float(s / float(cnt))


def run_metrics(
    *,
    gt_report_json: Path,
    gt_windows_npz: Path,
    model_samples_npz: Path,
    out_json: Path,
    cfg: Config,
) -> Dict[str, object]:
    report = json.loads(gt_report_json.read_text(encoding="utf-8"))
    cases = report.get("selected_cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("gt_report_json missing selected_cases")

    gt = np.load(str(gt_windows_npz), allow_pickle=True)
    need_gt = {"start_pos", "targets", "traj_idx", "start_t"}
    if not need_gt.issubset(set(gt.files)):
        raise ValueError(f"gt_windows_npz must contain {sorted(need_gt)}, got {sorted(list(gt.files))}")
    gt_start = np.asarray(gt["start_pos"], dtype=np.float32)
    gt_targets = np.asarray(gt["targets"], dtype=np.float32)
    gt_dest = np.asarray(gt["dest_pos"], dtype=np.float32) if "dest_pos" in gt.files else None
    gt_key = _key64(np.asarray(gt["traj_idx"]), np.asarray(gt["start_t"]))
    gt_map = {int(k): int(i) for i, k in enumerate(gt_key.tolist())}

    ms = np.load(str(model_samples_npz), allow_pickle=True)
    need_ms = {"preds_k", "start_pos", "traj_idx", "start_t"}
    if not need_ms.issubset(set(ms.files)):
        raise ValueError(f"model_samples_npz must contain {sorted(need_ms)}, got {sorted(list(ms.files))}")
    preds_k = np.asarray(ms["preds_k"], dtype=np.float32)  # (N,K,F,2)
    ms_start = np.asarray(ms["start_pos"], dtype=np.float32)
    ms_dest = np.asarray(ms["dest_pos"], dtype=np.float32) if "dest_pos" in ms.files else None
    ms_key = _key64(np.asarray(ms["traj_idx"]), np.asarray(ms["start_t"]))
    ms_map = {int(k): int(i) for i, k in enumerate(ms_key.tolist())}

    od_end = str(report.get("config", {}).get("od_end", "dest_pos"))

    out_cases: List[Dict[str, object]] = []
    all_div = []
    all_cov = []
    all_collapse = []

    for c in cases:
        ids = c.get("window_ids")
        if not isinstance(ids, dict) or ("traj_idx" not in ids) or ("start_t" not in ids):
            continue
        tid = np.asarray(ids["traj_idx"], dtype=np.int64).reshape(-1)
        t0 = np.asarray(ids["start_t"], dtype=np.int64).reshape(-1)
        keys = _key64(tid, t0)

        gt_idx = [gt_map.get(int(k)) for k in keys.tolist()]
        gt_idx = [i for i in gt_idx if i is not None]
        if len(gt_idx) < 4:
            continue
        gt_idx_arr = np.asarray(gt_idx, dtype=np.int64)
        sp = gt_start[gt_idx_arr]
        tg = gt_targets[gt_idx_arr]
        dp = (gt_dest[gt_idx_arr] if gt_dest is not None else None)

        feats = []
        if od_end == "dest_pos" and dp is not None:
            for i in range(int(gt_idx_arr.size)):
                feats.append(_polyline_features_to_dest_single(sp[i], tg[i], dp[i]))
        else:
            for i in range(int(gt_idx_arr.size)):
                feats.append(_polyline_features_segment_end_single(sp[i], tg[i]))
        feats_arr = np.stack(feats, axis=0)  # (n,3)
        cl = _fit_two_corridors(feats_arr, seed=int(cfg.seed))

        # Match model windows.
        ms_idx = [ms_map.get(int(k)) for k in keys.tolist()]
        ms_idx = [i for i in ms_idx if i is not None]
        if len(ms_idx) == 0:
            continue
        ms_idx_arr = np.asarray(ms_idx, dtype=np.int64)
        div_list = []
        cov_list = []
        collapse_list = []

        d_gt = float(c.get("gt_jaccard_distance", {}).get("mean", 0.0))
        thr = float(cfg.collapse_ratio) * float(d_gt)

        for i in ms_idx_arr.tolist():
            start_i = ms_start[int(i)]
            dest_i = ms_dest[int(i)] if ms_dest is not None else None
            pk = preds_k[int(i)]  # (K,F,2)

            sets = [_occupancy_set(start_i, pk[k], cell=float(cfg.jacc_cell)) for k in range(int(pk.shape[0]))]
            d_model = _mean_pairwise_jaccard_distance(sets)
            div_list.append(float(d_model))

            # Coverage vs GT clusters (2 modes).
            hit = set()
            for k in range(int(pk.shape[0])):
                if od_end == "dest_pos" and dest_i is not None:
                    feat = _polyline_features_to_dest_single(start_i, pk[k], dest_i)
                else:
                    feat = _polyline_features_segment_end_single(start_i, pk[k])
                lab = _assign_cluster(feat, mu=cl["mu"], sig=cl["sig"], centers=cl["centers"])
                hit.add(int(lab))
            cov = float(len(hit)) / 2.0
            cov_list.append(float(cov))

            collapse_list.append(bool(d_model < thr) if d_gt > 0 else False)

        out_cases.append(
            {
                "case_id": c.get("case_id"),
                "n_model_matched": int(len(ms_idx_arr)),
                "gt_div_mean": float(d_gt),
                "collapse_threshold": float(thr),
                "model_div": {
                    "mean": float(np.mean(div_list)) if div_list else 0.0,
                    "p50": float(np.percentile(div_list, 50)) if div_list else 0.0,
                },
                "model_coverage": {
                    "mean": float(np.mean(cov_list)) if cov_list else 0.0,
                    "p50": float(np.percentile(cov_list, 50)) if cov_list else 0.0,
                },
                "collapse_rate": float(np.mean(collapse_list)) if collapse_list else 0.0,
            }
        )

        all_div.extend(div_list)
        all_cov.extend(cov_list)
        all_collapse.extend(collapse_list)

    out = {
        "inputs": {
            "gt_report_json": str(gt_report_json),
            "gt_windows_npz": str(gt_windows_npz),
            "model_samples_npz": str(model_samples_npz),
        },
        "config": {
            "od_end": str(od_end),
            "jacc_cell": float(cfg.jacc_cell),
            "collapse_ratio": float(cfg.collapse_ratio),
            "seed": int(cfg.seed),
        },
        "overall": {
            "num_cases_evaluated": int(len(out_cases)),
            "model_div_mean": float(np.mean(all_div)) if all_div else 0.0,
            "model_cov_mean": float(np.mean(all_cov)) if all_cov else 0.0,
            "collapse_rate_mean": float(np.mean(all_collapse)) if all_collapse else 0.0,
        },
        "per_case": out_cases,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute mode-collapse metrics vs GT baseline for fixed diagnostic cases (JSON-only).")
    p.add_argument("--gt_report_json", type=str, required=True)
    p.add_argument("--gt_windows_npz", type=str, required=True)
    p.add_argument("--model_samples_npz", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--jacc_cell", type=float, default=8.0)
    p.add_argument("--collapse_ratio", type=float, default=0.5, help="collapse if D_model < collapse_ratio * D_GT")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config(jacc_cell=float(args.jacc_cell), collapse_ratio=float(args.collapse_ratio), seed=int(args.seed))
    out = run_metrics(
        gt_report_json=Path(args.gt_report_json),
        gt_windows_npz=Path(args.gt_windows_npz),
        model_samples_npz=Path(args.model_samples_npz),
        out_json=Path(args.out_json),
        cfg=cfg,
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

