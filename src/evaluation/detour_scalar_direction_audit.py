from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Loaded:
    start_pos: np.ndarray  # (N,2)
    targets: np.ndarray  # (N,F,2)
    preds: np.ndarray  # (N,F,2)
    traj_idx: Optional[np.ndarray] = None  # (N,)
    start_t: Optional[np.ndarray] = None  # (N,)


def _parse_inputs(items: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in items:
        if ":" not in raw:
            raise ValueError(f"Invalid --inputs item '{raw}'. Expected 'Label:Path'.")
        label, path = raw.split(":", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid --inputs item '{raw}'. Expected 'Label:Path'.")
        out[label] = path
    return out


def _window_keys(traj_idx: np.ndarray, start_t: np.ndarray) -> np.ndarray:
    traj_idx = np.asarray(traj_idx, dtype=np.int64).reshape(-1)
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    if traj_idx.shape[0] != start_t.shape[0]:
        raise ValueError(f"Bad traj_idx/start_t shapes: traj_idx={traj_idx.shape} start_t={start_t.shape}")
    return (traj_idx << np.int64(32)) | (start_t & np.int64(0xFFFFFFFF))


def _align_by_window_ids(loaded: Dict[str, Loaded]) -> Tuple[Dict[str, Loaded], Dict[str, object]]:
    names = list(loaded.keys())
    if not names:
        raise ValueError("No inputs.")
    if any(loaded[n].traj_idx is None or loaded[n].start_t is None for n in names):
        raise ValueError("All inputs must contain traj_idx and start_t for alignment.")

    keys_by_name: Dict[str, np.ndarray] = {}
    for name in names:
        keys_by_name[name] = _window_keys(loaded[name].traj_idx, loaded[name].start_t)  # type: ignore[arg-type]
        if int(np.unique(keys_by_name[name]).size) != int(keys_by_name[name].size):
            raise ValueError(f"Duplicate window ids detected in {name}; cannot align safely.")

    common = keys_by_name[names[0]]
    for name in names[1:]:
        common = np.intersect1d(common, keys_by_name[name], assume_unique=False)
    if common.size == 0:
        raise ValueError("No common windows across inputs (traj_idx/start_t intersection is empty).")

    # Preserve reference order (the first input).
    ref_keys = keys_by_name[names[0]]
    keep_mask = np.isin(ref_keys, common)
    order = ref_keys[keep_mask]

    stats: Dict[str, object] = {
        "aligned_by": "traj_idx_start_t",
        "common_N": int(order.size),
        "dropped": {},
    }

    aligned: Dict[str, Loaded] = {}
    for name in names:
        k = keys_by_name[name]
        idx_map = {int(v): int(i) for i, v in enumerate(k)}
        idx = np.asarray([idx_map[int(v)] for v in order], dtype=np.int64)
        stats["dropped"][name] = int(k.size - idx.size)
        src = loaded[name]
        aligned[name] = Loaded(
            start_pos=np.asarray(src.start_pos)[idx],
            targets=np.asarray(src.targets)[idx],
            preds=np.asarray(src.preds)[idx],
            traj_idx=np.asarray(src.traj_idx)[idx] if src.traj_idx is not None else None,
            start_t=np.asarray(src.start_t)[idx] if src.start_t is not None else None,
        )

    return aligned, stats


def _load_one(path: Path) -> Loaded:
    data = np.load(str(path), allow_pickle=True)
    need = {"start_pos", "targets"}
    miss = [k for k in sorted(need) if k not in data.files]
    if miss:
        raise ValueError(f"{path} missing keys: {miss}. got={list(data.files)}")

    start_pos = np.asarray(data["start_pos"], dtype=np.float32)
    targets = np.asarray(data["targets"], dtype=np.float32)

    if "preds" in data.files:
        preds = np.asarray(data["preds"], dtype=np.float32)
    elif "preds_k" in data.files:
        pk = np.asarray(data["preds_k"], dtype=np.float32)
        if pk.ndim != 4:
            raise ValueError(f"{path} bad preds_k shape: {pk.shape} (expected N,K,F,2)")
        preds = pk[:, 0]
    else:
        raise ValueError(f"{path} missing preds/preds_k. got={list(data.files)}")

    traj_idx = np.asarray(data["traj_idx"], dtype=np.int64) if "traj_idx" in data.files else None
    start_t = np.asarray(data["start_t"], dtype=np.int64) if "start_t" in data.files else None

    if start_pos.ndim != 2 or start_pos.shape[1] != 2:
        raise ValueError(f"{path} bad start_pos shape: {start_pos.shape}")
    if targets.ndim != 3 or targets.shape[2] != 2:
        raise ValueError(f"{path} bad targets shape: {targets.shape}")
    if preds.ndim != 3 or preds.shape[2] != 2:
        raise ValueError(f"{path} bad preds shape: {preds.shape}")
    if int(start_pos.shape[0]) != int(targets.shape[0]) or int(start_pos.shape[0]) != int(preds.shape[0]):
        raise ValueError(f"{path} N mismatch: start={start_pos.shape} targets={targets.shape} preds={preds.shape}")

    return Loaded(start_pos=start_pos, targets=targets, preds=preds, traj_idx=traj_idx, start_t=start_t)


def _max_lateral_deviation_ratio(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return 0.0
    a = points[0].astype(np.float64)
    b = points[-1].astype(np.float64)
    ab = b - a
    chord = float(np.linalg.norm(ab))
    if chord <= 1e-6:
        return 0.0
    denom = chord + 1e-12
    ap = points.astype(np.float64) - a[None, :]
    cross = np.abs(ab[0] * ap[:, 1] - ab[1] * ap[:, 0])
    d = cross / denom  # perpendicular distance to chord line
    d[0] = 0.0
    d[-1] = 0.0
    return float(np.max(d) / chord)


def _path_length_ratio(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return 1.0
    seg = points[1:] - points[:-1]
    length = float(np.sum(np.linalg.norm(seg, axis=1)))
    chord = float(np.linalg.norm(points[-1] - points[0])) + 1e-12
    return float(length / chord)


def _poly_from_start_and_seq(start_pos: np.ndarray, seq: np.ndarray) -> np.ndarray:
    return np.concatenate([start_pos[None, :], seq], axis=0).astype(np.float32, copy=False)


def _quantiles(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "mean": float("nan")}
    p10, p50, p90 = np.percentile(x, [10, 50, 90]).tolist()
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90), "mean": float(np.mean(x))}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detour scalar directionality audit: report raw quantiles of max_dev_ratio/len_ratio for GT vs each method."
    )
    p.add_argument("--inputs", type=str, nargs="+", required=True, help="Repeatable: 'Label:/path/to/samples.npz'")
    p.add_argument("--detour_pct", type=float, default=10.0, help="Define detour subset as top pct by GT max_dev_ratio.")
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--quiet", action="store_true", help="Suppress console prints (write JSON only).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    inputs = _parse_inputs(args.inputs)
    loaded: Dict[str, Loaded] = {name: _load_one(Path(p)) for name, p in inputs.items()}

    align_stats = None
    if all(v.traj_idx is not None and v.start_t is not None for v in loaded.values()):
        loaded, align_stats = _align_by_window_ids(loaded)

    names = list(loaded.keys())
    ref = loaded[names[0]]
    N = int(ref.start_pos.shape[0])
    F = int(ref.targets.shape[1])

    gt_dev = np.zeros((N,), dtype=np.float64)
    gt_len = np.zeros((N,), dtype=np.float64)
    for i in range(N):
        poly_gt = _poly_from_start_and_seq(ref.start_pos[i], ref.targets[i])
        gt_dev[i] = _max_lateral_deviation_ratio(poly_gt)
        gt_len[i] = _path_length_ratio(poly_gt)

    detour_pct = float(args.detour_pct)
    detour_mask = np.zeros((N,), dtype=bool)
    detour_thr = float("nan")
    if np.isfinite(detour_pct) and detour_pct > 0.0 and detour_pct < 100.0:
        detour_thr = float(np.percentile(gt_dev, 100.0 - detour_pct))
        detour_mask = gt_dev >= detour_thr
    elif detour_pct >= 100.0:
        detour_thr = float(np.min(gt_dev)) if gt_dev.size else float("nan")
        detour_mask[:] = True

    out: Dict[str, object] = {
        "stats": {
            "N": N,
            "F": F,
            "detour_pct": detour_pct,
            "detour_thr_max_dev_ratio": detour_thr,
            "detour_size": int(np.sum(detour_mask)),
        },
        "gt": {
            "overall": {"max_dev_ratio": _quantiles(gt_dev), "len_ratio": _quantiles(gt_len)},
            "detour": {
                "max_dev_ratio": _quantiles(gt_dev[detour_mask]) if int(np.sum(detour_mask)) > 0 else _quantiles(np.zeros((0,))),
                "len_ratio": _quantiles(gt_len[detour_mask]) if int(np.sum(detour_mask)) > 0 else _quantiles(np.zeros((0,))),
            },
        },
        "methods": {},
    }
    if align_stats is not None:
        out["stats"]["alignment"] = align_stats

    for name, data in loaded.items():
        pred_dev = np.zeros((N,), dtype=np.float64)
        pred_len = np.zeros((N,), dtype=np.float64)
        for i in range(N):
            poly = _poly_from_start_and_seq(data.start_pos[i], data.preds[i])
            pred_dev[i] = _max_lateral_deviation_ratio(poly)
            pred_len[i] = _path_length_ratio(poly)

        meth = {
            "overall": {
                "max_dev_ratio": _quantiles(pred_dev),
                "len_ratio": _quantiles(pred_len),
                "delta_mean_vs_gt": {
                    "max_dev_ratio": float(np.mean(pred_dev) - np.mean(gt_dev)),
                    "len_ratio": float(np.mean(pred_len) - np.mean(gt_len)),
                },
                "delta_p50_vs_gt": {
                    "max_dev_ratio": float(np.percentile(pred_dev, 50) - np.percentile(gt_dev, 50)),
                    "len_ratio": float(np.percentile(pred_len, 50) - np.percentile(gt_len, 50)),
                },
            },
            "detour": {
                "max_dev_ratio": _quantiles(pred_dev[detour_mask]) if int(np.sum(detour_mask)) > 0 else _quantiles(np.zeros((0,))),
                "len_ratio": _quantiles(pred_len[detour_mask]) if int(np.sum(detour_mask)) > 0 else _quantiles(np.zeros((0,))),
                "delta_mean_vs_gt": {
                    "max_dev_ratio": float(np.mean(pred_dev[detour_mask]) - np.mean(gt_dev[detour_mask])) if int(np.sum(detour_mask)) > 0 else float("nan"),
                    "len_ratio": float(np.mean(pred_len[detour_mask]) - np.mean(gt_len[detour_mask])) if int(np.sum(detour_mask)) > 0 else float("nan"),
                },
                "delta_p50_vs_gt": {
                    "max_dev_ratio": float(np.percentile(pred_dev[detour_mask], 50) - np.percentile(gt_dev[detour_mask], 50)) if int(np.sum(detour_mask)) > 0 else float("nan"),
                    "len_ratio": float(np.percentile(pred_len[detour_mask], 50) - np.percentile(gt_len[detour_mask], 50)) if int(np.sum(detour_mask)) > 0 else float("nan"),
                },
            },
        }
        out["methods"][name] = meth

    if not bool(args.quiet):
        print("============================================================")
        print("DET0UR SCALAR DIRECTION AUDIT (raw quantiles)")
        print("============================================================")
        print(f"N={N}  detour_pct={detour_pct}  detour_size={int(np.sum(detour_mask))}")
        print("--- GT ---")
        gt_o = out["gt"]["overall"]  # type: ignore[assignment]
        print(f"GT max_dev_ratio p50={gt_o['max_dev_ratio']['p50']:.4f}  len_ratio p50={gt_o['len_ratio']['p50']:.4f}")
        for name in inputs.keys():
            m = out["methods"][name]["overall"]  # type: ignore[index]
            d = m["delta_p50_vs_gt"]  # type: ignore[index]
            print(f"- {name}: Δp50 max_dev_ratio={d['max_dev_ratio']:+.4f}  Δp50 len_ratio={d['len_ratio']:+.4f}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        if not bool(args.quiet):
            print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
