from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.features.semantic_od import load_osm_road_prob
from src.training.route_npz_utils import load_route_windows_npz


@dataclass(frozen=True)
class Config:
    jacc_cell: float
    road_prob_thr: float
    seed: int


def _key64(traj_idx: np.ndarray, start_t: np.ndarray) -> np.ndarray:
    traj_idx = np.asarray(traj_idx, dtype=np.int64).reshape(-1)
    start_t = np.asarray(start_t, dtype=np.int64).reshape(-1)
    return (traj_idx << np.int64(32)) | (start_t & np.int64(0xFFFFFFFF))


def _occupancy_set(start_pos: np.ndarray, path: np.ndarray, *, cell: float) -> set[int]:
    c = max(float(cell), 1e-6)
    pts = np.concatenate(
        [np.asarray(start_pos, dtype=np.float64).reshape(1, 2), np.asarray(path, dtype=np.float64).reshape(-1, 2)],
        axis=0,
    )
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


def _sample_road_prob(road_prob: np.ndarray, pos: np.ndarray) -> np.ndarray:
    road_prob = np.asarray(road_prob, dtype=np.float32)
    if road_prob.ndim != 2:
        raise ValueError(f"Expected road_prob (H,W), got {road_prob.shape}")
    H, W = int(road_prob.shape[0]), int(road_prob.shape[1])

    pos = np.asarray(pos, dtype=np.float32)
    yy = np.rint(pos[..., 0]).astype(np.int64)
    xx = np.rint(pos[..., 1]).astype(np.int64)
    inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    yy_c = np.clip(yy, 0, H - 1)
    xx_c = np.clip(xx, 0, W - 1)
    out = road_prob[yy_c, xx_c].astype(np.float32, copy=False)
    out = out * inb.astype(np.float32, copy=False)
    return out


def _compute_micro_metrics(preds_k: np.ndarray, gt: np.ndarray) -> Dict[str, np.ndarray]:
    preds_k = np.asarray(preds_k, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if preds_k.ndim != 4 or preds_k.shape[-1] != 2:
        raise ValueError(f"Expected preds_k (N,K,F,2), got {preds_k.shape}")
    if gt.ndim != 3 or gt.shape[-1] != 2:
        raise ValueError(f"Expected gt (N,F,2), got {gt.shape}")
    if preds_k.shape[0] != gt.shape[0] or preds_k.shape[2] != gt.shape[1]:
        raise ValueError(f"N/F mismatch: preds_k={preds_k.shape} gt={gt.shape}")

    diff = preds_k - gt[:, None, :, :]
    dist = np.linalg.norm(diff.astype(np.float64), axis=-1).astype(np.float32)  # (N,K,F)
    ade_k = dist.mean(axis=2)  # (N,K)
    fde_k = dist[:, :, -1]  # (N,K)
    return {
        "ade_mean": ade_k.mean(axis=1),
        "ade_std": ade_k.std(axis=1),
        "ade_best": ade_k.min(axis=1),
        "fde_mean": fde_k.mean(axis=1),
        "fde_std": fde_k.std(axis=1),
        "fde_best": fde_k.min(axis=1),
    }


def run_metrics(
    *,
    gt_windows_npz: Path,
    model_samples_npz: Path,
    out_json: Path,
    cfg: Config,
    semantic_dir: Optional[Path],
    max_n: Optional[int],
) -> Dict[str, object]:
    gt = load_route_windows_npz(str(gt_windows_npz), max_n=None, seed=int(cfg.seed))
    gt_start = np.asarray(gt["start_pos"], dtype=np.float32)
    gt_targets = np.asarray(gt["targets"], dtype=np.float32)
    gt_dest = np.asarray(gt["dest_pos"], dtype=np.float32)
    gt_traj_idx = np.asarray(gt["traj_idx"], dtype=np.int64)
    gt_start_t = np.asarray(gt["start_t"], dtype=np.int64)
    gt_key = _key64(gt_traj_idx, gt_start_t)
    gt_map = {int(k): int(i) for i, k in enumerate(gt_key.tolist())}

    ms = np.load(str(model_samples_npz), allow_pickle=True)
    need_ms = {"preds_k", "start_pos", "traj_idx", "start_t"}
    if not need_ms.issubset(set(ms.files)):
        raise ValueError(f"model_samples_npz must contain {sorted(need_ms)}, got {sorted(list(ms.files))}")
    preds_k_all = np.asarray(ms["preds_k"], dtype=np.float32)
    ms_start = np.asarray(ms["start_pos"], dtype=np.float32)
    ms_dest = np.asarray(ms["dest_pos"], dtype=np.float32) if "dest_pos" in ms.files else None
    ms_traj_idx = np.asarray(ms["traj_idx"], dtype=np.int64)
    ms_start_t = np.asarray(ms["start_t"], dtype=np.int64)
    ms_key = _key64(ms_traj_idx, ms_start_t)
    ms_map = {int(k): int(i) for i, k in enumerate(ms_key.tolist())}

    keys_common = [k for k in gt_map.keys() if k in ms_map]
    if not keys_common:
        raise RuntimeError("No matched windows between gt_windows_npz and model_samples_npz (traj_idx/start_t mismatch).")

    if max_n is not None:
        k = int(max_n)
        if k > 0 and len(keys_common) > k:
            rng = np.random.default_rng(int(cfg.seed))
            pick = rng.choice(len(keys_common), size=k, replace=False)
            keys_common = [keys_common[int(i)] for i in np.sort(pick)]

    gt_idx = np.asarray([gt_map[int(k)] for k in keys_common], dtype=np.int64)
    ms_idx = np.asarray([ms_map[int(k)] for k in keys_common], dtype=np.int64)

    gt_targets_m = gt_targets[gt_idx]
    preds_k = preds_k_all[ms_idx]

    micro = _compute_micro_metrics(preds_k, gt_targets_m)

    k_samples = int(preds_k.shape[1])
    div_list = []
    for i in range(int(preds_k.shape[0])):
        sets = [_occupancy_set(ms_start[int(ms_idx[i])], preds_k[i, kk], cell=float(cfg.jacc_cell)) for kk in range(k_samples)]
        div_list.append(_mean_pairwise_jaccard_distance(sets))
    div_arr = np.asarray(div_list, dtype=np.float32)

    road_prob = None
    if semantic_dir is not None:
        road_prob = load_osm_road_prob(str(semantic_dir))
        if road_prob.shape[0] < 2 or road_prob.shape[1] < 2:
            raise ValueError(f"Bad osm_road_prob.npy shape: {road_prob.shape}")

    road_metrics = {}
    if road_prob is not None:
        rp_pred = _sample_road_prob(road_prob, preds_k)  # (N,K,F)
        on_pred = (rp_pred >= float(cfg.road_prob_thr)).astype(np.float32, copy=False)
        road_metrics.update(
            {
                "pred_road_prob_mean": float(np.mean(rp_pred)),
                "pred_onroad_rate": float(np.mean(on_pred)),
            }
        )

        rp_gt = _sample_road_prob(road_prob, gt_targets_m)  # (N,F)
        on_gt = (rp_gt >= float(cfg.road_prob_thr)).astype(np.float32, copy=False)
        road_metrics.update(
            {
                "gt_road_prob_mean": float(np.mean(rp_gt)),
                "gt_onroad_rate": float(np.mean(on_gt)),
            }
        )

    out: Dict[str, object] = {
        "inputs": {"gt_windows_npz": str(gt_windows_npz), "model_samples_npz": str(model_samples_npz)},
        "config": {
            "K": int(k_samples),
            "max_n": (int(max_n) if max_n is not None else None),
            "jacc_cell": float(cfg.jacc_cell),
            "road_prob_thr": float(cfg.road_prob_thr),
            "semantic_dir": (str(semantic_dir) if semantic_dir is not None else None),
            "seed": int(cfg.seed),
        },
        "stats": {"num_windows_matched": int(preds_k.shape[0]), "F": int(preds_k.shape[2])},
        "micro": {
            "ADE_mean": float(np.mean(micro["ade_mean"])),
            "ADE_std": float(np.mean(micro["ade_std"])),
            "ADE_best": float(np.mean(micro["ade_best"])),
            "FDE_mean": float(np.mean(micro["fde_mean"])),
            "FDE_std": float(np.mean(micro["fde_std"])),
            "FDE_best": float(np.mean(micro["fde_best"])),
        },
        "diversity": {
            "jaccard_mean": float(np.mean(div_arr)),
            "jaccard_p50": float(np.percentile(div_arr, 50)),
        },
        "feasibility": road_metrics,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full-scale metrics for route generation (JSON-only): ADE/FDE (mean/std/best), diversity (Jaccard), and optional on-road rate from OSM road_prob.")
    p.add_argument("--gt_windows_npz", type=str, required=True)
    p.add_argument("--model_samples_npz", type=str, required=True, help="samples.npz containing preds_k and traj_idx/start_t.")
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--semantic_dir", type=str, default=None, help="If set, loads osm_road_prob.npy from this directory and reports on-road rate.")
    p.add_argument("--road_prob_thr", type=float, default=0.5)
    p.add_argument("--jacc_cell", type=float, default=8.0)
    p.add_argument("--max_n", type=int, default=None, help="Optional: subsample matched windows for a quick run.")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config(jacc_cell=float(args.jacc_cell), road_prob_thr=float(args.road_prob_thr), seed=int(args.seed))
    report = run_metrics(
        gt_windows_npz=Path(args.gt_windows_npz),
        model_samples_npz=Path(args.model_samples_npz),
        out_json=Path(args.out_json),
        cfg=cfg,
        semantic_dir=(Path(args.semantic_dir) if args.semantic_dir else None),
        max_n=(int(args.max_n) if args.max_n is not None else None),
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

