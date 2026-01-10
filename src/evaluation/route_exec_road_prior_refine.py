from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):  # type: ignore[no-redef]
        return x


@dataclass(frozen=True)
class Config:
    iters: int
    step: float
    weight_scale_m: float
    chunk_n: int
    seed: int


def _load_raster(path: Path, *, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    a = np.load(path).astype(np.float32, copy=False)
    if a.ndim != 2:
        raise ValueError(f"Bad {name} shape: {a.shape} (expected H,W)")
    return a


def _load_dist_and_grad(semantic_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist = _load_raster(semantic_dir / "osm_dist_to_road_m.npy", name="osm_dist_to_road_m.npy")
    gy, gx = np.gradient(dist.astype(np.float32, copy=False))
    return dist, gy.astype(np.float32, copy=False), gx.astype(np.float32, copy=False)


def _load_road_prob_and_grad(semantic_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rp = _load_raster(semantic_dir / "osm_road_prob.npy", name="osm_road_prob.npy")
    gy, gx = np.gradient(rp.astype(np.float32, copy=False))
    return rp, gy.astype(np.float32, copy=False), gx.astype(np.float32, copy=False)


def _refine_chunk_with_dist(
    preds: np.ndarray,  # (n,K,F,2)
    *,
    dest_pos: Optional[np.ndarray],  # (n,2)
    dist: np.ndarray,
    gy: np.ndarray,
    gx: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    H, W = int(dist.shape[0]), int(dist.shape[1])
    out = np.asarray(preds, dtype=np.float32, copy=True)
    eps = 1e-6
    for _ in range(int(cfg.iters)):
        pos = out[:, :, :-1, :]  # exclude last (keep dest fixed)
        yy = np.rint(pos[..., 0]).astype(np.int32)
        xx = np.rint(pos[..., 1]).astype(np.int32)
        yy = np.clip(yy, 0, H - 1)
        xx = np.clip(xx, 0, W - 1)

        d = dist[yy, xx]  # (n,K,F-1)
        w = np.clip(d / max(float(cfg.weight_scale_m), 1e-6), 0.0, 1.0).astype(np.float32, copy=False)

        g_y = gy[yy, xx]
        g_x = gx[yy, xx]
        g_norm = np.sqrt(g_y * g_y + g_x * g_x) + float(eps)
        # Move towards smaller dist (towards road): -grad(dist).
        step = float(cfg.step)
        pos[..., 0] -= (step * w) * (g_y / g_norm)
        pos[..., 1] -= (step * w) * (g_x / g_norm)
        pos[..., 0] = np.clip(pos[..., 0], 0.0, float(H - 1))
        pos[..., 1] = np.clip(pos[..., 1], 0.0, float(W - 1))

        out[:, :, :-1, :] = pos.astype(np.float32, copy=False)
        if dest_pos is not None:
            out[:, :, -1, :] = dest_pos[:, None, :].astype(np.float32, copy=False)
    return out.astype(np.float32, copy=False)


def _refine_chunk_with_road_prob(
    preds: np.ndarray,  # (n,K,F,2)
    *,
    dest_pos: Optional[np.ndarray],  # (n,2)
    road_prob: np.ndarray,
    gy: np.ndarray,
    gx: np.ndarray,
    road_prob_thr: float,
    cfg: Config,
) -> np.ndarray:
    H, W = int(road_prob.shape[0]), int(road_prob.shape[1])
    out = np.asarray(preds, dtype=np.float32, copy=True)
    eps = 1e-6
    thr = float(road_prob_thr)
    for _ in range(int(cfg.iters)):
        pos = out[:, :, :-1, :]
        yy = np.rint(pos[..., 0]).astype(np.int32)
        xx = np.rint(pos[..., 1]).astype(np.int32)
        yy = np.clip(yy, 0, H - 1)
        xx = np.clip(xx, 0, W - 1)

        rp = road_prob[yy, xx]
        # Push more when below threshold.
        w = np.clip((thr - rp) / max(thr, 1e-6), 0.0, 1.0).astype(np.float32, copy=False)

        g_y = gy[yy, xx]
        g_x = gx[yy, xx]
        g_norm = np.sqrt(g_y * g_y + g_x * g_x) + float(eps)
        step = float(cfg.step)
        pos[..., 0] += (step * w) * (g_y / g_norm)
        pos[..., 1] += (step * w) * (g_x / g_norm)
        pos[..., 0] = np.clip(pos[..., 0], 0.0, float(H - 1))
        pos[..., 1] = np.clip(pos[..., 1], 0.0, float(W - 1))

        out[:, :, :-1, :] = pos.astype(np.float32, copy=False)
        if dest_pos is not None:
            out[:, :, -1, :] = dest_pos[:, None, :].astype(np.float32, copy=False)
    return out.astype(np.float32, copy=False)


def run_refine(
    *,
    in_samples_npz: Path,
    semantic_dir: Path,
    out_samples_npz: Path,
    out_json: Path,
    cfg: Config,
    mode: str,
    road_prob_thr: float,
) -> Dict[str, object]:
    data = np.load(str(in_samples_npz), allow_pickle=True)
    if "preds_k" not in data.files:
        raise ValueError(f"in_samples_npz missing preds_k: {sorted(list(data.files))}")
    preds_k = np.asarray(data["preds_k"], dtype=np.float32)
    if preds_k.ndim != 4 or preds_k.shape[-1] != 2:
        raise ValueError(f"Expected preds_k (N,K,F,2), got {preds_k.shape}")
    dest_pos = np.asarray(data["dest_pos"], dtype=np.float32) if "dest_pos" in data.files else None

    N = int(preds_k.shape[0])
    chunk_n = int(cfg.chunk_n)
    if chunk_n <= 0:
        chunk_n = N

    if mode == "dist":
        dist, gy, gx = _load_dist_and_grad(semantic_dir)
        ref_fn = lambda chunk, dp: _refine_chunk_with_dist(chunk, dest_pos=dp, dist=dist, gy=gy, gx=gx, cfg=cfg)
        prior_info = {"mode": "dist", "dist_file": str((semantic_dir / "osm_dist_to_road_m.npy").resolve())}
    elif mode == "road_prob":
        road_prob, gy, gx = _load_road_prob_and_grad(semantic_dir)
        ref_fn = lambda chunk, dp: _refine_chunk_with_road_prob(
            chunk,
            dest_pos=dp,
            road_prob=road_prob,
            gy=gy,
            gx=gx,
            road_prob_thr=float(road_prob_thr),
            cfg=cfg,
        )
        prior_info = {"mode": "road_prob", "road_prob_file": str((semantic_dir / "osm_road_prob.npy").resolve()), "road_prob_thr": float(road_prob_thr)}
    else:
        raise ValueError("--mode must be one of {dist, road_prob}")

    out_preds = np.empty_like(preds_k, dtype=np.float32)
    for i0 in tqdm(range(0, N, chunk_n), desc="refine", dynamic_ncols=True):
        i1 = min(N, int(i0 + chunk_n))
        dp = (dest_pos[i0:i1] if dest_pos is not None else None)
        out_preds[i0:i1] = ref_fn(preds_k[i0:i1], dp)

    out_kwargs = {}
    for k in data.files:
        if k == "preds_k":
            continue
        out_kwargs[k] = data[k]
    out_kwargs["preds_k"] = out_preds

    out_samples_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_samples_npz, **out_kwargs)

    report: Dict[str, object] = {
        "inputs": {"in_samples_npz": str(in_samples_npz), "semantic_dir": str(semantic_dir)},
        "config": {
            "mode": str(mode),
            "iters": int(cfg.iters),
            "step": float(cfg.step),
            "weight_scale_m": float(cfg.weight_scale_m),
            "chunk_n": int(chunk_n),
            "seed": int(cfg.seed),
        },
        "prior": prior_info,
        "stats": {"N": int(preds_k.shape[0]), "K": int(preds_k.shape[1]), "F": int(preds_k.shape[2])},
        "outputs": {"out_samples_npz": str(out_samples_npz)},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Execution-stage feasibility refinement using OSM priors (JSON-only meta).")
    p.add_argument("--in_samples_npz", type=str, required=True, help="samples.npz containing preds_k (N,K,F,2).")
    p.add_argument("--semantic_dir", type=str, required=True, help="Directory containing osm_dist_to_road_m.npy and/or osm_road_prob.npy.")
    p.add_argument("--out_samples_npz", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["dist", "road_prob"], default="dist")
    p.add_argument("--road_prob_thr", type=float, default=0.5, help="Only used when mode=road_prob.")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--step", type=float, default=0.5, help="Step size in grid units.")
    p.add_argument("--weight_scale_m", type=float, default=50.0, help="(mode=dist) scale for dist-based weights in meters.")
    p.add_argument("--chunk_n", type=int, default=256, help="Process this many windows per chunk for memory control.")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config(
        iters=int(args.iters),
        step=float(args.step),
        weight_scale_m=float(args.weight_scale_m),
        chunk_n=int(args.chunk_n),
        seed=int(args.seed),
    )
    report = run_refine(
        in_samples_npz=Path(args.in_samples_npz),
        semantic_dir=Path(args.semantic_dir),
        out_samples_npz=Path(args.out_samples_npz),
        out_json=Path(args.out_json),
        cfg=cfg,
        mode=str(args.mode),
        road_prob_thr=float(args.road_prob_thr),
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

