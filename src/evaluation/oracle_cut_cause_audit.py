from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class GateStats:
    collision_rate_any: float
    cut_only_rate: float
    waypoint_any_offroad_rate: float
    waypoint_offroad_rate: float
    collision_seg0_rate: float
    collision_seg1_rate: float
    collision_seg2_rate: float


def _dilate_bool(mask: np.ndarray, iters: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    iters_i = int(max(0, iters))
    for _ in range(iters_i):
        src = m
        out = src.copy()
        out[:-1, :] |= src[1:, :]
        out[1:, :] |= src[:-1, :]
        out[:, :-1] |= src[:, 1:]
        out[:, 1:] |= src[:, :-1]
        out[:-1, :-1] |= src[1:, 1:]
        out[1:, 1:] |= src[:-1, :-1]
        out[:-1, 1:] |= src[1:, :-1]
        out[1:, :-1] |= src[:-1, 1:]
        m = out
    return m


def _load_nav_drivable(nav_file: Path, *, count_thr: float) -> np.ndarray:
    data = np.load(str(nav_file), allow_pickle=True)
    if "count" not in data.files:
        raise ValueError(f"nav_file must contain 'count', got {data.files}")
    count = np.asarray(data["count"], dtype=np.float32)
    if count.ndim != 2:
        raise ValueError(f"Expected count (H,W), got {count.shape}")
    return np.asarray(count >= float(count_thr), dtype=bool)


def _load_macro_points(samples_npz: Path, *, k_index: int, max_n: Optional[int]) -> Dict[str, np.ndarray]:
    data = np.load(str(samples_npz), allow_pickle=True)
    need = {"start_pos", "z_k_grid"}
    miss = [k for k in sorted(need) if k not in data.files]
    if miss:
        raise ValueError(f"samples_npz missing keys: {miss}. got={list(data.files)}")
    start_pos = np.asarray(data["start_pos"], dtype=np.float32)
    z = np.asarray(data["z_k_grid"], dtype=np.float32)
    if z.ndim != 4 or z.shape[-2:] != (3, 2):
        raise ValueError(f"Expected z_k_grid (N,K,3,2), got {z.shape}")
    if start_pos.ndim != 2 or start_pos.shape[1] != 2:
        raise ValueError(f"Expected start_pos (N,2), got {start_pos.shape}")
    if int(z.shape[0]) != int(start_pos.shape[0]):
        raise ValueError("N mismatch between start_pos and z_k_grid")

    N = int(start_pos.shape[0])
    if max_n is not None:
        N = min(N, int(max_n))
        start_pos = start_pos[:N]
        z = z[:N]
    k = int(k_index)
    if k < 0 or k >= int(z.shape[1]):
        raise ValueError(f"k_index out of range: {k} not in [0,{int(z.shape[1]) - 1}]")
    pts = z[:, k]  # (N,3,2)
    return {"start_pos": start_pos, "wp": pts}


def _segment_collision(
    a: np.ndarray,
    b: np.ndarray,
    *,
    drivable: np.ndarray,
    sample_step: float,
    max_samples: int,
) -> bool:
    a = np.asarray(a, dtype=np.float32).reshape(2)
    b = np.asarray(b, dtype=np.float32).reshape(2)
    H, W = int(drivable.shape[0]), int(drivable.shape[1])
    d = b - a
    seg_len = float(np.linalg.norm(d))
    n = int(np.clip(np.ceil(seg_len / max(float(sample_step), 1e-6)) + 1, 2, int(max_samples)))
    ts = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    pts = a[None, :] + ts[:, None] * d[None, :]
    yy = np.rint(pts[:, 0]).astype(np.int64)
    xx = np.rint(pts[:, 1]).astype(np.int64)
    inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    if not bool(np.all(inb)):
        return True
    safe = drivable[yy, xx]
    return bool(np.any(~safe))


def _point_offroad(p: np.ndarray, drivable: np.ndarray) -> bool:
    H, W = int(drivable.shape[0]), int(drivable.shape[1])
    yy = int(np.rint(float(p[0])))
    xx = int(np.rint(float(p[1])))
    if yy < 0 or yy >= H or xx < 0 or xx >= W:
        return True
    return not bool(drivable[yy, xx])


def _run_gate(
    *,
    start_pos: np.ndarray,
    wp: np.ndarray,  # (N,3,2)
    drivable: np.ndarray,
    sample_step: float,
    max_samples: int,
) -> Dict[str, object]:
    N = int(start_pos.shape[0])
    seg0 = np.zeros((N,), dtype=bool)
    seg1 = np.zeros((N,), dtype=bool)
    seg2 = np.zeros((N,), dtype=bool)
    wp_off = np.zeros((N, 3), dtype=bool)

    for i in range(N):
        s = start_pos[i]
        w1, w2, e = wp[i, 0], wp[i, 1], wp[i, 2]
        wp_off[i, 0] = _point_offroad(w1, drivable)
        wp_off[i, 1] = _point_offroad(w2, drivable)
        wp_off[i, 2] = _point_offroad(e, drivable)
        seg0[i] = _segment_collision(s, w1, drivable=drivable, sample_step=sample_step, max_samples=max_samples)
        seg1[i] = _segment_collision(w1, w2, drivable=drivable, sample_step=sample_step, max_samples=max_samples)
        seg2[i] = _segment_collision(w2, e, drivable=drivable, sample_step=sample_step, max_samples=max_samples)

    coll_any = seg0 | seg1 | seg2 | np.any(wp_off, axis=1)
    wp_any = np.any(wp_off, axis=1)
    cut_only = (~wp_any) & (seg0 | seg1 | seg2)

    stats = GateStats(
        collision_rate_any=float(np.mean(coll_any)),
        cut_only_rate=float(np.mean(cut_only)),
        waypoint_any_offroad_rate=float(np.mean(wp_any)),
        waypoint_offroad_rate=float(np.mean(wp_off)),
        collision_seg0_rate=float(np.mean(seg0)),
        collision_seg1_rate=float(np.mean(seg1)),
        collision_seg2_rate=float(np.mean(seg2)),
    )
    return {
        "N": int(N),
        "stats": stats.__dict__,
        "masks": {
            "collision_any": coll_any,
            "cut_only": cut_only,
            "wp_any_offroad": wp_any,
            "collision_seg0": seg0,
            "collision_seg1": seg1,
            "collision_seg2": seg2,
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit why ORACLE_PROJ still has CUT: test drivable-mask dilation sensitivity (mask strictness vs straight-line skeleton limit)."
    )
    p.add_argument("--samples_npz", type=str, required=True, help="samples.npz containing start_pos and z_k_grid.")
    p.add_argument("--nav_file", type=str, required=True, help="nav_field.npz containing count.")
    p.add_argument("--count_thr", type=float, default=1.0)
    p.add_argument("--k_index", type=int, default=0, help="Which K to audit in z_k_grid.")
    p.add_argument("--dilate_iters", type=int, nargs="+", default=[0, 1, 2], help="Dilation iterations to test.")
    p.add_argument("--sample_step", type=float, default=0.5)
    p.add_argument("--max_samples_per_segment", type=int, default=256)
    p.add_argument("--max_n", type=int, default=None)
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--quiet", action="store_true", help="Suppress console prints (write JSON only).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    samples_npz = Path(args.samples_npz)
    nav_file = Path(args.nav_file)

    base_drv = _load_nav_drivable(nav_file, count_thr=float(args.count_thr))
    blob = _load_macro_points(samples_npz, k_index=int(args.k_index), max_n=(int(args.max_n) if args.max_n is not None else None))
    start_pos = blob["start_pos"]
    wp = blob["wp"]

    out: Dict[str, object] = {
        "meta": {
            "samples_npz": str(samples_npz),
            "nav_file": str(nav_file),
            "count_thr": float(args.count_thr),
            "k_index": int(args.k_index),
            "sample_step": float(args.sample_step),
            "max_samples_per_segment": int(args.max_samples_per_segment),
            "dilate_iters": [int(x) for x in args.dilate_iters],
        },
        "N": int(start_pos.shape[0]),
        "results": {},
        "cut_only_resolution": {},
    }

    base = None
    for it in [int(x) for x in args.dilate_iters]:
        drv = _dilate_bool(base_drv, iters=int(it)) if int(it) > 0 else base_drv
        rep = _run_gate(
            start_pos=start_pos,
            wp=wp,
            drivable=drv,
            sample_step=float(args.sample_step),
            max_samples=int(args.max_samples_per_segment),
        )
        out["results"][f"dilate_{it}"] = rep["stats"]
        if base is None and int(it) == 0:
            base = rep

    if base is not None:
        cut_only = np.asarray(base["masks"]["cut_only"], dtype=bool)
        n_cut = int(np.sum(cut_only))
        out["cut_only_resolution"]["base_cut_only_n"] = n_cut
        for it in [int(x) for x in args.dilate_iters if int(x) > 0]:
            drv = _dilate_bool(base_drv, iters=int(it))
            rep_it = _run_gate(
                start_pos=start_pos,
                wp=wp,
                drivable=drv,
                sample_step=float(args.sample_step),
                max_samples=int(args.max_samples_per_segment),
            )
            coll_any_it = np.asarray(rep_it["masks"]["collision_any"], dtype=bool)
            if n_cut > 0:
                resolved = cut_only & (~coll_any_it)
                out["cut_only_resolution"][f"resolved_by_dilate_{it}"] = float(np.mean(resolved[cut_only]))
                out["cut_only_resolution"][f"remaining_by_dilate_{it}"] = float(1.0 - float(np.mean(resolved[cut_only])))
            else:
                out["cut_only_resolution"][f"resolved_by_dilate_{it}"] = float("nan")
                out["cut_only_resolution"][f"remaining_by_dilate_{it}"] = float("nan")

    if not bool(args.quiet):
        print("============================================================")
        print("ORACLE CUT CAUSE AUDIT (mask dilation sensitivity)")
        print("============================================================")
        print(f"N={out['N']}  count_thr={float(args.count_thr)}  sample_step={float(args.sample_step)}")
        for k, v in out["results"].items():
            print(f"- {k}: COLL={v['collision_rate_any']:.4f}  CUT={v['cut_only_rate']:.4f}")
        if out["cut_only_resolution"]:
            n_cut = out["cut_only_resolution"].get("base_cut_only_n", 0)
            print(f"Base CUT-only N={n_cut}")
            for it in [int(x) for x in args.dilate_iters if int(x) > 0]:
                r = out["cut_only_resolution"].get(f"resolved_by_dilate_{it}", float("nan"))
                print(f"  resolved_by_dilate_{it}: {r:.4f}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        if not bool(args.quiet):
            print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
