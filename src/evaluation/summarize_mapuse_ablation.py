from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_gate(path: Path) -> Dict[str, float]:
    r = json.loads(path.read_text())
    res = r.get("results", {})
    if not isinstance(res, dict):
        raise TypeError(f"Bad gate.json format (missing results dict): {path}")
    return {
        "collision_rate_any": float(res["collision_rate_any"]),
        "waypoint_any_offroad_rate": float(res["waypoint_any_offroad_rate"]),
        "waypoint_offroad_rate": float(res["waypoint_offroad_rate"]),
        "end_offroad_rate": float(res["end_offroad_rate"]),
        "cut_only_rate": float(res["cut_only_rate"]),
        "collision_seg0_rate": float(res["collision_seg0_rate"]),
        "collision_seg1_rate": float(res["collision_seg1_rate"]),
        "collision_seg2_rate": float(res["collision_seg2_rate"]),
    }


def _find_npz_key(data: np.lib.npyio.NpzFile, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in data.files:
            return k
    return None


def _l2_summary(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    d = a.astype(np.float32, copy=False) - b.astype(np.float32, copy=False)
    l2 = np.sqrt(np.sum(d * d, axis=-1))
    return float(l2.mean()), float(np.max(l2))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize map-usage ablation runs (gate + sensitivity).")
    p.add_argument("--prefix", type=str, required=True, help="Experiment prefix, e.g. phys_mapuse_test")
    p.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["none", "shuffle", "zeros", "dir_zero", "ch2_zero"],
        help="List of modes appended to prefix: {prefix}_{mode}",
    )
    p.add_argument("--base_dir", type=str, default="data/experiments", help="Base experiments directory")
    p.add_argument("--baseline", type=str, default="none", help="Mode name used as baseline for sensitivity checks")
    p.add_argument("--no_sensitivity", action="store_true", help="Skip npz sensitivity checks")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    base_dir = Path(args.base_dir)
    prefix = str(args.prefix)
    modes = [str(m) for m in args.modes]
    baseline = str(args.baseline)

    # --- Gate table ---
    print("MODE       | COLL   | WP_ANY | WP_AVG | END    | CUT    | SEG0   SEG1   SEG2")
    print("-" * 90)
    gates: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        exp_dir = base_dir / f"{prefix}_{mode}"
        gate_path = exp_dir / "macro_waypoint_gate.json"
        if not gate_path.exists():
            print(f"{mode:10s} | MISSING gate: {gate_path}")
            continue
        g = _load_gate(gate_path)
        gates[mode] = g
        print(
            f"{mode:10s} | {g['collision_rate_any']:.4f} | {g['waypoint_any_offroad_rate']:.4f} | "
            f"{g['waypoint_offroad_rate']:.4f} | {g['end_offroad_rate']:.4f} | {g['cut_only_rate']:.4f} | "
            f"{g['collision_seg0_rate']:.4f} {g['collision_seg1_rate']:.4f} {g['collision_seg2_rate']:.4f}"
        )

    if bool(args.no_sensitivity):
        return

    # --- Sensitivity checks ---
    base_npz = base_dir / f"{prefix}_{baseline}" / "samples.npz"
    if not base_npz.exists():
        print(f"\n[WARN] baseline samples not found: {base_npz}")
        return

    with np.load(base_npz, allow_pickle=True) as data0:
        z_key = _find_npz_key(data0, ["z_k_grid", "z_k"])
        p_key = _find_npz_key(data0, ["preds_k", "preds", "samples"])
        if z_key is None:
            print(f"\n[WARN] baseline samples.npz missing z_k_grid/z_k: {base_npz}")
            return
        z0 = np.asarray(data0[z_key], dtype=np.float32)
        p0 = np.asarray(data0[p_key], dtype=np.float32) if p_key is not None else None

    print("\nSENSITIVITY (vs baseline=%s)" % baseline)
    print("mode       | z_key     | meanL2(all/wp1/wp2/end) | maxL2(all/wp1/wp2/end)" + (" | meanL2(preds)" if p0 is not None else ""))
    print("-" * 120)

    for mode in modes:
        if mode == baseline:
            continue
        p_npz = base_dir / f"{prefix}_{mode}" / "samples.npz"
        if not p_npz.exists():
            print(f"{mode:10s} | MISSING samples: {p_npz}")
            continue
        with np.load(p_npz, allow_pickle=True) as data1:
            if z_key not in data1.files:
                alt = _find_npz_key(data1, ["z_k_grid", "z_k"])
                if alt is None:
                    print(f"{mode:10s} | MISSING z in samples: {p_npz}")
                    continue
                z_key_use = alt
            else:
                z_key_use = z_key
            z1 = np.asarray(data1[z_key_use], dtype=np.float32)

            # align by min common (avoid over-engineering; we expect exact match)
            n = min(int(z0.shape[0]), int(z1.shape[0]))
            k = min(int(z0.shape[1]), int(z1.shape[1]))
            z0c = z0[:n, :k]
            z1c = z1[:n, :k]

            # (N,K,3,2) -> L2 per point
            mean_all, max_all = _l2_summary(z0c.reshape(-1, 2), z1c.reshape(-1, 2))
            mean_wp1, max_wp1 = _l2_summary(z0c[:, :, 0, :].reshape(-1, 2), z1c[:, :, 0, :].reshape(-1, 2))
            mean_wp2, max_wp2 = _l2_summary(z0c[:, :, 1, :].reshape(-1, 2), z1c[:, :, 1, :].reshape(-1, 2))
            mean_end, max_end = _l2_summary(z0c[:, :, 2, :].reshape(-1, 2), z1c[:, :, 2, :].reshape(-1, 2))

            extra = ""
            if p0 is not None:
                p_key1 = _find_npz_key(data1, ["preds_k", "preds", "samples"])
                if p_key1 is not None:
                    p1 = np.asarray(data1[p_key1], dtype=np.float32)
                    n2 = min(int(p0.shape[0]), int(p1.shape[0]))
                    k2 = min(int(p0.shape[1]), int(p1.shape[1]))
                    p0c = p0[:n2, :k2]
                    p1c = p1[:n2, :k2]
                    mean_pred, _ = _l2_summary(p0c.reshape(-1, 2), p1c.reshape(-1, 2))
                    extra = f" | {mean_pred:.4f}"
                else:
                    extra = " | NA"

        print(
            f"{mode:10s} | {z_key_use:9s} | "
            f"{mean_all:.4f}/{mean_wp1:.4f}/{mean_wp2:.4f}/{mean_end:.4f} | "
            f"{max_all:.2f}/{max_wp1:.2f}/{max_wp2:.2f}/{max_end:.2f}"
            f"{extra}"
        )


if __name__ == "__main__":
    main()

