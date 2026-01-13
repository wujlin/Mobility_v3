from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


TZ_SHANGHAI = timezone(timedelta(hours=8))


def _load_meta(data: np.lib.npyio.NpzFile) -> Optional[dict]:
    if "meta" not in data.files:
        return None
    meta = data["meta"]
    if isinstance(meta, np.ndarray) and meta.shape == ():
        meta = meta.item()
    return meta if isinstance(meta, dict) else None


def _load_paths_graph_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    need = {
        "traj_idx",
        "start_t",
        "start_pos",
        "dest_pos",
        "start_node",
        "dest_node",
        "node_seq_pad",
        "node_seq_len",
    }
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"paths_graph.npz missing keys: {missing} in {path}")
    out = {
        "traj_idx": np.asarray(data["traj_idx"], dtype=np.int64).reshape(-1),
        "start_t": np.asarray(data["start_t"], dtype=np.int64).reshape(-1),
        "start_pos": np.asarray(data["start_pos"], dtype=np.float32).reshape(-1, 2),
        "dest_pos": np.asarray(data["dest_pos"], dtype=np.float32).reshape(-1, 2),
        "start_node": np.asarray(data["start_node"], dtype=np.int32).reshape(-1),
        "dest_node": np.asarray(data["dest_node"], dtype=np.int32).reshape(-1),
        "node_seq_pad": np.asarray(data["node_seq_pad"], dtype=np.int32),
        "node_seq_len": np.asarray(data["node_seq_len"], dtype=np.int32).reshape(-1),
        "kept_index": np.asarray(data["kept_index"], dtype=np.int64).reshape(-1) if "kept_index" in data.files else None,
        "meta": _load_meta(data),
    }
    n = int(out["traj_idx"].shape[0])
    if (
        int(out["start_t"].shape[0]) != n
        or int(out["start_pos"].shape[0]) != n
        or int(out["dest_pos"].shape[0]) != n
        or int(out["start_node"].shape[0]) != n
        or int(out["dest_node"].shape[0]) != n
        or int(out["node_seq_pad"].shape[0]) != n
        or int(out["node_seq_len"].shape[0]) != n
    ):
        raise ValueError(f"N mismatch inside {path}")
    return out


def _load_road_graph_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    need = {"node_y", "node_x", "edge_u", "edge_v", "edge_w_m", "edge_tier", "meta"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"road_graph.npz missing keys: {missing} in {path}")
    meta = _load_meta(data)
    if meta is None:
        raise ValueError(f"road_graph.npz meta is missing or invalid in {path}")
    out = {
        "node_y": np.asarray(data["node_y"], dtype=np.float32).reshape(-1),
        "node_x": np.asarray(data["node_x"], dtype=np.float32).reshape(-1),
        "edge_u": np.asarray(data["edge_u"], dtype=np.int32).reshape(-1),
        "edge_v": np.asarray(data["edge_v"], dtype=np.int32).reshape(-1),
        "edge_w_m": np.asarray(data["edge_w_m"], dtype=np.float32).reshape(-1),
        "edge_tier": np.asarray(data["edge_tier"], dtype=np.uint8).reshape(-1),
        "meta": meta,
    }
    if not (out["edge_u"].shape[0] == out["edge_v"].shape[0] == out["edge_w_m"].shape[0] == out["edge_tier"].shape[0]):
        raise ValueError(f"edge arrays length mismatch in {path}")
    return out


def _offset_i32(a: np.ndarray, *, offset: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.int32)
    off = int(offset)
    out = a.copy()
    m = out >= 0
    out[m] = (out[m].astype(np.int64) + np.int64(off)).astype(np.int32)
    return out


def _pad_sequences(a_pad: np.ndarray, b_pad: np.ndarray, *, pad_val: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a_pad, dtype=np.int32)
    b = np.asarray(b_pad, dtype=np.int32)
    La = int(a.shape[1])
    Lb = int(b.shape[1])
    Lmax = int(max(La, Lb))
    if La == Lmax and Lb == Lmax:
        return a, b
    out_a = np.full((a.shape[0], Lmax), int(pad_val), dtype=np.int32)
    out_b = np.full((b.shape[0], Lmax), int(pad_val), dtype=np.int32)
    out_a[:, :La] = a
    out_b[:, :Lb] = b
    return out_a, out_b


@dataclass(frozen=True)
class ComboCfg:
    b_traj_idx_offset: int
    city_a: Optional[str]
    city_b: Optional[str]


def build_combo(
    *,
    a_paths_graph_npz: Path,
    b_paths_graph_npz: Path,
    a_road_graph_npz: Path,
    b_road_graph_npz: Path,
    out_dir: Path,
    cfg: ComboCfg,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_graph = out_dir / "road_graph_combo.npz"
    out_paths = out_dir / "paths_graph_combo.npz"
    out_report = out_dir / "combo_report.json"

    a_g = _load_road_graph_npz(a_road_graph_npz)
    b_g = _load_road_graph_npz(b_road_graph_npz)
    n_a = int(a_g["node_y"].shape[0])
    n_b = int(b_g["node_y"].shape[0])
    node_offset_b = int(n_a)

    node_y = np.concatenate([a_g["node_y"], b_g["node_y"]], axis=0).astype(np.float32, copy=False)
    node_x = np.concatenate([a_g["node_x"], b_g["node_x"]], axis=0).astype(np.float32, copy=False)
    node_city = np.concatenate([np.zeros((n_a,), dtype=np.int8), np.ones((n_b,), dtype=np.int8)], axis=0)

    edge_u = np.concatenate([a_g["edge_u"], b_g["edge_u"].astype(np.int64) + np.int64(node_offset_b)], axis=0).astype(np.int32, copy=False)
    edge_v = np.concatenate([a_g["edge_v"], b_g["edge_v"].astype(np.int64) + np.int64(node_offset_b)], axis=0).astype(np.int32, copy=False)
    edge_w_m = np.concatenate([a_g["edge_w_m"], b_g["edge_w_m"]], axis=0).astype(np.float32, copy=False)
    edge_tier = np.concatenate([a_g["edge_tier"], b_g["edge_tier"]], axis=0).astype(np.uint8, copy=False)
    edge_city = np.concatenate([np.zeros((a_g["edge_u"].shape[0],), dtype=np.int8), np.ones((b_g["edge_u"].shape[0],), dtype=np.int8)], axis=0)

    meta = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "task": "build_combo_graph_paths_dataset",
        "inputs": {
            "a_paths_graph_npz": str(a_paths_graph_npz),
            "b_paths_graph_npz": str(b_paths_graph_npz),
            "a_road_graph_npz": str(a_road_graph_npz),
            "b_road_graph_npz": str(b_road_graph_npz),
            "city_a": (str(cfg.city_a) if cfg.city_a else None),
            "city_b": (str(cfg.city_b) if cfg.city_b else None),
        },
        "config": {"b_traj_idx_offset": int(cfg.b_traj_idx_offset), "node_offset_b": int(node_offset_b)},
        # Keep grid meta from city A for compatibility with existing graph loaders.
        "grid": (a_g["meta"].get("grid") if isinstance(a_g.get("meta"), dict) else None),
        "cities": {"a_meta": a_g["meta"], "b_meta": b_g["meta"]},
        "stats": {
            "nodes": {"N_a": int(n_a), "N_b": int(n_b), "N_total": int(n_a + n_b)},
            "edges": {"E_a": int(a_g["edge_u"].shape[0]), "E_b": int(b_g["edge_u"].shape[0]), "E_total": int(edge_u.shape[0])},
        },
    }

    np.savez_compressed(
        out_graph,
        node_y=node_y,
        node_x=node_x,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w_m=edge_w_m,
        edge_tier=edge_tier,
        node_city=node_city,
        edge_city=edge_city,
        meta=meta,
    )

    a_p = _load_paths_graph_npz(a_paths_graph_npz)
    b_p = _load_paths_graph_npz(b_paths_graph_npz)
    a_pad, b_pad = _pad_sequences(a_p["node_seq_pad"], b_p["node_seq_pad"], pad_val=-1)
    b_pad = _offset_i32(b_pad, offset=int(node_offset_b))
    b_start_node = _offset_i32(b_p["start_node"], offset=int(node_offset_b))
    b_dest_node = _offset_i32(b_p["dest_node"], offset=int(node_offset_b))

    # traj_idx offset for b (avoid collisions)
    b_traj_idx = b_p["traj_idx"].astype(np.int64, copy=False) + np.int64(int(cfg.b_traj_idx_offset))
    inter = np.intersect1d(np.unique(a_p["traj_idx"]), np.unique(b_traj_idx))
    if int(inter.size) > 0:
        raise ValueError(f"traj_idx collision after b_traj_idx_offset={int(cfg.b_traj_idx_offset)}; increase offset.")

    traj_idx = np.concatenate([a_p["traj_idx"], b_traj_idx], axis=0).astype(np.int64, copy=False)
    start_t = np.concatenate([a_p["start_t"], b_p["start_t"]], axis=0).astype(np.int64, copy=False)
    start_pos = np.concatenate([a_p["start_pos"], b_p["start_pos"]], axis=0).astype(np.float32, copy=False)
    dest_pos = np.concatenate([a_p["dest_pos"], b_p["dest_pos"]], axis=0).astype(np.float32, copy=False)
    start_node = np.concatenate([a_p["start_node"], b_start_node], axis=0).astype(np.int32, copy=False)
    dest_node = np.concatenate([a_p["dest_node"], b_dest_node], axis=0).astype(np.int32, copy=False)
    node_seq_pad = np.concatenate([a_pad, b_pad], axis=0).astype(np.int32, copy=False)
    node_seq_len = np.concatenate([a_p["node_seq_len"], b_p["node_seq_len"]], axis=0).astype(np.int32, copy=False)
    route_city = np.concatenate([np.zeros((a_p["traj_idx"].shape[0],), dtype=np.int8), np.ones((b_p["traj_idx"].shape[0],), dtype=np.int8)], axis=0)

    np.savez_compressed(
        out_paths,
        traj_idx=traj_idx,
        start_t=start_t,
        start_pos=start_pos,
        dest_pos=dest_pos,
        start_node=start_node,
        dest_node=dest_node,
        node_seq_pad=node_seq_pad,
        node_seq_len=node_seq_len,
        route_city=route_city,
        meta={
            "created_at": meta["created_at"],
            "task": "build_combo_graph_paths_dataset",
            "inputs": meta["inputs"],
            "config": meta["config"],
            "stats": {"N_a": int(a_p["traj_idx"].shape[0]), "N_b": int(b_p["traj_idx"].shape[0]), "N_total": int(traj_idx.shape[0]), "Lmax": int(node_seq_pad.shape[1])},
            "sources_meta": {"a_paths_meta": a_p.get("meta"), "b_paths_meta": b_p.get("meta")},
        },
    )

    report = {
        "ok": True,
        "created_at": meta["created_at"],
        "task": "build_combo_graph_paths_dataset",
        "inputs": meta["inputs"],
        "config": meta["config"],
        "outputs": {"out_dir": str(out_dir), "road_graph_npz": str(out_graph), "paths_graph_npz": str(out_paths), "report_json": str(out_report)},
        "stats": {"nodes": meta["stats"]["nodes"], "edges": meta["stats"]["edges"], "paths": {"N_total": int(traj_idx.shape[0]), "Lmax": int(node_seq_pad.shape[1])}},
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a combo (disjoint-union) graph+paths dataset by concatenating city A/B road graphs and offsetting B node indices.")
    p.add_argument("--a_city", type=str, default=None)
    p.add_argument("--b_city", type=str, default=None)
    p.add_argument("--a_paths_graph_npz", type=Path, required=True)
    p.add_argument("--b_paths_graph_npz", type=Path, required=True)
    p.add_argument("--a_road_graph_npz", type=Path, required=True)
    p.add_argument("--b_road_graph_npz", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--b_traj_idx_offset", type=int, default=1_000_000_000)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = build_combo(
        a_paths_graph_npz=Path(args.a_paths_graph_npz),
        b_paths_graph_npz=Path(args.b_paths_graph_npz),
        a_road_graph_npz=Path(args.a_road_graph_npz),
        b_road_graph_npz=Path(args.b_road_graph_npz),
        out_dir=Path(args.out_dir),
        cfg=ComboCfg(b_traj_idx_offset=int(args.b_traj_idx_offset), city_a=(str(args.a_city) if args.a_city else None), city_b=(str(args.b_city) if args.b_city else None)),
    )
    compact = {
        "ok": True,
        "out_dir": report["outputs"]["out_dir"],
        "road_graph_npz": report["outputs"]["road_graph_npz"],
        "paths_graph_npz": report["outputs"]["paths_graph_npz"],
        "report_json": report["outputs"]["report_json"],
        "N_paths": report["stats"]["paths"]["N_total"],
        "N_nodes": report["stats"]["nodes"]["N_total"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

