from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data.way_graph.region_sequence_dataset import load_region_ar_dataset
from src.models.way_casd.region_ar import RegionARCfg, RegionARModel

TZ_SHANGHAI = timezone(timedelta(hours=8))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def _decode_meta(meta_obj: object) -> Optional[dict]:
    if meta_obj is None:
        return None
    if isinstance(meta_obj, np.ndarray):
        if meta_obj.size != 1:
            return None
        meta_obj = meta_obj.item()
    return meta_obj if isinstance(meta_obj, dict) else None


def _load_region_meta(
    *, way_regions_npz: Path, way_features_npz: Path, coord_scale: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """
    Build:
      - region_city: (R,) long
      - region_static: (R,4) float = [centroid_y_norm, centroid_x_norm, log1p(n_ways), log1p(deg)]
      - region_adj: (R,R) bool (diag=True)
    """
    wr = np.load(str(way_regions_npz), allow_pickle=True)
    need = {"region_way_ptr", "region_way_idx", "region_adj_ptr", "region_adj_idx"}
    missing = sorted(list(need - set(wr.files)))
    if missing:
        raise SystemExit(f"[FATAL] way_regions_npz missing keys: {missing}")

    region_way_ptr = np.asarray(wr["region_way_ptr"], dtype=np.int64).reshape(-1)
    region_way_idx = np.asarray(wr["region_way_idx"], dtype=np.int64).reshape(-1)
    region_adj_ptr = np.asarray(wr["region_adj_ptr"], dtype=np.int64).reshape(-1)
    region_adj_idx = np.asarray(wr["region_adj_idx"], dtype=np.int64).reshape(-1)
    n_regions = int(region_way_ptr.size) - 1
    meta = _decode_meta(wr["meta"]) if "meta" in wr.files else None
    per_city = (meta or {}).get("per_city", {})
    if not isinstance(per_city, dict) or not per_city:
        log.warning("way_regions_npz meta/per_city missing; fallback to single-city mapping (all regions -> city 0).")
        region_city = np.zeros((n_regions,), dtype=np.int64)
        n_cities = 1
    else:
        region_city = np.full((n_regions,), -1, dtype=np.int64)
        n_cities = 0
        for k, v in per_city.items():
            try:
                city = int(k)
                off = int(v.get("region_id_offset", 0))
                nr = int(v.get("n_regions", 0))
            except Exception:
                continue
            if nr <= 0:
                continue
            region_city[off : off + nr] = int(city)
            n_cities = max(n_cities, city + 1)
        if int(np.sum(region_city < 0)) > 0:
            raise SystemExit(f"[FATAL] region_city has unassigned entries: {int(np.sum(region_city < 0))}/{n_regions}")

    wf = np.load(str(way_features_npz), allow_pickle=True)
    need = {"way_center_y", "way_center_x"}
    missing = sorted(list(need - set(wf.files)))
    if missing:
        raise SystemExit(f"[FATAL] way_features_npz missing keys: {missing}")
    way_center_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
    way_center_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)

    coord_scale = float(coord_scale)
    cent_y = np.zeros((n_regions,), dtype=np.float64)
    cent_x = np.zeros((n_regions,), dtype=np.float64)
    n_ways = np.zeros((n_regions,), dtype=np.float64)
    for r in range(n_regions):
        s = int(region_way_ptr[r])
        e = int(region_way_ptr[r + 1])
        ways = region_way_idx[s:e]
        n = int(ways.size)
        n_ways[r] = float(n)
        if n <= 0:
            continue
        cent_y[r] = float(np.mean(way_center_y[ways]))
        cent_x[r] = float(np.mean(way_center_x[ways]))

    deg = (region_adj_ptr[1:] - region_adj_ptr[:-1]).astype(np.int64, copy=False)
    deg_f = deg.astype(np.float64, copy=False)
    static = np.stack(
        [cent_y / coord_scale, cent_x / coord_scale, np.log1p(n_ways), np.log1p(deg_f)],
        axis=1,
    ).astype(np.float32, copy=False)

    # dense adjacency for masking / diagnostics
    adj = np.zeros((n_regions, n_regions), dtype=bool)
    np.fill_diagonal(adj, True)
    for r in range(n_regions):
        s = int(region_adj_ptr[r])
        e = int(region_adj_ptr[r + 1])
        for nb in region_adj_idx[s:e].tolist():
            b = int(nb)
            if 0 <= b < n_regions:
                adj[r, b] = True

    report = {
        "n_regions": int(n_regions),
        "n_cities": int(n_cities),
        "region_city_counts": {str(i): int(np.sum(region_city == i)) for i in range(int(n_cities))},
        "static_dim": int(static.shape[1]),
        "deg": {"p50": float(np.percentile(deg_f, 50)), "p90": float(np.percentile(deg_f, 90)), "max": int(deg.max()) if deg.size else 0},
        "n_ways": {"p50": float(np.percentile(n_ways, 50)), "p90": float(np.percentile(n_ways, 90)), "max": float(np.max(n_ways)) if n_ways.size else 0.0},
        "coord_scale": float(coord_scale),
    }
    return (
        torch.as_tensor(region_city, dtype=torch.long),
        torch.as_tensor(static, dtype=torch.float32),
        torch.as_tensor(adj, dtype=torch.bool),
        report,
    )


def _mask_next_logits(*, logits: torch.Tensor, prev_region: int, region_adj: torch.Tensor) -> torch.Tensor:
    # logits: (R,)
    R = int(logits.numel())
    prev = int(prev_region)
    allowed = region_adj[prev].clone()
    if 0 <= prev < R:
        allowed[prev] = False  # disallow staying (compressed seq)
    if bool(allowed.sum().item() == 0):
        # numerical safety fallback
        if 0 <= prev < R:
            allowed[prev] = True
        else:
            allowed[0] = True
    return logits.masked_fill(~allowed, -1e9)


@dataclass(frozen=True)
class DecodeResult:
    pred_seq: List[int]
    n_invalid_steps: int
    has_backtrack: bool


@torch.no_grad()
def _decode_greedy_model(
    *,
    model: RegionARModel,
    route_cond: Dict[str, torch.Tensor],
    o_region: int,
    d_region: int,
    max_len: int,
    region_adj: torch.Tensor,
    use_candidate_mask: bool,
) -> DecodeResult:
    seq: List[int] = [int(o_region)]
    n_invalid = 0
    visited = {int(o_region)}
    has_backtrack = False

    for _ in range(int(max_len) - 1):
        cur = int(seq[-1])
        if cur == int(d_region):
            break

        x = torch.as_tensor(np.asarray(seq, dtype=np.int64)[None, :], dtype=torch.long, device=route_cond["route_city"].device)
        logits = model(
            region_seq_in=x,
            o_region=torch.as_tensor([int(o_region)], dtype=torch.long, device=x.device),
            d_region=torch.as_tensor([int(d_region)], dtype=torch.long, device=x.device),
            route_cond=route_cond,
        )
        next_logits = logits[0, -1]  # (R,)
        if bool(use_candidate_mask):
            next_logits = _mask_next_logits(logits=next_logits, prev_region=cur, region_adj=region_adj)
        else:
            # diagnostics: count invalid transitions
            pred_raw = int(torch.argmax(next_logits).item())
            if not bool(region_adj[cur, pred_raw].item()):
                n_invalid += 1

        nxt = int(torch.argmax(next_logits).item())
        seq.append(nxt)
        if nxt in visited:
            has_backtrack = True
        visited.add(nxt)

    return DecodeResult(pred_seq=seq, n_invalid_steps=int(n_invalid), has_backtrack=bool(has_backtrack))


@torch.no_grad()
def _decode_beam_model(
    *,
    model: RegionARModel,
    route_cond: Dict[str, torch.Tensor],
    o_region: int,
    d_region: int,
    max_len: int,
    region_adj: torch.Tensor,
    use_candidate_mask: bool,
    beam_size: int,
) -> DecodeResult:
    beam_size = int(max(1, beam_size))
    beams: List[Tuple[float, List[int]]] = [(0.0, [int(o_region)])]  # (score, seq)

    for _ in range(int(max_len) - 1):
        new_beams: List[Tuple[float, List[int]]] = []
        all_done = True
        for score, seq in beams:
            cur = int(seq[-1])
            if cur == int(d_region):
                new_beams.append((float(score), list(seq)))
                continue
            all_done = False

            x = torch.as_tensor(np.asarray(seq, dtype=np.int64)[None, :], dtype=torch.long, device=route_cond["route_city"].device)
            logits = model(
                region_seq_in=x,
                o_region=torch.as_tensor([int(o_region)], dtype=torch.long, device=x.device),
                d_region=torch.as_tensor([int(d_region)], dtype=torch.long, device=x.device),
                route_cond=route_cond,
            )
            next_logits = logits[0, -1]  # (R,)
            if bool(use_candidate_mask):
                next_logits = _mask_next_logits(logits=next_logits, prev_region=cur, region_adj=region_adj)

            lp = F.log_softmax(next_logits, dim=-1)
            topv, topi = torch.topk(lp, k=min(int(beam_size), int(lp.numel())))
            for v, i in zip(topv.tolist(), topi.tolist()):
                new_beams.append((float(score + float(v)), list(seq) + [int(i)]))

        # keep best K
        new_beams.sort(key=lambda t: t[0], reverse=True)
        beams = new_beams[:beam_size]
        if bool(all_done):
            break

    best = max(beams, key=lambda t: t[0])[1] if beams else [int(o_region)]
    visited: set[int] = set()
    has_backtrack = False
    for r in best:
        if int(r) in visited:
            has_backtrack = True
            break
        visited.add(int(r))
    n_invalid = 0
    for a, b in zip(best[:-1], best[1:]):
        if not bool(region_adj[int(a), int(b)].item()):
            n_invalid += 1
    return DecodeResult(pred_seq=best, n_invalid_steps=int(n_invalid), has_backtrack=bool(has_backtrack))


def _build_markov_next(region_seq_npz: Path, *, n_regions: int) -> np.ndarray:
    data = np.load(str(region_seq_npz), allow_pickle=True)
    need = {"region_seq_ptr", "region_seq_idx", "region_seq_len"}
    missing = sorted(list(need - set(data.files)))
    if missing:
        raise ValueError(f"{region_seq_npz} missing keys: {missing}")
    ptr = np.asarray(data["region_seq_ptr"], dtype=np.int64).reshape(-1)
    idx = np.asarray(data["region_seq_idx"], dtype=np.int64).reshape(-1)
    ln = np.asarray(data["region_seq_len"], dtype=np.int64).reshape(-1)

    counts = np.zeros((int(n_regions), int(n_regions)), dtype=np.int64)
    K = int(ln.size)
    for k in range(K):
        L = int(ln[k])
        if L <= 1:
            continue
        s = int(ptr[k])
        e = int(ptr[k + 1])
        seq = idx[s:e].astype(np.int64, copy=False)
        if seq.size <= 1:
            continue
        a = seq[:-1]
        b = seq[1:]
        for u, v in zip(a.tolist(), b.tolist()):
            uu = int(u)
            vv = int(v)
            if 0 <= uu < n_regions and 0 <= vv < n_regions:
                counts[uu, vv] += 1
    # argmax per row; ties -> smallest id
    nxt = np.argmax(counts, axis=1).astype(np.int64, copy=False)
    return nxt


def _decode_greedy_baseline(
    *,
    baseline: str,
    o_region: int,
    d_region: int,
    max_len: int,
    region_adj: np.ndarray,  # (R,R) bool
    markov_next: Optional[np.ndarray],
    rng: np.random.Generator,
) -> DecodeResult:
    seq: List[int] = [int(o_region)]
    visited = {int(o_region)}
    has_backtrack = False
    n_invalid = 0

    R = int(region_adj.shape[0])
    for _ in range(int(max_len) - 1):
        cur = int(seq[-1])
        if cur == int(d_region):
            break
        allowed = np.asarray(region_adj[cur], dtype=bool).copy()
        if 0 <= cur < R:
            allowed[cur] = False
        cand = np.nonzero(allowed)[0].astype(np.int64, copy=False)
        if cand.size == 0:
            cand = np.asarray([cur], dtype=np.int64)

        nxt = int(cand[0])
        if baseline == "dest":
            if 0 <= int(d_region) < R and bool(region_adj[cur, int(d_region)]) and int(d_region) != cur:
                nxt = int(d_region)
            elif markov_next is not None and 0 <= cur < markov_next.size:
                nxt = int(markov_next[cur])
            else:
                nxt = int(cand[0])
        elif baseline == "markov":
            if markov_next is not None and 0 <= cur < markov_next.size:
                nxt = int(markov_next[cur])
            else:
                nxt = int(cand[0])
        elif baseline == "random":
            nxt = int(rng.choice(cand).item())
        else:
            raise ValueError(f"unknown baseline: {baseline}")

        if not bool(region_adj[cur, nxt]):
            n_invalid += 1
        seq.append(nxt)
        if nxt in visited:
            has_backtrack = True
        visited.add(nxt)

    return DecodeResult(pred_seq=seq, n_invalid_steps=int(n_invalid), has_backtrack=bool(has_backtrack))


def _summarize(records: List[dict]) -> dict:
    if not records:
        return {}
    n = len(records)
    reach = sum(int(r["reach_dest"]) for r in records)
    exact = sum(int(r["exact_match"]) for r in records)
    back = sum(int(r["has_backtrack"]) for r in records)
    inv = sum(int(r.get("n_invalid_steps", 0)) for r in records)
    steps = sum(int(r.get("n_steps", 0)) for r in records)
    return {
        "n_routes": int(n),
        "reach_dest_rate": float(reach / n),
        "exact_match_rate": float(exact / n),
        "has_backtrack_rate": float(back / n),
        "invalid_steps_rate": float(inv / max(1, steps)),
        "pred_len": {
            "p50": float(np.percentile([r["pred_len"] for r in records], 50)),
            "p95": float(np.percentile([r["pred_len"] for r in records], 95)),
            "max": int(max(r["pred_len"] for r in records)),
        },
        "gt_len": {
            "p50": float(np.percentile([r["gt_len"] for r in records], 50)),
            "p95": float(np.percentile([r["gt_len"] for r in records], 95)),
            "max": int(max(r["gt_len"] for r in records)),
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rollout evaluation for Region AR (greedy/beam).")
    p.add_argument("--way_routes_npz", type=Path, required=True)
    p.add_argument("--way_regions_npz", type=Path, required=True)
    p.add_argument("--region_seq_npz", type=Path, required=True)
    p.add_argument("--way_features_npz", type=Path, required=True)
    p.add_argument("--model_ckpt", type=Path, default=None, help="If omitted, run a baseline decoder.")
    p.add_argument("--baseline", type=str, default="none", choices=["none", "dest", "markov", "random"])
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tz_offset_hours", type=float, default=-5.0)
    p.add_argument("--n_routes", type=int, default=200)
    p.add_argument("--max_len", type=int, default=16)
    p.add_argument("--decode", type=str, default="greedy", choices=["greedy", "beam"])
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--coord_scale", type=float, default=1024.0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    rng = np.random.default_rng(int(args.seed))

    region_city, region_static, region_adj, region_meta = _load_region_meta(
        way_regions_npz=Path(args.way_regions_npz),
        way_features_npz=Path(args.way_features_npz),
        coord_scale=float(args.coord_scale),
    )
    R = int(region_city.numel())
    region_adj_np = region_adj.cpu().numpy().astype(bool, copy=False)

    ds = load_region_ar_dataset(
        way_routes_npz=Path(args.way_routes_npz),
        region_seq_npz=Path(args.region_seq_npz),
        way_regions_npz=Path(args.way_regions_npz),
        tz_offset_hours=float(args.tz_offset_hours),
        max_routes=None,
    )
    n_total = len(ds)
    n_eval = int(min(int(args.n_routes), int(n_total)))
    idx = rng.choice(np.arange(n_total, dtype=np.int64), size=n_eval, replace=False) if n_eval < n_total else np.arange(n_total, dtype=np.int64)

    device = torch.device(str(args.device) if (str(args.device) != "cuda" or torch.cuda.is_available()) else "cpu")
    use_model = (args.model_ckpt is not None) and (str(args.baseline) == "none")
    model: Optional[RegionARModel] = None
    use_candidate_mask = False

    if bool(use_model):
        ckpt = torch.load(str(args.model_ckpt), map_location=device)
        cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
        use_candidate_mask = bool(cfg.get("use_candidate_mask", False))
        model = RegionARModel(
            cfg=RegionARCfg(
                d_model=int(cfg.get("d_model", 256)),
                n_heads=int(cfg.get("n_heads", 8)),
                n_layers=int(cfg.get("n_layers", 4)),
                dropout=float(cfg.get("dropout", 0.1)),
                max_len=int(cfg.get("max_len", int(args.max_len))),
                n_regions=int(R),
                n_route_cities=int(cfg.get("n_route_cities", 2)),
                coord_scale=float(cfg.get("coord_scale", float(args.coord_scale))),
                use_candidate_mask=bool(use_candidate_mask),
            ),
            region_city=region_city.to(device=device),
            region_static=region_static.to(device=device),
            region_adj=region_adj.to(device=device),
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        log.info(f"Loaded model_ckpt={args.model_ckpt} use_candidate_mask={use_candidate_mask}")

    markov_next = None
    if str(args.baseline) in {"dest", "markov"}:
        markov_next = _build_markov_next(Path(args.region_seq_npz), n_regions=int(R))

    records: List[dict] = []
    for j, row in enumerate(idx.tolist()):
        item = ds[int(row)]
        rid = int(item["route_id"])
        city = int(item["route_city"])
        gt_seq = [int(x) for x in np.asarray(item["region_seq"], dtype=np.int64).reshape(-1).tolist()]
        o = int(item["o_region"])
        d = int(item["d_region"])

        if bool(use_model) and model is not None:
            route_cond = {
                "start_pos": torch.as_tensor(item["start_pos"][None, :], dtype=torch.float32, device=device),
                "dest_pos": torch.as_tensor(item["dest_pos"][None, :], dtype=torch.float32, device=device),
                # use hour/dow computed from unix in dataset
                "hour": torch.as_tensor([int(item["hour"])], dtype=torch.long, device=device),
                "dow": torch.as_tensor([int(item["dow"])], dtype=torch.long, device=device),
                "route_city": torch.as_tensor([int(city)], dtype=torch.long, device=device),
            }
            if str(args.decode) == "beam":
                dec = _decode_beam_model(
                    model=model,
                    route_cond=route_cond,
                    o_region=o,
                    d_region=d,
                    max_len=int(args.max_len),
                    region_adj=region_adj.to(device=device),
                    use_candidate_mask=bool(use_candidate_mask),
                    beam_size=int(args.beam_size),
                )
            else:
                dec = _decode_greedy_model(
                    model=model,
                    route_cond=route_cond,
                    o_region=o,
                    d_region=d,
                    max_len=int(args.max_len),
                    region_adj=region_adj.to(device=device),
                    use_candidate_mask=bool(use_candidate_mask),
                )
        else:
            dec = _decode_greedy_baseline(
                baseline=str(args.baseline),
                o_region=o,
                d_region=d,
                max_len=int(args.max_len),
                region_adj=region_adj_np,
                markov_next=markov_next,
                rng=rng,
            )

        pred_seq = dec.pred_seq
        reach_dest = bool(pred_seq and int(pred_seq[-1]) == int(d))
        exact_match = bool(pred_seq == gt_seq)
        rec = {
            "route_id": int(rid),
            "city": int(city),
            "o_region": int(o),
            "d_region": int(d),
            "gt_seq": gt_seq,
            "pred_seq": pred_seq,
            "gt_len": int(len(gt_seq)),
            "pred_len": int(len(pred_seq)),
            "reach_dest": bool(reach_dest),
            "exact_match": bool(exact_match),
            "has_backtrack": bool(dec.has_backtrack),
            "n_invalid_steps": int(dec.n_invalid_steps),
            "n_steps": int(max(0, len(pred_seq) - 1)),
        }
        records.append(rec)
        if (j + 1) % 200 == 0 or (j + 1) == len(idx):
            log.info(f"progress {j+1}/{len(idx)}")

    by_city: Dict[str, List[dict]] = {}
    for r in records:
        by_city.setdefault(str(int(r["city"])), []).append(r)

    out = {
        "ok": True,
        "task": "region_ar_rollout_eval",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "cfg": {
            "seed": int(args.seed),
            "n_routes": int(n_eval),
            "max_len": int(args.max_len),
            "decode": str(args.decode),
            "beam_size": int(args.beam_size),
            "baseline": str(args.baseline),
            "use_model": bool(use_model),
        },
        "inputs": {
            "way_routes_npz": str(args.way_routes_npz),
            "way_regions_npz": str(args.way_regions_npz),
            "region_seq_npz": str(args.region_seq_npz),
            "way_features_npz": str(args.way_features_npz),
            "model_ckpt": (str(args.model_ckpt) if args.model_ckpt is not None else None),
        },
        "region_meta": region_meta,
        "summary": _summarize(records),
        "summary_by_city": {k: _summarize(v) for k, v in by_city.items()},
        "records": records,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
