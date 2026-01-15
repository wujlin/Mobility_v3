#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TZ_SHANGHAI = timezone(timedelta(hours=8))

_ROUTE_NPZ_RE = re.compile(r"^(?P<city>[a-z0-9_]+)_segments_route_F(?P<F>\\d+)_epoch_seed(?P<seed>\\d+)\\.npz$", re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    key: Tuple[str, int, int]
    path: Path
    score: int
    mtime_s: float


def _load_env_raw_root() -> str:
    return str(os.environ.get("RAW_ROOT") or "/home/jinlin/data/geoexplicit_data")


def _score_path(p: Path) -> int:
    s = str(p)
    score = 0
    if "E_S1_segments_fixedlen" in s:
        score += 3
    if "segments_fixedlen" in s or "fixedlen" in s:
        score += 2
    if "gt_segments" in s:
        score += 1
    return int(score)


def _parse_key(name: str) -> Optional[Tuple[str, int, int]]:
    m = _ROUTE_NPZ_RE.match(name)
    if not m:
        return None
    city = str(m.group("city")).lower()
    try:
        F = int(m.group("F"))
        seed = int(m.group("seed"))
    except Exception:
        return None
    return (city, int(F), int(seed))


def _safe_rel(p: Path, *, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def _iter_candidates(exp_root: Path) -> List[Candidate]:
    out: List[Candidate] = []
    for p in exp_root.rglob("*_segments_route_F*_epoch_seed*.npz"):
        key = _parse_key(p.name)
        if key is None:
            continue
        try:
            st = p.stat()
        except Exception:
            continue
        out.append(Candidate(key=key, path=p, score=_score_path(p), mtime_s=float(st.st_mtime)))
    return out


def _pick_best(cands: List[Candidate]) -> Candidate:
    # Higher score first; then newer mtime.
    cands_sorted = sorted(cands, key=lambda c: (int(c.score), float(c.mtime_s)), reverse=True)
    return cands_sorted[0]


def _ensure_symlink(alias_path: Path, *, target: Path, dry_run: bool) -> str:
    if alias_path.exists() or alias_path.is_symlink():
        if alias_path.is_symlink():
            try:
                cur = alias_path.resolve()
            except Exception:
                cur = None
            if cur is not None and cur == target.resolve():
                return "unchanged"
            if dry_run:
                return "update(dry)"
            alias_path.unlink()
        else:
            # Do not overwrite real files.
            return "skip(non-symlink-exists)"
    if dry_run:
        return "create(dry)"
    alias_path.symlink_to(target)
    return "created"


def main() -> None:
    p = argparse.ArgumentParser(description="为工作站 $RAW_ROOT/experiments/icml2026_routegen 建立稳定别名（软链接），减少路径名工程量。")
    p.add_argument("--raw_root", type=str, default=_load_env_raw_root())
    p.add_argument("--exp_root", type=str, default=None, help="默认: <raw_root>/experiments/icml2026_routegen")
    p.add_argument("--alias_dir", type=str, default=None, help="默认: <exp_root>/gt_segments")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    raw_root = Path(args.raw_root)
    exp_root = Path(args.exp_root) if args.exp_root else (raw_root / "experiments" / "icml2026_routegen")
    alias_dir = Path(args.alias_dir) if args.alias_dir else (exp_root / "gt_segments")

    if not exp_root.exists():
        raise SystemExit(f"exp_root not found: {exp_root}")

    cands = _iter_candidates(exp_root)
    by_key: Dict[Tuple[str, int, int], List[Candidate]] = {}
    for c in cands:
        by_key.setdefault(c.key, []).append(c)

    keys = sorted(by_key.keys())
    print(f"[scan] exp_root={exp_root} candidates={len(cands)} keys={len(keys)}")
    if not keys:
        raise SystemExit("No *_segments_route_F*_epoch_seed*.npz found. Check your experiments directory.")

    if not args.dry_run:
        alias_dir.mkdir(parents=True, exist_ok=True)

    chosen: Dict[str, dict] = {}
    n_created = 0
    n_unchanged = 0
    n_skipped = 0
    for key in keys:
        group = by_key[key]
        best = _pick_best(group)
        city, F, seed = key
        alias_name = f"{city}_segments_route_F{F}_epoch_seed{seed}.npz"
        alias_path = alias_dir / alias_name
        action = _ensure_symlink(alias_path, target=best.path, dry_run=bool(args.dry_run))
        if action.startswith("skip"):
            n_skipped += 1
        elif action.startswith("unchanged"):
            n_unchanged += 1
        else:
            n_created += 1

        chosen[alias_name] = {
            "key": {"city": city, "F": int(F), "seed": int(seed)},
            "picked": _safe_rel(best.path, root=exp_root),
            "score": int(best.score),
            "mtime_s": float(best.mtime_s),
            "num_candidates": int(len(group)),
            "action": action,
        }
        print(f"[alias] {alias_name} -> {best.path} ({action})")

    manifest = {
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "exp_root": str(exp_root),
        "alias_dir": str(alias_dir),
        "dry_run": bool(args.dry_run),
        "summary": {"keys": int(len(keys)), "created": int(n_created), "unchanged": int(n_unchanged), "skipped": int(n_skipped)},
        "aliases": chosen,
    }
    if not args.dry_run:
        out_json = alias_dir / "aliases.json"
        out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[saved] {out_json}")
    else:
        print("[dry_run] no files written.")


if __name__ == "__main__":
    main()

