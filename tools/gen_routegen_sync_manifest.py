#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _rel(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _summarize_audit_report(data: dict) -> dict:
    runs = []
    for r in data.get("runs", []) or []:
        if not isinstance(r, dict):
            continue
        m = r.get("metrics_overall") or {}
        if not isinstance(m, dict):
            m = {}
        runs.append(
            {
                "intervention": r.get("intervention"),
                "mixture_jsd": m.get("mixture_jsd_mean"),
                "mixture_l1": m.get("mixture_l1_mean"),
                "coverage": m.get("model_cov_mean"),
                "diversity": m.get("model_div_mean"),
                "collapse_rate": m.get("collapse_rate_mean"),
            }
        )
    return {
        "checkpoint": (data.get("inputs") or {}).get("checkpoint"),
        "case_npz": (data.get("inputs") or {}).get("case_npz"),
        "semantic_dir": (data.get("inputs") or {}).get("semantic_dir"),
        "num_samples_per_condition": ((data.get("config") or {}).get("num_samples_per_condition")),
        "runs": runs,
    }


def _summarize_exec_prior_report(data: dict) -> dict:
    out: Dict[str, Any] = {"gate": data.get("gate")}
    for k in ("baseline_cascade", "exec_road", "exec_road_strong"):
        if k in data:
            out[k] = data.get(k)
    return out


def _summarize_fullscale_dir(dir_path: Path) -> dict:
    out: Dict[str, Any] = {}
    for name in ("metrics_cascade.json", "metrics_e2e.json"):
        p = dir_path / name
        d = _read_json(p)
        if isinstance(d, dict):
            out[name.replace(".json", "")] = d
    return out


def _guess_kind(dir_path: Path) -> str:
    if (dir_path / "train_summary.json").exists():
        return "train"
    if (dir_path / "report.json").exists():
        data = _read_json(dir_path / "report.json") or {}
        if isinstance(data, dict) and "runs" in data:
            return "audit"
        if isinstance(data, dict) and "baseline_cascade" in data:
            return "exec_prior"
        if isinstance(data, dict) and str(data.get("gate", "")).startswith("E13"):
            return "fullscale"
        return "report"
    return "other"


def _extract_gate_num(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.match(r"^E(\d+)", str(s))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_gate_num_from_path(path_str: Optional[str]) -> Optional[int]:
    if not path_str:
        return None
    try:
        parts = Path(str(path_str)).parts
    except Exception:
        parts = tuple(str(path_str).split("/"))
    for part in reversed(parts):
        g = _extract_gate_num(part)
        if g is not None:
            return g
    return None


def _build_entry(dir_path: Path, *, root: Path) -> dict:
    entry: Dict[str, Any] = {
        "name": dir_path.name,
        "path": _rel(dir_path, root=root),
        "is_symlink": dir_path.is_symlink(),
        "kind": _guess_kind(dir_path),
        "files": sorted(p.name for p in dir_path.iterdir() if p.is_file()),
        "subdirs": sorted(p.name for p in dir_path.iterdir() if p.is_dir()),
    }

    report_path = dir_path / "report.json"
    if report_path.exists():
        data = _read_json(report_path)
        if isinstance(data, dict):
            entry["gate"] = data.get("gate")
            if entry["kind"] == "audit":
                entry["summary"] = _summarize_audit_report(data)
            elif entry["kind"] == "exec_prior":
                entry["summary"] = _summarize_exec_prior_report(data)
            elif entry["kind"] == "fullscale":
                entry["summary"] = {"gate": data.get("gate"), **_summarize_fullscale_dir(dir_path)}
            else:
                entry["summary"] = {"gate": data.get("gate")}

    train_path = dir_path / "train_summary.json"
    if train_path.exists():
        data = _read_json(train_path)
        if isinstance(data, dict):
            cfg = data.get("config") or {}
            if not isinstance(cfg, dict):
                cfg = {}
            entry["summary"] = {
                "train_npz": (data.get("inputs") or {}).get("train_npz"),
                "waypoint_mode": cfg.get("waypoint_mode"),
                "semantic_mode": cfg.get("semantic_mode"),
                "grid_channels": cfg.get("grid_channels"),
                "temporal_effective": cfg.get("temporal_effective"),
                "checkpoint": (data.get("outputs") or {}).get("checkpoint"),
            }

    issues: List[str] = []
    if not any(dir_path.iterdir()):
        issues.append("empty_dir")
    if entry["is_symlink"]:
        try:
            target = dir_path.resolve().relative_to(root.resolve())
            entry["symlink_target"] = str(target)
        except Exception:
            entry["symlink_target"] = str(dir_path.resolve())

    # Record inferred gate ids (useful for debugging naming), but do not treat audit↔checkpoint
    # mismatches as errors since audits often reference earlier checkpoints by design.
    name_gate = _extract_gate_num(entry.get("name"))
    ckpt_path: Optional[str] = None
    summary = entry.get("summary")
    if isinstance(summary, dict):
        ckpt_path = summary.get("checkpoint")
    ckpt_gate = _extract_gate_num_from_path(ckpt_path)
    if name_gate is not None:
        entry["name_gate"] = name_gate
    if ckpt_gate is not None:
        entry["checkpoint_gate"] = ckpt_gate
    if entry.get("kind") == "train" and name_gate is not None and ckpt_gate is not None and name_gate != ckpt_gate:
        issues.append(f"gate_mismatch(name=E{name_gate}, checkpoint=E{ckpt_gate})")
    entry["issues"] = issues
    return entry


def _to_markdown(manifest: dict) -> str:
    lines: List[str] = []
    lines.append("# ICML 2026 RouteGen: 本地同步结果索引（_sync）")
    lines.append("")
    lines.append(f"- root: `{manifest['root']}`")
    lines.append(f"- num_entries: `{manifest['num_entries']}`")
    if manifest.get("issues"):
        lines.append(f"- issues: `{len(manifest['issues'])}`")
    lines.append("")
    lines.append("## Experiments")
    lines.append("")

    for e in manifest["entries"]:
        name = e["name"]
        path = e["path"]
        kind = e["kind"]
        gate = e.get("gate")
        suffix = f" ({gate})" if gate else ""
        lines.append(f"- `{name}`{suffix} — `{kind}` — `{path}`")

        if e.get("is_symlink"):
            lines.append(f"  - symlink -> `{e.get('symlink_target')}`")

        summary = e.get("summary")
        if isinstance(summary, dict):
            if kind == "audit":
                ckpt = summary.get("checkpoint")
                if ckpt:
                    lines.append(f"  - checkpoint: `{ckpt}`")
                runs = summary.get("runs") or []
                valid = [r for r in runs if isinstance(r, dict)]
                if valid:
                    best = min(
                        valid,
                        key=lambda r: float("inf") if r.get("mixture_jsd") is None else float(r.get("mixture_jsd")),
                    )
                    lines.append(
                        "  - best(mixture_jsd): "
                        f"{best.get('intervention')} jsd={best.get('mixture_jsd')} l1={best.get('mixture_l1')} cov={best.get('coverage')}"
                    )
            elif kind == "train":
                ckpt = summary.get("checkpoint")
                if ckpt:
                    lines.append(f"  - checkpoint: `{ckpt}`")

        if e.get("issues"):
            lines.append(f"  - issues: `{','.join(e['issues'])}`")

    if manifest.get("issues"):
        lines.append("")
        lines.append("## Issues")
        lines.append("")
        for it in manifest["issues"]:
            lines.append(f"- `{it['name']}`: `{it['issue']}`")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a manifest for _sync/wsa/icml2026_routegen results.")
    p.add_argument("--root", type=str, default="_sync/wsa/icml2026_routegen")
    p.add_argument("--out_json", type=str, default="docs/ICML_2026_ROUTEGEN_SYNC_MANIFEST.json")
    p.add_argument("--out_md", type=str, default="docs/ICML_2026_ROUTEGEN_SYNC_MANIFEST.md")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    entries: List[dict] = []
    issues: List[dict] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir() or p.is_symlink()], key=lambda x: x.name):
        e = _build_entry(d, root=root)
        entries.append(e)
        for issue in e.get("issues") or []:
            issues.append({"name": e["name"], "issue": issue})

    manifest = {"root": str(root), "num_entries": len(entries), "entries": entries, "issues": issues}

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_to_markdown(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
