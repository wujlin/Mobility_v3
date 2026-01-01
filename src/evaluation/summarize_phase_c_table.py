from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _get(d: Dict[str, Any], keys: Tuple[str, ...], default: Optional[Any] = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt(x: Any, *, nd: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return "NA"
    if v != v:  # NaN
        return "NA"
    return f"{v:.{nd}f}"


def _fmt_pct(x: Any, *, nd: int = 2) -> str:
    try:
        v = float(x)
    except Exception:
        return "NA"
    if v != v:
        return "NA"
    return f"{v * 100:.{nd}f}%"


def _load_g1(g1_json: Path) -> Dict[str, float]:
    r = _load_json(g1_json)
    res = r.get("results", r)
    return {
        "COLL": float(res["collision_rate_any"]),
        "CUT": float(res["cut_only_rate"]),
        "WP_ANY": float(res["waypoint_any_offroad_rate"]),
        "SEG0": float(res["collision_seg0_rate"]),
        "SEG1": float(res["collision_seg1_rate"]),
        "SEG2": float(res["collision_seg2_rate"]),
    }


def _load_align(align_json: Path) -> Dict[str, Any]:
    r = _load_json(align_json)
    valid = r.get("valid_rates", {})
    metrics = r.get("metrics", {})
    out: Dict[str, Any] = {
        "pred_valid_rate": float(valid.get("pred_valid_rate", 0.0)),
        "gt_proj_valid_rate": float(valid.get("gt_proj_valid_rate", 0.0)),
        "jsd_pref": {},
        "jsd_pref_rand": {},
        "jsd_rdist": {},
    }
    for stage in ("wp1", "wp2", "end"):
        m = metrics.get(stage, {})
        out["jsd_pref"][stage] = float(_get(m, ("heatmap_jsd_pref", "pred_vs_gt_proj"), 0.0))
        out["jsd_pref_rand"][stage] = float(_get(m, ("heatmap_jsd_pref", "rand_vs_gt_proj"), 0.0))
        out["jsd_rdist"][stage] = float(_get(m, ("JSD_Rdist_proj",), 0.0))
    return out


def _load_detour_validity(detour_json: Path) -> Dict[str, Any]:
    r = _load_json(detour_json)
    metrics = r.get("metrics", {})
    out: Dict[str, Any] = {}
    for name in metrics.keys():
        m = metrics[name]
        out[name] = {"overall": {}, "detour": {}}
        for scope in ("overall", "detour"):
            s = m.get(scope, {})
            for k in ("JSD_turn@4.0", "JSD_turn@8.0", "JSD_max_dev_ratio", "JSD_len_ratio"):
                v = _get(s, (k, "mean"), None)
                if v is not None:
                    out[name][scope][k] = float(v)
    return out


def _load_phy(phy_json: Path) -> Dict[str, Any]:
    r = _load_json(phy_json)
    metrics = r.get("metrics", {})
    out: Dict[str, Any] = {}
    for name, m in metrics.items():
        out[name] = {
            "JSD_TurnAngle": float(m.get("JSD_TurnAngle", 0.0)),
            "JSD_Speed": float(m.get("JSD_Speed", 0.0)),
            "JSD_Accel": float(m.get("JSD_Accel", 0.0)),
            "Vio_Speed_Rate": float(m.get("Vio_Speed_Rate", 0.0)),
            "Vio_Accel_Rate": float(m.get("Vio_Accel_Rate", 0.0)),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Phase C summary table (G1 + mask alignment + detour validity + physical stats).")
    p.add_argument("--tag", type=str, required=True, help="Row tag (e.g., ARGMAX / SAMPLE1).")
    p.add_argument("--g1_json", type=str, required=True)
    p.add_argument("--align_json", type=str, required=True)
    p.add_argument("--detour_validity_json", type=str, required=True)
    p.add_argument("--phy_json", type=str, required=True, help="plot_physical_stats *_validity.json")
    args = p.parse_args()

    tag = str(args.tag)
    g1 = _load_g1(Path(args.g1_json))
    al = _load_align(Path(args.align_json))
    dv = _load_detour_validity(Path(args.detour_validity_json))
    phy = _load_phy(Path(args.phy_json))

    print("-" * 60)
    print(
        f"[{tag}] G1: "
        f"COLL={_fmt(g1['COLL'])} CUT={_fmt(g1['CUT'])} WP_ANY={_fmt(g1['WP_ANY'])} "
        f"SEG0/1/2={_fmt(g1['SEG0'])}/{_fmt(g1['SEG1'])}/{_fmt(g1['SEG2'])}"
    )
    print(
        f"[{tag}] ALIGN: "
        f"valid(pred/gt_proj)={_fmt(al['pred_valid_rate'], nd=3)}/{_fmt(al['gt_proj_valid_rate'], nd=3)} "
        f"JSD_pref wp1/wp2/end={_fmt(al['jsd_pref']['wp1'])}/{_fmt(al['jsd_pref']['wp2'])}/{_fmt(al['jsd_pref']['end'])} "
        f"(rand={_fmt(al['jsd_pref_rand']['wp1'])}) "
        f"JSD_rdist={_fmt(al['jsd_rdist']['wp1'])}/{_fmt(al['jsd_rdist']['wp2'])}/{_fmt(al['jsd_rdist']['end'])}"
    )

    for method in ("MacroSkel", "Macro+DetRes"):
        if method in dv:
            o = dv[method].get("overall", {})
            d = dv[method].get("detour", {})
            print(
                f"[{tag}] {method}: "
                f"overall turn@4/8={_fmt(o.get('JSD_turn@4.0'))}/{_fmt(o.get('JSD_turn@8.0'))} "
                f"dev/len={_fmt(o.get('JSD_max_dev_ratio'))}/{_fmt(o.get('JSD_len_ratio'))} | "
                f"detour turn@4/8={_fmt(d.get('JSD_turn@4.0'))}/{_fmt(d.get('JSD_turn@8.0'))} "
                f"dev/len={_fmt(d.get('JSD_max_dev_ratio'))}/{_fmt(d.get('JSD_len_ratio'))}"
            )

    for method in ("MacroSkel", "Macro+DetRes"):
        if method in phy:
            m = phy[method]
            print(
                f"[{tag}] PHY {method}: "
                f"JSD_turn={_fmt(m['JSD_TurnAngle'])} JSD_speed={_fmt(m['JSD_Speed'])} JSD_accel={_fmt(m['JSD_Accel'])} "
                f"DCV_speed={_fmt_pct(m['Vio_Speed_Rate'])} DCV_accel={_fmt_pct(m['Vio_Accel_Rate'])}"
            )


if __name__ == "__main__":
    main()

