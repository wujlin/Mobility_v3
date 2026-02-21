from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _parse_city_kv(spec: str) -> Tuple[int, Path]:
    s = str(spec or "").strip()
    if "=" in s:
        k, v = s.split("=", 1)
    elif ":" in s:
        k, v = s.split(":", 1)
    else:
        raise ValueError(f"Bad spec (expect CITY=PATH): {spec!r}")
    return int(str(k).strip()), Path(str(v).strip()).expanduser()


def _grid_bbox_from_meta(meta: dict) -> Optional[Tuple[int, int, float, float, float, float]]:
    grid = meta.get("grid", {}) if isinstance(meta, dict) else {}
    if not isinstance(grid, dict):
        return None
    H = grid.get("H", None)
    W = grid.get("W", None)
    bbox = grid.get("bbox", None)
    if not isinstance(bbox, dict):
        return None
    try:
        H_i = int(H)
        W_i = int(W)
        min_lon = float(bbox["min_lon"])
        min_lat = float(bbox["min_lat"])
        max_lon = float(bbox["max_lon"])
        max_lat = float(bbox["max_lat"])
    except Exception:
        return None
    if H_i <= 0 or W_i <= 0:
        return None
    return (H_i, W_i, min_lon, min_lat, max_lon, max_lat)


def _first_loop_span(seq: Sequence[int]) -> Optional[Tuple[int, int, int]]:
    pos: Dict[int, int] = {}
    for i, x in enumerate(seq):
        xx = int(x)
        if xx in pos:
            s = int(pos[xx])
            e = int(i)
            return (s, e, max(0, e - s))
        pos[xx] = int(i)
    return None


def _to_latlon(ids: Sequence[int], way_y: np.ndarray, way_x: np.ndarray, meta: dict) -> List[List[float]]:
    bb = _grid_bbox_from_meta(meta)
    if bb is None:
        return []
    H, W, min_lon, min_lat, max_lon, max_lat = bb
    out: List[List[float]] = []
    n = int(way_y.size)
    for w in ids:
        wi = int(w)
        if wi < 0 or wi >= n:
            continue
        y = float(way_y[wi])
        x = float(way_x[wi])
        lon = min_lon + (x / float(W)) * (max_lon - min_lon)
        lat = max_lat - (y / float(H)) * (max_lat - min_lat)
        out.append([float(lat), float(lon)])
    return out


def _json_num(v: Any, nd: int = 3) -> str:
    try:
        x = float(v)
        if not math.isfinite(x):
            return "nan"
        return f"{x:.{int(nd)}f}"
    except Exception:
        return "nan"


def main() -> None:
    ap = argparse.ArgumentParser(description="Export 5-10 loop-heavy cases to a Leaflet interactive HTML.")
    ap.add_argument("--per_route_json", type=Path, required=True)
    ap.add_argument("--way_features_npz", type=Path, required=True)
    ap.add_argument("--city_grid_meta", action="append", default=[], help="CITY=PATH (repeatable)")
    ap.add_argument("--out_html", type=Path, required=True)
    ap.add_argument("--mode", choices=["greedy", "beam"], default="greedy")
    ap.add_argument("--city", type=int, default=0)
    ap.add_argument("--max_cases", type=int, default=10)
    ap.add_argument("--only_failed", action="store_true")
    ap.add_argument("--sort_by", choices=["loop_len", "len_ratio", "final_error_m"], default="loop_len")
    args = ap.parse_args()

    city_meta: Dict[int, dict] = {}
    for spec in list(args.city_grid_meta or []):
        c, p = _parse_city_kv(str(spec))
        city_meta[int(c)] = json.loads(p.read_text(encoding="utf-8"))
    if int(args.city) not in city_meta:
        raise SystemExit(f"[FATAL] missing city meta for city={int(args.city)}")

    wf = np.load(str(args.way_features_npz), allow_pickle=True)
    way_y = np.asarray(wf["way_center_y"], dtype=np.float64).reshape(-1)
    way_x = np.asarray(wf["way_center_x"], dtype=np.float64).reshape(-1)

    obj = json.loads(args.per_route_json.read_text(encoding="utf-8"))
    rows = obj.get("per_route", [])
    if not isinstance(rows, list):
        raise SystemExit("[FATAL] per_route_json missing list field: per_route")

    picked: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if int(r.get("city", -1)) != int(args.city):
            continue
        dec = r.get(str(args.mode), None)
        if not isinstance(dec, dict):
            continue
        if not bool(dec.get("has_loop", False)):
            continue
        if bool(args.only_failed) and bool(dec.get("success", False)):
            continue
        pred = dec.get("pred_way_ids", None)
        if not isinstance(pred, list) or len(pred) < 2:
            continue
        span = _first_loop_span([int(x) for x in pred])
        if span is None:
            continue
        s, e, ll = span
        rec = {
            "route_id": int(r.get("route_id", -1)),
            "city": int(r.get("city", -1)),
            "success": bool(dec.get("success", False)),
            "hit_wall": bool(dec.get("hit_wall", False)),
            "len_ratio": float(dec.get("len_ratio", float("nan"))),
            "final_error_m": float(dec.get("final_error_m", float("nan"))),
            "loop_len": int(ll),
            "loop_start": int(s),
            "loop_end": int(e),
            "pred_way_ids": [int(x) for x in pred],
            "gt_way_ids": [int(x) for x in r.get("gt_way_ids", [])] if isinstance(r.get("gt_way_ids", []), list) else [],
        }
        picked.append(rec)

    if not picked:
        raise SystemExit("[FATAL] no loop cases found under current filters.")

    key = str(args.sort_by)
    picked.sort(key=lambda z: float(z.get(key, 0.0)), reverse=True)
    picked = picked[: max(1, int(args.max_cases))]

    palette = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
        "#17becf",
        "#bcbd22",
    ]

    case_js: List[str] = []
    all_pts: List[List[float]] = []
    for i, rec in enumerate(picked):
        pred_ll = _to_latlon(rec["pred_way_ids"], way_y, way_x, city_meta[int(args.city)])
        if len(pred_ll) < 2:
            continue
        s = int(rec["loop_start"])
        e = int(rec["loop_end"])
        loop_ll = pred_ll[max(0, s) : min(len(pred_ll), e + 1)]
        gt_ll = _to_latlon(rec["gt_way_ids"], way_y, way_x, city_meta[int(args.city)]) if rec["gt_way_ids"] else []
        all_pts.extend(pred_ll)
        color = palette[i % len(palette)]
        popup = (
            f"rid={rec['route_id']} | succ={rec['success']} | hw={rec['hit_wall']} | "
            f"loop_len={rec['loop_len']} | lenR={_json_num(rec['len_ratio'])} | err={_json_num(rec['final_error_m'])}m"
        )
        case_js.append(
            json.dumps(
                {
                    "name": f"case_{i+1}_rid_{rec['route_id']}",
                    "color": color,
                    "popup": popup,
                    "pred": pred_ll,
                    "loop": loop_ll,
                    "gt": gt_ll,
                },
                ensure_ascii=False,
            )
        )

    if not all_pts:
        raise SystemExit("[FATAL] all selected routes are invalid after coord conversion.")

    lat0 = float(np.mean([p[0] for p in all_pts]))
    lon0 = float(np.mean([p[1] for p in all_pts]))
    zoom0 = 12

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Way-CASD Loop Cases (Leaflet)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
    .legend {{ background: rgba(255,255,255,0.9); padding: 8px 10px; line-height: 1.4; }}
  </style>
</head>
<body>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map = L.map('map').setView([{lat0}, {lon0}], {zoom0});
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);

const cases = [{",".join(case_js)}];
const overlays = {{}};
for (const c of cases) {{
  const g = L.layerGroup();
  const pred = L.polyline(c.pred, {{color: c.color, weight: 4, opacity: 0.9}}).addTo(g);
  pred.bindPopup(c.popup);
  if (c.loop.length >= 2) {{
    L.polyline(c.loop, {{color: '#d62728', weight: 6, opacity: 0.95}}).addTo(g);
  }}
  if (c.gt.length >= 2) {{
    L.polyline(c.gt, {{color: '#9e9e9e', weight: 2, opacity: 0.7, dashArray: '4,4'}}).addTo(g);
  }}
  overlays[c.name] = g;
  g.addTo(map);
}}
L.control.layers(null, overlays, {{collapsed: false}}).addTo(map);
</script>
</body>
</html>
"""

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")

    meta = {
        "ok": True,
        "task": "waycasd_plot_loop_leaflet",
        "created_at": datetime.now(tz=TZ_SHANGHAI).isoformat(),
        "inputs": {
            "per_route_json": str(args.per_route_json),
            "way_features_npz": str(args.way_features_npz),
            "mode": str(args.mode),
            "city": int(args.city),
            "only_failed": bool(args.only_failed),
            "sort_by": str(args.sort_by),
            "max_cases": int(args.max_cases),
        },
        "n_selected": int(len(picked)),
        "out_html": str(args.out_html),
    }
    meta_path = args.out_html.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {args.out_html}")
    print(f"[OK] saved: {meta_path}")


if __name__ == "__main__":
    main()

