"""
Utilities for drawing GeoJSON overlays in matplotlib without extra dependencies.

KISS note:
- We intentionally avoid geopandas/shapely to keep the visualization stack lightweight.
- Supported geometry types: Polygon, MultiPolygon, LineString, MultiLineString.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class GeoJSONFeature:
    geometry_type: str
    coordinates: object
    properties: Dict[str, object]


def load_geojson_features(path: Path) -> List[GeoJSONFeature]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("type") == "FeatureCollection":
        feats = obj.get("features", [])
        out: List[GeoJSONFeature] = []
        for f in feats:
            geom = f.get("geometry") or {}
            out.append(
                GeoJSONFeature(
                    geometry_type=str(geom.get("type", "")),
                    coordinates=geom.get("coordinates"),
                    properties=dict(f.get("properties") or {}),
                )
            )
        return out

    if obj.get("type") == "Feature":
        geom = obj.get("geometry") or {}
        return [
            GeoJSONFeature(
                geometry_type=str(geom.get("type", "")),
                coordinates=geom.get("coordinates"),
                properties=dict(obj.get("properties") or {}),
            )
        ]

    # Raw geometry object
    if "coordinates" in obj and "type" in obj:
        return [
            GeoJSONFeature(
                geometry_type=str(obj.get("type", "")),
                coordinates=obj.get("coordinates"),
                properties={},
            )
        ]

    raise ValueError(f"Unsupported GeoJSON format: {path}")


def _iter_lines_from_geom(geometry_type: str, coordinates: object) -> Iterator[List[Tuple[float, float]]]:
    if not coordinates:
        return

    if geometry_type == "LineString":
        line = [(float(x), float(y)) for x, y in coordinates]  # type: ignore[assignment]
        if len(line) >= 2:
            yield line
        return

    if geometry_type == "MultiLineString":
        for coords in coordinates:  # type: ignore[assignment]
            line = [(float(x), float(y)) for x, y in coords]
            if len(line) >= 2:
                yield line
        return

    if geometry_type == "Polygon":
        # Exterior ring first, then holes (we draw all as lines)
        for ring in coordinates:  # type: ignore[assignment]
            line = [(float(x), float(y)) for x, y in ring]
            if len(line) >= 2:
                yield line
        return

    if geometry_type == "MultiPolygon":
        for poly in coordinates:  # type: ignore[assignment]
            for ring in poly:
                line = [(float(x), float(y)) for x, y in ring]
                if len(line) >= 2:
                    yield line
        return


def iter_geojson_lines(feature: GeoJSONFeature) -> Iterator[List[Tuple[float, float]]]:
    yield from _iter_lines_from_geom(feature.geometry_type, feature.coordinates)


def geojson_bbox(features: Iterable[GeoJSONFeature]) -> Optional[Tuple[float, float, float, float]]:
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    any_pt = False
    for f in features:
        for line in _iter_lines_from_geom(f.geometry_type, f.coordinates):
            for x, y in line:
                any_pt = True
                minx = min(minx, x)
                maxx = max(maxx, x)
                miny = min(miny, y)
                maxy = max(maxy, y)
    if not any_pt:
        return None
    return (minx, miny, maxx, maxy)


def geojson_label_points(
    features: Iterable[GeoJSONFeature],
    name_prop: str = "name",
) -> List[Tuple[str, float, float]]:
    """
    Return (label, lon, lat) points computed by a cheap centroid approximation:
    average of exterior ring coordinates (first ring) for polygon-like geometries.
    """
    out: List[Tuple[str, float, float]] = []
    for f in features:
        label = str(f.properties.get(name_prop, "")).strip()
        if not label:
            continue
        if f.geometry_type not in ("Polygon", "MultiPolygon"):
            continue
        lines = list(_iter_lines_from_geom(f.geometry_type, f.coordinates))
        if not lines:
            continue
        # Pick the longest ring as "main boundary"
        ring = max(lines, key=len)
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        if not xs:
            continue
        out.append((label, float(sum(xs) / len(xs)), float(sum(ys) / len(ys))))
    return out
