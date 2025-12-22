"""
GeoJSON basemap overlay helpers (matplotlib-only, no geopandas/shapely).

Design goals:
- KISS: lightweight, dependency-free overlay for publication figures.
- Works with WGS84 lon/lat GeoJSON (Polygon/MultiPolygon/LineString/MultiLineString).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from src.visualization.geojson_utils import geojson_label_points, iter_geojson_lines, load_geojson_features


@dataclass(frozen=True)
class BasemapStyle:
    edgecolor: str = "#3A3A3A"
    facecolor: str = "none"  # use "none" for transparent fill
    linewidth: float = 0.7
    alpha: float = 0.55
    label: bool = False
    label_size: int = 8


def draw_geojson_basemap(
    ax: plt.Axes,
    geojson_path: Optional[Path],
    style: BasemapStyle,
    zorder_base: int = 0,
    name_prop: str = "name",
) -> None:
    if geojson_path is None:
        return
    if not geojson_path.exists():
        raise FileNotFoundError(geojson_path)
    feats = load_geojson_features(geojson_path)

    # Optional fill (only exterior ring; avoid filling holes).
    if str(style.facecolor).lower() != "none":
        for f in feats:
            if f.geometry_type == "Polygon" and f.coordinates:
                ring0 = f.coordinates[0]  # type: ignore[index]
                xs = [float(x) for x, _ in ring0]
                ys = [float(y) for _, y in ring0]
                ax.fill(
                    xs,
                    ys,
                    facecolor=style.facecolor,
                    edgecolor="none",
                    alpha=float(style.alpha) * 0.35,
                    zorder=int(zorder_base),
                )
            elif f.geometry_type == "MultiPolygon" and f.coordinates:
                for poly in f.coordinates:  # type: ignore[assignment]
                    if not poly:
                        continue
                    ring0 = poly[0]
                    xs = [float(x) for x, _ in ring0]
                    ys = [float(y) for _, y in ring0]
                    ax.fill(
                        xs,
                        ys,
                        facecolor=style.facecolor,
                        edgecolor="none",
                        alpha=float(style.alpha) * 0.35,
                        zorder=int(zorder_base),
                    )

    # Outlines
    for f in feats:
        for line in iter_geojson_lines(f):
            xs = [p[0] for p in line]
            ys = [p[1] for p in line]
            ax.plot(
                xs,
                ys,
                color=str(style.edgecolor),
                linewidth=float(style.linewidth),
                alpha=float(style.alpha),
                zorder=int(zorder_base) + 1,
            )

    # Labels (optional)
    if style.label:
        for name, x, y in geojson_label_points(feats, name_prop=name_prop):
            ax.text(
                x,
                y,
                name,
                fontsize=int(style.label_size),
                color=str(style.edgecolor),
                alpha=min(1.0, float(style.alpha) + 0.15),
                ha="center",
                va="center",
                zorder=int(zorder_base) + 2,
            )

