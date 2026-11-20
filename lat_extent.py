"""Most northern and most southern latitude across all dates.

Uses every multispectral (M) and panchromatic (P) image of each pair, based on
the valid-data footprints (nodata fill excluded) in EPSG:4326.
"""

from __future__ import annotations

import re
from pathlib import Path

import folium
from shapely.geometry import mapping

from footprint_area import (
    EXCLUDE_IMGS,
    EXTENSIONS,
    _EXCLUDED_UPPER,
    _footprint_wgs84,
)

# ---------------------------------------------------------------------------
# CONFIG: folder holding the imagery (one sub-folder per date)
# ---------------------------------------------------------------------------
ROOT = Path(r"C:\Users\alicia.canales\Projects\WhalesAI\imagery\GAIA Uplift Request #14")
# ---------------------------------------------------------------------------

OUTPUT_HTML = Path(__file__).with_name("lat_extent_map.html")

# Matches the M (multispectral) and P (panchromatic) product tags, e.g. -M2AS- / -P2AS-.
PAIR_PATTERN = re.compile(r"-[MP]\d[A-Z]{2}-", re.IGNORECASE)


def _image_paths(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
        and PAIR_PATTERN.search(p.name)
        and not (EXCLUDE_IMGS and p.name.upper() in _EXCLUDED_UPPER)
    )


def _style(color: str, weight: int, fill: float):
    return lambda _: {"color": color, "weight": weight, "fillOpacity": fill}


def main() -> None:
    footprints = []  # (label, geometry)
    for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        print(f"Processing {folder.name} ...")
        for path in _image_paths(folder):
            try:
                footprints.append((f"{folder.name} / {path.name}", _footprint_wgs84(path)))
            except Exception as exc:
                print(f"  skipped {path.name}: {exc}")

    if not footprints:
        print("No images found.")
        return

    north_label, north_geom = max(footprints, key=lambda f: f[1].bounds[3])
    south_label, south_geom = min(footprints, key=lambda f: f[1].bounds[1])
    north_lat = north_geom.bounds[3]
    south_lat = south_geom.bounds[1]
    print(f"\nMost northern latitude: {north_lat:.6f} ({north_label})")
    print(f"Most southern latitude: {south_lat:.6f} ({south_label})")

    min_lon = min(g.bounds[0] for _, g in footprints)
    max_lon = max(g.bounds[2] for _, g in footprints)
    centre = [(north_lat + south_lat) / 2, (min_lon + max_lon) / 2]
    fmap = folium.Map(location=centre, zoom_start=8, tiles="OpenStreetMap")
    folium.TileLayer("Esri.WorldImagery", name="Satellite").add_to(fmap)

    all_layer = folium.FeatureGroup(name="All footprints").add_to(fmap)
    for label, geom in footprints:
        folium.GeoJson(
            mapping(geom), style_function=_style("#3388ff", 1, 0.05), tooltip=label
        ).add_to(all_layer)

    for lat, label, geom, color, name in (
        (north_lat, north_label, north_geom, "red", "Most northern"),
        (south_lat, south_label, south_geom, "green", "Most southern"),
    ):
        text = f"{name}: {lat:.6f}<br>{label}"
        folium.GeoJson(
            mapping(geom), style_function=_style(color, 3, 0.15), tooltip=text
        ).add_to(fmap)
        folium.PolyLine(
            [[lat, min_lon], [lat, max_lon]], color=color, weight=2, dash_array="8", tooltip=text
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    fmap.fit_bounds([[south_lat, min_lon], [north_lat, max_lon]])
    fmap.save(OUTPUT_HTML)
    print(f"Map saved to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
