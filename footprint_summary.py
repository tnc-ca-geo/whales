"""Overall multispectral footprint statistics plus a map of per-date unions.

Prints the mean footprint area (km2) of every multispectral image across all
date folders, and writes a Leaflet map (same style as ``compare_pair.py``) with
each date's union footprint as its own toggleable, coloured layer.
Also writes the union footprints to GeoJSON (EPSG:4326).
"""

from __future__ import annotations

import json
import re
import webbrowser
from datetime import datetime
from pathlib import Path

import folium
import pandas as pd
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from footprint_area import (
    EXTENSIONS,
    NAME_FILTER,
    ROOT,
    _footprint_wgs84,
    _laea_crs,
)

OUT_DIR = Path(__file__).parent
MAP_PATH = OUT_DIR / "footprint_union_map.html"
GEOJSON_PATH = OUT_DIR / "footprint_unions.geojson"
IMAGE_CSV_PATH = OUT_DIR / "footprint_area_by_image.csv"

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _folder_date(name: str) -> datetime:
    """Parse the date embedded in a folder name like '2a_11Dec2025'."""
    match = re.search(r"(\d{1,2})([A-Za-z]{3})(\d{2,4})", name)
    if match:
        day, mon, year = match.groups()
        fmt = "%d%b%Y" if len(year) == 4 else "%d%b%y"
        try:
            return datetime.strptime(f"{day}{mon}{year}", fmt)
        except ValueError:
            pass
    return datetime.max


def collect_footprints() -> dict[str, list]:
    """Valid-data footprints (EPSG:4326) of every multispectral image, by date."""
    date_folders = sorted(
        (p for p in ROOT.iterdir() if p.is_dir()), key=lambda p: _folder_date(p.name)
    )

    by_date: dict[str, list] = {}
    for folder in date_folders:
        paths = sorted(
            p
            for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
            and (NAME_FILTER is None or NAME_FILTER.upper() in p.name.upper())
        )
        print(f"Processing {folder.name} ({len(paths)} images) ...")

        footprints = []
        for path in paths:
            try:
                footprints.append((path.name, _footprint_wgs84(path)))
            except Exception as exc:
                print(f"  skipped {path.name}: {exc}")
        if footprints:
            by_date[folder.name] = footprints
    return by_date


def _date_label(name: str) -> str:
    """'2a_11Dec2025' -> '11 Dec 2025', falling back to the folder name."""
    parsed = _folder_date(name)
    return name if parsed == datetime.max else parsed.strftime("%d %b %Y")


def _legend_html(entries: list[tuple[str, str]]) -> str:
    rows = "".join(
        f'<div style="margin:2px 0;">'
        f'<span style="display:inline-block;width:14px;height:14px;'
        f"background:{colour};opacity:0.6;border:1.5px solid {colour};"
        f'vertical-align:middle;margin-right:6px;"></span>{label}</div>'
        for label, colour in entries
    )
    return (
        '<div style="position:fixed;bottom:25px;left:15px;z-index:9999;'
        "background:rgba(255,255,255,0.9);padding:8px 12px;border:1px solid #999;"
        'border-radius:4px;font:13px/1.4 sans-serif;">'
        '<div style="font-weight:bold;margin-bottom:4px;">Image date</div>'
        f"{rows}</div>"
    )


def build_map(unions: dict, by_date: dict, areas: dict, out_html: Path) -> Path:
    """Leaflet map with one coloured, toggleable layer per acquisition date."""
    fmap = folium.Map(tiles=None)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri World Imagery",
    ).add_to(fmap)

    souths, wests, norths, easts = [], [], [], []
    legend_entries = []

    for i, (name, geom) in enumerate(unions.items()):
        colour = PALETTE[i % len(PALETTE)]
        n_images = len(by_date[name])
        label = f"{name} ({areas[name]:,.0f} km2, n={n_images})"
        legend_entries.append((_date_label(name), colour))

        folium.GeoJson(
            mapping(geom),
            name=label,
            style_function=lambda _f, colour=colour: {
                "color": colour,
                "weight": 2,
                "fillColor": colour,
                "fillOpacity": 0.25,
            },
            tooltip=label,
            popup=folium.Popup(
                f"<b>{name}</b><br>"
                f"images: {n_images}<br>"
                f"non-overlapping area: {areas[name]:,.2f} km2",
                max_width=320,
            ),
        ).add_to(fmap)

        west, south, east, north = geom.bounds
        souths.append(south)
        wests.append(west)
        norths.append(north)
        easts.append(east)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[min(souths), min(wests)], [max(norths), max(easts)]])
    fmap.get_root().html.add_child(folium.Element(_legend_html(legend_entries)))
    fmap.save(out_html)

    return out_html


def main() -> None:
    by_date = collect_footprints()
    if not by_date:
        print("No multispectral rasters found.")
        return

    all_geoms = [geom for items in by_date.values() for _, geom in items]
    centroid = unary_union(all_geoms).centroid
    # One equal-area projection for the whole study area keeps areas comparable.
    dst_crs = _laea_crs(centroid.x, centroid.y)

    def area_km2(geom) -> float:
        return shape(transform_geom("EPSG:4326", dst_crs, mapping(geom))).area / 1e6

    rows = []
    unions = {}
    for date_name, items in by_date.items():
        unions[date_name] = unary_union([geom for _, geom in items])
        for image_name, geom in items:
            rows.append(
                {
                    "date_folder": date_name,
                    "image": image_name,
                    "footprint_km2": area_km2(geom),
                }
            )

    union_areas = {name: area_km2(geom) for name, geom in unions.items()}

    df = pd.DataFrame(rows)
    df.to_csv(IMAGE_CSV_PATH, index=False)

    print()
    print(
        df.groupby("date_folder", sort=False)["footprint_km2"]
        .agg(n_images="count", mean_km2="mean", total_km2="sum")
        .to_string(float_format=lambda v: f"{v:,.2f}")
    )
    print(f"\nImages: {len(df)}")
    print(f"Mean footprint area, all dates: {df['footprint_km2'].mean():,.2f} km2")
    print(f"Median: {df['footprint_km2'].median():,.2f} km2")
    print(
        f"Range: {df['footprint_km2'].min():,.2f} - "
        f"{df['footprint_km2'].max():,.2f} km2"
    )

    with open(GEOJSON_PATH, "w") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "date_folder": name,
                            "union_area_km2": union_areas[name],
                            "n_images": len(by_date[name]),
                        },
                        "geometry": mapping(geom),
                    }
                    for name, geom in unions.items()
                ],
            },
            f,
        )

    build_map(unions, by_date, union_areas, MAP_PATH)
    print(f"\nMap written to {MAP_PATH}")
    webbrowser.open(MAP_PATH.resolve().as_uri())


if __name__ == "__main__":
    main()
