"""Per-date footprint statistics for the Whales_Uplift_15 imagery.

For each date folder, computes:
  * the non-overlapping (union) area of all image footprints, in km2
  * the mean area of an individual image footprint, in km2

Footprints are the valid-data outlines (black/nodata fill excluded), not the
full raster extents. Areas are measured in a Lambert azimuthal equal-area
projection centred on each date's own imagery.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import rasterio
from affine import Affine
from rasterio import features
from rasterio.enums import Resampling
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

ROOT = Path("/Users/annietaylor/Documents/Whales/Whales_Uplift_15")
EXTENSIONS = (".tif", ".tiff")
# Multispectral products are tagged M<level> (e.g. M2AS); panchromatic are P<level>.
NAME_FILTER = "-M2AS-"

# Footprints are traced from a decimated overview; this caps the longest side.
MAX_DIM = 2000
# WorldView 2A products pad the rotated swath with zeros rather than nodata.
TREAT_ZERO_AS_NODATA = True

# Drop specific rasters from the union and mean footprint calculations.
# THESE ARE IMGS I HAVEN'T BEEN ABLE TO UPLOAD, excluding for now
EXCLUDE_IMGS = True
EXCLUDED_NAMES = {
    "24NOV27190359-M2AS-510345993010_01_P004.TIF",
    "24DEC04191641-M2AS-510345993020_01_P004.TIF",
}
_EXCLUDED_UPPER = {n.upper() for n in EXCLUDED_NAMES}


def _footprint_wgs84(path: Path):
    """Valid-data footprint of one raster as a shapely geometry in EPSG:4326."""
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("raster has no CRS")

        scale = max(1, math.ceil(max(src.width, src.height) / MAX_DIM))
        out_h = max(1, src.height // scale)
        out_w = max(1, src.width // scale)

        mask = src.dataset_mask(
            out_shape=(out_h, out_w), resampling=Resampling.nearest
        ).astype(bool)

        if TREAT_ZERO_AS_NODATA:
            data = src.read(
                1, out_shape=(out_h, out_w), resampling=Resampling.nearest
            )
            mask &= data != 0

        if not mask.any():
            raise ValueError("raster contains no valid pixels")

        transform = src.transform * Affine.scale(
            src.width / out_w, src.height / out_h
        )
        pixel_size = max(abs(transform.a), abs(transform.e))

        polygons = [
            shape(geom)
            for geom, value in features.shapes(
                mask.astype("uint8"), mask=mask, transform=transform
            )
            if value == 1
        ]
        footprint = unary_union(polygons).buffer(0).simplify(pixel_size)
        return shape(transform_geom(src.crs, "EPSG:4326", mapping(footprint)))


def _laea_crs(lon: float, lat: float):
    """Equal-area CRS centred on the given point, for area measurement."""
    return rasterio.crs.CRS.from_proj4(
        f"+proj=laea +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 "
        "+datum=WGS84 +units=m +no_defs"
    )


def date_folder_stats(folder: Path) -> dict:
    """Union and mean footprint areas (km2) for every raster under `folder`."""
    paths = sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
        and (NAME_FILTER is None or NAME_FILTER.upper() in p.name.upper())
        and not (EXCLUDE_IMGS and p.name.upper() in _EXCLUDED_UPPER)
    )

    footprints = []
    failures = []
    for path in paths:
        try:
            footprints.append(_footprint_wgs84(path))
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")

    if not footprints:
        return {
            "date_folder": folder.name,
            "n_images": 0,
            "union_area_km2": float("nan"),
            "mean_footprint_km2": float("nan"),
            "sum_footprint_km2": float("nan"),
            "overlap_km2": float("nan"),
            "errors": "; ".join(failures) or None,
        }

    merged = unary_union(footprints)
    centroid = merged.centroid
    dst_crs = _laea_crs(centroid.x, centroid.y)

    def area_km2(geom) -> float:
        return shape(transform_geom("EPSG:4326", dst_crs, mapping(geom))).area / 1e6

    individual = [area_km2(g) for g in footprints]
    union_area = area_km2(merged)

    return {
        "date_folder": folder.name,
        "n_images": len(footprints),
        "union_area_km2": union_area,
        "mean_footprint_km2": sum(individual) / len(individual),
        "sum_footprint_km2": sum(individual),
        "overlap_km2": sum(individual) - union_area,
        "errors": "; ".join(failures) or None,
    }


def main() -> None:
    date_folders = sorted(p for p in ROOT.iterdir() if p.is_dir())

    records = []
    for folder in date_folders:
        print(f"Processing {folder.name} ...")
        records.append(date_folder_stats(folder))

    df = pd.DataFrame(records)
    pd.set_option("display.width", 200)
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    total_images = int(df["n_images"].sum())
    if total_images:
        mean_all = df["sum_footprint_km2"].sum() / total_images
        print(f"\nImages across all dates: {total_images}")
        print(f"Mean footprint area, all images: {mean_all:,.2f} km2")
        print(f"Total non-overlapping area, all dates: {df['union_area_km2'].sum():,.2f} km2")

    df.to_csv(Path(__file__).with_name("footprint_area_by_date.csv"), index=False)


if __name__ == "__main__":
    main()
