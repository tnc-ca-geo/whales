"""Calculate the surface area (km2) of geographic rasters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

EARTH_RADIUS_KM = 6371.0088

# Folder containing the rasters
RASTER_FOLDER = "/Users/annietaylor/Documents/Whales/Whales_Uplift_15/2a_11Dec2025"


def _row_areas_km2(transform, width: int, height: int) -> np.ndarray:
    """Area of a single cell in each raster row, in km2.

    Cell area shrinks towards the poles, so it is computed per row from the
    spherical band formula R^2 * dlon * (sin(lat_top) - sin(lat_bottom)).
    """
    dlon_rad = np.radians(abs(transform.a))

    row_edges = np.arange(height + 1)
    lat_edges = transform.f + row_edges * transform.e
    sin_lat = np.sin(np.radians(np.clip(lat_edges, -90.0, 90.0)))

    return np.abs(EARTH_RADIUS_KM**2 * dlon_rad * np.diff(sin_lat))


def _projected_row_areas_km2(transform, height: int, crs) -> np.ndarray:
    """Constant cell area for a projected CRS, repeated per row."""
    metres_per_unit = crs.linear_units_factor[1]
    cell_area_km2 = (
        abs(transform.a * transform.e - transform.b * transform.d)
        * metres_per_unit**2
        / 1e6
    )
    return np.full(height, cell_area_km2)


def raster_area_km2(path: str | Path, band: int = 1) -> dict:
    """Return extent and valid-data areas in km2 for one raster."""
    path = Path(path)

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path.name}: raster has no CRS, cannot compute area.")

        if src.crs.is_geographic:
            row_areas = _row_areas_km2(src.transform, src.width, src.height)
        else:
            row_areas = _projected_row_areas_km2(src.transform, src.height, src.crs)

        data = src.read(band, masked=True)
        valid = ~np.ma.getmaskarray(data)
        if np.issubdtype(data.dtype, np.floating):
            valid &= ~np.isnan(data.filled(np.nan))

        valid_per_row = valid.sum(axis=1)

        return {
            "name": path.name,
            "path": str(path),
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "total_cells": src.width * src.height,
            "valid_cells": int(valid_per_row.sum()),
            "valid_area_km2": float(np.dot(valid_per_row, row_areas)),
            "extent_area_km2": float(row_areas.sum() * src.width),
        }


def raster_areas_in_folder(
    folder: str | Path,
    extensions: tuple[str, ...] = (".tif", ".tiff"),
    band: int = 1,
) -> pd.DataFrame:
    """Recursively calculate areas for every raster in a folder."""
    folder = Path(folder)
    # Matched on the suffix rather than by glob so .TIF and .tif both qualify.
    paths = sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    records = []
    for path in paths:
        try:
            record = raster_area_km2(path, band=band)
            record["error"] = None
        except Exception as exc:
            record = {"name": path.name, "path": str(path), "error": str(exc)}
        records.append(record)

    columns = [
        "name",
        "crs",
        "width",
        "height",
        "total_cells",
        "valid_cells",
        "valid_area_km2",
        "extent_area_km2",
        "path",
        "error",
    ]
    return pd.DataFrame(records, columns=columns)


def main() -> None:
    df = raster_areas_in_folder(RASTER_FOLDER)
    print(df.to_string(index=False))

    if not df.empty:
        print(f"\nTotal valid area: {df['valid_area_km2'].sum():,.2f} km2")
        print(f"\nMean valid area: {df['valid_area_km2'].mean():,.2f} km2")


if __name__ == "__main__":
    main()
