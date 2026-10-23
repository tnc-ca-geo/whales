# WhalesAI Image Analysis

Scripts for inspecting WorldView-3 satellite imagery footprints and matching valid MS + Panchromatic image pairs for uploading to whalesAI.

## Scripts

- [check_pairs.py](check_pairs.py) — Matches panchromatic and multispectral WV3 rasters within a folder using their metadata (CatId, first line time, product catalog ID).
- [compare_pair.py](compare_pair.py) — Compares a matched PAN/MS pair's bounds and footprint IoU and writes an interactive Leaflet map with both rasters as image overlays (e.g. for one aquisition date).
- [raster_area.py](raster_area.py) — Reports the extent and valid-data area (km²) of every raster in a folder.
- [footprint_area.py](footprint_area.py) — Computes, per acquisition-date folder, the non-overlapping (union) area and mean footprint area of the multispectral images.
- [footprint_summary.py](footprint_summary.py) — Summarises mean footprint area across all dates and writes a colour-coded Leaflet map and GeoJSON of each date's union footprint.

## Setup

```sh
uv venv
uv pip install rasterio shapely pandas folium pillow numpy
```
