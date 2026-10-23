"""Compare a matched Maxar PAN/MS pair: bounds, IoU, and a side-by-side Leaflet map.

Takes the first matched pair found by ``check_pairs.find_pairs``, reports the
footprint geometry of each raster, computes the intersection-over-union of the
two footprints, and writes an HTML map with both rasters drawn as image
overlays so they can be compared visually.

Usage:
    python compare_pair.py [FOLDER] [PAIR_INDEX]   # one pair, with image overlays
    python compare_pair.py --all [FOLDER]          # every pair, combined footprint map
"""

from __future__ import annotations

import sys
import tempfile
import webbrowser
from pathlib import Path

import folium
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from check_pairs import find_pairs

DEFAULT_FOLDER = "/Users/annietaylor/Documents/Whales/Whales_Uplift_15/2a_11Dec2025"
WGS84 = "EPSG:4326"

# Longest edge, in pixels, of the PNG overlays written for the map.
OVERLAY_MAX_DIM = 1400

# WorldView-3 multispectral band order is coastal, blue, green, yellow, red,
# red-edge, NIR1, NIR2, so natural colour is bands 5/3/2.
MS_RGB_BANDS = (5, 3, 2)

STRETCH_PERCENTILES = (2.0, 98.0)


def raster_info(path: Path) -> dict:
    """Geometry summary of one raster, with bounds in both native CRS and WGS84."""
    with rasterio.open(path) as src:
        bounds = src.bounds
        return {
            "path": path,
            "name": path.name,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "res": src.res,
            "bounds": tuple(bounds),
            "bounds_wgs84": transform_bounds(src.crs, WGS84, *bounds, densify_pts=21),
        }


def bbox_iou(a: tuple, b: tuple) -> tuple[float, float, float]:
    """Intersection-over-union of two (left, bottom, right, top) boxes.

    Returns (iou, intersection_area, union_area) in the units of the boxes.
    """
    left = max(a[0], b[0])
    bottom = max(a[1], b[1])
    right = min(a[2], b[2])
    top = min(a[3], b[3])

    inter = max(0.0, right - left) * max(0.0, top - bottom)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter

    return (inter / union if union > 0 else 0.0), inter, union


def _stretch(band: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Percentile-stretch one band to 0-255."""
    sample = band[valid]
    if sample.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)

    lo, hi = np.percentile(sample, STRETCH_PERCENTILES)
    if hi <= lo:
        lo, hi = float(sample.min()), float(sample.max())
    if hi <= lo:
        return np.zeros(band.shape, dtype=np.uint8)

    scaled = (band.astype(np.float32) - lo) / (hi - lo)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def write_overlay_png(path: Path, out_png: Path, max_dim: int = OVERLAY_MAX_DIM) -> tuple:
    """Reproject a raster to WGS84, downsample it, and save an RGBA PNG.

    Leaflet image overlays are axis-aligned in lat/lon, so the raster is read
    through a WarpedVRT rather than reprojected after the fact. Returns the
    WGS84 bounds the PNG should be pinned to.
    """
    with rasterio.open(path) as src:
        with WarpedVRT(src, crs=WGS84, resampling=Resampling.bilinear) as vrt:
            scale = max(vrt.width, vrt.height) / max_dim
            out_w = max(1, round(vrt.width / scale)) if scale > 1 else vrt.width
            out_h = max(1, round(vrt.height / scale)) if scale > 1 else vrt.height

            band_idx = MS_RGB_BANDS if vrt.count >= max(MS_RGB_BANDS) else (1, 1, 1)
            data = vrt.read(
                band_idx,
                out_shape=(len(band_idx), out_h, out_w),
                resampling=Resampling.average,
                masked=True,
            )
            bounds = vrt.bounds

    filled = data.filled(0)
    # Warping leaves zero-filled corners outside the real footprint; treat those
    # (and any nodata) as transparent so the two overlays don't occlude each other.
    valid = ~np.ma.getmaskarray(data).any(axis=0) & (filled > 0).any(axis=0)

    rgb = np.dstack([_stretch(filled[i], valid) for i in range(filled.shape[0])])
    alpha = np.where(valid, 255, 0).astype(np.uint8)

    Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA").save(out_png)

    return bounds


def _leaflet_bounds(bounds_wgs84: tuple) -> list[list[float]]:
    """(left, bottom, right, top) -> [[south, west], [north, east]]."""
    left, bottom, right, top = bounds_wgs84
    return [[bottom, left], [top, right]]


def build_map(pan: dict, ms: dict, out_html: Path) -> Path:
    """Write a Leaflet map with both rasters as toggleable image overlays."""
    print("Rendering overlays (reprojecting to WGS84)...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pan_png = tmp_dir / "pan.png"
        ms_png = tmp_dir / "ms.png"

        pan_overlay_bounds = write_overlay_png(pan["path"], pan_png)
        ms_overlay_bounds = write_overlay_png(ms["path"], ms_png)

        fmap = _assemble_map(
            pan, ms, pan_png, ms_png, pan_overlay_bounds, ms_overlay_bounds
        )
        # Folium base64-embeds images given as paths, so the PNGs can be
        # discarded and the HTML stays a single portable file.
        fmap.save(out_html)

    return out_html


def _assemble_map(
    pan: dict,
    ms: dict,
    pan_png: Path,
    ms_png: Path,
    pan_overlay_bounds: tuple,
    ms_overlay_bounds: tuple,
) -> folium.Map:
    south = min(pan["bounds_wgs84"][1], ms["bounds_wgs84"][1])
    west = min(pan["bounds_wgs84"][0], ms["bounds_wgs84"][0])
    north = max(pan["bounds_wgs84"][3], ms["bounds_wgs84"][3])
    east = max(pan["bounds_wgs84"][2], ms["bounds_wgs84"][2])

    fmap = folium.Map(tiles=None)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri World Imagery",
    ).add_to(fmap)

    folium.raster_layers.ImageOverlay(
        name=f"MS: {ms['name']}",
        image=str(ms_png),
        bounds=_leaflet_bounds(ms_overlay_bounds),
        opacity=1.0,
    ).add_to(fmap)
    folium.raster_layers.ImageOverlay(
        name=f"PAN: {pan['name']}",
        image=str(pan_png),
        bounds=_leaflet_bounds(pan_overlay_bounds),
        opacity=0.6,
    ).add_to(fmap)

    folium.Rectangle(
        bounds=_leaflet_bounds(ms["bounds_wgs84"]),
        color="#ff7f0e",
        weight=2,
        fill=False,
        tooltip=f"MS footprint: {ms['name']}",
    ).add_to(fmap)
    folium.Rectangle(
        bounds=_leaflet_bounds(pan["bounds_wgs84"]),
        color="#1f77b4",
        weight=2,
        fill=False,
        tooltip=f"PAN footprint: {pan['name']}",
    ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[south, west], [north, east]])

    return fmap


def describe(label: str, info: dict) -> None:
    left, bottom, right, top = info["bounds"]
    w_left, w_bottom, w_right, w_top = info["bounds_wgs84"]
    print(f"{label}: {info['name']}")
    print(f"    CRS         : {info['crs']}")
    print(f"    Size        : {info['width']} x {info['height']} px, {info['bands']} band(s)")
    print(f"    Resolution  : {info['res'][0]:.3f} x {info['res'][1]:.3f} CRS units/px")
    print(f"    Bounds      : left={left:.2f} bottom={bottom:.2f} right={right:.2f} top={top:.2f}")
    print(
        f"    Bounds WGS84: W={w_left:.6f} S={w_bottom:.6f} E={w_right:.6f} N={w_top:.6f}"
    )


def pair_iou(pan: dict, ms: dict) -> tuple[float, float, float, str, str]:
    """IoU of a pair's footprints, in the shared CRS where there is one."""
    if pan["crs"] == ms["crs"]:
        iou, inter, union = bbox_iou(pan["bounds"], ms["bounds"])
        unit = "m2" if pan["crs"].linear_units == "metre" else "CRS units^2"
        return iou, inter, union, unit, str(pan["crs"])

    # Degree-space IoU is area-distorted but is the only common frame here.
    iou, inter, union = bbox_iou(pan["bounds_wgs84"], ms["bounds_wgs84"])
    return iou, inter, union, "deg2", f"{WGS84} (CRSs differ; areas are approximate)"


def product_folders(path: Path, root: Path) -> tuple[str, str]:
    """(product folder, scene folder) of a raster relative to the scanned root."""
    parts = path.relative_to(root).parts
    product = parts[0] if len(parts) > 1 else ""
    scene = parts[1] if len(parts) > 2 else ""
    return product, scene


def build_overview_map(rows: list[dict], root: Path, out_html: Path) -> Path:
    """One map showing every pair's footprint, each pair toggleable."""
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    fmap = folium.Map(tiles=None)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri World Imagery",
    ).add_to(fmap)

    souths, wests, norths, easts = [], [], [], []

    for row in rows:
        pan, ms = row["pan"], row["ms"]
        colour = palette[(row["index"] - 1) % len(palette)]
        group = folium.FeatureGroup(
            name=f"Pair {row['index']}: {row['pan_product']} + {row['ms_product']}"
        )

        for info, kind, dash in ((pan, "PAN", None), (ms, "MS", "6,6")):
            product, scene = product_folders(info["path"], root)
            folium.Rectangle(
                bounds=_leaflet_bounds(info["bounds_wgs84"]),
                color=colour,
                weight=2,
                dash_array=dash,
                fill=True,
                fill_opacity=0.08,
                tooltip=f"Pair {row['index']} {kind}: {product}",
                popup=folium.Popup(
                    f"<b>Pair {row['index']} &mdash; {kind}</b><br>"
                    f"folder: {product}/{scene}<br>"
                    f"file: {info['name']}<br>"
                    f"size: {info['width']} x {info['height']} px, {info['bands']} band(s)<br>"
                    f"IoU: {row['iou']:.6f}",
                    max_width=420,
                ),
            ).add_to(group)

        group.add_to(fmap)

        for info in (pan, ms):
            west, south, east, north = info["bounds_wgs84"]
            souths.append(south)
            wests.append(west)
            norths.append(north)
            easts.append(east)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[min(souths), min(wests)], [max(norths), max(easts)]])
    fmap.save(out_html)

    return out_html


def run_all(folder: Path) -> None:
    """Compare every matched pair and write one combined footprint map."""
    pairs, unmatched = find_pairs(folder)
    if not pairs:
        sys.exit(f"No matched PAN/MS pairs found in {folder}")

    print(f"Scanning {folder}")
    print(f"{len(pairs)} matched pair(s), {len(unmatched)} unmatched\n")

    rows = []
    for i, (pan_path, ms_path, same_collect, _, _) in enumerate(pairs, start=1):
        pan, ms = raster_info(pan_path), raster_info(ms_path)
        iou, inter, union, unit, frame = pair_iou(pan, ms)

        pan_product, pan_scene = product_folders(pan_path, folder)
        ms_product, ms_scene = product_folders(ms_path, folder)

        print(f"Pair {i}  (IoU {iou:.6f}, collect match: {'yes' if same_collect else 'NO'})")
        print(f"    PAN folder: {pan_product}/{pan_scene}")
        print(f"        file  : {pan['name']}")
        print(f"        size  : {pan['width']} x {pan['height']} px, {pan['bands']} band(s)"
              f", {pan['res'][0]:.2f} m/px")
        print(f"    MS  folder: {ms_product}/{ms_scene}")
        print(f"        file  : {ms['name']}")
        print(f"        size  : {ms['width']} x {ms['height']} px, {ms['bands']} band(s)"
              f", {ms['res'][0]:.2f} m/px")
        print(f"    Overlap   : {inter:,.1f} / {union:,.1f} {unit} in {frame}")
        print()

        rows.append(
            {
                "index": i,
                "pan": pan,
                "ms": ms,
                "iou": iou,
                "pan_product": pan_product,
                "ms_product": ms_product,
            }
        )

    print("Summary")
    print(f"    {'#':>2}  {'PAN folder':<24} {'MS folder':<24} {'IoU':>9}")
    for row in rows:
        print(
            f"    {row['index']:>2}  {row['pan_product']:<24} "
            f"{row['ms_product']:<24} {row['iou']:>9.6f}"
        )

    ious = [row["iou"] for row in rows]
    identical = sum(1 for v in ious if v >= 0.999999)
    print(
        f"\n    {identical}/{len(rows)} pair(s) have identical footprints; "
        f"IoU range {min(ious):.6f} - {max(ious):.6f}"
    )

    for p in unmatched:
        print(f"    UNMATCHED: {p.relative_to(folder)}")

    out_html = folder / "compare_pairs_overview.html"
    build_overview_map(rows, folder, out_html)
    print(f"\nOverview map written to {out_html}")
    webbrowser.open(out_html.resolve().as_uri())


def run_single(folder: Path, index: int) -> None:
    """Full comparison of one pair, including image overlays."""
    pairs, _ = find_pairs(folder)
    if not pairs:
        sys.exit(f"No matched PAN/MS pairs found in {folder}")
    if index >= len(pairs):
        sys.exit(f"Pair index {index} out of range ({len(pairs)} pair(s) found)")

    pan_path, ms_path, same_collect, _, _ = pairs[index]
    print(f"Pair {index + 1} of {len(pairs)} in {folder}")
    print(f"Metadata collect match: {'yes' if same_collect else 'NO'}\n")

    pan, ms = raster_info(pan_path), raster_info(ms_path)
    describe("PAN", pan)
    print()
    describe("MS ", ms)

    iou, inter, union, unit, frame = pair_iou(pan, ms)
    print("\nFootprint overlap")
    print(f"    Compared in : {frame}")
    print(f"    Intersection: {inter:,.1f} {unit}")
    print(f"    Union       : {union:,.1f} {unit}")
    print(f"    IoU         : {iou:.6f}")

    out_html = folder / f"compare_pair_{index + 1}.html"
    build_map(pan, ms, out_html)
    print(f"\nMap written to {out_html}")
    webbrowser.open(out_html.resolve().as_uri())


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--all"]
    compare_all = "--all" in sys.argv[1:]

    folder = Path(args[0] if args else DEFAULT_FOLDER)
    if not folder.is_dir():
        sys.exit(f"Not a directory: {folder}")

    if compare_all:
        run_all(folder)
    else:
        run_single(folder, int(args[1]) if len(args) > 1 else 0)


if __name__ == "__main__":
    main()
