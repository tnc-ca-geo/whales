"""Match Maxar PAN (P1BS) and MS (M1BS) rasters in a folder using metadata."""

import re
import sys
from pathlib import Path

RASTER_EXTS = {".tif", ".tiff", ".ntf", ".ntif"}
META_EXTS = (".IMD", ".imd", ".XML", ".xml")

META_KEYS = ("CatId", "firstLineTime", "productCatalogId")

# Maxar product codes: P1BS/M1BS (1B), P2AS/M2AS (2A), P3DS/M3DS, ...
PRODUCT_RE = re.compile(r"-([PM])\d[A-Z]{2}-")


def base_name(path: Path) -> str:
    """Filename with the PAN/MS product code neutralised."""
    return PRODUCT_RE.sub(lambda m: f"-X{m.group(0)[2:]}", path.stem)


def product_code(path: Path) -> str | None:
    m = PRODUCT_RE.search(path.stem)
    return m.group(1) if m else None


def read_metadata(path: Path) -> dict:
    """Pull CatId / firstLineTime / productCatalogId from a sidecar .IMD or .XML."""
    for ext in META_EXTS:
        meta = path.with_suffix(ext)
        if meta.exists():
            text = meta.read_text(errors="ignore")
            out = {}
            for key in META_KEYS:
                m = re.search(rf"{key}\s*[=>]\s*\"?([^\"<;\s]+)", text, re.IGNORECASE)
                if m:
                    out[key] = m.group(1).strip()
            return out
    return {}


def find_rasters(folder: Path) -> list[Path]:
    """All rasters anywhere below `folder` (Maxar nests them ~2 levels down)."""
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in RASTER_EXTS and not p.name.startswith(".")
    )


def match_key(path: Path, meta: dict) -> str:
    """Join key shared by a PAN/MS pair.

    PAN and MS tiles of the same scene get different filenames (different
    timestamps and _P00N part numbers), so the sidecar metadata is the only
    reliable link. Fall back to the filename if no metadata is present.
    """
    catalog_id = meta.get("productCatalogId")
    if catalog_id:
        return f"catalogId:{catalog_id}"
    return f"name:{base_name(path)}"


def find_pairs(folder: Path):
    rasters = find_rasters(folder)
    meta_by_path = {p: read_metadata(p) for p in rasters}

    pans = [p for p in rasters if product_code(p) == "P"]
    mss = [p for p in rasters if product_code(p) == "M"]
    unknown = [p for p in rasters if product_code(p) is None]

    ms_by_key: dict[str, list[Path]] = {}
    for ms in mss:
        ms_by_key.setdefault(match_key(ms, meta_by_path[ms]), []).append(ms)

    pairs, unmatched = [], list(unknown)
    paired_ms = set()

    for pan in pans:
        pan_meta = meta_by_path[pan]
        candidates = ms_by_key.get(match_key(pan, pan_meta), [])
        ms = next((c for c in candidates if c not in paired_ms), None)
        if ms is None:
            unmatched.append(pan)
            continue

        paired_ms.add(ms)
        ms_meta = meta_by_path[ms]
        same_collect = all(
            pan_meta.get(k) == ms_meta.get(k)
            for k in ("CatId", "productCatalogId")
            if k in pan_meta and k in ms_meta
        )
        pairs.append((pan, ms, same_collect, pan_meta, ms_meta))

    unmatched.extend(ms for ms in mss if ms not in paired_ms)
    return pairs, unmatched


if __name__ == "__main__":
    folder = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/Users/annietaylor/Documents/Whales/Whales_Uplift_15/2a_11Dec2025"
    )

    if not folder.is_dir():
        sys.exit(f"Not a directory: {folder}")

    rasters = find_rasters(folder)
    print(f"Scanning {folder}")
    print(f"Found {len(rasters)} raster(s)\n")

    pairs, unmatched = find_pairs(folder)

    for pan, ms, ok, pan_meta, ms_meta in pairs:
        flag = "OK " if ok else "!! "
        print(f"{flag}{pan.name}  <->  {ms.name}")
        print(f"    PAN: {pan_meta}")
        print(f"    MS : {ms_meta}")

    for p in unmatched:
        print(f"NO MATCH: {p.relative_to(folder)}")

    print(f"\n{len(pairs)} pair(s), {len(unmatched)} unmatched")
