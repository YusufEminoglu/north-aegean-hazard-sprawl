"""
Paper 2 — Move, Clip & Validate GEE Downloads
================================================
1. Moves all files from the directory configured by GEE_DOWNLOAD_DIR -> data/01_raw/paper2/gee_exports/
2. Clips each raster to Kuzey Ege Havzasi (EPSG:32635) -> data/02_interim/paper2/
3. Validates every clipped output (non-empty, correct CRS, valid pixel range)
4. Prints a full report: OK / WARN / FAIL per layer

IMPORTANT — nodata handling
----------------------------
Pixels outside the havza polygon (but inside the bounding box) are masked to a
declared nodata value written into the GeoTIFF metadata. Without this declaration
QGIS / downstream scripts treat those pixels as valid data.

Nodata values used:
  float32 / float64  ->  -9999.0
  int16 / int32      ->  -9999
  uint8              ->      0   (valid LULC classes start at ≥1 or ≥10)
  uint16 / uint32    ->      0

Set FORCE_RECLIP = True to overwrite already-existing interim files.
"""

import shutil
import os
import json
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.ops import unary_union
from shapely.geometry import shape

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS   = Path(os.environ.get("GEE_DOWNLOAD_DIR", PROJECT_ROOT / "data" / "downloads"))
RAW_GEE     = PROJECT_ROOT / "data" / "01_raw"  / "paper2" / "gee_exports"
INTERIM     = PROJECT_ROOT / "data" / "02_interim" / "paper2"
LOG_DIR     = PROJECT_ROOT / "logs"

RAW_GEE.mkdir(parents=True, exist_ok=True)
INTERIM.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CRS_TARGET   = "EPSG:32635"
FORCE_RECLIP = True   # overwrite existing clipped files to fix nodata declaration


def nodata_for_dtype(dtype_str: str) -> float:
    """Return a safe nodata sentinel for a given numpy/rasterio dtype string."""
    dt = str(dtype_str).lower()
    if "float" in dt:
        return -9999.0
    if "int16" in dt or "int32" in dt:
        return -9999
    # uint8, uint16, uint32 — use 0 (LULC classes never start at 0 in these products)
    return 0

# ── Load havza boundary ────────────────────────────────────────────────────
gj_path = PROJECT_ROOT / "data" / "01_raw" / "paper2" / "kuzey_ege_havzasi_v2.geojson"
with open(gj_path, encoding="utf-8") as f:
    gj = json.load(f)
gc = gj["features"][0]["geometry"]
polys = [shape(g) for g in gc["geometries"] if g["type"] in ("Polygon","MultiPolygon")]
havza_wgs84 = unary_union(polys)

gdf_wgs84 = gpd.GeoDataFrame(geometry=[havza_wgs84], crs="EPSG:4326")
gdf_utm   = gdf_wgs84.to_crs(CRS_TARGET)
havza_utm = gdf_utm.geometry.iloc[0]
havza_bounds = havza_utm.bounds  # (minx, miny, maxx, maxy)

print(f"Havza UTM bounds : {[round(v) for v in havza_bounds]}")
print(f"Havza area       : {havza_utm.area/1e6:.1f} km2")

# ── Layer catalogue ────────────────────────────────────────────────────────
# (filename_stem, expected_resolution_m, dtype_hint, valid_min, valid_max)
LAYERS = [
    # SRTM terrain
    ("p2_srtm_elevation",            30,   "float32",  -100,   4000),
    ("p2_srtm_slope",                30,   "float32",     0,     90),
    ("p2_srtm_aspect",               30,   "float32",     0,    360),
    ("p2_srtm_tpi",                  30,   "float32",  -200,    200),
    # GLAD GLCLU
    ("p2_glad_lulc_2000",            30,   "int16",       0,    255),
    ("p2_glad_lulc_2020",            30,   "int16",       0,    255),
    # GLC-FCS30D 1985-2022
    *[(f"p2_glcfcs_{y}",             30,   "int16",       0,    220) for y in [1985,1990,1995]+list(range(2000,2023))],
    # ESA WorldCover
    ("p2_worldcover_2021",           10,   "uint8",      10,    100),
    # CORINE
    *[(f"p2_corine_{y}",            100,   "int16",     111,    523) for y in [1990,2000,2006,2012,2018]],
    # JRC Surface Water
    ("p2_gsw_occurrence",            30,   "float32",     0,    100),
    ("p2_gsw_seasonality",           30,   "uint8",       0,     12),
    ("p2_gsw_recurrence",            30,   "float32",     0,    100),
    ("p2_water_distance",            30,   "float32",     0,  1e6),
    # HRSL population
    ("p2_hrsl_pop_general",          30,   "float32",     0,  5000),
    ("p2_hrsl_pop_women",            30,   "float32",     0,  2500),
    ("p2_hrsl_pop_children_u5",      30,   "float32",     0,   500),
    ("p2_hrsl_pop_elderly_60p",      30,   "float32",     0,   500),
    # VIIRS NTL 2014-2024
    *[(f"p2_viirs_ntl_{y}",         500,   "float32",     0,   200) for y in range(2014,2025)],
    # Soil
    ("p2_soilgrids_clay",           250,   "float32",     0,   100),
    ("p2_soil_texture",             250,   "uint8",       1,    12),
    # Climate
    ("p2_worldclim_bio13_maxprecip",1000,  "float32",     0,   600),
    ("p2_worldclim_bio01_meantemp", 1000,  "float32",   -10,    35),
    ("p2_olm_annual_precip",        1000,  "float32",     0,   200),
    # MERIT Hydro
    ("p2_merit_flow_dir",            90,   "int16",       1,    128),
    ("p2_merit_upstream_area",       90,   "float32",     0,  1e8),
    ("p2_merit_hydro_elev",          90,   "float32",  -100,  4000),
    ("p2_merit_hand",                90,   "float32",     0,  1000),
    ("p2_river_distance",            30,   "float32",     0,  1e6),
    # Global Flood DB
    ("p2_gfd_flood_count",          250,   "int16",       0,   200),
    ("p2_gfd_flood_duration",       250,   "int16",       0,  2000),
    ("p2_gfd_perm_water",           250,   "uint8",       0,     1),
    # Canopy + extras
    ("p2_meta_canopy_height",        10,   "float32",     0,    50),
    ("p2_road_distance",             30,   "float32",     0,  1e6),
    ("p2_alos_landforms",            30,   "uint8",       1,    44),
    # Thermal 2024
    ("p2_ndvi_summer_2024",          30,   "float32",  -0.5,     1),
    ("p2_ndwi_summer_2024",          30,   "float32",  -0.8,   0.8),
    ("p2_lst_summer_2024",          100,   "float32",    10,    70),
]

# Vector layers (just copy, no clip needed here)
VECTOR_LAYERS = ["p2_grip4_roads.geojson"]

# ── Resampling map ─────────────────────────────────────────────────────────
CATEGORICAL = {
    "p2_glad_lulc_2000","p2_glad_lulc_2020",
    "p2_worldcover_2021","p2_alos_landforms","p2_soil_texture",
    "p2_gfd_perm_water",
} | {f"p2_glcfcs_{y}" for y in [1985,1990,1995]+list(range(2000,2023))} \
  | {f"p2_corine_{y}" for y in [1990,2000,2006,2012,2018]}

# ── Helper: reproject + clip to havza ─────────────────────────────────────
def clip_to_havza(src_path: Path, dst_path: Path, stem: str) -> dict:
    """Reproject to EPSG:32635 and clip to havza boundary. Returns info dict.

    Pixels outside the polygon (within the bounding box) are set to the nodata
    sentinel AND that sentinel is explicitly declared in the output GeoTIFF
    metadata.  Without the declaration, viewers treat those pixels as valid data.
    """
    result = {"stem": stem, "status": "OK", "warnings": [], "pixels_valid": 0,
              "res_m": None, "crs": None, "nodata": None}

    resampling = Resampling.nearest if stem in CATEGORICAL else Resampling.bilinear

    with rasterio.open(src_path) as src:
        src_crs = src.crs.to_string()
        result["src_crs"] = src_crs

        # Determine the nodata value to use for this dtype
        fill_nodata = nodata_for_dtype(src.dtypes[0])

        # If already EPSG:32635 (GEE exports with CRS), just clip
        if src.crs.to_epsg() == 32635:
            out_img, out_transform = rio_mask(
                src, [havza_utm], crop=True, filled=True, nodata=fill_nodata
            )
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "crs": CRS_TARGET,
                "compress": "deflate",
                "nodata": fill_nodata,   # ← MUST declare so viewers mask correctly
            })
        else:
            # Reproject first, then clip
            dst_crs = CRS_TARGET
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            out_meta = src.meta.copy()
            out_meta.update({
                "crs": dst_crs, "transform": transform,
                "width": width, "height": height,
                "driver": "GTiff", "compress": "deflate",
                "nodata": fill_nodata,
            })
            import tempfile, os
            tmp = Path(tempfile.mktemp(suffix=".tif"))
            with rasterio.open(tmp, "w", **out_meta) as tmp_dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(tmp_dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=resampling,
                    )
            with rasterio.open(tmp) as reprojected:
                out_img, out_transform = rio_mask(
                    reprojected, [havza_utm], crop=True, filled=True, nodata=fill_nodata
                )
                out_meta = reprojected.meta.copy()
                out_meta.update({
                    "height": out_img.shape[1],
                    "width": out_img.shape[2],
                    "transform": out_transform,
                    "compress": "deflate",
                    "nodata": fill_nodata,   # ← declare here too
                })
            os.unlink(tmp)

    # Write clipped output
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **out_meta) as dst:
        dst.write(out_img)

    # Validate
    result["crs"] = CRS_TARGET
    result["res_m"] = round(abs(out_transform.a), 1)
    result["nodata"] = fill_nodata
    result["shape"] = (out_img.shape[1], out_img.shape[2])

    valid_mask = out_img[0] != fill_nodata

    valid_data = out_img[0][valid_mask]
    result["pixels_valid"] = int(valid_mask.sum())

    if result["pixels_valid"] == 0:
        result["status"] = "FAIL"
        result["warnings"].append("ALL PIXELS ARE NODATA — empty output!")
    elif result["pixels_valid"] < 1000:
        result["status"] = "WARN"
        result["warnings"].append(f"Very few valid pixels: {result['pixels_valid']}")

    if result["pixels_valid"] > 0:
        result["data_min"] = float(np.nanmin(valid_data))
        result["data_max"] = float(np.nanmax(valid_data))
    else:
        result["data_min"] = result["data_max"] = None

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 1 — Moving files to data/01_raw/paper2/gee_exports/")
print("="*70)

moved, already = 0, 0
for f in DOWNLOADS.iterdir():
    dst = RAW_GEE / f.name
    if dst.exists():
        already += 1
    else:
        shutil.copy2(f, dst)
        moved += 1

print(f"  Moved  : {moved} files")
print(f"  Already: {already} files")
print(f"  Total  : {moved + already} files in {RAW_GEE}")

# Vector: just copy to interim as-is
for vl in VECTOR_LAYERS:
    src = RAW_GEE / vl
    dst = INTERIM / vl
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print(f"  Vector copied: {vl}")

print("\n" + "="*70)
print("STEP 2 — Clipping all rasters to Kuzey Ege Havzasi")
print("="*70)

results = []
TOTAL = len(LAYERS)

for idx, (stem, expected_res, dtype_hint, vmin, vmax) in enumerate(LAYERS, 1):
    src_path = RAW_GEE / f"{stem}.tif"
    dst_path = INTERIM / f"{stem}_clipped.tif"

    prefix = f"  [{idx:>3}/{TOTAL}] {stem:<40}"

    if not src_path.exists():
        print(f"{prefix} -> MISSING SOURCE")
        results.append({"stem": stem, "status": "MISSING", "warnings": ["Source file not found"]})
        continue

    if dst_path.exists() and not FORCE_RECLIP:
        # Quick validate existing
        with rasterio.open(dst_path) as d:
            arr = d.read(1)
            nd = d.nodata
            valid = int((arr != nd).sum()) if nd is not None else int(arr.size)
            res_m = round(abs(d.transform.a), 1)
        print(f"{prefix} -> ALREADY EXISTS ({res_m}m, {valid:,} valid px)")
        results.append({"stem": stem, "status": "OK", "warnings": [], "pixels_valid": valid,
                        "res_m": res_m, "crs": CRS_TARGET})
        continue

    try:
        r = clip_to_havza(src_path, dst_path, stem)
        status_str = r["status"]
        valid_px = r.get("pixels_valid", 0)
        res = r.get("res_m", "?")
        dmin = r.get("data_min")
        dmax = r.get("data_max")
        range_str = f"[{dmin:.2f}, {dmax:.2f}]" if dmin is not None else "N/A"
        warn_str = " !! " + "; ".join(r["warnings"]) if r["warnings"] else ""
        print(f"{prefix} -> {status_str} | {res}m | {valid_px:,} px | range {range_str}{warn_str}")
        results.append(r)
    except Exception as e:
        print(f"{prefix} -> ERROR: {e}")
        results.append({"stem": stem, "status": "ERROR", "warnings": [str(e)]})

# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 3 — VALIDATION REPORT")
print("="*70)

ok    = [r for r in results if r["status"] == "OK"]
warn  = [r for r in results if r["status"] == "WARN"]
fail  = [r for r in results if r["status"] in ("FAIL","ERROR","MISSING")]

print(f"\n  OK OK      : {len(ok)}")
print(f"  !! WARN    : {len(warn)}")
print(f"  XX FAIL    : {len(fail)}")

if warn:
    print("\n--- WARNINGS ---")
    for r in warn:
        print(f"  {r['stem']}: {'; '.join(r['warnings'])}")

if fail:
    print("\n--- FAILURES (re-download needed) ---")
    for r in fail:
        print(f"  {r['stem']}: {r['status']} — {'; '.join(r['warnings'])}")

print("\n" + "="*70)
print("STEP 4 — OUTPUT INVENTORY")
print("="*70)

groups = {
    "Terrain (SRTM)": [r for r in results if "srtm" in r["stem"]],
    "LULC — GLAD":    [r for r in results if "glad" in r["stem"]],
    "LULC — GLC-FCS": [r for r in results if "glcfcs" in r["stem"]],
    "LULC — Other":   [r for r in results if any(x in r["stem"] for x in ["worldcover","corine"])],
    "Hydrology":      [r for r in results if any(x in r["stem"] for x in ["gsw","water","merit","river","gfd"])],
    "Population":     [r for r in results if "hrsl" in r["stem"]],
    "NTL":            [r for r in results if "viirs" in r["stem"]],
    "Soil/Climate":   [r for r in results if any(x in r["stem"] for x in ["soil","clay","worldclim","olm"])],
    "Thermal/Veg":    [r for r in results if any(x in r["stem"] for x in ["ndvi","ndwi","lst"])],
    "Other":          [r for r in results if any(x in r["stem"] for x in ["canopy","road","alos"])],
}

for group, items in groups.items():
    if not items:
        continue
    ok_cnt = sum(1 for r in items if r["status"] == "OK")
    print(f"\n  {group} ({ok_cnt}/{len(items)} OK)")
    for r in items:
        icon = "OK" if r["status"] == "OK" else ("!!" if r["status"] == "WARN" else "XX")
        res = r.get("res_m", "?")
        px  = r.get("pixels_valid", 0)
        print(f"    {icon} {r['stem']:<44} {res}m  {px:>12,} px")

print(f"\n  Clipped files : {INTERIM}")
print(f"  Raw GEE files : {RAW_GEE}")

# Write failure list for re-download
if fail:
    fail_log = LOG_DIR / "p2_redownload_needed.txt"
    with open(fail_log, "w") as f:
        f.write("# Paper 2 — layers that need re-download from GEE\n")
        for r in fail:
            f.write(f"{r['stem']}.tif\n")
    print(f"\n  Re-download list: {fail_log}")

print("\nDONE.")
