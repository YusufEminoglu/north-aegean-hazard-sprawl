"""
Paper 2 - Spatial Drivers (12 Drivers)
========================================
Aligns ALL 12 spatial drivers promised in the manuscript to a common
30 m EPSG:32635 grid covering the Kuzey Ege Havzasi.

Driver inventory (manuscript Table T3):
  1.  slope          - SRTM slope (degrees)
  2.  elevation      - SRTM elevation (m)
  3.  road_dist      - Distance to primary/secondary roads (GRIP4)
  4.  river_dist     - Distance to rivers/streams (HydroSHEDS / MERIT)
  5.  water_dist     - Distance to permanent water (JRC GSW)
  6.  pop_density    - Population density (Meta HRSL general)
  7.  ntl            - Nighttime light intensity (VIIRS 2020)
  8.  ndvi           - Summer NDVI (Landsat 8/9 2024)
  9.  clay           - Soil clay content (SoilGrids)
 10.  coast_dist      - Distance to coastline (derived from DEM sea-level mask)
 11.  urban_dist      - Distance to urban centres (derived from 2020 LULC)
 12.  canopy          - Canopy height (Meta 2023)

Drivers 10 and 11 are computed here because they are not available as
direct GEE exports - they are derived from existing interim rasters.
"""

import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt
from pathlib import Path

# Force ASCII-safe output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "02_interim" / "paper2"
PROCESSED = PROJECT_ROOT / "data" / "03_processed" / "paper2"
DRIVERS_OUT = PROCESSED / "drivers"
TABLES = PROJECT_ROOT / "outputs" / "tables" / "paper2"

DRIVERS_OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# Reference raster (30 m EPSG:32635)
REF_RASTER = INTERIM / "p2_srtm_elevation_clipped.tif"

# Direct GEE-derived drivers (just reproject/align)
DIRECT_DRIVERS = [
    ("slope",       "p2_srtm_slope_clipped.tif"),
    ("elevation",   "p2_srtm_elevation_clipped.tif"),
    ("road_dist",   "p2_road_distance_clipped.tif"),
    ("river_dist",  "p2_river_distance_clipped.tif"),
    ("water_dist",  "p2_water_distance_clipped.tif"),
    ("pop_density", "p2_hrsl_pop_general_clipped.tif"),
    ("ntl",         "p2_viirs_ntl_2020_clipped.tif"),
    # NDVI uses a 2018-2020 Landsat summer composite (NOT the 2024 composite) so
    # that no RF predictor postdates the 2020 reference state. Run
    # p2_00d_ndvi_2020_download.py to export this raster from GEE.
    ("ndvi",        "p2_ndvi_summer_2020_clipped.tif"),
    ("clay",        "p2_soilgrids_clay_clipped.tif"),
    # canopy retained for wildfire hazard only; dropped from the RF driver set.
    ("canopy",      "p2_meta_canopy_height_clipped.tif"),
]


def get_ref_meta():
    """Load reference raster metadata for alignment."""
    with rasterio.open(REF_RASTER) as ref:
        return ref.meta.copy(), ref.transform, (ref.height, ref.width)


def align_raster(src_path: Path, dst_path: Path, ref_meta: dict):
    """Reproject and align a raster to the reference grid."""
    with rasterio.open(src_path) as src:
        out_meta = ref_meta.copy()
        out_meta.update(dtype=rasterio.float32, nodata=-9999.0)

        with rasterio.open(dst_path, "w", **out_meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_meta["transform"],
                dst_crs=ref_meta["crs"],
                resampling=Resampling.bilinear,
            )


def compute_coast_dist(ref_meta: dict, shape: tuple) -> np.ndarray:
    """
    Derive distance-to-coastline from the SRTM elevation raster.
    Pixels with elevation <= 5 m are treated as potential coastal/sea cells.
    The Euclidean distance transform gives metre distance from nearest low-elevation cell.

    NOTE: This is an approximation. For publication replace with actual coastline vector
    (e.g. GSHHG) if available.
    """
    elev_path = INTERIM / "p2_srtm_elevation_clipped.tif"
    with rasterio.open(elev_path) as src:
        elev = src.read(1).astype(np.float32)
        nodata = src.nodata

    sea_mask = (elev <= 5.0)
    if nodata is not None:
        sea_mask |= (elev == nodata)

    pixel_m = abs(ref_meta["transform"].a)
    dist_pixels = distance_transform_edt(~sea_mask)
    dist_m = dist_pixels * pixel_m

    return dist_m.astype(np.float32)


def compute_urban_dist(ref_meta: dict, shape: tuple, year: int = 2020) -> np.ndarray:
    """
    Derive distance-to-nearest-urban-centre from the simplified LULC raster of
    the given year. Urban class = 1 (see p2_02_lulc_timeseries.py
    reclassification). year=2020 falls back to GLAD 2020 if the simplified
    LULC is not yet produced; year=2000 requires lulc_simplified_2000.tif.

    A 2000-vintage version of this driver exists so the CA hindcast
    (p2_10_model_validation.py, 2000->2020) does not condition on urban_dist
    computed from the 2020 endpoint it is trying to predict: reusing the 2020
    urban_dist there is a temporal-leakage channel, since distance-to-2020-
    urban already encodes where the 2020 urban edge sits, which trivially
    correlates with 2000-2020 growth adjacent to that edge.
    """
    simplified_path = PROCESSED / f"lulc_simplified_{year}.tif"
    fallback_path   = INTERIM / f"p2_glad_lulc_{year}_clipped.tif"

    if simplified_path.exists():
        lulc_path = simplified_path
        urban_class = 1
    elif year == 2020 and fallback_path.exists():
        lulc_path = fallback_path
        urban_class = 7  # GLAD GLCLU built-up approximate class
        print(f"  [urban_dist_{year}] Using GLAD GLCLU {year} fallback")
    else:
        print(f"  [urban_dist_{year}] No LULC available - setting uniform 50 km")
        return np.full(shape, 50000.0, dtype=np.float32)

    with rasterio.open(lulc_path) as src:
        lulc = src.read(1)

    urban_mask = (lulc == urban_class)
    pixel_m = abs(ref_meta["transform"].a)

    dist_pixels = distance_transform_edt(~urban_mask)
    dist_m = dist_pixels * pixel_m

    # Ensure shape matches reference grid
    if dist_m.shape != shape:
        from scipy.ndimage import zoom
        zoom_factors = (shape[0] / dist_m.shape[0], shape[1] / dist_m.shape[1])
        dist_m = zoom(dist_m, zoom_factors, order=0)

    return dist_m.astype(np.float32)


def save_derived_driver(arr: np.ndarray, name: str, ref_meta: dict):
    """Write a derived driver array to the drivers output folder, masking to study area."""
    # Create mask from reference raster to ensure perfect clipping
    with rasterio.open(REF_RASTER) as ref:
        ref_arr = ref.read(1)
        nodata_val = ref.nodata
        if nodata_val is not None:
            valid_mask = (ref_arr != nodata_val) & ~np.isnan(ref_arr)
        else:
            valid_mask = ~np.isnan(ref_arr)
            
    # Apply mask
    arr_masked = arr.copy()
    arr_masked[~valid_mask] = -9999.0
    
    dst_path = DRIVERS_OUT / f"driver_{name}.tif"
    out_meta = ref_meta.copy()
    out_meta.update(dtype=rasterio.float32, nodata=-9999.0, count=1)
    with rasterio.open(dst_path, "w", **out_meta) as dst:
        dst.write(arr_masked, 1)
    return dst_path


def align_direct_drivers(ref_meta: dict):
    """Align all GEE-derived drivers to the reference grid."""
    print("\n-- Aligning GEE-derived drivers ------------------------------------------")
    for name, filename in DIRECT_DRIVERS:
        src_path = INTERIM / filename
        dst_path = DRIVERS_OUT / f"driver_{name}.tif"

        if not src_path.exists():
            print(f"  [MISSING] {filename}  -> skipping driver '{name}'")
            continue

        align_raster(src_path, dst_path, ref_meta)
        print(f"  OK  {name:<20} <- {filename}")


def build_derived_drivers(ref_meta: dict, shape: tuple):
    """Compute and save derived drivers (coast_dist, urban_dist)."""
    print("\n-- Computing derived drivers ---------------------------------------------")

    # Driver 10 - coast_dist
    print("  Computing coast_dist ...")
    coast_arr = compute_coast_dist(ref_meta, shape)
    p = save_derived_driver(coast_arr, "coast_dist", ref_meta)
    print(f"  OK  coast_dist -> {p.name}  range=[{coast_arr.min():.0f}, {coast_arr.max():.0f}] m")

    # Driver 11 - urban_dist
    print("  Computing urban_dist ...")
    urban_arr = compute_urban_dist(ref_meta, shape, year=2020)
    p = save_derived_driver(urban_arr, "urban_dist", ref_meta)
    print(f"  OK  urban_dist -> {p.name}  range=[{urban_arr.min():.0f}, {urban_arr.max():.0f}] m")

    # Driver 11, 2000-vintage - used only by the leakage-free CA hindcast
    # validation (p2_10_model_validation.py), never by the production 2050
    # suitability model.
    print("  Computing urban_dist_2000 (hindcast-only) ...")
    urban_arr_2000 = compute_urban_dist(ref_meta, shape, year=2000)
    p = save_derived_driver(urban_arr_2000, "urban_dist_2000", ref_meta)
    print(f"  OK  urban_dist_2000 -> {p.name}  range=[{urban_arr_2000.min():.0f}, {urban_arr_2000.max():.0f}] m")


def verify_drivers():
    """Print a summary of all 12 expected drivers."""
    print("\n-- Driver inventory ------------------------------------------------------")
    expected = [
        "slope", "elevation", "road_dist", "river_dist", "water_dist",
        "pop_density", "ntl", "ndvi", "clay", "canopy",
        "coast_dist", "urban_dist",
    ]
    ok, missing = 0, 0
    for name in expected:
        path = DRIVERS_OUT / f"driver_{name}.tif"
        if path.exists():
            with rasterio.open(path) as src:
                arr = src.read(1)
                nd  = src.nodata
                valid = int((arr != nd).sum()) if nd is not None else arr.size
            print(f"  OK  {name:<20}  {valid:>12,} valid pixels")
            ok += 1
        else:
            print(f"  XX  {name:<20}  MISSING")
            missing += 1
    print(f"\n  {ok}/12 drivers ready, {missing} missing.")
    return ok, missing


if __name__ == "__main__":
    if not REF_RASTER.exists():
        print(f"ERROR: Reference raster not found: {REF_RASTER}")
        print("       Run p2_01_move_clip_validate.py first.")
        raise SystemExit(1)

    ref_meta, transform, shape = get_ref_meta()
    ref_meta.update(driver="GTiff", compress="deflate")

    align_direct_drivers(ref_meta)
    build_derived_drivers(ref_meta, shape)

    ok, missing = verify_drivers()
    if missing == 0:
        print("\nAll 12 drivers aligned and ready for p2_04_flus_ca_engine.py")
    else:
        print(f"\nWARNING: {missing} driver(s) missing - check above for details.")
