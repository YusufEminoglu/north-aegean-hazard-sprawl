"""
Paper 2 - Flood Hazard Surface (Model A & B Fusion)
=====================================================
Calculates a continuous 1-5 flood hazard score by fusing two WLC models
(weighted average of the two 1-5 factor scores; no z-standardisation).

Model A (Physical-Geomorphic):
  - Slope (30%)
  - Distance to river (30%)
  - Elevation (15%)
  - Soil clay content (10%)
  - Max precipitation (10%)
  - LULC permeability (5%)

Model B (Remote Sensing / Spectral):
  - Distance to permanent water (30%)
  - TPI / Topographic Position Index (30%)
  - Elevation (20%)
  - NDWI (15%)
  - NDVI (5%)

Fusion:
  Final Flood Hazard = 0.6 * Model A + 0.4 * Model B
"""

import rasterio
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM = PROJECT_ROOT / "data" / "02_interim" / "paper2"
PROCESSED = PROJECT_ROOT / "data" / "03_processed" / "paper2"

REF_RASTER = INTERIM / "p2_srtm_elevation_clipped.tif"

def reclass_slope(arr):
    # <2 (5), 2-5 (4), 5-10 (3), 10-20 (2), >20 (1)
    out = np.ones_like(arr)
    out[(arr >= 0) & (arr < 2)] = 5
    out[(arr >= 2) & (arr < 5)] = 4
    out[(arr >= 5) & (arr < 10)] = 3
    out[(arr >= 10) & (arr < 20)] = 2
    return out

def reclass_elevation(arr):
    # <10 (5), 10-30 (4), 30-100 (3), 100-300 (2), >300 (1)
    out = np.ones_like(arr)
    out[(arr >= -100) & (arr < 10)] = 5
    out[(arr >= 10) & (arr < 30)] = 4
    out[(arr >= 30) & (arr < 100)] = 3
    out[(arr >= 100) & (arr < 300)] = 2
    return out

def reclass_dist(arr):
    # <50 (5), 50-200 (4), 200-500 (3), 500-1000 (2), >1000 (1)
    out = np.ones_like(arr)
    out[(arr >= 0) & (arr < 50)] = 5
    out[(arr >= 50) & (arr < 200)] = 4
    out[(arr >= 200) & (arr < 500)] = 3
    out[(arr >= 500) & (arr < 1000)] = 2
    return out

def reclass_clay(arr):
    # >40% (5), 30-40 (4), 20-30 (3), 10-20 (2), <10 (1)
    # Note: soilgrids clay is 0-100 or 0-1000, let's assume 0-100 here.
    # Usually it's in % or g/kg. In our case it's mostly %.
    out = np.ones_like(arr)
    out[arr > 40] = 5
    out[(arr > 30) & (arr <= 40)] = 4
    out[(arr > 20) & (arr <= 30)] = 3
    out[(arr > 10) & (arr <= 20)] = 2
    return out

def reclass_tpi(arr):
    # <-5 (Valley) = 5, -5 to -1 = 4, -1 to 1 (Flat) = 3, 1 to 5 = 2, >5 (Ridge) = 1
    out = np.ones_like(arr)
    out[arr < -5] = 5
    out[(arr >= -5) & (arr < -1)] = 4
    out[(arr >= -1) & (arr <= 1)] = 3
    out[(arr > 1) & (arr <= 5)] = 2
    return out

def reclass_ndvi(arr):
    # <0.1 (5), 0.1-0.3 (4), 0.3-0.5 (3), 0.5-0.7 (2), >0.7 (1)
    out = np.ones_like(arr)
    out[arr < 0.1] = 5
    out[(arr >= 0.1) & (arr < 0.3)] = 4
    out[(arr >= 0.3) & (arr < 0.5)] = 3
    out[(arr >= 0.5) & (arr < 0.7)] = 2
    return out

def reclass_ndwi(arr):
    # >0.1 (wet) (5), 0 to 0.1 (4), -0.1 to 0 (3), -0.2 to -0.1 (2), <-0.2 (dry) (1)
    out = np.ones_like(arr)
    out[arr > 0.1] = 5
    out[(arr > 0) & (arr <= 0.1)] = 4
    out[(arr > -0.1) & (arr <= 0)] = 3
    out[(arr > -0.2) & (arr <= -0.1)] = 2
    return out

def read_resampled(path, ref_meta):
    """Read a raster and ensure it aligns with the reference."""
    if not path.exists():
        print(f"  [MISSING] {path.name}")
        return None
    out = np.zeros((ref_meta["height"], ref_meta["width"]), dtype=np.float32)
    with rasterio.open(path) as src:
        from rasterio.warp import reproject as warp_r, Resampling
        warp_r(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_meta["transform"],
            dst_crs=ref_meta["crs"],
            resampling=Resampling.bilinear,
        )
    return out

def run_flood_hazard():
    print("== Flood Hazard Processing (Model A & B Fusion) ==")
    
    with rasterio.open(REF_RASTER) as ref:
        ref_meta = ref.meta.copy()
        ref_arr = ref.read(1)
        nodata = ref.nodata
    
    valid_mask = (ref_arr != nodata) if nodata is not None else np.ones(ref_arr.shape, bool)
    
    # Load inputs
    print("  Loading rasters...")
    slope = read_resampled(INTERIM / "p2_srtm_slope_clipped.tif", ref_meta)
    elev = read_resampled(INTERIM / "p2_srtm_elevation_clipped.tif", ref_meta)
    riv_d = read_resampled(INTERIM / "p2_river_distance_clipped.tif", ref_meta)
    wat_d = read_resampled(INTERIM / "p2_water_distance_clipped.tif", ref_meta)
    clay = read_resampled(INTERIM / "p2_soilgrids_clay_clipped.tif", ref_meta)
    tpi = read_resampled(INTERIM / "p2_srtm_tpi_clipped.tif", ref_meta)
    ndvi = read_resampled(INTERIM / "p2_ndvi_summer_2024_clipped.tif", ref_meta)
    ndwi = read_resampled(INTERIM / "p2_ndwi_summer_2024_clipped.tif", ref_meta)
    precip = read_resampled(INTERIM / "p2_worldclim_bio13_maxprecip_clipped.tif", ref_meta)
    
    # Optional LULC 2020 for permeability
    lulc_path = PROCESSED / "lulc_simplified_2020.tif"
    if lulc_path.exists():
        with rasterio.open(lulc_path) as src:
            lulc = src.read(1)
        # Urban(1), Agriculture(2), Forest(3), Water(4), Bare(5)
        perm = np.ones_like(lulc, dtype=np.float32)
        perm[lulc == 4] = 5 # Water -> 5
        perm[lulc == 1] = 4 # Urban -> 4
        perm[lulc == 5] = 3 # Bare -> 3
        perm[lulc == 2] = 2 # Ag -> 2
        perm[lulc == 3] = 1 # Forest -> 1
    else:
        perm = np.full(ref_arr.shape, 3.0, dtype=np.float32)

    # Precipitation dynamic reclass (quintiles)
    precip_score = np.ones_like(precip)
    if precip is not None:
        p_valid = precip[valid_mask & (precip > 0)]
        if len(p_valid) > 0:
            q = np.percentile(p_valid, [20, 40, 60, 80])
            precip_score[precip > q[3]] = 5
            precip_score[(precip <= q[3]) & (precip > q[2])] = 4
            precip_score[(precip <= q[2]) & (precip > q[1])] = 3
            precip_score[(precip <= q[1]) & (precip > q[0])] = 2

    # Model A: Geomorphic (six factors; HAND is not an input of the surface used in the paper)
    print("  Computing Model A (Geomorphic)...")
    mod_A = (
        reclass_slope(slope) * 0.30 +
        reclass_dist(riv_d) * 0.30 +
        reclass_elevation(elev) * 0.15 +
        reclass_clay(clay) * 0.10 +
        precip_score * 0.10 +
        perm * 0.05
    )

    # Model B: Spectral
    print("  Computing Model B (Spectral)...")
    mod_B = (
        reclass_dist(wat_d) * 0.30 +
        reclass_tpi(tpi) * 0.30 +
        reclass_elevation(elev) * 0.20 +
        reclass_ndwi(ndwi) * 0.15 +
        reclass_ndvi(ndvi) * 0.05
    )

    # Fusion
    print("  Fusing A and B...")
    flood = mod_A * 0.60 + mod_B * 0.40
    flood[~valid_mask] = -9999.0

    out_path = PROCESSED / "flood_hazard.tif"
    ref_meta.update(dtype=rasterio.float32, nodata=-9999.0)
    with rasterio.open(out_path, "w", **ref_meta) as out:
        out.write(flood.astype(np.float32), 1)

    v_flood = flood[valid_mask]
    print(f"  Range: [{v_flood.min():.2f}, {v_flood.max():.2f}]")
    print(f"  Output saved to: {out_path.name}")
    print("  Done.")

if __name__ == "__main__":
    run_flood_hazard()
