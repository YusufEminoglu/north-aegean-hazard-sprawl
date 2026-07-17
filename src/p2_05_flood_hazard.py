"""Build the continuous two-model flood-susceptibility surface.

Model A combines HAND, slope, river distance, elevation, clay, maximum
precipitation, and LULC permeability. Model B combines water distance,
elevation, TPI, NDVI, and NDWI. Both WLC outputs are standardised before their
0.60/0.40 fusion, matching the revised manuscript.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "02_interim" / "paper2"
PROCESSED = PROJECT_ROOT / "data" / "03_processed" / "paper2"
REF_RASTER = INTERIM / "p2_srtm_elevation_clipped.tif"

MODEL_A_WEIGHTS = {
    "hand": 0.216,
    "slope": 0.209,
    "river_distance": 0.209,
    "elevation": 0.105,
    "clay": 0.105,
    "precipitation": 0.105,
    "permeability": 0.051,
}
MODEL_B_WEIGHTS = {
    "water_distance": 0.30,
    "elevation": 0.25,
    "tpi": 0.20,
    "ndvi": 0.15,
    "ndwi": 0.10,
}


def reclass_slope(arr: np.ndarray) -> np.ndarray:
    out = np.ones_like(arr, dtype=np.float32)
    out[(arr >= 0) & (arr < 2)] = 5
    out[(arr >= 2) & (arr < 5)] = 4
    out[(arr >= 5) & (arr < 10)] = 3
    out[(arr >= 10) & (arr < 20)] = 2
    return out


def reclass_elevation(arr: np.ndarray) -> np.ndarray:
    out = np.ones_like(arr, dtype=np.float32)
    out[(arr >= -100) & (arr < 10)] = 5
    out[(arr >= 10) & (arr < 30)] = 4
    out[(arr >= 30) & (arr < 100)] = 3
    out[(arr >= 100) & (arr < 300)] = 2
    return out


def reclass_distance(arr: np.ndarray) -> np.ndarray:
    out = np.ones_like(arr, dtype=np.float32)
    out[(arr >= 0) & (arr < 50)] = 5
    out[(arr >= 50) & (arr < 200)] = 4
    out[(arr >= 200) & (arr < 500)] = 3
    out[(arr >= 500) & (arr < 1000)] = 2
    return out


def reclass_clay(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    clay = arr.copy()
    values = clay[valid & np.isfinite(clay)]
    if values.size and np.nanpercentile(values, 99) > 100:
        clay /= 10.0
    out = np.ones_like(clay, dtype=np.float32)
    out[clay > 40] = 5
    out[(clay > 30) & (clay <= 40)] = 4
    out[(clay > 20) & (clay <= 30)] = 3
    out[(clay > 10) & (clay <= 20)] = 2
    return out


def reclass_tpi(arr: np.ndarray) -> np.ndarray:
    out = np.ones_like(arr, dtype=np.float32)
    out[arr < -5] = 5
    out[(arr >= -5) & (arr < -1)] = 4
    out[(arr >= -1) & (arr <= 1)] = 3
    out[(arr > 1) & (arr <= 5)] = 2
    return out


def reclass_ndvi(arr: np.ndarray) -> np.ndarray:
    out = np.ones_like(arr, dtype=np.float32)
    out[arr < 0.1] = 5
    out[(arr >= 0.1) & (arr < 0.3)] = 4
    out[(arr >= 0.3) & (arr < 0.5)] = 3
    out[(arr >= 0.5) & (arr < 0.7)] = 2
    return out


def reclass_ndwi(arr: np.ndarray) -> np.ndarray:
    out = np.ones_like(arr, dtype=np.float32)
    out[arr > 0.1] = 5
    out[(arr > 0) & (arr <= 0.1)] = 4
    out[(arr > -0.1) & (arr <= 0)] = 3
    out[(arr > -0.2) & (arr <= -0.1)] = 2
    return out


def quantile_score(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = arr[valid & np.isfinite(arr)]
    if not values.size:
        raise ValueError("Cannot score an empty precipitation surface.")
    breaks = np.percentile(values, [20, 40, 60, 80])
    return (np.digitize(arr, breaks, right=True) + 1).astype(np.float32)


def hand_score(arr: np.ndarray) -> np.ndarray:
    out = np.ones_like(arr, dtype=np.float32)
    out[(arr >= 0) & (arr < 2)] = 5
    out[(arr >= 2) & (arr < 5)] = 4
    out[(arr >= 5) & (arr < 10)] = 3
    out[(arr >= 10) & (arr < 20)] = 2
    return out


def read_aligned(path: Path, ref_meta: dict) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required flood input is missing: {path}")
    destination = np.full(
        (ref_meta["height"], ref_meta["width"]), np.nan, dtype=np.float32
    )
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=ref_meta["transform"],
            dst_crs=ref_meta["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return destination


def standardize(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = arr[valid & np.isfinite(arr)]
    if not values.size or np.isclose(values.std(), 0):
        raise ValueError("Cannot standardise an empty or constant flood component.")
    out = np.full_like(arr, np.nan, dtype=np.float32)
    out[valid] = (arr[valid] - values.mean()) / values.std()
    return out


def write_float(path: Path, arr: np.ndarray, meta: dict, valid: np.ndarray) -> None:
    output = arr.astype(np.float32, copy=True)
    output[~valid] = -9999.0
    out_meta = meta.copy()
    out_meta.update(dtype=rasterio.float32, count=1, nodata=-9999.0)
    with rasterio.open(path, "w", **out_meta) as target:
        target.write(output, 1)


def run_flood_hazard() -> None:
    print("== Flood hazard: standardised Model A/B fusion ==")
    PROCESSED.mkdir(parents=True, exist_ok=True)

    with rasterio.open(REF_RASTER) as reference:
        ref_meta = reference.meta.copy()
        ref_arr = reference.read(1).astype(np.float32)
        nodata = reference.nodata
    valid = np.isfinite(ref_arr)
    if nodata is not None:
        valid &= ref_arr != nodata

    inputs = {
        "slope": read_aligned(INTERIM / "p2_srtm_slope_clipped.tif", ref_meta),
        "elevation": read_aligned(INTERIM / "p2_srtm_elevation_clipped.tif", ref_meta),
        "river_distance": read_aligned(INTERIM / "p2_river_distance_clipped.tif", ref_meta),
        "water_distance": read_aligned(INTERIM / "p2_water_distance_clipped.tif", ref_meta),
        "clay": read_aligned(INTERIM / "p2_soilgrids_clay_clipped.tif", ref_meta),
        "tpi": read_aligned(INTERIM / "p2_srtm_tpi_clipped.tif", ref_meta),
        "ndvi": read_aligned(INTERIM / "p2_ndvi_summer_2024_clipped.tif", ref_meta),
        "ndwi": read_aligned(INTERIM / "p2_ndwi_summer_2024_clipped.tif", ref_meta),
        "precipitation": read_aligned(
            INTERIM / "p2_worldclim_bio13_maxprecip_clipped.tif", ref_meta
        ),
        "hand": read_aligned(INTERIM / "p2_merit_hand_clipped.tif", ref_meta),
    }
    for arr in inputs.values():
        valid &= np.isfinite(arr)

    lulc_path = PROCESSED / "lulc_simplified_2020.tif"
    if not lulc_path.exists():
        raise FileNotFoundError(f"Required permeability layer is missing: {lulc_path}")
    with rasterio.open(lulc_path) as source:
        lulc = source.read(1)
    if lulc.shape != ref_arr.shape:
        raise ValueError("LULC permeability raster is not aligned to the reference grid.")

    permeability = np.ones_like(lulc, dtype=np.float32)
    permeability[lulc == 4] = 5
    permeability[lulc == 1] = 4
    permeability[lulc == 5] = 3
    permeability[lulc == 2] = 2
    permeability[lulc == 3] = 1

    scores_a = {
        "hand": hand_score(inputs["hand"]),
        "slope": reclass_slope(inputs["slope"]),
        "river_distance": reclass_distance(inputs["river_distance"]),
        "elevation": reclass_elevation(inputs["elevation"]),
        "clay": reclass_clay(inputs["clay"], valid),
        "precipitation": quantile_score(inputs["precipitation"], valid),
        "permeability": permeability,
    }
    model_a = sum(scores_a[key] * weight for key, weight in MODEL_A_WEIGHTS.items())

    scores_b = {
        "water_distance": reclass_distance(inputs["water_distance"]),
        "elevation": reclass_elevation(inputs["elevation"]),
        "tpi": reclass_tpi(inputs["tpi"]),
        "ndvi": reclass_ndvi(inputs["ndvi"]),
        "ndwi": reclass_ndwi(inputs["ndwi"]),
    }
    model_b = sum(scores_b[key] * weight for key, weight in MODEL_B_WEIGHTS.items())

    model_a_z = standardize(model_a, valid)
    model_b_z = standardize(model_b, valid)
    flood = 0.60 * model_a_z + 0.40 * model_b_z

    write_float(PROCESSED / "flood_model_a_standardized.tif", model_a_z, ref_meta, valid)
    write_float(PROCESSED / "flood_model_b_standardized.tif", model_b_z, ref_meta, valid)
    write_float(PROCESSED / "flood_hazard.tif", flood, ref_meta, valid)

    values = flood[valid]
    print(f"  valid pixels: {values.size:,}")
    print(f"  fused range: [{values.min():.3f}, {values.max():.3f}]")
    print("  saved: flood_model_a_standardized.tif")
    print("  saved: flood_model_b_standardized.tif")
    print("  saved: flood_hazard.tif")


if __name__ == "__main__":
    run_flood_hazard()
