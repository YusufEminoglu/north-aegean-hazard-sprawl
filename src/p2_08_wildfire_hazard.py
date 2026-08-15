"""
Paper 2 — Enhanced Wildfire Hazard
====================================
Composites 3 sub-inputs into a single normalised wildfire hazard index.

Sub-inputs:
  1. GABAM burned frequency (count 1990-2021) — historical fire recurrence  higher = worse
  2. Summer EVI (Sentinel-2 SR JJA 2019-2024) — fuel biomass density        higher = worse
  3. Tree Canopy Height / TCH (Meta/WRI)       — fuel structure              higher = worse

Each min-max normalised to [0,1], then averaged over whichever sub-inputs
have data at each pixel. The composite receives its AHP weight in
p2_09_multi_hazard_fusion.py.

Output: data/03_processed/paper2/wildfire_hazard.tif  (float32, 0-1)
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM   = PROJECT_ROOT / "data" / "02_interim"  / "paper2"
PROCESSED = PROJECT_ROOT / "data" / "03_processed" / "paper2"
PROCESSED.mkdir(parents=True, exist_ok=True)

REF_PATH = INTERIM / "p2_srtm_elevation_clipped.tif"

INPUTS = {
    "gabam": INTERIM / "p2_gabam_burned_frequency.tif",
    "evi":   INTERIM / "p2_evi_summer_composite.tif",
    "tch":   INTERIM / "p2_meta_canopy_height_clipped.tif",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_aligned(path: Path, ref_meta: dict) -> np.ndarray | None:
    """Reproject a source raster to the reference grid, returning NaN at
    nodata pixels rather than letting bilinear resampling blend a numeric
    nodata sentinel (e.g. -9999) into neighbouring valid cells.

    BUG FIX: this previously called reproject() without src_nodata/dst_nodata,
    so canopy height's -9999 sentinel (present at ~60% of source pixels — a
    data gap, not "zero canopy") was blended by bilinear interpolation into a
    corrupted halo around every gap, and normalise()'s `arr >= 0` filter then
    silently mapped every affected pixel to composite-score 0 (the *lowest*
    possible fuel-structure score) instead of excluding it. This affected the
    majority of the study area and very plausibly explains why an earlier
    composite came out anti-correlated with independent FIRMS fire detections
    in validation.
    """
    if not path.exists():
        print(f"  [MISSING] {path.name}")
        return None
    dst = np.full((ref_meta["height"], ref_meta["width"]), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
            resampling=Resampling.bilinear,
            src_nodata=src.nodata, dst_nodata=np.nan,
        )
    return dst

def normalise(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0,1] over valid (non-nodata) pixels; returns NaN
    (not 0) at pixels with no source data, so a missing layer does not get
    silently scored as "lowest possible hazard" in the composite mean.
    """
    valid = arr[mask & ~np.isnan(arr)]
    if len(valid) == 0:
        return np.full_like(arr, np.nan)
    lo, hi = np.nanmin(valid), np.nanmax(valid)
    if hi == lo:
        return np.zeros_like(arr)
    out = np.full_like(arr, np.nan, dtype=np.float32)
    m = mask & ~np.isnan(arr)
    out[m] = (arr[m] - lo) / (hi - lo)
    return out

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("== Enhanced Wildfire Hazard ==")

    with rasterio.open(REF_PATH) as ref:
        ref_meta = ref.meta.copy()
        ref_arr  = ref.read(1)
        nodata   = ref.nodata

    valid = (ref_arr != nodata) if nodata is not None else np.ones(ref_arr.shape, bool)

    layers = {}
    for key, path in INPUTS.items():
        arr = load_aligned(path, ref_meta)
        if arr is None:
            arr = np.full(ref_arr.shape, np.nan, dtype=np.float32)
        layers[key] = normalise(arr, valid)
        finite = layers[key][valid]
        finite = finite[np.isfinite(finite)]
        cov = len(finite) / int(valid.sum()) * 100 if valid.sum() else 0.0
        rng = (finite.min(), finite.max()) if len(finite) else (float("nan"), float("nan"))
        print(f"  {key}: range [{rng[0]:.3f}, {rng[1]:.3f}]  coverage={cov:.1f}%")

    # NaN-aware mean: a pixel missing one input (e.g. canopy-height gaps) is
    # scored from whichever of the other layers have data, rather than the
    # missing layer being silently treated as 0.
    stack = np.stack(list(layers.values()), axis=0)
    with np.errstate(invalid="ignore"):
        wildfire = np.nanmean(stack, axis=0).astype(np.float32)
    all_missing = np.all(np.isnan(stack), axis=0)
    wildfire[~valid | all_missing] = -9999.0
    wildfire[np.isnan(wildfire)] = -9999.0

    out_meta = ref_meta.copy()
    out_meta.update(dtype=rasterio.float32, nodata=-9999.0)
    out_path = PROCESSED / "wildfire_hazard.tif"
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(wildfire, 1)

    v = wildfire[valid & (wildfire != -9999.0)]
    print(f"  Output range: [{v.min():.3f}, {v.max():.3f}]")
    print(f"  Saved: {out_path.name}")

if __name__ == "__main__":
    main()
