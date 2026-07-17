"""
Paper 2 — Bio-Climatic Stress Hazard
=====================================
Composites 4 sub-inputs into a single normalised bio-climatic stress index.

Sub-inputs:
  1. Summer LST (°C)           — thermal exposure          higher = worse
  2. Summer UHI z-score        — urban heat amplification  higher = worse
  3. Tropospheric NO2 (mol/m²) — air quality health risk   higher = worse
  4. Summer precip JJA (mm)    — drought stress amplifier  LOWER = worse (inverted)

Each is min-max normalised to [0,1]; precip is inverted before averaging.
Equal weights (0.25 each) applied to avoid unverified AHP bias at sub-input level;
the composite itself receives its AHP weight in p2_09_multi_hazard_fusion.py.

Output: data/03_processed/paper2/bioclimate_hazard.tif  (float32, 0-1)
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM   = PROJECT_ROOT / "data" / "02_interim"  / "paper2"
PROCESSED = PROJECT_ROOT / "data" / "03_processed" / "paper2"
PROCESSED.mkdir(parents=True, exist_ok=True)

REF_PATH  = INTERIM / "p2_srtm_elevation_clipped.tif"

INPUTS = {
    "lst":    INTERIM / "p2_lst_summer_composite.tif",
    "uhi":    INTERIM / "p2_uhi_summer_composite.tif",
    "no2":    INTERIM / "p2_s5p_no2_summer.tif",
    "precip": INTERIM / "p2_summer_precip_jja.tif",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_aligned(path: Path, ref_meta: dict) -> np.ndarray | None:
    if not path.exists():
        print(f"  [MISSING] {path.name}")
        return None
    dst = np.zeros((ref_meta["height"], ref_meta["width"]), dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
            resampling=Resampling.bilinear,
        )
    return dst

def normalise(arr: np.ndarray, mask: np.ndarray, invert: bool = False) -> np.ndarray:
    valid = arr[mask & ~np.isnan(arr)]
    if len(valid) == 0:
        return np.zeros_like(arr)
    lo, hi = np.nanmin(valid), np.nanmax(valid)
    if hi == lo:
        return np.zeros_like(arr)
    out = np.zeros_like(arr, dtype=np.float32)
    m = mask & ~np.isnan(arr)
    out[m] = (arr[m] - lo) / (hi - lo)
    if invert:
        out[m] = 1.0 - out[m]
    return out

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("== Bio-Climatic Stress Hazard ==")

    with rasterio.open(REF_PATH) as ref:
        ref_meta = ref.meta.copy()
        ref_arr  = ref.read(1)
        nodata   = ref.nodata

    valid = (ref_arr != nodata) if nodata is not None else np.ones(ref_arr.shape, bool)

    layers = {}
    for key, path in INPUTS.items():
        arr = load_aligned(path, ref_meta)
        if arr is None:
            arr = np.zeros(ref_arr.shape, dtype=np.float32)
        invert = (key == "precip")
        layers[key] = normalise(arr, valid, invert=invert)
        print(f"  {key}: range [{layers[key][valid].min():.3f}, {layers[key][valid].max():.3f}]  invert={invert}")

    bioclimate = np.mean(list(layers.values()), axis=0).astype(np.float32)
    bioclimate[~valid] = -9999.0

    out_meta = ref_meta.copy()
    out_meta.update(dtype=rasterio.float32, nodata=-9999.0)
    out_path = PROCESSED / "bioclimate_hazard.tif"
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(bioclimate, 1)

    v = bioclimate[valid & (bioclimate != -9999.0)]
    print(f"  Output range: [{v.min():.3f}, {v.max():.3f}]")
    print(f"  Saved: {out_path.name}")

if __name__ == "__main__":
    main()
