"""
Paper 2 — GEE Export: 2020-vintage Summer NDVI (RF driver)
==========================================================
Exports a single raster:
  p2_ndvi_summer_2020  : summer NDVI median, Landsat 8 Jun-Aug 2018-2020   30 m

WHY THIS EXISTS
---------------
The RF urban-suitability model (p2_04_flus_ca_engine.py) is calibrated on the
2020 land configuration. The previous NDVI driver was a 2024 Landsat composite,
which POSTDATES the 2020 reference state and leaks post-conversion landscape
information into the predictor set. This script produces a 2018-2020 NDVI
composite so that no RF predictor postdates 2020.

A 3-year JJA window (2018-2020) is used (not a single year) to suppress
cloud/scene noise while keeping every observation at or before the 2020
reference state. Landsat 8 only (no L9, which launched late 2021; no L7 to
avoid SLC-off striping).

Study area : EE_ROI_ASSET or local GeoJSON
CRS        : EPSG:32635
Folder     : GEE_DRIVE_FOLDER

Usage:
  earthengine authenticate
  python src/p2_00d_ndvi_2020_download.py

After the Drive export finishes, place/clip the file as:
  data/02_interim/paper2/p2_ndvi_summer_2020_clipped.tif
(via the existing p2_01_move_clip_validate.py step), then rerun
p2_03_spatial_drivers.py -> p2_04_flus_ca_engine.py.
"""

import ee
from _gee_config import drive_folder, initialize_ee, load_roi

initialize_ee()

# ── Config ────────────────────────────────────────────────────────────────────
ROI    = load_roi()
CRS    = 'EPSG:32635'
FOLDER = drive_folder()
MAX_PX = 1e13


def submit(image, name, scale):
    task = ee.batch.Export.image.toDrive(
        image          = image.toFloat(),
        description    = name,
        folder         = FOLDER,
        fileNamePrefix = name,
        region         = ROI,
        scale          = scale,
        crs            = CRS,
        maxPixels      = MAX_PX,
    )
    task.start()
    print(f'  Submitted: {name}  ({scale} m)')


# ── Landsat 8 cloud / shadow mask (Collection 2 L2 QA_PIXEL) ──────────────────
def _mask_landsat(image):
    qa   = image.select('QA_PIXEL')
    mask = (qa.bitwiseAnd(1 << 3).eq(0)            # cloud
              .And(qa.bitwiseAnd(1 << 4).eq(0)))    # cloud shadow
    return image.updateMask(mask)


def _scale_optical(image):
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    return image.addBands(optical, None, True)


# ── Landsat 8 summer (JJA) composite, 2018-2020 ───────────────────────────────
landsat_2020 = (
    ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(ROI)
    .filter(ee.Filter.calendarRange(6, 8, 'month'))
    .filterDate('2018-01-01', '2020-12-31')
    .map(_mask_landsat)
    .map(_scale_optical)
    .median()
    .clip(ROI)
)

# NDVI = (NIR - RED) / (NIR + RED) = (SR_B5 - SR_B4) / (SR_B5 + SR_B4)
ndvi_2020 = landsat_2020.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

# ── Export ────────────────────────────────────────────────────────────────────
print(f'Submitting 1 export task to Google Drive -> {FOLDER}/')
submit(ndvi_2020, 'p2_ndvi_summer_2020', 30)   # Landsat optical native = 30 m
print('Done. Monitor at code.earthengine.google.com -> Tasks tab.')
