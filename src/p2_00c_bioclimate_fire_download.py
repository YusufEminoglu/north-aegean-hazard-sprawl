"""
Paper 2 — GEE Export: Bio-Climate & Enhanced Wildfire Inputs
=============================================================
Exports 6 rasters for the revised hazard framework.

Bio-Climate Hazard (4 inputs):
  p2_lst_summer_composite  : summer LST median (°C), Landsat 8/9 Jun-Aug 2014-2024  100 m
  p2_uhi_summer_composite  : UHI z-score = (LST - spatial_mean) / std              100 m
  p2_s5p_no2_summer        : tropospheric NO2 mean, S5P OFFL Jun-Aug 2019-2024     1000 m
  p2_summer_precip_jja     : JJA mean precipitation (mm/month), OpenLandMap        1000 m
                             NOTE: lower precip = higher drought stress —
                             INVERT during normalization in processing step.

Wildfire Hazard (2 new inputs; TCH already on disk):
  p2_evi_summer_composite  : EVI median, Sentinel-2 SR Jun-Aug 2019-2024           100 m
                             S2 used (not Landsat) for better 10 m optical quality
                             and SCL-based cloud masking.
  p2_gabam_burned_frequency: cumulative burned count 1990-2021 (GABAM)             100 m

Already on disk — NOT re-downloaded:
  p2_meta_canopy_height_clipped.tif  (Tree Canopy Height, Meta/WRI)

LST note: composite extended to 2014-2024 (vs 2019-2024 previously) for a more
robust thermal baseline — more scenes reduce noise from anomalous single years.
TIRS thermal band native resolution is 100 m; exporting at 100 m is appropriate.

Study area : EE_ROI_ASSET or local GeoJSON
CRS        : EPSG:32635
Folder     : GEE_DRIVE_FOLDER

Usage:
  earthengine authenticate
  python src/p2_00c_bioclimate_fire_download.py
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

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — Landsat 8/9 Summer Composite 2014-2024 (Jun-Aug)
# Extended period for a more robust thermal baseline (more scenes, less noise)
# Yields: LST (°C), UHI (z-score)
# ═══════════════════════════════════════════════════════════════════════════════

def _mask_landsat(image):
    qa   = image.select('QA_PIXEL')
    mask = (qa.bitwiseAnd(1 << 3).eq(0)            # cloud
              .And(qa.bitwiseAnd(1 << 4).eq(0)))    # cloud shadow
    return image.updateMask(mask)

def _scale_landsat(image):
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal = image.select('ST_B10').multiply(0.00341802).add(149.0)   # → Kelvin
    return image.addBands(optical, None, True).addBands(thermal, None, True)

landsat_composite = (
    ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
    .filterBounds(ROI)
    .filter(ee.Filter.calendarRange(6, 8, 'month'))
    .filterDate('2014-01-01', '2024-12-31')
    .map(_mask_landsat)
    .map(_scale_landsat)
    .median()
    .clip(ROI)
)

# ── LST (°C) — emissivity-corrected Planck inversion ─────────────────────────
# Artis & Carnahan (1982), OLI-TIRS Collection 2 adaptation
# Safe NDVI min/max with fallback (ee.Algorithms.If) to avoid null errors

ndvi   = landsat_composite.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
nstats = ndvi.reduceRegion(ee.Reducer.minMax(), ROI, scale=300, maxPixels=1e9)

ndvi_min = ee.Number(ee.Algorithms.If(nstats.get('NDVI_min'), nstats.get('NDVI_min'), 0))
ndvi_max = ee.Number(ee.Algorithms.If(nstats.get('NDVI_max'), nstats.get('NDVI_max'), 1))

fv  = ndvi.subtract(ndvi_min).divide(ndvi_max.subtract(ndvi_min)).pow(2)
em  = fv.multiply(0.004).add(0.986)
tb  = landsat_composite.select('ST_B10')

lst = tb.expression(
    '(tb / (1.0 + (0.00115 * tb / 1.438) * log(em))) - 273.15',
    {'tb': tb, 'em': em}
).rename('LST')

# ── UHI — spatial z-score of LST ─────────────────────────────────────────────
lstats   = lst.reduceRegion(
    ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True),
    ROI, scale=300, maxPixels=1e9)
lst_mean = ee.Number(ee.Algorithms.If(lstats.get('LST_mean'),   lstats.get('LST_mean'),   0))
lst_std  = ee.Number(ee.Algorithms.If(lstats.get('LST_stdDev'), lstats.get('LST_stdDev'), 1))

uhi = lst.subtract(lst_mean).divide(lst_std).rename('UHI')

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Sentinel-2 EVI Summer Composite 2019-2024 (Jun-Aug)
# S2 chosen over Landsat for: 10 m optical bands, SCL cloud masking (more
# accurate than QA_PIXEL for Mediterranean mixed terrain)
# ═══════════════════════════════════════════════════════════════════════════════

def _mask_s2(image):
    scl  = image.select('SCL')
    # Exclude: cloud shadow (3), medium cloud (8), high cloud (9), cirrus (10)
    mask = scl.eq(3).Or(scl.gte(8).And(scl.lte(10))).eq(0)
    return image.select(['B.*']).divide(10000).updateMask(mask)

s2_composite = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(ROI)
    .filter(ee.Filter.calendarRange(6, 8, 'month'))
    .filterDate('2019-01-01', '2024-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    .map(_mask_s2)
    .median()
    .clip(ROI)
)

# EVI formula: Huete et al. (2002)
evi = s2_composite.expression(
    '2.5 * (NIR - RED) / (NIR + 6.0 * RED - 7.0 * BLUE + 1.0)',
    {
        'NIR':  s2_composite.select('B8'),    # 10 m
        'RED':  s2_composite.select('B4'),    # 10 m
        'BLUE': s2_composite.select('B2'),    # 10 m
    }
).rename('EVI')

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — Sentinel-5P Tropospheric NO2 Summer Mean 2019-2024
# Compound health stressor: high LST + high NO2 → respiratory/cardiovascular risk
# OFFL product used (offline processing = better quality than NRTI)
# Native ~5.5 km; exporting at 1000 m consistent with other coarse inputs
# ═══════════════════════════════════════════════════════════════════════════════

s5p_no2 = (
    ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
    .filterBounds(ROI)
    .filter(ee.Filter.calendarRange(6, 8, 'month'))
    .filterDate('2019-01-01', '2024-12-31')
    .select('tropospheric_NO2_column_number_density')
    .mean()
    .clip(ROI)
    .rename('NO2')
)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — Summer (JJA) Precipitation — OpenLandMap SM2RAIN Climatology
# Lower summer precip = higher drought-heat compound stress
# Static monthly climatology (single image with named monthly bands)
# ═══════════════════════════════════════════════════════════════════════════════

summer_precip = (
    ee.Image('OpenLandMap/CLM/CLM_PRECIPITATION_SM2RAIN_M/v01')
    .select(['jun', 'jul', 'aug'])
    .reduce(ee.Reducer.mean())
    .clip(ROI)
    .rename('summer_precip_jja')
)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — GABAM Cumulative Burned Frequency 1990-2021
# Binary annual maps summed → pixel-level fire recurrence count
# ═══════════════════════════════════════════════════════════════════════════════

burned_freq = (
    ee.ImageCollection('projects/sat-io/open-datasets/GABAM')
    .filter(ee.Filter.calendarRange(1990, 2021, 'year'))
    .sum()
    .clip(ROI)
    .rename('burned_frequency')
)

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT — 6 tasks
# ═══════════════════════════════════════════════════════════════════════════════
print(f'Submitting 6 export tasks to Google Drive → {FOLDER}/')
submit(lst,          'p2_lst_summer_composite',    100)   # Landsat TIRS native = 100 m
submit(uhi,          'p2_uhi_summer_composite',    100)
submit(evi,          'p2_evi_summer_composite',    100)   # S2 10 m → aggregated to 100 m
submit(s5p_no2,      'p2_s5p_no2_summer',         1000)   # S5P native ~5.5 km
submit(summer_precip,'p2_summer_precip_jja',      1000)   # OLM native ~1 km
submit(burned_freq,  'p2_gabam_burned_frequency',  100)
print('Done. Monitor at code.earthengine.google.com → Tasks tab.')
