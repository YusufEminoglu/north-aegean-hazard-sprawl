"""
Paper 2 — GEE Batch Download
=============================
Downloads ALL required Earth Engine data layers for Paper 2 to data/01_raw/paper2/.
Uses the Kuzey Ege Havzasi boundary from the local GeoJSON.

Layers downloaded:
  1. SRTM DEM (30m)
  2. GLAD GLCLU 2000 + 2020 (30m)
  3. GLC-FCS30D annual LULC 1985-2022 (30m)
  4. ESA WorldCover 2021 (10m)
  5. CORINE 1990/2000/2006/2012/2018 (100m)
  6. JRC Global Surface Water occurrence (30m)
  7. Meta HRSL population grids (30m)
  8. VIIRS NTL annual composites 2014-2024 (500m)
  9. SoilGrids clay content (250m)
  10. WorldClim bio13 max precipitation (1km)
  11. MERIT Hydro drainage direction (90m)
  12. Global Flood DB historical floods (250m)
  13. Meta Canopy Height (10m)

Usage:
  python src/p2_00_gee_batch_download.py
"""

import ee
from _gee_config import drive_folder, initialize_ee, load_roi
import json
import time
import os
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "01_raw" / "paper2"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CRS = "EPSG:32635"
initialize_ee()
GEE_DRIVE_FOLDER = drive_folder()
roi = load_roi()
roi_buffered = roi.buffer(1000)
roi_bounds = roi_buffered.bounds()
print(f"ROI loaded: {roi.area().divide(1e6).getInfo():.0f} km2")


def export_image(image, description, scale, folder=GEE_DRIVE_FOLDER,
                 region=None, crs=CRS, max_pixels=1e13):
    """Start a Drive export task and return the task object."""
    if region is None:
        region = roi_bounds
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=max_pixels,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def wait_for_tasks(tasks, poll_interval=30):
    """Poll until all tasks complete or fail."""
    print(f"\nWaiting for {len(tasks)} tasks...")
    while True:
        statuses = []
        for name, task in tasks:
            status = task.status()
            state = status["state"]
            statuses.append((name, state))

        active = [(n, s) for n, s in statuses if s in ("READY", "RUNNING")]
        done = [(n, s) for n, s in statuses if s == "COMPLETED"]
        failed = [(n, s) for n, s in statuses if s in ("FAILED", "CANCEL_REQUESTED", "CANCELLED")]

        print(f"  Active: {len(active)} | Done: {len(done)} | Failed: {len(failed)}", end="\r")

        if not active:
            print()
            for n, s in failed:
                t = [t for nn, t in tasks if nn == n][0]
                print(f"  FAILED: {n} — {t.status().get('error_message', '?')}")
            for n, s in done:
                print(f"  OK: {n}")
            break

        time.sleep(poll_interval)


# ── Collect all export tasks ────────────────────────────────────────────────
all_tasks = []

# ── 1. SRTM DEM (30m) ──────────────────────────────────────────────────────
print("\n[1/13] SRTM DEM...")
srtm = ee.Image("USGS/SRTMGL1_003").clip(roi_buffered)
slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).clip(roi_buffered)
aspect = ee.Terrain.aspect(ee.Image("USGS/SRTMGL1_003")).clip(roi_buffered)

all_tasks.append(("p2_srtm_elevation", export_image(srtm.toFloat(), "p2_srtm_elevation", 30)))
all_tasks.append(("p2_srtm_slope", export_image(slope.toFloat(), "p2_srtm_slope", 30)))
all_tasks.append(("p2_srtm_aspect", export_image(aspect.toFloat(), "p2_srtm_aspect", 30)))

# TPI (Topographic Position Index)
tpi = srtm.subtract(srtm.focal_mean(150, "circle", "meters")).rename("TPI")
all_tasks.append(("p2_srtm_tpi", export_image(tpi.toFloat(), "p2_srtm_tpi", 30)))

# ── 2. GLAD GLCLU 2000 + 2020 (30m) ───────────────────────────────────────
print("[2/13] GLAD GLCLU 2000 & 2020...")
landmask = ee.Image("projects/glad/landBuffer4").mask()
lc2000 = ee.Image("projects/glad/GLCLU2020/LCLUC_2000").updateMask(landmask).clip(roi_buffered)
lc2020 = ee.Image("projects/glad/GLCLU2020/LCLUC_2020").updateMask(landmask).clip(roi_buffered)

all_tasks.append(("p2_glad_lulc_2000", export_image(lc2000.toInt16(), "p2_glad_lulc_2000", 30)))
all_tasks.append(("p2_glad_lulc_2020", export_image(lc2020.toInt16(), "p2_glad_lulc_2020", 30)))

# ── 3. GLC-FCS30D annual LULC 1985-2022 (30m) ─────────────────────────────
print("[3/13] GLC-FCS30D time series...")
five_year = ee.ImageCollection("projects/sat-io/open-datasets/GLC-FCS30D/five-years-map") \
    .filterBounds(roi).mosaic().clip(roi_buffered)
annual = ee.ImageCollection("projects/sat-io/open-datasets/GLC-FCS30D/annual") \
    .filterBounds(roi).mosaic().clip(roi_buffered)

# 5-year maps: b1=1985, b2=1990, b3=1995
for i, year in enumerate([1985, 1990, 1995]):
    band = five_year.select(f"b{i+1}")
    all_tasks.append((f"p2_glcfcs_{year}", export_image(band.toInt16(), f"p2_glcfcs_{year}", 30)))

# Annual maps: b1=2000, ..., b23=2022
for i in range(1, 24):
    year = 1999 + i
    band = annual.select(f"b{i}")
    all_tasks.append((f"p2_glcfcs_{year}", export_image(band.toInt16(), f"p2_glcfcs_{year}", 30)))

# ── 4. ESA WorldCover 2021 (10m) ──────────────────────────────────────────
print("[4/13] ESA WorldCover 2021...")
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(roi_buffered)
all_tasks.append(("p2_worldcover_2021", export_image(worldcover.toByte(), "p2_worldcover_2021", 10)))

# ── 5. CORINE Land Cover (100m) ───────────────────────────────────────────
print("[5/13] CORINE multi-year...")
corine_years = ["1990", "2000", "2006", "2012", "2018"]
for yr in corine_years:
    corine = ee.Image(f"COPERNICUS/CORINE/V20/100m/{yr}").select("landcover").clip(roi_buffered)
    all_tasks.append((f"p2_corine_{yr}", export_image(corine.toInt16(), f"p2_corine_{yr}", 100)))

# ── 6. JRC Global Surface Water (30m) ─────────────────────────────────────
print("[6/13] JRC Global Surface Water...")
gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
gsw_occurrence = gsw.select("occurrence").clip(roi_buffered)
gsw_seasonality = gsw.select("seasonality").clip(roi_buffered)
gsw_recurrence = gsw.select("recurrence").clip(roi_buffered)

all_tasks.append(("p2_gsw_occurrence", export_image(gsw_occurrence.toFloat(), "p2_gsw_occurrence", 30)))
all_tasks.append(("p2_gsw_seasonality", export_image(gsw_seasonality.toByte(), "p2_gsw_seasonality", 30)))
all_tasks.append(("p2_gsw_recurrence", export_image(gsw_recurrence.toFloat(), "p2_gsw_recurrence", 30)))

# Distance from permanent water (>80% occurrence)
permanent_water = gsw_occurrence.gt(80).selfMask()
water_distance = permanent_water.fastDistanceTransform().sqrt() \
    .multiply(ee.Image.pixelArea().sqrt()).clip(roi_buffered).rename("water_dist")
all_tasks.append(("p2_water_distance", export_image(water_distance.toFloat(), "p2_water_distance", 30)))

# ── 7. Meta HRSL Population (30m) ─────────────────────────────────────────
print("[7/13] Meta HRSL population...")
hrsl_datasets = {
    "pop_general": "projects/sat-io/open-datasets/hrsl/hrslpop",
    "pop_women": "projects/sat-io/open-datasets/hrsl/hrsl_women",
    "pop_children_u5": "projects/sat-io/open-datasets/hrsl/hrsl_children_under_five",
    "pop_elderly_60p": "projects/sat-io/open-datasets/hrsl/hrsl_elderly_over_sixty",
}
for name, asset in hrsl_datasets.items():
    img = ee.ImageCollection(asset).mosaic().clip(roi_buffered)
    all_tasks.append((f"p2_hrsl_{name}", export_image(img.toFloat(), f"p2_hrsl_{name}", 30)))

# ── 8. VIIRS Nighttime Lights (500m) ──────────────────────────────────────
print("[8/13] VIIRS NTL annual composites...")
viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
for year in range(2014, 2025):
    yearly = viirs.filter(ee.Filter.calendarRange(year, year, "year")) \
        .mean().select("avg_rad").clip(roi_buffered)
    all_tasks.append((f"p2_viirs_ntl_{year}", export_image(yearly.toFloat(), f"p2_viirs_ntl_{year}", 500)))

# ── 9. SoilGrids clay content (250m) ──────────────────────────────────────
print("[9/13] SoilGrids clay...")
clay = ee.Image("projects/soilgrids-isric/clay_mean") \
    .select("clay_0-5cm_mean").clip(roi_buffered).divide(10)
all_tasks.append(("p2_soilgrids_clay", export_image(clay.toFloat(), "p2_soilgrids_clay", 250)))

# Soil texture
soil_texture = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02") \
    .select("b0").clip(roi_buffered)
all_tasks.append(("p2_soil_texture", export_image(soil_texture.toByte(), "p2_soil_texture", 250)))

# ── 10. WorldClim bio13 + OpenLandMap precipitation (1km) ─────────────────
print("[10/13] Climate data...")
bio13 = ee.Image("WORLDCLIM/V1/BIO").select("bio13").clip(roi_buffered)
all_tasks.append(("p2_worldclim_bio13_maxprecip", export_image(bio13.toFloat(), "p2_worldclim_bio13_maxprecip", 1000)))

bio01 = ee.Image("WORLDCLIM/V1/BIO").select("bio01").multiply(0.1).clip(roi_buffered)
all_tasks.append(("p2_worldclim_bio01_meantemp", export_image(bio01.toFloat(), "p2_worldclim_bio01_meantemp", 1000)))

# OpenLandMap annual mean precipitation
olm_precip = ee.Image("OpenLandMap/CLM/CLM_PRECIPITATION_SM2RAIN_M/v01") \
    .reduce(ee.Reducer.mean()).clip(roi_buffered)
all_tasks.append(("p2_olm_annual_precip", export_image(olm_precip.toFloat(), "p2_olm_annual_precip", 1000)))

# ── 11. MERIT Hydro (90m) ─────────────────────────────────────────────────
print("[11/13] MERIT Hydro...")
merit = ee.Image("MERIT/Hydro/v1_0_1").clip(roi_buffered)
merit_dir = merit.select("dir")
merit_upa = merit.select("upa")  # upstream area
merit_elv = merit.select("elv")  # hydrologically adjusted elevation
merit_hnd = merit.select("hnd")  # HAND (Height Above Nearest Drainage)

all_tasks.append(("p2_merit_flow_dir", export_image(merit_dir.toInt16(), "p2_merit_flow_dir", 90)))
all_tasks.append(("p2_merit_upstream_area", export_image(merit_upa.toFloat(), "p2_merit_upstream_area", 90)))
all_tasks.append(("p2_merit_hydro_elev", export_image(merit_elv.toFloat(), "p2_merit_hydro_elev", 90)))
all_tasks.append(("p2_merit_hand", export_image(merit_hnd.toFloat(), "p2_merit_hand", 90)))

# HydroSHEDS rivers
rivers = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(roi)
rivers_img = ee.Image().byte().paint(rivers, 1, 1).selfMask()
river_distance = rivers_img.fastDistanceTransform().sqrt() \
    .multiply(ee.Image.pixelArea().sqrt()).clip(roi_buffered).rename("river_dist")
all_tasks.append(("p2_river_distance", export_image(river_distance.toFloat(), "p2_river_distance", 30)))

# ── 12. Global Flood DB (250m) ────────────────────────────────────────────
print("[12/13] Global Flood DB...")
gfd = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1").filterBounds(roi)
gfd_flooded = gfd.select("flooded").sum().clip(roi_buffered)
gfd_duration = gfd.select("duration").sum().clip(roi_buffered)
gfd_perm_water = gfd.select("jrc_perm_water").sum().gte(1).clip(roi_buffered)

all_tasks.append(("p2_gfd_flood_count", export_image(gfd_flooded.toInt16(), "p2_gfd_flood_count", 250)))
all_tasks.append(("p2_gfd_flood_duration", export_image(gfd_duration.toInt16(), "p2_gfd_flood_duration", 250)))
all_tasks.append(("p2_gfd_perm_water", export_image(gfd_perm_water.toByte(), "p2_gfd_perm_water", 250)))

# ── 13. Meta Canopy Height (10m) ──────────────────────────────────────────
print("[13/13] Meta Canopy Height...")
canopy = ee.ImageCollection("projects/sat-io/open-datasets/facebook/meta-canopy-height") \
    .filterBounds(roi).mosaic().clip(roi_buffered)
all_tasks.append(("p2_meta_canopy_height", export_image(canopy.toFloat(), "p2_meta_canopy_height", 10)))

# ── ROAD NETWORK (vector export) ──────────────────────────────────────────
print("\n[BONUS] GRIP4 road network...")
roads = ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/Europe").filterBounds(roi)
road_task = ee.batch.Export.table.toDrive(
    collection=roads,
    description="p2_grip4_roads",
    folder=GEE_DRIVE_FOLDER,
    fileNamePrefix="p2_grip4_roads",
    fileFormat="GeoJSON",
)
road_task.start()
all_tasks.append(("p2_grip4_roads", road_task))

# Road distance raster
road_img = ee.Image().byte().paint(roads, 1, 1).selfMask()
road_distance = road_img.fastDistanceTransform().sqrt() \
    .multiply(ee.Image.pixelArea().sqrt()).clip(roi_buffered).rename("road_dist")
all_tasks.append(("p2_road_distance", export_image(road_distance.toFloat(), "p2_road_distance", 30)))

# ── LANDFORM (30m) ────────────────────────────────────────────────────────
print("[BONUS] ALOS Landforms...")
landform = ee.Image("CSP/ERGo/1_0/Global/ALOS_landforms").select("constant").clip(roi_buffered)
all_tasks.append(("p2_alos_landforms", export_image(landform.toByte(), "p2_alos_landforms", 30)))

# ── LST / UHI / Thermal (100m, summer 2024) ──────────────────────────────
print("[BONUS] Landsat LST summer 2024...")

def mask_l89(image):
    qa = image.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(mask)

def scale_l89(image):
    optical = image.select(["SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7"]) \
        .multiply(0.0000275).add(-0.2).rename(["blue","green","red","nir","swir1","swir2"])
    thermal = image.select("ST_B10").multiply(0.00341802).add(149.0).rename("temp")
    return optical.addBands(thermal)

l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
summer_2024 = ee.ImageCollection(l8.merge(l9)) \
    .filterDate("2024-01-01", "2024-12-31") \
    .filter(ee.Filter.calendarRange(6, 8, "month")) \
    .filterBounds(roi) \
    .map(mask_l89).map(scale_l89).median().clip(roi_buffered)

# NDVI
ndvi = summer_2024.normalizedDifference(["nir", "red"]).rename("NDVI")
# NDWI
ndwi = summer_2024.normalizedDifference(["green", "nir"]).rename("NDWI")
# LST
ndvi_stats = ndvi.reduceRegion(ee.Reducer.minMax(), roi, 100, maxPixels=1e9)
ndvi_min = ee.Number(ee.Algorithms.If(ndvi_stats.get("NDVI_min"), ndvi_stats.get("NDVI_min"), 0))
ndvi_max = ee.Number(ee.Algorithms.If(ndvi_stats.get("NDVI_max"), ndvi_stats.get("NDVI_max"), 1))
fv = ndvi.subtract(ndvi_min).divide(ndvi_max.subtract(ndvi_min)).pow(2).rename("FV")
em = fv.multiply(0.004).add(0.986).rename("EM")
thermal = summer_2024.select("temp")
lst = thermal.expression(
    "(tb / (1 + (0.00115 * tb / 1.438) * log(em))) - 273.15",
    {"tb": thermal, "em": em}
).rename("LST")

all_tasks.append(("p2_ndvi_summer_2024", export_image(ndvi.toFloat(), "p2_ndvi_summer_2024", 30)))
all_tasks.append(("p2_ndwi_summer_2024", export_image(ndwi.toFloat(), "p2_ndwi_summer_2024", 30)))
all_tasks.append(("p2_lst_summer_2024", export_image(lst.toFloat(), "p2_lst_summer_2024", 100)))

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"TOTAL TASKS SUBMITTED: {len(all_tasks)}")
print(f"All exports go to Google Drive folder: '{GEE_DRIVE_FOLDER}'")
print(f"{'='*60}")

for name, _ in all_tasks:
    print(f"  - {name}")

print(f"\nMonitoring tasks... (Ctrl+C to stop monitoring, exports continue on GEE server)")
wait_for_tasks(all_tasks, poll_interval=60)
