"""Independent wildfire & seismic hazard validation.

Flood is validated against GFD + DSI historical floods (p2_10, p2_11). This
script does the same for the two remaining components:

  Wildfire: MODIS/VIIRS thermal-anomaly detections (GEE `FIRMS` collection,
  ~1 km, 2001-2024) - a sensor/detection-method INDEPENDENT of the GABAM
  Landsat-derived burned-area product already used as a wildfire_hazard
  input, so this is a genuine independent check, not circular validation.

  Seismic: USGS ComCat historical earthquake catalogue (public REST API, no
  auth), M>=2.5, full historical record for the basin bounding box. This is
  OBSERVED seismicity checked against the MODELLED seismic_hazard surface
  (GEM fault buffers + ESHM20 PGA) - explicitly reported as epicentre
  density, NOT damage/loss records.

Usage:
  python src/p2_12_wildfire_seismic_validation.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.warp import Resampling, reproject
from sklearn.metrics import roc_auc_score

import _gee_config

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "03_processed" / "paper2"
INTERIM = ROOT / "data" / "02_interim" / "paper2"


def enrichment_report(score, valid, obs_mask, tag):
    auc = roc_auc_score(obs_mask[valid].astype(np.uint8), score[valid])
    print(f"  {tag}_auc = {auc:.3f}")
    print(f"{tag}_auc,{auc:.3f}")
    q = np.nanpercentile(score[valid], [20, 40, 60, 80])
    edges = [score[valid].min()] + list(q) + [score[valid].max()]
    for c in range(5):
        lo, hi = edges[c], edges[c + 1]
        cls_mask = valid & (score >= lo) & (score <= hi if c == 4 else score < hi)
        prev = float(obs_mask[cls_mask].sum()) / float(cls_mask.sum()) * 100 if cls_mask.sum() else 0.0
        print(f"    class {c+1}: prevalence={prev:.4f}%  n={int(cls_mask.sum()):,}")
        print(f"{tag}_enrichment_class{c+1},{prev:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# A. Wildfire — GEE FIRMS thermal-anomaly detection frequency (independent of GABAM)
# ════════════════════════════════════════════════════════════════════════════
def wildfire_validation():
    print("\n[A] Wildfire validation vs GEE FIRMS thermal-anomaly detections ...")
    _gee_config.initialize_ee()
    roi = _gee_config.load_roi()

    import ee

    # NOTE: 'T21' (brightness temperature) has a valid value for every MODIS
    # overpass regardless of whether a fire was detected - using T21>0 as a
    # "fire" filter massively over-counts (it is really an observation-
    # frequency mask, ~39% of the basin). 'confidence' is populated ONLY at
    # pixels MODIS actually flagged as a thermal anomaly, so its presence
    # (not its magnitude) is the correct per-pixel fire-detection indicator;
    # we additionally require confidence >= 30 (nominal-or-above, excluding
    # the lowest-confidence tier).
    firms = (ee.ImageCollection('FIRMS')
             .filterDate('2001-01-01', '2024-12-31')
             .filterBounds(roi)
             .select('confidence'))
    detect_freq = firms.map(lambda img: img.gte(30)).sum().clip(roi).rename('fire_freq')

    url = detect_freq.getDownloadURL({
        'scale': 1000, 'crs': 'EPSG:32635', 'region': roi, 'format': 'GEO_TIFF'})
    print("  Downloading FIRMS detection-frequency raster ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out_path = INTERIM / "p2_firms_fire_frequency.tif"
    out_path.write_bytes(r.content)
    print(f"  Saved: {out_path.relative_to(ROOT)} ({len(r.content):,} bytes)")

    with rasterio.open(PROC / "wildfire_hazard.tif") as ref:
        ref_meta = ref.meta.copy()
        wf = ref.read(1).astype(np.float32)
        nodata = ref.nodata

    fire = np.zeros((ref_meta["height"], ref_meta["width"]), dtype=np.float32)
    with rasterio.open(out_path) as src:
        reproject(source=rasterio.band(src, 1), destination=fire,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
                  resampling=Resampling.bilinear)

    valid = (wf != nodata) if nodata is not None else np.isfinite(wf)
    obs = valid & (fire > 0)
    n_detect = int(obs.sum())
    print(f"  FIRMS thermal-anomaly-positive pixels in basin (ALL land, "
          f"2001-2024): {n_detect:,} ({n_detect/int(valid.sum())*100:.3f}% of basin)")
    print(f"firms_positive_pct,{n_detect/int(valid.sum())*100:.3f}")

    if n_detect > 20:
        print("  -- basin-wide (includes agricultural burning) --")
        enrichment_report(wf, valid, obs, "wildfire_allland")

    # At ~1 km MODIS resolution over a 24-year record, raw FIRMS detections in
    # a basin this agricultural are dominated by low-intensity crop-residue
    # burning on farmland, not wildland fire in forest fuel - the construct
    # wildfire_hazard.tif actually targets (GABAM burned frequency + EVI +
    # canopy height). Restricting detections to forest/shrubland pixels
    # (LULC class 3) isolates the wildland-fire-relevant subset for a fair
    # like-for-like independent check.
    with rasterio.open(PROC / "lulc_simplified_2020.tif") as s:
        lulc = np.zeros((ref_meta["height"], ref_meta["width"]), dtype=np.float32)
        reproject(source=rasterio.band(s, 1), destination=lulc,
                  src_transform=s.transform, src_crs=s.crs,
                  dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
                  resampling=Resampling.nearest)
    forest = valid & (lulc == 3)
    obs_forest = forest & (fire > 0)
    n_detect_forest = int(obs_forest.sum())
    print(f"\n  Forest/shrubland-restricted FIRMS detections: {n_detect_forest:,} "
          f"({n_detect_forest/int(forest.sum())*100:.3f}% of forest/shrub area)")
    print(f"firms_forest_positive_pct,{n_detect_forest/int(forest.sum())*100:.3f}")
    if n_detect_forest > 20:
        print("  -- forest/shrubland only (wildland-fire-relevant subset) --")
        enrichment_report(wf, forest, obs_forest, "wildfire_forest")
    else:
        print("  [WARNING] Too few forest-restricted detections for a robust AUC.")

    wildfire_subcomponent_decomposition(ref_meta, forest, obs_forest)


# ════════════════════════════════════════════════════════════════════════════
# A2. Wildfire — decompose the composite by its 3 sub-inputs (GABAM burned
#     frequency, summer EVI, canopy height) against the same forest/shrub
#     FIRMS detections, to see whether the composite's negative validation
#     traces to one weak ingredient or is shared across all three.
# ════════════════════════════════════════════════════════════════════════════
SUBINPUTS = {
    "gabam": INTERIM / "p2_gabam_burned_frequency.tif",
    "evi":   INTERIM / "p2_evi_summer_composite.tif",
    "tch":   INTERIM / "p2_meta_canopy_height_clipped.tif",
}


def _load_aligned_nan(path, ref_meta):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        dst = np.full((ref_meta["height"], ref_meta["width"]), np.nan, dtype=np.float32)
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
                  resampling=Resampling.bilinear,
                  src_nodata=src.nodata, dst_nodata=np.nan)
        return dst


def wildfire_subcomponent_decomposition(ref_meta, forest, obs_forest):
    print("\n  -- sub-input decomposition (forest/shrubland domain) --")
    for key, path in SUBINPUTS.items():
        arr = _load_aligned_nan(path, ref_meta)
        has_data = forest & np.isfinite(arr)
        cov = float(has_data.sum()) / float(forest.sum()) * 100
        auc = roc_auc_score(obs_forest[has_data].astype(np.uint8), arr[has_data])
        pos_mean = float(np.nanmean(arr[has_data & obs_forest]))
        neg_mean = float(np.nanmean(arr[has_data & ~obs_forest]))
        print(f"    {key}: coverage={cov:.1f}%  AUC={auc:.3f}  "
              f"mean@fire={pos_mean:.2f}  mean@nofire={neg_mean:.2f}")
        print(f"wildfire_{key}_auc,{auc:.3f}")
        print(f"wildfire_{key}_coverage_pct,{cov:.1f}")
        print(f"wildfire_{key}_mean_fire,{pos_mean:.2f}")
        print(f"wildfire_{key}_mean_nofire,{neg_mean:.2f}")


# ════════════════════════════════════════════════════════════════════════════
# B. Seismic — USGS ComCat historical earthquake catalogue (epicentre density,
#    NOT damage records)
# ════════════════════════════════════════════════════════════════════════════
def seismic_validation():
    print("\n[B] Seismic validation vs USGS historical earthquake catalogue "
          "(epicentre density, not damage records) ...")

    basin = gpd.read_file(INTERIM / "kuzey_ege_havzasi_boundary.gpkg").to_crs("EPSG:4326")
    minx, miny, maxx, maxy = basin.total_bounds

    resp = requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params={"format": "geojson", "starttime": "1900-01-01", "endtime": "2024-12-31",
                "minlatitude": miny, "maxlatitude": maxy,
                "minlongitude": minx, "maxlongitude": maxx,
                "minmagnitude": 2.5},
        timeout=60)
    resp.raise_for_status()
    data = resp.json()
    feats = data["features"]
    print(f"  USGS events (M>=2.5, 1900-2024) in basin bbox: {len(feats)}")

    lons = [f["geometry"]["coordinates"][0] for f in feats]
    lats = [f["geometry"]["coordinates"][1] for f in feats]
    mags = [f["properties"]["mag"] for f in feats]
    gdf = gpd.GeoDataFrame({"mag": mags}, geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")
    gdf = gdf[gdf.geometry.within(basin.union_all())]
    print(f"  Events within basin polygon: {len(gdf)}")
    print(f"usgs_quake_n,{len(gdf)}")

    with rasterio.open(PROC / "seismic_hazard.tif") as ref:
        ref_meta = ref.meta.copy()
        sh = ref.read(1).astype(np.float32)
        nodata = ref.nodata
    valid = (sh != nodata) if nodata is not None else np.isfinite(sh)

    gdf32635 = gdf.to_crs(ref_meta["crs"])
    rows, cols = rasterio.transform.rowcol(ref_meta["transform"], gdf32635.geometry.x, gdf32635.geometry.y)
    rows = np.clip(np.array(rows), 0, sh.shape[0] - 1)
    cols = np.clip(np.array(cols), 0, sh.shape[1] - 1)
    epicentre_mask = np.zeros(sh.shape, dtype=bool)
    epicentre_mask[rows, cols] = True
    epicentre_mask &= valid

    vals_at_pts = sh[rows, cols]
    ok = np.isfinite(vals_at_pts)
    pctl = [float((sh[valid] <= v).sum()) / float(valid.sum()) * 100 for v in vals_at_pts[ok]]
    print(f"  Mean seismic-hazard percentile at epicentre locations: {np.mean(pctl):.1f} "
          f"(50 = no enrichment)")
    print(f"usgs_quake_mean_percentile,{np.mean(pctl):.1f}")

    if epicentre_mask.sum() > 20:
        enrichment_report(sh, valid, epicentre_mask, "seismic_usgs")
    print("\n  NOTE: this is an epicentre-DENSITY check on modelled ground-motion hazard "
          "(GEM faults + ESHM20 PGA), not validation against structural-damage or loss "
          "records.")


if __name__ == "__main__":
    wildfire_validation()
    seismic_validation()
    print("\nDone. Transcribe key,value lines into docs/manuscripts/paper2/src/mydata.dat")
