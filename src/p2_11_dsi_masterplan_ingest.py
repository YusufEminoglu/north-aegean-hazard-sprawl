"""Ingest DSI (State Hydraulic Works) Kuzey Ege Havzasi Master Plan GIS
layers, reproject, and clip to the basin boundary.

The DSI Master Plan shapefiles are restricted-distribution government data
and are not included in this repository. Place the source folder (as
supplied by DSI, containing TARIHI_TASKINLAR.shp, TASKIN_ALANLARI_4373.shp,
KORUNAN_ALANLAR.shp, KORUMA_BANTLARI.shp, FAYLAR.shp) somewhere local and
point DSI_SOURCE_DIR at it, or drop it at the default location below.

Source CRS is EPSG:23035 (ED50 / UTM 35N); reprojected to the project
working CRS EPSG:32635 (WGS84 / UTM 35N) and clipped to the basin boundary.

Layers pulled (all others in the master plan are hydrology/water-supply
infrastructure, not needed for hazard validation):

  TARIHI_TASKINLAR.shp        historical flood events (points, n=127)
                               -> independent flood-hazard validation
  TASKIN_ALANLARI_4373.shp    delineated flood-prone areas (polygons, n=25)
                               -> independent flood-hazard validation
  KORUNAN_ALANLAR.shp         protected areas (polygons, n=301)
                               -> real regulatory grounding for the ECO
                                  scenario's ecological penalty zones
  KORUMA_BANTLARI.shp         official water-source protection buffers
                               (polygons, n=90; tipi 1/2/3 = 300/1000/2000 m
                               bands per Icmesuyu Havzalari Yonetmeligi)
                               -> real regulatory buffer-distance evidence
  FAYLAR.shp                  mapped faults (lines, n=847)
                               -> cross-check vs the GEM fault-buffer layer
                                  already used in the seismic hazard model

TASKIN_TEHLIKE_ALANI / TASKIN_RISKI_ON_DEGERLENDIRME / YAS_KORUMA_ALANI are
present in the source but contain 0 features for this basin extract -
skipped.

Usage:
  python src/p2_11_dsi_masterplan_ingest.py
"""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "02_interim" / "paper2"
OUT_DIR = INTERIM / "dsi_masterplan"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DSI_SRC = Path(os.environ.get(
    "DSI_SOURCE_DIR", PROJECT_ROOT / "data" / "01_raw" / "paper2" / "dsi_masterplan_source"))
BASIN_GPKG = INTERIM / "kuzey_ege_havzasi_boundary.gpkg"
TARGET_CRS = "EPSG:32635"

LAYERS = {
    "TARIHI_TASKINLAR": "dsi_tarihi_taskinlar.gpkg",
    "TASKIN_ALANLARI_4373": "dsi_taskin_alanlari.gpkg",
    "KORUNAN_ALANLAR": "dsi_korunan_alanlar.gpkg",
    "KORUMA_BANTLARI": "dsi_koruma_bantlari.gpkg",
    "FAYLAR": "dsi_faylar.gpkg",
}


def main() -> None:
    if not DSI_SRC.exists():
        print(f"ERROR: DSI source folder not found: {DSI_SRC}")
        print("Set DSI_SOURCE_DIR to the folder containing the DSI Master "
              "Plan shapefiles, or place them at the path above.")
        raise SystemExit(1)
    if not BASIN_GPKG.exists():
        print(f"ERROR: Basin boundary not found: {BASIN_GPKG}")
        raise SystemExit(1)

    basin = gpd.read_file(BASIN_GPKG).to_crs(TARGET_CRS)
    basin_geom = basin.union_all()

    for name, out_name in LAYERS.items():
        src_path = DSI_SRC / f"{name}.shp"
        if not src_path.exists():
            print(f"  [MISSING] {src_path.name}")
            continue

        gdf = gpd.read_file(src_path)
        if len(gdf) == 0:
            print(f"  [EMPTY]   {name} (0 features in source, skipping)")
            continue

        gdf = gdf.to_crs(TARGET_CRS)
        is_poly = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        if is_poly.any():
            # buffer(0) fixes invalid rings but collapses Point/LineString
            # geometries to empty ones -- only apply to polygonal features.
            gdf.loc[is_poly, "geometry"] = gdf.loc[is_poly, "geometry"].buffer(0)
        clipped = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty
                      & gdf.geometry.intersects(basin_geom)].copy()

        out_path = OUT_DIR / out_name
        clipped.to_file(out_path, driver="GPKG")
        print(f"  OK  {name:<28} {len(gdf):>4} src -> {len(clipped):>4} in-basin -> {out_path.name}")

    print("\nDone. Clipped layers in:", OUT_DIR.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
