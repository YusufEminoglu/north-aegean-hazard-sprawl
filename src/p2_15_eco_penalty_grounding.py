"""ECO scenario penalty grounding against real DSI protection zones.

The ECO scenario penalises agriculture (x0.1), forest (x0.1), and a 500 m
riparian buffer (x0.1). Rather than asserting these are realistic, this
script tests them against real DSI Kuzey Ege Havzasi Master Plan regulatory
geometry (see p2_11_dsi_masterplan_ingest.py):

  - KORUMA_BANTLARI: official water-source protection buffers, tipi 1/2/3 =
    300/1000/2000 m bands (Icmesuyu Havzalari Yonetmeligi short/medium/long-
    distance protection zones).
  - KORUNAN_ALANLAR: protected-area polygons in the basin.

For each: (a) rasterise to the study grid, (b) report what fraction of the
official zones falls inside the ECO scenario's already-penalised area
(riparian buffer + agriculture + forest), and (c) report how much BAU vs
ECO new urban growth intrudes into these real protection zones - a direct,
data-grounded policy metric independent of the assumed penalty multipliers.

Usage:
  python src/p2_15_eco_penalty_grounding.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "03_processed" / "paper2"
INTERIM = ROOT / "data" / "02_interim" / "paper2"
DSI = INTERIM / "dsi_masterplan"
DRIVERS = PROC / "drivers"

PIXEL_HA = 0.09  # 30 m x 30 m


def rasterize_layer(gdf, ref_meta):
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros((ref_meta["height"], ref_meta["width"]), dtype=np.uint8)
    return rasterize(shapes, out_shape=(ref_meta["height"], ref_meta["width"]),
                      transform=ref_meta["transform"], fill=0, default_value=1,
                      dtype=np.uint8)


def main() -> None:
    print("== ECO penalty grounding vs DSI protection zones ==")

    ref_path = PROC / "lulc_simplified_2020.tif"
    with rasterio.open(ref_path) as ref:
        ref_meta = ref.meta.copy()
        lulc2020 = ref.read(1)
        nodata = ref.nodata
    valid = (lulc2020 != nodata) if nodata is not None else (lulc2020 > 0)

    with rasterio.open(DRIVERS / "driver_river_dist.tif") as r:
        riv = r.read(1)

    eco_penalised = valid & ((lulc2020 == 2) | (lulc2020 == 3) | (riv < 500))
    n_valid = int(valid.sum())
    print(f"  ECO-penalised area (agriculture + forest + 500m riparian): "
          f"{float(eco_penalised.sum())/n_valid*100:.1f}% of basin")

    kb_path = DSI / "dsi_koruma_bantlari.gpkg"
    ka_path = DSI / "dsi_korunan_alanlar.gpkg"
    if not (kb_path.exists() and ka_path.exists()):
        print("  [MISSING] DSI layers - run p2_11_dsi_masterplan_ingest.py first.")
        return

    kb = gpd.read_file(kb_path).to_crs(ref_meta["crs"])
    ka = gpd.read_file(ka_path).to_crs(ref_meta["crs"])

    kb_raster = rasterize_layer(kb, ref_meta).astype(bool) & valid
    ka_raster = rasterize_layer(ka, ref_meta).astype(bool) & valid

    for name, raster in [("KORUMA_BANTLARI (official water-source buffers)", kb_raster),
                          ("KORUNAN_ALANLAR (protected areas)", ka_raster)]:
        n_official = int(raster.sum())
        overlap = int((raster & eco_penalised).sum())
        pct_overlap = overlap / n_official * 100 if n_official > 0 else 0.0
        print(f"\n  {name}: {n_official*PIXEL_HA:,.0f} ha in basin")
        print(f"    Overlap with ECO-penalised zones: {pct_overlap:.1f}% "
              f"({overlap*PIXEL_HA:,.0f} ha of {n_official*PIXEL_HA:,.0f} ha)")

    key_kb = "dsi_koruma_overlap_pct"
    key_ka = "dsi_korunan_overlap_pct"
    print(f"\n{key_kb},{int((kb_raster & eco_penalised).sum())/max(int(kb_raster.sum()),1)*100:.1f}")
    print(f"{key_ka},{int((ka_raster & eco_penalised).sum())/max(int(ka_raster.sum()),1)*100:.1f}")

    # BAU vs ECO new-growth intrusion into real protection zones
    base_urban = (lulc2020 == 1) & valid
    for sc in ["sim_2050_bau", "sim_2050_eco"]:
        sim_path = PROC / f"{sc}.tif"
        if not sim_path.exists():
            continue
        with rasterio.open(sim_path) as s:
            sim = s.read(1)
        new_urban = (sim == 1) & (~base_urban) & valid
        total = int(new_urban.sum())
        into_kb = int((new_urban & kb_raster).sum())
        into_ka = int((new_urban & ka_raster).sum())
        print(f"\n  {sc}: {into_kb*PIXEL_HA:,.1f} ha of new growth inside official water-source "
              f"buffers ({into_kb/total*100 if total else 0:.2f}%), "
              f"{into_ka*PIXEL_HA:,.1f} ha inside protected areas "
              f"({into_ka/total*100 if total else 0:.2f}%)")
        tag = sc.replace("sim_2050_", "")
        print(f"dsi_koruma_intrusion_ha_{tag},{into_kb*PIXEL_HA:.1f}")
        print(f"dsi_korunan_intrusion_ha_{tag},{into_ka*PIXEL_HA:.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
