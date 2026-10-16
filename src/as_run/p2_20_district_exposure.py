"""
Paper 2 -- District-level BAU exposure table (manuscript Table 9, R4)
=====================================================================
Reproduces the district ranking of BAU 2050 new urban growth and its hazard
exposure by raster-polygon intersection, so that every value in the table is
traceable to a script (previous versions of the table were hand-typed).

For each district (OSM admin_level=6 polygons cached at
data/02_interim/paper2/admin_ilce.gpkg, clipped to the basin, features smaller
than 30 km2 dropped as sub-district units):
  * BAU new growth (ha): simulated 2050 urban pixels that are not urban in 2020;
  * class 4-5 share (%): share of that district's new growth in hazard classes
    4-5 of the Jenks multi-hazard surface;
  * dominant component: the hazard component (flood, seismic, compound =
    bio-climatic + wildfire) with the largest AHP-weighted mean normalised
    score over the district's new-growth pixels (weights 0.35/0.35/0.20/0.10).

Writes outputs/tables/paper2/district_exposure_top10.csv and a LaTeX body
(outputs/tables/paper2/table_7_districts_body.tex) for the manuscript table.

Usage:
  .venv/Scripts/python.exe src/paper2_hazard_sprawl/p2_20_district_exposure.py
"""

import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("p2_09", HERE / "p2_09_multi_hazard_fusion.py")
p2_09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2_09)

PROCESSED, INTERIM, TABLES = p2_09.PROCESSED, p2_09.INTERIM, p2_09.TABLES
PIXEL_HA = 0.09
TEX = {"ğ": r"\u{g}", "ı": r"\i{}", "ç": r"\c{c}", "ş": r"\c{s}", "ö": r'\"{o}', "ü": r'\"{u}',
       "İ": r"\.{I}", "Ç": r"\c{C}", "Ş": r"\c{S}", "Ö": r'\"{O}', "Ü": r'\"{U}', "Ğ": r"\u{G}"}


def tex(name):
    return "".join(TEX.get(c, c) for c in name)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with rasterio.open(PROCESSED / "multi_hazard_surface.tif") as ref:
        meta = ref.meta.copy()
        haz = ref.read(1).astype(np.float32)
        nd = ref.nodata
        transform = ref.transform
    valid = haz != nd
    with rasterio.open(PROCESSED / "lulc_simplified_2020.tif") as s:
        l20 = s.read(1)
    with rasterio.open(PROCESSED / "sim_2050_bau.tif") as s:
        bau = s.read(1)
    new = (bau == 1) & ~((l20 == 1) & valid) & valid

    comp = {}
    for key, fn in (("flood", "flood_hazard.tif"), ("seismic", "seismic_hazard.tif"),
                    ("bioclimate", "bioclimate_hazard.tif"), ("wildfire", "wildfire_hazard.tif")):
        arr = p2_09.load_aligned(PROCESSED / fn, meta)
        comp[key] = p2_09.normalise(arr, valid) * p2_09.WEIGHTS[key]
    contrib = {"F": comp["flood"], "S": comp["seismic"], "C": comp["bioclimate"] + comp["wildfire"]}

    gdf = gpd.read_file(INTERIM / "admin_ilce.gpkg").to_crs(meta["crs"])
    # The cache mixes administrative levels (country, regions, provinces,
    # districts, sub-district units); keep district-level units only.
    non_district = {"Türkiye", "Ege Bölgesi", "Marmara Bölgesi", "Balıkesir", "İzmir", "Çanakkale", "Manisa"}
    gdf = gdf[~gdf["name"].isin(non_district)]
    gdf = gdf[gdf.geometry.area / 1e6 >= 30.0].copy()
    gdf = gdf.dissolve(by="name", as_index=False)
    ids = {i + 1: n for i, n in enumerate(gdf["name"])}
    shapes = [(g, i + 1) for i, g in enumerate(gdf.geometry)]
    lab = features.rasterize(shapes, out_shape=haz.shape, transform=transform, fill=0, dtype="int32")

    rows = []
    for i, name in ids.items():
        m = new & (lab == i)
        n_px = int(m.sum())
        if n_px == 0:
            continue
        hi = int((m & (haz >= 4)).sum())
        dom = max(contrib, key=lambda k: float(contrib[k][m].mean()))
        rows.append((name, n_px * PIXEL_HA, hi * PIXEL_HA, hi / n_px * 100, dom))
    rows.sort(key=lambda r: -r[1])
    top = rows[:10]
    print(f"Districts with BAU new growth: {len(rows)}; total in districts {sum(r[1] for r in rows):,.1f} ha "
          f"of {new.sum() * PIXEL_HA:,.1f} ha")
    print("rank,district,new_ha,class45_ha,class45_pct,dominant")
    for k, r in enumerate(top, 1):
        print(f"{k},{r[0]},{r[1]:.1f},{r[2]:.1f},{r[3]:.1f},{r[4]}")

    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / "district_exposure_top10.csv", "w", encoding="utf-8") as f:
        f.write("rank,district,new_ha,class45_ha,class45_pct,dominant\n")
        for k, r in enumerate(top, 1):
            f.write(f"{k},{r[0]},{r[1]:.1f},{r[2]:.1f},{r[3]:.1f},{r[4]}\n")
    with open(TABLES / "table_7_districts_body.tex", "w", encoding="utf-8") as f:
        for k, r in enumerate(top, 1):
            nm = tex(r[0])
            f.write(f"{k:<2} & {nm:<20} & {r[1]:,.1f} & {r[2]:,.1f} & {r[3]:.1f} & {r[4]} \\\\\n".replace(",", "{,}"))
    print("Wrote district_exposure_top10.csv and table_7_districts_body.tex")


if __name__ == "__main__":
    main()
