"""
Paper 2 - Extra GEE Downloads: WorldPop 2050 Population + CMIP6 Climate
=========================================================================
Bu script, GEE Python API kullanarak:
  1. WorldPop 2050 nufus tahmini (lineer regresyon) - Kuzey Ege Havzasi
  2. CMIP6 SSP2-4.5 2040-2059 ortalamasi yillik yagis

WorldPop kaynagi:
  WorldPop Project / University of Southampton
  Veri: 100m cozunurluklu ulke nufus gridi (WorldPop/GP/100m/pop)
  Yontem: Mevcut yil verileri (2000-2020) icin lineer regresyon -> 2050 ekstrapolasyon
  Referans: https://developers.google.com/earth-engine/datasets/catalog/WorldPop_GP_100m_pop

Calistirmadan once:
  earthengine authenticate
  set EE_PROJECT=your-ee-project

Ciktilar (Google Drive -> GEE_DRIVE_FOLDER):
  p2_worldpop_2050_havza.tif   - 100m, EPSG:32635
  p2_cmip6_ssp245_pr_2050.tif  - 1000m, EPSG:32635
"""

import ee
from _gee_config import drive_folder, initialize_ee, load_roi
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "02_interim" / "paper2"


def get_havza_roi():
    return load_roi()


def make_worldpop_2050(roi):
    """
    WorldPop 2050 nufus tahmini - kullanicinin GEE scriptiyle ayni mantik.

    Adimlar:
    1. Turkiye'ye ait tum WorldPop yillik nufus goruntuleri yukle (2000-2020)
    2. Her goruntiye 'year' bandi ekle
    3. Lineer regresyon fit et (ee.Reducer.linearFit)
    4. scale * 2050 + offset -> pop2050
    5. Negatif pikselleri 0'a klample
    6. Havzaya kes ve Google Drive'a aktar
    """
    print("\n[1] WorldPop 2050 Nufus Tahmini hazirlaniyor...")

    worldPop = ee.ImageCollection("WorldPop/GP/100m/pop")

    # Turkiye ve havza bolgesindeki goruntuleri al
    turkeyPop = (worldPop
                 .filterBounds(roi)
                 .filter(ee.Filter.eq('country', 'TUR')))

    # Mevcut yil sayisini kontrol et
    count = turkeyPop.size()
    print(f"  WorldPop goruntu sayisi (TUR): kontrol edilecek GEE'de")

    # Her goruntiye yil bandi ekle (kullanicinin JS scriptiyle ayni)
    def add_year_band(img):
        year = ee.Number.parse(ee.String(img.get('year')))
        year_band = ee.Image.constant(year).rename('year').toFloat()
        return img.addBands(year_band).select(['year', 'population']).toFloat()

    pop_with_year = turkeyPop.map(add_year_band)

    # Lineer regresyon
    regression = pop_with_year.reduce(ee.Reducer.linearFit())
    # regression = {scale: egim, offset: y-kesimi}

    # 2050 tahmini: scale * 2050 + offset
    pop_2050 = (regression.select('scale').multiply(2050)
                .add(regression.select('offset')))

    # Negatif degerleri 0'a klample (nufus negatif olamaz)
    pop_2050_clamped = pop_2050.where(pop_2050.lt(0), 0).rename('population_2050')

    # Ek bilgi: 2020 ve 2023 tahminleri de ekle (referans icin)
    pop_2020_pred = (regression.select('scale').multiply(2020)
                     .add(regression.select('offset'))
                     .rename('population_2020_pred'))
    pop_2023_pred = (regression.select('scale').multiply(2023)
                     .add(regression.select('offset'))
                     .rename('population_2023_pred'))

    # Tum bant paketini havzaya kes
    output_img = ee.Image.cat([
        pop_2020_pred.where(pop_2020_pred.lt(0), 0),
        pop_2023_pred.where(pop_2023_pred.lt(0), 0),
        pop_2050_clamped
    ]).clip(roi)

    task = ee.batch.Export.image.toDrive(
        image=output_img,
        description='p2_worldpop_2050_havza',
        folder=drive_folder(),
        fileNamePrefix='p2_worldpop_2050_havza',
        region=roi.bounds(),
        scale=100,
        crs='EPSG:32635',
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    return task


def make_cmip6_precip_2050(roi):
    """
    CMIP6 SSP2-4.5 2040-2059 ortalama yillik yagis.
    Model: MIROC6 (NASA NEX-GDDP-CMIP6)
    Birim: kg/m2/s -> mm/yil cevrimi (x 86400 x 365)
    """
    print("\n[2] CMIP6 SSP2-4.5 Yagis Tahmini hazirlaniyor...")

    cmip6 = (ee.ImageCollection("NASA/GDDP-CMIP6")
             .filterBounds(roi)
             .filter(ee.Filter.date('2040-01-01', '2059-12-31'))
             .filter(ee.Filter.eq('scenario', 'ssp245'))
             .filter(ee.Filter.eq('model', 'MIROC6'))
             .select('pr'))

    # Gunluk kg/m2/s -> yillik mm/yil
    pr_annual_mm = cmip6.mean().multiply(86400 * 365).rename('annual_precip_2050_mm')

    task = ee.batch.Export.image.toDrive(
        image=pr_annual_mm.clip(roi),
        description='p2_cmip6_ssp245_pr_2050',
        folder=drive_folder(),
        fileNamePrefix='p2_cmip6_ssp245_pr_2050',
        region=roi.bounds(),
        scale=1000,
        crs='EPSG:32635',
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    return task


def make_worldpop_ilce_summary(roi):
    """
    Ilce bazli WorldPop 2020 ve 2050 tahmini tablosu.
    Bu, kullanicinin JS scriptindeki il bazli tablo mantiginin ilce versiyonu.
    Cikti: Google Drive'a CSV olarak aktarilir.
    """
    print("\n[3] Ilce bazli nufus ozeti hazirlaniyor...")

    worldPop = ee.ImageCollection("WorldPop/GP/100m/pop")
    turkeyPop = (worldPop
                 .filterBounds(roi)
                 .filter(ee.Filter.eq('country', 'TUR')))

    def add_year_band(img):
        year = ee.Number.parse(ee.String(img.get('year')))
        year_band = ee.Image.constant(year).rename('year').toFloat()
        return img.addBands(year_band).select(['year', 'population']).toFloat()

    pop_with_year = turkeyPop.map(add_year_band)
    regression = pop_with_year.reduce(ee.Reducer.linearFit())

    pop_2020_pred = (regression.select('scale').multiply(2020)
                     .add(regression.select('offset'))
                     .where(regression.select('scale').multiply(2020).add(regression.select('offset')).lt(0), 0))
    pop_2050_pred = (regression.select('scale').multiply(2050)
                     .add(regression.select('offset'))
                     .where(regression.select('scale').multiply(2050).add(regression.select('offset')).lt(0), 0))

    # Havza bolgesini 10km kareler ile grid'e bol (ilce yoksa yaklasim)
    # NOT: Eger GAUL veya Turkiye idari sinir asset'i varsa buraya eklenebilir
    # Su an sadece raster degerlerini aktariyoruz (ilce siniri eklenmeli)

    # Ilce bazli analiz icin: GAUL Level-2 kullan
    gaul2 = ee.FeatureCollection("FAO/GAUL/2015/level2")
    havza_ilceler = gaul2.filterBounds(roi)

    # 2020 nufus
    pop2020_by_ilce = pop_2020_pred.rename('Pop_2020').reduceRegions(
        collection=havza_ilceler,
        reducer=ee.Reducer.sum(),
        scale=100
    ).map(lambda f: f.set('Pop_2020', ee.Number(f.get('sum')).toUint32(), 'sum', None))

    # 2050 nufus tahmini
    pop2050_by_ilce = pop_2050_pred.rename('Pop_2050').reduceRegions(
        collection=pop2020_by_ilce,
        reducer=ee.Reducer.sum(),
        scale=100
    ).map(lambda f: f.set('Pop_2050', ee.Number(f.get('sum')).toUint32(), 'sum', None))

    task = ee.batch.Export.table.toDrive(
        collection=pop2050_by_ilce,
        description='p2_worldpop_ilce_2020_2050',
        folder=drive_folder(),
        fileNamePrefix='p2_worldpop_ilce_2020_2050',
        fileFormat='CSV',
        selectors=['ADM2_NAME', 'ADM1_NAME', 'Pop_2020', 'Pop_2050']
    )
    return task


def main():
    initialize_ee()
    roi = get_havza_roi()

    tasks = []

    # 1. WorldPop 2050 raster (3 bantli: 2020_pred, 2023_pred, 2050_pred)
    task_pop_raster = make_worldpop_2050(roi)
    tasks.append(task_pop_raster)

    # 2. CMIP6 yagis
    task_clim = make_cmip6_precip_2050(roi)
    tasks.append(task_clim)

    # 3. Ilce bazli CSV tablo (FLUS talebi icin)
    task_ilce = make_worldpop_ilce_summary(roi)
    tasks.append(task_ilce)

    # Gorevleri baslat
    print("\n" + "="*60)
    print("GEE Gorevleri baslatiliyor...")
    for t in tasks:
        try:
            t.start()
            status = t.status()
            print(f"  Baslatildi: {status['description']}  (ID: {t.id})")
        except Exception as e:
            print(f"  HATA: {e}")

    print("\nGorevler arka planda calisuyor.")
    print("Durumu kontrol et: https://code.earthengine.google.com/tasks")
    print("\nCiktilari indir ve su konuma kopyala:")
    print("  data/02_interim/paper2/p2_worldpop_2050_havza_clipped.tif")
    print("  data/02_interim/paper2/p2_cmip6_ssp245_pr_2050_clipped.tif")
    print("  outputs/tables/paper2/p2_worldpop_ilce_2020_2050.csv")


if __name__ == "__main__":
    main()
