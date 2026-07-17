# Reproducibility guide

## 1. Create the environment

Conda/Mamba is recommended because Rasterio, GDAL, GEOS, and PROJ must remain compatible.

~~~bash
mamba env create -f environment.yml
mamba activate north-aegean-hazard-sprawl
python scripts/verify_ahp.py
~~~

A pip-only setup is also pinned in <code>requirements.txt</code>, but system geospatial libraries may still be required.

## 2. Configure Earth Engine without committing secrets

Authenticate with your own Google Earth Engine account. Define these settings in your shell or an untracked local environment file:

~~~text
EE_PROJECT=your-ee-cloud-project
GEE_DRIVE_FOLDER=your-export-folder
EE_ROI_ASSET=your-private-boundary-asset-id
GEE_DOWNLOAD_DIR=/path/to/downloaded/exports
P2_DEMAND_METHOD=linear
~~~

If <code>EE_ROI_ASSET</code> is not set, <code>src/_gee_config.py</code> loads <code>data/01_raw/paper2/kuzey_ege_havzasi_v2.geojson</code>. The publication audit rejects private asset IDs, hard-coded Drive folders, user-specific absolute paths, and common token formats.

## 3. Obtain source data

Follow [data_sources.md](data_sources.md), keep a provenance ledger with download date/version, and confirm each provider’s terms. Do not add provider rasters to Git.

Expected ignored stages:

~~~text
data/01_raw/paper2/
data/02_interim/paper2/
data/03_processed/paper2/
data/04_models/paper2/
outputs/
logs/
~~~

## 4. Run the pipeline

The numeric prefixes encode dependency order.

~~~bash
python src/p2_00_gee_batch_download.py
python src/p2_00c_bioclimate_fire_download.py
python src/p2_00d_ndvi_2020_download.py
python src/p2_01_move_clip_validate.py
python src/p2_02_lulc_timeseries.py
python src/p2_03_spatial_drivers.py
python src/p2_04_flus_ca_engine.py
python src/p2_05_flood_hazard.py
python src/p2_06_seismic_hazard.py
python src/p2_07_bioclimate_hazard.py
python src/p2_08_wildfire_hazard.py
python src/p2_09_multi_hazard_fusion.py
python src/p2_10_model_validation.py
~~~

Some provider-specific exports are split into <code>p2_00b</code>–<code>p2_00d</code>. Run only the exporters required for inputs you do not already have.

## 5. Regenerate figures

Figure scripts are in <code>src/p2_fig01_*.py</code> through <code>src/p2_fig10_*.py</code>. They expect completed processed outputs. Final authored PNGs are included in <code>figures/png</code> for inspection without third-party inputs.

## 6. Change the demand pathway

The headline scenario uses linear continuation:

~~~powershell
$env:P2_DEMAND_METHOD = "linear"
python src/p2_04_flus_ca_engine.py
~~~

For the Markov sensitivity pathway, set the value to <code>markov</code>. A <code>population</code> option is available only when the documented district projection CSV is supplied.

## 7. Audit before publishing

~~~bash
python -m compileall -q src scripts
python scripts/verify_ahp.py
python scripts/publication_audit.py
~~~

The audit also asserts that exactly ten author figures are present and that no PDF, manuscript source, model object, raw raster, private Earth Engine identifier, or credential-like string is publishable.
