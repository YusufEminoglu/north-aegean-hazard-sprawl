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

Optional tuning and revision-round scripts, each re-runnable once the processed layers above exist:

~~~bash
python src/p2_04b_rf_grid_search.py                  # 27-configuration RF grid search (manuscript Table S6)
python src/p2_11_dsi_masterplan_ingest.py            # DSI master-plan layers
python src/p2_12_wildfire_seismic_validation.py      # wildfire / seismic independent checks
python src/p2_13_uncertainty_propagation.py          # AHP-weight Monte Carlo
python src/p2_14_markov_demand_rerun.py              # Markov demand pathway
python src/p2_15_eco_penalty_grounding.py            # ECO penalty vs DSI buffers
python src/p2_16_ahp_full_verification.py            # four weighting schemes
python src/p2_17_wildfire_exclusion_sensitivity.py   # wildfire-exclusion sensitivity
python src/p2_18_prioritization_weight_sensitivity.py # prioritisation weight sensitivity
python src/p2_19_r4_diagnostics.py                   # gross/net relocation, data-gap and NbS bookkeeping
python src/as_run/p2_20_district_exposure.py         # district table (as-run folder)
python src/as_run/p2_21_ca_realisations.py           # allocation-stochasticity check (needs the model object produced by p2_04)
~~~

**Scope of reproducibility.** The scripts exactly as executed to generate the reported layers are in <code>src/as_run/</code> (see its README); <code>src/</code> holds a cleaned, seeded re-implementation of the same method that reproduces the reported pattern but not the exact pixel allocation. The RF grid-search driver and the sensitivity and diagnostic scripts are included. Provider rasters and intermediate layers are not redistributed, so the pipeline can be re-run only after the inputs are obtained as described in <code>data_sources.md</code>; the figures in <code>figures/png</code> and the compact tables are the deposited results.

Some provider-specific exports are split into <code>p2_00b</code>–<code>p2_00d</code>. Run only the exporters required for inputs you do not already have.

## 5. Regenerate figures

Figure scripts are in <code>src/p2_fig01_*.py</code> through <code>src/p2_fig13_*.py</code>. They expect completed processed outputs. Final authored PNGs are included in <code>figures/png</code> for inspection without third-party inputs.

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

The audit also asserts that exactly thirteen author figures are present and that no PDF, manuscript source, model object, raw raster, private Earth Engine identifier, or credential-like string is publishable.
