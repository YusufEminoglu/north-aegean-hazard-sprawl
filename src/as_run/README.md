# As-run scripts

This folder holds the analysis scripts exactly as they were executed to generate the layers, tables, and figures reported in the manuscript (revision R4). Paths resolve relative to the repository root, so run them from there with the staged data directories described in `docs/reproducibility.md`.

Not included here: the Earth Engine export scripts `p2_00*` (the sanitised versions in `src/` are the ones to use) and `p2_12_wildfire_seismic_validation.py` (its sanitised twin is `src/p2_12_wildfire_seismic_validation.py`).

## How `src/` differs from `src/as_run/`

The scripts in `src/` are a cleaned re-implementation of the same method, kept for reuse. They reproduce the reported pattern, not the exact pixel allocation:

- **`p2_04_flus_ca_engine.py`.** The as-run allocation step draws unseeded random numbers, so the reported BAU and ECO rasters are one realisation. Eight runs with different seeds and the RF surface held fixed (`p2_21_ca_realisations.py`) give ECO high-hazard shares of 34.9-35.0% (BAU 36.1% in every run), ERR 3.0-3.3%, and a net class 4-5 reduction of 107-116 ha. The seeded `src/` engine gives one different realisation (BAU 36.9%, ECO 35.2%, ERR 4.7%, 172 ha net; Jaccard overlap of the new-growth footprint with the as-run raster about 0.6).
- **`p2_05_flood_hazard.py`.** Same method in both folders (six-parameter Model A without HAND; 0.6/0.4 weighted average of the two 1-5 scores).
- **Other scripts** differ in style and in some defensive checks; the as-run versions are the reference for any number in the manuscript.

## New in R4

- `p2_19_r4_diagnostics.py` re-derives the exposure keys read by the manuscript (and prints new gap, NbS, and relocation quantities).
- `p2_20_district_exposure.py` writes the district table.
- `p2_21_ca_realisations.py` measures the spread across random realisations of the allocation step (it needs the persisted production model `data/04_models/paper2/rf_flus_model.pkl`).
