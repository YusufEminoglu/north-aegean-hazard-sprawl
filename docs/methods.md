# Methods at a glance

This document maps the revised manuscript methods to the public implementation. It is a technical companion, not a replacement for the peer-reviewed article.

## Analysis frame

The study covers the approximately 9,860 km² North Aegean Basin in western Türkiye. Continuous layers are harmonised to a 30 m reference grid in EPSG:32635 using bilinear resampling; categorical and binary layers use nearest-neighbour resampling. Distances are calculated in the projected coordinate system.

## Stage I — observed urban dynamics

GLC-FCS30D provides the 1985–2022 land-cover trajectory. GLAD GLCLU 2000 and 2020 states support the suitability target and temporal hindcast. The urban footprint increased from 5,814.81 ha in 1985 to 23,988.24 ha in 2022 (+312.5%).

Relevant implementation:

- <code>src/p2_00_gee_batch_download.py</code> prepares provider exports.
- <code>src/p2_01_move_clip_validate.py</code> clips and validates inputs.
- <code>src/p2_02_lulc_timeseries.py</code> harmonises the LULC series.
- <code>src/p2_03_spatial_drivers.py</code> aligns eleven suitability drivers.

## Stage II — RF suitability and FLUS-style CA

A Random Forest models the 2020 urban configuration from eleven predictors: slope, elevation, distance to roads, rivers, water, coast, and existing urban land; population density; night-time lights; NDVI; and clay content. Predictors do not postdate the 2020 target state.

The production configuration used by `p2_04_flus_ca_engine.py::train_rf()` is:

| Parameter | Value |
|---|---:|
| Trees | 50 |
| Maximum depth | 10 |
| Minimum samples to split | 2 |
| Class weighting | None (natural ~2.5% urban share preserved) |
| Training sample | 50,000 pixels, uniform random, seed 42, not stratified |
| Random seed | 42 |

This configuration comes from a stratified 5-fold grid search over
`n_estimators` {50, 100, 200} x `max_depth` {10, 20, unbounded} x
`min_samples_split` {2, 5, 10} (27 configurations). The driver is
`src/p2_04b_rf_grid_search.py`; it writes the full ranked grid to
`outputs/tables/paper2/rf_hyperparameter_grid.csv` (manuscript Supplementary
Table S6). All 27 configurations saturate at cross-validated AUC = 1.0000
(spread below 1e-15) because distance-to-existing-urban nearly reproduces the
2020 urban mask, so the grid cannot discriminate among them and the smallest
forest (50 trees) is retained on parsimony grounds. Running the driver
requires the processed driver rasters, which are not redistributed (see
`data/README.md`).

Distance-to-existing-urban is intentionally retained as an agglomeration proximity prior, but its circularity in a cross-sectional classifier is explicit. The saturated cross-validated AUC is therefore treated as a diagnostic, not evidence of forecast skill. Temporal CA hindcasting is the allocation test.

**Hindcast variants:** `p2_10_model_validation.py` reports two hindcast variants, since the 2020-vintage distance-to-urban predictor used throughout the original hindcast is itself derived from the 2020 endpoint the hindcast is scored against (an endpoint-information leak). The **endpoint-informed** variant (kept for transparency, not treated as primary) reproduces the 66.3% Figure of Merit figure previously reported here. The **predictor-corrected** variant, which substitutes a distance-to-urban layer computed from the 2000 classification alone for this one predictor (see `p2_03_spatial_drivers.py::compute_urban_dist(year=2000)`), is the primary reported value (FoM 26.3%). Because the suitability model itself is still fitted on the 2020 cross-section, it is a retrospective allocation diagnostic with partial endpoint correction, not an independent forecast validation; rerun both functions in `p2_10_model_validation.py` to get current values for your own data vintage.

The CA combines RF suitability with a 3×3 Moore-neighbourhood term and a seeded stochastic perturbation. Exact demand is allocated in batches without threshold-tie over-allocation. BAU and ECO share the same demand. ECO penalises agricultural/forest conversion and riparian development.

The headline 2050 demand linearly extends the observed 2000–2020 urban-pixel increment. A Markov alternative raises the 20-year transition matrix to the 1.5 power, producing a 30-year operator. Set <code>P2_DEMAND_METHOD=markov</code> to use it; <code>linear</code> is the default.

## Stage III — multi-hazard construction

Four min–max-normalised components are fused:

| Component | Primary weight | Construction |
|---|---:|---|
| Flood | 0.35 | 0.60 physical-geomorphic + 0.40 remote-sensing model |
| Seismic | 0.35 | Fault proximity, PGA, site amplification |
| Bio-climatic stress | 0.20 | LST/UHI, NO₂, inverse summer precipitation |
| Wildfire | 0.10 | EVI, canopy structure, burned frequency |

**Fixed:** `p2_08_wildfire_hazard.py` previously reprojected its three inputs without declaring `src_nodata`/`dst_nodata`, so canopy height's numeric nodata sentinel (present at roughly 60% of source pixels — a coverage gap, not zero canopy) was blended by bilinear resampling and then silently scored as the *lowest* possible fuel-structure value. The reprojection and compositing now propagate nodata correctly and average only the sub-inputs that have data at each pixel. Independent validation against MODIS/VIIRS FIRMS thermal-anomaly detections after this fix still shows the composite anti-correlated with observed detections (a real ecological pattern: low-canopy/low-EVI shrubland registers more fire activity in this basin than closed forest, both because it is more flammable and because 1&nbsp;km-resolution thermal sensors resolve large shrub/grass fires better than sub-canopy fire); decomposed by sub-input, the historical-burn input (GABAM) is essentially indistinguishable from chance (AUC 0.524) and the EVI and canopy-height inputs are mildly anti-correlated (AUC 0.347 and 0.375). See the manuscript's methodology section for the full decomposition.

Flood Model A (physical-geomorphic) uses slope (0.30), river distance (0.30), elevation (0.15), clay (0.10), maximum precipitation (0.10), and LULC permeability (0.05); HAND is not an input. Model B (spectral) uses water distance (0.30), TPI (0.30), elevation (0.20), NDWI (0.15), and NDVI (0.05). The two 1-5 scores are fused as 0.6 A + 0.4 B without z-standardisation; the continuous score (1.00-4.64) is min-max normalised in the fusion step. `src/as_run/p2_05_flood_hazard.py` is the script that produced the surface used in the manuscript.

The continuous fusion surface is classified into five ordinal classes with Jenks natural breaks computed over the full valid basin grid. High and very-high classes cover 36.2% of the basin; that share is an empirical result rather than the fixed 40% produced by quintiles.

## Stage IV — exposure intersection

Newly allocated BAU and ECO urban pixels are intersected with the hazard classes. The main outcome is the share and area of new urban growth in high or very-high hazard zones.

| Outcome | BAU | ECO |
|---|---:|---:|
| New urban area | 9,866.7 ha | 9,866.7 ha |
| High / very-high exposure | 3,564 ha (36.1%) | 3,448 ha (34.9%) |
| Very-high exposure | 2,319 ha | 2,062 ha |

The ECO mask relocates 950 ha of BAU class 4-5 growth (gross) but allocates 833 ha to class 4-5 pixels that BAU leaves unconverted, so the net decrease is 116 ha (3,564.00 to 3,447.81 ha): 1.2 percentage points absolute, an Exposure Reduction Ratio of 3.3% (90% interval 0.88-9.64% across tested AHP-weight perturbations). The limited reduction motivates hazard-explicit zoning in addition to ecological constraints. `src/p2_19_r4_diagnostics.py` reproduces the gross/net decomposition.

Population exposure uses direct raster intersection of the 2018 HRSL grids with the new-growth pixels of each hazard class (no night-time-light redistribution). It is a static-demography exposure screen, not a 2050 population forecast: the BAU total of 135,207 people is the population recorded in 2018 within the BAU growth footprint across all hazard classes (48,385 in classes 4-5).

**Note:** the table reproduces the values reported in the revised manuscript (R4, October 2026), which come from the end-to-end run that includes the wildfire nodata fix and the predictor-corrected hindcast. Exact percentages can shift slightly with different data vintages.

## Validation and robustness

- Flood susceptibility is tested against MODIS Global Flood Database inundation using threshold-independent ROC AUC (0.782 in the revised manuscript, with GFD aligned to the hazard grid by bilinear resampling; the exact third decimal depends on the alignment method, so rerun `p2_10_model_validation.py::flood_validation()` for the authoritative value on your own data vintage).
- The maximum-CSI operating point is reported separately: POD 0.250, FAR 0.885, CSI 0.086.
- GFD prevalence rises from 0.482% in class 1 to 7.264% in class 5, a 15.1-fold enrichment.
- Fusion robustness is evaluated with AHP, equal, rank-based, and PCA-derived schemes (`data/ahp/weight_schemes.csv`, `sensitivity_correlations.csv`).
- The minimum pairwise Spearman correlation is 0.921; minimum Kendall τ is 0.796 (W3 rank-based vs. W4 PCA-derived, the two most divergent schemes).

**Resolved:** `src/p2_16_ahp_full_verification.py` now generates both files
from real computation rather than asserting them. W3 (rank-based) uses
Rank-Order-Centroid weights (Barron & Barrett, 1996) over a rank ordering of
the four components by the independent-validation strength established
elsewhere in this repository (flood > seismic > bio-climatic, unvalidated >
wildfire, anti-correlated). W4 (PCA-derived) takes the absolute, renormalised
loadings of the first principal component of the four normalised hazard
rasters (PC1 explains ~74% of variance and is dominated by seismic — a real,
somewhat striking result, since seismic hazard has by far the sharpest
spatial gradients of the four components). The same script also reconstructs
the flood Model A/B and seismic sub-component pairwise matrices via
`a_ij = w_i / w_j` from their already-published weight vectors — the unique
perfectly consistent Saaty matrix reproducing a given weight vector exactly
(CR = 0 by construction). This states the judgment matrix consistent with the
adopted sub-weights; it is not an independent expert re-elicitation, and
`data/ahp/subcomponent_weights.csv`'s `reported_CR` column is labelled
accordingly.

## Additional analyses

- `src/p2_11_dsi_masterplan_ingest.py` clips the DSİ (State Hydraulic Works) Kuzey Ege Havzası Master Plan layers (historical floods, delineated flood-prone areas, protected areas, water-source protection buffers, mapped faults) to the basin. The source shapefiles are restricted-distribution government data and are not included in this repository; set `DSI_SOURCE_DIR` to a local copy to run it.
- `src/p2_12_wildfire_seismic_validation.py` checks the wildfire and seismic hazard components against sources independent of their own inputs: GEE FIRMS thermal-anomaly detections (wildfire, including the per-sub-input GABAM/EVI/canopy-height decomposition referenced above) and the USGS historical earthquake catalogue (seismic, reported as epicentre density, not damage records).
- `src/p2_13_uncertainty_propagation.py` propagates AHP weight uncertainty (500 Dirichlet-perturbed draws around the indicative weights) through the fixed Jenks classification into a 90% interval on basin/BAU/ECO high-hazard share, HCI, and ERR, rather than reporting only the four discrete named schemes.
- `src/p2_14_markov_demand_rerun.py` reruns BAU/ECO under the Markov demand pathway end to end (not just projected, but actually simulated) and reports HCI/ERR/high-hazard exposure under both demand levels side by side, to test rather than assert demand-invariance.
- `src/p2_15_eco_penalty_grounding.py` checks how much of the DSİ's official water-source protection buffers and protected areas already fall inside the ECO scenario's penalised zones, and how much BAU vs. ECO new growth intrudes into them — a data-grounded check on the penalty multipliers independent of their assumed values.
- `src/p2_fig11_prioritization.py`, `p2_fig12_wildfire_subcomponents.py`, and `p2_fig13_location_map.py` generate the corresponding supplementary figures from the above.

`p2_fig07_sensitivity.py`'s W3/W4 weight vectors and its Spearman correlation-matrix inset are computed on the same 200,000-pixel SRTM-masked sample as `p2_16_ahp_full_verification.py` uses for `sensitivity_correlations.csv`, rather than a separately (and previously divergently) sampled full-basin computation.

## Determinism

RF and CA stochastic components use seed 42. Geospatial results still depend on the exact provider vintages and the GDAL/PROJ stack, so the environment files are pinned and provenance should be recorded for each rerun.
