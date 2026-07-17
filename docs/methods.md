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

The retained grid-search configuration is:

| Parameter | Value |
|---|---:|
| Trees | 100 |
| Maximum depth | 10 |
| Minimum samples to split | 5 |
| Class weighting | Balanced |
| Random seed | 42 |

Distance-to-existing-urban is intentionally retained as an agglomeration proximity prior, but its circularity in a cross-sectional classifier is explicit. The saturated cross-validated AUC is therefore treated as a diagnostic, not evidence of forecast skill. Temporal CA hindcasting is the allocation test; the reported Figure of Merit is 66.3%.

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

Flood Model A uses HAND (0.216), slope (0.209), river distance (0.209), elevation (0.105), clay (0.105), maximum precipitation (0.105), and LULC permeability (0.051). Model B uses water distance (0.30), elevation (0.25), TPI (0.20), NDVI (0.15), and NDWI (0.10).

The continuous fusion surface is classified into five ordinal classes with Jenks natural breaks computed over the full valid basin grid. High and very-high classes cover 35.6% of the basin; that share is an empirical result rather than the fixed 40% produced by quintiles.

## Stage IV — exposure intersection

Newly allocated BAU and ECO urban pixels are intersected with the hazard classes. The main outcome is the share and area of new urban growth in high or very-high hazard zones.

| Outcome | BAU | ECO |
|---|---:|---:|
| New urban area | 9,866.7 ha | 9,866.7 ha |
| High / very-high exposure | 3,126 ha (31.7%) | 2,956 ha (30.0%) |
| Very-high exposure | 2,020 ha | 1,774 ha |

The ECO scenario redirects 170 ha of high-hazard growth, an Exposure Reduction Ratio of 5.4%. The limited reduction motivates hazard-explicit zoning in addition to ecological constraints.

## Validation and robustness

- Flood susceptibility is tested against MODIS Global Flood Database inundation using ROC AUC (0.820).
- The maximum-CSI operating point is reported separately: POD 0.137, FAR 0.860, CSI 0.074.
- GFD prevalence rises from 0.152% in class 1 to 3.827% in class 5, a 25.2-fold enrichment.
- Fusion robustness is evaluated with AHP, equal, rank-based, and PCA-derived schemes.
- The minimum pairwise Spearman correlation is 0.960; minimum Kendall τ is 0.835.

## Determinism

RF and CA stochastic components use seed 42. Geospatial results still depend on the exact provider vintages and the GDAL/PROJ stack, so the environment files are pinned and provenance should be recorded for each rerun.
