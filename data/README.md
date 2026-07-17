# Data boundary

This repository does not redistribute third-party geospatial data.

Place provider downloads and Earth Engine exports in the ignored staging
directories below. The scripts use these paths consistently:

- data/01_raw/paper2 — original downloads, the study-boundary GeoJSON, and GEE exports
- data/02_interim/paper2 — clipped and harmonised rasters
- data/03_processed/paper2 — derived LULC, drivers, scenarios, and hazard surfaces
- data/04_models/paper2 — fitted RF model objects

Only data/ahp is versioned. It contains small author-created parameter tables
needed to verify the AHP fusion. The full provider catalogue, access links, and
licence notes are in ../docs/data_sources.md.

Before running any workflow, obtain each source from its provider and confirm
the terms that apply on your access date.
