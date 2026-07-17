# Data sources and redistribution boundary

No third-party raster or vector is redistributed in this repository. Obtain each input from the provider, record the version and access date, and apply the provider terms that are current for your use. The notes below were checked against provider pages on 18 July 2026; they are provenance guidance, not legal advice.

| Layer | Provider / period / nominal resolution | Role | Access and licence note |
|---|---|---|---|
| GLC-FCS30D LULC | Zhang et al.; 1985–2022; 30 m | Long-run land-cover trajectory | [Zenodo record 8239305](https://zenodo.org/records/8239305); record supplies version and licence metadata |
| GLAD GLCLU | University of Maryland GLAD; 2000, 2020; 30 m | 2020 RF target and CA hindcast | [GLCLUC2020 portal](https://glad.umd.edu/dataset/GLCLUC2020); follow the portal’s access and attribution instructions |
| CORINE Land Cover | Copernicus Land Monitoring Service; 1990–2018; 100 m | LULC transition cross-check | [CORINE overview](https://land.copernicus.eu/en/products/corine-land-cover?tab=overview); Copernicus data policy applies |
| ESA WorldCover | ESA; 2021; 10 m | Contemporary LULC reference | [WorldCover](https://esa-worldcover.org/en); consult product terms and citation guidance |
| SRTM terrain | NASA/USGS; static; 30 m | Elevation, slope and TPI | [USGS SRTM 1 Arc-Second Global](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1); US government source, attribution requested |
| MERIT Hydro / HAND | Yamazaki et al.; static; 90 m | Drainage and flood Model A | [Earth Engine catalogue](https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1); dual CC BY-NC 4.0 / ODbL 1.0 terms are documented by the catalogue—do not assume unrestricted commercial reuse |
| Global Surface Water v1.4 | EC JRC; 1984–2021; 30 m | Flood Model B and water exclusion | [JRC download portal](https://global-surface-water.appspot.com/download); free use with source attribution under the portal terms |
| Global Flood Database | Tellman et al.; 2000–2018; 250 m | Independent flood validation | [Earth Engine catalogue](https://developers.google.com/earth-engine/datasets/catalog/GLOBAL_FLOOD_DB_MODIS_EVENTS_V1); currently listed as CC BY-NC 4.0 |
| SoilGrids clay | ISRIC; static; 250 m | Flood and site-amplification input | [SoilGrids](https://soilgrids.org/) and [file service](https://files.isric.org/soilgrids/latest/); verify the release-specific licence |
| WorldClim v1.4 bio13 | WorldClim; climatology; ~1 km | Maximum monthly precipitation | [WorldClim v1.4](https://www.worldclim.org/data/v1.4/worldclim14.html); archived page identifies CC BY-SA 4.0, while general use conditions should also be checked |
| HRSL population | Meta / CIESIN; circa 2018; ~30 m | Demographic exposure and allocation | Search the [HDX catalogue](https://data.humdata.org/dataset/?q=High%20Resolution%20Population%20Density%20Maps) for the country release; dataset-specific HDX terms apply |
| VIIRS night-time lights | Earth Observation Group; 2020; ~500 m | RF accessibility/activity driver | [EOG VNL products](https://eogdata.mines.edu/products/vnl/); follow the requested EOG attribution and product conditions |
| Tree canopy height | Meta / WRI; 2020 product; high resolution | Wildfire fuel structure | [WRI dataset record](https://datasets.wri.org/datasets/meta-tree-canopy-height); verify vintage and licence on download |
| Landsat 8/9 | USGS; JJA 2014–2024; thermal native 100 m | Summer LST and UHI | [Landsat Collection 2](https://www.usgs.gov/landsat-missions/landsat-collection-2); USGS states Landsat data are no-cost and public domain |
| Sentinel-5P OFFL L3 NO₂ | Copernicus; 2019–2024; ~1 km analysis scale | Bio-climatic stress | [Earth Engine catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2); Copernicus Sentinel data terms apply |
| OpenLandMap SM2RAIN | OpenLandMap; climatology; ~1 km | Inverted summer precipitation stress | [Earth Engine catalogue](https://developers.google.com/earth-engine/datasets/catalog/OpenLandMap_CLM_CLM_PRECIPITATION_SM2RAIN_M_v01); CC BY-SA 4.0 |
| Sentinel-2 SR harmonised | Copernicus; JJA 2019–2024 | Summer EVI | [Earth Engine catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED); Copernicus Sentinel data terms apply |
| GABAM burned area | GABAM; 1990–2021 | Burn recurrence | [Community catalogue record](https://gee-community-catalog.org/projects/gabam/); follow cited source and catalogue terms |
| Active faults | GEM Global Active Faults / EMME-SHARE; static vector | Fault-proximity hazard | [GEM active-fault repository](https://github.com/cossatot/gem-global-active-faults); repository documents CC BY-SA terms |
| PGA | AFAD national model / ESHM20; static | Seismic ground motion | [ESHM20 overview](https://hazard.efehr.org/en/Documentation/specific-hazard-models/europe/eshm2020-overview/); follow EFEHR/AFAD access and citation requirements |
| GRIP road network | GLOBIO / GRIP4; static vector | Road accessibility | [GRIP download page](https://www.globio.info/download-grip-dataset); confirm product conditions at source |

## Processing services

Google Earth Engine is used for cloud-scale compositing and export. Users authenticate with their own account and Cloud project. Earth Engine dataset availability does not override the underlying provider licence.

## Why the data are absent

Several inputs restrict redistribution, have non-commercial or share-alike conditions, or are too large for a source repository. Keeping them out of Git also avoids presenting a stale mixed-vintage snapshot as canonical. The code expects users to rebuild a provenance-controlled local data stack.

## Citation ledger

For a rerun, record at minimum:

- provider and product title;
- version, DOI, asset ID, or catalogue ID;
- access date and original URL;
- temporal coverage and processing filters;
- licence/version observed on that date;
- any local reprojection, clipping, resampling, and nodata decisions.
