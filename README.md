<p align="center">
  <img src="docs/assets/banner.svg" alt="North Aegean Hazard–Sprawl" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img alt="Code: MIT" src="https://img.shields.io/badge/code-MIT-54c5a8?style=flat-square"></a>
  <a href="LICENSE-docs"><img alt="Figures and docs: CC BY 4.0" src="https://img.shields.io/badge/figures%20%26%20docs-CC%20BY%204.0-efb366?style=flat-square"></a>
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-78dce8?style=flat-square">
  <img alt="Resolution 30 metres" src="https://img.shields.io/badge/analysis%20grid-30%20m-d4ef9c?style=flat-square">
  <img alt="Publication audit" src="https://img.shields.io/badge/publication%20audit-pass-54c5a8?style=flat-square">
</p>

# Future spatial footprints of natural hazards

This is the open research companion to a study of how urban expansion may intersect flood, seismic, bio-climatic, and wildfire hazard across the North Aegean Basin, Türkiye. It connects a 37-year land-cover record to an RF–FLUS-style cellular automaton, tests two demand-equivalent 2050 planning scenarios, and turns the resulting exposure patterns into planning-ready evidence.

> The manuscript is under review at Urban Climate. No article DOI has been assigned. This repository contains code, author-created figures, and compact parameter tables—not the manuscript, publisher PDF, or third-party geospatial data.

<table>
  <tr>
    <td align="center"><strong>+312.5%</strong><br><sub>urban footprint, 1985–2022</sub></td>
    <td align="center"><strong>31.7%</strong><br><sub>BAU growth in high / very-high hazard</sub></td>
    <td align="center"><strong>5.4%</strong><br><sub>ECO exposure reduction ratio</sub></td>
    <td align="center"><strong>0.820</strong><br><sub>flood validation ROC AUC</sub></td>
    <td align="center"><strong>ρ ≥ 0.960</strong><br><sub>cross-scheme spatial robustness</sub></td>
  </tr>
</table>

<p align="center">
  <img src="figures/png/fig10_Policy_Synthesis.png" alt="Policy synthesis map and intervention framework" width="94%">
</p>

## Why this repository matters

The scientific contribution is not another single-hazard map. It is the explicit coupling of observed urbanisation, forward land allocation, independently constructed hazard components, and demographic/agricultural asset exposure on one harmonised 30 m grid.

- Historical urban land increased from 5,814.81 ha to 23,988.24 ha.
- BAU allocates 3,126 ha of projected new growth to high or very-high hazard zones.
- ECO redirects 170 ha of the most exposed growth, yet 30.0% remains in high-hazard corridors.
- Jenks natural breaks replace fixed quintiles, so the high-hazard share is learned from the surface.
- AHP, equal, rank-based, and PCA-derived fusion schemes remain strongly concordant.
- GFD validation uses threshold-independent ROC AUC and reports the maximum-CSI operating point separately.

## The analytical chain

~~~mermaid
flowchart LR
    A["1985–2022 LULC<br>+ 11 spatial drivers"] --> B["RF urban<br>suitability"]
    B --> C{"Seeded CA<br>same 2050 demand"}
    C --> D["BAU"]
    C --> E["ECO"]
    F["Flood"] --> J["AHP fusion<br>+ Jenks classes"]
    G["Seismic"] --> J
    H["Bio-climate"] --> J
    I["Wildfire"] --> J
    D --> K["Spatial exposure<br>intersection"]
    E --> K
    J --> K
    K --> L["No-build · TDR · NbS<br>planning priorities"]
~~~

Method details are documented in [docs/methods.md](docs/methods.md), with the exact fusion matrix and published weight vectors in [docs/ahp_matrices.md](docs/ahp_matrices.md).

## Reproduce the workflow

The fastest local setup uses Conda/Mamba:

~~~bash
git clone https://github.com/YOUR-ORG/north-aegean-hazard-sprawl.git
cd north-aegean-hazard-sprawl
mamba env create -f environment.yml
mamba activate north-aegean-hazard-sprawl
python scripts/verify_ahp.py
python scripts/publication_audit.py
~~~

For Earth Engine exports, copy <code>.env.example</code> to your own untracked environment configuration and set <code>EE_PROJECT</code>, <code>GEE_DRIVE_FOLDER</code>, and either <code>EE_ROI_ASSET</code> or a local boundary GeoJSON. Then follow the staged recipe in [docs/reproducibility.md](docs/reproducibility.md).

Third-party rasters are deliberately absent. [docs/data_sources.md](docs/data_sources.md) links each provider and records the redistribution boundary.

## Repository map

~~~text
north-aegean-hazard-sprawl/
├── src/                    analysis, GEE export, validation, and figure code
├── data/ahp/               versioned author-created weights and fusion matrix
├── figures/png/            ten final author-generated figures
├── scripts/                AHP verification and pre-publication audit
├── docs/                   methods, provenance, reproducibility, project site
├── CITATION.cff            software and preferred article citation metadata
├── environment.yml         geospatial Conda environment
└── requirements.txt        pinned Python environment
~~~

## Visual atlas

| Urban dynamics | 2050 scenarios |
|---|---|
| [![LULC dynamics](figures/png/fig02_Dynamics.png)](figures/png/fig02_Dynamics.png) | [![CA scenarios](figures/png/fig05_CA_Scenarios.png)](figures/png/fig05_CA_Scenarios.png) |

| Multi-hazard convergence | Weight sensitivity |
|---|---|
| [![Multi-hazard convergence](figures/png/fig06_MultiHazard_Convergence.png)](figures/png/fig06_MultiHazard_Convergence.png) | [![Sensitivity](figures/png/fig07_Sensitivity.png)](figures/png/fig07_Sensitivity.png) |

| Exposure intersection | Demographic exposure |
|---|---|
| [![Exposure](figures/png/fig08_Exposure.png)](figures/png/fig08_Exposure.png) | [![Demographic exposure](figures/png/fig09_Demographic_Exposure.png)](figures/png/fig09_Demographic_Exposure.png) |

## Reproducibility boundary

This repository makes the analysis logic inspectable, but it cannot legally or responsibly mirror every provider raster. Reproduction therefore has two levels:

1. Code-level reproduction: environment, AHP verification, workflow order, parameters, and authored figures are complete here.
2. Data-level rerun: users obtain the listed datasets under their original terms, authenticate with their own Earth Engine account, and populate ignored staging directories.

The only full pairwise matrix released here is the four-component fusion matrix available in the archived revision package. Sub-component final weights and reported consistency ratios are included without reverse-engineering absent judgement matrices. See [docs/ahp_matrices.md](docs/ahp_matrices.md).

## Citation

Use the repository’s “Cite this repository” control, powered by [CITATION.cff](CITATION.cff). Until the article is accepted, please treat the preferred article metadata as “under review” and do not invent a DOI.

## Licence and attribution

Code is MIT licensed. Documentation and author-generated figures are CC BY 4.0. Input datasets retain their providers’ terms and are not redistributed. See [NOTICE.md](NOTICE.md) and [docs/data_sources.md](docs/data_sources.md).

Created by Yusuf Eminoğlu and Kemal Mert Çubukçu at Dokuz Eylül University / LUQAA.
