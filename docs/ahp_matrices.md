# AHP parameters and consistency

The repository distinguishes between material that is fully recomputable and material reported from the revision archive.

## Fully specified fusion matrix

The four-component reciprocal judgement matrix is versioned at <code>data/ahp/fusion_pairwise.csv</code>.

|  | Flood | Seismic | Bio-climatic | Wildfire |
|---|---:|---:|---:|---:|
| Flood | 1 | 1 | 2 | 3 |
| Seismic | 1 | 1 | 2 | 3 |
| Bio-climatic | 1/2 | 1/2 | 1 | 2 |
| Wildfire | 1/3 | 1/3 | 1/2 | 1 |

The matrix geometric-mean priorities are 0.351, 0.351, 0.189, and 0.109. The operational vector reported and applied in the analysis is the nearby nominal set 0.35, 0.35, 0.20, and 0.10. The revision reports λmax = 4.010, CI = 0.003, and CR = 0.0038. Run:

~~~bash
python scripts/verify_ahp.py
~~~

The verifier reads the CSV, checks positivity and reciprocity, recomputes the priority vector and Saaty consistency ratio, and fails if the matrix or expected values drift.

## Sub-component weights

Final published weight vectors are stored in <code>data/ahp/subcomponent_weights.csv</code>.

| Model | Criteria and weights | Reported CR |
|---|---|---:|
| Flood A | HAND .216; slope .209; river distance .209; elevation .105; clay .105; maximum precipitation .105; permeability .051 | .001 |
| Flood B | water distance .30; elevation .25; TPI .20; NDVI .15; NDWI .10 | .012 |
| Seismic | fault proximity .40; PGA .40; site amplification .20 | .000 |

The archived revision material available when this repository was assembled contains these weight vectors and consistency ratios, but not every underlying reciprocal cell for the three sub-component matrices. Those missing judgements have not been reverse-engineered or fabricated. Consequently, users can reproduce the applied weighted combinations exactly, while only the top-level fusion consistency calculation is independently recomputable from a full matrix here.

## Sensitivity schemes

| Hazard | W1 AHP | W2 equal | W3 rank-based | W4 PCA |
|---|---:|---:|---:|---:|
| Flood | .35 | .25 | .39 | .07 |
| Seismic | .35 | .25 | .39 | .81 |
| Bio-climatic | .20 | .25 | .13 | .10 |
| Wildfire | .10 | .25 | .10 | .02 |

W3 follows the rank ordering of the primary weights. W4 is based on normalised absolute PC1 loadings; PC1 explains 64.9% of variance and loads chiefly on the seismic component. The printed two-decimal W3 values sum to 1.01 because of rounding; the implementation normalises every scheme to unit sum before fusion. The six cross-scheme comparisons are stored in <code>data/ahp/sensitivity_correlations.csv</code>.

## Formulae

For a reciprocal matrix A with n criteria, the geometric-mean priority is:

~~~text
g_i = (product over j of a_ij)^(1/n)
w_i = g_i / sum(g)
~~~

Consistency is evaluated as:

~~~text
CI = (lambda_max - n) / (n - 1)
CR = CI / RI_n
~~~

For n = 4, Saaty’s random index RI is 0.90. CR below 0.10 is the conventional acceptance threshold.
