"""Full AHP verification: primary matrix, sub-component matrices, and
genuinely computed rank-based / PCA-derived sensitivity schemes.

Closes a gap flagged in docs/methods.md: scripts/verify_ahp.py already
verifies the primary 4x4 fusion matrix, but the flood/seismic sub-component
matrices and the W3 ("rank-based")/W4 ("PCA-derived") sensitivity schemes in
data/ahp/{subcomponent_weights,weight_schemes,sensitivity_correlations}.csv
were asserted values with no generating computation anywhere in this
repository. This script provides one.

  A. Sub-component matrices. The flood Model A (7 factors), flood Model B
     (5 factors) and seismic (3 factors) sub-weights are already published
     (hardcoded in p2_05/p2_06). Given a target weight vector w, the pairwise
     matrix a_ij = w_i / w_j is the *unique* perfectly consistent Saaty
     matrix reproducing it exactly (CR=0 by construction) - a standard AHP
     technique for stating a judgment matrix consistent with an adopted
     weight vector, not a fresh independent elicitation, and reported as such.

  B. W3 (rank-based): Rank-Order-Centroid (Barron & Barrett, 1996) weights
     over a rank ordering of the four hazard components by independent-
     validation strength (flood AUC=0.782 vs GFD > seismic AUC=0.588 vs USGS
     epicentres > bio-climatic, no independent validation available >
     wildfire AUC=0.370, anti-correlated).

  C. W4 (PCA-derived): first principal component of the four normalised
     hazard rasters (sampled), absolute loadings renormalised to sum to 1.

  D. Spearman/Kendall rank correlation between all six scheme pairs on a real
     pixel sample of the fused continuous surfaces.

Usage:
  python src/p2_16_ahp_full_verification.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("p2_09", HERE / "p2_09_multi_hazard_fusion.py")
p2_09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2_09)

PROCESSED = p2_09.PROCESSED
REF_PATH = p2_09.REF_PATH
RNG = np.random.default_rng(42)
RANDOM_INDEX = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}


def reconstruct_and_verify(name: str, factor_names: list[str], target_weights: list[float]):
    w = np.array(target_weights, dtype=float)
    w = w / w.sum()
    n = len(w)
    matrix = np.outer(w, 1.0 / w)

    geo_means = np.prod(matrix, axis=1) ** (1.0 / n)
    recovered = geo_means / geo_means.sum()
    lambda_max = float(np.mean((matrix @ recovered) / recovered))
    ci = (lambda_max - n) / (n - 1) if n > 2 else 0.0
    ri = RANDOM_INDEX.get(n, 1.32)
    cr = ci / ri if ri else 0.0

    print(f"\n  {name} ({n} factors):")
    for fname, wt in zip(factor_names, recovered):
        print(f"    {fname:<22} {wt:.4f}")
    print(f"    lambda_max={lambda_max:.4f}  CI={ci:.5f}  CR={cr:.5f}")
    if not np.allclose(recovered, w, atol=1e-6):
        raise ValueError(f"{name}: recovery failed")
    return matrix, recovered, lambda_max, ci, cr


def run_subcomponent_verification() -> None:
    print("[A] Sub-component pairwise matrix reconstruction + verification")
    flood_a = reconstruct_and_verify(
        "Flood Model A",
        ["HAND", "Slope", "River_dist", "Elevation", "Clay", "Max_precip", "LULC_perm"],
        [0.216, 0.209, 0.209, 0.105, 0.105, 0.105, 0.051],
    )
    flood_b = reconstruct_and_verify(
        "Flood Model B",
        ["Water_dist", "Elevation", "TPI", "NDVI", "NDWI"],
        [0.30, 0.25, 0.20, 0.15, 0.10],
    )
    seismic = reconstruct_and_verify(
        "Seismic", ["Fault_proximity", "PGA", "Site_amplification"], [0.40, 0.40, 0.20],
    )
    print(f"\nfloodA_cr,{flood_a[4]:.5f}")
    print(f"floodB_cr,{flood_b[4]:.5f}")
    print(f"seismic_cr,{seismic[4]:.5f}")


def roc_weights(n: int) -> np.ndarray:
    """Rank-Order Centroid weights for n ranked items (rank 1 = most important)."""
    return np.array([sum(1.0 / k for k in range(i, n + 1)) / n for i in range(1, n + 1)])


def run_w3_w4_and_correlations() -> None:
    print("\n[B/C/D] W3 (rank-based) + W4 (PCA-derived) + real scheme correlations")

    with rasterio.open(REF_PATH) as ref:
        ref_meta = ref.meta.copy()
        ref_arr = ref.read(1)
        nodata = ref.nodata
    valid = (ref_arr != nodata) if nodata is not None else np.ones(ref_arr.shape, bool)

    keys = ["flood", "seismic", "bioclimate", "wildfire"]
    normed = {}
    for key in keys:
        arr = p2_09.load_aligned(PROCESSED / f"{key}_hazard.tif", ref_meta)
        normed[key] = p2_09.normalize(arr, valid)

    n_valid = int(valid.sum())
    sample_idx = RNG.choice(n_valid, min(200_000, n_valid), replace=False)
    rows, cols = np.where(valid)
    srows, scols = rows[sample_idx], cols[sample_idx]
    X = np.column_stack([np.nan_to_num(normed[k][srows, scols], nan=0.0) for k in keys])

    rank_order = ["flood", "seismic", "bioclimate", "wildfire"]
    roc = roc_weights(4)
    w3 = {k: float(w) for k, w in zip(rank_order, roc)}
    print("\n  W3 (Rank-Order Centroid, validation-strength rank):")
    for k in keys:
        print(f"    {k:<12} {w3[k]:.4f}")

    pca = PCA(n_components=1)
    pca.fit(X)
    loadings = np.abs(pca.components_[0])
    w4 = {k: float(w) for k, w in zip(keys, loadings / loadings.sum())}
    print(f"\n  W4 (PCA, PC1 explains {pca.explained_variance_ratio_[0] * 100:.1f}% of variance):")
    for k in keys:
        print(f"    {k:<12} {w4[k]:.4f}")

    print(f"\nw3_flood,{w3['flood']:.4f}")
    print(f"w3_seismic,{w3['seismic']:.4f}")
    print(f"w3_bioclimate,{w3['bioclimate']:.4f}")
    print(f"w3_wildfire,{w3['wildfire']:.4f}")
    print(f"w4_flood,{w4['flood']:.4f}")
    print(f"w4_seismic,{w4['seismic']:.4f}")
    print(f"w4_bioclimate,{w4['bioclimate']:.4f}")
    print(f"w4_wildfire,{w4['wildfire']:.4f}")
    print(f"pca_pc1_variance_pct,{pca.explained_variance_ratio_[0] * 100:.1f}")

    schemes = {
        "W1_AHP": {"flood": 0.35, "seismic": 0.35, "bioclimate": 0.20, "wildfire": 0.10},
        "W2_Equal": {"flood": 0.25, "seismic": 0.25, "bioclimate": 0.25, "wildfire": 0.25},
        "W3_Rank": w3,
        "W4_PCA": w4,
    }
    fused = {name: sum(X[:, i] * w[k] for i, k in enumerate(keys)) for name, w in schemes.items()}

    names = list(schemes.keys())
    print("\n  Pairwise Spearman rho / Kendall tau (200k-pixel sample):")
    rho_min, tau_min = 1.0, 1.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho, _ = spearmanr(fused[names[i]], fused[names[j]])
            tau, _ = kendalltau(fused[names[i]], fused[names[j]])
            rho_min, tau_min = min(rho_min, rho), min(tau_min, tau)
            print(f"    {names[i]:<9} vs {names[j]:<9} rho={rho:.3f}  tau={tau:.3f}")
            print(f"corr_{names[i]}_vs_{names[j]}_rho,{rho:.3f}")
            print(f"corr_{names[i]}_vs_{names[j]}_tau,{tau:.3f}")

    print(f"\nsensitivity_rho_min,{rho_min:.3f}")
    print(f"sensitivity_tau_min,{tau_min:.3f}")


if __name__ == "__main__":
    run_subcomponent_verification()
    run_w3_w4_and_correlations()
    print("\nDone.")
