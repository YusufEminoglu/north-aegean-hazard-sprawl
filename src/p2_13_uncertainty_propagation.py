"""Uncertainty propagation: AHP weights -> 2050 exposure estimates.

The discrete-scheme sensitivity analysis (p2_fig07_sensitivity.py) compares
four named weight schemes and reports their rank agreement - it does not
propagate weight uncertainty into a distribution on the reported exposure
percentages. This script does that directly:

  1. Draw N perturbed AHP weight vectors from a Dirichlet distribution
     centred on the indicative weights (flood 0.35, seismic 0.35,
     bioclimate 0.20, wildfire 0.10), with a concentration chosen so each
     weight varies by roughly +-10-15% of its value (a plausible range for
     AHP pairwise-comparison uncertainty at CR < 0.10).
  2. For each draw, fuse the 4 normalized hazard layers into a continuous
     surface and reclassify with the SAME Jenks edges fit on the baseline
     (indicative-weight) surface - this isolates how much classification
     outcome moves under weight uncertainty, holding the classification
     rule fixed, rather than conflating it with re-fitting noise.
  3. Recompute basin high+very-high share, BAU/ECO new-growth high-hazard
     share, ERR, and HCI for every draw; report the empirical 90% interval.

Usage:
  python src/p2_13_uncertainty_propagation.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "03_processed" / "paper2"
INTERIM = ROOT / "data" / "02_interim" / "paper2"

_spec = importlib.util.spec_from_file_location(
    "p2_09", Path(__file__).with_name("p2_09_multi_hazard_fusion.py"))
p2_09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2_09)

N_DRAWS = 500
CONCENTRATION = 50.0  # Dirichlet concentration: higher = tighter around base
BASE_WEIGHTS = np.array([0.35, 0.35, 0.20, 0.10])  # flood, seismic, bioclimate, wildfire
KEYS = ["flood", "seismic", "bioclimate", "wildfire"]


def main() -> None:
    print("== AHP weight -> exposure uncertainty propagation ==")
    rng = np.random.default_rng(42)

    with rasterio.open(p2_09.REF_PATH) as ref:
        ref_meta = ref.meta.copy()
        ref_arr = ref.read(1)
        nodata = ref.nodata
    valid = (ref_arr != nodata) if nodata is not None else np.ones(ref_arr.shape, bool)

    normed = {}
    for key in KEYS:
        arr = p2_09.load_aligned(PROC / f"{key}_hazard.tif", ref_meta)
        normed[key] = p2_09.normalize(arr, valid) if arr is not None else np.zeros(ref_arr.shape, np.float32)

    # Baseline continuous surface + Jenks edges (fixed classification rule).
    multi_base = sum(normed[k] * w for k, w in zip(KEYS, BASE_WEIGHTS)).astype(np.float32)
    _, edges = p2_09.jenks_classify(multi_base, valid)

    def classify_fixed_edges(arr):
        ordinal = np.ones_like(arr, dtype=np.float32)
        for c in range(2, 6):
            ordinal[arr > edges[c - 1]] = c
        return ordinal

    lulc_path = PROC / "lulc_simplified_2020.tif"
    lulc2020 = p2_09.load_aligned(lulc_path, ref_meta)
    base_urban = (lulc2020 == 1) & valid

    sims = {}
    for sc in ["sim_2050_bau", "sim_2050_eco"]:
        sim = p2_09.load_aligned(PROC / f"{sc}.tif", ref_meta)
        sims[sc] = (sim == 1) & (~base_urban) & valid if sim is not None else None

    n_valid = int(valid.sum())
    alpha = BASE_WEIGHTS * CONCENTRATION

    records = {"basin_high": [], "bau_high": [], "eco_high": [], "err": [],
               "hci_bau": [], "hci_eco": []}

    print(f"  Running {N_DRAWS} Dirichlet-perturbed weight draws (concentration={CONCENTRATION:.0f})...")
    for _ in range(N_DRAWS):
        w = rng.dirichlet(alpha)
        multi = sum(normed[k] * wk for k, wk in zip(KEYS, w)).astype(np.float32)
        ordinal = classify_fixed_edges(multi)

        basin_high = float(((ordinal >= 4) & valid).sum()) / n_valid * 100
        records["basin_high"].append(basin_high)

        highs = {}
        for sc in ["sim_2050_bau", "sim_2050_eco"]:
            mask = sims[sc]
            if mask is None:
                continue
            total = int(mask.sum())
            high = int((mask & (ordinal >= 4)).sum())
            highs[sc] = (high / total * 100) if total > 0 else 0.0

        if "sim_2050_bau" in highs and "sim_2050_eco" in highs:
            records["bau_high"].append(highs["sim_2050_bau"])
            records["eco_high"].append(highs["sim_2050_eco"])
            err = ((highs["sim_2050_bau"] - highs["sim_2050_eco"]) / highs["sim_2050_bau"] * 100
                   if highs["sim_2050_bau"] > 0 else 0.0)
            records["err"].append(err)
            records["hci_bau"].append(highs["sim_2050_bau"] / basin_high if basin_high > 0 else np.nan)
            records["hci_eco"].append(highs["sim_2050_eco"] / basin_high if basin_high > 0 else np.nan)

    print("\n  == Empirical 90% intervals over weight perturbation ==")
    for name, vals in records.items():
        if not vals:
            continue
        arr = np.array(vals)
        lo, med, hi = np.percentile(arr, [5, 50, 95])
        print(f"  {name:10s}: median={med:.2f}  90% CI=[{lo:.2f}, {hi:.2f}]  (n={len(arr)})")
        print(f"uncertainty_{name}_p5,{lo:.2f}")
        print(f"uncertainty_{name}_p50,{med:.2f}")
        print(f"uncertainty_{name}_p95,{hi:.2f}")

    print("\nDone. Transcribe key,value lines into docs/manuscripts/paper2/src/mydata.dat")


if __name__ == "__main__":
    main()
