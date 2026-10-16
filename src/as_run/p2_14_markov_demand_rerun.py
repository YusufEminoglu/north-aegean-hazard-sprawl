"""
Paper 2 — Markov-chain demand: full BAU/ECO CA rerun
========================================================
Reviewer 2 (Aug review, point 4): "the statement that changing absolute
demand simply rescales both scenario footprints and leaves HCI and ERR
invariant is not generally guaranteed ... please rerun BAU and ECO using at
least the alternative Markov demand and report HCI, ERR and high-hazard
exposure under both demand levels."

This was previously an unverified claim (no script ever reran the CA under
the Markov demand). This script:
  1. Projects the 2000->2020 transition-probability matrix forward to 2050
     (fractional matrix power, same method as p2_fig02's 25-year projection)
     to get an independent, non-linear urban-demand estimate.
  2. Reruns the full BAU/ECO FLUS-CA simulation at that demand (reusing the
     exact suitability model and scenario masks from p2_04).
  3. Recomputes high-hazard exposure share, HCI and ERR under this demand
     using the already-fitted Jenks multi-hazard classification (the hazard
     surface and its classification do not depend on urban demand), and
     reports them next to the linear-demand results for direct comparison.

Usage:
  .venv/Scripts/python.exe src/paper2_hazard_sprawl/p2_14_markov_demand_rerun.py
"""

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.linalg import fractional_matrix_power

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "03_processed" / "paper2"
MODELS = ROOT / "data" / "04_models" / "paper2"
TABLES = ROOT / "outputs" / "tables" / "paper2"

spec = importlib.util.spec_from_file_location(
    "p2_04", Path(__file__).with_name("p2_04_flus_ca_engine.py"))
p2_04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2_04)

PIXEL_HA = (30 * 30) / 10000.0


def markov_2050_urban_pixels(lulc_2020):
    tpm_df = pd.read_csv(TABLES / "transition_probability_matrix.csv", index_col=0)
    classes = tpm_df.index.tolist()  # Urban, Agriculture, Forest/Shrub, Water, Bareland, Other
    T = tpm_df.values.astype(float)
    T_30 = fractional_matrix_power(T, 30 / 20).real  # 2020 -> 2050
    T_30 = np.clip(T_30, 0, None)
    T_30 = T_30 / T_30.sum(axis=1, keepdims=True)

    total_px = int((lulc_2020 > 0).sum())
    s0 = np.array([float((lulc_2020 == i + 1).sum()) / total_px for i in range(len(classes))])
    s0 = s0 / s0.sum()
    s2050 = s0 @ T_30

    urban_share_2050 = float(s2050[classes.index("Urban")])
    urban_px_2050 = int(round(urban_share_2050 * total_px))
    return urban_px_2050, urban_share_2050, T_30


def high_hazard_exposure(hazard_ordinal, base_urban, sim, valid):
    new_urban = (sim == 1) & (~base_urban) & valid
    total = int(new_urban.sum())
    high = int((new_urban & (hazard_ordinal >= 4)).sum())
    return (high / total * 100) if total > 0 else 0.0, total * PIXEL_HA


def main():
    print("== Markov-demand BAU/ECO rerun ==")

    with rasterio.open(PROC / "lulc_simplified_2000.tif") as src:
        lulc_2000 = src.read(1)
        meta = src.meta.copy()
    with rasterio.open(PROC / "lulc_simplified_2020.tif") as src:
        lulc_2020 = src.read(1)

    urban_2020 = int(np.sum(lulc_2020 == 1))
    markov_px, markov_share, _ = markov_2050_urban_pixels(lulc_2020)
    markov_new_px = markov_px - urban_2020
    print(f"  Markov-projected 2050 urban share: {markov_share*100:.2f}%")
    print(f"  Markov-projected 2050 urban pixels: {markov_px:,}  "
          f"(new growth: {markov_new_px:,} px = {markov_new_px*PIXEL_HA:,.1f} ha)")
    print(f"markov_demand_new_ha,{markov_new_px*PIXEL_HA:.1f}")

    if markov_new_px <= 0:
        print("  Markov demand <= current urban extent - nothing to simulate.")
        return

    drivers, _ = p2_04.load_drivers()
    with open(MODELS / "rf_flus_model.pkl", "rb") as f:
        rf = pickle.load(f)
    valid = (lulc_2020 > 0) & (lulc_2020 != 4)
    suitability = np.zeros(lulc_2020.shape, np.float32)
    suitability[valid] = rf.predict_proba(drivers[valid])[:, 1]

    bau_mask, eco_mask = p2_04.build_scenario_masks(lulc_2020)

    with rasterio.open(PROC / "multi_hazard_surface.tif") as src:
        hazard_ordinal = src.read(1)
    base_urban = (lulc_2020 == 1) & valid

    print("\n--- BAU (Markov demand) ---")
    sim_bau_mk = p2_04.run_ca_simulation(lulc_2020, suitability, urban_2020, markov_px, bau_mask)
    with rasterio.open(PROC / "sim_2050_bau_markov.tif", "w", **meta) as out:
        out.write(sim_bau_mk, 1)
    bau_high_mk, bau_total_mk = high_hazard_exposure(hazard_ordinal, base_urban, sim_bau_mk, valid)

    print("\n--- ECO (Markov demand) ---")
    sim_eco_mk = p2_04.run_ca_simulation(lulc_2020, suitability, urban_2020, markov_px, eco_mask)
    with rasterio.open(PROC / "sim_2050_eco_markov.tif", "w", **meta) as out:
        out.write(sim_eco_mk, 1)
    eco_high_mk, eco_total_mk = high_hazard_exposure(hazard_ordinal, base_urban, sim_eco_mk, valid)

    with rasterio.open(PROC / "lulc_simplified_2020.tif") as src:
        pass
    valid_all = valid
    basin_high_pct = float(((hazard_ordinal >= 4) & valid_all).sum()) / float(valid_all.sum()) * 100

    hci_bau_mk = bau_high_mk / basin_high_pct
    hci_eco_mk = eco_high_mk / basin_high_pct
    err_mk = (bau_high_mk - eco_high_mk) / bau_high_mk * 100 if bau_high_mk > 0 else 0.0

    print("\n== Markov-demand results (compare against linear-demand exposure_sim_2050_*.csv) ==")
    print(f"  BAU total new growth: {bau_total_mk:,.1f} ha  |  high-hazard share: {bau_high_mk:.2f}%  |  HCI={hci_bau_mk:.3f}")
    print(f"  ECO total new growth: {eco_total_mk:,.1f} ha  |  high-hazard share: {eco_high_mk:.2f}%  |  HCI={hci_eco_mk:.3f}")
    print(f"  ERR (Markov demand): {err_mk:.2f}%")
    print(f"bau_high_hazard_pct_markov,{bau_high_mk:.2f}")
    print(f"eco_high_hazard_pct_markov,{eco_high_mk:.2f}")
    print(f"hci_bau_markov,{hci_bau_mk:.3f}")
    print(f"hci_eco_markov,{hci_eco_mk:.3f}")
    print(f"err_pct_markov,{err_mk:.2f}")

    print("\nDone. Compare *_markov keys against the linear-demand run to test the invariance claim.")


if __name__ == "__main__":
    main()
