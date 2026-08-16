"""Markov-chain demand: full BAU/ECO CA rerun.

The default pipeline (p2_04_flus_ca_engine.py) projects 2050 urban demand by
linear extrapolation of the 2000-2020 increment. That same module also
implements an independent, non-linear demand pathway (`markov_demand`,
selectable via P2_DEMAND_METHOD=markov): the 2000-2020 class-transition
matrix raised to a fractional power and applied to the 2020 class totals.

Whether ERR and HCI are invariant to the choice of absolute demand level is
an empirical question, not something that can be asserted from the linear
run alone. This script reruns BAU/ECO under the Markov demand (reusing the
exact suitability model and scenario-mask logic from p2_04) and reports
high-hazard exposure share, HCI, and ERR under both demand levels side by
side.

Usage:
  python src/p2_14_markov_demand_rerun.py
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "03_processed" / "paper2"
MODELS = ROOT / "data" / "04_models" / "paper2"
DRIVERS_DIR = PROC / "drivers"

_spec = importlib.util.spec_from_file_location(
    "p2_04", Path(__file__).with_name("p2_04_flus_ca_engine.py"))
p2_04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2_04)

PIXEL_HA = 0.09  # 30 m x 30 m


def high_hazard_exposure(hazard_ordinal, base_urban, sim, valid):
    new_urban = (sim == 1) & (~base_urban) & valid
    total = int(new_urban.sum())
    high = int((new_urban & (hazard_ordinal >= 4)).sum())
    return (high / total * 100) if total > 0 else 0.0, total * PIXEL_HA


def build_scenario_masks(lulc_2020):
    """Reproduce p2_04.main()'s BAU/ECO mask construction exactly."""
    bau_mask = (lulc_2020 != 4).astype(np.float32)
    eco_mask = bau_mask.copy()
    eco_mask[np.isin(lulc_2020, [2, 3])] = 0.1
    river_path = DRIVERS_DIR / "driver_river_dist.tif"
    if river_path.exists():
        with rasterio.open(river_path) as source:
            river_distance = source.read(1)
        eco_mask[river_distance < 500] *= 0.1
    return bau_mask, eco_mask


def main() -> None:
    print("== Markov-demand BAU/ECO rerun ==")

    with rasterio.open(PROC / "lulc_simplified_2000.tif") as src:
        lulc_2000 = src.read(1)
        meta = src.meta.copy()
    with rasterio.open(PROC / "lulc_simplified_2020.tif") as src:
        lulc_2020 = src.read(1)

    urban_2020 = int(np.sum(lulc_2020 == 1))
    markov_px = p2_04.markov_demand(lulc_2000, lulc_2020)
    markov_new_px = markov_px - urban_2020
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

    bau_mask, eco_mask = build_scenario_masks(lulc_2020)

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

    basin_high_pct = float(((hazard_ordinal >= 4) & valid).sum()) / float(valid.sum()) * 100

    hci_bau_mk = bau_high_mk / basin_high_pct
    hci_eco_mk = eco_high_mk / basin_high_pct
    err_mk = (bau_high_mk - eco_high_mk) / bau_high_mk * 100 if bau_high_mk > 0 else 0.0

    print("\n== Markov-demand results (compare against linear-demand exposure) ==")
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
