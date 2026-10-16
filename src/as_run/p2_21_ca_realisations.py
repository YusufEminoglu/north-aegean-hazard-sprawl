"""
Paper 2 -- Allocation-stochasticity check for the ECO benefit (R4)
===================================================================
The CA allocation step perturbs each pixel's suitability with a random factor
(p2_04_flus_ca_engine.run_ca_simulation), so a single BAU/ECO pair is one
realisation. This script re-runs the BAU and ECO simulations N times with
different numpy seeds while holding the fitted RF suitability surface fixed
(the persisted production model data/04_models/paper2/rf_flus_model.pkl), and
reports the spread of the headline contrast metrics against the multi-hazard
surface. It writes nothing to data/ or outputs/figures/.

Prints `key,value` lines (ca_real_*).

Usage:
  .venv/Scripts/python.exe src/paper2_hazard_sprawl/p2_21_ca_realisations.py [N]
"""

import importlib.util
import pickle
import statistics as st
import sys
from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("p2_04", HERE / "p2_04_flus_ca_engine.py")
p2_04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2_04)

PIXEL_HA = 0.09
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def main():
    with rasterio.open(p2_04.PROCESSED / "multi_hazard_surface.tif") as s:
        hz = s.read(1)
        nd = s.nodata
    valid = hz != nd
    with rasterio.open(p2_04.PROCESSED / "lulc_simplified_2000.tif") as s:
        l00 = s.read(1)
    with rasterio.open(p2_04.PROCESSED / "lulc_simplified_2020.tif") as s:
        l20 = s.read(1)
    drivers, _ = p2_04.load_drivers()
    target, current = p2_04.calculate_2050_demand(l00, l20)
    with open(p2_04.MODELS / "rf_flus_model.pkl", "rb") as f:
        rf = pickle.load(f)
    suit = np.zeros(l20.shape, np.float32)
    m = (l20 > 0) & (l20 != 4)
    suit[m] = rf.predict_proba(drivers[m])[:, 1]
    bau_mask, eco_mask = p2_04.build_scenario_masks(l20)
    base = (l20 == 1) & valid
    high = hz >= 4

    rows = []
    for i in range(N):
        np.random.seed(1000 + i)
        b = p2_04.run_ca_simulation(l20, suit, current, target, bau_mask)
        e = p2_04.run_ca_simulation(l20, suit, current, target, eco_mask)
        nb = (b == 1) & ~base & valid
        ne = (e == 1) & ~base & valid
        hb, he = nb & high, ne & high
        rows.append(dict(
            seed=1000 + i,
            bau_pct=hb.sum() / nb.sum() * 100, eco_pct=he.sum() / ne.sum() * 100,
            err_pct=(hb.sum() - he.sum()) / hb.sum() * 100,
            err_pp=(hb.sum() / nb.sum() - he.sum() / ne.sum()) * 100,
            net_ha=(hb.sum() - he.sum()) * PIXEL_HA,
            bau_only=(hb & ~he).sum() * PIXEL_HA, eco_only=(he & ~hb).sum() * PIXEL_HA))
        print("  ", {k: round(float(v), 2) for k, v in rows[-1].items()})

    def rng(k):
        v = [r[k] for r in rows]
        return min(v), max(v), st.mean(v)

    print(f"ca_real_n,{N}")
    lo, hi, mean = rng("err_pct")
    print(f"ca_real_err_min,{lo:.1f}\nca_real_err_max,{hi:.1f}\nca_real_err_mean,{mean:.1f}")
    lo, hi, mean = rng("net_ha")
    print(f"ca_real_net_min,{lo:.0f}\nca_real_net_max,{hi:.0f}")
    lo, hi, mean = rng("bau_pct")
    print(f"ca_real_bau_min,{lo:.1f}\nca_real_bau_max,{hi:.1f}")
    lo, hi, mean = rng("eco_pct")
    print(f"ca_real_eco_min,{lo:.1f}\nca_real_eco_max,{hi:.1f}")


if __name__ == "__main__":
    main()
