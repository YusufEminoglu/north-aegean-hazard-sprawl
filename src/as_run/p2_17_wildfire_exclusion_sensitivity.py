"""
Paper 2 -- Wildfire-exclusion sensitivity analysis (R3, Reviewer 2, Point 1)
=============================================================================
Reviewer 2 (Aug review, point 1): the wildfire composite validates with
ROC-AUC = 0.370 against independent MODIS/VIIRS FIRMS detections (i.e. it is
anti-correlated with observed fire occurrence), so it may represent standing
fuel/vegetation structure rather than observed ignition probability. Because
it still contributes 0.10 of the primary multi-hazard fusion weight, the
reviewer asks for "a sensitivity analysis excluding the wildfire component"
that reports "whether the principal BAU-ECO comparison, high-hazard exposure
shares and policy-priority areas remain materially unchanged."

This script builds a second, 3-component (flood/seismic/bio-climatic) fused
hazard surface with the wildfire term dropped and the remaining three AHP
weights renormalised proportionally (0.35/0.9, 0.35/0.9, 0.20/0.9), re-fits a
fresh Jenks classification on that surface (its own class edges -- reusing
the primary surface's edges would silently import the wildfire term's
influence on where the breaks fall), and recomputes:
  (a) basin-wide high+very-high share,
  (b) BAU / ECO high-hazard exposure share, ERR, and HCI,
  (c) the no-build/TDR prioritisation candidate and top-tier footprints
      (Supplementary Figure S1's rule, applied to the 3-component ordinal),
      including their spatial overlap (IoU) against the primary, wildfire-
      included footprints.

All under the identical BAU/ECO simulation rasters, RF suitability surface,
population and night-time-light layers used everywhere else in the study --
only the hazard input to the fusion and classification changes.

Usage:
  .venv/Scripts/python.exe src/paper2_hazard_sprawl/p2_17_wildfire_exclusion_sensitivity.py
"""

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

HERE = Path(__file__).resolve().parent


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p2_04 = _load_module("p2_04", "p2_04_flus_ca_engine.py")
p2_09 = _load_module("p2_09", "p2_09_multi_hazard_fusion.py")

PROCESSED = p2_09.PROCESSED
INTERIM = p2_09.INTERIM
REF_PATH = p2_09.REF_PATH
MODELS = HERE.resolve().parents[1] / "data" / "04_models" / "paper2"

# Primary AHP weights (unchanged) minus wildfire, renormalised to sum to 1.
PRIMARY_WEIGHTS = p2_09.WEIGHTS  # {"flood":0.35, "seismic":0.35, "bioclimate":0.20, "wildfire":0.10}
_remaining = {k: v for k, v in PRIMARY_WEIGHTS.items() if k != "wildfire"}
_denom = sum(_remaining.values())
WEIGHTS_NO_WILDFIRE = {k: v / _denom for k, v in _remaining.items()}


def build_ordinal(weights, hazard_keys):
    """Fuse the given hazard components under `weights` and Jenks-classify
    the result fresh (own class edges). Returns (ordinal, valid_mask,
    ref_meta, hazard_continuous)."""
    with rasterio.open(REF_PATH) as ref:
        ref_meta = ref.meta.copy()
        ref_arr = ref.read(1)
        nodata = ref.nodata
    valid = (ref_arr != nodata) if nodata is not None else np.ones(ref_arr.shape, bool)

    hazard_paths = {
        "flood": PROCESSED / "flood_hazard.tif",
        "seismic": PROCESSED / "seismic_hazard.tif",
        "bioclimate": PROCESSED / "bioclimate_hazard.tif",
        "wildfire": PROCESSED / "wildfire_hazard.tif",
    }
    normed = {}
    for key in hazard_keys:
        arr = p2_09.load_aligned(hazard_paths[key], ref_meta)
        normed[key] = p2_09.normalise(arr, valid)

    multi = sum(normed[k] * weights[k] for k in hazard_keys).astype(np.float32)
    ordinal, edges = p2_09.jenks_classify(multi, valid, k=5, seed=42)
    return ordinal, valid, ref_meta, multi, edges


def exposure_summary(ordinal, valid, ref_meta, tag):
    """Replicates p2_09's exposure-intersection logic for one hazard ordinal,
    returns dict of basin/BAU/ECO high-hazard shares, ERR, HCI."""
    lulc2020 = p2_09.load_aligned(PROCESSED / "lulc_simplified_2020.tif", ref_meta)
    base_urban = (lulc2020 == 1)
    pixel_ha = (30 * 30) / 10000.0
    n_valid = int(valid.sum())
    basin_high_pct = float(((ordinal >= 4) & valid).sum()) / n_valid * 100

    out = {"basin_high_pct": basin_high_pct}
    for sc in ["sim_2050_bau", "sim_2050_eco"]:
        sim = p2_09.load_aligned(PROCESSED / f"{sc}.tif", ref_meta)
        new_urban = (sim == 1) & (~base_urban) & valid
        tot = int(new_urban.sum())
        high_ha = int((new_urban & (ordinal >= 4)).sum()) * pixel_ha
        tot_ha = tot * pixel_ha
        high_pct = high_ha / tot_ha * 100 if tot_ha > 0 else 0.0
        key = "bau" if sc.endswith("bau") else "eco"
        out[f"{key}_high_pct"] = high_pct
        out[f"{key}_hci"] = high_pct / basin_high_pct if basin_high_pct > 0 else float("nan")

    out["err_pct"] = (
        (out["bau_high_pct"] - out["eco_high_pct"]) / out["bau_high_pct"] * 100
        if out["bau_high_pct"] > 0 else 0.0
    )
    out["err_pp"] = out["bau_high_pct"] - out["eco_high_pct"]
    print(f"\n  [{tag}] basin high+very-high: {out['basin_high_pct']:.1f}%  "
          f"BAU: {out['bau_high_pct']:.1f}% (HCI={out['bau_hci']:.2f})  "
          f"ECO: {out['eco_high_pct']:.1f}% (HCI={out['eco_hci']:.2f})  "
          f"ERR={out['err_pct']:.1f}% ({out['err_pp']:.1f} pp)")
    return out


def prioritisation_footprint(ordinal, valid, ref_meta):
    """Reimplements p2_fig11's four-criterion score for one hazard ordinal,
    returns (candidate_mask, top_tier_mask)."""
    with rasterio.open(PROCESSED / "lulc_simplified_2020.tif") as s:
        lulc2020 = s.read(1)
    with rasterio.open(PROCESSED / "sim_2050_bau.tif") as s:
        sim_bau = s.read(1)
    base_urban = (lulc2020 == 1) & valid
    new_bau = (sim_bau == 1) & (~base_urban) & valid
    candidate = new_bau & (ordinal >= 4)

    drivers, _ = p2_04.load_drivers()
    with open(MODELS / "rf_flus_model.pkl", "rb") as f:
        rf = pickle.load(f)
    dev_pressure = np.zeros(ordinal.shape, np.float32)
    m = (lulc2020 > 0) & (lulc2020 != 4)
    dev_pressure[m] = rf.predict_proba(drivers[m])[:, 1]

    def load_aligned_bilinear(path):
        # nodata-aware (matches the fix applied to p2_fig11_prioritization.py):
        # declare the source nodata so bilinear resampling does not blend it
        # into neighbouring valid cells, then treat a genuine data gap as 0.
        dst = np.full((ref_meta["height"], ref_meta["width"]), np.nan, dtype=np.float32)
        with rasterio.open(path) as src:
            reproject(source=rasterio.band(src, 1), destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
                      resampling=Resampling.bilinear, src_nodata=src.nodata, dst_nodata=np.nan)
        return np.where(np.isfinite(dst), dst, 0.0).astype(np.float32)

    pop = load_aligned_bilinear(INTERIM / "p2_hrsl_pop_general_clipped.tif")
    ntl = load_aligned_bilinear(PROCESSED / "drivers" / "driver_ntl.tif")

    def norm01(arr, mask):
        v = arr[mask]
        v = v[np.isfinite(v)]
        if len(v) == 0 or v.max() == v.min():
            return np.zeros_like(arr)
        out = np.zeros_like(arr, dtype=np.float32)
        out[mask] = np.clip((arr[mask] - v.min()) / (v.max() - v.min()), 0, 1)
        return out

    if candidate.sum() == 0:
        return candidate, candidate.copy()

    hazard_n = norm01(ordinal, candidate)
    pressure_n = norm01(dev_pressure, candidate)
    pop_n = norm01(pop, candidate)
    feas_n = 1.0 - norm01(ntl, candidate)

    score = np.full(ordinal.shape, np.nan, dtype=np.float32)
    score[candidate] = (hazard_n[candidate] + pressure_n[candidate] +
                         pop_n[candidate] + feas_n[candidate]) / 4.0
    q75 = float(np.nanpercentile(score[candidate], 75))
    top_tier = candidate & (score >= q75)
    return candidate, top_tier


def iou(mask_a, mask_b):
    inter = int((mask_a & mask_b).sum())
    union = int((mask_a | mask_b).sum())
    return inter / union if union > 0 else float("nan")


def main():
    print("== Wildfire-exclusion sensitivity analysis (Reviewer 2, Point 1) ==")
    pixel_ha = (30 * 30) / 10000.0

    print("\n[1] Primary 4-component surface (flood/seismic/bioclimate/wildfire, "
          f"weights={PRIMARY_WEIGHTS}) ...")
    ord_primary, valid, ref_meta, _, edges_primary = build_ordinal(
        PRIMARY_WEIGHTS, ["flood", "seismic", "bioclimate", "wildfire"])
    summary_primary = exposure_summary(ord_primary, valid, ref_meta, "4-component (primary)")

    print("\n[2] 3-component surface excluding wildfire "
          f"(renormalised weights={ {k: round(v,4) for k,v in WEIGHTS_NO_WILDFIRE.items()} }) ...")
    ord_nowf, valid2, ref_meta2, _, edges_nowf = build_ordinal(
        WEIGHTS_NO_WILDFIRE, ["flood", "seismic", "bioclimate"])
    summary_nowf = exposure_summary(ord_nowf, valid2, ref_meta2, "3-component (wildfire excluded)")

    print("\n[3] Prioritisation footprint under each hazard surface ...")
    cand_primary, top_primary = prioritisation_footprint(ord_primary, valid, ref_meta)
    cand_nowf, top_nowf = prioritisation_footprint(ord_nowf, valid2, ref_meta2)

    cand_ha_primary = int(cand_primary.sum()) * pixel_ha
    cand_ha_nowf = int(cand_nowf.sum()) * pixel_ha
    top_ha_primary = int(top_primary.sum()) * pixel_ha
    top_ha_nowf = int(top_nowf.sum()) * pixel_ha
    cand_iou = iou(cand_primary, cand_nowf)
    top_iou = iou(top_primary, top_nowf)

    print(f"  Candidate area: primary={cand_ha_primary:,.0f} ha, "
          f"no-wildfire={cand_ha_nowf:,.0f} ha, IoU={cand_iou:.3f}")
    print(f"  Top-tier area:  primary={top_ha_primary:,.0f} ha, "
          f"no-wildfire={top_ha_nowf:,.0f} ha, IoU={top_iou:.3f}")

    print("\n== mydata.dat keys ==")
    print(f"nowf_jenks_edges,{'/'.join(f'{v:.4f}' for v in edges_nowf)}")
    print(f"nowf_basin_high_pct,{summary_nowf['basin_high_pct']:.1f}")
    print(f"nowf_bau_high_pct,{summary_nowf['bau_high_pct']:.1f}")
    print(f"nowf_eco_high_pct,{summary_nowf['eco_high_pct']:.1f}")
    print(f"nowf_hci_bau,{summary_nowf['bau_hci']:.2f}")
    print(f"nowf_hci_eco,{summary_nowf['eco_hci']:.2f}")
    print(f"nowf_err_pct,{summary_nowf['err_pct']:.1f}")
    print(f"nowf_err_pp,{summary_nowf['err_pp']:.1f}")
    print(f"primary_err_pp,{summary_primary['err_pp']:.1f}")
    print(f"nowf_candidate_ha,{cand_ha_nowf:.0f}")
    print(f"nowf_top_tier_ha,{top_ha_nowf:.0f}")
    print(f"nowf_candidate_iou,{cand_iou:.3f}")
    print(f"nowf_top_tier_iou,{top_iou:.3f}")

    print("\nDone. Compare nowf_* keys against the primary (wildfire-included) "
          "values already in mydata.dat (basin_high_pct/jenks_basin_high_pct, "
          "bau_high_hazard_pct, eco_high_hazard_pct, hci_bau, hci_eco, err_pct, "
          "prioritization_candidate_ha, prioritization_top_tier_ha).")


if __name__ == "__main__":
    main()
