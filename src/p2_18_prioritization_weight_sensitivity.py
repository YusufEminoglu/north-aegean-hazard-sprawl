"""Prioritisation-score weight-sensitivity check (R3, Reviewer 2, point 6).

Reviewer 2 (Aug review, point 6): the no-build/TDR prioritisation score
(Supplementary Figure S1) equally weights hazard severity, development
pressure, population exposure, and implementation feasibility (inverse NTL).
The reviewer asks that this be presented as an exploratory planning construct
and that the authors "either provide a brief weight-sensitivity assessment or
explicitly acknowledge that the resulting priority area depends on normative
weighting and a proxy-based representation of feasibility."

This script recomputes the composite score under four alternative weight
vectors (the equal-weight scheme already reported, plus three schemes that
each place double weight on one criterion, renormalised to sum to 1) over the
identical candidate footprint (new BAU growth, hazard class >= 4; the
candidate set itself does not depend on the score weights), and reports the
top-quartile ("first-tier priority") area and its spatial overlap (IoU)
against the equal-weight footprint under each alternative.

Usage:
  python src/p2_18_prioritization_weight_sensitivity.py
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.stats import spearmanr

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
MODELS = p2_09.PROJECT_ROOT / "data" / "04_models" / "paper2"

CRITERIA = ["hazard", "pressure", "pop", "feasibility"]
SCHEMES = {
    "equal":              {"hazard": 0.25, "pressure": 0.25, "pop": 0.25, "feasibility": 0.25},
    "hazard_heavy":       {"hazard": 0.40, "pressure": 0.20, "pop": 0.20, "feasibility": 0.20},
    "pop_heavy":          {"hazard": 0.20, "pressure": 0.20, "pop": 0.40, "feasibility": 0.20},
    "feasibility_heavy":  {"hazard": 0.20, "pressure": 0.20, "pop": 0.20, "feasibility": 0.40},
}


def norm01(arr, mask):
    v = arr[mask]
    v = v[np.isfinite(v)]
    if len(v) == 0 or v.max() == v.min():
        return np.zeros_like(arr)
    out = np.zeros_like(arr, dtype=np.float32)
    out[mask] = np.clip((arr[mask] - v.min()) / (v.max() - v.min()), 0, 1)
    return out


def iou(mask_a, mask_b):
    inter = int((mask_a & mask_b).sum())
    union = int((mask_a | mask_b).sum())
    return inter / union if union > 0 else float("nan")


def main():
    print("== Prioritisation-score weight-sensitivity check (Reviewer 2, point 6) ==")
    pixel_ha = (30 * 30) / 10000.0

    ref_path = PROCESSED / "multi_hazard_surface.tif"
    with rasterio.open(ref_path) as ref:
        ref_meta = ref.meta.copy()
        hazard = ref.read(1).astype(np.float32)
        nodata = ref.nodata
    valid = (hazard != nodata)

    with rasterio.open(PROCESSED / "lulc_simplified_2020.tif") as s:
        lulc2020 = s.read(1)
    with rasterio.open(PROCESSED / "sim_2050_bau.tif") as s:
        sim_bau = s.read(1)
    base_urban = (lulc2020 == 1) & valid
    new_bau = (sim_bau == 1) & (~base_urban) & valid
    candidate = new_bau & (hazard >= 4)
    print(f"  Candidate pixels (fixed across all schemes): {int(candidate.sum()):,} "
          f"({int(candidate.sum()) * pixel_ha:,.0f} ha)")

    drivers, _ = p2_04.load_drivers()
    with open(MODELS / "rf_flus_model.pkl", "rb") as f:
        rf = pickle.load(f)
    dev_pressure = np.zeros(hazard.shape, np.float32)
    m = (lulc2020 > 0) & (lulc2020 != 4)
    dev_pressure[m] = rf.predict_proba(drivers[m])[:, 1]

    def load_aligned(path):
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

    pop = load_aligned(INTERIM / "p2_hrsl_pop_general_clipped.tif")
    ntl = load_aligned(PROCESSED / "drivers" / "driver_ntl.tif")

    criteria_n = {
        "hazard": norm01(hazard, candidate),
        "pressure": norm01(dev_pressure, candidate),
        "pop": norm01(pop, candidate),
        "feasibility": 1.0 - norm01(ntl, candidate),
    }

    footprints = {}
    scores = {}
    for name, w in SCHEMES.items():
        score = np.full(hazard.shape, np.nan, dtype=np.float32)
        score[candidate] = sum(criteria_n[c][candidate] * w[c] for c in CRITERIA)
        q75 = float(np.nanpercentile(score[candidate], 75))
        top_tier = candidate & (score >= q75)
        footprints[name] = top_tier
        scores[name] = score
        ha = int(top_tier.sum()) * pixel_ha
        print(f"  [{name:18s}] weights={w}  top-tier: {int(top_tier.sum()):,} px "
              f"({ha:,.0f} ha, {ha/(int(candidate.sum())*pixel_ha)*100:.1f}% of candidates)")

    print("\n  Overlap of each alternative's top-tier footprint against the equal-weight footprint:")
    for name in SCHEMES:
        if name == "equal":
            continue
        ov = iou(footprints["equal"], footprints[name])
        print(f"    equal vs {name:18s} IoU = {ov:.3f}")

    print("\n  Spearman rank correlation of pixel-level composite scores among candidates:")
    cand_idx = np.where(candidate)
    pairs = [("equal", "hazard_heavy"), ("equal", "pop_heavy"), ("equal", "feasibility_heavy")]
    corr_out = {}
    for a, b in pairs:
        rho, _ = spearmanr(scores[a][cand_idx], scores[b][cand_idx])
        corr_out[f"{a}_vs_{b}"] = rho
        print(f"    {a} vs {b}: rho = {rho:.3f}")

    print("\n== key,value summary ==")
    print(f"prioritization_ws_hazard_heavy_ha,{int(footprints['hazard_heavy'].sum())*pixel_ha:.0f}")
    print(f"prioritization_ws_pop_heavy_ha,{int(footprints['pop_heavy'].sum())*pixel_ha:.0f}")
    print(f"prioritization_ws_feasibility_heavy_ha,{int(footprints['feasibility_heavy'].sum())*pixel_ha:.0f}")
    print(f"prioritization_ws_iou_hazard_heavy,{iou(footprints['equal'], footprints['hazard_heavy']):.3f}")
    print(f"prioritization_ws_iou_pop_heavy,{iou(footprints['equal'], footprints['pop_heavy']):.3f}")
    print(f"prioritization_ws_iou_feasibility_heavy,{iou(footprints['equal'], footprints['feasibility_heavy']):.3f}")
    print(f"prioritization_ws_rho_min,{min(corr_out.values()):.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
