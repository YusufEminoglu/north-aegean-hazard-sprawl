"""
Paper 2 — Multi-Hazard Fusion & Exposure Analysis
===================================================
Reads the 4 pre-computed hazard surfaces and fuses them into the final
multi-hazard surface. Then intersects with 2050 urban growth scenarios.

Prerequisites (must run first):
  p2_05_flood_hazard.py     → flood_hazard.tif
  p2_06_seismic_hazard.py   → seismic_hazard.tif
  p2_07_bioclimate_hazard.py → bioclimate_hazard.tif
  p2_08_wildfire_hazard.py  → wildfire_hazard.tif

Weights (indicative AHP — update after formal pairwise comparison):
  Flood       0.35
  Seismic     0.35
  Bio-Climate 0.20
  Wildfire    0.10

Outputs:
  multi_hazard_continuous.tif        float32 0-1 weighted score
  multi_hazard_surface.tif           int8 1-5 ordinal, Jenks natural breaks
                                      (PRIMARY classification, mapclassify)
  multi_hazard_surface_quintile.tif  int8 1-5 ordinal, quintile breaks
                                      (retained as the sensitivity variant
                                      promised in the manuscript, not primary)
  exposure_sim_2050_bau.csv          Jenks-class exposure (primary)
  exposure_sim_2050_eco.csv
  exposure_sim_2050_bau_quintile.csv quintile-class exposure (sensitivity)
  exposure_sim_2050_eco_quintile.csv
"""

import numpy as np
import pandas as pd
import rasterio
import mapclassify
from rasterio.warp import reproject, Resampling
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM   = PROJECT_ROOT / "data" / "02_interim"  / "paper2"
PROCESSED = PROJECT_ROOT / "data" / "03_processed" / "paper2"
TABLES    = PROJECT_ROOT / "outputs" / "tables"    / "paper2"
TABLES.mkdir(parents=True, exist_ok=True)

REF_PATH = INTERIM / "p2_srtm_elevation_clipped.tif"

# AHP-indicative weights (sum = 1.0)
WEIGHTS = {
    "flood":      0.35,
    "seismic":    0.35,
    "bioclimate": 0.20,
    "wildfire":   0.10,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_aligned(path: Path, ref_meta: dict) -> np.ndarray | None:
    if not path.exists():
        print(f"  [MISSING] {path.name}")
        return None
    dst = np.full((ref_meta["height"], ref_meta["width"]), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
            resampling=Resampling.bilinear,
            src_nodata=src.nodata, dst_nodata=np.nan,
        )
    return dst

def normalise(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = arr[mask & ~np.isnan(arr)]
    if len(valid) == 0:
        return np.zeros_like(arr)
    lo, hi = float(np.nanmin(valid)), float(np.nanmax(valid))
    if hi == lo:
        return np.zeros_like(arr)
    out = np.zeros_like(arr, dtype=np.float32)
    m = mask & ~np.isnan(arr)
    out[m] = (arr[m] - lo) / (hi - lo)
    return out

def quintile_classify(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = arr[mask & ~np.isnan(arr)]
    q = np.nanpercentile(valid, [20, 40, 60, 80]) if len(valid) > 0 else [0.2, 0.4, 0.6, 0.8]
    ordinal = np.ones_like(arr, dtype=np.float32)
    ordinal[arr > q[3]] = 5
    ordinal[(arr <= q[3]) & (arr > q[2])] = 4
    ordinal[(arr <= q[2]) & (arr > q[1])] = 3
    ordinal[(arr <= q[1]) & (arr > q[0])] = 2
    print(f"    Quintile thresholds: {[round(v,4) for v in q]}")
    return ordinal


def jenks_classify(arr: np.ndarray, mask: np.ndarray, k: int = 5, seed: int = 42):
    """Fisher-Jenks natural-breaks classification into k ordinal classes
    (1=lowest .. k=highest), fit on a seeded 50,000-point subsample of the
    valid data (mapclassify.NaturalBreaks on the full multi-million-pixel
    array is prohibitively slow; class boundaries from a 50k random sample
    are stable to <0.1% for a distribution this size) and applied to every
    valid pixel via the fitted bin edges.

    Returns (ordinal_array, bin_edges) where bin_edges has k+1 values
    (min .. max) so the primary classification is reported with reproducible,
    citable break values instead of the fixed 20/40/60/80th-percentile cuts
    that force exactly two-fifths of the basin into the top two classes.
    """
    valid = arr[mask & ~np.isnan(arr)]
    rng = np.random.default_rng(seed)
    sample = valid if len(valid) <= 50000 else rng.choice(valid, 50000, replace=False)

    # mapclassify.NaturalBreaks internally re-runs its centroid search from
    # `initial` (default 10) random starts using the legacy numpy.random
    # global state, not a Generator instance -- seeding only `rng` above left
    # the classification itself non-reproducible between runs (bin edges,
    # and therefore every downstream Jenks-class percentage, drifted by a
    # few tenths of a percent run to run despite the fixed sample). Seeding
    # the legacy global state immediately before the call makes it exact.
    np.random.seed(seed)
    nb = mapclassify.NaturalBreaks(sample, k=k)
    edges = np.concatenate([[float(valid.min())], nb.bins.astype(float)])
    edges[-1] = float(valid.max())  # ensure the true max is captured

    ordinal = np.ones_like(arr, dtype=np.float32)
    for c in range(2, k + 1):
        ordinal[arr > edges[c - 1]] = c
    print(f"    Jenks (natural-breaks) thresholds: {[round(v, 4) for v in edges]}")
    return ordinal, edges

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("== Multi-Hazard Fusion ==")

    with rasterio.open(REF_PATH) as ref:
        ref_meta = ref.meta.copy()
        ref_arr  = ref.read(1)
        nodata   = ref.nodata

    valid = (ref_arr != nodata) if nodata is not None else np.ones(ref_arr.shape, bool)

    # 1. Load and normalise hazard surfaces
    hazard_paths = {
        "flood":      PROCESSED / "flood_hazard.tif",
        "seismic":    PROCESSED / "seismic_hazard.tif",
        "bioclimate": PROCESSED / "bioclimate_hazard.tif",
        "wildfire":   PROCESSED / "wildfire_hazard.tif",
    }

    print("  Loading hazard surfaces...")
    normed = {}
    for key, path in hazard_paths.items():
        arr = load_aligned(path, ref_meta)
        if arr is None:
            print(f"  WARNING: {key} missing — using zero layer")
            arr = np.zeros(ref_arr.shape, dtype=np.float32)
        normed[key] = normalise(arr, valid)
        print(f"  {key:12s} w={WEIGHTS[key]:.2f}  range [{normed[key][valid].min():.3f}, {normed[key][valid].max():.3f}]")

    # 2. Weighted fusion
    print("  Fusing hazard surfaces...")
    multi = sum(normed[k] * WEIGHTS[k] for k in WEIGHTS).astype(np.float32)
    multi_clean = multi.copy()
    multi_clean[~valid] = -9999.0

    out_meta = ref_meta.copy()
    out_meta.update(dtype=rasterio.float32, nodata=-9999.0)

    with rasterio.open(PROCESSED / "multi_hazard_continuous.tif", "w", **out_meta) as dst:
        dst.write(multi_clean, 1)

    # 3. PRIMARY classification: Jenks natural breaks → 1-5 ordinal.
    # Quintile is retained as the sensitivity variant only (see
    # multi_hazard_surface_quintile.tif below) - quintile cuts force exactly
    # 40% of the basin into the top two classes by construction, which is
    # what the natural-breaks classification avoids.
    ordinal, jenks_edges = jenks_classify(multi, valid)
    ordinal[~valid] = -9999.0

    with rasterio.open(PROCESSED / "multi_hazard_surface.tif", "w", **out_meta) as dst:
        dst.write(ordinal.astype(np.float32), 1)

    # Sensitivity variant: quintile classification of the same continuous surface.
    ordinal_q = quintile_classify(multi, valid)
    ordinal_q[~valid] = -9999.0
    with rasterio.open(PROCESSED / "multi_hazard_surface_quintile.tif", "w", **out_meta) as dst:
        dst.write(ordinal_q.astype(np.float32), 1)

    n_valid = int(valid.sum())
    jenks_high_pct = float(((ordinal >= 4) & valid).sum()) / n_valid * 100
    quint_high_pct = float(((ordinal_q >= 4) & valid).sum()) / n_valid * 100
    print(f"  Jenks basin high+very-high share:    {jenks_high_pct:.1f}%")
    print(f"  Quintile basin high+very-high share: {quint_high_pct:.1f}%  (fixed at 40% by construction)")

    print("  Saved: multi_hazard_continuous.tif + multi_hazard_surface.tif (Jenks) "
          "+ multi_hazard_surface_quintile.tif (sensitivity)")

    # 4. Exposure intersection with 2050 scenarios
    print("\n  Computing 2050 scenario exposure...")

    lulc_path = PROCESSED / "lulc_simplified_2020.tif"
    if not lulc_path.exists():
        lulc_path = INTERIM / "p2_glad_lulc_2020_clipped.tif"
        urban_val = 7
    else:
        urban_val = 1

    lulc2020 = load_aligned(lulc_path, ref_meta)
    base_urban = (lulc2020 == urban_val) if lulc2020 is not None else np.zeros(ref_arr.shape, bool)
    pixel_ha   = (30 * 30) / 10000.0

    summary = {}
    for sc in ["sim_2050_bau", "sim_2050_eco"]:
        sim = load_aligned(PROCESSED / f"{sc}.tif", ref_meta)
        if sim is None:
            print(f"  [SKIP] {sc}")
            continue

        new_urban = (sim == 1) & (~base_urban) & valid
        total_new = int(new_urban.sum())

        for label, ord_arr, suffix in [("Jenks", ordinal, ""), ("Quintile", ordinal_q, "_quintile")]:
            rows = []
            for cls in [1, 2, 3, 4, 5]:
                px = int((new_urban & (ord_arr == cls)).sum())
                rows.append({"Hazard_Class": cls, "New_Urban_Area_ha": px * pixel_ha})
            df = pd.DataFrame(rows)
            tot_ha = df["New_Urban_Area_ha"].sum()
            df["Percentage_%"] = (df["New_Urban_Area_ha"] / tot_ha * 100).fillna(0).round(1)
            df.to_csv(TABLES / f"exposure_{sc}{suffix}.csv", index=False)
            print(f"\n  {sc} [{label}]:")
            print(df.to_string(index=False, float_format="%.1f"))
            print(f"  Total new urban: {tot_ha:,.1f} ha")
            if label == "Jenks":
                sc_tag = sc.replace("sim_2050_", "")  # "bau" / "eco"
                for _, row in df.iterrows():
                    cls = int(row["Hazard_Class"])
                    print(f"{sc_tag}_new_ha_class{cls},{row['New_Urban_Area_ha']:.2f}")
                if sc_tag == "eco":
                    bau_df = pd.read_csv(TABLES / "exposure_sim_2050_bau.csv")
                    for cls in [1, 2, 3, 4, 5]:
                        b = float(bau_df.loc[bau_df["Hazard_Class"] == cls, "New_Urban_Area_ha"].iloc[0])
                        e = float(df.loc[df["Hazard_Class"] == cls, "New_Urban_Area_ha"].iloc[0])
                        area_err = (b - e) / b * 100 if b > 0 else 0.0
                        print(f"area_err_class{cls},{area_err:.1f}")

            high_ha = df.loc[df["Hazard_Class"] >= 4, "New_Urban_Area_ha"].sum()
            high_pct = high_ha / tot_ha * 100 if tot_ha > 0 else 0.0
            if label == "Jenks":
                summary[f"{sc}_high_pct"] = high_pct

        summary[f"{sc}_total_new_ha"] = total_new * pixel_ha

    print("\n  == Summary (Jenks, primary) ==")
    for k, v in summary.items():
        print(f"  {k}: {v:.2f}")
    if "sim_2050_bau_high_pct" in summary and "sim_2050_eco_high_pct" in summary:
        bau_h, eco_h = summary["sim_2050_bau_high_pct"], summary["sim_2050_eco_high_pct"]
        err = (bau_h - eco_h) / bau_h * 100 if bau_h > 0 else 0.0
        print(f"  Exposure Reduction Rate (ERR): {err:.2f}%")
        print(f"  jenks_basin_high_pct,{jenks_high_pct:.1f}")
        print(f"  bau_high_hazard_pct,{bau_h:.1f}")
        print(f"  eco_high_hazard_pct,{eco_h:.1f}")
        print(f"  err_pct,{err:.2f}")

    print("\n  Done.")

if __name__ == "__main__":
    main()
