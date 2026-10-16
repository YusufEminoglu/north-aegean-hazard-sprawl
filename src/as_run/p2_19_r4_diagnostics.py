"""
Paper 2 -- R4 reviewer-response diagnostics (UCLIM-D-26-02395, October 2026)
=============================================================================
Reproduces, from the persisted rasters, four quantities requested in the R4
(October 2026) reports, using exactly the same masks and loaders as the
scripts that produced the previously reported figures:

  A. Gross relocation vs. net high-hazard reduction (Reviewer 2, Sec. 4.7):
     decomposes the 950 ha "risk avoided" (p2_fig10_policy.py) into BAU-only
     and ECO-only high-hazard pixels and reconciles it with the 116 ha net
     decrease in combined class 4-5 area (Table 8 / exposure table).
  B. Genuine data gaps vs. valid zeros in the prioritisation score
     (Reviewer 2, Sec. 3.5.3), plus alternative gap treatments.
  C. Eligible area and denominator behind the NbS "top quartile"
     (Reviewer 2, Sec. 4.7).
  D. How the HRSL population exposure is actually computed (Reviewer 2,
     Sec. 3.5.2 / Table 8): direct raster intersection, with gap bookkeeping.

Prints `key,value` lines for mydata.dat.

Usage:
  .venv/Scripts/python.exe src/paper2_hazard_sprawl/p2_19_r4_diagnostics.py
"""

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

HERE = Path(__file__).resolve().parent


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROJECT_ROOT = HERE.resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "02_interim" / "paper2"
PROCESSED = PROJECT_ROOT / "data" / "03_processed" / "paper2"
MODELS = PROJECT_ROOT / "data" / "04_models" / "paper2"
PIXEL_HA = (30 * 30) / 10000.0


def jaccard(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else float("nan")


def norm01(arr, mask):
    v = arr[mask]
    v = v[np.isfinite(v)]
    out = np.zeros_like(arr, dtype=np.float32)
    if len(v) == 0 or v.max() == v.min():
        return out
    out[mask] = np.clip((arr[mask] - v.min()) / (v.max() - v.min()), 0, 1)
    return out


def main():
    with rasterio.open(PROCESSED / "multi_hazard_surface.tif") as ref:
        ref_meta = ref.meta.copy()
        hazard = ref.read(1).astype(np.float32)
        nodata = ref.nodata
    valid = hazard != nodata
    H, W = hazard.shape

    with rasterio.open(PROCESSED / "lulc_simplified_2020.tif") as s:
        lulc2020 = s.read(1)
    with rasterio.open(PROCESSED / "sim_2050_bau.tif") as s:
        sim_bau = s.read(1)
    with rasterio.open(PROCESSED / "sim_2050_eco.tif") as s:
        sim_eco = s.read(1)
    base_urban = (lulc2020 == 1) & valid
    new_bau = (sim_bau == 1) & ~base_urban & valid
    new_eco = (sim_eco == 1) & ~base_urban & valid
    high = np.nan_to_num(np.where(valid, hazard, np.nan), nan=0) >= 4

    print("== A. Gross relocation vs net high-hazard reduction ==")
    bau_high = new_bau & high
    eco_high = new_eco & high
    shared_high = bau_high & eco_high
    bau_only_high = bau_high & ~eco_high      # == p2_fig10 'risk_avoided'
    eco_only_high = eco_high & ~bau_high      # countervailing allocations
    ha = lambda m: float(m.sum()) * PIXEL_HA
    print(f"  BAU new growth {ha(new_bau):,.2f} ha | ECO new growth {ha(new_eco):,.2f} ha "
          f"| shared (both scenarios convert the same pixel) {ha(new_bau & new_eco):,.2f} ha")
    print(f"  BAU class 4-5: {ha(bau_high):,.2f} ha | ECO class 4-5: {ha(eco_high):,.2f} ha | "
          f"net decrease {ha(bau_high) - ha(eco_high):,.2f} ha")
    print(f"  shared class 4-5 pixels: {ha(shared_high):,.2f} ha")
    print(f"  BAU-only class 4-5 (gross 'avoided'): {ha(bau_only_high):,.2f} ha")
    print(f"  ECO-only class 4-5 (countervailing):  {ha(eco_only_high):,.2f} ha")
    ident = ha(bau_only_high) - ha(eco_only_high) - (ha(bau_high) - ha(eco_high))
    print(f"  identity check (gross_avoided - countervailing - net) = {ident:.4f} ha (must be 0)")
    # Where BAU-only class 4-5 growth is relocated: ECO-only growth by class
    eco_only_all = new_eco & ~new_bau
    bau_only_all = new_bau & ~new_eco
    print(f"  all-class gross relocation: BAU-only {ha(bau_only_all):,.2f} ha "
          f"({ha(bau_only_all) / ha(new_bau) * 100:.1f}% of BAU growth), "
          f"ECO-only {ha(eco_only_all):,.2f} ha")
    for k in range(1, 6):
        ck = valid & (hazard == k)
        print(f"   class {k}: BAU-only {ha(bau_only_all & ck):9.2f} ha | "
              f"ECO-only {ha(eco_only_all & ck):9.2f} ha | "
              f"net (ECO-BAU) {ha(new_eco & ck) - ha(new_bau & ck):9.2f} ha")
    print(f"reloc_bau_high_ha,{ha(bau_high):.2f}")
    print(f"reloc_eco_high_ha,{ha(eco_high):.2f}")
    print(f"reloc_net_high_ha,{ha(bau_high) - ha(eco_high):.2f}")
    print(f"reloc_bau_only_high_ha,{ha(bau_only_high):.2f}")
    print(f"reloc_eco_only_high_ha,{ha(eco_only_high):.2f}")
    print(f"reloc_shared_high_ha,{ha(shared_high):.2f}")
    print(f"reloc_bau_only_all_ha,{ha(bau_only_all):.2f}")
    print(f"reloc_bau_only_all_pct,{ha(bau_only_all) / ha(new_bau) * 100:.1f}")
    for k in (4, 5):
        ck = valid & (hazard == k)
        print(f"reloc_bau_only_c{k},{ha(bau_only_all & ck):.2f}")
        print(f"reloc_eco_only_c{k},{ha(eco_only_all & ck):.2f}")

    # ----- E. Regression cross-check of the exposure keys in mydata.dat ------
    # Re-derives, from the persisted rasters, the headline exposure quantities
    # so that every one of them can be diffed against mydata.dat.
    print("\n== E. Exposure keys re-derived from rasters ==")
    nb = int(new_bau.sum())
    for k in range(1, 6):
        ck = valid & (hazard == k)
        print(f"bau_new_ha_class{k},{ha(new_bau & ck):.2f}")
        print(f"eco_new_ha_class{k},{ha(new_eco & ck):.2f}")
        a, b = ha(new_bau & ck), ha(new_eco & ck)
        print(f"area_err_class{k},{(a - b) / a * 100:.1f}")
    print(f"bau_high_ha,{ha(bau_high):.0f}")
    print(f"eco_high_ha,{ha(eco_high):.0f}")
    print(f"bau_high_hazard_pct,{ha(bau_high) / ha(new_bau) * 100:.1f}")
    print(f"eco_high_hazard_pct,{ha(eco_high) / ha(new_eco) * 100:.1f}")
    print(f"bau_vh_ha,{ha(new_bau & valid & (hazard == 5)):.0f}")
    print(f"eco_vh_ha,{ha(new_eco & valid & (hazard == 5)):.0f}")
    print(f"bau_vh_pct,{ha(new_bau & valid & (hazard == 5)) / ha(new_bau) * 100:.1f}")
    print(f"eco_vh_pct,{ha(new_eco & valid & (hazard == 5)) / ha(new_eco) * 100:.1f}")
    print(f"err_pct,{(ha(bau_high) - ha(eco_high)) / ha(bau_high) * 100:.1f}")
    print(f"err_pp,{(ha(bau_high) / ha(new_bau) - ha(eco_high) / ha(new_eco)) * 100:.1f}")
    print(f"no_build_ha,{ha(bau_high):.0f}")
    print(f"risk_avoided_ha,{ha(bau_only_high):.0f}")
    basin_high = valid & (hazard >= 4)
    print(f"high_hazard_area_ha,{ha(basin_high):.0f}")
    print(f"high_hazard_pct,{basin_high.sum() / valid.sum() * 100:.1f}")
    hci = lambda m: (ha(m & high) / ha(m)) / (basin_high.sum() / valid.sum())
    print(f"hci_bau,{hci(new_bau):.2f}")
    print(f"hci_eco,{hci(new_eco):.2f}")
    wmh = lambda m: float(hazard[m & valid].mean())
    print(f"wmh_bau,{wmh(new_bau):.2f}")
    print(f"wmh_eco,{wmh(new_eco):.2f}")
    print(f"wmh_delta,{wmh(new_bau) - wmh(new_eco):.2f}")

    # ----- loaders (same nodata handling as p2_fig11) -----------------------
    def load_raw(path, resampling=Resampling.bilinear):
        dst = np.full((H, W), np.nan, dtype=np.float32)
        with rasterio.open(path) as src:
            reproject(source=rasterio.band(src, 1), destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
                      resampling=resampling, src_nodata=src.nodata, dst_nodata=np.nan)
        return dst

    print("\n== D. HRSL population: grid alignment and gap bookkeeping ==")
    with rasterio.open(INTERIM / "p2_hrsl_pop_general_clipped.tif") as s:
        same_grid = (s.height == H and s.width == W and s.crs == ref_meta["crs"]
                     and s.transform == ref_meta["transform"])
        print(f"  HRSL grid identical to hazard grid (no resampling in p2_fig09): {same_grid} "
              f"(HRSL {s.height}x{s.width}, nodata={s.nodata})")
        hrsl_native = s.read(1).astype(np.float32)
        hrsl_nd = s.nodata
    gap_native = ~np.isfinite(hrsl_native)
    if hrsl_nd is not None:
        gap_native |= hrsl_native == hrsl_nd
    pos_native = ~gap_native & (hrsl_native > 0)
    zero_native = ~gap_native & (hrsl_native == 0)
    print(f"  Basin: HRSL gap {gap_native[valid].mean() * 100:.2f}% | "
          f"valid zero {zero_native[valid].mean() * 100:.2f}% | "
          f"positive {pos_native[valid].mean() * 100:.2f}% of valid basin pixels")
    # p2_fig09 treats gap as NaN -> np.nansum (i.e. contributes 0)
    for label, m in (("BAU", new_bau), ("ECO", new_eco)):
        tot = float(np.nansum(np.where(gap_native, np.nan, hrsl_native)[m]))
        print(f"  {label} new-growth footprint: {int(m.sum()):,} px | "
              f"HRSL gap {gap_native[m].mean() * 100:.2f}% | valid zero {zero_native[m].mean() * 100:.2f}% | "
              f"positive {pos_native[m].mean() * 100:.2f}% | direct-intersection sum {tot:,.0f}")
    bau_tot = float(np.nansum(np.where(gap_native, np.nan, hrsl_native)[new_bau]))
    bau_hi = float(np.nansum(np.where(gap_native, np.nan, hrsl_native)[new_bau & high]))
    pop_clean = np.where(gap_native, np.nan, hrsl_native)
    for k in range(1, 6):
        ck = valid & (hazard == k)
        b_k = float(np.nansum(pop_clean[new_bau & ck]))
        e_k = float(np.nansum(pop_clean[new_eco & ck]))
        print(f"bau_pop_exposed_class{k},{b_k:.0f}")
        print(f"eco_pop_exposed_class{k},{e_k:.0f}")
        print(f"pop_err_class{k},{(b_k - e_k) / b_k * 100:.1f}")
    print(f"eco_pop_exposed,{float(np.nansum(pop_clean[new_eco])):.0f}")
    print(f"  BAU population in class 4-5 new growth: {bau_hi:,.0f} ({bau_hi / bau_tot * 100:.1f}% of {bau_tot:,.0f})")
    print(f"pop_bau_total_direct,{bau_tot:.0f}")
    # sum of the rounded class-4 and class-5 values, so the figure equals what a reader
    # obtains by adding the two class rows of the exposure table
    b45 = sum(round(float(np.nansum(pop_clean[new_bau & valid & (hazard == k)]))) for k in (4, 5))
    print(f"pop_bau_high45,{b45}")
    print(f"pop_bau_high45_pct,{b45 / bau_tot * 100:.1f}")
    print(f"pop_bau_footprint_pos_pct,{pos_native[new_bau].mean() * 100:.1f}")
    print(f"pop_bau_footprint_gap_pct,{gap_native[new_bau].mean() * 100:.1f}")
    # Age composition of the directly intersected BAU population (traces the
    # elderly / child shares quoted in Results to a script).
    def hrsl_sum(fname, mask):
        with rasterio.open(INTERIM / fname) as s:
            a = s.read(1).astype(np.float32)
            nd = s.nodata
        a = np.where(np.isfinite(a) & ((a != nd) if nd is not None else True), a, np.nan)
        return float(np.nansum(a[mask]))
    for label, m in (("bau", new_bau), ("eco", new_eco)):
        tot = hrsl_sum("p2_hrsl_pop_general_clipped.tif", m)
        eld = hrsl_sum("p2_hrsl_pop_elderly_60p_clipped.tif", m)
        chi = hrsl_sum("p2_hrsl_pop_children_u5_clipped.tif", m)
        print(f"  {label.upper()} footprint: total {tot:,.0f}; elderly(>=60) {eld:,.0f} ({eld / tot * 100:.1f}%); "
              f"children(<5) {chi:,.0f} ({chi / tot * 100:.1f}%)")
        print(f"{label}_elderly_pct,{eld / tot * 100:.1f}")
        print(f"{label}_child_pct,{chi / tot * 100:.1f}")
    base_pop = float(np.nansum(np.where(gap_native, np.nan, hrsl_native)[valid]))
    print(f"  Basin HRSL total {base_pop:,.0f}; BAU footprint holds {bau_tot / base_pop * 100:.2f}% of it")
    print(f"pop_basin_total,{base_pop:.0f}")

    print("\n== B. Prioritisation score: genuine gaps vs valid zeros ==")
    candidate = new_bau & high
    nc = int(candidate.sum())
    pop_raw = load_raw(INTERIM / "p2_hrsl_pop_general_clipped.tif")
    ntl_raw = load_raw(PROCESSED / "drivers" / "driver_ntl.tif")
    with rasterio.open(PROCESSED / "drivers" / "driver_ntl.tif") as s:
        print(f"  NTL driver nodata={s.nodata}, grid identical={s.height == H and s.width == W}")
    stats = {}
    for name, raw in (("pop", pop_raw), ("ntl", ntl_raw)):
        r = raw[candidate]
        gap = int((~np.isfinite(r)).sum())
        zero = int((np.isfinite(r) & (r == 0)).sum())
        pos = int((np.isfinite(r) & (r > 0)).sum())
        stats[name] = (gap, zero, pos)
        print(f"  {name.upper()}: candidates {nc:,} | gap {gap:,} ({gap / nc * 100:.1f}%) | "
              f"valid zero {zero:,} ({zero / nc * 100:.1f}%) | positive {pos:,} ({pos / nc * 100:.1f}%)")
        print(f"prioritization_gap_{name}_n,{gap}")
        print(f"prioritization_gap_{name}_pct,{gap / nc * 100:.1f}")
        print(f"prioritization_validzero_{name}_pct,{zero / nc * 100:.1f}")
    any_gap = ~np.isfinite(pop_raw) | ~np.isfinite(ntl_raw)
    both_gap = ~np.isfinite(pop_raw) & ~np.isfinite(ntl_raw)
    print(f"  any gap {int(any_gap[candidate].sum()):,} ({any_gap[candidate].mean() * 100:.1f}%) | "
          f"both {int(both_gap[candidate].sum()):,}")
    print(f"prioritization_gap_any_pct,{any_gap[candidate].mean() * 100:.1f}")

    drivers, _ = _load_module("p2_04", "p2_04_flus_ca_engine.py").load_drivers()
    with open(MODELS / "rf_flus_model.pkl", "rb") as f:
        rf = pickle.load(f)
    dev = np.zeros((H, W), np.float32)
    m = (lulc2020 > 0) & (lulc2020 != 4)
    dev[m] = rf.predict_proba(drivers[m])[:, 1]

    def score_from(pop, ntl, use_mean_available=False, pop_gap=None, ntl_gap=None):
        hz, pr = norm01(hazard, candidate), norm01(dev, candidate)
        pn, fn = norm01(pop, candidate), 1.0 - norm01(ntl, candidate)
        if not use_mean_available:
            s = (hz + pr + pn + fn) / 4.0
        else:
            num = hz + pr
            den = np.full((H, W), 2.0, np.float32)
            num = num + np.where(pop_gap, 0.0, pn); den = den + np.where(pop_gap, 0.0, 1.0)
            num = num + np.where(ntl_gap, 0.0, fn); den = den + np.where(ntl_gap, 0.0, 1.0)
            s = num / den
        return s

    def top(s, mask):
        q = float(np.nanpercentile(s[mask], 75))
        return mask & (s >= q)

    pop_gap = ~np.isfinite(pop_raw)
    ntl_gap = ~np.isfinite(ntl_raw)
    zero_fill = lambda a: np.where(np.isfinite(a), a, 0.0).astype(np.float32)

    base_score = score_from(zero_fill(pop_raw), zero_fill(ntl_raw))
    base_top = top(base_score, candidate)
    print(f"  Baseline (gaps=0, reported): top-tier {ha(base_top):,.0f} ha")
    print(f"prioritization_baseline_top_ha,{ha(base_top):.0f}")

    # (1) mean of available criteria (gap criterion dropped for that pixel)
    s1 = score_from(zero_fill(pop_raw), zero_fill(ntl_raw), True, pop_gap, ntl_gap)
    t1 = top(s1, candidate)
    # (2) median imputation over observed candidate values
    pm = np.nanmedian(pop_raw[candidate]); nm = np.nanmedian(ntl_raw[candidate])
    s2 = score_from(np.where(np.isfinite(pop_raw), pop_raw, pm).astype(np.float32),
                    np.where(np.isfinite(ntl_raw), ntl_raw, nm).astype(np.float32))
    t2 = top(s2, candidate)
    # (3) complete-case: rank only candidates with no gap in either criterion
    cc = candidate & ~any_gap
    s3 = score_from(zero_fill(pop_raw), zero_fill(ntl_raw))
    # re-normalise on complete cases only
    hz, pr = norm01(hazard, cc), norm01(dev, cc)
    pn, fn = norm01(zero_fill(pop_raw), cc), 1.0 - norm01(zero_fill(ntl_raw), cc)
    s3 = (hz + pr + pn + fn) / 4.0
    t3 = top(s3, cc)
    for lab, t, key in (("mean of available criteria", t1, "avail"),
                        ("median imputation", t2, "median"),
                        ("complete cases only", t3, "complete")):
        j = jaccard(base_top, t)
        j_cc = jaccard(base_top & cc, t & cc) if key == "complete" else float("nan")
        print(f"  alt [{lab:28s}] top-tier {ha(t):,.0f} ha | Jaccard vs baseline {j:.3f}"
              + (f" | Jaccard within complete cases {j_cc:.3f}" if key == "complete" else ""))
        print(f"prioritization_gap_alt_{key}_ha,{ha(t):.0f}")
        print(f"prioritization_gap_alt_{key}_iou,{j:.3f}")
    # (4)-(6) separate the two sources: genuine NTL gaps vs HRSL "absence"
    pn0, hz0, pr0 = norm01(zero_fill(pop_raw), candidate), norm01(hazard, candidate), norm01(dev, candidate)
    # (4) genuine NTL gaps -> median NTL (pop still zero-filled)
    s4 = score_from(zero_fill(pop_raw), np.where(np.isfinite(ntl_raw), ntl_raw, nm).astype(np.float32))
    t4 = top(s4, candidate)
    # (5) genuine NTL gaps -> feasibility criterion dropped for those pixels only
    num = hz0 + pr0 + pn0 + np.where(ntl_gap, 0.0, 1.0 - norm01(zero_fill(ntl_raw), candidate))
    den = 3.0 + np.where(ntl_gap, 0.0, 1.0)
    s5 = (num / den).astype(np.float32)
    t5 = top(s5, candidate)
    # (6) population criterion removed altogether (no assumption about HRSL absence)
    s6 = ((hz0 + pr0 + (1.0 - norm01(zero_fill(ntl_raw), candidate))) / 3.0).astype(np.float32)
    t6 = top(s6, candidate)
    for lab, t, key in (("NTL gaps -> median", t4, "ntl_median"),
                        ("NTL gaps -> criterion dropped", t5, "ntl_drop"),
                        ("population criterion removed", t6, "nopop")):
        j = jaccard(base_top, t)
        print(f"  alt [{lab:28s}] top-tier {ha(t):,.0f} ha | Jaccard vs baseline {j:.3f}")
        print(f"prioritization_gap_alt_{key}_ha,{ha(t):.0f}")
        print(f"prioritization_gap_alt_{key}_iou,{j:.3f}")
    # share of baseline top tier that is gap-affected
    print(f"  Share of baseline first-tier pixels with >=1 gap: {any_gap[base_top].mean() * 100:.1f}% "
          f"(vs {any_gap[candidate].mean() * 100:.1f}% of all candidates)")
    print(f"prioritization_gap_in_toptier_pct,{any_gap[base_top].mean() * 100:.1f}")

    print("\n== C. NbS priority: eligible area and denominator ==")
    ref_arr = None
    with rasterio.open(PROCESSED / "drivers" / "driver_elevation.tif") as s:
        ref_arr = s.read(1); nd = s.nodata
        e_valid = (ref_arr != nd) if nd is not None else ~np.isnan(ref_arr)
    f10 = _load_module("p2_fig10", "p2_fig10_policy.py")
    flood_h = f10.load_aligned(PROCESSED / "flood_hazard.tif", ref_meta, Resampling.bilinear)
    bio_h = f10.load_aligned(PROCESSED / "bioclimate_hazard.tif", ref_meta, Resampling.bilinear)
    canopy = f10.load_aligned(INTERIM / "p2_meta_canopy_height_clipped.tif", ref_meta)
    v10 = e_valid
    canopy_low = (canopy < 2) & v10
    nbs = (f10.normalise(flood_h, v10) * 0.50 + f10.normalise(bio_h, v10) * 0.30 +
           f10.normalise(np.where(canopy_low, 1.0, 0.0).astype(np.float32), v10) * 0.20)
    nbs[~v10] = np.nan
    q75 = np.nanpercentile(nbs[v10], 75)
    nbs_prio = nbs > q75
    den_ha = float(v10.sum()) * PIXEL_HA
    print(f"  Valid basin area (denominator): {den_ha:,.0f} ha ({int(v10.sum()):,} px)")
    print(f"  NbS priority: {ha(nbs_prio):,.0f} ha = {ha(nbs_prio) / den_ha * 100:.1f}% of valid basin; q75={q75:.4f}")
    print(f"  Canopy-low (<2 m) share of valid basin: {canopy_low[v10].mean() * 100:.1f}%")
    names = {1: "urban", 2: "agriculture", 3: "forest/shrub/grass", 4: "water", 5: "bareland", 6: "other/wetland"}
    for k, nme in names.items():
        mk = nbs_prio & (lulc2020 == k)
        print(f"    NbS on LULC2020 {k} ({nme}): {ha(mk):,.0f} ha ({ha(mk) / ha(nbs_prio) * 100:.1f}%)")
    print(f"  NbS priority overlapping BAU no-build candidates: {ha(nbs_prio & candidate):,.0f} ha "
          f"({ha(nbs_prio & candidate) / ha(candidate) * 100:.1f}% of candidates)")
    with rasterio.open(PROCESSED / "drivers" / "driver_river_dist.tif") as s:
        rd = s.read(1).astype(np.float32); rnd = s.nodata
    rd_ok = np.isfinite(rd) & ((rd != rnd) if rnd is not None else True) & v10
    for d in (500, 1000):
        near = rd_ok & (rd <= d)
        share_basin = near.sum() / v10.sum() * 100
        share_nbs = (nbs_prio & near).sum() / nbs_prio.sum() * 100
        enr = share_nbs / share_basin
        print(f"  Within {d} m of a river: {share_basin:.1f}% of basin, {share_nbs:.1f}% of NbS priority "
              f"-> enrichment x{enr:.2f}; {ha(nbs_prio & near) / ha(near) * 100:.1f}% of that corridor is NbS priority")
        print(f"nbs_riparian{d}_basin_pct,{share_basin:.1f}")
        print(f"nbs_riparian{d}_nbs_pct,{share_nbs:.1f}")
        print(f"nbs_riparian{d}_enrichment,{enr:.2f}")
    print(f"  Seismic hazard is NOT an input to the NbS composite (flood 0.50, bio-climatic 0.30, canopy deficit 0.20)")
    print(f"nbs_valid_basin_ha,{den_ha:.0f}")
    print(f"nbs_priority_check_ha,{ha(nbs_prio):.0f}")
    print(f"nbs_priority_share_pct,{ha(nbs_prio) / den_ha * 100:.1f}")
    print(f"nbs_on_agri_pct,{ha(nbs_prio & (lulc2020 == 2)) / ha(nbs_prio) * 100:.1f}")
    print(f"nbs_on_forest_pct,{ha(nbs_prio & (lulc2020 == 3)) / ha(nbs_prio) * 100:.1f}")
    print(f"nbs_on_urban_pct,{ha(nbs_prio & (lulc2020 == 1)) / ha(nbs_prio) * 100:.1f}")
    print(f"nbs_candidate_overlap_ha,{ha(nbs_prio & candidate):.0f}")
    print("\n== F. Quintile-variant exposure and urban trajectory (from p2_02 / p2_09 CSV outputs) ==")
    import csv
    tabs = PROJECT_ROOT / "outputs" / "tables" / "paper2"
    qs = {}
    for sc in ("bau", "eco"):
        rows = list(csv.DictReader(open(tabs / f"exposure_sim_2050_{sc}_quintile.csv")))
        ha_by = {int(r["Hazard_Class"]): float(r["New_Urban_Area_ha"]) for r in rows}
        tot = sum(ha_by.values())
        qs[sc] = (ha_by[4] + ha_by[5], tot)
        print(f"quintile_{sc}_high_pct,{(ha_by[4] + ha_by[5]) / tot * 100:.1f}")
    print(f"quintile_err_pct,{(qs['bau'][0] - qs['eco'][0]) / qs['bau'][0] * 100:.1f}")
    ann = {int(r["Year"]): float(r["Urban_Area_ha"])
           for r in csv.DictReader(open(tabs / "annual_impervious_area_1985_2022.csv"))}
    print(f"urban_cagr,{((ann[2022] / ann[1985]) ** (1 / 37) - 1) * 100:.1f}")
    print(f"urban_rate_2000_2012,{(ann[2012] - ann[2000]) / 12:.0f}")
    print(f"urban_rate_2012_2022,{(ann[2022] - ann[2012]) / 10:.0f}")
    pk = max(ann, key=ann.get)
    print(f"urban_peak_year,{pk}")
    print(f"urban_peak_ha,{ann[pk]:.0f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
