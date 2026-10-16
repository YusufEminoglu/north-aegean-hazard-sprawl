"""
Paper 2 — Supplementary Figure: Intervention Prioritisation Score
=====================================================================
Implements the four-criterion prioritisation rule defined in Methodology
sec:policy_instruments ("Prioritisation criteria", R1-Aug review comment 8):
an equally weighted, normalised (0-1) composite of
  (i)   hazard severity          — MHI value
  (ii)  development pressure     — RF suitability probability (proxy for
                                    likelihood of development absent
                                    intervention; the persisted suitability
                                    surface, not a saved per-iteration CA
                                    roulette-wheel probability)
  (iii) population exposure      — HRSL general population density
  (iv)  implementation feasibility — inverse VIIRS NTL intensity (lower
                                    sunk-cost / easier to designate no-build)
computed on BAU no-build candidate pixels (new BAU growth, hazard class >= 4).
Top-quartile pixels are first-tier priorities.

Style matches fig10 (reuses its admin-boundary/hillshade/scale-bar helpers).

Usage:
  .venv/Scripts/python.exe src/paper2_hazard_sprawl/p2_fig11_prioritization.py
"""

import pickle
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_fill_holes
from pyproj import Transformer
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("p2_fig10_policy", HERE / "p2_fig10_policy.py")
fig10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fig10)

spec4 = importlib.util.spec_from_file_location("p2_04", HERE / "p2_04_flus_ca_engine.py")
p2_04 = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(p2_04)

PROJECT_ROOT = fig10.PROJECT_ROOT
INTERIM      = fig10.INTERIM
PROCESSED    = fig10.PROCESSED
MODELS       = PROJECT_ROOT / "data" / "04_models" / "paper2"
FIG_DIR      = fig10.FIG_DIR

elite_cmap  = fig10.elite_cmap
BOUNDARY_IL = fig10.BOUNDARY_IL


def create_fig11():
    print("Generating Supplementary Figure: Intervention Prioritisation Score...")

    ref_path = PROCESSED / "multi_hazard_surface.tif"
    with rasterio.open(ref_path) as ref:
        ref_meta = ref.meta.copy()
        hazard   = ref.read(1).astype(np.float32)
        nodata   = ref.nodata
    valid = (hazard != nodata)

    with rasterio.open(PROCESSED / "lulc_simplified_2020.tif") as s:
        lulc2020 = s.read(1)
    with rasterio.open(PROCESSED / "sim_2050_bau.tif") as s:
        sim_bau = s.read(1)
    base_urban = (lulc2020 == 1) & valid
    new_bau = (sim_bau == 1) & (~base_urban) & valid
    candidate = new_bau & (hazard >= 4)
    print(f"  No-build/TDR candidate pixels: {int(candidate.sum()):,}")

    drivers, _ = p2_04.load_drivers()
    with open(MODELS / "rf_flus_model.pkl", "rb") as f:
        rf = pickle.load(f)
    dev_pressure = np.zeros(hazard.shape, np.float32)
    m = (lulc2020 > 0) & (lulc2020 != 4)
    dev_pressure[m] = rf.predict_proba(drivers[m])[:, 1]

    # R3 fix (Reviewer 2, point 6 audit): the previous version of this loader
    # did not declare the source nodata value to `reproject`, so bilinear
    # resampling blended the HRSL raster's -9999 sentinel into neighbouring
    # valid cells (the same defect class as the wildfire nodata bug fixed in
    # R2 for p2_08_wildfire_hazard.py) -- this dragged norm01()'s min() far
    # below the true data range and silently compressed nearly every real
    # population value into a thin sliver near 1.0. The HRSL source raster
    # also mixes literal NaN pixels with its declared -9999 sentinel; both
    # are propagated as NaN by `dst_nodata=np.nan` below, and then explicitly
    # mapped to 0 (no population/no night-time light recorded) rather than
    # left as NaN, so a data gap in one criterion no longer poisons the
    # 4-criterion mean for a candidate pixel.
    def load_aligned_raw(path, resampling=Resampling.bilinear):
        dst = np.full((ref_meta["height"], ref_meta["width"]), np.nan, dtype=np.float32)
        with rasterio.open(path) as src:
            reproject(source=rasterio.band(src, 1), destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
                      resampling=resampling, src_nodata=src.nodata, dst_nodata=np.nan)
        return dst

    def load_aligned(path, resampling=Resampling.bilinear):
        raw = load_aligned_raw(path, resampling)
        return np.where(np.isfinite(raw), raw, 0.0).astype(np.float32)

    pop_raw = load_aligned_raw(INTERIM / "p2_hrsl_pop_general_clipped.tif")
    ntl_raw = load_aligned_raw(PROCESSED / "drivers" / "driver_ntl.tif")
    print(f"  [nodata fix] candidate pixels with no HRSL population record, "
          f"filled as 0: {int((~np.isfinite(pop_raw[candidate])).sum()):,} / "
          f"{int(candidate.sum()):,}")
    print(f"  [nodata fix] candidate pixels with no NTL record, filled as 0: "
          f"{int((~np.isfinite(ntl_raw[candidate])).sum()):,} / {int(candidate.sum()):,}")
    pop = np.where(np.isfinite(pop_raw), pop_raw, 0.0).astype(np.float32)
    ntl = np.where(np.isfinite(ntl_raw), ntl_raw, 0.0).astype(np.float32)

    def norm01(arr, mask):
        v = arr[mask]
        v = v[np.isfinite(v)]
        if len(v) == 0 or v.max() == v.min():
            return np.zeros_like(arr)
        out = np.zeros_like(arr, dtype=np.float32)
        out[mask] = np.clip((arr[mask] - v.min()) / (v.max() - v.min()), 0, 1)
        return out

    hazard_n   = norm01(hazard, candidate)
    pressure_n = norm01(dev_pressure, candidate)
    pop_n      = norm01(pop, candidate)
    feas_n     = 1.0 - norm01(ntl, candidate)  # lower NTL -> higher feasibility

    score = np.full(hazard.shape, np.nan, dtype=np.float32)
    score[candidate] = (hazard_n[candidate] + pressure_n[candidate] +
                         pop_n[candidate] + feas_n[candidate]) / 4.0

    q75 = float(np.nanpercentile(score[candidate], 75))
    top_tier = candidate & (score >= q75)
    pixel_ha = (30 * 30) / 10000.0
    print(f"  Top-quartile (first-tier priority) pixels: {int(top_tier.sum()):,}"
          f"  ({int(top_tier.sum()) * pixel_ha:,.0f} ha)")
    print(f"prioritization_top_tier_ha,{int(top_tier.sum()) * pixel_ha:.0f}")
    print(f"prioritization_candidate_ha,{int(candidate.sum()) * pixel_ha:.0f}")

    # ── Map ────────────────────────────────────────────────────────────────
    with rasterio.open(PROCESSED / "drivers" / "driver_elevation.tif") as src:
        elev = src.read(1); nd = src.nodata
        emask = (elev == nd); bounds = src.bounds; crs = src.crs
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = np.ma.masked_where(
        emask, ls.hillshade(np.where(emask, 0, elev).astype(float), vert_exag=3, dx=30, dy=30))
    with rasterio.open(PROCESSED / "lulc_simplified_2000.tif") as s:
        lulc_bg = s.read(1); nd2 = s.nodata
    boundary = binary_fill_holes((lulc_bg != nd2) if nd2 is not None else (lulc_bg > 0))
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    fig, ax = plt.subplots(figsize=(9, 8))
    shift = 15
    sm = np.zeros_like(boundary, bool)
    sm[shift:, shift:] = boundary[:-shift, :-shift]
    sa = np.zeros(boundary.shape, float); sa[~sm] = np.nan
    ax.imshow(sa, cmap="Greys", vmin=0, vmax=1, alpha=0.4, extent=extent, zorder=1)
    ax.imshow(hillshade, cmap="gray", extent=extent, zorder=2, alpha=0.6)

    # Context layer: full BAU new-growth footprint (light), so the sparse
    # candidate/priority pixels read as highlights within a visible growth
    # pattern rather than isolated specks on an empty map (candidate pixels
    # are only 3,564 ha of the ~986,000 ha basin).
    rgba = np.zeros((*hazard.shape, 4), np.float32)
    rgba[new_bau] = [*elite_cmap(0.30)[:3], 0.55]        # light teal — all new growth
    rgba[candidate] = [*elite_cmap(0.65)[:3], 0.85]      # amber — candidate (hazard>=4)
    rgba[top_tier] = [*elite_cmap(1.00)[:3], 0.95]       # coral — first-tier priority
    ax.imshow(rgba, extent=extent, origin="upper", zorder=3)

    # Cluster contour for first-tier priority pixels (dilated for visibility
    # at basin scale, matching the technique used in fig10 panel a).
    _tt_s = gaussian_filter(top_tier.astype(float), sigma=3)
    ax.contour(_tt_s, levels=[0.15], colors=[elite_cmap(1.0)], linewidths=0.7,
               extent=extent, origin="upper", alpha=0.75, zorder=5)
    ax.contour(boundary, levels=[0.5], colors=BOUNDARY_IL, linewidths=0.8,
               extent=extent, origin="upper", zorder=4)

    il_gdf, ilce_gdf = fig10.load_admin_boundaries(crs)
    import geopandas as gpd
    basin_gpkg = INTERIM / "kuzey_ege_havzasi_boundary.gpkg"
    basin_geom = (gpd.read_file(basin_gpkg).to_crs(crs).union_all()
                  if basin_gpkg.exists() else None)
    if basin_geom is not None:
        fig10.add_boundaries(ax, il_gdf, ilce_gdf, basin_geom)
        fig10.add_province_labels(ax, il_gdf, basin_geom)

    _tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    _ymid = bounds.bottom + (bounds.top - bounds.bottom) / 2
    _xmid = bounds.left + (bounds.right - bounds.left) / 2
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{_tr.transform(v, _ymid)[0]:.2f}°E"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{_tr.transform(_xmid, v)[1]:.2f}°N"))
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")
    fig10.add_scale_north(ax, bounds)

    leg = [
        Patch(facecolor=elite_cmap(0.30), alpha=0.55, label="All BAU new growth"),
        Patch(facecolor=elite_cmap(0.65), alpha=0.85, label="Candidate (hazard ≥4)"),
        Patch(facecolor=elite_cmap(1.00), alpha=0.95, label="First-tier priority (top quartile)"),
    ]
    ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.0, 0.09),
              fontsize=8, frameon=True, facecolor="white",
              edgecolor=fig10.BOUNDARY_ILCE, title="Prioritisation score", title_fontsize=8.5)
    ax.set_title("No-build/TDR intervention prioritisation "
                  "(hazard × pressure × exposure × feasibility)",
                  fontsize=11, fontweight="bold", color=BOUNDARY_IL, pad=10)

    stats = (f"Candidate area: {int(candidate.sum())*pixel_ha:,.0f} ha\n"
             f"First-tier priority: {int(top_tier.sum())*pixel_ha:,.0f} ha "
             f"({int(top_tier.sum())/max(int(candidate.sum()),1)*100:.0f}% of candidates)")
    ax.text(0.98, 0.98, stats, transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(facecolor="white", edgecolor=fig10.BOUNDARY_ILCE, alpha=0.9,
                      pad=4, boxstyle="round,pad=0.4"))

    out_path = FIG_DIR / "fig11_Prioritization.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    create_fig11()
