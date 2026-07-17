"""
Paper 2 — Figure 10: Policy Synthesis
======================================
(a) No-Build Zones     — BAU new growth pixels in high/very-high hazard areas
(b) NbS Priority Areas — composite score: canopy deficit + flood + bio-climatic
(c) ECO Effectiveness  — spatial risk avoided vs BAU

Style: elite colormap, hillshade, admin boundaries, scale/north — matching fig06/fig07.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, LightSource
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_fill_holes, gaussian_filter
from pyproj import Transformer
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "axes.titlesize": 11, "axes.labelsize": 9, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM     = PROJECT_ROOT / "data" / "02_interim"  / "paper2"
PROCESSED   = PROJECT_ROOT / "data" / "03_processed" / "paper2"
DRIVERS_DIR = PROCESSED / "drivers"
FIG_DIR     = PROJECT_ROOT / "outputs" / "figures"  / "paper2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

elite_cmap    = LinearSegmentedColormap.from_list("elite", ["#dfe6e9", "#00cec9", "#ff7675"])
BAU_COLOR     = "#ff7675"
ECO_COLOR     = "#00cec9"
BOUNDARY_IL   = "#2c3e50"
BOUNDARY_ILCE = "#636e72"
GRID_COLOR    = "#b2bec3"


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_aligned(path, ref_meta, method=Resampling.nearest):
    if not path.exists():
        print(f"  WARNING: {path.name} not found")
        return None
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd  = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        if (src.height == ref_meta["height"] and
                src.width  == ref_meta["width"]  and
                src.crs    == ref_meta["crs"]):
            return arr
        dst = np.full((ref_meta["height"], ref_meta["width"]), np.nan, dtype=np.float32)
        reproject(source=arr, destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
                  resampling=method)
        return dst


def normalise(arr, mask):
    v = arr[mask & ~np.isnan(arr)] if arr is not None else np.array([])
    if len(v) == 0 or v.max() == v.min():
        return np.zeros_like(arr, np.float32) if arr is not None else None
    out = np.zeros_like(arr, np.float32)
    out[mask] = (arr[mask] - v.min()) / (v.max() - v.min())
    return out


def _name_col(gdf):
    for c in ("name","NAME","ADM1_EN","ADM1_TR","il_adi","PROVINCE","name:tr"):
        if c in gdf.columns:
            return c
    return None


def load_admin_boundaries(crs_target):
    import geopandas as gpd
    cache_il   = INTERIM / "admin_il.gpkg"
    cache_ilce = INTERIM / "admin_ilce.gpkg"
    if cache_il.exists() and cache_ilce.exists():
        il   = gpd.read_file(cache_il).to_crs(crs_target)
        ilce = gpd.read_file(cache_ilce).to_crs(crs_target)
        if len(il) > 0 and len(ilce) > 0:
            return il, ilce
    return None, None


def add_boundaries(ax, il_gdf, ilce_gdf, basin_geom):
    try:
        if ilce_gdf is not None:
            cl = ilce_gdf.clip(basin_geom)
            cl = cl[~cl.is_empty & cl.is_valid]
            if len(cl):
                cl.boundary.plot(ax=ax, color=BOUNDARY_ILCE,
                                 linewidth=0.35, linestyle="--", zorder=5, alpha=0.65)
        if il_gdf is not None:
            cl = il_gdf.clip(basin_geom)
            cl = cl[~cl.is_empty & cl.is_valid]
            if len(cl):
                cl.boundary.plot(ax=ax, color=BOUNDARY_IL,
                                 linewidth=0.75, zorder=5, alpha=0.80)
    except Exception as exc:
        print(f"  Boundaries skipped: {exc}")


def add_province_labels(ax, il_gdf, basin_geom):
    if il_gdf is None:
        return
    try:
        nc      = _name_col(il_gdf)
        clipped = il_gdf.clip(basin_geom)
        clipped = clipped[~clipped.geometry.is_empty]
        for _, row in clipped.iterrows():
            name = str(row.get(nc, "")).strip() if nc else ""
            if not name or name.lower() in ("none","nan",""):
                continue
            pt = row.geometry.representative_point()
            ax.text(pt.x, pt.y, name.upper(), fontsize=7,
                    ha="center", va="center", color=BOUNDARY_IL,
                    fontweight="bold", alpha=0.90, zorder=8,
                    path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
    except Exception as exc:
        print(f"  Labels skipped: {exc}")


def add_scale_north(ax, bounds):
    W, H  = bounds.right - bounds.left, bounds.top - bounds.bottom
    cx    = bounds.right - W * 0.08
    slen  = 20000
    sb_x0 = cx - slen / 2
    sb_y0 = bounds.bottom + H * 0.04
    sb_h  = H * 0.013
    for i in range(2):
        ax.add_patch(Rectangle(
            (sb_x0 + i * slen / 2, sb_y0), slen / 2, sb_h,
            facecolor="black" if i == 0 else "white",
            edgecolor="black", zorder=6))
        ax.text(sb_x0 + i * slen / 2, sb_y0 + sb_h * 1.65,
                f"{int(i * slen / 2 / 1000)}",
                ha="center", va="bottom", fontsize=7, fontweight="bold", zorder=6)
    ax.text(sb_x0 + slen, sb_y0 + sb_h * 1.65, f"{int(slen/1000)} km",
            ha="center", va="bottom", fontsize=7, fontweight="bold", zorder=6)
    na_base = sb_y0 + sb_h * 5.2
    na_w, na_h = W * 0.028, H * 0.055
    ax.add_patch(Polygon(
        [[cx, na_base + na_h], [cx + na_w/2, na_base],
         [cx, na_base + na_h * 0.25], [cx - na_w/2, na_base]],
        facecolor=BOUNDARY_IL, edgecolor="white", lw=1.2, zorder=6))
    ax.text(cx, na_base + na_h + H * 0.005, "N",
            ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=BOUNDARY_IL, zorder=6)


def draw_base(ax, hillshade, boundary, bounds):
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    shift  = 15
    sm = np.zeros_like(boundary, bool)
    sm[shift:, shift:] = boundary[:-shift, :-shift]
    sa = np.zeros(boundary.shape, float); sa[~sm] = np.nan
    ax.imshow(sa, cmap="Greys", vmin=0, vmax=1, alpha=0.4, extent=extent, zorder=1)
    ax.imshow(hillshade, cmap="gray", extent=extent, zorder=2, alpha=0.6)
    ax.contour(boundary, levels=[0.5], colors=BOUNDARY_IL,
               linewidths=0.8, extent=extent, origin="upper", zorder=4)
    ax.set_xticks([]); ax.set_yticks([])
    return extent


def stats_box(ax, text, loc="upper left"):
    x = 0.02 if "left" in loc else 0.98
    y = 0.97 if "upper" in loc else 0.03
    ha = "left" if "left" in loc else "right"
    va = "top"  if "upper" in loc else "bottom"
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=8, va=va, ha=ha, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=BOUNDARY_IL, alpha=0.92, lw=1.0))


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def create_fig10():
    print("Generating Fig 10: Policy Synthesis...")

    # Reference grid from driver_elevation (same as all processed files)
    with rasterio.open(DRIVERS_DIR / "driver_elevation.tif") as ref:
        ref_meta = ref.meta.copy()
        ref_arr  = ref.read(1)
        nodata   = ref.nodata
        bounds   = ref.bounds
        crs      = ref.crs
    valid = (ref_arr != nodata) if nodata is not None else ~np.isnan(ref_arr)

    # Load rasters
    hazard   = load_aligned(PROCESSED / "multi_hazard_surface.tif",    ref_meta)
    bau_sim  = load_aligned(PROCESSED / "sim_2050_bau.tif",             ref_meta)
    eco_sim  = load_aligned(PROCESSED / "sim_2050_eco.tif",             ref_meta)
    canopy   = load_aligned(INTERIM   / "p2_meta_canopy_height_clipped.tif", ref_meta)
    flood_h  = load_aligned(PROCESSED / "flood_hazard.tif",             ref_meta, Resampling.bilinear)
    bio_h    = load_aligned(PROCESSED / "bioclimate_hazard.tif",        ref_meta, Resampling.bilinear)
    lulc2020 = load_aligned(PROCESSED / "lulc_simplified_2020.tif",    ref_meta)

    base_urban  = (lulc2020 == 1) & valid if lulc2020 is not None else np.zeros_like(valid)
    new_bau     = (bau_sim == 1) & (~base_urban) & valid if bau_sim is not None else np.zeros_like(valid, bool)
    new_eco     = (eco_sim == 1) & (~base_urban) & valid if eco_sim is not None else np.zeros_like(valid, bool)
    hazard_arr  = np.nan_to_num(hazard, nan=0) if hazard is not None else np.zeros_like(ref_arr)
    high_haz    = hazard_arr >= 4
    no_build    = new_bau & high_haz

    # NbS composite score (flood 50% + bio-climatic 30% + canopy deficit 20%)
    canopy_low = (canopy < 2) & valid if canopy is not None else np.zeros_like(valid, bool)
    nbs_score  = (normalise(flood_h, valid)                                  * 0.50 +
                  normalise(bio_h,   valid)                                  * 0.30 +
                  normalise(np.where(canopy_low, 1.0, 0.0).astype(np.float32), valid) * 0.20)
    nbs_score[~valid] = np.nan
    nbs_q75    = np.nanpercentile(nbs_score[valid], 75)
    nbs_prio   = nbs_score > nbs_q75

    # Risk avoided: BAU would have grown here in high-hazard, ECO avoided it
    risk_avoided = new_bau & high_haz & ~(new_eco & high_haz)

    pixel_ha = (30 * 30) / 10_000.0
    nb_ha    = float(no_build.sum())     * pixel_ha
    nbs_ha   = float(nbs_prio.sum())     * pixel_ha
    saved_ha = float(risk_avoided.sum()) * pixel_ha

    # Background
    ls        = LightSource(azdeg=315, altdeg=45)
    hillshade = np.ma.masked_where(
        (ref_arr == nodata),
        ls.hillshade(np.where(ref_arr == nodata, 0, ref_arr).astype(float),
                     vert_exag=3, dx=30, dy=30))
    with rasterio.open(PROCESSED / "lulc_simplified_2000.tif") as src:
        lulc_bg = src.read(1); nd2 = src.nodata
    boundary = binary_fill_holes((lulc_bg != nd2) if nd2 is not None else (lulc_bg > 0))

    # Admin boundaries (from cache)
    import geopandas as gpd
    basin_gpkg = INTERIM / "kuzey_ege_havzasi_boundary.gpkg"
    if basin_gpkg.exists():
        basin_geom = gpd.read_file(basin_gpkg).to_crs(crs).union_all()
    else:
        from shapely.geometry import box as _bbox
        basin_geom = _bbox(bounds.left, bounds.bottom, bounds.right, bounds.top)
    il_gdf, ilce_gdf = load_admin_boundaries(crs)
    has_admin = (il_gdf is not None) or (ilce_gdf is not None)

    # Coordinate formatter
    _tr   = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    _ymid = bounds.bottom + (bounds.top   - bounds.bottom) / 2
    _xmid = bounds.left   + (bounds.right - bounds.left)   / 2

    def fmt_x(v, _):
        try: lon, _ = _tr.transform(v, _ymid); return f"{lon:.2f}°E"
        except: return ""

    def fmt_y(v, _):
        try: _, lat = _tr.transform(_xmid, v); return f"{lat:.2f}°N"
        except: return ""

    # ── Layout ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    plt.subplots_adjust(wspace=0.04)

    for ax in axes:
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.spines["left"].set_linewidth(1.2)

    # ── (a) No-Build Zones ────────────────────────────────────────────────────
    extent = draw_base(axes[0], hillshade, boundary, bounds)

    rgba_a = np.zeros((*ref_arr.shape, 4), np.float32)
    rgba_a[new_bau & ~high_haz] = [*elite_cmap(0.45)[:3], 0.75]   # teal — acceptable
    rgba_a[no_build]             = [*elite_cmap(1.00)[:3], 0.92]   # coral — no-build
    axes[0].imshow(rgba_a, extent=extent, origin="upper", zorder=3)

    # Cluster contours — smooth no-build mask and draw outline
    _nb_s = gaussian_filter(no_build.astype(float), sigma=4)
    axes[0].contour(_nb_s, levels=[0.25],
                    colors=[elite_cmap(1.0)], linewidths=0.65,
                    extent=extent, origin="upper", alpha=0.70, zorder=6)

    if has_admin:
        add_boundaries(axes[0], il_gdf, ilce_gdf, basin_geom)
    add_province_labels(axes[0], il_gdf, basin_geom)
    add_scale_north(axes[0], bounds)

    stats_box(axes[0],
              f"BAU new growth : {float(new_bau.sum())*pixel_ha:,.0f} ha\n"
              f"No-build zones : {nb_ha:,.0f} ha\n"
              f"Share in hazard ≥4 : {nb_ha/max(float(new_bau.sum())*pixel_ha,1)*100:.1f}%",
              loc="upper right")

    leg_a = [Patch(facecolor=elite_cmap(0.45), label="Acceptable growth"),
             Patch(facecolor=elite_cmap(1.00), label="No-build zone (hazard ≥4)")]
    if has_admin:
        leg_a += [Line2D([0],[0], color=BOUNDARY_IL,   lw=0.85, label="Province"),
                  Line2D([0],[0], color=BOUNDARY_ILCE, lw=0.40, linestyle="--", label="District")]
    axes[0].legend(handles=leg_a, loc="lower left", fontsize=7.5,
                   frameon=True, facecolor="white", edgecolor=BOUNDARY_ILCE,
                   title="No-Build", title_fontsize=8)
    axes[0].set_title("(a)", loc="left", fontweight="bold", fontsize=18, pad=8)

    # Coordinate ticks — panel (a)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # ── (b) NbS Priority Areas — discrete 3 classes ──────────────────────────
    extent  = draw_base(axes[1], hillshade, boundary, bounds)
    nbs_q875 = np.nanpercentile(nbs_score[valid], 87.5)
    nbs_q95  = np.nanpercentile(nbs_score[valid], 95.0)

    rgba_b = np.zeros((*ref_arr.shape, 4), np.float32)
    rgba_b[nbs_prio & (nbs_score <= nbs_q875) & valid] = [*elite_cmap(0.35)[:3], 0.82]
    rgba_b[nbs_prio & (nbs_score >  nbs_q875) & (nbs_score <= nbs_q95) & valid] = [*elite_cmap(0.65)[:3], 0.88]
    rgba_b[nbs_prio & (nbs_score >  nbs_q95)  & valid] = [*elite_cmap(0.95)[:3], 0.93]
    axes[1].imshow(rgba_b, extent=extent, origin="upper", zorder=3)

    if has_admin:
        add_boundaries(axes[1], il_gdf, ilce_gdf, basin_geom)
    add_province_labels(axes[1], il_gdf, basin_geom)

    stats_box(axes[1],
              f"NbS priority area  : {nbs_ha:,.0f} ha\n"
              f"Score threshold    : >{nbs_q75:.2f}\n"
              f"Weights: Flood 50% | Bio 30% | Canopy 20%",
              loc="upper right")

    leg_b = [Patch(facecolor=elite_cmap(0.35), label="Moderate (75–87.5th pct)"),
             Patch(facecolor=elite_cmap(0.65), label="High (87.5–95th pct)"),
             Patch(facecolor=elite_cmap(0.95), label="Very High (>95th pct)")]
    if has_admin:
        leg_b += [Line2D([0],[0], color=BOUNDARY_IL,   lw=0.85, label="Province"),
                  Line2D([0],[0], color=BOUNDARY_ILCE, lw=0.40, linestyle="--", label="District")]
    axes[1].legend(handles=leg_b, loc="lower left", fontsize=7.5,
                   frameon=True, facecolor="white", edgecolor=BOUNDARY_ILCE,
                   title="NbS Priority", title_fontsize=8)
    axes[1].set_title("(b)", loc="left", fontweight="bold", fontsize=18, pad=8)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # ── (c) ECO Effectiveness ─────────────────────────────────────────────────
    extent = draw_base(axes[2], hillshade, boundary, bounds)

    rgba_c = np.zeros((*ref_arr.shape, 4), np.float32)
    rgba_c[new_eco & ~high_haz & valid]  = [*elite_cmap(0.35)[:3], 0.75]   # teal — safe ECO
    rgba_c[risk_avoided]                  = [0.961, 0.616, 0.043, 0.92]     # amber-orange — risk avoided
    rgba_c[new_eco & high_haz & valid]   = [*elite_cmap(1.00)[:3], 0.65]   # coral — remaining risk
    axes[2].imshow(rgba_c, extent=extent, origin="upper", zorder=3)

    # Cluster contours — smooth risk_avoided mask and draw outline
    _ra_s = gaussian_filter(risk_avoided.astype(float), sigma=4)
    axes[2].contour(_ra_s, levels=[0.25],
                    colors=[(0.961, 0.616, 0.043)], linewidths=0.65,
                    extent=extent, origin="upper", alpha=0.70, zorder=6)

    if has_admin:
        add_boundaries(axes[2], il_gdf, ilce_gdf, basin_geom)
    add_province_labels(axes[2], il_gdf, basin_geom)
    add_scale_north(axes[2], bounds)

    # Coordinate ticks — panel (c)
    axes[2].xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    axes[2].yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(fmt_x))
    axes[2].yaxis.set_major_formatter(ticker.FuncFormatter(fmt_y))
    plt.setp(axes[2].get_yticklabels(), rotation=90, va="center")
    axes[2].tick_params(labelsize=8)
    axes[2].spines["bottom"].set_visible(True)
    axes[2].spines["left"].set_visible(True)

    eco_safe_ha = float((new_eco & ~high_haz & valid).sum()) * pixel_ha
    eco_risk_ha = float((new_eco & high_haz  & valid).sum()) * pixel_ha
    stats_box(axes[2],
              f"Risk avoided (vs BAU) : {saved_ha:,.0f} ha\n"
              f"ECO safe growth        : {eco_safe_ha:,.0f} ha\n"
              f"ECO remaining risk     : {eco_risk_ha:,.0f} ha",
              loc="upper right")

    leg_c = [Patch(facecolor=elite_cmap(0.35), label="ECO — safe growth"),
             Patch(facecolor=(0.961, 0.616, 0.043), label=f"Risk avoided ({saved_ha:,.0f} ha)"),
             Patch(facecolor=elite_cmap(1.00), label="ECO — remaining risk")]
    if has_admin:
        leg_c += [Line2D([0],[0], color=BOUNDARY_IL,   lw=0.85, label="Province"),
                  Line2D([0],[0], color=BOUNDARY_ILCE, lw=0.40, linestyle="--", label="District")]
    axes[2].legend(handles=leg_c, loc="lower left", fontsize=7.5,
                   frameon=True, facecolor="white", edgecolor=BOUNDARY_ILCE,
                   title="ECO Scenario", title_fontsize=8)
    axes[2].set_title("(c)", loc="left", fontweight="bold", fontsize=18, pad=8)

    out_path = FIG_DIR / "fig10_Policy_Synthesis.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"  No-build: {nb_ha:,.0f} ha | NbS: {nbs_ha:,.0f} ha | Risk avoided: {saved_ha:,.0f} ha")


if __name__ == "__main__":
    create_fig10()
