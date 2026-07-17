"""
Paper 2 — Figure 9: Socio-Demographic Exposure Disaggregation
==============================================================
(a) Grouped bar — exposed population by demographic group × hazard class
(b) VPEI map  — fig07-panel-b style; agri-loss inset upper-right
Style: elite colormap, hillshade, admin boundaries — matching fig07.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, LightSource
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_fill_holes
from pyproj import Transformer
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "axes.titlesize": 11, "axes.labelsize": 10, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
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

GROUP_COLORS = {
    "Total":       BOUNDARY_IL,
    "Women":       elite_cmap(0.65),
    "Children <5": BAU_COLOR,
    "Elderly >60": BOUNDARY_ILCE,
}
HRSL_PATHS = {
    "Total":       INTERIM / "p2_hrsl_pop_general_clipped.tif",
    "Women":       INTERIM / "p2_hrsl_pop_women_clipped.tif",
    "Children <5": INTERIM / "p2_hrsl_pop_children_u5_clipped.tif",
    "Elderly >60": INTERIM / "p2_hrsl_pop_elderly_60p_clipped.tif",
}
HAZARD_LABELS = ["Very Low", "Low", "Moderate", "High", "Very High"]


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_aligned(path, ref_meta):
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
                  resampling=Resampling.bilinear)
        return dst


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
            ax.text(pt.x, pt.y, name.upper(), fontsize=7.5,
                    ha="center", va="center", color=BOUNDARY_IL,
                    fontweight="bold", alpha=0.90, zorder=8,
                    path_effects=[pe.withStroke(linewidth=3.5,
                                                foreground="white")])
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
                ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=6)
    ax.text(sb_x0 + slen, sb_y0 + sb_h * 1.65, f"{int(slen/1000)} km",
            ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=6)
    na_base = sb_y0 + sb_h * 5.2
    na_w, na_h = W * 0.028, H * 0.055
    ax.add_patch(Polygon(
        [[cx, na_base + na_h], [cx + na_w/2, na_base],
         [cx, na_base + na_h * 0.25], [cx - na_w/2, na_base]],
        facecolor=BOUNDARY_IL, edgecolor="white", lw=1.2, zorder=6))
    ax.text(cx, na_base + na_h + H * 0.005, "N",
            ha="center", va="bottom", fontsize=10,
            fontweight="bold", color=BOUNDARY_IL, zorder=6)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def create_fig09():
    print("Generating Fig 09: Demographic Exposure...")

    ref_path = PROCESSED / "multi_hazard_surface.tif"
    if not ref_path.exists():
        ref_path = PROCESSED / "lulc_simplified_2020.tif"

    with rasterio.open(ref_path) as ref:
        ref_meta = ref.meta.copy()
        ref_arr  = ref.read(1)
        nodata   = ref.nodata
    valid = (ref_arr != nodata) if nodata is not None else ~np.isnan(ref_arr)

    hazard   = load_aligned(PROCESSED / "multi_hazard_surface.tif", ref_meta)
    bau_sim  = load_aligned(PROCESSED / "sim_2050_bau.tif",          ref_meta)
    eco_sim  = load_aligned(PROCESSED / "sim_2050_eco.tif",          ref_meta)
    lulc2020 = load_aligned(PROCESSED / "lulc_simplified_2020.tif",  ref_meta)

    base_urban = (lulc2020 == 1) & valid if lulc2020 is not None else np.zeros_like(valid)
    new_bau    = (bau_sim == 1) & (~base_urban) & valid if bau_sim is not None else np.zeros_like(valid, bool)
    new_eco    = (eco_sim == 1) & (~base_urban) & valid if eco_sim is not None else np.zeros_like(valid, bool)
    hazard_arr = hazard if hazard is not None else np.zeros_like(ref_arr)

    hrsl = {}
    for g, p in HRSL_PATHS.items():
        arr = load_aligned(p, ref_meta)
        hrsl[g] = arr if arr is not None else np.zeros(ref_arr.shape, np.float32)

    classes  = [1, 2, 3, 4, 5]
    exposure = {g: {"bau": [], "eco": []} for g in hrsl}
    for cls in classes:
        cls_mask = hazard_arr == cls
        for g in hrsl:
            exposure[g]["bau"].append(float(np.nansum(hrsl[g][new_bau & cls_mask])))
            exposure[g]["eco"].append(float(np.nansum(hrsl[g][new_eco & cls_mask])))

    # Agricultural loss (for inset)
    pixel_ha = (30 * 30) / 10_000.0
    agri_mask = (lulc2020 == 2) & valid if lulc2020 is not None else np.zeros_like(valid, bool)
    agri_bau  = [float((new_bau & agri_mask & (hazard_arr == c)).sum()) * pixel_ha for c in classes]
    agri_eco  = [float((new_eco & agri_mask & (hazard_arr == c)).sum()) * pixel_ha for c in classes]

    # VPEI
    total = hrsl["Total"]; vuln = hrsl["Children <5"] + hrsl["Elderly >60"]
    vpei  = np.zeros_like(ref_arr, np.float32)
    m     = valid & (total > 0)
    vpei[m] = vuln[m] / total[m]
    vpei_map = np.where(new_bau, vpei, np.nan)

    # Background
    with rasterio.open(DRIVERS_DIR / "driver_elevation.tif") as src:
        elev = src.read(1); nd = src.nodata
        emask = (elev == nd); bounds = src.bounds; crs = src.crs
    ls        = LightSource(azdeg=315, altdeg=45)
    hillshade = np.ma.masked_where(
        emask,
        ls.hillshade(np.where(emask, 0, elev).astype(float), vert_exag=3, dx=30, dy=30))
    with rasterio.open(PROCESSED / "lulc_simplified_2000.tif") as src:
        lulc_bg = src.read(1); nd2 = src.nodata
    boundary = binary_fill_holes((lulc_bg != nd2) if nd2 is not None else (lulc_bg > 0))

    # ── Layout: 2 panels ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.06)
    ax_bar  = fig.add_subplot(gs[0])
    ax_vpei = fig.add_subplot(gs[1])

    # ── (a) Grouped bar ───────────────────────────────────────────────────────
    groups  = list(hrsl.keys())
    x       = np.arange(len(classes))
    bw      = 0.18
    n_g     = len(groups)
    offsets = np.linspace(-(n_g - 1) / 2, (n_g - 1) / 2, n_g) * bw * 1.1
    any_data = False

    for idx, grp in enumerate(groups):
        color    = GROUP_COLORS[grp]
        vals_bau = [v / 1000 for v in exposure[grp]["bau"]]
        vals_eco = [v / 1000 for v in exposure[grp]["eco"]]
        if sum(vals_bau) + sum(vals_eco) > 0:
            any_data = True
        ax_bar.bar(x + offsets[idx] - bw * 0.15, vals_bau, bw * 0.48,
                   color=color, alpha=0.92, edgecolor="none", zorder=3)
        ax_bar.bar(x + offsets[idx] + bw * 0.15, vals_eco, bw * 0.48,
                   color=color, alpha=0.38, hatch="///",
                   edgecolor=color, linewidth=0.4, zorder=3)

    if not any_data:
        ax_bar.text(0.5, 0.5, "HRSL rasters not found",
                    transform=ax_bar.transAxes, ha="center", va="center",
                    fontsize=10, color=BOUNDARY_ILCE, style="italic")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(HAZARD_LABELS, fontsize=9,
                            color=BOUNDARY_IL, fontweight="bold")
    ax_bar.set_ylabel("Exposed Population (thousands)", fontweight="bold")
    ax_bar.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:,.1f}"))
    ax_bar.grid(axis="y", linestyle="--", linewidth=0.5,
                alpha=0.45, color=GRID_COLOR, zorder=0)
    ax_bar.spines["bottom"].set_visible(True)
    ax_bar.spines["bottom"].set_linewidth(1.5)
    ax_bar.spines["left"].set_linewidth(1.5)
    ax_bar.tick_params(axis="x", length=0, pad=6)

    leg_handles = [Patch(facecolor=GROUP_COLORS[g], label=g) for g in groups]
    leg_handles += [
        Patch(facecolor="#888", alpha=0.92, label="BAU (solid)"),
        Patch(facecolor="#888", alpha=0.38, hatch="///",
              edgecolor="#888", linewidth=0.4, label="ECO (hatched)"),
    ]
    ax_bar.legend(handles=leg_handles, fontsize=8, loc="upper right",
                  frameon=True, facecolor="white", edgecolor=GRID_COLOR)
    ax_bar.set_title("(a)", loc="left", fontweight="bold", fontsize=18, pad=8)

    # ── Connector lines BAU → ECO per group per class ─────────────────────────
    if any_data:
        for cls_idx in range(len(classes)):
            for grp_idx, grp in enumerate(groups):
                bv = exposure[grp]["bau"][cls_idx] / 1000
                ev = exposure[grp]["eco"][cls_idx] / 1000
                if bv == 0 and ev == 0:
                    continue
                cx_b = x[cls_idx] + offsets[grp_idx] - bw * 0.15
                cx_e = x[cls_idx] + offsets[grp_idx] + bw * 0.15
                ax_bar.plot([cx_b, cx_e], [bv, ev],
                            color=ECO_COLOR if ev <= bv else BAU_COLOR,
                            lw=0.85, alpha=0.55, zorder=4,
                            solid_capstyle="round")


    # ── (b) VPEI map — fig07-panel-b style ───────────────────────────────────
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    shift  = 15
    sm = np.zeros_like(boundary, bool)
    sm[shift:, shift:] = boundary[:-shift, :-shift]
    sa = np.zeros(boundary.shape, float); sa[~sm] = np.nan
    ax_vpei.imshow(sa, cmap="Greys", vmin=0, vmax=1,
                   alpha=0.4, extent=extent, zorder=1)
    ax_vpei.imshow(hillshade, cmap="gray", extent=extent, zorder=2, alpha=0.6)

    # Quantile breaks from actual non-zero VPEI values on BAU growth pixels
    _vv = vpei_map[~np.isnan(vpei_map)]
    _vv_nz = _vv[_vv > 1e-6]   # exclude true-zero pixels (no HRSL)
    if len(_vv_nz) > 10 and (_vv_nz.max() - _vv_nz.min()) > 1e-4:
        _q        = np.percentile(_vv_nz, [25, 50, 75])
        vpei_bins = [float(_vv_nz.min()) - 1e-6,
                     float(_q[0]), float(_q[1]), float(_q[2]),
                     float(_vv_nz.max()) + 1e-6]
    else:
        # Fallback fixed bins (HRSL data absent or flat)
        vpei_bins = [0.0, 0.15, 0.30, 0.45, 0.60]
    vpei_colors = [elite_cmap(v) for v in [0.05, 0.35, 0.65, 0.95]]
    vpei_cmap   = mcolors.ListedColormap(vpei_colors)
    vpei_norm   = mcolors.BoundaryNorm(vpei_bins, vpei_cmap.N, clip=True)
    im = ax_vpei.imshow(vpei_map, cmap=vpei_cmap, norm=vpei_norm,
                        extent=extent, origin="upper",
                        interpolation="nearest", zorder=3, alpha=0.85)

    # VPEI quartile contours — thin strokes at Q1/Q2/Q3 boundaries
    # NaN filled with value below minimum so contour ignores non-BAU areas
    _vfill   = np.where(np.isnan(vpei_map), vpei_bins[0] - 1.0, vpei_map)
    _levels  = vpei_bins[1:-1]   # [Q25, Q50, Q75] boundary values
    _ccolors = [vpei_colors[i + 1] for i in range(len(_levels))]
    if len(_levels) > 0 and _levels[-1] > _levels[0]:   # skip if flat data
        ax_vpei.contour(_vfill, levels=_levels,
                        colors=_ccolors, linewidths=0.65,
                        extent=extent, origin="upper",
                        alpha=0.65, zorder=6)

    ax_vpei.contour(boundary, levels=[0.5], colors=BOUNDARY_IL,
                    linewidths=0.8, extent=extent, origin="upper", zorder=4)

    # Admin boundaries + labels
    import geopandas as gpd
    basin_gpkg = INTERIM / "kuzey_ege_havzasi_boundary.gpkg"
    if basin_gpkg.exists():
        basin_geom = gpd.read_file(basin_gpkg).to_crs(crs).union_all()
    else:
        from shapely.geometry import box as _bbox
        basin_geom = _bbox(bounds.left, bounds.bottom, bounds.right, bounds.top)
    il_gdf, ilce_gdf = load_admin_boundaries(crs)
    has_admin = (il_gdf is not None) or (ilce_gdf is not None)
    if has_admin:
        add_boundaries(ax_vpei, il_gdf, ilce_gdf, basin_geom)
    add_province_labels(ax_vpei, il_gdf, basin_geom)

    # Coordinate ticks
    _tr   = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    _ymid = bounds.bottom + (bounds.top   - bounds.bottom) / 2
    _xmid = bounds.left   + (bounds.right - bounds.left)   / 2

    def fmt_x(v, _):
        try: lon, _ = _tr.transform(v, _ymid); return f"{lon:.2f}°E"
        except: return ""

    def fmt_y(v, _):
        try: _, lat = _tr.transform(_xmid, v); return f"{lat:.2f}°N"
        except: return ""

    ax_vpei.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax_vpei.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax_vpei.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_x))
    ax_vpei.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_y))
    plt.setp(ax_vpei.get_yticklabels(), rotation=90, va="center")
    ax_vpei.spines["bottom"].set_visible(True)
    ax_vpei.spines["left"].set_visible(True)
    ax_vpei.spines["bottom"].set_linewidth(1.5)
    ax_vpei.spines["left"].set_linewidth(1.5)

    add_scale_north(ax_vpei, bounds)

    # Discrete VPEI legend + admin boundaries in one legend
    vpei_labels  = [f"Q{i+1}  ({vpei_bins[i]:.2f}–{vpei_bins[i+1]:.2f})"
                    for i in range(4)]
    leg_items = [Patch(facecolor=c, label=l, edgecolor="white", linewidth=0.4)
                 for c, l in zip(vpei_colors, vpei_labels)]
    if has_admin:
        leg_items += [
            Line2D([0],[0], color=BOUNDARY_IL,   lw=0.85, label="Province"),
            Line2D([0],[0], color=BOUNDARY_ILCE, lw=0.40,
                   linestyle="--", label="District"),
        ]
    ax_vpei.legend(handles=leg_items, loc="lower left",
                   bbox_to_anchor=(0.0, 0.09),
                   fontsize=8, frameon=True,
                   facecolor="white", edgecolor=BOUNDARY_ILCE,
                   title="VPEI", title_fontsize=8.5)

    ax_vpei.set_title("(b)", loc="left", fontweight="bold", fontsize=18, pad=8)

    # ── Agricultural loss inset — upper right of panel (b) ────────────────────
    fig.canvas.draw()
    pos_b = ax_vpei.get_position()
    _iw   = pos_b.width  * 0.26 * 1.2
    _ih   = pos_b.height * 0.24 * 1.2
    ax_ins = fig.add_axes(
        [pos_b.x0 + pos_b.width - _iw,
         pos_b.y0 + pos_b.height - _ih,
         _iw, _ih]
    )

    xb  = np.arange(len(classes))
    bwi = 0.24
    ax_ins.bar(xb - bwi/2, agri_bau, bwi, color=BAU_COLOR,
               alpha=0.90, edgecolor="white", linewidth=0.4,
               label="BAU", zorder=3)
    ax_ins.bar(xb + bwi/2, agri_eco, bwi, color=ECO_COLOR,
               alpha=0.90, edgecolor="white", linewidth=0.4,
               label="ECO", zorder=3)

    ax_ins.set_xticks(xb)
    ax_ins.set_xticklabels(["VL","L","M","H","VH"], fontsize=6,
                            color=BOUNDARY_IL, fontweight="bold")
    ax_ins.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax_ins.tick_params(axis="y", labelsize=6, length=2, pad=2,
                       colors=BOUNDARY_ILCE)
    ax_ins.tick_params(axis="x", length=0, pad=3)
    ax_ins.set_ylabel("ha", fontsize=6, labelpad=2, color=BOUNDARY_ILCE)
    ax_ins.grid(axis="y", linestyle=":", linewidth=0.4,
                alpha=0.40, color=GRID_COLOR)
    ax_ins.set_facecolor("none")
    ax_ins.patch.set_alpha(0.0)
    ax_ins.set_title("Agri. Land Loss",
                     fontsize=7, fontweight="bold",
                     color=BOUNDARY_IL, pad=4, loc="left")
    ax_ins.legend(fontsize=6, frameon=False,
                  loc="upper center", ncol=2,
                  handlelength=1, handletextpad=0.4, columnspacing=0.8)
    for sp in ax_ins.spines.values():
        sp.set_visible(False)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = FIG_DIR / "fig09_Demographic_Exposure.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    create_fig09()
