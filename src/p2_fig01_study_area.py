import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from scipy.ndimage import binary_fill_holes
import rasterio
from rasterio.warp import transform
from matplotlib.colors import ListedColormap, LightSource
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in locals() else Path(".")
TABLES    = PROJECT_ROOT / "outputs" / "tables" / "paper2"
PROCESSED = PROJECT_ROOT / "data" / "03_processed" / "paper2"
DRIVERS   = PROCESSED / "drivers"
FIG_DIR   = PROJECT_ROOT / "outputs" / "figures" / "paper2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY_IL   = "#2c3e50"   # dark navy  — province boundary
BOUNDARY_ILCE = "#636e72"   # medium gray — district boundary


# ─── ADMIN BOUNDARIES (same logic as fig05) ───────────────────────────────────

def _name_col(gdf):
    for c in ("name", "NAME", "ADM1_EN", "ADM1_TR", "province",
              "il_adi", "PROVINCE", "name:tr"):
        if c in gdf.columns:
            return c
    return None


def load_admin_boundaries(crs_target):
    """Province and district boundaries via OSMnx geocode (cached after first run)."""
    import geopandas as gpd
    import pandas as pd

    CACHE_DIR  = PROJECT_ROOT / "data" / "02_interim" / "paper2"
    cache_il   = CACHE_DIR / "admin_il.gpkg"
    cache_ilce = CACHE_DIR / "admin_ilce.gpkg"

    if cache_il.exists() and cache_ilce.exists():
        il   = gpd.read_file(cache_il).to_crs(crs_target)
        ilce = gpd.read_file(cache_ilce).to_crs(crs_target)
        if len(il) > 0 and len(ilce) > 0:
            print(f"  Cache: {len(il)} provinces, {len(ilce)} districts")
            return il, ilce

    try:
        import osmnx as ox

        # İzmir R223167 ve Çanakkale R223471 by_osmid=True gerektiriyor —
        # Nominatim doğru poligonu sadece bu şekilde döndürüyor.
        PROVINCES = [
            ("R223167",               "Izmir",     True),
            ("Manisa Province, Turkey", "Manisa",  False),
            ("Balikesir, Turkey",     "Balikesir", False),
            ("R223471",               "Canakkale", True),
            ("Usak, Turkey",          "Usak",      False),
        ]

        print("  Fetching province boundaries via OSMnx geocode ...")
        il_parts = []
        for q, short, by_id in PROVINCES:
            try:
                g = ox.geocode_to_gdf(q, by_osmid=by_id)
                g["name"] = short
                il_parts.append(g[["name", "geometry"]])
                print(f"    {short}: OK")
            except Exception as e:
                print(f"    {short}: {e}")

        il_gdf = None
        if il_parts:
            import pandas as _pd
            il_gdf = (gpd.GeoDataFrame(_pd.concat(il_parts, ignore_index=True),
                                        crs="EPSG:4326")
                      .to_crs(crs_target))
            il_gdf[["name", "geometry"]].to_file(cache_il, driver="GPKG")
            print(f"  Provinces cached: {len(il_gdf)}")

        MIN_ILCE_AREA_M2 = 30_000_000  # 30 km²
        ilce_gdf_raw = None
        try:
            basin_gpkg = PROJECT_ROOT / "data" / "02_interim" / "paper2" / "kuzey_ege_havzasi_boundary.gpkg"
            if basin_gpkg.exists():
                basin_wgs  = gpd.read_file(basin_gpkg).to_crs("EPSG:4326")
                basin_poly = basin_wgs.union_all()
                print("  Fetching district boundaries within basin polygon ...")
                tags = {"boundary": "administrative", "admin_level": "6"}
                g    = ox.features_from_polygon(basin_poly, tags=tags)
                polys   = g[g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
                polys_m = polys.to_crs("EPSG:32635")
                polys   = polys[polys_m.geometry.area >= MIN_ILCE_AREA_M2].reset_index(drop=True)
                if "name" not in polys.columns:
                    polys["name"] = ""
                ilce_gdf_raw = polys[["name", "geometry"]]
                print(f"  Basin ilce: {len(ilce_gdf_raw)} districts after area filter")
        except Exception as e:
            print(f"  Basin ilce fetch failed: {e}")

        ilce_gdf = None
        if ilce_gdf_raw is not None:
            ilce_gdf = (gpd.GeoDataFrame(ilce_gdf_raw, crs="EPSG:4326")
                        .to_crs(crs_target))
            ilce_gdf[["name", "geometry"]].to_file(cache_ilce, driver="GPKG")
            print(f"  Districts cached: {len(ilce_gdf)}")

        return il_gdf, ilce_gdf

    except Exception as exc:
        print(f"  OSMnx unavailable ({exc}). Trying local fallback ...")
        local = (PROJECT_ROOT / "data" / "00_external"
                 / "akilli_sehir_db" / "ilce.shp")
        if local.exists():
            ilce = gpd.read_file(local).to_crs(crs_target)
            print(f"  Local fallback: {local.name} ({len(ilce)} features)")
            return None, ilce
        return None, None


def add_boundaries(ax, il_gdf, ilce_gdf, basin_geom):
    """Clip to basin polygon before plotting — nothing outside study area."""
    try:
        if ilce_gdf is not None:
            clipped = ilce_gdf.clip(basin_geom)
            clipped = clipped[~clipped.is_empty & clipped.is_valid]
            if len(clipped):
                clipped.boundary.plot(
                    ax=ax, color=BOUNDARY_ILCE, linewidth=0.35,
                    linestyle="--", zorder=5, alpha=0.65)
        if il_gdf is not None:
            clipped = il_gdf.clip(basin_geom)
            clipped = clipped[~clipped.is_empty & clipped.is_valid]
            if len(clipped):
                clipped.boundary.plot(
                    ax=ax, color=BOUNDARY_IL, linewidth=0.75,
                    zorder=5, alpha=0.80)
    except Exception as exc:
        print(f"  Boundary overlay skipped: {exc}")


def add_province_labels(ax, il_gdf, basin_geom):
    """Labels centred on the clipped (within-study-area) part of each province."""
    if il_gdf is None:
        return
    try:
        nc = _name_col(il_gdf)
        if nc is None:
            return
        clipped = il_gdf.clip(basin_geom)
        clipped = clipped[~clipped.geometry.is_empty]
        for _, row in clipped.iterrows():
            name = str(row.get(nc, "")).strip()
            if not name or name.lower() in ("none", "nan", ""):
                continue
            pt = row.geometry.representative_point()
            ax.text(pt.x, pt.y, name.upper(), fontsize=7.5,
                    ha="center", va="center",
                    color=BOUNDARY_IL, fontweight="bold",
                    alpha=0.90, zorder=8,
                    path_effects=[pe.withStroke(linewidth=3.5,
                                                foreground="white")])
    except Exception as exc:
        print(f"  Province labels skipped: {exc}")


# ─── SCALE BAR + NORTH ARROW (same positioning as fig05) ─────────────────────

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
    ax.text(sb_x0 + slen, sb_y0 + sb_h * 1.65,
            f"{int(slen / 1000)} km",
            ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=6)

    na_base = sb_y0 + sb_h * 5.2
    na_w, na_h = W * 0.028, H * 0.055
    ax.add_patch(Polygon(
        [[cx,          na_base + na_h],
         [cx + na_w/2, na_base],
         [cx,          na_base + na_h * 0.25],
         [cx - na_w/2, na_base]],
        facecolor=BOUNDARY_IL, edgecolor="white", lw=1.2, zorder=6))
    ax.text(cx, na_base + na_h + H * 0.005, "N",
            ha="center", va="bottom", fontsize=10,
            fontweight="bold", color=BOUNDARY_IL, zorder=6)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def create_fig01_advanced():
    print("Generating Fig 01 (Historical Growth & LULC Base Map)...")

    fig = plt.figure(figsize=(15, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.4], wspace=0.08)

    # ── Panel A: Historical urban expansion ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    df_area = pd.read_csv(TABLES / "annual_impervious_area_1985_2022.csv")

    years_obs = df_area["Year"].values
    area_obs  = df_area["Urban_Area_ha"].values

    # Extend series to 2026 using a linear trend fit on the full observed period.
    TARGET_YEAR = 2026
    slope, intercept = np.polyfit(years_obs, area_obs, 1)
    ext_years = np.arange(years_obs[-1] + 1, TARGET_YEAR + 1)
    ext_area  = slope * ext_years + intercept

    years = np.concatenate([years_obs, ext_years])
    area  = np.concatenate([area_obs,  ext_area])

    ax1.plot(years, area, color="#ff7675", lw=2.5, marker="o",
             markersize=5, zorder=3, markeredgecolor="white")
    ax1.fill_between(years, area, color="#ff7675", alpha=0.2, zorder=2)

    ax1.set_title("(a)", loc="left", fontweight="bold", fontsize=14)
    ax1.set_ylabel("Total Urban Area (Hectares)", fontweight="bold")
    ax1.set_xlabel("Year", fontweight="bold")
    ax1.set_xlim(years.min(), years.max())
    ax1.set_ylim(0, area.max() * 1.15)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)

    start_val = area[0]
    end_val   = area[-1]
    end_year  = int(years[-1])
    total_inc = end_val - start_val
    pct_inc   = (total_inc / start_val) * 100
    n_years   = years[-1] - years[0]
    cagr      = ((end_val / start_val) ** (1 / n_years) - 1) * 100

    stats_text = (
        f"Statistical Highlights\n"
        f"-------------------------\n"
        f"Initial Area (1985)  : {start_val:,.0f} ha\n"
        f"Final Area ({end_year}) : {end_val:,.0f} ha\n"
        f"Absolute Increase  : +{total_inc:,.0f} ha\n"
        f"Total Growth Rate  : +{pct_inc:.1f}%\n"
        f"Comp. Annual Growth : {cagr:.2f}% / yr"
    )
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment="top", linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.8", facecolor="white",
                       alpha=0.9, edgecolor="#ff7675", lw=1.5),
             zorder=4)

    # ── Panel B: LULC 2020 map ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    lulc_path = PROCESSED / "lulc_simplified_2020.tif"
    elev_path = DRIVERS   / "driver_elevation.tif"

    with rasterio.open(lulc_path) as src:
        lulc          = src.read(1)
        src_crs       = src.crs
        bounds        = src.bounds
        nodata        = src.nodata
        valid_mask    = (lulc != nodata) if nodata is not None else ~np.isnan(lulc)
    boundary_mask = binary_fill_holes(valid_mask)

    with rasterio.open(elev_path) as src_elev:
        elev = src_elev.read(1)
        elev[~valid_mask] = 0

    ls        = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(elev, vert_exag=3, dx=30, dy=30)

    shift_px    = 15
    shadow_mask = np.zeros_like(boundary_mask, dtype=bool)
    shadow_mask[shift_px:, shift_px:] = boundary_mask[:-shift_px, :-shift_px]

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    shadow_arr = np.zeros_like(lulc, dtype=float)
    shadow_arr[~shadow_mask] = np.nan
    ax2.imshow(shadow_arr, cmap="Greys", vmin=0, vmax=1,
               alpha=0.4, extent=extent, zorder=1)

    hill_masked = np.ma.masked_where(~valid_mask, hillshade)
    ax2.imshow(hill_masked, cmap="gray", extent=extent, zorder=2, alpha=0.6)

    ax2.contour(boundary_mask, levels=[0.5], colors=BOUNDARY_IL,
                linewidths=0.8, extent=extent, origin="upper", zorder=4)

    lulc_colors = ["#ff7675", "#fab1a0", "#00cec9", "#74b9ff", "#dfe6e9", "#a29bfe"]
    cmap_lulc   = ListedColormap(lulc_colors)
    lulc_masked = np.ma.masked_where(~valid_mask, lulc)
    ax2.imshow(lulc_masked, cmap=cmap_lulc, extent=extent,
               zorder=3, alpha=0.85, interpolation="nearest")

    # Admin boundaries + province labels (same logic as fig05 panel a)
    import geopandas as gpd
    basin_gpkg = PROJECT_ROOT / "data" / "02_interim" / "paper2" / "kuzey_ege_havzasi_boundary.gpkg"
    if basin_gpkg.exists():
        basin_geom = gpd.read_file(basin_gpkg).to_crs(src_crs).union_all()
    else:
        from shapely.geometry import box as _bbox
        basin_geom = _bbox(bounds.left, bounds.bottom, bounds.right, bounds.top)

    il_gdf, ilce_gdf = load_admin_boundaries(src_crs)
    has_admin = (il_gdf is not None) or (ilce_gdf is not None)

    if has_admin:
        add_boundaries(ax2, il_gdf, ilce_gdf, basin_geom)

    add_province_labels(ax2, il_gdf, basin_geom)

    # Coordinate tick labels
    def format_x(value, _):
        try:
            lon, _ = transform(src_crs, "EPSG:4326",
                               [value],
                               [bounds.bottom + (bounds.top - bounds.bottom) / 2])
            return f"{lon[0]:.2f}°E"
        except:
            return ""

    def format_y(value, _):
        try:
            _, lat = transform(src_crs, "EPSG:4326",
                               [bounds.left + (bounds.right - bounds.left) / 2],
                               [value])
            return f"{lat[0]:.2f}°N"
        except:
            return ""

    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_x))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_y))
    plt.setp(ax2.get_yticklabels(), rotation=90, va="center")

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_linewidth(1.5)
    ax2.spines["left"].set_linewidth(1.5)

    ax2.set_title("(b)", loc="left", fontweight="bold", fontsize=14)

    # Scale bar + north arrow — same positioning as fig05
    add_scale_north(ax2, bounds)

    # LULC legend (lower left) + admin boundary entries
    lulc_labels  = ["Urban", "Agriculture", "Forest", "Water", "Bareland", "Other"]
    lulc_patches = [Patch(color=c, label=l)
                    for c, l in zip(lulc_colors, lulc_labels)]
    if has_admin:
        lulc_patches += [
            Line2D([0], [0], color=BOUNDARY_IL,   lw=0.85, label="Province boundary"),
            Line2D([0], [0], color=BOUNDARY_ILCE, lw=0.40,
                   linestyle="--", label="District boundary"),
        ]
    ax2.legend(handles=lulc_patches, loc="lower left",
               framealpha=0.9, title="LULC Classes",
               title_fontproperties={"weight": "bold"})

    out_path = FIG_DIR / "fig01_Baseline.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path.name}")


if __name__ == "__main__":
    create_fig01_advanced()
