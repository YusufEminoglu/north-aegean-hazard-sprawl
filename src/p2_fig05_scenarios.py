import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path
from scipy.ndimage import binary_fill_holes
from matplotlib.colors import LinearSegmentedColormap, LightSource
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.lines import Line2D
from pyproj import Transformer
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": 14,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in locals() else Path(".")
PROCESSED    = PROJECT_ROOT / "data" / "03_processed" / "paper2"
DRIVERS_DIR  = PROCESSED / "drivers"
FIG_DIR      = PROJECT_ROOT / "outputs" / "figures" / "paper2"

# ── Colour palette ────────────────────────────────────────────────────────────
COLOR_EXISTING = "#00cec9"   # bright teal  — existing urban 2020 (both panels)
COLOR_GROWTH   = "#ff7675"   # coral        — new urban growth   (both panels)
BOUNDARY_IL    = "#2c3e50"   # dark navy    — province boundary
BOUNDARY_ILCE  = "#636e72"   # medium gray  — district boundary

# Short LULC class names for inset x-axis (vertical bars)
SHORT_LULC = {2: "Agri.", 3: "Forest", 5: "Bare.", 6: "Other"}


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def get_sprawl_arrays(sim_path, lulc_path):
    with rasterio.open(lulc_path) as src:
        lulc   = src.read(1)
        nodata = src.nodata
        bounds = src.bounds
        crs    = src.crs
    with rasterio.open(sim_path) as src:
        sim = src.read(1)
    valid = (lulc != nodata) if nodata is not None else np.ones_like(lulc, bool)
    out   = np.zeros_like(lulc, dtype=np.uint8)
    out[valid & (lulc == 1)]              = 1   # existing urban
    out[valid & (lulc != 1) & (sim == 1)] = 2   # new growth
    arr = np.ma.masked_where((out == 0) | (~valid), out)
    return arr, bounds, crs, valid, lulc, nodata


def load_admin_boundaries(crs_target):
    """Province and district boundaries via OSMnx geocode (cached after first run)."""
    import geopandas as gpd
    import pandas as pd

    CACHE_DIR  = PROJECT_ROOT / "data" / "02_interim" / "paper2"
    cache_il   = CACHE_DIR / "admin_il.gpkg"
    cache_ilce = CACHE_DIR / "admin_ilce.gpkg"

    # 1. Load from cache
    if cache_il.exists() and cache_ilce.exists():
        il   = gpd.read_file(cache_il).to_crs(crs_target)
        ilce = gpd.read_file(cache_ilce).to_crs(crs_target)
        if len(il) > 0 and len(ilce) > 0:
            print(f"  Cache: {len(il)} provinces, {len(ilce)} districts")
            return il, ilce

    # 2. OSMnx via geocode — one query per province (fast & reliable)
    try:
        import osmnx as ox

        # (query, short_label, by_osmid)
        # Izmir R223167 and Canakkale R223471 must use by_osmid — Nominatim
        # returns wrong geometry for these two without the OSM relation ID.
        PROVINCES = [
            ("R223167",               "Izmir",     True),
            ("Manisa Province, Turkey", "Manisa",  False),
            ("Balikesir, Turkey",     "Balikesir", False),
            ("R223471",               "Canakkale", True),
            ("Usak, Turkey",          "Usak",      False),
        ]

        # --- Province boundaries
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
            il_gdf = (gpd.GeoDataFrame(pd.concat(il_parts, ignore_index=True),
                                       crs="EPSG:4326")
                      .to_crs(crs_target))
            il_gdf[["name", "geometry"]].to_file(cache_il, driver="GPKG")
            print(f"  Provinces cached: {len(il_gdf)}")

        # --- District boundaries (features_from_place per province)
        print("  Fetching district boundaries via OSMnx features_from_place ...")
        # Fetch all admin_level=6 features within the basin polygon in one query.
        # This avoids per-province fetching and the mahalle contamination caused
        # by Izmir's non-standard OSM tagging (mahalle tagged as admin_level=6).
        # After fetching, keep only features whose area is >= 30 km² to
        # exclude any residual sub-district polygons.
        MIN_ILCE_AREA_M2 = 30_000_000  # 30 km²

        ilce_gdf_raw = None
        try:
            basin_gpkg = PROJECT_ROOT / "data" / "02_interim" / "paper2" / "kuzey_ege_havzasi_boundary.gpkg"
            if basin_gpkg.exists():
                basin_wgs = gpd.read_file(basin_gpkg).to_crs("EPSG:4326")
                basin_poly = basin_wgs.union_all()
                print("  Fetching district boundaries within basin polygon ...")
                tags = {"boundary": "administrative", "admin_level": "6"}
                g = ox.features_from_polygon(basin_poly, tags=tags)
                polys = g[g.geometry.geom_type.isin(
                    ["Polygon", "MultiPolygon"])].reset_index(drop=True)
                polys_m = polys.to_crs("EPSG:32635")
                polys = polys[polys_m.geometry.area >= MIN_ILCE_AREA_M2].reset_index(drop=True)
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


def _name_col(gdf):
    for c in ("name", "NAME", "ADM1_EN", "ADM1_TR", "province",
              "il_adi", "PROVINCE", "name:tr"):
        if c in gdf.columns:
            return c
    return None


# ─── MAP FURNITURE ────────────────────────────────────────────────────────────

def add_basemap(ax, elev, valid, bounds):
    bnd  = binary_fill_holes(valid)
    ev   = elev.copy().astype(float);  ev[~valid] = 0
    hs   = LightSource(azdeg=315, altdeg=45).hillshade(ev, vert_exag=3,
                                                        dx=30, dy=30)
    ext  = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    sm   = np.zeros_like(bnd, bool);  sm[15:, 15:] = bnd[:-15, :-15]
    ax.imshow(np.where(~sm, np.nan, 0.0), cmap="Greys",
              vmin=0, vmax=1, alpha=0.4, extent=ext, zorder=1)
    ax.imshow(np.ma.masked_where(~valid, hs),
              cmap="gray", extent=ext, zorder=2, alpha=0.6)
    ax.contour(bnd, levels=[0.5], colors=BOUNDARY_IL,
               linewidths=0.8, extent=ext, origin="upper", zorder=4)
    return ext


def style_axes(ax, bounds, crs):
    _tr   = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    _ymid = bounds.bottom + (bounds.top   - bounds.bottom) / 2
    _xmid = bounds.left   + (bounds.right - bounds.left)   / 2

    def fmt_x(v, _):
        try:
            lon, _ = _tr.transform(v, _ymid)
            return f"{lon:.2f}°E"
        except: return ""

    def fmt_y(v, _):
        try:
            _, lat = _tr.transform(_xmid, v)
            return f"{lat:.2f}°N"
        except: return ""

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_x))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_y))
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")
    ax.tick_params(labelsize=8)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(1.5)


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


def add_boundaries(ax, il_gdf, ilce_gdf, basin_geom):
    """Clip to actual basin polygon before plotting — nothing outside study area."""
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
            # representative_point() is guaranteed inside the polygon
            # (centroid can fall outside for crescent-shaped clipped regions)
            pt = row.geometry.representative_point()
            cx, cy = pt.x, pt.y
            ax.text(cx, cy, name.upper(), fontsize=7.5,
                    ha="center", va="center",
                    color=BOUNDARY_IL, fontweight="bold",
                    alpha=0.90, zorder=8,
                    path_effects=[pe.withStroke(linewidth=3.5,
                                                foreground="white")])
    except Exception as exc:
        print(f"  Province labels skipped: {exc}")


def add_panel_legend(ax, has_admin):
    """In-axes legend at upper-right — 'Map Keys', no Turkish names."""
    handles = [
        Patch(facecolor=COLOR_EXISTING, edgecolor="none",
              label="Existing Urban (2020)"),
        Patch(facecolor=COLOR_GROWTH, edgecolor="none",
              label="New Urban Growth (2050)"),
    ]
    if has_admin:
        handles += [
            Line2D([0], [0], color=BOUNDARY_IL, lw=0.85,
                   label="Province boundary"),
            Line2D([0], [0], color=BOUNDARY_ILCE, lw=0.40,
                   linestyle="--", label="District boundary"),
        ]
    ax.legend(handles=handles, loc="upper right",
              title="Map Keys", title_fontsize=9,
              fontsize=8.5, frameon=True,
              facecolor="white", edgecolor=BOUNDARY_IL,
              framealpha=0.92, borderpad=0.8)


# ─── INSET CHART (vertical bars, transparent background) ─────────────────────

def add_inset(ax_map, lulc, growth_mask, nodata):
    """Vertical bar chart of converted land type — lower-left, no background."""
    nd_ok     = (lulc != nodata) if nodata is not None else np.ones_like(lulc, bool)
    converted = lulc[growth_mask & nd_ok]

    labels, pcts = [], []
    for cls, lbl in SHORT_LULC.items():
        ha  = float(np.sum(converted == cls)) * 900.0 / 10_000.0
        total_ha = float(np.sum(growth_mask)) * 900.0 / 10_000.0
        if ha > 0.0 and total_ha > 0.0:
            labels.append(lbl)
            pcts.append(ha / total_ha * 100.0)

    if not pcts:
        return

    ax_ins = inset_axes(ax_map, width="27%", height="28%",
                         loc="lower left", borderpad=2.5)

    bars = ax_ins.bar(labels, pcts, color=COLOR_GROWTH, width=0.72,
                       edgecolor="white", linewidth=0.5, alpha=0.88)

    for bar, p in zip(bars, pcts):
        ax_ins.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{p:.0f}%", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold", color="#2c3e50")

    ax_ins.set_ylim(0, max(pcts) * 1.35)
    ax_ins.set_ylabel("Share (%)", fontsize=6.5, labelpad=2)
    ax_ins.set_title("Converted land type",
                      fontsize=7, fontweight="bold", pad=3, loc="left")

    # X-axis labels horizontal — short names fit without rotation
    ax_ins.tick_params(axis="x", labelsize=7, length=2, pad=2)
    ax_ins.tick_params(axis="y", labelsize=6.5, length=2, pad=2)

    for sp in ("top", "right"):
        ax_ins.spines[sp].set_visible(False)
    ax_ins.spines["left"].set_visible(True)
    ax_ins.spines["left"].set_linewidth(0.7)
    ax_ins.spines["bottom"].set_visible(True)
    ax_ins.spines["bottom"].set_linewidth(0.7)

    # Transparent background — map shows through
    ax_ins.set_facecolor("none")
    ax_ins.patch.set_alpha(0.0)
    ax_ins.grid(axis="y", linestyle="--", alpha=0.30, linewidth=0.5)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def create_fig3():
    print("Generating Figure 5: CA Simulation Scenarios ...")

    lulc_path = PROCESSED / "lulc_simplified_2020.tif"
    bau_path  = PROCESSED / "sim_2050_bau.tif"
    eco_path  = PROCESSED / "sim_2050_eco.tif"

    if not bau_path.exists() or not eco_path.exists():
        print("Error: Missing CA simulation files."); return

    arr_bau, bounds, crs, valid, lulc_base, lulc_nd = \
        get_sprawl_arrays(bau_path, lulc_path)
    arr_eco, *_ = get_sprawl_arrays(eco_path, lulc_path)

    def gmask(arr):
        if isinstance(arr, np.ma.MaskedArray):
            return (~arr.mask) & (arr.data == 2)
        return arr == 2

    with rasterio.open(DRIVERS_DIR / "driver_elevation.tif") as src:
        elev = src.read(1)

    il_gdf, ilce_gdf = load_admin_boundaries(crs)
    has_admin = (il_gdf is not None) or (ilce_gdf is not None)

    # Basin polygon for clipping admin boundaries to study area only
    import geopandas as gpd
    basin_path = PROJECT_ROOT / "data" / "02_interim" / "paper2" / "kuzey_ege_havzasi_boundary.gpkg"
    if basin_path.exists():
        basin_geom = gpd.read_file(basin_path).to_crs(crs).union_all()
    else:
        # Fallback: raster extent
        from shapely.geometry import box as _bbox
        basin_geom = _bbox(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Both panels share the same colormap
    cm  = mcolors.ListedColormap([COLOR_EXISTING, COLOR_GROWTH])
    nrm = mcolors.BoundaryNorm([0.5, 1.5, 2.5], cm.N)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    panels = [
        (axes[0], arr_bau, "(a)",
         "2050 Business-as-Usual (BAU) Scenario",  gmask(arr_bau)),
        (axes[1], arr_eco, "(b)",
         "2050 Ecological Protection (ECO) Scenario", gmask(arr_eco)),
    ]

    for ax, arr, lbl, title, gm in panels:
        extent = add_basemap(ax, elev, valid, bounds)
        style_axes(ax, bounds, crs)

        if has_admin:
            add_boundaries(ax, il_gdf, ilce_gdf, basin_geom)

        ax.imshow(arr, cmap=cm, norm=nrm, interpolation="nearest",
                  extent=extent, zorder=3, alpha=0.90)

        # Province name labels — centred on within-basin portion only
        add_province_labels(ax, il_gdf, basin_geom)

        add_scale_north(ax, bounds)
        add_panel_legend(ax, has_admin)
        add_inset(ax, lulc_base, gm, lulc_nd)

        ax.text(0.0,  1.02, lbl,   transform=ax.transAxes,
                fontweight="bold", fontsize=18, va="bottom")
        ax.text(0.08, 1.02, title, transform=ax.transAxes,
                fontweight="bold", fontsize=12, va="bottom")

    plt.tight_layout(pad=0.5)
    out_path = FIG_DIR / "fig05_CA_Scenarios.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close()
    print(f"  Saved: {out_path.name}")


if __name__ == "__main__":
    create_fig3()
