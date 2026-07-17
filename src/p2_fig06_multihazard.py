import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, LightSource
from matplotlib.patches import Polygon, Rectangle, Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from scipy.ndimage import binary_fill_holes
from pyproj import Transformer
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in locals() else Path(".")
INTERIM     = PROJECT_ROOT / "data" / "02_interim"  / "paper2"
PROCESSED   = PROJECT_ROOT / "data" / "03_processed" / "paper2"
DRIVERS_DIR = PROCESSED / "drivers"
FIG_DIR     = PROJECT_ROOT / "outputs" / "figures"   / "paper2"

elite_cmap = LinearSegmentedColormap.from_list("elite", ["#dfe6e9", "#00cec9", "#ff7675"])

BOUNDARY_IL   = "#2c3e50"
BOUNDARY_ILCE = "#636e72"


# ─── ADMIN BOUNDARIES (same logic as fig01 / fig05) ──────────────────────────

def _name_col(gdf):
    for c in ("name", "NAME", "ADM1_EN", "ADM1_TR", "province",
              "il_adi", "PROVINCE", "name:tr"):
        if c in gdf.columns:
            return c
    return None


def load_admin_boundaries(crs_target):
    import geopandas as gpd
    import pandas as _pd

    cache_il   = INTERIM / "admin_il.gpkg"
    cache_ilce = INTERIM / "admin_ilce.gpkg"

    if cache_il.exists() and cache_ilce.exists():
        il   = gpd.read_file(cache_il).to_crs(crs_target)
        ilce = gpd.read_file(cache_ilce).to_crs(crs_target)
        if len(il) > 0 and len(ilce) > 0:
            print(f"  Cache: {len(il)} provinces, {len(ilce)} districts")
            return il, ilce

    try:
        import osmnx as ox
        PROVINCES = [
            ("R223167",                 "Izmir",     True),
            ("Manisa Province, Turkey", "Manisa",    False),
            ("Balikesir, Turkey",       "Balikesir", False),
            ("R223471",                 "Canakkale",  True),
            ("Usak, Turkey",            "Usak",      False),
        ]
        parts = []
        for q, short, by_id in PROVINCES:
            try:
                g = ox.geocode_to_gdf(q, by_osmid=by_id)
                g["name"] = short
                parts.append(g[["name", "geometry"]])
            except Exception as e:
                print(f"    {short}: {e}")

        il_gdf = None
        if parts:
            il_gdf = (gpd.GeoDataFrame(_pd.concat(parts, ignore_index=True),
                                        crs="EPSG:4326").to_crs(crs_target))
            il_gdf[["name", "geometry"]].to_file(cache_il, driver="GPKG")

        MIN_AREA = 30_000_000
        ilce_gdf = None
        basin_gpkg = INTERIM / "kuzey_ege_havzasi_boundary.gpkg"
        if basin_gpkg.exists():
            basin_wgs = gpd.read_file(basin_gpkg).to_crs("EPSG:4326")
            tags = {"boundary": "administrative", "admin_level": "6"}
            g    = ox.features_from_polygon(basin_wgs.union_all(), tags=tags)
            polys   = g[g.geometry.geom_type.isin(["Polygon","MultiPolygon"])].reset_index(drop=True)
            polys_m = polys.to_crs("EPSG:32635")
            polys   = polys[polys_m.geometry.area >= MIN_AREA].reset_index(drop=True)
            if "name" not in polys.columns:
                polys["name"] = ""
            ilce_gdf = (gpd.GeoDataFrame(polys[["name","geometry"]], crs="EPSG:4326")
                        .to_crs(crs_target))
            ilce_gdf[["name","geometry"]].to_file(cache_ilce, driver="GPKG")
        return il_gdf, ilce_gdf

    except Exception as exc:
        print(f"  OSMnx unavailable ({exc})")
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
        print(f"  Boundary overlay skipped: {exc}")


def add_province_labels(ax, il_gdf, basin_geom):
    if il_gdf is None:
        return
    try:
        nc      = _name_col(il_gdf)
        clipped = il_gdf.clip(basin_geom)
        clipped = clipped[~clipped.geometry.is_empty]
        for _, row in clipped.iterrows():
            name = str(row.get(nc, "")).strip() if nc else ""
            if not name or name.lower() in ("none", "nan", ""):
                continue
            pt = row.geometry.representative_point()
            ax.text(pt.x, pt.y, name.upper(), fontsize=7.5,
                    ha="center", va="center", color=BOUNDARY_IL,
                    fontweight="bold", alpha=0.90, zorder=8,
                    path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])
    except Exception as exc:
        print(f"  Province labels skipped: {exc}")


# ─── SCALE BAR + NORTH ARROW (same as fig01 / fig05) ─────────────────────────

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
        [[cx,          na_base + na_h],
         [cx + na_w/2, na_base],
         [cx,          na_base + na_h * 0.25],
         [cx - na_w/2, na_base]],
        facecolor=BOUNDARY_IL, edgecolor="white", lw=1.2, zorder=6))
    ax.text(cx, na_base + na_h + H * 0.005, "N",
            ha="center", va="bottom", fontsize=10,
            fontweight="bold", color=BOUNDARY_IL, zorder=6)


# ─── RASTER LOADER ────────────────────────────────────────────────────────────

def load_raster(path, vmin_p=5, vmax_p=95, ordinal=False):
    if not path.exists():
        return None, None, None, None, None
    with rasterio.open(path) as src:
        arr    = src.read(1)
        nodata = src.nodata
        bounds = src.bounds
        crs    = src.crs
        if nodata is not None:
            mask = (arr == nodata) | np.isnan(arr)
            arr  = np.ma.masked_array(arr, mask=mask)
        else:
            arr  = np.ma.masked_array(arr, mask=np.isnan(arr))
        if ordinal:
            return arr, 1, 5, bounds, crs
        valid = arr.compressed()
        if len(valid) == 0:
            return arr, 0, 1, bounds, crs
        return arr, np.percentile(valid, vmin_p), np.percentile(valid, vmax_p), bounds, crs


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def create_fig05_advanced():
    print("Generating Advanced Fig 06: Multi-Hazard Convergence...")

    fig = plt.figure(figsize=(16, 9))

    col0_x, col1_x = 0.03, 0.245
    sm_w  = 0.205
    e_x, e_w = 0.480, 0.510
    top_y = 0.510
    bot_y = 0.040
    sm_h  = 0.440
    e_y   = 0.040
    e_h   = 0.910

    ax_a    = fig.add_axes([col0_x, top_y, sm_w, sm_h])
    ax_b    = fig.add_axes([col1_x, top_y, sm_w, sm_h])
    ax_c    = fig.add_axes([col0_x, bot_y, sm_w, sm_h])
    ax_d    = fig.add_axes([col1_x, bot_y, sm_w, sm_h])
    ax_main = fig.add_axes([e_x,    e_y,   e_w,  e_h ])

    # ── Shared data ───────────────────────────────────────────────────────────
    with rasterio.open(DRIVERS_DIR / "driver_elevation.tif") as src:
        elev        = src.read(1)
        elev_nodata = src.nodata
        elev_mask   = (elev == elev_nodata)
    ls               = LightSource(azdeg=315, altdeg=45)
    hillshade        = ls.hillshade(elev, vert_exag=3, dx=30, dy=30)
    hillshade_masked = np.ma.masked_where(elev_mask, hillshade)

    with rasterio.open(PROCESSED / "lulc_simplified_2000.tif") as src:
        lulc_base = src.read(1)
        nodata    = src.nodata
    valid_mask    = (lulc_base != nodata) if nodata is not None else (lulc_base > 0)
    boundary_mask = binary_fill_holes(valid_mask)

    # ── Sub-hazard panels (a–d) ───────────────────────────────────────────────
    hazards = [
        {"path": PROCESSED / "flood_hazard.tif",     "label": "(a)",
         "title": "Flood Hazard",       "unit": "Score (1–5)", "ax": ax_a, "ordinal": False},
        {"path": PROCESSED / "seismic_hazard.tif",   "label": "(b)",
         "title": "Seismic Hazard",     "unit": "Class (1–5)", "ax": ax_b, "ordinal": True},
        {"path": PROCESSED / "bioclimate_hazard.tif","label": "(c)",
         "title": "Bio-Climatic Stress","unit": "Index (0–1)", "ax": ax_c, "ordinal": False},
        {"path": PROCESSED / "wildfire_hazard.tif",  "label": "(d)",
         "title": "Wildfire Hazard",    "unit": "Index (0–1)", "ax": ax_d, "ordinal": False},
    ]

    for h in hazards:
        arr, vmin, vmax, bounds_h, _ = load_raster(h["path"], ordinal=h["ordinal"])
        ax = h["ax"]
        if arr is None:
            ax.text(0.5, 0.5, "Data Missing", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        x_r = bounds_h.right - bounds_h.left
        y_r = bounds_h.top   - bounds_h.bottom
        ext = [bounds_h.left, bounds_h.right, bounds_h.bottom, bounds_h.top]

        ax.set_xlim(bounds_h.left  - x_r * 0.02, bounds_h.right + x_r * 0.02)
        ax.set_ylim(bounds_h.bottom - y_r * 0.22, bounds_h.top   + y_r * 0.04)
        ax.imshow(hillshade_masked, cmap="gray", extent=ext, alpha=0.5)

        if h["ordinal"]:
            colors_5  = [elite_cmap(i / 4) for i in range(5)]
            cmap_disc = mcolors.ListedColormap(colors_5)
            norm_disc = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap_disc.N)
            im  = ax.imshow(np.ma.masked_where(arr < 1, arr), cmap=cmap_disc,
                            norm=norm_disc, extent=ext, alpha=0.8, interpolation="nearest")
            cax = inset_axes(ax, width="65%", height="5%", loc="lower center", borderpad=0.8)
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[1, 2, 3, 4, 5])
            cbar.ax.set_xticklabels(["1", "2", "3", "4", "5"])
        else:
            im   = ax.imshow(arr, cmap=elite_cmap, vmin=vmin, vmax=vmax,
                             extent=ext, alpha=0.8)
            cax  = inset_axes(ax, width="65%", height="5%", loc="lower center", borderpad=0.8)
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")

        ax.contour(boundary_mask, levels=[0.5], colors=BOUNDARY_IL,
                   linewidths=0.5, extent=ext, origin="upper", zorder=4)
        cbar.ax.tick_params(labelsize=8, length=2, pad=1)
        cbar.outline.set_visible(False)
        cbar.ax.text(1.0, 1.3, h["unit"], transform=cbar.ax.transAxes,
                     fontsize=7, ha="right", va="bottom", style="italic", color="#636e72")
        ax.text(0.0,  1.02, h["label"], transform=ax.transAxes,
                fontweight="bold", fontsize=18, va="bottom", ha="left")
        ax.text(0.14, 1.02, h["title"], transform=ax.transAxes,
                fontweight="bold", fontsize=11, va="bottom", ha="left", color=BOUNDARY_IL)
        ax.set_xticks([]); ax.set_yticks([])

    # ── Panel (e): Integrated Multi-Hazard Surface ────────────────────────────
    multi_path = PROCESSED / "multi_hazard_surface.tif"
    arr_multi, _, _, bounds, src_crs = load_raster(multi_path, ordinal=True)

    if arr_multi is not None:
        extent   = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        colors_5 = [elite_cmap(i / 4) for i in range(5)]
        cmap_multi = mcolors.ListedColormap(colors_5)
        norm       = mcolors.BoundaryNorm([0.5,1.5,2.5,3.5,4.5,5.5], cmap_multi.N)
        arr_multi  = np.ma.masked_where(arr_multi < 1, arr_multi)

        ax_main.imshow(hillshade_masked, cmap="gray", extent=extent, alpha=0.5)
        ax_main.imshow(arr_multi, cmap=cmap_multi, norm=norm,
                       extent=extent, alpha=0.85, interpolation="nearest")
        ax_main.contour(boundary_mask, levels=[0.5], colors=BOUNDARY_IL,
                        linewidths=1.0, extent=extent, origin="upper", zorder=4)

        # Admin boundaries + province labels
        import geopandas as gpd
        basin_gpkg = INTERIM / "kuzey_ege_havzasi_boundary.gpkg"
        if basin_gpkg.exists():
            basin_geom = gpd.read_file(basin_gpkg).to_crs(src_crs).union_all()
        else:
            from shapely.geometry import box as _bbox
            basin_geom = _bbox(bounds.left, bounds.bottom, bounds.right, bounds.top)

        il_gdf, ilce_gdf = load_admin_boundaries(src_crs)
        has_admin = (il_gdf is not None) or (ilce_gdf is not None)

        if has_admin:
            add_boundaries(ax_main, il_gdf, ilce_gdf, basin_geom)
        add_province_labels(ax_main, il_gdf, basin_geom)

        ax_main.text(0.0,  1.02, "(e)", transform=ax_main.transAxes,
                     fontweight="bold", fontsize=18, va="bottom", ha="left")
        ax_main.text(0.06, 1.02, "Integrated Multi-Hazard Surface",
                     transform=ax_main.transAxes,
                     fontweight="bold", fontsize=14, va="bottom", ha="left")

        # Legend + admin boundary entries
        labels_5    = ["1: Very Low", "2: Low", "3: Moderate", "4: High", "5: Very High"]
        leg_items   = [Patch(color=c, label=l) for c, l in zip(colors_5, labels_5)]
        if has_admin:
            leg_items += [
                Line2D([0],[0], color=BOUNDARY_IL,   lw=0.85, label="Province boundary"),
                Line2D([0],[0], color=BOUNDARY_ILCE, lw=0.40,
                       linestyle="--", label="District boundary"),
            ]
        ax_main.legend(handles=leg_items, loc="lower left", frameon=True,
                       facecolor="white", edgecolor="black",
                       fontsize=11, title="Multi-Hazard Class", title_fontsize=12)

        # Coordinate ticks
        _tr   = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        _ymid = bounds.bottom + (bounds.top   - bounds.bottom) / 2
        _xmid = bounds.left   + (bounds.right - bounds.left)   / 2

        def format_x(value, _):
            try:
                lon, _ = _tr.transform(value, _ymid)
                return f"{lon:.2f}°E"
            except: return ""

        def format_y(value, _):
            try:
                _, lat = _tr.transform(_xmid, value)
                return f"{lat:.2f}°N"
            except: return ""

        ax_main.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax_main.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax_main.xaxis.set_major_formatter(ticker.FuncFormatter(format_x))
        ax_main.yaxis.set_major_formatter(ticker.FuncFormatter(format_y))
        plt.setp(ax_main.get_yticklabels(), rotation=90, va="center")
        ax_main.spines["bottom"].set_visible(True)
        ax_main.spines["left"].set_visible(True)
        ax_main.spines["bottom"].set_linewidth(1.5)
        ax_main.spines["left"].set_linewidth(1.5)

        # Scale bar + north arrow
        add_scale_north(ax_main, bounds)

        # ── Province multi-hazard class distribution — radial donut chart ────
        if il_gdf is not None:
            try:
                from rasterio.features import geometry_mask as _geo_mask
                from shapely.geometry import mapping as _mapping

                nc       = _name_col(il_gdf)
                il_clip  = il_gdf.clip(basin_geom)
                il_clip  = il_clip[~il_clip.geometry.is_empty]

                with rasterio.open(multi_path) as _src:
                    ras_tr    = _src.transform
                    mh_raw    = _src.read(1).astype(float)
                    mh_nd     = _src.nodata
                mh_flat = np.where(
                    (mh_raw == mh_nd) | (mh_raw < 1) if mh_nd is not None
                    else (mh_raw < 1),
                    np.nan, mh_raw,
                )

                suit_rows = []
                for _, row in il_clip.iterrows():
                    try:
                        pmask = ~_geo_mask(
                            [_mapping(row.geometry)],
                            out_shape=mh_flat.shape,
                            transform=ras_tr,
                        )
                        pix = mh_flat[pmask]
                        pix = pix[~np.isnan(pix)]
                        if len(pix) >= 100:
                            name = str(row.get(nc, "")).strip() if nc else ""
                            # Classes 1–5, all included, renorm to 100%
                            raw  = [float(np.mean(pix == cls) * 100) for cls in range(1, 6)]
                            tot  = sum(raw) or 1.0
                            pcts = [v / tot * 100.0 for v in raw]
                            suit_rows.append({"name": name, "pcts": pcts,
                                              "mean": float(np.mean(pix))})
                    except Exception:
                        pass
                suit_rows.sort(key=lambda x: x["mean"])

                if suit_rows:
                    cls_labels = ["Very Low", "Low", "Moderate", "High", "Very High"]

                    # Horizontal inset — upper-right, %5 smaller, shifted up
                    _iw = e_w * 0.40 * 0.4 * 2.5 * 0.8 * 0.95
                    _ih = e_h * 0.37 * 0.4 * 2.5 * 0.8 * 0.95
                    ax_ins = fig.add_axes(
                        [e_x + e_w * 0.97 - _iw, e_y + e_h * 1.00 - _ih, _iw, _ih]
                    )

                    names = [d["name"] for d in suit_rows]
                    y_pos = np.arange(len(names))
                    left  = np.zeros(len(names))

                    for c_idx, (clr, lbl) in enumerate(zip(colors_5, cls_labels)):
                        vals = np.array([d["pcts"][c_idx] for d in suit_rows])
                        ax_ins.barh(y_pos, vals, left=left,
                                    color=clr, edgecolor="white",
                                    linewidth=0.4, height=0.82,
                                    alpha=0.93, zorder=3, label=lbl)
                        for yi, (v, l) in enumerate(zip(vals, left)):
                            if v >= 8:
                                ax_ins.text(l + v / 2, yi, f"{v:.0f}%",
                                            ha="center", va="center",
                                            fontsize=5.5, fontweight="bold",
                                            color=BOUNDARY_IL,
                                            path_effects=[pe.withStroke(
                                                linewidth=2.0, foreground="white")])
                        left += vals

                    ax_ins.set_yticks(y_pos)
                    ax_ins.set_yticklabels([n.upper() for n in names],
                                           fontsize=6.2, fontweight="bold",
                                           color=BOUNDARY_IL)
                    ax_ins.set_xlim(0, 100)
                    ax_ins.xaxis.set_major_locator(ticker.MultipleLocator(25))
                    ax_ins.xaxis.set_major_formatter(
                        ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
                    ax_ins.tick_params(axis="x", labelsize=6, length=2, pad=2,
                                       colors="#636e72")
                    ax_ins.tick_params(axis="y", length=0, pad=3)
                    ax_ins.grid(axis="x", linestyle=":", linewidth=0.4,
                                alpha=0.40, color="#bdc3c7")
                    ax_ins.set_facecolor("none")
                    ax_ins.patch.set_alpha(0.0)
                    ax_ins.set_title("Hazard Class by Province",
                                     fontsize=7.5, fontweight="bold",
                                     color=BOUNDARY_IL, pad=5, loc="left")
                    for sp in ax_ins.spines.values():
                        sp.set_visible(False)

            except Exception as exc:
                print(f"  Radial chart skipped: {exc}")

    else:
        ax_main.text(0.5, 0.5, "Multi-Hazard Surface Missing", ha="center", va="center")
        ax_main.set_xticks([]); ax_main.set_yticks([])

    out_path = FIG_DIR / "fig06_MultiHazard_Convergence.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    create_fig05_advanced()
