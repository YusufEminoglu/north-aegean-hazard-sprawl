import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, LightSource, ListedColormap
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pickle
import rasterio
from pathlib import Path
from scipy.ndimage import binary_fill_holes, gaussian_filter
from pyproj import Transformer

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.bottom": False,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in locals() else Path(".")
MODELS_DIR  = PROJECT_ROOT / "data" / "04_models"  / "paper2"
PROCESSED   = PROJECT_ROOT / "data" / "03_processed" / "paper2"
DRIVERS_DIR = PROCESSED / "drivers"
INTERIM_DIR = PROJECT_ROOT / "data" / "02_interim"  / "paper2"
FIG_DIR     = PROJECT_ROOT / "outputs" / "figures"   / "paper2"

BOUNDARY_IL   = "#2c3e50"
BOUNDARY_ILCE = "#636e72"

DRIVER_NAMES = [
    "slope", "elevation", "road_dist", "river_dist",
    "water_dist", "pop_density", "ntl", "ndvi",
    "clay", "coast_dist", "urban_dist",
]
PRETTY_NAMES = {
    "slope": "Slope", "elevation": "Elevation", "road_dist": "Dist. to Road",
    "river_dist": "Dist. to River", "water_dist": "Dist. to Water",
    "pop_density": "Pop Density", "ntl": "Night Lights", "ndvi": "NDVI",
    "clay": "Clay Content", "coast_dist": "Dist. to Coast",
    "urban_dist": "Dist. to Urban", "canopy": "Canopy",
}


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

    cache_il   = INTERIM_DIR / "admin_il.gpkg"
    cache_ilce = INTERIM_DIR / "admin_ilce.gpkg"

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

        print("  Fetching province boundaries via OSMnx ...")
        parts = []
        for q, short, by_id in PROVINCES:
            try:
                g = ox.geocode_to_gdf(q, by_osmid=by_id)
                g["name"] = short
                parts.append(g[["name", "geometry"]])
                print(f"    {short}: OK")
            except Exception as e:
                print(f"    {short}: {e}")

        il_gdf = None
        if parts:
            il_gdf = (gpd.GeoDataFrame(_pd.concat(parts, ignore_index=True), crs="EPSG:4326")
                      .to_crs(crs_target))
            il_gdf[["name", "geometry"]].to_file(cache_il, driver="GPKG")

        MIN_AREA = 30_000_000
        ilce_gdf = None
        basin_gpkg = INTERIM_DIR / "kuzey_ege_havzasi_boundary.gpkg"
        if basin_gpkg.exists():
            basin_wgs  = gpd.read_file(basin_gpkg).to_crs("EPSG:4326")
            basin_poly = basin_wgs.union_all()
            tags = {"boundary": "administrative", "admin_level": "6"}
            g    = ox.features_from_polygon(basin_poly, tags=tags)
            polys   = g[g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
            polys_m = polys.to_crs("EPSG:32635")
            polys   = polys[polys_m.geometry.area >= MIN_AREA].reset_index(drop=True)
            if "name" not in polys.columns:
                polys["name"] = ""
            ilce_gdf = (gpd.GeoDataFrame(polys[["name", "geometry"]], crs="EPSG:4326")
                        .to_crs(crs_target))
            ilce_gdf[["name", "geometry"]].to_file(cache_ilce, driver="GPKG")

        return il_gdf, ilce_gdf

    except Exception as exc:
        print(f"  OSMnx unavailable ({exc})")
        return None, None


def add_boundaries(ax, il_gdf, ilce_gdf, basin_geom):
    try:
        if ilce_gdf is not None:
            clipped = ilce_gdf.clip(basin_geom)
            clipped = clipped[~clipped.is_empty & clipped.is_valid]
            if len(clipped):
                clipped.boundary.plot(ax=ax, color=BOUNDARY_ILCE, linewidth=0.35,
                                      linestyle="--", zorder=5, alpha=0.65)
        if il_gdf is not None:
            clipped = il_gdf.clip(basin_geom)
            clipped = clipped[~clipped.is_empty & clipped.is_valid]
            if len(clipped):
                clipped.boundary.plot(ax=ax, color=BOUNDARY_IL, linewidth=0.75,
                                      zorder=5, alpha=0.80)
    except Exception as exc:
        print(f"  Boundary overlay skipped: {exc}")


def add_province_labels(ax, il_gdf, basin_geom):
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
                    color=BOUNDARY_IL, fontweight="bold", alpha=0.90, zorder=8,
                    path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])
    except Exception as exc:
        print(f"  Province labels skipped: {exc}")


# ─── SCALE BAR + NORTH ARROW (same positioning as fig01 / fig05) ─────────────

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
        [[cx,           na_base + na_h],
         [cx + na_w/2,  na_base],
         [cx,           na_base + na_h * 0.25],
         [cx - na_w/2,  na_base]],
        facecolor=BOUNDARY_IL, edgecolor="white", lw=1.2, zorder=6))
    ax.text(cx, na_base + na_h + H * 0.005, "N",
            ha="center", va="bottom", fontsize=10,
            fontweight="bold", color=BOUNDARY_IL, zorder=6)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def create_fig04_advanced():
    print("Generating Advanced Fig 04 (RF Suitability - Bar, Donut & Map)...")

    # 1. Load model & feature importance
    model_path = MODELS_DIR / "rf_flus_model.pkl"
    with open(model_path, "rb") as f:
        rf = pickle.load(f)

    df = pd.DataFrame({
        "Driver":     [PRETTY_NAMES[n] for n in DRIVER_NAMES],
        "Importance": rf.feature_importances_ * 100,
    }).sort_values(by="Importance", ascending=True).reset_index(drop=True)

    # 2. Predict suitability map
    print("  Loading drivers for Suitability Map...")
    stack, meta = [], None
    for name in DRIVER_NAMES:
        with rasterio.open(DRIVERS_DIR / f"driver_{name}.tif") as src:
            if meta is None:
                meta    = src.meta.copy()
                bounds  = src.bounds
                src_crs = src.crs
            arr = src.read(1)
            stack.append(np.nan_to_num(arr, nan=0.0))

    drivers = np.stack(stack, axis=-1)

    with rasterio.open(PROCESSED / "lulc_simplified_2000.tif") as src:
        lulc_base = src.read(1)
        nodata    = src.nodata

    valid_mask    = (lulc_base != nodata) if nodata is not None else (lulc_base > 0)
    boundary_mask = binary_fill_holes(valid_mask)

    valid_indices = np.where(valid_mask)
    X_valid       = drivers[valid_indices]

    print("  Predicting suitability probabilities...")
    probs      = np.zeros(len(X_valid), dtype=np.float32)
    chunk_size = 500000
    for i in range(0, len(X_valid), chunk_size):
        probs[i:i+chunk_size] = rf.predict_proba(X_valid[i:i+chunk_size])[:, 1]

    prob_surface              = np.zeros(lulc_base.shape, dtype=np.float32)
    prob_surface[valid_indices] = probs
    prob_surface              = gaussian_filter(prob_surface, sigma=2.0)
    prob_surface              = np.ma.masked_where(~valid_mask, prob_surface)

    # 3. Layout
    fig = plt.figure(figsize=(15, 8))
    gs  = fig.add_gridspec(1, 2, width_ratios=[0.85, 1], wspace=0.15)

    # ── Panel A: Feature importance ───────────────────────────────────────────
    ax_bar    = fig.add_subplot(gs[0])
    elite_cmap = LinearSegmentedColormap.from_list("elite", ["#dfe6e9", "#00cec9", "#ff7675"])
    colors    = [elite_cmap(i / (len(df) - 1)) for i in range(len(df))]

    bars = ax_bar.barh(df["Driver"], df["Importance"], color=colors,
                       height=0.6, edgecolor="none")
    ax_bar.set_xlabel("Gini Importance (%)", fontweight="bold", fontsize=12, labelpad=4)
    ax_bar.tick_params(axis="y", length=0, labelsize=11)
    ax_bar.tick_params(axis="x", labelsize=11)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.3, color="#bdc3c7")
    ax_bar.spines["bottom"].set_visible(True)
    ax_bar.spines["bottom"].set_linewidth(1.5)

    for bar in bars:
        w = bar.get_width()
        ax_bar.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}%", ha="left", va="center",
                    fontweight="bold", fontsize=10, color="#2c3e50")

    ax_bar.set_xlim(0, max(df["Importance"]) * 1.05)

    # Inset donut chart
    # [x0, y0, w, h] in axes fraction — shifted ~1/4 to the left vs lower-right default
    ax_donut   = ax_bar.inset_axes([0.22, 0.01, 0.60, 0.58])
    df_donut   = df[df["Driver"] != "Dist. to Urban"].copy()
    total      = df_donut["Importance"].sum()
    df_donut["Donut_Pct"] = df_donut["Importance"] / total * 100
    df_donut   = df_donut.sort_values(by="Importance", ascending=False)
    color_map  = {row["Driver"]: colors[idx] for idx, row in df.iterrows()}
    colors_d   = [color_map[d] for d in df_donut["Driver"]]

    wedges, _ = ax_donut.pie(df_donut["Donut_Pct"], colors=colors_d,
                              startangle=90, wedgeprops=dict(width=0.4, edgecolor="white"))
    for i, p in enumerate(wedges):
        pct  = df_donut.iloc[i]["Donut_Pct"]
        name = df_donut.iloc[i]["Driver"]
        if pct >= 10.0:
            ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
            r   = 0.8
            ax_donut.text(np.cos(np.deg2rad(ang)) * r,
                          np.sin(np.deg2rad(ang)) * r,
                          f"{name}\n{pct:.1f}%",
                          ha="center", va="center", fontsize=10,
                          fontweight="bold", color="#2c3e50",
                          path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    ax_donut.text(0, 0, "Excl.\nDist. to Urban",
                  ha="center", va="center", fontsize=9,
                  fontweight="bold", color="#7f8c8d")

    # ── Panel B: Suitability map ──────────────────────────────────────────────
    ax_map = fig.add_subplot(gs[1])

    with rasterio.open(DRIVERS_DIR / "driver_elevation.tif") as src:
        elev = src.read(1)
        elev[~valid_mask] = 0

    ls        = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(elev, vert_exag=3, dx=30, dy=30)
    extent    = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    shift_px    = 15
    shadow_mask = np.zeros_like(boundary_mask, dtype=bool)
    shadow_mask[shift_px:, shift_px:] = boundary_mask[:-shift_px, :-shift_px]
    shadow_arr  = np.zeros_like(prob_surface, dtype=float)
    shadow_arr[~shadow_mask] = np.nan
    ax_map.imshow(shadow_arr, cmap="Greys", vmin=0, vmax=1,
                  alpha=0.4, extent=extent, zorder=1)
    ax_map.imshow(np.ma.masked_where(~valid_mask, hillshade),
                  cmap="gray", extent=extent, zorder=2, alpha=0.6)
    ax_map.contour(boundary_mask, levels=[0.5], colors=BOUNDARY_IL,
                   linewidths=0.8, extent=extent, origin="upper", zorder=4)

    bins       = [0, 0.166, 0.333, 0.5, 0.666, 0.833, 1.0]
    prob_class = np.digitize(prob_surface, bins) - 1
    prob_class = np.clip(prob_class, 0, 5)
    prob_class = np.ma.masked_where(~valid_mask, prob_class)
    colors_6   = [elite_cmap(i / 5) for i in range(6)]
    cmap_6     = ListedColormap(colors_6)
    ax_map.imshow(prob_class, cmap=cmap_6, vmin=0, vmax=5,
                  extent=extent, zorder=3, alpha=0.85, interpolation="nearest")

    # Admin boundaries + province labels
    import geopandas as gpd
    basin_gpkg = INTERIM_DIR / "kuzey_ege_havzasi_boundary.gpkg"
    if basin_gpkg.exists():
        basin_geom = gpd.read_file(basin_gpkg).to_crs(src_crs).union_all()
    else:
        from shapely.geometry import box as _bbox
        basin_geom = _bbox(bounds.left, bounds.bottom, bounds.right, bounds.top)

    il_gdf, ilce_gdf = load_admin_boundaries(src_crs)
    has_admin = (il_gdf is not None) or (ilce_gdf is not None)

    if has_admin:
        add_boundaries(ax_map, il_gdf, ilce_gdf, basin_geom)
    add_province_labels(ax_map, il_gdf, basin_geom)

    # Coordinate tick labels
    _to_wgs84 = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    _y_mid    = bounds.bottom + (bounds.top  - bounds.bottom) / 2
    _x_mid    = bounds.left   + (bounds.right - bounds.left)  / 2

    def format_x(value, _):
        try:
            lon, _ = _to_wgs84.transform(value, _y_mid)
            return f"{lon:.2f}°E"
        except:
            return ""

    def format_y(value, _):
        try:
            _, lat = _to_wgs84.transform(_x_mid, value)
            return f"{lat:.2f}°N"
        except:
            return ""

    ax_map.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax_map.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax_map.xaxis.set_major_formatter(ticker.FuncFormatter(format_x))
    ax_map.yaxis.set_major_formatter(ticker.FuncFormatter(format_y))
    plt.setp(ax_map.get_yticklabels(), rotation=90, va="center")

    ax_map.spines["top"].set_visible(False)
    ax_map.spines["right"].set_visible(False)
    ax_map.spines["bottom"].set_visible(True)
    ax_map.spines["left"].set_visible(True)
    ax_map.spines["bottom"].set_linewidth(1.5)
    ax_map.spines["left"].set_linewidth(1.5)

    # Scale bar + north arrow — same positioning as fig01 / fig05
    add_scale_north(ax_map, bounds)

    # Legend: suitability classes + admin boundaries
    suit_labels  = ["Very Low (<0.16)", "Low (0.16–0.33)", "Moderate (0.33–0.50)",
                    "High (0.50–0.66)", "Very High (0.66–0.83)", "Extreme (>0.83)"]
    legend_items = [Patch(color=c, label=l) for c, l in zip(colors_6, suit_labels)]
    if has_admin:
        legend_items += [
            Line2D([0], [0], color=BOUNDARY_IL,   lw=0.85, label="Province boundary"),
            Line2D([0], [0], color=BOUNDARY_ILCE, lw=0.40,
                   linestyle="--", label="District boundary"),
        ]
    ax_map.legend(handles=legend_items, loc="lower left", frameon=True,
                  facecolor="white", edgecolor="black",
                  fontsize=9, title="Suitability Class", title_fontsize=10)

    # Align panel heights
    plt.subplots_adjust(bottom=0.05)
    fig.canvas.draw()
    pos_map = ax_map.get_position()
    pos_bar = ax_bar.get_position()
    ax_bar.set_position([pos_bar.x0, pos_map.y0, pos_bar.width, pos_map.height])

    ax_bar.text(0.0, 1.02, "(a)", transform=ax_bar.transAxes,
                fontweight="bold", fontsize=18, va="bottom")
    ax_map.text(0.0, 1.02, "(b)", transform=ax_map.transAxes,
                fontweight="bold", fontsize=18, va="bottom")

    out_path = FIG_DIR / "fig04_Suitability.png"
    plt.savefig(out_path, dpi=300, facecolor="white",
                bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {out_path.name}")


if __name__ == "__main__":
    create_fig04_advanced()
