"""
Paper 2 — Figure 7: Hazard Model Sensitivity Analysis
=======================================================
Panel (a): Spearman rank correlation matrix between weight schemes
Panel (b): Spatial agreement map (fig06-panel-e style) + weight schemes radar inset
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, LightSource
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import spearmanr, gaussian_kde
from scipy.ndimage import binary_fill_holes
from scipy.interpolate import interp1d
from pyproj import Transformer
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import itertools
import jenkspy

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "axes.titlesize": 11, "axes.labelsize": 10, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED    = PROJECT_ROOT / "data" / "03_processed" / "paper2"
INTERIM      = PROJECT_ROOT / "data" / "02_interim"  / "paper2"
DRIVERS_DIR  = PROCESSED / "drivers"
FIG_DIR      = PROJECT_ROOT / "outputs" / "figures"  / "paper2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REF_PATH = PROCESSED / "lulc_simplified_2020.tif"

elite_cmap = LinearSegmentedColormap.from_list("elite", ["#dfe6e9", "#00cec9", "#ff7675"])

BOUNDARY_IL   = "#2c3e50"
BOUNDARY_ILCE = "#636e72"

HAZARD_KEYS   = ["flood", "seismic", "bioclimate", "wildfire"]
HAZARD_LABELS = ["Flood", "Seismic", "Bio-Climate", "Wildfire"]

WEIGHT_SCHEMES = {
    "W1 AHP\n(primary)":    [0.35, 0.35, 0.20, 0.10],
    "W2 Equal\nweight":     [0.25, 0.25, 0.25, 0.25],
    "W3 Rank\nbased":       [0.39, 0.39, 0.13, 0.10],
    "W4 PCA\nderived":      [0.07, 0.81, 0.10, 0.02],
}
SCHEME_COLORS = [elite_cmap(v) for v in [0.9, 0.6, 0.3, 0.05]]


# ─── ADMIN BOUNDARIES (same logic as fig01 / fig05 / fig06) ──────────────────

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


# ─── COMPUTATION HELPERS ──────────────────────────────────────────────────────

def load_aligned(path, ref_meta):
    if not path.exists():
        print(f"  WARNING: {path.name} not found")
        return None
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd  = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        # Same grid → return directly, no reproject needed
        if (src.height == ref_meta["height"] and
                src.width  == ref_meta["width"]  and
                src.crs    == ref_meta["crs"]):
            return arr
        # Different grid → reproject
        dst = np.full((ref_meta["height"], ref_meta["width"]), np.nan, dtype=np.float32)
        reproject(
            source=arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_meta["transform"],
            dst_crs=ref_meta["crs"],
            resampling=Resampling.bilinear,
        )
        return dst


def normalise(arr, mask):
    valid = mask & ~np.isnan(arr)
    v = arr[valid]
    if len(v) == 0 or v.max() == v.min():
        return np.zeros_like(arr, np.float32)
    out = np.zeros_like(arr, np.float32)
    out[valid] = (arr[valid] - v.min()) / (v.max() - v.min())
    return out


def jenks_ordinal(arr, mask):
    values = arr[mask & np.isfinite(arr)]
    breaks = jenkspy.jenks_breaks(values, n_classes=5)
    ordinal = np.zeros_like(arr, dtype=np.uint8)
    ordinal[mask] = np.digitize(arr[mask], breaks[1:-1], right=True) + 1
    return ordinal

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def create_fig07():
    print("Generating Fig 07: Sensitivity Analysis...")

    # Load reference raster
    with rasterio.open(REF_PATH) as ref:
        ref_meta = ref.meta.copy()
        ref_arr  = ref.read(1)
        nodata   = ref.nodata
    valid = (ref_arr != nodata) if nodata is not None else np.ones(ref_arr.shape, bool)

    # Compute weighted hazard surfaces
    normed = {}
    for key in HAZARD_KEYS:
        arr = load_aligned(PROCESSED / f"{key}_hazard.tif", ref_meta)
        normed[key] = normalise(arr, valid) if arr is not None else np.zeros(ref_arr.shape, np.float32)

    surfaces, ordinals = {}, {}
    for name, weights in WEIGHT_SCHEMES.items():
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()  # absorb published two-decimal rounding
        s = sum(normed[k] * w for k, w in zip(HAZARD_KEYS, weights)).astype(np.float32)
        surfaces[name] = s
        ordinals[name] = jenks_ordinal(s, valid)

    # Spearman correlation matrix
    scheme_names = list(WEIGHT_SCHEMES.keys())
    n    = len(scheme_names)
    flat = np.column_stack([surfaces[nm][valid].ravel() for nm in scheme_names])
    corr_mat = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        xi, xj = flat[:, i], flat[:, j]
        ok = np.isfinite(xi) & np.isfinite(xj)
        r, _ = spearmanr(xi[ok], xj[ok])
        corr_mat[i, j] = corr_mat[j, i] = float(r)

    # Spatial agreement fraction
    stack    = np.stack([ordinals[nm] for nm in scheme_names], axis=0)
    from scipy.stats import mode as sp_mode
    mode_arr = sp_mode(stack, axis=0, keepdims=False).mode.squeeze()
    agree_frac = np.sum(stack == mode_arr[np.newaxis], axis=0) / n
    agree_frac[~valid] = np.nan

    # Background layers
    with rasterio.open(DRIVERS_DIR / "driver_elevation.tif") as src:
        elev   = src.read(1)
        nd     = src.nodata
        emask  = (elev == nd)
        bounds = src.bounds
        crs    = src.crs
    ls        = LightSource(azdeg=315, altdeg=45)
    hillshade = np.ma.masked_where(
        emask,
        ls.hillshade(np.where(emask, 0, elev).astype(float), vert_exag=3, dx=30, dy=30)
    )
    with rasterio.open(PROCESSED / "lulc_simplified_2000.tif") as src:
        lulc = src.read(1); nd2 = src.nodata
    valid_lulc = (lulc != nd2) if nd2 is not None else (lulc > 0)
    boundary   = binary_fill_holes(valid_lulc)

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # ── Layout: 2 panels ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.18)
    ax_radar = fig.add_subplot(gs[0], polar=True)
    ax_map   = fig.add_subplot(gs[1])

    # ── (a) Weight Schemes — full radar panel ─────────────────────────────────
    N_h    = len(HAZARD_LABELS)
    angles_base = [i / N_h * 2 * np.pi for i in range(N_h)]
    angles = angles_base + [angles_base[0]]

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles_base)
    ax_radar.set_xticklabels(HAZARD_LABELS, fontsize=10,
                              color=BOUNDARY_IL, fontweight="bold")
    ax_radar.set_ylim(0, 0.5)
    ax_radar.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax_radar.set_yticklabels(["0.1", "0.2", "0.3", "0.4", "0.5"],
                              fontsize=7.5, color="#95a5a6")
    ax_radar.grid(color="#b2bec3", linestyle="--", linewidth=0.6, alpha=0.55)
    ax_radar.spines["polar"].set_visible(False)

    for (name, weights), color in zip(WEIGHT_SCHEMES.items(), SCHEME_COLORS):
        a_wrap   = angles_base + [angles_base[0] + 2 * np.pi]
        v_wrap   = list(weights) + [weights[0]]
        f_spl    = interp1d(a_wrap, v_wrap, kind="cubic")
        a_smooth = np.linspace(0, 2 * np.pi, 300)
        v_smooth = np.clip(f_spl(a_smooth), 0, None)
        ax_radar.plot(a_smooth, v_smooth, color=color, linewidth=2.2,
                      label=name.replace("\n", " "))
        ax_radar.fill(a_smooth, v_smooth, color=color, alpha=0.12)
        # Actual weight vertices
        ax_radar.scatter(angles_base, weights, s=36, color=color,
                         zorder=5, edgecolors="white", linewidths=0.8)

    ax_radar.set_facecolor("white")
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.30, 1.02),
                    fontsize=9, frameon=True, facecolor="white",
                    edgecolor=BOUNDARY_ILCE, framealpha=0.90)
    ax_radar.set_title("(a)", loc="left", fontweight="bold", fontsize=18,
                        pad=20, color=BOUNDARY_IL)
    # subtitle via figure text positioned just above ax
    fig.text(0.0, 1.0, "Weight Schemes", fontweight="bold",
             fontsize=11, color=BOUNDARY_IL,
             transform=ax_radar.transAxes, va="bottom", ha="left")

    # ── (b) Spatial Agreement Map — fig06-panel-e style ──────────────────────
    shift = 15
    sm    = np.zeros_like(boundary, bool)
    sm[shift:, shift:] = boundary[:-shift, :-shift]
    sa    = np.zeros(boundary.shape, float); sa[~sm] = np.nan
    ax_map.imshow(sa, cmap="Greys", vmin=0, vmax=1,
                  alpha=0.4, extent=extent, zorder=1)
    ax_map.imshow(hillshade, cmap="gray", extent=extent, zorder=2, alpha=0.6)

    # Discrete 4-class colormap: 1/4, 2/4, 3/4, 4/4 agreement
    agree_colors = [elite_cmap(v) for v in [0.05, 0.38, 0.68, 0.95]]
    cmap_disc    = mcolors.ListedColormap(agree_colors)
    # Midpoints between actual agree_frac values (0.25, 0.5, 0.75, 1.0)
    bnorm        = mcolors.BoundaryNorm([0.0, 0.375, 0.625, 0.875, 1.01], cmap_disc.N)

    im2 = ax_map.imshow(agree_frac, cmap=cmap_disc, norm=bnorm,
                        extent=extent, origin="upper",
                        interpolation="nearest", zorder=3, alpha=0.85)
    ax_map.contour(boundary, levels=[0.5], colors=BOUNDARY_IL,
                   linewidths=0.8, extent=extent, origin="upper", zorder=4)

    # Discrete agreement legend (lower centre area, above full-agreement text)
    agree_labels = ["1/4 schemes agree", "2/4 schemes agree",
                    "3/4 schemes agree", "4/4 schemes agree"]
    agree_patches = [Patch(facecolor=c, label=l, edgecolor="white", linewidth=0.4)
                     for c, l in zip(agree_colors, agree_labels)]

    # Admin boundaries + province labels
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
        add_boundaries(ax_map, il_gdf, ilce_gdf, basin_geom)
    add_province_labels(ax_map, il_gdf, basin_geom)

    # Coordinate ticks
    _tr   = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    _ymid = bounds.bottom + (bounds.top   - bounds.bottom) / 2
    _xmid = bounds.left   + (bounds.right - bounds.left)   / 2

    def fmt_x(v, _):
        try:
            lon, _ = _tr.transform(v, _ymid); return f"{lon:.2f}°E"
        except: return ""

    def fmt_y(v, _):
        try:
            _, lat = _tr.transform(_xmid, v); return f"{lat:.2f}°N"
        except: return ""

    ax_map.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax_map.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax_map.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_x))
    ax_map.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_y))
    plt.setp(ax_map.get_yticklabels(), rotation=90, va="center")
    ax_map.spines["bottom"].set_visible(True)
    ax_map.spines["left"].set_visible(True)
    ax_map.spines["bottom"].set_linewidth(1.5)
    ax_map.spines["left"].set_linewidth(1.5)

    # Scale bar + north arrow
    add_scale_north(ax_map, bounds)

    # Full-agreement annotation
    all_agree = float((agree_frac >= 1.0).sum()) / float(valid.sum()) * 100
    ax_map.text(0.02, 0.02, f"Full agreement: {all_agree:.1f}% of basin",
                transform=ax_map.transAxes, fontsize=8.5, va="bottom",
                bbox=dict(facecolor="white", edgecolor=BOUNDARY_ILCE,
                          alpha=0.88, pad=3, boxstyle="round,pad=0.4"))

    # Combined legend: agreement classes + admin boundaries
    leg_items = list(agree_patches)
    if has_admin:
        leg_items += [
            Line2D([0],[0], color=BOUNDARY_IL,   lw=0.85, label="Province boundary"),
            Line2D([0],[0], color=BOUNDARY_ILCE, lw=0.40,
                   linestyle="--", label="District boundary"),
        ]
    ax_map.legend(handles=leg_items, loc="lower left",
                  bbox_to_anchor=(0.0, 0.09),
                  frameon=True, facecolor="white",
                  edgecolor=BOUNDARY_ILCE, fontsize=8.5,
                  title="Agreement", title_fontsize=9)

    ax_map.set_title("(b)", loc="left", fontweight="bold", fontsize=18, pad=8)

    # ── Rank correlation scatter matrix — inset upper-right of map panel ──────
    fig.canvas.draw()
    pos_map = ax_map.get_position()

    _cw    = pos_map.width  * 0.33
    _ch    = pos_map.height * 0.36
    ins_x0 = pos_map.x0 + pos_map.width  - _cw
    ins_y0 = pos_map.y0 + pos_map.height - _ch
    cell_w = _cw / n
    cell_h = _ch / n

    # W3 color = lowest corr end, W4 color = highest corr end
    corr_cmap_custom = LinearSegmentedColormap.from_list(
        "corr", [SCHEME_COLORS[2], SCHEME_COLORS[3]])
    off_vals = [corr_mat[i, j] for i in range(n) for j in range(n) if i != j]
    c_vmin, c_vmax = min(off_vals), 1.0

    short_labels = ["W1", "W2", "W3", "W4"]
    rng = np.random.default_rng(42)

    for i in range(n):
        for j in range(n):
            cx    = ins_x0 + j * cell_w
            cy    = ins_y0 + (n - 1 - i) * cell_h
            ax_c  = fig.add_axes([cx, cy, cell_w, cell_h])
            ax_c.set_xticks([]); ax_c.set_yticks([])
            for sp in ax_c.spines.values():
                sp.set_linewidth(0.25); sp.set_edgecolor("#bdc3c7")

            if i == j:                         # ── diagonal: KDE ──
                nm   = scheme_names[i]
                vals = surfaces[nm][valid].ravel()
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    sub = rng.choice(vals, size=min(20000, len(vals)), replace=False)
                    kde = gaussian_kde(sub)
                    xr  = np.linspace(sub.min(), sub.max(), 100)
                    yr  = kde(xr)
                    ax_c.fill_between(xr, yr, color=SCHEME_COLORS[i], alpha=0.50)
                    ax_c.plot(xr, yr, color=SCHEME_COLORS[i], lw=0.9)
                ax_c.set_facecolor("none")
                ax_c.text(0.5, 0.88, short_labels[i],
                          transform=ax_c.transAxes, ha="center", va="top",
                          fontsize=6.5, fontweight="bold", color=BOUNDARY_IL)

            elif j > i:                        # ── upper triangle: bubble ──
                r      = corr_mat[i, j]
                norm_r = (r - c_vmin) / (c_vmax - c_vmin + 1e-9)
                color  = corr_cmap_custom(norm_r)
                ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1)
                ax_c.scatter(0.5, 0.5, s=norm_r * 600 + 80, c=[color],
                             zorder=3, edgecolors="white", linewidths=0.4)
                ax_c.text(0.5, 0.5, f"{r:.2f}",
                          transform=ax_c.transAxes, ha="center", va="center",
                          fontsize=5.5, fontweight="bold",
                          color="white" if norm_r > 0.5 else BOUNDARY_IL)
                ax_c.set_facecolor("none")

            else:                              # ── lower triangle: text ──
                ax_c.text(0.5, 0.5, f"ρ={corr_mat[i,j]:.3f}",
                          transform=ax_c.transAxes, ha="center", va="center",
                          fontsize=5.5, color=BOUNDARY_IL)
                ax_c.set_facecolor("none")

    # Title above grid
    fig.text(ins_x0 + _cw / 2, ins_y0 + _ch + 0.004,
             "Spearman ρ — Rank Correlation",
             fontsize=6.8, fontweight="bold", color=BOUNDARY_IL,
             ha="center", va="bottom")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = FIG_DIR / "fig07_Sensitivity.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    create_fig07()
