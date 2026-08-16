"""Supplementary Figure S3: Study area location map.

(a) Top, full width: location of the North Aegean Basin within Turkiye
    (national context, districts shown as line work, no fill, basin
    highlighted, major reference cities for orientation, CRS inset,
    scale bar + north arrow).
(b) Bottom-left: Esri World Imagery satellite basemap of the basin, with
    the 2020 urban footprint and basin/province/district boundaries.
(c) Bottom-right: basin elevation (hillshaded DEM) with a MERIT-Hydro
    upstream-drainage-area-derived river network and the DSI Kuzey Ege
    Havzasi Master Plan's official flood-prone-area polygons overlaid
    (source attributed in the caption only, not labelled on the map).

Administrative boundaries source: ArcGIS Online "Turkiye Ilce Sinirlari"
(Districts of Turkey) feature service, item 6ed6703aa1db41c7879f3f9689ae2a65,
970 features with il_adi/ilce_adi fields, cached locally as GeoJSON. The
national outline is dissolved from these 970 districts; dissolve-induced
interior holes are removed per-polygon (exterior ring only) without
touching separate exterior polygons, so offshore islands are preserved.

Panel (a) is rendered and cropped independently of panels (b)/(c) and the
two blocks are composited with PIL at a fixed, small gap, because
matplotlib's fixed-aspect axes (needed to keep Turkiye's outline
undistorted) leaves an unpredictable, aspect-driven blank gap when placed
in a GridSpec row sized by height_ratios alone.

Usage:
  python src/p2_fig13_location_map.py
"""

from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import rasterio
import requests
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from pyproj import Transformer
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("p2_fig10", HERE / "p2_fig10_policy.py")
fig10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig10)

PROJECT_ROOT = fig10.PROJECT_ROOT
INTERIM      = fig10.INTERIM
PROCESSED    = fig10.PROCESSED
DRIVERS_DIR  = PROCESSED / "drivers"
FIG_DIR      = PROJECT_ROOT / "outputs" / "figures" / "paper2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY_IL   = fig10.BOUNDARY_IL
BOUNDARY_ILCE = fig10.BOUNDARY_ILCE
BASIN_COLOR   = "#ff7675"
URBAN_COLOR   = "#ffeaa7"
RIVER_COLOR   = "#0984e3"
FLOOD_COLOR   = "#6c5ce7"
GRID_COLOR    = "#7f8c8d"

ILCE_SERVICE_URL = ("https://services5.arcgis.com/26ObnytmBZ58Wbj8/arcgis/rest/"
                     "services/turkiye_ilce_sinirlari/FeatureServer/0/query")
ILCE_CACHE = INTERIM / "turkiye_ilce_sinirlari_arcgis.geojson"
RIVER_UPSTREAM_KM2 = 30

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "axes.titlesize": 12, "figure.dpi": 300,
})

CITIES = {
    "Istanbul": (28.9784, 41.0082),
    "Ankara":   (32.8597, 39.9334),
    "Izmir":    (27.1428, 38.4237),
}


def load_turkiye_ilce():
    if ILCE_CACHE.exists():
        return gpd.read_file(ILCE_CACHE, use_arrow=False)
    print("  Downloading Turkiye ilce sinirlari (ArcGIS Online)...")
    params = {"where": "1=1", "outFields": "*", "f": "geojson",
              "resultRecordCount": 2000, "outSR": 4326}
    r = requests.get(ILCE_SERVICE_URL, params=params, timeout=60)
    r.raise_for_status()
    ILCE_CACHE.write_bytes(r.content)
    return gpd.read_file(ILCE_CACHE, use_arrow=False)


def remove_holes(geom):
    if geom.geom_type == "Polygon":
        return ShapelyPolygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([ShapelyPolygon(p.exterior) for p in geom.geoms])
    return geom


def add_lonlat_grid(ax, crs, is_geographic, step=None):
    if is_geographic:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(step or 2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(step or 2))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}°E"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    else:
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xmid, ymid = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2

        def fmt_x(v, _):
            lon, _lat = tr.transform(v, ymid)
            return f"{lon:.2f}°E"

        def fmt_y(v, _):
            _lon, lat = tr.transform(xmid, v)
            return f"{lat:.2f}°N"

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_x))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_y))

    ax.tick_params(labelsize=7, color=GRID_COLOR, length=3)
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.4, alpha=0.45, zorder=2)


def add_scale_north_geographic(ax, lat_mid):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    W, H = xlim[1] - xlim[0], ylim[1] - ylim[0]
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat_mid))
    slen_km  = 200
    slen_deg = slen_km / km_per_deg_lon
    cx    = xlim[1] - W * 0.10
    sb_x0 = cx - slen_deg / 2
    sb_y0 = ylim[0] + H * 0.06
    sb_h  = H * 0.02
    for i in range(2):
        ax.add_patch(Rectangle((sb_x0 + i * slen_deg / 2, sb_y0), slen_deg / 2, sb_h,
                                facecolor="black" if i == 0 else "white",
                                edgecolor="black", zorder=9))
        ax.text(sb_x0 + i * slen_deg / 2, sb_y0 + sb_h * 1.7, f"{int(i * slen_km / 2)}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", zorder=9)
    ax.text(sb_x0 + slen_deg, sb_y0 + sb_h * 1.7, f"{slen_km} km",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold", zorder=9)
    na_base = sb_y0 + sb_h * 3.6
    na_w, na_h = W * 0.018, H * 0.07
    ax.add_patch(MplPolygon(
        [[cx, na_base + na_h], [cx + na_w / 2, na_base],
         [cx, na_base + na_h * 0.25], [cx - na_w / 2, na_base]],
        facecolor=BOUNDARY_IL, edgecolor="white", lw=1.2, zorder=9))
    ax.text(cx, na_base + na_h + H * 0.012, "N", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=BOUNDARY_IL, zorder=9)
    return sb_x0, sb_y0, sb_h, na_base, na_h


def add_admin_overlay(ax, basin_geom, il_gdf, ilce_gdf):
    """Basin boundary on top, then province, then district."""
    ilce_clip = ilce_gdf.clip(basin_geom)
    ilce_clip = ilce_clip[~ilce_clip.is_empty & ilce_clip.is_valid]
    if len(ilce_clip):
        ilce_clip.boundary.plot(ax=ax, color=BOUNDARY_ILCE, linewidth=0.4,
                                linestyle="--", zorder=6, alpha=0.75)
    il_clip = il_gdf.clip(basin_geom)
    il_clip = il_clip[~il_clip.is_empty & il_clip.is_valid]
    if len(il_clip):
        il_clip.boundary.plot(ax=ax, color=BOUNDARY_IL, linewidth=0.9,
                              zorder=7, alpha=0.9)
    gpd.GeoSeries([basin_geom], crs=il_gdf.crs).boundary.plot(
        ax=ax, color=BASIN_COLOR, linewidth=1.4, zorder=8)
    return il_clip


def _load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd = src.nodata
        mask = (arr == nd) if nd is not None else ~np.isfinite(arr)
        bounds, crs, transform = src.bounds, src.crs, src.transform
    return arr, mask, bounds, crs, transform


# ─── Panel (a): Turkiye location, rendered standalone ────────────────────────

def render_panel_a(basin_wgs84, ilce_wgs84, turkey_outline):
    fig, ax = plt.subplots(figsize=(15, 5.5))

    turkey_outline.plot(ax=ax, facecolor="none", edgecolor=BOUNDARY_IL,
                        linewidth=1.2, zorder=3)
    ilce_wgs84.boundary.plot(ax=ax, color=BOUNDARY_ILCE, linewidth=0.15,
                             alpha=0.55, zorder=2)
    ax.set_xlim(24.5, 45.0)
    ax.set_ylim(35.5, 42.5)
    basin_wgs84.plot(ax=ax, facecolor=BASIN_COLOR, edgecolor="#c0392b",
                      linewidth=1.3, alpha=0.92, zorder=5)

    label_offsets = {"Istanbul": (8, 6), "Ankara": (8, 6), "Izmir": (8, -11)}
    for name, (lon, lat) in CITIES.items():
        ax.scatter(lon, lat, s=34, color=BOUNDARY_IL, zorder=6,
                   edgecolor="white", linewidths=0.9)
        ax.annotate(name, (lon, lat), xytext=label_offsets[name],
                    textcoords="offset points", fontsize=9, fontweight="bold",
                    color=BOUNDARY_IL, zorder=7,
                    path_effects=[pe.withStroke(linewidth=3.2, foreground="white")])

    add_lonlat_grid(ax, "EPSG:4326", is_geographic=True, step=5)
    sb_x0, sb_y0, sb_h, na_base, na_h = add_scale_north_geographic(ax, lat_mid=39.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_edgecolor(BOUNDARY_IL); ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_edgecolor(BOUNDARY_IL);   ax.spines["left"].set_linewidth(1.0)

    ax.set_title("(a)", loc="left", fontweight="bold", fontsize=18, pad=8)
    ax.legend(handles=[Patch(facecolor=BASIN_COLOR, edgecolor="#c0392b",
                              label="North Aegean Basin (Kuzey Ege Havzası)"),
                        Patch(facecolor="none", edgecolor=BOUNDARY_ILCE,
                              label="District (ilçe) boundary")],
              loc="lower left", fontsize=8, frameon=True, framealpha=0.92,
              edgecolor=BOUNDARY_ILCE)

    # CRS inset, positioned immediately left of the scale bar, horizontally
    # aligned with it (same y as the scale bar's own vertical centre), using
    # the block's own data-coordinate geometry so the two stay aligned
    # regardless of map extent.
    ax.text(sb_x0 - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.015,
            sb_y0 + sb_h * 0.5, "CRS: WGS 84 (EPSG:4326)\ngeographic coordinates",
            ha="right", va="center", fontsize=7.5, color=BOUNDARY_IL, zorder=9,
            bbox=dict(facecolor="white", edgecolor=BOUNDARY_ILCE, alpha=0.92,
                      pad=4, boxstyle="round,pad=0.4"))

    buf = BytesIO()
    plt.savefig(buf, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ─── Panel (b): Esri World Imagery ───────────────────────────────────────────

def panel_b_imagery(ax, basin_3857, crs_3857, il_3857, ilce_3857):
    import contextily as ctx

    bounds = basin_3857.total_bounds
    pad = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.04
    xlim = (bounds[0] - pad, bounds[2] + pad)
    ylim = (bounds[1] - pad, bounds[3] + pad)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    print("  Fetching Esri World Imagery basemap tiles...")
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=crs_3857,
                     attribution=False, zoom="auto")

    with rasterio.open(PROCESSED / "lulc_simplified_2020.tif") as src:
        dst_transform, w, h = calculate_default_transform(
            src.crs, crs_3857, src.width, src.height, *src.bounds)
        lulc_3857 = np.zeros((h, w), dtype=np.float32)
        reproject(source=rasterio.band(src, 1), destination=lulc_3857,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dst_transform, dst_crs=crs_3857,
                  resampling=Resampling.nearest)

    urban_mask = (lulc_3857 == 1)
    rgba = np.zeros((*urban_mask.shape, 4), dtype=np.float32)
    rgba[urban_mask] = [*mcolors.to_rgb(URBAN_COLOR), 0.55]
    extent = [dst_transform.c, dst_transform.c + w * dst_transform.a,
              dst_transform.f + h * dst_transform.e, dst_transform.f]
    ax.imshow(rgba, extent=extent, origin="upper", zorder=4, interpolation="nearest")

    basin_geom_3857 = basin_3857.union_all()
    add_admin_overlay(ax, basin_geom_3857, il_3857, ilce_3857)
    for _, row in il_3857.clip(basin_geom_3857).iterrows():
        pt = row.geometry.representative_point()
        ax.text(pt.x, pt.y, str(row["il_adi"]).upper(), fontsize=7.5, ha="center",
                va="center", color=BOUNDARY_IL, fontweight="bold", alpha=0.92, zorder=9,
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])

    fig10.add_scale_north(ax, type("B", (), {"left": xlim[0], "bottom": ylim[0],
                                              "right": xlim[1], "top": ylim[1]})())
    add_lonlat_grid(ax, crs_3857, is_geographic=False)
    ax.set_title("(b)", loc="left", fontweight="bold", fontsize=18, pad=8)

    leg_items = [Patch(facecolor=URBAN_COLOR, edgecolor="none", alpha=0.75, label="Urban / impervious (2020)"),
                 Patch(facecolor="none", edgecolor=BASIN_COLOR, linewidth=1.5, label="Basin boundary"),
                 Line2D([0], [0], color=BOUNDARY_IL, lw=0.9, label="Province boundary (il)"),
                 Line2D([0], [0], color=BOUNDARY_ILCE, lw=0.4, linestyle="--", label="District boundary (ilçe)")]
    ax.legend(handles=leg_items, loc="lower left", fontsize=6.8, frameon=True,
              framealpha=0.92, edgecolor=BOUNDARY_ILCE)
    ax.text(0.99, 0.01, "Imagery © Esri, Maxar, Earthstar Geographics",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
            color="white", path_effects=[pe.withStroke(linewidth=2, foreground="black")])


# ─── Panel (c): elevation + rivers + flood-prone areas ───────────────────────

def panel_c_elevation(ax, basin_geom_32635, il_32635, ilce_32635, dsi_flood_32635):
    elev, emask, bounds, crs, transform = _load_raster(DRIVERS_DIR / "driver_elevation.tif")
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # Restrict the raster display to the basin polygon itself -- the driver
    # grid is a rectangular bounding box, and the area outside the basin
    # (still inside that rectangle) is not part of the study area. Crop the
    # axes view to the basin's own bounds (+ small pad) too, matching panel
    # (b), rather than the full rectangular driver-grid extent.
    from rasterio.features import geometry_mask
    basin_pixel_mask = geometry_mask([basin_geom_32635], transform=transform,
                                     out_shape=elev.shape, invert=True)
    outside = ~basin_pixel_mask
    bminx, bminy, bmaxx, bmaxy = basin_geom_32635.bounds
    bpad = max(bmaxx - bminx, bmaxy - bminy) * 0.04
    ax.set_xlim(bminx - bpad, bmaxx + bpad)
    ax.set_ylim(bminy - bpad, bmaxy + bpad)

    ls = LightSource(azdeg=315, altdeg=45)
    terrain_cmap = LinearSegmentedColormap.from_list(
        "terrain2", ["#2d6a4f", "#95d5b2", "#e9edc9", "#d4a373", "#7f5539"])
    vmin, vmax = np.nanpercentile(np.where(emask, np.nan, elev), [1, 99])
    rgb = ls.shade(np.where(emask, vmin, elev), cmap=terrain_cmap, blend_mode="soft",
                   vert_exag=2, dx=30, dy=30, vmin=vmin, vmax=vmax)
    rgb[outside] = [1, 1, 1, 0]
    ax.imshow(rgb, extent=extent, origin="upper", zorder=1)

    upa, umask, ubounds, ucrs, utransform = _load_raster(
        INTERIM / "p2_merit_upstream_area_clipped.tif")
    river = np.zeros_like(elev)
    reproject(source=upa, destination=river,
              src_transform=utransform, src_crs=ucrs,
              dst_transform=transform, dst_crs=crs,
              resampling=Resampling.max)
    river_mask = (river >= RIVER_UPSTREAM_KM2) & basin_pixel_mask
    river_rgba = np.zeros((*river_mask.shape, 4), dtype=np.float32)
    river_rgba[river_mask] = [*mcolors.to_rgb(RIVER_COLOR), 0.9]
    ax.imshow(river_rgba, extent=extent, origin="upper", zorder=3, interpolation="nearest")

    if dsi_flood_32635 is not None and len(dsi_flood_32635):
        dsi_flood_32635.plot(ax=ax, facecolor=FLOOD_COLOR, alpha=0.35, zorder=4)
        dsi_flood_32635.boundary.plot(ax=ax, color=FLOOD_COLOR, linewidth=1.6, zorder=5)

    add_admin_overlay(ax, basin_geom_32635, il_32635, ilce_32635)
    view_bounds = type("B", (), {"left": bminx - bpad, "bottom": bminy - bpad,
                                  "right": bmaxx + bpad, "top": bmaxy + bpad})()
    fig10.add_scale_north(ax, view_bounds)
    add_lonlat_grid(ax, crs, is_geographic=False)
    ax.set_title("(c)", loc="left", fontweight="bold", fontsize=18, pad=8)

    sm = plt.cm.ScalarMappable(cmap=terrain_cmap, norm=mcolors.Normalize(vmin, vmax))
    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.15)
    cb = plt.colorbar(sm, cax=cax)
    cb.set_label("Elevation (m)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    leg_items = [Line2D([0], [0], color=RIVER_COLOR, lw=1.8,
                        label=f"River network (≥{RIVER_UPSTREAM_KM2} km² upstream area)"),
                 Patch(facecolor=FLOOD_COLOR, edgecolor=FLOOD_COLOR, alpha=0.5,
                       label="Flood-prone area (n=25)"),
                 Patch(facecolor="none", edgecolor=BASIN_COLOR, linewidth=1.4, label="Basin boundary"),
                 Line2D([0], [0], color=BOUNDARY_IL, lw=0.9, label="Province boundary (il)"),
                 Line2D([0], [0], color=BOUNDARY_ILCE, lw=0.4, linestyle="--", label="District boundary (ilçe)")]
    ax.legend(handles=leg_items, loc="lower left", fontsize=6.5, frameon=True,
              framealpha=0.92, edgecolor=BOUNDARY_ILCE)


def render_panels_bc(basin_3857, crs_3857, il_3857, ilce_3857,
                      basin_geom_32635, il_32635, ilce_32635, dsi_flood_32635):
    fig, (axB, axC) = plt.subplots(1, 2, figsize=(15, 7.3))
    panel_b_imagery(axB, basin_3857, crs_3857, il_3857, ilce_3857)
    panel_c_elevation(axC, basin_geom_32635, il_32635, ilce_32635, dsi_flood_32635)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def create_fig_s3() -> None:
    print("Generating Supplementary Figure S3: Study area location map...")
    basin_wgs84 = gpd.read_file(INTERIM / "kuzey_ege_havzasi_boundary.gpkg", use_arrow=False).to_crs("EPSG:4326")
    basin_32635 = gpd.read_file(INTERIM / "kuzey_ege_havzasi_boundary.gpkg", use_arrow=False)
    basin_geom_32635 = basin_32635.union_all()
    crs_3857 = "EPSG:3857"
    basin_3857 = basin_wgs84.to_crs(crs_3857)

    ilce_wgs84 = load_turkiye_ilce()
    print(f"  Loaded {len(ilce_wgs84)} districts (il_adi/ilce_adi, ArcGIS Online)")
    turkey_dissolved = ilce_wgs84.union_all()
    turkey_outline = gpd.GeoDataFrame(geometry=[remove_holes(turkey_dissolved)], crs="EPSG:4326")

    il_wgs84 = ilce_wgs84.dissolve(by="il_adi", as_index=False)[["il_adi", "geometry"]]
    il_3857    = il_wgs84.to_crs(crs_3857)
    ilce_3857  = ilce_wgs84.to_crs(crs_3857)
    il_32635   = il_wgs84.to_crs(basin_32635.crs)
    ilce_32635 = ilce_wgs84.to_crs(basin_32635.crs)

    dsi_flood_path = INTERIM / "dsi_masterplan" / "dsi_taskin_alanlari.gpkg"
    dsi_flood_32635 = gpd.read_file(dsi_flood_path, use_arrow=False) if dsi_flood_path.exists() else None
    if dsi_flood_32635 is not None and dsi_flood_32635.crs != basin_32635.crs:
        dsi_flood_32635 = dsi_flood_32635.to_crs(basin_32635.crs)

    img_a  = render_panel_a(basin_wgs84, ilce_wgs84, turkey_outline)
    img_bc = render_panels_bc(basin_3857, crs_3857, il_3857, ilce_3857,
                              basin_geom_32635, il_32635, ilce_32635, dsi_flood_32635)

    # Composite with a small, fixed gap -- no aspect-driven blank space.
    target_w = max(img_a.width, img_bc.width)
    if img_a.width != target_w:
        img_a = img_a.resize((target_w, int(img_a.height * target_w / img_a.width)))
    if img_bc.width != target_w:
        img_bc = img_bc.resize((target_w, int(img_bc.height * target_w / img_bc.width)))

    gap = 24
    combined = Image.new("RGB", (target_w, img_a.height + gap + img_bc.height), "white")
    combined.paste(img_a, (0, 0))
    combined.paste(img_bc, (0, img_a.height + gap))

    out_path = FIG_DIR / "fig13_LocationMap.png"
    combined.save(out_path, dpi=(300, 300))
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    create_fig_s3()
