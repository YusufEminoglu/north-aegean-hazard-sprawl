import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes
import rasterio
from matplotlib.colors import LinearSegmentedColormap, LightSource
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import string

# UNIFIED AESTHETICS
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in locals() else Path(".")
DRIVERS_DIR = PROJECT_ROOT / "data" / "03_processed" / "paper2" / "drivers"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "paper2"

INTERIM_DIR = PROJECT_ROOT / "data" / "02_interim" / "paper2"

# Additional hazard-specific inputs not in the drivers folder
HAZARD_INPUTS = [
    # ── Flood hazard ──────────────────────────────────────────────────────────
    (INTERIM_DIR / "p2_merit_hand_clipped.tif",              "HAND",             "(m)"),
    (INTERIM_DIR / "p2_srtm_tpi_clipped.tif",               "TPI",              "(Index)"),
    # NDWI dropped (correlated with NDVI already in drivers)
    # (INTERIM_DIR / "p2_ndwi_summer_2024_clipped.tif",      "NDWI",             "(Index)"),
    (INTERIM_DIR / "p2_worldclim_bio13_maxprecip_clipped.tif","Max Precipitation","(mm)"),
    # ── Bio-climatic hazard ───────────────────────────────────────────────────
    (INTERIM_DIR / "p2_lst_summer_composite.tif",            "LST Summer",       "(°C)"),
    (INTERIM_DIR / "p2_uhi_summer_composite.tif",            "UHI",              "(z-score)"),
    (INTERIM_DIR / "p2_s5p_no2_summer.tif",                  "S5P NO2",          "(mol/m2)"),
    (INTERIM_DIR / "p2_summer_precip_jja.tif",               "Precip JJA",       "(mm)"),
    # ── Wildfire hazard ───────────────────────────────────────────────────────
    (INTERIM_DIR / "p2_gabam_burned_frequency.tif",          "GABAM Burned Freq","(count)"),
    (INTERIM_DIR / "p2_evi_summer_composite.tif",            "EVI Summer",       "(Index)"),
]

def create_fig03_advanced():
    print("Generating Advanced Fig 03 (Spatial Drivers + Hazard Inputs Grid)...")

    # 1. Gather driver paths (FLUS model inputs)
    driver_paths = sorted(DRIVERS_DIR.glob("driver_*.tif"))
    
    # Elite Colormap from Fig02 Panel A
    elite_cmap = LinearSegmentedColormap.from_list("elite", ["#f5f6fa", "#00cec9", "#ff7675"])
    
    # Read elevation explicitly for Hillshade across all maps
    elev_path = DRIVERS_DIR / "driver_elevation.tif"
    with rasterio.open(elev_path) as src_elev:
        elev = src_elev.read(1)
        valid_mask = elev != src_elev.nodata
        boundary_mask = binary_fill_holes(valid_mask)
        elev[~valid_mask] = 0
        
    # Generate unified Hillshade
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(elev, vert_exag=3, dx=30, dy=30)
    
        # Pre-calculate Drop Shadow
    shift_px = 15 # Lowered shift for more subtle shadow
    shadow_mask = np.zeros_like(boundary_mask, dtype=bool)
    shadow_mask[shift_px:, shift_px:] = boundary_mask[:-shift_px, :-shift_px]
    
    # 2. Build combined panel list: (path, label, unit)
    # Coast Dist dropped (less relevant for inland hazard analysis)
    SKIP_DRIVERS = {"driver_coast_dist"}
    panels = []
    for p in driver_paths:
        if p.stem in SKIP_DRIVERS:
            continue
        name = p.stem.replace("driver_", "").replace("_", " ").title()
        if "Dist" in name:
            unit = "(km)"
        elif "Elevation" in name:
            unit = "(m)"
        elif "Slope" in name:
            unit = "(°)"
        elif "Canopy" in name or "Clay" in name:
            unit = "(%)"
        elif "Ndvi" in name:
            unit = "(Index)"
        elif "Ntl" in name:
            unit = "(DN)"
        elif "Pop" in name:
            unit = "(pop/km²)"
        else:
            unit = ""
        panels.append((p, name, unit))

    for p, label, unit in HAZARD_INPUTS:
        if p.exists():
            panels.append((p, label, unit))
        else:
            print(f"  Skipped (not found): {p.name}")

    n_cols  = 4
    n_rows  = -(-len(panels) // n_cols)   # ceiling division → 20/4 = 5 rows
    fig_h   = 4.2 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, fig_h))
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    axes_flat = axes.flatten()

    hill_masked = np.ma.masked_where(~valid_mask, hillshade)

    for idx, (path, label, unit) in enumerate(panels):
        ax = axes_flat[idx]

        with rasterio.open(path) as src:
            data   = src.read(1).astype(float)
            nodata = src.nodata
            bounds = src.bounds

        extent  = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        x_range = bounds.right - bounds.left
        y_range = bounds.top   - bounds.bottom

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(bounds.left  - x_range * 0.05, bounds.right + x_range * 0.05)
        ax.set_ylim(bounds.bottom - y_range * 0.20, bounds.top  + y_range * 0.10)

        # Drop shadow — always use boundary_mask shape; extent handles geo alignment
        shadow_arr = np.zeros_like(boundary_mask, dtype=float)
        shadow_arr[~shadow_mask] = np.nan
        ax.imshow(shadow_arr, cmap='Greys', vmin=0, vmax=1,
                  alpha=0.4, extent=extent, zorder=1)

        ax.imshow(hill_masked, cmap='gray', extent=extent, zorder=2, alpha=0.6)
        ax.contour(boundary_mask, levels=[0.5], colors='#2c3e50',
                   linewidths=0.8, extent=extent, origin='upper', zorder=4)

        # Mask nodata
        data_mask = ~np.isnan(data)
        if nodata is not None:
            data_mask = data_mask & (data != nodata)
        data_masked = np.ma.masked_where(~data_mask, data)

        # Convert distances to km
        if "(km)" in unit:
            data_masked = data_masked / 1000.0

        valid_data = data_masked.compressed()
        if len(valid_data) == 0:
            ax.set_visible(False); continue
        vmin, vmax = np.percentile(valid_data, [5, 95])

        im = ax.imshow(data_masked, cmap=elite_cmap, vmin=vmin, vmax=vmax,
                       extent=extent, zorder=3, alpha=0.85)

        letter = string.ascii_lowercase[idx] if idx < 26 else f"a{string.ascii_lowercase[idx-26]}"
        ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left', zorder=5)

        cax  = inset_axes(ax, width="60%", height="4%", loc='lower center', borderpad=1)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label(f"{label} {unit}".strip(), fontsize=10,
                       fontweight='bold', labelpad=4)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)

    # Hide unused cells
    for idx in range(len(panels), n_rows * n_cols):
        axes_flat[idx].set_visible(False)

    out_path = FIG_DIR / "fig03_Drivers.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path.name}")

if __name__ == "__main__":
    create_fig03_advanced()
