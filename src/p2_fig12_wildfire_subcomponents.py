"""Supplementary Figure S2: Wildfire sub-component decomposition.

Visualises the per-sub-input validation: none of the wildfire composite's
three sub-inputs (GABAM burned frequency, summer EVI, canopy height) is
individually a meaningful positive discriminator against independent
MODIS/VIIRS FIRMS thermal-anomaly detections within the forest/shrubland
domain the construct targets.

Re-derives the exact numbers p2_12_wildfire_seismic_validation.py's
wildfire_subcomponent_decomposition() prints (same rasters, same forest
mask, same nodata-safe reproject) so the figure is guaranteed consistent
with the reported AUC values rather than a separately-computed, potentially
drifting rendering of the same claim.

Style matches fig07 (elite_cmap, same rcParams, same KDE-panel convention).

Usage:
  python src/p2_fig12_wildfire_subcomponents.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from rasterio.warp import Resampling, reproject
from scipy.stats import gaussian_kde, mannwhitneyu
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("p2_12", HERE / "p2_12_wildfire_seismic_validation.py")
p2_12 = importlib.util.module_from_spec(_spec)
# p2_12's module body only defines constants/functions at import time (no
# GEE/network calls happen until wildfire_validation()/seismic_validation()
# are explicitly called), so exec_module is safe and cheap here.
_spec.loader.exec_module(p2_12)

PROJECT_ROOT = p2_12.ROOT
PROCESSED    = p2_12.PROC
INTERIM      = p2_12.INTERIM
FIG_DIR      = PROJECT_ROOT / "outputs" / "figures" / "paper2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "axes.titlesize": 11, "axes.labelsize": 10, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

elite_cmap  = LinearSegmentedColormap.from_list("elite", ["#dfe6e9", "#00cec9", "#ff7675"])
BOUNDARY_IL = "#2c3e50"
COLOR_FIRE     = "#ff7675"  # FIRMS-positive
COLOR_NOFIRE   = "#00cec9"  # FIRMS-negative

PANELS = [
    ("gabam", "GABAM Cumulative Burned Frequency", "count (1990-2021)"),
    ("evi",   "Summer EVI",                        "EVI (unitless)"),
    ("tch",   "Canopy Height",                      "metres"),
]


def create_fig_s2() -> None:
    print("Generating Supplementary Figure S2: Wildfire sub-component decomposition...")

    with rasterio.open(p2_12.PROC / "wildfire_hazard.tif") as ref:
        ref_meta = ref.meta.copy()
        wf       = ref.read(1).astype(np.float32)
        nodata   = ref.nodata
    valid = (wf != nodata) if nodata is not None else np.isfinite(wf)

    fire = np.zeros((ref_meta["height"], ref_meta["width"]), dtype=np.float32)
    with rasterio.open(INTERIM / "p2_firms_fire_frequency.tif") as src:
        reproject(source=rasterio.band(src, 1), destination=fire,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
                  resampling=Resampling.bilinear)

    with rasterio.open(PROCESSED / "lulc_simplified_2020.tif") as s:
        lulc = np.zeros((ref_meta["height"], ref_meta["width"]), dtype=np.float32)
        reproject(source=rasterio.band(s, 1), destination=lulc,
                  src_transform=s.transform, src_crs=s.crs,
                  dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
                  resampling=Resampling.nearest)
    forest     = valid & (lulc == 3)
    obs_forest = forest & (fire > 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panel_letters = ["(a)", "(b)", "(c)"]

    for ax, letter, (key, title, xlabel) in zip(axes, panel_letters, PANELS):
        arr      = p2_12._load_aligned_nan(p2_12.SUBINPUTS[key], ref_meta)
        has_data = forest & np.isfinite(arr)
        cov      = float(has_data.sum()) / float(forest.sum()) * 100

        v_fire   = arr[has_data & obs_forest]
        v_nofire = arr[has_data & ~obs_forest]
        auc      = roc_auc_score(obs_forest[has_data].astype(np.uint8), arr[has_data])
        _, p_mw  = mannwhitneyu(v_fire, v_nofire, alternative="two-sided")

        for vals, color, label in [(v_nofire, COLOR_NOFIRE, "FIRMS-negative"),
                                    (v_fire,   COLOR_FIRE,   "FIRMS-positive")]:
            if len(vals) < 2 or np.ptp(vals) == 0:
                continue
            sub = vals if len(vals) <= 20000 else np.random.default_rng(42).choice(vals, 20000, replace=False)
            kde = gaussian_kde(sub)
            xr  = np.linspace(vals.min(), vals.max(), 200)
            ax.fill_between(xr, kde(xr), color=color, alpha=0.35)
            ax.plot(xr, kde(xr), color=color, lw=1.6, label=f"{label} (n={len(vals):,})")
            ax.axvline(np.mean(vals), color=color, lw=1.2, linestyle="--", alpha=0.85)

        ax.set_title(letter, loc="left", fontweight="bold", color=BOUNDARY_IL, fontsize=18, pad=8)
        ax.text(0.5, 1.02, title, transform=ax.transAxes, ha="center", va="bottom",
                fontweight="bold", color=BOUNDARY_IL, fontsize=10.5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_yticks([])
        ax.set_ylabel("Density", fontsize=9)
        sig = "n.s." if p_mw >= 0.001 else "p<0.001"
        ax.text(0.97, 0.94, f"AUC = {auc:.3f}\ncoverage = {cov:.1f}% of forest/shrub\nMann-Whitney {sig}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                color=BOUNDARY_IL,
                bbox=dict(facecolor="white", edgecolor="#b2bec3", alpha=0.92,
                          pad=4, boxstyle="round,pad=0.4"))
        ax.legend(loc="upper left", fontsize=7.8, frameon=True, framealpha=0.92,
                  edgecolor="#b2bec3")
        ax.axhline(0, color="#b2bec3", lw=0.6)

    fig.suptitle(
        "Wildfire hazard sub-inputs vs. independent FIRMS thermal-anomaly detections "
        "(forest/shrubland domain)",
        fontsize=12.5, fontweight="bold", color=BOUNDARY_IL, y=1.02)
    fig.text(0.5, -0.02,
              "Dashed lines mark class means. AUC $\\approx$ 0.5 = chance discrimination; none of the three "
              "sub-inputs individually validates well, consistent with the composite's own negative result.",
              ha="center", fontsize=8.5, color="#636e72", style="italic")

    plt.tight_layout()
    out_path = FIG_DIR / "fig12_WildfireSubcomponents.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    create_fig_s2()
