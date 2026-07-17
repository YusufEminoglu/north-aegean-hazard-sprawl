"""
Paper 2 — Figure 8: Urban Exposure Under Multi-Hazard Scenarios
================================================================
Panel (a): New urban area by hazard class — BAU vs ECO grouped bars
Panel (b): Net risk reduction (BAU − ECO) — diverging horizontal bar
Style: elite colormap palette, matching fig03–fig07 aesthetic.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in locals() else Path(".")
TABLES  = PROJECT_ROOT / "outputs" / "tables"  / "paper2"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "paper2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

elite_cmap = LinearSegmentedColormap.from_list("elite", ["#dfe6e9", "#00cec9", "#ff7675"])

BAU_COLOR    = "#ff7675"   # coral
ECO_COLOR    = "#00cec9"   # teal
BOUNDARY_IL  = "#2c3e50"
GRID_COLOR   = "#b2bec3"

HAZARD_LABELS = [
    "1 — Very Low", "2 — Low", "3 — Moderate",
    "4 — High",     "5 — Very High"
]


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def fig08_exposure():
    print("Generating Fig 08: Urban Exposure Under Scenarios...")

    bau_path = TABLES / "exposure_sim_2050_bau.csv"
    eco_path = TABLES / "exposure_sim_2050_eco.csv"

    if not bau_path.exists() or not eco_path.exists():
        print("  ERROR: exposure CSV files not found."); return

    df_bau = pd.read_csv(bau_path)
    df_eco = pd.read_csv(eco_path)

    bau_ha  = df_bau["New_Urban_Area_ha"].values
    eco_ha  = df_eco["New_Urban_Area_ha"].values
    bau_pct = df_bau["Percentage_%"].values
    eco_pct = df_eco["Percentage_%"].values
    n_cls   = len(bau_ha)
    x       = np.arange(n_cls)
    bw      = 0.36   # bar width

    # ECO − BAU: positive = ECO has more, negative = ECO has less
    reduction_ha  = eco_ha  - bau_ha
    reduction_pct = eco_pct - bau_pct

    fig = plt.figure(figsize=(16, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.28)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # ── Panel (a): Grouped bar — BAU vs ECO ───────────────────────────────────
    bars_bau = ax_a.bar(x - bw/2, bau_ha, width=bw, color=BAU_COLOR,
                         edgecolor="white", linewidth=0.5, alpha=0.92,
                         label="BAU 2050", zorder=3)
    bars_eco = ax_a.bar(x + bw/2, eco_ha, width=bw, color=ECO_COLOR,
                         edgecolor="white", linewidth=0.5, alpha=0.92,
                         label="ECO 2050", zorder=3)

    # Percentage labels above each bar
    for bar, pct in zip(bars_bau, bau_pct):
        ax_a.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + max(bau_ha) * 0.012,
                  f"{pct:.1f}%", ha="center", va="bottom",
                  fontsize=8, color=BAU_COLOR, fontweight="bold")
    for bar, pct in zip(bars_eco, eco_pct):
        ax_a.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + max(bau_ha) * 0.012,
                  f"{pct:.1f}%", ha="center", va="bottom",
                  fontsize=8, color=ECO_COLOR, fontweight="bold")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(HAZARD_LABELS[:n_cls], fontsize=9.5,
                          color=BOUNDARY_IL, fontweight="bold")
    ax_a.set_ylabel("New Urban Area (ha)", fontweight="bold", labelpad=8)
    ax_a.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_a.set_ylim(0, max(bau_ha) * 1.22)
    ax_a.grid(axis="y", linestyle="--", linewidth=0.5,
               alpha=0.45, color=GRID_COLOR, zorder=0)
    ax_a.spines["bottom"].set_linewidth(1.5)
    ax_a.spines["left"].set_linewidth(1.5)
    ax_a.tick_params(axis="x", length=0, pad=6)
    ax_a.tick_params(axis="y", labelsize=9)

    # Summary stats box
    total_bau = bau_ha.sum()
    total_eco = eco_ha.sum()
    # Weighted mean hazard class (demand is equal; risk differs by spatial allocation)
    classes   = np.arange(1, n_cls + 1)
    wmh_bau   = (classes * bau_ha).sum() / total_bau if total_bau > 0 else 0
    wmh_eco   = (classes * eco_ha).sum() / total_eco if total_eco > 0 else 0
    # Very High hazard (class 5) shift
    vh_delta_ha  = eco_ha[-1] - bau_ha[-1]
    vh_delta_pct = eco_pct[-1] - bau_pct[-1]
    stats_txt = (
        f"Total new urban (equal demand) : {total_bau:,.0f} ha\n"
        f"Weighted mean hazard — BAU     : {wmh_bau:.2f}\n"
        f"Weighted mean hazard — ECO     : {wmh_eco:.2f}  (Δ {wmh_eco-wmh_bau:+.2f})\n"
        f"Very High (cl.5) shift              : {vh_delta_ha:+,.0f} ha  ({vh_delta_pct:+.1f} pp)"
    )
    ax_a.text(0.97, 1.01, stats_txt, transform=ax_a.transAxes,
              fontsize=8.5, va="top", ha="right", linespacing=1.6,
              bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                        edgecolor=BOUNDARY_IL, alpha=0.92, lw=1.2))

    ax_a.legend(handles=[
        Patch(facecolor=BAU_COLOR, edgecolor="none", label="BAU 2050"),
        Patch(facecolor=ECO_COLOR, edgecolor="none", label="ECO 2050"),
    ], loc="upper left", fontsize=9.5, frameon=True,
       facecolor="white", edgecolor=GRID_COLOR, framealpha=0.9)

    ax_a.set_title("(a)", loc="left", fontweight="bold", fontsize=18, pad=8)

    # ── Panel (b): Diverging bar — net risk reduction ──────────────────────────
    y_pos  = np.arange(n_cls)
    # Beneficial: more growth in low-hazard (class 1-3, positive) OR
    #             less growth in high-hazard (class 4-5, negative)
    colors = [
        ECO_COLOR if (rha >= 0 and i <= 2) or (rha < 0 and i >= 3)
        else BAU_COLOR
        for i, rha in enumerate(reduction_ha)
    ]

    ax_b.barh(y_pos, reduction_ha, color=colors,
               edgecolor="white", linewidth=0.5, alpha=0.92,
               height=0.58, zorder=3)

    # Zero reference line
    ax_b.axvline(0, color=BOUNDARY_IL, linewidth=1.2, zorder=4)

    # Value labels — Very High (last bar) goes inside the bar
    xmax = max(abs(reduction_ha))
    for yi, (rha, rpct) in enumerate(zip(reduction_ha, reduction_pct)):
        sign = "+" if rha >= 0 else ""
        label = f"{sign}{rha:,.0f} ha  ({sign}{rpct:.1f}%)"
        is_last = (yi == len(reduction_ha) - 1)
        if is_last:
            # Inside the bar — centred
            ax_b.text(rha / 2, yi, label,
                      ha="center", va="center", fontsize=8.5,
                      color="white", fontweight="bold",
                      path_effects=[pe.withStroke(linewidth=2.0,
                                                  foreground=colors[yi])])
        else:
            xoff = xmax * 0.03
            ha_  = "left" if rha >= 0 else "right"
            xpos = rha + xoff if rha >= 0 else rha - xoff
            ax_b.text(xpos, yi, label,
                      ha=ha_, va="center", fontsize=8.5,
                      color=BOUNDARY_IL, fontweight="bold",
                      path_effects=[pe.withStroke(linewidth=2.5,
                                                  foreground="white")])

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(HAZARD_LABELS[:n_cls], fontsize=9.5,
                          color=BOUNDARY_IL, fontweight="bold")
    ax_b.set_xlabel("ECO − BAU  (ha)", fontweight="bold", labelpad=8)
    ax_b.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:+,.0f}"))
    ax_b.grid(axis="x", linestyle="--", linewidth=0.5,
               alpha=0.45, color=GRID_COLOR, zorder=0)
    ax_b.spines["bottom"].set_linewidth(1.5)
    ax_b.spines["left"].set_linewidth(1.5)
    ax_b.tick_params(axis="y", length=0, pad=6)
    ax_b.tick_params(axis="x", labelsize=9)

    # Annotation: positive = risk saved
    ax_b.text(0.04, 0.08,
              "Teal = beneficial shift\nCoral = adverse shift",
              transform=ax_b.transAxes, fontsize=8, va="bottom", ha="left",
              style="italic", color=BOUNDARY_IL)

    ax_b.set_title("(b)", loc="left", fontweight="bold", fontsize=18, pad=8)

    out_path = FIG_DIR / "fig08_Exposure.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close()
    print(f"  Saved: {out_path.name}")


if __name__ == "__main__":
    fig08_exposure()
