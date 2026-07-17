import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# UNIFIED AESTHETICS
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in locals() else Path(".")
TABLES = PROJECT_ROOT / "outputs" / "tables" / "paper2"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "paper2"

def create_fig02():
    print("Generating Fig 02...")
    
    tpm_df  = pd.read_csv(TABLES / "transition_probability_matrix.csv", index_col=0)
    classes = tpm_df.index.tolist()

    # Project 2000→2020 matrix forward to 2000→2025 via Markov chain scaling.
    # T^(25/20) gives the 25-year transition under the same annual dynamics.
    from scipy.linalg import fractional_matrix_power
    T       = tpm_df.values.astype(float)
    T_proj  = fractional_matrix_power(T, 25 / 20).real
    T_proj  = np.clip(T_proj, 0, None)
    T_proj  = T_proj / T_proj.sum(axis=1, keepdims=True)  # re-normalise rows
    tpm_df  = pd.DataFrame(T_proj, index=classes, columns=classes)
    
    # Elite LULC Palette harmonized with the Heatmap colors (Salmon, Teal, Tinted Grey)
    lulc_colors = ['#ff7675', '#fab1a0', '#00cec9', '#74b9ff', '#dfe6e9', '#a29bfe']
    
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.15)
    
    # Absolute Title Alignment to ensure (a) and (b) are perfectly on the same Y-axis
    # Lowered slightly to reduce distance to the graphs
    fig.text(0.12, 0.92, "(a)", fontweight='bold', fontsize=16)
    fig.text(0.52, 0.92, "(b)", fontweight='bold', fontsize=16)
    
    # ========================================================
    # PANEL A: PROFESSIONAL TPM HEATMAP
    # ========================================================
    ax_a = fig.add_subplot(gs[0])
    
    # Custom Elite Colormap (Tinted Grey -> Teal -> Yavruagzi/Salmon)
    from matplotlib.colors import LinearSegmentedColormap
    elite_cmap = LinearSegmentedColormap.from_list("elite", ["#f5f6fa", "#00cec9", "#ff7675"])
    
    sns.heatmap(tpm_df, annot=True, fmt=".3f", cmap=elite_cmap, 
                linewidths=1, linecolor='white', square=True, 
                cbar_kws={"shrink": .8, "label": "Transition Probability"}, 
                ax=ax_a, annot_kws={"size": 11, "weight": "bold"})
                
    ax_a.set_ylabel("Transition From (2000)", fontweight='bold')
    ax_a.set_xlabel("Transition To (2025)", fontweight='bold')
    
    # X-axis label removed to be drawn with absolute figure coordinates later
    ax_a.set_xlabel("")
    
    plt.setp(ax_a.get_xticklabels(), rotation=45, ha='right', fontweight='bold')
    plt.setp(ax_a.get_yticklabels(), rotation=0, fontweight='bold')

    # ========================================================
    # PANEL B: LULC TRANSITION FLOW (Stacked Bar)
    # ========================================================
    ax_b = fig.add_subplot(gs[1])
    
    tpm_df.plot(kind='bar', stacked=True, color=lulc_colors, ax=ax_b, edgecolor='white', width=0.7)
    
    ax_b.set_ylabel("Transition Proportion", fontweight='bold')
    
    # X-axis label removed to be drawn with absolute figure coordinates later
    ax_b.set_xlabel("")
    
    # Hide standard labels completely to avoid layout clashes
    ax_b.set_xticklabels([])
    ax_b.tick_params(axis='x', length=0)
    
    # Add percentage text with white buffer inside bars if >= 10%
    for p in ax_b.patches:
        height = p.get_height()
        if height >= 0.10:
            x_pos = p.get_x() + p.get_width() / 2
            y_pos = p.get_y() + height / 2
            ax_b.text(x_pos, y_pos, f"{height*100:.0f}%", ha='center', va='center', 
                      fontsize=9, fontweight='bold', color='#2c3e50',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none'))
                      
    # Drop Legend
    if ax_b.get_legend():
        ax_b.get_legend().remove()
        
    # Manually draw text and dots perfectly aligned below the axis
    for idx, (cls_name, color) in enumerate(zip(classes, lulc_colors)):
        # Text just below the axis line
        ax_b.text(idx, -0.02, cls_name, ha='center', va='top', fontweight='bold', 
                  transform=ax_b.get_xaxis_transform(), clip_on=False)
        # Dot strictly below the text, no stroke (edgecolor='none')
        ax_b.scatter(idx, -0.09, color=color, s=250, clip_on=False, edgecolor='none', zorder=10, 
                     transform=ax_b.get_xaxis_transform())

    # Absolute positioning for X-axis labels so they are perfectly aligned on the Y-axis
    # Raised y from 0.01 to 0.04 to decrease distance to the x-axis labels
    fig.text(0.28, 0.04, "Transition To (2025)", ha='center', va='center', fontweight='bold', fontsize=12)
    fig.text(0.74, 0.04, "Original Land Cover (2000)", ha='center', va='center', fontweight='bold', fontsize=12)

    # Adding a subtle pad at the bottom
    plt.subplots_adjust(bottom=0.18)

    out_path = FIG_DIR / "fig02_Dynamics.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path.name}")

if __name__ == "__main__":
    create_fig02()
