"""
Paper 2 — FLUS/CA Engine
===========================
1. Estimate 2050 urban demand (using historical 1985-2020 growth rate extrapolation)
2. Train RF on 2000->2020 LULC urban transitions using 12 drivers
3. Run FLUS Cellular Automata simulation to 2050
   - BAU: Physical exclusions only
   - ECO: Environmental exclusions (protected areas, high canopy, riparian)
"""

import numpy as np
import rasterio
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from scipy.ndimage import convolve
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "03_processed" / "paper2"
DRIVERS_DIR = PROCESSED / "drivers"
MODELS = PROJECT_ROOT / "data" / "04_models" / "paper2"
TABLES = PROJECT_ROOT / "outputs" / "tables" / "paper2"

MODELS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

RF_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
}

# 11 spatial drivers (manuscript Table T3).
# NOTE: canopy height (Meta 2023) was removed from the RF driver set to avoid
# using a post-period landscape-state predictor in the suitability model; it is
# retained only in the wildfire hazard composite (Stage III). NDVI now uses a
# 2018-2020 Landsat composite (driver_ndvi.tif sourced from the 2020 raster)
# so that no predictor postdates the 2020 reference state.
DRIVER_NAMES = [
    "slope", "elevation", "road_dist", "river_dist",
    "water_dist", "pop_density", "ntl", "ndvi",
    "clay", "coast_dist", "urban_dist",
]

def load_drivers():
    """Load all spatial drivers into a stacked array."""
    print("Loading spatial drivers...")
    stack = []
    meta = None
    for name in DRIVER_NAMES:
        path = DRIVERS_DIR / f"driver_{name}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Missing driver: {name}")
        with rasterio.open(path) as src:
            if meta is None:
                meta = src.meta.copy()
            arr = src.read(1)
            arr = np.nan_to_num(arr, nan=0.0) # Handle nans
            stack.append(arr)
    return np.stack(stack, axis=-1), meta

def calculate_2050_demand(lulc_2000, lulc_2020):
    """
    Estimate 2050 urban pixel demand.

    Path A (preferred): WorldPop ilce CSV (p2_worldpop_ilce_2020_2050.csv)
      - Reads district-level Pop_2020 and Pop_2050 from GEE output
      - Growth rate = Pop_2050 / Pop_2020
      - Assumes urban land demand scales with population growth rate
      - Method: urban_2020_px * population_growth_rate

    Path B (fallback): Historical LULC pixel trend
      - Linear extrapolation of 2000->2020 pixel-level urban growth
      - Simpler but robust; sufficient for preliminary analysis
    """
    print("Calculating 2050 Urban Demand...")
    urban_2000 = int(np.sum(lulc_2000 == 1))
    urban_2020 = int(np.sum(lulc_2020 == 1))
    pixel_area_ha = (30 * 30) / 10000.0

    # Path A: WorldPop district-level demand
    csv_path = TABLES / "p2_worldpop_ilce_2020_2050.csv"
    if csv_path.exists():
        print(f"  WorldPop ilce CSV bulundu: {csv_path.name}")
        wp = pd.read_csv(csv_path)
        if "Pop_2020" in wp.columns and "Pop_2050" in wp.columns:
            pop_2020_total = wp["Pop_2020"].sum()
            pop_2050_total = wp["Pop_2050"].sum()
            if pop_2020_total > 0:
                pop_growth_rate = pop_2050_total / pop_2020_total
                demand_2050 = int(urban_2020 * pop_growth_rate)
                print(f"  [Path A - WorldPop CSV]")
                print(f"  Nufus 2020 (havza)   : {pop_2020_total:,.0f}")
                print(f"  Nufus 2050 (tahmin)  : {pop_2050_total:,.0f}")
                print(f"  Nufus buyume orani   : x{pop_growth_rate:.3f}")
                print(f"  Urban Pixels 2020    : {urban_2020:,}  ({urban_2020*pixel_area_ha:,.0f} ha)")
                print(f"  Target 2050          : {demand_2050:,}  ({demand_2050*pixel_area_ha:,.0f} ha)")
                return demand_2050, urban_2020

    # Path B: Pixel trend fallback
    print("  [Path B - Pixel trend extrapolation (2000->2020)]")
    annual_growth = (urban_2020 - urban_2000) / 20.0
    demand_2050   = int(urban_2020 + annual_growth * 30)  # 30 years to 2050
    print(f"  Urban Pixels 2000    : {urban_2000:,}  ({urban_2000*pixel_area_ha:,.0f} ha)")
    print(f"  Urban Pixels 2020    : {urban_2020:,}  ({urban_2020*pixel_area_ha:,.0f} ha)")
    print(f"  Annual growth (px)   : {annual_growth:,.0f}")
    print(f"  Target 2050          : {demand_2050:,}  ({demand_2050*pixel_area_ha:,.0f} ha)")
    return demand_2050, urban_2020


def grid_search_rf(X_train, y_train):
    """Stratified 5-fold grid search over the RF hyperparameter space reported
    in the manuscript (n_estimators x max_depth x min_samples_split, 27
    configurations). Returns the fitted GridSearchCV and writes the full grid
    to outputs/tables/paper2/rf_hyperparameter_grid.csv so every reported
    tuning number traces to an actual run.
    """
    print("Grid-searching RF hyperparameters (5-fold CV, 27 configurations)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base = RandomForestClassifier(n_jobs=1, random_state=42)
    gs = GridSearchCV(base, RF_GRID, scoring="roc_auc", cv=cv, n_jobs=-1, refit=False)
    gs.fit(X_train, y_train)

    results = pd.DataFrame(gs.cv_results_)
    cols = ["param_n_estimators", "param_max_depth", "param_min_samples_split",
            "mean_test_score", "std_test_score", "rank_test_score"]
    grid_table = results[cols].sort_values("rank_test_score").reset_index(drop=True)
    grid_table.columns = ["n_estimators", "max_depth", "min_samples_split",
                           "cv_auc_mean", "cv_auc_std", "rank"]
    grid_path = TABLES / "rf_hyperparameter_grid.csv"
    grid_table.to_csv(grid_path, index=False)
    print(f"  Saved grid ({len(grid_table)} configs) -> {grid_path.relative_to(PROJECT_ROOT)}")
    print(f"  Best: {gs.best_params_}  cv_auc={gs.best_score_:.4f}")

    return gs.best_params_, gs.best_score_, grid_table


def train_rf(drivers, lulc_2000, lulc_2020, rf_params=None):
    """Train a cross-sectional RF urban-suitability model (FLUS framing).

    The target is the *2020 urban configuration* (urban vs non-urban land),
    modelled as a function of contemporaneous spatial drivers, rather than the
    2000->2020 transition. Predicting a historical change event with predictors
    whose vintage postdates the change introduces temporal data leakage (the
    post-conversion landscape state trivially separates transitioned pixels).
    By learning the static suitability of the 2020 land configuration from
    drivers that do not postdate the 2020 reference state, the surface is a
    valid suitability prior for forward CA allocation.

    rf_params: dict of {n_estimators, max_depth, min_samples_split}. If None,
    a grid search is run (see grid_search_rf) and its winning configuration is
    used - the production model is always the grid-search optimum, never a
    hardcoded default.
    """
    print("Training RF suitability model (2020 cross-sectional urban state)...")

    # Training/prediction domain: all land pixels, excluding permanent water and
    # nodata. Currently-urban pixels are valid positive examples; the CA later
    # zeroes them out during allocation and grows new urban via the neighbourhood
    # enrichment term.
    valid_mask = (lulc_2020 > 0) & (lulc_2020 != 4)

    # Target: urban (1) vs non-urban (0) in the 2020 reference state.
    y = (lulc_2020 == 1).astype(np.uint8)

    X_train = drivers[valid_mask]
    y_train = y[valid_mask]

    # Subsample for training speed (e.g. 50,000 points); seeded for
    # reproducibility (a fixed sample id set every run).
    if len(y_train) > 50000:
        idx = RNG.choice(len(y_train), 50000, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    if rf_params is None:
        rf_params, cv_auc, _ = grid_search_rf(X_train, y_train)
    else:
        cv_auc = None

    rf = RandomForestClassifier(n_jobs=-1, random_state=42, **rf_params)
    rf.fit(X_train, y_train)

    # Predict development probability for all valid pixels
    print("  Predicting suitability surface...")
    prob_surface = np.zeros(lulc_2020.shape, dtype=np.float32)

    # Predict in chunks to save memory
    chunk_size = 1000000
    valid_indices = np.where(valid_mask)
    X_valid = drivers[valid_mask]

    probs = np.zeros(len(X_valid), dtype=np.float32)
    for i in range(0, len(X_valid), chunk_size):
        probs[i:i+chunk_size] = rf.predict_proba(X_valid[i:i+chunk_size])[:, 1]

    prob_surface[valid_indices] = probs

    with open(MODELS / "rf_flus_model.pkl", "wb") as f:
        pickle.dump(rf, f)

    import json
    with open(MODELS / "rf_best_params.json", "w") as f:
        json.dump({"n_estimators": rf.n_estimators,
                   "max_depth": rf.max_depth,
                   "min_samples_split": rf.min_samples_split,
                   "cv_auc": cv_auc}, f, indent=2)

    return prob_surface

def run_ca_simulation(lulc_base, suitability, demand, target_demand, scenario_mask):
    """
    Run Cellular Automata with Moore neighborhood and roulette wheel selection.
    """
    print(f"Running CA Simulation (Target: +{target_demand - demand:,} pixels)...")
    
    current_state = lulc_base.copy()
    current_urban = (current_state == 1).astype(np.float32)
    
    # Moore neighborhood kernel (3x3)
    from scipy.ndimage import convolve
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.float32)
                       
    pixels_needed = target_demand - demand
    iteration = 1
    
    # Pre-calculate base suitability where mask is valid
    base_suit = suitability * scenario_mask
    
    while pixels_needed > 0:
        # 1. Neighborhood enrichment
        neighborhood_density = convolve(current_urban, kernel, mode='constant') / 8.0
        
        # 2. Combined probability (add small base prob to allow sprawl jumps beyond immediate neighbors)
        combined_prob = base_suit * (neighborhood_density + 0.05)
        
        # Zero out already urban
        combined_prob[current_urban == 1] = 0
        
        # 3. Roulette wheel selection / stochastic perturbation
        alpha = 1.0 + (-np.log(np.random.random(combined_prob.shape) + 1e-10)) * 0.1
        final_prob = combined_prob * alpha
        
        # Allocate top N pixels
        alloc_size = min(pixels_needed, 5000) # Add in batches
        
        # To avoid sorting massive arrays, we only look at non-zero probabilities
        valid_probs = final_prob[final_prob > 0]
        if len(valid_probs) < alloc_size:
            alloc_size = len(valid_probs)
            if alloc_size == 0:
                print("  CA Simulation stalled. No valid pixels to allocate.")
                break
                
        threshold = np.sort(valid_probs)[-alloc_size]
        new_urban = (final_prob >= threshold) & (final_prob > 0)
        
        current_urban[new_urban] = 1
        current_state[new_urban] = 1
        
        added = np.sum(new_urban)
        pixels_needed -= added
        if iteration % 5 == 0:
            print(f"  Iteration {iteration}: Allocated {added:,} px. Remaining: {pixels_needed:,}")
        iteration += 1
        
    return current_state


def build_scenario_masks(lulc_2020):
    """BAU (physical exclusions only) and ECO (+ agriculture/forest/riparian
    penalties) scenario constraint masks. Factored out so any script that
    needs to rerun the CA under a different demand target (see
    p2_14_markov_demand_rerun.py) uses the identical scenario definitions as
    the main 2050 simulation.
    """
    bau_mask = (lulc_2020 != 4).astype(np.float32)

    eco_mask = bau_mask.copy()
    eco_mask[lulc_2020 == 2] = 0.1  # Strongly penalize agriculture
    eco_mask[lulc_2020 == 3] = 0.1  # Strongly penalize forest

    riv_path = DRIVERS_DIR / "driver_river_dist.tif"
    if riv_path.exists():
        with rasterio.open(riv_path) as r:
            riv = r.read(1)
        eco_mask[riv < 500] *= 0.1  # Strongly penalize riparian
    return bau_mask, eco_mask


def main():
    if not DRIVERS_DIR.exists():
        print("Drivers not aligned. Please run p2_03_spatial_drivers.py first.")
        return

    # Load LULC

    with rasterio.open(PROCESSED / "lulc_simplified_2000.tif") as src:
        lulc_2000 = src.read(1)
        meta = src.meta.copy()
    with rasterio.open(PROCESSED / "lulc_simplified_2020.tif") as src:
        lulc_2020 = src.read(1)

    # 1. Load Drivers & Demand
    drivers, _ = load_drivers()
    target_demand, current_demand = calculate_2050_demand(lulc_2000, lulc_2020)

    # 2. Train RF
    suitability = train_rf(drivers, lulc_2000, lulc_2020)

    # 3. Create Scenarios Masks
    bau_mask, eco_mask = build_scenario_masks(lulc_2020)

    # 4. Simulate
    print("\n--- BAU Scenario 2050 ---")
    sim_bau = run_ca_simulation(lulc_2020, suitability, current_demand, target_demand, bau_mask)
    with rasterio.open(PROCESSED / "sim_2050_bau.tif", "w", **meta) as out:
        out.write(sim_bau, 1)
        
    print("\n--- ECO Scenario 2050 ---")
    sim_eco = run_ca_simulation(lulc_2020, suitability, current_demand, target_demand, eco_mask)
    with rasterio.open(PROCESSED / "sim_2050_eco.tif", "w", **meta) as out:
        out.write(sim_eco, 1)
        
    print("\nSimulations completed successfully.")

if __name__ == "__main__":
    main()
