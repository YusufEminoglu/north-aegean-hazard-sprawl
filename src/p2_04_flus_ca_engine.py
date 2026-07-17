"""
Paper 2 — RF/FLUS-style cellular-automata engine.

The headline 2050 demand follows a transparent linear extrapolation of the
2000–2020 urban-pixel increment. A Markov-chain pathway is available as a
sensitivity check. Both BAU and ECO scenarios use the same demand.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.linalg import fractional_matrix_power
from scipy.ndimage import convolve
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "03_processed" / "paper2"
DRIVERS_DIR = PROCESSED / "drivers"
MODELS = PROJECT_ROOT / "data" / "04_models" / "paper2"
TABLES = PROJECT_ROOT / "outputs" / "tables" / "paper2"
MODELS.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42

DRIVER_NAMES = [
    "slope", "elevation", "road_dist", "river_dist", "water_dist",
    "pop_density", "ntl", "ndvi", "clay", "coast_dist", "urban_dist",
]


def load_drivers():
    """Load aligned spatial drivers and fail if their grids differ."""
    print("Loading spatial drivers...")
    stack = []
    reference = None
    for name in DRIVER_NAMES:
        path = DRIVERS_DIR / f"driver_{name}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Missing driver: {path}")
        with rasterio.open(path) as src:
            signature = (src.height, src.width, src.crs, src.transform)
            if reference is None:
                reference = signature
                meta = src.meta.copy()
            elif signature != reference:
                raise ValueError(f"Driver grid does not match reference: {path}")
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            stack.append(np.nan_to_num(arr, nan=0.0))
    return np.stack(stack, axis=-1), meta


def linear_demand(lulc_2000, lulc_2020):
    """Extrapolate the 2000–2020 urban increment over the next 30 years."""
    urban_2000 = int(np.sum(lulc_2000 == 1))
    urban_2020 = int(np.sum(lulc_2020 == 1))
    annual_increment = (urban_2020 - urban_2000) / 20.0
    return int(round(urban_2020 + annual_increment * 30.0))


def markov_demand(lulc_2000, lulc_2020):
    """Project class totals with a 20-year transition matrix raised to 1.5."""
    classes = np.array(sorted(
        set(np.unique(lulc_2000)).intersection(np.unique(lulc_2020)) - {0}
    ), dtype=int)
    valid = np.isin(lulc_2000, classes) & np.isin(lulc_2020, classes)
    matrix = np.zeros((len(classes), len(classes)), dtype=np.float64)
    for i, source_class in enumerate(classes):
        source = valid & (lulc_2000 == source_class)
        denominator = source.sum()
        if denominator:
            for j, target_class in enumerate(classes):
                matrix[i, j] = np.sum(source & (lulc_2020 == target_class)) / denominator
        else:
            matrix[i, i] = 1.0
    operator_30y = np.real_if_close(fractional_matrix_power(matrix, 1.5)).real
    operator_30y = np.clip(operator_30y, 0.0, None)
    operator_30y /= operator_30y.sum(axis=1, keepdims=True)
    counts_2020 = np.array([np.sum(lulc_2020 == value) for value in classes])
    projected = counts_2020 @ operator_30y
    urban_index = int(np.where(classes == 1)[0][0])
    return int(round(projected[urban_index]))


def population_demand(lulc_2000, lulc_2020):
    """Optional population-scaled demand from a user-supplied district CSV."""
    del lulc_2000
    csv_path = TABLES / "p2_worldpop_ilce_2020_2050.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            "Population pathway requested but district projection CSV is missing: "
            f"{csv_path}"
        )
    table = pd.read_csv(csv_path)
    required = {"Pop_2020", "Pop_2050"}
    if not required.issubset(table.columns):
        raise ValueError(f"{csv_path.name} must contain {sorted(required)}")
    pop_2020 = float(table["Pop_2020"].sum())
    if pop_2020 <= 0:
        raise ValueError("Pop_2020 total must be positive")
    return int(round(np.sum(lulc_2020 == 1) * table["Pop_2050"].sum() / pop_2020))


def calculate_2050_demand(lulc_2000, lulc_2020, method=None):
    """Return target and current urban pixel counts for the chosen pathway."""
    method = (method or os.environ.get("P2_DEMAND_METHOD", "linear")).lower()
    functions = {
        "linear": linear_demand,
        "markov": markov_demand,
        "population": population_demand,
    }
    if method not in functions:
        raise ValueError(f"Unknown P2_DEMAND_METHOD={method!r}; choose {sorted(functions)}")
    current = int(np.sum(lulc_2020 == 1))
    target = functions[method](lulc_2000, lulc_2020)
    pixel_area_ha = 0.09
    print(f"2050 demand pathway: {method}")
    print(f"  Urban 2020: {current:,} px ({current * pixel_area_ha:,.1f} ha)")
    print(f"  Target 2050: {target:,} px ({target * pixel_area_ha:,.1f} ha)")
    return target, current


def train_rf(drivers, lulc_2000, lulc_2020):
    """Fit the selected R1 RF configuration to the 2020 urban state."""
    del lulc_2000
    valid = (lulc_2020 > 0) & (lulc_2020 != 4)
    target = (lulc_2020 == 1).astype(np.uint8)
    features = drivers[valid]
    labels = target[valid]
    rng = np.random.default_rng(RANDOM_SEED)
    if len(labels) > 50_000:
        sample = rng.choice(len(labels), 50_000, replace=False)
        features, labels = features[sample], labels[sample]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        oob_score=True,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(features, labels)

    probability = np.zeros(lulc_2020.shape, dtype=np.float32)
    valid_features = drivers[valid]
    predicted = np.empty(len(valid_features), dtype=np.float32)
    for start in range(0, len(valid_features), 1_000_000):
        stop = start + 1_000_000
        predicted[start:stop] = model.predict_proba(valid_features[start:stop])[:, 1]
    probability[np.where(valid)] = predicted
    with open(MODELS / "rf_flus_model.pkl", "wb") as stream:
        pickle.dump(model, stream)
    return probability


def run_ca_simulation(
    lulc_base, suitability, demand, target_demand, scenario_mask,
    random_seed=RANDOM_SEED,
):
    """Allocate exactly the requested demand with seeded stochastic perturbation."""
    current_state = lulc_base.copy()
    current_urban = current_state == 1
    pixels_needed = max(0, int(target_demand - demand))
    rng = np.random.default_rng(random_seed)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    base_suitability = np.nan_to_num(suitability * scenario_mask, nan=0.0)

    iteration = 0
    while pixels_needed:
        iteration += 1
        density = convolve(current_urban.astype(np.float32), kernel, mode="constant") / 8.0
        probability = base_suitability * (density + 0.05)
        probability[current_urban] = 0.0
        perturbation = 1.0 - np.log(np.maximum(rng.random(probability.shape), 1e-12)) * 0.1
        scores = probability * perturbation
        candidates = np.flatnonzero(scores.ravel() > 0)
        allocation = min(pixels_needed, 5_000, len(candidates))
        if allocation == 0:
            raise RuntimeError(
                f"CA stalled with {pixels_needed:,} pixels unallocated; "
                "check the scenario mask and suitability surface"
            )
        candidate_scores = scores.ravel()[candidates]
        chosen_local = np.argpartition(candidate_scores, -allocation)[-allocation:]
        chosen = candidates[chosen_local]
        current_urban.ravel()[chosen] = True
        current_state.ravel()[chosen] = 1
        pixels_needed -= allocation
        if iteration % 5 == 0 or not pixels_needed:
            print(f"  Iteration {iteration}: {pixels_needed:,} pixels remaining")
    return current_state


def _write_raster(path, array, meta):
    output_meta = meta.copy()
    output_meta.update(count=1, dtype=str(array.dtype), compress="deflate")
    with rasterio.open(path, "w", **output_meta) as destination:
        destination.write(array, 1)


def main():
    drivers, _ = load_drivers()
    with rasterio.open(PROCESSED / "lulc_simplified_2000.tif") as source:
        lulc_2000 = source.read(1)
    with rasterio.open(PROCESSED / "lulc_simplified_2020.tif") as source:
        lulc_2020 = source.read(1)
        meta = source.meta.copy()

    target_demand, current_demand = calculate_2050_demand(lulc_2000, lulc_2020)
    suitability = train_rf(drivers, lulc_2000, lulc_2020)
    bau_mask = (lulc_2020 != 4).astype(np.float32)
    eco_mask = bau_mask.copy()
    eco_mask[np.isin(lulc_2020, [2, 3])] = 0.1
    river_path = DRIVERS_DIR / "driver_river_dist.tif"
    if river_path.exists():
        with rasterio.open(river_path) as source:
            river_distance = source.read(1)
        eco_mask[river_distance < 500] *= 0.1

    print("BAU scenario")
    bau = run_ca_simulation(
        lulc_2020, suitability, current_demand, target_demand, bau_mask, RANDOM_SEED
    )
    _write_raster(PROCESSED / "sim_2050_bau.tif", bau, meta)
    print("ECO scenario")
    eco = run_ca_simulation(
        lulc_2020, suitability, current_demand, target_demand, eco_mask, RANDOM_SEED
    )
    _write_raster(PROCESSED / "sim_2050_eco.tif", eco, meta)
    print("Simulations completed.")


if __name__ == "__main__":
    main()
