"""Random Forest hyperparameter grid search (R4, Reviewer 2, reproducibility).

Re-runs the stratified 5-fold grid search that selects the production RF
configuration used by ``p2_04_flus_ca_engine.train_rf`` and reported in
Supplementary Table S6 of the manuscript:

    n_estimators      in {50, 100, 200}
    max_depth         in {10, 20, None (unbounded)}
    min_samples_split in {2, 5, 10}        -> 27 configurations

Protocol (identical to the manuscript):
  * target   : observed 2020 urban (1) vs non-urban (0) state;
  * domain   : every 2020 land pixel except permanent water (class 4) and nodata;
  * sample   : 50,000 pixels drawn uniformly at random, seed 42, no class
               balancing (natural urban share ~2.5%);
  * features : the 11 spatial drivers in ``p2_04_flus_ca_engine.DRIVER_NAMES``;
  * CV       : StratifiedKFold(n_splits=5, shuffle=True, random_state=42);
  * metric   : ROC-AUC; forest random_state=42.

Expected outcome: every configuration saturates at CV-AUC = 1.0000 (variation
below 1e-15), because distance-to-existing-urban nearly reproduces the 2020
urban mask. The grid therefore cannot discriminate among the 27 settings;
the production model keeps the smallest forest (50 trees) on parsimony
grounds. The full ranked grid is written to
``outputs/tables/paper2/rf_hyperparameter_grid.csv`` and is the source of
Supplementary Table S6.

Usage:
  python src/p2_04b_rf_grid_search.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

HERE = Path(__file__).resolve().parent

RF_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
}
SAMPLE_SIZE = 50_000
SEED = 42


def _load_engine():
    spec = importlib.util.spec_from_file_location("p2_04", HERE / "p2_04_flus_ca_engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    engine = _load_engine()
    drivers, _ = engine.load_drivers()
    with rasterio.open(engine.PROCESSED / "lulc_simplified_2020.tif") as src:
        lulc_2020 = src.read(1)

    valid = (lulc_2020 > 0) & (lulc_2020 != 4)
    labels = (lulc_2020 == 1).astype(np.uint8)[valid]
    features = drivers[valid]
    rng = np.random.default_rng(SEED)
    if len(labels) > SAMPLE_SIZE:
        sample = rng.choice(len(labels), SAMPLE_SIZE, replace=False)
        features, labels = features[sample], labels[sample]
    print(f"Training sample: {len(labels):,} pixels, urban share {labels.mean() * 100:.2f}%")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    base = RandomForestClassifier(n_jobs=1, random_state=SEED)
    search = GridSearchCV(base, RF_GRID, scoring="roc_auc", cv=cv, n_jobs=-1, refit=False)
    print("Grid-searching RF hyperparameters (5-fold CV, 27 configurations)...")
    search.fit(features, labels)

    results = pd.DataFrame(search.cv_results_)
    table = results[[
        "param_n_estimators", "param_max_depth", "param_min_samples_split",
        "mean_test_score", "std_test_score", "rank_test_score",
    ]].sort_values(["rank_test_score", "param_n_estimators"]).reset_index(drop=True)
    table.columns = [
        "n_estimators", "max_depth", "min_samples_split",
        "cv_auc_mean", "cv_auc_std", "rank",
    ]
    out_path = engine.TABLES / "rf_hyperparameter_grid.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    spread = float(table["cv_auc_mean"].max() - table["cv_auc_mean"].min())
    tied = int((table["rank"] == table["rank"].min()).sum())
    print(f"Saved {len(table)} configurations -> {out_path}")
    print(f"Best CV-AUC {table['cv_auc_mean'].max():.4f}; spread across grid {spread:.2e}; "
          f"{tied} of {len(table)} configurations share rank 1")
    print(f"rf_cv_auc,{table['cv_auc_mean'].max():.4f}")


if __name__ == "__main__":
    main()
