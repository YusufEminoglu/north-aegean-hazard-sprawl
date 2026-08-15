"""
Paper 2 — model validation and diagnostics.

Reports RF spatial-block diagnostics, a 2000–2020 CA hindcast, and
threshold-independent flood validation against observed GFD inundation.
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import jenkspy
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "03_processed" / "paper2"
INTERIM = ROOT / "data" / "02_interim" / "paper2"
MODELS = ROOT / "data" / "04_models" / "paper2"
RNG = np.random.default_rng(42)

spec = importlib.util.spec_from_file_location(
    "p2_04", Path(__file__).with_name("p2_04_flus_ca_engine.py")
)
p2_04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2_04)


def read_raster(path):
    with rasterio.open(path) as source:
        array = source.read(1).astype(np.float32)
        if source.nodata is not None:
            array[array == source.nodata] = np.nan
        return array, source.meta.copy()


def align_to(path, reference_meta, resampling=Resampling.nearest):
    destination = np.full(
        (reference_meta["height"], reference_meta["width"]), np.nan, np.float32
    )
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=reference_meta["transform"],
            dst_crs=reference_meta["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return destination


def rf_diagnostics():
    """Eight-neighbour Moran I and four-quadrant spatial hold-out AUC."""
    print("[A] RF residual diagnostics")
    drivers, _ = p2_04.load_drivers()
    lulc_2020, _ = read_raster(PROCESSED / "lulc_simplified_2020.tif")
    valid = np.isfinite(lulc_2020) & (lulc_2020 > 0) & (lulc_2020 != 4)
    target = (lulc_2020 == 1).astype(np.uint8)
    rows, columns = np.where(valid)
    features = drivers[valid]
    labels = target[valid]

    with open(MODELS / "rf_flus_model.pkl", "rb") as stream:
        model = pickle.load(stream)
    residuals = labels.astype(np.float32) - model.predict_proba(features)[:, 1]

    sample_size = min(50_000, len(rows))
    sample = RNG.choice(len(rows), sample_size, replace=False)
    coordinates = np.column_stack([columns[sample], rows[sample]]).astype(float)
    centred = residuals[sample] - residuals[sample].mean()
    neighbours = cKDTree(coordinates).query(coordinates, k=9)[1][:, 1:]
    lag = centred[neighbours].mean(axis=1)
    moran = np.sum(centred * lag) / np.sum(centred * centred)

    row_mid = (rows.min() + rows.max()) / 2
    col_mid = (columns.min() + columns.max()) / 2
    quadrants = (rows > row_mid).astype(int) * 2 + (columns > col_mid).astype(int)
    aucs = []
    for quadrant in range(4):
        train = quadrants != quadrant
        test = quadrants == quadrant
        train_indices = np.flatnonzero(train)
        if len(train_indices) > 50_000:
            train_indices = RNG.choice(train_indices, 50_000, replace=False)
        fold_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        fold_model.fit(features[train_indices], labels[train_indices])
        aucs.append(
            roc_auc_score(labels[test], fold_model.predict_proba(features[test])[:, 1])
        )

    print(f"rf_resid_moran_knn8,{moran:.3f}")
    print(f"rf_spatial_block_auc_min,{min(aucs):.3f}")
    print(f"rf_spatial_block_auc_mean,{np.mean(aucs):.3f}")
    print(f"rf_spatial_block_auc_max,{max(aucs):.3f}")


def _pontius_stats(domain, observed_change, simulated_change):
    hits = np.sum(observed_change & simulated_change)
    misses = np.sum(observed_change & ~simulated_change)
    false_alarms = np.sum(~observed_change & simulated_change & domain)
    correct_persistence = np.sum(~observed_change & ~simulated_change & domain)
    fom = hits / (hits + misses + false_alarms)
    oa = (hits + correct_persistence) / domain.sum()
    kappa = cohen_kappa_score(
        observed_change[domain].astype(np.uint8),
        simulated_change[domain].astype(np.uint8),
    )
    n_dom = domain.sum()
    n_obs_change = hits + misses
    n_sim_change = hits + false_alarms
    quantity_dis = abs(n_sim_change - n_obs_change) / n_dom
    total_dis = (misses + false_alarms) / n_dom
    allocation_dis = max(total_dis - quantity_dis, 0.0)
    return dict(fom=fom, oa=oa, kappa=kappa, quantity_dis=quantity_dis,
                allocation_dis=allocation_dis)


def ca_hindcast():
    """Evaluate change allocation from the 2000 state to observed 2020.

    Reports two variants. The RF suitability surface used for the 2050
    projection is trained on 2020-vintage drivers, including distance to the
    *2020* urban footprint; reusing that surface unmodified to hindcast
    2000->2020 growth lets the model see where the 2020 urban edge already is
    before "predicting" the very growth that produced that edge - an
    endpoint-information leak, not a genuine out-of-sample test. The
    "endpoint-informed" variant below reproduces that (kept for transparency,
    not treated as the primary validation result); the "leakage-free" variant
    substitutes a distance-to-urban layer computed from the *2000*
    classification alone for this one run, holding the trained suitability
    model and every other (time-invariant) driver fixed.
    """
    print("[B] CA hindcast 2000-2020")
    drivers, _ = p2_04.load_drivers()
    lulc_2000, _ = read_raster(PROCESSED / "lulc_simplified_2000.tif")
    lulc_2020, _ = read_raster(PROCESSED / "lulc_simplified_2020.tif")

    with open(MODELS / "rf_flus_model.pkl", "rb") as stream:
        model = pickle.load(stream)
    valid = np.isfinite(lulc_2020) & (lulc_2020 > 0) & (lulc_2020 != 4)

    urban_2000 = int(np.sum(lulc_2000 == 1))
    urban_2020 = int(np.sum(lulc_2020 == 1))
    mask = (np.isfinite(lulc_2000) & (lulc_2000 != 4)).astype(np.float32)

    domain = (
        np.isfinite(lulc_2000)
        & np.isfinite(lulc_2020)
        & (lulc_2000 != 1)
        & (lulc_2000 != 4)
        & (lulc_2020 != 4)
    )
    observed_change = domain & (lulc_2020 == 1)

    # --- (i) endpoint-informed: 2020-vintage drivers throughout ---
    suitability_ei = np.zeros(lulc_2020.shape, np.float32)
    suitability_ei[valid] = model.predict_proba(drivers[valid])[:, 1]
    simulated_ei = p2_04.run_ca_simulation(
        lulc_2000, suitability_ei, urban_2000, urban_2020, mask, random_seed=42
    )
    stats_ei = _pontius_stats(domain, observed_change, domain & (simulated_ei == 1))

    # --- (ii) leakage-free: swap urban_dist for its 2000-vintage version ---
    urban_dist_idx = p2_04.DRIVER_NAMES.index("urban_dist")
    urban_dist_2000, _ = read_raster(PROCESSED / "drivers" / "driver_urban_dist_2000.tif")
    drivers_lf = drivers.copy()
    drivers_lf[..., urban_dist_idx] = np.nan_to_num(urban_dist_2000, nan=0.0)
    suitability_lf = np.zeros(lulc_2020.shape, np.float32)
    suitability_lf[valid] = model.predict_proba(drivers_lf[valid])[:, 1]
    simulated_lf = p2_04.run_ca_simulation(
        lulc_2000, suitability_lf, urban_2000, urban_2020, mask, random_seed=42
    )
    stats_lf = _pontius_stats(domain, observed_change, domain & (simulated_lf == 1))

    print("  -- endpoint-informed (2020-vintage urban_dist throughout) --")
    print(f"flus_fom_endpoint_informed,{100 * stats_ei['fom']:.1f}")
    print(f"flus_oa_endpoint_informed,{100 * stats_ei['oa']:.1f}")
    print(f"flus_kappa_endpoint_informed,{stats_ei['kappa']:.3f}")
    print(f"flus_quantity_dis_endpoint_informed,{100 * stats_ei['quantity_dis']:.1f}")
    print(f"flus_allocation_dis_endpoint_informed,{100 * stats_ei['allocation_dis']:.1f}")

    print("  -- leakage-free (2000-vintage urban_dist) - PRIMARY RESULT --")
    print(f"flus_fom,{100 * stats_lf['fom']:.1f}")
    print(f"flus_oa,{100 * stats_lf['oa']:.1f}")
    print(f"flus_kappa,{stats_lf['kappa']:.3f}")
    print(f"flus_quantity_dis,{100 * stats_lf['quantity_dis']:.1f}")
    print(f"flus_allocation_dis,{100 * stats_lf['allocation_dis']:.1f}")


def flood_validation():
    """Compute AUC, maximum-CSI statistics, and physical monotonicity checks."""
    print("[C] Flood validation against GFD")
    flood, flood_meta = read_raster(PROCESSED / "flood_hazard.tif")
    gfd = align_to(
        INTERIM / "p2_gfd_flood_count_clipped.tif",
        flood_meta,
        Resampling.nearest,
    )
    valid = np.isfinite(flood) & np.isfinite(gfd)
    labels = (gfd[valid] > 0).astype(np.uint8)
    scores = flood[valid]
    if len(np.unique(labels)) != 2:
        raise ValueError("GFD reference must contain observed and unobserved pixels")

    auc = roc_auc_score(labels, scores)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    positives = labels.sum()
    negatives = len(labels) - positives
    hits = true_positive_rate * positives
    false_alarms = false_positive_rate * negatives
    misses = positives - hits
    csi = np.divide(
        hits,
        hits + misses + false_alarms,
        out=np.zeros_like(hits),
        where=(hits + misses + false_alarms) > 0,
    )
    best = int(np.nanargmax(csi))
    pod = true_positive_rate[best]
    far = false_alarms[best] / (hits[best] + false_alarms[best])

    breaks = jenkspy.jenks_breaks(scores, n_classes=5)
    classes = np.digitize(scores, breaks[1:-1], right=True) + 1
    class_rates = {
        class_id: 100 * labels[classes == class_id].mean()
        for class_id in range(1, 6)
    }
    enrichment = class_rates[5] / class_rates[1]

    hand = align_to(INTERIM / "p2_merit_hand_clipped.tif", flood_meta)
    hand_valid = valid & np.isfinite(hand)
    hand_low = 100 * (gfd[hand_valid & (hand < 2)] > 0).mean()
    hand_high = 100 * (gfd[hand_valid & (hand > 20)] > 0).mean()

    print(f"flood_auc,{auc:.3f}")
    print(f"flood_max_csi_threshold,{thresholds[best]:.6f}")
    print(f"flood_pod,{pod:.3f}")
    print(f"flood_far,{far:.3f}")
    print(f"flood_csi,{csi[best]:.3f}")
    print(f"gfd_prevalence_pct,{100 * labels.mean():.3f}")
    print(f"gfd_class1_pct,{class_rates[1]:.3f}")
    print(f"gfd_class5_pct,{class_rates[5]:.3f}")
    print(f"gfd_enrichment,{enrichment:.1f}")
    print(f"gfd_hand_lt2_pct,{hand_low:.3f}")
    print(f"gfd_hand_gt20_pct,{hand_high:.3f}")


if __name__ == "__main__":
    rf_diagnostics()
    ca_hindcast()
    flood_validation()
