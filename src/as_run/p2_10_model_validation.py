"""
Paper 2 — Model validation & diagnostics
========================================
Computes the validation statistics reported in the manuscript.

  A. RF residual diagnostics
     - 8-nearest-neighbour Moran's I on a 50k residual sub-sample
     - quadrant spatial-block hold-out AUC (min/mean/max over 4 folds)
  B. CA hindcast (2000 -> 2020) allocation accuracy
     - endpoint-informed reconstruction (original approach, kept for
       transparency): suitability from the 2020-conditioned RF model,
       including urban_dist computed from the 2020 endpoint
     - leakage-free hindcast (primary, reported result): the same RF model
       applied to a driver stack where urban_dist is swapped for a
       2000-vintage version (driver_urban_dist_2000.tif), so the hindcast no
       longer conditions on the 2020 urban footprint it is being scored
       against (fixes the leakage raised in Reviewer 2, Aug review, point 2)
     - Overall Accuracy, Cohen's Kappa, Figure of Merit (Pontius 2008),
       Quantity/Allocation Disagreement (Pontius & Millones 2011)
  C. Driver ablation: RF with vs without urban_dist (Reviewer 2, point 4)
  D. Flood hazard validation
     - continuous ROC-AUC + monotonic class-enrichment vs the Global Flood
       Database (threshold-independent, replaces the single-cut POD/FAR/CSI)
     - independent validation against DSI Kuzey Ege Havzasi Master Plan
       historical flood records (127 events) and delineated flood-prone
       areas (25 polygons) - see p2_11_dsi_masterplan_ingest.py

All values are printed in `key,value` form for transcription into mydata.dat.

Usage:
  .venv/Scripts/python.exe src/paper2_hazard_sprawl/p2_10_model_validation.py
"""

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "03_processed" / "paper2"
INTERIM = ROOT / "data" / "02_interim" / "paper2"
MODELS = ROOT / "data" / "04_models" / "paper2"
DSI = INTERIM / "dsi_masterplan"

# import p2_04 (load_drivers, run_ca_simulation, DRIVER_NAMES) without running main
spec = importlib.util.spec_from_file_location(
    "p2_04", Path(__file__).with_name("p2_04_flus_ca_engine.py"))
p2_04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2_04)

RNG = np.random.default_rng(42)


def _read(path):
    with rasterio.open(path) as s:
        return s.read(1), s.meta.copy()


# ════════════════════════════════════════════════════════════════════════════
# A. RF residual diagnostics
# ════════════════════════════════════════════════════════════════════════════
def rf_diagnostics():
    print("\n[A] RF residual diagnostics ...")
    drivers, _ = p2_04.load_drivers()
    lulc20, _ = _read(PROC / "lulc_simplified_2020.tif")

    valid = (lulc20 > 0) & (lulc20 != 4)
    y = (lulc20 == 1).astype(np.uint8)
    rows, cols = np.where(valid)
    X = drivers[valid]
    yv = y[valid]

    with open(MODELS / "rf_flus_model.pkl", "rb") as f:
        rf = pickle.load(f)
    phat = rf.predict_proba(X)[:, 1]
    resid = yv.astype(np.float32) - phat

    n_sub = min(50000, len(rows))
    idx = RNG.choice(len(rows), n_sub, replace=False)
    coords = np.column_stack([cols[idx], rows[idx]]).astype(np.float64)
    z = resid[idx] - resid[idx].mean()
    tree = cKDTree(coords)
    _, nn = tree.query(coords, k=9)
    nn = nn[:, 1:]
    lag = z[nn].mean(axis=1)
    moran = np.sum(z * lag) / np.sum(z * z)

    rmid = (rows.min() + rows.max()) / 2
    cmid = (cols.min() + cols.max()) / 2
    quad = ((rows > rmid).astype(int) * 2 + (cols > cmid).astype(int))
    aucs = []
    for q in range(4):
        tr = quad != q
        te = quad == q
        if yv[te].sum() < 50 or yv[tr].sum() < 50:
            continue
        tr_idx = np.where(tr)[0]
        if len(tr_idx) > 50000:
            tr_idx = RNG.choice(tr_idx, 50000, replace=False)
        m = RandomForestClassifier(n_estimators=50, max_depth=10,
                                    min_samples_split=2, n_jobs=-1, random_state=42)
        m.fit(X[tr_idx], yv[tr_idx])
        p_te = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(yv[te], p_te))
    aucs = np.array(aucs)

    print(f"rf_resid_moran_knn8,{moran:.3f}")
    print(f"rf_spatial_block_auc_min,{aucs.min():.3f}")
    print(f"rf_spatial_block_auc_mean,{aucs.mean():.3f}")
    print(f"rf_spatial_block_auc_max,{aucs.max():.3f}")


# ════════════════════════════════════════════════════════════════════════════
# B. CA hindcast 2000 -> 2020 (endpoint-informed + leakage-free)
# ════════════════════════════════════════════════════════════════════════════
def _pontius_stats(obs_change, sim_change, dom):
    B = int(np.sum(obs_change & sim_change))
    A = int(np.sum(obs_change & ~sim_change))
    C = int(np.sum(~obs_change & sim_change & dom))
    D = int(np.sum(~obs_change & ~sim_change & dom))
    fom = 100 * B / (A + B + C) if (A + B + C) else float("nan")
    oa = 100 * (B + D) / (A + B + C + D)
    yt = obs_change[dom].astype(np.uint8)
    yp = sim_change[dom].astype(np.uint8)
    kappa = cohen_kappa_score(yt, yp)

    # Pontius & Millones (2011) Quantity / Allocation Disagreement, as a
    # fraction of the domain: Quantity = |predicted-change count - observed-
    # change count| relative to domain; Allocation = the remaining
    # disagreement (misses + false alarms not explained by the quantity gap).
    n_obs_change = A + B
    n_sim_change = B + C
    n_dom = A + B + C + D
    quantity_dis = abs(n_sim_change - n_obs_change) / n_dom * 100
    total_dis = (A + C) / n_dom * 100
    allocation_dis = max(total_dis - quantity_dis, 0.0)
    return dict(A=A, B=B, C=C, D=D, fom=fom, oa=oa, kappa=kappa,
                quantity_dis=quantity_dis, allocation_dis=allocation_dis)


def ca_hindcast():
    print("\n[B] CA hindcast 2000 -> 2020 ...")
    lulc00, _ = _read(PROC / "lulc_simplified_2000.tif")
    lulc20, _ = _read(PROC / "lulc_simplified_2020.tif")

    with open(MODELS / "rf_flus_model.pkl", "rb") as f:
        rf = pickle.load(f)
    valid = (lulc20 > 0) & (lulc20 != 4)
    dom = (lulc00 != 1) & (lulc00 != 4) & (lulc20 != 4)
    obs_change = dom & (lulc20 == 1)

    u00 = int(np.sum(lulc00 == 1))
    u20 = int(np.sum(lulc20 == 1))
    mask = (lulc00 != 4).astype(np.float32)

    # --- (i) endpoint-informed reconstruction: 2020-vintage drivers throughout
    drivers_2020, _ = p2_04.load_drivers()
    suit_endpoint = np.zeros(lulc20.shape, np.float32)
    suit_endpoint[valid] = rf.predict_proba(drivers_2020[valid])[:, 1]
    sim_ei = p2_04.run_ca_simulation(lulc00, suit_endpoint, u00, u20, mask)
    sim_change_ei = dom & (sim_ei == 1)
    stats_ei = _pontius_stats(obs_change, sim_change_ei, dom)

    # --- (ii) leakage-free: swap urban_dist for its 2000-vintage version
    urban_dist_idx = p2_04.DRIVER_NAMES.index("urban_dist")
    with rasterio.open(PROC / "drivers" / "driver_urban_dist_2000.tif") as src:
        urban_dist_2000 = np.nan_to_num(src.read(1), nan=0.0)
    drivers_2000 = drivers_2020.copy()
    drivers_2000[..., urban_dist_idx] = urban_dist_2000
    suit_lf = np.zeros(lulc20.shape, np.float32)
    suit_lf[valid] = rf.predict_proba(drivers_2000[valid])[:, 1]
    sim_lf = p2_04.run_ca_simulation(lulc00, suit_lf, u00, u20, mask)
    sim_change_lf = dom & (sim_lf == 1)
    stats_lf = _pontius_stats(obs_change, sim_change_lf, dom)

    print("  -- endpoint-informed (2020-vintage urban_dist throughout) --")
    print(f"flus_fom_endpoint_informed,{stats_ei['fom']:.1f}")
    print(f"flus_oa_endpoint_informed,{stats_ei['oa']:.1f}")
    print(f"flus_kappa_endpoint_informed,{stats_ei['kappa']:.3f}")
    print(f"flus_quantity_dis_endpoint_informed,{stats_ei['quantity_dis']:.1f}")
    print(f"flus_allocation_dis_endpoint_informed,{stats_ei['allocation_dis']:.1f}")

    print("  -- leakage-free (2000-vintage urban_dist) — PRIMARY RESULT --")
    print(f"flus_fom,{stats_lf['fom']:.1f}")
    print(f"flus_oa,{stats_lf['oa']:.1f}")
    print(f"flus_kappa,{stats_lf['kappa']:.3f}")
    print(f"flus_quantity_dis,{stats_lf['quantity_dis']:.1f}")
    print(f"flus_allocation_dis,{stats_lf['allocation_dis']:.1f}")


# ════════════════════════════════════════════════════════════════════════════
# C. Driver ablation: with vs without urban_dist
# ════════════════════════════════════════════════════════════════════════════
def driver_ablation():
    print("\n[C] Driver ablation (with vs without urban_dist) ...")
    drivers, _ = p2_04.load_drivers()
    lulc20, _ = _read(PROC / "lulc_simplified_2020.tif")
    valid = (lulc20 > 0) & (lulc20 != 4)
    y = (lulc20 == 1).astype(np.uint8)

    urban_dist_idx = p2_04.DRIVER_NAMES.index("urban_dist")
    keep_idx = [i for i in range(len(p2_04.DRIVER_NAMES)) if i != urban_dist_idx]

    X_full = drivers[valid]
    X_abl = drivers[valid][:, keep_idx]
    y_valid = y[valid]

    idx = RNG.choice(len(y_valid), min(50000, len(y_valid)), replace=False)
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import f1_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for label, X in [("with_urban_dist", X_full[idx]), ("without_urban_dist", X_abl[idx])]:
        yb = y_valid[idx]
        m = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        auc_scores = cross_val_score(m, X, yb, cv=cv, scoring="roc_auc")
        m.fit(X, yb)
        oob_m = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1,
                                        random_state=42, oob_score=True)
        oob_m.fit(X, yb)
        f1 = f1_score(yb, m.predict(X), average="macro")
        tag = "" if label == "with_urban_dist" else "_ablated"
        print(f"  {label}: cv_auc={auc_scores.mean():.4f}  oob_acc={oob_m.oob_score_:.4f}  macro_f1={f1:.4f}")
        print(f"rf_cv_auc{tag},{auc_scores.mean():.4f}")
        print(f"rf_oob_acc{tag},{oob_m.oob_score_:.4f}")
        print(f"rf_macro_f1{tag},{f1:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# D. Flood validation vs Global Flood Database (continuous, threshold-free)
#    + independent validation vs DSI historical floods
# ════════════════════════════════════════════════════════════════════════════
def flood_validation():
    print("\n[D] Flood validation ...")
    flood, fmeta = _read(PROC / "flood_hazard.tif")

    gfd_src = INTERIM / "p2_gfd_flood_count_clipped.tif"
    gfd = np.zeros(flood.shape, np.float32)
    with rasterio.open(gfd_src) as s:
        reproject(
            source=rasterio.band(s, 1), destination=gfd,
            src_transform=s.transform, src_crs=s.crs,
            dst_transform=fmeta["transform"], dst_crs=fmeta["crs"],
            resampling=Resampling.bilinear,
        )

    valid = np.isfinite(flood) & (flood > 0)
    obs = valid & (gfd > 0)

    # --- continuous ROC-AUC (threshold-independent) ---
    auc = roc_auc_score(obs[valid].astype(np.uint8), flood[valid])
    print(f"flood_auc,{auc:.3f}")

    # --- monotonic class enrichment (quintile of continuous score) ---
    q = np.nanpercentile(flood[valid], [20, 40, 60, 80])
    edges = [flood[valid].min()] + list(q) + [flood[valid].max()]
    print("  Class-enrichment (observed-flood prevalence by susceptibility class):")
    for c in range(5):
        lo, hi = edges[c], edges[c + 1]
        cls_mask = valid & (flood >= lo) & (flood <= hi if c == 4 else flood < hi)
        prev = float(obs[cls_mask].sum()) / float(cls_mask.sum()) * 100 if cls_mask.sum() else 0.0
        print(f"    class {c+1}: prevalence={prev:.3f}%  n={int(cls_mask.sum()):,}")
        print(f"flood_enrichment_class{c+1},{prev:.3f}")

    # --- point-optimum categorical stats (for readers who want one number) ---
    best = None
    for thr in np.linspace(flood[valid].min(), flood[valid].max(), 40):
        high = valid & (flood >= thr)
        H = int(np.sum(high & obs)); M = int(np.sum(~high & obs)); F = int(np.sum(high & ~obs))
        csi = H / (H + M + F) if (H + M + F) else 0.0
        if best is None or csi > best[0]:
            best = (csi, thr, H, M, F)
    csi, thr, H, M, F = best
    pod = H / (H + M) if (H + M) else float("nan")
    far = F / (H + F) if (H + F) else float("nan")
    bias = (H + F) / (H + M) if (H + M) else float("nan")
    print(f"  Max-CSI operating point: threshold={thr:.3f}  POD={pod:.3f}  FAR={far:.3f}  CSI={csi:.3f}  bias={bias:.2f}")
    print(f"flood_pod,{pod:.3f}")
    print(f"flood_far,{far:.3f}")
    print(f"flood_csi,{csi:.3f}")
    print(f"flood_bias,{bias:.2f}")

    # --- independent validation: DSI historical floods + flood-prone areas ---
    import geopandas as gpd
    tt_path = DSI / "dsi_tarihi_taskinlar.gpkg"
    ta_path = DSI / "dsi_taskin_alanlari.gpkg"
    if tt_path.exists():
        pts = gpd.read_file(tt_path).to_crs(fmeta["crs"])
        rows_c, cols_c = rasterio.transform.rowcol(fmeta["transform"], pts.geometry.x, pts.geometry.y)
        rows_c = np.clip(np.array(rows_c), 0, flood.shape[0] - 1)
        cols_c = np.clip(np.array(cols_c), 0, flood.shape[1] - 1)
        vals_at_pts = flood[rows_c, cols_c]
        ok = np.isfinite(vals_at_pts) & (vals_at_pts > 0)
        pctl_at_pts = [float((flood[valid] <= v).sum()) / float(valid.sum()) * 100
                       for v in vals_at_pts[ok]]
        print(f"\n  DSI historical floods (n={len(pts)}): {ok.sum()} fall inside valid flood_hazard extent")
        print(f"  Mean flood-susceptibility percentile at historical-flood points: {np.mean(pctl_at_pts):.1f}"
              f" (50.0 = no enrichment, 100 = all in the highest-susceptibility pixel)")
        print(f"dsi_flood_pts_n,{len(pts)}")
        print(f"dsi_flood_pts_mean_percentile,{np.mean(pctl_at_pts):.1f}")
        above_median = float(np.mean([p > 50 for p in pctl_at_pts])) * 100
        print(f"dsi_flood_pts_above_median_pct,{above_median:.1f}")
    else:
        print("  [MISSING] DSI historical floods layer - run p2_11_dsi_masterplan_ingest.py")

    if ta_path.exists():
        areas = gpd.read_file(ta_path).to_crs(fmeta["crs"])
        area_geom = areas.union_all()
        from rasterio.features import geometry_mask
        area_raster_mask = ~geometry_mask([area_geom], out_shape=flood.shape,
                                           transform=fmeta["transform"], invert=False)
        in_area = valid & area_raster_mask
        if in_area.sum() > 0:
            mean_score_in = float(flood[in_area].mean())
            mean_score_out = float(flood[valid & ~area_raster_mask].mean())
            print(f"  DSI delineated flood-prone areas (n={len(areas)}): mean flood_hazard score "
                  f"inside={mean_score_in:.3f} vs outside={mean_score_out:.3f}")
            print(f"dsi_flood_area_mean_score_inside,{mean_score_in:.3f}")
            print(f"dsi_flood_area_mean_score_outside,{mean_score_out:.3f}")
    else:
        print("  [MISSING] DSI flood-prone areas layer - run p2_11_dsi_masterplan_ingest.py")


if __name__ == "__main__":
    rf_diagnostics()
    ca_hindcast()
    driver_ablation()
    flood_validation()
    print("\nDone. Transcribe key,value lines into docs/manuscripts/paper2/src/mydata.dat")
