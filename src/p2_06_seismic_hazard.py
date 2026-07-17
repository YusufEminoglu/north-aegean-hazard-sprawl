"""
Paper 2 - Seismic Hazard Surface
==================================
Produces a 0-5 ordinal seismic hazard raster for the Kuzey Ege Havzasi.

Data sources (automatic priority order):
  1. Earthquake_Bibliometric.gpkg  (gem_active_faults_harmonized layer)
     -> ALREADY IN data/01_raw/paper2/ - NO DOWNLOAD NEEDED!
  2. Clipped faults from p2_01 (gem_faults_havza.gpkg)
  3. AFAD/MTA standalone shapefiles
  4. Tier-C fallback (AFAD seismic zone 1 placeholder)

Three-tier pipeline:
  Tier A - FULL:     Fault proximity (0.40) + PGA (0.40) + Site amp (0.20)
  Tier B - MODERATE: Fault proximity (0.70) + Site amp (0.30)  [no PGA]
  Tier C - MINIMAL:  Uniform score = 4 (AFAD Zone 1 placeholder)

PGA download (optional, for Tier A):
  ESHM20: https://www.efehr.org/access/  -> Hazard maps -> PGA 475yr
  Place as: data/01_raw/paper2/pga_475yr.tif
"""

import sqlite3
import struct
import warnings
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject as warp_reproject, Resampling
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW         = PROJECT_ROOT / "data" / "01_raw" / "paper2"
INTERIM     = PROJECT_ROOT / "data" / "02_interim" / "paper2"
PROCESSED   = PROJECT_ROOT / "data" / "03_processed" / "paper2"
DRIVERS_DIR = PROCESSED / "drivers"
PROCESSED.mkdir(parents=True, exist_ok=True)

# Reference raster (30 m EPSG:32635)
REF_RASTER = INTERIM / "p2_srtm_elevation_clipped.tif"

# Kuzey Ege Havzasi bbox (WGS84) for spatial filtering
HAVZA_LON_MIN, HAVZA_LON_MAX = 25.5, 30.5
HAVZA_LAT_MIN, HAVZA_LAT_MAX = 36.5, 40.5

# Fault proximity buffer zones
# We must iterate from largest to smallest, so the smaller buffers overwrite the larger ones.
FAULT_BUFFERS = [
    (20_000, 2),   # 10 - 20 km  -> Low
    (10_000, 3),   #  5 - 10 km  -> Moderate
    (5_000,  4),   #  1 -  5 km  -> High
    (1_000,  5),   #  0 -  1 km  -> Very High
]
# >20 km -> score = 1 (Very Low)



# =============================================================================
# Geometry helpers for GPKG binary blob parsing (no geopandas dependency)
# =============================================================================
def _parse_gpkg_envelope(blob: bytes):
    """Extract bounding box from a GPKG geometry blob header."""
    if blob is None:
        return None
    try:
        flags = blob[3]
        n_env = {0: 0, 1: 4, 2: 6, 3: 6, 4: 8}.get((flags >> 1) & 0x07, 0)
        if n_env == 0:
            return None
        env = struct.unpack_from(f"<{n_env}d", blob, 8)
        return env[0], env[2], env[1], env[3]  # minx, miny, maxx, maxy
    except Exception:
        return None


def _parse_gpkg_wkb(blob: bytes):
    """Extract ISO WKB bytes from a GPKG geometry blob."""
    if blob is None:
        return None
    try:
        flags = blob[3]
        n_env = {0: 0, 1: 4, 2: 6, 3: 6, 4: 8}.get((flags >> 1) & 0x07, 0)
        wkb_offset = 8 + n_env * 8
        return bytes(blob[wkb_offset:])
    except Exception:
        return None


# =============================================================================
# Source detection
# =============================================================================
def find_fault_source():
    """
    Returns (path, source_type) where source_type is one of:
      'bibliometric_gpkg' - Earthquake_Bibliometric.gpkg
      'vector_file'       - shapefile or single-layer GPKG
      None                - no data found
    """
    # 1. Previously clipped GEM faults
    clipped = INTERIM / "gem_faults_havza.gpkg"
    if clipped.exists():
        print(f"  Found clipped faults: {clipped.name}")
        return clipped, "vector_file"

    # 2. Earthquake_Bibliometric.gpkg (PRIORITY - already present!)
    biblio = RAW / "Earthquake_Bibliometric.gpkg"
    if biblio.exists():
        print(f"  Found Earthquake_Bibliometric.gpkg -> using gem_active_faults_harmonized layer")
        return biblio, "bibliometric_gpkg"

    # 3. Standalone files
    for p in [RAW / "afad_faults.shp", RAW / "afad_faults.gpkg",
              RAW / "mta_active_faults.shp", RAW / "gem_faults.gpkg"]:
        if p.exists():
            print(f"  Found: {p.name}")
            return p, "vector_file"
    for p in list(RAW.glob("*fault*.gpkg")) + list(RAW.glob("*fault*.shp")):
        print(f"  Found: {p.name}")
        return p, "vector_file"

    return None, None


def find_pga_file():
    """Search for a PGA GeoTIFF."""
    candidates = list(RAW.glob("pga*.tif")) + list(RAW.glob("eshm20_pga*.tif"))
    candidates += [RAW / "pga_475yr.tif"]
    for p in candidates:
        if p.exists():
            print(f"  Found PGA raster: {p.name}")
            return p
    return None


# =============================================================================
# Stage 1 - Fault Proximity
# =============================================================================
def load_faults_from_bibliometric(gpkg_path: Path):
    """
    Extract gem_active_faults_harmonized geometries that intersect
    Kuzey Ege Havzasi from Earthquake_Bibliometric.gpkg.
    Returns a list of Shapely geometries (WGS84).
    """
    from shapely import wkb as shapely_wkb

    conn = sqlite3.connect(str(gpkg_path))
    cur  = conn.cursor()
    cur.execute("SELECT geom FROM gem_active_faults_harmonized")
    rows = cur.fetchall()
    conn.close()

    geometries = []
    skipped = 0
    for (blob,) in rows:
        bb = _parse_gpkg_envelope(blob)
        if bb is None:
            continue
        minx, miny, maxx, maxy = bb
        # Spatial filter: keep only those overlapping the havza bbox
        if (maxx < HAVZA_LON_MIN or minx > HAVZA_LON_MAX or
                maxy < HAVZA_LAT_MIN or miny > HAVZA_LAT_MAX):
            continue
        wkb_bytes = _parse_gpkg_wkb(blob)
        if wkb_bytes is None:
            continue
        try:
            geom = shapely_wkb.loads(wkb_bytes)
            if geom is not None and not geom.is_empty:
                geometries.append(geom)
        except Exception:
            skipped += 1

    print(f"  Extracted {len(geometries)} fault segments in Kuzey Ege bbox "
          f"({skipped} parse errors skipped)")
    return geometries


def make_fault_proximity(fault_path: Path, source_type: str,
                         shape: tuple, transform, crs_str: str) -> np.ndarray:
    """
    Rasterise fault buffer rings. Returns float32 array (values 1-5).
    """
    print("  Building fault proximity raster ...")

    if source_type == "bibliometric_gpkg":
        from shapely.ops import unary_union
        import geopandas as gpd

        geoms_wgs84 = load_faults_from_bibliometric(fault_path)
        if not geoms_wgs84:
            print("  WARNING: No fault geometries found in havza bbox.")
            return np.full(shape, 4.0, dtype=np.float32)

        faults = gpd.GeoDataFrame(geometry=geoms_wgs84, crs="EPSG:4326")
        faults = faults.to_crs("EPSG:32635")

    else:
        import geopandas as gpd
        faults = gpd.read_file(fault_path)
        if faults.crs is None:
            faults = faults.set_crs("EPSG:4326")
        if faults.crs.to_epsg() != 32635:
            faults = faults.to_crs("EPSG:32635")

    proximity = np.ones(shape, dtype=np.float32)  # base = 1 (>20 km)

    for dist_m, score in FAULT_BUFFERS:
        buf_geoms    = faults.geometry.buffer(dist_m)
        shapes_iter  = [(g.__geo_interface__, score)
                        for g in buf_geoms if g is not None and not g.is_empty]
        if shapes_iter:
            rasterize(shapes=shapes_iter, out=proximity,
                      transform=transform,
                      merge_alg=rasterio.enums.MergeAlg.replace)
        print(f"    Buffer {dist_m/1000:.0f} km -> score {score}  "
              f"({(proximity >= score).sum():,} px)")

    return proximity


# =============================================================================
# Stage 2 - PGA Score
# =============================================================================
def make_pga_score(pga_path: Path, ref_meta: dict, shape: tuple) -> np.ndarray:
    """Reproject PGA raster and rescale to 1-5."""
    print("  Reprojecting PGA raster ...")
    pga_arr = np.zeros(shape, dtype=np.float32)
    with rasterio.open(pga_path) as src:
        warp_reproject(
            source=rasterio.band(src, 1), destination=pga_arr,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
            resampling=Resampling.bilinear,
        )
    pga_max = float(np.nanmax(pga_arr[pga_arr > 0]))
    if pga_max > 5.0:
        pga_arr /= 981.0  # cm/s2 -> g
        print(f"  PGA converted from cm/s2 to g (original max={pga_max:.1f})")
    else:
        print(f"  PGA in g units (max={pga_max:.3f})")

    score = np.ones(shape, dtype=np.float32)
    score[pga_arr >= 0.06] = 2
    score[pga_arr >= 0.12] = 3
    score[pga_arr >= 0.20] = 4
    score[pga_arr >= 0.30] = 5
    return score


# =============================================================================
# Stage 3 - Site Amplification
# =============================================================================
def make_site_amplification(shape: tuple, ref_meta: dict = None) -> np.ndarray:
    """Low slope + high clay = higher amplification (alluvial plains).
    Reads directly from INTERIM clipped rasters to guarantee shape matches ref grid.
    """
    # Use original INTERIM clipped files (not the driver-aligned ones) to avoid shape mismatch
    slope_path = INTERIM / "p2_srtm_slope_clipped.tif"
    clay_path  = INTERIM / "p2_soilgrids_clay_clipped.tif"
    elev_path  = INTERIM / "p2_srtm_elevation_clipped.tif"

    def read_to_ref(path):
        """Read a raster and reproject to ref grid shape/transform."""
        out = np.zeros(shape, dtype=np.float32)
        with rasterio.open(path) as src:
            from rasterio.warp import reproject as warp_r, Resampling
            warp_r(
                source=rasterio.band(src, 1),
                destination=out,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_meta["transform"],
                dst_crs=ref_meta["crs"],
                resampling=Resampling.bilinear,
            )
        return out

    if ref_meta is None:
        # Fallback: use elevation raster directly
        with rasterio.open(elev_path) as e:
            elev = e.read(1).astype(np.float32)
        return np.clip(1.0 - elev / 500.0, 0, 1) * 5.0

    if slope_path.exists() and clay_path.exists():
        slope = np.clip(read_to_ref(slope_path), 0, None)
        clay  = np.clip(read_to_ref(clay_path),  0, None)
        amp = (np.clip(1.0 - slope / 30.0, 0, 1) * 0.5 +
               np.clip(clay / 100.0, 0, 1) * 0.5) * 5.0
        print("  Site amplification: slope + clay (INTERIM) used.")
    elif elev_path.exists():
        elev = read_to_ref(elev_path)
        amp = np.clip(1.0 - elev / 500.0, 0, 1) * 5.0
        print("  Site amplification: elevation-only fallback.")
    else:
        amp = np.full(shape, 3.0, dtype=np.float32)
        print("  Site amplification: uniform 3.0 (no data).")

    return amp.astype(np.float32)



# =============================================================================
# Main
# =============================================================================
def process_seismic_hazard():
    print("\n== Seismic Hazard Processing =========================================")

    if not REF_RASTER.exists():
        print(f"ERROR: Reference raster not found: {REF_RASTER}")
        print("       Run p2_01_move_clip_validate.py first.")
        return

    with rasterio.open(REF_RASTER) as ref:
        ref_meta  = ref.meta.copy()
        transform = ref.transform
        shape     = (ref.height, ref.width)
        nodata    = ref.nodata
        ref_arr   = ref.read(1)

    ref_meta.update(dtype=rasterio.float32, nodata=-9999.0,
                    driver="GTiff", compress="deflate")
    valid_mask = (ref_arr != nodata) if nodata is not None else np.ones(shape, bool)

    # Detect data
    fault_path, source_type = find_fault_source()
    pga_path = find_pga_file()

    # Site amplification (always computed - pass ref_meta for correct grid alignment)
    print("\n[Stage 3] Site Amplification ...")
    site_amp = make_site_amplification(shape, ref_meta)

    # Branch by availability
    if fault_path is not None:
        tier_label = "A" if pga_path else "B"
        print(f"\n[Stage 1] Fault Proximity (Tier {tier_label}) ...")
        fault_prox = make_fault_proximity(fault_path, source_type, shape, transform,
                                          ref_meta["crs"].to_string())

        # Ensure site_amp matches ref shape (driver grids may differ slightly)
        if site_amp.shape != shape:
            from scipy.ndimage import zoom as sci_zoom
            zf = (shape[0] / site_amp.shape[0], shape[1] / site_amp.shape[1])
            site_amp = sci_zoom(site_amp, zf, order=1).astype(np.float32)

        if pga_path is not None:
            print("\n[Stage 2] PGA Score ...")
            pga_score = make_pga_score(pga_path, ref_meta, shape)
            seismic = fault_prox * 0.40 + pga_score * 0.40 + site_amp * 0.20
            tier = "A (fault + PGA + site amp)"
        else:
            print("\n[Stage 2] No PGA -> Tier B weights ...")
            seismic = fault_prox * 0.70 + site_amp * 0.30
            tier = "B (fault + site amp, no PGA)"
    else:
        print("\n  WARNING: No fault data found -> Tier C fallback")
        print("  Entire Kuzey Ege = AFAD Zone 1 -> score = 4 (High)")
        print("  *** MUST flag as placeholder in manuscript Limitations ***")
        seismic = np.full(shape, 4.0, dtype=np.float32)
        tier = "C (uniform AFAD Zone-1 placeholder)"

    seismic[~valid_mask] = -9999.0

    # Quintile reclassification → 1-5 ordinal (ensures each class has ~20% of pixels)
    # Raw weighted score range is narrow (e.g., 0.7-2.2 when far from faults),
    # so class 5 is nearly empty without reclassification.
    print("\n  Applying quintile reclassification to 1-5 ordinal ...")
    valid_scores = seismic[valid_mask & (seismic != -9999.0)]
    q = np.nanpercentile(valid_scores, [20, 40, 60, 80])
    print(f"  Quintile breaks: {q[0]:.3f} | {q[1]:.3f} | {q[2]:.3f} | {q[3]:.3f}")
    ordinal = np.full(shape, -9999.0, dtype=np.float32)
    m = valid_mask & (seismic != -9999.0)
    ordinal[m] = 1
    ordinal[m & (seismic > q[0])] = 2
    ordinal[m & (seismic > q[1])] = 3
    ordinal[m & (seismic > q[2])] = 4
    ordinal[m & (seismic > q[3])] = 5
    seismic = ordinal

    # Save
    out_path = PROCESSED / "seismic_hazard.tif"
    with rasterio.open(out_path, "w", **ref_meta) as out:
        out.write(seismic, 1)

    valid_data = seismic[valid_mask & (seismic != -9999.0)]
    print(f"\n  Tier applied : {tier}")
    print(f"  Score range  : [{float(np.nanmin(valid_data)):.2f}, {float(np.nanmax(valid_data)):.2f}]")
    print(f"  Mean score   : {float(np.nanmean(valid_data)):.2f}")
    print(f"  Output       : {out_path.relative_to(PROJECT_ROOT)}")
    print()
    total = valid_data.size
    for cls, label in [(1, "Very Low"), (2, "Low"), (3, "Moderate"),
                       (4, "High"), (5, "Very High")]:
        n = int(((valid_data >= cls - 0.5) & (valid_data < cls + 0.5)).sum())
        print(f"    ~{cls} {label:<12}: {n:>9,} px  ({100*n/total:.1f}%)")



if __name__ == "__main__":
    process_seismic_hazard()
