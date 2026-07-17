"""Fuse four continuous hazard surfaces and intersect 2050 urban scenarios.

The final ordinal surface uses Jenks natural breaks on the full valid basin
grid. Missing hazard components are fatal: silently substituting a zero layer
would change the scientific model.
"""

from __future__ import annotations

from pathlib import Path

import jenkspy
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "02_interim" / "paper2"
PROCESSED = PROJECT_ROOT / "data" / "03_processed" / "paper2"
TABLES = PROJECT_ROOT / "outputs" / "tables" / "paper2"
TABLES.mkdir(parents=True, exist_ok=True)

REF_PATH = INTERIM / "p2_srtm_elevation_clipped.tif"
WEIGHTS = {
    "flood": 0.35,
    "seismic": 0.35,
    "bioclimate": 0.20,
    "wildfire": 0.10,
}


def load_aligned(
    path: Path,
    ref_meta: dict,
    *,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    destination = np.full(
        (ref_meta["height"], ref_meta["width"]), np.nan, dtype=np.float32
    )
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=ref_meta["transform"],
            dst_crs=ref_meta["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return destination


def normalize(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = mask & np.isfinite(arr)
    values = arr[selected]
    if not values.size or np.isclose(values.max(), values.min()):
        raise ValueError("A hazard component is empty or constant after alignment.")
    output = np.full_like(arr, np.nan, dtype=np.float32)
    output[selected] = (arr[selected] - values.min()) / (values.max() - values.min())
    return output


def jenks_classify(
    arr: np.ndarray,
    mask: np.ndarray,
    n_classes: int = 5,
) -> tuple[np.ndarray, list[float]]:
    values = arr[mask & np.isfinite(arr)].astype(np.float64)
    if values.size < n_classes:
        raise ValueError("Too few valid cells for Jenks classification.")
    breaks = [float(value) for value in jenkspy.jenks_breaks(values, n_classes=n_classes)]
    ordinal = np.zeros(arr.shape, dtype=np.uint8)
    ordinal[mask] = np.digitize(arr[mask], breaks[1:-1], right=True) + 1
    return ordinal, breaks


def write_continuous(path: Path, arr: np.ndarray, meta: dict, valid: np.ndarray) -> None:
    output = arr.astype(np.float32, copy=True)
    output[~valid] = -9999.0
    out_meta = meta.copy()
    out_meta.update(dtype=rasterio.float32, count=1, nodata=-9999.0)
    with rasterio.open(path, "w", **out_meta) as target:
        target.write(output, 1)


def write_ordinal(path: Path, arr: np.ndarray, meta: dict) -> None:
    out_meta = meta.copy()
    out_meta.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(path, "w", **out_meta) as target:
        target.write(arr.astype(np.uint8), 1)


def main() -> None:
    print("== Multi-hazard fusion and Jenks classification ==")
    with rasterio.open(REF_PATH) as reference:
        ref_meta = reference.meta.copy()
        ref_arr = reference.read(1).astype(np.float32)
        nodata = reference.nodata

    valid = np.isfinite(ref_arr)
    if nodata is not None:
        valid &= ref_arr != nodata

    paths = {
        key: PROCESSED / f"{key}_hazard.tif"
        for key in WEIGHTS
    }
    components = {}
    for key, path in paths.items():
        aligned = load_aligned(path, ref_meta)
        valid &= np.isfinite(aligned)
        components[key] = aligned

    normalized = {
        key: normalize(arr, valid)
        for key, arr in components.items()
    }
    multi = sum(normalized[key] * WEIGHTS[key] for key in WEIGHTS).astype(np.float32)
    write_continuous(
        PROCESSED / "multi_hazard_continuous.tif", multi, ref_meta, valid
    )

    ordinal, breaks = jenks_classify(multi, valid)
    write_ordinal(PROCESSED / "multi_hazard_surface.tif", ordinal, ref_meta)

    print("  Jenks breaks: " + " | ".join(f"{value:.5f}" for value in breaks))
    high_share = 100.0 * np.count_nonzero(valid & (ordinal >= 4)) / np.count_nonzero(valid)
    print(f"  basin high + very-high share: {high_share:.1f}%")

    lulc_path = PROCESSED / "lulc_simplified_2020.tif"
    lulc_2020 = load_aligned(lulc_path, ref_meta, resampling=Resampling.nearest)
    base_urban = np.isclose(lulc_2020, 1)
    pixel_area_ha = abs(ref_meta["transform"].a * ref_meta["transform"].e) / 10000.0

    for scenario in ("sim_2050_bau", "sim_2050_eco"):
        simulation = load_aligned(
            PROCESSED / f"{scenario}.tif",
            ref_meta,
            resampling=Resampling.nearest,
        )
        new_urban = np.isclose(simulation, 1) & ~base_urban & valid
        rows = []
        for hazard_class in range(1, 6):
            pixels = int(np.count_nonzero(new_urban & (ordinal == hazard_class)))
            rows.append(
                {
                    "Hazard_Class": hazard_class,
                    "New_Urban_Area_ha": pixels * pixel_area_ha,
                }
            )

        table = pd.DataFrame(rows)
        total = table["New_Urban_Area_ha"].sum()
        table["Percentage_%"] = (
            table["New_Urban_Area_ha"].div(total).mul(100).fillna(0).round(1)
        )
        output_path = TABLES / f"exposure_{scenario}.csv"
        table.to_csv(output_path, index=False)
        high_area = table.loc[
            table["Hazard_Class"].isin([4, 5]), "New_Urban_Area_ha"
        ].sum()
        print(
            f"  {scenario}: {total:,.1f} ha new urban; "
            f"{high_area:,.1f} ha high/very-high"
        )

    print("  saved continuous surface, Jenks classes, and exposure tables")


if __name__ == "__main__":
    main()
