"""Shared, secret-free Google Earth Engine configuration.

Set EE_PROJECT and GEE_DRIVE_FOLDER in the shell before running an export
script. The study boundary is read from the local data directory by default.
Users who keep the boundary as an Earth Engine asset may additionally set
EE_ROI_ASSET.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import ee


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROI_PATH = (
    PROJECT_ROOT / "data" / "01_raw" / "paper2" / "kuzey_ege_havzasi_v2.geojson"
)


def require_setting(name: str) -> str:
    """Return a required environment setting without exposing its value."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example values into your shell "
            "environment; never commit private account identifiers."
        )
    return value


def initialize_ee() -> None:
    """Initialize Earth Engine with the caller's project."""
    ee.Initialize(project=require_setting("EE_PROJECT"))


def drive_folder() -> str:
    """Return the caller-supplied Google Drive export folder."""
    return require_setting("GEE_DRIVE_FOLDER")


def load_roi() -> ee.Geometry:
    """Load the basin geometry from a private EE asset or a local GeoJSON."""
    asset_id = os.environ.get("EE_ROI_ASSET", "").strip()
    if asset_id:
        return ee.FeatureCollection(asset_id).geometry()

    roi_path = Path(os.environ.get("EE_ROI_GEOJSON", str(DEFAULT_ROI_PATH)))
    if not roi_path.exists():
        raise FileNotFoundError(
            f"Study boundary not found at {roi_path}. Set EE_ROI_ASSET or "
            "EE_ROI_GEOJSON to your own boundary source."
        )

    with roi_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)

    geometries: list[dict] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry", {})
        if geometry.get("type") == "GeometryCollection":
            geometries.extend(geometry.get("geometries", []))
        else:
            geometries.append(geometry)

    polygons = [
        geometry
        for geometry in geometries
        if geometry.get("type") in {"Polygon", "MultiPolygon"}
    ]
    if not polygons:
        raise ValueError(f"No polygon geometry found in {roi_path}")

    largest = max(polygons, key=lambda geometry: len(json.dumps(geometry)))
    return ee.Geometry(largest)
