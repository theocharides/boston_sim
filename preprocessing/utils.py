"""Shared utilities for preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))




def load_parcels_csv(csv_path: Path, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Load parcels CSV with WKT geometry column and return as GeoDataFrame."""
    df = pd.read_csv(csv_path, low_memory=False)
    if "geometry" not in df.columns:
        raise ValueError("Input CSV must contain a 'geometry' WKT column.")

    geoms = df["geometry"].map(
        lambda value: wkt.loads(value) if isinstance(value, str) and value.strip() else None
    )
    return gpd.GeoDataFrame(df, geometry=geoms, crs=crs)


def save_parcels_csv(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Save GeoDataFrame to CSV with geometry as WKT."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
    df["geometry"] = gdf.geometry.map(
        lambda geom: geom.wkt if getattr(geom, "wkt", None) is not None else pd.NA
    )
    df.to_csv(output_path, index=False)


def to_point_geometry(geom):
    """Convert any geometry to a point for distance calculations."""
    if geom is None:
        return None
    return geom if geom.geom_type == "Point" else geom.representative_point()


def clean_numeric(series: pd.Series) -> pd.Series:
    """Parse numeric-looking fields that may include commas or symbols."""
    as_text = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    as_text = as_text.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(as_text, errors="coerce")


def clean_year_series(series: pd.Series) -> pd.Series:
    """Parse year fields and correct obvious single-digit suffix typos.

    Known source correction: 20198 -> 2019.
    """
    raw_text = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    raw_text = raw_text.replace({"": np.nan, "nan": np.nan, "None": np.nan, "20198": "2019"})
    years = pd.to_numeric(raw_text, errors="coerce")
    malformed_mask = (years >= 10000) & (years <= 99999)
    if malformed_mask.any():
        lower, upper = 1600.0, 2030.0
        shortened = (years[malformed_mask] // 10).astype("Int64")
        plausible_shortened = shortened.between(lower, upper)
        years.loc[malformed_mask[malformed_mask].index[plausible_shortened]] = shortened[plausible_shortened].astype(float)
    return years
