"""Tests for FY2015 parcel cleaning and spatial land-use join step."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from shapely.geometry import Polygon
from preprocessing.steps.add_2015_data import clean_geom, join_2015_land_use_to_2025


def test_clean_geom_handles_none_and_invalid() -> None:
    assert clean_geom(None) is None

    valid_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    cleaned = clean_geom(valid_poly)
    assert cleaned is not None
    assert cleaned.is_valid


def test_join_2015_land_use_selects_largest_overlap() -> None:
    # 2025 parcel covering (0,0) to (10,10)
    poly_2025 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    p25_gdf = gpd.GeoDataFrame(
        {"PID": ["2025_001"]},
        geometry=[poly_2025],
        crs="EPSG:4326",
    )

    # 2015 parcel A: small overlap covering (0,0) to (2,10) -> area 20 -> LU="R1"
    poly_2015_a = Polygon([(0, 0), (2, 0), (2, 10), (0, 10)])
    # 2015 parcel B: large overlap covering (2,0) to (10,10) -> area 80 -> LU="C"
    poly_2015_b = Polygon([(2, 0), (10, 0), (10, 10), (2, 10)])

    p15_gdf = gpd.GeoDataFrame(
        {
            "lu_2015": ["R1", "C"],
            "land_sf_2015": [200.0, 800.0],
        },
        geometry=[poly_2015_a, poly_2015_b],
        crs="EPSG:4326",
    )

    joined = join_2015_land_use_to_2025(p25_gdf, p15_gdf)

    assert len(joined) == 1
    assert "lu_2015" in joined.columns
    assert "LU_2015" not in joined.columns
    assert "_p25_row" not in joined.columns
    assert "flag_multiple_2015_in_2025" in joined.columns
    assert "flag_multiple_2025_in_2015" in joined.columns
    # Must pick 'C' because parcel B has 80% overlap vs parcel A's 20%
    assert joined.iloc[0]["lu_2015"] == "C"
    # Parcel 2025_001 intersected 2 2015 parcels (poly_2015_a and poly_2015_b)
    assert bool(joined.iloc[0]["flag_multiple_2015_in_2025"]) is True

