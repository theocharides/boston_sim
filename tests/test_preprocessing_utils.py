from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from preprocessing import utils


def test_clean_numeric_strips_symbols() -> None:
    series = pd.Series(["$1,200", " 300 ", "", "None"])
    cleaned = utils.clean_numeric(series)

    assert cleaned.iloc[0] == 1200
    assert cleaned.iloc[1] == 300
    assert np.isnan(cleaned.iloc[2])
    assert np.isnan(cleaned.iloc[3])


def test_clean_year_series_fixes_known_and_malformed_values() -> None:
    series = pd.Series(["20198", "19876", "1999", "bad"])
    cleaned = utils.clean_year_series(series)

    assert cleaned.iloc[0] == 2019
    assert cleaned.iloc[1] == 1987
    assert cleaned.iloc[2] == 1999
    assert np.isnan(cleaned.iloc[3])


def test_to_point_geometry_returns_representative_point() -> None:
    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    result = utils.to_point_geometry(poly)

    assert result.geom_type == "Point"


def test_save_and_load_parcels_csv_roundtrip(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {"PID": [1, 2]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )
    out_path = tmp_path / "parcels.csv"

    utils.save_parcels_csv(gdf, out_path)
    loaded = utils.load_parcels_csv(out_path)

    assert len(loaded) == 2
    assert "geometry" in loaded.columns
