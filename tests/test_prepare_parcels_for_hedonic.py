from __future__ import annotations

import pandas as pd

from hedonic.prepare_parcels_for_hedonic import prepare_hedonic_dataframe


def test_prepare_hedonic_dataframe_filters_and_clips() -> None:
    df = pd.DataFrame(
        {
            "PID": [1, 2, 3],
            "LU": ["R1", "A", "X"],
            "TOTAL_VALUE": [10000, 200000, 30000],
            "LIVING_AREA": [800, 1500, 1200],
            "LAND_SF": [100, 2000, 400],
            "GROSS_AREA": [500, 4000, 700],
            "INT_COND": ["A", "F", "E"],
            "OVERALL_COND": ["A", "E", "G"],
            "RES_FLOOR": [1, 2, 3],
            "YR_BUILT": [1940, 2020, 1900],
            "BLDG_TYPE": ["SFR", "MF", "SFR"],
            "neighborhood_walkability": [50.0, 80.0, 90.0],
            "emp_dist_m": [1000.0, 2000.0, 1500.0],
            "median_hh_income": [50000.0, 80000.0, 60000.0],
            "neighborhood_name": ["A", "B", "C"],
            "geometry": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)"],
        }
    )

    result = prepare_hedonic_dataframe(df)

    assert result["LU"].isin(["R1", "A"]).all()
    assert list(result.columns) == [
        "TOTAL_VALUE",
        "LU",
        "LIVING_AREA",
        "LAND_SF",
        "GROSS_AREA",
        "INT_COND",
        "OVERALL_COND",
        "RES_FLOOR",
        "YR_BUILT",
        "BLDG_TYPE",
        "neighborhood_walkability",
        "emp_dist_m",
        "median_hh_income",
        "neighborhood_name",
        "geometry",
    ]
    assert result["TOTAL_VALUE"].ge(0).all()
    assert result["RES_FLOOR"].ge(1).all()
