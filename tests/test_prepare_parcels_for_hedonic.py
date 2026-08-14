from __future__ import annotations

import pandas as pd

from hedonic.common.modeling_common import prepare_model_df
from hedonic.prepare_parcels_for_hedonic import prepare_hedonic_dataframe


def _base_row() -> dict[str, object]:
    return {
        "TOTAL_VALUE": 500000,
        "INT_COND": "A - Average",
        "OVERALL_COND": "A - Average",
        "BLDG_TYPE": "CL - Colonial",
        "GROSS_AREA": 2400,
        "LIVING_AREA": 1800,
        "LAND_SF": 3200,
    }


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


def test_prepare_model_df_keeps_reasonable_rows_and_filters_invalid_targets() -> None:
    rows = []

    valid_a = _base_row()
    rows.append(valid_a)

    bad_int_cond = _base_row()
    bad_int_cond["INT_COND"] = ""
    rows.append(bad_int_cond)

    bad_overall_cond = _base_row()
    bad_overall_cond["OVERALL_COND"] = "Z - Unknown"
    rows.append(bad_overall_cond)

    bad_bldg_type = _base_row()
    bad_bldg_type["BLDG_TYPE"] = ""
    rows.append(bad_bldg_type)

    bad_gross = _base_row()
    bad_gross["GROSS_AREA"] = -100
    rows.append(bad_gross)

    bad_living = _base_row()
    bad_living["LIVING_AREA"] = 0
    rows.append(bad_living)

    bad_land = _base_row()
    bad_land["LAND_SF"] = -1
    rows.append(bad_land)

    valid_b = _base_row()
    valid_b["TOTAL_VALUE"] = 650000
    valid_b["LIVING_AREA"] = 2200
    rows.append(valid_b)

    df = pd.DataFrame(rows)

    model_df = prepare_model_df(
        df,
        numeric_features=["LAND_SF", "LIVING_AREA", "GROSS_AREA"],
        categorical_features=["INT_COND"],
        target_col="TOTAL_VALUE",
    )

    assert len(model_df) == len(df)

    df2 = pd.DataFrame([_base_row(), _base_row(), {**_base_row(), "TOTAL_VALUE": 0}])
    model_df2 = prepare_model_df(
        df2,
        numeric_features=["LAND_SF", "LIVING_AREA", "GROSS_AREA"],
        categorical_features=["INT_COND", "OVERALL_COND", "BLDG_TYPE"],
        target_col="TOTAL_VALUE",
    )

    assert len(model_df2) == 2


def test_prepare_model_df_keeps_reasonable_structural_and_locational_rows() -> None:
    rows = []

    valid = _base_row()
    valid.update(
        {
            "RES_UNITS": 2,
            "RES_FLOOR": 2,
            "YR_BUILT": 2000,
            "STRUCTURE_CLASS": "Wood frame",
            "neighborhood_walkability": 60,
            "emp_dist_m": 1200,
            "median_hh_income": 90000,
        }
    )
    rows.append(valid)

    bad_units = _base_row()
    bad_units.update({
        "RES_UNITS": 0,
        "RES_FLOOR": 2,
        "YR_BUILT": 2000,
        "STRUCTURE_CLASS": "Wood frame",
        "neighborhood_walkability": 60,
        "emp_dist_m": 1200,
        "median_hh_income": 90000,
    })
    rows.append(bad_units)

    bad_floor = _base_row()
    bad_floor.update({
        "RES_UNITS": 2,
        "RES_FLOOR": 0,
        "YR_BUILT": 2000,
        "STRUCTURE_CLASS": "Wood frame",
        "neighborhood_walkability": 60,
        "emp_dist_m": 1200,
        "median_hh_income": 90000,
    })
    rows.append(bad_floor)

    bad_year = _base_row()
    bad_year.update({
        "RES_UNITS": 2,
        "RES_FLOOR": 2,
        "YR_BUILT": 1200,
        "STRUCTURE_CLASS": "Wood frame",
        "neighborhood_walkability": 60,
        "emp_dist_m": 1200,
        "median_hh_income": 90000,
    })
    rows.append(bad_year)

    bad_structure = _base_row()
    bad_structure.update({
        "RES_UNITS": 2,
        "RES_FLOOR": 2,
        "YR_BUILT": 2000,
        "STRUCTURE_CLASS": "",
        "neighborhood_walkability": 60,
        "emp_dist_m": 1200,
        "median_hh_income": 90000,
    })
    rows.append(bad_structure)

    bad_walkability = _base_row()
    bad_walkability.update({
        "RES_UNITS": 2,
        "RES_FLOOR": 2,
        "YR_BUILT": 2000,
        "STRUCTURE_CLASS": "Wood frame",
        "neighborhood_walkability": -5,
        "emp_dist_m": 1200,
        "median_hh_income": 90000,
    })
    rows.append(bad_walkability)

    bad_emp = _base_row()
    bad_emp.update({
        "RES_UNITS": 2,
        "RES_FLOOR": 2,
        "YR_BUILT": 2000,
        "STRUCTURE_CLASS": "Wood frame",
        "neighborhood_walkability": 60,
        "emp_dist_m": 0,
        "median_hh_income": 90000,
    })
    rows.append(bad_emp)

    bad_income = _base_row()
    bad_income.update({
        "RES_UNITS": 2,
        "RES_FLOOR": 2,
        "YR_BUILT": 2000,
        "STRUCTURE_CLASS": "Wood frame",
        "neighborhood_walkability": 60,
        "emp_dist_m": 1200,
        "median_hh_income": -666666666,
    })
    rows.append(bad_income)

    df = pd.DataFrame(rows)

    model_df = prepare_model_df(
        df,
        numeric_features=[
            "LAND_SF",
            "LIVING_AREA",
            "GROSS_AREA",
            "RES_UNITS",
            "RES_FLOOR",
            "YR_BUILT",
            "neighborhood_walkability",
            "emp_dist_m",
            "median_hh_income",
        ],
        categorical_features=["INT_COND", "OVERALL_COND", "BLDG_TYPE", "STRUCTURE_CLASS"],
        target_col="TOTAL_VALUE",
    )

    assert len(model_df) == len(df)
