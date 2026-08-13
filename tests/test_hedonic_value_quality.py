"""Tests for model-prep behavior now that hedonic row filters are disabled."""

from __future__ import annotations

import pandas as pd

from hedonic.common.modeling_common import prepare_model_df


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


def test_prepare_model_df_does_not_drop_rows_for_hedonic_quality_values() -> None:
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


def test_prepare_model_df_does_not_drop_rows_for_structural_and_locational_values() -> None:
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
    bad_units.update({"RES_UNITS": 0, "RES_FLOOR": 2, "YR_BUILT": 2000, "STRUCTURE_CLASS": "Wood frame", "neighborhood_walkability": 60, "emp_dist_m": 1200, "median_hh_income": 90000})
    rows.append(bad_units)

    bad_floor = _base_row()
    bad_floor.update({"RES_UNITS": 2, "RES_FLOOR": 0, "YR_BUILT": 2000, "STRUCTURE_CLASS": "Wood frame", "neighborhood_walkability": 60, "emp_dist_m": 1200, "median_hh_income": 90000})
    rows.append(bad_floor)

    bad_year = _base_row()
    bad_year.update({"RES_UNITS": 2, "RES_FLOOR": 2, "YR_BUILT": 1200, "STRUCTURE_CLASS": "Wood frame", "neighborhood_walkability": 60, "emp_dist_m": 1200, "median_hh_income": 90000})
    rows.append(bad_year)

    bad_structure = _base_row()
    bad_structure.update({"RES_UNITS": 2, "RES_FLOOR": 2, "YR_BUILT": 2000, "STRUCTURE_CLASS": "", "neighborhood_walkability": 60, "emp_dist_m": 1200, "median_hh_income": 90000})
    rows.append(bad_structure)

    bad_walkability = _base_row()
    bad_walkability.update({"RES_UNITS": 2, "RES_FLOOR": 2, "YR_BUILT": 2000, "STRUCTURE_CLASS": "Wood frame", "neighborhood_walkability": -5, "emp_dist_m": 1200, "median_hh_income": 90000})
    rows.append(bad_walkability)

    bad_emp = _base_row()
    bad_emp.update({"RES_UNITS": 2, "RES_FLOOR": 2, "YR_BUILT": 2000, "STRUCTURE_CLASS": "Wood frame", "neighborhood_walkability": 60, "emp_dist_m": 0, "median_hh_income": 90000})
    rows.append(bad_emp)

    bad_income = _base_row()
    bad_income.update({"RES_UNITS": 2, "RES_FLOOR": 2, "YR_BUILT": 2000, "STRUCTURE_CLASS": "Wood frame", "neighborhood_walkability": 60, "emp_dist_m": 1200, "median_hh_income": -666666666})
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


def test_prepare_model_df_still_filters_non_positive_target_rows() -> None:
    df = pd.DataFrame([_base_row(), _base_row(), {**_base_row(), "TOTAL_VALUE": 0}])

    model_df = prepare_model_df(
        df,
        numeric_features=["LAND_SF", "LIVING_AREA", "GROSS_AREA"],
        categorical_features=["INT_COND", "OVERALL_COND", "BLDG_TYPE"],
        target_col="TOTAL_VALUE",
    )

    assert len(model_df) == 2


def test_prepare_model_df_has_no_quality_warnings_when_rows_are_reasonable() -> None:
    df = pd.DataFrame([_base_row(), _base_row()])

    model_df = prepare_model_df(
        df,
        numeric_features=["LAND_SF", "LIVING_AREA", "GROSS_AREA"],
        categorical_features=["INT_COND", "OVERALL_COND", "BLDG_TYPE"],
        target_col="TOTAL_VALUE",
    )

    assert len(model_df) == 2
