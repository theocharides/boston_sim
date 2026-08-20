from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simulation.steps import common


class _Model:
    feature_names_in_ = ["feature_a"]

    def predict(self, rows: pd.DataFrame):
        return np.log1p(np.array([100.0] * len(rows), dtype=float))


def test_allocate_units_by_step_spreads_remainder() -> None:
    assert common.allocate_units_by_step(10, 3) == [4, 3, 3]


def test_update_res_units_with_allocations_adds_allocated_units() -> None:
    df = pd.DataFrame({"RES_UNITS": [1, 2], "allocated_units": [3, 4]})
    out = common.update_res_units_with_allocations(df)

    assert out["RES_UNITS"].tolist() == [4, 6]


def test_update_res_units_with_allocations_requires_columns() -> None:
    with pytest.raises(ValueError):
        common.update_res_units_with_allocations(pd.DataFrame({"RES_UNITS": [1]}))


def test_predict_and_update_values_with_model_updates_residential_rows() -> None:
    df = pd.DataFrame(
        {
            "LU": ["R1", "I"],
            "feature_a": [1.0, 2.0],
            "TOTAL_VALUE": [0.0, 0.0],
        }
    )
    out = common.predict_and_update_values_with_model(df, _Model())

    assert out.loc[0, "TOTAL_VALUE"] > 0
    assert out.loc[1, "TOTAL_VALUE"] == 0
