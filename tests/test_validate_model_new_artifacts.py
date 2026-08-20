from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hedonic.validation import validate_model


def test_parse_selected_features_json_accepts_list_and_dict(tmp_path: Path) -> None:
    list_path = tmp_path / "selected_list.json"
    list_path.write_text(json.dumps(["LAND_SF", "INT_COND"]), encoding="utf-8")

    dict_path = tmp_path / "selected_dict.json"
    dict_path.write_text(json.dumps({"features": ["LAND_SF", "INT_COND"]}), encoding="utf-8")

    assert validate_model._parse_selected_features_json(list_path) == ["LAND_SF", "INT_COND"]
    assert validate_model._parse_selected_features_json(dict_path) == ["LAND_SF", "INT_COND"]


def test_build_ols_design_from_selected_features_creates_const_and_dummies() -> None:
    model_df = pd.DataFrame(
        {
            "LAND_SF": [1000.0, 1200.0, 1500.0],
            "INT_COND": ["A", "B", "A"],
        }
    )

    design_df, numeric, categorical = validate_model._build_ols_design_from_selected_features(
        model_df=model_df,
        selected_features=["LAND_SF", "INT_COND"],
    )

    assert design_df.columns[0] == "const"
    assert numeric == ["LAND_SF"]
    assert categorical == ["INT_COND"]
    assert any(col.startswith("INT_COND_") for col in design_df.columns)


def test_predict_from_ols_coefficients_uses_term_intersection(tmp_path: Path) -> None:
    design_df = pd.DataFrame(
        {
            "const": [1.0, 1.0],
            "LAND_SF": [1000.0, 2000.0],
            "INT_COND_B": [0.0, 1.0],
        }
    )

    coef_path = tmp_path / "coefs.csv"
    pd.DataFrame(
        {
            "term": ["const", "LAND_SF", "INT_COND_B", "UNUSED_TERM"],
            "coefficient_log": [0.5, 0.001, 0.25, 9.0],
        }
    ).to_csv(coef_path, index=False)

    pred_log, terms = validate_model._predict_from_ols_coefficients(design_df, coef_path)

    expected = np.array([1.5, 2.75])
    assert np.allclose(pred_log, expected)
    assert "UNUSED_TERM" not in terms


def test_merge_missing_model_columns_fills_by_pid(tmp_path: Path) -> None:
    base = pd.DataFrame({"PID": [1, 2], "LAND_SF": [1000.0, 1200.0]})
    aux_path = tmp_path / "aux.csv"
    pd.DataFrame({"PID": [1, 2], "INT_COND": ["A", "B"]}).to_csv(aux_path, index=False)

    merged = validate_model._merge_missing_model_columns(base, aux_path, ["LAND_SF", "INT_COND"])

    assert "INT_COND" in merged.columns
    assert merged["INT_COND"].tolist() == ["A", "B"]


def test_predict_from_ols_coefficients_rejects_missing_required_columns(tmp_path: Path) -> None:
    design_df = pd.DataFrame({"const": [1.0], "x": [2.0]})
    bad_path = tmp_path / "bad_coefs.csv"
    pd.DataFrame({"term": ["const"], "wrong": [1.0]}).to_csv(bad_path, index=False)

    with pytest.raises(ValueError):
        validate_model._predict_from_ols_coefficients(design_df, bad_path)
