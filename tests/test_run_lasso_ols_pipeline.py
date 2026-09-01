from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from hedonic.workflow import run_lasso_ols_pipeline as pipeline


class _DummyPreprocess:
    def get_feature_names_out(self):
        return ["num__LAND_SF", "cat__INT_COND_A", "cat__INT_COND_B"]


class _DummyLasso:
    def __init__(self) -> None:
        self.coef_ = [0.2, 0.3, 0.15]
        self.alpha_ = 0.01


class _DummyPipeline:
    def __init__(self) -> None:
        self.named_steps = {"preprocess": _DummyPreprocess(), "lasso": _DummyLasso()}

    def fit(self, X, y):
        return self


def test_resolve_feature_set_uses_default_set() -> None:
    df = pd.DataFrame({"LAND_SF": [1], "INT_COND": ["A"]})

    features, source = pipeline._resolve_feature_set(df)

    assert set(features) == {"LAND_SF", "INT_COND"}
    assert source == "DEFAULT_FEATURE_SET"


def test_select_features_with_lasso_honors_min_categorical_dummies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_build_lasso_pipeline", lambda **kwargs: _DummyPipeline())

    model_df = pd.DataFrame(
        {
            "LAND_SF": [1000, 1200, 1500, 1600],
            "INT_COND": ["A", "B", "A", "B"],
            "TOTAL_VALUE": [300000, 350000, 400000, 420000],
        }
    )

    result = pipeline._select_features_with_lasso(
        model_df=model_df,
        feature_list=["LAND_SF", "INT_COND"],
        numeric_features=["LAND_SF"],
        categorical_features=["INT_COND"],
        cv_folds=2,
        random_seed=42,
        lasso_max_iter=1000,
        coef_threshold=0.1,
        min_categorical_dummies=2,
    )

    assert result["selected_features"] == ["LAND_SF", "INT_COND"]
    assert result["categorical_surviving_counts"]["INT_COND"] == 2


def test_final_ols_inference_uses_log_price_per_sqft_target() -> None:
    model_df = pd.DataFrame(
        {
            "LAND_SF": [1000, 1200, 1500, 1800, 2000],
            "INT_COND": ["A", "B", "A", "B", "A"],
            "LOG_PRICE_PER_SQFT": [4.1, 4.2, 4.3, 4.35, 4.4],
            "TOTAL_VALUE": [300000, 340000, 410000, 470000, 520000],
        }
    )

    results, coef_table = pipeline._final_ols_inference(model_df, ["LAND_SF", "INT_COND"])

    assert results.nobs == len(model_df)
    assert "term" in coef_table.columns
    assert "coefficient_log" in coef_table.columns
