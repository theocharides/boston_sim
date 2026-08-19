import numpy as np
import pytest

from hedonic.validation.validate_model import _classify_spatial_dependence, _spatial_lm_diagnostics


def test_classify_spatial_lag_when_robust_lag_is_stronger() -> None:
    result = _classify_spatial_dependence(
        {
            "lme": {"statistic": 2.0, "p_value": 0.2},
            "lml": {"statistic": 4.0, "p_value": 0.05},
            "rlme": {"statistic": 1.0, "p_value": 0.4},
            "rlml": {"statistic": 5.0, "p_value": 0.03},
            "sarma": {"statistic": 10.0, "p_value": 0.02},
        }
    )
    assert result["classification"] == "spatial lag"


def test_classify_spatial_error_when_robust_error_is_stronger() -> None:
    result = _classify_spatial_dependence(
        {
            "lme": {"statistic": 5.0, "p_value": 0.04},
            "lml": {"statistic": 1.0, "p_value": 0.5},
            "rlme": {"statistic": 7.0, "p_value": 0.02},
            "rlml": {"statistic": 1.5, "p_value": 0.2},
            "sarma": {"statistic": 9.0, "p_value": 0.01},
        }
    )
    assert result["classification"] == "spatial error"


def test_classify_ambiguous_when_neither_test_is_clear() -> None:
    result = _classify_spatial_dependence(
        {
            "lme": {"statistic": 1.0, "p_value": 0.5},
            "lml": {"statistic": 1.2, "p_value": 0.4},
            "rlme": {"statistic": 0.9, "p_value": 0.7},
            "rlml": {"statistic": 1.1, "p_value": 0.6},
            "sarma": {"statistic": 1.5, "p_value": 0.3},
        }
    )
    assert result["classification"] == "ambiguous"


def test_spatial_lm_diagnostics_are_skipped_when_design_matrix_is_singular() -> None:
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    result = _spatial_lm_diagnostics(X, y, coords, k_neighbors=2)

    assert result["available"] is False
    reason = result["reason"].lower()
    assert any(term in reason for term in ("singular", "collinear", "constant"))
