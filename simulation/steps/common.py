"""Shared helpers for development simulation step modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hedonic.common.modeling_common import TARGET_COL

RESIDENTIAL_LU_CODES = {"A", "CD", "CM", "R1", "R2", "R3", "R4", "RC", "RL"}


def write_simple_yaml(path: Path, values: dict[str, object]) -> None:
    lines: list[str] = []
    for key, value in values.items():
        lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def allocate_units_by_step(total_units: int, time_steps: int) -> list[int]:
    base = total_units // time_steps
    remainder = total_units % time_steps
    values = [base] * time_steps
    for index in range(remainder):
        values[index] += 1
    return values


def update_res_units_with_allocations(df: pd.DataFrame, allocated_col: str = "allocated_units") -> pd.DataFrame:
    if allocated_col not in df.columns:
        raise ValueError(f"Required allocation column missing: {allocated_col}")
    if "RES_UNITS" not in df.columns:
        raise ValueError("Required column missing: RES_UNITS")

    out = df.copy()
    current_units = pd.to_numeric(out["RES_UNITS"], errors="coerce").fillna(0)
    allocated = pd.to_numeric(out[allocated_col], errors="coerce").fillna(0)
    out["RES_UNITS"] = (current_units + allocated).round().astype("Int64")
    return out


def predict_and_update_values_with_model(parcels_df: pd.DataFrame, model: object) -> pd.DataFrame:
    """Update residential TOTAL_VALUE using a pre-trained hedonic model."""
    if not hasattr(model, "predict"):
        raise ValueError("Loaded hedonic object does not support predict().")
    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Loaded hedonic model is missing feature_names_in_. "
            "Re-train the model with a compatible scikit-learn version."
        )

    out = parcels_df.copy()
    lu_codes = out["LU"].astype("string").fillna("").str.strip().str.upper()
    residential_mask = lu_codes.isin(RESIDENTIAL_LU_CODES)

    feature_names = [str(name) for name in model.feature_names_in_]
    missing = [name for name in feature_names if name not in out.columns]
    if missing:
        raise ValueError(f"Input parcels are missing model features required for prediction: {missing}")

    pred_rows = out.loc[residential_mask, feature_names].copy()
    pred_log_all = model.predict(pred_rows)
    pred_level_all = np.expm1(pred_log_all)
    out.loc[residential_mask, TARGET_COL] = np.maximum(pred_level_all, 0.0)
    return out
