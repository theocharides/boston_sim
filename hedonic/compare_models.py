"""
Compare hedonic models using different variable combinations.

This script tests multiple feature specifications on the same train/test split,
computes fit metrics for each, and writes a summary table. It helps identify
which variables contribute most to model performance.

Specifications are built as different combinations of structural features and
locational/neighborhood features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from modeling_common import (
    LOC_NEIGHBORHOOD_FEATURES,
    STRUCTURAL_FEATURES,
    TARGET_COL,
    available_features,
    build_ridge_pipeline,
    evaluate_log_and_level,
    infer_feature_types,
    prepare_model_df,
    require_existing_path,
    split_features_target,
    train_test_split_indices,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Compare hedonic models with different variable combinations."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels.csv",
        help="Path to parcel-level input CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "hedonic" / "artifacts" / "model_comparison.csv",
        help="Path to write comparison results table.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for holdout evaluation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )

    args = parser.parse_args()
    args.input_csv = require_existing_path(args.input_csv, "Parcel CSV")
    args.output_csv = args.output_csv.expanduser().resolve()
    return args


def fit_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict:
    """Fit a ridge model on the given features and return metrics."""
    feature_names = [*numeric_features, *categorical_features]
    if len(X_train) == 0 or len(X_test) == 0:
        return {
            "n_features": len(feature_names),
            "features": ", ".join(feature_names),
            "rows_train": 0,
            "rows_test": 0,
            "r2_log": np.nan,
            "rmse_log": np.nan,
            "r2_level": np.nan,
            "mae_level": np.nan,
            "alpha": np.nan,
            "error": "No data",
        }

    model = build_ridge_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    try:
        model.fit(X_train[feature_names], y_train)

        pred_log = model.predict(X_test[feature_names])
        eval_metrics = evaluate_log_and_level(
            y_train=y_train,
            y_test=y_test,
            pred_log=pred_log,
        )

        return {
            "n_features": len(feature_names),
            "features": ", ".join(feature_names),
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
            "r2_log": eval_metrics["r2_log"],
            "rmse_log": eval_metrics["rmse_log"],
            "r2_level": eval_metrics["r2_level"],
            "mae_level": eval_metrics["mae_level"],
            "alpha": float(model.named_steps["ridge"].alpha_),
            "error": None,
        }
    except Exception as exc:
        return {
            "n_features": len(feature_names),
            "features": ", ".join(feature_names),
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
            "r2_log": np.nan,
            "rmse_log": np.nan,
            "r2_level": np.nan,
            "mae_level": np.nan,
            "alpha": np.nan,
            "error": str(exc),
        }


def main() -> None:
    args = parse_args()

    print(f"Reading parcel data: {args.input_csv}")
    # Use the Python parser for resilience with very wide geometry text rows.
    df = pd.read_csv(args.input_csv, engine="python")

    target_col = TARGET_COL
    excluded_columns = {target_col}
    if target_col not in df.columns:
        raise ValueError(f"Required target column missing: {target_col}")

    feature_pool = available_features(df, [*STRUCTURAL_FEATURES, *LOC_NEIGHBORHOOD_FEATURES])
    numeric_pool, categorical_pool = infer_feature_types(df, feature_pool)
    existing_features = [*numeric_pool, *categorical_pool]
    model_df = prepare_model_df(
        df,
        numeric_features=numeric_pool,
        categorical_features=categorical_pool,
        target_col=target_col,
    )

    if len(model_df) < 100:
        raise ValueError("Too few valid rows to estimate model after filtering.")

    _, y = split_features_target(model_df, existing_features, target_col=target_col)

    X_train_idx, X_test_idx = train_test_split_indices(
        row_count=len(model_df),
        test_size=args.test_size,
        random_seed=args.random_seed,
    )
    y_train = y.iloc[X_train_idx].reset_index(drop=True)
    y_test = y.iloc[X_test_idx].reset_index(drop=True)
    X_train_full = model_df.iloc[X_train_idx].reset_index(drop=True)
    X_test_full = model_df.iloc[X_test_idx].reset_index(drop=True)

    structural_features = STRUCTURAL_FEATURES
    locational_neighborhood_features = LOC_NEIGHBORHOOD_FEATURES
    core_structural = [
        "LAND_SF",
        "GROSS_AREA",
        "LIVING_AREA",
        "BED_RMS",
        "FULL_BTH",
        "HLF_BTH",
        "NUM_PARKING",
        "YR_BUILT",
        "YR_REMODEL",
    ]

    specifications = [
        core_structural,
        [*structural_features, "max_far", "max_height", "max_dua", "max_floors"],
        [
            *structural_features,
            *["front_setback", "side_setback", "rear_setback"],
        ],
        core_structural[1:],
        ["LAND_SF", "GROSS_AREA", "LIVING_AREA"],
        [*core_structural, "OVERALL_COND", "zoning_use"],
        [*structural_features, *locational_neighborhood_features],
        [*structural_features, *locational_neighborhood_features, *categorical_pool],
    ]

    results = []
    for i, spec in enumerate(specifications, start=1):
        spec_features = [
            feature
            for feature in spec
            if feature in model_df.columns and feature not in excluded_columns
        ]
        spec_numeric = [feature for feature in spec_features if feature in numeric_pool]
        spec_categorical = [
            feature for feature in spec_features if feature in categorical_pool
        ]

        if not spec_numeric and not spec_categorical:
            print(f"Spec {i}: No available non-leaky features, skipping.")
            continue

        print(
            f"Testing specification {i}/{len(specifications)}: "
            f"{len(spec_numeric) + len(spec_categorical)} features"
        )
        metrics = fit_and_evaluate(
            X_train_full,
            X_test_full,
            y_train,
            y_test,
            numeric_features=spec_numeric,
            categorical_features=spec_categorical,
        )
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("r2_log", ascending=False).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)

    print(f"\nComparison results written: {args.output_csv}")
    print("\nTop models by R2 (log):")
    print(
        results_df[
            ["n_features", "r2_log", "rmse_log", "r2_level", "alpha", "features"]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
