"""
Estimate a hedonic price model from the parcels table.

This script fits a regularized linear regression model to predict parcel
assessed value using structural and locational/neighborhood proxy variables.

It estimates one combined specification that uses all available features from
the shared structural and locational/neighborhood pools.

It writes model artifacts, fit metrics, and coefficients to hedonic/artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from modeling_common import (
    LOC_NEIGHBORHOOD_FEATURES,
    TARGET_COL,
    STRUCTURAL_FEATURES,
    available_features,
    build_ridge_pipeline,
    evaluate_log_and_level,
    extract_coefficients,
    infer_feature_types,
    prepare_model_df,
    require_existing_path,
    split_features_target,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Estimate a hedonic model from parcel-level assessor data."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels.csv",
        help="Path to parcel-level input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "hedonic" / "artifacts",
        help="Directory where model artifacts are written.",
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
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()

    print(f"Reading parcel data: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)

    target_col = TARGET_COL
    if target_col not in df.columns:
        raise ValueError(f"Required target column missing: {target_col}")

    candidate_features = [*STRUCTURAL_FEATURES, *LOC_NEIGHBORHOOD_FEATURES]
    existing_features = available_features(df, candidate_features)
    numeric_features, categorical_features = infer_feature_types(df, existing_features)
    all_features = [*numeric_features, *categorical_features]
    if not all_features:
        raise ValueError("No model features found in input CSV.")

    model_df = prepare_model_df(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_col=target_col,
    )
    if len(model_df) < 100:
        raise ValueError("Too few valid rows to estimate model after filtering.")

    X, y = split_features_target(model_df, all_features, target_col=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    model = build_ridge_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    print("Training hedonic model...")
    model.fit(X_train, y_train)

    pred_log = model.predict(X_test)
    eval_metrics = evaluate_log_and_level(y_train=y_train, y_test=y_test, pred_log=pred_log)

    metrics = {
        "rows_total": int(len(model_df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "target": target_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "r2_log": eval_metrics["r2_log"],
        "rmse_log": eval_metrics["rmse_log"],
        "r2_level": eval_metrics["r2_level"],
        "rmse_level": eval_metrics["rmse_level"],
        "mae_level": eval_metrics["mae_level"],
        "alpha": float(model.named_steps["ridge"].alpha_),
    }

    coef_df = extract_coefficients(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "hedonic_model.joblib"
    metrics_path = args.output_dir / "hedonic_metrics.json"
    coefficients_path = args.output_dir / "hedonic_coefficients.csv"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    coef_df.to_csv(coefficients_path, index=False)

    print(f"Model written: {model_path}")
    print(f"Metrics written: {metrics_path}")
    print(f"Coefficients written: {coefficients_path}")
    print(f"Holdout R2 (log): {metrics['r2_log']:.4f}")
    print(f"Holdout RMSE (level): {metrics['rmse_level']:.2f}")


if __name__ == "__main__":
    main()
