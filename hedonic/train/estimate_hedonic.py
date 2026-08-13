"""Train a residential ridge hedonic model and write artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from hedonic.common.modeling_common import (
    LOC_NEIGHBORHOOD_FEATURES,
    STRUCTURAL_FEATURES,
    TARGET_COL,
    available_features,
    build_ridge_pipeline,
    cross_validate_log_and_level,
    evaluate_log_and_level,
    extract_coefficients,
    get_ridge_alpha,
    infer_feature_types,
    prepare_model_df,
    require_existing_path,
    split_features_target,
    subset_residential_rows,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Estimate a residential hedonic model from parcel-level assessor data."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root / "parcels_preprocessed.csv",
        help="Path to parcel-level input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "hedonic" / "artifacts",
        help="Directory where residential hedonic artifacts are written.",
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
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Optional K-fold cross-validation count (set >=2 to enable).",
    )
    parser.add_argument(
        "--feature-list",
        type=str,
        default=None,
        help=(
            "Optional comma-separated feature list to train on. "
            "Example: LAND_SF,neighborhood_walkability,emp_dist_m,INT_COND"
        ),
    )
    parser.add_argument(
        "--feature-spec-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing selected features under a 'features' key "
            "or as a top-level string array."
        ),
    )

    args = parser.parse_args()
    args.input_csv = require_existing_path(args.input_csv, "Parcel CSV")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.feature_spec_json is not None:
        args.feature_spec_json = require_existing_path(args.feature_spec_json, "Feature spec JSON")
    return args


def _parse_feature_spec_json(path: Path) -> list[str]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, dict) and isinstance(raw.get("features"), list):
        values = raw["features"]
    else:
        raise ValueError(
            "Feature spec JSON must be a list of feature names or an object with a 'features' list."
        )

    parsed = [str(value).strip() for value in values if str(value).strip()]
    if not parsed:
        raise ValueError("Feature spec JSON did not contain any feature names.")
    return parsed


def resolve_feature_set(df: pd.DataFrame, args: argparse.Namespace) -> tuple[list[str], str]:
    if args.feature_list:
        selected = [value.strip() for value in args.feature_list.split(",") if value.strip()]
        if not selected:
            raise ValueError("--feature-list was provided but no feature names were parsed.")
        source = "--feature-list"
    elif args.feature_spec_json is not None:
        selected = _parse_feature_spec_json(args.feature_spec_json)
        source = f"--feature-spec-json ({args.feature_spec_json})"
    else:
        candidate_features = [*STRUCTURAL_FEATURES, *LOC_NEIGHBORHOOD_FEATURES]
        selected = candidate_features
        source = "default feature pools (STRUCTURAL_FEATURES + LOC_NEIGHBORHOOD_FEATURES)"

    existing_features = available_features(df, selected)
    if not existing_features:
        raise ValueError(
            "No selected features were found in the input CSV. "
            f"Requested: {selected}"
        )

    missing = [feature for feature in selected if feature not in existing_features]
    if missing:
        print(f"Warning: selected features not found in input and excluded: {missing}")

    return existing_features, source


def main() -> None:
    args = parse_args()

    print(f"Reading parcel data: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)
    before_filter = len(df)
    df = subset_residential_rows(df, strict=True)
    print(f"Residential subset rows: {len(df)} of {before_filter}")

    target_col = TARGET_COL
    if target_col not in df.columns:
        raise ValueError(f"Required target column missing: {target_col}")

    existing_features, feature_source = resolve_feature_set(df, args)
    numeric_features, categorical_features = infer_feature_types(df, existing_features)
    all_features = [*numeric_features, *categorical_features]
    if not all_features:
        raise ValueError("No model features found in input CSV.")

    print(f"Using feature source: {feature_source}")
    print("Using features:", ", ".join(all_features))

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

    print("Training residential hedonic model...")
    model.fit(X_train, y_train)

    pred_log = model.predict(X_test)
    eval_metrics = evaluate_log_and_level(y_train=y_train, y_test=y_test, pred_log=pred_log)

    metrics = {
        "rows_total": int(len(model_df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "target": target_col,
        "feature_source": feature_source,
        "selected_features": all_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "r2_log": eval_metrics["r2_log"],
        "rmse_log": eval_metrics["rmse_log"],
        "r2_level": eval_metrics["r2_level"],
        "rmse_level": eval_metrics["rmse_level"],
        "mae_level": eval_metrics["mae_level"],
        "alpha": get_ridge_alpha(model),
    }

    if args.cv_folds >= 2:
        print(f"Running {args.cv_folds}-fold cross-validation...")
        cv_results = cross_validate_log_and_level(
            model=build_ridge_pipeline(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            ),
            X=X,
            y=y,
            cv_folds=args.cv_folds,
            random_seed=args.random_seed,
        )
        metrics["cv"] = cv_results

    coef_df = extract_coefficients(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "residential_hedonic_model.joblib"
    metrics_path = args.output_dir / "residential_hedonic_metrics.json"
    cv_metrics_path = args.output_dir / "residential_hedonic_cv_metrics.json"
    coefficients_path = args.output_dir / "residential_hedonic_coefficients.csv"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if "cv" in metrics:
        cv_metrics_path.write_text(json.dumps(metrics["cv"], indent=2), encoding="utf-8")
    coef_df.to_csv(coefficients_path, index=False)

    print(f"Model written: {model_path}")
    print(f"Metrics written: {metrics_path}")
    if "cv" in metrics:
        print(f"CV metrics written: {cv_metrics_path}")
    print(f"Coefficients written: {coefficients_path}")
    print(f"Holdout R2 (log): {metrics['r2_log']:.4f}")
    print(f"Holdout RMSE (level): {metrics['rmse_level']:.2f}")


if __name__ == "__main__":
    main()
