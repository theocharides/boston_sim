"""Run a cross-validation evaluation for a specified hedonic feature set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from hedonic.common.modeling_common import (
    DEFAULT_FEATURE_SET,
    TARGET_COL,
    available_features,
    build_ridge_pipeline,
    cross_validate_log_and_level,
    infer_feature_types,
    prepare_model_df,
    subset_residential_rows,
)
from shared_utils import require_existing_path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run K-fold cross-validation for a hedonic feature configuration, without using a tests folder."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels_processed_for_hedonic.csv",
        help="Parcel CSV to validate against.",
    )
    parser.add_argument(
        "--feature-list",
        type=str,
        default=None,
        help="Optional comma-separated feature list to evaluate with K-fold CV.",
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
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds to use for cross-validation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for fold splitting.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=repo_root / "hedonic" / "workflow" / "residential_hedonic_cv_metrics.json",
        help="Path to save the CV summary JSON.",
    )
    args = parser.parse_args()
    args.input_csv = require_existing_path(args.input_csv, "Input CSV")
    if args.feature_spec_json is not None:
        args.feature_spec_json = require_existing_path(args.feature_spec_json, "Feature spec JSON")
    args.output_json = args.output_json.expanduser().resolve()
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


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv, low_memory=False)
    df = subset_residential_rows(df, strict=True)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Required target column missing: {TARGET_COL}")

    if args.feature_list:
        feature_list = [value.strip() for value in args.feature_list.split(",") if value.strip()]
        feature_source = "--feature-list"
    elif args.feature_spec_json is not None:
        feature_list = _parse_feature_spec_json(args.feature_spec_json)
        feature_source = f"--feature-spec-json ({args.feature_spec_json})"
    else:
        feature_list = list(DEFAULT_FEATURE_SET)
        feature_source = "DEFAULT_FEATURE_SET"

    feature_list = available_features(df, feature_list)
    if not feature_list:
        raise ValueError("None of the requested features existed in the input data.")

    numeric_features, categorical_features = infer_feature_types(df, feature_list)
    model_df = prepare_model_df(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_col=TARGET_COL,
    )

    X = model_df[[*numeric_features, *categorical_features]]
    y = model_df[TARGET_COL].pipe(lambda s: pd.Series(__import__("numpy").log1p(s), index=s.index))

    model = build_ridge_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    cv_results = cross_validate_log_and_level(
        model=model,
        X=X,
        y=y,
        cv_folds=args.cv_folds,
        random_seed=args.random_seed,
    )
    cv_results["selected_features"] = feature_list
    cv_results["numeric_features"] = numeric_features
    cv_results["categorical_features"] = categorical_features

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(cv_results, indent=2), encoding="utf-8")

    print(f"Cross-validation results written to: {args.output_json}")
    print(f"Feature source: {feature_source}")
    print(f"Selected features: {', '.join(feature_list)}")
    print(json.dumps(cv_results["summary"], indent=2))


if __name__ == "__main__":
    main()
