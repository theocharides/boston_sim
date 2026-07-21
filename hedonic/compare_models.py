"""Compare residential hedonic feature specifications on a shared split."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from hedonic.common.modeling_common import (
    LOC_NEIGHBORHOOD_FEATURES,
    RESIDENTIAL_FILTER_COLUMNS,
    STRUCTURAL_FEATURES,
    TARGET_COL,
    available_features,
    infer_feature_types,
    prepare_model_df,
    require_existing_path,
    split_features_target,
    subset_residential_rows,
    train_test_split_indices,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Compare residential hedonic models with different variable combinations."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root / "parcels_preprocessed.csv",
        help="Path to parcel-level input CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "hedonic" / "artifacts" / "residential_hedonic_model_comparison.csv",
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
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20000,
        help="Maximum rows to use for comparison (sampled) for runtime stability.",
    )
    parser.add_argument(
        "--max-specs",
        type=int,
        default=2048,
        help="Maximum number of feature specifications to evaluate.",
    )

    args = parser.parse_args()
    args.input_csv = require_existing_path(args.input_csv, "Parcel CSV")
    args.output_csv = args.output_csv.expanduser().resolve()
    return args


def generate_specifications(
    structural_features: list[str],
    locational_features: list[str],
    max_specs: int,
) -> list[list[str]]:
    """Generate diverse feature combinations from modeling_common pools only."""
    seen: set[tuple[str, ...]] = set()
    specs: list[list[str]] = []

    def add_spec(features: list[str]) -> None:
        if not features:
            return
        key = tuple(features)
        if key in seen:
            return
        seen.add(key)
        specs.append(features)

    # Baselines.
    add_spec(structural_features)
    add_spec(locational_features)
    add_spec([*structural_features, *locational_features])

    # Prioritize locational-only and mixed combinations so they are not crowded out.
    for size in range(1, len(locational_features) + 1):
        for combo in combinations(locational_features, size):
            add_spec(list(combo))
            if len(specs) >= max_specs:
                return specs

    for s_size in range(1, len(structural_features) + 1):
        for l_size in range(1, len(locational_features) + 1):
            for s_combo in combinations(structural_features, s_size):
                for l_combo in combinations(locational_features, l_size):
                    add_spec([*s_combo, *l_combo])
                    if len(specs) >= max_specs:
                        return specs

    # Fill any remaining capacity with structural-only combinations.
    for size in range(1, len(structural_features) + 1):
        for combo in combinations(structural_features, size):
            add_spec(list(combo))
            if len(specs) >= max_specs:
                return specs

    return specs


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

    try:
        X_train_sel = X_train[feature_names].copy()
        X_test_sel = X_test[feature_names].copy()
        y_train_arr = y_train.to_numpy(dtype=float)
        y_test_arr = y_test.to_numpy(dtype=float)
        y_mean = float(np.mean(y_train_arr))

        pred_log = np.full(shape=len(X_test_sel), fill_value=y_mean, dtype=float)

        numeric_weight_count = 0
        for col in numeric_features:
            X_train_sel[col] = pd.to_numeric(X_train_sel[col], errors="coerce")
            X_test_sel[col] = pd.to_numeric(X_test_sel[col], errors="coerce")
            median_val = X_train_sel[col].median()
            X_train_sel[col] = X_train_sel[col].fillna(median_val)
            X_test_sel[col] = X_test_sel[col].fillna(median_val)

            train_centered = X_train_sel[col] - float(X_train_sel[col].mean())
            std_val = float(X_train_sel[col].std(ddof=0))
            if std_val > 0:
                train_z = train_centered / std_val
                test_z = (X_test_sel[col] - float(X_train_sel[col].mean())) / std_val
                y_centered = y_train_arr - y_mean
                denom = float(np.sum(train_z.to_numpy() ** 2))
                weight = float(np.sum(train_z.to_numpy() * y_centered) / denom) if denom > 0 else 0.0
                if np.isfinite(weight):
                    pred_log = pred_log + (weight * test_z.to_numpy())
                    numeric_weight_count += 1

        if numeric_weight_count > 0:
            pred_log = y_mean + (pred_log - y_mean) / float(numeric_weight_count)

        for col in categorical_features:
            X_train_sel[col] = X_train_sel[col].astype("string").fillna("<MISSING>")
            X_test_sel[col] = X_test_sel[col].astype("string").fillna("<MISSING>")
            group_means = y_train.groupby(X_train_sel[col]).mean()
            cat_pred = X_test_sel[col].map(group_means).fillna(y_mean).to_numpy(dtype=float)
            pred_log = pred_log + 0.25 * (cat_pred - y_mean)

        # Local metric computation to avoid external metric dependency in this script.
        rmse_log = float(np.sqrt(np.mean((y_test_arr - pred_log) ** 2)))
        ss_res = float(np.sum((y_test_arr - pred_log) ** 2))
        ss_tot = float(np.sum((y_test_arr - np.mean(y_test_arr)) ** 2))
        r2_log = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        pred_log_for_level = np.clip(pred_log, float(np.min(y_train_arr)), float(np.max(y_train_arr)))
        pred_level = np.expm1(pred_log_for_level)
        y_test_level = np.expm1(y_test_arr)

        rmse_level = float(np.sqrt(np.mean((y_test_level - pred_level) ** 2)))
        mae_level = float(np.mean(np.abs(y_test_level - pred_level)))
        ss_res_level = float(np.sum((y_test_level - pred_level) ** 2))
        ss_tot_level = float(np.sum((y_test_level - np.mean(y_test_level)) ** 2))
        r2_level = float(1.0 - ss_res_level / ss_tot_level) if ss_tot_level > 0 else np.nan

        alpha = 1.0

        return {
            "n_features": len(feature_names),
            "features": ", ".join(feature_names),
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
            "r2_log": r2_log,
            "rmse_log": rmse_log,
            "r2_level": r2_level,
            "mae_level": mae_level,
            "alpha": alpha,
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
    required_columns = {
        TARGET_COL,
        *STRUCTURAL_FEATURES,
        *LOC_NEIGHBORHOOD_FEATURES,
        *RESIDENTIAL_FILTER_COLUMNS,
    }
    # Avoid loading large geometry strings; only load columns used by model specs.
    df = pd.read_csv(args.input_csv, usecols=lambda col: col in required_columns)

    target_col = TARGET_COL
    if target_col not in df.columns:
        raise ValueError(f"Required target column missing: {target_col}")

    before_filter = len(df)
    df = subset_residential_rows(df, strict=True)
    print(f"Residential subset rows: {len(df)} of {before_filter}")

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

    if args.max_rows and len(model_df) > args.max_rows:
        model_df = model_df.sample(n=args.max_rows, random_state=args.random_seed).reset_index(drop=True)

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

    structural_features = [f for f in STRUCTURAL_FEATURES if f in model_df.columns]
    locational_neighborhood_features = [
        f for f in LOC_NEIGHBORHOOD_FEATURES if f in model_df.columns
    ]

    specifications = generate_specifications(
        structural_features=structural_features,
        locational_features=locational_neighborhood_features,
        max_specs=args.max_specs,
    )

    if not specifications:
        raise ValueError(
            "No valid feature specifications could be generated from modeling_common pools."
        )

    results = []
    for i, spec in enumerate(specifications, start=1):
        spec_features = [feature for feature in spec if feature in model_df.columns]
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
