"""Run benchmark OLS, cross-validated LASSO selection, and final OLS inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hedonic.common.modeling_common import (
    DEFAULT_FEATURE_SET,
    LOG_PRICE_PER_SQFT_COL,
    TARGET_COL,
    available_features,
    infer_feature_types,
    prepare_price_per_sqft_model_df,
    prepare_model_df,
    subset_residential_rows,
)
from shared_utils import require_existing_path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Three-step hedonic pipeline: "
            "(A) benchmark OLS, (B) K-fold CV LASSO selection, (C) final OLS inference."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels_processed_for_hedonic.csv",
        help="Parcel CSV used for modeling.",
    )
    parser.add_argument(
        "--feature-list",
        type=str,
        default=None,
        help="Optional comma-separated feature list.",
    )
    parser.add_argument(
        "--feature-spec-json",
        type=Path,
        default=None,
        help="Optional feature spec JSON with a top-level list or a 'features' key.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Evaluation-set share held out before any feature selection.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for LASSO cross-validation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for splits/CV reproducibility.",
    )
    parser.add_argument(
        "--lasso-max-iter",
        type=int,
        default=20000,
        help="Max iterations for LassoCV.",
    )
    parser.add_argument(
        "--coef-threshold",
        type=float,
        default=1e-1,
        help="Absolute coefficient threshold for considering a transformed feature non-zero.",
    )
    parser.add_argument(
        "--min-categorical-dummies",
        type=int,
        default=2,
        help=(
            "Minimum number of one-hot dummy coefficients for a categorical base feature "
            "that must exceed --coef-threshold to keep that base feature."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "hedonic" / "artifacts",
        help="Output directory for pipeline artifacts.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional row sample size for faster test runs (0 uses all rows).",
    )

    args = parser.parse_args()
    args.input_csv = require_existing_path(args.input_csv, "Input CSV")
    if args.feature_spec_json is not None:
        args.feature_spec_json = require_existing_path(args.feature_spec_json, "Feature spec JSON")
    args.output_dir = args.output_dir.expanduser().resolve()

    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2")
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test-size must be between 0 and 1")
    if args.lasso_max_iter < 1000:
        raise ValueError("--lasso-max-iter should be >= 1000")
    if args.coef_threshold <= 0:
        raise ValueError("--coef-threshold must be > 0")
    if args.min_categorical_dummies < 1:
        raise ValueError("--min-categorical-dummies must be >= 1")
    if args.sample_size < 0:
        raise ValueError("--sample-size must be >= 0")

    return args


def _parse_feature_spec_json(path: Path) -> list[str]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, dict) and isinstance(raw.get("features"), list):
        values = raw["features"]
    else:
        raise ValueError("Feature spec JSON must be a list or an object with a 'features' list.")

    parsed = [str(value).strip() for value in values if str(value).strip()]
    if not parsed:
        raise ValueError("Feature spec JSON did not contain any feature names.")
    return parsed


def _resolve_feature_set(df: pd.DataFrame, args: argparse.Namespace) -> tuple[list[str], str]:
    if args.feature_list:
        selected = [value.strip() for value in args.feature_list.split(",") if value.strip()]
        source = "--feature-list"
    elif args.feature_spec_json is not None:
        selected = _parse_feature_spec_json(args.feature_spec_json)
        source = f"--feature-spec-json ({args.feature_spec_json})"
    else:
        selected = list(DEFAULT_FEATURE_SET)
        source = "DEFAULT_FEATURE_SET"

    existing = available_features(df, selected)
    if not existing:
        raise ValueError("None of the requested features exist in the input data.")

    missing = [feature for feature in selected if feature not in existing]
    if missing:
        print(f"Warning: excluded missing features: {missing}")

    return existing, source


def _build_benchmark_ols_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
                    ]
                ),
                categorical_features,
            )
        )

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(steps=[("preprocess", preprocess), ("ols", LinearRegression())])


def _build_lasso_pipeline(numeric_features: list[str], categorical_features: list[str], cv_folds: int, random_seed: int, lasso_max_iter: int) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                categorical_features,
            )
        )

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    lasso = LassoCV(cv=cv, max_iter=lasso_max_iter, n_jobs=1)

    return Pipeline(steps=[("preprocess", preprocess), ("lasso", lasso)])


def _evaluate_log_and_level(y_train_log: np.ndarray, y_test_log: np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    pred_log_for_level = np.clip(pred_log, float(np.min(y_train_log)), float(np.max(y_train_log)))
    pred_level = np.expm1(pred_log_for_level)
    y_test_level = np.expm1(y_test_log)

    return {
        "r2_log": float(r2_score(y_test_log, pred_log)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_test_log, pred_log))),
        "r2_level": float(r2_score(y_test_level, pred_level)),
        "rmse_level": float(np.sqrt(mean_squared_error(y_test_level, pred_level))),
        "mae_level": float(mean_absolute_error(y_test_level, pred_level)),
    }


def _infer_base_feature(transformed_name: str, numeric_features: list[str], categorical_features: list[str]) -> str | None:
    if transformed_name.startswith("num__"):
        candidate = transformed_name.replace("num__", "", 1)
        return candidate if candidate in numeric_features else None

    if transformed_name.startswith("cat__"):
        raw = transformed_name.replace("cat__", "", 1)
        for column in categorical_features:
            prefix = f"{column}_"
            if raw == column or raw.startswith(prefix):
                return column
        return None

    return None


def _select_features_with_lasso(
    model_df: pd.DataFrame,
    feature_list: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    cv_folds: int,
    random_seed: int,
    lasso_max_iter: int,
    coef_threshold: float,
    min_categorical_dummies: int,
) -> dict[str, Any]:
    X = model_df[[*numeric_features, *categorical_features]]
    y_log = np.log1p(model_df[TARGET_COL].to_numpy(dtype=float))

    pipeline = _build_lasso_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        cv_folds=cv_folds,
        random_seed=random_seed,
        lasso_max_iter=lasso_max_iter,
    )
    pipeline.fit(X, y_log)

    preprocess = pipeline.named_steps["preprocess"]
    lasso = pipeline.named_steps["lasso"]
    transformed_names = [str(name) for name in preprocess.get_feature_names_out()]
    coefs = np.asarray(lasso.coef_, dtype=float)

    surviving_transformed: list[dict[str, Any]] = []
    surviving_numeric_base: set[str] = set()
    categorical_surviving_counts: dict[str, int] = {}

    for name, coef in zip(transformed_names, coefs):
        coef_value = float(coef)
        if abs(coef_value) <= coef_threshold:
            continue

        base = _infer_base_feature(name, numeric_features, categorical_features)
        surviving_transformed.append(
            {
                "transformed_feature": name,
                "coefficient": coef_value,
                "base_feature": base,
            }
        )
        if base is None:
            continue
        if base in numeric_features:
            surviving_numeric_base.add(base)
        elif base in categorical_features:
            categorical_surviving_counts[base] = categorical_surviving_counts.get(base, 0) + 1

    selected_features: list[str] = []
    for feature in feature_list:
        if feature in numeric_features and feature in surviving_numeric_base:
            selected_features.append(feature)
            continue
        if feature in categorical_features and categorical_surviving_counts.get(feature, 0) >= min_categorical_dummies:
            selected_features.append(feature)

    if not selected_features:
        raise ValueError(
            "No features survived selection with the current thresholds. "
            f"Try lowering --coef-threshold or --min-categorical-dummies (current: {min_categorical_dummies})."
        )

    return {
        "alpha": float(lasso.alpha_),
        "cv_folds": int(cv_folds),
        "coef_threshold": float(coef_threshold),
        "min_categorical_dummies": int(min_categorical_dummies),
        "selected_features": selected_features,
        "surviving_transformed_features": surviving_transformed,
        "categorical_surviving_counts": categorical_surviving_counts,
        "n_selected_base_features": int(len(selected_features)),
        "n_surviving_transformed_features": int(len(surviving_transformed)),
    }


def _build_statsmodels_design(
    model_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    if numeric_features:
        numeric_df = model_df[numeric_features].apply(pd.to_numeric, errors="coerce")
        for column in numeric_df.columns:
            numeric_df[column] = numeric_df[column].fillna(numeric_df[column].median())
        parts.append(numeric_df.astype(float))

    if categorical_features:
        cat_df = model_df[categorical_features].copy()
        for column in cat_df.columns:
            mode = cat_df[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "MISSING"
            cat_df[column] = cat_df[column].fillna(fill_value).astype(str)
        cat_dummies = pd.get_dummies(cat_df, drop_first=True, dtype=float)
        parts.append(cat_dummies)

    if not parts:
        raise ValueError("No design matrix columns were created for final OLS.")

    X = pd.concat(parts, axis=1)
    X = sm.add_constant(X, has_constant="add")
    return X


def _final_ols_inference(model_df: pd.DataFrame, selected_features: list[str]) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    numeric_selected, categorical_selected = infer_feature_types(model_df, selected_features)
    X = _build_statsmodels_design(model_df, numeric_selected, categorical_selected)
    if LOG_PRICE_PER_SQFT_COL in model_df.columns:
        y_log = model_df[LOG_PRICE_PER_SQFT_COL].to_numpy(dtype=float)
    else:
        y_log = np.log1p(model_df[TARGET_COL].to_numpy(dtype=float))

    results = sm.OLS(y_log, X).fit()

    conf_int = results.conf_int()
    coef_table = pd.DataFrame(
        {
            "term": results.params.index,
            "coefficient_log": results.params.values,
            "std_error": results.bse.values,
            "t_stat": results.tvalues.values,
            "p_value": results.pvalues.values,
            "ci_low": conf_int.iloc[:, 0].values,
            "ci_high": conf_int.iloc[:, 1].values,
        }
    )
    coef_table["pct_effect_1_unit"] = np.expm1(coef_table["coefficient_log"]) * 100.0

    return results, coef_table


def main() -> None:
    args = parse_args()

    print(f"Reading parcel data: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)
    raw_rows = len(df)
    df = subset_residential_rows(df, strict=True)
    print(f"Residential subset rows: {len(df)} of {raw_rows}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Required target column missing: {TARGET_COL}")

    feature_list, feature_source = _resolve_feature_set(df, args)
    numeric_features, categorical_features = infer_feature_types(df, feature_list)

    model_df = prepare_price_per_sqft_model_df(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_value_col=TARGET_COL,
    )
    if args.sample_size > 0 and args.sample_size < len(model_df):
        model_df = model_df.sample(n=args.sample_size, random_state=args.random_seed)
    if model_df.empty:
        raise ValueError("No rows available after model data preparation.")

    discovery_idx, evaluation_idx = train_test_split(
        np.arange(len(model_df)),
        test_size=args.test_size,
        random_state=args.random_seed,
    )
    discovery_df = model_df.iloc[discovery_idx].copy()
    evaluation_df = model_df.iloc[evaluation_idx].copy()

    X_discovery = discovery_df[[*numeric_features, *categorical_features]]
    y_discovery_log = discovery_df[LOG_PRICE_PER_SQFT_COL].to_numpy(dtype=float)
    X_evaluation = evaluation_df[[*numeric_features, *categorical_features]]
    y_evaluation_log = evaluation_df[LOG_PRICE_PER_SQFT_COL].to_numpy(dtype=float)

    # Step A: benchmark OLS on discovery variables, evaluated on untouched evaluation set.
    benchmark_model = _build_benchmark_ols_pipeline(numeric_features, categorical_features)
    benchmark_model.fit(X_discovery, y_discovery_log)
    benchmark_pred_log = benchmark_model.predict(X_evaluation)
    benchmark_metrics = _evaluate_log_and_level(
        y_train_log=y_discovery_log,
        y_test_log=y_evaluation_log,
        pred_log=benchmark_pred_log,
    )

    # Step B: K-fold CV LASSO chooses alpha using discovery set only.
    lasso_selection = _select_features_with_lasso(
        model_df=discovery_df,
        feature_list=feature_list,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        cv_folds=args.cv_folds,
        random_seed=args.random_seed,
        lasso_max_iter=args.lasso_max_iter,
        coef_threshold=args.coef_threshold,
        min_categorical_dummies=args.min_categorical_dummies,
    )
    selected_features = lasso_selection["selected_features"]

    selected_numeric, selected_categorical = infer_feature_types(discovery_df, selected_features)
    selected_discovery_df = discovery_df[[*selected_numeric, *selected_categorical, LOG_PRICE_PER_SQFT_COL]].copy()
    selected_evaluation_df = evaluation_df[[*selected_numeric, *selected_categorical, LOG_PRICE_PER_SQFT_COL]].copy()

    X_sel_discovery = selected_discovery_df[[*selected_numeric, *selected_categorical]]
    y_sel_discovery_log = selected_discovery_df[LOG_PRICE_PER_SQFT_COL].to_numpy(dtype=float)
    X_sel_evaluation = selected_evaluation_df[[*selected_numeric, *selected_categorical]]
    y_sel_evaluation_log = selected_evaluation_df[LOG_PRICE_PER_SQFT_COL].to_numpy(dtype=float)

    selected_benchmark_model = _build_benchmark_ols_pipeline(selected_numeric, selected_categorical)
    selected_benchmark_model.fit(X_sel_discovery, y_sel_discovery_log)
    selected_pred_log = selected_benchmark_model.predict(X_sel_evaluation)
    selected_evaluation_metrics = _evaluate_log_and_level(
        y_train_log=y_sel_discovery_log,
        y_test_log=y_sel_evaluation_log,
        pred_log=selected_pred_log,
    )

    # Step C: final standard OLS inference on discovery set with LASSO-selected variables.
    final_results, coef_table = _final_ols_inference(selected_discovery_df, selected_features)

    pipeline_summary = {
        "input_csv": str(args.input_csv),
        "rows_residential": int(len(df)),
        "rows_model": int(len(model_df)),
        "rows_discovery": int(len(discovery_df)),
        "rows_evaluation": int(len(evaluation_df)),
        "sample_size": int(args.sample_size),
        "target": TARGET_COL,
        "model_target": LOG_PRICE_PER_SQFT_COL,
        "log_transformed_numeric_features": [
            feature for feature in numeric_features if feature.upper().endswith(("_SF", "_AREA")) or feature.upper() == "LIVING_AREA"
        ],
        "feature_source": feature_source,
        "candidate_features": feature_list,
        "split_design": {
            "evaluation_share": float(args.test_size),
            "leakage_rule": "feature_selection_and_inference_use_discovery_only",
        },
        "step_a_benchmark_ols": {
            "description": "Benchmark OLS trained on discovery and evaluated on held-out evaluation set",
            "metrics": benchmark_metrics,
        },
        "step_b_lasso_cv": lasso_selection,
        "step_c_final_ols": {
            "description": "Final OLS inference on discovery set using LASSO-selected variables",
            "selected_features": selected_features,
            "nobs": int(final_results.nobs),
            "r_squared": float(final_results.rsquared),
            "adj_r_squared": float(final_results.rsquared_adj),
            "f_statistic": float(final_results.fvalue) if final_results.fvalue is not None else None,
            "f_pvalue": float(final_results.f_pvalue) if final_results.f_pvalue is not None else None,
            "aic": float(final_results.aic),
            "bic": float(final_results.bic),
            "evaluation_metrics_selected_subset": selected_evaluation_metrics,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "residential_hedonic_lasso_ols_pipeline.json"
    selected_features_path = args.output_dir / "residential_hedonic_lasso_selected_features.json"
    coefficients_path = args.output_dir / "residential_hedonic_final_ols_coefficients.csv"

    summary_path.write_text(json.dumps(pipeline_summary, indent=2), encoding="utf-8")
    selected_features_path.write_text(
        json.dumps(
            {
                "features": selected_features,
                "selected_from": feature_list,
                "selection": {
                    "method": "LassoCV",
                    "alpha": lasso_selection["alpha"],
                    "cv_folds": lasso_selection["cv_folds"],
                    "coef_threshold": lasso_selection["coef_threshold"],
                    "min_categorical_dummies": lasso_selection["min_categorical_dummies"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    coef_table.to_csv(coefficients_path, index=False)

    print(f"Pipeline summary written: {summary_path}")
    print(f"Selected feature spec written: {selected_features_path}")
    print(f"Final OLS coefficients written: {coefficients_path}")
    print("Step A benchmark OLS evaluation RMSE (level): " f"{benchmark_metrics['rmse_level']:.2f}")
    print("Step C selected-subset evaluation RMSE (level): " f"{selected_evaluation_metrics['rmse_level']:.2f}")


if __name__ == "__main__":
    main()
