"""Shared hedonic modeling utilities used by training and analysis scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COL = "TOTAL_VALUE"

RESIDENTIAL_FILTER_COLUMNS = [
    "LU",
]

RESIDENTIAL_LU_CODES = {
    "A",   # Residential 7 or more units
    "CD",  # Residential condominium unit
    "CM",  # Condominium main
    "R1",  # Residential 1-family
    "R2",  # Residential 2-family
    "R3",  # Residential 3-family
    "R4",  # Residential 4 or more family
    "RC",  # Mixed use (residential and commercial)
    "RL",  # Residential land
}

STRUCTURAL_FEATURES = [
    "LAND_SF", 
    "INT_COND",
    "LU"
]

LOC_NEIGHBORHOOD_FEATURES = [
    "neighborhood_walkability",
    "emp_dist_m"
]

DEFAULT_FEATURE_SET = [*STRUCTURAL_FEATURES, *LOC_NEIGHBORHOOD_FEATURES]

def available_features(df: pd.DataFrame, feature_pool: list[str]) -> list[str]:
    """Return features from feature_pool that exist in the dataframe."""
    return [feature for feature in feature_pool if feature in df.columns]


def subset_residential_rows(
    df: pd.DataFrame,
    strict: bool = True,
) -> pd.DataFrame:
    """Return only rows with residential LU codes."""
    available_filter_cols = [
        col for col in RESIDENTIAL_FILTER_COLUMNS if col in df.columns
    ]

    if not available_filter_cols:
        if strict:
            raise ValueError(
                "Cannot subset to residential rows: none of "
                f"{RESIDENTIAL_FILTER_COLUMNS} found in input data."
            )
        return df.copy()

    lu_col = available_filter_cols[0]
    lu_codes = (
        df[lu_col]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
    )
    residential_mask = lu_codes.isin(RESIDENTIAL_LU_CODES)

    residential_df = df.loc[residential_mask].copy()
    if strict and residential_df.empty:
        raise ValueError(
            "Residential filter removed all rows. Check assessor type values in "
            f"{available_filter_cols}."
        )

    return residential_df


def infer_feature_types(
    df: pd.DataFrame,
    feature_names: list[str],
    min_numeric_share: float = 0.95,
) -> tuple[list[str], list[str]]:
    """Infer numeric vs categorical features from observed values."""
    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for feature in feature_names:
        series = df[feature]
        non_missing = int(series.notna().sum())

        if non_missing == 0:
            # Default all-missing columns to numeric; they will be imputed/dropped safely.
            numeric_features.append(feature)
            continue

        numeric_cast = pd.to_numeric(series, errors="coerce")
        numeric_share = float(numeric_cast.notna().sum()) / float(non_missing)

        if numeric_share >= min_numeric_share:
            numeric_features.append(feature)
        else:
            categorical_features.append(feature)

    return numeric_features, categorical_features


def prepare_model_df(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str = TARGET_COL,
    apply_quality_filters: bool = False,
) -> pd.DataFrame:
    """Prepare numeric/categorical features and filter to valid target rows."""

    model_cols = [*numeric_features, *categorical_features, target_col]
    model_df = df[model_cols].copy()

    for col in [*numeric_features, target_col]:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    for col in categorical_features:
        as_text = model_df[col].astype("object")
        as_text = as_text.where(pd.notna(as_text), np.nan)
        as_text = as_text.map(lambda value: value.strip() if isinstance(value, str) else value)
        model_df[col] = as_text.replace({"": np.nan, "nan": np.nan, "None": np.nan})

    model_df = model_df[model_df[target_col].notna() & (model_df[target_col] > 0)]
    return model_df


def apply_hedonic_value_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility shim: quality filtering is disabled and data passes through unchanged."""
    return df.copy()


def split_features_target(
    model_df: pd.DataFrame,
    feature_names: list[str],
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and log-transformed target y."""
    X = model_df[feature_names]
    y = np.log1p(model_df[target_col])
    return X, y


def build_ridge_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Create a preprocessing + Ridge pipeline for mixed feature types."""
    transformers = []

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
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("ridge", Ridge(alpha=1.0, solver="auto", max_iter=10000)),
        ]
    )


def get_ridge_alpha(model: Pipeline) -> float:
    """Return ridge regularization value from a fitted pipeline."""
    ridge = model.named_steps["ridge"]
    if hasattr(ridge, "alpha_"):
        return float(ridge.alpha_)
    return float(ridge.alpha)


def evaluate_log_and_level(
    y_train: pd.Series,
    y_test: pd.Series,
    pred_log: np.ndarray,
) -> dict[str, float]:
    """Compute standard holdout metrics on log and level scales."""
    pred_log_for_level = np.clip(pred_log, y_train.min(), y_train.max())
    pred_level = np.expm1(pred_log_for_level)
    y_test_level = np.expm1(y_test)
    mae_level = mean_absolute_error(y_test_level, pred_level)

    return {
        "r2_log": float(r2_score(y_test, pred_log)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_test, pred_log))),
        "r2_level": float(r2_score(y_test_level, pred_level)),
        "rmse_level": float(np.sqrt(mean_squared_error(y_test_level, pred_level))),
        "mae_level": float(mae_level),
    }


def cross_validate_log_and_level(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_seed: int,
) -> dict[str, object]:
    """Run K-fold CV and return fold metrics plus summary statistics."""
    if cv_folds < 2:
        raise ValueError("cv_folds must be >= 2 for cross-validation.")
    if len(X) < cv_folds:
        raise ValueError(
            f"Cannot run {cv_folds}-fold CV with only {len(X)} rows."
        )

    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    fold_metrics: list[dict[str, float]] = []

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        fold_model = clone(model)
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        fold_model.fit(X_train, y_train)
        pred_log = fold_model.predict(X_test)
        metrics = evaluate_log_and_level(y_train=y_train, y_test=y_test, pred_log=pred_log)
        metrics["fold"] = float(fold_index)
        fold_metrics.append(metrics)

    metric_names = ["r2_log", "rmse_log", "r2_level", "rmse_level", "mae_level"]
    summary: dict[str, float] = {}
    for metric in metric_names:
        values = np.array([fold[metric] for fold in fold_metrics], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))

    return {
        "cv_folds": int(cv_folds),
        "rows_total": int(len(X)),
        "fold_metrics": fold_metrics,
        "summary": summary,
    }


def train_test_split_indices(
    row_count: int,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train and test index arrays for a fixed random split."""
    return train_test_split(
        np.arange(row_count),
        test_size=test_size,
        random_state=random_seed,
    )


def extract_coefficients(model: Pipeline) -> pd.DataFrame:
    """Return sorted feature coefficients from fitted ridge pipeline."""

    def _parse_feature(feature: str) -> tuple[str, str, str]:
        if feature.startswith("num__"):
            name = feature.replace("num__", "", 1)
            readable = name.replace("_", " ").title()
            return "numeric", readable, ""

        if feature.startswith("cat__"):
            raw = feature.replace("cat__", "", 1)
            prefixes = [
                "neighborhood_name_",
                "OVERALL_COND_",
                "INT_COND_",
                "BLDG_TYPE_",
                "LU_",
            ]
            for prefix in prefixes:
                if raw.startswith(prefix):
                    base = prefix.rstrip("_")
                    level = raw[len(prefix) :]
                    readable = base.replace("_", " ").title()
                    return "categorical", readable, level

            if "_" in raw:
                base, level = raw.split("_", 1)
                readable = base.replace("_", " ").title()
                return "categorical", readable, level

            return "categorical", raw.replace("_", " ").title(), ""

        return "other", feature.replace("_", " ").title(), ""

    feature_names = [
        str(name) for name in model.named_steps["preprocess"].get_feature_names_out()
    ]
    coefficients = model.named_steps["ridge"].coef_

    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})

    coef_df["pct_effect_1_unit"] = (np.expm1(coef_df["coefficient"])) * 100.0
    coef_df["pct_effect_100_units"] = np.where(
        coef_df["feature"].str.startswith("num__"),
        np.expm1(coef_df["coefficient"] * 100.0) * 100.0,
        np.nan,
    )
    coef_df["pct_effect_1000_units"] = np.where(
        coef_df["feature"].str.startswith("num__"),
        np.expm1(coef_df["coefficient"] * 1000.0) * 100.0,
        np.nan,
    )

    parsed = coef_df["feature"].map(_parse_feature)
    coef_df[["feature_type", "feature_readable", "feature_level"]] = pd.DataFrame(
        parsed.tolist(),
        index=coef_df.index,
    )

    coef_df["_feature_type_rank"] = coef_df["feature_type"].map(
        {"numeric": 0, "categorical": 1, "other": 2}
    ).fillna(3)
    coef_df = coef_df.sort_values(
        by=["_feature_type_rank", "feature_readable", "feature_level", "coefficient"],
        ascending=[True, True, True, False],
    ).drop(columns=["_feature_type_rank"])

    coef_df["effect_1_unit_label"] = coef_df["pct_effect_1_unit"].map(
        lambda value: f"{value:.6f}%"
    )
    coef_df["effect_100_units_label"] = coef_df["pct_effect_100_units"].map(
        lambda value: f"{value:.3f}%" if pd.notna(value) else "N/A (categorical)"
    )
    coef_df["effect_1000_units_label"] = coef_df["pct_effect_1000_units"].map(
        lambda value: f"{value:.3f}%" if pd.notna(value) else "N/A (categorical)"
    )

    coef_df = coef_df[
        [
            "feature",
            "feature_readable",
            "feature_level",
            "feature_type",
            "coefficient",
            "pct_effect_1_unit",
            "effect_1_unit_label",
            "pct_effect_100_units",
            "effect_100_units_label",
            "pct_effect_1000_units",
            "effect_1000_units_label",
        ]
    ]

    return coef_df
