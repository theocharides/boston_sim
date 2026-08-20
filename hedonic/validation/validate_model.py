"""Run post-estimation diagnostics for the residential hedonic model.

Current diagnostics:
- Moran's I on model residuals using k-nearest-neighbor spatial weights.
- Standard regression diagnostics and plots for residuals and fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import libpysal
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import stats
from shapely import wkt
from spreg import OLS
from spreg.diagnostics_sp import LMtests

from hedonic.common.modeling_common import (
    DEFAULT_FEATURE_SET,
    LOG_PRICE_PER_SQFT_COL,
    PRICE_PER_SQFT_COL,
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
        description="Validate a fitted residential hedonic model with spatial residual diagnostics."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels_processed_for_hedonic.csv",
        help="Parcel CSV used for validation. This must match the processed parcel table used for the final model.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional path to legacy fitted hedonic model artifact (.joblib).",
    )
    parser.add_argument(
        "--selected-features-json",
        type=Path,
        default=repo_root / "hedonic" / "artifacts" / "residential_hedonic_lasso_selected_features.json",
        help="Path to selected-features JSON from the LASSO/OLS pipeline.",
    )
    parser.add_argument(
        "--coefficients-csv",
        type=Path,
        default=repo_root / "hedonic" / "artifacts" / "residential_hedonic_final_ols_coefficients.csv",
        help="Path to final OLS coefficients CSV from the LASSO/OLS pipeline.",
    )
    parser.add_argument(
        "--merge-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels_preprocessed.csv",
        help=(
            "Optional auxiliary parcel CSV used to fill missing model columns by PID "
            "before validation."
        ),
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=8,
        help="Number of nearest neighbors used for Moran's I weights.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=199,
        help="Retained for CLI compatibility. spreg Moran's I uses analytical inference and ignores permutations.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for permutation testing.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional sample size for faster diagnostics (0 uses all rows).",
    )
    parser.add_argument(
        "--residual-scale",
        type=str,
        choices=["log", "level"],
        default="log",
        help="Residual scale used for Moran's I: 'log' uses log(TOTAL_VALUE) residuals, 'level' uses dollar residuals.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=repo_root / "hedonic" / "artifacts" / "residential_hedonic_validation.json",
        help="Path to write validation metrics JSON.",
    )

    args = parser.parse_args()
    args.input_csv = require_existing_path(args.input_csv, "Input CSV")
    if args.model_path is not None:
        args.model_path = args.model_path.expanduser().resolve()
        if not args.model_path.exists():
            args.model_path = None
    if args.selected_features_json is not None:
        args.selected_features_json = require_existing_path(args.selected_features_json, "Selected features JSON")
    if args.coefficients_csv is not None:
        args.coefficients_csv = require_existing_path(args.coefficients_csv, "Coefficients CSV")
    if args.merge_csv is not None:
        args.merge_csv = require_existing_path(args.merge_csv, "Merge CSV")
    args.output_json = args.output_json.expanduser().resolve()

    if args.model_path is None and (args.selected_features_json is None or args.coefficients_csv is None):
        raise ValueError(
            "Provide --model-path for legacy validation or both --selected-features-json and "
            "--coefficients-csv for LASSO/OLS validation."
        )

    if args.k_neighbors < 1:
        raise ValueError("--k-neighbors must be >= 1")
    if args.permutations < 0:
        raise ValueError("--permutations must be >= 0")
    if args.sample_size < 0:
        raise ValueError("--sample-size must be >= 0")

    return args


def _fitted_model_feature_names(model: Any) -> list[str] | None:
    """Return fitted model input feature names when available."""
    if hasattr(model, "feature_names_in_"):
        return [str(name) for name in model.feature_names_in_]

    preprocess = getattr(model, "named_steps", {}).get("preprocess") if hasattr(model, "named_steps") else None
    if preprocess is not None and hasattr(preprocess, "feature_names_in_"):
        return [str(name) for name in preprocess.feature_names_in_]

    return None


def _parse_selected_features_json(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("features"), list):
        values = raw["features"]
    elif isinstance(raw, list):
        values = raw
    else:
        raise ValueError("Selected-features JSON must be a list or an object with a 'features' list.")

    parsed = [str(value).strip() for value in values if str(value).strip()]
    if not parsed:
        raise ValueError("Selected-features JSON did not contain any features.")
    return parsed


def _build_ols_design_from_selected_features(
    model_df: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric_selected, categorical_selected = infer_feature_types(model_df, selected_features)
    parts: list[pd.DataFrame] = []

    if numeric_selected:
        numeric_df = model_df[numeric_selected].apply(pd.to_numeric, errors="coerce")
        for column in numeric_df.columns:
            numeric_df[column] = numeric_df[column].fillna(numeric_df[column].median())
        parts.append(numeric_df.astype(float))

    if categorical_selected:
        cat_df = model_df[categorical_selected].copy()
        for column in cat_df.columns:
            mode = cat_df[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "MISSING"
            cat_df[column] = cat_df[column].fillna(fill_value).astype(str)
        cat_dummies = pd.get_dummies(cat_df, drop_first=True, dtype=float)
        parts.append(cat_dummies)

    if not parts:
        raise ValueError("No design matrix columns could be built from selected features.")

    design_df = pd.concat(parts, axis=1)
    design_df.insert(0, "const", 1.0)
    return design_df, numeric_selected, categorical_selected


def _predict_from_ols_coefficients(design_df: pd.DataFrame, coefficients_csv: Path) -> tuple[np.ndarray, list[str]]:
    coef_df = pd.read_csv(coefficients_csv)
    if "term" not in coef_df.columns or "coefficient_log" not in coef_df.columns:
        raise ValueError("Coefficients CSV must include 'term' and 'coefficient_log' columns.")

    coef_series = pd.Series(
        pd.to_numeric(coef_df["coefficient_log"], errors="coerce").to_numpy(dtype=float),
        index=coef_df["term"].astype(str),
    )
    coef_series = coef_series.dropna()

    common_terms = [column for column in design_df.columns if column in coef_series.index]
    if "const" not in common_terms:
        common_terms = ["const", *common_terms]
        if "const" not in coef_series.index:
            coef_series.loc["const"] = 0.0

    design_used = design_df.reindex(columns=common_terms, fill_value=0.0)
    coef_used = coef_series.reindex(common_terms, fill_value=0.0)
    pred_log = design_used.to_numpy(dtype=float) @ coef_used.to_numpy(dtype=float)
    return pred_log.astype(float), common_terms


def _repo_relative_path(path: Path | str, repo_root: Path) -> str:
    """Render a path relative to the repo root when possible, otherwise use the original string."""
    candidate = Path(path).expanduser() if isinstance(path, str) else path.expanduser()
    resolved_path = candidate.resolve()
    resolved_root = repo_root.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        # Keep absolute paths for artifacts generated outside of the repository root.
        return resolved_path.as_posix()


def _build_connected_knn_weights(coords: np.ndarray, k_neighbors: int) -> tuple[Any, int, int]:
    """Build KNN weights and increase k until the graph is connected (or max feasible k)."""
    n_obs = int(coords.shape[0])
    if n_obs < 2:
        raise ValueError("At least two observations are required to build spatial weights")

    k = max(1, min(int(k_neighbors), n_obs - 1))
    max_k = min(64, n_obs - 1)

    while True:
        w = libpysal.weights.KNN.from_array(coords, k=k)
        component_labels = w.component_labels
        component_count = len(set(component_labels)) if component_labels is not None else 1
        if component_count <= 1 or k >= max_k:
            w.transform = "r"
            return w, k, component_count
        k = min(max_k, k + 2)


def _parse_centroid_xy(geometry_wkt: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    keep_mask = np.zeros(len(geometry_wkt), dtype=bool)

    for idx, value in enumerate(geometry_wkt.astype("object")):
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            geom = wkt.loads(value)
        except Exception:
            continue
        if geom is None or geom.is_empty:
            continue
        point = geom if geom.geom_type == "Point" else geom.representative_point()
        xs.append(float(point.x))
        ys.append(float(point.y))
        keep_mask[idx] = True

    return np.column_stack([xs, ys]), keep_mask


def _coerce_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _residual_diagnostics(
    residual_values: np.ndarray,
    fitted_values: np.ndarray,
) -> dict[str, float]:
    residual_values = residual_values.astype(float)
    fitted_values = fitted_values.astype(float)
    mean_resid = float(np.mean(residual_values))
    std_resid = float(np.std(residual_values, ddof=1)) if residual_values.size > 1 else float("nan")
    centered = residual_values - mean_resid
    skewness = float(stats.skew(residual_values, bias=False)) if residual_values.size > 2 else float("nan")
    kurtosis = float(stats.kurtosis(residual_values, fisher=True, bias=False)) if residual_values.size > 3 else float("nan")
    fitted_resid_corr = float(stats.pearsonr(fitted_values, residual_values).statistic) if residual_values.size > 1 else float("nan")

    normaltest_stat = float("nan")
    normaltest_p = float("nan")
    if residual_values.size >= 8:
        normaltest = stats.normaltest(residual_values)
        normaltest_stat = float(normaltest.statistic)
        normaltest_p = float(normaltest.pvalue)

    abs_sqrt_std = np.sqrt(np.abs(centered / std_resid)) if std_resid > 0 else np.full_like(centered, np.nan)
    return {
        "residual_mean": mean_resid,
        "residual_std": std_resid,
        "residual_min": float(np.min(residual_values)),
        "residual_max": float(np.max(residual_values)),
        "residual_median": float(np.median(residual_values)),
        "residual_skewness": skewness,
        "residual_excess_kurtosis": kurtosis,
        "fitted_residual_correlation": fitted_resid_corr,
        "residual_normaltest_statistic": normaltest_stat,
        "residual_normaltest_pvalue": normaltest_p,
        "scaled_location_mean": float(np.nanmean(abs_sqrt_std)),
    }


def _save_validation_plots(
    output_dir: Path,
    scale_name: str,
    fitted_values: np.ndarray,
    y_true: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    standardized = residuals / (np.std(residuals, ddof=1) if residuals.size > 1 else 1.0)
    abs_sqrt_std = np.sqrt(np.abs(standardized))

    plots: dict[str, str] = {}

    width, height = 900, 700
    margin_left, margin_right, margin_top, margin_bottom = 90, 30, 60, 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    font = ImageFont.load_default()

    def _new_canvas(title: str, x_label: str, y_label: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((margin_left, 18), title, fill="black", font=font)
        draw.text((margin_left + plot_width // 2 - 25, height - 35), x_label, fill="black", font=font)
        draw.text((10, margin_top + plot_height // 2), y_label, fill="black", font=font)
        draw.rectangle(
            [margin_left, margin_top, margin_left + plot_width, margin_top + plot_height],
            outline="#444444",
            width=1,
        )
        return image, draw

    def _scale_points(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
        x_min = float(np.min(x_values))
        x_max = float(np.max(x_values))
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5
        x_scaled = margin_left + (x_values - x_min) / (x_max - x_min) * plot_width
        y_scaled = margin_top + plot_height - (y_values - y_min) / (y_max - y_min) * plot_height
        return x_scaled, y_scaled, x_min, x_max, y_min, y_max

    def _save_image(image: Image.Image, name: str) -> None:
        path = output_dir / f"{name}.png"
        image.save(path)
        plots[name] = str(path)

    image, draw = _new_canvas(f"Observed vs Predicted ({scale_name})", "Observed", "Predicted")
    x_scaled, y_scaled, x_min, x_max, y_min, y_max = _scale_points(y_true, fitted_values)
    for x_value, y_value in zip(x_scaled, y_scaled):
        draw.ellipse((x_value - 2, y_value - 2, x_value + 2, y_value + 2), fill="#4C78A8", outline="#4C78A8")
    line_lo = min(x_min, y_min)
    line_hi = max(x_max, y_max)
    x1 = margin_left + (line_lo - x_min) / (x_max - x_min) * plot_width
    y1 = margin_top + plot_height - (line_lo - y_min) / (y_max - y_min) * plot_height
    x2 = margin_left + (line_hi - x_min) / (x_max - x_min) * plot_width
    y2 = margin_top + plot_height - (line_hi - y_min) / (y_max - y_min) * plot_height
    draw.line((x1, y1, x2, y2), fill="crimson", width=2)
    _save_image(image, "observed_vs_predicted")

    image, draw = _new_canvas(f"Residuals vs Fitted ({scale_name})", "Fitted", "Residuals")
    x_scaled, y_scaled, _, _, _, _ = _scale_points(fitted_values, residuals)
    for x_value, y_value in zip(x_scaled, y_scaled):
        draw.ellipse((x_value - 2, y_value - 2, x_value + 2, y_value + 2), fill="#4C78A8", outline="#4C78A8")
    y_zero = margin_top + plot_height - (0.0 - float(np.min(residuals))) / (float(np.max(residuals)) - float(np.min(residuals))) * plot_height if float(np.max(residuals)) != float(np.min(residuals)) else margin_top + plot_height / 2
    draw.line((margin_left, y_zero, margin_left + plot_width, y_zero), fill="crimson", width=2)
    _save_image(image, "residuals_vs_fitted")

    image, draw = _new_canvas(f"Residual Distribution ({scale_name})", "Residual", "Density")
    hist, bin_edges = np.histogram(residuals, bins=50, density=True)
    bar_width = plot_width / len(hist)
    y_max_hist = float(np.max(hist)) if float(np.max(hist)) > 0.0 else 1.0
    for idx, density in enumerate(hist):
        left = margin_left + idx * bar_width
        right = left + bar_width * 0.9
        top = margin_top + plot_height - (float(density) / y_max_hist) * plot_height
        draw.rectangle((left, top, right, margin_top + plot_height), fill="#4C78A8", outline="#4C78A8")
    mu, sigma = stats.norm.fit(residuals)
    xs = np.linspace(float(np.min(residuals)), float(np.max(residuals)), 200)
    pdf = stats.norm.pdf(xs, mu, sigma)
    x_scaled, y_scaled, _, _, _, _ = _scale_points(xs, pdf)
    draw.line(list(zip(x_scaled, y_scaled)), fill="crimson", width=2)
    _save_image(image, "residual_distribution")

    image, draw = _new_canvas(f"Q-Q Plot of Standardized Residuals ({scale_name})", "Theoretical Quantiles", "Standardized Residuals")
    if standardized.size > 1:
        ordered = np.sort(standardized)
        probs = (np.arange(1, standardized.size + 1) - 0.5) / standardized.size
        theoretical = stats.norm.ppf(probs)
        x_scaled, y_scaled, x_min, x_max, y_min, y_max = _scale_points(theoretical, ordered)
        for x_value, y_value in zip(x_scaled, y_scaled):
            draw.ellipse((x_value - 2, y_value - 2, x_value + 2, y_value + 2), fill="#4C78A8", outline="#4C78A8")
        qq_lo = min(x_min, y_min)
        qq_hi = max(x_max, y_max)
        x1 = margin_left + (qq_lo - x_min) / (x_max - x_min) * plot_width
        y1 = margin_top + plot_height - (qq_lo - y_min) / (y_max - y_min) * plot_height
        x2 = margin_left + (qq_hi - x_min) / (x_max - x_min) * plot_width
        y2 = margin_top + plot_height - (qq_hi - y_min) / (y_max - y_min) * plot_height
        draw.line((x1, y1, x2, y2), fill="crimson", width=2)
    _save_image(image, "qq_plot_standardized_residuals")

    image, draw = _new_canvas(f"Scale-Location ({scale_name})", "Fitted", "Sqrt(|Standardized Residual|)")
    x_scaled, y_scaled, _, _, _, _ = _scale_points(fitted_values, abs_sqrt_std)
    for x_value, y_value in zip(x_scaled, y_scaled):
        draw.ellipse((x_value - 2, y_value - 2, x_value + 2, y_value + 2), fill="#4C78A8", outline="#4C78A8")
    _save_image(image, "scale_location")

    return plots


def _morans_i_knn(
    values: np.ndarray,
    coords: np.ndarray,
    k_neighbors: int,
    permutations: int,
    random_seed: int,
) -> dict[str, float | int | None]:
    n = int(values.shape[0])
    if n < 3:
        raise ValueError("Need at least 3 rows with valid geometry for Moran's I.")

    k_eff = min(k_neighbors, n - 1)
    if k_eff < 1:
        raise ValueError("Not enough rows for requested k-nearest-neighbor computation.")

    if float(np.var(values)) <= 0.0:
        raise ValueError("Residual variance is zero; Moran's I is undefined.")

    weights, k_used, component_count = _build_connected_knn_weights(coords, k_eff)

    weight_matrix = weights.sparse.tocsr()
    centered = values.astype(float) - float(np.mean(values))
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        raise ValueError("Residual variance is zero; Moran's I is undefined.")

    def _morans_i_for_series(series: np.ndarray) -> float:
        centered_series = series.astype(float) - float(np.mean(series))
        return float((n / weights.s0) * (centered_series @ (weight_matrix @ centered_series)) / float(np.dot(centered_series, centered_series)))

    morans_i = _morans_i_for_series(values)

    permutations_used = int(permutations) if permutations > 0 else 199
    rng = np.random.default_rng(random_seed)
    if permutations_used > 0:
        permuted = np.empty(permutations_used, dtype=float)
        for idx in range(permutations_used):
            permuted[idx] = _morans_i_for_series(rng.permutation(values))
        perm_mean = float(np.mean(permuted))
        perm_std = float(np.std(permuted, ddof=1)) if permutations_used > 1 else float("nan")
        if perm_std > 0.0:
            z_score = float((morans_i - perm_mean) / perm_std)
        else:
            z_score = float("nan")
        tail = float(np.mean(np.abs(permuted) >= abs(morans_i)))
        p_value_two_sided = max(tail, 1.0 / (permutations_used + 1))
    else:
        perm_mean = float("nan")
        perm_std = float("nan")
        z_score = float("nan")
        p_value_two_sided = float("nan")

    return {
        "n": n,
        "k_neighbors": k_eff,
        "k_neighbors_used": int(k_used),
        "connected_components": int(component_count),
        "morans_i": float(morans_i),
        "z_score": z_score,
        "permutations_requested": int(permutations),
        "permutations_used": permutations_used,
        "inference": "permutation",
        "permutation_mean": perm_mean,
        "permutation_std": perm_std,
        "p_value_two_sided": p_value_two_sided,
    }


def _design_matrix(model: Any, X: pd.DataFrame) -> np.ndarray:
    preprocess = model.named_steps["preprocess"]
    matrix = preprocess.transform(X)
    dense = _coerce_dense(matrix)
    if dense.ndim == 1:
        dense = dense.reshape(-1, 1)
    return dense.astype(float)


def _extract_test_pair(test_value: Any) -> tuple[float, float]:
    if isinstance(test_value, tuple) and len(test_value) >= 2:
        return float(test_value[0]), float(test_value[1])
    if isinstance(test_value, dict):
        stat = test_value.get("statistic", test_value.get("value", np.nan))
        p_value = test_value.get("p_value", test_value.get("p", np.nan))
        return float(stat), float(p_value)
    if test_value is None:
        return float("nan"), float("nan")
    return float("nan"), float("nan")


def _classify_spatial_dependence(lm_tests: dict[str, Any]) -> dict[str, Any]:
    """Choose the more likely spatial process from robust LM diagnostics."""
    def _get_stats(name: str) -> tuple[float, float]:
        if name not in lm_tests:
            return float("nan"), float("nan")
        return _extract_test_pair(lm_tests[name])

    lme_stat, lme_p = _get_stats("lme")
    lml_stat, lml_p = _get_stats("lml")
    rlme_stat, rlme_p = _get_stats("rlme")
    rlml_stat, rlml_p = _get_stats("rlml")
    sarma_stat, sarma_p = _get_stats("sarma")

    strong_error = np.isfinite(rlme_stat) and np.isfinite(rlme_p) and rlme_p < 0.05
    strong_lag = np.isfinite(rlml_stat) and np.isfinite(rlml_p) and rlml_p < 0.05
    strong_sarma = np.isfinite(sarma_stat) and np.isfinite(sarma_p) and sarma_p < 0.05

    if strong_lag and (np.isnan(rlme_stat) or rlml_stat >= rlme_stat):
        classification = "spatial lag"
        reason = "robust LM lag test is significant and exceeds robust LM error evidence"
    elif strong_error and (np.isnan(rlml_stat) or rlme_stat >= rlml_stat):
        classification = "spatial error"
        reason = "robust LM error test is significant and exceeds robust LM lag evidence"
    elif np.isfinite(lml_stat) and np.isfinite(lme_stat) and lml_p < 0.05 and lme_p >= 0.05 and lml_stat >= lme_stat:
        classification = "spatial lag"
        reason = "LM lag is significant while LM error is not"
    elif np.isfinite(lme_stat) and np.isfinite(lml_stat) and lme_p < 0.05 and lml_p >= 0.05 and lme_stat >= lml_stat:
        classification = "spatial error"
        reason = "LM error is significant while LM lag is not"
    elif strong_sarma:
        classification = "ambiguous"
        reason = "SARMA test is significant, suggesting a combined lag/error process"
    else:
        classification = "ambiguous"
        reason = "no LM test clearly dominates; evidence is weak or mixed"

    return {
        "classification": classification,
        "reason": reason,
        "lm_error": {"statistic": lme_stat, "p_value": lme_p},
        "lm_lag": {"statistic": lml_stat, "p_value": lml_p},
        "robust_lm_error": {"statistic": rlme_stat, "p_value": rlme_p},
        "robust_lm_lag": {"statistic": rlml_stat, "p_value": rlml_p},
        "sarma": {"statistic": sarma_stat, "p_value": sarma_p},
    }


def _spatial_lm_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    k_neighbors: int,
) -> dict[str, Any]:
    """Run LM lag/error diagnostics using a KNN-weighted OLS baseline."""
    n = int(X.shape[0])
    if n < 3:
        return {
            "available": False,
            "reason": "fewer than 3 rows available for spatial diagnostics",
        }

    y_2d = np.asarray(y, dtype=float).reshape(-1, 1)
    X_2d = np.asarray(X, dtype=float)
    if X_2d.ndim == 1:
        X_2d = X_2d.reshape(-1, 1)

    if not np.all(np.isfinite(X_2d)) or not np.all(np.isfinite(y_2d)):
        return {
            "available": False,
            "reason": "design matrix or target contains non-finite values; spatial LM diagnostics skipped",
        }

    # Stabilize the matrix passed to spreg by removing constant and collinear columns.
    col_stds = np.std(X_2d, axis=0)
    non_constant_mask = np.isfinite(col_stds) & (col_stds > 1e-12)
    if not np.any(non_constant_mask):
        return {
            "available": False,
            "reason": "design matrix has no non-constant columns after filtering; spatial LM diagnostics skipped",
        }

    X_work = X_2d[:, non_constant_mask]
    full_rank_idx: list[int] = []
    for local_idx in range(X_work.shape[1]):
        trial_idx = full_rank_idx + [local_idx]
        trial_matrix = X_work[:, trial_idx]
        if np.linalg.matrix_rank(trial_matrix) > len(full_rank_idx):
            full_rank_idx.append(local_idx)

    if not full_rank_idx:
        return {
            "available": False,
            "reason": "design matrix is rank-deficient; spatial LM diagnostics skipped",
        }

    dropped_for_collinearity = int(X_work.shape[1] - len(full_rank_idx))
    X_work = X_work[:, full_rank_idx]
    kept_feature_count = int(X_work.shape[1])

    k_eff = min(int(k_neighbors), max(1, n - 1))
    candidate_ks = [k_eff, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64]
    candidate_ks = [k for k in dict.fromkeys(candidate_ks) if 1 <= k <= max(1, n - 1)]

    lm = None
    k_used = k_eff
    component_count = 0
    rows_used = n
    fallback_applied = False
    last_error = "unknown failure"

    for k_try in candidate_ks:
        try:
            weights, k_used_try, component_count_try = _build_connected_knn_weights(coords[:n], k_try)
        except Exception as exc:
            last_error = f"k={k_try}: unable to build weights ({exc})"
            continue

        if weights.s0 <= 0:
            last_error = f"k={k_try}: weights matrix is empty"
            continue

        try:
            ols_model = OLS(y_2d, X_work)
            lm = LMtests(ols_model, weights, tests=["all"])
            k_used = k_used_try
            component_count = component_count_try
            rows_used = n
            break
        except np.linalg.LinAlgError as exc:
            last_error = f"k={k_try}: linear algebra instability ({exc})"
            continue
        except ValueError as exc:
            last_error = f"k={k_try}: {exc}"
            if "math domain error" not in str(exc).lower() or component_count_try <= 1:
                continue

            labels = np.asarray(weights.component_labels)
            unique_labels, counts = np.unique(labels, return_counts=True)
            keep_label = unique_labels[int(np.argmax(counts))]
            keep_mask = labels == keep_label
            n_component = int(np.sum(keep_mask))
            if n_component < 30:
                last_error = (
                    f"k={k_try}: math domain error and largest component too small "
                    f"(n={n_component})"
                )
                continue

            X_component = X_work[keep_mask]
            y_component = y_2d[keep_mask]
            coords_component = coords[:n][keep_mask]
            k_component = min(k_used_try, max(1, n_component - 1))
            try:
                weights_component, k_component_used, component_count_component = _build_connected_knn_weights(
                    coords_component,
                    k_component,
                )
                ols_model = OLS(y_component, X_component)
                lm = LMtests(ols_model, weights_component, tests=["all"])
                k_used = k_component_used
                component_count = component_count_component
                rows_used = n_component
                fallback_applied = True
                break
            except Exception as fallback_exc:
                last_error = f"k={k_try}: largest-component fallback failed ({fallback_exc})"
                continue
        except Exception as exc:  # pragma: no cover - defensive trap for unexpected spreg failures
            last_error = f"k={k_try}: unexpected LM failure ({exc})"
            continue

    if lm is None:
        return {
            "available": False,
            "reason": f"spatial LM diagnostics could not be computed: {last_error}",
            "k_neighbors_attempted": [int(k) for k in candidate_ks],
        }

    result = {
        "available": True,
        "k_neighbors": k_eff,
        "k_neighbors_used": int(k_used),
        "connected_components": int(component_count),
        "rows_used": int(rows_used),
        "largest_component_fallback": bool(fallback_applied),
        "features_used": kept_feature_count,
        "features_dropped_for_collinearity": dropped_for_collinearity,
        "lme": {"statistic": float(lm.lme[0]), "p_value": float(lm.lme[1])},
        "lml": {"statistic": float(lm.lml[0]), "p_value": float(lm.lml[1])},
        "rlme": {"statistic": float(lm.rlme[0]), "p_value": float(lm.rlme[1])},
        "rlml": {"statistic": float(lm.rlml[0]), "p_value": float(lm.rlml[1])},
        "sarma": {"statistic": float(lm.sarma[0]), "p_value": float(lm.sarma[1])},
    }
    result["dependence_classification"] = _classify_spatial_dependence(result)
    return result


def _vif_from_design_matrix(design_matrix: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    """Compute variance inflation factors for the transformed design matrix."""
    if not feature_names:
        return {
            "feature_vif": {},
            "max_vif": None,
            "n_features": 0,
            "features_above_5": [],
            "features_above_10": [],
        }

    matrix = np.asarray(design_matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    vif_by_feature: dict[str, float] = {}
    above_5: list[str] = []
    above_10: list[str] = []

    for idx, feature_name in enumerate(feature_names):
        if idx >= matrix.shape[1]:
            vif_by_feature[feature_name] = float("nan")
            continue

        target = matrix[:, idx]
        if np.allclose(target, target[0]):
            vif_by_feature[feature_name] = float("nan")
            continue

        others = np.delete(matrix, idx, axis=1)
        if others.shape[1] == 0:
            vif = 1.0
        else:
            intercept = np.ones((others.shape[0], 1))
            design = np.hstack([intercept, others])
            coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
            fitted = design @ coeffs
            ss_total = float(np.sum((target - np.mean(target)) ** 2))
            ss_residual = float(np.sum((target - fitted) ** 2))
            r_squared = 0.0 if ss_total <= 0 else 1.0 - (ss_residual / ss_total)
            vif = 1.0 / (1.0 - r_squared) if r_squared < 0.999999 else float("inf")

        if np.isfinite(vif):
            vif_by_feature[feature_name] = float(vif)
            if vif > 10:
                above_10.append(feature_name)
            elif vif > 5:
                above_5.append(feature_name)
        else:
            vif_by_feature[feature_name] = float("inf")
            above_10.append(feature_name)

    max_vif = max((value for value in vif_by_feature.values() if np.isfinite(value)), default=None)
    if max_vif is not None and max_vif == float("inf"):
        max_vif = None

    return {
        "feature_vif": vif_by_feature,
        "max_vif": max_vif,
        "n_features": len(feature_names),
        "features_above_5": above_5,
        "features_above_10": above_10,
    }


def _predict_from_design_matrix(
    design_matrix: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    chunk_size: int = 10,
) -> np.ndarray:
    predictions = np.empty(design_matrix.shape[0], dtype=float)
    for start in range(0, design_matrix.shape[0], chunk_size):
        stop = min(start + chunk_size, design_matrix.shape[0])
        predictions[start:stop] = design_matrix[start:stop] @ coef + intercept
    return predictions


def _merge_missing_model_columns(
    df: pd.DataFrame,
    merge_csv: Path | None,
    required_columns: list[str],
) -> pd.DataFrame:
    if merge_csv is None:
        return df

    missing = [column for column in required_columns if column not in df.columns]
    if not missing:
        return df

    if "PID" not in df.columns:
        raise ValueError("Input CSV is missing PID, so missing model columns cannot be merged.")

    auxiliary = pd.read_csv(merge_csv, low_memory=False)
    if "PID" not in auxiliary.columns:
        raise ValueError(f"Merge CSV {merge_csv} must contain PID for alignment.")

    join_columns = ["PID", *[column for column in missing if column in auxiliary.columns]]
    if len(join_columns) == 1:
        return df

    working = df.reset_index()
    index_column = working.columns[0]

    merged = working.merge(
        auxiliary[join_columns].drop_duplicates(subset=["PID"]),
        on="PID",
        how="left",
        suffixes=("", "_merge"),
    )

    for column in join_columns[1:]:
        merge_column = f"{column}_merge"
        if merge_column in merged.columns:
            merged[column] = merged[column].combine_first(merged[merge_column])
            merged = merged.drop(columns=[merge_column])

    merged = merged.set_index(index_column)
    return merged


def main() -> None:
    args = parse_args()

    print(f"Reading validation data: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)
    df = subset_residential_rows(df, strict=True)
    model = joblib.load(args.model_path) if args.model_path is not None else None

    if TARGET_COL not in df.columns:
        raise ValueError(f"Required target column missing: {TARGET_COL}")
    if "geometry" not in df.columns:
        raise ValueError(
            "Input CSV must contain a geometry column in WKT format. "
            "Use inputs/parcels_preprocessed.csv or another table that retains geometry."
        )

    if model is not None:
        feature_source = "fitted_model"
        feature_list = _fitted_model_feature_names(model)
        if feature_list is None:
            feature_source = "default_feature_set"
            feature_list = available_features(df, list(DEFAULT_FEATURE_SET))
            if not feature_list:
                raise ValueError("No default features were found in the input data.")
    else:
        feature_source = "selected_features_json"
        feature_list = _parse_selected_features_json(args.selected_features_json)

    missing_for_model = [feature for feature in feature_list if feature not in df.columns]
    if missing_for_model:
        df = _merge_missing_model_columns(df, args.merge_csv, missing_for_model)
        missing_for_model = [feature for feature in feature_list if feature not in df.columns]
        if missing_for_model:
            raise ValueError(
                "Input CSV is missing columns required by the selected model even after "
                f"merging {args.merge_csv}: {missing_for_model}"
            )

    numeric_features, categorical_features = infer_feature_types(df, feature_list)
    if model is not None:
        model_df = prepare_model_df(
            df,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_col=TARGET_COL,
        )
    else:
        model_df = prepare_price_per_sqft_model_df(
            df,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_value_col=TARGET_COL,
        )
    if model_df.empty:
        raise ValueError("No rows available after model data preparation.")

    geometry_series = df.loc[model_df.index, "geometry"]
    X = model_df[[*numeric_features, *categorical_features]]
    if model is not None:
        y_level = model_df[TARGET_COL].to_numpy(dtype=float)
        y_log = np.log1p(y_level)
    else:
        y_level = model_df[PRICE_PER_SQFT_COL].to_numpy(dtype=float)
        y_log = model_df[LOG_PRICE_PER_SQFT_COL].to_numpy(dtype=float)

    if model is not None:
        preprocess = model.named_steps["preprocess"]
        design_matrix = _design_matrix(model, X)
        feature_names = list(preprocess.get_feature_names_out())
        vif = _vif_from_design_matrix(design_matrix, feature_names)
        ridge = model.named_steps["ridge"]
        pred_log = _predict_from_design_matrix(
            design_matrix=design_matrix,
            coef=np.asarray(ridge.coef_, dtype=float),
            intercept=float(ridge.intercept_),
        )
    else:
        design_df, numeric_features, categorical_features = _build_ols_design_from_selected_features(model_df, feature_list)
        pred_log, used_terms = _predict_from_ols_coefficients(design_df, args.coefficients_csv)
        vif_design = design_df.drop(columns=["const"], errors="ignore")
        vif = _vif_from_design_matrix(vif_design.to_numpy(dtype=float), list(vif_design.columns))
        design_matrix = vif_design.to_numpy(dtype=float)
    pred_level = np.expm1(np.clip(pred_log, float(np.min(y_log)), float(np.max(y_log))))

    if args.residual_scale == "level":
        pred_log_clipped = np.clip(pred_log, float(np.min(y_log)), float(np.max(y_log)))
        pred_level = np.expm1(pred_log_clipped)
        residual_values = y_level - pred_level
    else:
        residual_values = y_log - pred_log

    if args.residual_scale == "level":
        plot_fitted_values = pred_level
        plot_true_values = y_level
    else:
        plot_fitted_values = pred_log
        plot_true_values = y_log

    coords, keep_mask = _parse_centroid_xy(geometry_series.reset_index(drop=True))
    valid_idx = np.flatnonzero(keep_mask)

    residual_for_moran = residual_values[valid_idx]
    fitted_for_moran = plot_fitted_values[valid_idx]
    y_true_for_moran = plot_true_values[valid_idx]
    design_for_lm = np.asarray(design_matrix[valid_idx], dtype=float)
    y_for_lm = np.asarray(y_log[valid_idx], dtype=float)

    if args.sample_size > 0 and args.sample_size < residual_for_moran.shape[0]:
        rng = np.random.default_rng(args.random_seed)
        sample_idx = rng.choice(residual_for_moran.shape[0], size=args.sample_size, replace=False)
        residual_for_moran = residual_for_moran[sample_idx]
        fitted_for_moran = fitted_for_moran[sample_idx]
        y_true_for_moran = y_true_for_moran[sample_idx]
        design_for_lm = design_for_lm[sample_idx]
        y_for_lm = y_for_lm[sample_idx]
        coords = coords[sample_idx]

    residual_metrics = _residual_diagnostics(
        residual_values=residual_for_moran,
        fitted_values=fitted_for_moran,
    )

    moran = _morans_i_knn(
        values=residual_for_moran,
        coords=coords,
        k_neighbors=args.k_neighbors,
        permutations=args.permutations,
        random_seed=args.random_seed,
    )

    lm_diagnostics = _spatial_lm_diagnostics(
        X=design_for_lm,
        y=y_for_lm,
        coords=coords,
        k_neighbors=args.k_neighbors,
    )

    plots_dir = args.output_json.parent / f"{args.output_json.stem}_plots"
    plot_paths = _save_validation_plots(
        output_dir=plots_dir,
        scale_name=args.residual_scale,
        fitted_values=fitted_for_moran,
        y_true=y_true_for_moran,
        residuals=residual_for_moran,
    )

    repo_root = Path(__file__).resolve().parents[2]
    output = {
        "input_csv": _repo_relative_path(args.input_csv, repo_root),
        "model_path": _repo_relative_path(args.model_path, repo_root) if args.model_path is not None else None,
        "selected_features_json": _repo_relative_path(args.selected_features_json, repo_root) if args.selected_features_json is not None else None,
        "coefficients_csv": _repo_relative_path(args.coefficients_csv, repo_root) if args.coefficients_csv is not None else None,
        "residual_scale": args.residual_scale,
        "feature_source": feature_source,
        "model_target": PRICE_PER_SQFT_COL if model is None else TARGET_COL,
        "selected_features": feature_list,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "rows_residential": int(len(df)),
        "rows_model": int(len(model_df)),
        "rows_with_valid_geometry": int(int(keep_mask.sum())),
        "sample_size": int(args.sample_size),
        "residual_diagnostics": residual_metrics,
        "morans_i": moran,
        "spatial_lm_tests": lm_diagnostics,
        "spatial_dependence": lm_diagnostics.get("dependence_classification", {}),
        "vif": vif,
        "plots_dir": _repo_relative_path(plots_dir, repo_root),
        "plot_paths": {name: _repo_relative_path(path, repo_root) for name, path in plot_paths.items()},
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Validation output written to: {args.output_json}")
    print(json.dumps({"morans_i": moran}, indent=2))


if __name__ == "__main__":
    main()
