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

from hedonic.common.modeling_common import (
    DEFAULT_FEATURE_SET,
    TARGET_COL,
    available_features,
    infer_feature_types,
    prepare_model_df,
    subset_residential_rows,
)
from shared_utils import require_existing_path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
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
        default=repo_root / "hedonic" / "artifacts" / "residential_hedonic_model.joblib",
        help="Path to fitted hedonic model artifact.",
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
    args.model_path = require_existing_path(args.model_path, "Model artifact")
    if args.merge_csv is not None:
        args.merge_csv = require_existing_path(args.merge_csv, "Merge CSV")
    args.output_json = args.output_json.expanduser().resolve()

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

    weights = libpysal.weights.KNN.from_array(coords, k=k_eff)
    weights.transform = "r"

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
    model = joblib.load(args.model_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Required target column missing: {TARGET_COL}")
    if "geometry" not in df.columns:
        raise ValueError(
            "Input CSV must contain a geometry column in WKT format. "
            "Use inputs/parcels_preprocessed_with_baseline_vars.csv or another table that retains geometry."
        )

    feature_source = "fitted_model"
    feature_list = _fitted_model_feature_names(model)

    if feature_list is None:
        feature_source = "default_feature_set"
        feature_list = available_features(df, list(DEFAULT_FEATURE_SET))
        if not feature_list:
            raise ValueError("No default features were found in the input data.")
    else:
        missing_for_model = [feature for feature in feature_list if feature not in df.columns]
        if missing_for_model:
            df = _merge_missing_model_columns(df, args.merge_csv, missing_for_model)
            missing_for_model = [feature for feature in feature_list if feature not in df.columns]
            if missing_for_model:
                raise ValueError(
                    "Input CSV is missing columns required by the fitted model even after "
                    f"merging {args.merge_csv}: {missing_for_model}"
                )

    numeric_features, categorical_features = infer_feature_types(df, feature_list)
    model_df = prepare_model_df(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_col=TARGET_COL,
    )
    if model_df.empty:
        raise ValueError("No rows available after model data preparation.")

    geometry_series = df.loc[model_df.index, "geometry"]
    X = model_df[[*numeric_features, *categorical_features]]
    y_level = model_df[TARGET_COL].to_numpy(dtype=float)
    y_log = np.log1p(y_level)

    design_matrix = _design_matrix(model, X)
    ridge = model.named_steps["ridge"]
    pred_log = _predict_from_design_matrix(
        design_matrix=design_matrix,
        coef=np.asarray(ridge.coef_, dtype=float),
        intercept=float(ridge.intercept_),
    )
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
    residual_for_moran = residual_values[keep_mask]
    fitted_for_moran = plot_fitted_values[keep_mask]
    y_true_for_moran = plot_true_values[keep_mask]

    if args.sample_size > 0 and args.sample_size < residual_for_moran.shape[0]:
        rng = np.random.default_rng(args.random_seed)
        sample_idx = rng.choice(residual_for_moran.shape[0], size=args.sample_size, replace=False)
        residual_for_moran = residual_for_moran[sample_idx]
        fitted_for_moran = fitted_for_moran[sample_idx]
        y_true_for_moran = y_true_for_moran[sample_idx]
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

    plots_dir = args.output_json.parent / f"{args.output_json.stem}_plots"
    plot_paths = _save_validation_plots(
        output_dir=plots_dir,
        scale_name=args.residual_scale,
        fitted_values=fitted_for_moran,
        y_true=y_true_for_moran,
        residuals=residual_for_moran,
    )

    output = {
        "input_csv": str(args.input_csv),
        "model_path": str(args.model_path),
        "residual_scale": args.residual_scale,
        "feature_source": feature_source,
        "selected_features": feature_list,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "rows_residential": int(len(df)),
        "rows_model": int(len(model_df)),
        "rows_with_valid_geometry": int(int(keep_mask.sum())),
        "sample_size": int(args.sample_size),
        "residual_diagnostics": residual_metrics,
        "morans_i": moran,
        "plots_dir": str(plots_dir),
        "plot_paths": plot_paths,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Validation output written to: {args.output_json}")
    print(json.dumps({"morans_i": moran}, indent=2))


if __name__ == "__main__":
    main()
