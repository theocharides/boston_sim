"""Compare hedonic feature combinations and optionally write the best spec JSON."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hedonic.common.modeling_common import (
    LOC_NEIGHBORHOOD_FEATURES,
    STRUCTURAL_FEATURES,
    TARGET_COL,
    available_features,
    build_ridge_pipeline,
    evaluate_log_and_level,
    get_ridge_alpha,
    infer_feature_types,
    prepare_model_df,
    subset_residential_rows,
)
from shared_utils import require_existing_path


def _unique_output_json_path(output_json: Path) -> Path:
    if not output_json.exists():
        return output_json

    parent = output_json.parent
    stem = output_json.stem
    suffix = output_json.suffix
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_csv = repo_root / "inputs" / "parcels_processed_for_hedonic.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate many feature combinations on holdout metrics, save a comparison CSV, "
            "and write the best spec JSON."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=default_input_csv,
        help="Path to processed parcel CSV for hedonic modeling.",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=repo_root / "hedonic" / "workflow" / "residential_hedonic_model_comparison.csv",
        help="Output CSV with metrics for each tested feature combination.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=repo_root / "hedonic" / "workflow" / "residential_hedonic_feature_spec.json",
        help="Output path for selected feature spec JSON.",
    )
    parser.add_argument(
        "--skip-selected-spec",
        action="store_true",
        help="Skip writing selected spec JSON and only output comparison CSV.",
    )
    parser.add_argument(
        "--selection-rule",
        type=str,
        default="balanced_holdout",
        choices=["balanced_holdout", "best_primary"],
        help="How to select best model from comparison rows.",
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        default="r2_level",
        choices=["r2_level", "r2_log"],
        help="Primary metric if --selection-rule best_primary is used.",
    )
    parser.add_argument(
        "--secondary-metric",
        type=str,
        default="r2_log",
        choices=["r2_level", "r2_log"],
        help="Tie-break metric if --selection-rule best_primary is used.",
    )
    parser.add_argument(
        "--min-r2-log",
        type=float,
        default=.55,
        help="Minimum acceptable R-squared on the log-price holdout.",
    )
    parser.add_argument(
        "--min-r2-level",
        type=float,
        default=0.50,
        help="Minimum acceptable level-scale R-squared on the holdout.",
    )
    parser.add_argument(
        "--max-rmse-log",
        type=float,
        default=0.25,
        help="Maximum acceptable RMSE on the log-price holdout.",
    )
    parser.add_argument(
        "--max-mae-level",
        type=float,
        default=300000.0,
        help="Maximum acceptable mean absolute error on raw dollar value.",
    )
    parser.add_argument(
        "--require-walkability",
        action="store_true",
        help="Only evaluate combinations that include neighborhood_walkability.",
    )
    parser.add_argument(
        "--require-neighborhood",
        action="store_true",
        help="Only evaluate combinations that include the neighborhood fixed-effect feature.",
    )
    parser.add_argument(
        "--neighborhood-feature",
        type=str,
        default="neighborhood_name",
        help="Column name to use for neighborhood fixed effects when --require-neighborhood is set.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout share for each model evaluation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for all train/test splits.",
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=1,
        help="Minimum feature count per tested combination.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="Maximum feature count per tested combination. 0 means no limit.",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=50000,
        help=(
            "Safety cap on number of feature combinations to evaluate. "
            "Reduce with --max-features or increase this cap explicitly."
        ),
    )

    args = parser.parse_args()
    args.input_csv = require_existing_path(args.input_csv, "Input CSV")
    args.comparison_csv = args.comparison_csv.expanduser().resolve()
    args.output_json = args.output_json.expanduser().resolve()
    return args


def _normalize_metric(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    _min = values.min()
    _max = values.max()
    if pd.isna(_min) or pd.isna(_max) or _max == _min:
        return pd.Series(0.0, index=values.index)
    return (values - _min) / (_max - _min)


def _balanced_holdout_score(df: pd.DataFrame) -> pd.Series:
    metrics: list[pd.Series] = []

    for metric in ["r2_log", "r2_level"]:
        if metric in df.columns:
            metrics.append(_normalize_metric(df[metric]).rename(f"{metric}_score"))

    for metric in ["rmse_log", "mae_level"]:
        if metric in df.columns:
            loss = pd.to_numeric(df[metric], errors="coerce")
            inverted = loss.max() - loss
            metrics.append(_normalize_metric(inverted).rename(f"{metric}_score"))

    if not metrics:
        raise ValueError("No compatible holdout metrics found for balanced scoring.")

    return pd.concat(metrics, axis=1).mean(axis=1)


def _features_to_str(features: list[str]) -> str:
    return ", ".join(features)


def _meets_acceptance_thresholds(row: pd.Series, args: argparse.Namespace) -> bool:
    metric_checks = [
        ("r2_log", row.get("r2_log", float("nan")) >= args.min_r2_log),
        ("r2_level", row.get("r2_level", float("nan")) >= args.min_r2_level),
        ("rmse_log", row.get("rmse_log", float("nan")) <= args.max_rmse_log),
        ("mae_level", row.get("mae_level", float("nan")) <= args.max_mae_level),
    ]
    return all(pass_check for _, pass_check in metric_checks)


def _select_best_row(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, str]:
    if args.selection_rule == "best_primary":
        best = df.sort_values(
            [args.primary_metric, args.secondary_metric],
            ascending=[False, False],
        ).iloc[0]
        criterion = (
            f"highest {args.primary_metric}, tie-break highest {args.secondary_metric}, "
            "excluding errored rows and requiring acceptable holdout thresholds"
        )
        return best, criterion

    scored = df.copy()
    scored["balanced_holdout_score"] = _balanced_holdout_score(scored)
    best = scored.sort_values(
        ["balanced_holdout_score", "r2_level", "r2_log"],
        ascending=[False, False, False],
    ).iloc[0]
    criterion = (
        "acceptable holdout thresholds must be met for r2_log, rmse_log, r2_level, and mae_level; "
        "among accepted rows, choose the highest balanced holdout score"
    )
    return best, criterion


def _candidate_feature_sets(
    available_pool: list[str],
    min_features: int,
    max_features: int,
    require_walkability: bool,
    require_neighborhood: bool,
    neighborhood_feature: str,
) -> list[list[str]]:
    if not available_pool:
        return []

    max_k = max_features if max_features > 0 else len(available_pool)
    max_k = min(max_k, len(available_pool))
    min_k = max(1, min_features)
    if min_k > max_k:
        raise ValueError(f"Invalid feature bounds: min_features={min_features}, max_features={max_features}")

    combos: list[list[str]] = []
    for k in range(min_k, max_k + 1):
        for combo in itertools.combinations(available_pool, k):
            combo_list = list(combo)
            if require_walkability and "neighborhood_walkability" not in combo_list:
                continue
            if require_neighborhood and neighborhood_feature not in combo_list:
                continue
            combos.append(combo_list)
    return combos


def main() -> None:
    args = parse_args()

    print(f"Reading parcel data: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)
    raw_rows = len(df)
    df = subset_residential_rows(df, strict=True)
    print(f"Residential subset rows: {len(df)} of {raw_rows}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Required target column missing: {TARGET_COL}")

    feature_pool = [*STRUCTURAL_FEATURES, *LOC_NEIGHBORHOOD_FEATURES]
    if args.neighborhood_feature and args.neighborhood_feature not in feature_pool:
        feature_pool.append(args.neighborhood_feature)
    available_pool = available_features(df, feature_pool)

    if args.require_walkability and "neighborhood_walkability" not in available_pool:
        raise ValueError(
            "--require-walkability was set, but neighborhood_walkability is not present in the input data."
        )
    if args.require_neighborhood and args.neighborhood_feature not in available_pool:
        raise ValueError(
            "--require-neighborhood was set, but the neighborhood feature is not present in the input data: "
            f"{args.neighborhood_feature}"
        )

    candidate_sets = _candidate_feature_sets(
        available_pool=available_pool,
        min_features=args.min_features,
        max_features=args.max_features,
        require_walkability=args.require_walkability,
        require_neighborhood=args.require_neighborhood,
        neighborhood_feature=args.neighborhood_feature,
    )
    if not candidate_sets:
        raise ValueError("No feature combinations were generated for evaluation.")

    if len(candidate_sets) > args.max_combinations:
        raise ValueError(
            "Requested feature search is too large: "
            f"{len(candidate_sets):,} combinations exceeds --max-combinations={args.max_combinations:,}. "
            "Use --max-features to narrow the search or raise --max-combinations explicitly."
        )

    print(f"Evaluating {len(candidate_sets)} feature combinations...")
    rows: list[dict[str, object]] = []

    for features in candidate_sets:
        row: dict[str, object] = {
            "n_features": len(features),
            "features": _features_to_str(features),
            "rows_train": 0,
            "rows_test": 0,
            "r2_log": float("nan"),
            "rmse_log": float("nan"),
            "r2_level": float("nan"),
            "mae_level": float("nan"),
            "alpha": float("nan"),
            "error": "",
        }
        try:
            numeric_features, categorical_features = infer_feature_types(df, features)
            model_df = prepare_model_df(
                df,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                target_col=TARGET_COL,
            )
            if len(model_df) < 100:
                raise ValueError("Too few valid rows after filtering for this feature set.")

            X = model_df[[*numeric_features, *categorical_features]]
            y = model_df[TARGET_COL].pipe(lambda s: pd.Series(np.log1p(s), index=s.index))

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
            model.fit(X_train, y_train)
            pred_log = model.predict(X_test)
            metrics = evaluate_log_and_level(y_train=y_train, y_test=y_test, pred_log=pred_log)

            row.update(
                {
                    "rows_train": int(len(X_train)),
                    "rows_test": int(len(X_test)),
                    "r2_log": float(metrics["r2_log"]),
                    "rmse_log": float(metrics["rmse_log"]),
                    "r2_level": float(metrics["r2_level"]),
                    "mae_level": float(metrics["mae_level"]),
                    "alpha": float(get_ridge_alpha(model)),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive capture for full comparison output
            row["error"] = str(exc)

        rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("No comparison rows were produced.")

    valid = results[results["error"].astype("string").str.strip() == ""].copy()
    if valid.empty:
        raise ValueError("All model combinations failed; see comparison CSV for errors.")

    acceptable = valid[valid.apply(lambda row: _meets_acceptance_thresholds(row, args), axis=1)].copy()
    if acceptable.empty:
        raise ValueError(
            "No feature set met the acceptance thresholds: "
            f"r2_log >= {args.min_r2_log}, r2_level >= {args.min_r2_level}, "
            f"rmse_log <= {args.max_rmse_log}, mae_level <= {args.max_mae_level}. "
            "Relax the thresholds or inspect the comparison CSV."
        )

    if args.selection_rule == "balanced_holdout":
        acceptable["balanced_holdout_score"] = _balanced_holdout_score(acceptable)
        results = results.merge(
            acceptable[["features", "balanced_holdout_score"]],
            on="features",
            how="left",
        )
        results = results.sort_values(
            ["balanced_holdout_score", "r2_level", "r2_log"],
            ascending=[False, False, False],
        )
    else:
        results = results.sort_values(
            [args.primary_metric, args.secondary_metric],
            ascending=[False, False],
        )

    args.comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.comparison_csv, index=False)
    print(f"Comparison CSV written: {args.comparison_csv}")

    write_selected_spec = not args.skip_selected_spec
    if not write_selected_spec:
        return

    best, selection_criterion = _select_best_row(acceptable, args)
    features = [part.strip() for part in str(best["features"]).split(",") if part.strip()]

    payload = {
        "name": "residential_level_balanced_selected",
        "selected_from": str(args.comparison_csv).replace("\\", "/"),
        "selection_criterion": selection_criterion,
        "acceptance_thresholds": {
            "min_r2_log": args.min_r2_log,
            "min_r2_level": args.min_r2_level,
            "max_rmse_log": args.max_rmse_log,
            "max_mae_level": args.max_mae_level,
        },
        "features": features,
        "metrics_snapshot": {
            "r2_log": float(best["r2_log"]),
            "rmse_log": float(best["rmse_log"]),
            "r2_level": float(best["r2_level"]),
            "mae_level": float(best["mae_level"]),
            "rows_train": int(best["rows_train"]) if pd.notna(best["rows_train"]) else 0,
            "rows_test": int(best["rows_test"]) if pd.notna(best["rows_test"]) else 0,
            "balanced_holdout_score": float(best.get("balanced_holdout_score", 0.0)),
        },
    }

    selected_spec_path = _unique_output_json_path(args.output_json)
    selected_spec_path.parent.mkdir(parents=True, exist_ok=True)
    selected_spec_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if selected_spec_path != args.output_json:
        print(f"Output JSON already existed; wrote new file instead: {selected_spec_path}")
    print(f"Selected spec written: {selected_spec_path}")
    print("Best features:", ", ".join(features))


if __name__ == "__main__":
    main()