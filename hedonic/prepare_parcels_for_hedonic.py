"""Prepare the canonical preprocessed parcel table for residential hedonic modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hedonic.common.modeling_common import RESIDENTIAL_LU_CODES, subset_residential_rows
from shared_utils import require_existing_path


def prepare_hedonic_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to residential rows, retain hedonic columns, and clip obvious outliers."""
    residential = subset_residential_rows(df, strict=True)

    columns = [
        "TOTAL_VALUE",
        "LU",
        "LIVING_AREA",
        "LAND_SF",
        "GROSS_AREA",
        "INT_COND",
        "OVERALL_COND",
        "RES_FLOOR",
        "YR_BUILT",
        "BLDG_TYPE",
        "neighborhood_walkability",
        "emp_dist_m",
        "median_hh_income",
        "neighborhood_name",
        "geometry",
    ]
    missing = [column for column in columns if column not in residential.columns]
    if missing:
        raise ValueError(f"Input data is missing required hedonic columns: {missing}")

    prepared = residential.loc[:, columns].copy()

    prepared["TOTAL_VALUE"] = pd.to_numeric(prepared["TOTAL_VALUE"], errors="coerce")
    prepared["LAND_SF"] = pd.to_numeric(prepared["LAND_SF"], errors="coerce")
    prepared["GROSS_AREA"] = pd.to_numeric(prepared["GROSS_AREA"], errors="coerce")
    prepared["LIVING_AREA"] = pd.to_numeric(prepared["LIVING_AREA"], errors="coerce")
    prepared["RES_FLOOR"] = pd.to_numeric(prepared["RES_FLOOR"], errors="coerce")
    prepared["YR_BUILT"] = pd.to_numeric(prepared["YR_BUILT"], errors="coerce")
    prepared["median_hh_income"] = pd.to_numeric(prepared["median_hh_income"], errors="coerce")
    prepared["emp_dist_m"] = pd.to_numeric(prepared["emp_dist_m"], errors="coerce")
    prepared["neighborhood_walkability"] = pd.to_numeric(prepared["neighborhood_walkability"], errors="coerce")

    prepared["TOTAL_VALUE"] = prepared["TOTAL_VALUE"].clip(lower=prepared["TOTAL_VALUE"].quantile(0.02))
    prepared["TOTAL_VALUE"] = prepared["TOTAL_VALUE"].clip(upper=prepared["TOTAL_VALUE"].quantile(0.98))

    prepared["LAND_SF"] = prepared["LAND_SF"].clip(lower=500)
    prepared["LAND_SF"] = prepared["LAND_SF"].clip(upper=prepared["LAND_SF"].quantile(0.99))

    prepared["GROSS_AREA"] = prepared["GROSS_AREA"].clip(lower=500)
    prepared["GROSS_AREA"] = prepared["GROSS_AREA"].clip(upper=prepared["GROSS_AREA"].quantile(0.95))

    prepared["LIVING_AREA"] = prepared["LIVING_AREA"].clip(lower=500)
    prepared["LIVING_AREA"] = prepared["LIVING_AREA"].clip(upper=prepared["LIVING_AREA"].quantile(0.95))

    prepared["RES_FLOOR"] = prepared["RES_FLOOR"].clip(lower=1)
    prepared["YR_BUILT"] = prepared["YR_BUILT"].clip(upper=2030)
    prepared["median_hh_income"] = prepared["median_hh_income"].clip(lower=prepared["median_hh_income"].quantile(0.01))

    prepared = prepared.dropna(subset=[
        "TOTAL_VALUE",
        "LAND_SF",
        "GROSS_AREA",
        "LIVING_AREA",
        "INT_COND",
        "OVERALL_COND",
        "RES_FLOOR",
        "YR_BUILT",
        "BLDG_TYPE",
        "neighborhood_walkability",
        "emp_dist_m",
        "median_hh_income",
        "neighborhood_name",
        "geometry",
    ]).copy()

    return prepared


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Prepare the canonical preprocessed parcel table for residential hedonic modeling."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels_preprocessed.csv",
        help="Canonical preprocessed parcel CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels_processed_for_hedonic.csv",
        help="Prepared parcel CSV for hedonic model estimation.",
    )
    args = parser.parse_args()
    args.input_csv = require_existing_path(args.input_csv, "Input parcel CSV")
    args.output_csv = args.output_csv.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv, low_memory=False)
    prepared = prepare_hedonic_dataframe(df)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output_csv, index=False)
    print(f"Prepared hedonic input rows: {len(prepared):,}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
