"""
Collapse assessor account rows to one record per parcel geometry and
subset data to relevant columns.

This script converts the assessor table from account-level records (including
condo units) to parcel-level records. Condo/unit PIDs that do not directly
match a parcel polygon are mapped to a base parcel PID (first 7 digits + 000)
when possible, then all rows are aggregated to one row per parcel.

The final output is guaranteed to have one row per parcel geometry by joining
aggregated assessor attributes onto the full parcel geometry key list.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


# Map source assessor columns to final cleaned output names.
OUTPUT_COLUMN_MAP: dict[str, str] = {
    "PID": "PID",
    "CM_ID": "CONDO_ID",
    "NUM_BLDGS": "NUM_BLDGS",
    "LU": "LU",
    "LU_DESC": "LU_DESC",
    "BLDG_TYPE": "BLDG_TYPE",
    "RES_FLOOR": "RES_FLOOR",
    "RES_UNITS": "CONDO_RES_UNITS",
    "TT_RMS": "TT_RMS",
    "BED_RMS": "BED_RMS",
    "FULL_BTH": "FULL_BTH",
    "HLF_BTH": "HLF_BTH",
    "KITCHENS": "KITCHENS",
    "OVERALL_COND": "OVERALL_COND",
    "INT_COND": "INT_COND",
    "EXT_COND": "EXT_COND",
    "NUM_PARKING": "NUM_PARKING",
    "STRUCTURE_CLASS": "STRUCTURE_CLASS",
    "YR_REMODEL": "YR_REMODEL",
    "YR_BUILT": "YR_BUILT",
    "LAND_VALUE": "LAND_VALUE",
    "BLDG_VALUE": "BLDG_VALUE",
    "TOTAL_VALUE": "TOTAL_VALUE",
    "LAND_SF": "LAND_SF",
    "GROSS_AREA": "GROSS_AREA",
    "LIVING_AREA": "LIVING_AREA",
}

SOURCE_COLUMNS: list[str] = list(OUTPUT_COLUMN_MAP.keys())
OUTPUT_COLUMNS: list[str] = list(OUTPUT_COLUMN_MAP.values())

CONDO_LU_CODES = {"CD", "CP", "CC", "CM"}

# Columns to sum for condos
SUM_COLUMNS = {
    "LAND_VALUE",
    "BLDG_VALUE",
    "TOTAL_VALUE",
    "RES_UNITS",
    "GROSS_AREA",
    "LIVING_AREA",
    "BED_RMS",
    "FULL_BTH",
    "HLF_BTH",
    "KITCHENS",
    "TT_RMS",
    "NUM_PARKING",
}

# Columns not to sum for condos
MAX_COLUMNS = {
    "CM_ID",
    "NUM_BLDGS",
    "RES_FLOOR",
    "LAND_SF",
    "YR_BUILT",
    "YR_REMODEL",
}

def normalize_pid(series: pd.Series) -> pd.Series:
    """Normalize parcel/account IDs to a 10-digit numeric string."""
    return series.astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)


def to_base_pid(pid_norm: pd.Series) -> pd.Series:
    """Convert an account PID to a likely base parcel PID."""
    return pid_norm.str.slice(0, 7) + "000"


def clean_numeric(series: pd.Series) -> pd.Series:
    """Parse numeric-looking assessor fields that may include commas or symbols."""
    as_text = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    as_text = as_text.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(as_text, errors="coerce")


def require_existing_path(path: Path, label: str) -> Path:
    """Return a resolved path or raise a clear error for missing inputs."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Collapse Boston assessors to one row per parcel geometry."
    )
    parser.add_argument(
        "--assessors-csv",
        type=Path,
        default=repo_root / "data" / "boston_parcel_assessors.csv",
        help="Path to assessor CSV.",
    )
    parser.add_argument(
        "--parcel-shapes",
        type=Path,
        default=repo_root / "data" / "boston_parcel_shapes.geojson",
        help="Path to parcel polygon file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "data" / "boston_parcel_assessors_cleaned.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()
    args.assessors_csv = require_existing_path(args.assessors_csv, "Assessors CSV")
    args.parcel_shapes = require_existing_path(args.parcel_shapes, "Parcel shapes file")
    args.output_csv = args.output_csv.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()

    print(f"Reading assessors: {args.assessors_csv}")
    assessors = pd.read_csv(args.assessors_csv, low_memory=False)

    print(f"Reading parcel geometry keys: {args.parcel_shapes}")
    parcels = gpd.read_file(args.parcel_shapes, columns=["MAP_PAR_ID", "geometry"])
    parcels = parcels[["MAP_PAR_ID", "geometry"]].copy()
    parcels["_parcel_key"] = normalize_pid(parcels["MAP_PAR_ID"])
    parcels = parcels.drop_duplicates(subset=["_parcel_key"], keep="first")

    parcel_keys = set(parcels["_parcel_key"])

    assessors["_pid_norm"] = normalize_pid(assessors["PID"])
    assessors["_base_pid"] = to_base_pid(assessors["_pid_norm"])
    assessors["_parcel_key"] = np.where(
        assessors["_pid_norm"].isin(parcel_keys),
        assessors["_pid_norm"],
        np.where(assessors["_base_pid"].isin(parcel_keys), assessors["_base_pid"], pd.NA),
    )

    matched = assessors[assessors["_parcel_key"].notna()].copy()

    print("Preparing columns for aggregation...")
    output_source_columns = [
        col for col in SOURCE_COLUMNS if col != "PID" and col in matched.columns
    ]

    # Restrict numeric parsing to columns that are explicitly numeric in the cleaned schema.
    numeric_sum_columns = [col for col in SUM_COLUMNS if col in output_source_columns]
    numeric_max_columns = [col for col in MAX_COLUMNS if col in output_source_columns]
    numeric_columns = numeric_sum_columns + [
        col for col in numeric_max_columns if col not in numeric_sum_columns
    ]
    text_columns = [
        col for col in output_source_columns if col not in set(numeric_columns)
    ]

    if numeric_columns:
        print(f"Parsing numeric columns ({len(numeric_columns)})...")
        for col in numeric_columns:
            matched[col] = clean_numeric(matched[col])

    if text_columns:
        # Normalize blank strings once so groupby.first can skip them efficiently.
        matched[text_columns] = matched[text_columns].replace(r"^\s*$", pd.NA, regex=True)

    print("Collapsing assessor rows to parcel level...")
    grouped = matched.groupby("_parcel_key", sort=False, observed=True)

    collapsed_parts: list[pd.DataFrame] = []

    if numeric_sum_columns:
        print(f"Aggregating numeric sum columns ({len(numeric_sum_columns)})...")
        collapsed_parts.append(grouped[numeric_sum_columns].sum(min_count=1))

    if numeric_max_columns:
        print(f"Aggregating numeric max columns ({len(numeric_max_columns)})...")
        collapsed_parts.append(grouped[numeric_max_columns].max())

    if text_columns:
        print(f"Aggregating text columns ({len(text_columns)})...")
        collapsed_parts.append(grouped[text_columns].first())

    if collapsed_parts:
        collapsed = pd.concat(collapsed_parts, axis=1).reset_index()
    else:
        collapsed = grouped.size().rename("_unused").reset_index()[["_parcel_key"]]

    print("Joining to full parcel key list to guarantee row parity with geometry...")
    result = parcels[["_parcel_key", "MAP_PAR_ID"]].merge(
        collapsed,
        on="_parcel_key",
        how="left",
    )

    result["PID"] = result["_parcel_key"]

    rename_map = {
        source_col: output_col
        for source_col, output_col in OUTPUT_COLUMN_MAP.items()
        if source_col != output_col
    }
    result = result.rename(columns=rename_map)

    # Emit only the requested final schema in the requested order.
    for column_name in OUTPUT_COLUMNS:
        if column_name not in result.columns:
            result[column_name] = pd.NA
    result = result[OUTPUT_COLUMNS]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)

    print(f"Output rows: {len(result):,}")
    print(f"Unique parcel keys in geometry: {parcels['_parcel_key'].nunique():,}")
    print(f"Matched assessor rows used: {len(matched):,}")
    print(f"Unmatched assessor rows dropped: {len(assessors) - len(matched):,}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
