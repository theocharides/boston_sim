"""
Add one zoning capacity column to parcels.csv.

This script filters to residential LU codes and calculates parcel capacity in units
from zoning constraints, then writes a single new column named zoned_units.

Method:
1) Residential parcels are LU in {A, CD, CM, R1, R2, R3, R4, RC, RL}.
2) FAR units = floor((LAND_SF * max_far) / unit_size_sf) when LAND_SF and max_far > 0.
3) DUA units = floor(max_dua * (LAND_SF / 43,560)) when LAND_SF and max_dua > 0.
4) zoned_units = min(FAR units, DUA units) when both exist, otherwise the one that exists.
5) Non-residential or missing-zoning rows keep zoned_units as blank.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESIDENTIAL_LU_CODES = {"A", "CD", "CM", "R1", "R2", "R3", "R4", "RC", "RL"}
SQFT_PER_ACRE = 43560.0


def require_existing_path(path: Path, label: str) -> Path:
    """Resolve a path and raise a clear error when it does not exist."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Calculate residential zoning unit capacity for each parcel."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels.csv",
        help="Path to parcel CSV.",
    )
    parser.add_argument(
        "--unit-size-sf",
        type=float,
        default=1000.0,
        help="Assumed average unit size in square feet for FAR-based capacity.",
    )

    args = parser.parse_args()
    args.input_csv = require_existing_path(args.input_csv, "Parcel CSV")
    args.output_csv = args.input_csv

    if args.unit_size_sf <= 0:
        raise ValueError("--unit-size-sf must be positive.")

    return args


def to_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric with coercion."""
    return pd.to_numeric(series, errors="coerce")


def main() -> None:
    args = parse_args()

    print(f"Reading parcel data: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)

    required_cols = ["LU", "LAND_SF", "max_far", "max_dua"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    lu = df["LU"].astype("string").fillna("").str.strip().str.upper()
    is_residential = lu.isin(RESIDENTIAL_LU_CODES)

    land_sf = to_numeric(df["LAND_SF"])
    max_far = to_numeric(df["max_far"])
    max_dua = to_numeric(df["max_dua"])
    lot_area_acres = land_sf / SQFT_PER_ACRE

    far_units = np.floor((land_sf * max_far) / args.unit_size_sf)
    far_units = pd.Series(far_units, index=df.index)
    far_valid = is_residential & land_sf.gt(0) & max_far.gt(0)
    far_units = far_units.where(far_valid)

    dua_units = np.floor(max_dua * lot_area_acres)
    dua_units = pd.Series(dua_units, index=df.index)
    dua_valid = is_residential & land_sf.gt(0) & max_dua.gt(0)
    dua_units = dua_units.where(dua_valid)

    both = far_units.notna() & dua_units.notna()
    only_far = far_units.notna() & dua_units.isna()
    only_dua = far_units.isna() & dua_units.notna()

    zoning_capacity = pd.Series(np.nan, index=df.index, dtype="float64")
    zoning_capacity.loc[both] = np.minimum(far_units.loc[both], dua_units.loc[both])
    zoning_capacity.loc[only_far] = far_units.loc[only_far]
    zoning_capacity.loc[only_dua] = dua_units.loc[only_dua]
    zoning_capacity = zoning_capacity.clip(lower=0)

    df["zoned_units"] = zoning_capacity.round().astype("Int64")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    residential_count = int(is_residential.sum())
    known_capacity_count = int(df["zoned_units"].notna().sum())
    total_units = float(df["zoned_units"].fillna(0).sum())

    print(f"Wrote capacity output: {args.output_csv}")
    print(f"Residential parcels: {residential_count}")
    print(f"Residential parcels with zoned_units: {known_capacity_count}")
    print(f"Total zoned_units (sum): {total_units:,.0f}")


if __name__ == "__main__":
    main()
