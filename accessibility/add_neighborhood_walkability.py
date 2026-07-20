"""
Adds a neighborhood walkability score to parcel records.

This script computes a fast, deterministic walkability proxy from existing
parcel and zoning attributes already present in `inputs/parcels.csv`.

The score is a weighted composite of available components:
- Built intensity proxy (GROSS_AREA / LAND_SF)
- Zoning walkability capacity (max_far, max_floors)

The final score is normalized to a 0-100 scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import require_existing_path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Add neighborhood walkability score to parcel records."
    )
    parser.add_argument(
        "--parcels-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels.csv",
        help="Path to parcels CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels.csv",
        help="Output CSV path (can be same as input for in-place update).",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default="neighborhood_walkability",
        help="Name of output walkability score column.",
    )

    args = parser.parse_args()
    args.parcels_csv = require_existing_path(args.parcels_csv, "Parcels CSV")
    args.output_csv = args.output_csv.expanduser().resolve()
    return args


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _minmax(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)

    s_min = float(valid.min())
    s_max = float(valid.max())
    if np.isclose(s_min, s_max):
        out = pd.Series(np.nan, index=series.index, dtype=float)
        out.loc[valid.index] = 0.5
        return out

    out = (series - s_min) / (s_max - s_min)
    return out.clip(0.0, 1.0)


def main() -> None:
    args = parse_args()

    print(f"Reading parcels: {args.parcels_csv}")
    df = pd.read_csv(args.parcels_csv, low_memory=False)

    land_sf = _safe_numeric(df, "LAND_SF")
    gross_area = _safe_numeric(df, "GROSS_AREA")
    max_far = _safe_numeric(df, "max_far")
    max_floors = _safe_numeric(df, "max_floors")

    # Built intensity proxy: higher gross area per land area suggests denser, more walkable fabric.
    intensity_ratio = gross_area / land_sf.replace(0, np.nan)
    intensity_score = _minmax(np.log1p(intensity_ratio))

    # Zoning capacity proxy: allows more mixed/dense development potential.
    far_score = _minmax(max_far)
    floors_score = _minmax(max_floors)
    zoning_score = pd.concat([far_score, floors_score], axis=1).mean(axis=1, skipna=True)

    components = pd.DataFrame(
        {
            "intensity": intensity_score,
            "zoning": zoning_score,
        }
    )

    weights = {
        "intensity": 0.50,
        "zoning": 0.50,
    }

    weighted_sum = pd.Series(0.0, index=df.index, dtype=float)
    weight_total = pd.Series(0.0, index=df.index, dtype=float)

    for name, weight in weights.items():
        comp = components[name]
        valid = comp.notna()
        weighted_sum.loc[valid] += comp.loc[valid] * weight
        weight_total.loc[valid] += weight

    score = pd.Series(np.nan, index=df.index, dtype=float)
    valid_rows = weight_total > 0
    score.loc[valid_rows] = (weighted_sum.loc[valid_rows] / weight_total.loc[valid_rows]) * 100.0

    df[args.score_column] = score.round(2)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    n_with_score = int(df[args.score_column].notna().sum())
    print(f"Rows written: {len(df):,}")
    print(f"Rows with walkability score: {n_with_score:,}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
