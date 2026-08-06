"""
Score residential development opportunity and allocate units by parcel capacity.

# Scoring Logic
- `additional_units = max(zoned_units - RES_UNITS, 0)`
- `capacity_upside_score = percentile_rank(additional_units)`
- `market_strength_score = mean(percentile_rank(median_hh_income),`
  `percentile_rank(neighborhood_walkability),`
  `percentile_rank(emp_dist_m, ascending=False))`
- `acquisition_cost_intensity = TOTAL_VALUE / max(zoned_units, 1)`
- `acquisition_cost_score = percentile_rank(acquisition_cost_intensity)`

Final weighted score:
- `opportunity_score = w_capacity * capacity_upside_score`
  `+ w_market * market_strength_score`
  `- w_cost * acquisition_cost_score`

Parcels are labeled by score percentile among scored residential rows:
- `high`: top 30% (`>= 0.70`)
- `medium`: middle 40% (`0.30` to `< 0.70`)
- `low`: bottom 30% (`< 0.30`)

Allocation Rules
- Parcels are sorted by `opportunity_score` (descending).
- Only eligible parcels can receive units:
  residential, scored, and `additional_units > 0`.
- Each parcel is capped at its `additional_units`.
- Allocation stops when `units_to_add` is fully assigned or no eligible
  capacity remains.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared_utils import load_simple_yaml, require_existing_path

RESIDENTIAL_LU_CODES = {"A", "CD", "CM", "R1", "R2", "R3", "R4", "RC", "RL"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Score simple residential development opportunity."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root / "parcels_preprocessed.csv",
        help="Path to parcel CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "development_opportunity_scored.csv",
        help="Path to output scored CSV.",
    )
    parser.add_argument(
        "--units-to-add",
        type=int,
        default=None,
        help="Optional units target. If provided, overrides config YAML units_to_add.",
    )
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=repo_root / "development_sim.yaml",
        help="Path to YAML config with units_to_add.",
    )
    parser.add_argument(
        "--w-capacity",
        type=float,
        default=1.0,
        help="Weight on capacity-upside component.",
    )
    parser.add_argument(
        "--w-market",
        type=float,
        default=1.0,
        help="Weight on market-strength component.",
    )
    parser.add_argument(
        "--w-cost",
        type=float,
        default=1.0,
        help="Weight on acquisition-cost-intensity component.",
    )

    args = parser.parse_args()
    args.input_csv = require_existing_path(args.input_csv, "Parcel CSV")
    if args.units_to_add is None:
        args.config_yaml = require_existing_path(args.config_yaml, "Config YAML")
    else:
        args.config_yaml = args.config_yaml.expanduser().resolve()
    args.output_csv = args.output_csv.expanduser().resolve()
    return args


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def percentile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Return percentile rank in [0, 1] for non-missing values."""
    ranked = series.rank(method="average", pct=True, ascending=ascending)
    return ranked.where(series.notna())


def main() -> None:
    args = parse_args()

    print(f"Reading parcel data: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)
    if args.units_to_add is not None:
        units_to_add = int(args.units_to_add)
    else:
        config = load_simple_yaml(args.config_yaml)
        if "units_to_add" not in config:
            raise ValueError(
                "Config YAML must include units_to_add."
            )
        try:
            units_to_add = int(config["units_to_add"])
        except (TypeError, ValueError) as exc:
            raise ValueError("units_to_add must be an integer.") from exc
    if units_to_add < 0:
        raise ValueError("units_to_add must be >= 0.")

    required_cols = [
        "LU",
        "TOTAL_VALUE",
        "RES_UNITS",
        "zoned_units",
        "median_hh_income",
        "neighborhood_walkability",
        "emp_dist_m",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Run capacity/calculate_unit_capacity.py first if zoned_units is missing."
        )

    lu = df["LU"].astype("string").fillna("").str.strip().str.upper()
    is_residential = lu.isin(RESIDENTIAL_LU_CODES)

    out = df.copy()

    zoned_units = to_numeric(out["zoned_units"])
    current_units = to_numeric(out["RES_UNITS"])
    total_value = to_numeric(out["TOTAL_VALUE"])

    additional_units = (zoned_units - current_units).clip(lower=0.0)
    additional_units = additional_units.where(is_residential)

    capacity_score = percentile_rank(additional_units, ascending=True)

    market_parts: list[pd.Series] = [
        percentile_rank(to_numeric(out["median_hh_income"]), ascending=True),
        percentile_rank(to_numeric(out["neighborhood_walkability"]), ascending=True),
        percentile_rank(to_numeric(out["emp_dist_m"]), ascending=False),
    ]
    market_strength = pd.concat(market_parts, axis=1).mean(axis=1, skipna=True)

    market_strength = market_strength.where(is_residential)

    zoned_units_for_cost = zoned_units.clip(lower=1.0)
    acquisition_cost_intensity = total_value / zoned_units_for_cost
    acquisition_cost_intensity = acquisition_cost_intensity.where(is_residential)
    cost_score = percentile_rank(acquisition_cost_intensity, ascending=True)
    no_zoned_capacity = zoned_units.isna() | (zoned_units <= 0)
    acquisition_cost_intensity = acquisition_cost_intensity.where(~(is_residential & no_zoned_capacity))
    cost_score = cost_score.mask(is_residential & no_zoned_capacity, 0.0)

    opportunity_score = (
        args.w_capacity * capacity_score
        + args.w_market * market_strength
        - args.w_cost * cost_score
    )
    opportunity_score = opportunity_score.where(is_residential)

    score_pct = percentile_rank(opportunity_score, ascending=True)
    tier = pd.Series(pd.NA, index=out.index, dtype="string")
    tier.loc[score_pct >= 0.70] = "high"
    tier.loc[(score_pct >= 0.30) & (score_pct < 0.70)] = "medium"
    tier.loc[score_pct < 0.30] = "low"

    out["current_units"] = current_units.where(is_residential).round().astype("Int64")
    out["additional_units"] = additional_units.round().astype("Int64")
    out["capacity_upside_score"] = capacity_score
    out["market_strength_score"] = market_strength
    out["acquisition_cost_intensity"] = acquisition_cost_intensity
    out["acquisition_cost_score"] = cost_score
    out["opportunity_score"] = opportunity_score
    out["opportunity_tier"] = tier

    eligible_mask = (
        is_residential
        & out["opportunity_score"].notna()
        & out["additional_units"].notna()
        & (out["additional_units"] > 0)
    )
    eligible = out.loc[eligible_mask, ["opportunity_score", "additional_units"]].copy()
    eligible = eligible.sort_values("opportunity_score", ascending=False)
    eligible_cap = pd.to_numeric(eligible["additional_units"], errors="coerce").fillna(0)

    cum_cap = eligible_cap.cumsum()
    prev_cap = cum_cap - eligible_cap
    alloc_values = np.clip(units_to_add - prev_cap, a_min=0, a_max=None)
    alloc_values = np.minimum(alloc_values, eligible_cap)

    allocated_units = pd.Series(0, index=out.index, dtype="int64")
    allocated_units.loc[eligible.index] = alloc_values.astype("int64")
    out["allocated_units"] = allocated_units.astype("Int64")
    out["selected_for_development"] = out["allocated_units"] > 0

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    residential_rows = int(is_residential.sum())
    scored_rows = int(out["opportunity_score"].notna().sum())
    high_rows = int((out["opportunity_tier"] == "high").sum())
    total_allocated_units = int(out["allocated_units"].fillna(0).sum())
    unmet_units = max(int(units_to_add - total_allocated_units), 0)

    print(f"Wrote scored output: {args.output_csv}")
    print(f"Target units: {units_to_add}")
    print(f"Allocated units: {total_allocated_units}")
    print(f"Unmet units (if capacity limited): {unmet_units}")
    print(f"Residential rows: {residential_rows}")
    print(f"Scored rows: {scored_rows}")
    print(f"High opportunity rows: {high_rows}")


if __name__ == "__main__":
    main()
