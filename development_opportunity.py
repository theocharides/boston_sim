"""
Score parcel development opportunity using three components.

This script computes a simple residential development opportunity score from:
1) Capacity upside: additional units allowed by zoning.
2) Market strength: neighborhood demand proxies.
3) Acquisition cost intensity: current value per existing unit.

Required inputs in parcels.csv:
- LU
- TOTAL_VALUE
- zoned_units

Current-units input:
- Uses RES_UNITS when present.
- Otherwise falls back to another *_RES_UNITS field if available.

Required market-strength inputs:
- median_hh_income (higher is stronger)
- neighborhood_walkability (higher is stronger)
- emp_dist_m (lower is stronger)

Scoring approach:
- additional_units = max(zoned_units - current_units, 0)
- capacity_upside_score = percentile_rank(additional_units)
- market_strength_score = mean of available market proxy percentile ranks
- acquisition_cost_intensity = TOTAL_VALUE / max(current_units, 1)
- acquisition_cost_score = percentile_rank(acquisition_cost_intensity)
- opportunity_score =
    w_capacity * capacity_upside_score
    + w_market * market_strength_score
    - w_cost * acquisition_cost_score

Tiers are assigned by opportunity_score percentile among residential rows:
- high: top 30%
- medium: middle 40%
- low: bottom 30%

Allocation from config:
- The script reads boston_units_to_add from a YAML config file.
- It allocates units to parcels in descending opportunity_score order.
- Each parcel can receive up to additional_units.
- Allocation stops once the Boston unit target is met or capacity is exhausted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESIDENTIAL_LU_CODES = {"A", "CD", "CM", "R1", "R2", "R3", "R4", "RC", "RL"}


def require_existing_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Score simple residential development opportunity."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root / "parcels.csv",
        help="Path to parcel CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "development_opportunity_scored.csv",
        help="Path to output scored CSV.",
    )
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=repo_root / "config" / "development_opportunity.yaml",
        help="Path to YAML config with boston_units_to_add.",
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
    args.config_yaml = require_existing_path(args.config_yaml, "Config YAML")
    args.output_csv = args.output_csv.expanduser().resolve()
    return args


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def percentile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Return percentile rank in [0, 1] for non-missing values."""
    ranked = series.rank(method="average", pct=True, ascending=ascending)
    return ranked.where(series.notna())


def get_current_units_series(df: pd.DataFrame) -> pd.Series:
    """Return current unit counts using RES_UNITS or a compatible fallback field."""
    if "RES_UNITS" in df.columns:
        return to_numeric(df["RES_UNITS"]).fillna(0.0).clip(lower=0.0)

    fallback_columns = [
        col
        for col in df.columns
        if col.upper().endswith("RES_UNITS") and col.upper() != "RES_UNITS"
    ]
    if fallback_columns:
        return to_numeric(df[fallback_columns[0]]).fillna(0.0).clip(lower=0.0)

    raise ValueError("Missing current-units field. Expected RES_UNITS or a compatible *_RES_UNITS column.")


def parse_yaml_scalar(raw_value: str):
    """Parse a simple YAML scalar value without external dependencies."""
    value = raw_value.strip()
    if not value:
        return ""
    if (value.startswith("\"") and value.endswith("\"")) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict[str, object]:
    """Load top-level key-value YAML pairs from a config file."""
    parsed: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key:
            continue
        parsed[key] = parse_yaml_scalar(raw_value)
    return parsed


def main() -> None:
    args = parse_args()

    print(f"Reading parcel data: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)
    config = load_simple_yaml(args.config_yaml)

    if "boston_units_to_add" not in config:
        raise ValueError(
            "Config YAML must include boston_units_to_add."
        )
    try:
        boston_units_to_add = int(config["boston_units_to_add"])
    except (TypeError, ValueError) as exc:
        raise ValueError("boston_units_to_add must be an integer.") from exc
    if boston_units_to_add < 0:
        raise ValueError("boston_units_to_add must be >= 0.")

    required_cols = [
        "LU",
        "TOTAL_VALUE",
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
    current_units = get_current_units_series(out)
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

    current_units_for_cost = current_units.clip(lower=1.0)
    acquisition_cost_intensity = total_value / current_units_for_cost
    acquisition_cost_intensity = acquisition_cost_intensity.where(is_residential)
    cost_score = percentile_rank(acquisition_cost_intensity, ascending=True)

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
    alloc_values = np.clip(boston_units_to_add - prev_cap, a_min=0, a_max=None)
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
    unmet_units = max(int(boston_units_to_add - total_allocated_units), 0)

    print(f"Wrote scored output: {args.output_csv}")
    print(f"Target units from config: {boston_units_to_add}")
    print(f"Allocated units: {total_allocated_units}")
    print(f"Unmet units (if capacity limited): {unmet_units}")
    print(f"Residential rows: {residential_rows}")
    print(f"Scored rows: {scored_rows}")
    print(f"High opportunity rows: {high_rows}")


if __name__ == "__main__":
    main()
