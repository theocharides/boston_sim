"""Post-process simulation outputs into top-level summary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared_utils import require_existing_path


RESIDENTIAL_LU_CODES = {"A", "CD", "CM", "R1", "R2", "R3", "R4", "RC", "RL"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Summarize where simulated housing units were added in Boston."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root / "outputs" / "parcels_simulated.csv",
        help="Path to simulated parcel CSV with allocated_units.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs",
        help="Directory where top-level post-processing outputs are written.",
    )
    parser.add_argument(
        "--step-summaries-json",
        type=Path,
        default=None,
        help="Optional JSON file containing per-step simulation summaries.",
    )
    parser.add_argument(
        "--neighborhood-step-summaries-json",
        type=Path,
        default=None,
        help="Optional JSON file containing per-step neighborhood summaries.",
    )
    parser.add_argument(
        "--lu-step-summaries-json",
        type=Path,
        default=None,
        help="Optional JSON file containing per-step land-use summaries.",
    )

    args = parser.parse_args()
    args.input_csv = require_existing_path(args.input_csv, "Simulated parcels CSV")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.step_summaries_json is not None:
        args.step_summaries_json = require_existing_path(
            args.step_summaries_json,
            "Step summaries JSON",
        )
    if args.neighborhood_step_summaries_json is not None:
        args.neighborhood_step_summaries_json = require_existing_path(
            args.neighborhood_step_summaries_json,
            "Neighborhood step summaries JSON",
        )
    if args.lu_step_summaries_json is not None:
        args.lu_step_summaries_json = require_existing_path(
            args.lu_step_summaries_json,
            "Land-use step summaries JSON",
        )
    return args


def choose_area_column(df: pd.DataFrame) -> str:
    """Use land use category (LU) for area summaries."""
    if "LU" not in df.columns:
        raise ValueError("No LU column found in input CSV; cannot summarize by land use category.")
    return "LU"


def choose_neighborhood_column(df: pd.DataFrame) -> str | None:
    """Use neighborhood_name for neighborhood summaries when available."""
    if "neighborhood_name" in df.columns:
        return "neighborhood_name"
    return None


def first_non_empty(values: pd.Series) -> str:
    cleaned = values.astype("string").fillna("").str.strip()
    non_empty = cleaned[cleaned != ""]
    if non_empty.empty:
        return "Unknown"
    return str(non_empty.iloc[0])


def main() -> None:
    args = parse_args()

    print(f"Reading simulated parcels: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)

    if "allocated_units" not in df.columns:
        raise ValueError("Input CSV must include allocated_units.")

    allocated = pd.to_numeric(df["allocated_units"], errors="coerce").fillna(0)
    df = df.copy()
    df["allocated_units"] = allocated

    total_added_units = int(df["allocated_units"].sum())
    parcels_with_added_units = int((df["allocated_units"] > 0).sum())

    area_column = choose_area_column(df)
    neighborhood_column = choose_neighborhood_column(df)
    developed = df[df["allocated_units"] > 0].copy()

    if developed.empty:
        area_summary = pd.DataFrame(
            columns=[
                "area_type",
                "area_name",
                "area_description",
                "parcels_with_added_units",
                "units_added",
                "share_of_added_units",
            ]
        )
    else:
        grouping = developed[area_column].astype("string").fillna("Unknown").replace("", "Unknown")
        description_source = (
            developed["LU_DESC"] if "LU_DESC" in developed.columns else developed[area_column]
        )
        area_summary = (
            developed.assign(_area_name=grouping, _area_description=description_source)
            .groupby("_area_name", as_index=False)
            .agg(
                area_description=("_area_description", first_non_empty),
                parcels_with_added_units=("PID", "count") if "PID" in developed.columns else ("allocated_units", "size"),
                units_added=("allocated_units", "sum"),
            )
            .rename(columns={"_area_name": "area_name"})
        )
        area_summary["units_added"] = area_summary["units_added"].round().astype("int64")
        denom = float(total_added_units) if total_added_units > 0 else 1.0
        area_summary["share_of_added_units"] = (area_summary["units_added"] / denom).round(6)
        area_summary.insert(0, "area_type", area_column)
        area_summary = area_summary.sort_values("units_added", ascending=False).reset_index(drop=True)

    if developed.empty:
        neighborhood_summary = pd.DataFrame(
            columns=[
                "neighborhood_name",
                "parcels_with_added_units",
                "units_added",
                "share_of_added_units",
                "mean_walkability",
                "mean_residential_total_value",
            ]
        )
    else:
        if neighborhood_column is None:
            neighborhood_grouping = pd.Series("Unknown", index=developed.index, dtype="string")
        else:
            neighborhood_grouping = (
                developed[neighborhood_column].astype("string").fillna("Unknown").replace("", "Unknown")
            )
        walkability_numeric = pd.to_numeric(developed.get("neighborhood_walkability"), errors="coerce")
        total_value_numeric = pd.to_numeric(developed.get("TOTAL_VALUE"), errors="coerce")
        residential_mask = (
            developed.get("LU", pd.Series("", index=developed.index))
            .astype("string")
            .fillna("")
            .str.strip()
            .str.upper()
            .isin(RESIDENTIAL_LU_CODES)
        )
        residential_total_value = total_value_numeric.where(residential_mask)
        neighborhood_summary = (
            developed.assign(
                _neighborhood_name=neighborhood_grouping,
                _walkability=walkability_numeric,
                _residential_total_value=residential_total_value,
            )
            .groupby("_neighborhood_name", as_index=False)
            .agg(
                parcels_with_added_units=("PID", "count") if "PID" in developed.columns else ("allocated_units", "size"),
                units_added=("allocated_units", "sum"),
                mean_walkability=("_walkability", "mean"),
                mean_residential_total_value=("_residential_total_value", "mean"),
            )
            .rename(columns={"_neighborhood_name": "neighborhood_name"})
        )
        neighborhood_summary["units_added"] = neighborhood_summary["units_added"].round().astype("int64")
        neighborhood_summary["share_of_added_units"] = (
            neighborhood_summary["units_added"] / float(total_added_units if total_added_units > 0 else 1)
        ).round(6)
        neighborhood_summary["mean_walkability"] = neighborhood_summary["mean_walkability"].round(4)
        neighborhood_summary["mean_residential_total_value"] = (
            neighborhood_summary["mean_residential_total_value"].round(2)
        )
        neighborhood_summary = neighborhood_summary.sort_values("units_added", ascending=False).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "simulation_summary.csv"
    areas_path = args.output_dir / "simulation_units_by_lu.csv"
    neighborhoods_path = args.output_dir / "simulation_units_by_neighborhood.csv"

    if args.step_summaries_json is not None:
        with args.step_summaries_json.open("r", encoding="utf-8") as stream:
            step_summaries = json.load(stream)
        if not isinstance(step_summaries, list):
            raise ValueError("Step summaries JSON must contain a list of objects.")
        pd.DataFrame(step_summaries).to_csv(summary_path, index=False)

    if args.neighborhood_step_summaries_json is not None:
        with args.neighborhood_step_summaries_json.open("r", encoding="utf-8") as stream:
            neighborhood_step_summaries = json.load(stream)
        if not isinstance(neighborhood_step_summaries, list):
            raise ValueError("Neighborhood step summaries JSON must contain a list of objects.")
        neighborhood_summary = pd.DataFrame(neighborhood_step_summaries)
        if not neighborhood_summary.empty and "step" in neighborhood_summary.columns:
            neighborhood_summary = neighborhood_summary.sort_values(
                ["step", "units_added"],
                ascending=[True, False],
            ).reset_index(drop=True)

    if args.lu_step_summaries_json is not None:
        with args.lu_step_summaries_json.open("r", encoding="utf-8") as stream:
            lu_step_summaries = json.load(stream)
        if not isinstance(lu_step_summaries, list):
            raise ValueError("Land-use step summaries JSON must contain a list of objects.")
        area_summary = pd.DataFrame(lu_step_summaries)
        if not area_summary.empty and "step" in area_summary.columns:
            area_summary = area_summary.sort_values(
                ["step", "units_added"],
                ascending=[True, False],
            ).reset_index(drop=True)

    area_summary.to_csv(areas_path, index=False)
    neighborhood_summary.to_csv(neighborhoods_path, index=False)

    if args.step_summaries_json is not None:
        print(f"Wrote simulation summary: {summary_path}")
    if args.neighborhood_step_summaries_json is not None:
        print("Wrote neighborhood summary with timestep rows")
    if args.lu_step_summaries_json is not None:
        print("Wrote land-use summary with timestep rows")
    print(f"Wrote area summary: {areas_path}")
    print(f"Wrote neighborhood summary: {neighborhoods_path}")
    print(f"Total added units: {total_added_units}")
    print(f"Parcels with added units: {parcels_with_added_units}")


if __name__ == "__main__":
    main()
