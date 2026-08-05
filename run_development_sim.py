"""
Run a multi-step residential development simulation.

Workflow per time step:
0) Run capacity and walkability steps to create baseline parcel values.
1) Score development opportunity and allocate units under parcel capacity.
2) Apply allocated units to parcel RES_UNITS.
3) Recompute neighborhood walkability from the updated parcel table, adding
    synthetic amenities near newly developed parcels.
4) Update residential TOTAL_VALUE predictions using a fixed pre-trained hedonic model.

Configuration is read from development_sim.yaml:
- units_to_add: total units to allocate across all time steps.
- time_steps: number of simulation steps to run.
- w_capacity: weight on capacity score in development opportunity.
- w_market: weight on market score in development opportunity.
- w_cost: weight on acquisition cost score in development opportunity.
- max_walk_distance_m: walkability max distance passed to accessibility step.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hedonic.common.modeling_common import TARGET_COL, subset_residential_rows
from shared_utils import load_simple_yaml, require_existing_path
from simulation.steps.common import allocate_units_by_step
from simulation.steps.step_01_development_allocation import run as run_development_allocation_step
from simulation.steps.step_02_walkability_update import run as run_walkability_update_step
from simulation.steps.step_03_hedonic_update import run as run_hedonic_update_step


RESIDENTIAL_LU_CODES = {"A", "CD", "CM", "R1", "R2", "R3", "R4", "RC", "RL"}


def ensure_required_input_columns(
    repo_root: Path,
    working_csv: Path,
    max_walk_distance_m: float,
    distance_decay_exponent: float,
    synthetic_units_per_feature: dict[str, float],
) -> None:
    """Ensure allocation prerequisites exist, auto-building selected missing columns."""
    probe = pd.read_csv(working_csv, nrows=1, low_memory=False)
    available = set(probe.columns)

    if "zoned_units" not in available:
        capacity_script = repo_root / "residential_capacity" / "calculate_unit_capacity.py"
        capacity_cmd = [
            sys.executable,
            str(capacity_script),
            "--input-csv",
            str(working_csv),
        ]
        print(f"\n{'=' * 72}")
        print("Preflight: add zoned_units")
        print("Command:", " ".join(capacity_cmd))
        print(f"{'=' * 72}")
        subprocess.run(capacity_cmd, check=True)

    probe = pd.read_csv(working_csv, nrows=1, low_memory=False)
    available = set(probe.columns)
    if "neighborhood_walkability" not in available:
        walkability_script = repo_root / "accessibility" / "neighborhood_walkability.py"
        walkability_cmd = [
            sys.executable,
            str(walkability_script),
            "--parcels-csv",
            str(working_csv),
            "--output-csv",
            str(working_csv),
            "--max-walk-distance-m",
            str(max_walk_distance_m),
            "--distance-decay-exponent",
            str(distance_decay_exponent),
        ]
        for category, units_per_feature in synthetic_units_per_feature.items():
            walkability_cmd.extend(["--synthetic-units-per-feature", f"{category}={units_per_feature}"])
        print(f"\n{'=' * 72}")
        print("Preflight: add neighborhood_walkability")
        print("Command:", " ".join(walkability_cmd))
        print(f"{'=' * 72}")
        subprocess.run(walkability_cmd, check=True)

    required_for_allocation = [
        "LU",
        "TOTAL_VALUE",
        "RES_UNITS",
        "zoned_units",
        "median_hh_income",
        "neighborhood_walkability",
        "emp_dist_m",
    ]
    probe = pd.read_csv(working_csv, nrows=1, low_memory=False)
    missing = [column for column in required_for_allocation if column not in probe.columns]
    if missing:
        raise ValueError(
            "Input parcel table is missing required columns for development allocation: "
            f"{missing}. Run preprocessing/run_data_prep.py to rebuild parcel inputs."
        )

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Run multi-step Boston residential development simulation."
    )
    parser.add_argument(
        "--parcels-csv",
        type=Path,
        default=repo_root / "outputs/parcels_preprocessed.csv",
        help="Preprocessed parcel CSV to simulate.",
    )
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=repo_root / "development_sim.yaml",
        help="Simulation config YAML.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "outputs" / "parcels_simulated.csv",
        help="Final simulated parcel CSV output.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=repo_root / "outputs",
        help="Directory for working/intermediate simulation files.",
    )
    parser.add_argument(
        "--hedonic-model-path",
        type=Path,
        default=repo_root / "hedonic" / "artifacts" / "residential_hedonic_model.joblib",
        help="Path to pre-trained hedonic model used for price updates.",
    )
    args = parser.parse_args()
    args.parcels_csv = require_existing_path(args.parcels_csv, "Parcels CSV")
    args.config_yaml = require_existing_path(args.config_yaml, "Simulation config")
    args.hedonic_model_path = args.hedonic_model_path.expanduser().resolve()
    args.output_csv = args.output_csv.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    config = load_simple_yaml(args.config_yaml)
    repo_root = Path(__file__).resolve().parent

    if "units_to_add" not in config:
        raise ValueError("Config must include units_to_add.")

    total_units = int(config["units_to_add"])
    time_steps = int(config.get("time_steps", 2))
    w_capacity = float(config.get("w_capacity", 1.0))
    w_market = float(config.get("w_market", 1.0))
    w_cost = float(config.get("w_cost", 1.0))
    max_walk_distance_m = float(config.get("max_walk_distance_m", 1600.0))
    distance_decay_exponent = float(config.get("distance_decay_exponent", 1.0))
    synthetic_units_per_feature = {
        "food": float(config.get("synthetic_units_per_food", 120.0)),
        "grocery": float(config.get("synthetic_units_per_grocery", 300.0)),
        "park": float(config.get("synthetic_units_per_park", 450.0)),
        "transit": float(config.get("synthetic_units_per_transit", 700.0)),
        "education": float(config.get("synthetic_units_per_education", 1200.0)),
    }

    if total_units < 0:
        raise ValueError("units_to_add must be >= 0")
    if time_steps < 1:
        raise ValueError("time_steps must be >= 1")
    if max_walk_distance_m <= 0:
        raise ValueError("max_walk_distance_m must be > 0")
    if distance_decay_exponent <= 0:
        raise ValueError("distance_decay_exponent must be > 0")
    invalid_thresholds = [
        category for category, units_per_feature in synthetic_units_per_feature.items() if units_per_feature <= 0
    ]
    if invalid_thresholds:
        raise ValueError(
            "Synthetic amenity thresholds must be > 0 for all categories. Invalid: "
            f"{invalid_thresholds}"
        )

    units_per_step = allocate_units_by_step(total_units, time_steps)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    working_csv = args.run_dir / "parcels_working.csv"

    base = pd.read_csv(args.parcels_csv, low_memory=False)
    base.to_csv(working_csv, index=False)

    ensure_required_input_columns(
        repo_root=repo_root,
        working_csv=working_csv,
        max_walk_distance_m=max_walk_distance_m,
        distance_decay_exponent=distance_decay_exponent,
        synthetic_units_per_feature=synthetic_units_per_feature,
    )

    model_path = require_existing_path(args.hedonic_model_path, "Hedonic model")
    fixed_hedonic_model = joblib.load(model_path)
    print(f"Using fixed hedonic model: {model_path}")

    summaries: list[dict[str, object]] = []
    neighborhood_step_summaries: list[dict[str, object]] = []
    lu_step_summaries: list[dict[str, object]] = []
    cumulative_allocated = 0

    for step, units_this_step in enumerate(units_per_step, start=1):
        allocated_now = run_development_allocation_step(
            repo_root=repo_root,
            working_csv=working_csv,
            units_this_step=units_this_step,
            w_capacity=w_capacity,
            w_market=w_market,
            w_cost=w_cost,
        )
        cumulative_allocated += allocated_now

        run_walkability_update_step(
            repo_root=repo_root,
            working_csv=working_csv,
            max_walk_distance_m=max_walk_distance_m,
            distance_decay_exponent=distance_decay_exponent,
            synthetic_units_per_feature=synthetic_units_per_feature,
        )

        current = run_hedonic_update_step(working_csv=working_csv, model=fixed_hedonic_model)

        residential_rows = subset_residential_rows(current, strict=True)
        mean_walkability = float(
            pd.to_numeric(residential_rows.get("neighborhood_walkability"), errors="coerce").mean()
        )
        residential_values = pd.to_numeric(residential_rows[TARGET_COL], errors="coerce")
        mean_residential_value = float(residential_values.mean())

        summaries.append(
            {
                "step": step,
                "units_requested": units_this_step,
                "units_allocated": allocated_now,
                "units_allocated_cumulative": cumulative_allocated,
                "mean_walkability": round(mean_walkability, 4) if not np.isnan(mean_walkability) else None,
                "mean_residential_total_value": (
                    round(mean_residential_value, 2) if not np.isnan(mean_residential_value) else None
                ),
            }
        )

        neighborhood_column = "neighborhood_name" if "neighborhood_name" in current.columns else None
        step_rows = current.copy()
        if neighborhood_column is None:
            step_rows["_neighborhood_name"] = "Unknown"
        else:
            step_rows["_neighborhood_name"] = (
                step_rows[neighborhood_column].astype("string").fillna("Unknown").replace("", "Unknown")
            )

        step_rows["_allocated_units"] = pd.to_numeric(step_rows.get("allocated_units"), errors="coerce").fillna(0)
        step_rows["_walkability"] = pd.to_numeric(step_rows.get("neighborhood_walkability"), errors="coerce")
        step_rows["_total_value"] = pd.to_numeric(step_rows.get(TARGET_COL), errors="coerce")
        lu_series = step_rows["LU"] if "LU" in step_rows.columns else pd.Series("", index=step_rows.index)
        step_rows["_is_residential"] = (
            lu_series.astype("string").fillna("").str.strip().str.upper().isin(RESIDENTIAL_LU_CODES)
        )
        step_rows["_residential_total_value"] = step_rows["_total_value"].where(step_rows["_is_residential"])

        step_developed = step_rows[step_rows["_allocated_units"] > 0].copy()
        if step_developed.empty:
            neighborhood_step_summaries.append(
                {
                    "step": step,
                    "neighborhood_name": "Unknown",
                    "parcels_with_added_units": 0,
                    "units_added": 0,
                    "share_of_added_units": 0.0,
                    "mean_walkability": None,
                    "mean_residential_total_value": None,
                }
            )
            lu_step_summaries.append(
                {
                    "step": step,
                    "area_type": "LU",
                    "area_name": "Unknown",
                    "area_description": "Unknown",
                    "parcels_with_added_units": 0,
                    "units_added": 0,
                    "share_of_added_units": 0.0,
                }
            )
        else:
            total_step_added_units = float(step_developed["_allocated_units"].sum())
            by_neighborhood = (
                step_developed.groupby("_neighborhood_name", as_index=False)
                .agg(
                    parcels_with_added_units=("PID", "count") if "PID" in step_developed.columns else ("_allocated_units", "size"),
                    units_added=("_allocated_units", "sum"),
                    mean_walkability=("_walkability", "mean"),
                    mean_residential_total_value=("_residential_total_value", "mean"),
                )
                .rename(columns={"_neighborhood_name": "neighborhood_name"})
            )
            by_neighborhood["units_added"] = by_neighborhood["units_added"].round().astype("int64")
            denominator = total_step_added_units if total_step_added_units > 0 else 1.0
            by_neighborhood["share_of_added_units"] = (by_neighborhood["units_added"] / denominator).round(6)
            by_neighborhood["mean_walkability"] = by_neighborhood["mean_walkability"].round(4)
            by_neighborhood["mean_residential_total_value"] = (
                by_neighborhood["mean_residential_total_value"].round(2)
            )
            by_neighborhood = by_neighborhood.sort_values("units_added", ascending=False).reset_index(drop=True)

            for row in by_neighborhood.to_dict(orient="records"):
                neighborhood_step_summaries.append({"step": step, **row})

            area_grouping = step_developed["LU"].astype("string").fillna("Unknown").replace("", "Unknown")
            description_source = step_developed["LU_DESC"] if "LU_DESC" in step_developed.columns else area_grouping
            by_area = (
                step_developed.assign(_area_name=area_grouping, _area_description=description_source)
                .groupby("_area_name", as_index=False)
                .agg(
                    area_description=("_area_description", lambda values: str(values.astype("string").fillna("").str.strip()[values.astype("string").fillna("").str.strip() != ""].iloc[0]) if (values.astype("string").fillna("").str.strip() != "").any() else "Unknown"),
                    parcels_with_added_units=("PID", "count") if "PID" in step_developed.columns else ("_allocated_units", "size"),
                    units_added=("_allocated_units", "sum"),
                )
                .rename(columns={"_area_name": "area_name"})
            )
            by_area["units_added"] = by_area["units_added"].round().astype("int64")
            by_area["share_of_added_units"] = (by_area["units_added"] / denominator).round(6)
            by_area.insert(0, "area_type", "LU")
            by_area = by_area.sort_values("units_added", ascending=False).reset_index(drop=True)

            for row in by_area.to_dict(orient="records"):
                lu_step_summaries.append({"step": step, **row})

        print(
            f"Step {step} complete | requested={units_this_step} allocated={allocated_now} "
            f"cumulative={cumulative_allocated}"
        )

    final_df = pd.read_csv(working_csv, low_memory=False)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output_csv, index=False)

    step_summaries_path = args.run_dir / "step_summaries.json"
    with step_summaries_path.open("w", encoding="utf-8") as stream:
        json.dump(summaries, stream, ensure_ascii=True, indent=2)
    neighborhood_step_summaries_path = args.run_dir / "neighborhood_step_summaries.json"
    with neighborhood_step_summaries_path.open("w", encoding="utf-8") as stream:
        json.dump(neighborhood_step_summaries, stream, ensure_ascii=True, indent=2)
    lu_step_summaries_path = args.run_dir / "lu_step_summaries.json"
    with lu_step_summaries_path.open("w", encoding="utf-8") as stream:
        json.dump(lu_step_summaries, stream, ensure_ascii=True, indent=2)

    postprocess_script = repo_root / "simulation" / "postprocess_simulation_outputs.py"
    postprocess_cmd = [
        sys.executable,
        str(postprocess_script),
        "--input-csv",
        str(args.output_csv),
        "--output-dir",
        str(args.run_dir),
        "--step-summaries-json",
        str(step_summaries_path),
        "--neighborhood-step-summaries-json",
        str(neighborhood_step_summaries_path),
        "--lu-step-summaries-json",
        str(lu_step_summaries_path),
    ]
    print(f"\n{'=' * 72}")
    print("Post-process: summarize added units")
    print("Command:", " ".join(postprocess_cmd))
    print(f"{'=' * 72}")
    subprocess.run(postprocess_cmd, check=True)
    step_summaries_path.unlink(missing_ok=True)
    neighborhood_step_summaries_path.unlink(missing_ok=True)
    lu_step_summaries_path.unlink(missing_ok=True)

    print("\nSimulation complete")
    print(f"Input parcels: {args.parcels_csv}")
    print(f"Final output: {args.output_csv}")
    print(f"Summary: {args.run_dir / 'simulation_summary.csv'}")
    print(f"Total units target: {total_units}")
    print(f"Total units allocated: {cumulative_allocated}")


if __name__ == "__main__":
    main()
