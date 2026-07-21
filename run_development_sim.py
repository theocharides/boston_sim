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


def ensure_required_input_columns(
    repo_root: Path,
    working_csv: Path,
    max_walk_distance_m: float,
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
        ]
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
        default=repo_root / "parcels_preprocessed.csv",
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
        default=repo_root / "simulation_ouputs" / "parcels_simulated.csv",
        help="Final simulated parcel CSV output.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=repo_root / "simulation_ouputs",
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

    if total_units < 0:
        raise ValueError("units_to_add must be >= 0")
    if time_steps < 1:
        raise ValueError("time_steps must be >= 1")
    if max_walk_distance_m <= 0:
        raise ValueError("max_walk_distance_m must be > 0")

    units_per_step = allocate_units_by_step(total_units, time_steps)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    working_csv = args.run_dir / "parcels_working.csv"

    base = pd.read_csv(args.parcels_csv, low_memory=False)
    base.to_csv(working_csv, index=False)

    ensure_required_input_columns(
        repo_root=repo_root,
        working_csv=working_csv,
        max_walk_distance_m=max_walk_distance_m,
    )

    model_path = require_existing_path(args.hedonic_model_path, "Hedonic model")
    fixed_hedonic_model = joblib.load(model_path)
    print(f"Using fixed hedonic model: {model_path}")

    summaries: list[dict[str, object]] = []
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
        )

        current = run_hedonic_update_step(working_csv=working_csv, model=fixed_hedonic_model)

        mean_walkability = float(pd.to_numeric(current.get("neighborhood_walkability"), errors="coerce").mean())
        residential_values = pd.to_numeric(
            subset_residential_rows(current, strict=True)[TARGET_COL], errors="coerce"
        )
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
                "hedonic_mode": "predict-only",
            }
        )

        print(
            f"Step {step} complete | requested={units_this_step} allocated={allocated_now} "
            f"cumulative={cumulative_allocated}"
        )

    final_df = pd.read_csv(working_csv, low_memory=False)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output_csv, index=False)

    summary_path = args.run_dir / "simulation_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    postprocess_script = repo_root / "simulation" / "postprocess_simulation_outputs.py"
    postprocess_cmd = [
        sys.executable,
        str(postprocess_script),
        "--input-csv",
        str(args.output_csv),
        "--output-dir",
        str(args.run_dir),
    ]
    print(f"\n{'=' * 72}")
    print("Post-process: summarize added units")
    print("Command:", " ".join(postprocess_cmd))
    print(f"{'=' * 72}")
    subprocess.run(postprocess_cmd, check=True)

    print("\nSimulation complete")
    print(f"Input parcels: {args.parcels_csv}")
    print(f"Final output: {args.output_csv}")
    print(f"Summary: {summary_path}")
    print(f"Total units target: {total_units}")
    print(f"Total units allocated: {cumulative_allocated}")


if __name__ == "__main__":
    main()
