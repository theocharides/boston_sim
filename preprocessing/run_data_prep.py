"""
Master pipeline orchestrator for parcel data preprocessing.

This script runs all preprocessing steps in sequence:
1. clean_parcels.py - Collapse assessor data to parcel level
2. add_zoning.py - Add zoning attributes via spatial join
3. add_neighborhood.py - Add Boston neighborhood tag via spatial join
4. add_income.py - Add tract median household income
5. add_employment_dist.py - Add employment center distances
6. neighborhood_walkability.py - Add the baseline neighborhood walkability score

Each step reads the output of the previous step and adds new columns,
producing one canonical preprocessed table with baseline variables included.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared_utils import require_existing_path


def run_step(script_name: str, script_path: Path, **kwargs) -> None:
    """Run a preprocessing script with the given keyword arguments."""
    cmd = [sys.executable, str(script_path)]
    
    for key, value in kwargs.items():
        flag = f"--{key.replace('_', '-')}"
        cmd.extend([flag, str(value)])
    
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Script {script_name} failed with return code {result.returncode}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    
    parser = argparse.ArgumentParser(
        description="Run full parcel preprocessing pipeline end-to-end."
    )
    parser.add_argument(
        "--raw-assessors",
        type=Path,
        default=repo_root / "preprocessing" / "raw_data" / "boston_parcel_assessors.csv",
        help="Path to raw assessor CSV.",
    )
    parser.add_argument(
        "--raw-parcel-shapes",
        type=Path,
        default=repo_root / "preprocessing" / "raw_data" / "boston_parcel_shapes.geojson",
        help="Path to parcel polygon file.",
    )
    parser.add_argument(
        "--zoning-shapefile",
        type=Path,
        default=repo_root / "preprocessing" / "raw_data" / "boston_zoning_subdistricts" / "Boston_Zoning_Subdistricts.shp",
        help="Path to zoning subdistrict shapefile.",
    )
    parser.add_argument(
        "--neighborhood-geojson",
        type=Path,
        default=repo_root / "preprocessing" / "raw_data" / "boston_neighborhood_boundaries.geojson",
        help="Path to Boston neighborhood boundary GeoJSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels_preprocessed.csv",
        help="Canonical parcel table with preprocessing and baseline-context variables included.",
    )
    parser.add_argument(
        "--skip-income",
        action="store_true",
        help="Skip ACS income enrichment step (useful when Census API key is unavailable).",
    )
    parser.add_argument(
        "--census-api-key",
        type=str,
        default=None,
        help="Optional Census API key passed to add_income.py. If omitted, CENSUS_API_KEY is used.",
    )
    parser.add_argument(
        "--max-walk-distance-m",
        type=float,
        default=1600.0,
        help="Maximum walking distance used in the neighborhood_walkability baseline score.",
    )
    parser.add_argument(
        "--distance-decay-exponent",
        type=float,
        default=1.0,
        help="Distance decay exponent used for the walkability score.",
    )
    parser.add_argument(
        "--synthetic-units-per-feature",
        action="append",
        default=None,
        help="Optional category=value overrides used when generating synthetic amenity points for walkability. Repeat per category.",
    )
    
    args = parser.parse_args()
    args.raw_assessors = require_existing_path(args.raw_assessors, "Raw assessor CSV")
    args.raw_parcel_shapes = require_existing_path(args.raw_parcel_shapes, "Raw parcel shapes")
    args.zoning_shapefile = require_existing_path(args.zoning_shapefile, "Zoning shapefile")
    args.neighborhood_geojson = require_existing_path(args.neighborhood_geojson, "Neighborhood boundaries")
    args.output_csv = args.output_csv.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    
    preprocessing_dir = Path(__file__).resolve().parent
    steps_dir = preprocessing_dir / "steps"
    final_csv = args.output_csv
    final_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Clean parcels
    run_step(
        "clean_parcels.py",
        steps_dir / "clean_parcels.py",
        assessors_csv=args.raw_assessors,
        parcel_shapes=args.raw_parcel_shapes,
        output_csv=final_csv,
    )
    
    # Step 2: Add zoning
    run_step(
        "add_zoning.py",
        steps_dir / "add_zoning.py",
        parcels_cleaned=final_csv,
        zoning_shapefile=args.zoning_shapefile,
        output_csv=final_csv,
    )
    
    # Step 3: Add neighborhood tag
    run_step(
        "add_neighborhood.py",
        steps_dir / "add_neighborhood.py",
        parcels_csv=final_csv,
        neighborhood_geojson=args.neighborhood_geojson,
        output_csv=final_csv,
    )

    # Step 4: Add tract income (optional)
    if args.skip_income:
        print("\nSkipping add_income.py (--skip-income set).")
    else:
        income_kwargs = {
            "parcels_csv": final_csv,
            "output_csv": final_csv,
        }
        api_key = args.census_api_key or os.getenv("CENSUS_API_KEY")
        if api_key:
            income_kwargs["census_api_key"] = api_key
        run_step(
            "add_income.py",
            steps_dir / "add_income.py",
            **income_kwargs,
        )

    # Step 5: Add employment distance
    run_step(
        "add_employment_dist.py",
        steps_dir / "add_employment_dist.py",
        parcels_csv=final_csv,
        output_csv=final_csv,
    )

    # Step 6: Add neighborhood walkability baseline score
    walkability_kwargs = {
        "parcels_csv": final_csv,
        "output_csv": final_csv,
        "max_walk_distance_m": args.max_walk_distance_m,
        "distance_decay_exponent": args.distance_decay_exponent,
    }
    if args.synthetic_units_per_feature:
        walkability_kwargs["synthetic_units_per_feature"] = args.synthetic_units_per_feature
    run_step(
        "neighborhood_walkability.py",
        Path(__file__).resolve().parents[1] / "accessibility" / "neighborhood_walkability.py",
        **walkability_kwargs,
    )

    print("\n" + "="*70)
    print("Pipeline complete!")
    print(f"Final output: {final_csv}")
    print("="*70)


if __name__ == "__main__":
    main()
