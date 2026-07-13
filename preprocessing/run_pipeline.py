"""
Master pipeline orchestrator for parcel data preprocessing.

This script runs all preprocessing steps in sequence:
1. clean_parcels.py - Collapse assessor data to parcel level
2. add_zoning.py - Add zoning attributes via spatial join
3. add_income.py - Add tract median household income
4. add_employment_accessibility.py - Add employment center distances
5. add_transit_accessibility.py - Add transit network distances (final output)

Each step reads the output of the previous step and adds new columns,
producing the final `inputs/parcels.csv` with all attributes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def require_existing_path(path: Path, label: str) -> Path:
    """Return a resolved path or raise a clear error for missing inputs."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


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
        "--output-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels.csv",
        help="Single output CSV updated in-place by each pipeline step.",
    )
    
    args = parser.parse_args()
    args.raw_assessors = require_existing_path(args.raw_assessors, "Raw assessor CSV")
    args.raw_parcel_shapes = require_existing_path(args.raw_parcel_shapes, "Raw parcel shapes")
    args.zoning_shapefile = require_existing_path(args.zoning_shapefile, "Zoning shapefile")
    args.output_csv = args.output_csv.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    
    preprocessing_dir = Path(__file__).resolve().parent
    steps_dir = preprocessing_dir / "steps"
    repo_root = preprocessing_dir.parent
    inputs_dir = repo_root / "inputs"
    final_csv = args.output_csv

    steps_dir = require_existing_path(steps_dir, "Preprocessing steps folder")
    
    inputs_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Step 3: Add tract income
    run_step(
        "add_income.py",
        steps_dir / "add_income.py",
        parcels_csv=final_csv,
        output_csv=final_csv,
    )

    # Step 4: Add employment accessibility
    run_step(
        "add_employment_accessibility.py",
        steps_dir / "add_employment_accessibility.py",
        parcels_csv=final_csv,
        output_csv=final_csv,
    )
    
    # Step 5: Add transit accessibility
    run_step(
        "add_transit_accessibility.py",
        steps_dir / "add_transit_accessibility.py",
        parcels_csv=final_csv,
        output_csv=final_csv,
    )
    
    print("\n" + "="*70)
    print("Pipeline complete!")
    print(f"Final output: {final_csv}")
    print("="*70)


if __name__ == "__main__":
    main()
