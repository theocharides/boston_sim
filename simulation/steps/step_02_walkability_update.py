"""Simulation step 02: recompute neighborhood walkability."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(
    repo_root: Path,
    working_csv: Path,
    max_walk_distance_m: float,
    distance_decay_exponent: float,
    synthetic_units_per_feature: dict[str, float],
) -> None:
    """Recompute neighborhood walkability in-place for the working parcel CSV."""
    walkability_script = repo_root / "accessibility" / "neighborhood_walkability.py"

    cmd = [
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
        "--housing-growth-column",
        "allocated_units",
    ]
    for category, units_per_feature in synthetic_units_per_feature.items():
        cmd.extend(["--synthetic-units-per-feature", f"{category}={units_per_feature}"])

    print(f"\n{'=' * 72}")
    print("Step 2: recompute neighborhood walkability")
    print("Command:", " ".join(cmd))
    print(f"{'=' * 72}")
    subprocess.run(cmd, check=True)
