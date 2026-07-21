"""Simulation step 02: recompute neighborhood walkability."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(repo_root: Path, working_csv: Path, max_walk_distance_m: float) -> None:
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
        "--housing-growth-column",
        "allocated_units",
    ]

    print(f"\n{'=' * 72}")
    print("Step 2: recompute neighborhood walkability")
    print("Command:", " ".join(cmd))
    print(f"{'=' * 72}")
    subprocess.run(cmd, check=True)
