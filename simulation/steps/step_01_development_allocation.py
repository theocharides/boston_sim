"""Simulation step 01: score development opportunity and allocate units."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from .common import update_res_units_with_allocations


def run(
    repo_root: Path,
    working_csv: Path,
    units_this_step: int,
    w_capacity: float,
    w_market: float,
    w_cost: float,
) -> int:
    """Run development scoring and allocation, update working CSV, return allocated units."""
    development_script = repo_root / "development" / "allocation.py"

    cmd = [
        sys.executable,
        str(development_script),
        "--input-csv",
        str(working_csv),
        "--output-csv",
        str(working_csv),
        "--units-to-add",
        str(units_this_step),
        "--w-capacity",
        str(w_capacity),
        "--w-market",
        str(w_market),
        "--w-cost",
        str(w_cost),
    ]

    print(f"\n{'=' * 72}")
    print("Step 1: development opportunity and allocation")
    print("Command:", " ".join(cmd))
    print(f"{'=' * 72}")
    subprocess.run(cmd, check=True)

    scored = pd.read_csv(working_csv, low_memory=False)
    scored = update_res_units_with_allocations(scored, allocated_col="allocated_units")
    scored.to_csv(working_csv, index=False)

    allocated_now = int(pd.to_numeric(scored.get("allocated_units", 0), errors="coerce").fillna(0).sum())
    return allocated_now
