"""Simulation step 01: score development opportunity and allocate units."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from .common import update_res_units_with_allocations, write_simple_yaml


def run(
    repo_root: Path,
    working_csv: Path,
    step_dir: Path,
    units_this_step: int,
    w_capacity: float,
    w_market: float,
    w_cost: float,
) -> int:
    """Run development scoring and allocation, update working CSV, return allocated units."""
    development_script = repo_root / "development" / "allocation.py"
    scored_csv = step_dir / "development_opportunity_scored.csv"
    step_config_path = step_dir / "development_step_config.yaml"
    write_simple_yaml(step_config_path, {"units_to_add": units_this_step})

    cmd = [
        sys.executable,
        str(development_script),
        "--input-csv",
        str(working_csv),
        "--output-csv",
        str(scored_csv),
        "--config-yaml",
        str(step_config_path),
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

    scored = pd.read_csv(scored_csv, low_memory=False)
    scored = update_res_units_with_allocations(scored, allocated_col="allocated_units")
    scored.to_csv(working_csv, index=False)

    allocated_now = int(pd.to_numeric(scored.get("allocated_units", 0), errors="coerce").fillna(0).sum())
    return allocated_now
