"""Simulation step 03: update parcel values with fixed hedonic model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import predict_and_update_values_with_model


def run(working_csv: Path, model: object) -> pd.DataFrame:
    """Load working parcels, update TOTAL_VALUE via fixed model, and write in-place."""
    current = pd.read_csv(working_csv, low_memory=False)
    current = predict_and_update_values_with_model(parcels_df=current, model=model)
    current.to_csv(working_csv, index=False)
    return current
