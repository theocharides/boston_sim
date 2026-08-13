from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hedonic.train import estimate_hedonic


def _write_real_data_sample(source_csv: Path, dest_csv: Path, rows: int = 5000) -> None:
    required_cols = ["PID", "LU", "TOTAL_VALUE", "LAND_SF", "LIVING_AREA", "INT_COND", "emp_dist_m"]
    df = pd.read_csv(source_csv, usecols=required_cols, low_memory=False)
    if len(df) > rows:
        df = df.sample(n=rows, random_state=42)
    df.to_csv(dest_csv, index=False)

def test_estimate_hedonic_writes_cv_metrics(tmp_path, monkeypatch) -> None:
    print("[hedonic-cv-test] Starting cross-validation artifact regression test")
    # Keep BLAS thread counts low to avoid known Windows native crashes in this env.
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    source_csv = REPO_ROOT / "outputs" / "parcels_preprocessed.csv"
    if not source_csv.exists():
        pytest.skip(f"Preprocessed parcel CSV not found: {source_csv}")

    use_full_data = os.getenv("HEDONIC_CV_USE_FULL_DATA", "1") == "1"
    if use_full_data:
        input_csv = source_csv
        print(f"[hedonic-cv-test] Using full real parcel input: {input_csv}")
    else:
        input_csv = tmp_path / "parcels_preprocessed_sample.csv"
        print(f"[hedonic-cv-test] Building sample from real parcel input: {source_csv}")
        _write_real_data_sample(source_csv=source_csv, dest_csv=input_csv)
        print(f"[hedonic-cv-test] Sample CSV written at: {input_csv}")

    output_dir = tmp_path / "hedonic_artifacts"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "estimate_hedonic",
            "--input-csv",
            str(input_csv),
            "--output-dir",
            str(output_dir),
            "--feature-list",
            "LIVING_AREA,LAND_SF,emp_dist_m",
            "--cv-folds",
            "3",
            "--test-size",
            "0.2",
            "--random-seed",
            "42",
        ],
    )

    print("[hedonic-cv-test] Running estimate_hedonic.main() with --cv-folds=3")
    estimate_hedonic.main()

    metrics_path = output_dir / "residential_hedonic_metrics.json"
    cv_metrics_path = output_dir / "residential_hedonic_cv_metrics.json"
    print(f"[hedonic-cv-test] Checking output files in: {output_dir}")

    assert metrics_path.exists(), "Expected residential_hedonic_metrics.json to be written."
    assert cv_metrics_path.exists(), "Expected residential_hedonic_cv_metrics.json to be written."

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cv_metrics = json.loads(cv_metrics_path.read_text(encoding="utf-8"))

    print(
        "[hedonic-cv-test] CV summary: "
        f"folds={cv_metrics.get('cv_folds')}, "
        f"r2_log_mean={cv_metrics.get('summary', {}).get('r2_log_mean')}"
    )

    assert "cv" in metrics
    assert metrics["cv"]["cv_folds"] == 3
    assert isinstance(metrics["cv"]["fold_metrics"], list)
    assert len(metrics["cv"]["fold_metrics"]) == 3

    assert cv_metrics["cv_folds"] == 3
    assert isinstance(cv_metrics["summary"], dict)
    assert "r2_log_mean" in cv_metrics["summary"]
    assert "rmse_log_mean" in cv_metrics["summary"]
    print("[hedonic-cv-test] Completed successfully")
