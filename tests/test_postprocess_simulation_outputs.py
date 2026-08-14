import json
import sys
from pathlib import Path

import pandas as pd

from simulation import postprocess_simulation_outputs as postprocess


def test_postprocess_writes_lu_step_rows(tmp_path, monkeypatch):
    input_path = tmp_path / "parcels.csv"
    pd.DataFrame(
        [
            {"allocated_units": 5, "LU": "A", "LU_DESC": "APT 7-30 UNITS", "PID": 1},
            {"allocated_units": 0, "LU": "R1", "LU_DESC": "SINGLE FAM DWELLING", "PID": 2},
        ]
    ).to_csv(input_path, index=False)

    lu_step_summaries_path = tmp_path / "lu_step_summaries.json"
    lu_step_summaries_path.write_text(
        json.dumps(
            [
                {
                    "step": 1,
                    "area_type": "LU",
                    "area_name": "A",
                    "area_description": "APT 7-30 UNITS",
                    "parcels_with_added_units": 1,
                    "units_added": 5,
                    "share_of_added_units": 1.0,
                },
                {
                    "step": 2,
                    "area_type": "LU",
                    "area_name": "R1",
                    "area_description": "SINGLE FAM DWELLING",
                    "parcels_with_added_units": 1,
                    "units_added": 2,
                    "share_of_added_units": 0.4,
                },
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "postprocess",
            "--input-csv",
            str(input_path),
            "--output-dir",
            str(tmp_path),
            "--lu-step-summaries-json",
            str(lu_step_summaries_path),
        ],
    )

    postprocess.main()

    output = pd.read_csv(tmp_path / "simulation_units_by_lu.csv")
    assert "step" in output.columns
    assert output["step"].tolist() == [1, 2]
    assert output.loc[0, "area_name"] == "A"
    assert output.loc[1, "area_name"] == "R1"


def test_selected_spec_creates_new_file_instead_of_overwriting(tmp_path):
    from hedonic import select_model

    base_path = tmp_path / "residential_hedonic_feature_spec.json"
    base_path.write_text('{"features": ["old"]}', encoding="utf-8")

    next_path = select_model._unique_output_json_path(base_path)

    assert next_path != base_path
    assert next_path.parent == tmp_path
    assert next_path.suffix == ".json"
    assert next_path.name.startswith("residential_hedonic_feature_spec")
