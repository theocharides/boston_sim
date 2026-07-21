# Residential Hedonic Scripts

This folder contains scripts for estimating and comparing residential hedonic price models from parcel-level data.

All scripts read from `parcels_preprocessed.csv` by default and subset to residential building types before modeling.

Core feature pools and common modeling helpers are defined in
`common/modeling_common.py` and shared by all scripts in this folder.

### 1) train/estimate_hedonic.py

Purpose:
- Trains a regularized linear residential hedonic model (RidgeCV) on log-transformed TOTAL_VALUE.
- Uses available features from common/modeling_common.py.
- Filters to residential parcel rows before training.

Default arguments:
- --input-csv parcels_preprocessed.csv
- --output-dir hedonic/artifacts
- --test-size 0.2
- --random-seed 42

Outputs (in hedonic/artifacts):
- residential_hedonic_model.joblib
- residential_hedonic_metrics.json
- residential_hedonic_coefficients.csv

### 2) compare_models.py

Purpose:
- Tests multiple residential feature specifications on a shared train/test split.
- Writes model performance comparison table.
- Filters to residential parcel rows before evaluation.

Default arguments:
- --input-csv parcels_preprocessed.csv
- --output-csv hedonic/artifacts/residential_hedonic_model_comparison.csv
- --test-size 0.2
- --random-seed 42

Output:
- hedonic/artifacts/residential_hedonic_model_comparison.csv

### 3) fast_feature_analysis.py

Purpose:
- Fast, non-modeling screen using correlations with log(TOTAL_VALUE).
- Prints ranked summary to console only.
- Filters to residential parcel rows before analysis.

Default input:
- parcels_preprocessed.csv

Output:
- Console report only (no file written).

## How This Feeds The Simulation

- `run_development_sim.py` uses the trained model artifact
	`hedonic/artifacts/residential_hedonic_model.joblib`.
- During each simulation step, after development allocation and walkability
	updates, the simulation applies the fixed hedonic model to update
	residential `TOTAL_VALUE`.

## Typical Workflow

1. Define a set of test variables in compare_models.py.
2. Run fast_feature_analysis.py to observe variable correlation.
3. Run compare_models.py to compare alternative feature sets.
4. Save a reasonable model for the simulation by running train/estimate_hedonic.py.

## Example Commands

```bash
python hedonic/fast_feature_analysis.py
python hedonic/compare_models.py --max-specs 512
python hedonic/train/estimate_hedonic.py
```

If you trained a new model and want the simulation to use it explicitly:

```bash
python run_development_sim.py --hedonic-model-path hedonic/artifacts/residential_hedonic_model.joblib
```

## Current Folder Layout

- common/modeling_common.py: shared feature pools and modeling utilities.
- train/estimate_hedonic.py: model training and artifact export.
- compare_models.py: specification comparison.
- fast_feature_analysis.py: quick feature screen.