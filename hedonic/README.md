# Residential Hedonic Scripts

This folder contains scripts for estimating and comparing residential hedonic price models from parcel-level data.

All scripts read from inputs/parcels.csv by default and subset to residential building types before modeling.

Core structural features are currently configured in modeling_common.py, which contains
the target variable and independent variables used across all scripts in this folder.

### 1) estimate_hedonic.py

Purpose:
- Trains a regularized linear residential hedonic model (RidgeCV) on log-transformed TOTAL_VALUE.
- Uses available features from modeling_common.py.
- Filters to residential parcel rows before training.

Default arguments:
- --input-csv inputs/parcels.csv
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
- --input-csv inputs/parcels.csv
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
- inputs/parcels.csv

Output:
- Console report only (no file written).

## Typical Workflow

1. Define a set of test variables in model_comparison.py.
2. Run fast_feature_analysis.py to observe variable correlation.
3. Run compare_models.py to compare alternative feature sets.
4. Save a reasonable model for the simulation by running estimate_hedonic.py.