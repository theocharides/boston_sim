# Residential Hedonic Scripts

This folder contains scripts for estimating and comparing residential hedonic price models from parcel-level data.

All scripts read from `inputs/parcels_processed_for_hedonic.csv` by default and subset to residential building types before modeling.

**common/modeling_common.py**: shared feature pools and modeling utilities.

If you do not pass features to `estimate_hedonic.py`, it uses the default feature list from this script.

**train/estimate_hedonic.py**: model training and artifact export.

## Single-purpose workflow

Each step below has one script location:
- Feature selection: `hedonic/workflow/select_model.py`
- Final model estimation: `hedonic/train/estimate_hedonic.py`
- Cross-validation reporting: `hedonic/workflow/run_cross_validation.py`
- Validation diagnostics: `hedonic/validation/validate_model.py`

### 1) Select features
Compare many candidate feature specifications, save performance results, and write the selected feature spec JSON.

```bash
python -m hedonic.workflow.select_model
```
Outputs:
- `hedonic/workflow/residential_hedonic_model_comparison.csv`
- `hedonic/workflow/residential_hedonic_feature_spec.json`

Optional: require `neighborhood_walkability` in every tested combination.
```bash
python -m hedonic.workflow.select_model --require-walkability
```

Recommended for larger feature pools: cap feature count to keep the search tractable.
```bash
python -m hedonic.workflow.select_model --require-walkability --max-features 6
```

Optional: override the hard safety cap on total combinations.
```bash
python -m hedonic.workflow.select_model --max-combinations 100000
```

Optional: only write the comparison table and skip best-spec JSON.
```bash
python -m hedonic.workflow.select_model --skip-selected-spec
```

### 2) Estimate the final model
Estimate the production model using the selected spec file.
```bash
python -m hedonic.train.estimate_hedonic --feature-spec-json hedonic/workflow/residential_hedonic_feature_spec.json
```

Outputs (in `hedonic/artifacts`):
- `residential_hedonic_model.joblib`
- `residential_hedonic_metrics.json`
- `residential_hedonic_coefficients.csv`

### 3) Run cross-validation
Run K-fold CV in one dedicated script. This writes CV metrics only and does not train/export the production model artifact.
```bash
python -m hedonic.workflow.run_cross_validation --feature-spec-json hedonic/workflow/residential_hedonic_feature_spec.json --cv-folds 5
```

Output:
- `hedonic/workflow/residential_hedonic_cv_metrics.json`

### 4) Run validation diagnostics (starting with Moran's I)
Compute Moran's I on log-residuals from the fitted production model to check for
spatial autocorrelation in errors, using KNN row-standardized weights from
`libpysal`.

The validation script also includes a VIF (variance inflation factor) check on the
transformed model design matrix, so you can flag likely multicollinearity among
features as part of the same post-estimation pass.

The validation script writes standard regression diagnostics and plots,
including observed-vs-predicted, residuals-vs-fitted, residual histogram, Q-Q,
scale-location charts, plus summary metrics for RMSE, MAE, R2, residual
distribution shape, fitted/residual correlation, normality, Moran's I, and VIF.

```bash
python -m hedonic.validation.validate_model --input-csv inputs/parcels_processed_for_hedonic.csv --model-path hedonic/artifacts/residential_hedonic_model.joblib --k-neighbors 8 --permutations 199
```

This validation script is meant to assess the final fitted model on the same parcel
input used to train it. It does not compute holdout metrics; it focuses on residual
diagnostics and Moran's I.

Output:
- `hedonic/artifacts/residential_hedonic_validation.json`
- `hedonic/artifacts/residential_hedonic_validation_plots/`

The JSON includes a top-level `vif` section with per-feature values and a summary
of features above 5 and 10, alongside the residual diagnostics and Moran's I
output. The Moran output uses permutation-based inference when permutations are
requested, and the validator still defaults to a small permutation sample when
`--permutations 0` is passed for smoke tests.

If you trained a new model and want the simulation to use it explicitly:
```bash
python run_development_sim.py --hedonic-model-path hedonic/artifacts/residential_hedonic_model.joblib
```
