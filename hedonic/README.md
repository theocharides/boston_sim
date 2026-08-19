# Residential Hedonic Scripts

This folder contains the active residential hedonic modeling workflow.

All scripts read from `inputs/parcels_processed_for_hedonic.csv` by default and subset to residential rows before modeling.

## Active Pipeline

Run the full pipeline in one command:

```bash
python -m hedonic.workflow.run_lasso_ols_pipeline
```

This script implements:

1. Step A: benchmark OLS on all candidate variables with out-of-sample metrics.
2. Step B: K-fold cross-validated LASSO (`LassoCV`) to select surviving variables.
3. Step C: final OLS inference on selected variables with coefficients, standard errors, and p-values.

Data splitting rule enforced:

- The script first splits data into a Discovery set and an Evaluation set.
- Feature selection (LASSO) and final OLS inference run on Discovery only.
- Reported model performance is evaluated on the untouched Evaluation set.

Outputs (in `hedonic/artifacts`):

- `residential_hedonic_lasso_ols_pipeline.json`
- `residential_hedonic_lasso_selected_features.json`
- `residential_hedonic_final_ols_coefficients.csv`

Useful options:

```bash
python -m hedonic.workflow.run_lasso_ols_pipeline --cv-folds 5 --test-size 0.2 --random-seed 42
```

Tighten categorical selection by requiring multiple dummy coefficients above threshold:

```bash
python -m hedonic.workflow.run_lasso_ols_pipeline --coef-threshold 0.01 --min-categorical-dummies 2
```

Fast smoke test:

```bash
python -m hedonic.workflow.run_lasso_ols_pipeline --sample-size 5000 --cv-folds 3
```

Run with an explicit feature list:

```bash
python -m hedonic.workflow.run_lasso_ols_pipeline --feature-list LAND_SF,INT_COND,LU,neighborhood_walkability,emp_dist_m
```

Run with a feature spec JSON:

```bash
python -m hedonic.workflow.run_lasso_ols_pipeline --feature-spec-json path/to/feature_spec.json
```
