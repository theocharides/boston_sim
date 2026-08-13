# Hedonic Tests

This folder contains hedonic-modeling specific regression tests.

## Files

- `test_cross_validation.py`
  - Executes `hedonic.train.estimate_hedonic` with `--cv-folds 3` using real parcel data from `outputs/parcels_preprocessed.csv`.
  - Default mode uses the full dataset (normal train/test and CV row counts).
  - Set `HEDONIC_CV_USE_FULL_DATA=0` to run in sampled mode for quicker checks.
  - Verifies that both artifacts are written:
    - `residential_hedonic_metrics.json`
    - `residential_hedonic_cv_metrics.json`
  - Confirms CV payload fields and fold results exist (`cv_folds == 3`, `len(fold_metrics) == 3`, summary keys like `r2_log_mean` and `rmse_log_mean`).

## Run

From repository root:

```bash
python -m pytest hedonic/tests/test_cross_validation.py -v
```

To display print statements during test execution:

```bash
python -m pytest hedonic/tests/test_cross_validation.py -v -s
```

Run with sampled mode (faster check):

```bash
set HEDONIC_CV_USE_FULL_DATA=0
python -m pytest hedonic/tests/test_cross_validation.py -v -s
```
