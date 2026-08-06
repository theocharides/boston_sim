# Residential Hedonic Scripts

This folder contains scripts for estimating and comparing residential hedonic price models from parcel-level data.

All scripts read from `parcels_preprocessed.csv` by default and subset to residential building types before modeling.

**common/modeling_common.py**: shared feature pools and modeling utilities.

If you do not pass features to `estimate_hedonic.py`, it uses the default feature list from this script.

**train/estimate_hedonic.py**: model training and artifact export.

### 1) Compare specs
Compare candidate feature specifications.

```bash
python -m hedonic.compare_models 
```
Output:
- `hedonic/artifacts/residential_hedonic_model_comparison.csv`

### Stage 2) Select a spec
Select one winning specification into a saved JSON spec file.

```bash
python -m hedonic.train.select_spec 
```
Outputs:
- `hedonic/artifacts/residential_hedonic_selected_spec.json`
- Selected feature list and snapshot metrics recorded in that JSON.

### 3) Train with selected spec
Train the production model using that selected spec file.
```bash
python -m hedonic.train.estimate_hedonic
```

Outputs (in `hedonic/artifacts`):
- `residential_hedonic_model.joblib`
- `residential_hedonic_metrics.json`
- `residential_hedonic_coefficients.csv`

If you trained a new model and want the simulation to use it explicitly:
```bash
python run_development_sim.py --hedonic-model-path hedonic/artifacts/residential_hedonic_model.joblib
```
