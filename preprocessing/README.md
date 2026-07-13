# Preprocessing Scripts

This folder contains the parcel preprocessing pipeline.

## What The Pipeline Does

The scripts build and enrich `inputs/parcels.csv` in this order:

1. `clean_parcels.py`
- Reads assessor records and parcel shapes.
- Collapses assessor rows to one row per parcel geometry.
- Writes `inputs/parcels.csv`.

2. `add_zoning.py`
- Reads `inputs/parcels.csv`.
- Spatially joins zoning attributes to parcels.
- Updates `inputs/parcels.csv`.

3. `add_employment_accessibility.py`
- Reads `inputs/parcels.csv`.
- Adds `emp_dist_m` (distance to nearest employment center/CBD).
- Updates `inputs/parcels.csv`.

4. `add_income.py`
- Reads `inputs/parcels.csv`.
- Pulls ACS tract-level median household income (`B19013_001E`) and spatially joins by tract.
- Adds `median_hh_income`.
- Updates `inputs/parcels.csv`.

5. `add_transit_accessibility.py` (optional for now)
- Reads `inputs/parcels.csv`.
- Adds transit network distance column (default: `transit_walk_dist_m`).
- Updates `inputs/parcels.csv`.


Any script can be run standalone to modify column construction:
```
python preprocessing/steps/clean_parcels.py
python preprocessing/steps/add_zoning.py
python preprocessing/steps/add_income.py
python preprocessing/steps/add_employment_accessibility.py
python preprocessing/steps/add_transit_accessibility.py
```

## Run Pipeline Orchestrator

Alternatively, `run_pipeline.py` runs all steps.
```
python preprocessing/run_pipeline.py
```

Notes:
- `run_pipeline.py` currently always runs transit.
- If you want to skip transit, run scripts manually as shown in "Current Workflow (Transit Skipped)".

## Inputs And Outputs

Inputs:
- `preprocessing/raw_data/boston_parcel_assessors.csv`
- `preprocessing/raw_data/boston_parcel_shapes.geojson`
- `preprocessing/raw_data/boston_zoning_subdistricts/Boston_Zoning_Subdistricts.shp`

Primary output:
- `inputs/parcels.csv`