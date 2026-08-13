# Preprocessing Scripts

This folder contains parcel preprocessing scripts that build and enrich the
project parcel table.

## Current Scripts In This Folder

1. `steps/clean_parcels.py`
- Builds parcel-level records from assessor rows and parcel geometries.
- Adds/normalizes core assessor attributes and writes one row per parcel.

2. `steps/add_zoning.py`
- Spatially joins zoning subdistrict attributes to parcel geometries.
- Adds fields such as `zoning_use`, `max_far`, `max_height`, and setbacks.

3. `steps/add_income.py`
- Pulls ACS tract-level median household income (`B19013_001E`).
- Joins tract income to parcel points as `median_hh_income`.
- Requires a Census API key (`CENSUS_API_KEY` environment variable or `--census-api-key`).

4. `steps/add_neighborhood.py`
- Spatially joins Boston neighborhood boundaries to parcel points.
- Adds `neighborhood_name` (and optional neighborhood ID if available).

5. `steps/add_employment_dist.py`
- Computes straight-line distance in meters from each parcel to the nearest
	employment center/CBD.
- Writes `emp_dist_m`.

5. `run_data_prep.py`
- Orchestrator script intended to run preprocessing steps in sequence.

6. `utils.py`
- Shared helpers for path checks, parcel CSV geometry loading/saving, and
	numeric cleanup.

## Typical Manual Run Order

Run these scripts in sequence when building the table manually:

```bash
python preprocessing/steps/clean_parcels.py
python preprocessing/steps/add_zoning.py
python preprocessing/steps/add_neighborhood.py
python preprocessing/steps/add_income.py
python preprocessing/steps/add_employment_dist.py
```

## Run The Orchestrator

```bash
python preprocessing/run_data_prep.py
```

If you do not have a Census API key, run the full pipeline without the income step:

```bash
python preprocessing/run_data_prep.py --skip-income
```

## Inputs And Outputs

Common raw inputs:
- `preprocessing/raw_data/boston_parcel_assessors.csv`
- `preprocessing/raw_data/boston_parcel_shapes.geojson`
- `preprocessing/raw_data/boston_zoning_subdistricts/Boston_Zoning_Subdistricts.shp`
- `preprocessing/raw_data/boston_neighborhood_boundaries.geojson`

Primary working output table:
- `inputs/parcels_preprocessed.csv`