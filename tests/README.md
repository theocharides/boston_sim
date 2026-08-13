# Tests

This folder contains two test modules.

## 1. Preprocessed parcel validation

File: `tests/test_preprocessed_parcels.py`

Purpose:
Validate that the fully preprocessed parcel table in `inputs/parcels_preprocessed.csv` has the expected schema, identifiers, land-use values, value columns, locational enrichment columns, and geometry formatting.

Data source used by the tests:
- `inputs/parcels_preprocessed.csv`

Test groups and checks:

### Schema
- `test_required_columns_present`
  Checks that core output columns required by downstream workflows exist.
- `test_optional_columns_present`
  Checks that enrichment columns expected from the full pipeline exist.
- `test_no_duplicate_column_names`
  Ensures the CSV does not contain duplicate headers.

### Rows and identifiers
- `test_has_rows`
  Confirms the preprocessed table is not empty.
- `test_pid_is_unique`
  Confirms one row per parcel by requiring unique `PID` values.
- `test_pid_not_null`
  Confirms no parcel IDs are missing.
- `test_pid_is_numeric_string_10_digits`
  Confirms parcel IDs are numeric and within the expected 10-digit format.

### Land use
- `test_lu_not_all_null`
  Ensures `LU` is populated for most rows.
- `test_lu_values_are_known_codes`
  Confirms land-use codes are in the project data dictionary.
- `test_residential_lu_rows_exist`
  Confirms the output includes residential parcel types.

### Value columns
- `test_total_value_not_all_null`
  Ensures `TOTAL_VALUE` is populated for most rows.
- `test_total_value_non_negative`
  Confirms no negative total assessed values.
- `test_land_value_non_negative`
  Confirms no negative land assessed values.
- `test_bldg_value_non_negative`
  Confirms no negative building assessed values.
- `test_total_value_matches_sum_of_components`
  Checks that `TOTAL_VALUE` is broadly consistent with `LAND_VALUE + BLDG_VALUE`, with tolerance for known assessor aggregation quirks.
- `test_land_sf_positive_where_present`
  Confirms `LAND_SF` is positive when populated.
- `test_living_area_non_negative_where_present`
  Confirms `LIVING_AREA` is non-negative when populated.
- `test_yr_built_plausible`
  Checks that `YR_BUILT` values are mostly within a plausible range and tolerates only a small number of obvious data-entry errors.

### Locational enrichment
- `test_emp_dist_m_present_and_positive`
  Confirms `emp_dist_m` exists for most rows and is positive when populated.
- `test_neighborhood_name_mostly_present`
  Confirms `neighborhood_name` was added for most rows.
- `test_median_hh_income_mostly_present`
  Confirms `median_hh_income` exists for most rows, excluding the known missing-value sentinel.
- `test_median_hh_income_no_unexpected_negatives`
  Confirms income values do not contain invalid negatives beyond the known sentinel value.
- `test_zoning_use_mostly_present`
  Confirms `zoning_use` was added for most rows.

### Geometry
- `test_geometry_mostly_present`
  Confirms geometry exists for nearly all rows.
- `test_geometry_looks_like_wkt`
  Confirms non-null geometry strings look like WKT.

## 2. Simulation postprocessing regression

File: `tests/test_postprocess_simulation_outputs.py`

Purpose:
Validate that simulation postprocessing preserves time-step rows when land-use step summaries are supplied.

Test:
- `test_postprocess_writes_lu_step_rows`
  Creates a temporary parcel input and temporary per-step LU summary JSON, runs `simulation/postprocess_simulation_outputs.py`, and verifies that:
  - `simulation_units_by_lu.csv` is written
  - the output contains a `step` column
  - rows are preserved in step order
  - expected `area_name` values are present

## Running tests

Run all tests:

```bash
python -m pytest tests/ -v
```

Run only parcel validation tests:

```bash
python -m pytest tests/test_preprocessed_parcels.py -v
```

Run only the postprocessing regression test:

```bash
python -m pytest tests/test_postprocess_simulation_outputs.py -v
```

## Current count

- `test_preprocessed_parcels.py`: 33 tests
- `test_postprocess_simulation_outputs.py`: 1 test
- Total: 34 tests
