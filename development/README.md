# Development Allocation

This folder contains development opportunity scoring and housing unit allocation
logic used by the simulation.

## Main Script

- `allocation.py`: scores residential parcel opportunity and allocates
  user-specified unit targets subject to parcel-level capacity.

## What The Script Does

`allocation.py` calculates a residential development opportunity score using
three components:

1. Capacity upside: additional units allowed by zoning.
2. Market strength: neighborhood demand proxies.
3. Acquisition cost intensity: current parcel value per zoned unit.

After scoring, the script allocates new units to parcels in descending
`opportunity_score` order until the step target is met or capacity is
exhausted.

## Required Inputs

Required parcel columns:
- `LU`
- `TOTAL_VALUE`
- `RES_UNITS`
- `zoned_units`
- `median_hh_income`
- `neighborhood_walkability`
- `emp_dist_m`

Configuration input:
- YAML file with `units_to_add`

## Scoring Logic

For residential rows (`LU` in A, CD, CM, R1, R2, R3, R4, RC, RL), the script
computes:

- `additional_units = max(zoned_units - RES_UNITS, 0)`
- `capacity_upside_score = percentile_rank(additional_units)`
- `market_strength_score = mean(percentile_rank(median_hh_income),`
  `percentile_rank(neighborhood_walkability),`
  `percentile_rank(emp_dist_m, ascending=False))`
- `acquisition_cost_intensity = TOTAL_VALUE / max(zoned_units, 1)`
- `acquisition_cost_score = percentile_rank(acquisition_cost_intensity)`

Final weighted score:

- `opportunity_score = w_capacity * capacity_upside_score`
  `+ w_market * market_strength_score`
  `- w_cost * acquisition_cost_score`

## Opportunity Tiers

Parcels are labeled by score percentile among scored residential rows:

- `high`: top 30% (`>= 0.70`)
- `medium`: middle 40% (`0.30` to `< 0.70`)
- `low`: bottom 30% (`< 0.30`)

## Allocation Rules

- Parcels are sorted by `opportunity_score` (descending).
- Only eligible parcels can receive units:
  residential, scored, and `additional_units > 0`.
- Each parcel is capped at its `additional_units`.
- Allocation stops when `units_to_add` is fully assigned or no eligible
  capacity remains.

Outputs include:
- `allocated_units`
- `selected_for_development`
- scoring and tier columns for diagnostics.

## CLI Usage

Run directly:

```bash
python development/allocation.py --input-csv parcels_preprocessed.csv --output-csv development_opportunity_scored.csv --config-yaml development_sim.yaml
```

Simulation integration:
- `simulation/steps/step_01_development_allocation.py` runs this script each
  time step and writes a step-level config containing that step's
  `units_to_add` target.
