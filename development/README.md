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

## CLI Usage

Run directly:

```bash
python development/allocation.py --input-csv parcels_preprocessed.csv --output-csv development_opportunity_scored.csv --config-yaml development_sim.yaml
```

Simulation integration:
- `simulation/steps/step_01_development_allocation.py` runs this script each
  time step and passes that step's `units_to_add` target directly via
  CLI (`--units-to-add`).
