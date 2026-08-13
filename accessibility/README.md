# Accessibility Scripts

This folder contains walkability and accessibility logic used by the simulation.

## Main Script

- `neighborhood_walkability.py`: recomputes parcel walkability from a walking
  network and destination amenities.

## Walkability Update Method

`neighborhood_walkability.py` computes parcel walkability from an OSM
walking network and a set of daily destination categories.

When the simulation passes `allocated_units` into the script, it creates
synthetic amenities near parcels that received new housing. The added amenities are combined with the OSM amenities, and the final walkability score is recomputed from the expanded destination set.

## CLI Usage

Run directly:

```bash
python accessibility/neighborhood_walkability.py --parcels-csv inputs/parcels_preprocessed.csv --output-csv inputs/parcels_preprocessed_with_baseline_vars.csv
```

Simulation integration:
- `simulation/steps/step_02_walkability_update.py` runs this script each time
  step and passes `--housing-growth-column allocated_units` so new development
  can influence destination supply and walkability.
