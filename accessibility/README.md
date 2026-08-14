# Accessibility Scripts

This folder contains walkability and accessibility logic used by the simulation.

## Main Script

- `neighborhood_walkability.py`: recomputes parcel walkability from a walking
  network and destination amenities.

## Walkability Update Method

`neighborhood_walkability.py` computes parcel walkability from an OSM
walking network and a set of daily destination categories.

For each destination category, the script converts network distance to a
0-100 score using a power-decay curve controlled by
`--distance-decay-exponent`. The default exponent is `1.0`, which makes the
distance decay linear: scores start at 100 at 0 m and decline to 0 at the
`--max-walk-distance-m` threshold, which defaults to 1,600 m.

When the simulation passes `allocated_units` into the script, it creates
synthetic amenities near parcels that received new housing. The added amenities are combined with the OSM amenities, and the final walkability score is recomputed from the expanded destination set.

## CLI Usage

Run directly:

```bash
python accessibility/neighborhood_walkability.py --parcels-csv inputs/parcels_preprocessed.csv --output-csv inputs/parcels_preprocessed.csv
```

Simulation integration:
- `simulation/steps/step_02_walkability_update.py` runs this script each time
  step and passes `--housing-growth-column allocated_units` so new development
  can influence destination supply and walkability.
