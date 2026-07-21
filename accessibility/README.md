# Accessibility Scripts

This folder contains walkability and accessibility logic used by the simulation.

## Main Script

- `neighborhood_walkability.py`: recomputes parcel walkability from a walking
  network and destination amenities.

## Walkability Update Method

`neighborhood_walkability.py` computes parcel walkability from an OSM
walking network and a set of daily destination categories.

Categories used in scoring:
- grocery
- food
- education
- park
- transit

The script workflow is:
1. Read parcel geometry and convert parcels to representative points.
2. Download a local walking network for the parcel extent.
3. Download destination amenities from OpenStreetMap.
4. Snap parcels and destinations to the network.
5. Compute shortest-path network distance to nearest destination by category.
6. Convert distances to category scores and average to one parcel score.

Scoring notes:
- Category scores decay linearly from 100 at 0 m to 0 at `--max-walk-distance-m`.
- Final walkability is the mean of available category scores.
- Parcels with no reachable destinations for sampled categories remain blank.

## How New Housing Affects Walkability

When the simulation passes `allocated_units` into the script, it creates
synthetic amenities near parcels that received new housing.

How the amenities are chosen:
- Parcels with more `allocated_units` are more likely to be selected.
- Synthetic amenities are jittered a short distance around the selected growth
  parcels so they appear nearby rather than on the exact parcel centroid.
- The selection uses a fixed random seed so the pattern is reproducible.

How many amenities are added:
- food: about 1 amenity per 120 new units
- grocery: about 1 amenity per 300 new units
- park: about 1 amenity per 450 new units
- transit: about 1 amenity per 700 new units
- education: about 1 amenity per 1,200 new units

The added amenities are combined with the OSM amenities, and the final
walkability score is recomputed from the expanded destination set.

## CLI Usage

Run directly:

```bash
python accessibility/neighborhood_walkability.py --parcels-csv parcels_preprocessed.csv --output-csv parcels_preprocessed.csv
```

Simulation integration:
- `simulation/steps/step_02_walkability_update.py` runs this script each time
  step and passes `--housing-growth-column allocated_units` so new development
  can influence destination supply and walkability.
