# Accessibility Scripts

This folder contains walkability and accessibility logic used by the simulation.

## Walkability Update

`add_neighborhood_walkability.py` computes parcel walkability from an OSM
walking network and a set of daily destination categories:
- grocery
- food
- education
- park
- transit

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

## Main Script

- `add_neighborhood_walkability.py`: recomputes parcel walkability from the
  walking network and amenities.
