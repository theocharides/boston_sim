## Overview

This Boston simulator allocates housing units to Boston parcels where residential unit capacity exists. Residential units are allocated based on market strength and acquisition cost. The simulator recomputes neighborhood accessibility for each parcel based on housing and amenity growth. Parcel prices then update as parcel walkability scores change.

## Preprocessing

The preprocessing pipeline lives in `preprocessing/`
and is orchestrated by `preprocessing/run_data_prep.py`.

It runs these steps in order:
1. `clean_parcels.py` builds one row per parcel from assessor and geometry data.
2. `add_zoning.py` adds zoning fields to the table.
3. `add_neighborhood.py` adds Boston neighborhood tags from neighborhood boundaries.
4. `add_income.py` joins tract-level median household income using the Census API.
5. `add_employment_dist.py` adds straight-line distance to major employment centers in Boston.

The preprocessing output is stored as `inputs/parcels_preprocessed.csv`.
This is the single canonical parcel table used for downstream modeling and simulation.

## Configuration

The main simulation config is `development_sim.yaml`.

- `units_to_add` sets the total number of new units to allocate.
- `time_steps` sets how many simulation rounds to split that allocation across.
- `w_capacity`, `w_market`, `w_cost` set development opportunity score weights.
- `max_walk_distance_m` sets the walkability scoring distance cap used in each step.

## Simulation

`run_development_sim.py` runs the simulation in three steps per time period:
1. Calculate development opportunity and allocate units within parcel capacity.
2. Recompute neighborhood walkability after the new units are placed, with
	synthetic amenities added near newly developed parcels. 
3. Update residential `TOTAL_VALUE` using the pre-trained hedonic model. 

Housing production is allocated over a user-defined number time steps, with walkability raising market strength and higher total value raising acquisition cost intensity.

The final output is the post-simulation parcel table
`outputs/parcels_simulated.csv`.

After simulation completes, a post-processing step writes summary
outputs:
- `outputs/simulation_summary.csv`: summary of
	total units added and number of parcels that received added units.
- `outputs/simulation_units_by_lu.csv`: where units were
	added, summarized by land use category (`LU`).
- `outputs/simulation_units_by_neighborhood.csv`: where units were
	added, summarized by neighborhood.

### Future development
1. Add prototype differentiation to development decisions, incorporating hedonic coefficients.
2. Add an employment location choice model (supercede current amenity assignment). Housing development may be responsive to employment locations per the hedonic coefficient.
