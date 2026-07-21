## Overview

The simulator adds specified housing production targets to Boston parcels where residential unit capacity exists. Residential units are allocated to parcels based on market strength and acquisition cost. The simulator recomputes neighborhood accessibility for each parcel based on housing and ammenity growth, and updates parcel values based on changing walkability scores for a user-defined number of time steps.

## Preprocessing

The preprocessing pipeline lives in [preprocessing/](preprocessing/README.md)
and is orchestrated by `preprocessing/run_data_prep.py`.

It runs these steps in order:
1. `clean_parcels.py` builds one row per parcel from assessor and geometry data.
2. `add_zoning.py` adds zoning fields to the table.
3. `add_income.py` joins tract-level median household income using the Census API.
4. `add_employment_dist.py` adds straight-line distance to major employment centers in Boston.

The preprocessing output is the canonical parcel table
`parcels_preprocessed.csv`.

## Configuration

The main simulation config is `development_sim.yaml`.

- `units_to_add` sets the total number of new units to allocate.
- `time_steps` sets how many simulation rounds to split that allocation across.

## Simulation

`run_development_sim.py` runs the simulation in three steps per time period:
1. Calculate development opportunity and allocate units within parcel capacity.
2. Recompute neighborhood walkability after the new units are placed, with
	synthetic amenities added near newly developed parcels. Parcels with more new units are more likely to receive new amenitie
3. Update residential `TOTAL_VALUE` using the pre-trained hedonic model. 

Housing production is allocated over a user-defined number time steps, with walkability raising market strength and higher total value raising acquisition cost intensity.

### Development Opportunity Scoring

Development allocation uses `development/allocation.py`.

The opportunity score combines:
1. Capacity upside: additional units allowed by zoning.
2. Market strength: neighborhood proxies (income, walkability, employment proximity).
3. Acquisition cost intensity: current value relative to zoned unit capacity.

Parcels are ranked by `opportunity_score`, assigned to high/medium/low tiers,
and allocated units in descending score order until each step target is met or
eligible parcel capacity is exhausted.

Weight controls are exposed in `run_development_sim.py`:
- `--w-capacity`
- `--w-market`
- `--w-cost`

The final output is the post-simulation parcel table `parcels_simulated.csv`.

Future development
1. Use a location choice model to add new ammenities.
2. Use household growth projections to create housing production targets.
3. Add a light visualizer to inspect simulation results. 
