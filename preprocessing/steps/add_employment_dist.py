"""
Adds parcel-level distance to major employment centers/CBDs.

This script computes straight-line distances (in meters) from each parcel
to the nearest employment center/CBD.

Input parcels must include a `geometry` column in WKT format.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.utils import load_parcels_csv, save_parcels_csv, to_point_geometry
from shared_utils import require_existing_path


DEFAULT_CENTERS = [
    {"name": "Downtown Boston CBD", "lat": 42.3555, "lon": -71.0605},
    {"name": "Back Bay", "lat": 42.3493, "lon": -71.0799},
    {"name": "Seaport", "lat": 42.3503, "lon": -71.0446},
    {"name": "Longwood Medical Area", "lat": 42.3360, "lon": -71.1057},
    {"name": "Kendall Square", "lat": 42.3626, "lon": -71.0863},
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Add employment-center/CBD straight-line distance to parcels."
    )
    parser.add_argument(
        "--parcels-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels_preprocessed.csv",
        help="Path to parcels CSV with geometry in WKT.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels_preprocessed.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()
    args.parcels_csv = require_existing_path(args.parcels_csv, "Parcels CSV")
    args.output_csv = args.output_csv.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()

    print(f"Reading parcels: {args.parcels_csv}")
    parcels = load_parcels_csv(args.parcels_csv)

    valid = parcels.geometry.notna()
    if int(valid.sum()) == 0:
        raise ValueError("No valid parcel geometries found in input.")

    parcel_points = parcels.loc[valid].copy()
    parcel_points["geometry"] = parcel_points.geometry.map(to_point_geometry)

    metric_crs = parcel_points.estimate_utm_crs()
    if metric_crs is None:
        raise ValueError("Unable to infer projected CRS for parcel extent.")

    parcel_points_metric = parcel_points.to_crs(metric_crs)

    centers_df = pd.DataFrame(DEFAULT_CENTERS)
    if centers_df.empty:
        raise ValueError("No center points available for distance calculation.")

    centers_gdf = gpd.GeoDataFrame(
        centers_df, geometry=gpd.points_from_xy(centers_df["lon"], centers_df["lat"]), crs="EPSG:4326"
    )
    centers_metric = centers_gdf.to_crs(metric_crs)

    parcel_x = parcel_points_metric.geometry.x.to_numpy()
    parcel_y = parcel_points_metric.geometry.y.to_numpy()

    nearest_distances = np.full(shape=len(parcel_points_metric), fill_value=np.inf, dtype=float)
    print("Computing nearest employment-center straight-line distances...")
    for center_geom in centers_metric.geometry:
        dx = parcel_x - float(center_geom.x)
        dy = parcel_y - float(center_geom.y)
        distances = np.hypot(dx, dy)
        improved = distances < nearest_distances
        nearest_distances[improved] = distances[improved]

    parcels["emp_dist_m"] = np.nan
    parcels.loc[valid, "emp_dist_m"] = nearest_distances.tolist()

    save_parcels_csv(parcels, args.output_csv)

    n_with_distance = int(pd.notna(parcels["emp_dist_m"]).sum())
    print(f"Centers used: {len(centers_df):,}")
    print(f"Rows written: {len(parcels):,}")
    print(f"Rows with center distance: {n_with_distance:,}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
