"""Add neighborhood tags to parcels via spatial join with Boston boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.utils import load_parcels_csv, save_parcels_csv, to_point_geometry
from shared_utils import require_existing_path

NEIGHBORHOOD_NAME_CANDIDATES = ["name", "neighborhood", "neighborhood_name", "NBHD", "NBH_NAME"]
NEIGHBORHOOD_ID_CANDIDATES = ["neighborhood_id", "id", "OBJECTID"]


def choose_column(columns: list[str], candidates: list[str]) -> str | None:
    by_lower = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Spatially join Boston neighborhood boundaries onto parcel records."
    )
    parser.add_argument(
        "--parcels-csv",
        type=Path,
        default=repo_root / "outputs" / "parcels_preprocessed.csv",
        help="Path to parcel CSV with geometry in WKT.",
    )
    parser.add_argument(
        "--neighborhood-geojson",
        type=Path,
        default=repo_root / "preprocessing" / "raw_data" / "boston_neighborhood_boundaries.geojson",
        help="Path to neighborhood boundary polygons.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "outputs" / "parcels_preprocessed.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--neighborhood-column",
        type=str,
        default="neighborhood_name",
        help="Output column name for neighborhood label.",
    )
    parser.add_argument(
        "--neighborhood-id-column",
        type=str,
        default="neighborhood_id",
        help="Output column name for neighborhood ID (if available).",
    )

    args = parser.parse_args()
    args.parcels_csv = require_existing_path(args.parcels_csv, "Parcels CSV")
    args.neighborhood_geojson = require_existing_path(args.neighborhood_geojson, "Neighborhood boundary file")
    args.output_csv = args.output_csv.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()

    print(f"Reading parcels: {args.parcels_csv}")
    parcels = load_parcels_csv(args.parcels_csv)

    print(f"Reading neighborhood boundaries: {args.neighborhood_geojson}")
    neighborhoods = gpd.read_file(args.neighborhood_geojson)
    if neighborhoods.empty:
        raise ValueError("Neighborhood boundary file has no rows.")
    if neighborhoods.geometry.isna().all():
        raise ValueError("Neighborhood boundary file has no valid geometries.")

    name_source_col = choose_column(list(neighborhoods.columns), NEIGHBORHOOD_NAME_CANDIDATES)
    if name_source_col is None:
        raise ValueError(
            "Could not find a neighborhood name column in boundary file. "
            f"Expected one of {NEIGHBORHOOD_NAME_CANDIDATES}."
        )

    id_source_col = choose_column(list(neighborhoods.columns), NEIGHBORHOOD_ID_CANDIDATES)

    keep_cols = [name_source_col, "geometry"]
    if id_source_col is not None and id_source_col not in keep_cols:
        keep_cols.insert(1, id_source_col)
    neighborhoods = neighborhoods[keep_cols].copy()

    valid = parcels.geometry.notna()
    if int(valid.sum()) == 0:
        raise ValueError("No valid parcel geometries found in input.")

    parcel_points = parcels.loc[valid].copy()
    parcel_points["geometry"] = parcel_points.geometry.map(to_point_geometry)
    parcel_points = gpd.GeoDataFrame(parcel_points, geometry="geometry", crs=parcels.crs)

    if parcel_points.crs != neighborhoods.crs:
        neighborhoods = neighborhoods.to_crs(parcel_points.crs)

    print("Joining parcel points to neighborhood boundaries...")
    join_cols = ["geometry", name_source_col]
    if id_source_col is not None:
        join_cols.append(id_source_col)
    joined = gpd.sjoin(
        parcel_points[["geometry"]],
        neighborhoods[join_cols],
        how="left",
        predicate="within",
    )

    neighborhood_name_by_parcel = joined.groupby(joined.index)[name_source_col].first()
    parcels[args.neighborhood_column] = pd.NA
    parcels.loc[neighborhood_name_by_parcel.index, args.neighborhood_column] = neighborhood_name_by_parcel.values

    if id_source_col is not None:
        neighborhood_id_by_parcel = joined.groupby(joined.index)[id_source_col].first()
        parcels[args.neighborhood_id_column] = pd.NA
        parcels.loc[neighborhood_id_by_parcel.index, args.neighborhood_id_column] = neighborhood_id_by_parcel.values

    save_parcels_csv(parcels, args.output_csv)

    print(f"Rows written: {len(parcels):,}")
    print(f"Rows with neighborhood match: {int(parcels[args.neighborhood_column].notna().sum()):,}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
