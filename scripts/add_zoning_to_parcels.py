"""
Add zoning subdistrict attributes to Boston assessor parcels and
rename columns to simplified names.

This script reads the assessor CSV, joins parcel geometries from the parcel
GeoJSON using normalized parcel IDs, then performs a spatial join against the
Boston zoning subdistrict shapefile. It writes both a tabular CSV and a
spatial GeoJSON output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def normalize_pid(series: pd.Series) -> pd.Series:
    """Normalize parcel IDs so assessor PIDs and MAP_PAR_ID can be joined."""
    return series.astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)


def require_existing_path(path: Path, label: str) -> Path:
    """Return a resolved path or raise a clear error for missing inputs."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    parser = argparse.ArgumentParser(
        description="Attach zoning attributes to parcel assessors using a spatial join."
    )
    parser.add_argument(
        "--assessors-csv",
        type=Path,
        default=data_dir / "boston_parcel_assessors_cleaned.csv",
        help="Path to assessor CSV.",
    )
    parser.add_argument(
        "--zoning-shapefile",
        type=Path,
        default=data_dir
        / "boston_zoning_subdistricts"
        / "Boston_Zoning_Subdistricts.shp",
        help="Path to zoning subdistrict shapefile.",
    )
    parser.add_argument(
        "--parcel-shapes",
        type=Path,
        default=data_dir / "boston_parcel_shapes.geojson",
        help="Path to parcel geometry GeoJSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-spatial",
        type=Path,
        default=repo_root / "data" / "parcels_spatial.geojson",
        help="Output spatial file path (GeoJSON).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of assessor rows to process for testing.",
    )

    args = parser.parse_args()
    args.assessors_csv = require_existing_path(args.assessors_csv, "Assessors CSV")
    args.parcel_shapes = require_existing_path(args.parcel_shapes, "Parcel shapes file")
    args.zoning_shapefile = require_existing_path(args.zoning_shapefile, "Zoning shapefile")
    args.output_csv = args.output_csv.expanduser().resolve()
    args.output_spatial = args.output_spatial.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()

    zoning_columns = {
        "Zoning_Sub": "zoning_use",
        "Max_FAR": "max_far",
        "Max_Height": "max_height",
        "Front_Setb": "front_setback",
        "Side_Setba": "side_setback",
        "Rear_Setba": "rear_setback",
        "Max_Dwelli": "max_dua",
        "Max_Number": "max_floors",
    }

    print(f"Reading assessors CSV: {args.assessors_csv}")
    assessors = pd.read_csv(args.assessors_csv)
    if args.sample_size is not None:
        assessors = assessors.head(args.sample_size)

    assessors = assessors.copy()
    assessors["_row_id"] = assessors.index
    assessors["_parcel_key"] = normalize_pid(assessors["PID"])

    print(f"Reading parcel geometry: {args.parcel_shapes}")
    parcels = gpd.read_file(args.parcel_shapes, columns=["MAP_PAR_ID", "geometry"])
    parcels = parcels[["MAP_PAR_ID", "geometry"]].copy()
    parcels["_parcel_key"] = normalize_pid(parcels["MAP_PAR_ID"])
    parcels = parcels.drop_duplicates(subset=["_parcel_key"], keep="first")

    assessors_geo = assessors.merge(
        parcels[["_parcel_key", "geometry"]],
        on="_parcel_key",
        how="left",
    )
    assessors_geo = gpd.GeoDataFrame(assessors_geo, geometry="geometry", crs=parcels.crs)

    print(f"Reading zoning shapefile: {args.zoning_shapefile}")
    zoning = gpd.read_file(args.zoning_shapefile, columns=[*zoning_columns.keys(), "geometry"])
    zoning = zoning[[*zoning_columns.keys(), "geometry"]].rename(columns=zoning_columns)

    if assessors_geo.crs != zoning.crs:
        zoning = zoning.to_crs(assessors_geo.crs)

    valid_geoms = assessors_geo.geometry.notna()
    assessors_with_geom = assessors_geo.loc[valid_geoms].copy()

    print("Running spatial join...")
    joined = gpd.sjoin(
        assessors_with_geom,
        zoning,
        how="left",
        predicate="intersects",
    )

    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    joined = joined.sort_values(by=["_row_id"]).drop_duplicates(subset=["_row_id"], keep="first")

    result = assessors.copy()
    zoning_out_cols = list(zoning_columns.values())
    result = result.merge(joined[["_row_id", *zoning_out_cols]], on="_row_id", how="left")
    result = result.drop(columns=["_row_id", "_parcel_key"])

    spatial_result = assessors_geo.merge(
        joined[["_row_id", *zoning_out_cols]],
        on="_row_id",
        how="left",
    )
    spatial_result = spatial_result.drop(columns=["_row_id", "_parcel_key"])
    spatial_result = gpd.GeoDataFrame(spatial_result, geometry="geometry", crs=assessors_geo.crs)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_spatial.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    spatial_result.to_file(args.output_spatial, driver="GeoJSON")

    matched_geom = valid_geoms.sum()
    with_zoning = result["zoning_use"].notna().sum()
    print(f"Rows written: {len(result):,}")
    print(f"Rows with parcel geometry: {matched_geom:,}")
    print(f"Rows with zoning match: {with_zoning:,}")
    print(f"CSV output: {args.output_csv}")
    print(f"Spatial output: {args.output_spatial}")


if __name__ == "__main__":
    main()
