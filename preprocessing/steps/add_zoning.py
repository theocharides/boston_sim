"""
Adds zoning attributes to parcel data via spatial join.

This script reads cleaned parcel data (with geometry in WKT format) and
performs a spatial join with zoning subdistrict boundaries to add zoning
attributes to each parcel.

Inputs:
- `parcels_preprocessed.csv` (output from clean_parcels.py)
- `boston_zoning_subdistricts/Boston_Zoning_Subdistricts.shp`

Outputs:
- `parcels_preprocessed.csv`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import geopandas as gpd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.utils import load_parcels_csv, save_parcels_csv
from shared_utils import require_existing_path


ZONING_COLUMN_MAP: dict[str, str] = {
    "Zoning_Sub": "zoning_use",
    "Max_FAR": "max_far",
    "Max_Height": "max_height",
    "Front_Setb": "front_setback",
    "Side_Setba": "side_setback",
    "Rear_Setba": "rear_setback",
    "Max_Dwelli": "max_dua",
    "Max_Number": "max_floors",
}

OUTPUT_COLUMNS: list[str] = [
    "PID",
    "CONDO_ID",
    "NUM_BLDGS",
    "LU",
    "LU_DESC",
    "BLDG_TYPE",
    "RES_FLOOR",
    "RES_UNITS",
    "TT_RMS",
    "BED_RMS",
    "FULL_BTH",
    "HLF_BTH",
    "KITCHENS",
    "OVERALL_COND",
    "INT_COND",
    "EXT_COND",
    "NUM_PARKING",
    "STRUCTURE_CLASS",
    "YR_REMODEL",
    "YR_BUILT",
    "LAND_VALUE",
    "BLDG_VALUE",
    "TOTAL_VALUE",
    "LAND_SF",
    "GROSS_AREA",
    "LIVING_AREA",
    *ZONING_COLUMN_MAP.values(),
    "geometry",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Add zoning attributes to cleaned parcel data."
    )
    parser.add_argument(
        "--parcels-cleaned",
        type=Path,
        default=repo_root / "parcels_preprocessed.csv",
        help="Path to cleaned parcels CSV (output from clean_parcels.py).",
    )
    parser.add_argument(
        "--zoning-shapefile",
        type=Path,
        default=repo_root
        / "preprocessing"
        / "raw_data"
        / "boston_zoning_subdistricts"
        / "Boston_Zoning_Subdistricts.shp",
        help="Path to zoning subdistrict shapefile.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "parcels_preprocessed.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()
    args.parcels_cleaned = require_existing_path(args.parcels_cleaned, "Cleaned parcels CSV")
    args.zoning_shapefile = require_existing_path(args.zoning_shapefile, "Zoning shapefile")
    args.output_csv = args.output_csv.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()

    print(f"Reading cleaned parcels: {args.parcels_cleaned}")
    result_geo = load_parcels_csv(args.parcels_cleaned)

    print(f"Reading zoning shapefile: {args.zoning_shapefile}")
    zoning = gpd.read_file(args.zoning_shapefile, columns=[*ZONING_COLUMN_MAP.keys(), "geometry"])
    zoning = zoning[[*ZONING_COLUMN_MAP.keys(), "geometry"]].rename(columns=ZONING_COLUMN_MAP)

    if result_geo.crs != zoning.crs:
        print(f"Reprojecting zoning to match parcels CRS ({result_geo.crs})...")
        zoning = zoning.to_crs(result_geo.crs)

    valid_geoms = result_geo.geometry.notna()
    result_with_geom = result_geo.loc[valid_geoms, ["PID", "geometry"]].copy()

    print("Joining zoning attributes by parcel geometry...")
    joined = gpd.sjoin(result_with_geom, zoning, how="left", predicate="intersects")
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    zoning_out_cols = list(ZONING_COLUMN_MAP.values())
    zoning_by_parcel = (
        joined[["PID", *zoning_out_cols]]
        .sort_values(by=["PID"])
        .drop_duplicates(subset=["PID"], keep="first")
    )
    result_geo = result_geo.merge(zoning_by_parcel, on="PID", how="left")

    # Emit only the requested final schema in the requested order.
    for column_name in OUTPUT_COLUMNS:
        if column_name not in result_geo.columns:
            result_geo[column_name] = pd.NA
    result_geo = result_geo[OUTPUT_COLUMNS]

    save_parcels_csv(result_geo, args.output_csv)

    print(f"Output rows: {len(result_geo):,}")
    print(f"Rows with zoning match: {result_geo['zoning_use'].notna().sum():,}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
