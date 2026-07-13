"""Adds tract median household income to parcel records.

This script downloads ACS 5-year tract-level median household income
(B19013_001E), joins it to tract geometry, spatially matches parcels
to tracts, and writes the tract income onto each parcel row.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from json import JSONDecodeError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import pandas as pd

from utils import load_parcels_csv, require_existing_path, save_parcels_csv, to_point_geometry


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Add tract median household income to parcel records."
    )
    parser.add_argument(
        "--parcels-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels.csv",
        help="Path to parcels CSV with geometry in WKT.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels.csv",
        help="Output CSV path (can be same as input for in-place update).",
    )
    parser.add_argument(
        "--acs-year",
        type=int,
        default=2023,
        help="ACS 5-year dataset year (e.g., 2022 or 2023).",
    )
    parser.add_argument(
        "--state-fips",
        type=str,
        default="25",
        help="State FIPS code for ACS query and tract geometry.",
    )
    parser.add_argument(
        "--county-fips",
        type=str,
        default="025",
        help="County FIPS code for ACS query and tract geometry.",
    )
    parser.add_argument(
        "--income-column",
        type=str,
        default="median_hh_income",
        help="Name of output income column.",
    )
    parser.add_argument(
        "--census-api-key",
        type=str,
        default=None,
        help="Optional Census API key. If omitted, uses env var CENSUS_API_KEY.",
    )

    args = parser.parse_args()
    args.parcels_csv = require_existing_path(args.parcels_csv, "Parcels CSV")
    args.output_csv = args.output_csv.expanduser().resolve()
    args.state_fips = args.state_fips.zfill(2)
    args.county_fips = args.county_fips.zfill(3)
    return args


def fetch_acs_income(
    acs_year: int,
    state_fips: str,
    county_fips: str,
    census_api_key: str | None,
) -> pd.DataFrame:
    """Fetch tract median household income from ACS API."""
    base_url = f"https://api.census.gov/data/{acs_year}/acs/acs5"
    query = {
        "get": "B19013_001E",
        "for": "tract:*",
        "in": f"state:{state_fips} county:{county_fips}",
    }
    if census_api_key:
        query["key"] = census_api_key

    url = f"{base_url}?{urlencode(query)}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response:
        body = response.read().decode("utf-8", errors="replace")

    try:
        payload = json.loads(body)
    except JSONDecodeError as exc:
        preview = body.strip().replace("\n", " ")[:240]
        raise ValueError(
            "ACS API returned a non-JSON response. "
            "This is commonly caused by a missing/invalid Census API key. "
            f"Response preview: {preview}"
        ) from exc

    if isinstance(payload, dict) and "error" in payload:
        raise ValueError(f"ACS API error: {payload['error']}")

    if not payload or len(payload) < 2:
        raise ValueError("ACS API returned no tract rows for the selected geography.")

    header = payload[0]
    rows = payload[1:]
    df = pd.DataFrame(rows, columns=header)
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["B19013_001E"] = pd.to_numeric(df["B19013_001E"], errors="coerce")
    return df[["GEOID", "B19013_001E"]]


def load_tract_geometry(acs_year: int, state_fips: str, county_fips: str) -> gpd.GeoDataFrame:
    """Download tract geometry from TIGER/Line and filter to county."""
    tiger_url = f"https://www2.census.gov/geo/tiger/TIGER{acs_year}/TRACT/tl_{acs_year}_{state_fips}_tract.zip"
    tracts = gpd.read_file(tiger_url)

    if "COUNTYFP" not in tracts.columns or "GEOID" not in tracts.columns:
        raise ValueError("Unexpected tract geometry schema from TIGER source.")

    tracts = tracts[tracts["COUNTYFP"] == county_fips].copy()
    if tracts.empty:
        raise ValueError("No tract geometry rows found for requested county FIPS.")

    return tracts[["GEOID", "geometry"]]


def main() -> None:
    args = parse_args()

    census_api_key = args.census_api_key or os.getenv("CENSUS_API_KEY")
    if not census_api_key:
        raise ValueError(
            "Census API key is required. Set environment variable CENSUS_API_KEY "
            "or pass --census-api-key."
        )

    print(f"Reading parcels: {args.parcels_csv}")
    parcels = load_parcels_csv(args.parcels_csv)

    valid = parcels.geometry.notna()
    if int(valid.sum()) == 0:
        raise ValueError("No valid parcel geometries found in input.")

    print(
        f"Fetching ACS tract income for state={args.state_fips}, county={args.county_fips}, year={args.acs_year}..."
    )
    income_df = fetch_acs_income(
        acs_year=args.acs_year,
        state_fips=args.state_fips,
        county_fips=args.county_fips,
        census_api_key=census_api_key,
    )

    print("Downloading tract geometry from TIGER/Line...")
    tracts = load_tract_geometry(
        acs_year=args.acs_year,
        state_fips=args.state_fips,
        county_fips=args.county_fips,
    )

    tracts_income = tracts.merge(income_df, on="GEOID", how="left")
    tracts_income = tracts_income.rename(columns={"B19013_001E": args.income_column})

    parcel_points = parcels.loc[valid].copy()
    parcel_points["geometry"] = parcel_points.geometry.map(to_point_geometry)
    parcel_points = gpd.GeoDataFrame(parcel_points, geometry="geometry", crs=parcels.crs)

    if parcel_points.crs != tracts_income.crs:
        tracts_income = tracts_income.to_crs(parcel_points.crs)

    print("Joining parcel points to tracts...")
    joined = gpd.sjoin(
        parcel_points[["geometry"]],
        tracts_income[["GEOID", args.income_column, "geometry"]],
        how="left",
        predicate="within",
    )

    income_by_parcel = joined.groupby(joined.index)[args.income_column].first()

    parcels[args.income_column] = np.nan
    parcels.loc[income_by_parcel.index, args.income_column] = income_by_parcel.values

    save_parcels_csv(parcels, args.output_csv)

    rows_with_income = int(pd.notna(parcels[args.income_column]).sum())
    print(f"Rows written: {len(parcels):,}")
    print(f"Rows with income value: {rows_with_income:,}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()