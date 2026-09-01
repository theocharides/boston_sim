"""
Cleans FY2015 Boston parcel data and spatially joins FY2015 land use (and land area)
onto the FY2025 parcel preprocessed table.

Inputs:
- `preprocessing/raw_data/boston_parcel_assessors_2015.csv`
- `preprocessing/raw_data/boston_parcel_shapes_2015.geojson`
- `inputs/parcels_preprocessed.csv` (2025 preprocessed parcels)

Outputs:
- `inputs/parcels_preprocessed.csv` (with `lu_2015` added)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.utils import load_parcels_csv, save_parcels_csv, clean_numeric
from shared_utils import require_existing_path


def clean_geom(geom):
    """Repair and validate geometry if invalid or empty."""
    if geom is None or geom.is_empty:
        return None
    try:
        fixed = make_valid(geom) if not geom.is_valid else geom
    except Exception:
        fixed = geom
    if fixed is None or fixed.is_empty:
        return None
    if not fixed.is_valid:
        try:
            fixed = fixed.buffer(0)
        except Exception:
            return None
    return fixed if fixed is not None and not fixed.is_empty else None


def normalize_pid(series: pd.Series) -> pd.Series:
    """Normalize parcel/account IDs to a 10-digit numeric string."""
    return series.astype(str).str.replace(r"\D", "", regex=True).str.strip().str.zfill(10)


def to_base_pid(pid_norm: pd.Series) -> pd.Series:
    """Convert an account PID to a likely base parcel PID."""
    return pid_norm.str.slice(0, 7) + "000"


def clean_2015_parcels(assessors_csv: Path, shapes_geojson: Path) -> gpd.GeoDataFrame:
    """Read and collapse 2015 assessor records, then join with 2015 parcel geometries."""
    print(f"Reading 2015 assessors: {assessors_csv}")
    p1 = pd.read_csv(assessors_csv, low_memory=False)

    print(f"Reading 2015 parcel shapes: {shapes_geojson}")
    ps1 = gpd.read_file(shapes_geojson)

    # 1) Collapse condo records (where CM_ID is present)
    source = p1[["CM_ID", "PID", "LU", "LAND_SF"]].copy()
    if "LAND_SF" in source.columns:
        source["LAND_SF"] = clean_numeric(source["LAND_SF"])

    mask_cm = source["CM_ID"].notna()
    p1_no_cm = source.loc[~mask_cm, ["CM_ID", "PID", "LU", "LAND_SF"]]

    p1_cm = (
        source.loc[mask_cm, ["CM_ID", "PID", "LU", "LAND_SF"]]
        .assign(anchor_lu=lambda d: d["LU"].where(d["PID"].astype(str).eq(d["CM_ID"].astype(str))))
        .groupby("CM_ID", as_index=False)
        .agg(
            LAND_SF=("LAND_SF", "sum"),
            LU=("anchor_lu", "first"),
            fallback_lu=("LU", "first"),
        )
    )
    p1_cm["LU"] = p1_cm["LU"].fillna(p1_cm["fallback_lu"])
    p1_cm["PID"] = p1_cm["CM_ID"]
    p1_cm = p1_cm[["CM_ID", "PID", "LU", "LAND_SF"]]

    p1_clean = pd.concat([p1_no_cm, p1_cm], ignore_index=True)
    p1_clean["_pid_norm"] = normalize_pid(p1_clean["PID"])
    p1_clean["_base_pid"] = to_base_pid(p1_clean["_pid_norm"])

    # Deduplicate 2015 parcel shapes by GIS ID / PID
    shape_id_col = "TaxData2015_GIS_ID" if "TaxData2015_GIS_ID" in ps1.columns else ps1.columns[0]
    ps1["_gis_id_norm"] = normalize_pid(ps1[shape_id_col])
    ps1 = ps1.drop_duplicates(subset=["_gis_id_norm"], keep="first").copy().reset_index(drop=True)

    shape_keys = set(ps1["_gis_id_norm"])

    # Match 2015 assessors to shape key
    p1_clean["_parcel_key"] = np.where(
        p1_clean["_pid_norm"].isin(shape_keys),
        p1_clean["_pid_norm"],
        np.where(p1_clean["_base_pid"].isin(shape_keys), p1_clean["_base_pid"], pd.NA),
    )

    matched = p1_clean[p1_clean["_parcel_key"].notna()].copy()

    # Aggregate matched records per 2015 parcel geometry
    collapsed = (
        matched.groupby("_parcel_key", as_index=False)
        .agg(
            lu_2015=("LU", "first"),
            land_sf_2015=("LAND_SF", "sum"),
        )
    )

    p15_gdf = ps1[["_gis_id_norm", "geometry"]].merge(
        collapsed,
        left_on="_gis_id_norm",
        right_on="_parcel_key",
        how="left",
    )
    p15_gdf = gpd.GeoDataFrame(p15_gdf, geometry="geometry", crs=ps1.crs)
    p15_gdf = p15_gdf[p15_gdf.geometry.notna() & ~p15_gdf.geometry.is_empty].copy()
    return p15_gdf


def join_2015_land_use_to_2025(
    parcels_2025_gdf: gpd.GeoDataFrame,
    parcels_2015_gdf: gpd.GeoDataFrame,
    target_crs: str = "EPSG:26986",
) -> gpd.GeoDataFrame:
    """Spatially join 2015 land use onto 2025 parcels by selecting maximum intersection area."""
    print("Reprojecting geometries to projected CRS for spatial join...")
    cols_to_drop = [
        col for col in parcels_2025_gdf.columns
        if col in {"_p25_row", "LU_2015", "lu_2015", "land_sf_2015", "flag_multiple_2015_in_2025", "flag_multiple_2025_in_2015"}
        or col.startswith("lu_2015_") or col.startswith("land_sf_2015_")
    ]
    p25 = parcels_2025_gdf.drop(columns=cols_to_drop, errors="ignore").to_crs(target_crs).copy()
    p15 = parcels_2015_gdf.to_crs(target_crs).copy()

    print("Repairing invalid geometries...")
    p25["geometry"] = p25["geometry"].apply(clean_geom)
    p15["geometry"] = p15["geometry"].apply(clean_geom)

    p25 = p25[p25.geometry.notna() & ~p25.geometry.is_empty].copy()
    p15 = p15[p15.geometry.notna() & ~p15.geometry.is_empty].copy()

    p25_base = p25.reset_index().rename(columns={"index": "_p25_row"})
    p15_base = p15[["lu_2015", "land_sf_2015", "geometry"]].reset_index().rename(columns={"index": "_p15_row"})

    print("Performing spatial join (intersects)...")
    joined = gpd.sjoin(
        p25_base[["_p25_row", "geometry"]],
        p15_base[["_p15_row", "lu_2015", "land_sf_2015", "geometry"]],
        how="left",
        predicate="intersects",
    )

    matched_pairs = joined[joined["_p15_row"].notna()].copy()

    if not matched_pairs.empty:
        matched_pairs["_p15_row"] = matched_pairs["_p15_row"].astype(int)
        matched_pairs = matched_pairs.merge(
            p15_base[["_p15_row", "geometry"]].rename(columns={"geometry": "_p15_geom"}),
            on="_p15_row",
            how="left",
        )

        print("Computing intersection areas for multi-match resolution...")
        def calc_overlap(row):
            g25 = row["geometry"]
            g15 = row["_p15_geom"]
            if g25 is None or g15 is None:
                return 0.0
            try:
                return g25.intersection(g15).area
            except Exception:
                try:
                    return g25.buffer(0).intersection(g15.buffer(0)).area
                except Exception:
                    return 0.0

        matched_pairs["overlap_area"] = matched_pairs.apply(calc_overlap, axis=1)

        # Flag 1: 2025 parcel intersects multiple 2015 parcels
        multi_2015_for_2025 = (
            matched_pairs.groupby("_p25_row")["_p15_row"]
            .nunique()
            .gt(1)
            .rename("flag_multiple_2015_in_2025")
            .reset_index()
        )

        best_matches = (
            matched_pairs.sort_values(
                ["_p25_row", "overlap_area"],
                ascending=[True, False],
            )
            .drop_duplicates(subset=["_p25_row"], keep="first")
            .copy()
        )

        # Flag 2: 2015 parcel is matched to multiple 2025 parcels
        multi_2025_ids = (
            best_matches.groupby("_p15_row")["_p25_row"]
            .nunique()
            .loc[lambda s: s > 1]
            .index
        )
        best_matches["flag_multiple_2025_in_2015"] = best_matches["_p15_row"].isin(multi_2025_ids)

        result_df = p25_base.merge(multi_2015_for_2025, on="_p25_row", how="left")
        result_df = result_df.merge(
            best_matches[["_p25_row", "lu_2015", "land_sf_2015", "flag_multiple_2025_in_2015"]],
            on="_p25_row",
            how="left",
        )
        result_df["flag_multiple_2015_in_2025"] = result_df["flag_multiple_2015_in_2025"].fillna(False).astype(bool)
        result_df["flag_multiple_2025_in_2015"] = result_df["flag_multiple_2025_in_2015"].fillna(False).astype(bool)
    else:
        result_df = p25_base.copy()
        result_df["lu_2015"] = pd.NA
        result_df["land_sf_2015"] = pd.NA
        result_df["flag_multiple_2015_in_2025"] = False
        result_df["flag_multiple_2025_in_2015"] = False

    result_df = result_df.drop(columns=["_p25_row", "LU_2015"], errors="ignore")

    # Re-assign to original GeoDataFrame with original CRS
    out_gdf = gpd.GeoDataFrame(result_df, geometry="geometry", crs=target_crs)
    out_gdf = out_gdf.to_crs(parcels_2025_gdf.crs)
    return out_gdf


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Clean 2015 parcels and spatially join 2015 land use to 2025 parcels."
    )
    parser.add_argument(
        "--parcels-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels_preprocessed.csv",
        help="Path to 2025 preprocessed parcel CSV.",
    )
    parser.add_argument(
        "--assessors-2015",
        type=Path,
        default=repo_root / "preprocessing" / "raw_data" / "boston_parcel_assessors_2015.csv",
        help="Path to 2015 assessors CSV.",
    )
    parser.add_argument(
        "--shapes-2015",
        type=Path,
        default=repo_root / "preprocessing" / "raw_data" / "boston_parcel_shapes_2015.geojson",
        help="Path to 2015 parcel shapes GeoJSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "inputs" / "parcels_preprocessed.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()
    args.parcels_csv = require_existing_path(args.parcels_csv, "2025 Parcels CSV")
    args.assessors_2015 = require_existing_path(args.assessors_2015, "2015 Assessors CSV")
    args.shapes_2015 = require_existing_path(args.shapes_2015, "2015 Parcel Shapes")
    args.output_csv = args.output_csv.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()

    print(f"Loading 2025 parcels: {args.parcels_csv}")
    p25_gdf = load_parcels_csv(args.parcels_csv)

    p15_gdf = clean_2015_parcels(args.assessors_2015, args.shapes_2015)
    print(f"Cleaned 2015 parcel shapes count: {len(p15_gdf):,}")

    enriched_gdf = join_2015_land_use_to_2025(p25_gdf, p15_gdf)

    print(f"Saving enriched 2025 parcels to {args.output_csv}")
    save_parcels_csv(enriched_gdf, args.output_csv)

    matched_count = enriched_gdf["lu_2015"].notna().sum()
    total_count = len(enriched_gdf)
    print(f"2015 LU joined: {matched_count:,} / {total_count:,} parcels ({matched_count / total_count:.1%})")


if __name__ == "__main__":
    main()
