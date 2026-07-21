"""Compute neighborhood walkability from OSM network distance to key amenities."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.utils import (
    load_parcels_csv,
    save_parcels_csv,
    to_point_geometry,
)
from shared_utils import require_existing_path


def parse_args() -> argparse.Namespace:
    repo_root = REPO_ROOT

    parser = argparse.ArgumentParser(
        description="Add neighborhood walkability score to parcel records."
    )
    parser.add_argument(
        "--parcels-csv",
        type=Path,
        default=repo_root / "parcels_preprocessed.csv",
        help="Path to parcels CSV with geometry in WKT.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "parcels_preprocessed.csv",
        help="Output CSV path (can be same as input for in-place update).",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default="neighborhood_walkability",
        help="Name of output walkability score column.",
    )
    parser.add_argument(
        "--max-walk-distance-m",
        type=float,
        default=1600.0,
        help="Distance where category score decays to zero.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional number of parcels to score for a faster validation run.",
    )
    parser.add_argument(
        "--housing-growth-column",
        type=str,
        default=None,
        help="Optional parcel column used to create synthetic amenities near new housing.",
    )

    args = parser.parse_args()
    args.parcels_csv = require_existing_path(args.parcels_csv, "Parcels CSV")
    args.output_csv = args.output_csv.expanduser().resolve()
    return args


AMENITY_TAGS: dict[str, dict[str, object]] = {
    "grocery": {"shop": ["supermarket", "grocery", "convenience"]},
    "food": {"amenity": ["restaurant", "cafe", "fast_food"]},
    "education": {"amenity": ["school", "college", "university", "library"]},
    "park": {"leisure": ["park", "playground"]},
    "transit": {"public_transport": True, "railway": ["station", "tram_stop"], "highway": "bus_stop"},
}


def _linear_distance_score(distance_m: pd.Series, max_distance_m: float) -> pd.Series:
    """Convert distances to 0-100 scores with linear decay."""
    score = 100.0 * (1.0 - (distance_m / max_distance_m))
    return score.clip(lower=0.0, upper=100.0).where(distance_m.notna())


def _ensure_points(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert mixed OSM geometries to representative points."""
    out = features.copy()
    out = out[out.geometry.notna()].copy()
    out["geometry"] = out.geometry.map(lambda geom: geom if geom.geom_type == "Point" else geom.representative_point())
    return out


def _flatten_feature_columns(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reset multi-index columns returned by OSMnx feature queries."""
    out = features.copy()
    out.columns = ["_".join(str(part) for part in col if str(part) != "") if isinstance(col, tuple) else str(col) for col in out.columns]
    return out


def _download_amenity_points(parcel_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Download amenity features by category within parcel extent."""
    parcel_polygon = parcel_points.to_crs("EPSG:4326").unary_union.convex_hull.buffer(0.01)
    amenity_frames: list[gpd.GeoDataFrame] = []

    for category, tags in AMENITY_TAGS.items():
        features = ox.features_from_polygon(parcel_polygon, tags=tags)
        if features.empty:
            continue
        features = _flatten_feature_columns(features)
        features = _ensure_points(features)
        if features.empty:
            continue
        features = features[["geometry"]].copy()
        features["category"] = category
        amenity_frames.append(features)

    if not amenity_frames:
        raise ValueError("No walkability destination features were downloaded from OpenStreetMap.")

    return pd.concat(amenity_frames, ignore_index=True)


SYNTHETIC_AMENITY_UNITS_PER_FEATURE: dict[str, float] = {
    "food": 120.0,
    "grocery": 300.0,
    "park": 450.0,
    "transit": 700.0,
    "education": 1200.0,
}
SYNTHETIC_AMENITY_JITTER_M = 100.0


def _build_synthetic_amenities(
    parcel_points: gpd.GeoDataFrame,
    growth_column: str | None,
    graph_crs: str,
) -> gpd.GeoDataFrame:
    """Create small synthetic amenity clusters near parcels with new housing."""
    if not growth_column or growth_column not in parcel_points.columns:
        return gpd.GeoDataFrame(columns=["geometry", "category"], geometry="geometry", crs=graph_crs)

    growth_values = pd.to_numeric(parcel_points[growth_column], errors="coerce").fillna(0.0)
    growth_values = growth_values[growth_values.gt(0)]
    if growth_values.empty:
        return gpd.GeoDataFrame(columns=["geometry", "category"], geometry="geometry", crs=graph_crs)

    total_growth = float(growth_values.sum())
    growth_index = growth_values.index.to_numpy()
    growth_weights = growth_values.to_numpy(dtype=float)
    growth_weights = growth_weights / growth_weights.sum()
    rng = np.random.default_rng(42)

    synthetic_frames: list[gpd.GeoDataFrame] = []
    for category, units_per_feature in SYNTHETIC_AMENITY_UNITS_PER_FEATURE.items():
        feature_count = int(np.floor(total_growth / units_per_feature))
        if feature_count <= 0:
            continue

        sampled_indices = rng.choice(growth_index, size=feature_count, replace=True, p=growth_weights)
        points: list[Point] = []
        for parcel_index in sampled_indices:
            base_point = parcel_points.loc[parcel_index].geometry
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            radius = float(rng.uniform(0.0, SYNTHETIC_AMENITY_JITTER_M))
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            points.append(Point(float(base_point.x + dx), float(base_point.y + dy)))

        synthetic_frames.append(
            gpd.GeoDataFrame({"geometry": points, "category": category}, geometry="geometry", crs=graph_crs)
        )

    if not synthetic_frames:
        return gpd.GeoDataFrame(columns=["geometry", "category"], geometry="geometry", crs=graph_crs)

    return pd.concat(synthetic_frames, ignore_index=True)


def main() -> None:
    args = parse_args()

    print(f"Reading parcels: {args.parcels_csv}")
    parcels = load_parcels_csv(args.parcels_csv)
    valid = parcels.geometry.notna()
    if int(valid.sum()) == 0:
        raise ValueError("No valid parcel geometries found in input.")

    parcel_points = parcels.loc[valid].copy()
    parcel_points["geometry"] = parcel_points.geometry.map(to_point_geometry)
    if args.sample_size and args.sample_size > 0 and len(parcel_points) > args.sample_size:
        parcel_points = parcel_points.sample(n=args.sample_size, random_state=42).copy()

    hull = parcel_points.to_crs("EPSG:4326").unary_union.convex_hull
    print("Downloading walking network from OpenStreetMap...")
    graph = ox.graph_from_polygon(hull.buffer(0.01), network_type="walk", simplify=True)
    if len(graph.nodes) == 0:
        raise ValueError("Downloaded walking network has no nodes.")

    graph = ox.project_graph(graph)
    graph_crs = graph.graph.get("crs")
    parcel_points = parcel_points.to_crs(graph_crs)

    print("Downloading walk-destination amenities from OpenStreetMap...")
    amenities = _download_amenity_points(parcel_points)
    amenities = gpd.GeoDataFrame(amenities, geometry="geometry", crs="EPSG:4326").to_crs(graph_crs)

    synthetic_amenities = _build_synthetic_amenities(parcel_points, args.housing_growth_column, graph_crs)
    if not synthetic_amenities.empty:
        print(f"Adding {len(synthetic_amenities):,} synthetic amenities from housing growth...")
        amenities = pd.concat([amenities, synthetic_amenities], ignore_index=True)

    parcel_node_ids = ox.distance.nearest_nodes(
        graph,
        X=parcel_points.geometry.x.to_numpy(),
        Y=parcel_points.geometry.y.to_numpy(),
    )
    parcel_node_series = pd.Series(parcel_node_ids, index=parcel_points.index)

    category_scores: list[pd.Series] = []
    print("Computing network distance to nearest amenities by category...")
    for category in sorted(amenities["category"].unique()):
        amenity_subset = amenities.loc[amenities["category"] == category].copy()
        amenity_node_ids = ox.distance.nearest_nodes(
            graph,
            X=amenity_subset.geometry.x.to_numpy(),
            Y=amenity_subset.geometry.y.to_numpy(),
        )
        amenity_node_ids = pd.Index(pd.Series(amenity_node_ids).dropna().astype(int).unique())
        if amenity_node_ids.empty:
            continue

        distances_to_targets = nx.multi_source_dijkstra_path_length(
            graph,
            sources=list(amenity_node_ids),
            weight="length",
        )
        parcel_distances = parcel_node_series.map(distances_to_targets)
        category_score = _linear_distance_score(parcel_distances.astype(float), args.max_walk_distance_m)
        category_score.name = f"walk_{category}_score"
        category_scores.append(category_score)

    if not category_scores:
        raise ValueError("No walkability category scores could be computed from the downloaded network and amenities.")

    score_components = pd.concat(category_scores, axis=1)
    final_score = score_components.mean(axis=1, skipna=True)

    parcels[args.score_column] = np.nan
    parcels.loc[score_components.index, args.score_column] = final_score.round(2)

    save_parcels_csv(parcels, args.output_csv)

    n_with_score = int(pd.notna(parcels[args.score_column]).sum())
    print(f"Rows written: {len(parcels):,}")
    print(f"Rows with walkability score: {n_with_score:,}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
