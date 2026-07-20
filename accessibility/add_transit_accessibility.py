"""
Adds parcel-level transit accessibility using a walking street network.

This script uses OSMnx + NetworkX to compute the shortest walking-network
(distance in meters) from each parcel to the nearest transit stop/station.

Input parcels must include a `geometry` column in WKT format.

This script is standalone since it's time intenstive.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from utils import require_existing_path, load_parcels_csv, save_parcels_csv, to_point_geometry


TRANSIT_TAGS: dict[str, object] = {
    "public_transport": ["platform", "stop_position", "station"],
    "highway": "bus_stop",
    "railway": ["station", "tram_stop", "halt", "subway_entrance"],
}


def log(message: str) -> None:
    """Print a timestamped progress message."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Add transit network accessibility to parcel records."
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
        help="Output CSV path (final parcels file).",
    )
    parser.add_argument(
        "--distance-column",
        type=str,
        default="transit_walk_dist_m",
        help="Name of output distance column in meters.",
    )
    parser.add_argument(
        "--network-buffer-m",
        type=float,
        default=1200.0,
        help="Buffer around parcel extent used to download street/transit data.",
    )
    parser.add_argument(
        "--network-type",
        type=str,
        default="walk",
        help="OSMnx network type (typically walk, bike, or drive).",
    )
    parser.add_argument(
        "--max-query-area-km2",
        type=float,
        default=150.0,
        help="Fail fast if buffered parcel extent exceeds this area in square kilometers.",
    )
    parser.add_argument(
        "--overpass-timeout-sec",
        type=int,
        default=180,
        help="OSM Overpass timeout (seconds) for downloads.",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable OSMnx HTTP cache (enabled by default).",
    )

    args = parser.parse_args()
    args.parcels_csv = require_existing_path(args.parcels_csv, "Parcels CSV")
    args.output_csv = args.output_csv.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    started = perf_counter()

    ox.settings.use_cache = not args.disable_cache
    ox.settings.timeout = args.overpass_timeout_sec

    log(f"Reading parcels: {args.parcels_csv}")
    parcels = load_parcels_csv(args.parcels_csv)
    log(f"Loaded {len(parcels):,} parcel rows.")

    valid = parcels.geometry.notna()
    if int(valid.sum()) == 0:
        raise ValueError("No valid parcel geometries found in input.")

    parcel_points = parcels.loc[valid].copy()
    parcel_points["geometry"] = parcel_points.geometry.map(to_point_geometry)

    metric_crs = parcel_points.estimate_utm_crs()
    if metric_crs is None:
        raise ValueError("Unable to infer projected CRS for parcel extent.")

    extent_metric = parcel_points.to_crs(metric_crs)
    query_metric = extent_metric.geometry.unary_union.buffer(args.network_buffer_m)
    query_area_km2 = float(query_metric.area) / 1_000_000.0
    if query_area_km2 > args.max_query_area_km2:
        raise ValueError(
            "Buffered query area is too large for reliable OSM download "
            f"({query_area_km2:.1f} km^2 > {args.max_query_area_km2:.1f} km^2). "
            "Reduce --network-buffer-m or increase --max-query-area-km2."
        )

    log(
        f"Query area: {query_area_km2:.1f} km^2 | cache={'off' if args.disable_cache else 'on'} "
        f"| timeout={args.overpass_timeout_sec}s"
    )
    query_polygon = gpd.GeoSeries([query_metric], crs=metric_crs).to_crs("EPSG:4326").iloc[0]

    t_network = perf_counter()
    log("Downloading walking network from OSM...")
    graph = ox.graph_from_polygon(query_polygon, network_type=args.network_type, simplify=True)
    log(
        f"Walking network ready in {perf_counter() - t_network:.1f}s "
        f"({graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges)."
    )

    t_features = perf_counter()
    log("Downloading transit features from OSM...")
    transit = ox.features_from_polygon(query_polygon, tags=TRANSIT_TAGS)
    log(f"Transit features downloaded in {perf_counter() - t_features:.1f}s ({len(transit):,} rows).")
    if transit.empty or "geometry" not in transit.columns:
        raise ValueError("No transit features were found in the parcel area.")

    transit_points = transit.geometry.dropna().map(to_point_geometry)
    transit_points = gpd.GeoSeries(transit_points, crs="EPSG:4326")
    if transit_points.empty:
        raise ValueError("Transit features were found, but none had usable geometries.")

    t_map = perf_counter()
    log("Mapping parcels and transit to network nodes...")
    transit_node_ids = ox.distance.nearest_nodes(
        graph,
        X=transit_points.x.to_numpy(),
        Y=transit_points.y.to_numpy(),
    )
    transit_node_ids = list(set(transit_node_ids))
    if not transit_node_ids:
        raise ValueError("No transit nodes could be mapped to the street network.")

    parcel_node_ids = ox.distance.nearest_nodes(
        graph,
        X=parcel_points.geometry.x.to_numpy(),
        Y=parcel_points.geometry.y.to_numpy(),
    )
    log(
        f"Mapped nodes in {perf_counter() - t_map:.1f}s "
        f"({len(transit_node_ids):,} transit sources, {len(parcel_node_ids):,} parcel points)."
    )

    t_dijkstra = perf_counter()
    log("Computing nearest transit walking distances...")
    distance_by_node = nx.multi_source_dijkstra_path_length(
        graph,
        sources=transit_node_ids,
        weight="length",
    )
    log(
        f"Dijkstra completed in {perf_counter() - t_dijkstra:.1f}s "
        f"({len(distance_by_node):,} reachable nodes)."
    )

    parcel_distances = [distance_by_node.get(node_id, np.nan) for node_id in parcel_node_ids]

    parcels[args.distance_column] = np.nan
    parcels.loc[valid, args.distance_column] = parcel_distances

    save_parcels_csv(parcels, args.output_csv)

    n_with_value = int(pd.notna(parcels[args.distance_column]).sum())
    log(f"Rows written: {len(parcels):,}")
    log(f"Rows with transit distance: {n_with_value:,}")
    log(f"Output: {args.output_csv}")
    log(f"Transit step total runtime: {perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
