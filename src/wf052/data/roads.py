"""Roads network ingestion and GeoParquet export via OSMnx.

Fetches the driveable road network for the domain bounding box.
Attempts an Overpass historical snapshot at 2018-11-01; if unavailable or
timing out, falls back to current OSM data with an explicit audit flag.
Projects graph to UTM 10N (EPSG:32610) and persists edges and nodes as GeoParquet.
"""

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox
import yaml

logger = logging.getLogger(__name__)


def fetch_road_graph(
    bbox_wgs84: dict[str, float],
    historical_date: str = "2018-11-01",
    network_type: str = "drive",
    timeout: int = 25,
) -> tuple[nx.MultiDiGraph, bool]:
    """Retrieve road graph via OSMnx.

    Returns:
        tuple of (graph, is_historical_snapshot)
    """
    # OSMnx 2.x bbox format is (left, bottom, right, top)
    bbox_tuple = (
        bbox_wgs84["lon_min"],
        bbox_wgs84["lat_min"],
        bbox_wgs84["lon_max"],
        bbox_wgs84["lat_max"],
    )

    ox.settings.timeout = timeout
    is_historical = False

    # 1. Attempt historical snapshot
    try:
        ox.settings.overpass_settings = (
            f'[out:json][timeout:{{timeout}}][date:"{historical_date}T00:00:00Z"]'
        )
        logger.info("Attempting Overpass historical snapshot for %s...", historical_date)
        graph = ox.graph_from_bbox(bbox=bbox_tuple, network_type=network_type)
        is_historical = True
        logger.info("Historical snapshot succeeded (%d nodes)", len(graph.nodes))
    except Exception as exc:
        logger.warning(
            "Historical Overpass query failed or timed out (%s). Falling back to current OSM.", exc
        )
        ox.settings.overpass_settings = "[out:json][timeout:{timeout}]"
        graph = ox.graph_from_bbox(bbox=bbox_tuple, network_type=network_type)
        is_historical = False
        logger.info("Current OSM query succeeded (%d nodes)", len(graph.nodes))

    return graph, is_historical


def project_and_export_graph(
    graph: nx.MultiDiGraph,
    tiles_dir: Path,
    target_crs: str = "EPSG:32610",
) -> tuple[Path, Path]:
    """Project graph to target CRS and write nodes/edges to GeoParquet."""
    graph_proj = ox.project_graph(graph, to_crs=target_crs)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph_proj)

    tiles_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = tiles_dir / "roads_nodes.parquet"
    edges_path = tiles_dir / "roads_edges.parquet"

    # Clean list-valued attributes in edges for Arrow compatibility
    for col in gdf_edges.columns:
        if gdf_edges[col].apply(lambda x: isinstance(x, list)).any():
            gdf_edges[col] = gdf_edges[col].apply(
                lambda x: ",".join(map(str, x)) if isinstance(x, list) else str(x)
            )

    gdf_nodes.to_parquet(nodes_path)
    gdf_edges.to_parquet(edges_path)

    return nodes_path, edges_path


def ingest_roads_pipeline(config_path: str = "configs/phase0.yaml") -> dict[str, Any]:
    """Execute road network ingestion and persistence pipeline."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bbox_wgs84 = cfg["domain"]["bbox_wgs84"]
    target_crs = cfg["domain"]["crs"]
    tiles_dir = Path(cfg["data"]["tiles_dir"])
    historical_date = cfg["data"]["roads"].get("date_snapshot", "2018-11-01")
    network_type = cfg["data"]["roads"].get("network_type", "drive")

    graph, is_historical = fetch_road_graph(
        bbox_wgs84=bbox_wgs84,
        historical_date=historical_date,
        network_type=network_type,
    )

    nodes_path, edges_path = project_and_export_graph(
        graph=graph,
        tiles_dir=tiles_dir,
        target_crs=target_crs,
    )

    return {
        "nodes_path": nodes_path,
        "edges_path": edges_path,
        "is_historical": is_historical,
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
    }


if __name__ == "__main__":
    res = ingest_roads_pipeline()
    print("Roads pipeline completed.")
    print(f"Historical 2018 snapshot used: {res['is_historical']}")
    print(f"Nodes: {res['num_nodes']}, Edges: {res['num_edges']}")
    print(f"Saved: {res['nodes_path']} and {res['edges_path']}")
