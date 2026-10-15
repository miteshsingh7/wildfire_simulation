"""Build and cache full-detail OSM road network for the Camp Fire domain.

Fetches network including residential and service roads via OSMnx and exports
projected GeoParquet tiles to data/tiles/roads_{nodes,edges}_full.parquet (EPSG:32610).
"""

import logging
from pathlib import Path

import osmnx as ox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bounding box for the Camp Fire analysis domain (EPSG:4326 lat/lon)
# Matches the extent of the 60m elevation / fuel grid
NORTH = 39.95
SOUTH = 39.65
EAST = -121.35
WEST = -121.75

CUSTOM_FILTER = (
    '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service"]'
)


def build_full_detail_graph(output_dir: Path | str = "data/tiles") -> None:
    p_out = Path(output_dir)
    p_out.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading road network within bbox: [%.2f, %.2f, %.2f, %.2f]", SOUTH, NORTH, WEST, EAST
    )
    graph = ox.graph_from_bbox(
        bbox=(NORTH, SOUTH, EAST, WEST),
        custom_filter=CUSTOM_FILTER,
        simplify=False,
        retain_all=True,
    )

    logger.info("Projecting road graph to EPSG:32610 (UTM Zone 10N)...")
    graph_proj = ox.project_graph(graph, to_crs="EPSG:32610")

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph_proj)

    logger.info("Saving %d nodes and %d edges to %s...", len(nodes_gdf), len(edges_gdf), p_out)
    nodes_gdf.to_parquet(p_out / "roads_nodes_full.parquet")
    edges_gdf.to_parquet(p_out / "roads_edges_full.parquet")
    logger.info("Full detail road graph built successfully.")


if __name__ == "__main__":
    build_full_detail_graph()
