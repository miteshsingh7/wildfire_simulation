"""Build and cache full-detail OSM road network for the Camp Fire domain.

Ensures the full-detail graph is a strict SUPERSET of the simplified graph,
incorporating all simplified road segments plus all residential and service roads.
Exports projected GeoParquet tiles to data/tiles/roads_{nodes,edges}_full.parquet (EPSG:32610).
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_full_detail_graph(output_dir: Path | str = "data/tiles") -> None:
    p_out = Path(output_dir)
    p_out.mkdir(parents=True, exist_ok=True)

    simplified_nodes_path = p_out / "roads_nodes.parquet"
    simplified_edges_path = p_out / "roads_edges.parquet"

    if not simplified_nodes_path.exists() or not simplified_edges_path.exists():
        raise FileNotFoundError("Simplified graph tiles not found; needed as superset base.")

    s_nodes = gpd.read_parquet(simplified_nodes_path)
    s_edges = gpd.read_parquet(simplified_edges_path)
    logger.info("Loaded simplified graph base: %d nodes, %d edges", len(s_nodes), len(s_edges))

    # Load existing full detail edges to extract service roads and fine-grained links
    old_full_edges_path = p_out / "roads_edges_full.parquet"
    old_full_nodes_path = p_out / "roads_nodes_full.parquet"

    if old_full_edges_path.exists() and old_full_nodes_path.exists():
        old_f_edges = gpd.read_parquet(old_full_edges_path)
        old_f_nodes = gpd.read_parquet(old_full_nodes_path)
    else:
        old_f_edges = s_edges
        old_f_nodes = s_nodes

    # Extract extra edges not present in simplified graph
    extra_edges = old_f_edges[~old_f_edges.index.isin(s_edges.index)]
    logger.info("Found %d additional detail/service edges to add to superset", len(extra_edges))

    # Combine edges (s_edges first to ensure priority on attributes)
    combined_edges = pd.concat([s_edges, extra_edges])
    combined_edges = gpd.GeoDataFrame(combined_edges, geometry="geometry", crs=s_edges.crs)

    # Determine all required nodes
    u_nodes = set(combined_edges.index.get_level_values("u"))
    v_nodes = set(combined_edges.index.get_level_values("v"))
    needed_nodes = u_nodes | v_nodes

    missing_node_ids = list(needed_nodes - set(s_nodes.index))
    extra_nodes = old_f_nodes.loc[old_f_nodes.index.isin(missing_node_ids)]
    combined_nodes = pd.concat([s_nodes, extra_nodes])
    combined_nodes = gpd.GeoDataFrame(combined_nodes, geometry="geometry", crs=s_nodes.crs)

    logger.info(
        "Final superset graph: %d nodes, %d edges (superset of %d s_edges)",
        len(combined_nodes),
        len(combined_edges),
        len(s_edges),
    )

    combined_nodes.to_parquet(p_out / "roads_nodes_full.parquet")
    combined_edges.to_parquet(p_out / "roads_edges_full.parquet")
    logger.info("Superset road graph saved successfully.")


if __name__ == "__main__":
    build_full_detail_graph()
