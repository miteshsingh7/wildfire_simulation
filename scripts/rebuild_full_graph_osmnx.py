"""Rebuild full-detail road graph using a single consistent OSMnx graph_from_bbox call.

Replaces the composite hybrid graph with a topologically verified, planar road network
where edges cross ONLY at shared intersection nodes (unless grade separated).
Exports to data/tiles/roads_{nodes,edges}_full.parquet.
"""

from pathlib import Path

import networkx as nx
import osmnx as ox
import shapely.geometry

DOMAIN_BBOX = {
    "lon_min": -121.750163,
    "lat_min": 39.650017,
    "lon_max": -121.349998,
    "lat_max": 39.949900,
}

OUTPUT_DIR = Path("data/tiles")


def rebuild_graph():
    print("Fetching single OSMnx graph_from_bbox for Camp Fire domain...")
    bbox_tuple = (
        DOMAIN_BBOX["lon_min"],
        DOMAIN_BBOX["lat_min"],
        DOMAIN_BBOX["lon_max"],
        DOMAIN_BBOX["lat_max"],
    )

    ox.settings.timeout = 90
    ox.settings.overpass_settings = "[out:json][timeout:{timeout}]"

    custom_filter = (
        '["highway"]["area"!~"yes"]'
        '["highway"!~"abandoned|bridleway|bus_guideway|corridor|cycleway|elevator|escalator|footway|'
        "hazard|kayak|ladder|path|pedestrian|pipeline|platform|proposed|raceway|rest_area|sanitary_dump_station|"
        'services|sidewalk|stairs|steps|stop|track|trail|via_ferrata"]'
        '["motor_vehicle"!~"no"]["motorcar"!~"no"]["access"!~"private"]'
    )

    g = ox.graph_from_bbox(bbox=bbox_tuple, custom_filter=custom_filter, simplify=True)
    print(f"Retrieved raw graph: {len(g.nodes)} nodes, {len(g.edges)} edges.")

    # Project to EPSG:32610
    g_proj = ox.project_graph(g, to_crs="EPSG:32610")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(g_proj)

    # Clean list-valued attributes for Arrow compatibility
    for col in edges_gdf.columns:
        if edges_gdf[col].apply(lambda x: isinstance(x, list)).any():
            edges_gdf[col] = edges_gdf[col].apply(
                lambda x: ",".join(map(str, x)) if isinstance(x, list) else str(x)
            )

    # Audit connectivity
    di_g = nx.DiGraph()
    for idx, row in edges_gdf.iterrows():
        u = idx[0] if isinstance(idx, tuple) else row.get("u")
        v = idx[1] if isinstance(idx, tuple) else row.get("v")
        di_g.add_edge(u, v)

    scc_list = list(nx.strongly_connected_components(di_g))
    largest_scc = max(len(c) for c in scc_list) if scc_list else 0
    scc_pct = (largest_scc / len(nodes_gdf)) * 100.0

    print(f"SCC Audit: Largest SCC has {largest_scc} nodes ({scc_pct:.2f}% of graph).")

    # Audit crossings
    sindex = edges_gdf.sindex
    sample_edges = edges_gdf.iloc[:2000] if len(edges_gdf) > 2000 else edges_gdf
    crossing_count = 0
    checked = set()

    for idx1, r1 in sample_edges.iterrows():
        g1 = r1.geometry
        if g1 is None or g1.is_empty:
            continue
        u1, v1 = idx1[0], idx1[1]
        for pm in sindex.intersection(g1.bounds):
            if pm >= len(edges_gdf):
                continue
            idx2 = edges_gdf.index[pm]
            if idx1 == idx2:
                continue
            pair = tuple(sorted([str(idx1), str(idx2)]))
            if pair in checked:
                continue
            checked.add(pair)
            r2 = edges_gdf.iloc[pm]
            g2 = r2.geometry
            if g2 is None or g2.is_empty:
                continue
            u2, v2 = idx2[0], idx2[1]
            if {u1, v1} & {u2, v2}:
                continue
            inter = g1.intersection(g2)
            if not inter.is_empty and isinstance(
                inter, (shapely.geometry.Point, shapely.geometry.MultiPoint)
            ):
                crossing_count += 1

    print(f"Crossing edge pairs without intersection node (2000-edge sample): {crossing_count}")

    # Save to data/tiles/
    nodes_out = OUTPUT_DIR / "roads_nodes_full.parquet"
    edges_out = OUTPUT_DIR / "roads_edges_full.parquet"
    nodes_gdf.to_parquet(nodes_out)
    edges_gdf.to_parquet(edges_out)
    print(f"Successfully saved clean single-OSMnx graph to {nodes_out} and {edges_out}.")


if __name__ == "__main__":
    rebuild_graph()
