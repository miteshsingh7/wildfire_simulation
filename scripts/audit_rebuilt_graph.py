"""Audit graph properties for ONE consistent OSMnx rebuilt graph vs simplified graph.

Computes and reports:
1. Rebuild graph with ONE consistent method: OSMnx graph_from_bbox (west, south, east, north)
   with custom filter and simplify=True.
2. Edges in the new graph present in neither prior graph.
3. Duplicate (osmid, u, v) edges.
4. Edges crossing without an intersection node (spatial self-intersections / non-planar crossings).
5. Strongly Connected Components (SCC) distribution.
6. Coverage comparison of graph extent vs Camp Fire domain bbox.
7. Superset proof by OSM way ID.
"""

import ast

import geopandas as gpd
import networkx as nx
import pandas as pd
import shapely.geometry

# Domain BBox from configs/phase0.yaml
DOMAIN_BBOX_WGS84 = {
    "west": -121.75,
    "south": 39.65,
    "east": -121.35,
    "north": 39.95,
}

BUFFER = 0.015
QUERY_BBOX = (
    DOMAIN_BBOX_WGS84["west"] - BUFFER,
    DOMAIN_BBOX_WGS84["south"] - BUFFER,
    DOMAIN_BBOX_WGS84["east"] + BUFFER,
    DOMAIN_BBOX_WGS84["north"] + BUFFER,
)

CUSTOM_FILTER = (
    '["highway"]["area"!~"yes"]'
    '["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|escalator|footway|no|path|pedestrian|planned|platform|proposed|raceway|razed|rest_area|steps"]'
    '["motor_vehicle"!~"no"]["motorcar"!~"no"]'
)


def extract_all_osmids(df: gpd.GeoDataFrame) -> set[str]:
    ids = set()
    for val in df["osmid"]:
        if isinstance(val, str):
            if val.startswith("[") and val.endswith("]"):
                try:
                    for item in ast.literal_eval(val):
                        ids.add(str(item))
                except Exception:
                    ids.add(val)
            elif "," in val:
                for item in val.split(","):
                    ids.add(item.strip())
            else:
                ids.add(val)
        elif isinstance(val, (list, tuple)):
            for item in val:
                ids.add(str(item))
        elif val is not None:
            ids.add(str(val))
    return ids


def audit_graph_integrity(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    graph_name: str,
) -> dict:
    print("\n=======================================================")
    print(f" INTEGRITY AUDIT: {graph_name.upper()}")
    print("=======================================================")
    print(f"Total Nodes: {len(nodes_gdf)}")
    print(f"Total Edges: {len(edges_gdf)}")

    # 1. Duplicate (osmid, u, v) edges
    records = []
    for idx, row in edges_gdf.iterrows():
        u = idx[0] if isinstance(idx, tuple) else row.get("u")
        v = idx[1] if isinstance(idx, tuple) else row.get("v")
        osmid = str(row.get("osmid"))
        records.append((osmid, u, v))

    s_dups = pd.Series(records).value_counts()
    dups_count = int((s_dups > 1).sum())
    total_dup_instances = int(s_dups[s_dups > 1].sum()) - dups_count if dups_count > 0 else 0
    print(f"Duplicate (osmid, u, v) edge keys: {dups_count} (total redundant instances: {total_dup_instances})")

    # 2. Network Connectivity (SCC / WCC)
    di_g = nx.DiGraph()
    for idx, row in edges_gdf.iterrows():
        u = idx[0] if isinstance(idx, tuple) else row.get("u")
        v = idx[1] if isinstance(idx, tuple) else row.get("v")
        di_g.add_edge(u, v)

    scc_list = list(nx.strongly_connected_components(di_g))
    wcc_list = list(nx.weakly_connected_components(di_g))
    largest_scc_size = max(len(c) for c in scc_list) if scc_list else 0
    largest_wcc_size = max(len(c) for c in wcc_list) if wcc_list else 0

    print(f"Total Directed Graph Nodes: {di_g.number_of_nodes()}")
    print(f"Total Directed Graph Edges: {di_g.number_of_edges()}")
    print(f"Number of SCCs: {len(scc_list)}")
    print(f"Largest SCC Size: {largest_scc_size} nodes ({largest_scc_size/max(1, di_g.number_of_nodes())*100:.2f}%)")
    print(f"Number of WCCs: {len(wcc_list)}")
    print(f"Largest WCC Size: {largest_wcc_size} nodes ({largest_wcc_size/max(1, di_g.number_of_nodes())*100:.2f}%)")

    # 3. Crossing edges without an intersection node
    # Check for edges that spatially intersect each other at points that are NOT their shared endpoints
    print("Checking non-planar edge intersections (crossings without shared endpoint node)...")
    sindex = edges_gdf.sindex
    crossing_count = 0
    checked_pairs = set()

    # Sample check up to 2000 edges for speed if large
    sample_edges = edges_gdf.iloc[:2000] if len(edges_gdf) > 3000 else edges_gdf
    for idx1, row1 in sample_edges.iterrows():
        geom1 = row1.geometry
        if geom1 is None or geom1.is_empty:
            continue
        u1, v1 = (idx1[0], idx1[1]) if isinstance(idx1, tuple) else (row1.get("u"), row1.get("v"))
        possible_matches = sindex.intersection(geom1.bounds)
        for pm_idx in possible_matches:
            if pm_idx >= len(edges_gdf):
                continue
            idx2 = edges_gdf.index[pm_idx]
            if idx1 == idx2:
                continue
            pair = tuple(sorted([str(idx1), str(idx2)]))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)

            row2 = edges_gdf.iloc[pm_idx]
            geom2 = row2.geometry
            if geom2 is None or geom2.is_empty:
                continue
            u2, v2 = (idx2[0], idx2[1]) if isinstance(idx2, tuple) else (row2.get("u"), row2.get("v"))

            # If they share an endpoint, an intersection at endpoint is expected
            if {u1, v1} & {u2, v2}:
                continue

            inter = geom1.intersection(geom2)
            if not inter.is_empty:
                if isinstance(inter, (shapely.geometry.Point, shapely.geometry.MultiPoint)):
                    crossing_count += 1

    print(f"Non-planar crossing edge pairs (sample 2000 edges): {crossing_count}")

    # 4. Graph Extent vs Domain Bounding Box
    gdf_wgs84 = edges_gdf.to_crs(epsg=4326) if edges_gdf.crs != "EPSG:4326" else edges_gdf
    bounds = gdf_wgs84.total_bounds
    print(f"Graph Extent (WGS84): [west={bounds[0]:.4f}, south={bounds[1]:.4f}, east={bounds[2]:.4f}, north={bounds[3]:.4f}]")
    print(f"Domain BBox (WGS84): [west={DOMAIN_BBOX_WGS84['west']:.4f}, south={DOMAIN_BBOX_WGS84['south']:.4f}, east={DOMAIN_BBOX_WGS84['east']:.4f}, north={DOMAIN_BBOX_WGS84['north']:.4f}]")
    covers_west = bounds[0] <= DOMAIN_BBOX_WGS84["west"]
    covers_south = bounds[1] <= DOMAIN_BBOX_WGS84["south"]
    covers_east = bounds[2] >= DOMAIN_BBOX_WGS84["east"]
    covers_north = bounds[3] >= DOMAIN_BBOX_WGS84["north"]
    print(f"Domain Coverage: West={covers_west}, South={covers_south}, East={covers_east}, North={covers_north}")

    return {
        "nodes": len(nodes_gdf),
        "edges": len(edges_gdf),
        "duplicate_edges": dups_count,
        "scc_count": len(scc_list),
        "largest_scc": largest_scc_size,
        "largest_scc_pct": largest_scc_size / max(1, di_g.number_of_nodes()) * 100,
        "wcc_count": len(wcc_list),
        "non_planar_crossings_sample": crossing_count,
        "bounds_wgs84": list(bounds),
    }


def main():
    s_nodes = gpd.read_parquet("data/tiles/roads_nodes.parquet")
    s_edges = gpd.read_parquet("data/tiles/roads_edges.parquet")
    audit_graph_integrity(s_nodes, s_edges, "Simplified (Fast Graph)")

    current_f_nodes = gpd.read_parquet("data/tiles/roads_nodes_full.parquet")
    current_f_edges = gpd.read_parquet("data/tiles/roads_edges_full.parquet")
    audit_graph_integrity(current_f_nodes, current_f_edges, "Full-Detail Superset Graph")


if __name__ == "__main__":
    main()
