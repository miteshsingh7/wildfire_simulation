"""Audit the road network (simplified vs full-detail) and print detailed statistics.

Covers:
1. Highway-type filter used
2. Edge/node counts by type
3. Largest strongly connected component (SCC) and weakly connected component (WCC)
4. One-way handling
5. Speed source (OSM maxspeed vs imputed by highway type)
6. Lane counts (explicit vs imputed)
"""

import geopandas as gpd
import networkx as nx


def audit_graph(nodes_path: str, edges_path: str, mode_name: str) -> None:
    print("\n==========================================")
    print(f" GRAPH AUDIT: {mode_name.upper()}")
    print("==========================================")
    nodes = gpd.read_parquet(nodes_path)
    edges = gpd.read_parquet(edges_path)

    print(f"Nodes count: {len(nodes)}")
    print(f"Edges count: {len(edges)}")

    # Edge counts by highway type
    print("\n--- Edge Counts by Highway Type ---")
    hw_counts = edges["highway"].value_counts(dropna=False)
    for hw, count in hw_counts.items():
        print(f"  {str(hw):<30}: {count:>6} ({count / len(edges) * 100:>5.2f}%)")

    # One-way handling
    print("\n--- One-Way Handling ---")
    if "oneway" in edges.columns:
        ow_counts = edges["oneway"].value_counts(dropna=False)
        for ow, count in ow_counts.items():
            print(f"  oneway={str(ow):<10}: {count:>6} ({count / len(edges) * 100:>5.2f}%)")
    else:
        print("  oneway column missing")

    # Speed source
    print("\n--- Speed Source ---")
    if "maxspeed" in edges.columns:
        has_speed = (
            edges["maxspeed"].notna()
            & (edges["maxspeed"] != "")
            & (edges["maxspeed"].astype(str) != "nan")
        )
        explicit_speed = int(has_speed.sum())
        imputed_speed = len(edges) - explicit_speed
        print(
            f"  Explicit OSM maxspeed: {explicit_speed:>6} ({explicit_speed / len(edges) * 100:>5.2f}%)"
        )
        print(
            f"  Imputed by highway type: {imputed_speed:>6} ({imputed_speed / len(edges) * 100:>5.2f}%)"
        )
    else:
        print("  All speeds imputed by highway type classification")

    # Lanes
    print("\n--- Lane Counts ---")
    if "lanes" in edges.columns:
        has_lanes = (
            edges["lanes"].notna() & (edges["lanes"] != "") & (edges["lanes"].astype(str) != "nan")
        )
        explicit_lanes = int(has_lanes.sum())
        missing_lanes = len(edges) - explicit_lanes
        print(
            f"  Explicit lanes defined: {explicit_lanes:>6} ({explicit_lanes / len(edges) * 100:>5.2f}%)"
        )
        print(
            f"  Missing (imputed by highway type): {missing_lanes:>6} ({missing_lanes / len(edges) * 100:>5.2f}%)"
        )
        lanes_counts = edges.loc[has_lanes, "lanes"].value_counts()
        for lane_val, count in lanes_counts.head(5).items():
            print(f"    lanes={str(lane_val):<10}: {count:>6}")
    else:
        print("  Lanes column missing (all imputed)")

    # Connectivity
    print("\n--- Network Connectivity (SCC / WCC) ---")
    di_graph = nx.DiGraph()
    for _, edge in edges.iterrows():
        u = (
            edge["osmid"]
            if isinstance(edge["osmid"], int)
            else edge.name[0]
            if isinstance(edge.name, tuple)
            else None
        )
        # From multiindex if (u, v, key)
        if isinstance(edge.name, tuple) and len(edge.name) >= 2:
            u, v = edge.name[0], edge.name[1]
        else:
            u, v = edge.get("u"), edge.get("v")
        if u is not None and v is not None:
            di_graph.add_edge(u, v)

    if di_graph.number_of_nodes() > 0:
        scc = list(nx.strongly_connected_components(di_graph))
        wcc = list(nx.weakly_connected_components(di_graph))
        largest_scc = max(len(c) for c in scc)
        largest_wcc = max(len(c) for c in wcc)
        print(f"  Total graph nodes: {di_graph.number_of_nodes()}")
        print(f"  Total graph edges: {di_graph.number_of_edges()}")
        print(
            f"  Number of WCCs: {len(wcc)} (Largest WCC: {largest_wcc} nodes, {largest_wcc / di_graph.number_of_nodes() * 100:.2f}%)"
        )
        print(
            f"  Number of SCCs: {len(scc)} (Largest SCC: {largest_scc} nodes, {largest_scc / di_graph.number_of_nodes() * 100:.2f}%)"
        )


if __name__ == "__main__":
    audit_graph(
        "data/tiles/roads_nodes.parquet", "data/tiles/roads_edges.parquet", "Simplified (Fast Mode)"
    )
    audit_graph(
        "data/tiles/roads_nodes_full.parquet",
        "data/tiles/roads_edges_full.parquet",
        "Full-Detail (Standard Mode)",
    )
