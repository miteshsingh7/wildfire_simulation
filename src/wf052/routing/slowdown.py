"""Post-processing module for critical free-flow slowdown multiplier (k*).

Computes the maximum traffic slowdown factor k >= 1.0 (where edge traversal time
t_e(k) = k * t_e_free) such that the best evacuation route from an origin at departure
time t0 maintains a safety margin >= B minutes ahead of fire arrival.
"""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from wf052.routing.graph import RoadGraph
from wf052.routing.router import _run_constrained_dijkstra

logger = logging.getLogger(__name__)


def compute_critical_slowdown_k_star(
    road_graph: RoadGraph,
    origin_node: int,
    destination_nodes: Sequence[int] | set[int],
    t0: float,
    buffer_minutes: float = 15.0,
    edge_fire_arrivals: np.ndarray | None = None,
    max_k: float = 50.0,
    precision: float = 0.05,
) -> float:
    """Compute maximum free-flow slowdown multiplier k* at which a route maintains margin >= B.

    Args:
        road_graph: Road network graph.
        origin_node: Departure origin node ID.
        destination_nodes: Collection of boundary destination node IDs.
        t0: Departure time in minutes from fire ignition.
        buffer_minutes: Required safety margin threshold B in minutes.
        edge_fire_arrivals: (E,) array of fire arrival times in minutes, or None for no fire.
        max_k: Upper bound search multiplier (default 50.0x free-flow time).
        precision: Binary search stopping precision on k.

    Returns:
        k_star: Maximum slowdown multiplier (float). If even free flow (k=1.0) fails
        to achieve margin >= buffer_minutes, returns 0.0. If safe up to max_k, returns max_k.
    """
    dest_set = set(destination_nodes)
    if origin_node not in road_graph.nodes:
        raise ValueError(f"Origin node {origin_node} not in road graph.")
    if not dest_set.intersection(road_graph.nodes):
        raise ValueError("None of destination nodes exist in road graph.")

    num_edges = len(road_graph.edges)
    if edge_fire_arrivals is None:
        edge_fire_arrivals = np.full(num_edges, np.inf, dtype=np.float64)

    # Base free-flow traversal times
    base_times = np.array([e.traversal_time_min for e in road_graph.edges], dtype=np.float64)

    # 1. Test k = 1.0 (baseline free flow)
    best_dest, _, _ = _run_constrained_dijkstra(
        road_graph=road_graph,
        origin_node=origin_node,
        destination_nodes=dest_set,
        t0=t0,
        buffer_minutes=buffer_minutes,
        edge_fire_arrivals=edge_fire_arrivals,
        dynamic_edge_traversal_times=base_times,
    )
    if best_dest is None:
        # Cannot maintain safety buffer even under uncongested free flow
        return 0.0

    # 2. Test k = max_k
    times_max = base_times * max_k
    best_dest_max, _, _ = _run_constrained_dijkstra(
        road_graph=road_graph,
        origin_node=origin_node,
        destination_nodes=dest_set,
        t0=t0,
        buffer_minutes=buffer_minutes,
        edge_fire_arrivals=edge_fire_arrivals,
        dynamic_edge_traversal_times=times_max,
    )
    if best_dest_max is not None:
        # Fire is far enough that even max_k slowdown is completely safe
        return float(max_k)

    # 3. Monotone binary search for critical multiplier k*
    k_low = 1.0
    k_high = float(max_k)

    while (k_high - k_low) > precision:
        k_mid = 0.5 * (k_low + k_high)
        test_times = base_times * k_mid
        dest_test, _, _ = _run_constrained_dijkstra(
            road_graph=road_graph,
            origin_node=origin_node,
            destination_nodes=dest_set,
            t0=t0,
            buffer_minutes=buffer_minutes,
            edge_fire_arrivals=edge_fire_arrivals,
            dynamic_edge_traversal_times=test_times,
        )
        if dest_test is not None:
            # Feasible with margin >= B
            k_low = k_mid
        else:
            k_high = k_mid

    return float(np.round(k_low, 2))


def compute_k_star_grid(
    road_graph: RoadGraph,
    origins: dict[str, int],
    destination_nodes: Sequence[int] | set[int],
    departure_times_min: list[float],
    buffer_minutes: float = 15.0,
    edge_fire_arrivals: np.ndarray | None = None,
    max_k: float = 50.0,
) -> pd.DataFrame:
    """Compute k* grid across multiple origin zones and departure times."""
    dest_set = set(destination_nodes)
    rows: list[dict[str, Any]] = []

    for orig_name, orig_node in origins.items():
        for t0 in departure_times_min:
            k_val = compute_critical_slowdown_k_star(
                road_graph=road_graph,
                origin_node=orig_node,
                destination_nodes=dest_set,
                t0=t0,
                buffer_minutes=buffer_minutes,
                edge_fire_arrivals=edge_fire_arrivals,
                max_k=max_k,
            )
            rows.append(
                {
                    "origin_name": orig_name,
                    "origin_node": orig_node,
                    "departure_time_min": t0,
                    "buffer_minutes": buffer_minutes,
                    "k_star": k_val,
                }
            )

    return pd.DataFrame(rows)
