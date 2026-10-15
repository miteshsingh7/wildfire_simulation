"""Time-dependent Dijkstra (FIFO) evacuation router with safety margin guarantees.

Implements:
1. Mode (a): Fastest route with safety margin >= B minutes.
2. Mode (b): If none exists (or requested), maximum-bottleneck-margin route via
   monotone binary search over feasible buffer thresholds.
3. Strict FIFO traversal time integration where exit_time = entry_time + dt_min,
   and edge safety margin = edge_fire_arrival_q10 - exit_time.
"""

import heapq
from collections.abc import Sequence

import numpy as np
import shapely.geometry

from wf052.routing.contracts import (
    EvacuationRoute,
    RouteEdge,
    RouteStatus,
    RoutingMode,
)
from wf052.routing.graph import RoadGraph


def _run_constrained_dijkstra(
    road_graph: RoadGraph,
    origin_node: int,
    destination_nodes: set[int],
    t0: float,
    buffer_minutes: float,
    edge_fire_arrivals: np.ndarray,
) -> tuple[int | None, dict[int, float], dict[int, tuple[int, int]]]:
    """Single-source FIFO Dijkstra constrained by edge_exit_time + buffer <= edge_fire_arrival.

    Returns:
        (best_dest, earliest_arrival, parent_map)
        best_dest is None if no destination is reachable under the margin constraint.
    """
    # Priority queue stores (arrival_time_at_u, u)
    pq: list[tuple[float, int]] = [(t0, origin_node)]
    earliest_arrival: dict[int, float] = {origin_node: t0}
    parent: dict[int, tuple[int, int]] = {}  # v -> (u, edge_idx)

    best_dest: int | None = None

    while pq:
        t_u, u = heapq.heappop(pq)

        if t_u > earliest_arrival.get(u, float("inf")):
            continue

        if u in destination_nodes:
            best_dest = u
            break

        for v, _k, dt_min, edge_idx in road_graph.adj.get(u, []):
            t_exit = t_u + dt_min
            fire_arr = float(edge_fire_arrivals[edge_idx])

            # FIFO Safety Condition: exit_time + buffer <= edge fire arrival
            if t_exit + buffer_minutes <= fire_arr:
                if t_exit < earliest_arrival.get(v, float("inf")):
                    earliest_arrival[v] = t_exit
                    parent[v] = (u, edge_idx)
                    heapq.heappush(pq, (t_exit, v))

    return best_dest, earliest_arrival, parent


def _reconstruct_route(
    road_graph: RoadGraph,
    origin_node: int,
    dest_node: int,
    t0: float,
    parent: dict[int, tuple[int, int]],
    edge_fire_arrivals: np.ndarray,
    status: RouteStatus,
    mode_used: RoutingMode,
) -> EvacuationRoute:
    """Reconstruct an EvacuationRoute from the Dijkstra parent trace."""
    # Trace backward from dest_node to origin_node
    curr = dest_node
    edge_indices_rev: list[int] = []
    node_ids_rev: list[int] = [curr]

    while curr != origin_node:
        p_node, e_idx = parent[curr]
        edge_indices_rev.append(e_idx)
        curr = p_node
        node_ids_rev.append(curr)

    edge_indices = list(reversed(edge_indices_rev))
    node_ids = list(reversed(node_ids_rev))

    # Re-integrate forward to build RouteEdge records
    route_edges: list[RouteEdge] = []
    curr_t = t0
    min_margin = float("inf")
    bottleneck_edge: tuple[int, int, int] | None = None
    geoms = []

    for e_idx in edge_indices:
        ed = road_graph.edges[e_idx]
        entry_t = curr_t
        exit_t = entry_t + ed.traversal_time_min
        fire_arr = float(edge_fire_arrivals[e_idx])
        margin = fire_arr - exit_t

        if margin < min_margin:
            min_margin = margin
            bottleneck_edge = (ed.u, ed.v, ed.key)

        route_edges.append(
            RouteEdge(
                u=ed.u,
                v=ed.v,
                key=ed.key,
                length_m=ed.length_m,
                speed_mps=ed.speed_mps,
                traversal_time_min=ed.traversal_time_min,
                entry_time_min=entry_t,
                exit_time_min=exit_t,
                fire_arrival_min=fire_arr,
                margin_min=margin,
            )
        )
        curr_t = exit_t
        geoms.append(ed.geometry)

    total_travel_time = curr_t - t0
    geom_wkt = None
    if geoms:
        # Combine into MultiLineString or LineString
        try:
            geom_wkt = shapely.geometry.MultiLineString(geoms).wkt
        except Exception:
            geom_wkt = None

    return EvacuationRoute(
        origin_id=origin_node,
        destination_id=dest_node,
        departure_time_min=float(t0),
        arrival_time_min=float(curr_t),
        total_travel_time_min=float(total_travel_time),
        min_margin_min=float(min_margin),
        bottleneck_edge=bottleneck_edge,
        edges=tuple(route_edges),
        node_ids=tuple(node_ids),
        status=status,
        mode_used=mode_used,
        geometry_wkt=geom_wkt,
    )


class TimeDependentRouter:
    """Time-dependent wildland evacuation router satisfying the ArrivalField contract."""

    def __init__(self, road_graph: RoadGraph) -> None:
        self.road_graph = road_graph

    def find_route(
        self,
        origin_node: int,
        destination_nodes: Sequence[int] | set[int],
        t0: float,
        buffer_minutes: float = 15.0,
        edge_fire_arrivals: np.ndarray | None = None,
        mode: RoutingMode = RoutingMode.FASTEST_SAFE,
    ) -> EvacuationRoute | None:
        """Find an evacuation route from origin to any destination.

        If mode is FASTEST_SAFE and a route exists with margin >= buffer_minutes,
        returns the fastest safe route.
        If no such route exists, falls back to MAX_MIN_MARGIN (Mode b) to return
        the route that maximizes the minimum margin.
        If the graph is disconnected from origin to all destinations, returns None.
        """
        dest_set = set(destination_nodes)
        if origin_node not in self.road_graph.nodes:
            raise ValueError(f"Origin node {origin_node} not in road graph.")
        if not dest_set.intersection(self.road_graph.nodes):
            raise ValueError("None of the destination nodes exist in road graph.")

        if edge_fire_arrivals is None:
            edge_fire_arrivals = np.full(len(self.road_graph.edges), np.inf, dtype=np.float64)

        # 1. Mode (a): Try fastest route with margin >= buffer_minutes
        if mode == RoutingMode.FASTEST_SAFE:
            best_dest, _arr, parent = _run_constrained_dijkstra(
                road_graph=self.road_graph,
                origin_node=origin_node,
                destination_nodes=dest_set,
                t0=t0,
                buffer_minutes=buffer_minutes,
                edge_fire_arrivals=edge_fire_arrivals,
            )
            if best_dest is not None:
                return _reconstruct_route(
                    road_graph=self.road_graph,
                    origin_node=origin_node,
                    dest_node=best_dest,
                    t0=t0,
                    parent=parent,
                    edge_fire_arrivals=edge_fire_arrivals,
                    status=RouteStatus.SAFE,
                    mode_used=RoutingMode.FASTEST_SAFE,
                )

        # 2. Mode (b): Maximize minimum safety margin via monotone binary search
        # First check physical reachability ignoring fire constraints (buffer = -inf)
        unconstrained_dest, _arr_unc, unconstrained_parent = _run_constrained_dijkstra(
            road_graph=self.road_graph,
            origin_node=origin_node,
            destination_nodes=dest_set,
            t0=t0,
            buffer_minutes=-1e9,
            edge_fire_arrivals=edge_fire_arrivals,
        )
        if unconstrained_dest is None:
            # Physically disconnected graph
            return None

        # Binary search over achievable safety margin M in [M_low, M_high]
        m_low = -1440.0  # -24 hours
        m_high = float(buffer_minutes)

        best_found_dest: int | None = unconstrained_dest
        best_found_parent = unconstrained_parent

        # 16 iterations gives precision < 0.05 minutes
        for _ in range(16):
            m_mid = 0.5 * (m_low + m_high)
            test_dest, _test_arr, test_parent = _run_constrained_dijkstra(
                road_graph=self.road_graph,
                origin_node=origin_node,
                destination_nodes=dest_set,
                t0=t0,
                buffer_minutes=m_mid,
                edge_fire_arrivals=edge_fire_arrivals,
            )
            if test_dest is not None:
                m_low = m_mid
                best_found_dest = test_dest
                best_found_parent = test_parent
            else:
                m_high = m_mid

        assert best_found_dest is not None
        return _reconstruct_route(
            road_graph=self.road_graph,
            origin_node=origin_node,
            dest_node=best_found_dest,
            t0=t0,
            parent=best_found_parent,
            edge_fire_arrivals=edge_fire_arrivals,
            status=RouteStatus.BEST_EFFORT,
            mode_used=RoutingMode.MAX_MIN_MARGIN,
        )
