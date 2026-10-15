"""En-route vehicle position representation and dynamic diversion re-planning.

Accepts real-time vehicle positions (edge u->v + offset in meters along the edge)
and re-routes from the vehicle's projected arrival at the downstream node when
a new ArrivalField arrives or an edge closes.
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from wf052.routing.contracts import (
    EvacuationRoute,
    RouteEdge,
    RouteStatus,
    RoutingMode,
    validate_arrival_field,
)
from wf052.routing.graph import RoadGraph
from wf052.routing.router import TimeDependentRouter


@dataclass(frozen=True)
class VehiclePosition:
    """Current spatio-temporal position of a vehicle in transit."""

    vehicle_id: str
    edge_u: int
    edge_v: int
    edge_key: int
    offset_m: float  # Distance along edge from u towards v
    current_time_min: float


def replan_en_route_vehicle(
    vehicle: VehiclePosition,
    arrival_field: xr.Dataset,
    road_graph: RoadGraph,
    destination_node_ids: set[int],
    buffer_minutes: float = 15.0,
    closed_edges: set[tuple[int, int, int]] | None = None,
    mode: RoutingMode = RoutingMode.FASTEST_SAFE,
) -> EvacuationRoute | None:
    """Re-plan an evacuation route for a vehicle in transit.

    1. Identifies remaining travel on the current edge to downstream node v.
    2. Re-routes from node v starting at arrival time t_v against the new ArrivalField.
    3. Prepends the remaining current-edge traversal segment to produce a complete route.
    """
    validate_arrival_field(arrival_field)

    # 1. Locate current edge in road graph
    current_edge_rec = None
    for cand in road_graph.edges:
        if cand.u == vehicle.edge_u and cand.v == vehicle.edge_v and cand.key == vehicle.edge_key:
            current_edge_rec = cand
            break

    if current_edge_rec is None:
        raise ValueError(
            f"Vehicle {vehicle.vehicle_id} is on edge ({vehicle.edge_u}, {vehicle.edge_v}, {vehicle.edge_key}) "
            f"which does not exist in road graph."
        )

    # 2. Compute remaining traversal on current edge to downstream node v
    rem_dist_m = max(0.0, current_edge_rec.length_m - vehicle.offset_m)
    rem_dt_min = (rem_dist_m / current_edge_rec.speed_mps) / 60.0
    t_v = vehicle.current_time_min + rem_dt_min

    # Check fire arrival on current edge
    edge_fire_arrivals = road_graph.compute_edge_fire_arrivals(arrival_field)
    current_fire_arr = float(edge_fire_arrivals[current_edge_rec.edge_idx])
    current_margin = current_fire_arr - t_v

    # Zero out capacity / mark infinity for closed edges
    if closed_edges:
        edge_fire_arrivals = edge_fire_arrivals.copy()
        for cand in road_graph.edges:
            if (cand.u, cand.v, cand.key) in closed_edges:
                edge_fire_arrivals[cand.edge_idx] = -np.inf  # Impassable / burned

    # 3. If downstream node v is already a destination, exit immediately
    if current_edge_rec.v in destination_node_ids:
        cur_edge_seg = RouteEdge(
            u=current_edge_rec.u,
            v=current_edge_rec.v,
            key=current_edge_rec.key,
            length_m=rem_dist_m,
            speed_mps=current_edge_rec.speed_mps,
            traversal_time_min=rem_dt_min,
            entry_time_min=vehicle.current_time_min,
            exit_time_min=t_v,
            fire_arrival_min=current_fire_arr,
            margin_min=current_margin,
        )
        return EvacuationRoute(
            origin_id=current_edge_rec.u,
            destination_id=current_edge_rec.v,
            departure_time_min=vehicle.current_time_min,
            arrival_time_min=t_v,
            total_travel_time_min=rem_dt_min,
            min_margin_min=current_margin,
            bottleneck_edge=(current_edge_rec.u, current_edge_rec.v, current_edge_rec.key),
            edges=(cur_edge_seg,),
            node_ids=(current_edge_rec.u, current_edge_rec.v),
            status=RouteStatus.SAFE
            if current_margin >= buffer_minutes
            else RouteStatus.BEST_EFFORT,
            mode_used=mode,
        )

    # 4. Route from downstream node v to destination set
    router = TimeDependentRouter(road_graph=road_graph)
    downstream_route = router.find_route(
        origin_node=current_edge_rec.v,
        destination_nodes=destination_node_ids,
        t0=t_v,
        buffer_minutes=buffer_minutes,
        edge_fire_arrivals=edge_fire_arrivals,
        mode=mode,
    )

    if downstream_route is None:
        return None

    # 5. Prepend remaining segment of current edge to downstream route
    cur_edge_seg = RouteEdge(
        u=current_edge_rec.u,
        v=current_edge_rec.v,
        key=current_edge_rec.key,
        length_m=rem_dist_m,
        speed_mps=current_edge_rec.speed_mps,
        traversal_time_min=rem_dt_min,
        entry_time_min=vehicle.current_time_min,
        exit_time_min=t_v,
        fire_arrival_min=current_fire_arr,
        margin_min=current_margin,
    )

    combined_edges = (cur_edge_seg,) + downstream_route.edges
    combined_nodes = (current_edge_rec.u,) + downstream_route.node_ids
    combined_min_margin = min(current_margin, downstream_route.min_margin_min)
    total_travel_time = downstream_route.arrival_time_min - vehicle.current_time_min

    status = (
        RouteStatus.SAFE
        if (combined_min_margin >= buffer_minutes and downstream_route.status == RouteStatus.SAFE)
        else RouteStatus.BEST_EFFORT
    )

    return EvacuationRoute(
        origin_id=current_edge_rec.u,
        destination_id=downstream_route.destination_id,
        departure_time_min=float(vehicle.current_time_min),
        arrival_time_min=float(downstream_route.arrival_time_min),
        total_travel_time_min=float(total_travel_time),
        min_margin_min=float(combined_min_margin),
        bottleneck_edge=(
            (current_edge_rec.u, current_edge_rec.v, current_edge_rec.key)
            if current_margin < downstream_route.min_margin_min
            else downstream_route.bottleneck_edge
        ),
        edges=combined_edges,
        node_ids=combined_nodes,
        status=status,
        mode_used=downstream_route.mode_used,
    )
