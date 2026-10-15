"""Evacuation route re-planning engine.

Accepts a new ArrivalField, recomputes edge fire arrivals across the graph,
re-plans routes for all requested origins, independently validates all routes,
and measures re-plan latency (target < 5 seconds).
"""

import hashlib
import time
from collections.abc import Sequence
from datetime import UTC, datetime

import xarray as xr

from wf052.routing.contracts import (
    EvacuationRoute,
    ReplanResult,
    RoutingMode,
    validate_arrival_field,
)
from wf052.routing.graph import RoadGraph
from wf052.routing.router import TimeDependentRouter
from wf052.routing.validator import IndependentRouteValidator


class ReplanEngine:
    """Operational re-plan engine evaluating evacuation routes against dynamic ArrivalFields."""

    def __init__(self, road_graph: RoadGraph) -> None:
        self.road_graph = road_graph
        self.router = TimeDependentRouter(road_graph=self.road_graph)
        self.validator = IndependentRouteValidator()

    def replan(
        self,
        arrival_field: xr.Dataset,
        origins: Sequence[str | int] | None = None,
        destinations: Sequence[str | int] | None = None,
        t0: float = 0.0,
        buffer_minutes: float = 15.0,
        mode: RoutingMode = RoutingMode.FASTEST_SAFE,
        validate_routes: bool = True,
    ) -> ReplanResult:
        """Execute complete evacuation re-plan for specified origins against a new ArrivalField.

        Args:
            arrival_field: xarray Dataset strictly conforming to ArrivalField contract.
            origins: list of landmark names ('Paradise', 'Magalia', etc.) or integer node IDs.
                     Defaults to all named landmarks in road_graph.
            destinations: list of destination names ('West_Chico', etc.) or integer node IDs.
                          Defaults to all boundary highway exits in road_graph.
            t0: departure time in minutes since ignition.
            buffer_minutes: safety buffer in minutes.
            mode: RoutingMode (FASTEST_SAFE or MAX_MIN_MARGIN).
            validate_routes: if True, runs IndependentRouteValidator and asserts validity.

        Returns:
            ReplanResult containing computed routes, margins, and re-plan latency.
        """
        validate_arrival_field(arrival_field)

        t_start = time.perf_counter()

        # Compute arrival field hash for data provenance
        q10_bytes = arrival_field["arrival_q10"].values.tobytes()
        arr_hash = hashlib.sha256(q10_bytes).hexdigest()

        # 1. Update edge fire arrival times across the entire road network
        edge_fire_arrivals = self.road_graph.compute_edge_fire_arrivals(arrival_field)

        # 2. Resolve destination node IDs
        dest_node_ids: set[int] = set()
        if destinations is None:
            dest_node_ids = set(self.road_graph.boundary_destinations.values())
        else:
            for d in destinations:
                if isinstance(d, str):
                    if d in self.road_graph.boundary_destinations:
                        dest_node_ids.add(self.road_graph.boundary_destinations[d])
                    elif d in self.road_graph.named_origins:
                        dest_node_ids.add(self.road_graph.named_origins[d])
                    else:
                        raise ValueError(f"Unknown destination landmark: '{d}'")
                else:
                    dest_node_ids.add(int(d))

        # 3. Resolve origin node IDs
        origin_map: dict[str, int] = {}
        if origins is None:
            origin_map = dict(self.road_graph.named_origins)
        else:
            for o in origins:
                if isinstance(o, str):
                    if o in self.road_graph.named_origins:
                        origin_map[o] = self.road_graph.named_origins[o]
                    else:
                        raise ValueError(f"Unknown origin landmark: '{o}'")
                else:
                    origin_map[f"node_{o}"] = int(o)

        # 4. Route each origin to the best reachable destination exit
        routes: dict[str, EvacuationRoute] = {}
        for origin_name, origin_nid in origin_map.items():
            route = self.router.find_route(
                origin_node=origin_nid,
                destination_nodes=dest_node_ids,
                t0=t0,
                buffer_minutes=buffer_minutes,
                edge_fire_arrivals=edge_fire_arrivals,
                mode=mode,
            )
            if route is not None:
                if validate_routes:
                    val_res = self.validator.validate(
                        route=route,
                        arrival_field=arrival_field,
                        road_graph=self.road_graph,
                        buffer_minutes=buffer_minutes,
                    )
                    if not val_res.is_valid:
                        raise AssertionError(
                            f"Independent validation failed for origin '{origin_name}': "
                            f"{val_res.failure_reasons}"
                        )
                routes[origin_name] = route

        latency_sec = time.perf_counter() - t_start

        return ReplanResult(
            timestamp_utc=datetime.now(UTC).isoformat(),
            departure_time_min=float(t0),
            buffer_minutes=float(buffer_minutes),
            routes=routes,
            replan_latency_sec=float(latency_sec),
            arrival_field_hash=arr_hash,
            metadata={
                "num_origins": len(origin_map),
                "num_destinations": len(dest_node_ids),
                "num_edges": len(self.road_graph.edges),
                "osm_snapshot": self.road_graph.osm_snapshot_flag,
            },
        )
