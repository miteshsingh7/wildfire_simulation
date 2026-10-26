"""Simulation runners for evaluating the 4 evacuation methods in causal replay.

Methods evaluated on the same 200 block origins with departure times from zone schedules:
1. StaticMethod: Static fastest route ignoring fire and roadblocks.
2. ReactiveMethod: Re-routes around active roadblocks and visible fire observations as they appear.
3. AgentProviderA: Evacuation agent using Provider A (teacher_v1 frozen raster).
4. AgentProviderB: Evacuation agent using Provider B (observation-inflated heuristic field).
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wf052.replay.contracts import ReplayOriginResult, TruthHazard
from wf052.replay.harness import CausalReplayHarness
from wf052.replay.metrics import (
    check_route_hazard_miss_by_type,
    compute_warning_lead_time,
)
from wf052.routing.contracts import (
    EvacuationRoute,
)
from wf052.routing.demand import EvacuationDemandModel
from wf052.routing.en_route import VehiclePosition
from wf052.routing.graph import RoadGraph
from wf052.routing.router import TimeDependentRouter


@dataclass(frozen=True)
class EvacuationOrigin:
    """Sampled evacuation origin from inhabited census blocks."""

    origin_id: int
    geoid10: str
    assigned_zone: str
    centroid_utm: tuple[float, float]
    departure_time_min: float


def sample_block_origins(
    road_graph: RoadGraph,
    num_samples: int = 200,
    seed: int = 42,
    config_path: str | Path = "configs/evacuation_demand.yaml",
) -> list[EvacuationOrigin]:
    """Sample inhabited census block origins across Paradise zones.

    Returns deterministic sample of num_samples origins with departure times.
    """
    demand_model = EvacuationDemandModel(config_path=config_path, road_graph=road_graph)
    paradise_blocks = [b for b in demand_model.block_demands if b.assigned_zone.startswith("Zone_")]
    if len(paradise_blocks) < num_samples:
        raise ValueError(
            f"Requested {num_samples} blocks, but only {len(paradise_blocks)} available."
        )

    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(paradise_blocks), size=num_samples, replace=False)
    chosen_indices.sort()

    origins: list[EvacuationOrigin] = []
    for idx in chosen_indices:
        b = paradise_blocks[idx]
        dep_time = min(b.departure_schedule.keys()) if b.departure_schedule else 60.0
        origins.append(
            EvacuationOrigin(
                origin_id=b.origin_node_id,
                geoid10=b.geoid10,
                assigned_zone=b.assigned_zone,
                centroid_utm=b.centroid_utm,
                departure_time_min=float(dep_time),
            )
        )

    return origins


def _find_vehicle_position_on_route(
    route: EvacuationRoute,
    current_time_min: float,
) -> VehiclePosition | None:
    """Locate vehicle position along an EvacuationRoute at clock time current_time_min."""
    if not route.edges:
        return None
    if current_time_min <= route.departure_time_min:
        first_e = route.edges[0]
        return VehiclePosition(
            vehicle_id=f"veh_{route.origin_id}",
            edge_u=first_e.u,
            edge_v=first_e.v,
            edge_key=first_e.key,
            offset_m=0.0,
            current_time_min=current_time_min,
        )
    if current_time_min >= route.arrival_time_min:
        return None  # Vehicle has exited to safe destination

    for e in route.edges:
        if e.entry_time_min <= current_time_min <= e.exit_time_min:
            dt_elapsed = current_time_min - e.entry_time_min
            offset = min(e.length_m, dt_elapsed * 60.0 * e.speed_mps)
            return VehiclePosition(
                vehicle_id=f"veh_{route.origin_id}",
                edge_u=e.u,
                edge_v=e.v,
                edge_key=e.key,
                offset_m=offset,
                current_time_min=current_time_min,
            )

    return None


def run_static_baseline(
    origins: Sequence[EvacuationOrigin],
    road_graph: RoadGraph,
    destinations: Sequence[int] | set[int],
    truth_hazards: Sequence[TruthHazard],
) -> tuple[list[ReplayOriginResult], float]:
    """Evaluate Method (i): Static fastest route ignoring fire and roadblocks."""
    t0_start = time.perf_counter()
    router = TimeDependentRouter(road_graph=road_graph)
    dest_set = set(destinations)
    results: list[ReplayOriginResult] = []

    for orig in origins:
        route = router.find_route(
            origin_node=orig.origin_id,
            destination_nodes=dest_set,
            t0=orig.departure_time_min,
            buffer_minutes=0.0,
            edge_fire_arrivals=None,
        )

        by_type = check_route_hazard_miss_by_type(route, truth_hazards, road_graph)
        is_miss, _ = by_type["total"]
        tt = route.total_travel_time_min if route is not None else float("inf")
        margin = route.min_margin_min if route is not None else -float("inf")

        results.append(
            ReplayOriginResult(
                origin_id=orig.origin_id,
                method_name="static_fastest",
                departure_time_min=orig.departure_time_min,
                route=route,
                travel_time_min=tt,
                hazard_miss=is_miss,
                min_margin_min=margin,
                replan_count=0,
                warning_lead_time_min=None,
                flagged_before_onset=False,  # Static route cannot flag or divert anything
                hazard_misses_by_type={k: v[0] for k, v in by_type.items()},
            )
        )

    elapsed_ms = (time.perf_counter() - t0_start) * 1000.0
    return results, elapsed_ms


def run_reactive_baseline(
    origins: Sequence[EvacuationOrigin],
    road_graph: RoadGraph,
    destinations: Sequence[int] | set[int],
    truth_hazards: Sequence[TruthHazard],
    harness: CausalReplayHarness,
    step_duration_min: float = 10.0,
    horizon_min: float = 360.0,
) -> tuple[list[ReplayOriginResult], float]:
    """Evaluate Method (ii): Reactive routing around active roadblocks and visible observations."""
    router = TimeDependentRouter(road_graph=road_graph)
    dest_set = set(destinations)

    # Initial routes planned without fire
    routes: dict[int, EvacuationRoute | None] = {}
    for orig in origins:
        routes[orig.origin_id] = router.find_route(
            origin_node=orig.origin_id,
            destination_nodes=dest_set,
            t0=orig.departure_time_min,
            buffer_minutes=0.0,
            edge_fire_arrivals=None,
        )

    initial_routes = {orig.origin_id: routes[orig.origin_id] for orig in origins}
    replan_histories: dict[int, list[tuple[float, EvacuationRoute | None]]] = {
        orig.origin_id: [(0.0, routes[orig.origin_id])] for orig in origins
    }
    replan_counts: dict[int, int] = {orig.origin_id: 0 for orig in origins}

    steps = np.arange(step_duration_min, horizon_min + 1e-6, step_duration_min)
    step_latencies: list[float] = []

    for step_t in steps:
        t_step_start = time.perf_counter()
        env = harness.step(step_t)

        closed_rb = harness.get_closed_edges_for_roadblocks(env.active_roadblocks)
        closed_obs = harness.get_closed_edges_for_observations(env.visible_observations)
        closed_edges = closed_rb | closed_obs

        # Build dynamic traversal times
        dynamic_times = None
        if closed_edges:
            dynamic_times = np.array(
                [e.traversal_time_min for e in road_graph.edges], dtype=np.float64
            )
            for idx, ge in enumerate(road_graph.edges):
                if (ge.u, ge.v) in closed_edges or (ge.v, ge.u) in closed_edges:
                    dynamic_times[idx] = float("inf")

        for orig in origins:
            cur_r = routes[orig.origin_id]
            if cur_r is None:
                continue
            if cur_r.arrival_time_min <= step_t:
                continue  # Already safely arrived

            # Check if current route traverses any newly closed edge
            intersects_closure = any(
                (e.u, e.v) in closed_edges or (e.v, e.u) in closed_edges for e in cur_r.edges
            )

            if not intersects_closure:
                continue

            # Need to re-route!
            replan_counts[orig.origin_id] += 1
            if step_t <= orig.departure_time_min:
                # Pre-departure: re-route from origin
                new_r = router.find_route(
                    origin_node=orig.origin_id,
                    destination_nodes=dest_set,
                    t0=orig.departure_time_min,
                    buffer_minutes=0.0,
                    dynamic_edge_traversal_times=dynamic_times,
                )
                routes[orig.origin_id] = new_r
                replan_histories[orig.origin_id].append((step_t, new_r))
            else:
                # En-route: locate vehicle position
                pos = _find_vehicle_position_on_route(cur_r, step_t)
                if pos is not None:
                    # Reroute downstream
                    down_r = router.find_route(
                        origin_node=pos.edge_v,
                        destination_nodes=dest_set,
                        t0=step_t,
                        buffer_minutes=0.0,
                        dynamic_edge_traversal_times=dynamic_times,
                    )
                    routes[orig.origin_id] = down_r
                    replan_histories[orig.origin_id].append((step_t, down_r))
                else:
                    routes[orig.origin_id] = None
                    replan_histories[orig.origin_id].append((step_t, None))

        step_latencies.append((time.perf_counter() - t_step_start) * 1000.0)

    # Compile results
    results: list[ReplayOriginResult] = []
    for orig in origins:
        final_r = routes[orig.origin_id]
        by_type = check_route_hazard_miss_by_type(final_r, truth_hazards, road_graph)
        is_miss, _ = by_type["total"]
        lead_time, flagged_before = compute_warning_lead_time(
            initial_route=initial_routes[orig.origin_id],
            replan_history=replan_histories[orig.origin_id],
            hazards=truth_hazards,
            road_graph=road_graph,
        )
        tt = final_r.total_travel_time_min if final_r is not None else float("inf")
        margin = final_r.min_margin_min if final_r is not None else -float("inf")

        results.append(
            ReplayOriginResult(
                origin_id=orig.origin_id,
                method_name="reactive",
                departure_time_min=orig.departure_time_min,
                route=final_r,
                travel_time_min=tt,
                hazard_miss=is_miss,
                min_margin_min=margin,
                replan_count=replan_counts[orig.origin_id],
                warning_lead_time_min=lead_time,
                flagged_before_onset=flagged_before,
                hazard_misses_by_type={k: v[0] for k, v in by_type.items()},
            )
        )

    mean_step_ms = float(np.mean(step_latencies)) if step_latencies else 0.0
    return results, mean_step_ms


def run_agent_simulation(
    method_name: str,
    origins: Sequence[EvacuationOrigin],
    road_graph: RoadGraph,
    destinations: Sequence[int] | set[int],
    truth_hazards: Sequence[TruthHazard],
    harness: CausalReplayHarness,
    buffer_minutes: float = 15.0,
    step_duration_min: float = 10.0,
    horizon_min: float = 360.0,
) -> tuple[list[ReplayOriginResult], float]:
    """Evaluate Method (iii) / (iv): Evacuation agent with Field Provider A or B."""
    router = TimeDependentRouter(road_graph=road_graph)
    dest_set = set(destinations)

    # Initial step at t=0
    env0 = harness.step(0.0)
    edge_arrivals = road_graph.compute_edge_fire_arrivals(env0.arrival_field)

    routes: dict[int, EvacuationRoute | None] = {}
    for orig in origins:
        routes[orig.origin_id] = router.find_route(
            origin_node=orig.origin_id,
            destination_nodes=dest_set,
            t0=orig.departure_time_min,
            buffer_minutes=buffer_minutes,
            edge_fire_arrivals=edge_arrivals,
        )

    initial_routes = {orig.origin_id: routes[orig.origin_id] for orig in origins}
    replan_histories: dict[int, list[tuple[float, EvacuationRoute | None]]] = {
        orig.origin_id: [(0.0, routes[orig.origin_id])] for orig in origins
    }
    replan_counts: dict[int, int] = {orig.origin_id: 0 for orig in origins}

    steps = np.arange(step_duration_min, horizon_min + 1e-6, step_duration_min)
    step_latencies: list[float] = []

    for step_t in steps:
        t_step_start = time.perf_counter()
        env = harness.step(step_t)

        edge_arrivals = road_graph.compute_edge_fire_arrivals(env.arrival_field)
        closed_rb = harness.get_closed_edges_for_roadblocks(env.active_roadblocks)

        dynamic_times = None
        if closed_rb:
            dynamic_times = np.array(
                [e.traversal_time_min for e in road_graph.edges], dtype=np.float64
            )
            for idx, ge in enumerate(road_graph.edges):
                if (ge.u, ge.v) in closed_rb or (ge.v, ge.u) in closed_rb:
                    dynamic_times[idx] = float("inf")

        for orig in origins:
            cur_r = routes[orig.origin_id]
            if cur_r is None:
                continue
            if cur_r.arrival_time_min <= step_t:
                continue

            # Check if road is severed by roadblock
            severed = any((e.u, e.v) in closed_rb or (e.v, e.u) in closed_rb for e in cur_r.edges)

            # Re-evaluate route margin under updated fire field
            edge_lookup = {(e.u, e.v): e for e in road_graph.edges}
            min_cur_margin = float("inf")
            for re in cur_r.edges:
                ed = edge_lookup.get((re.u, re.v))
                if ed is not None:
                    fire_t = float(edge_arrivals[ed.edge_idx])
                    m = fire_t - re.exit_time_min
                    if m < min_cur_margin:
                        min_cur_margin = m

            margin_eroded = min_cur_margin < buffer_minutes

            if not (severed or margin_eroded):
                continue

            # Replan triggered!
            replan_counts[orig.origin_id] += 1
            if step_t <= orig.departure_time_min:
                new_r = router.find_route(
                    origin_node=orig.origin_id,
                    destination_nodes=dest_set,
                    t0=orig.departure_time_min,
                    buffer_minutes=buffer_minutes,
                    edge_fire_arrivals=edge_arrivals,
                    dynamic_edge_traversal_times=dynamic_times,
                )
                routes[orig.origin_id] = new_r
                replan_histories[orig.origin_id].append((step_t, new_r))
            else:
                pos = _find_vehicle_position_on_route(cur_r, step_t)
                if pos is not None:
                    down_r = router.find_route(
                        origin_node=pos.edge_v,
                        destination_nodes=dest_set,
                        t0=step_t,
                        buffer_minutes=buffer_minutes,
                        edge_fire_arrivals=edge_arrivals,
                        dynamic_edge_traversal_times=dynamic_times,
                    )
                    routes[orig.origin_id] = down_r
                    replan_histories[orig.origin_id].append((step_t, down_r))
                else:
                    routes[orig.origin_id] = None
                    replan_histories[orig.origin_id].append((step_t, None))

        step_latencies.append((time.perf_counter() - t_step_start) * 1000.0)

    # Compile results
    results: list[ReplayOriginResult] = []
    for orig in origins:
        final_r = routes[orig.origin_id]
        by_type = check_route_hazard_miss_by_type(final_r, truth_hazards, road_graph)
        is_miss, _ = by_type["total"]
        lead_time, flagged_before = compute_warning_lead_time(
            initial_route=initial_routes[orig.origin_id],
            replan_history=replan_histories[orig.origin_id],
            hazards=truth_hazards,
            road_graph=road_graph,
        )
        tt = final_r.total_travel_time_min if final_r is not None else float("inf")
        margin = final_r.min_margin_min if final_r is not None else -float("inf")

        results.append(
            ReplayOriginResult(
                origin_id=orig.origin_id,
                method_name=method_name,
                departure_time_min=orig.departure_time_min,
                route=final_r,
                travel_time_min=tt,
                hazard_miss=is_miss,
                min_margin_min=margin,
                replan_count=replan_counts[orig.origin_id],
                warning_lead_time_min=lead_time,
                flagged_before_onset=flagged_before,
                hazard_misses_by_type={k: v[0] for k, v in by_type.items()},
            )
        )

    mean_step_ms = float(np.mean(step_latencies)) if step_latencies else 0.0
    return results, mean_step_ms
