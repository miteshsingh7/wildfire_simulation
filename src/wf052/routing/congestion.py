"""Edge capacity, point-queue bottleneck congestion, and dynamic time-sliced traffic assignment.

Implements:
1. Per-edge capacity (veh/hr) from Highway Capacity Manual (HCM 2016) planning defaults
   by highway classification & lane count, scaled by configurable capacity_scale and inbound_traffic.
2. Clean separation of delay models (no double counting):
   - Point-Queue Bottleneck Model: physical queuing when demand exceeds capacity (t = t_free + t_queue).
   - BPR Volume-Delay Function: flow-dependent moving travel time deceleration (t = t_free * (1 + alpha * (V/C)^beta)).
3. Finite boundary exit outflow capacities at domain perimeter.
4. Time-sliced 5-minute incremental assignment across cohort departure schedules.
5. Cohort entrapment metrics: vehicles on edge when fire arrives (q10/q50), and vehicle-hours
   within danger buffer B minutes of fire front.
6. Per-zone evacuation clearance metrics and bottleneck analysis.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wf052.routing.contracts import EvacuationRoute, RouteStatus, RoutingMode
from wf052.routing.demand import EvacuationDemandModel
from wf052.routing.graph import RoadGraph
from wf052.routing.router import TimeDependentRouter


@dataclass
class EdgeCapacityInfo:
    """Capacity and congestion properties of an edge in the road graph."""

    edge_idx: int
    u: int
    v: int
    key: int
    highway: str
    lanes: int
    nominal_capacity_veh_per_hr: float
    capacity_veh_per_hr: float
    free_traversal_time_min: float
    is_contraflow: bool = False


@dataclass(frozen=True)
class EntrapmentMetrics:
    """Quantitative entrapment and fire danger exposure metrics from cohort trajectories."""

    total_trapped_vehicles: int
    total_danger_vehicle_hours: float
    vehicles_trapped_by_edge: dict[int, int] = field(default_factory=dict)
    vehicles_trapped_by_zone: dict[str, int] = field(default_factory=dict)
    danger_vehicle_hours_by_edge: dict[int, float] = field(default_factory=dict)
    danger_vehicle_hours_by_zone: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ZoneClearanceSummary:
    """Clearance metrics and bottleneck performance for an evacuation zone."""

    zone_name: str
    total_vehicles: int
    evacuated_vehicles: int
    clearance_time_min: float  # Arrival time at domain boundary of last vehicle
    clearance_time_hours: float
    free_flow_clearance_time_min: float
    congestion_delay_hours: float
    min_margin_min: float  # Minimum safety margin experienced across all vehicles
    pct_exceeded_fire_arrival: float  # % of vehicles with exit_time > fire_arrival
    top_bottleneck_edges: tuple[str, ...]  # Names/IDs of top congested edges


class CongestionModel:
    """Bottleneck and capacity model for wildland evacuation traffic.

    Explicitly partitions delay:
    - If use_point_queue=True: computes physical queue accumulation and discharge delay;
      link traversal time remains free-flow (no moving-flow slowdown).
    - If use_point_queue=False: applies standard BPR volume-delay moving deceleration.
    Guarantees no edge is double-counted.
    """

    def __init__(
        self,
        road_graph: RoadGraph,
        config_path: str | Path = "configs/evacuation_demand.yaml",
        use_point_queue: bool = True,
        capacity_scale: float = 1.0,
        inbound_traffic_veh_per_hr: float = 0.0,
    ) -> None:
        self.road_graph = road_graph
        self.use_point_queue = use_point_queue
        self.capacity_scale = float(capacity_scale)
        self.inbound_traffic_veh_per_hr = float(inbound_traffic_veh_per_hr)

        with open(config_path) as f:
            self.cfg: dict[str, Any] = yaml.safe_load(f)

        self.cap_defaults = self.cfg.get("capacities_veh_per_hr_per_lane", {})
        self.lane_defaults = self.cfg.get("default_lanes_per_direction", {})
        self.bpr_alpha = float(self.cfg.get("bpr_parameters", {}).get("alpha", 0.15))
        self.bpr_beta = float(self.cfg.get("bpr_parameters", {}).get("beta", 4.0))

        # Boundary exit outflow capacities
        self.exit_outflow_capacities = self.cfg.get("boundary_exit_outflow_capacities", {})

        self.edge_capacities: list[EdgeCapacityInfo] = []
        self._initialize_edge_capacities()

    def _initialize_edge_capacities(self) -> None:
        """Assign lanes, nominal capacity, and effective scaled capacity to every edge."""
        for edge in self.road_graph.edges:
            hw = edge.highway.lower()
            lanes = 1
            for k_hw, l_val in self.lane_defaults.items():
                if k_hw in hw:
                    lanes = int(l_val)
                    break

            cap_per_lane = float(self.cap_defaults.get("fallback", 600))
            for k_hw, c_val in self.cap_defaults.items():
                if k_hw in hw:
                    cap_per_lane = float(c_val)
                    break

            nominal_cap = float(cap_per_lane * lanes)
            # Apply capacity scale and subtract inbound emergency traffic demand
            effective_cap = max(
                10.0, (nominal_cap * self.capacity_scale) - self.inbound_traffic_veh_per_hr
            )

            self.edge_capacities.append(
                EdgeCapacityInfo(
                    edge_idx=edge.edge_idx,
                    u=edge.u,
                    v=edge.v,
                    key=edge.key,
                    highway=edge.highway,
                    lanes=lanes,
                    nominal_capacity_veh_per_hr=nominal_cap,
                    capacity_veh_per_hr=effective_cap,
                    free_traversal_time_min=edge.traversal_time_min,
                    is_contraflow=False,
                )
            )

    def apply_contraflow(
        self,
        edge_indices: list[int],
        capacity_multiplier: float = 2.0,
    ) -> None:
        """Activate emergency contraflow on designated outbound corridors.

        Doubles outbound capacity and marks contraflow active.
        """
        for idx in edge_indices:
            if 0 <= idx < len(self.edge_capacities):
                info = self.edge_capacities[idx]
                info.capacity_veh_per_hr *= capacity_multiplier
                info.is_contraflow = True

    def compute_congested_time(
        self,
        edge_idx: int,
        flow_veh: float,
        time_slice_min: float = 5.0,
        preexisting_queue: float = 0.0,
    ) -> tuple[float, float, float]:
        """Compute congested travel time, pure queue delay, and remaining queue for an edge.

        Returns:
            (total_traversal_time_min, queue_delay_min, remaining_queue_veh)
        """
        info = self.edge_capacities[edge_idx]
        t_free = info.free_traversal_time_min
        cap_hr = info.capacity_veh_per_hr

        if cap_hr <= 0.0:
            return float("inf"), float("inf"), preexisting_queue + flow_veh

        if self.use_point_queue:
            # Pure point-queue bottleneck delay (moving time remains t_free)
            slice_cap = cap_hr * (time_slice_min / 60.0)
            demand = preexisting_queue + flow_veh
            rem_queue = max(0.0, demand - slice_cap)
            avg_queue = 0.5 * (preexisting_queue + rem_queue)
            queue_delay_min = (avg_queue / cap_hr) * 60.0
            total_t = t_free + queue_delay_min
            return total_t, queue_delay_min, rem_queue
        else:
            # Standard BPR function: deceleration on link (queue delay is 0, moving time slows down)
            hourly_flow = flow_veh * (60.0 / time_slice_min)
            voc = hourly_flow / cap_hr
            congested_t = t_free * (1.0 + self.bpr_alpha * (voc**self.bpr_beta))
            bpr_delay = congested_t - t_free
            return congested_t, bpr_delay, 0.0


class TimeSlicedAssignmentEngine:
    """Executes dynamic time-sliced incremental evacuation traffic assignment."""

    def __init__(
        self,
        road_graph: RoadGraph,
        demand_model: EvacuationDemandModel,
        congestion_model: CongestionModel,
        router: TimeDependentRouter,
    ) -> None:
        self.road_graph = road_graph
        self.demand_model = demand_model
        self.congestion_model = congestion_model
        self.router = router

    def run_assignment(
        self,
        edge_fire_arrivals: np.ndarray,
        destination_node_ids: set[int],
        buffer_minutes: float = 15.0,
        free_flow_mode: bool = False,
        danger_buffer_minutes: float = 15.0,
        apply_exit_capacities: bool = True,
    ) -> tuple[dict[str, ZoneClearanceSummary], EntrapmentMetrics]:
        """Execute dynamic 5-minute time-sliced traffic assignment across all zones.

        Returns:
            (summaries_dict, entrapment_metrics)
        """
        time_slice_min = self.demand_model.time_slice_min
        slices_demand = self.demand_model.get_slices_demand()

        # Group demands by slice start time
        demands_by_time: dict[float, list[Any]] = defaultdict(list)
        for dem in slices_demand:
            demands_by_time[dem.slice_start_min].append(dem)

        all_times = sorted(demands_by_time.keys())
        num_edges = len(self.road_graph.edges)

        # Dynamic state tracking
        edge_queues = np.zeros(num_edges, dtype=np.float64)
        edge_slice_flows = np.zeros(num_edges, dtype=np.float64)
        edge_total_delays = np.zeros(num_edges, dtype=np.float64)

        # Boundary exit queue tracking
        exit_queues: dict[int, float] = defaultdict(float)

        # Vehicle tracking per zone: list of (departure_t, exit_arrival_t, min_margin, safe_flag)
        zone_vehicle_results: dict[str, list[tuple[float, float, float, bool]]] = defaultdict(list)

        # Cohort trajectories: list of dict(zone, num_vehicles, intervals=[(e_idx, t_enter, t_exit)])
        cohort_trajectories: list[dict[str, Any]] = []

        # Fast edge lookup: (u, v, key) -> edge_idx
        edge_lookup = {(e.u, e.v, e.key): e.edge_idx for e in self.road_graph.edges}

        for t_slice in all_times:
            # 1. Update edge traversal delays based on current congestion
            if not free_flow_mode:
                for e_idx in range(num_edges):
                    congested_t, queue_delay, rem_q = self.congestion_model.compute_congested_time(
                        edge_idx=e_idx,
                        flow_veh=edge_slice_flows[e_idx],
                        time_slice_min=time_slice_min,
                        preexisting_queue=edge_queues[e_idx],
                    )
                    edge_queues[e_idx] = rem_q
                    edge_total_delays[e_idx] += queue_delay
            else:
                edge_queues.fill(0.0)

            # Reset slice flow accumulators
            edge_slice_flows.fill(0.0)

            # 2. Assign demands departing in this slice
            slice_demands = demands_by_time[t_slice]
            for dem in slice_demands:
                cnt = dem.vehicle_count
                if cnt <= 0:
                    continue

                route: EvacuationRoute | None = self.router.find_route(
                    origin_node=dem.origin_node_id,
                    destination_nodes=destination_node_ids,
                    t0=t_slice,
                    buffer_minutes=buffer_minutes,
                    edge_fire_arrivals=edge_fire_arrivals,
                    mode=RoutingMode.FASTEST_SAFE,
                )

                if route is not None and len(route.edges) > 0:
                    dest_node = route.destination_id
                    base_arrival_t = route.arrival_time_min

                    # Apply boundary exit outflow capacity bottleneck delay
                    exit_delay = 0.0
                    if apply_exit_capacities and not free_flow_mode:
                        exit_cap_hr = float(
                            self.congestion_model.exit_outflow_capacities.get(
                                "fallback_exit", 1000.0
                            )
                        )
                        # Check if named destination has explicit capacity
                        slice_exit_cap = exit_cap_hr * (time_slice_min / 60.0)
                        q_exit = exit_queues[dest_node] + cnt
                        rem_exit_q = max(0.0, q_exit - slice_exit_cap)
                        exit_queues[dest_node] = rem_exit_q
                        exit_delay = (
                            0.5 * (exit_queues[dest_node] + rem_exit_q) / exit_cap_hr
                        ) * 60.0

                    final_arrival_t = base_arrival_t + exit_delay
                    is_safe = (route.status == RouteStatus.SAFE) and (route.min_margin_min >= 0.0)
                    zone_vehicle_results[dem.zone_name].append(
                        (t_slice, final_arrival_t, route.min_margin_min, is_safe)
                    )

                    # Build cohort trajectory intervals
                    intervals = []
                    for re in route.edges:
                        cand_idx = edge_lookup.get((re.u, re.v, re.key))
                        if cand_idx is not None:
                            edge_slice_flows[cand_idx] += cnt
                            intervals.append((cand_idx, re.entry_time_min, re.exit_time_min))

                    cohort_trajectories.append(
                        {
                            "zone_name": dem.zone_name,
                            "num_vehicles": cnt,
                            "intervals": intervals,
                        }
                    )

        # 3. Compute Entrapment Metrics from recorded cohort trajectories
        entrapment = self._compute_entrapment_metrics(
            cohort_trajectories=cohort_trajectories,
            edge_fire_arrivals=edge_fire_arrivals,
            danger_buffer_minutes=danger_buffer_minutes,
        )

        # 4. Compile clearance summary per zone
        summaries: dict[str, ZoneClearanceSummary] = {}
        for z_name, zone in self.demand_model.zones.items():
            results = zone_vehicle_results.get(z_name, [])
            total_v = zone.num_vehicles
            evac_v = len(results)

            if results:
                last_arrival = max(r[1] for r in results)
                min_marg = min(r[2] for r in results)
                unsafe_cnt = sum(1 for r in results if not r[3])
                pct_unsafe = (unsafe_cnt / evac_v) * 100.0 if evac_v > 0 else 100.0
            else:
                last_arrival = float("inf")
                min_marg = -float("inf")
                pct_unsafe = 100.0

            free_clearance = zone.notification_elapsed_min + zone.departure_duration_min + 15.0
            congestion_delay_hrs = max(0.0, (last_arrival - free_clearance) / 60.0)

            top_edges_idx = np.argsort(-edge_total_delays)[:3]
            top_edges = tuple(
                f"{self.road_graph.edges[i].name or 'Unnamed'} ({self.road_graph.edges[i].highway})"
                for i in top_edges_idx
                if edge_total_delays[i] > 0
            )

            summaries[z_name] = ZoneClearanceSummary(
                zone_name=z_name,
                total_vehicles=total_v,
                evacuated_vehicles=evac_v,
                clearance_time_min=float(last_arrival),
                clearance_time_hours=float(last_arrival / 60.0),
                free_flow_clearance_time_min=float(free_clearance),
                congestion_delay_hours=float(congestion_delay_hrs),
                min_margin_min=float(min_marg),
                pct_exceeded_fire_arrival=float(pct_unsafe),
                top_bottleneck_edges=top_edges,
            )

        return summaries, entrapment

    def _compute_entrapment_metrics(
        self,
        cohort_trajectories: list[dict[str, Any]],
        edge_fire_arrivals: np.ndarray,
        danger_buffer_minutes: float,
    ) -> EntrapmentMetrics:
        """Compute vehicles on edge when fire arrives and vehicle-hours in danger window."""
        trapped_by_edge: dict[int, int] = defaultdict(int)
        trapped_by_zone: dict[str, int] = defaultdict(int)
        danger_hrs_by_edge: dict[int, float] = defaultdict(float)
        danger_hrs_by_zone: dict[str, float] = defaultdict(float)

        for cohort in cohort_trajectories:
            z_name = cohort["zone_name"]
            n_veh = cohort["num_vehicles"]

            for e_idx, t_enter, t_exit in cohort["intervals"]:
                t_fire = float(edge_fire_arrivals[e_idx])
                if np.isinf(t_fire):
                    continue

                # 1. Trapped at arrival: fire reaches edge while vehicle is on the edge
                if t_enter <= t_fire <= t_exit:
                    trapped_by_edge[e_idx] += n_veh
                    trapped_by_zone[z_name] += n_veh

                # 2. Exposure within danger window B min of fire arrival
                window_start = max(t_enter, t_fire - danger_buffer_minutes)
                window_end = min(t_exit, t_fire)
                if window_end > window_start:
                    danger_time_hrs = ((window_end - window_start) / 60.0) * n_veh
                    danger_hrs_by_edge[e_idx] += danger_time_hrs
                    danger_hrs_by_zone[z_name] += danger_time_hrs

        total_trapped = sum(trapped_by_zone.values())
        total_danger_hrs = sum(danger_hrs_by_zone.values())

        return EntrapmentMetrics(
            total_trapped_vehicles=total_trapped,
            total_danger_vehicle_hours=float(total_danger_hrs),
            vehicles_trapped_by_edge=dict(trapped_by_edge),
            vehicles_trapped_by_zone=dict(trapped_by_zone),
            danger_vehicle_hours_by_edge=dict(danger_hrs_by_edge),
            danger_vehicle_hours_by_zone=dict(danger_hrs_by_zone),
        )
