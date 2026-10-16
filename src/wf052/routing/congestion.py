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

from wf052.routing.contracts import RouteStatus, RoutingMode
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

    def get_exit_outflow_capacity(self, dest_node: int, dest_name: str | None = None) -> float:
        """Look up outflow capacity for a boundary destination exit."""
        if dest_name is None:
            for name, nid in self.road_graph.boundary_destinations.items():
                if nid == dest_node:
                    dest_name = name
                    break
            if dest_name is None:
                dest_name = ""

        d_lower = dest_name.lower()
        # Direct exact or sanitized key match first
        if d_lower in self.exit_outflow_capacities:
            return float(self.exit_outflow_capacities[d_lower])
        if f"{d_lower}_exit" in self.exit_outflow_capacities:
            return float(self.exit_outflow_capacities[f"{d_lower}_exit"])

        # Keyword mapping for Camp Fire primary evacuation corridors
        if "skyway" in d_lower or "chico" in d_lower:
            return float(self.exit_outflow_capacities.get("skyway_south_exit", 1600.0))
        elif "clark" in d_lower:
            return float(self.exit_outflow_capacities.get("clark_south_exit", 1200.0))
        elif "pentz" in d_lower or "oroville" in d_lower:
            return float(self.exit_outflow_capacities.get("pentz_south_exit", 1000.0))
        elif "70" in d_lower or "feather" in d_lower:
            return float(self.exit_outflow_capacities.get("highway_70_south_exit", 1400.0))
        else:
            return float(self.exit_outflow_capacities.get("fallback_exit", 1000.0))

    def get_exit_commute_traffic(self, dest_node: int, dest_name: str | None = None) -> float:
        """Return background commute traffic reducing exit merge capacity (NIST TN 2252 sup Table A-2 TD-086)."""
        commute = float(
            self.cfg.get("background_commute_traffic", {}).get(
                "chico_ca99_commute_veh_per_hr", 500.0
            )
        )
        if dest_name is None:
            for name, nid in self.road_graph.boundary_destinations.items():
                if nid == dest_node:
                    dest_name = name
                    break
            if dest_name is None:
                dest_name = ""
        d_lower = dest_name.lower()
        if "chico" in d_lower or "skyway" in d_lower:
            return commute
        return 0.0


class TimeSlicedAssignmentEngine:
    """Executes dynamic time-sliced incremental evacuation traffic assignment with queue-aware routing.

    Implements:
    1. Dynamic FIFO Dijkstra path evaluation incorporating internal link queuing delays.
    2. Boundary exit outflow bottleneck delays entering path utility costs.
    3. Multi-destination Logit / MSA equilibrium flow splitting across egress corridors.
    4. Sub-horizon replanning (t >= t_now).
    5. Quantitative tracking of vehicles exiting per corridor over time and per-edge queue lengths.
    """

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

        # Persistent diagnostics from last assignment run
        self.last_vehicles_per_exit_over_time: dict[str, dict[float, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.last_vehicles_per_exit_by_id: dict[int, dict[float, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.last_edge_queues: np.ndarray = np.zeros(len(self.road_graph.edges), dtype=np.float64)
        self.last_edge_queues_over_time: dict[int, dict[float, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.last_cohort_trajectories: list[dict[str, Any]] = []
        self.last_edge_total_delays: np.ndarray = np.zeros(
            len(self.road_graph.edges), dtype=np.float64
        )

    def run_assignment(
        self,
        edge_fire_arrivals: np.ndarray,
        destination_node_ids: set[int],
        buffer_minutes: float = 15.0,
        free_flow_mode: bool = False,
        danger_buffer_minutes: float = 15.0,
        apply_exit_capacities: bool = True,
        t_now: float = 0.0,
        theta: float = 0.15,
        msa_iterations: int = 4,
    ) -> tuple[dict[str, ZoneClearanceSummary], EntrapmentMetrics]:
        """Execute dynamic 5-minute time-sliced traffic assignment across all zones.

        Args:
            edge_fire_arrivals: (E,) array of fire arrival times.
            destination_node_ids: Set of domain boundary destination node IDs.
            buffer_minutes: Safety margin threshold B in minutes.
            free_flow_mode: If True, disables queue delays and uses pure free-flow.
            danger_buffer_minutes: Exposure danger buffer for entrapment metrics.
            apply_exit_capacities: If True, applies finite exit outflow bottlenecks.
            t_now: Re-assign only from 'now' forward (t_slice >= t_now).
            theta: Logit choice dispersion parameter (sensitivity to travel + queue delay differences).
            msa_iterations: Method of Successive Averages iterations for flow balancing.

        Returns:
            (summaries_dict, entrapment_metrics)
        """
        from wf052.routing.router import _reconstruct_route, _run_constrained_dijkstra

        time_slice_min = self.demand_model.time_slice_min
        slices_demand = self.demand_model.get_slices_demand()

        # Group demands by slice start time
        demands_by_time: dict[float, list[Any]] = defaultdict(list)
        for dem in slices_demand:
            if dem.slice_start_min >= t_now:
                demands_by_time[dem.slice_start_min].append(dem)

        all_times = sorted(demands_by_time.keys())
        num_edges = len(self.road_graph.edges)

        # Dynamic state tracking
        edge_queues = np.zeros(num_edges, dtype=np.float64)
        edge_slice_flows = np.zeros(num_edges, dtype=np.float64)
        edge_total_delays = np.zeros(num_edges, dtype=np.float64)

        # Boundary exit queue tracking
        exit_queues: dict[int, float] = defaultdict(float)

        # Vehicle tracking per zone: list of (departure_t, exit_arrival_t, min_margin, safe_flag, count)
        zone_vehicle_results: dict[str, list[tuple[float, float, float, bool, int]]] = defaultdict(
            list
        )

        # Detailed tracking
        self.last_vehicles_per_exit_over_time = defaultdict(lambda: defaultdict(int))
        self.last_vehicles_per_exit_by_id = defaultdict(lambda: defaultdict(int))
        self.last_edge_queues_over_time = defaultdict(lambda: defaultdict(float))
        cohort_trajectories: list[dict[str, Any]] = []

        # Fast edge lookup: (u, v, key) -> edge_idx
        edge_lookup = {(e.u, e.v, e.key): e.edge_idx for e in self.road_graph.edges}
        dest_node_to_name = {
            nid: name for name, nid in self.road_graph.boundary_destinations.items()
        }

        # Compute effective exit capacities
        effective_exit_capacities: dict[int, float] = {}
        for d in destination_node_ids:
            cap = self.congestion_model.get_exit_outflow_capacity(d)
            commute = self.congestion_model.get_exit_commute_traffic(d)
            effective_exit_capacities[d] = max(100.0, cap - commute)

        for t_slice in all_times:
            slice_demands = demands_by_time[t_slice]
            total_slice_demand = sum(d.vehicle_count for d in slice_demands)

            if (
                total_slice_demand <= 0
                and np.all(edge_queues <= 0)
                and all(q <= 0 for q in exit_queues.values())
            ):
                continue

            # Free-flow vs. Congested Mode assignment
            if free_flow_mode:
                edge_slice_flows.fill(0.0)
                for dem in slice_demands:
                    cnt = dem.vehicle_count
                    if cnt <= 0:
                        continue
                    route = self.router.find_route(
                        origin_node=dem.origin_node_id,
                        destination_nodes=destination_node_ids,
                        t0=t_slice,
                        buffer_minutes=buffer_minutes,
                        edge_fire_arrivals=edge_fire_arrivals,
                        mode=RoutingMode.FASTEST_SAFE,
                    )
                    if route is not None and len(route.edges) > 0:
                        dest_node = route.destination_id
                        final_arrival_t = route.arrival_time_min
                        is_safe = (route.status == RouteStatus.SAFE) and (
                            route.min_margin_min >= 0.0
                        )
                        zone_vehicle_results[dem.zone_name].append(
                            (t_slice, final_arrival_t, route.min_margin_min, is_safe, cnt)
                        )
                        dest_name = dest_node_to_name.get(dest_node, str(dest_node))
                        self.last_vehicles_per_exit_over_time[dest_name][t_slice] += cnt
                        self.last_vehicles_per_exit_by_id[dest_node][t_slice] += cnt

                        intervals = []
                        for re in route.edges:
                            cand_idx = edge_lookup.get((re.u, re.v, re.key))
                            if cand_idx is not None:
                                edge_slice_flows[cand_idx] += cnt
                                intervals.append((cand_idx, re.entry_time_min, re.exit_time_min))
                        cohort_trajectories.append(
                            {
                                "zone_name": dem.zone_name,
                                "destination_id": dest_node,
                                "num_vehicles": cnt,
                                "intervals": intervals,
                            }
                        )
                continue

            # CONGESTED QUEUE-AWARE ASSIGNMENT WITH MSA LOGIT FLOW SPLITTING
            tentative_edge_flows = np.zeros(num_edges, dtype=np.float64)
            tentative_exit_flows: dict[int, float] = {d: 0.0 for d in destination_node_ids}
            dem_assigned_split: dict[Any, dict[int, float]] = {}
            dem_parents: dict[Any, dict[int, tuple[int, int]]] = {}
            dem_reached: dict[Any, list[int]] = {}

            k_max = max(1, msa_iterations)
            for k in range(1, k_max + 1):
                alpha_k = 1.0 / float(k)

                # 1. Update dynamic edge traversal times (t_free + queue_delay)
                dyn_edge_times = np.zeros(num_edges, dtype=np.float64)
                for e_idx in range(num_edges):
                    cap_e = self.congestion_model.edge_capacities[e_idx].capacity_veh_per_hr
                    q_e = edge_queues[e_idx] + 0.5 * tentative_edge_flows[e_idx]
                    q_delay = (q_e / cap_e) * 60.0 if cap_e > 0 else 0.0
                    dyn_edge_times[e_idx] = (
                        self.congestion_model.edge_capacities[e_idx].free_traversal_time_min
                        + q_delay
                    )

                # 2. Update boundary exit bottleneck delays
                exit_delays: dict[int, float] = {}
                for d in destination_node_ids:
                    if apply_exit_capacities:
                        cap_d = effective_exit_capacities[d]
                        q_d = exit_queues.get(d, 0.0) + 0.5 * tentative_exit_flows.get(d, 0.0)
                        exit_delays[d] = (q_d / cap_d) * 60.0
                    else:
                        exit_delays[d] = 0.0

                # 3. Settle all destinations via FIFO Dijkstra & compute Logit choice probabilities
                for dem in slice_demands:
                    cnt = dem.vehicle_count
                    if cnt <= 0:
                        continue

                    # Try constrained fastest safe route to all destinations
                    _best_dest, earliest_arrival, parent = _run_constrained_dijkstra(
                        road_graph=self.road_graph,
                        origin_node=dem.origin_node_id,
                        destination_nodes=destination_node_ids,
                        t0=t_slice,
                        buffer_minutes=buffer_minutes,
                        edge_fire_arrivals=edge_fire_arrivals,
                        dynamic_edge_traversal_times=dyn_edge_times,
                        destination_exit_delays=exit_delays,
                        settle_all_destinations=True,
                    )
                    reached_dests = [d for d in destination_node_ids if d in earliest_arrival]

                    # Fallback to unconstrained reachability if fire cuts all safe routes
                    if not reached_dests:
                        _best_dest, earliest_arrival, parent = _run_constrained_dijkstra(
                            road_graph=self.road_graph,
                            origin_node=dem.origin_node_id,
                            destination_nodes=destination_node_ids,
                            t0=t_slice,
                            buffer_minutes=-1e9,
                            edge_fire_arrivals=edge_fire_arrivals,
                            dynamic_edge_traversal_times=dyn_edge_times,
                            destination_exit_delays=exit_delays,
                            settle_all_destinations=True,
                        )
                        reached_dests = [d for d in destination_node_ids if d in earliest_arrival]

                    if not reached_dests:
                        continue

                    dem_parents[dem] = parent
                    dem_reached[dem] = reached_dests

                    # Cost = path travel time + boundary exit bottleneck queue delay
                    costs = np.array(
                        [
                            (earliest_arrival[d] - t_slice) + exit_delays.get(d, 0.0)
                            for d in reached_dests
                        ]
                    )
                    v = -theta * costs
                    v_norm = v - np.max(v)
                    probs = np.exp(v_norm) / np.sum(np.exp(v_norm))
                    aux_split = {
                        d: float(cnt * p) for d, p in zip(reached_dests, probs, strict=True)
                    }

                    if k == 1:
                        dem_assigned_split[dem] = aux_split
                    else:
                        curr_split = dem_assigned_split.get(dem, {})
                        dem_assigned_split[dem] = {
                            d: (1.0 - alpha_k) * curr_split.get(d, 0.0)
                            + alpha_k * aux_split.get(d, 0.0)
                            for d in reached_dests
                        }

                # 4. Update tentative flows for next MSA iteration
                tentative_edge_flows.fill(0.0)
                tentative_exit_flows = {d: 0.0 for d in destination_node_ids}
                for dem, split in dem_assigned_split.items():
                    parent = dem_parents.get(dem, {})
                    for d, flow_d in split.items():
                        if flow_d <= 0.0:
                            continue
                        tentative_exit_flows[d] += flow_d
                        curr = d
                        while curr != dem.origin_node_id and curr in parent:
                            p_node, e_idx = parent[curr]
                            tentative_edge_flows[e_idx] += flow_d
                            curr = p_node

            # Finalize integer vehicle assignments across reached corridors
            edge_slice_flows.fill(0.0)
            exit_slice_flows: dict[int, float] = {d: 0.0 for d in destination_node_ids}

            for dem in slice_demands:
                cnt = dem.vehicle_count
                if cnt <= 0 or dem not in dem_assigned_split:
                    continue

                split = dem_assigned_split[dem]
                d_list = list(split.keys())
                exact_vals = np.array([split[d] for d in d_list], dtype=np.float64)
                floor_vals = np.floor(exact_vals).astype(int)
                rem = cnt - int(np.sum(floor_vals))
                if rem > 0:
                    fracs = exact_vals - floor_vals
                    top_indices = np.argsort(-fracs)[:rem]
                    for idx in top_indices:
                        floor_vals[idx] += 1

                for d, n_d in zip(d_list, floor_vals, strict=True):
                    if n_d <= 0:
                        continue

                    route = _reconstruct_route(
                        road_graph=self.road_graph,
                        origin_node=dem.origin_node_id,
                        dest_node=d,
                        t0=t_slice,
                        parent=dem_parents[dem],
                        edge_fire_arrivals=edge_fire_arrivals,
                        status=RouteStatus.SAFE
                        if d in dem_reached.get(dem, [])
                        else RouteStatus.BEST_EFFORT,
                        mode_used=RoutingMode.FASTEST_SAFE,
                        dynamic_edge_traversal_times=dyn_edge_times,
                    )

                    exit_delay = exit_delays.get(d, 0.0)
                    final_arrival_t = route.arrival_time_min + exit_delay
                    is_safe = (route.status == RouteStatus.SAFE) and (route.min_margin_min >= 0.0)

                    zone_vehicle_results[dem.zone_name].append(
                        (t_slice, final_arrival_t, route.min_margin_min, is_safe, int(n_d))
                    )

                    dest_name = dest_node_to_name.get(d, str(d))
                    self.last_vehicles_per_exit_over_time[dest_name][t_slice] += int(n_d)
                    self.last_vehicles_per_exit_by_id[d][t_slice] += int(n_d)

                    intervals = []
                    for re in route.edges:
                        cand_idx = edge_lookup.get((re.u, re.v, re.key))
                        if cand_idx is not None:
                            edge_slice_flows[cand_idx] += n_d
                            intervals.append((cand_idx, re.entry_time_min, re.exit_time_min))

                    exit_slice_flows[d] += n_d
                    cohort_trajectories.append(
                        {
                            "zone_name": dem.zone_name,
                            "destination_id": d,
                            "num_vehicles": int(n_d),
                            "intervals": intervals,
                        }
                    )

            # Physical Queue Updating at End of Time Slice
            for e_idx in range(num_edges):
                cap_e = self.congestion_model.edge_capacities[e_idx].capacity_veh_per_hr
                slice_cap = cap_e * (time_slice_min / 60.0)
                old_q = edge_queues[e_idx]
                rem_q = max(0.0, old_q + edge_slice_flows[e_idx] - slice_cap)
                edge_queues[e_idx] = rem_q
                avg_q = 0.5 * (old_q + rem_q)
                q_del = (avg_q / cap_e) * 60.0 if cap_e > 0 else 0.0
                edge_total_delays[e_idx] += q_del
                if rem_q > 0:
                    self.last_edge_queues_over_time[e_idx][t_slice] = rem_q

            for d in destination_node_ids:
                cap_d = effective_exit_capacities[d]
                slice_exit_cap = cap_d * (time_slice_min / 60.0)
                old_exit_q = exit_queues.get(d, 0.0)
                rem_exit_q = max(0.0, old_exit_q + exit_slice_flows.get(d, 0.0) - slice_exit_cap)
                exit_queues[d] = rem_exit_q

        # Save persistent run state
        self.last_edge_queues = edge_queues
        self.last_cohort_trajectories = cohort_trajectories
        self.last_edge_total_delays = edge_total_delays

        # Compute Entrapment Metrics from recorded cohort trajectories
        entrapment = self._compute_entrapment_metrics(
            cohort_trajectories=cohort_trajectories,
            edge_fire_arrivals=edge_fire_arrivals,
            danger_buffer_minutes=danger_buffer_minutes,
        )

        # Compile clearance summary per zone
        summaries: dict[str, ZoneClearanceSummary] = {}
        for z_name, zone in self.demand_model.zones.items():
            results = zone_vehicle_results.get(z_name, [])
            total_v = zone.num_vehicles
            evac_v = sum(r[4] for r in results) if results else 0

            if results and evac_v > 0:
                last_arrival = max(r[1] for r in results)
                min_marg = min(r[2] for r in results)
                unsafe_cnt = sum(r[4] for r in results if not r[3])
                pct_unsafe = (unsafe_cnt / evac_v) * 100.0
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

    def print_vehicles_per_exit_over_time(self) -> None:
        """Print table of vehicle departures per exit corridor across time slices."""
        print("\n=== VEHICLES PER EXIT CORRIDOR OVER TIME (Baseline Assignment) ===")
        all_exits = sorted(self.last_vehicles_per_exit_over_time.keys())
        all_times = sorted(
            {t for d in all_exits for t in self.last_vehicles_per_exit_over_time[d].keys()}
        )
        if not all_times:
            print("No vehicles assigned to any exit.")
            return

        header = f"{'Time (min)':<12}" + "".join(f"{d:>18}" for d in all_exits) + f"{'Total':>14}"
        print(header)
        print("-" * len(header))
        cum_by_exit: dict[str, int] = defaultdict(int)
        for t in all_times:
            row_vals = [self.last_vehicles_per_exit_over_time[d].get(t, 0) for d in all_exits]
            row_total = sum(row_vals)
            for d, v in zip(all_exits, row_vals, strict=True):
                cum_by_exit[d] += v
            print(f"{t:<12.1f}" + "".join(f"{v:>18}" for v in row_vals) + f"{row_total:>14}")

        print("-" * len(header))
        total_evac = sum(cum_by_exit.values())
        print(
            f"{'CUMULATIVE':<12}"
            + "".join(f"{cum_by_exit[d]:>18}" for d in all_exits)
            + f"{total_evac:>14}"
        )

    def print_edge_queue_lengths(self, top_n: int = 15) -> None:
        """Print table of maximum and persistent edge queue lengths and bottleneck locations."""
        print(f"\n=== TOP {top_n} EDGE QUEUE BOTTLENECKS (Max Vehicles Queued) ===")
        max_queues: list[tuple[int, float, str, str]] = []
        for e_idx, time_dict in self.last_edge_queues_over_time.items():
            if time_dict:
                max_q = max(time_dict.values())
                ed = self.road_graph.edges[e_idx]
                max_queues.append((e_idx, max_q, ed.name or "Unnamed", ed.highway))

        max_queues.sort(key=lambda x: -x[1])
        if not max_queues:
            print("No internal edge queues accumulated (all links within slice capacity).")
            return

        header = f"{'Edge Index':<12} {'Max Queue (veh)':<18} {'Highway Type':<16} {'Road Name'}"
        print(header)
        print("-" * len(header))
        for e_idx, max_q, name, hw in max_queues[:top_n]:
            print(f"{e_idx:<12} {max_q:<18.1f} {hw:<16} {name}")

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
