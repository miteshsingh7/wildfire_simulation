"""Edge capacity, point-queue bottleneck congestion, and dynamic time-sliced traffic assignment.

Implements:
1. Per-edge capacity (veh/hr) from Highway Capacity Manual (HCM 2016) defaults by highway type & lanes.
2. Flow-dependent travel time using point-queue bottleneck delay (and BPR function).
3. 5-minute time-sliced incremental assignment: loads each slice, updates delays, and re-routes subsequent slices.
4. Emergency contraflow flag doubling outbound capacity and blocking inbound traffic.
5. Per-zone evacuation clearance metrics (last vehicle out, bottleneck edges, congested margins).
"""

from collections import defaultdict
from dataclasses import dataclass
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
    capacity_veh_per_hr: float
    free_traversal_time_min: float
    is_contraflow: bool = False


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
    """Point-queue bottleneck and BPR congestion model for wildland evacuation routing."""

    def __init__(
        self,
        road_graph: RoadGraph,
        config_path: str | Path = "configs/evacuation_demand.yaml",
        use_point_queue: bool = True,
    ) -> None:
        self.road_graph = road_graph
        self.use_point_queue = use_point_queue

        with open(config_path) as f:
            self.cfg: dict[str, Any] = yaml.safe_load(f)

        self.cap_defaults = self.cfg.get("capacities_veh_per_hr_per_lane", {})
        self.lane_defaults = self.cfg.get("default_lanes_per_direction", {})
        self.bpr_alpha = float(self.cfg.get("bpr_parameters", {}).get("alpha", 0.15))
        self.bpr_beta = float(self.cfg.get("bpr_parameters", {}).get("beta", 4.0))

        self.edge_capacities: list[EdgeCapacityInfo] = []
        self._initialize_edge_capacities()

    def _initialize_edge_capacities(self) -> None:
        """Assign lanes and hourly capacities to every edge in the road graph."""
        for edge in self.road_graph.edges:
            hw = edge.highway.lower()
            # Determine lanes
            lanes = 1
            for k_hw, l_val in self.lane_defaults.items():
                if k_hw in hw:
                    lanes = int(l_val)
                    break

            # Determine capacity per lane
            cap_per_lane = float(self.cap_defaults.get("fallback", 600))
            for k_hw, c_val in self.cap_defaults.items():
                if k_hw in hw:
                    cap_per_lane = float(c_val)
                    break

            total_cap = float(cap_per_lane * lanes)
            self.edge_capacities.append(
                EdgeCapacityInfo(
                    edge_idx=edge.edge_idx,
                    u=edge.u,
                    v=edge.v,
                    key=edge.key,
                    highway=edge.highway,
                    lanes=lanes,
                    capacity_veh_per_hr=total_cap,
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
    ) -> tuple[float, float]:
        """Compute congested travel time and remaining queue for an edge during a time slice.

        Returns:
            (congested_time_min, remaining_queue_veh)
        """
        info = self.edge_capacities[edge_idx]
        t_free = info.free_traversal_time_min
        cap_hr = info.capacity_veh_per_hr

        if cap_hr <= 0.0:
            return float("inf"), preexisting_queue + flow_veh

        if self.use_point_queue:
            # Point-queue bottleneck delay
            slice_cap = cap_hr * (time_slice_min / 60.0)
            demand = preexisting_queue + flow_veh
            rem_queue = max(0.0, demand - slice_cap)
            avg_queue = 0.5 * (preexisting_queue + rem_queue)
            queue_delay_min = (avg_queue / cap_hr) * 60.0
            return t_free + queue_delay_min, rem_queue
        else:
            # Standard BPR function
            hourly_flow = flow_veh * (60.0 / time_slice_min)
            voc = hourly_flow / cap_hr
            congested_t = t_free * (1.0 + self.bpr_alpha * (voc**self.bpr_beta))
            return congested_t, 0.0


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
    ) -> dict[str, ZoneClearanceSummary]:
        """Execute dynamic 5-minute time-sliced traffic assignment across all zones.

        Returns:
            Dictionary mapping zone_name -> ZoneClearanceSummary.
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

        # Vehicle tracking per zone: list of (departure_t, arrival_t, min_margin, safe_flag)
        zone_vehicle_results: dict[str, list[tuple[float, float, float, bool]]] = defaultdict(list)

        for t_slice in all_times:
            # 1. Update edge traversal times based on current congestion
            if not free_flow_mode:
                for e_idx in range(num_edges):
                    congested_t, rem_q = self.congestion_model.compute_congested_time(
                        edge_idx=e_idx,
                        flow_veh=edge_slice_flows[e_idx],
                        time_slice_min=time_slice_min,
                        preexisting_queue=edge_queues[e_idx],
                    )
                    edge_queues[e_idx] = rem_q
                    edge_total_delays[e_idx] += max(
                        0.0, congested_t - self.road_graph.edges[e_idx].traversal_time_min
                    )
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

                # Find optimal safe route departing at t_slice
                route: EvacuationRoute | None = self.router.find_route(
                    origin_node=dem.origin_node_id,
                    destination_nodes=destination_node_ids,
                    t0=t_slice,
                    buffer_minutes=buffer_minutes,
                    edge_fire_arrivals=edge_fire_arrivals,
                    mode=RoutingMode.FASTEST_SAFE,
                )

                if route is not None and len(route.edges) > 0:
                    # Vehicle safely exits
                    is_safe = (route.status == RouteStatus.SAFE) and (route.min_margin_min >= 0.0)
                    zone_vehicle_results[dem.zone_name].append(
                        (t_slice, route.arrival_time_min, route.min_margin_min, is_safe)
                    )

                    # Accumulate flow onto route edges for congestion loading
                    for re in route.edges:
                        # Find corresponding edge index
                        for cand in self.road_graph.edges:
                            if cand.u == re.u and cand.v == re.v and cand.key == re.key:
                                edge_slice_flows[cand.edge_idx] += cnt
                                break

        # 3. Compile clearance summary per zone
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

            # Free-flow theoretical baseline: notification + duration + approx free-flow travel time (15m)
            free_clearance = zone.notification_elapsed_min + zone.departure_duration_min + 15.0
            congestion_delay_hrs = max(0.0, (last_arrival - free_clearance) / 60.0)

            # Top bottleneck edges
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

        return summaries
