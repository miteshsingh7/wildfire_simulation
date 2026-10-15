"""Wildland evacuation demand modeling and time-sliced departure generation.

Implements:
1. Documented census populations for Town of Paradise (split into 4 sub-zones),
   Magalia, and Concow.
2. Vehicle fleet estimation: vehicles = persons / occupancy.
3. Time-sliced departure schedules (5-minute slices) driven by notification
   times and response curves.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wf052.routing.graph import RoadGraph


@dataclass(frozen=True)
class EvacuationZone:
    """Evacuation origin zone specification."""

    name: str
    census_source: str
    population: int
    num_vehicles: int
    centroid_utm: tuple[float, float]
    notification_elapsed_min: float
    departure_duration_min: float
    origin_node_id: int


@dataclass(frozen=True)
class TimeSliceDemand:
    """Vehicles departing a specific zone in a 5-minute time slice."""

    zone_name: str
    slice_start_min: float
    slice_end_min: float
    vehicle_count: int
    origin_node_id: int


def calculate_departure_schedule(
    total_vehicles: int,
    notification_min: float,
    duration_min: float,
    time_slice_min: float = 5.0,
    horizon_min: float = 720.0,
) -> dict[float, int]:
    """Generate time-sliced vehicle departure counts using a standard sigmoid response curve.

    F(t) = 1 / (1 + exp(-k * (t - t_mid)))
    where t_mid = notification_min + duration_min / 2, k = 6.0 / duration_min.
    """
    if total_vehicles <= 0:
        return {}

    t_mid = notification_min + duration_min / 2.0
    k = 6.0 / max(1.0, duration_min)

    num_slices = int(np.ceil(horizon_min / time_slice_min))
    slice_times = np.arange(num_slices) * time_slice_min

    # Evaluate CDF at slice boundaries
    t_bounds = np.arange(num_slices + 1) * time_slice_min
    cdf_vals = 1.0 / (1.0 + np.exp(-k * (t_bounds - t_mid)))
    # Zero out before notification
    cdf_vals[t_bounds < notification_min] = 0.0

    # Incremental fractions per slice
    slice_fractions = np.diff(cdf_vals)
    # Normalize fractions to sum to 1.0 if any departed
    sum_frac = slice_fractions.sum()
    if sum_frac > 0:
        slice_fractions = slice_fractions / sum_frac

    # Distribute vehicles as integers
    exact_counts = slice_fractions * total_vehicles
    counts = np.floor(exact_counts).astype(int)
    remainder = total_vehicles - counts.sum()
    if remainder > 0:
        # Give remaining vehicles to the peak slices
        top_indices = np.argsort(-slice_fractions)[:remainder]
        for idx in top_indices:
            counts[idx] += 1

    schedule: dict[float, int] = {}
    for t_start, cnt in zip(slice_times, counts, strict=True):
        if cnt > 0:
            schedule[float(t_start)] = int(cnt)

    return schedule


class EvacuationDemandModel:
    """Manages evacuation zone populations, fleet sizes, and dynamic departure loads."""

    def __init__(
        self,
        config_path: str | Path = "configs/evacuation_demand.yaml",
        road_graph: RoadGraph | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.cfg: dict[str, Any] = yaml.safe_load(f)

        self.occupancy = float(self.cfg.get("occupancy", {}).get("persons_per_vehicle", 1.6))
        self.time_slice_min = float(self.cfg.get("scenario", {}).get("time_slice_minutes", 5.0))
        self.horizon_min = float(self.cfg.get("scenario", {}).get("total_horizon_minutes", 720.0))

        self.zones: dict[str, EvacuationZone] = {}
        self.departure_schedules: dict[str, dict[float, int]] = {}

        self._initialize_zones(road_graph)

    def _initialize_zones(self, road_graph: RoadGraph | None) -> None:
        """Parse zones, compute vehicle demands, and map to graph origin nodes."""
        raw_zones = self.cfg.get("zones", {})

        for name, z_cfg in raw_zones.items():
            pop = int(z_cfg["population"])
            num_veh = int(np.round(pop / self.occupancy))
            coords = (float(z_cfg["centroid_utm"][0]), float(z_cfg["centroid_utm"][1]))
            notif_t = float(z_cfg.get("notification_elapsed_min", 60.0))
            dur_t = float(z_cfg.get("departure_duration_min", 90.0))

            # Resolve nearest graph node
            node_id = 0
            if road_graph is not None:
                # Check if already in named_origins
                if name in road_graph.named_origins:
                    node_id = road_graph.named_origins[name]
                else:
                    node_ids = np.array(list(road_graph.nodes.keys()), dtype=np.int64)
                    node_coords = np.array([road_graph.nodes[nid] for nid in node_ids])
                    dists = np.hypot(node_coords[:, 0] - coords[0], node_coords[:, 1] - coords[1])
                    node_id = int(node_ids[np.argmin(dists)])

            zone = EvacuationZone(
                name=name,
                census_source=str(z_cfg.get("census_source", "")),
                population=pop,
                num_vehicles=num_veh,
                centroid_utm=coords,
                notification_elapsed_min=notif_t,
                departure_duration_min=dur_t,
                origin_node_id=node_id,
            )
            self.zones[name] = zone
            self.departure_schedules[name] = calculate_departure_schedule(
                total_vehicles=num_veh,
                notification_min=notif_t,
                duration_min=dur_t,
                time_slice_min=self.time_slice_min,
                horizon_min=self.horizon_min,
            )

    def get_total_population(self) -> int:
        """Return total regional population across all zones."""
        return sum(z.population for z in self.zones.values())

    def get_total_vehicles(self) -> int:
        """Return total regional vehicle fleet size across all zones."""
        return sum(z.num_vehicles for z in self.zones.values())

    def get_slices_demand(self) -> list[TimeSliceDemand]:
        """Return flattened list of all time-slice demands sorted chronologically."""
        demands: list[TimeSliceDemand] = []
        for name, sched in self.departure_schedules.items():
            node_id = self.zones[name].origin_node_id
            for t_start, cnt in sched.items():
                demands.append(
                    TimeSliceDemand(
                        zone_name=name,
                        slice_start_min=t_start,
                        slice_end_min=t_start + self.time_slice_min,
                        vehicle_count=cnt,
                        origin_node_id=node_id,
                    )
                )
        demands.sort(key=lambda d: (d.slice_start_min, d.zone_name))
        return demands
