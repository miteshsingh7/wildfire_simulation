"""Wildland evacuation demand modeling and time-sliced departure generation.

Implements:
1. Documented census populations for Town of Paradise (split into 4 sub-zones),
   Magalia, and Concow.
2. Vehicle fleet estimation: vehicles = persons / occupancy.
3. Time-sliced departure schedules (5-minute slices) driven by notification
   times and response curves.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wf052.routing.graph import RoadGraph


@dataclass(frozen=True)
class CensusBlockDemand:
    """Census block vehicle demand with snapped road network origin node."""

    geoid10: str
    population: int
    num_vehicles: int
    assigned_zone: str
    centroid_utm: tuple[float, float]
    origin_node_id: int
    departure_schedule: dict[float, int]


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
    """Vehicles departing a specific origin node in a 5-minute time slice."""

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
        self.t0_ignition_utc = str(
            self.cfg.get("scenario", {}).get(
                "t0_ignition_utc",
                self.cfg.get("scenario", {}).get("event_ignition_utc", "2018-11-08T14:33:00Z"),
            )
        )

        self.zones: dict[str, EvacuationZone] = {}
        self.departure_schedules: dict[str, dict[float, int]] = {}
        self.block_demands: list[CensusBlockDemand] = []
        self.node_vehicles: dict[int, int] = {}
        self.max_per_node_load: int = 0
        self.max_load_node_id: int = 0
        self.unique_origin_nodes_count: int = 0

        self._initialize_zones(road_graph)
        if road_graph is not None and len(road_graph.nodes) >= 100:
            self._initialize_block_demands(road_graph)

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

    def _initialize_block_demands(self, road_graph: RoadGraph) -> None:
        """Load census block centroids and map vehicles to nearest road graph origin nodes."""
        blocks_path = Path("data/tiles/census2010_butte_blocks_pophu.parquet")
        zones_path = Path("data/tiles/paradise_14_zones.parquet")
        places_path = Path("data/raw/census2010_places/tl_2010_06_place10.shp")

        if not blocks_path.exists():
            return

        import geopandas as gpd

        node_ids = np.array(list(road_graph.nodes.keys()), dtype=np.int64)
        node_coords = np.array([road_graph.nodes[nid] for nid in node_ids])

        blocks_gdf = gpd.read_parquet(blocks_path).to_crs("EPSG:32610")

        # Load 14 Paradise zones polygons
        if not zones_path.exists():
            from scripts.plot_paradise_zones_overlay import build_and_save_14_zones

            zones_gdf, _, _ = build_and_save_14_zones()
        else:
            zones_gdf = gpd.read_parquet(zones_path)

        town_poly = zones_gdf.union_all()
        p_blocks = blocks_gdf[
            blocks_gdf.geometry.centroid.within(town_poly) & (blocks_gdf["POP10"] > 0)
        ].copy()

        # CDPs: Magalia and Concow
        m_blocks = gpd.GeoDataFrame()
        c_blocks = gpd.GeoDataFrame()
        if places_path.exists():
            places = gpd.read_file(places_path).to_crs("EPSG:32610")
            mag_poly = places[places["NAMELSAD10"] == "Magalia CDP"].iloc[0].geometry
            con_poly = places[places["NAMELSAD10"] == "Concow CDP"].iloc[0].geometry
            m_blocks = blocks_gdf[
                blocks_gdf.geometry.centroid.within(mag_poly) & (blocks_gdf["POP10"] > 0)
            ].copy()
            c_blocks = blocks_gdf[
                blocks_gdf.geometry.centroid.within(con_poly) & (blocks_gdf["POP10"] > 0)
            ].copy()

        def apportion_zone_vehicles(
            blocks_info: list[dict[str, Any]], target_veh: int
        ) -> list[int]:
            if not blocks_info:
                return []
            if len(blocks_info) == 1:
                return [target_veh]
            total_pop = sum(b["pop"] for b in blocks_info)
            if total_pop <= 0:
                return [max(1, target_veh // len(blocks_info))] * len(blocks_info)
            shares = np.array(
                [b["pop"] / total_pop * target_veh for b in blocks_info], dtype=np.float64
            )
            floor_vals = np.floor(shares).astype(int)
            if target_veh >= len(blocks_info):
                for i in range(len(floor_vals)):
                    if floor_vals[i] < 1:
                        floor_vals[i] = 1
            rem = target_veh - int(np.sum(floor_vals))
            if rem > 0:
                fracs = shares - floor_vals
                top_indices = np.argsort(-fracs)[:rem]
                for idx in top_indices:
                    floor_vals[idx] += 1
            elif rem < 0:
                sort_desc = np.argsort(-floor_vals)
                for idx in sort_desc:
                    if rem == 0:
                        break
                    if floor_vals[idx] > 1:
                        deduct = min(floor_vals[idx] - 1, -rem)
                        floor_vals[idx] -= deduct
                        rem += deduct
            return floor_vals.tolist()

        # Group Paradise blocks by assigned zone
        paradise_zone_blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for _, blk in p_blocks.iterrows():
            b_pt = blk.geometry.centroid
            pop = int(blk["POP10"])
            assigned_z = "Zone_1"
            for _, z_row in zones_gdf.iterrows():
                if z_row.geometry.contains(b_pt):
                    assigned_z = z_row.zone_name
                    break
            dists = np.hypot(node_coords[:, 0] - b_pt.x, node_coords[:, 1] - b_pt.y)
            snapped_nid = int(node_ids[np.argmin(dists)])
            paradise_zone_blocks[assigned_z].append(
                {
                    "geoid10": str(blk["GEOID10"]),
                    "pop": pop,
                    "centroid_utm": (float(b_pt.x), float(b_pt.y)),
                    "origin_node_id": snapped_nid,
                }
            )

        block_demands: list[CensusBlockDemand] = []
        node_vehicles: dict[int, int] = defaultdict(int)

        # Build Paradise block demands with exact vehicle apportionment
        for z_name, b_list in paradise_zone_blocks.items():
            z_obj = self.zones.get(z_name, self.zones.get("Zone_1"))
            target_v = (
                z_obj.num_vehicles if z_obj else sum(int(b["pop"] / self.occupancy) for b in b_list)
            )
            v_alloc = apportion_zone_vehicles(b_list, target_v)
            notif_t = z_obj.notification_elapsed_min if z_obj else 60.0
            dur_t = z_obj.departure_duration_min if z_obj else 90.0

            for b_info, n_veh in zip(b_list, v_alloc, strict=True):
                sched = calculate_departure_schedule(
                    total_vehicles=n_veh,
                    notification_min=notif_t,
                    duration_min=dur_t,
                    time_slice_min=self.time_slice_min,
                    horizon_min=self.horizon_min,
                )
                block_demands.append(
                    CensusBlockDemand(
                        geoid10=b_info["geoid10"],
                        population=b_info["pop"],
                        num_vehicles=n_veh,
                        assigned_zone=z_name,
                        centroid_utm=b_info["centroid_utm"],
                        origin_node_id=b_info["origin_node_id"],
                        departure_schedule=sched,
                    )
                )
                node_vehicles[b_info["origin_node_id"]] += n_veh

        # Process Magalia blocks with exact vehicle apportionment
        if len(m_blocks) > 0 and "Magalia" in self.zones:
            z_mag = self.zones["Magalia"]
            m_list = []
            for _, blk in m_blocks.iterrows():
                b_pt = blk.geometry.centroid
                pop = int(blk["POP10"])
                dists = np.hypot(node_coords[:, 0] - b_pt.x, node_coords[:, 1] - b_pt.y)
                snapped_nid = int(node_ids[np.argmin(dists)])
                m_list.append(
                    {
                        "geoid10": str(blk["GEOID10"]),
                        "pop": pop,
                        "centroid_utm": (float(b_pt.x), float(b_pt.y)),
                        "origin_node_id": snapped_nid,
                    }
                )

            m_alloc = apportion_zone_vehicles(m_list, z_mag.num_vehicles)
            for b_info, n_veh in zip(m_list, m_alloc, strict=True):
                sched = calculate_departure_schedule(
                    total_vehicles=n_veh,
                    notification_min=z_mag.notification_elapsed_min,
                    duration_min=z_mag.departure_duration_min,
                    time_slice_min=self.time_slice_min,
                    horizon_min=self.horizon_min,
                )
                block_demands.append(
                    CensusBlockDemand(
                        geoid10=b_info["geoid10"],
                        population=b_info["pop"],
                        num_vehicles=n_veh,
                        assigned_zone="Magalia",
                        centroid_utm=b_info["centroid_utm"],
                        origin_node_id=b_info["origin_node_id"],
                        departure_schedule=sched,
                    )
                )
                node_vehicles[b_info["origin_node_id"]] += n_veh

        # Process Concow blocks with exact vehicle apportionment
        if len(c_blocks) > 0 and "Concow" in self.zones:
            z_con = self.zones["Concow"]
            c_list = []
            for _, blk in c_blocks.iterrows():
                b_pt = blk.geometry.centroid
                pop = int(blk["POP10"])
                dists = np.hypot(node_coords[:, 0] - b_pt.x, node_coords[:, 1] - b_pt.y)
                snapped_nid = int(node_ids[np.argmin(dists)])
                c_list.append(
                    {
                        "geoid10": str(blk["GEOID10"]),
                        "pop": pop,
                        "centroid_utm": (float(b_pt.x), float(b_pt.y)),
                        "origin_node_id": snapped_nid,
                    }
                )

            c_alloc = apportion_zone_vehicles(c_list, z_con.num_vehicles)
            for b_info, n_veh in zip(c_list, c_alloc, strict=True):
                sched = calculate_departure_schedule(
                    total_vehicles=n_veh,
                    notification_min=z_con.notification_elapsed_min,
                    duration_min=z_con.departure_duration_min,
                    time_slice_min=self.time_slice_min,
                    horizon_min=self.horizon_min,
                )
                block_demands.append(
                    CensusBlockDemand(
                        geoid10=b_info["geoid10"],
                        population=b_info["pop"],
                        num_vehicles=n_veh,
                        assigned_zone="Concow",
                        centroid_utm=b_info["centroid_utm"],
                        origin_node_id=b_info["origin_node_id"],
                        departure_schedule=sched,
                    )
                )
                node_vehicles[b_info["origin_node_id"]] += n_veh

        # Pulga point demand
        if "Pulga" in self.zones:
            z_pulga = self.zones["Pulga"]
            node_vehicles[z_pulga.origin_node_id] += z_pulga.num_vehicles

        self.block_demands = block_demands
        self.node_vehicles = dict(node_vehicles)
        if self.node_vehicles:
            self.max_load_node_id = max(self.node_vehicles, key=self.node_vehicles.get)
            self.max_per_node_load = self.node_vehicles[self.max_load_node_id]
            self.unique_origin_nodes_count = len(self.node_vehicles)

    def get_total_population(self) -> int:
        """Return total regional population across all zones."""
        return sum(z.population for z in self.zones.values())

    def get_paradise_population(self) -> int:
        """Return total population across all Paradise zones, reconciling to Census total."""
        return sum(
            z.population
            for name, z in self.zones.items()
            if name.startswith("Zone_") or "Paradise" in name
        )

    def get_total_vehicles(self) -> int:
        """Return total regional vehicle fleet size across all zones."""
        return sum(z.num_vehicles for z in self.zones.values())

    def set_ignition_clock(self, ignition_time_utc: str) -> None:
        """Update master ignition clock from ArrivalField and shift departure curves.

        CONTRACT: Shifting ArrivalField.ignition_time shifts all zone departures by the exact offset.
        """
        import datetime

        old_dt = datetime.datetime.fromisoformat(self.t0_ignition_utc.replace("Z", "+00:00"))
        new_dt = datetime.datetime.fromisoformat(ignition_time_utc.replace("Z", "+00:00"))
        shift_min = (old_dt - new_dt).total_seconds() / 60.0

        self.t0_ignition_utc = ignition_time_utc
        # Shift all zone notification times and recalculate schedules
        new_zones = {}
        for name, zone in self.zones.items():
            new_notif = max(0.0, zone.notification_elapsed_min + shift_min)
            new_z = EvacuationZone(
                name=zone.name,
                census_source=zone.census_source,
                population=zone.population,
                num_vehicles=zone.num_vehicles,
                centroid_utm=zone.centroid_utm,
                notification_elapsed_min=new_notif,
                departure_duration_min=zone.departure_duration_min,
                origin_node_id=zone.origin_node_id,
            )
            new_zones[name] = new_z
            self.departure_schedules[name] = calculate_departure_schedule(
                total_vehicles=zone.num_vehicles,
                notification_min=new_notif,
                duration_min=zone.departure_duration_min,
                time_slice_min=self.time_slice_min,
                horizon_min=self.horizon_min,
            )
        self.zones = new_zones

        # Shift block departure schedules
        if self.block_demands:
            new_blocks = []
            for blk in self.block_demands:
                z_obj = self.zones.get(blk.assigned_zone)
                notif_t = z_obj.notification_elapsed_min if z_obj else 60.0
                dur_t = z_obj.departure_duration_min if z_obj else 90.0
                new_sched = calculate_departure_schedule(
                    total_vehicles=blk.num_vehicles,
                    notification_min=notif_t,
                    duration_min=dur_t,
                    time_slice_min=self.time_slice_min,
                    horizon_min=self.horizon_min,
                )
                new_blocks.append(
                    CensusBlockDemand(
                        geoid10=blk.geoid10,
                        population=blk.population,
                        num_vehicles=blk.num_vehicles,
                        assigned_zone=blk.assigned_zone,
                        centroid_utm=blk.centroid_utm,
                        origin_node_id=blk.origin_node_id,
                        departure_schedule=new_sched,
                    )
                )
            self.block_demands = new_blocks

    def get_slices_demand(self) -> list[TimeSliceDemand]:
        """Return flattened list of all time-slice demands sorted chronologically."""
        if self.block_demands:
            # Group block schedules by (assigned_zone, origin_node_id, slice_start_min)
            node_sched: dict[tuple[str, int, float], int] = defaultdict(int)
            for blk in self.block_demands:
                for t_start, cnt in blk.departure_schedule.items():
                    if cnt > 0:
                        node_sched[(blk.assigned_zone, blk.origin_node_id, t_start)] += cnt

            # Add Pulga if not in blocks
            if "Pulga" in self.zones and not any(
                b.assigned_zone == "Pulga" for b in self.block_demands
            ):
                z_pulga = self.zones["Pulga"]
                for t_start, cnt in self.departure_schedules.get("Pulga", {}).items():
                    if cnt > 0:
                        node_sched[("Pulga", z_pulga.origin_node_id, t_start)] += cnt

            demands = [
                TimeSliceDemand(
                    zone_name=z_name,
                    slice_start_min=t_start,
                    slice_end_min=t_start + self.time_slice_min,
                    vehicle_count=cnt,
                    origin_node_id=nid,
                )
                for (z_name, nid, t_start), cnt in node_sched.items()
                if cnt > 0
            ]
            demands.sort(key=lambda d: (d.slice_start_min, d.zone_name, d.origin_node_id))
            return demands

        # Fallback to zone centroids
        demands = []
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
