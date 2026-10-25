"""Causal replay simulation harness for the 2018 Camp Fire.

Enforces strict causal filtering:
- At simulation step t:
  - Observations visible only if timestamp <= t - latency_min
  - Roadblocks active only if onset_time <= t
  - Future data (t' > t) is strictly invisible and cannot affect state.
"""

import re
from pathlib import Path

import geopandas as gpd
from shapely import STRtree

from wf052.replay.contracts import CausalStepEnvironment, TruthHazard
from wf052.replay.providers import FieldProvider, TeacherFieldProvider
from wf052.routing.graph import RoadGraph


def parse_clock_to_elapsed_min(time_str: str, base_hour: int = 6, base_min: int = 33) -> float:
    """Convert HH:MM clock string into elapsed minutes from base clock (06:33 PST)."""
    match = re.search(r"(\d{1,2}):(\d{2})", str(time_str))
    if not match:
        return 0.0
    h, m = int(match.group(1)), int(match.group(2))
    return (h - base_hour) * 60.0 + (m - base_min)


class CausalReplayHarness:
    """Causal simulation environment stepping from ignition (06:33 PST) forward in 10-minute intervals."""

    def __init__(
        self,
        road_graph: RoadGraph,
        provider: FieldProvider | None = None,
        latency_min: float = 0.0,
        tiles_dir: str | Path = "data/tiles",
        step_duration_min: float = 10.0,
        horizon_min: float = 360.0,
    ) -> None:
        self.road_graph = road_graph
        self.provider = provider or TeacherFieldProvider()
        self.latency_min = float(latency_min)
        self.tiles_dir = Path(tiles_dir)
        self.step_duration_min = step_duration_min
        self.horizon_min = horizon_min

        # Load datasets
        self._raw_observations = self._load_raw_observations()
        self._raw_roadblocks = self._load_raw_roadblocks()
        self._truth_hazards = self._load_all_truth_hazards()
        self._edge_tree = STRtree([e.geometry for e in self.road_graph.edges])

    def _load_raw_observations(self) -> gpd.GeoDataFrame:
        p_obs = self.tiles_dir / "observations_nist.parquet"
        if not p_obs.exists():
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:32610")

        gdf = gpd.read_parquet(p_obs)
        if gdf.crs is None or gdf.crs.to_epsg() != 32610:
            gdf = gdf.to_crs("EPSG:32610")

        # In observations_nist.parquet, elapsed_minutes=0.0 corresponds to 06:25 PST.
        # Ignition is 06:33 PST (8 minutes later).
        # Adjust elapsed minutes to align with 06:33 PST ignition:
        gdf["elapsed_minutes_adj"] = gdf["elapsed_minutes"] - 8.0
        return gdf

    def _load_raw_roadblocks(self) -> gpd.GeoDataFrame:
        p_rb = self.tiles_dir / "nist_roadblocks_a2.parquet"
        if not p_rb.exists():
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:32610")

        gdf = gpd.read_parquet(p_rb)
        if gdf.crs is None or gdf.crs.to_epsg() != 32610:
            gdf = gdf.to_crs("EPSG:32610")

        # Parse onset elapsed minutes from time_pst (relative to 06:33 PST)
        gdf["onset_elapsed_min"] = [parse_clock_to_elapsed_min(t) for t in gdf["time_pst"]]
        return gdf

    def _load_all_truth_hazards(self) -> list[TruthHazard]:
        hazards: list[TruthHazard] = []

        # 1. NIST Burnovers (23 incidents)
        p_bo = self.tiles_dir / "nist_burnovers_23.parquet"
        if p_bo.exists():
            gdf_bo = gpd.read_parquet(p_bo).to_crs("EPSG:32610")
            for _, row in gdf_bo.iterrows():
                onset = parse_clock_to_elapsed_min(row.get("datetime_str", "07:50"))
                geom = row.geometry if hasattr(row.geometry, "x") else row.geometry.centroid
                hazards.append(
                    TruthHazard(
                        hazard_id=str(row.get("burnover_id", f"BO_{len(hazards)}")),
                        hazard_type="burnover",
                        onset_elapsed_min=max(0.0, onset),
                        geometry_utm=geom,
                        description=str(row.get("name", "NIST Burnover Incident")),
                        source_ref=str(row.get("source_doc", "NIST Table 4-1")),
                    )
                )

        # 2. NIST Roadblocks (178 records)
        p_rb = self.tiles_dir / "nist_roadblocks_a2.parquet"
        if p_rb.exists():
            gdf_rb = gpd.read_parquet(p_rb).to_crs("EPSG:32610")
            for _, row in gdf_rb.iterrows():
                onset = parse_clock_to_elapsed_min(row.get("time_pst", "07:29"))
                geom = row.geometry if hasattr(row.geometry, "x") else row.geometry.centroid
                hazards.append(
                    TruthHazard(
                        hazard_id=str(row.get("source_id", f"RB_{len(hazards)}")),
                        hazard_type="roadblock",
                        onset_elapsed_min=max(0.0, onset),
                        geometry_utm=geom,
                        description=str(row.get("description", "NIST Law Enforcement Roadblock")),
                        source_ref="NIST Table A-2",
                    )
                )

        # 3. NIST Abandoned Vehicles (235 records)
        p_av = self.tiles_dir / "nist_abandoned_vehicles_h2.parquet"
        if p_av.exists():
            gdf_av = gpd.read_parquet(p_av).to_crs("EPSG:32610")
            for _, row in gdf_av.iterrows():
                # Vehicles in Table H-2 abandoned during peak corridor entrapment ~08:00 - 08:30 PST (87 - 117 min)
                geom = row.geometry if hasattr(row.geometry, "x") else row.geometry.centroid
                hazards.append(
                    TruthHazard(
                        hazard_id=str(row.get("av_id", f"AV_{len(hazards)}")),
                        hazard_type="abandoned_vehicle",
                        onset_elapsed_min=87.0,  # 08:00 PST town-wide overrun onset
                        geometry_utm=geom,
                        description=f"Abandoned vehicle on {row.get('street', 'corridor')}",
                        source_ref="NIST Table H-2",
                    )
                )

        return hazards

    @property
    def truth_hazards(self) -> list[TruthHazard]:
        return self._truth_hazards

    def step(self, current_step_min: float) -> CausalStepEnvironment:
        """Construct the causally filtered environment at clock time current_step_min."""
        t_now = float(current_step_min)
        t_obs_limit = t_now - self.latency_min

        # Filter observations to timestamp <= t_now - latency_min
        if len(self._raw_observations) > 0 and t_obs_limit >= 0.0:
            visible_obs = self._raw_observations[
                self._raw_observations["elapsed_minutes_adj"] <= t_obs_limit
            ].copy()
        else:
            visible_obs = gpd.GeoDataFrame(geometry=[], crs="EPSG:32610")

        # Filter active roadblocks to onset <= t_now
        if len(self._raw_roadblocks) > 0:
            active_rb = self._raw_roadblocks[
                self._raw_roadblocks["onset_elapsed_min"] <= t_now
            ].copy()
        else:
            active_rb = gpd.GeoDataFrame(geometry=[], crs="EPSG:32610")

        # Get field from provider
        field = self.provider.get_field(
            current_step_min=t_now,
            visible_observations=visible_obs,
        )

        return CausalStepEnvironment(
            step_elapsed_min=t_now,
            latency_min=self.latency_min,
            visible_observations=visible_obs,
            active_roadblocks=active_rb,
            arrival_field=field,
        )

    def get_closed_edges_for_roadblocks(
        self,
        active_roadblocks: gpd.GeoDataFrame,
        max_distance_m: float = 120.0,
    ) -> set[tuple[int, int]]:
        """Identify road graph edges severed by active roadblocks within spatial radius."""
        closed_edges: set[tuple[int, int]] = set()
        if len(active_roadblocks) == 0:
            return closed_edges

        rb_geoms = [
            row.geometry if hasattr(row.geometry, "x") else row.geometry.centroid
            for _, row in active_roadblocks.iterrows()
            if row.geometry is not None
        ]

        for pt in rb_geoms:
            hit_indices = self._edge_tree.query(pt.buffer(max_distance_m))
            for idx in hit_indices:
                e = self.road_graph.edges[idx]
                if e.geometry is not None and e.geometry.distance(pt) <= max_distance_m:
                    closed_edges.add((e.u, e.v))
                    closed_edges.add((e.v, e.u))

        return closed_edges

    def get_closed_edges_for_observations(
        self,
        visible_observations: gpd.GeoDataFrame,
        max_distance_m: float = 200.0,
    ) -> set[tuple[int, int]]:
        """Identify road graph edges severed by visible fire observations within spatial radius."""
        closed_edges: set[tuple[int, int]] = set()
        if len(visible_observations) == 0:
            return closed_edges

        obs_geoms = [
            row.geometry if hasattr(row.geometry, "x") else row.geometry.centroid
            for _, row in visible_observations.iterrows()
            if row.geometry is not None
        ]

        for pt in obs_geoms:
            hit_indices = self._edge_tree.query(pt.buffer(max_distance_m))
            for idx in hit_indices:
                e = self.road_graph.edges[idx]
                if e.geometry is not None and e.geometry.distance(pt) <= max_distance_m:
                    closed_edges.add((e.u, e.v))
                    closed_edges.add((e.v, e.u))

        return closed_edges
