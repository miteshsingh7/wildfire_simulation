"""Road graph ingestion, pre-indexing, and edge fire arrival mapping.

Builds a time-dependent directed road network from OSMnx / GeoParquet for the
Camp Fire domain (EPSG:32610 UTM Zone 10N).
Flags contemporary OSM snapshot (2018 Overpass attic data not retrieved).
Pre-indexes spatial intersection of each edge geometry onto the 60m ArrivalField
grid for sub-20ms edge fire arrival lookup.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import shapely.geometry
import xarray as xr

from wf052.routing.contracts import validate_arrival_field

logger = logging.getLogger(__name__)

# Standard evacuation travel speed defaults by OSM highway tag (mph -> m/s)
MPH_TO_MPS = 0.44704
DEFAULT_SPEEDS_MPH: dict[str, float] = {
    "motorway": 65.0,
    "motorway_link": 45.0,
    "trunk": 55.0,
    "trunk_link": 40.0,
    "primary": 45.0,
    "primary_link": 30.0,
    "secondary": 35.0,
    "secondary_link": 25.0,
    "tertiary": 30.0,
    "tertiary_link": 20.0,
    "residential": 25.0,
    "living_street": 15.0,
    "unclassified": 25.0,
    "service": 15.0,
}
FALLBACK_SPEED_MPH = 25.0

# Canonical Camp Fire evacuation landmarks (EPSG:32610 UTM Zone 10N Easting, Northing)
# Sourced from NIST TN 2135 and USGS GNIS
CAMP_FIRE_LANDMARKS_UTM: dict[str, tuple[float, float]] = {
    "Paradise": (618047.8, 4401983.8),  # Central Paradise (Skyway / Pearson)
    "Upper_Paradise": (620099.0, 4404502.5),  # Upper Paradise (Skyway / Clark)
    "Magalia": (621690.1, 4407869.3),  # Magalia center
    "Concow": (627156.9, 4399353.8),  # Concow / Camelot
    "Stirling_City": (625821.5, 4418516.6),  # Stirling City
    "Pulga": (633391.0, 4407338.0),  # Ignition origin
}


def parse_speed_mps(maxspeed_val: Any, highway_val: Any) -> float:
    """Parse edge maximum speed into meters per second.

    Prefers numerical maxspeed if available, otherwise maps highway classification.
    """
    if maxspeed_val is not None and not (
        isinstance(maxspeed_val, float) and np.isnan(maxspeed_val)
    ):
        s_str = str(maxspeed_val).lower().strip()
        # Handle comma-separated lists (e.g. '55 mph,35 mph') by taking the minimum
        parts = [p.strip() for p in s_str.split(",") if p.strip()]
        numeric_speeds = []
        for p in parts:
            match = re.search(r"(\d+(?:\.\d+)?)", p)
            if match:
                val = float(match.group(1))
                # Check for km/h vs mph
                if "km/h" in p or "kph" in p:
                    numeric_speeds.append(val / 3.6)
                else:
                    numeric_speeds.append(val * MPH_TO_MPS)
        if numeric_speeds:
            return float(min(numeric_speeds))

    # Fallback to highway type
    if highway_val is not None and not (isinstance(highway_val, float) and np.isnan(highway_val)):
        hw_str = str(highway_val).lower().strip()
        for hw_part in hw_str.split(","):
            hw_clean = hw_part.strip()
            if hw_clean in DEFAULT_SPEEDS_MPH:
                return DEFAULT_SPEEDS_MPH[hw_clean] * MPH_TO_MPS

    return FALLBACK_SPEED_MPH * MPH_TO_MPS


@dataclass
class EdgeRecord:
    """Internal representation of a directed road graph edge."""

    edge_idx: int
    u: int
    v: int
    key: int
    length_m: float
    speed_mps: float
    traversal_time_min: float
    name: str
    highway: str
    geometry: shapely.geometry.LineString
    cell_rows: np.ndarray  # (N,) int array of intersected grid rows
    cell_cols: np.ndarray  # (N,) int array of intersected grid cols


class RoadGraph:
    """Directed, time-dependent road network for wildland evacuation routing."""

    def __init__(
        self,
        nodes_gdf: gpd.GeoDataFrame,
        edges_gdf: gpd.GeoDataFrame,
        grid_x_coords: np.ndarray | None = None,
        grid_y_coords: np.ndarray | None = None,
        osm_snapshot_flag: str = "contemporary OSM snapshot (2018 Overpass attic data not retrieved)",
    ) -> None:
        self.osm_snapshot_flag = osm_snapshot_flag
        self.nodes_gdf = nodes_gdf.copy()
        self.edges_gdf = edges_gdf.copy()

        # Node index mapping: node_id -> (x, y)
        self.nodes: dict[int, tuple[float, float]] = {}
        for node_id, geom in zip(self.nodes_gdf.index, self.nodes_gdf.geometry, strict=True):
            self.nodes[int(node_id)] = (float(geom.x), float(geom.y))

        # Adjacency list: u -> list of (v, key, traversal_time_min, edge_idx)
        self.adj: dict[int, list[tuple[int, int, float, int]]] = {n: [] for n in self.nodes}
        self.edges: list[EdgeRecord] = []

        # Grid specification for spatial indexing
        self.grid_x: np.ndarray | None = grid_x_coords
        self.grid_y: np.ndarray | None = grid_y_coords

        self._build_graph_and_spatial_index()

        # Evacuation landmarks & boundary exits
        self.named_origins: dict[str, int] = {}
        self.boundary_destinations: dict[str, int] = {}
        self._identify_landmarks_and_exits()

    def _build_graph_and_spatial_index(self) -> None:
        """Process edges, compute traversal times, and pre-index intersected grid cells."""
        has_grid = (
            self.grid_x is not None
            and self.grid_y is not None
            and len(self.grid_x) > 1
            and len(self.grid_y) > 1
        )
        if has_grid:
            assert self.grid_x is not None and self.grid_y is not None
            dx = float(abs(self.grid_x[1] - self.grid_x[0]))
            dy = float(abs(self.grid_y[1] - self.grid_y[0]))
            x_min = float(self.grid_x.min()) - 0.5 * dx
            y_max = float(self.grid_y.max()) + 0.5 * dy
            num_rows = len(self.grid_y)
            num_cols = len(self.grid_x)

        edge_idx = 0
        for idx, row in self.edges_gdf.iterrows():
            if isinstance(idx, tuple) and len(idx) >= 3:
                u, v, k = int(idx[0]), int(idx[1]), int(idx[2])
            else:
                u = int(row.get("u", 0))
                v = int(row.get("v", 0))
                k = int(row.get("key", 0))

            if u not in self.nodes or v not in self.nodes:
                continue

            geom = row.geometry
            if geom is None or geom.is_empty:
                geom = shapely.geometry.LineString([self.nodes[u], self.nodes[v]])

            length_m = float(row.get("length", geom.length))
            if length_m <= 0.0:
                length_m = float(geom.length) if geom.length > 0 else 1.0

            speed_mps = parse_speed_mps(row.get("maxspeed", None), row.get("highway", None))
            traversal_time_min = (length_m / speed_mps) / 60.0

            # Precompute intersected grid cells
            c_rows: list[int] = []
            c_cols: list[int] = []
            if has_grid:
                # Sample points along geometry every 25m
                if geom.length <= 30.0:
                    pts = list(geom.coords)
                else:
                    n_pts = int(np.ceil(geom.length / 25.0)) + 1
                    dists = np.linspace(0, geom.length, n_pts)
                    pts = [(float(p.x), float(p.y)) for p in [geom.interpolate(d) for d in dists]]

                seen_cells: set[tuple[int, int]] = set()
                for px, py in pts:
                    col = int((px - x_min) / dx)
                    row_idx = int((y_max - py) / dy)
                    if 0 <= row_idx < num_rows and 0 <= col < num_cols:
                        cell = (row_idx, col)
                        if cell not in seen_cells:
                            seen_cells.add(cell)
                            c_rows.append(row_idx)
                            c_cols.append(col)

            rec = EdgeRecord(
                edge_idx=edge_idx,
                u=u,
                v=v,
                key=k,
                length_m=length_m,
                speed_mps=speed_mps,
                traversal_time_min=traversal_time_min,
                name=str(row.get("name", "")),
                highway=str(row.get("highway", "")),
                geometry=geom,
                cell_rows=np.array(c_rows, dtype=np.int32),
                cell_cols=np.array(c_cols, dtype=np.int32),
            )
            self.edges.append(rec)
            self.adj[u].append((v, k, traversal_time_min, edge_idx))
            edge_idx += 1

    def _identify_landmarks_and_exits(self) -> None:
        """Map canonical Camp Fire origins and boundary highway exits to graph nodes."""
        node_ids = np.array(list(self.nodes.keys()), dtype=np.int64)
        node_coords = np.array([self.nodes[nid] for nid in node_ids], dtype=np.float64)

        # 1. Map origins
        for name, (x, y) in CAMP_FIRE_LANDMARKS_UTM.items():
            dists = np.hypot(node_coords[:, 0] - x, node_coords[:, 1] - y)
            nearest_idx = int(np.argmin(dists))
            self.named_origins[name] = int(node_ids[nearest_idx])

        # 2. Identify boundary highway exits
        # Nodes within 1500m of the graph boundary bounding box
        min_x = node_coords[:, 0].min()
        max_x = node_coords[:, 0].max()
        min_y = node_coords[:, 1].min()
        max_y = node_coords[:, 1].max()

        boundary_margin = 1500.0
        is_west = node_coords[:, 0] <= min_x + boundary_margin
        is_east = node_coords[:, 0] >= max_x - boundary_margin
        is_south = node_coords[:, 1] <= min_y + boundary_margin
        is_north = node_coords[:, 1] >= max_y - boundary_margin

        # Prioritize major highway edges (trunk, primary, secondary, tertiary)
        major_highways = {"trunk", "primary", "secondary", "tertiary"}
        boundary_nodes: dict[str, list[int]] = {
            "West_Chico": [],
            "South_Oroville": [],
            "East_FeatherRiver": [],
            "North_Lassen": [],
        }

        for edge in self.edges:
            hw = edge.highway.lower()
            if any(m in hw for m in major_highways):
                if is_west[np.where(node_ids == edge.v)[0][0]]:
                    boundary_nodes["West_Chico"].append(edge.v)
                if is_south[np.where(node_ids == edge.v)[0][0]]:
                    boundary_nodes["South_Oroville"].append(edge.v)
                if is_east[np.where(node_ids == edge.v)[0][0]]:
                    boundary_nodes["East_FeatherRiver"].append(edge.v)
                if is_north[np.where(node_ids == edge.v)[0][0]]:
                    boundary_nodes["North_Lassen"].append(edge.v)

        for direction, n_list in boundary_nodes.items():
            if n_list:
                # Pick the most connected or first major node
                self.boundary_destinations[direction] = n_list[0]
            else:
                # Fallback to extreme geometric boundary node
                if direction == "West_Chico":
                    self.boundary_destinations[direction] = int(
                        node_ids[np.argmin(node_coords[:, 0])]
                    )
                elif direction == "South_Oroville":
                    self.boundary_destinations[direction] = int(
                        node_ids[np.argmin(node_coords[:, 1])]
                    )
                elif direction == "East_FeatherRiver":
                    self.boundary_destinations[direction] = int(
                        node_ids[np.argmax(node_coords[:, 0])]
                    )
                elif direction == "North_Lassen":
                    self.boundary_destinations[direction] = int(
                        node_ids[np.argmax(node_coords[:, 1])]
                    )

    def compute_edge_fire_arrivals(self, arrival_field: xr.Dataset) -> np.ndarray:
        """Compute the minimum fire arrival time q10 for each edge in the graph.

        Returns:
            np.ndarray of shape (num_edges,) containing min q10 arrival in minutes
            (+inf if fire never touches the edge).
        """
        validate_arrival_field(arrival_field)
        q10_grid = arrival_field["arrival_q10"].values

        num_edges = len(self.edges)
        edge_arrivals = np.full(num_edges, np.inf, dtype=np.float64)

        for i, edge in enumerate(self.edges):
            if len(edge.cell_rows) > 0:
                vals = q10_grid[edge.cell_rows, edge.cell_cols]
                edge_arrivals[i] = float(np.min(vals))

        return edge_arrivals

    def check_destination_safety(
        self,
        arrival_field: xr.Dataset,
        safety_buffer_km: float = 2.0,
        horizon_minutes: float = 720.0,
    ) -> tuple[set[int], dict[int, float]]:
        """Check all boundary destination exits against fire encroachment within X km.

        Returns:
            (safe_destination_node_ids, dict of unsafe_destination_node_id -> min_arrival_time_within_buffer)
        """
        validate_arrival_field(arrival_field)
        q10_grid = arrival_field["arrival_q10"].values
        grid_x = arrival_field["x"].values
        grid_y = arrival_field["y"].values

        safe_dests: set[int] = set()
        unsafe_dests: dict[int, float] = {}
        buffer_m = safety_buffer_km * 1000.0

        for _name, dest_id in self.boundary_destinations.items():
            if dest_id not in self.nodes:
                continue
            dx_node, dy_node = self.nodes[dest_id]

            # Mask cells within safety radius
            dist_sq = (grid_x[None, :] - dx_node) ** 2 + (grid_y[:, None] - dy_node) ** 2
            mask = dist_sq <= (buffer_m * buffer_m)

            if np.any(mask):
                min_arr = float(np.min(q10_grid[mask]))
                if min_arr <= horizon_minutes:
                    unsafe_dests[dest_id] = min_arr
                else:
                    safe_dests.add(dest_id)
            else:
                safe_dests.add(dest_id)

        return safe_dests, unsafe_dests


def load_camp_fire_road_graph(
    tiles_dir: Path | str = "data/tiles",
    grid_x_coords: np.ndarray | None = None,
    grid_y_coords: np.ndarray | None = None,
    full_detail: bool = True,
) -> RoadGraph:
    """Load the Camp Fire road graph from GeoParquet tiles.

    Args:
        tiles_dir: Directory containing roads GeoParquet files.
        grid_x_coords: 1D array of 60m grid easting coordinates.
        grid_y_coords: 1D array of 60m grid northing coordinates.
        full_detail: If True, loads full-detail graph with service roads (12,379 edges).
                     If False, loads simplified/fast drive graph (8,085 edges).
    """
    p_tiles = Path(tiles_dir)
    if full_detail:
        nodes_path = p_tiles / "roads_nodes_full.parquet"
        edges_path = p_tiles / "roads_edges_full.parquet"
        if not nodes_path.exists() or not edges_path.exists():
            # Fallback to standard if full not yet generated
            nodes_path = p_tiles / "roads_nodes.parquet"
            edges_path = p_tiles / "roads_edges.parquet"
    else:
        nodes_path = p_tiles / "roads_nodes.parquet"
        edges_path = p_tiles / "roads_edges.parquet"

    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            f"Road parquet files not found in {tiles_dir}. Expected {nodes_path} and {edges_path}."
        )

    nodes_gdf = gpd.read_parquet(nodes_path)
    edges_gdf = gpd.read_parquet(edges_path)

    # If grid coordinates not provided, synthesize default 60m grid from bbox
    if grid_x_coords is None or grid_y_coords is None:
        bbox = nodes_gdf.total_bounds  # [minx, miny, maxx, maxy]
        grid_x_coords = np.arange(bbox[0], bbox[2] + 60.0, 60.0)
        grid_y_coords = np.arange(bbox[3], bbox[1] - 60.0, -60.0)

    mode_label = "full-detail" if full_detail else "simplified/fast"
    return RoadGraph(
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        grid_x_coords=grid_x_coords,
        grid_y_coords=grid_y_coords,
        osm_snapshot_flag=f"contemporary OSM snapshot ({mode_label}; 2018 Overpass attic data not retrieved)",
    )
