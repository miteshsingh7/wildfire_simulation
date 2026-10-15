"""Contract specifications, data structures, and validation for the time-aware evacuation router.

CONTRACT:
ArrivalField = xarray.Dataset with:
  - Data variables:
      - arrival_q10: (y, x) float, arrival time in minutes since ignition on the 60m grid (+inf where fire never arrives)
      - arrival_q50: (y, x) float, median arrival time in minutes since ignition (+inf where unburned)
      - arrival_q90: (y, x) float, 90th percentile arrival time in minutes since ignition (+inf where unburned)
  - Coordinates:
      - y: northing in projected CRS (EPSG:32610 UTM Zone 10N)
      - x: easting in projected CRS (EPSG:32610 UTM Zone 10N)
  - Attributes:
      - crs: CRS string (e.g. 'EPSG:32610')
      - ignition_time: ISO timestamp string (e.g. '2018-11-08T14:33:00Z')
The router reads only this structure.
"""

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import xarray as xr

ArrivalField = xr.Dataset


class RoutingMode(StrEnum):
    """Routing optimization modes."""

    FASTEST_SAFE = "fastest_safe"  # (a) Fastest route with margin >= B minutes
    MAX_MIN_MARGIN = "max_min_margin"  # (b) Route that maximizes the minimum margin


class RouteStatus(StrEnum):
    """Evacuation route feasibility status."""

    SAFE = "safe"  # Route satisfies margin >= buffer minutes on all edges
    BEST_EFFORT = "best_effort"  # Route maximizes minimum margin (margin < buffer)
    NO_ROUTE = "no_route"  # Graph disconnected or completely impassable


@dataclass(frozen=True)
class RouteEdge:
    """Directed edge traversal segment in an evacuation route."""

    u: int
    v: int
    key: int
    length_m: float
    speed_mps: float
    traversal_time_min: float
    entry_time_min: float
    exit_time_min: float
    fire_arrival_min: float
    margin_min: float  # fire_arrival_min - exit_time_min


@dataclass(frozen=True)
class EvacuationRoute:
    """Complete computed evacuation route from origin to destination."""

    origin_id: int
    destination_id: int
    departure_time_min: float
    arrival_time_min: float
    total_travel_time_min: float
    min_margin_min: float
    bottleneck_edge: tuple[int, int, int] | None
    edges: tuple[RouteEdge, ...]
    node_ids: tuple[int, ...]
    status: RouteStatus
    mode_used: RoutingMode
    geometry_wkt: str | None = None


@dataclass(frozen=True)
class ReplanResult:
    """Structured result returned by the evacuation re-plan entry point."""

    timestamp_utc: str
    departure_time_min: float
    buffer_minutes: float
    routes: dict[str, EvacuationRoute]  # origin_name -> EvacuationRoute
    replan_latency_sec: float
    arrival_field_hash: str
    metadata: dict[str, str | float | int] = field(default_factory=dict)


def validate_arrival_field(ds: xr.Dataset) -> None:
    """Validate that an xarray Dataset conforms strictly to the ArrivalField contract.

    Raises:
        ValueError or TypeError if the dataset does not satisfy the contract.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"ArrivalField must be an xarray.Dataset, got {type(ds).__name__}")

    required_vars = ("arrival_q10", "arrival_q50", "arrival_q90")
    for var in required_vars:
        if var not in ds.data_vars:
            raise ValueError(f"ArrivalField missing required data variable: '{var}'")
        arr = ds[var].values
        if arr.ndim != 2:
            raise ValueError(f"ArrivalField variable '{var}' must be 2D, got shape {arr.shape}")
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError(
                f"ArrivalField variable '{var}' must have float dtype, got {arr.dtype}"
            )

    if "y" not in ds.coords or "x" not in ds.coords:
        raise ValueError("ArrivalField must have 1D coordinates 'y' and 'x'")

    if "crs" not in ds.attrs:
        raise ValueError("ArrivalField attributes must contain 'crs' string")

    if "ignition_time" not in ds.attrs:
        raise ValueError("ArrivalField attributes must contain 'ignition_time' timestamp string")
