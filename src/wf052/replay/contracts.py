"""Typed contracts and data classes for the Camp Fire causal replay harness."""

from dataclasses import asdict, dataclass
from typing import Any

import geopandas as gpd
import shapely.geometry

from wf052.routing.contracts import ArrivalField, EvacuationRoute


@dataclass(frozen=True)
class TruthHazard:
    """Ground truth hazard observation from NIST TN 2252 empirical extracts.

    Source categories:
    - Burnovers (23 incidents, NIST Table 4-1 / Table 23)
    - Roadblocks (178 records, NIST Table A-2)
    - Abandoned vehicles (235 records, NIST Table H-2)
    """

    hazard_id: str
    hazard_type: str  # "burnover", "roadblock", "abandoned_vehicle"
    onset_elapsed_min: float  # Elapsed minutes since 06:33 PST ignition (14:33 UTC)
    geometry_utm: shapely.geometry.Point
    description: str
    source_ref: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hazard_id": self.hazard_id,
            "hazard_type": self.hazard_type,
            "onset_elapsed_min": self.onset_elapsed_min,
            "x": self.geometry_utm.x,
            "y": self.geometry_utm.y,
            "description": self.description,
            "source_ref": self.source_ref,
        }


@dataclass(frozen=True)
class CausalStepEnvironment:
    """Isolated, causally filtered environmental state at simulation clock time t."""

    step_elapsed_min: float
    latency_min: float
    visible_observations: gpd.GeoDataFrame
    active_roadblocks: gpd.GeoDataFrame
    arrival_field: ArrivalField

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_elapsed_min": self.step_elapsed_min,
            "latency_min": self.latency_min,
            "visible_observations_count": len(self.visible_observations),
            "active_roadblocks_count": len(self.active_roadblocks),
        }


@dataclass(frozen=True)
class ReplayOriginResult:
    """Evaluation output for a single origin under a specific evacuation method."""

    origin_id: int
    method_name: str
    departure_time_min: float
    route: EvacuationRoute | None
    travel_time_min: float
    hazard_miss: bool
    min_margin_min: float
    replan_count: int
    warning_lead_time_min: float | None
    flagged_before_onset: bool
    hazard_misses_by_type: dict[str, bool] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MethodMetrics:
    """Aggregated evaluation metrics with bootstrap confidence intervals."""

    method_name: str
    total_origins: int
    hazard_miss_count: int
    hazard_miss_rate: float
    hazard_miss_ci: tuple[float, float]
    extra_travel_time_mean: float
    extra_travel_time_ci: tuple[float, float]
    warning_lead_time_mean: float
    warning_lead_time_ci: tuple[float, float]
    warning_flagged_fraction: float
    replan_count_mean: float
    min_margin_median: float
    min_margin_p10: float
    mean_step_latency_ms: float
    miss_count_burnovers: int = 0
    miss_count_roadblocks: int = 0
    miss_count_abandoned: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
