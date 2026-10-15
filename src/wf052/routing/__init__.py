"""Time-aware wildland evacuation routing and dynamic traffic assignment module."""

from wf052.routing.congestion import (
    CongestionModel,
    EdgeCapacityInfo,
    TimeSlicedAssignmentEngine,
    ZoneClearanceSummary,
)
from wf052.routing.contracts import (
    ArrivalField,
    EvacuationRoute,
    ReplanResult,
    RouteEdge,
    RouteStatus,
    RoutingMode,
    validate_arrival_field,
)
from wf052.routing.demand import (
    EvacuationDemandModel,
    EvacuationZone,
    TimeSliceDemand,
    calculate_departure_schedule,
)
from wf052.routing.en_route import VehiclePosition, replan_en_route_vehicle
from wf052.routing.graph import EdgeRecord, RoadGraph, load_camp_fire_road_graph
from wf052.routing.hazard_check import (
    HazardCheckResult,
    HazardPoint,
    score_route_against_hazards,
)
from wf052.routing.replan import ReplanEngine
from wf052.routing.router import TimeDependentRouter
from wf052.routing.synthetic_fire import (
    create_expanding_circle_arrival_field,
    create_unburned_arrival_field,
    create_wind_ellipse_arrival_field,
)
from wf052.routing.validator import IndependentRouteValidator, ValidationResult

__all__ = [
    "ArrivalField",
    "CongestionModel",
    "EdgeCapacityInfo",
    "EdgeRecord",
    "EvacuationDemandModel",
    "EvacuationRoute",
    "EvacuationZone",
    "HazardCheckResult",
    "HazardPoint",
    "IndependentRouteValidator",
    "ReplanEngine",
    "ReplanResult",
    "RoadGraph",
    "RouteEdge",
    "RouteStatus",
    "RoutingMode",
    "TimeDependentRouter",
    "TimeSliceDemand",
    "TimeSlicedAssignmentEngine",
    "ValidationResult",
    "VehiclePosition",
    "ZoneClearanceSummary",
    "calculate_departure_schedule",
    "create_expanding_circle_arrival_field",
    "create_unburned_arrival_field",
    "create_wind_ellipse_arrival_field",
    "load_camp_fire_road_graph",
    "replan_en_route_vehicle",
    "score_route_against_hazards",
    "validate_arrival_field",
]
