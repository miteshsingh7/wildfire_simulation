"""Plain typed Python tools for agent state machine.

Rules:
- The LLM never computes routes, geometry or numbers.
- All routing, validation, geometry, and metric computations happen inside these typed tools.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from wf052.agent.contracts import (
    ArrivalFieldMeta,
    ExposureSummary,
    ObservationConsistencyResult,
    RouteLevelExposure,
    WeatherReport,
)
from wf052.routing.contracts import (
    ArrivalField,
    EvacuationRoute,
    RouteStatus,
    validate_arrival_field,
)
from wf052.routing.graph import RoadGraph
from wf052.routing.router import TimeDependentRouter
from wf052.routing.slowdown import compute_critical_slowdown_k_star
from wf052.routing.synthetic_fire import (
    create_expanding_circle_arrival_field,
    create_unburned_arrival_field,
    create_wind_ellipse_arrival_field,
)
from wf052.routing.validator import IndependentRouteValidator, ValidationResult

# In-memory weather cache stub (HRRR cache)
_HRRR_WEATHER_CACHE: dict[str, WeatherReport] = {
    "default": WeatherReport(
        wind_speed_mps=6.5,
        wind_dir_deg=45.0,
        temp_c=22.0,
        rh_pct=18.0,
        is_stub=True,
        station_id="HRRR_STUB_DEFAULT",
    ),
    "high_wind": WeatherReport(
        wind_speed_mps=15.0,
        wind_dir_deg=65.0,
        temp_c=28.0,
        rh_pct=10.0,
        is_stub=True,
        station_id="HRRR_STUB_HIGH_WIND",
    ),
}


def get_weather(
    location: str | tuple[float, float] = "default",
    timestamp: str | None = None,
    custom_weather: WeatherReport | None = None,
) -> WeatherReport:
    """Retrieve weather observations or forecast from cached HRRR stub.

    Args:
        location: Named weather scenario or (x, y) coordinate.
        timestamp: Optional timestamp for time-sliced weather.
        custom_weather: Override report for testing.
    """
    if custom_weather is not None:
        return custom_weather

    loc_key = str(location).lower()
    if loc_key in _HRRR_WEATHER_CACHE:
        return _HRRR_WEATHER_CACHE[loc_key]
    return _HRRR_WEATHER_CACHE["default"]


def check_field_vs_observations(
    arrival_field: ArrivalField,
    detections: Any,
    tolerance_minutes: float = 15.0,
    horizon_minutes: float = 720.0,
) -> ObservationConsistencyResult:
    """Verify arrival field consistency against field observations (e.g. NIST points/polygons).

    Flags low confidence when:
    1. Any detection arrives earlier than field's arrival_q10 by more than tolerance_minutes.
    2. Any detection lies where the field predicts no arrival (inf, unburned, or outside grid).

    Args:
        arrival_field: Arrival field xarray Dataset.
        detections: GeoDataFrame, DataFrame, or list of detection dicts with coords and timestamps.
        tolerance_minutes: Config tolerance threshold in minutes (default 15.0).
        horizon_minutes: Simulation time horizon in minutes (default 720.0).

    Returns:
        ObservationConsistencyResult with consistency flag, counts, and diagnostic reason.
    """
    if detections is None or len(detections) == 0:
        return ObservationConsistencyResult(
            is_consistent=True,
            is_low_confidence=False,
            reason="No detections provided for consistency check.",
            earlier_detections_count=0,
            unmodeled_detections_count=0,
            total_detections=0,
        )

    var_name = "arrival_q10" if "arrival_q10" in arrival_field else "arrival_q50"
    q_arr = arrival_field[var_name].values
    gx = arrival_field["x"].values
    gy = arrival_field["y"].values

    min_x, max_x = float(np.min(gx)), float(np.max(gx))
    min_y, max_y = float(np.min(gy)), float(np.max(gy))
    dx = float(abs(gx[1] - gx[0])) if len(gx) > 1 else 60.0
    dy = float(abs(gy[1] - gy[0])) if len(gy) > 1 else 60.0

    earlier_count = 0
    unmodeled_count = 0
    total = 0

    records: list[Any] = []
    if hasattr(detections, "iterrows"):
        for _, row in detections.iterrows():
            records.append(row)
    elif isinstance(detections, list):
        records = detections
    else:
        records = [detections]

    for det in records:
        total += 1
        x_val: float | None = None
        y_val: float | None = None
        t_obs: float | None = None

        if hasattr(det, "geometry") and det.geometry is not None:
            geom = det.geometry
            if hasattr(geom, "x") and hasattr(geom, "y"):
                x_val = float(geom.x)
                y_val = float(geom.y)
            elif hasattr(geom, "centroid"):
                x_val = float(geom.centroid.x)
                y_val = float(geom.centroid.y)
        if x_val is None:
            if hasattr(det, "utm_x") and hasattr(det, "utm_y"):
                x_val = float(det.utm_x)
                y_val = float(det.utm_y)
            elif isinstance(det, dict) and "utm_x" in det and "utm_y" in det:
                x_val = float(det["utm_x"])
                y_val = float(det["utm_y"])
            elif isinstance(det, dict) and "x" in det and "y" in det:
                x_val = float(det["x"])
                y_val = float(det["y"])
            elif hasattr(det, "x") and hasattr(det, "y"):
                x_val = float(det.x)
                y_val = float(det.y)

        if hasattr(det, "elapsed_minutes"):
            t_obs = float(det.elapsed_minutes)
        elif isinstance(det, dict) and "elapsed_minutes" in det:
            t_obs = float(det["elapsed_minutes"])
        elif hasattr(det, "time_min"):
            t_obs = float(det.time_min)
        elif isinstance(det, dict) and "time_min" in det:
            t_obs = float(det["time_min"])
        elif hasattr(det, "t_min"):
            t_obs = float(det.t_min)
        elif isinstance(det, dict) and "t_min" in det:
            t_obs = float(det["t_min"])

        if x_val is None or y_val is None or t_obs is None:
            continue

        if x_val < min_x - dx or x_val > max_x + dx or y_val < min_y - dy or y_val > max_y + dy:
            unmodeled_count += 1
            continue

        ix = int(np.argmin(np.abs(gx - x_val)))
        iy = int(np.argmin(np.abs(gy - y_val)))

        t_field = float(q_arr[iy, ix])

        if not np.isfinite(t_field) or t_field >= horizon_minutes:
            unmodeled_count += 1
        elif t_obs < t_field - tolerance_minutes:
            earlier_count += 1

    is_low_conf = earlier_count > 0 or unmodeled_count > 0
    is_consistent = not is_low_conf

    if is_low_conf:
        reasons = []
        if earlier_count > 0:
            reasons.append(
                f"{earlier_count} detection(s) arrived earlier than field q10 by > {tolerance_minutes:.1f} min"
            )
        if unmodeled_count > 0:
            reasons.append(
                f"{unmodeled_count} detection(s) lie where field predicts no arrival / unmodeled"
            )
        reason_str = f"Observation inconsistency: {'; '.join(reasons)}."
    else:
        reason_str = f"Field consistent with {total} observation detection(s) within {tolerance_minutes:.1f} min."

    return ObservationConsistencyResult(
        is_consistent=is_consistent,
        is_low_confidence=is_low_conf,
        reason=reason_str,
        earlier_detections_count=earlier_count,
        unmodeled_detections_count=unmodeled_count,
        total_detections=total,
    )


def get_arrival_field(
    mode: str = "synthetic",
    config: dict[str, Any] | None = None,
    road_graph: RoadGraph | None = None,
    custom_field: ArrivalField | None = None,
    detections: Any | None = None,
    min_spread_area_cells: int = 5,
    observation_tolerance_minutes: float = 15.0,
) -> tuple[ArrivalField, ArrivalFieldMeta]:
    """Retrieve or generate an ArrivalField and evaluate its confidence.

    Args:
        mode: 'synthetic_circle', 'synthetic_ellipse', 'unburned', or 'custom'.
        config: Configuration dict for ignition and spread rate.
        road_graph: Road network graph supplying spatial grid bounds.
        custom_field: Direct ArrivalField for testing.
        detections: Optional field observations (NIST points) to check consistency against.
        min_spread_area_cells: Threshold below which field is flagged low confidence.
        observation_tolerance_minutes: Observation vs q10 tolerance.
    """
    if custom_field is not None:
        arr_field = custom_field
    else:
        cfg = config or {}
        if road_graph is None or road_graph.grid_x is None or road_graph.grid_y is None:
            gx = np.linspace(-500.0, 1500.0, 40)
            gy = np.linspace(1000.0, -1000.0, 40)
        else:
            gx, gy = road_graph.grid_x, road_graph.grid_y

        if mode == "unburned":
            arr_field = create_unburned_arrival_field(x_coords=gx, y_coords=gy)
        elif mode == "synthetic_circle":
            arr_field = create_expanding_circle_arrival_field(
                x_coords=gx,
                y_coords=gy,
                ignition_x=cfg.get("ignition_x", 0.0),
                ignition_y=cfg.get("ignition_y", 0.0),
                spread_rate_m_per_min=cfg.get("spread_rate_m_per_min", 25.0),
                horizon_minutes=cfg.get("horizon_minutes", 720.0),
            )
        else:
            # synthetic_ellipse default
            arr_field = create_wind_ellipse_arrival_field(
                x_coords=gx,
                y_coords=gy,
                ignition_x=cfg.get("ignition_x", 0.0),
                ignition_y=cfg.get("ignition_y", 0.0),
                wind_speed_mps=cfg.get("wind_speed_mps", 8.0),
                wind_dir_deg=cfg.get("wind_dir_deg", 45.0),
                horizon_minutes=cfg.get("horizon_minutes", 720.0),
            )

    validate_arrival_field(arr_field)

    # Compute spread metrics
    data_var = "arrival_q50" if "arrival_q50" in arr_field else "arrival_q10" if "arrival_q10" in arr_field else next(iter(arr_field.data_vars.keys()))
    arr_vals = arr_field[data_var].values
    burned_mask = np.isfinite(arr_vals) & (arr_vals < 720.0)
    burned_cells = int(np.sum(burned_mask))
    total_cells = int(arr_vals.size)
    min_arr = float(np.min(arr_vals)) if burned_cells > 0 else float("inf")

    # Low confidence triggers:
    # 1. Shows minimal spread beyond ignition area (<= 5 cells)
    # 2. Inconsistent with field observations (earlier detection or unmodeled location)
    reasons: list[str] = []
    is_low_conf = False

    if burned_cells <= min_spread_area_cells:
        is_low_conf = True
        reasons.append(
            f"Fire model indicates minimal or zero spread ({burned_cells} cells <= {min_spread_area_cells} threshold); "
            "spread may be stalled or observation incomplete."
        )

    obs_result: ObservationConsistencyResult | None = None
    if detections is not None:
        obs_result = check_field_vs_observations(
            arr_field, detections, tolerance_minutes=observation_tolerance_minutes
        )
        if obs_result.is_low_confidence:
            is_low_conf = True
            reasons.append(obs_result.reason)

    combined_reason = " | ".join(reasons)

    meta = ArrivalFieldMeta(
        is_low_confidence=is_low_conf,
        burned_cells=burned_cells,
        total_cells=total_cells,
        min_arrival_min=min_arr,
        reason=combined_reason,
        observation_consistency=obs_result,
    )

    return arr_field, meta


def plan_routes(
    road_graph: RoadGraph,
    origins: Sequence[int],
    destinations: Sequence[int] | set[int],
    departure_time: float,
    arrival_field: ArrivalField,
    buffer_minutes: float = 15.0,
    closed_edges: set[tuple[int, int]] | None = None,
    conservative_mode: bool = False,
    conservative_buffer_minutes: float = 30.0,
) -> dict[int, EvacuationRoute]:
    """Execute time-dependent evacuation routing using the existing router."""
    dest_set = set(destinations)
    effective_buffer = (
        max(buffer_minutes, conservative_buffer_minutes)
        if conservative_mode
        else buffer_minutes
    )
    edge_arrivals = road_graph.compute_edge_fire_arrivals(arrival_field)
    router = TimeDependentRouter(road_graph=road_graph)

    dynamic_times = None
    if closed_edges:
        dynamic_times = np.array([e.traversal_time_min for e in road_graph.edges], dtype=np.float64)
        for idx, e in enumerate(road_graph.edges):
            if (e.u, e.v) in closed_edges or (e.v, e.u) in closed_edges:
                dynamic_times[idx] = float("inf")

    routes: dict[int, EvacuationRoute] = {}
    for orig in origins:
        route = router.find_route(
            origin_node=orig,
            destination_nodes=dest_set,
            t0=departure_time,
            buffer_minutes=effective_buffer,
            edge_fire_arrivals=edge_arrivals,
            dynamic_edge_traversal_times=dynamic_times,
        )
        if route is not None:
            routes[orig] = route

    return routes


def validate_routes(
    road_graph: RoadGraph,
    routes: dict[int, EvacuationRoute],
    arrival_field: ArrivalField,
    buffer_minutes: float = 15.0,
) -> dict[int, ValidationResult]:
    """Verify route validity using the existing independent route validator."""
    validator = IndependentRouteValidator()
    results: dict[int, ValidationResult] = {}
    for orig, route in routes.items():
        res = validator.validate(
            route=route,
            arrival_field=arrival_field,
            road_graph=road_graph,
            buffer_minutes=buffer_minutes,
        )
        results[orig] = res
    return results


def compute_exposure(
    routes: dict[int, EvacuationRoute],
    road_graph: RoadGraph,
    buffer_minutes: float = 15.0,
) -> RouteLevelExposure:
    """Compute route-level safety margin exposure and qualitative traffic conditions.

    Note: This computes static route-level margins (t_fire - t_exit); it does NOT
    track dynamic time-sliced vehicle cohort trajectories.
    """
    trapped_count = 0
    safe_count = 0
    margins = []
    bottleneck_edges: list[tuple[int, int]] = []

    for r in routes.values():
        if r.status != RouteStatus.SAFE or r.min_margin_min < buffer_minutes:
            trapped_count += 1
        else:
            safe_count += 1

        if np.isfinite(r.min_margin_min):
            margins.append(r.min_margin_min)

        if r.bottleneck_edge is not None:
            if isinstance(r.bottleneck_edge, tuple):
                bottleneck_edges.append((r.bottleneck_edge[0], r.bottleneck_edge[1]))
            else:
                bottleneck_edges.append((r.bottleneck_edge.u, r.bottleneck_edge.v))

    lowest_margin = float(min(margins)) if margins else -float("inf")

    # Qualitative description of traffic conditions
    if trapped_count > 0:
        qualitative = "Severe bottleneck detected; one or more routes lack required safety margin."
    elif lowest_margin < buffer_minutes * 1.5:
        qualitative = "Moderate congestion and narrowing margins; delay expected on primary egress."
    else:
        qualitative = "Free-flowing conditions; healthy safety buffer ahead of fire arrival."

    return RouteLevelExposure(
        total_routes=len(routes),
        trapped_count=trapped_count,
        safe_count=safe_count,
        lowest_margin_min=lowest_margin,
        bottleneck_edges=list(set(bottleneck_edges)),
        qualitative_traffic=qualitative,
    )


def compute_k_star(
    road_graph: RoadGraph,
    origins: Sequence[int],
    destinations: Sequence[int] | set[int],
    departure_time: float,
    arrival_field: ArrivalField,
    buffer_minutes: float = 15.0,
) -> dict[int, float]:
    """Compute critical traffic slowdown multiplier k* (congestion headroom) per origin."""
    edge_arrivals = road_graph.compute_edge_fire_arrivals(arrival_field)
    dest_set = set(destinations)

    k_stars: dict[int, float] = {}
    for orig in origins:
        k_val = compute_critical_slowdown_k_star(
            road_graph=road_graph,
            origin_node=orig,
            destination_nodes=dest_set,
            t0=departure_time,
            buffer_minutes=buffer_minutes,
            edge_fire_arrivals=edge_arrivals,
        )
        k_stars[orig] = float(k_val)

    return k_stars
