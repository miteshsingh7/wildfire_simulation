"""Evaluation metrics and bootstrap confidence interval estimators for causal replay.

Evaluates:
- Hazard-miss rate: Fraction of routes intersecting a truth hazard at or after onset time.
- Warning lead time: Difference between hazard onset time and the time the route was first
  flagged or diverted away (median, distribution, fraction flagged before onset).
- Extra travel time vs static fastest baseline.
- Minimum safety margin distribution (median, p10).
- Re-plan frequency per origin.
- Bootstrap 95% confidence intervals over the 200 sampled origins.
"""

from collections.abc import Sequence

import numpy as np
import shapely.geometry

from wf052.replay.contracts import MethodMetrics, ReplayOriginResult, TruthHazard
from wf052.routing.contracts import EvacuationRoute
from wf052.routing.graph import RoadGraph


def check_route_hazard_miss(
    route: EvacuationRoute | None,
    hazards: Sequence[TruthHazard],
    road_graph: RoadGraph,
    spatial_radius_m: float = 120.0,
) -> tuple[bool, list[str]]:
    """Check if an EvacuationRoute intersects any truth hazard at or after its onset time.

    A 'hazard miss' occurs when the planned route traverses within spatial_radius_m
    of a truth hazard (burnover, roadblock, or abandoned vehicle) at a time
    t_traversal >= hazard.onset_elapsed_min.

    Args:
        route: The evaluated evacuation route.
        hazards: Sequence of truth hazards from NIST extracts.
        road_graph: RoadGraph containing edge geometries.
        spatial_radius_m: Spatial buffer around hazard point in meters.

    Returns:
        (hazard_miss_bool, list_of_encountered_hazard_ids)
    """
    if route is None or not route.edges:
        # Route could not be found or disconnected (trapped at origin)
        return True, ["UNREACHABLE_NO_ROUTE"]

    encountered_ids: list[str] = []
    # Build fast edge lookup: (u, v, key) -> EdgeRecord
    edge_map = {(e.u, e.v, e.key): e for e in road_graph.edges}

    for re in route.edges:
        ed = edge_map.get((re.u, re.v, re.key))
        geom = ed.geometry if ed is not None else None

        for h in hazards:
            # If the vehicle traversed and exited the edge BEFORE hazard onset, it is safe
            if re.exit_time_min < h.onset_elapsed_min:
                continue

            # Check spatial distance
            if geom is not None and not geom.is_empty:
                dist = float(geom.distance(h.geometry_utm))
            else:
                u_coord = road_graph.nodes.get(re.u)
                v_coord = road_graph.nodes.get(re.v)
                if u_coord and v_coord:
                    dist = float(
                        shapely.geometry.LineString([u_coord, v_coord]).distance(h.geometry_utm)
                    )
                else:
                    dist = float("inf")

            if dist <= spatial_radius_m:
                encountered_ids.append(h.hazard_id)

    is_miss = len(encountered_ids) > 0
    return is_miss, sorted(set(encountered_ids))


def compute_warning_lead_time(
    initial_route: EvacuationRoute | None,
    replan_history: list[tuple[float, EvacuationRoute | None]],
    hazards: Sequence[TruthHazard],
    road_graph: RoadGraph,
    spatial_radius_m: float = 120.0,
) -> tuple[float | None, bool]:
    """Compute warning lead time for an origin.

    Warning lead time = hazard_onset - time_route_first_diverted_or_flagged.
    If the initial route would have encountered a hazard at onset t_onset, and
    at step t_replan the route was re-planned to safely avoid that hazard before onset:
        lead_time = t_onset - t_replan.
    If flagged before onset (lead_time > 0), flagged_before_onset is True.

    If the initial route never encounters any hazard, returns (None, True).
    If a hazard was encountered and never diverted before onset, returns (0.0, False).
    """
    initial_miss, initial_hazards = check_route_hazard_miss(
        initial_route, hazards, road_graph, spatial_radius_m
    )
    if not initial_miss or not initial_hazards:
        return None, True

    # Find the earliest hazard onset time among those initially encountered
    hazard_dict = {h.hazard_id: h for h in hazards}
    earliest_onset = min(
        hazard_dict[hid].onset_elapsed_min for hid in initial_hazards if hid in hazard_dict
    )

    # Check replan history for first time the hazard was avoided
    for step_t, r in replan_history:
        if r is None:
            continue
        miss, curr_hazards = check_route_hazard_miss(r, hazards, road_graph, spatial_radius_m)
        # Check if the initially threatening hazards are now avoided
        still_threatened = any(hid in curr_hazards for hid in initial_hazards)
        if not still_threatened:
            lead_time = earliest_onset - step_t
            return float(lead_time), lead_time > 0.0

    return 0.0, False


def compute_bootstrap_ci(
    data: Sequence[float] | np.ndarray,
    stat_fn: str = "mean",
    num_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval (percentile method) over sample.

    Args:
        data: Array or sequence of sample values.
        stat_fn: 'mean' or 'median'.
        num_resamples: Number of bootstrap resamples (default 1000).
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (ci_lower, ci_upper)
    """
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (0.0, 0.0)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))

    rng = np.random.default_rng(seed)
    n = len(arr)
    indices = rng.integers(0, n, size=(num_resamples, n))
    resamples = arr[indices]

    if stat_fn == "median":
        boot_stats = np.median(resamples, axis=1)
    else:
        boot_stats = np.mean(resamples, axis=1)

    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_low = float(np.percentile(boot_stats, low_pct))
    ci_high = float(np.percentile(boot_stats, high_pct))
    return (ci_low, ci_high)


def aggregate_method_metrics(
    method_name: str,
    results: list[ReplayOriginResult],
    static_results: list[ReplayOriginResult] | None = None,
    mean_step_latency_ms: float = 0.0,
) -> MethodMetrics:
    """Aggregate per-origin simulation results into MethodMetrics with bootstrap CIs."""
    total = len(results)
    if total == 0:
        return MethodMetrics(
            method_name=method_name,
            total_origins=0,
            hazard_miss_count=0,
            hazard_miss_rate=0.0,
            hazard_miss_ci=(0.0, 0.0),
            extra_travel_time_mean=0.0,
            extra_travel_time_ci=(0.0, 0.0),
            warning_lead_time_mean=0.0,
            warning_lead_time_ci=(0.0, 0.0),
            warning_flagged_fraction=0.0,
            replan_count_mean=0.0,
            min_margin_median=0.0,
            min_margin_p10=0.0,
            mean_step_latency_ms=mean_step_latency_ms,
        )

    # 1. Hazard misses
    miss_flags = np.array([1.0 if r.hazard_miss else 0.0 for r in results])
    miss_count = int(np.sum(miss_flags))
    miss_rate = miss_count / total
    miss_ci = compute_bootstrap_ci(miss_flags, stat_fn="mean")

    # 2. Extra travel time vs static (computed over completed evacuations)
    static_tt_map = (
        {sr.origin_id: sr.travel_time_min for sr in static_results} if static_results else {}
    )
    extra_tts = []
    for r in results:
        base_tt = static_tt_map.get(r.origin_id, r.travel_time_min)
        if np.isfinite(r.travel_time_min) and np.isfinite(base_tt):
            extra = max(0.0, r.travel_time_min - base_tt)
            extra_tts.append(extra)
    extra_tts_arr = np.array(extra_tts) if extra_tts else np.array([0.0])
    extra_tt_mean = float(np.mean(extra_tts_arr))
    extra_tt_ci = compute_bootstrap_ci(extra_tts_arr, stat_fn="mean")

    # 3. Warning lead time
    lead_times = [r.warning_lead_time_min for r in results if r.warning_lead_time_min is not None]
    lead_times_arr = np.array(lead_times) if lead_times else np.array([0.0])
    lead_time_mean = float(np.mean(lead_times_arr))
    lead_time_ci = compute_bootstrap_ci(lead_times_arr, stat_fn="mean")

    flagged_count = sum(1 for r in results if r.flagged_before_onset)
    flagged_fraction = flagged_count / total

    # 4. Replan count
    replan_counts = np.array([r.replan_count for r in results], dtype=np.float64)
    replan_mean = float(np.mean(replan_counts))

    # 5. Margins
    margins = np.array([r.min_margin_min for r in results if np.isfinite(r.min_margin_min)])
    if len(margins) > 0:
        margin_med = float(np.median(margins))
        margin_p10 = float(np.percentile(margins, 10.0))
    else:
        margin_med = 0.0
        margin_p10 = 0.0

    return MethodMetrics(
        method_name=method_name,
        total_origins=total,
        hazard_miss_count=miss_count,
        hazard_miss_rate=miss_rate,
        hazard_miss_ci=miss_ci,
        extra_travel_time_mean=extra_tt_mean,
        extra_travel_time_ci=extra_tt_ci,
        warning_lead_time_mean=lead_time_mean,
        warning_lead_time_ci=lead_time_ci,
        warning_flagged_fraction=flagged_fraction,
        replan_count_mean=replan_mean,
        min_margin_median=margin_med,
        min_margin_p10=margin_p10,
        mean_step_latency_ms=mean_step_latency_ms,
    )
