"""Evacuation re-planning policy with hysteresis and change detection."""


import numpy as np

from wf052.agent.contracts import ReplanDecision
from wf052.routing.contracts import ArrivalField, EvacuationRoute, RouteStatus
from wf052.routing.graph import RoadGraph


def evaluate_arrival_field_difference(
    field_a: ArrivalField | None,
    field_b: ArrivalField,
    horizon_minutes: float = 720.0,
) -> float:
    """Compute mean absolute arrival time shift across cells burned in either field.

    Returns:
        Mean absolute difference in minutes. Returns 0.0 if field_a is None.
    """
    if field_a is None:
        return 0.0

    var_a = (
        "arrival_q50"
        if "arrival_q50" in field_a
        else "arrival_q10"
        if "arrival_q10" in field_a
        else next(iter(field_a.data_vars.keys()))
    )
    var_b = (
        "arrival_q50"
        if "arrival_q50" in field_b
        else "arrival_q10"
        if "arrival_q10" in field_b
        else next(iter(field_b.data_vars.keys()))
    )
    a_vals = field_a[var_a].values
    b_vals = field_b[var_b].values

    active_mask = (a_vals < horizon_minutes) | (b_vals < horizon_minutes)
    if not np.any(active_mask):
        return 0.0

    diff = np.abs(a_vals[active_mask] - b_vals[active_mask])
    # Replace non-finite or extreme values
    diff = np.where(np.isfinite(diff), diff, horizon_minutes)
    return float(np.mean(diff))


def reevaluate_route_margin(
    route: EvacuationRoute,
    new_arrival_field: ArrivalField,
    road_graph: RoadGraph | None = None,
) -> float:
    """Re-evaluate route's minimum safety margin under updated fire arrival field."""
    if road_graph is None:
        return route.min_margin_min

    edge_arrivals = road_graph.compute_edge_fire_arrivals(new_arrival_field)
    min_margin = float("inf")

    for edge in route.edges:
        for idx, ge in enumerate(road_graph.edges):
            if ge.u == edge.u and ge.v == edge.v:
                margin = float(edge_arrivals[idx]) - edge.exit_time_min
                if margin < min_margin:
                    min_margin = margin
                break

    return min_margin


class ReplanPolicy:
    """Policy engine deciding when to trigger evacuation route replanning with hysteresis."""

    def __init__(
        self,
        buffer_minutes: float = 15.0,
        arrival_field_change_threshold: float = 15.0,
        hysteresis_margin_min: float = 5.0,
        hysteresis_time_pct: float = 0.10,
    ) -> None:
        self.buffer_minutes = buffer_minutes
        self.arrival_field_change_threshold = arrival_field_change_threshold
        self.hysteresis_margin_min = hysteresis_margin_min
        self.hysteresis_time_pct = hysteresis_time_pct

    def decide(
        self,
        current_routes: dict[int, EvacuationRoute] | None,
        candidate_routes: dict[int, EvacuationRoute],
        previous_arrival_field: ArrivalField | None,
        new_arrival_field: ArrivalField,
        road_cut_edges: set[tuple[int, int]] | None = None,
        road_graph: RoadGraph | None = None,
    ) -> tuple[dict[int, EvacuationRoute], ReplanDecision]:
        """Evaluate re-plan criteria across all origins with anti-flapping hysteresis.

        Args:
            current_routes: Currently active route assignments, or None/empty if initial.
            candidate_routes: Newly calculated optimal routes under the new arrival field.
            previous_arrival_field: ArrivalField from previous update.
            new_arrival_field: Updated ArrivalField from current update.
            road_cut_edges: Optional set of newly severed (u, v) edges.
            road_graph: Road graph for evaluating current route against new fire field.

        Returns:
            (active_routes, ReplanDecision)
        """
        # Case 1: Initial assignment
        if not current_routes:
            return candidate_routes, ReplanDecision(
                needs_replan=True,
                reason="Initial evacuation route assignment.",
                changed_origins=list(candidate_routes.keys()),
                flap_prevented_origins=[],
            )

        road_cuts = road_cut_edges or set()
        field_shift = evaluate_arrival_field_difference(
            previous_arrival_field, new_arrival_field
        )
        field_changed_significantly = field_shift >= self.arrival_field_change_threshold

        active_routes: dict[int, EvacuationRoute] = {}
        changed_origins: list[int] = []
        flap_prevented_origins: list[int] = []
        reasons: list[str] = []

        if field_changed_significantly:
            reasons.append(
                f"Arrival field shifted by {field_shift:.1f} min >= {self.arrival_field_change_threshold:.1f} min threshold."
            )

        for orig, cand_route in candidate_routes.items():
            curr_route = current_routes.get(orig)

            if curr_route is None:
                active_routes[orig] = cand_route
                changed_origins.append(orig)
                continue

            # Check if current route passes through a severed/cut road edge
            curr_edges = {(e.u, e.v) for e in curr_route.edges}
            is_cut = bool(curr_edges.intersection(road_cuts))

            # Re-evaluate current route margin under the NEW arrival field
            curr_margin = reevaluate_route_margin(curr_route, new_arrival_field, road_graph)
            margin_violated = (
                curr_route.status != RouteStatus.SAFE
                or curr_margin < self.buffer_minutes
            )

            # Check route node equality
            same_path = curr_route.node_ids == cand_route.node_ids

            if is_cut:
                active_routes[orig] = cand_route
                changed_origins.append(orig)
                reasons.append(f"Origin {orig}: road along active route was severed.")
            elif margin_violated:
                active_routes[orig] = cand_route
                if not same_path:
                    changed_origins.append(orig)
                reasons.append(
                    f"Origin {orig}: safety margin under updated fire field ({curr_margin:.1f} min) fell below buffer ({self.buffer_minutes:.1f} min)."
                )
            elif same_path:
                # Path identical, adopt candidate route with updated timestamps
                active_routes[orig] = cand_route
            else:
                # Candidate route is a different physical path. Apply hysteresis!
                margin_gain = cand_route.min_margin_min - curr_margin
                time_reduction = (
                    curr_route.total_travel_time_min - cand_route.total_travel_time_min
                ) / max(1e-3, curr_route.total_travel_time_min)

                substantially_better = (
                    margin_gain >= self.hysteresis_margin_min
                    or time_reduction >= self.hysteresis_time_pct
                )

                if substantially_better:
                    active_routes[orig] = cand_route
                    changed_origins.append(orig)
                    reasons.append(
                        f"Origin {orig}: candidate route offers substantial improvement (margin +{margin_gain:.1f} min, time {time_reduction*100:.1f}%)."
                    )
                else:
                    # Near-equal option: suppress flapping! Keep current route
                    active_routes[orig] = curr_route
                    flap_prevented_origins.append(orig)

        needs_replan = len(changed_origins) > 0 or field_changed_significantly
        combined_reason = (
            " | ".join(sorted(set(reasons)))
            if reasons
            else "Routes stable within hysteresis bounds; no re-plan required."
        )

        decision = ReplanDecision(
            needs_replan=needs_replan,
            reason=combined_reason,
            changed_origins=changed_origins,
            flap_prevented_origins=flap_prevented_origins,
        )

        return active_routes, decision
