"""Hazard-miss scoring scaffold for evaluating planned routes against ground-truth hazard points.

Evaluates whether any vehicle's 4D spatio-temporal trajectory (x(t), y(t))
intersects active danger radii of timestamped ground-truth hazards (e.g. NIST
burnover points, tree fall, active fireline intersections).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import shapely.geometry

from wf052.routing.contracts import EvacuationRoute
from wf052.routing.graph import RoadGraph


@dataclass(frozen=True)
class HazardPoint:
    """A ground-truth timestamped hazard observation."""

    hazard_id: str
    x: float
    y: float
    time_min: float  # Time in minutes post-ignition when hazard becomes active
    radius_m: float = 60.0  # Danger radius around hazard
    hazard_type: str = "burnover"


@dataclass(frozen=True)
class HazardCheckResult:
    """Outcome of scoring a route against ground-truth hazard points."""

    total_hazards_checked: int
    encountered_hazard_count: (
        int  # Number of hazards where vehicle entered danger zone while active
    )
    missed_hazard_count: int  # Number of hazards safely avoided (total - encountered)
    min_hazard_distance_m: float  # Minimum spatial distance to any active hazard
    encountered_hazard_ids: tuple[str, ...]


def score_route_against_hazards(
    route: EvacuationRoute,
    hazards: Sequence[HazardPoint],
    road_graph: RoadGraph | None = None,
    spatial_tolerance_m: float = 0.0,
) -> HazardCheckResult:
    """Score an EvacuationRoute against ground-truth timestamped hazards.

    A hazard is encountered if the vehicle enters within hazard.radius_m + spatial_tolerance_m
    of (hazard.x, hazard.y) at a time t >= hazard.time_min.
    """
    if not hazards or not route.edges:
        return HazardCheckResult(
            total_hazards_checked=len(hazards),
            encountered_hazard_count=0,
            missed_hazard_count=len(hazards),
            min_hazard_distance_m=float("inf"),
            encountered_hazard_ids=(),
        )

    encountered_ids: set[str] = set()
    global_min_dist = float("inf")

    for edge in route.edges:
        # Determine geometry of this edge
        geom = None
        if road_graph is not None:
            for ed_rec in road_graph.edges:
                if ed_rec.u == edge.u and ed_rec.v == edge.v and ed_rec.key == edge.key:
                    geom = ed_rec.geometry
                    break

        for h in hazards:
            # If vehicle passed this edge entirely before hazard became active, it is safe
            if edge.exit_time_min < h.time_min:
                continue

            h_pt = shapely.geometry.Point(h.x, h.y)
            if geom is not None and not geom.is_empty:
                dist = float(geom.distance(h_pt))
            else:
                # Fallback to endpoint proxy
                dist = np.hypot(h.x - float(edge.u), h.y - float(edge.v))

            if dist < global_min_dist:
                global_min_dist = dist

            if dist <= h.radius_m + spatial_tolerance_m:
                encountered_ids.add(h.hazard_id)

    encountered_count = len(encountered_ids)
    missed_count = len(hazards) - encountered_count

    return HazardCheckResult(
        total_hazards_checked=len(hazards),
        encountered_hazard_count=encountered_count,
        missed_hazard_count=missed_count,
        min_hazard_distance_m=float(global_min_dist),
        encountered_hazard_ids=tuple(sorted(encountered_ids)),
    )
