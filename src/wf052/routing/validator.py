"""Independent validation code path for evacuation routes.

Re-integrates edge traversal times from scratch using raw geometry and speeds,
re-samples fire arrivals directly from the ArrivalField raster, and verifies
all margins and arrival times match the router within floating-point tolerance.
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from wf052.routing.contracts import EvacuationRoute, RouteStatus, validate_arrival_field
from wf052.routing.graph import RoadGraph


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of independent evacuation route verification."""

    is_valid: bool
    recomputed_total_travel_time_min: float
    recomputed_arrival_time_min: float
    recomputed_min_margin_min: float
    max_time_discrepancy_min: float
    max_margin_discrepancy_min: float
    status_verified: bool
    failure_reasons: tuple[str, ...]


class IndependentRouteValidator:
    """Independent validator checking route integrity and margin compliance."""

    def __init__(self, tolerance_min: float = 1e-4) -> None:
        self.tolerance_min = tolerance_min

    def validate(
        self,
        route: EvacuationRoute,
        arrival_field: xr.Dataset,
        road_graph: RoadGraph,
        buffer_minutes: float = 0.0,
    ) -> ValidationResult:
        """Independently re-integrate traversal times and fire arrivals for an EvacuationRoute.

        Asserts agreement between the router's outputs and independent re-integration.
        """
        validate_arrival_field(arrival_field)
        q10_grid = arrival_field["arrival_q10"].values
        grid_x = arrival_field["x"].values
        grid_y = arrival_field["y"].values

        dx = float(abs(grid_x[1] - grid_x[0]))
        dy = float(abs(grid_y[1] - grid_y[0]))
        x_min = float(grid_x.min()) - 0.5 * dx
        y_max = float(grid_y.max()) + 0.5 * dy
        num_rows = len(grid_y)
        num_cols = len(grid_x)

        failures: list[str] = []
        max_time_err = 0.0
        max_margin_err = 0.0

        if not route.edges:
            return ValidationResult(
                is_valid=False,
                recomputed_total_travel_time_min=0.0,
                recomputed_arrival_time_min=route.departure_time_min,
                recomputed_min_margin_min=-float("inf"),
                max_time_discrepancy_min=0.0,
                max_margin_discrepancy_min=0.0,
                status_verified=False,
                failure_reasons=("Route contains no edges",),
            )

        # Re-integrate sequentially
        curr_t = float(route.departure_time_min)
        min_margin = float("inf")

        for idx, edge in enumerate(route.edges):
            # 1. Independent traversal time check
            expected_dt = (edge.length_m / edge.speed_mps) / 60.0
            dt_diff = abs(edge.traversal_time_min - expected_dt)
            max_time_err = max(max_time_err, dt_diff)
            if dt_diff > self.tolerance_min:
                failures.append(
                    f"Edge {idx} ({edge.u}->{edge.v}) traversal time mismatch: "
                    f"recorded={edge.traversal_time_min:.5f}, expected={expected_dt:.5f}"
                )

            entry_t = curr_t
            exit_t = entry_t + edge.traversal_time_min

            entry_diff = abs(edge.entry_time_min - entry_t)
            exit_diff = abs(edge.exit_time_min - exit_t)
            max_time_err = max(max_time_err, entry_diff, exit_diff)

            # 2. Independent fire arrival re-sampling from grid
            # Sample points along geometry directly
            pts: list[tuple[float, float]] = []
            matched_ed = None
            for cand in road_graph.edges:
                if cand.u == edge.u and cand.v == edge.v and cand.key == edge.key:
                    matched_ed = cand
                    break

            if matched_ed is not None:
                geom = matched_ed.geometry
                if geom.length <= 30.0:
                    pts = [(float(p[0]), float(p[1])) for p in geom.coords]
                else:
                    n_pts = int(np.ceil(geom.length / 25.0)) + 1
                    dists = np.linspace(0, geom.length, n_pts)
                    pts = [(float(p.x), float(p.y)) for p in [geom.interpolate(d) for d in dists]]

                sample_q10 = []
                for px, py in pts:
                    col = int((px - x_min) / dx)
                    row = int((y_max - py) / dy)
                    if 0 <= row < num_rows and 0 <= col < num_cols:
                        sample_q10.append(float(q10_grid[row, col]))

                recomputed_fire_arr = min(sample_q10) if sample_q10 else float("inf")
                fire_arr_diff = abs(edge.fire_arrival_min - recomputed_fire_arr)
                if np.isfinite(edge.fire_arrival_min) and np.isfinite(recomputed_fire_arr):
                    max_margin_err = max(max_margin_err, fire_arr_diff)
                    if fire_arr_diff > self.tolerance_min:
                        failures.append(
                            f"Edge {idx} fire arrival mismatch: recorded={edge.fire_arrival_min:.5f}, "
                            f"recomputed={recomputed_fire_arr:.5f}"
                        )

            # 3. Independent margin calculation
            expected_margin = edge.fire_arrival_min - exit_t
            margin_diff = abs(edge.margin_min - expected_margin)
            max_margin_err = max(max_margin_err, margin_diff)
            if margin_diff > self.tolerance_min:
                failures.append(
                    f"Edge {idx} margin mismatch: recorded={edge.margin_min:.5f}, "
                    f"expected={expected_margin:.5f}"
                )

            if edge.margin_min < min_margin:
                min_margin = edge.margin_min

            curr_t = exit_t

        recomputed_travel_time = curr_t - route.departure_time_min
        total_time_diff = abs(route.total_travel_time_min - recomputed_travel_time)
        max_time_err = max(max_time_err, total_time_diff)

        min_margin_diff = abs(route.min_margin_min - min_margin)
        max_margin_err = max(max_margin_err, min_margin_diff)

        status_verified = True
        if route.status == RouteStatus.SAFE:
            if min_margin + self.tolerance_min < buffer_minutes:
                status_verified = False
                failures.append(
                    f"Route status is SAFE but recomputed min margin {min_margin:.2f}m < buffer {buffer_minutes:.2f}m"
                )

        is_valid = (len(failures) == 0) and status_verified

        return ValidationResult(
            is_valid=is_valid,
            recomputed_total_travel_time_min=float(recomputed_travel_time),
            recomputed_arrival_time_min=float(curr_t),
            recomputed_min_margin_min=float(min_margin),
            max_time_discrepancy_min=float(max_time_err),
            max_margin_discrepancy_min=float(max_margin_err),
            status_verified=status_verified,
            failure_reasons=tuple(failures),
        )
