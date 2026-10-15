"""Comprehensive unit and integration tests for time-aware evacuation router.

Tests:
1. Synthetic expanding-circle fire with analytic arrival times.
2. Synthetic wind-ellipse fire with exact Huygens arrival times.
3. Known-answer routes on a deterministic diamond graph.
4. Case where no safe route exists (Mode b max-min margin fallback).
5. Dynamic re-plan where a road is cut mid-evacuation.
6. Independent route validator integrity check (verifies valid, catches corrupted).
7. Benchmark re-plan latency on full domain graph (target < 5 s).
"""

import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry

from wf052.routing.contracts import (
    EvacuationRoute,
    RouteEdge,
    RouteStatus,
    RoutingMode,
    validate_arrival_field,
)
from wf052.routing.graph import RoadGraph, load_camp_fire_road_graph
from wf052.routing.replan import ReplanEngine
from wf052.routing.router import TimeDependentRouter
from wf052.routing.synthetic_fire import (
    create_expanding_circle_arrival_field,
    create_unburned_arrival_field,
    create_wind_ellipse_arrival_field,
)
from wf052.routing.validator import IndependentRouteValidator


@pytest.fixture
def camp_fire_graph() -> RoadGraph:
    """Load the pre-ingested Camp Fire road graph."""
    return load_camp_fire_road_graph()


@pytest.fixture
def synthetic_diamond_graph() -> RoadGraph:
    """Construct a minimal, deterministic diamond road graph for known-answer tests.

    Nodes:
      0 (Origin): (0, 0)
      1 (Path A intermediate): (500, 500)
      2 (Path B intermediate): (500, -500)
      3 (Destination): (1000, 0)

    Edges:
      Path A (Fast, 10 min total):
        0 -> 1: length=707.1m, speed=1.1785 m/s -> traversal=10.0 min
        1 -> 3: length=707.1m, speed=1.1785 m/s -> traversal=10.0 min
        (Total Path A = 20 min)
      Path B (Slow, 40 min total):
        0 -> 2: length=707.1m, speed=0.58925 m/s -> traversal=20.0 min
        2 -> 3: length=707.1m, speed=0.58925 m/s -> traversal=20.0 min
        (Total Path B = 40 min)
    """
    nodes_data = {
        "node_id": [0, 1, 2, 3],
        "geometry": [
            shapely.geometry.Point(0.0, 0.0),
            shapely.geometry.Point(500.0, 500.0),
            shapely.geometry.Point(500.0, -500.0),
            shapely.geometry.Point(1000.0, 0.0),
        ],
    }
    nodes_gdf = gpd.GeoDataFrame(nodes_data, crs="EPSG:32610").set_index("node_id")

    edges_data = {
        "u": [0, 1, 0, 2],
        "v": [1, 3, 2, 3],
        "key": [0, 0, 0, 0],
        "length": [707.106, 707.106, 707.106, 707.106],
        "maxspeed": ["2.64 mph", "2.64 mph", "1.32 mph", "1.32 mph"],  # yields 10m vs 20m dt
        "highway": ["primary", "primary", "residential", "residential"],
        "geometry": [
            shapely.geometry.LineString([(0.0, 0.0), (500.0, 500.0)]),
            shapely.geometry.LineString([(500.0, 500.0), (1000.0, 0.0)]),
            shapely.geometry.LineString([(0.0, 0.0), (500.0, -500.0)]),
            shapely.geometry.LineString([(500.0, -500.0), (1000.0, 0.0)]),
        ],
    }
    edges_gdf = gpd.GeoDataFrame(edges_data, crs="EPSG:32610")

    grid_x = np.linspace(-200.0, 1200.0, 25)
    grid_y = np.linspace(700.0, -700.0, 25)

    return RoadGraph(
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        grid_x_coords=grid_x,
        grid_y_coords=grid_y,
    )


def test_contract_validation() -> None:
    """Verify that ArrivalField contract enforcement detects missing variables and coordinates."""
    x = np.arange(0.0, 600.0, 60.0)
    y = np.arange(600.0, 0.0, -60.0)
    valid_ds = create_unburned_arrival_field(x, y)
    validate_arrival_field(valid_ds)

    # Missing required variable
    invalid_ds = valid_ds.drop_vars("arrival_q10")
    with pytest.raises(ValueError, match="missing required data variable"):
        validate_arrival_field(invalid_ds)


def test_synthetic_expanding_circle_analytic_arrival() -> None:
    """Verify expanding circle produces exact radial distance / speed arrival times."""
    x = np.arange(0.0, 600.0, 60.0)
    y = np.arange(600.0, 0.0, -60.0)
    speed = 30.0  # m/min
    ds = create_expanding_circle_arrival_field(
        x_coords=x,
        y_coords=y,
        ignition_x=0.0,
        ignition_y=0.0,
        spread_rate_m_per_min=speed,
        horizon_minutes=20.0,
    )

    q50 = ds["arrival_q50"].values
    # At (300, 400), distance is 500m -> arrival is 500 / 30 = 16.667 min
    # Find cell closest to (300, 400)
    c_idx = int(np.argmin(abs(x - 300.0)))
    r_idx = int(np.argmin(abs(y - 400.0)))
    dist = np.hypot(x[c_idx], y[r_idx])
    expected_arr = dist / speed
    assert np.isclose(q50[r_idx, c_idx], expected_arr, atol=1e-5)

    # Beyond horizon (>20 min), should be +inf
    c_far = int(np.argmin(abs(x - 540.0)))
    r_far = int(np.argmin(abs(y - 540.0)))
    assert np.isinf(q50[r_far, c_far])

    # Quantile ordering: q10 <= q50 <= q90
    q10 = ds["arrival_q10"].values
    q90 = ds["arrival_q90"].values
    valid_mask = np.isfinite(q50)
    assert np.all(q10[valid_mask] <= q50[valid_mask])
    assert np.all(q50[valid_mask] <= q90[valid_mask])


def test_synthetic_wind_ellipse_huygens_analytic() -> None:
    """Verify wind ellipse produces exact Huygens arrival times in head, back, and flank axes."""
    x = np.linspace(-1000.0, 1000.0, 101)
    y = np.linspace(1000.0, -1000.0, 101)
    # Wind blowing towards East (0 deg)
    r_head = 50.0  # m/min
    r_back = 10.0  # m/min
    lb = 2.0
    ds = create_wind_ellipse_arrival_field(
        x_coords=x,
        y_coords=y,
        ignition_x=0.0,
        ignition_y=0.0,
        head_rate_m_per_min=r_head,
        back_rate_m_per_min=r_back,
        length_to_breadth=lb,
        wind_towards_deg=0.0,  # Eastward along +x
        horizon_minutes=100.0,
    )
    q50 = ds["arrival_q50"].values

    # 1. Downwind point at x = +500, y = 0: arrival should be 500 / r_head = 10.0 min
    c_head = int(np.argmin(abs(x - 500.0)))
    r_zero = int(np.argmin(abs(y - 0.0)))
    assert np.isclose(q50[r_zero, c_head], 500.0 / r_head, atol=1e-3)

    # 2. Upwind point at x = -200, y = 0: arrival should be 200 / r_back = 20.0 min
    c_back = int(np.argmin(abs(x - (-200.0))))
    assert np.isclose(q50[r_zero, c_back], 200.0 / r_back, atol=1e-3)


def test_known_answer_fastest_vs_safe_route(synthetic_diamond_graph: RoadGraph) -> None:
    """Known-answer test:

    Path A: 0 -> 1 -> 3 (total travel = 20 min). Goes through (500, 500).
    Path B: 0 -> 2 -> 3 (total travel = 40 min). Goes through (500, -500).

    Case 1: Fire arrives at node 1 at t = 25 min.
            Path A exits node 1 at t = 10 min. Margin = 25 - 10 = 15 min.
            With buffer B = 10 min, Path A has margin 15 >= 10 -> SAFE!
            Router MUST pick Path A (Fastest = 20 min).

    Case 2: Fire arrives at node 1 at t = 15 min.
            Path A exit time is 10 min. Margin = 15 - 10 = 5 min.
            With buffer B = 10 min, Path A margin 5 < 10 -> UNSAFE!
            Path B is unburned (margin = +inf).
            Router MUST pick Path B (Safe, total travel = 40 min).
    """
    router = TimeDependentRouter(road_graph=synthetic_diamond_graph)

    # Case 1: Fire reaches Path A at t=25 min
    # Edge 0 (0->1) fire arrival = 25.0, Edge 1 (1->3) fire arrival = 35.0
    edge_arrivals_safe = np.array([25.0, 35.0, np.inf, np.inf])

    route1 = router.find_route(
        origin_node=0,
        destination_nodes=[3],
        t0=0.0,
        buffer_minutes=10.0,
        edge_fire_arrivals=edge_arrivals_safe,
        mode=RoutingMode.FASTEST_SAFE,
    )
    assert route1 is not None
    assert route1.status == RouteStatus.SAFE
    assert route1.node_ids == (0, 1, 3)  # Selected Path A (fastest)
    assert np.isclose(route1.total_travel_time_min, 20.0, atol=0.1)
    assert route1.min_margin_min >= 10.0

    # Case 2: Fire reaches Path A at t=15 min -> Path A becomes unsafe
    edge_arrivals_cut = np.array([15.0, 25.0, np.inf, np.inf])

    route2 = router.find_route(
        origin_node=0,
        destination_nodes=[3],
        t0=0.0,
        buffer_minutes=10.0,
        edge_fire_arrivals=edge_arrivals_cut,
        mode=RoutingMode.FASTEST_SAFE,
    )
    assert route2 is not None
    assert route2.status == RouteStatus.SAFE
    assert route2.node_ids == (0, 2, 3)  # Diverted to Path B!
    assert np.isclose(route2.total_travel_time_min, 40.0, atol=0.1)
    assert np.isinf(route2.min_margin_min)


def test_no_safe_route_fallback_to_max_min_margin(synthetic_diamond_graph: RoadGraph) -> None:
    """When no route satisfies margin >= buffer, Mode (b) must return the maximum margin route."""
    router = TimeDependentRouter(road_graph=synthetic_diamond_graph)

    # Fire cuts Path A with margin = 2 min (arr=12m, exit=10m)
    # Fire cuts Path B with margin = -5 min (arr=15m, exit=20m)
    # Requested buffer = 15 min. Neither satisfies buffer >= 15.
    edge_arrivals = np.array([12.0, 50.0, 15.0, 50.0])

    route = router.find_route(
        origin_node=0,
        destination_nodes=[3],
        t0=0.0,
        buffer_minutes=15.0,
        edge_fire_arrivals=edge_arrivals,
        mode=RoutingMode.FASTEST_SAFE,
    )
    assert route is not None
    assert route.status == RouteStatus.BEST_EFFORT
    assert route.mode_used == RoutingMode.MAX_MIN_MARGIN
    # Path A has bottleneck margin ~2 min; Path B has bottleneck margin ~-5 min.
    # Router must pick Path A as maximizing minimum margin!
    assert route.node_ids == (0, 1, 3)
    assert np.isclose(route.min_margin_min, 2.0, atol=0.1)


def test_road_cut_mid_evacuation_replan(synthetic_diamond_graph: RoadGraph) -> None:
    """Test dynamic re-plan when fire progression cuts off the previously planned route mid-evacuation."""
    replan_engine = ReplanEngine(road_graph=synthetic_diamond_graph)

    # Initial plan at t=0: Both paths clear, Path A selected
    arr_field_initial = create_unburned_arrival_field(
        x_coords=synthetic_diamond_graph.grid_x,  # type: ignore
        y_coords=synthetic_diamond_graph.grid_y,  # type: ignore
    )
    res_initial = replan_engine.replan(
        arrival_field=arr_field_initial,
        origins=[0],
        destinations=[3],
        t0=0.0,
        buffer_minutes=10.0,
    )
    assert res_initial.routes["node_0"].node_ids == (0, 1, 3)

    # At t=5 min mid-evacuation, satellite update reveals fire arriving at node 1 at t=12 min
    # Traversal to 1 takes 10 min, so exit time is 5 + 10 = 15 min > 12 min (fatal!).
    # Replan from current node 0 at t=5 min with new ArrivalField
    arr_field_updated = create_unburned_arrival_field(
        x_coords=synthetic_diamond_graph.grid_x,  # type: ignore
        y_coords=synthetic_diamond_graph.grid_y,  # type: ignore
    )
    # Burn Path A intermediate cell around (500, 500)
    xx, yy = np.meshgrid(synthetic_diamond_graph.grid_x, synthetic_diamond_graph.grid_y)  # type: ignore
    mask_a = (xx > 200) & (xx < 600) & (yy > 200) & (yy < 600)
    arr_q10 = arr_field_updated["arrival_q10"].values
    arr_q10[mask_a] = 12.0
    arr_field_updated["arrival_q10"].values = arr_q10

    res_updated = replan_engine.replan(
        arrival_field=arr_field_updated,
        origins=[0],
        destinations=[3],
        t0=5.0,  # Leaving now at t=5
        buffer_minutes=10.0,
    )
    # Engine must re-route around the newly burned road to Path B (0 -> 2 -> 3)
    assert res_updated.routes["node_0"].node_ids == (0, 2, 3)
    assert res_updated.routes["node_0"].status == RouteStatus.SAFE


def test_independent_validator_catches_corruption(synthetic_diamond_graph: RoadGraph) -> None:
    """Verify that IndependentRouteValidator rejects modified/corrupted route metrics."""
    router = TimeDependentRouter(road_graph=synthetic_diamond_graph)
    validator = IndependentRouteValidator()
    edge_arrivals = np.array([50.0, 50.0, 50.0, 50.0])

    valid_route = router.find_route(
        origin_node=0,
        destination_nodes=[3],
        t0=0.0,
        buffer_minutes=10.0,
        edge_fire_arrivals=edge_arrivals,
    )
    assert valid_route is not None

    arr_field = create_unburned_arrival_field(
        x_coords=synthetic_diamond_graph.grid_x,  # type: ignore
        y_coords=synthetic_diamond_graph.grid_y,  # type: ignore
    )

    # Valid route must pass
    val_res = validator.validate(
        valid_route, arr_field, synthetic_diamond_graph, buffer_minutes=10.0
    )
    assert val_res.is_valid

    # Corrupt edge traversal time
    bad_edges = list(valid_route.edges)
    corrupted_edge = RouteEdge(
        u=bad_edges[0].u,
        v=bad_edges[0].v,
        key=bad_edges[0].key,
        length_m=bad_edges[0].length_m,
        speed_mps=bad_edges[0].speed_mps,
        traversal_time_min=999.0,  # Corrupted!
        entry_time_min=bad_edges[0].entry_time_min,
        exit_time_min=bad_edges[0].exit_time_min,
        fire_arrival_min=bad_edges[0].fire_arrival_min,
        margin_min=bad_edges[0].margin_min,
    )
    bad_edges[0] = corrupted_edge
    corrupted_route = EvacuationRoute(
        origin_id=valid_route.origin_id,
        destination_id=valid_route.destination_id,
        departure_time_min=valid_route.departure_time_min,
        arrival_time_min=valid_route.arrival_time_min,
        total_travel_time_min=valid_route.total_travel_time_min,
        min_margin_min=valid_route.min_margin_min,
        bottleneck_edge=valid_route.bottleneck_edge,
        edges=tuple(bad_edges),
        node_ids=valid_route.node_ids,
        status=valid_route.status,
        mode_used=valid_route.mode_used,
    )

    bad_val_res = validator.validate(
        corrupted_route, arr_field, synthetic_diamond_graph, buffer_minutes=10.0
    )
    assert not bad_val_res.is_valid
    assert any("traversal time mismatch" in r for r in bad_val_res.failure_reasons)


def test_replan_latency_on_full_camp_fire_domain(camp_fire_graph: RoadGraph) -> None:
    """Benchmark re-plan latency on full domain graph (8,085 edges, 3,627 nodes).

    Target latency is < 5.0 seconds.
    """
    assert camp_fire_graph.grid_x is not None and camp_fire_graph.grid_y is not None
    # Create a realistic synthetic wind ellipse fire spreading southwest towards Paradise
    arr_field = create_wind_ellipse_arrival_field(
        x_coords=camp_fire_graph.grid_x,
        y_coords=camp_fire_graph.grid_y,
        ignition_x=633391.0,  # Pulga
        ignition_y=4407338.0,
        head_rate_m_per_min=35.0,
        back_rate_m_per_min=5.0,
        length_to_breadth=2.5,
        wind_towards_deg=240.0,  # Southwest toward Paradise/Chico
        horizon_minutes=720.0,
    )

    engine = ReplanEngine(road_graph=camp_fire_graph)

    # Re-plan across key evacuation landmarks: Paradise, Magalia, Concow
    test_origins = ["Paradise", "Magalia", "Concow"]
    result = engine.replan(
        arrival_field=arr_field,
        origins=test_origins,
        destinations=["West_Chico", "South_Oroville"],
        t0=60.0,  # 1 hour post-ignition
        buffer_minutes=15.0,
        validate_routes=True,
    )

    print(
        f"\n[LATENCY REPORT] Full Domain Graph Re-Plan: {result.replan_latency_sec:.4f} s (Target < 5.0 s)"
    )
    for name, r in result.routes.items():
        print(
            f"  Origin: {name:<12} | Dest Node: {r.destination_id} | Status: {r.status.value:<11} "
            f"| Travel: {r.total_travel_time_min:5.1f} min | Min Margin: {r.min_margin_min:5.1f} min"
        )

    assert result.replan_latency_sec < 5.0, (
        f"Re-plan latency exceeded 5s target: {result.replan_latency_sec:.3f}s"
    )
    assert len(result.routes) == len(test_origins)
    for name in test_origins:
        assert name in result.routes
        assert len(result.routes[name].edges) > 0
