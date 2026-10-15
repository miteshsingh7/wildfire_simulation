"""Unit and integration tests for congestion, demand, en-route diversion, destination safety, and latency.

Tests:
1. Known-answer single bottleneck queue delay.
2. Emergency contraflow capacity doubling and queue dissipation.
3. En-route dynamic diversion from in-transit vehicle position (edge + offset).
4. Destination safety buffer X km filtering out encroached boundary exits.
5. Hazard-miss check scoring trajectory against timestamped truth hazards.
6. Camp Fire full-detail graph demand & clearance comparison (free-flow vs congested).
7. Latency benchmark across 100 random/heavily-blocked arrival fields (reporting p50, p95, max).
"""

import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry

from wf052.routing.congestion import (
    CongestionModel,
    TimeSlicedAssignmentEngine,
)
from wf052.routing.contracts import (
    EvacuationRoute,
    RouteEdge,
    RouteStatus,
    RoutingMode,
)
from wf052.routing.demand import (
    EvacuationDemandModel,
)
from wf052.routing.en_route import VehiclePosition, replan_en_route_vehicle
from wf052.routing.graph import RoadGraph
from wf052.routing.hazard_check import HazardPoint, score_route_against_hazards
from wf052.routing.replan import ReplanEngine
from wf052.routing.router import TimeDependentRouter
from wf052.routing.synthetic_fire import (
    create_unburned_arrival_field,
    create_wind_ellipse_arrival_field,
)


@pytest.fixture
def simple_bottleneck_graph() -> RoadGraph:
    """Two-node, single-edge bottleneck graph for exact queue delay tests."""
    nodes_gdf = gpd.GeoDataFrame(
        {"geometry": [shapely.geometry.Point(0.0, 0.0), shapely.geometry.Point(1000.0, 0.0)]},
        index=[0, 1],
        crs="EPSG:32610",
    )
    edges_gdf = gpd.GeoDataFrame(
        {
            "u": [0],
            "v": [1],
            "key": [0],
            "length": [1000.0],
            "maxspeed": ["37.28 mph"],  # 16.667 m/s -> traversal = 1.0 min
            "highway": ["primary"],
            "geometry": [shapely.geometry.LineString([(0.0, 0.0), (1000.0, 0.0)])],
        },
        crs="EPSG:32610",
    )
    return RoadGraph(
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        grid_x_coords=np.linspace(-100.0, 1100.0, 10),
        grid_y_coords=np.linspace(100.0, -100.0, 10),
    )


def test_known_answer_single_bottleneck_queue_delay(simple_bottleneck_graph: RoadGraph) -> None:
    """Verify point-queue bottleneck computes exact queue delays.

    Given:
      Edge length = 1000m, free traversal = 1.0 min
      Capacity = 600 veh/hr -> Slice capacity in 5 min = 600 * (5/60) = 50 veh
      Demand in 5-min slice = 150 veh (preexisting queue = 0)
    Then:
      Exit flow = 50 veh
      Remaining queue at end of slice = 150 - 50 = 100 veh
      Average queue = (0 + 100) / 2 = 50 veh
      Queue delay = (50 / 600) * 60 = 5.0 min
      Congested travel time = 1.0 + 5.0 = 6.0 min
    """
    model = CongestionModel(road_graph=simple_bottleneck_graph, use_point_queue=True)
    # Set capacity explicitly to 600 veh/hr
    model.edge_capacities[0].capacity_veh_per_hr = 600.0
    model.edge_capacities[0].free_traversal_time_min = 1.0

    congested_t, rem_q = model.compute_congested_time(
        edge_idx=0,
        flow_veh=150.0,
        time_slice_min=5.0,
        preexisting_queue=0.0,
    )

    assert np.isclose(rem_q, 100.0, atol=1e-5)
    assert np.isclose(congested_t, 6.0, atol=1e-5)

    # Next 5-min slice: 0 new demand, queue dissipates
    congested_t2, rem_q2 = model.compute_congested_time(
        edge_idx=0,
        flow_veh=0.0,
        time_slice_min=5.0,
        preexisting_queue=rem_q,  # 100 veh
    )
    # Exits 50 veh -> remaining queue = 50 veh
    # Average queue = (100 + 50)/2 = 75 veh -> delay = (75/600)*60 = 7.5 min
    assert np.isclose(rem_q2, 50.0, atol=1e-5)
    assert np.isclose(congested_t2, 1.0 + 7.5, atol=1e-5)


def test_emergency_contraflow_capacity_and_dissipation(simple_bottleneck_graph: RoadGraph) -> None:
    """Verify contraflow doubles capacity and reduces queue delay."""
    model = CongestionModel(road_graph=simple_bottleneck_graph, use_point_queue=True)
    model.edge_capacities[0].capacity_veh_per_hr = 600.0
    model.edge_capacities[0].free_traversal_time_min = 1.0

    # Normal congestion with 120 veh
    t_normal, q_normal = model.compute_congested_time(0, flow_veh=120.0, time_slice_min=5.0)
    # slice_cap = 50 -> rem_q = 70 -> avg_q = 35 -> delay = 3.5m -> t = 4.5m
    assert np.isclose(q_normal, 70.0, atol=1e-5)
    assert np.isclose(t_normal, 4.5, atol=1e-5)

    # Apply 2x contraflow -> capacity becomes 1200 veh/hr (slice_cap = 100 veh)
    model.apply_contraflow([0], capacity_multiplier=2.0)
    assert model.edge_capacities[0].capacity_veh_per_hr == 1200.0
    assert model.edge_capacities[0].is_contraflow

    t_contra, q_contra = model.compute_congested_time(0, flow_veh=120.0, time_slice_min=5.0)
    # slice_cap = 100 -> rem_q = 20 -> avg_q = 10 -> delay = (10/1200)*60 = 0.5m -> t = 1.5m
    assert np.isclose(q_contra, 20.0, atol=1e-5)
    assert np.isclose(t_contra, 1.5, atol=1e-5)
    assert t_contra < t_normal


def test_en_route_vehicle_replan_diversion() -> None:
    """Verify in-transit vehicle re-plans and diverts from mid-edge position."""
    # Build a Y-graph: 0 -> 1, then 1 -> 2 (Path A), and 1 -> 3 (Path B)
    nodes_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                shapely.geometry.Point(0.0, 0.0),
                shapely.geometry.Point(500.0, 0.0),
                shapely.geometry.Point(1000.0, 500.0),
                shapely.geometry.Point(1000.0, -500.0),
            ]
        },
        index=[0, 1, 2, 3],
        crs="EPSG:32610",
    )
    edges_gdf = gpd.GeoDataFrame(
        {
            "u": [0, 1, 1],
            "v": [1, 2, 3],
            "key": [0, 0, 0],
            "length": [500.0, 707.1, 707.1],
            "maxspeed": ["30 mph", "30 mph", "30 mph"],
            "highway": ["primary", "secondary", "secondary"],
            "geometry": [
                shapely.geometry.LineString([(0.0, 0.0), (500.0, 0.0)]),
                shapely.geometry.LineString([(500.0, 0.0), (1000.0, 500.0)]),
                shapely.geometry.LineString([(500.0, 0.0), (1000.0, -500.0)]),
            ],
        },
        crs="EPSG:32610",
    )
    rg = RoadGraph(nodes_gdf=nodes_gdf, edges_gdf=edges_gdf)
    x = np.linspace(-100.0, 1100.0, 20)
    y = np.linspace(600.0, -600.0, 20)
    arr_field = create_unburned_arrival_field(x, y)

    # Vehicle is on edge 0->1, 300m in, at current time t = 5.0 min
    veh = VehiclePosition(
        vehicle_id="evac_42",
        edge_u=0,
        edge_v=1,
        edge_key=0,
        offset_m=300.0,
        current_time_min=5.0,
    )

    # 1. Clear conditions: routes to nearest exit (either 2 or 3)
    route_clear = replan_en_route_vehicle(
        vehicle=veh,
        arrival_field=arr_field,
        road_graph=rg,
        destination_node_ids={2, 3},
        buffer_minutes=5.0,
    )
    assert route_clear is not None
    assert route_clear.edges[0].u == 0 and route_clear.edges[0].v == 1
    # Remaining length on edge 0->1 is 500 - 300 = 200m
    assert np.isclose(route_clear.edges[0].length_m, 200.0, atol=1e-5)

    # 2. Road 1->2 is closed mid-evacuation: vehicle must divert to 1->3
    route_diverted = replan_en_route_vehicle(
        vehicle=veh,
        arrival_field=arr_field,
        road_graph=rg,
        destination_node_ids={2, 3},
        buffer_minutes=5.0,
        closed_edges={(1, 2, 0)},
    )
    assert route_diverted is not None
    assert route_diverted.destination_id == 3  # Successfully diverted to alternate exit!
    assert route_diverted.node_ids == (0, 1, 3)


def test_destination_safety_buffer_filtering() -> None:
    """Verify exits within safety buffer of fire encroachment are flagged unsafe and avoided."""
    nodes_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                shapely.geometry.Point(0.0, 0.0),
                shapely.geometry.Point(1000.0, 0.0),  # Exit A (near fire)
                shapely.geometry.Point(0.0, 3000.0),  # Exit B (safe, 3km away)
            ]
        },
        index=[0, 10, 20],
        crs="EPSG:32610",
    )
    edges_gdf = gpd.GeoDataFrame(
        {
            "u": [0, 0],
            "v": [10, 20],
            "key": [0, 0],
            "length": [1000.0, 3000.0],
            "maxspeed": ["30 mph", "30 mph"],
            "highway": ["primary", "primary"],
            "geometry": [
                shapely.geometry.LineString([(0.0, 0.0), (1000.0, 0.0)]),
                shapely.geometry.LineString([(0.0, 0.0), (0.0, 3000.0)]),
            ],
        },
        crs="EPSG:32610",
    )
    rg = RoadGraph(nodes_gdf=nodes_gdf, edges_gdf=edges_gdf)
    rg.boundary_destinations = {"Exit_A": 10, "Exit_B": 20}

    x = np.linspace(-500.0, 4000.0, 50)
    y = np.linspace(4000.0, -500.0, 50)
    arr_field = create_unburned_arrival_field(x, y)

    # Burn area within 500m of Exit A (1000, 0) at t = 60 min
    xx, yy = np.meshgrid(x, y)
    dist_a = np.hypot(xx - 1000.0, yy - 0.0)
    arr_q10 = arr_field["arrival_q10"].values
    arr_q10[dist_a < 500.0] = 60.0
    arr_field["arrival_q10"].values = arr_q10

    # With safety buffer 1.0 km, Exit A is unsafe; Exit B (at (0, 3000), >3km away) is safe
    safe_dests, unsafe_dests = rg.check_destination_safety(arr_field, safety_buffer_km=1.0)
    assert 10 in unsafe_dests  # Exit A flagged unsafe!
    assert 20 in safe_dests  # Exit B remains safe

    # Test ReplanEngine integration
    engine = ReplanEngine(road_graph=rg)
    res = engine.replan(
        arrival_field=arr_field,
        origins=[0],
        destinations=[10, 20],
        t0=0.0,
        destination_safety_buffer_km=1.0,
    )
    # Must route to Exit B (node 20)
    assert res.routes["node_0"].destination_id == 20
    assert "node_10" in res.metadata.get("unsafe_destinations", "")


def test_hazard_miss_scoring(simple_bottleneck_graph: RoadGraph) -> None:
    """Verify hazard-miss scoring distinguishes between avoided and encountered hazards."""
    # Synthetic route from (0,0) to (1000,0) with exit at t = 20 min
    edge = RouteEdge(
        u=0,
        v=1,
        key=0,
        length_m=1000.0,
        speed_mps=15.0,
        traversal_time_min=20.0,
        entry_time_min=0.0,
        exit_time_min=20.0,
        fire_arrival_min=np.inf,
        margin_min=np.inf,
    )
    route = EvacuationRoute(
        origin_id=0,
        destination_id=1,
        departure_time_min=0.0,
        arrival_time_min=20.0,
        total_travel_time_min=20.0,
        min_margin_min=100.0,
        bottleneck_edge=None,
        edges=(edge,),
        node_ids=(0, 1),
        status=RouteStatus.SAFE,
        mode_used=RoutingMode.FASTEST_SAFE,
    )

    # Hazard 1: active at t = 100 min (vehicle passed long before -> MISSED/AVOIDED)
    h1 = HazardPoint(hazard_id="H1_Late", x=500.0, y=0.0, time_min=100.0, radius_m=50.0)
    # Hazard 2: active at t = 5 min (vehicle is on edge during [0, 20] -> ENCOUNTERED)
    h2 = HazardPoint(hazard_id="H2_Active", x=500.0, y=0.0, time_min=5.0, radius_m=50.0)

    res = score_route_against_hazards(route, hazards=[h1, h2], road_graph=simple_bottleneck_graph)
    assert res.total_hazards_checked == 2
    assert "H2_Active" in res.encountered_hazard_ids
    assert "H1_Late" not in res.encountered_hazard_ids
    assert res.encountered_hazard_count == 1
    assert res.missed_hazard_count == 1


def test_full_detail_camp_fire_demand_clearance_sanity(camp_fire_graph: RoadGraph) -> None:
    """Camp Fire demand scenario qualitative check: free-flow vs congested clearance."""
    demand_model = EvacuationDemandModel(
        config_path="configs/evacuation_demand.yaml",
        road_graph=camp_fire_graph,
    )
    congestion_model = CongestionModel(road_graph=camp_fire_graph, use_point_queue=True)
    router = TimeDependentRouter(road_graph=camp_fire_graph)
    engine = TimeSlicedAssignmentEngine(
        road_graph=camp_fire_graph,
        demand_model=demand_model,
        congestion_model=congestion_model,
        router=router,
    )

    edge_arrivals = np.full(len(camp_fire_graph.edges), np.inf, dtype=np.float64)
    destinations = set(camp_fire_graph.boundary_destinations.values())

    # Run free-flow assignment
    free_flow_results = engine.run_assignment(
        edge_fire_arrivals=edge_arrivals,
        destination_node_ids=destinations,
        free_flow_mode=True,
    )

    # Run congested assignment
    congested_results = engine.run_assignment(
        edge_fire_arrivals=edge_arrivals,
        destination_node_ids=destinations,
        free_flow_mode=False,
    )

    print("\n=== Camp Fire Evacuation Demand Clearance Sanity Check ===")
    for z_name, c_res in congested_results.items():
        f_res = free_flow_results[z_name]
        print(
            f"Zone: {z_name:<16} | Pop: {demand_model.zones[z_name].population:5d} | Veh: {demand_model.zones[z_name].num_vehicles:4d} | "
            f"Free-Flow: {f_res.clearance_time_hours:4.1f}h | Congested: {c_res.clearance_time_hours:4.1f}h | Delay: +{c_res.congestion_delay_hours:4.1f}h"
        )
        # Congested clearance should be >= free flow clearance
        assert c_res.clearance_time_min >= f_res.clearance_time_min


def test_latency_benchmark_100_random_fields(camp_fire_graph: RoadGraph) -> None:
    """Benchmark p50, p95, and max re-plan latency across 100 arrival fields on full-detail graph."""
    np.random.seed(42)
    assert camp_fire_graph.grid_x is not None and camp_fire_graph.grid_y is not None
    x = camp_fire_graph.grid_x
    y = camp_fire_graph.grid_y

    engine = ReplanEngine(road_graph=camp_fire_graph)
    origins = ["Paradise", "Magalia", "Concow"]
    destinations = ["West_Chico", "South_Oroville"]

    latencies_mode_a: list[float] = []
    latencies_mode_b: list[float] = []

    # Run 100 trials with randomized ignition locations and spread rates
    for _ in range(100):
        # Vary ignition coordinates and spread rate (some fast/heavily blocked)
        ign_x = float(np.random.uniform(615000.0, 635000.0))
        ign_y = float(np.random.uniform(4395000.0, 4415000.0))
        head_rate = float(np.random.uniform(20.0, 80.0))  # 20 to 80 m/min

        arr_field = create_wind_ellipse_arrival_field(
            x_coords=x,
            y_coords=y,
            ignition_x=ign_x,
            ignition_y=ign_y,
            head_rate_m_per_min=head_rate,
            back_rate_m_per_min=5.0,
            length_to_breadth=2.5,
            wind_towards_deg=float(np.random.uniform(200.0, 260.0)),
            horizon_minutes=720.0,
        )

        # Mode (a) test
        res_a = engine.replan(
            arrival_field=arr_field,
            origins=origins,
            destinations=destinations,
            t0=float(np.random.uniform(30.0, 120.0)),
            buffer_minutes=15.0,
            mode=RoutingMode.FASTEST_SAFE,
            validate_routes=False,  # Benchmark pure routing engine latency
        )
        latencies_mode_a.append(res_a.replan_latency_sec)

        # Mode (b) test (heavily constrained, force max-min margin)
        res_b = engine.replan(
            arrival_field=arr_field,
            origins=origins,
            destinations=destinations,
            t0=float(np.random.uniform(30.0, 120.0)),
            buffer_minutes=15.0,
            mode=RoutingMode.MAX_MIN_MARGIN,
            validate_routes=False,
        )
        latencies_mode_b.append(res_b.replan_latency_sec)

    p50_a = float(np.percentile(latencies_mode_a, 50))
    p95_a = float(np.percentile(latencies_mode_a, 95))
    max_a = float(np.max(latencies_mode_a))

    p50_b = float(np.percentile(latencies_mode_b, 50))
    p95_b = float(np.percentile(latencies_mode_b, 95))
    max_b = float(np.max(latencies_mode_b))

    print("\n=== Re-Plan Latency Benchmark (100 Arrival Fields, Full-Detail 12,379 Edges) ===")
    print(
        f"Mode (a) FASTEST_SAFE : p50 = {p50_a * 1000:5.1f} ms | p95 = {p95_a * 1000:5.1f} ms | max = {max_a * 1000:5.1f} ms"
    )
    print(
        f"Mode (b) MAX_MIN_MARGIN: p50 = {p50_b * 1000:5.1f} ms | p95 = {p95_b * 1000:5.1f} ms | max = {max_b * 1000:5.1f} ms"
    )

    assert max_a < 5.0, f"Mode (a) latency exceeded 5s target: {max_a:.3f}s"
    assert max_b < 5.0, f"Mode (b) latency exceeded 5s target: {max_b:.3f}s"
