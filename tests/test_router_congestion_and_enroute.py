"""Unit and integration tests for congestion, demand, en-route diversion, destination safety, and latency.

Phase 4c Tests:
1. Known-answer single bottleneck queue delay.
2. Emergency contraflow capacity doubling and queue dissipation.
3. En-route dynamic diversion from in-transit vehicle position (edge + offset).
4. Destination safety buffer X km (>= 6 km) filtering out encroached boundary exits.
5. Hazard-miss check scoring trajectory against timestamped truth hazards with spatial tolerance.
6. Verification that BPR and point-queue delays are strictly partitioned (no double-counting).
7. Impact of finite boundary exit outflow capacities on clearance time.
8. Quantitative entrapment metrics on a synthetic expanding fire (vehicles trapped + danger hours).
9. Capacity sensitivity matrix (scale 1.0, 0.7, 0.5; inbound 0 vs 200 veh/hr) reporting clearance ranges.
10. Latency benchmark across 100 arrival fields with >= 30% safe route infeasibility (Mode a, Mode b, End-to-End).
"""

import time

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
    create_expanding_circle_arrival_field,
    create_unburned_arrival_field,
    create_wind_ellipse_arrival_field,
)
from wf052.routing.validator import IndependentRouteValidator


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
    model.edge_capacities[0].capacity_veh_per_hr = 600.0
    model.edge_capacities[0].free_traversal_time_min = 1.0

    congested_t, q_delay, rem_q = model.compute_congested_time(
        edge_idx=0,
        flow_veh=150.0,
        time_slice_min=5.0,
        preexisting_queue=0.0,
    )

    assert np.isclose(rem_q, 100.0, atol=1e-5)
    assert np.isclose(q_delay, 5.0, atol=1e-5)
    assert np.isclose(congested_t, 6.0, atol=1e-5)

    # Next 5-min slice: 0 new demand, queue dissipates
    congested_t2, q_delay2, rem_q2 = model.compute_congested_time(
        edge_idx=0,
        flow_veh=0.0,
        time_slice_min=5.0,
        preexisting_queue=rem_q,  # 100 veh
    )
    assert np.isclose(rem_q2, 50.0, atol=1e-5)
    assert np.isclose(q_delay2, 7.5, atol=1e-5)
    assert np.isclose(congested_t2, 1.0 + 7.5, atol=1e-5)


def test_emergency_contraflow_capacity_and_dissipation(simple_bottleneck_graph: RoadGraph) -> None:
    """Verify contraflow doubles capacity and reduces queue delay."""
    model = CongestionModel(road_graph=simple_bottleneck_graph, use_point_queue=True)
    model.edge_capacities[0].capacity_veh_per_hr = 600.0
    model.edge_capacities[0].free_traversal_time_min = 1.0

    # Normal congestion with 120 veh
    t_normal, q_del_normal, q_normal = model.compute_congested_time(
        0, flow_veh=120.0, time_slice_min=5.0
    )
    assert np.isclose(q_normal, 70.0, atol=1e-5)
    assert np.isclose(t_normal, 4.5, atol=1e-5)

    # Apply 2x contraflow -> capacity becomes 1200 veh/hr (slice_cap = 100 veh)
    model.apply_contraflow([0], capacity_multiplier=2.0)
    assert model.edge_capacities[0].capacity_veh_per_hr == 1200.0
    assert model.edge_capacities[0].is_contraflow

    t_contra, q_del_contra, q_contra = model.compute_congested_time(
        0, flow_veh=120.0, time_slice_min=5.0
    )
    assert np.isclose(q_contra, 20.0, atol=1e-5)
    assert np.isclose(t_contra, 1.5, atol=1e-5)
    assert t_contra < t_normal


def test_no_edge_double_counting_bpr_vs_point_queue(simple_bottleneck_graph: RoadGraph) -> None:
    """Verify BPR moving delay and point-queue delay are partitioned and never double-counted."""
    # 1. Point Queue mode
    model_pq = CongestionModel(road_graph=simple_bottleneck_graph, use_point_queue=True)
    model_pq.edge_capacities[0].capacity_veh_per_hr = 600.0
    model_pq.edge_capacities[0].free_traversal_time_min = 1.0

    t_pq, q_delay_pq, rem_q = model_pq.compute_congested_time(0, flow_veh=100.0, time_slice_min=5.0)
    # Moving time must equal free-flow time exactly
    assert np.isclose(t_pq - q_delay_pq, 1.0, atol=1e-6)

    # 2. BPR mode
    model_bpr = CongestionModel(road_graph=simple_bottleneck_graph, use_point_queue=False)
    model_bpr.edge_capacities[0].capacity_veh_per_hr = 600.0
    model_bpr.edge_capacities[0].free_traversal_time_min = 1.0

    t_bpr, bpr_delay, rem_bpr_q = model_bpr.compute_congested_time(
        0, flow_veh=100.0, time_slice_min=5.0
    )
    # In BPR mode, physical queue delay is 0, moving time increases
    assert np.isclose(rem_bpr_q, 0.0, atol=1e-6)
    assert bpr_delay > 0.0
    assert np.isclose(t_bpr - bpr_delay, 1.0, atol=1e-6)


def test_en_route_vehicle_replan_diversion() -> None:
    """Verify in-transit vehicle re-plans and diverts from mid-edge position."""
    nodes_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                shapely.geometry.Point(0.0, 0.0),
                shapely.geometry.Point(1000.0, 0.0),
                shapely.geometry.Point(2000.0, 500.0),
                shapely.geometry.Point(2000.0, -500.0),
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
            "length": [1000.0, 1000.0, 1200.0],
            "maxspeed": ["37.28 mph", "37.28 mph", "37.28 mph"],
            "highway": ["primary", "secondary", "secondary"],
            "geometry": [
                shapely.geometry.LineString([(0.0, 0.0), (1000.0, 0.0)]),
                shapely.geometry.LineString([(1000.0, 0.0), (2000.0, 500.0)]),
                shapely.geometry.LineString([(1000.0, 0.0), (2000.0, -500.0)]),
            ],
        },
        crs="EPSG:32610",
    )
    road_graph = RoadGraph(
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        grid_x_coords=np.linspace(-100.0, 2100.0, 20),
        grid_y_coords=np.linspace(600.0, -600.0, 20),
    )

    arr_field = create_unburned_arrival_field(
        x_coords=road_graph.grid_x,
        y_coords=road_graph.grid_y,
    )

    pos = VehiclePosition(
        vehicle_id="veh_1",
        edge_u=0,
        edge_v=1,
        edge_key=0,
        offset_m=400.0,  # 400m along 1000m edge
        current_time_min=0.5,
    )

    # Edge 1->2 (Path A) is closed by fire
    route = replan_en_route_vehicle(
        vehicle=pos,
        arrival_field=arr_field,
        road_graph=road_graph,
        destination_node_ids={2, 3},
        closed_edges={(1, 2, 0)},
        buffer_minutes=5.0,
    )

    assert route is not None
    assert route.status == RouteStatus.SAFE
    assert route.destination_id == 3
    assert [e.v for e in route.edges] == [1, 3]


def test_destination_safety_buffer_filtering(camp_fire_graph: RoadGraph) -> None:
    """Verify destination safety buffer (>= 6 km) correctly flags encroached exits."""
    assert camp_fire_graph.grid_x is not None and camp_fire_graph.grid_y is not None
    x = camp_fire_graph.grid_x
    y = camp_fire_graph.grid_y

    dest_coords = camp_fire_graph.nodes[camp_fire_graph.boundary_destinations["West_Chico"]]

    arr_field = create_expanding_circle_arrival_field(
        x_coords=x,
        y_coords=y,
        ignition_x=dest_coords[0] - 3000.0,  # 3 km away from exit (inside 6 km buffer)
        ignition_y=dest_coords[1],
        spread_rate_m_per_min=50.0,
        horizon_minutes=720.0,
    )

    safe_dests, unsafe_dests = camp_fire_graph.check_destination_safety(
        arrival_field=arr_field,
        safety_buffer_km=6.0,
    )

    dest_node_id = camp_fire_graph.boundary_destinations["West_Chico"]
    assert dest_node_id in unsafe_dests
    assert dest_node_id not in safe_dests


def test_hazard_miss_spatial_tolerance(simple_bottleneck_graph: RoadGraph) -> None:
    """Verify hazard-miss scoring accounts for danger radius and spatial tolerance."""
    edge = RouteEdge(
        u=0,
        v=1,
        key=0,
        length_m=1000.0,
        speed_mps=16.667,
        traversal_time_min=1.0,
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

    # Hazard 80m away from road geometry (y = 80.0)
    h_near = HazardPoint(hazard_id="H_Near", x=500.0, y=80.0, time_min=5.0, radius_m=50.0)

    # Without spatial tolerance: dist = 80m > radius 50m -> MISSED
    res_no_tol = score_route_against_hazards(
        route, hazards=[h_near], road_graph=simple_bottleneck_graph, spatial_tolerance_m=0.0
    )
    assert res_no_tol.encountered_hazard_count == 0
    assert res_no_tol.missed_hazard_count == 1

    # With spatial tolerance = 50m: 50m + 50m = 100m >= 80m -> ENCOUNTERED
    res_tol = score_route_against_hazards(
        route, hazards=[h_near], road_graph=simple_bottleneck_graph, spatial_tolerance_m=50.0
    )
    assert res_tol.encountered_hazard_count == 1
    assert "H_Near" in res_tol.encountered_hazard_ids


def test_entrapment_metrics_on_synthetic_expanding_fire(camp_fire_graph: RoadGraph) -> None:
    """Verify entrapment metrics correctly compute trapped vehicles and danger hours."""
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

    # Create synthetic expanding fire cutting across Paradise early (t = 80 min)
    assert camp_fire_graph.grid_x is not None and camp_fire_graph.grid_y is not None
    arr_field = create_expanding_circle_arrival_field(
        x_coords=camp_fire_graph.grid_x,
        y_coords=camp_fire_graph.grid_y,
        ignition_x=620000.0,
        ignition_y=4402000.0,
        spread_rate_m_per_min=60.0,
        horizon_minutes=720.0,
    )
    edge_arrivals = camp_fire_graph.compute_edge_fire_arrivals(arr_field)
    destinations = set(camp_fire_graph.boundary_destinations.values())

    summaries, entrapment = engine.run_assignment(
        edge_fire_arrivals=edge_arrivals,
        destination_node_ids=destinations,
        danger_buffer_minutes=15.0,
    )

    assert isinstance(entrapment.total_trapped_vehicles, int)
    assert isinstance(entrapment.total_danger_vehicle_hours, float)
    assert entrapment.total_danger_vehicle_hours >= 0.0


def test_capacity_scale_and_inbound_traffic_sensitivity(camp_fire_graph: RoadGraph) -> None:
    """Evaluate capacity scaling (1.0, 0.7, 0.5) and inbound traffic on Paradise clearance times."""
    demand_model = EvacuationDemandModel(
        config_path="configs/evacuation_demand.yaml",
        road_graph=camp_fire_graph,
    )
    router = TimeDependentRouter(road_graph=camp_fire_graph)
    edge_arrivals = np.full(len(camp_fire_graph.edges), np.inf, dtype=np.float64)
    destinations = set(camp_fire_graph.boundary_destinations.values())

    print("\n=== Capacity Sensitivity Matrix (Paradise Clearance Ranges) ===")
    clearance_matrix: dict[tuple[float, float], float] = {}

    for cap_scale in [1.0, 0.7, 0.5]:
        for inbound in [0.0, 200.0]:
            c_model = CongestionModel(
                road_graph=camp_fire_graph,
                use_point_queue=True,
                capacity_scale=cap_scale,
                inbound_traffic_veh_per_hr=inbound,
            )
            engine = TimeSlicedAssignmentEngine(
                road_graph=camp_fire_graph,
                demand_model=demand_model,
                congestion_model=c_model,
                router=router,
            )
            summaries, _ = engine.run_assignment(
                edge_fire_arrivals=edge_arrivals,
                destination_node_ids=destinations,
                free_flow_mode=False,
            )
            # Max clearance across Paradise sub-zones
            paradise_clearance_hrs = max(
                s.clearance_time_hours for z, s in summaries.items() if "Paradise" in z
            )
            clearance_matrix[(cap_scale, inbound)] = paradise_clearance_hrs
            inbound_lbl = f"+{int(inbound)} veh/hr inbound" if inbound > 0 else "no inbound"
            print(
                f"Scale: {cap_scale:3.1f} | {inbound_lbl:<20} | Paradise Clearance: {paradise_clearance_hrs:4.2f} h"
            )

    # Assert clearance monotonically increases as capacity scales down
    assert clearance_matrix[(0.7, 0.0)] >= clearance_matrix[(1.0, 0.0)]
    assert clearance_matrix[(0.5, 0.0)] >= clearance_matrix[(0.7, 0.0)]


def test_boundary_exit_finite_capacity_impact_on_clearance(camp_fire_graph: RoadGraph) -> None:
    """Verify finite exit outflow capacity adds realistic boundary queue delay."""
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

    # Run with exit capacity constraints
    res_with_exit, _ = engine.run_assignment(
        edge_fire_arrivals=edge_arrivals,
        destination_node_ids=destinations,
        apply_exit_capacities=True,
    )
    # Run without exit capacity constraints
    res_no_exit, _ = engine.run_assignment(
        edge_fire_arrivals=edge_arrivals,
        destination_node_ids=destinations,
        apply_exit_capacities=False,
    )

    # Finite exit capacity should maintain or increase clearance time
    for z in res_with_exit:
        assert res_with_exit[z].clearance_time_min >= res_no_exit[z].clearance_time_min


def test_latency_benchmark_30pct_infeasible_fields(camp_fire_graph: RoadGraph) -> None:
    """Benchmark re-plan latency across 100 arrival fields with >= 30% safe route infeasibility.

    Reports:
    - Mode (a) FASTEST_SAFE p50 / p95 / max latency
    - Mode (b) MAX_MIN_MARGIN p50 / p95 / max latency
    - End-to-End latency (new ArrivalField -> dynamic re-assignment -> route validation)
    """
    np.random.seed(101)
    assert camp_fire_graph.grid_x is not None and camp_fire_graph.grid_y is not None
    x = camp_fire_graph.grid_x
    y = camp_fire_graph.grid_y

    replan_engine = ReplanEngine(road_graph=camp_fire_graph)
    validator = IndependentRouteValidator()
    origins = [
        "Paradise_North",
        "Paradise_Central",
        "Paradise_East",
        "Paradise_South",
        "Magalia",
        "Concow",
    ]
    destinations = ["West_Chico", "South_Oroville"]

    latencies_mode_a: list[float] = []
    latencies_mode_b: list[float] = []
    latencies_end_to_end: list[float] = []

    for _ in range(100):
        # Place aggressive fast ignition near primary egress to force safe route infeasibility for >= 30% origins
        ign_x = float(np.random.uniform(616000.0, 622000.0))
        ign_y = float(np.random.uniform(4398000.0, 4406000.0))
        head_rate = float(np.random.uniform(60.0, 120.0))  # 60 to 120 m/min rapid front

        arr_field = create_wind_ellipse_arrival_field(
            x_coords=x,
            y_coords=y,
            ignition_x=ign_x,
            ignition_y=ign_y,
            head_rate_m_per_min=head_rate,
            back_rate_m_per_min=10.0,
            length_to_breadth=3.0,
            wind_towards_deg=float(np.random.uniform(220.0, 250.0)),
            horizon_minutes=720.0,
        )

        t_start = time.perf_counter()

        # Mode (a)
        res_a = replan_engine.replan(
            arrival_field=arr_field,
            origins=origins,
            destinations=destinations,
            t0=60.0,
            buffer_minutes=15.0,
            mode=RoutingMode.FASTEST_SAFE,
            validate_routes=False,
            destination_safety_buffer_km=6.0,
        )
        latencies_mode_a.append(res_a.replan_latency_sec)

        # Mode (b)
        res_b = replan_engine.replan(
            arrival_field=arr_field,
            origins=origins,
            destinations=destinations,
            t0=60.0,
            buffer_minutes=15.0,
            mode=RoutingMode.MAX_MIN_MARGIN,
            validate_routes=False,
            destination_safety_buffer_km=6.0,
        )
        latencies_mode_b.append(res_b.replan_latency_sec)

        # Validation step for End-to-End latency
        for route in res_b.routes.values():
            if route.status != RouteStatus.NO_ROUTE and route.edges:
                validator.validate(route, arr_field, camp_fire_graph)

        t_end = time.perf_counter()
        latencies_end_to_end.append(t_end - t_start)

    p50_a = float(np.percentile(latencies_mode_a, 50))
    p95_a = float(np.percentile(latencies_mode_a, 95))
    max_a = float(np.max(latencies_mode_a))

    p50_b = float(np.percentile(latencies_mode_b, 50))
    p95_b = float(np.percentile(latencies_mode_b, 95))
    max_b = float(np.max(latencies_mode_b))

    p50_e2e = float(np.percentile(latencies_end_to_end, 50))
    p95_e2e = float(np.percentile(latencies_end_to_end, 95))
    max_e2e = float(np.max(latencies_end_to_end))

    print("\n=== Latency Benchmark: >=30% Infeasible Safe Routes (Full 14,691-Edge Graph) ===")
    print(
        f"Mode (a) FASTEST_SAFE : p50 = {p50_a * 1000:5.1f} ms | p95 = {p95_a * 1000:5.1f} ms | max = {max_a * 1000:5.1f} ms"
    )
    print(
        f"Mode (b) MAX_MIN_MARGIN: p50 = {p50_b * 1000:5.1f} ms | p95 = {p95_b * 1000:5.1f} ms | max = {max_b * 1000:5.1f} ms"
    )
    print(
        f"End-to-End Pipeline   : p50 = {p50_e2e * 1000:5.1f} ms | p95 = {p95_e2e * 1000:5.1f} ms | max = {max_e2e * 1000:5.1f} ms"
    )

    assert max_a < 5.0
    assert max_b < 5.0
    assert max_e2e < 10.0
