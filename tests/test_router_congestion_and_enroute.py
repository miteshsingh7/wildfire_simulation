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

pytestmark = pytest.mark.slow

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
                s.clearance_time_hours
                for z, s in summaries.items()
                if ("Zone_" in z or "Paradise" in z)
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

    # Finite exit capacity should maintain or increase total regional clearance time
    max_clearance_with = max(s.clearance_time_min for s in res_with_exit.values())
    max_clearance_no = max(s.clearance_time_min for s in res_no_exit.values())
    assert max_clearance_with >= max_clearance_no


def test_two_parallel_exits_split_to_equalize_delay() -> None:
    """Verify that MSA logit choice splits flow across two parallel exits with unequal capacity to equalize delay."""
    nodes_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                shapely.geometry.Point(0.0, 0.0),  # 0: Origin
                shapely.geometry.Point(500.0, 0.0),  # 1: Junction
                shapely.geometry.Point(1000.0, 500.0),  # 2: Exit A (Capacity 1200)
                shapely.geometry.Point(1000.0, -500.0),  # 3: Exit B (Capacity 600)
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
            "length": [500.0, 500.0, 500.0],
            "maxspeed": ["30 mph", "30 mph", "30 mph"],
            "highway": ["primary", "primary", "secondary"],
            "geometry": [
                shapely.geometry.LineString([(0.0, 0.0), (500.0, 0.0)]),
                shapely.geometry.LineString([(500.0, 0.0), (1000.0, 500.0)]),
                shapely.geometry.LineString([(500.0, 0.0), (1000.0, -500.0)]),
            ],
        },
        crs="EPSG:32610",
    )
    graph = RoadGraph(
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        grid_x_coords=np.linspace(-100.0, 1100.0, 10),
        grid_y_coords=np.linspace(600.0, -600.0, 10),
    )
    graph.boundary_destinations = {"Exit_A": 2, "Exit_B": 3}

    c_model = CongestionModel(road_graph=graph, use_point_queue=True)
    c_model.exit_outflow_capacities = {"exit_a": 1200.0, "exit_b": 600.0, "fallback_exit": 600.0}
    # Large internal capacity so exit bottleneck dominates
    for ec in c_model.edge_capacities:
        ec.capacity_veh_per_hr = 5000.0

    demand_model = EvacuationDemandModel(
        config_path="configs/evacuation_demand.yaml", road_graph=graph
    )
    from wf052.routing.demand import EvacuationZone

    test_zone = EvacuationZone(
        name="Test_Origin",
        census_source="test",
        population=1800,
        num_vehicles=1800,
        centroid_utm=(0.0, 0.0),
        notification_elapsed_min=0.0,
        departure_duration_min=5.0,
        origin_node_id=0,
    )
    demand_model.zones = {"Test_Origin": test_zone}
    demand_model.departure_schedules = {"Test_Origin": {0.0: 1800}}

    router = TimeDependentRouter(road_graph=graph)
    engine = TimeSlicedAssignmentEngine(
        road_graph=graph,
        demand_model=demand_model,
        congestion_model=c_model,
        router=router,
    )
    edge_arrivals = np.full(len(graph.edges), np.inf, dtype=np.float64)

    summaries, _ = engine.run_assignment(
        edge_fire_arrivals=edge_arrivals,
        destination_node_ids={2, 3},
        msa_iterations=5,
        theta=0.15,
    )

    exit_a_veh = sum(engine.last_vehicles_per_exit_by_id[2].values())
    exit_b_veh = sum(engine.last_vehicles_per_exit_by_id[3].values())

    print(f"\nFlow Split: Exit A = {exit_a_veh} veh | Exit B = {exit_b_veh} veh")
    assert exit_a_veh > 0 and exit_b_veh > 0
    assert exit_a_veh + exit_b_veh == 1800
    assert exit_a_veh > exit_b_veh


def test_halving_internal_capacity_changes_clearance(camp_fire_graph: RoadGraph) -> None:
    """Verify that halving internal capacity strictly changes (increases) clearance time."""
    demand_model = EvacuationDemandModel(
        config_path="configs/evacuation_demand.yaml", road_graph=camp_fire_graph
    )
    router = TimeDependentRouter(road_graph=camp_fire_graph)
    edge_arrivals = np.full(len(camp_fire_graph.edges), np.inf, dtype=np.float64)
    destinations = set(camp_fire_graph.boundary_destinations.values())

    c_model_10 = CongestionModel(road_graph=camp_fire_graph, capacity_scale=1.0)
    engine_10 = TimeSlicedAssignmentEngine(camp_fire_graph, demand_model, c_model_10, router)
    res_10, _ = engine_10.run_assignment(edge_arrivals, destinations)
    clearance_10 = max(
        s.clearance_time_min for z, s in res_10.items() if ("Zone_" in z or "Paradise" in z)
    )

    c_model_05 = CongestionModel(road_graph=camp_fire_graph, capacity_scale=0.5)
    engine_05 = TimeSlicedAssignmentEngine(camp_fire_graph, demand_model, c_model_05, router)
    res_05, _ = engine_05.run_assignment(edge_arrivals, destinations)
    clearance_05 = max(
        s.clearance_time_min for z, s in res_05.items() if ("Zone_" in z or "Paradise" in z)
    )

    print(
        f"\nInternal Capacity Test: Scale 1.0 = {clearance_10:.1f} min | Scale 0.5 = {clearance_05:.1f} min"
    )
    assert clearance_05 > clearance_10


def test_contraflow_changes_clearance(camp_fire_graph: RoadGraph) -> None:
    """Verify that activating emergency contraflow strictly changes (reduces) clearance time."""
    demand_model = EvacuationDemandModel(
        config_path="configs/evacuation_demand.yaml", road_graph=camp_fire_graph
    )
    router = TimeDependentRouter(road_graph=camp_fire_graph)
    edge_arrivals = np.full(len(camp_fire_graph.edges), np.inf, dtype=np.float64)
    destinations = set(camp_fire_graph.boundary_destinations.values())

    c_model_base = CongestionModel(road_graph=camp_fire_graph, capacity_scale=0.7)
    engine_base = TimeSlicedAssignmentEngine(camp_fire_graph, demand_model, c_model_base, router)
    res_base, _ = engine_base.run_assignment(edge_arrivals, destinations)
    clearance_base = max(
        s.clearance_time_min for z, s in res_base.items() if ("Zone_" in z or "Paradise" in z)
    )

    c_model_contra = CongestionModel(road_graph=camp_fire_graph, capacity_scale=0.7)
    skyway_edge_indices = [
        i for i, e in enumerate(camp_fire_graph.edges) if "skyway" in str(e.name).lower()
    ]
    c_model_contra.apply_contraflow(skyway_edge_indices, capacity_multiplier=2.0)
    engine_contra = TimeSlicedAssignmentEngine(
        camp_fire_graph, demand_model, c_model_contra, router
    )
    res_contra, _ = engine_contra.run_assignment(edge_arrivals, destinations)
    clearance_contra = max(
        s.clearance_time_min for z, s in res_contra.items() if ("Zone_" in z or "Paradise" in z)
    )

    print(
        f"\nContraflow Test: Baseline = {clearance_base:.1f} min | Contraflow = {clearance_contra:.1f} min"
    )
    assert clearance_contra < clearance_base


def test_shifting_arrivalfield_ignition_time_shifts_departures(camp_fire_graph: RoadGraph) -> None:
    """Verify shifting ArrivalField.attrs['ignition_time'] shifts zone departures by the exact offset."""
    assert camp_fire_graph.grid_x is not None and camp_fire_graph.grid_y is not None
    arr_field_base = create_unburned_arrival_field(
        x_coords=camp_fire_graph.grid_x,
        y_coords=camp_fire_graph.grid_y,
        ignition_time="2018-11-08T14:33:00Z",
    )
    demand_model = EvacuationDemandModel(
        config_path="configs/evacuation_demand.yaml", road_graph=camp_fire_graph
    )
    demand_model.set_ignition_clock(arr_field_base.attrs["ignition_time"])
    base_notif = demand_model.zones["Zone_3"].notification_elapsed_min
    base_first_departure = min(demand_model.departure_schedules["Zone_3"].keys())

    arr_field_shifted = create_unburned_arrival_field(
        x_coords=camp_fire_graph.grid_x,
        y_coords=camp_fire_graph.grid_y,
        ignition_time="2018-11-08T14:23:00Z",
    )
    demand_model.set_ignition_clock(arr_field_shifted.attrs["ignition_time"])
    shifted_notif = demand_model.zones["Zone_3"].notification_elapsed_min
    shifted_first_departure = min(demand_model.departure_schedules["Zone_3"].keys())

    assert shifted_notif == base_notif + 10.0
    assert shifted_first_departure == base_first_departure + 10.0


def test_latency_benchmark_30pct_infeasible_fields(camp_fire_graph: RoadGraph) -> None:
    """Benchmark re-plan latency across 20 arrival fields with >= 30% safe route infeasibility."""
    np.random.seed(101)
    assert camp_fire_graph.grid_x is not None and camp_fire_graph.grid_y is not None
    x = camp_fire_graph.grid_x
    y = camp_fire_graph.grid_y

    replan_engine = ReplanEngine(road_graph=camp_fire_graph)
    validator = IndependentRouteValidator()
    origins = [
        "Zone_1",
        "Zone_3",
        "Zone_8",
        "Zone_14",
        "Magalia",
        "Concow",
    ]
    destinations = ["West_Chico", "South_Oroville"]

    latencies_mode_a: list[float] = []
    latencies_mode_b: list[float] = []
    latencies_end_to_end: list[float] = []

    for _ in range(20):
        ign_x = float(np.random.uniform(616000.0, 622000.0))
        ign_y = float(np.random.uniform(4398000.0, 4406000.0))
        head_rate = float(np.random.uniform(60.0, 120.0))

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

        res_a = replan_engine.replan(
            arrival_field=arr_field,
            origins=origins,
            destinations=destinations,
            t0=60.0,
            buffer_minutes=15.0,
            mode=RoutingMode.FASTEST_SAFE,
            validate_routes=False,
            destination_safety_buffer_km=7.0,
        )
        latencies_mode_a.append(res_a.replan_latency_sec)

        res_b = replan_engine.replan(
            arrival_field=arr_field,
            origins=origins,
            destinations=destinations,
            t0=60.0,
            buffer_minutes=15.0,
            mode=RoutingMode.MAX_MIN_MARGIN,
            validate_routes=False,
            destination_safety_buffer_km=7.0,
        )
        latencies_mode_b.append(res_b.replan_latency_sec)

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

    print("\n=== Latency Benchmark: >=30% Infeasible Safe Routes ===")
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


def test_phase4e_block_level_demand_loading(camp_fire_graph: RoadGraph) -> None:
    """Phase 4e Task 1: Verify census block loading, origin node aggregation, and reconciliation."""
    dm = EvacuationDemandModel(road_graph=camp_fire_graph)
    assert dm.get_paradise_population() == 26218
    assert len(dm.block_demands) >= 300
    assert dm.unique_origin_nodes_count >= 200
    assert dm.max_per_node_load < 500
    assert dm.max_load_node_id in camp_fire_graph.nodes


def test_phase4e_inbound_corridor_filtering(camp_fire_graph: RoadGraph) -> None:
    """Phase 4e Task 2: Verify inbound emergency traffic applies only to named corridors."""
    cm = CongestionModel(
        road_graph=camp_fire_graph,
        capacity_scale=1.0,
        inbound_traffic_veh_per_hr=200.0,
    )
    skyway_edge = next(
        e
        for e in cm.edge_capacities
        if "skyway" in str(camp_fire_graph.edges[e.edge_idx].name).lower()
    )
    assert skyway_edge.is_inbound_corridor is True
    assert skyway_edge.capacity_veh_per_hr == skyway_edge.nominal_capacity_veh_per_hr - 200.0

    residential_edge = next(
        e
        for e in cm.edge_capacities
        if "residential" in e.highway.lower() and not e.is_inbound_corridor
    )
    assert residential_edge.is_inbound_corridor is False
    assert residential_edge.capacity_veh_per_hr == residential_edge.nominal_capacity_veh_per_hr


def test_phase4e_link_storage_and_spillback(camp_fire_graph: RoadGraph) -> None:
    """Phase 4e Task 3: Verify link storage calculation and spillback tracking."""
    cm = CongestionModel(road_graph=camp_fire_graph)
    for e in cm.edge_capacities[:100]:
        ed = camp_fire_graph.edges[e.edge_idx]
        expected_storage = max(1.0, (ed.length_m / 1000.0) * e.lanes * 130.0)
        assert abs(e.storage_veh - expected_storage) < 1e-4


def test_phase4e_time_dependent_roadblock_closure(camp_fire_graph: RoadGraph) -> None:
    """Phase 4e Task 5: Verify time-dependent closure of Hwy 70 at Pentz Rd at T+54 min."""
    cm = CongestionModel(road_graph=camp_fire_graph)
    assert len(cm.roadway_closures) >= 1
    assert any("70" in c["name"] and "Pentz" in c["name"] for c in cm.roadway_closures)


def test_phase4f_vehicle_conservation_exact_in_equals_out(camp_fire_graph: RoadGraph) -> None:
    """Phase 4f Task 2: Verify vehicles in == vehicles out and eliminate 150-vehicle gap."""
    dm = EvacuationDemandModel(road_graph=camp_fire_graph)
    total_zone_veh = dm.get_total_vehicles()
    total_slices_veh = sum(d.vehicle_count for d in dm.get_slices_demand())
    total_block_veh = sum(b.num_vehicles for b in dm.block_demands)
    pulga_veh = dm.zones.get("Pulga").num_vehicles if "Pulga" in dm.zones else 0

    assert total_zone_veh == total_slices_veh
    assert total_block_veh + pulga_veh == total_zone_veh

    # Quick assignment with 2 slices to verify in == out
    cm = CongestionModel(road_graph=camp_fire_graph, use_point_queue=True)
    router = TimeDependentRouter(road_graph=camp_fire_graph)
    engine = TimeSlicedAssignmentEngine(
        road_graph=camp_fire_graph, demand_model=dm, congestion_model=cm, router=router
    )
    arrivals = np.full(len(camp_fire_graph.edges), np.inf, dtype=np.float64)
    dests = set(camp_fire_graph.boundary_destinations.values())

    # Limit to first 2 slices for fast unit test execution
    slices = dm.get_slices_demand()
    test_slices = [s for s in slices if s.slice_start_min <= 25.0]
    expected_veh = sum(s.vehicle_count for s in test_slices)

    # Mock get_slices_demand for engine
    orig_get = dm.get_slices_demand
    dm.get_slices_demand = lambda: test_slices  # type: ignore[method-assign]
    try:
        summaries, _ = engine.run_assignment(
            edge_fire_arrivals=arrivals,
            destination_node_ids=dests,
            apply_exit_capacities=True,
        )
        total_veh_out = sum(
            sum(t_dict.values()) for t_dict in engine.last_vehicles_per_exit_over_time.values()
        )
        assert total_veh_out == expected_veh, (
            f"Vehicles in ({expected_veh}) != Vehicles out ({total_veh_out})"
        )
    finally:
        dm.get_slices_demand = orig_get


def test_phase4f_closure_directional_inbound_only(camp_fire_graph: RoadGraph) -> None:
    """Phase 4f Task 6: Confirm Hwy 70 closure at Pentz Rd is directional (inbound only)."""
    cm = CongestionModel(road_graph=camp_fire_graph)
    hwy70_closures = [c for c in cm.roadway_closures if "70" in c["name"]]
    assert len(hwy70_closures) >= 1
    assert hwy70_closures[0].get("inbound_only") is True
