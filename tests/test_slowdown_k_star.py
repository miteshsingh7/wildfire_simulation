"""Unit tests for critical slowdown multiplier k* on synthetic fires."""

import pytest

from wf052.routing.graph import RoadGraph, load_camp_fire_road_graph
from wf052.routing.slowdown import compute_critical_slowdown_k_star, compute_k_star_grid
from wf052.routing.synthetic_fire import create_wind_ellipse_arrival_field


@pytest.fixture(scope="module")
def camp_fire_graph() -> RoadGraph:
    return load_camp_fire_road_graph(full_detail=False)


def test_k_star_no_fire(camp_fire_graph: RoadGraph) -> None:
    """When no fire is present, k* should reach the search ceiling max_k."""
    origin = list(camp_fire_graph.nodes.keys())[0]
    destinations = set(camp_fire_graph.boundary_destinations.values())

    k_star = compute_critical_slowdown_k_star(
        road_graph=camp_fire_graph,
        origin_node=origin,
        destination_nodes=destinations,
        t0=0.0,
        buffer_minutes=15.0,
        edge_fire_arrivals=None,
        max_k=25.0,
    )
    assert k_star == 25.0


def test_k_star_monotonicity_with_departure_time(camp_fire_graph: RoadGraph) -> None:
    """Earlier departure allows greater slowdown tolerance than later departure."""
    assert camp_fire_graph.grid_x is not None and camp_fire_graph.grid_y is not None
    af = create_wind_ellipse_arrival_field(
        x_coords=camp_fire_graph.grid_x,
        y_coords=camp_fire_graph.grid_y,
        ignition_x=633391.0,
        ignition_y=4407338.0,
        head_rate_m_per_min=40.0,
        wind_towards_deg=225.0,
    )
    edge_arrivals = camp_fire_graph.compute_edge_fire_arrivals(af)
    destinations = set(camp_fire_graph.boundary_destinations.values())

    # Find a node in Paradise with finite fire arrival in domain
    origin = None
    for nid, pt in camp_fire_graph.nodes.items():
        if 618000.0 < pt[0] < 622000.0 and 4400000.0 < pt[1] < 4404000.0:
            origin = nid
            break
    assert origin is not None

    k_early = compute_critical_slowdown_k_star(
        road_graph=camp_fire_graph,
        origin_node=origin,
        destination_nodes=destinations,
        t0=10.0,
        buffer_minutes=15.0,
        edge_fire_arrivals=edge_arrivals,
        max_k=20.0,
    )
    k_late = compute_critical_slowdown_k_star(
        road_graph=camp_fire_graph,
        origin_node=origin,
        destination_nodes=destinations,
        t0=60.0,
        buffer_minutes=15.0,
        edge_fire_arrivals=edge_arrivals,
        max_k=20.0,
    )

    assert k_early >= k_late
    assert k_early > 0.0


def test_k_star_buffer_sensitivity(camp_fire_graph: RoadGraph) -> None:
    """A stricter safety buffer B reduces allowable slowdown k*."""
    assert camp_fire_graph.grid_x is not None and camp_fire_graph.grid_y is not None
    af = create_wind_ellipse_arrival_field(
        x_coords=camp_fire_graph.grid_x,
        y_coords=camp_fire_graph.grid_y,
        ignition_x=633391.0,
        ignition_y=4407338.0,
        head_rate_m_per_min=40.0,
        wind_towards_deg=225.0,
    )
    edge_arrivals = camp_fire_graph.compute_edge_fire_arrivals(af)
    destinations = set(camp_fire_graph.boundary_destinations.values())

    origin = None
    for nid, pt in camp_fire_graph.nodes.items():
        if 618000.0 < pt[0] < 622000.0 and 4400000.0 < pt[1] < 4404000.0:
            origin = nid
            break
    assert origin is not None

    k_small_b = compute_critical_slowdown_k_star(
        road_graph=camp_fire_graph,
        origin_node=origin,
        destination_nodes=destinations,
        t0=30.0,
        buffer_minutes=5.0,
        edge_fire_arrivals=edge_arrivals,
        max_k=20.0,
    )
    k_large_b = compute_critical_slowdown_k_star(
        road_graph=camp_fire_graph,
        origin_node=origin,
        destination_nodes=destinations,
        t0=30.0,
        buffer_minutes=45.0,
        edge_fire_arrivals=edge_arrivals,
        max_k=20.0,
    )

    assert k_small_b >= k_large_b


def test_k_star_grid(camp_fire_graph: RoadGraph) -> None:
    """Verify compute_k_star_grid generates structured tabular output."""
    destinations = set(camp_fire_graph.boundary_destinations.values())
    node_keys = list(camp_fire_graph.nodes.keys())
    origins = {"NodeA": node_keys[0], "NodeB": node_keys[1]}

    df = compute_k_star_grid(
        road_graph=camp_fire_graph,
        origins=origins,
        destination_nodes=destinations,
        departure_times_min=[0.0, 30.0],
        buffer_minutes=15.0,
        edge_fire_arrivals=None,
        max_k=10.0,
    )
    assert len(df) == 4
    assert set(df.columns) == {
        "origin_name",
        "origin_node",
        "departure_time_min",
        "buffer_minutes",
        "k_star",
    }
    assert all(df["k_star"] == 10.0)
