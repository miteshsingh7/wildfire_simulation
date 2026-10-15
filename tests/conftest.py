"""Pytest shared fixtures for wf052 routing and evaluation tests."""

import pytest

from wf052.routing.graph import RoadGraph, load_camp_fire_road_graph


@pytest.fixture(scope="session")
def camp_fire_graph() -> RoadGraph:
    """Load the full-detail Camp Fire road graph once per test session."""
    return load_camp_fire_road_graph(full_detail=True)
