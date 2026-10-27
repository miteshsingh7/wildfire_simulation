"""Unit test verifying that replay agent simulation is sensitive to distinct arrival fields.

Specifically ensures that:
- When presented with two distinct arrival fields (an unburned field vs.
  a field with early fire cutting off evacuation corridors), the agent's routes
  and/or hazard misses differ.
- The test asserts that outcomes differ (assert sig1 != sig2) and would fail
  if the replay harness or agent were insensitive to the arrival field.
"""

import numpy as np
import pytest

from wf052.replay.contracts import TruthHazard
from wf052.replay.harness import CausalReplayHarness
from wf052.replay.methods import run_agent_simulation, sample_block_origins
from wf052.replay.providers import DynamicTeacherV2Provider, TeacherFieldProvider
from wf052.routing.graph import load_camp_fire_road_graph


def test_replay_field_sensitivity_fails_if_identical():
    """Verify that distinct fire arrival fields produce distinct routing/hazard outcomes."""
    # Obtain spatial coordinate grid from Camp Fire extent
    tf = TeacherFieldProvider().get_field(0.0)
    gx, gy = tf["x"].values, tf["y"].values

    road_graph = load_camp_fire_road_graph(
        full_detail=True,
        grid_x_coords=gx,
        grid_y_coords=gy,
    )
    destinations = set(road_graph.boundary_destinations.values())
    assert len(destinations) > 0, "Road graph should have boundary destinations"

    origins = sample_block_origins(road_graph, num_samples=5, seed=42)

    # Field 1: Completely unburned (all +inf)
    arr_unburned = np.full((len(gy), len(gx)), np.inf, dtype=np.float64)

    # Field 2: Fire encroaches early (at t=30 min) across central corridors
    arr_encroaching = np.full((len(gy), len(gx)), np.inf, dtype=np.float64)
    arr_encroaching[200:400, 100:300] = 30.0

    # Run agent simulation with Field 1 (unburned)
    prov1 = DynamicTeacherV2Provider(arr_unburned, gx, gy, assimilate=False)
    harness1 = CausalReplayHarness(road_graph=road_graph, provider=prov1, latency_min=0.0)
    results1, _ = run_agent_simulation("agent_field1", origins, road_graph, destinations, [], harness1)

    # Run agent simulation with Field 2 (encroaching)
    prov2 = DynamicTeacherV2Provider(arr_encroaching, gx, gy, assimilate=False)
    harness2 = CausalReplayHarness(road_graph=road_graph, provider=prov2, latency_min=0.0)
    results2, _ = run_agent_simulation("agent_field2", origins, road_graph, destinations, [], harness2)

    # Extract route signatures: (origin_id, edge_count, arrival_time)
    def get_route_signature(results):
        sig = []
        for r in results:
            if r.route is None or not r.route.edges:
                sig.append((r.origin_id, "trapped", r.hazard_miss))
            else:
                edge_keys = tuple((e.u, e.v) for e in r.route.edges)
                sig.append((r.origin_id, len(edge_keys), r.hazard_miss, round(r.route.arrival_time_min, 2)))
        return sig

    sig1 = get_route_signature(results1)
    sig2 = get_route_signature(results2)

    # The simulation outcomes MUST differ between the unburned field and the encroaching field
    # If the simulator or agent ignores the arrival field, sig1 == sig2 and the assertion fails!
    assert sig1 != sig2, (
        f"Replay simulation produced identical outcomes under two completely distinct arrival fields!\n"
        f"Field 1 outcome: {sig1}\n"
        f"Field 2 outcome: {sig2}"
    )
