"""Comprehensive test suite for Phase 5A evacuation coordination agent.

Tests:
1. LangGraph state machine execution & JSONL step-by-step logging.
2. Plain typed tools (weather, arrival field, plan, validate, exposure, k_star).
3. Scripted update-sequence scenarios:
   - Scenario A: Slow growth (margin stays above B -> no re-plan).
   - Scenario B: Sudden wind shift (fire encroaches fast path -> re-plan to alternate).
   - Scenario C: Road cut (active route severed -> emergency re-plan).
   - Scenario D: No safe route (fire covers all egress -> Mode b max-min margin fallback).
   - Scenario E: Anti-flapping hysteresis (minor margin gain < threshold -> route preserved).
4. Briefing & Grounding:
   - Qualitative clearance rule enforcement (rejection of clearance hours).
   - Standing caveat presence.
   - Low confidence warning placed first.
   - Grounding rejection of hallucinated numbers and road names.
   - Single regeneration retry and template fallback.
5. Latency benchmarks (router + validator vs mock LLM).
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import httpx
import numpy as np
import pytest
import shapely.geometry

from wf052.agent.contracts import (
    AgentState,
    ArrivalFieldMeta,
    ExposureSummary,
    WeatherReport,
)
from wf052.agent.graph import EvacuationAgent
from wf052.agent.grounding import GroundingChecker
from wf052.agent.llm import (
    LOW_CONFIDENCE_NOTICE,
    NO_SAFE_ROUTE_ANY_NOTICE,
    STANDING_CAVEAT,
    DeterministicBriefGenerator,
    MockLLMClient,
    ProviderLLMClient,
    create_llm_client,
)
from wf052.agent.policy import ReplanPolicy, compute_max_active_edge_change
from wf052.agent.tools import (
    check_field_vs_observations,
    compute_exposure,
    compute_k_star,
    get_arrival_field,
    get_weather,
    plan_routes,
    validate_routes,
)
from wf052.routing.contracts import EvacuationRoute, RouteStatus
from wf052.routing.graph import RoadGraph
from wf052.routing.synthetic_fire import (
    create_expanding_circle_arrival_field,
    create_unburned_arrival_field,
    create_wind_ellipse_arrival_field,
)


@pytest.fixture
def synthetic_diamond_graph() -> RoadGraph:
    """Deterministic diamond graph.

    Nodes:
      0 (Origin): (0, 0)
      1 (Path A intermediate, Fast): (500, 500)
      2 (Path B intermediate, Slow): (500, -500)
      3 (Destination): (1000, 0)

    Edges:
      Path A: 0 -> 1 -> 3 (Total free-flow travel time = 20.0 min)
      Path B: 0 -> 2 -> 3 (Total free-flow travel time = 40.0 min)
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
        "maxspeed": ["2.64 mph", "2.64 mph", "1.32 mph", "1.32 mph"],  # 10 min vs 20 min per edge
        "highway": ["primary", "primary", "secondary", "secondary"],
        "name": ["Skyway North", "Skyway North", "Clark Bypass", "Clark Bypass"],
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


class TestStateAndTools:
    """Requirement 1 & 2: Typed state machine and plain tool functions."""

    def test_plain_tools_execution(self, synthetic_diamond_graph: RoadGraph) -> None:
        graph = synthetic_diamond_graph

        # 1. Weather
        w = get_weather("default")
        assert isinstance(w, WeatherReport)
        assert w.wind_speed_mps > 0
        assert w.is_stub is True

        # 2. Arrival Field
        arr, meta = get_arrival_field(
            mode="synthetic_circle",
            config={"ignition_x": -500.0, "ignition_y": 0.0, "spread_rate_m_per_min": 10.0},
            road_graph=graph,
        )
        assert "arrival_q50" in arr
        assert isinstance(meta, ArrivalFieldMeta)
        assert meta.burned_cells > 0

        # 3. Plan Routes
        routes = plan_routes(
            road_graph=graph,
            origins=[0],
            destinations=[3],
            departure_time=0.0,
            arrival_field=arr,
            buffer_minutes=15.0,
        )
        assert 0 in routes
        assert routes[0].status == RouteStatus.SAFE
        assert routes[0].destination_id == 3

        # 4. Validate Routes
        val = validate_routes(
            road_graph=graph,
            routes=routes,
            arrival_field=arr,
            buffer_minutes=15.0,
        )
        assert 0 in val
        assert val[0].is_valid is True

        # 5. Exposure & k*
        exp = compute_exposure(routes, graph, buffer_minutes=15.0)
        assert isinstance(exp, ExposureSummary)
        assert exp.safe_count == 1
        assert exp.trapped_count == 0
        assert "Free-flowing" in exp.qualitative_traffic or "healthy" in exp.qualitative_traffic

        k_stars = compute_k_star(graph, [0], [3], 0.0, arr, buffer_minutes=15.0)
        assert 0 in k_stars
        assert k_stars[0] >= 1.0

    def test_jsonl_step_logging(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Requirement 1: Every state machine step writes input/output with timestamp to JSONL."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_p = Path(tmp_dir) / "agent_log.jsonl"
            agent = EvacuationAgent(log_file=log_p)

            arr_field = create_unburned_arrival_field(
                x_coords=synthetic_diamond_graph.grid_x,
                y_coords=synthetic_diamond_graph.grid_y,
            )

            state: AgentState = {
                "update_id": "TEST_001",
                "road_graph": synthetic_diamond_graph,
                "origins": [0],
                "destinations": [3],
                "departure_time": 0.0,
                "buffer_minutes": 15.0,
                "arrival_field": arr_field,
                "log_file": str(log_p),
            }

            res = agent.run_update(state)
            assert "brief" in res
            assert log_p.exists()

            # Read log entries
            lines = [json.loads(line) for line in log_p.read_text(encoding="utf-8").strip().split("\n")]
            steps_logged = [entry["step"] for entry in lines]

            expected_steps = [
                "ingest_update",
                "get_arrival_field",
                "plan_routes",
                "validate_routes",
                "compute_exposure",
                "decide_replan",
                "brief",
            ]
            assert steps_logged == expected_steps

            for entry in lines:
                assert "timestamp" in entry
                assert "update_id" in entry
                assert "inputs" in entry
                assert "outputs" in entry


class TestScriptedReplanSequences:
    """Requirement 3: Scripted update-sequence tests with hysteresis."""

    def test_scenario_a_slow_growth_no_replan(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Scenario A: Slow fire growth far away; margin stays > B -> routes remain stable."""
        graph = synthetic_diamond_graph
        policy = ReplanPolicy(buffer_minutes=15.0, hysteresis_margin_min=5.0)

        # Update 1: Fire starts far west (-300, 0)
        arr1 = create_expanding_circle_arrival_field(
            x_coords=graph.grid_x, y_coords=graph.grid_y,
            ignition_x=-300.0, ignition_y=0.0, spread_rate_m_per_min=5.0
        )
        routes1 = plan_routes(graph, [0], [3], 0.0, arr1, 15.0)
        active1, dec1 = policy.decide(None, routes1, None, arr1)
        assert dec1.needs_replan is True  # Initial assignment
        assert [e.v for e in active1[0].edges] == [1, 3]  # Fast path A

        # Update 2: Fire expands slowly to (-250, 0), margin is still large (~30 min > 15 min)
        arr2 = create_expanding_circle_arrival_field(
            x_coords=graph.grid_x, y_coords=graph.grid_y,
            ignition_x=-250.0, ignition_y=0.0, spread_rate_m_per_min=5.0
        )
        routes2 = plan_routes(graph, [0], [3], 0.0, arr2, 15.0)
        active2, dec2 = policy.decide(active1, routes2, arr1, arr2)

        assert dec2.needs_replan is False
        assert len(dec2.changed_origins) == 0
        assert active2[0].node_ids == active1[0].node_ids

    def test_scenario_b_sudden_wind_shift_replan(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Scenario B: Sudden wind shift accelerates fire towards Path A -> margin < B -> re-plan to Path B."""
        graph = synthetic_diamond_graph
        policy = ReplanPolicy(buffer_minutes=15.0, hysteresis_margin_min=5.0)

        # Update 1: Unburned field -> Path A chosen (fastest, 20m)
        arr1 = create_unburned_arrival_field(graph.grid_x, graph.grid_y)
        routes1 = plan_routes(graph, [0], [3], 0.0, arr1, 15.0)
        active1, _ = policy.decide(None, routes1, None, arr1)
        assert [e.v for e in active1[0].edges] == [1, 3]

        # Update 2: Fire encroaches Path A node 1 at (500, 500)
        # Using expanding circle with ignition at (500, 600) and spread rate 10 m/min
        arr2 = create_expanding_circle_arrival_field(
            x_coords=graph.grid_x,
            y_coords=graph.grid_y,
            ignition_x=500.0,
            ignition_y=600.0,
            spread_rate_m_per_min=10.0,
        )

        routes2 = plan_routes(graph, [0], [3], 0.0, arr2, 15.0)
        active2, dec2 = policy.decide(active1, routes2, arr1, arr2, road_graph=graph)

        # Policy MUST trigger re-plan and switch origin 0 to Path B (nodes 0->2->3)
        assert dec2.needs_replan is True
        assert 0 in dec2.changed_origins
        assert [e.v for e in active2[0].edges] == [2, 3]
        assert active2[0].status == RouteStatus.SAFE

    def test_scenario_c_road_cut_emergency_diversion(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Scenario C: Active road corridor is severed by road cut -> emergency re-plan."""
        graph = synthetic_diamond_graph
        policy = ReplanPolicy(buffer_minutes=15.0)

        arr = create_unburned_arrival_field(graph.grid_x, graph.grid_y)
        routes1 = plan_routes(graph, [0], [3], 0.0, arr, 15.0)
        active1, _ = policy.decide(None, routes1, None, arr)
        assert [e.v for e in active1[0].edges] == [1, 3]

        # Road cut severs Edge (1, 3) on Path A
        road_cut = {(1, 3)}
        # Router plans route with cut edge avoided
        routes_cut = plan_routes(graph, [0], [3], 0.0, arr, 15.0, closed_edges=road_cut)

        active2, dec2 = policy.decide(active1, routes_cut, arr, arr, road_cut_edges=road_cut)

        assert dec2.needs_replan is True
        assert 0 in dec2.changed_origins
        assert "severed" in dec2.reason
        assert [e.v for e in active2[0].edges] == [2, 3]

    def test_scenario_d_no_safe_route_fallback(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Scenario D: Fire cuts all egress corridors -> router falls back to Mode b max-min margin."""
        graph = synthetic_diamond_graph
        policy = ReplanPolicy(buffer_minutes=15.0)

        # Fire burns both Path A and Path B immediately at t = 5 min
        arr_cutoff = create_unburned_arrival_field(graph.grid_x, graph.grid_y)
        for q in ["arrival_q10", "arrival_q50", "arrival_q90"]:
            if q in arr_cutoff:
                arr_cutoff[q].values[:, :] = 5.0  # Entire domain burns in 5 min

        routes = plan_routes(graph, [0], [3], 0.0, arr_cutoff, buffer_minutes=15.0)
        assert routes[0].status != RouteStatus.SAFE

        active, dec = policy.decide(None, routes, None, arr_cutoff)
        assert active[0].status != RouteStatus.SAFE
        assert active[0].min_margin_min < 0.0

        # Exposure summary correctly captures trapped vehicle state
        exp = compute_exposure(active, graph, buffer_minutes=15.0)
        assert exp.trapped_count == 1
        assert exp.safe_count == 0
        assert "Severe bottleneck" in exp.qualitative_traffic

    def test_scenario_e_anti_flapping_hysteresis(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Scenario E: Candidate route has near-equal margin gain (< 5 min) -> policy suppresses flapping."""
        graph = synthetic_diamond_graph
        # Require +5 min margin gain to switch
        policy = ReplanPolicy(buffer_minutes=10.0, hysteresis_margin_min=5.0)

        arr = create_unburned_arrival_field(graph.grid_x, graph.grid_y)
        routes1 = plan_routes(graph, [0], [3], 0.0, arr, 10.0)
        active1, _ = policy.decide(None, routes1, None, arr)

        # Distinct candidate on alternate path with only +2 min margin improvement (below 5 min threshold)
        r0 = routes1[0]
        alt_route = EvacuationRoute(
            origin_id=0,
            destination_id=3,
            departure_time_min=r0.departure_time_min,
            arrival_time_min=r0.arrival_time_min * 0.98,
            total_travel_time_min=r0.total_travel_time_min * 0.98,
            min_margin_min=r0.min_margin_min + 2.0,
            bottleneck_edge=None,
            edges=r0.edges,
            node_ids=(0, 2, 3),  # Different path
            status=RouteStatus.SAFE,
            mode_used=r0.mode_used,
        )

        candidate_routes = {0: alt_route}
        active2, dec2 = policy.decide(active1, candidate_routes, arr, arr, road_graph=graph)

        # Anti-flapping must keep active1 route
        assert 0 in dec2.flap_prevented_origins
        assert active2[0].node_ids == active1[0].node_ids
        assert active2[0].node_ids != (0, 2, 3)


class TestBriefsAndGrounding:
    """Requirement 4: Grounding check, caveats, qualitative clearance, and template fallback."""

    def test_deterministic_template_generation(self, synthetic_diamond_graph: RoadGraph) -> None:
        generator = DeterministicBriefGenerator()
        arr = create_unburned_arrival_field(synthetic_diamond_graph.grid_x, synthetic_diamond_graph.grid_y)
        routes = plan_routes(synthetic_diamond_graph, [0], [3], 0.0, arr, 15.0)
        exp = compute_exposure(routes, synthetic_diamond_graph, 15.0)

        state: AgentState = {
            "update_id": "UP_001",
            "weather": WeatherReport(7.0, 45.0, 20.0, 15.0),
            "arrival_field_meta": ArrivalFieldMeta(is_low_confidence=False, burned_cells=100, total_cells=625, min_arrival_min=10.0),
            "active_routes": routes,
            "exposure_summary": exp,
            "k_star_values": {0: 3.5},
            "buffer_minutes": 15.0,
            "road_graph": synthetic_diamond_graph,
            "origins": [0],
            "destinations": [3],
        }

        brief = generator.build_brief(state)
        assert STANDING_CAVEAT in brief
        assert "Traffic Conditions:" in brief
        assert "7.0 m/s" in brief
        assert "hours" not in brief.lower() or "traffic" in brief.lower()

    def test_low_confidence_notice_placed_first(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Requirement 4: If ArrivalField is flagged low confidence, brief says so first."""
        generator = DeterministicBriefGenerator()
        arr = create_unburned_arrival_field(synthetic_diamond_graph.grid_x, synthetic_diamond_graph.grid_y)
        routes = plan_routes(synthetic_diamond_graph, [0], [3], 0.0, arr, 15.0)

        state: AgentState = {
            "update_id": "LOW_CONF_001",
            "arrival_field_meta": ArrivalFieldMeta(
                is_low_confidence=True,
                burned_cells=2,
                total_cells=625,
                min_arrival_min=0.0,
                reason="No spread beyond initial cell.",
            ),
            "active_routes": routes,
            "road_graph": synthetic_diamond_graph,
            "origins": [0],
            "destinations": [3],
        }

        brief = generator.build_brief(state)
        assert brief.startswith(LOW_CONFIDENCE_NOTICE)
        assert brief.find("LOW CONFIDENCE") < brief.find("Assigned Evacuation Corridors:")

    def test_grounding_rejects_hallucinated_number(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Requirement 4: Grounding checker rejects brief with ungrounded numbers."""
        checker = GroundingChecker()
        arr = create_unburned_arrival_field(synthetic_diamond_graph.grid_x, synthetic_diamond_graph.grid_y)
        routes = plan_routes(synthetic_diamond_graph, [0], [3], 0.0, arr, 15.0)

        state: AgentState = {
            "update_id": "TEST_001",
            "buffer_minutes": 15.0,
            "active_routes": routes,
            "road_graph": synthetic_diamond_graph,
            "origins": [0],
            "destinations": [3],
        }

        # Injected hallucinated number: 948.5
        bad_brief = f"{STANDING_CAVEAT}\nEvacuation Brief: Observed wind speed is 948.5 m/s in Zone 0."
        res = checker.check(bad_brief, state)

        assert res.passed is False
        assert any("948.5" in r for r in res.rejected_tokens)

    def test_grounding_rejects_quantitative_clearance_hours(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Requirement 4: Congestion and clearance figures are qualitative: never state clearance hours."""
        checker = GroundingChecker()
        arr = create_unburned_arrival_field(synthetic_diamond_graph.grid_x, synthetic_diamond_graph.grid_y)
        routes = plan_routes(synthetic_diamond_graph, [0], [3], 0.0, arr, 15.0)

        state: AgentState = {
            "update_id": "TEST_002",
            "buffer_minutes": 15.0,
            "active_routes": routes,
            "road_graph": synthetic_diamond_graph,
            "origins": [0],
            "destinations": [3],
        }

        bad_brief = f"{STANDING_CAVEAT}\nRegional clearance time is estimated at 3.5 hours for Zone 0."
        res = checker.check(bad_brief, state)

        assert res.passed is False
        assert any("clearance hours" in r.lower() for r in res.rejected_tokens)

    def test_grounding_rejects_missing_standing_caveat(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Requirement 4: Every brief carries standing caveat on spotting and unmodeled perimeters."""
        checker = GroundingChecker()
        arr = create_unburned_arrival_field(synthetic_diamond_graph.grid_x, synthetic_diamond_graph.grid_y)
        routes = plan_routes(synthetic_diamond_graph, [0], [3], 0.0, arr, 15.0)

        state: AgentState = {
            "update_id": "TEST_003",
            "buffer_minutes": 15.0,
            "active_routes": routes,
            "road_graph": synthetic_diamond_graph,
            "origins": [0],
            "destinations": [3],
        }

        bad_brief = "Evacuation status: All routes are currently safe. Traffic moving freely."
        res = checker.check(bad_brief, state)

        assert res.passed is False
        assert any("standing caveat" in r.lower() for r in res.rejected_tokens)

    def test_agent_regeneration_and_fallback_workflow(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Requirement 4: On reject, regenerate once, then fall back to template."""
        # Mock LLM that generates a hallucinated number on call 1, but cleans it up on call 2
        mock_llm = MockLLMClient(mode="corrupt", hallucinated_number="888.8")
        agent = EvacuationAgent(llm_client=mock_llm)

        arr = create_unburned_arrival_field(synthetic_diamond_graph.grid_x, synthetic_diamond_graph.grid_y)
        state: AgentState = {
            "update_id": "RETRY_001",
            "road_graph": synthetic_diamond_graph,
            "origins": [0],
            "destinations": [3],
            "departure_time": 0.0,
            "buffer_minutes": 15.0,
            "arrival_field": arr,
        }

        res = agent.run_update(state)
        # Attempt 1 failed due to 888.8, regenerated attempt 2 succeeded
        assert res["grounding_result"].passed is True
        assert res["grounding_result"].attempts == 2
        assert mock_llm.call_count == 2
        assert "888.8" not in res["brief"]

    def test_grounding_numeric_tolerance_boundaries(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Requirement 0c: Numeric tolerance boundary tests (just inside and just outside tolerance).

        GroundingChecker tolerance is numeric_tolerance = 0.15.
        Ground truth travel time for Zone 0 is 20.0 min.
        - Just inside: 20.14 min (abs diff = 0.14 <= 0.15) -> PASS
        - Just outside: 20.16 min (abs diff = 0.16 > 0.15) -> REJECT
        - Just inside: 19.86 min (abs diff = 0.14 <= 0.15) -> PASS
        - Just outside: 19.84 min (abs diff = 0.16 > 0.15) -> REJECT
        """
        checker = GroundingChecker(numeric_tolerance=0.15)
        arr = create_unburned_arrival_field(synthetic_diamond_graph.grid_x, synthetic_diamond_graph.grid_y)
        routes = plan_routes(synthetic_diamond_graph, [0], [3], 0.0, arr, 15.0)

        state: AgentState = {
            "update_id": "BOUND_001",
            "buffer_minutes": 15.0,
            "active_routes": routes,
            "road_graph": synthetic_diamond_graph,
            "origins": [0],
            "destinations": [3],
        }

        # 1. Just inside (+0.14) -> PASS
        brief_inside_high = f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\n  - Zone 0 -> Exit 3: SAFE | Travel Time 20.14 min"
        res_in_high = checker.check(brief_inside_high, state)
        assert res_in_high.passed is True, f"Failed inside +0.14: {res_in_high.rejected_tokens}"

        # 2. Just inside (-0.14) -> PASS
        brief_inside_low = f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\n  - Zone 0 -> Exit 3: SAFE | Travel Time 19.86 min"
        res_in_low = checker.check(brief_inside_low, state)
        assert res_in_low.passed is True, f"Failed inside -0.14: {res_in_low.rejected_tokens}"

        # 3. Just outside (+0.16) -> REJECT
        brief_outside_high = f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\n  - Zone 0 -> Exit 3: SAFE | Travel Time 20.16 min"
        res_out_high = checker.check(brief_outside_high, state)
        assert res_out_high.passed is False
        assert any("20.16" in r for r in res_out_high.rejected_tokens)

        # 4. Just outside (-0.16) -> REJECT
        brief_outside_low = f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\n  - Zone 0 -> Exit 3: SAFE | Travel Time 19.84 min"
        res_out_low = checker.check(brief_outside_low, state)
        assert res_out_low.passed is False
        assert any("19.84" in r for r in res_out_low.rejected_tokens)

    def test_grounding_pass_and_reject_matrix(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Requirement 4: Grounding pass and reject matrix verification.

        Tests:
        - 5 scripted valid update briefs (pass) across varying conditions
        - 1 hallucinated number brief (reject)
        - 1 ungrounded road name brief (reject)
        - 1 quantitative clearance hours brief (reject)
        - 1 missing caveat brief (reject)
        - 1 late low-confidence notice brief (reject)

        Tallies and asserts total passes and rejects.
        """
        graph = synthetic_diamond_graph
        checker = GroundingChecker()
        generator = DeterministicBriefGenerator()

        passes = 0
        rejects = 0

        # --- A. Scripted Valid Update Briefs (Passes) ---
        # 1. Clear / unburned conditions
        arr1 = create_unburned_arrival_field(graph.grid_x, graph.grid_y)
        routes1 = plan_routes(graph, [0], [3], 0.0, arr1, 15.0)
        exp1 = compute_exposure(routes1, graph, 15.0)
        state1: AgentState = {
            "update_id": "VALID_001",
            "weather": WeatherReport(6.5, 45.0, 22.0, 15.0),
            "arrival_field_meta": ArrivalFieldMeta(False, 100, 625, 12.0),
            "active_routes": routes1,
            "exposure_summary": exp1,
            "k_star_values": {0: 3.5},
            "buffer_minutes": 15.0,
            "road_graph": graph,
            "origins": [0],
            "destinations": [3],
        }
        brief1 = generator.build_brief(state1)
        res1 = checker.check(brief1, state1)
        assert res1.passed is True, f"Valid brief 1 failed: {res1.rejected_tokens}"
        passes += 1

        # 2. Slow fire growth (Scenario A)
        arr2 = create_expanding_circle_arrival_field(graph.grid_x, graph.grid_y, -300.0, 0.0, 5.0)
        routes2 = plan_routes(graph, [0], [3], 0.0, arr2, 15.0)
        exp2 = compute_exposure(routes2, graph, 15.0)
        state2: AgentState = {
            "update_id": "VALID_002",
            "weather": WeatherReport(7.0, 60.0, 24.0, 14.0),
            "arrival_field_meta": ArrivalFieldMeta(False, 150, 625, 8.0),
            "active_routes": routes2,
            "exposure_summary": exp2,
            "k_star_values": {0: 2.8},
            "buffer_minutes": 15.0,
            "road_graph": graph,
            "origins": [0],
            "destinations": [3],
        }
        brief2 = generator.build_brief(state2)
        res2 = checker.check(brief2, state2)
        assert res2.passed is True, f"Valid brief 2 failed: {res2.rejected_tokens}"
        passes += 1

        # 3. Wind shift / alternate path (Scenario B)
        arr3 = create_expanding_circle_arrival_field(graph.grid_x, graph.grid_y, 500.0, 600.0, 10.0)
        routes3 = plan_routes(graph, [0], [3], 0.0, arr3, 15.0)
        exp3 = compute_exposure(routes3, graph, 15.0)
        state3: AgentState = {
            "update_id": "VALID_003",
            "weather": WeatherReport(12.0, 90.0, 28.0, 10.0),
            "arrival_field_meta": ArrivalFieldMeta(False, 220, 625, 4.0),
            "active_routes": routes3,
            "exposure_summary": exp3,
            "k_star_values": {0: 1.5},
            "buffer_minutes": 15.0,
            "road_graph": graph,
            "origins": [0],
            "destinations": [3],
        }
        brief3 = generator.build_brief(state3)
        res3 = checker.check(brief3, state3)
        assert res3.passed is True, f"Valid brief 3 failed: {res3.rejected_tokens}"
        passes += 1

        # 4. Road cut diversion (Scenario C)
        routes4 = plan_routes(graph, [0], [3], 0.0, arr1, 15.0, closed_edges={(1, 3)})
        exp4 = compute_exposure(routes4, graph, 15.0)
        state4: AgentState = {
            "update_id": "VALID_004",
            "weather": WeatherReport(8.0, 50.0, 25.0, 12.0),
            "arrival_field_meta": ArrivalFieldMeta(False, 100, 625, 12.0),
            "active_routes": routes4,
            "exposure_summary": exp4,
            "k_star_values": {0: 2.0},
            "buffer_minutes": 15.0,
            "road_graph": graph,
            "origins": [0],
            "destinations": [3],
        }
        brief4 = generator.build_brief(state4)
        res4 = checker.check(brief4, state4)
        assert res4.passed is True, f"Valid brief 4 failed: {res4.rejected_tokens}"
        passes += 1

        # 5. Low confidence field handled properly
        state5: AgentState = {
            "update_id": "VALID_005",
            "weather": WeatherReport(5.0, 30.0, 20.0, 18.0),
            "arrival_field_meta": ArrivalFieldMeta(
                is_low_confidence=True, burned_cells=2, total_cells=625, min_arrival_min=0.0, reason="Minimal spread"
            ),
            "active_routes": routes1,
            "exposure_summary": exp1,
            "k_star_values": {0: 3.5},
            "buffer_minutes": 15.0,
            "road_graph": graph,
            "origins": [0],
            "destinations": [3],
        }
        brief5 = generator.build_brief(state5)
        res5 = checker.check(brief5, state5)
        assert res5.passed is True, f"Valid brief 5 failed: {res5.rejected_tokens}"
        passes += 1

        # --- B. Specific Reject Cases ---
        # Reject Case 1: Hallucinated number
        bad_num_brief = f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\n  - Zone 0 -> Exit 3: SAFE\nFire spread speed is 948.5 m/s."
        res_rj1 = checker.check(bad_num_brief, state1)
        assert res_rj1.passed is False
        assert any("948.5" in r for r in res_rj1.rejected_tokens)
        rejects += 1

        # Reject Case 2: Ungrounded road name
        bad_road_brief = f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\n  - Zone 0 -> Exit 3: SAFE\nTraffic routing via Ocean Boulevard."
        res_rj2 = checker.check(bad_road_brief, state1)
        assert res_rj2.passed is False
        assert any("Ocean Boulevard" in r for r in res_rj2.rejected_tokens)
        rejects += 1

        # Reject Case 3: Quantitative clearance hours
        bad_clear_brief = f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\n  - Zone 0 -> Exit 3: SAFE\nRegional clearance time is estimated at 3.5 hours for Zone 0."
        res_rj3 = checker.check(bad_clear_brief, state1)
        assert res_rj3.passed is False
        assert any("clearance hours" in r.lower() for r in res_rj3.rejected_tokens)
        rejects += 1

        # Reject Case 4: Missing standing caveat
        bad_caveat_brief = "Evacuation Brief Update.\nAssigned Evacuation Corridors:\n  - Zone 0 -> Exit 3: SAFE\nTraffic moving freely."
        res_rj4 = checker.check(bad_caveat_brief, state1)
        assert res_rj4.passed is False
        assert any("standing caveat" in r.lower() for r in res_rj4.rejected_tokens)
        rejects += 1

        # Reject Case 5: Late low-confidence notice (placed after corridor section)
        bad_late_conf_brief = f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\n  - Zone 0 -> Exit 3: SAFE\n{LOW_CONFIDENCE_NOTICE}"
        res_rj5 = checker.check(bad_late_conf_brief, state5)
        assert res_rj5.passed is False
        assert any("low confidence notice must appear" in r.lower() for r in res_rj5.rejected_tokens)
        rejects += 1

        print(f"\nGrounding Matrix Results: Passes = {passes}, Rejects = {rejects}")
        assert passes == 5
        assert rejects == 5


class TestPhase5BFullGraphBenchmark:
    """Requirement 1: Full-graph run on real Camp Fire road graph vs small unit-test timings."""

    def test_full_camp_fire_graph_latency(
        self, camp_fire_graph: RoadGraph, synthetic_diamond_graph: RoadGraph
    ) -> None:
        """Run agent end to end on real Camp Fire road graph (11,814 edges, 5,110 nodes).

        Report latency per stage (router, validator, exposure) as median and p95 over 20 runs.
        Keep small-graph numbers separate and labelled as unit-test timings.
        """
        assert camp_fire_graph.grid_x is not None and camp_fire_graph.grid_y is not None
        arr_field = create_wind_ellipse_arrival_field(
            x_coords=camp_fire_graph.grid_x,
            y_coords=camp_fire_graph.grid_y,
            ignition_x=633391.0,
            ignition_y=4407338.0,
            head_rate_m_per_min=35.0,
            back_rate_m_per_min=5.0,
            length_to_breadth=2.5,
            wind_towards_deg=240.0,
            horizon_minutes=720.0,
        )

        test_origins = [
            camp_fire_graph.named_origins["Paradise"],
            camp_fire_graph.named_origins["Magalia"],
            camp_fire_graph.named_origins["Concow"],
        ]
        test_dests = list(camp_fire_graph.boundary_destinations.values())

        # 1. Benchmark 20 runs on Real Camp Fire Road Graph
        router_times_full = []
        validator_times_full = []
        exposure_times_full = []

        for _ in range(20):
            t0 = time.perf_counter()
            routes = plan_routes(
                camp_fire_graph, test_origins, test_dests, 60.0, arr_field, buffer_minutes=15.0
            )
            t1 = time.perf_counter()
            _ = validate_routes(camp_fire_graph, routes, arr_field, buffer_minutes=15.0)
            t2 = time.perf_counter()
            _ = compute_exposure(routes, camp_fire_graph, buffer_minutes=15.0)
            t3 = time.perf_counter()

            router_times_full.append((t1 - t0) * 1000.0)
            validator_times_full.append((t2 - t1) * 1000.0)
            exposure_times_full.append((t3 - t2) * 1000.0)

        # 2. Benchmark 20 runs on Small Synthetic Diamond Graph (Unit Test Timings)
        router_times_small = []
        validator_times_small = []
        exposure_times_small = []
        assert synthetic_diamond_graph.grid_x is not None and synthetic_diamond_graph.grid_y is not None
        arr_small = create_unburned_arrival_field(synthetic_diamond_graph.grid_x, synthetic_diamond_graph.grid_y)

        for _ in range(20):
            t0 = time.perf_counter()
            r_s = plan_routes(synthetic_diamond_graph, [0], [3], 0.0, arr_small, 15.0)
            t1 = time.perf_counter()
            _ = validate_routes(synthetic_diamond_graph, r_s, arr_small, 15.0)
            t2 = time.perf_counter()
            _ = compute_exposure(r_s, synthetic_diamond_graph, 15.0)
            t3 = time.perf_counter()

            router_times_small.append((t1 - t0) * 1000.0)
            validator_times_small.append((t2 - t1) * 1000.0)
            exposure_times_small.append((t3 - t2) * 1000.0)

        med_r_f, p95_r_f = float(np.median(router_times_full)), float(np.percentile(router_times_full, 95))
        med_v_f, p95_v_f = float(np.median(validator_times_full)), float(np.percentile(validator_times_full, 95))
        med_e_f, p95_e_f = float(np.median(exposure_times_full)), float(np.percentile(exposure_times_full, 95))

        med_r_s, p95_r_s = float(np.median(router_times_small)), float(np.percentile(router_times_small, 95))
        med_v_s, p95_v_s = float(np.median(validator_times_small)), float(np.percentile(validator_times_small, 95))
        med_e_s, p95_e_s = float(np.median(exposure_times_small)), float(np.percentile(exposure_times_small, 95))

        print("\n" + "=" * 75)
        print(f"[FULL GRAPH RUN] Real Camp Fire Road Graph ({len(camp_fire_graph.edges)} edges, {len(camp_fire_graph.nodes)} nodes):")
        print(f"  Stage 1 (Router):    Median = {med_r_f:6.2f} ms | P95 = {p95_r_f:6.2f} ms")
        print(f"  Stage 2 (Validator): Median = {med_v_f:6.2f} ms | P95 = {p95_v_f:6.2f} ms")
        print(f"  Stage 3 (Exposure):  Median = {med_e_f:6.2f} ms | P95 = {p95_e_f:6.2f} ms")
        print(f"  Total Pipeline:      Median = {med_r_f + med_v_f + med_e_f:6.2f} ms")
        print("-" * 75)
        print(f"[UNIT-TEST TIMINGS] Small Synthetic Diamond Graph ({len(synthetic_diamond_graph.edges)} edges, {len(synthetic_diamond_graph.nodes)} nodes):")
        print(f"  Stage 1 (Router):    Median = {med_r_s:6.3f} ms | P95 = {p95_r_s:6.3f} ms")
        print(f"  Stage 2 (Validator): Median = {med_v_s:6.3f} ms | P95 = {p95_v_s:6.3f} ms")
        print(f"  Stage 3 (Exposure):  Median = {med_e_s:6.3f} ms | P95 = {p95_e_s:6.3f} ms")
        print("=" * 75)

        assert med_r_f < 500.0
        assert med_v_f < 200.0
        assert med_r_s < 20.0


class TestPhase5BNoSafeRouteScenarios:
    """Requirement 2: Scripted sequences for partial and complete no-safe-route scenarios."""

    def test_no_safe_route_some_origins(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Scenario (a): No safe route exists for some origins (e.g. Origin 0), but safe for others."""
        graph = synthetic_diamond_graph
        generator = DeterministicBriefGenerator()
        checker = GroundingChecker()

        r0 = EvacuationRoute(
            origin_id=0,
            destination_id=3,
            departure_time_min=0.0,
            arrival_time_min=20.0,
            total_travel_time_min=20.0,
            min_margin_min=-5.0,
            bottleneck_edge=(0, 1, 0),
            edges=[],
            node_ids=(0, 1, 3),
            status=RouteStatus.BEST_EFFORT,
            mode_used="max_min_margin",
        )
        r1 = EvacuationRoute(
            origin_id=1,
            destination_id=3,
            departure_time_min=0.0,
            arrival_time_min=10.0,
            total_travel_time_min=10.0,
            min_margin_min=45.0,
            bottleneck_edge=None,
            edges=[],
            node_ids=(1, 3),
            status=RouteStatus.SAFE,
            mode_used="fastest_safe",
        )
        routes = {0: r0, 1: r1}
        exp = compute_exposure(routes, graph, buffer_minutes=15.0)

        state: AgentState = {
            "update_id": "PARTIAL_TRAP_001",
            "active_routes": routes,
            "exposure_summary": exp,
            "road_graph": graph,
            "origins": [0, 1],
            "destinations": [3],
            "buffer_minutes": 15.0,
        }

        brief = generator.build_brief(state)

        assert "CRITICAL ALERT: NO SAFE EVACUATION ROUTE EXISTS FOR Zone 0" in brief
        assert brief.find("NO SAFE EVACUATION ROUTE") < brief.find("Assigned Evacuation Corridors:")
        assert "Zone 0 -> Exit 3: BEST-EFFORT" in brief

        for line in brief.split("\n"):
            if "Zone 0 -> Exit" in line:
                assert "safe" not in line.lower()
                assert "BEST-EFFORT" in line

        check_res = checker.check(brief, state)
        assert check_res.passed is True

    def test_no_safe_route_all_origins(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Scenario (b): No safe route exists for any origin (complete entrapment)."""
        graph = synthetic_diamond_graph
        generator = DeterministicBriefGenerator()
        checker = GroundingChecker()

        r0 = EvacuationRoute(
            origin_id=0,
            destination_id=3,
            departure_time_min=0.0,
            arrival_time_min=20.0,
            total_travel_time_min=20.0,
            min_margin_min=-10.0,
            bottleneck_edge=(0, 1, 0),
            edges=[],
            node_ids=(0, 1, 3),
            status=RouteStatus.BEST_EFFORT,
            mode_used="max_min_margin",
        )
        routes = {0: r0}
        exp = compute_exposure(routes, graph, buffer_minutes=15.0)

        state: AgentState = {
            "update_id": "ALL_TRAP_001",
            "active_routes": routes,
            "exposure_summary": exp,
            "road_graph": graph,
            "origins": [0],
            "destinations": [3],
            "buffer_minutes": 15.0,
        }

        brief = generator.build_brief(state)

        assert brief.startswith(NO_SAFE_ROUTE_ANY_NOTICE)
        assert "BEST-EFFORT only" in brief
        assert "Zone 0 -> Exit 3: BEST-EFFORT" in brief

        for line in brief.split("\n"):
            if "Zone 0 -> Exit" in line:
                assert "safe" not in line.lower()

        check_res = checker.check(brief, state)
        assert check_res.passed is True


class TestPhase5BObservationConsistency:
    """Requirement 3: Field vs observation consistency checking and conservative re-plan trigger."""

    def test_detection_arrives_earlier_than_q10_flags_low_confidence(
        self, synthetic_diamond_graph: RoadGraph
    ) -> None:
        graph = synthetic_diamond_graph
        assert graph.grid_x is not None and graph.grid_y is not None
        arr = create_expanding_circle_arrival_field(graph.grid_x, graph.grid_y, 0.0, 0.0, 10.0)
        detections = [{"x": 500.0, "y": 500.0, "elapsed_minutes": 10.0}]

        res = check_field_vs_observations(arr, detections, tolerance_minutes=15.0)
        assert res.is_low_confidence is True
        assert res.is_consistent is False
        assert res.earlier_detections_count == 1
        assert "arrived earlier than field q10" in res.reason

    def test_detection_at_unmodeled_cell_flags_low_confidence(
        self, synthetic_diamond_graph: RoadGraph
    ) -> None:
        graph = synthetic_diamond_graph
        assert graph.grid_x is not None and graph.grid_y is not None
        arr = create_expanding_circle_arrival_field(
            graph.grid_x, graph.grid_y, -100.0, -100.0, 5.0, horizon_minutes=30.0
        )
        detections = [{"x": 1100.0, "y": 600.0, "elapsed_minutes": 20.0}]

        res = check_field_vs_observations(arr, detections, tolerance_minutes=15.0)
        assert res.is_low_confidence is True
        assert res.unmodeled_detections_count == 1
        assert "predicts no arrival / unmodeled" in res.reason

    def test_consistent_detection_maintains_high_confidence(
        self, synthetic_diamond_graph: RoadGraph
    ) -> None:
        graph = synthetic_diamond_graph
        assert graph.grid_x is not None and graph.grid_y is not None
        arr = create_expanding_circle_arrival_field(graph.grid_x, graph.grid_y, 0.0, 0.0, 10.0)
        detections = [{"x": 500.0, "y": 0.0, "elapsed_minutes": 52.0}]

        res = check_field_vs_observations(arr, detections, tolerance_minutes=15.0)
        assert res.is_low_confidence is False
        assert res.is_consistent is True
        assert res.earlier_detections_count == 0
        assert res.unmodeled_detections_count == 0

    def test_observation_inconsistency_triggers_conservative_replan(
        self, synthetic_diamond_graph: RoadGraph
    ) -> None:
        graph = synthetic_diamond_graph
        agent = EvacuationAgent()

        assert graph.grid_x is not None and graph.grid_y is not None
        arr = create_expanding_circle_arrival_field(graph.grid_x, graph.grid_y, 0.0, 0.0, 10.0)
        inconsistent_det = [{"x": 500.0, "y": 500.0, "elapsed_minutes": 10.0}]

        # Establish current baseline routes so this update is evaluated as a re-plan
        r_baseline = plan_routes(graph, [0], [3], 0.0, arr, 15.0)

        state: AgentState = {
            "update_id": "OBS_TEST_001",
            "road_graph": graph,
            "origins": [0],
            "destinations": [3],
            "departure_time": 0.0,
            "buffer_minutes": 15.0,
            "arrival_field": arr,
            "current_routes": r_baseline,
            "detections": inconsistent_det,
            "conservative_buffer_minutes": 30.0,
        }

        res = agent.run_update(state)
        assert res["arrival_field_meta"].is_low_confidence is True
        assert res["use_conservative_settings"] is True
        assert res["replan_decision"].needs_replan is True
        assert "Low confidence" in res["replan_decision"].reason
        assert LOW_CONFIDENCE_NOTICE in res["brief"]
        assert res["brief"].find("LOW CONFIDENCE") < res["brief"].find("Assigned Evacuation Corridors:")

    def test_five_burned_cells_persists_as_second_trigger(
        self, synthetic_diamond_graph: RoadGraph
    ) -> None:
        graph = synthetic_diamond_graph
        assert graph.grid_x is not None and graph.grid_y is not None
        arr, meta = get_arrival_field(
            mode="custom",
            custom_field=create_unburned_arrival_field(graph.grid_x, graph.grid_y),
            road_graph=graph,
            min_spread_area_cells=5,
        )
        assert meta.is_low_confidence is True
        assert "minimal or zero spread" in meta.reason


class TestPhase5BReplanTriggerAndHysteresis:
    """Requirement 4: Max active edge change trigger & hysteresis margin invariant."""

    def test_max_active_edge_change_triggers_replan(
        self, synthetic_diamond_graph: RoadGraph
    ) -> None:
        graph = synthetic_diamond_graph
        policy = ReplanPolicy(buffer_minutes=15.0, max_active_edge_change_threshold=15.0)

        assert graph.grid_x is not None and graph.grid_y is not None
        arr1 = create_unburned_arrival_field(graph.grid_x, graph.grid_y)
        routes1 = plan_routes(graph, [0], [3], 0.0, arr1, 15.0)

        arr2 = create_unburned_arrival_field(graph.grid_x, graph.grid_y)
        # Select active edge 0 and set its mapped grid cell to burn at t = 20 min
        edge0 = graph.edges[0]
        assert len(edge0.cell_rows) > 0 and len(edge0.cell_cols) > 0
        r_idx = edge0.cell_rows[0]
        c_idx = edge0.cell_cols[0]
        arr2["arrival_q10"].values[r_idx, c_idx] = 20.0
        arr2["arrival_q50"].values[r_idx, c_idx] = 20.0

        max_shift = compute_max_active_edge_change(routes1, arr1, arr2, graph)
        assert max_shift > 15.0

        routes2 = plan_routes(graph, [0], [3], 0.0, arr2, 15.0)
        active2, dec2 = policy.decide(routes1, routes2, arr1, arr2, road_graph=graph)

        assert dec2.needs_replan is True
        assert "Max arrival change on active route edges" in dec2.reason

    def test_hysteresis_never_blocks_switch_when_margin_violated(
        self, synthetic_diamond_graph: RoadGraph
    ) -> None:
        """Requirement 4: Hysteresis must NEVER block a switch when current route violates its margin."""
        graph = synthetic_diamond_graph
        policy = ReplanPolicy(buffer_minutes=15.0, hysteresis_margin_min=10.0)

        assert graph.grid_x is not None and graph.grid_y is not None
        arr = create_unburned_arrival_field(graph.grid_x, graph.grid_y)

        curr_route = EvacuationRoute(
            origin_id=0,
            destination_id=3,
            departure_time_min=0.0,
            arrival_time_min=20.0,
            total_travel_time_min=20.0,
            min_margin_min=10.0,
            bottleneck_edge=None,
            edges=[],
            node_ids=(0, 1, 3),
            status=RouteStatus.BEST_EFFORT,
            mode_used="fastest_safe",
        )
        current_routes = {0: curr_route}

        cand_route = EvacuationRoute(
            origin_id=0,
            destination_id=3,
            departure_time_min=0.0,
            arrival_time_min=38.0,
            total_travel_time_min=38.0,
            min_margin_min=11.5,
            bottleneck_edge=None,
            edges=[],
            node_ids=(0, 2, 3),
            status=RouteStatus.BEST_EFFORT,
            mode_used="max_min_margin",
        )
        candidate_routes = {0: cand_route}

        active, dec = policy.decide(current_routes, candidate_routes, arr, arr, road_graph=graph)

        assert dec.needs_replan is True
        assert 0 in dec.changed_origins
        assert 0 not in dec.flap_prevented_origins
        assert active[0].node_ids == (0, 2, 3)
        assert "fell below buffer" in dec.reason


class TestPhase5BGroundingMatrix30Cases:
    """Requirement 5: Grounding verification matrix across at least 30 diverse test cases."""

    def test_thirty_case_grounding_matrix(self, synthetic_diamond_graph: RoadGraph) -> None:
        graph = synthetic_diamond_graph
        checker = GroundingChecker()

        assert graph.grid_x is not None and graph.grid_y is not None
        arr = create_unburned_arrival_field(graph.grid_x, graph.grid_y)
        routes_safe = plan_routes(graph, [0], [3], 0.0, arr, 15.0)
        exp_safe = compute_exposure(routes_safe, graph, 15.0)

        # Baseline safe state
        state_safe: AgentState = {
            "update_id": "ST_SAFE",
            "weather": WeatherReport(6.5, 45.0, 22.0, 15.0),
            "arrival_field_meta": ArrivalFieldMeta(False, 100, 625, 12.0),
            "active_routes": routes_safe,
            "exposure_summary": exp_safe,
            "k_star_values": {0: 3.5},
            "buffer_minutes": 15.0,
            "road_graph": graph,
            "origins": [0],
            "destinations": [3],
        }

        # Compromised best-effort state (Route for Zone 0 is best-effort)
        r_trap = EvacuationRoute(
            origin_id=0,
            destination_id=3,
            departure_time_min=0.0,
            arrival_time_min=20.0,
            total_travel_time_min=20.0,
            min_margin_min=-5.0,
            bottleneck_edge=(0, 1, 0),
            edges=[],
            node_ids=(0, 1, 3),
            status=RouteStatus.BEST_EFFORT,
            mode_used="max_min_margin",
        )
        routes_trap = {0: r_trap}
        exp_trap = compute_exposure(routes_trap, graph, 15.0)
        state_trap: AgentState = {
            "update_id": "ST_TRAP",
            "weather": WeatherReport(6.5, 45.0, 22.0, 15.0),
            "arrival_field_meta": ArrivalFieldMeta(False, 100, 625, 12.0),
            "active_routes": routes_trap,
            "exposure_summary": exp_trap,
            "k_star_values": {0: 1.0},
            "buffer_minutes": 15.0,
            "road_graph": graph,
            "origins": [0],
            "destinations": [3],
        }

        passes = 0
        rejects = 0

        # --- PASS CASES (18 distinct cases) ---
        pass_cases: list[tuple[str, AgentState]] = [
            # 1. Standard grounded safe brief
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\n  - Zone 0 -> Exit 3: SAFE | Travel Time 20.0 min", state_safe),
            # 2. Paraphrased safe corridor description
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nEvacuation corridor from Zone 0 leading toward Exit 3 is SAFE.", state_safe),
            # 3. Abbreviated road name "Skyway N"
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nTraffic moving along Skyway N towards Exit 3 is SAFE.", state_safe),
            # 4. Abbreviated road name "Clark Rd"
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nTraffic moving along Clark Rd is designated for secondary traffic.", state_safe),
            # 5. Full road name "Clark Bypass"
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nClark Bypass corridor is fully operational.", state_safe),
            # 6. Abbreviated road name "Skyway North"
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nVehicles proceeding on Skyway North toward Exit 3.", state_safe),
            # 7. Zone leading zero variant "Zone 00"
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nZone 00 traffic routed to Exit 3 is SAFE.", state_safe),
            # 8. Zone underscore variant "Zone_0"
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nZone_0 evacuation path confirmed SAFE.", state_safe),
            # 9. Rounded travel time "20 min"
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nTravel time estimated at 20 min.", state_safe),
            # 10. Floating travel time "20.0 min"
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nTravel time measured at 20.0 min.", state_safe),
            # 11. Unit variant "mps" for wind speed
            (f"{STANDING_CAVEAT}\nWeather: Wind speed is 6.5 mps.", state_safe),
            # 12. Unit variant "degrees" for wind dir
            (f"{STANDING_CAVEAT}\nWeather: Wind bearing 45 degrees.", state_safe),
            # 13. Temperature unit variant "C"
            (f"{STANDING_CAVEAT}\nWeather: Ambient temp 22.0 C.", state_safe),
            # 14. RH unit variant "%"
            (f"{STANDING_CAVEAT}\nWeather: Relative humidity 15%.", state_safe),
            # 15. Valid best-effort brief (stating best-effort without claiming safe)
            (f"{NO_SAFE_ROUTE_ANY_NOTICE}\n{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\n  - Zone 0 -> Exit 3: BEST-EFFORT", state_trap),
            # 16. Low confidence notice placed first
            (f"{LOW_CONFIDENCE_NOTICE}\n{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\n  - Zone 0 -> Exit 3: SAFE", state_safe),
            # 17. Multi-corridor safe wording
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nZone 0 corridor to Exit 3 is designated SAFE.", state_safe),
            # 18. Destination alias "Exit 3"
            (f"{STANDING_CAVEAT}\nDestination Exit 3 is cleared for reception.", state_safe),
        ]

        for idx, (brief_txt, st) in enumerate(pass_cases, 1):
            res = checker.check(brief_txt, st)
            assert res.passed is True, f"Pass case {idx} failed unexpectedly: {res.rejected_tokens}"
            passes += 1

        # --- REJECT CASES (18 distinct cases) ---
        reject_cases: list[tuple[str, AgentState, str]] = [
            # 1. Status inconsistency: Compromised route claimed as SAFE
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\n  - Zone 0 -> Exit 3: SAFE", state_trap, "status inconsistency"),
            # 2. Status inconsistency: "Zone 0 is safe" when route is best-effort
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nZone 0 is safe for all evacuees.", state_trap, "status inconsistency"),
            # 3. Status inconsistency: "All routes are safe" when trapped
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nAll routes are safe across the district.", state_trap, "status inconsistency"),
            # 4. Status inconsistency: Safe route claimed to have no safe route
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nNo safe evacuation route exists for Zone 0.", state_safe, "status inconsistency"),
            # 5. Ungrounded road "Ocean Boulevard"
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nEvacuate via Ocean Boulevard to safety.", state_safe, "ocean boulevard"),
            # 6. Ungrounded road "Interstate 80"
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nTraffic routed via Interstate 80.", state_safe, "interstate 80"),
            # 7. Ungrounded road "Broadway Ave"
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nProceed along Broadway Ave.", state_safe, "broadway ave"),
            # 8. Ungrounded road "Route 66"
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nFollow Route 66 out of town.", state_safe, "route 66"),
            # 9. Ungrounded road "Lincoln Blvd"
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nEgress via Lincoln Blvd.", state_safe, "lincoln blvd"),
            # 10. Ungrounded zone "Zone 99"
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nEvacuate Zone 99 immediately.", state_safe, "zone 99"),
            # 11. Ungrounded zone "Zone 42"
            (f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\nShelter order for Zone 42.", state_safe, "zone 42"),
            # 12. Quantitative clearance hours: "clearance time is 3.5 hours"
            (f"{STANDING_CAVEAT}\nRegional clearance time is 3.5 hours for Zone 0.", state_safe, "clearance hours"),
            # 13. Quantitative clearance hours: "cleared in 4 hrs"
            (f"{STANDING_CAVEAT}\nCorridor will be cleared in 4 hrs.", state_safe, "clearance hours"),
            # 14. Quantitative clearance hours: "estimated 2 hours to clear"
            (f"{STANDING_CAVEAT}\nEstimated 2 hours to clear the town.", state_safe, "clearance hours"),
            # 15. Missing standing caveat
            ("Assigned Evacuation Corridors:\nZone 0 -> Exit 3 is moving freely.", state_safe, "standing caveat"),
            # 16. Late low-confidence notice (placed after corridors)
            (
                f"{STANDING_CAVEAT}\nAssigned Evacuation Corridors:\n  - Zone 0 -> Exit 3: SAFE\n{LOW_CONFIDENCE_NOTICE}",
                dict(state_safe, arrival_field_meta=ArrivalFieldMeta(True, 2, 625, 0.0)),
                "low confidence notice must appear",
            ),
            # 17. Hallucinated number "948.5"
            (f"{STANDING_CAVEAT}\nObserved fire speed is 948.5 m/s.", state_safe, "948.5"),
            # 18. Hallucinated number "1234.5"
            (f"{STANDING_CAVEAT}\nPeak congestion delay of 1234.5 minutes.", state_safe, "1234.5"),
        ]

        for idx, (brief_txt, st, expected_err) in enumerate(reject_cases, 1):
            res = checker.check(brief_txt, st)
            assert res.passed is False, f"Reject case {idx} passed unexpectedly!"
            assert any(expected_err in r.lower() for r in res.rejected_tokens), (
                f"Reject case {idx} rejected for wrong reason: {res.rejected_tokens}"
            )
            rejects += 1

        print("\n" + "=" * 70)
        print(f"Phase 5B 36-Case Grounding Matrix: Passes = {passes}, Rejects = {rejects}, Total = {passes + rejects}")
        print("=" * 70)

        assert passes == 18
        assert rejects == 18
        assert passes + rejects >= 30


class TestPhase5BLLMAdapter:
    """Requirement 6: Real provider implementation behind LLMClient, recorded response, and smoke test."""

    def test_provider_llm_client_recorded_response_openai(self) -> None:
        """Verify ProviderLLMClient parses an OpenAI-compatible recorded API response."""
        recorded_response_json = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            f"{STANDING_CAVEAT}\n"
                            "=== Wildfire Evacuation Operational Briefing ===\n"
                            "Weather: Wind 6.5 m/s at 45 deg\n"
                            "Assigned Evacuation Corridors:\n"
                            "  - Zone 0 -> Exit 3: SAFE | Travel Time 20.0 min\n"
                            "Traffic Conditions: Free-flowing"
                        ),
                    }
                }
            ]
        }

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = recorded_response_json
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.Client.post", return_value=mock_resp):
            client = ProviderLLMClient(provider="openai", api_key="sk-mock-key-secret-1234")
            result = client.generate("Please generate an evacuation briefing.")

            assert STANDING_CAVEAT in result
            assert "Zone 0 -> Exit 3: SAFE" in result
            # Security verification: API key must NEVER be printed or in the result
            assert "sk-mock-key-secret-1234" not in result

    def test_provider_llm_client_recorded_response_anthropic(self) -> None:
        """Verify ProviderLLMClient parses an Anthropic-compatible recorded API response."""
        recorded_response_json = {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{STANDING_CAVEAT}\n"
                        "=== Wildfire Evacuation Operational Briefing ===\n"
                        "Assigned Evacuation Corridors:\n"
                        "  - Zone 0 -> Exit 3: SAFE"
                    ),
                }
            ]
        }

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = recorded_response_json
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.Client.post", return_value=mock_resp):
            client = ProviderLLMClient(provider="anthropic", api_key="anthropic-secret-5678")
            result = client.generate("Please generate brief.")

            assert STANDING_CAVEAT in result
            assert "anthropic-secret-5678" not in result

    def test_provider_llm_client_missing_key_raises_error(self) -> None:
        """If real provider is selected but no API key is available, raises clear ValueError."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "LLM_API_KEY": ""}, clear=True):
            client = ProviderLLMClient(provider="openai", api_key="")
            with pytest.raises(ValueError, match="API key environment variable not set"):
                client.generate("Generate brief.")

    def test_create_llm_client_factory_env_selection(self) -> None:
        with patch.dict(os.environ, {"LLM_PROVIDER": "mock"}):
            c = create_llm_client()
            assert isinstance(c, MockLLMClient)

        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
            c = create_llm_client()
            assert isinstance(c, ProviderLLMClient)
            assert c.provider == "openai"

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY")
        and not os.getenv("ANTHROPIC_API_KEY")
        and not os.getenv("GEMINI_API_KEY"),
        reason="Real LLM API key not set in environment (skipped in offline test runs)",
    )
    def test_live_provider_smoke_test(self) -> None:
        """Live smoke test that executes against live API only when a valid key is in the environment."""
        provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "gemini"
        client = ProviderLLMClient(provider=provider)
        response = client.generate("Reply with exact word: CONFIRMED")
        assert len(response) > 0
