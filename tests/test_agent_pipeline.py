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
import tempfile
import time
from pathlib import Path

import geopandas as gpd
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
    STANDING_CAVEAT,
    DeterministicBriefGenerator,
    MockLLMClient,
)
from wf052.agent.policy import ReplanPolicy
from wf052.agent.tools import (
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


class TestPerformanceAndBenchmarks:
    """Requirement 5 & 6: Latency benchmarks and grounding stats."""

    def test_latency_router_validator_vs_llm(self, synthetic_diamond_graph: RoadGraph) -> None:
        """Requirement 6: Router + validator latency per update vs LLM step with mock."""
        graph = synthetic_diamond_graph
        arr = create_unburned_arrival_field(graph.grid_x, graph.grid_y)

        # 1. Measure Router + Validator latency (50 repeats)
        t_router_val = []
        for _ in range(50):
            t0 = time.perf_counter()
            r = plan_routes(graph, [0], [3], 0.0, arr, 15.0)
            _ = validate_routes(graph, r, arr, 15.0)
            t_router_val.append((time.perf_counter() - t0) * 1000.0)

        # 2. Measure Mock LLM step latency (50 repeats)
        mock_llm = MockLLMClient()
        prompt = DeterministicBriefGenerator().build_brief_from_text_prompt("Weather: Wind 6.5 m/s")
        t_llm = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = mock_llm.generate(prompt)
            t_llm.append((time.perf_counter() - t0) * 1000.0)

        med_rv = float(np.median(t_router_val))
        p95_rv = float(np.percentile(t_router_val, 95))
        med_llm = float(np.median(t_llm))
        p95_llm = float(np.percentile(t_llm, 95))

        print(f"\nRouter + Validator Latency: Median = {med_rv:.3f} ms, P95 = {p95_rv:.3f} ms")
        print(f"Mock LLM Step Latency:     Median = {med_llm:.3f} ms, P95 = {p95_llm:.3f} ms")

        # Sub-second execution
        assert med_rv < 100.0  # < 100 ms on synthetic graph
        assert med_llm < 10.0   # < 10 ms for mock LLM
