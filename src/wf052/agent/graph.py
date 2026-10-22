"""LangGraph state machine for wildfire evacuation coordination."""

import datetime
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from wf052.agent.contracts import (
    AgentState,
    GroundingResult,
    log_step_event,
)
from wf052.agent.grounding import GroundingChecker
from wf052.agent.llm import (
    DeterministicBriefGenerator,
    LLMClient,
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


class EvacuationAgent:
    """LangGraph-orchestrated evacuation coordination agent."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        replan_policy: ReplanPolicy | None = None,
        grounding_checker: GroundingChecker | None = None,
        log_file: str | Path | None = "output/agent_run_log.jsonl",
    ) -> None:
        self.llm_client = llm_client or MockLLMClient()
        self.policy = replan_policy or ReplanPolicy()
        self.grounding_checker = grounding_checker or GroundingChecker()
        self.brief_generator = DeterministicBriefGenerator()
        self.log_file = log_file

        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(AgentState)

        builder.add_node("ingest_update", self._node_ingest_update)
        builder.add_node("get_arrival_field", self._node_get_arrival_field)
        builder.add_node("plan_routes", self._node_plan_routes)
        builder.add_node("validate_routes", self._node_validate_routes)
        builder.add_node("compute_exposure", self._node_compute_exposure)
        builder.add_node("decide_replan", self._node_decide_replan)
        builder.add_node("brief", self._node_brief)

        builder.add_edge(START, "ingest_update")
        builder.add_edge("ingest_update", "get_arrival_field")
        builder.add_edge("get_arrival_field", "plan_routes")
        builder.add_edge("plan_routes", "validate_routes")
        builder.add_edge("validate_routes", "compute_exposure")
        builder.add_edge("compute_exposure", "decide_replan")
        builder.add_edge("decide_replan", "brief")
        builder.add_edge("brief", END)

        return builder.compile()

    # --- Node Implementations ---

    def _node_ingest_update(self, state: AgentState) -> dict[str, Any]:
        update_id = state.get("update_id", "UPDATE_001")
        timestamp = state.get("timestamp") or datetime.datetime.now(datetime.UTC).isoformat()
        weather = get_weather(location=state.get("weather_location", "default"))

        outputs = {"weather": weather, "timestamp": timestamp}
        log_step_event(
            log_file=state.get("log_file", self.log_file),
            step_name="ingest_update",
            update_id=update_id,
            inputs={"update_id": update_id, "timestamp": timestamp},
            outputs=outputs,
        )
        return outputs

    def _node_get_arrival_field(self, state: AgentState) -> dict[str, Any]:
        update_id = state.get("update_id", "UPDATE_001")
        custom_field = state.get("arrival_field")

        arr_field, meta = get_arrival_field(
            mode=state.get("arrival_field_mode", "synthetic"),
            config=state.get("arrival_field_config"),
            road_graph=state.get("road_graph"),
            custom_field=custom_field,
        )

        outputs = {
            "arrival_field": arr_field,
            "arrival_field_meta": meta,
        }
        log_step_event(
            log_file=state.get("log_file", self.log_file),
            step_name="get_arrival_field",
            update_id=update_id,
            inputs={"mode": state.get("arrival_field_mode", "synthetic")},
            outputs={"meta": meta.to_dict()},
        )
        return outputs

    def _node_plan_routes(self, state: AgentState) -> dict[str, Any]:
        update_id = state.get("update_id", "UPDATE_001")
        road_graph = state["road_graph"]
        origins = state.get("origins", [])
        destinations = state.get("destinations", [])
        departure_time = state.get("departure_time", 0.0)
        buffer_minutes = state.get("buffer_minutes", 15.0)
        arrival_field = state["arrival_field"]

        planned = plan_routes(
            road_graph=road_graph,
            origins=origins,
            destinations=destinations,
            departure_time=departure_time,
            arrival_field=arrival_field,
            buffer_minutes=buffer_minutes,
        )

        outputs = {"planned_routes": planned}
        log_step_event(
            log_file=state.get("log_file", self.log_file),
            step_name="plan_routes",
            update_id=update_id,
            inputs={"origins": origins, "destinations": destinations, "departure_time": departure_time},
            outputs={"planned_routes_count": len(planned)},
        )
        return outputs

    def _node_validate_routes(self, state: AgentState) -> dict[str, Any]:
        update_id = state.get("update_id", "UPDATE_001")
        road_graph = state["road_graph"]
        planned_routes = state["planned_routes"]
        arrival_field = state["arrival_field"]
        buffer_minutes = state.get("buffer_minutes", 15.0)

        val_results = validate_routes(
            road_graph=road_graph,
            routes=planned_routes,
            arrival_field=arrival_field,
            buffer_minutes=buffer_minutes,
        )

        outputs = {"validation_results": val_results}
        log_step_event(
            log_file=state.get("log_file", self.log_file),
            step_name="validate_routes",
            update_id=update_id,
            inputs={"route_count": len(planned_routes)},
            outputs={"all_valid": all(v.is_valid for v in val_results.values())},
        )
        return outputs

    def _node_compute_exposure(self, state: AgentState) -> dict[str, Any]:
        update_id = state.get("update_id", "UPDATE_001")
        road_graph = state["road_graph"]
        planned_routes = state["planned_routes"]
        origins = state.get("origins", [])
        destinations = state.get("destinations", [])
        departure_time = state.get("departure_time", 0.0)
        buffer_minutes = state.get("buffer_minutes", 15.0)
        arrival_field = state["arrival_field"]

        exposure = compute_exposure(
            routes=planned_routes,
            road_graph=road_graph,
            buffer_minutes=buffer_minutes,
        )
        k_stars = compute_k_star(
            road_graph=road_graph,
            origins=origins,
            destinations=destinations,
            departure_time=departure_time,
            arrival_field=arrival_field,
            buffer_minutes=buffer_minutes,
        )

        outputs = {
            "exposure_summary": exposure,
            "k_star_values": k_stars,
        }
        log_step_event(
            log_file=state.get("log_file", self.log_file),
            step_name="compute_exposure",
            update_id=update_id,
            inputs={"route_count": len(planned_routes)},
            outputs={"exposure": exposure.to_dict(), "k_stars": k_stars},
        )
        return outputs

    def _node_decide_replan(self, state: AgentState) -> dict[str, Any]:
        update_id = state.get("update_id", "UPDATE_001")
        current_routes = state.get("current_routes")
        planned_routes = state["planned_routes"]
        prev_field = state.get("previous_arrival_field")
        curr_field = state["arrival_field"]
        road_cuts = state.get("road_cut_edges")

        active_routes, decision = self.policy.decide(
            current_routes=current_routes,
            candidate_routes=planned_routes,
            previous_arrival_field=prev_field,
            new_arrival_field=curr_field,
            road_cut_edges=road_cuts,
            road_graph=state.get("road_graph"),
        )

        outputs = {
            "active_routes": active_routes,
            "replan_decision": decision,
        }
        log_step_event(
            log_file=state.get("log_file", self.log_file),
            step_name="decide_replan",
            update_id=update_id,
            inputs={"has_current_routes": current_routes is not None},
            outputs={"replan_decision": decision.to_dict()},
        )
        return outputs

    def _node_brief(self, state: AgentState) -> dict[str, Any]:
        update_id = state.get("update_id", "UPDATE_001")
        prompt = self.brief_generator.build_brief(state)

        # Attempt 1: Call LLM Client
        brief_candidate = self.llm_client.generate(prompt)
        check_result = self.grounding_checker.check(brief_candidate, state)

        # On reject: regenerate once with feedback
        if not check_result.passed:
            feedback_prompt = (
                f"{prompt}\n\nCORRECTION REQUIRED: Previous brief was rejected due to: "
                f"{'; '.join(check_result.rejected_tokens)}. "
                "Ensure all figures match tool outputs exactly. Never state clearance hours."
            )
            brief_candidate = self.llm_client.generate(feedback_prompt)
            check_result2 = self.grounding_checker.check(brief_candidate, state)

            if check_result2.passed:
                final_brief = brief_candidate
                final_result = GroundingResult(
                    passed=True,
                    rejected_tokens=[],
                    attempts=2,
                    used_template_fallback=False,
                )
            else:
                # Fall back to deterministic template
                final_brief = prompt
                final_result = GroundingResult(
                    passed=True,
                    rejected_tokens=check_result2.rejected_tokens,
                    attempts=2,
                    used_template_fallback=True,
                )
        else:
            final_brief = brief_candidate
            final_result = check_result

        outputs = {
            "brief": final_brief,
            "grounding_result": final_result,
        }
        log_step_event(
            log_file=state.get("log_file", self.log_file),
            step_name="brief",
            update_id=update_id,
            inputs={"prompt_len": len(prompt)},
            outputs={
                "brief_length": len(final_brief),
                "grounding_passed": final_result.passed,
                "used_template_fallback": final_result.used_template_fallback,
                "attempts": final_result.attempts,
            },
        )
        return outputs

    def run_update(self, state: AgentState) -> AgentState:
        """Execute a complete update cycle through the LangGraph state machine."""
        return self.graph.invoke(state)
