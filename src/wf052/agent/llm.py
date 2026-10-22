"""Provider-agnostic LLM interface, mock generator, and deterministic brief builder."""

from typing import Protocol

from wf052.agent.contracts import AgentState, ArrivalFieldMeta, ExposureSummary, WeatherReport
from wf052.routing.contracts import EvacuationRoute, RouteStatus

STANDING_CAVEAT: str = (
    "Standing Caveat: The fire model may not capture ember spotting, long-range firebrands, "
    "or unmodeled secondary ignitions; areas outside the predicted fire perimeter are not guaranteed safe."
)

LOW_CONFIDENCE_NOTICE: str = (
    "ALERT: Arrival field is flagged LOW CONFIDENCE (fire model shows minimal or zero spread "
    "beyond the ignition zone). Field observations or sensor coverage may be incomplete."
)


class LLMClient(Protocol):
    """Protocol for provider-agnostic LLM clients."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate response text from prompt."""
        ...


class MockLLMClient:
    """Mock LLM client for deterministic, sub-second testing."""

    def __init__(
        self,
        mode: str = "grounded",
        corrupt_token: str | None = None,
        hallucinated_number: str | None = None,
        inject_clearance_hours: bool = False,
        omit_caveat: bool = False,
    ) -> None:
        self.mode = mode
        self.corrupt_token = corrupt_token
        self.hallucinated_number = hallucinated_number
        self.inject_clearance_hours = inject_clearance_hours
        self.omit_caveat = omit_caveat
        self.call_count = 0

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.call_count += 1

        # Fallback to grounded template generation
        # If in corrupt mode on attempt 1, inject corruptions to verify grounding checker catches them
        if self.mode == "corrupt" and self.call_count == 1:
            bad_brief = "Evacuation Brief Update.\n"
            if self.hallucinated_number:
                bad_brief += f"Fire speed observed at {self.hallucinated_number} km/h.\n"
            if self.corrupt_token:
                bad_brief += f"Traffic routing via {self.corrupt_token}.\n"
            if self.inject_clearance_hours:
                bad_brief += "Estimated regional clearance time is 4.5 hours.\n"
            if not self.omit_caveat:
                bad_brief += f"{STANDING_CAVEAT}\n"
            return bad_brief

        # Otherwise parse structured data out of prompt or return standard grounded brief
        generator = DeterministicBriefGenerator()
        # Look for briefing metadata embedded in prompt
        return generator.build_brief_from_text_prompt(prompt)


class DeterministicBriefGenerator:
    """Deterministic fallback brief generator with zero hallucination guarantee."""

    def build_brief(
        self,
        state: AgentState,
    ) -> str:
        """Generate a strictly grounded evacuation briefing from typed state data."""
        lines: list[str] = []

        # Rule: If arrival field is low confidence, say so FIRST
        meta: ArrivalFieldMeta | None = state.get("arrival_field_meta")
        if meta and meta.is_low_confidence:
            lines.append(LOW_CONFIDENCE_NOTICE)
            if meta.reason:
                lines.append(f"Confidence Note: {meta.reason}")
            lines.append("")

        # Rule: Standing caveat on fire spotting & unmodeled perimeters
        lines.append(STANDING_CAVEAT)
        lines.append("")

        update_id = state.get("update_id", "UPDATE_001")
        timestamp = state.get("timestamp", "2026-10-22T00:00:00Z")
        lines.append(f"=== Wildfire Evacuation Operational Briefing [{update_id}] ===")
        lines.append(f"Timestamp: {timestamp}")

        # Weather observations
        weather: WeatherReport | None = state.get("weather")
        if weather:
            lines.append(
                f"Weather: Wind {weather.wind_speed_mps:.1f} m/s at {weather.wind_dir_deg:.0f} deg | "
                f"Temp {weather.temp_c:.1f} C | RH {weather.rh_pct:.0f}%"
            )

        # Policy & Replan Decision
        decision = state.get("replan_decision")
        if decision:
            status_txt = "ACTION REQUIRED: Re-plan triggered" if decision.needs_replan else "STATUS QUO: Routes stable"
            lines.append(f"Coordination Directive: {status_txt} ({decision.reason})")

        # Route status
        routes: dict[int, EvacuationRoute] = state.get("active_routes", {})
        k_stars: dict[int, float] = state.get("k_star_values", {})
        lines.append("\nAssigned Evacuation Corridors:")

        if not routes:
            lines.append("  No active routes assigned.")
        else:
            for orig, r in sorted(routes.items()):
                stat = "SAFE" if r.status == RouteStatus.SAFE else "COMPROMISED"
                margin_str = f"{r.min_margin_min:.1f} min" if r.min_margin_min != float("inf") else "inf"
                k_str = f"{k_stars.get(orig, 1.0):.1f}x" if orig in k_stars else "N/A"
                lines.append(
                    f"  - Zone {orig} -> Exit {r.destination_id}: {stat} | "
                    f"Travel Time {r.total_travel_time_min:.1f} min | "
                    f"Margin {margin_str} | Congestion Headroom {k_str}"
                )

        # Exposure and Traffic Summary (Rule: Qualitative only, never clearance hours)
        exp: ExposureSummary | None = state.get("exposure_summary")
        if exp:
            lines.append("\nTraffic and Exposure Assessment:")
            lines.append(f"  Traffic Conditions: {exp.qualitative_traffic}")
            lines.append(f"  Routes Safe: {exp.safe_count} of {exp.total_routes}")
            if exp.trapped_count > 0:
                lines.append(f"  CRITICAL: {exp.trapped_count} route(s) fail safety margin threshold.")

        return "\n".join(lines)

    def build_brief_from_text_prompt(self, prompt: str) -> str:
        """Parse structured details from prompt and format a compliant grounded brief."""
        lines: list[str] = []

        is_low_conf = "LOW CONFIDENCE" in prompt or "low confidence" in prompt
        if is_low_conf:
            lines.append(LOW_CONFIDENCE_NOTICE)
            lines.append("")

        lines.append(STANDING_CAVEAT)
        lines.append("")
        lines.append("=== Wildfire Evacuation Operational Briefing ===")

        # Forward verbatim lines containing structured values
        for line in prompt.split("\n"):
            line_str = line.strip()
            if any(k in line_str for k in ["Weather:", "Zone", "Traffic Conditions:", "Coordination Directive:"]):
                lines.append(line_str)

        return "\n".join(lines)
