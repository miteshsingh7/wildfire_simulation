"""Provider-agnostic LLM interface, mock generator, real provider adapter, and deterministic brief builder."""

import os
from typing import Protocol

import httpx

from wf052.agent.contracts import AgentState, ArrivalFieldMeta, ExposureSummary, WeatherReport
from wf052.routing.contracts import EvacuationRoute, RouteStatus

STANDING_CAVEAT: str = (
    "Standing Caveat: The fire model may not capture ember spotting, long-range firebrands, "
    "or unmodeled secondary ignitions; areas outside the predicted fire perimeter are not guaranteed safe."
)

LOW_CONFIDENCE_NOTICE: str = (
    "ALERT: Arrival field is flagged LOW CONFIDENCE (fire model shows minimal or zero spread "
    "beyond the ignition zone or contradicts observations). Field observations or sensor coverage may be incomplete."
)

NO_SAFE_ROUTE_ANY_NOTICE: str = (
    "CRITICAL ALERT: NO SAFE EVACUATION ROUTE EXISTS FOR ANY ORIGIN. All computed paths are "
    "BEST-EFFORT only (required safety margin buffer violated or corridors impassable)."
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
        status_corruption: str | None = None,
    ) -> None:
        self.mode = mode
        self.corrupt_token = corrupt_token
        self.hallucinated_number = hallucinated_number
        self.inject_clearance_hours = inject_clearance_hours
        self.omit_caveat = omit_caveat
        self.status_corruption = status_corruption
        self.call_count = 0

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.call_count += 1

        # If in corrupt mode on attempt 1, inject corruptions to verify grounding checker catches them
        if self.mode == "corrupt" and self.call_count == 1:
            bad_brief = "Evacuation Brief Update.\n"
            if self.hallucinated_number:
                bad_brief += f"Fire speed observed at {self.hallucinated_number} km/h.\n"
            if self.corrupt_token:
                bad_brief += f"Traffic routing via {self.corrupt_token}.\n"
            if self.status_corruption:
                bad_brief += f"{self.status_corruption}\n"
            if self.inject_clearance_hours:
                bad_brief += "Estimated regional clearance time is 4.5 hours.\n"
            if not self.omit_caveat:
                bad_brief += f"{STANDING_CAVEAT}\n"
            return bad_brief

        # Otherwise parse structured data out of prompt or return standard grounded brief
        generator = DeterministicBriefGenerator()
        return generator.build_brief_from_text_prompt(prompt)


class ProviderLLMClient:
    """Real LLM provider adapter using httpx to call provider endpoints.

    The API key is strictly read from environment variables and is never written
    to logs, files, or state dictionaries.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        api_key: str | None = None,
    ) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()
        self.timeout = timeout
        self.model = model
        # API key is kept in memory only; never printed or logged
        if api_key:
            self._api_key = api_key
        elif self.provider == "openai":
            self._api_key = os.getenv("OPENAI_API_KEY", "")
        elif self.provider == "anthropic":
            self._api_key = os.getenv("ANTHROPIC_API_KEY", "")
        elif self.provider == "gemini":
            self._api_key = os.getenv("GEMINI_API_KEY", "")
        else:
            self._api_key = os.getenv("LLM_API_KEY", "")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        if self.provider == "mock":
            return MockLLMClient().generate(prompt, system_prompt=system_prompt)

        if not self._api_key:
            raise ValueError(
                f"Cannot invoke real LLM provider '{self.provider}': API key environment variable not set."
            )

        if self.provider == "openai":
            return self._call_openai(prompt, system_prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, system_prompt)
        elif self.provider == "gemini":
            return self._call_gemini(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _call_openai(self, prompt: str, system_prompt: str | None) -> str:
        url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": messages,
            "temperature": 0.0,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"]).strip()

    def _call_anthropic(self, prompt: str, system_prompt: str | None) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data["content"][0]["text"]).strip()

    def _call_gemini(self, prompt: str, system_prompt: str | None) -> str:
        model_name = self.model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self._api_key}"
        headers = {"Content-Type": "application/json"}
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System instructions: {system_prompt}"}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        payload = {"contents": contents}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data["candidates"][0]["content"]["parts"][0]["text"]).strip()


def create_llm_client(provider: str | None = None) -> LLMClient:
    """Instantiate appropriate LLM client based on provider parameter or environment."""
    selected = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()
    if selected == "mock":
        return MockLLMClient()
    return ProviderLLMClient(provider=selected)


class DeterministicBriefGenerator:
    """Deterministic fallback brief generator with zero hallucination guarantee."""

    def build_brief(
        self,
        state: AgentState,
    ) -> str:
        """Generate a strictly grounded evacuation briefing from typed state data."""
        lines: list[str] = []

        routes: dict[int, EvacuationRoute] = (
            state.get("active_routes") or state.get("planned_routes") or {}
        )
        safe_routes = [r for r in routes.values() if r.status == RouteStatus.SAFE]
        best_effort_routes = [r for r in routes.values() if r.status != RouteStatus.SAFE]
        no_safe_any = len(safe_routes) == 0 and len(routes) > 0
        no_safe_some = len(safe_routes) > 0 and len(best_effort_routes) > 0

        # Rule 2: If no safe route exists for some or all origins, say so FIRST!
        if no_safe_any:
            lines.append(NO_SAFE_ROUTE_ANY_NOTICE)
            lines.append("")
        elif no_safe_some:
            compromised_str = ", ".join(f"Zone {r.origin_id}" for r in sorted(best_effort_routes, key=lambda x: x.origin_id))
            lines.append(
                f"CRITICAL ALERT: NO SAFE EVACUATION ROUTE EXISTS FOR {compromised_str}. "
                "Routes for these zones are BEST-EFFORT only (required safety margin buffer violated)."
            )
            lines.append("")

        # Rule: If arrival field is low confidence, say so FIRST (or immediately after critical alert)
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
        k_stars: dict[int, float] = state.get("k_star_values", {})
        lines.append("\nAssigned Evacuation Corridors:")

        if not routes:
            lines.append("  No active routes assigned.")
        else:
            for orig, r in sorted(routes.items()):
                # Requirement 2: Never use the word "safe" for a best-effort or no-route result!
                if r.status == RouteStatus.SAFE:
                    stat = "SAFE"
                elif r.status == RouteStatus.BEST_EFFORT:
                    stat = "BEST-EFFORT"
                else:
                    stat = "NO ROUTE"

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
            if no_safe_any:
                lines.append(f"  Routes Meeting Safety Buffer: 0 of {exp.total_routes}")
                lines.append(
                    f"  CRITICAL ENTRAPMENT HAZARD: All {exp.total_routes} route(s) fail buffer threshold. "
                    "Assigned paths are best-effort egress only."
                )
            elif no_safe_some:
                lines.append(f"  Routes Meeting Safety Buffer: {exp.safe_count} of {exp.total_routes}")
                lines.append(
                    f"  CRITICAL: {exp.trapped_count} route(s) fail safety margin threshold (best-effort status)."
                )
            else:
                lines.append(f"  Routes Meeting Safety Buffer: {exp.safe_count} of {exp.total_routes}")

        return "\n".join(lines)

    def build_brief_from_text_prompt(self, prompt: str) -> str:
        """Parse structured details from prompt and format a compliant grounded brief."""
        lines: list[str] = []

        is_no_safe = "NO SAFE EVACUATION ROUTE" in prompt or "no safe evacuation route" in prompt.lower()
        if is_no_safe:
            for pl in prompt.split("\n"):
                if "NO SAFE EVACUATION ROUTE" in pl:
                    lines.append(pl.strip())
                    lines.append("")
                    break

        is_low_conf = "LOW CONFIDENCE" in prompt or "low confidence" in prompt.lower()
        if is_low_conf:
            lines.append(LOW_CONFIDENCE_NOTICE)
            lines.append("")

        lines.append(STANDING_CAVEAT)
        lines.append("")
        lines.append("=== Wildfire Evacuation Operational Briefing ===")

        for line in prompt.split("\n"):
            line_str = line.strip()
            if any(k in line_str for k in ["Weather:", "Zone", "Traffic Conditions:", "Coordination Directive:", "Routes Meeting Safety Buffer:"]):
                lines.append(line_str)

        return "\n".join(lines)
