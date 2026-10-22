"""Grounding verification checker for evacuation briefing text."""

import re
from typing import Any

import numpy as np

from wf052.agent.contracts import AgentState, GroundingResult

# Prohibited quantitative clearance regex (e.g. "clearance time is estimated at 3.5 hours", "cleared in 3 hrs")
CLEARANCE_HOURS_PATTERN = re.compile(
    r"(?:clearance|clear)[\w\s]{0,35}?\d+(?:\.\d+)?\s*(?:hours|hrs|hr\b|h\b)|(?:\d+(?:\.\d+)?)\s*(?:hours|hrs|hr\b|h\b)[\w\s]{0,35}?(?:clearance|to clear)",
    re.IGNORECASE,
)

# Road candidate pattern (e.g. "Highway 101", "Ocean Boulevard", "Clark Bypass")
ROAD_CANDIDATE_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z0-9]+\s+)?(?:Highway|Hwy|Road|Rd|Boulevard|Blvd|Avenue|Ave|Street|St|Lane|Ln|Drive|Dr|Way|Bypass|Expressway|Route|Interstate)(?:\s+[A-Z0-9][a-z0-9]*)?\b",
    re.IGNORECASE,
)

# Zone candidate pattern (e.g. "Zone 99", "Zone B")
ZONE_CANDIDATE_PATTERN = re.compile(
    r"\bZone\s+([A-Za-z0-9_]+)\b",
    re.IGNORECASE,
)


class GroundingChecker:
    """Verifies that briefing prose does not hallucinate numbers, road names, or zone names."""

    def __init__(self, numeric_tolerance: float = 0.15) -> None:
        self.numeric_tolerance = numeric_tolerance

    def extract_allowed_data(self, state: AgentState) -> dict[str, Any]:
        """Extract all valid ground-truth numbers, road names, and zone names from state."""
        allowed_numbers: set[float] = set()
        allowed_strings: set[str] = set()

        # Buffer minutes & departure time
        if "buffer_minutes" in state:
            allowed_numbers.add(float(state["buffer_minutes"]))
        if "departure_time" in state:
            allowed_numbers.add(float(state["departure_time"]))

        # Weather numbers
        weather = state.get("weather")
        if weather:
            allowed_numbers.add(round(float(weather.wind_speed_mps), 1))
            allowed_numbers.add(round(float(weather.wind_dir_deg), 0))
            allowed_numbers.add(round(float(weather.temp_c), 1))
            allowed_numbers.add(round(float(weather.rh_pct), 0))

        # Arrival field metadata
        meta = state.get("arrival_field_meta")
        if meta:
            allowed_numbers.add(float(meta.burned_cells))
            allowed_numbers.add(float(meta.total_cells))
            if np.isfinite(meta.min_arrival_min):
                allowed_numbers.add(round(float(meta.min_arrival_min), 1))

        # Route numbers and node IDs
        routes = state.get("active_routes") or state.get("planned_routes") or {}
        for orig, route in routes.items():
            allowed_numbers.add(float(orig))
            allowed_numbers.add(float(route.destination_id))
            for nid in route.node_ids:
                allowed_numbers.add(float(nid))
            if np.isfinite(route.total_travel_time_min):
                allowed_numbers.add(round(float(route.total_travel_time_min), 1))
            if np.isfinite(route.min_margin_min):
                allowed_numbers.add(round(float(route.min_margin_min), 1))

            # Edge lengths and traversal times
            for edge in route.edges:
                allowed_numbers.add(round(float(edge.length_m), 1))
                allowed_numbers.add(round(float(edge.traversal_time_min), 1))

        # k* values
        k_stars = state.get("k_star_values") or {}
        for k_val in k_stars.values():
            allowed_numbers.add(round(float(k_val), 1))

        # Exposure numbers
        exp = state.get("exposure_summary")
        if exp:
            allowed_numbers.add(float(exp.total_routes))
            allowed_numbers.add(float(exp.trapped_count))
            allowed_numbers.add(float(exp.safe_count))
            if np.isfinite(exp.lowest_margin_min):
                allowed_numbers.add(round(float(exp.lowest_margin_min), 1))

        # Allowed road names and zone names from road_graph
        graph = state.get("road_graph")
        if graph:
            if hasattr(graph, "edges_gdf") and graph.edges_gdf is not None:
                if "name" in graph.edges_gdf.columns:
                    for val in graph.edges_gdf["name"].dropna():
                        allowed_strings.add(str(val).lower())
                if "highway" in graph.edges_gdf.columns:
                    for val in graph.edges_gdf["highway"].dropna():
                        allowed_strings.add(str(val).lower())

        # Origins & Destinations as valid zone names
        for orig in state.get("origins", []):
            allowed_strings.add(f"zone {orig}".lower())
            allowed_strings.add(f"zone_{orig}".lower())
            allowed_strings.add(f"origin {orig}".lower())
        for dest in state.get("destinations", []):
            allowed_strings.add(f"exit {dest}".lower())
            allowed_strings.add(f"destination {dest}".lower())

        return {
            "allowed_numbers": allowed_numbers,
            "allowed_strings": allowed_strings,
        }

    def check(
        self,
        brief_text: str,
        state: AgentState,
    ) -> GroundingResult:
        """Evaluate grounding, standing caveat, low confidence notice, and qualitative clearance rules."""
        rejected_reasons: list[str] = []

        # 1. Rule: Congestion and clearance figures are qualitative: never state clearance hours
        clearance_match = CLEARANCE_HOURS_PATTERN.search(brief_text)
        if clearance_match:
            rejected_reasons.append(
                f"Prohibited quantitative clearance hours detected: '{clearance_match.group(0)}'. "
                "Clearance must be described qualitatively."
            )

        # 2. Rule: Every brief carries standing caveat
        caveat_key = "fire model may not capture ember spotting"
        if caveat_key.lower() not in brief_text.lower() and "fire model can miss spotting" not in brief_text.lower():
            rejected_reasons.append(
                "Missing required standing caveat on ember spotting and unmodeled fire perimeter safety."
            )

        # 3. Rule: If arrival field is low confidence, say so first
        meta = state.get("arrival_field_meta")
        if meta and meta.is_low_confidence:
            conf_phrase = "low confidence"
            if conf_phrase not in brief_text.lower():
                rejected_reasons.append(
                    "Arrival field is flagged low confidence, but brief failed to state this."
                )
            else:
                # Check that low confidence notice appears before route corridors
                conf_pos = brief_text.lower().find(conf_phrase)
                corridor_pos = brief_text.lower().find("corridor")
                if corridor_pos != -1 and conf_pos > corridor_pos:
                    rejected_reasons.append(
                        "Low confidence notice must appear at the beginning of the brief before route details."
                    )

        # 4. Extract numbers from brief and verify they are grounded in tool outputs
        ground_truth = self.extract_allowed_data(state)
        allowed_nums = ground_truth["allowed_numbers"]

        # Tokenize floating point / integer numbers (ignoring timestamps like 2026-10-22)
        # Strip timestamps first to prevent ISO date tokens from tripping numeric check
        clean_text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?", " ", brief_text)
        clean_text = re.sub(r"\[UPDATE_\d+\]", " ", clean_text)

        # Find numbers
        num_tokens = re.findall(r"\b\d+(?:\.\d+)?\b", clean_text)
        for tok in num_tokens:
            try:
                val = float(tok)
            except ValueError:
                continue

            # Check if val matches any allowed number within tolerance
            matches = any(abs(val - a) <= self.numeric_tolerance for a in allowed_nums)
            if not matches:
                # Also allow small structural ordinals (0, 1, 2)
                if val not in [0.0, 1.0, 2.0]:
                    rejected_reasons.append(f"Ungrounded number '{tok}' ({val}) not present in tool outputs.")

        # 5. Check candidate road names and zone names in brief
        allowed_strings = ground_truth["allowed_strings"]
        if allowed_strings:
            road_matches = ROAD_CANDIDATE_PATTERN.findall(brief_text)
            for rm in road_matches:
                rm_clean = rm.strip()
                rm_lower = rm_clean.lower()
                if rm_lower in ["highway", "road", "street", "way", "bypass"]:
                    continue
                is_valid_road = any(
                    rm_lower == s or rm_lower in s or s in rm_lower
                    for s in allowed_strings
                    if len(s) > 3
                )
                if not is_valid_road:
                    rejected_reasons.append(
                        f"Ungrounded road name '{rm_clean}' not present in road network."
                    )

            zone_matches = ZONE_CANDIDATE_PATTERN.findall(brief_text)
            for zm in zone_matches:
                zone_str = f"zone {zm}".lower()
                is_valid_zone = any(
                    zone_str == s or zm.lower() == s or s in zone_str
                    for s in allowed_strings
                )
                if not is_valid_zone:
                    rejected_reasons.append(
                        f"Ungrounded zone name 'Zone {zm}' not present in scenario origins/destinations."
                    )

        passed = len(rejected_reasons) == 0
        return GroundingResult(
            passed=passed,
            rejected_tokens=rejected_reasons,
            attempts=1,
            used_template_fallback=not passed,
        )
