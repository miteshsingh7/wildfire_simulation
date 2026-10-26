"""Grounding verification checker for evacuation briefing text."""

import re
from typing import Any

import numpy as np

from wf052.agent.contracts import AgentState, GroundingResult
from wf052.routing.contracts import RouteStatus

# Prohibited quantitative clearance regex (e.g. "clearance time is estimated at 3.5 hours", "cleared in 3 hrs")
CLEARANCE_HOURS_PATTERN = re.compile(
    r"(?:clearance|clear)[\w\s]{0,35}?\d+(?:\.\d+)?\s*(?:hours|hrs|hr\b|h\b)|(?:\d+(?:\.\d+)?)\s*(?:hours|hrs|hr\b|h\b)[\w\s]{0,35}?(?:clearance|to clear)",
    re.IGNORECASE,
)

# Road candidate pattern (e.g. "Highway 101", "Ocean Boulevard", "Clark Bypass", "Skyway N", "Clark Rd")
ROAD_CANDIDATE_PATTERN = re.compile(
    r"\b(?:[A-Za-z0-9]+)\s+(?:Highway|Hwy|Road|Rd|Boulevard|Blvd|Avenue|Ave|Street|St|Lane|Ln|Drive|Dr|Bypass|Expressway|Expwy|Parkway|Pkwy)(?:\s+(?:North|South|East|West|N|S|E|W|[0-9]+))?\b|\b(?:Highway|Hwy|Interstate|Route|I-)\s*[0-9]+[A-Za-z]?\b",
    re.IGNORECASE,
)

# Zone candidate pattern (e.g. "Zone 99", "Zone 01", "Zone_1", "Zone 1A")
ZONE_CANDIDATE_PATTERN = re.compile(
    r"\bZone\s*[_]?\s*(\d+[A-Za-z]?|[A-Za-z]\b)\b",
    re.IGNORECASE,
)

ROAD_ABBREVIATIONS: dict[str, str] = {
    r"\bhwy\b": "highway",
    r"\brd\b": "road",
    r"\bblvd\b": "boulevard",
    r"\bave\b": "avenue",
    r"\bst\b": "street",
    r"\bln\b": "lane",
    r"\bdr\b": "drive",
    r"\bexpwy\b": "expressway",
    r"\bpkwy\b": "parkway",
    r"\bn\b": "north",
    r"\bs\b": "south",
    r"\be\b": "east",
    r"\bw\b": "west",
}

ROAD_GENERIC_TOKENS: set[str] = {
    "highway", "hwy", "road", "rd", "boulevard", "blvd", "avenue", "ave",
    "street", "st", "lane", "ln", "drive", "dr", "way", "bypass", "expressway",
    "expwy", "parkway", "pkwy", "interstate", "route", "north", "south", "east", "west",
    "n", "s", "e", "w", "corridor", "path", "evacuation", "alert", "notice", "directive"
}


def normalize_road_name(name: str) -> str:
    """Normalize road name by expanding standard abbreviations and standardizing whitespace."""
    norm = name.strip().lower()
    for pattern, repl in ROAD_ABBREVIATIONS.items():
        norm = re.sub(pattern, repl, norm)
    return re.sub(r"\s+", " ", norm).strip()


def extract_core_road_tokens(name: str) -> set[str]:
    """Extract distinct non-generic named tokens for a road."""
    tokens = re.findall(r"[a-z0-9]+", normalize_road_name(name))
    return {t for t in tokens if t not in ROAD_GENERIC_TOKENS and len(t) > 1}


def is_road_match(candidate: str, allowed_strings: set[str]) -> bool:
    """Check if candidate road name matches any allowed road in the network."""
    cand_norm = normalize_road_name(candidate)
    cand_core = extract_core_road_tokens(candidate)
    if not cand_core:
        return True

    for allowed in allowed_strings:
        allowed_norm = normalize_road_name(allowed)
        if cand_norm == allowed_norm or cand_norm in allowed_norm or allowed_norm in cand_norm:
            return True
        allowed_core = extract_core_road_tokens(allowed)
        if cand_core and allowed_core and cand_core.issubset(allowed_core):
            return True
    return False


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
                allowed_numbers.add(abs(round(float(route.min_margin_min), 1)))

            # Edge lengths and traversal times
            for edge in route.edges:
                allowed_numbers.add(round(float(edge.length_m), 1))
                allowed_numbers.add(round(float(edge.traversal_time_min), 1))
                if np.isfinite(edge.margin_min):
                    allowed_numbers.add(round(float(edge.margin_min), 1))
                    allowed_numbers.add(abs(round(float(edge.margin_min), 1)))

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
                allowed_numbers.add(abs(round(float(exp.lowest_margin_min), 1)))

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
            try:
                orig_int = int(orig)
                allowed_strings.add(f"zone {orig_int:02d}".lower())
                allowed_strings.add(f"zone_{orig_int:02d}".lower())
            except (ValueError, TypeError):
                pass
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

        # 1. Status consistency check (claims of safety must match actual route status)
        routes = state.get("active_routes") or state.get("planned_routes") or {}
        for orig, r in routes.items():
            orig_str = str(orig)
            if r.status != RouteStatus.SAFE:
                # Route is compromised / best-effort: must NEVER claim safety
                zone_safe_pattern = re.compile(
                    rf"(?:Zone|Origin)\s*[_]?\s*0*{orig_str}\b[^|\n]*?:\s*SAFE\b|"
                    rf"(?:Zone|Origin)\s*[_]?\s*0*{orig_str}\b[^\.\n]*?\b(?:is\s+safe|are\s+safe|safe\s+corridor|safe\s+route|safe\s+path|safe\s+evacuation)",
                    re.IGNORECASE,
                )
                if zone_safe_pattern.search(brief_text):
                    rejected_reasons.append(
                        f"Status inconsistency: Route for Zone {orig} has status '{r.status.value}', "
                        "but brief falsely claimed it is 'safe'."
                    )
            elif r.status == RouteStatus.SAFE:
                # Route is safe: must not falsely claim no safe route exists
                false_unsafe_pattern = re.compile(
                    rf"no\s+safe\s+evacuation\s+route\s+exists\s+for[^\.\n]*?(?:Zone|Origin)\s*[_]?\s*0*{orig_str}\b",
                    re.IGNORECASE,
                )
                if false_unsafe_pattern.search(brief_text):
                    rejected_reasons.append(
                        f"Status inconsistency: Route for Zone {orig} is SAFE, "
                        "but brief falsely claimed no safe route exists for it."
                    )

        all_unsafe = len(routes) > 0 and all(r.status != RouteStatus.SAFE for r in routes.values())
        if all_unsafe:
            if re.search(r"\b(?:all\s+routes\s+are\s+safe|all\s+corridors\s+are\s+safe)\b", brief_text, re.IGNORECASE):
                rejected_reasons.append(
                    "Status inconsistency: All routes fail safety margin, but brief claimed all routes are safe."
                )

        # 2. Rule: Congestion and clearance figures are qualitative: never state clearance hours
        clearance_match = CLEARANCE_HOURS_PATTERN.search(brief_text)
        if clearance_match:
            rejected_reasons.append(
                f"Prohibited quantitative clearance hours detected: '{clearance_match.group(0)}'. "
                "Clearance must be described qualitatively."
            )

        # 3. Rule: Every brief carries standing caveat
        caveat_key = "fire model may not capture ember spotting"
        if caveat_key.lower() not in brief_text.lower() and "fire model can miss spotting" not in brief_text.lower():
            rejected_reasons.append(
                "Missing required standing caveat on ember spotting and unmodeled fire perimeter safety."
            )

        # 4. Rule: If arrival field is low confidence, say so first
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

        # 5. Extract numbers from brief and verify they are grounded in tool outputs
        ground_truth = self.extract_allowed_data(state)
        allowed_nums = ground_truth["allowed_numbers"]

        # Tokenize floating point / integer numbers (ignoring timestamps like 2026-10-22)
        clean_text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?", " ", brief_text)
        clean_text = re.sub(r"\[UPDATE_\d+\]", " ", clean_text)

        num_tokens = re.findall(r"\b\d+(?:\.\d+)?\b", clean_text)
        for tok in num_tokens:
            try:
                val = float(tok)
            except ValueError:
                continue

            matches = any(abs(val - a) <= self.numeric_tolerance for a in allowed_nums)
            if not matches:
                # Also allow small structural ordinals (0, 1, 2)
                if val not in [0.0, 1.0, 2.0]:
                    rejected_reasons.append(f"Ungrounded number '{tok}' ({val}) not present in tool outputs.")

        # 6. Check candidate road names and zone names in brief
        allowed_strings = ground_truth["allowed_strings"]
        if allowed_strings:
            road_matches = ROAD_CANDIDATE_PATTERN.findall(brief_text)
            for rm in road_matches:
                rm_clean = rm.strip()
                rm_norm = normalize_road_name(rm_clean)
                if rm_norm in [
                    "highway", "road", "street", "way", "bypass",
                    "boulevard", "avenue", "drive", "lane", "parkway", "expressway",
                    "route", "routes", "evacuation route", "safe route",
                ]:
                    continue
                if any(w in rm_norm for w in ["evacuation", "alert", "notice", "directive", "traffic conditions"]):
                    continue
                if not is_road_match(rm_clean, allowed_strings):
                    rejected_reasons.append(
                        f"Ungrounded road name '{rm_clean}' not present in road network."
                    )

            zone_matches = ZONE_CANDIDATE_PATTERN.findall(brief_text)
            for zm in zone_matches:
                zm_clean = zm.strip()
                if zm_clean.lower() in {"s", "or", "and", "of", "to", "in", "the", "for", "by", "from", "with", "is", "are"}:
                    continue
                zm_norm = zm_clean.lstrip("0") if zm_clean.lstrip("0") else "0"
                zone_str_norm = f"zone {zm_norm}".lower()
                zone_str_raw = f"zone {zm_clean.lower()}"
                is_valid_zone = any(
                    zone_str_norm == s
                    or zone_str_raw == s
                    or zm_clean.lower() == s
                    or zm_norm == s
                    or s == f"zone_{zm_norm}"
                    or s == f"zone_{zm_clean.lower()}"
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
