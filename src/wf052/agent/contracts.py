"""Data contracts, state definitions, and logging events for evacuation agent."""

import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from wf052.routing.contracts import ArrivalField, EvacuationRoute
from wf052.routing.graph import RoadGraph
from wf052.routing.validator import ValidationResult


@dataclass(frozen=True)
class WeatherReport:
    """Cached weather stub or observed conditions."""

    wind_speed_mps: float
    wind_dir_deg: float
    temp_c: float
    rh_pct: float
    is_stub: bool = True
    station_id: str = "HRRR_STUB"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArrivalFieldMeta:
    """Metadata regarding arrival field spread extent and confidence."""

    is_low_confidence: bool
    burned_cells: int
    total_cells: int
    min_arrival_min: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExposureSummary:
    """Summary of vehicle exposure and entrapment risk."""

    total_routes: int
    trapped_count: int
    safe_count: int
    lowest_margin_min: float
    bottleneck_edges: list[tuple[int, int]]
    qualitative_traffic: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_routes": self.total_routes,
            "trapped_count": self.trapped_count,
            "safe_count": self.safe_count,
            "lowest_margin_min": round(self.lowest_margin_min, 2),
            "bottleneck_edges": [list(e) for e in self.bottleneck_edges],
            "qualitative_traffic": self.qualitative_traffic,
        }


@dataclass(frozen=True)
class ReplanDecision:
    """Decision produced by the re-plan policy with hysteresis."""

    needs_replan: bool
    reason: str
    changed_origins: list[int]
    flap_prevented_origins: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "needs_replan": self.needs_replan,
            "reason": self.reason,
            "changed_origins": self.changed_origins,
            "flap_prevented_origins": self.flap_prevented_origins,
        }


@dataclass(frozen=True)
class GroundingResult:
    """Output from the brief grounding check."""

    passed: bool
    rejected_tokens: list[str]
    attempts: int
    used_template_fallback: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentState(TypedDict, total=False):
    """Typed state object passed between nodes in the LangGraph state machine."""

    update_id: str
    timestamp: str
    road_graph: RoadGraph
    origins: list[int]
    destinations: list[int]
    departure_time: float
    buffer_minutes: float

    # Observation inputs
    weather: WeatherReport
    arrival_field: ArrivalField
    arrival_field_meta: ArrivalFieldMeta

    # Route and validation state
    current_routes: dict[int, EvacuationRoute]
    planned_routes: dict[int, EvacuationRoute]
    active_routes: dict[int, EvacuationRoute]
    validation_results: dict[int, ValidationResult]

    # Exposure & slowdown metrics
    exposure_summary: ExposureSummary
    k_star_values: dict[int, float]

    # Policy decisions
    replan_decision: ReplanDecision

    # Communication
    brief: str
    grounding_result: GroundingResult

    # Logging
    log_file: str
    history: list[dict[str, Any]]


def log_step_event(
    log_file: str | Path | None,
    step_name: str,
    update_id: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> dict[str, Any]:
    """Write an input/output record for a state machine step to a JSONL log file."""
    record = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "update_id": update_id,
        "step": step_name,
        "inputs": _sanitize_for_json(inputs),
        "outputs": _sanitize_for_json(outputs),
    }

    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    return record


def _sanitize_for_json(obj: Any) -> Any:
    """Helper to convert complex objects (numpy, dataclasses, routes) into json-serializable structures."""
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        if isinstance(obj, float) and np.isneginf(obj):
            return "-inf"
        if isinstance(obj, float) and np.isposinf(obj):
            return "inf"
        if isinstance(obj, float) and np.isnan(obj):
            return "nan"
        return obj
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: _sanitize_for_json(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)
