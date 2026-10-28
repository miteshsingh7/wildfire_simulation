"""Extract real, abridged source excerpts from the repository by regex anchors (no hand-typed code)."""

import re
from pathlib import Path

ROOT = Path("/vercel/share/v0-project")


def _lines(rel):
    return (ROOT / rel).read_text(encoding="utf-8").splitlines()


def grab(rel, start, end=None, *, after=None, extra_after=0, dedent=0):
    """Return lines from the first line matching `start` (searched after the first `after` match) through the
    first following line matching `end` (inclusive), plus `extra_after` lines. With no `end`, only the start line."""
    lines = _lines(rel)
    i0 = 0
    if after:
        i0 = next(i for i, ln in enumerate(lines) if re.search(after, ln))
    s = next(i for i in range(i0, len(lines)) if re.search(start, lines[i]))
    if end is None:
        e = s
    else:
        e = next(i for i in range(s, len(lines)) if re.search(end, lines[i]))
    seg = lines[s : e + 1 + extra_after]
    if dedent:
        seg = [ln[dedent:] if ln.startswith(" " * dedent) else ln for ln in seg]
    return seg


def cat(*segments):
    out = []
    for seg in segments:
        out.extend([seg] if isinstance(seg, str) else seg)
    return out


ELIDE = "    ..."

SNIPPETS = {
    # ---- Milestone 1 ----
    "landfire": cat(
        grab("src/wf052/data/landfire.py", r"^LANDFIRE_SERVICES = \{", r'"elevation":'),
        "    # ... slope, aspect, canopy_cover, canopy_height, canopy_base_height, canopy_bulk_density",
        "}",
        "",
        "# align_and_save_layer (excerpt): force every layer onto the exact canonical grid",
        grab(
            "src/wf052/data/landfire.py",
            r"# Validate or resize to exact grid dimensions",
            r"data = zoom\(data, \(zoom_y, zoom_x\), order=order\)",
            dedent=4,
        ),
    ),
    "weather_obs": cat(
        "# src/wf052/data/weather.py (fetch_hrrr_hourly, excerpt)",
        grab(
            "src/wf052/data/weather.py",
            r"ws_ms = float\(np.sqrt",
            r"temp_c = temp_k - 273.15",
            dedent=8,
        ),
        "",
        "# src/wf052/data/observations.py (parse_nist_appendix_f, excerpt)",
        grab(
            "src/wf052/data/observations.py",
            r"# Elapsed minutes from ignition on Nov 8",
            r"utm_x, utm_y = transformer.transform\(lon, lat\)",
            dedent=16,
        ),
    ),
    "roads": grab(
        "src/wf052/data/roads.py",
        r"# 1\. Attempt historical snapshot",
        r"logger.info\(\"Current OSM query succeeded",
        dedent=4,
    ),
    # ---- Milestone 2 ----
    "fuel_map": cat(
        grab("src/wf052/sim/pytorchfire_engine.py", r"^FBFM40_TO_P_VEG", r"92: -1\.0"),
        "    # ... 98 Open Water and 99 Barren (-1.0); GR 101-108 (0.30-0.60); GS 121-124; SH 141-147;",
        "    # ... TU 161-165; TL 181-189 (0.15-0.45); SB 201-204 (0.30-0.60)",
        "}",
        "",
        grab(
            "src/wf052/sim/pytorchfire_engine.py",
            r"# 2\. Map Canopy Cover to p_den",
            r"p_den_arr = np.clip",
            dedent=4,
        ),
    ),
    "teacher_v2": grab("configs/teacher_v2.yaml", r"^version: teacher_v2", r"wind_lofting_threshold_mps"),
    "ember": cat(
        grab(
            "src/wf052/sim/ember_kernel.py",
            r"v_eff = max\(float\(wind_speed_ms\)",
            r"launched_mask = burning",
            dedent=4,
        ),
        "",
        grab(
            "src/wf052/sim/ember_kernel.py",
            r"# 2\. Lognormal landing distance",
            r"dist_cells = dist_m / res_m",
            dedent=4,
        ),
        "",
        grab(
            "src/wf052/sim/ember_kernel.py",
            r"hit_counts = torch\.zeros",
            r"hit_mask = hit_counts",
            dedent=4,
        ),
    ),
    # ---- Milestone 3 ----
    "leak_guard": cat(
        grab(
            "src/wf052/surrogate/standin_generator.py",
            r"^PROHIBITED_OR_HELD_OUT_FIRES",
            r"^\]",
        ),
        "",
        "# src/wf052/surrogate/trainer.py (SurrogateTrainer.__init__, excerpt)",
        grab(
            "src/wf052/surrogate/trainer.py",
            r"train_fires = _extract_fires\(train_samples\)",
            r"Set allow_same_fire_leak=True to override with explicit warning",
            extra_after=1,
            dedent=8,
        ),
    ),
    "film": grab(
        "src/wf052/surrogate/model.py",
        r"def forward\(self, x: torch.Tensor, z_wind",
        r"^        return h$",
        after=r"^class FiLMBlock",
        dedent=4,
    ),
    "loss": cat(
        grab(
            "src/wf052/surrogate/losses.py",
            r"pinball_total = \(l_q10 \+ l_q50 \+ l_q90\)",
            r"total = pinball_total",
            dedent=8,
        ),
    ),
    "heads": cat(
        grab(
            "src/wf052/surrogate/model.py",
            r"# Quantile predictions: bounded strictly",
            r"burn_logits = self\.burn_logit_head\(y1\)",
            dedent=8,
        ),
        "",
        grab(
            "src/wf052/surrogate/model.py",
            r"def init_burn_prob_bias",
            r"return bias_val",
            dedent=4,
        ),
    ),
    # ---- Milestone 4 ----
    "route_contract": cat(
        grab("src/wf052/routing/contracts.py", r"^class RoutingMode", r"MAX_MIN_MARGIN"),
        "",
        grab("src/wf052/routing/contracts.py", r"^class RouteStatus", r"NO_ROUTE"),
        "",
        grab(
            "src/wf052/routing/contracts.py",
            r"required_vars = \(",
            r"raise ValueError\(f\"ArrivalField missing required data variable",
            extra_after=0,
            dedent=4,
        ),
    ),
    "dijkstra": cat(
        grab(
            "src/wf052/routing/router.py",
            r"for v, _k, dt_min, edge_idx in road_graph\.adj\.get\(u",
            r"heapq\.heappush\(pq, \(t_exit, v\)\)",
            dedent=8,
        ),
        "",
        "# TimeDependentRouter.find_route (Mode b, excerpt)",
        grab(
            "src/wf052/routing/router.py",
            r"# Binary search over achievable safety margin",
            r"m_high = m_mid",
            dedent=8,
        ),
    ),
    "congestion": cat(
        grab(
            "src/wf052/routing/congestion.py",
            r"nominal_cap = float\(cap_per_lane \* lanes\)",
            r"effective_cap = max\(10\.0",
            dedent=12,
        ),
        "",
        grab(
            "src/wf052/routing/congestion.py",
            r"# Physical link vehicle storage",
            r"jam_density_veh_per_km_per_lane",
            extra_after=1,
            dedent=12,
        ),
    ),
    "replan_hazard": cat(
        grab(
            "src/wf052/routing/replan.py",
            r"# 1\. Update edge fire arrival times",
            r"edge_fire_arrivals = self\.road_graph\.compute_edge_fire_arrivals",
            dedent=8,
        ),
        "",
        grab(
            "src/wf052/routing/hazard_check.py",
            r"# If vehicle passed this edge entirely before hazard became active",
            r"encountered_ids\.add\(h\.hazard_id\)",
            dedent=12,
        ),
    ),
    # ---- Milestone 5 ----
    "agent_graph": grab(
        "src/wf052/agent/graph.py",
        r"def _build_graph\(self\)",
        r"return builder\.compile\(\)",
        dedent=4,
    ),
    "policy_grounding": cat(
        grab("src/wf052/agent/policy.py", r"^class ReplanPolicy", r"self\.hysteresis_time_pct = "),
        "",
        "# src/wf052/agent/grounding.py (excerpt)",
        grab("src/wf052/agent/grounding.py", r"^# Prohibited quantitative clearance regex", r"^\)"),
    ),
    "harness": cat(
        grab(
            "src/wf052/replay/harness.py",
            r"# In observations_nist\.parquet, elapsed_minutes=0\.0",
            r"gdf\[\"elapsed_minutes_adj\"\]",
            dedent=8,
        ),
        "",
        grab(
            "src/wf052/replay/harness.py",
            r"t_now = float\(current_step_min\)",
            r"\]\.copy\(\)$",
            dedent=8,
            after=r"def step\(self",
        ),
    ),
    # ---- Milestone 6 ----
    "serve_web": cat(
        grab("scripts/serve_web.py", r"^PORT = 8000", r"Press Ctrl\+C to stop"),
        "        ...",
    ),
    "pytest_ruff": cat(
        grab("pyproject.toml", r"^\[tool\.pytest\.ini_options\]", r"^\]"),
        "",
        grab("pyproject.toml", r"^\[tool\.ruff\]", r"^src = "),
    ),
}

if __name__ == "__main__":
    for name, seg in SNIPPETS.items():
        print(f"===== {name} ({len(seg)} lines) =====")
        print("\n".join(seg))
        print()
