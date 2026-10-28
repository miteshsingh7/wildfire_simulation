"""Build the wf052 documentation DOCX by reusing the sample package (styles, numbering, header logos) as a template."""

import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from docx_lib import (
    bullet, body, caption, code_table, data_table, h1, h3, hyperlink, image, label, note, run, source_label, title,
)
from snippets import SNIPPETS

ROOT = Path("/vercel/share/v0-project")
SCRATCH = ROOT / ".scratch"
TEMPLATE = ROOT / "data" / "Air-Traffic_Conflict-Risk_Advisory_Agent_Documentation_Updated-a29c2f.docx"
OUT = Path("/vercel/share/artifacts/wf052_Wildfire_Evacuation_Planner_Documentation.docx")

FRAMED = Path("/tmp/agent-browser")  # framed 2880x1800 screenshots rendered by the browser
FLOWCHART = SCRATCH / "arch_flowchart_final.png"

# ---------------------------------------------------------------- relationship registry
_rels = []  # (rid, kind, target)
_next = [30]


def _rid(kind, target):
    for rid, k, t in _rels:
        if k == kind and t == target:
            return rid
    rid = f"rId{_next[0]}"
    _next[0] += 1
    _rels.append((rid, kind, target))
    return rid


def link(text, url):
    return hyperlink(_rid("hyperlink", url), text)


def links(*pairs):
    """Hyperlinks separated by ' | ' in the sample's style."""
    out = ""
    for i, (text, url) in enumerate(pairs):
        if i:
            out += run(" | ")
        out += link(text, url)
    return out


# ---------------------------------------------------------------- timeline from git log
def build_timeline():
    ist = timezone(timedelta(hours=5, minutes=30))
    raw = subprocess.check_output(
        ["git", "log", "--reverse", "--format=%h|%at|%s"], cwd=ROOT, text=True
    ).strip().splitlines()
    commits = []
    for line in raw:
        h, at, subj = line.split("|", 2)
        commits.append((h, datetime.fromtimestamp(int(at), ist), subj))
    by_hash = {h: (t, s) for h, t, s in commits}

    def when(h):
        for ch, t, _ in commits:
            if ch.startswith(h):
                return t
        raise KeyError(h)

    stages = [
        ("Foundational commit: repository scaffold (pyproject.toml, package layout, tests)", "05ec330"),
        ("Phase 0: seeding utilities, data ingestion pipelines, data card, PyTorchFire and Cell2Fire runners, feasibility report", "0e100e9"),
        ("Phase 0.5 audit of the feasibility spike", "8894033"),
        ("Phase 1A steps 0-2: perimeter timeline audit, resolution sensitivity, training-fire selection", "1b0793e"),
        ("Phase 1A step 3: leave-one-fire-out calibration, teacher_v1 freeze, held-out Camp Fire evaluation", "2432523"),
        ("Phase 1B: criticality sweep, assimilation mode, ember kernel, fresh held-out fire proposal", "22f7aae"),
        ("Time-aware evacuation router and synthetic-fire tests", "7953fca"),
        ("Phases 4b-4f: demand, congestion assignment, en-route re-planning, graph audit, k* slowdown module", "86082b8"),
        ("Wind-conditioned FiLM U-Net pipeline with stand-in data contract", "c8993f2"),
        ("Surrogate Phases 2B-2C: diagnostics, compact storage, router adapter, conformal calibration, Kaggle packaging", "6adb0e6"),
        ("Agent Phases 5A-5B: LangGraph state machine, typed tools, re-plan policy, grounded briefs, full-graph benchmark", "77ccd90"),
        ("Causal replay harness, Phase 6A evaluation and Phase 6B leakage audit", "fa59655"),
        ("teacher_v2 freeze, Phase 6C-6D replay integrity audits and teacher attribution", "06fa630"),
        ("Surrogate Phases 2D-2I: teacher_v2 ensemble labels, hardening, loss normalization, sampler fixes", "411a577"),
        ("Surrogate training and evaluation (Phases 3A, 3C) with surrogate-driven replay and Streamlit demo (Phases 4A-4C)", "4cc72bd"),
        ("Claims audit and freeze (Phases 4D-4F): hazard-type margins, limitations, dependency audit", "2d121ba"),
        ("Offline Stitch dashboard UI (Phases 5A-UI to 5C-UI-LITE): INCIDENT-052 web app, content audit, zone table proof", "fdd9eec"),
    ]

    def fmt_clock(t):
        return t.strftime("%d %b %H:%M")

    def fmt_elapsed(delta):
        mins = int(round(delta.total_seconds() / 60))
        d, rem = divmod(mins, 1440)
        h, m = divmod(rem, 60)
        parts = []
        if d:
            parts.append(f"{d}d")
        if d or h:
            parts.append(f"{h}h")
        parts.append(f"{m}m")
        return "~" + " ".join(parts)

    rows = []
    prev_end = None
    first = when(stages[0][1])
    for i, (name, end_hash) in enumerate(stages):
        end = when(end_hash)
        if prev_end is None:
            rows.append([name, fmt_clock(end), "N/A (first commit)"])
        else:
            same_day = prev_end.date() == end.date()
            window = (
                f"{fmt_clock(prev_end)} – {end.strftime('%H:%M')}"
                if same_day
                else f"{fmt_clock(prev_end)} – {fmt_clock(end)}"
            )
            rows.append([name, window, fmt_elapsed(end - prev_end)])
        prev_end = end
    last = commits[-1][1]
    rows.append(
        [
            "TOTAL (first commit → latest commit)",
            f"{fmt_clock(first)} – {fmt_clock(last)}",
            fmt_elapsed(last - first),
        ]
    )
    # longest gap, for the explanatory note
    gaps = [(commits[i + 1][1] - commits[i][1], commits[i], commits[i + 1]) for i in range(len(commits) - 1)]
    longest = max(gaps, key=lambda g: g[0])
    return rows, len(commits), longest


# ---------------------------------------------------------------- document content
def build_body():
    timeline_rows, n_commits, (gap, gap_a, gap_b) = build_timeline()
    X = []  # xml parts
    a = X.append

    a(title("Wildfire Spread Simulation and Agentic Evacuation Route Planner (wf052)"))
    a(label("Project Description:"))
    a(body(
        "Wildfire Spread Simulation and Agentic Evacuation Route Planner is an offline, research-only prototype for "
        "investigating physics-based wildfire spread simulation, deep-surrogate arrival-time emulation, risk-aware "
        "evacuation routing, and tool-augmented agentic coordination, developed and audited on the 2018 Camp Fire "
        "(Butte County, California) as an unseen development case. This software is for academic research and "
        "post-incident replay only and is not for operational use — it does not certify safe evacuation routes and "
        "must never be used for real-time emergency navigation or life-safety decisions."
    ))
    a(body(
        "The project pairs a calibrated 60 m cellular-automaton physics teacher (PyTorchFire contact spread plus a "
        "switchable non-local ember-spotting kernel) with a 4,094,852-parameter wind-conditioned FiLM U-Net surrogate "
        "that predicts arrival-time quantiles (q10 / q50 / q90) and a horizon burn probability, a time-dependent FIFO "
        "Dijkstra router with point-queue and BPR congestion models, and a LangGraph EvacuationAgent whose typed tools — "
        "never the language model — compute every route, margin and metric. A causal replay harness steps the Camp Fire "
        "forward in 10-minute intervals against 436 NIST ground-truth hazards (23 burnovers, 178 roadblocks and 235 "
        "abandoned vehicles) and compares seven routing methods, and a standalone offline dashboard (INCIDENT-052) "
        "exposes the precomputed replay for interactive inspection."
    ))
    a(body("Repository: github.com/miteshsingh7/wildfire_simulation"))

    # ---- Scenarios
    a(h1("Scenarios"))
    a(body(
        ("b", "Scenario 1: Offline Fire-Arrival Forecasting and Safety-Margin Routing: "),
        "A researcher loads a 256 x 256 terrain tile (60 m cells, 15.36 km x 15.36 km) and its 12-hour hourly wind "
        "series into the WindConditionedUNet. The surrogate returns arrival-time quantiles (q10 / q50 / q90, in minutes "
        "since ignition) and a burn-probability logit for every cell; an adapter wraps them as an ArrivalField, and the "
        "TimeDependentRouter returns, for each origin block, the fastest route whose every edge keeps at least a "
        "15-minute margin between vehicle exit time and the conservative q10 fire-arrival time — or, when no such route "
        "exists, the route that maximizes the minimum margin — before an independent validator re-checks it.",
    ))
    a(body(
        ("b", "Scenario 2: Causal Replay of the 2018 Camp Fire Against NIST Hazards: "),
        "An evaluator wants to know whether routes planned from a forecast fire field and delayed observations would "
        "have avoided the hazards that actually occurred. The CausalReplayHarness steps from the 06:33 PST ignition in "
        "10-minute intervals to a 6-hour horizon, exposes only observations whose timestamp is at or before t minus "
        "latency and roadblocks whose onset has passed, and routes 200 sampled block origins with seven comparative "
        "methods. Each route is scored against 436 NIST hazards, reporting encounters, trapped origins and safety "
        "margins with both plain and corridor-clustered bootstrap intervals.",
    ))
    a(body(
        ("b", "Scenario 3: Claims Governance and Offline Dashboard Verification: "),
        "A reviewer opens the INCIDENT-052 dashboard by double-clicking web/dist/index.html. Every figure on screen "
        "cites the CSV row it came from, the Status & Limitations tab separates what the system can and cannot claim, "
        "and a persistent strip states that the surrogate runs on untrained stand-in weights and that the tool is "
        "research-only. A headless Playwright script (scripts/verify_web_demo.py) re-asserts those phrases, confirms "
        "zero external network requests and zero console errors, and checks that the Zones table renders exactly 17 "
        "rows (14 zones and 3 corridors).",
    ))

    # ---- Architecture
    a(h1("Technical Architecture:"))
    a(image(_rid("image", "media/arch_flowchart.png"), "arch_flowchart",
            "Figure: wf052 Surrogate-to-Evacuation-Route Pipeline", 301, 4.0, 6.0))
    a(caption("Figure: wf052 Surrogate-to-Evacuation-Route Pipeline (from src/wf052/surrogate/model.py, "
              "src/wf052/routing/router.py, src/wf052/agent/graph.py)"))
    a(body(
        "The technical architecture consists of five interconnected subsystems: the data ingestion layer, the physics "
        "teacher simulator, the wind-conditioned neural surrogate, the risk-aware routing engine, and the agentic "
        "coordination layer together with its causal replay harness and offline dashboard."
    ))
    a(bullet(("b", "Data Ingestion & Canonical Grid: "),
        "Aligns LANDFIRE 2016 Remap fuels and topography, 12 hourly NOAA HRRR wind analyses (cross-checked against the "
        "Jarbo Gap RAWS station), NIST TN 2135 fire-progression observations and an OpenStreetMap drive network onto one "
        "EPSG:32610 grid (30 m, 1,128 x 1,124 cells for the Phase 0 domain; 60 m for the operating teacher), persisted "
        "as GeoTIFF and GeoParquet with a written data card."))
    a(bullet(("b", "Physics Teacher (wf052.sim): "),
        "A calibrated 60 m cellular-automaton model (PyTorchFire contact spread, p_base = 0.22) with a switchable "
        "non-local ember-spotting kernel (alpha_launch = 0.004, lognormal landing distance capped at 6,000 m, "
        "p_urban_wui = 0.15), frozen as configs/teacher_v2.yaml under a SHA-256 hash; calibrated on Kincade, Glass and "
        "Zogg and never tuned on the Camp Fire."))
    a(bullet(("b", "Wind-Conditioned Surrogate (wf052.surrogate): "),
        "A 4,094,852-parameter U-Net that consumes a 13-channel spatial tensor and a 12 x 2 hourly wind series, "
        "conditions every convolutional block on the wind embedding through FiLM, and emits q10 / q50 / q90 arrival "
        "rasters plus a burn logit; trained with a pinball + quantile-ordering + BCE multi-task loss under a trainer "
        "that refuses any split placing one fire in two sets."))
    a(bullet(("b", "Risk-Aware Routing (wf052.routing): "),
        "A time-dependent FIFO Dijkstra router constrained by exit_time + buffer <= fire arrival with a binary-search "
        "max-min-margin fallback, a point-queue / BPR congestion and demand model, a re-plan engine, an independent "
        "route validator and a hazard-miss scorer, all reading only the ArrivalField contract."))
    a(bullet(("b", "Agent, Replay Harness & Dashboard (wf052.agent, wf052.replay, web/): "),
        "A LangGraph EvacuationAgent whose typed tools compute every route and number and whose briefs pass a grounding "
        "checker; a causal replay harness scoring seven methods against NIST hazards; and a single-file offline HTML "
        "dashboard, INCIDENT-052, that replays the precomputed results."))

    # ---- Pre-requisites
    a(h1("Pre-requisites:"))
    a(bullet(("b", "Python Environment: "),
        "Python 3.11 (pyproject.toml pins >=3.11,<3.12) with PyTorch 2.14.1 as the only deep learning framework (no "
        "TensorFlow or Keras), managed with uv.", run(" "),
        links(("Python Documentation", "https://docs.python.org/3/"),
              ("PyTorch Documentation", "https://pytorch.org/docs/stable/index.html"))))
    a(bullet(("b", "Geospatial & Data Libraries: "),
        "NumPy, pandas, Rasterio, GeoPandas, Shapely, PyProj, PyArrow (GeoParquet), Zarr, NetworkX, Pydantic and PyYAML "
        "(run configurations in configs/*.yaml); OSMnx and Herbie are used by the road and HRRR ingestion modules. Docs: ",
        links(("NumPy", "https://numpy.org/doc/"), ("pandas", "https://pandas.pydata.org/docs/"),
              ("Rasterio", "https://rasterio.readthedocs.io/"), ("GeoPandas", "https://geopandas.org/"),
              ("Shapely", "https://shapely.readthedocs.io/"), ("NetworkX", "https://networkx.org/documentation/stable/"),
              ("PyArrow", "https://arrow.apache.org/docs/python/"))))
    a(bullet(("b", "Simulation Engines: "),
        "PyTorchFire 1.0.3.post2 (the selected differentiable cellular-automaton engine) and Cell2Fire (C++, evaluated "
        "in the Phase 0 spike and deprioritized because it cannot run native Scott & Burgan FBFM40 fuels). Docs: ",
        links(("PyTorchFire", "https://github.com/xiazeyu/PyTorchFire"),
              ("Cell2Fire", "https://github.com/cell2fire/Cell2Fire"))))
    a(bullet(("b", "Agent Orchestration: "),
        "LangGraph (>=0.2.0) for the seven-node evacuation state machine, with a MockLLMClient and a deterministic "
        "template brief so that no external LLM key is required. Docs: ",
        links(("LangGraph Documentation", "https://langchain-ai.github.io/langgraph/"))))
    a(bullet(("b", "Data Sources: "),
        "USGS LANDFIRE 2016 Remap (FBFM40, canopy, topography), NOAA HRRR surface analyses through Herbie, NIST TN 2135 "
        "and TN 2252 for fire-progression and hazard records, GOFER and FEDS satellite perimeters, GeoMAC overflight "
        "perimeters and OpenStreetMap. Docs: ",
        links(("LANDFIRE", "https://landfire.gov/"), ("Herbie", "https://herbie.readthedocs.io/"),
              ("NIST TN 2135", "https://doi.org/10.6028/NIST.TN.2135"), ("OSMnx", "https://osmnx.readthedocs.io/"))))
    a(bullet(("b", "Testing & Linting: "),
        "pytest 9.1.1 (184 test functions across 23 test files, slow tests deselected by default) and Ruff 0.16.10 "
        "(target py311, line length 100). Docs: ",
        links(("pytest Documentation", "https://docs.pytest.org/"),
              ("Ruff Documentation", "https://docs.astral.sh/ruff/"))))
    a(bullet(("b", "Compute: "),
        "Local CPU (Apple M1) for ingestion, graph construction, routing, agent tools and tests; Apple M1 MPS for the "
        "measured latency benchmarks; Kaggle GPU notebooks (notebooks/*.ipynb) for bulk teacher runs and surrogate "
        "training."))
    a(bullet(("b", "Version Control: "),
        "Git, with frozen configurations and the precomputed demo assets under out/demo/ (19 MB, 100% tracked) committed "
        "so the dashboard replays without untracked files. Docs: ",
        links(("Git Documentation", "https://git-scm.com/doc"))))

    # ---- Workflow table
    a(h1("Project Workflow:"))
    a(data_table([2621, 6739], ["Milestone", "Activities & Core Focus"], [
        ["Milestone 1: Data Ingestion, Canonical Grid & Provenance", [
            "Activity 1.1: LANDFIRE fuels & topography on a canonical UTM grid",
            "Activity 1.2: HRRR wind with RAWS cross-check & NIST observation rasterization",
            "Activity 1.3: OpenStreetMap road network & data-card provenance"]],
        ["Milestone 2: Physics Teacher Simulation & Calibration", [
            "Activity 2.1: PyTorchFire engine & FBFM40 fuel-model mapping",
            "Activity 2.2: Leave-one-fire-out calibration & frozen, hashed teacher configs",
            "Activity 2.3: Switchable ember-spotting kernel & assimilation mode"]],
        ["Milestone 3: Wind-Conditioned U-Net Surrogate", [
            "Activity 3.1: Data contract, leakage guards & shard pipeline",
            "Activity 3.2: FiLM U-Net architecture & multi-task quantile loss",
            "Activity 3.3: Base-rate initialization, evaluation & latency benchmark"]],
        ["Milestone 4: Risk-Aware Evacuation Routing", [
            "Activity 4.1: Road graph & ArrivalField contract",
            "Activity 4.2: Time-dependent FIFO Dijkstra with max-min-margin fallback",
            "Activity 4.3: Congestion, demand & independent route validation",
            "Activity 4.4: Re-plan engine & hazard-miss scoring"]],
        ["Milestone 5: Agentic Coordination & Causal Replay", [
            "Activity 5.1: LangGraph EvacuationAgent with typed tools",
            "Activity 5.2: Re-plan policy with hysteresis & grounded briefs",
            "Activity 5.3: Causal replay harness against NIST hazards"]],
        ["Milestone 6: Dashboard, Claims Audit & Testing", [
            "Activity 6.1: Standalone offline dashboard (INCIDENT-052) & Streamlit demo",
            "Activity 6.2: Claims audit, limitations & retractions",
            "Activity 6.3: Test suite, lint gate & reproducibility hashes"]],
    ]))

    S = SNIPPETS

    # ================= Milestone 1
    a(h1("Milestone 1: Data Ingestion, Canonical Grid and Provenance"))
    a(body(
        "Milestone 1 builds the data foundation: every environmental layer, observation and road network is aligned to "
        "one canonical projected grid, persisted in GIS-native formats, and documented in a data card that records "
        "vintage, source and any fallback so that downstream results trace back to verifiable inputs."
    ))
    a(h3("Activity 1.1: LANDFIRE Fuels and Topography on a Canonical Grid"))
    a(body(
        "Step 1: Request Pre-Fire Layers: landfire.py downloads eight LANDFIRE layers (fbfm40, elevation, slope, aspect, "
        "canopy_cover, canopy_height, canopy_base_height, canopy_bulk_density) from the official USGS ImageServer, using "
        "the LF 2016 Remap vintage for fuels and canopy and deliberately avoiding the 2023/2024 products whose burn "
        "scars would leak post-fire information into a 2018 case."
    ))
    a(body(
        "Step 2: Force the Canonical Grid: align_and_save_layer resamples any layer whose shape differs from the "
        "configured grid (nearest-neighbor for the categorical fbfm40, linear otherwise) and writes an LZW-compressed "
        "GeoTIFF in EPSG:32610 (int32 for fuels, float32 for everything else) so that every raster shares one transform."
    ))
    a(source_label("Source Code: src/wf052/data/landfire.py (LANDFIRE_SERVICES and align_and_save_layer, abridged):"))
    a(code_table(S["landfire"]))
    a(h3("Activity 1.2: HRRR Wind with RAWS Cross-Check and NIST Observation Rasterization"))
    a(body(
        "Step 1: Ingest Hourly HRRR Wind: weather.py retrieves 12 hourly HRRR surface analyses (fxx = 0) from "
        "2018-11-08 14:00 UTC through Herbie, averages 10 m U/V wind, gust and 2 m temperature over the domain, converts "
        "them to speed and direction, and cross-checks them against Jarbo Gap RAWS observations (sustained 25-31 mph, "
        "gusts 43-50 mph, direction 35-44 degrees). A synthetic Diablo-wind fallback exists in the code for any hour "
        "whose retrieval fails."
    ))
    a(body(
        "Step 2: Rasterize NIST Progression Points: observations.py parses Appendix F of NIST TN 2135 into roughly 2,800 "
        "timestamped, geolocated points (2,817 per the Phase 0 report, 2,818 per the data card), converts clock times to "
        "minutes after the 06:25 PST first record, reprojects them to UTM 10N, persists them as GeoParquet, and "
        "rasterizes the minimum arrival time per 30 m cell within the 12-hour window."
    ))
    a(note(
        "Note (governance): ",
        "the GOFER hourly satellite-progression dataset was examined and set aside for the Camp Fire because it covers "
        "28 California fires from 2019-2021 and does not include 2018; FEDS 375 m VIIRS detections and GeoMAC infrared "
        "overflight perimeters are used instead, and GOFER is used only for the seven training fires.",
    ))
    a(source_label("Source Code: src/wf052/data/weather.py and observations.py (excerpts):"))
    a(code_table(S["weather_obs"]))
    a(h3("Activity 1.3: OpenStreetMap Road Network and the Data Card"))
    a(body(
        "Step 1: Attempt a Historical Snapshot, Flag the Fallback: roads.py asks the Overpass API for the network as of "
        "2018-11-01 and, if that query fails or times out, falls back to the current OSM network and returns "
        "is_historical = False so the substitution is explicit rather than silent. The Phase 0 attempt timed out, so the "
        "current drive network was used and flagged."
    ))
    a(body(
        "Step 2: Project, Export and Document: the graph is projected to EPSG:32610, list-valued attributes are "
        "flattened for Arrow compatibility, and nodes and edges are written as GeoParquet; reports/data_card.md records "
        "the domain, vintage, source, license and flags of each of its six layers."
    ))
    a(source_label("Source Code: src/wf052/data/roads.py (fetch_road_graph, abridged):"))
    a(code_table(S["roads"]))

    # ================= Milestone 2
    a(h1("Milestone 2: Physics Teacher Simulation, Calibration and Ember Spotting"))
    a(body(
        "Milestone 2 turns the raw layers into a trustworthy physics teacher: a PyTorch cellular-automaton engine, a "
        "leave-one-fire-out calibration that freezes a single configuration before the Camp Fire is touched, and a "
        "switchable ember-spotting kernel that addresses the long-range spotting the contact model cannot reproduce."
    ))
    a(h3("Activity 2.1: PyTorchFire Engine and Fuel-Model Mapping"))
    a(body(
        "Step 1: Map Fuels to Spread Probabilities: pytorchfire_engine.py maps each Scott & Burgan FBFM40 code to "
        "PyTorchFire's p_veg: non-burnable classes (91 Developed, 92 Snow/Ice, 93 Agriculture, 98 Water, 99 Barren) get "
        "-1.0 so the automaton cannot spread through them, while grass, shrub, timber-understory, litter and slash models "
        "map to 0.15-0.60 by fuel-bed depth, load and spread rate; canopy cover becomes a density factor p_den in "
        "[-0.2, +0.2]."
    ))
    a(body(
        "Step 2: Benchmark Two Engines: a 10-seed Phase 0 benchmark ran PyTorchFire and Cell2Fire on the Camp Fire "
        "domain. PyTorchFire completed the 12-hour run on 1.27 million cells in 47.93 s (mean of 10 seeds), whereas "
        "Cell2Fire burned only 364 cells before contact spread stalled at non-burnable barriers. Phase 1A then "
        "re-audited the perimeter timeline, finding that GeoMAC epoch timestamps are local Pacific time, which places "
        "the first infrared overflight at +11.4 h (684 min) after ignition."
    ))
    a(source_label("Source Code: src/wf052/sim/pytorchfire_engine.py (FBFM40_TO_P_VEG and p_den mapping, abridged):"))
    a(code_table(S["fuel_map"]))
    a(h3("Activity 2.2: Leave-One-Fire-Out Calibration and Frozen Teacher Configurations"))
    a(body(
        "Step 1: Calibrate Without Touching the Camp Fire: Phase 1A standardizes seven training fires (Kincade, Bobcat, "
        "Glass, Creek, Zogg, Caldor, Tamarack) onto a 720 x 720, 60 m grid with one LANDFIRE 2016 vintage and real HRRR "
        "wind, runs an 82-simulation one-at-a-time sensitivity sweep, and compares an uncalibrated model, a 1-parameter "
        "global slowdown, a 6-parameter fit and a wind-only fit under leave-one-fire-out validation, keeping the "
        "simplest model within 0.02 IoU of the best (+12 h IoU 0.2436 for the 1-parameter model versus 0.1036 "
        "uncalibrated)."
    ))
    a(body(
        "Step 2: Freeze and Hash: configs/teacher_v1.yaml (p_base = 0.21) and teacher_v1_1.yaml (p_base = 0.22) are "
        "written to disk with SHA-256 hashes recorded before any Camp Fire evaluation; configs/teacher_v2.yaml adds the "
        "ember kernel and is verified against hash a6a92f81...2b4e85a5, with a bit-identical copy kept for the Kaggle "
        "notebooks."
    ))
    a(source_label("Source Code: configs/teacher_v2.yaml (frozen contact and ember parameters, abridged):"))
    a(code_table(S["teacher_v2"]))
    a(h3("Activity 2.3: Switchable Ember Spotting and Assimilation Mode"))
    a(body(
        "Step 1: Add Non-Local Spotting With Three Free Parameters: ember_kernel.py samples per-cell launches "
        "(alpha_launch scaled by a fuel spotting potential and by wind above a 4 m/s lofting threshold), lands them "
        "downwind with a lognormal distance (median beta_dist x wind speed, sigma 0.65, capped at 6,000 m) and 18-degree "
        "lateral scatter, and ignites receptive cells. A static-shape scatter_add_ avoids CPU-GPU synchronization inside "
        "the time-step loop, and the three parameters are fitted only on the wind-driven fires Kincade, Glass and Zogg."
    ))
    a(body(
        "Step 2: Attribute the Gain and Test Assimilation: embers account for 75.5% of the simulated burned-area gain and "
        "87.0% of the IoU gain from teacher_v1 to teacher_v2 over 20 seeds (Camp Fire median IoU 0.3147 against the "
        "22,158 ha GeoMAC perimeter at +11.4 h), and an assimilation mode initializes the fire from an observed extent "
        "at T0 (+1, +2 or +3 h) to forecast the next three hours."
    ))
    a(source_label("Source Code: src/wf052/sim/ember_kernel.py (sample_ember_spotting_step, abridged):"))
    a(code_table(S["ember"]))

    # ================= Milestone 3
    a(h1("Milestone 3: Wind-Conditioned U-Net Surrogate"))
    a(body(
        "Milestone 3 distills the teacher into a fast neural emulator. The pipeline is built and verified on stand-in "
        "and dry-run data under a strict data contract, so every reported number measures fidelity to the teacher "
        "rather than to ground truth, and no trained-surrogate accuracy is claimed in this version."
    ))
    a(h3("Activity 3.1: Data Contract, Leakage Guards and Shard Pipeline"))
    a(body(
        "Step 1: Fix the Tensor Contract: each sample is a (13, 256, 256) spatial tensor — seven one-hot Scott & Burgan "
        "fuel groups (NB, GR, GS, SH, TU, TL, SB), slope, sin and cos of aspect, z-scored elevation, normalized canopy "
        "cover and a binary ignition mask — plus a (12, 2) hourly wind series, with arrival time in minutes [0, 720] and "
        "a censor mask as targets. The 62-sample dry-run shard splits by whole fire: 34 train (kincade, glass, zogg, "
        "creek), 10 validation (bobcat) and 18 test (caldor, tamarack)."
    ))
    a(body(
        "Step 2: Make Leakage Impossible to Do Silently: the generator raises ProhibitedFireError for the Camp Fire and "
        "the three fresh held-out fires (Slater and Devil, Walker, W-5 Cold Springs), the trainer raises a ValueError if "
        "any fire appears in more than one split, and analytic wind-ellipse samples are flagged baseline_only and "
        "rejected as training data."
    ))
    a(source_label("Source Code: src/wf052/surrogate/standin_generator.py and trainer.py (leakage guards, abridged):"))
    a(code_table(S["leak_guard"]))
    a(h3("Activity 3.2: FiLM U-Net Architecture and Multi-Task Quantile Loss"))
    a(body(
        "Step 1: Condition Every Block on Wind: a 1D-convolution + MLP WindEncoder turns the 12-hour wind series into a "
        "64-dimensional embedding; each residual FiLMBlock projects it to (gamma, beta) and applies (1 + gamma) x "
        "GroupNorm(h) + beta, with the projection zero-initialized so training starts from identity modulation. Three "
        "encoder stages, two bottleneck blocks and three decoder stages with skip connections total 4,094,852 trainable "
        "parameters, inside the 2-6 M design band."
    ))
    a(body(
        "Step 2: Bound the Outputs and Train Three Tasks: quantile arrival times are a sigmoid scaled to [0, 720] "
        "minutes, the burn head emits unconstrained logits, and the loss is the mean of pinball losses at tau = 0.1, 0.5 "
        "and 0.9 on uncensored cells plus a non-crossing ordering penalty (lambda = 1.0) and a binary cross-entropy burn "
        "term (lambda = 0.5)."
    ))
    a(source_label("Source Code: src/wf052/surrogate/model.py (FiLMBlock.forward, abridged):"))
    a(code_table(S["film"]))
    a(source_label("Source Code: src/wf052/surrogate/losses.py (SurrogateMultiTaskLoss.forward, abridged):"))
    a(code_table(S["loss"]))
    a(h3("Activity 3.3: Base-Rate Initialization, Evaluation and Latency Benchmark"))
    a(body(
        "Step 1: Start the Burn Head at the Training Base Rate: init_burn_prob_bias sets the final bias of the "
        "burn-probability head to logit(clip(mean training burn rate, 1e-4, 0.5)), computed from training shards only, "
        "so the head begins at the observed base rate instead of 0.5; evaluation reports per-bin quantile error, Brier "
        "score and IoU per split and per fire."
    ))
    a(body(
        "Step 2: Measure Latency, Not Accuracy: a 100-run benchmark on Apple M1 MPS measures 34.51 ms per 256 x 256 tile "
        "(p50 34.43 ms, p95 35.42 ms) and 310.59 ms across the nine tiles that cover the Camp Fire domain, against a "
        "2,747.21 ms mean for the full teacher simulation over five seeds. Because the checkpoints are 2-epoch stand-in "
        "or 1-epoch dry-run weights these are inference-time figures only, and the earlier 24.1 ms and 56.4x narrative "
        "figures were retracted as unmeasured estimates."
    ))
    a(source_label("Source Code: src/wf052/surrogate/model.py (output heads and init_burn_prob_bias, abridged):"))
    a(code_table(S["heads"]))

    # ================= Milestone 4
    a(h1("Milestone 4: Risk-Aware Evacuation Routing"))
    a(body(
        "Milestone 4 converts a fire-arrival field into evacuation decisions. A road graph, a strict ArrivalField "
        "contract, a time-dependent router with a safety-margin guarantee, a congestion model and an independent "
        "validator are built so that every route can be checked against the fire field without trusting the planner."
    ))
    a(h3("Activity 4.1: Road Graph and the ArrivalField Contract"))
    a(body(
        "Step 1: Define One Input Contract: the router reads only an xarray Dataset holding arrival_q10, arrival_q50 and "
        "arrival_q90 rasters (minutes since ignition, +inf where the fire never arrives) with y / x coordinates and crs "
        "and ignition_time attributes; validate_arrival_field rejects anything else, so teacher outputs, surrogate "
        "outputs (via surrogate_output_to_arrival_field and stitch_tiles_to_arrival_field) and synthetic fire fields are "
        "interchangeable."
    ))
    a(body(
        "Step 2: Build the Graph: graph.py builds the drive network with per-edge traversal times and projects the field "
        "onto edges; the full rebuilt graph used in the replay has 14,691 edges, with named origin landmarks and "
        "boundary destination exits grouped into three egress corridors (Skyway, Clark Rd, Pentz/Hwy 70)."
    ))
    a(source_label("Source Code: src/wf052/routing/contracts.py (routing modes and contract validation, abridged):"))
    a(code_table(S["route_contract"]))
    a(h3("Activity 4.2: Time-Dependent FIFO Dijkstra with a Max-Min-Margin Fallback"))
    a(body(
        "Step 1: Enforce the Safety Condition on Every Edge: _run_constrained_dijkstra relaxes an edge only if "
        "exit_time + buffer <= the edge's fire-arrival time, with exit_time = entry_time + traversal time (strict FIFO). "
        "The default buffer is 15 minutes, and traversal times can be replaced by congested times and by destination "
        "exit-queue delays."
    ))
    a(body(
        "Step 2: Fall Back to the Safest Feasible Route: if no route meets the buffer, find_route first checks physical "
        "reachability with the fire constraint removed, then binary-searches the achievable minimum margin between "
        "-1,440 minutes and the buffer (16 iterations, precision under 0.05 min) and returns a BEST_EFFORT route, or "
        "None when the graph is disconnected."
    ))
    a(source_label("Source Code: src/wf052/routing/router.py (edge relaxation and Mode b search, abridged):"))
    a(code_table(S["dijkstra"]))
    a(h3("Activity 4.3: Congestion, Demand and Independent Validation"))
    a(body(
        "Step 1: Model Capacity and Delay Without Double Counting: congestion.py assigns each edge a Highway Capacity "
        "Manual planning capacity by highway class and lane count (scaled, minus inbound emergency traffic on Skyway, "
        "Clark, Neal, Pentz and Hwy 70) and a physical storage of length x lanes x jam density (130 veh/km/lane by "
        "default). Delay is either a point-queue bottleneck or a BPR volume-delay function (alpha 0.15, beta 4.0), never "
        "both on one edge, with 5-minute time-sliced assignment, finite boundary exit outflow and entrapment metrics."
    ))
    a(body(
        "Step 2: Re-Check Every Route Independently: ReplanEngine.replan runs IndependentRouteValidator on every returned "
        "route against the original arrival field and raises an AssertionError if any route fails; the measured "
        "end-to-end cost is 2,115.74 ms for all 200 origins (10.58 ms per origin) per 30-minute re-plan event."
    ))
    a(source_label("Source Code: src/wf052/routing/congestion.py (edge capacity and storage, abridged):"))
    a(code_table(S["congestion"]))
    a(h3("Activity 4.4: Re-Plan Engine and Hazard-Miss Scoring"))
    a(body(
        "Step 1: Re-Plan Against a New Field: replan() validates the field, hashes arrival_q10 with SHA-256 for "
        "provenance, recomputes edge fire arrivals once, resolves origin and destination names, optionally drops "
        "destinations the fire reaches within a safety buffer, and routes every origin to its best reachable exit."
    ))
    a(body(
        "Step 2: Score Routes Against Ground-Truth Hazards: score_route_against_hazards skips any edge the vehicle "
        "exits before a hazard's onset time and counts a hazard as encountered when an edge passes within the hazard "
        "radius (60 m by default), which is how the replay turns routes into burnover, roadblock and abandoned-vehicle "
        "encounters."
    ))
    a(source_label("Source Code: src/wf052/routing/replan.py and hazard_check.py (excerpts):"))
    a(code_table(S["replan_hazard"]))

    # ================= Milestone 5
    a(h1("Milestone 5: Agentic Coordination and Causal Replay"))
    a(body(
        "Milestone 5 wraps the routing stack in an agent that may decide and narrate but never calculate, and then "
        "evaluates that agent under a causal replay that cannot peek at the future."
    ))
    a(h3("Activity 5.1: LangGraph EvacuationAgent with Typed Tools"))
    a(body(
        "Step 1: Run a Fixed Seven-Node State Machine: EvacuationAgent compiles a LangGraph StateGraph — ingest_update, "
        "get_arrival_field, plan_routes, validate_routes, compute_exposure, decide_replan and brief — in which every "
        "numeric result comes from plain typed tools in tools.py (get_weather, get_arrival_field, plan_routes, "
        "validate_routes, compute_exposure, compute_k_star); the module rules state that the LLM never computes routes, "
        "geometry or numbers."
    ))
    a(body(
        "Step 2: Fall Back When the Field Is Not Credible: check_field_vs_observations flags an arrival field as "
        "low-confidence when a detection arrives more than 15 minutes earlier than arrival_q10 or lies where the field "
        "predicts no arrival, and the planner then switches to conservative settings (a 30-minute buffer)."
    ))
    a(source_label("Source Code: src/wf052/agent/graph.py (EvacuationAgent._build_graph):"))
    a(code_table(S["agent_graph"]))
    a(h3("Activity 5.2: Re-Plan Policy with Hysteresis and Grounded Briefs"))
    a(body(
        "Step 1: Re-Plan With Hysteresis: ReplanPolicy.decide re-plans when the largest arrival change on an active route "
        "edge or the domain-mean arrival shift reaches 15 minutes, when confidence is low, when a road on an active route "
        "is cut, or when a route's margin under the new field falls below the buffer; a different physical path replaces "
        "the current one only if it gains at least 5 minutes of margin or saves at least 10% travel time, so near-equal "
        "options do not flap."
    ))
    a(body(
        "Step 2: Reject Hallucinated Briefs: GroundingChecker extracts every allowed number, road name and zone name from "
        "tool outputs and rejects a brief that states an unsupported figure, an unknown road or zone, or any quantitative "
        "clearance time (CLEARANCE_HOURS_PATTERN). On rejection the agent regenerates once with feedback and then falls "
        "back to a deterministic template brief, which is what the dashboard shows (\"LLM NOT RUN\")."
    ))
    a(source_label("Source Code: src/wf052/agent/policy.py and grounding.py (excerpts):"))
    a(code_table(S["policy_grounding"]))
    a(h3("Activity 5.3: Causal Replay Harness Against NIST Hazards"))
    a(body(
        "Step 1: Hide the Future: CausalReplayHarness.step(t) returns only observations with timestamp <= t - latency and "
        "roadblocks whose onset <= t. NIST observation clocks are shifted by 8 minutes because elapsed_minutes = 0 is the "
        "06:25 PST first dispatch record while ignition was 06:33 PST; the offset was specified a priori from NIST TN "
        "2252 and not tuned post hoc."
    ))
    a(body(
        "Step 2: Compare Seven Methods and Report the Null Result: seven comparative methods (A1, A2, A2-assim, A3, B, "
        "static and live) route 200 sampled block origins. A plain bootstrap suggests nominal gains over the static "
        "baseline (for example -13.5% combined hazard misses for Agent B on A2), but zero comparisons survive a cluster "
        "bootstrap by static egress corridor (K = 3), whose intervals are 1.8x to 2.3x wider; A2 and B-on-A2 results are "
        "identical across five teacher seeds spanning 2,437 to 25,683 ha."
    ))
    a(source_label("Source Code: src/wf052/replay/harness.py (causal filtering and clock alignment, abridged):"))
    a(code_table(S["harness"]))

    # ================= Milestone 6
    a(h1("Milestone 6: Offline Dashboard, Claims Audit and Testing"))
    a(body(
        "Milestone 6 packages the results for inspection and keeps them honest: an offline dashboard that cannot make "
        "network calls, a claims-and-limitations audit that retracts unsupported statements, and a test and "
        "reproducibility layer."
    ))
    a(h3("Activity 6.1: Standalone Offline Dashboard (INCIDENT-052) and Streamlit Demo"))
    a(body(
        "Step 1: Inline Everything Into One File: scripts/build_web_demo.py reads the precomputed assets in out/demo/ "
        "(routes.json, hazards.geojson, updates.geojson, zones.geojson, isochrones and perimeters), simplifies geometries "
        "with Douglas-Peucker (0.0001 degrees, about 11 m, coordinates rounded to 5 decimals), and inlines the CSS tokens, "
        "ES6 logic, bundled Latin WOFF2 fonts and JSON payloads into web/dist/index.html (5.85 MB, under a 10 MB limit)."
    ))
    a(body(
        "Step 2: Open It or Serve It: the file opens by double-click or through scripts/serve_web.py on "
        "http://localhost:8000, and app/demo.py provides an equivalent Streamlit replay (PYTHONPATH=src streamlit run "
        "app/demo.py). Routes are precomputed across the seven methods — the dashboard never runs Dijkstra or A* live, "
        "and its Re-plan button only redraws the stored route cards."
    ))
    a(source_label("Source Code: scripts/serve_web.py (local server for web/dist, abridged):"))
    a(code_table(S["serve_web"]))
    a(h3("Activity 6.2: Claims Audit, Limitations and Retractions"))
    a(body(
        "Step 1: State What the System Can and Cannot Claim: REPORT.md and the dashboard's Status & Limitations tab list "
        "four supported claims (inference time of an untrained architecture, interactive re-planning, inconclusive "
        "burnover margins, burnover timing error in the teacher) and four unsupported ones (no safety guarantees, no "
        "superiority of A2-assim in overall margins, no independent real-time ground truth, no production readiness from "
        "stand-in weights)."
    ))
    a(body(
        "Step 2: Retract Rather Than Defend: thirteen audited limitations are recorded, including post-event HRRR "
        "analysis instead of a forecast, NIST-reconstructed hazard timestamps, K = 3 corridor clusters, a teacher that "
        "under-burns the Camp Fire by about 2x yet over-burns most training fires, shared provenance between assimilated "
        "observations and evaluation hazards, and the retraction of the 24.1 ms forward-pass and 56.4x speedup figures "
        "as unmeasured estimates."
    ))
    a(h3("Activity 6.3: Test Suite, Lint Gate and Reproducibility Hashes"))
    a(body(
        "Step 1: Test Every Phase as It Landed: the repository carries 184 test functions in 23 files — surrogate "
        "contract, model and evaluation tests per phase (94), router, congestion and k* slowdown tests (32), the agent "
        "pipeline (31), the causal replay harness (12), ember, engine and calibrator tests (11), scaffold tests (3) and a "
        "headless demo test (1) — with slow tests deselected by default."
    ))
    a(body(
        "Step 2: Hash What Was Frozen: README.md and REPORT.md list SHA-256 checksums for teacher_v2.yaml, the 62-sample "
        "dry-run shard, the dry-run and stand-in checkpoints, the NIST hazards GeoJSON, routes.json and metadata.json, "
        "every claim in the dashboard cites the CSV and row it came from, and Ruff (py311, line length 100) gates style."
    ))
    a(source_label("Source Code: pyproject.toml (pytest and Ruff configuration, abridged):"))
    a(code_table(S["pytest_ruff"]))

    # ---- Important notes
    a(h1("Important Notes:"))
    a(bullet(("b", "Non-Operational Research Prototype: "),
        "this software must never be used for real-time emergency navigation or life-safety decisions; it does not "
        "certify safe evacuation routes, and every dashboard view carries a persistent research-only strip."))
    a(bullet(("b", "Stand-In Surrogate Weights, Not a Trained Result: "),
        "every surrogate (A3) field and route shown comes from a 2-epoch stand-in checkpoint trained on 66 smoke shards "
        "or a 1-epoch dry run; the stand-in A3-reignite tile burned 85.5% of its area, so no trained-surrogate result is "
        "reported and no fidelity claim is made."))
    a(bullet(("b", "Camp Fire Is a Development Case, Never Training Data: "),
        "it is excluded from all surrogate training and tuning (only the criticality sweep and the assimilation test "
        "mode use it), and the teacher under-burns it by about 2x while over-burning most training fires."))
    a(bullet(("b", "Honest Null Result on Routing Benefit: "),
        "no comparison against the static baseline survives the corridor-clustered bootstrap, and A2-assim does not "
        "eliminate negative margins — its median truth margin over all 436 NIST hazards is -37.0 min versus -31.5 min "
        "for A0 static (cause not tested)."))
    a(bullet(("b", "Modelled, Not Observed, Abandoned-Vehicle Onset: "),
        "the 235 abandoned vehicles are given a constant 87.0-minute onset (08:00 PST) in harness.py; the input file has "
        "no timestamp column, so this is a harness assumption and every related figure is labelled as modelled."))
    a(bullet(("b", "Vacuous Predicted Margins: "),
        "evacuation vehicles finish before the simulated fire reaches Paradise (smallest positive predicted margin "
        "149.8 min), which is why the dashboard labels such cells \"No predicted fire arrival on route\" instead of "
        "presenting them as safety margins."))
    a(bullet(("b", "Documentation Discrepancy (Unresolved): "),
        "reports/data_card.md lists 3,396 road nodes, 7,583 edges and 2,818 NIST points, while reports/phase_0.md lists "
        "3,627 nodes, 8,085 edges and 2,817 points; both figures were found verbatim in the current repository and the "
        "discrepancy has not been reconciled as of this documentation pass."))

    # ---- Web application
    a(h1("Exploring the Web Application:"))
    a(body(
        "The INCIDENT-052 Wildfire Spread & Evacuation Routing dashboard (web/index.html, web/app.js, web/style.css, "
        "built into web/dist/index.html) is opened by double-clicking web/dist/index.html or by running \"python "
        "scripts/serve_web.py\" and visiting http://localhost:8000. It organizes around four tabs — Tactical Command "
        "Console, Evacuation Zones (14 Zones), System Latency & Brief, and Status & Limitations — with a persistent "
        "disclaimer strip and a replay timeline, described below."
    ))
    a(h1("Tactical Command Console and Route Evaluation Views"))
    a(body(
        "The Tactical Command Console renders a GIS canvas in Web Mercator (EPSG:3857) with the observed +11.4 h GeoMAC "
        "perimeter, NIST burnover, roadblock and abandoned-vehicle markers and the evacuation-zone outlines. A layer "
        "panel toggles the predicted fire perimeter, 1-6 h arrival isochrones, hazards, routes and 30-minute update "
        "markers and switches the field source between Surrogate and Teacher v2, while a routing-algorithm legend toggles "
        "A3 surrogate stand-in, B on A3, A2-assim, A2-pure and A0 static. Selecting one of 193 origin blocks (dropdown or "
        "Previous / Next Origin) fills the Route Evaluation Summary: each method's card reports route length from path "
        "geometry, the predicted margin on the agent's own field, whether the route hits a NIST hazard, and \"Margin to "
        "NIST burnovers: NOT AVAILABLE\", citing the source line (src/wf052/routing/router.py:174)."
    ))
    a(image(_rid("image", "media/nav_console.png"), "nav_console",
            "Figure: INCIDENT-052 Dashboard — Tactical Command Console view", 302, 6.0, 3.75))
    a(caption("Figure: INCIDENT-052 Dashboard — Tactical Command Console view"))
    a(image(_rid("image", "media/nav_route_evaluation.png"), "nav_route_evaluation",
            "Figure: INCIDENT-052 Dashboard — Route Evaluation view (Teacher v2 field source)", 303, 6.0, 3.75))
    a(caption("Figure: INCIDENT-052 Dashboard — Route Evaluation view (Teacher v2 field source)"))
    a(h1("Evacuation Zones, System Latency and Status & Limitations"))
    a(body(
        "The Evacuation Zones tab lists 14 evacuation zones and 3 egress corridors (17 sortable rows) from "
        "reports/step5d_per_zone_metrics.csv with origin counts, route changes versus the static baseline, P10 margin on "
        "the A3 field and trapped counts — for example Clark_Rd covers 93 origins with 63 route changes (67.7%), a 49.6 "
        "min P10 margin and 1 trapped origin — under a caveat that the stand-in A3 field was used. The System Latency & "
        "Brief tab shows three cards (2,115.74 ms router plus validator per re-plan across 200 origins, 34.51 ms per "
        "surrogate tile, 2,747.21 ms mean Teacher v2 simulation), each citing its CSV row, above a deterministic template "
        "Evacuation Operations Brief badged \"LLM NOT RUN\". The Status & Limitations tab tabulates specification "
        "metrics and splits what the system can claim from what it cannot, followed by the audited limitations."
    ))
    a(image(_rid("image", "media/nav_zones.png"), "nav_zones",
            "Figure: INCIDENT-052 Dashboard — Evacuation Zones view", 304, 6.0, 3.75))
    a(caption("Figure: INCIDENT-052 Dashboard — Evacuation Zones view"))
    a(image(_rid("image", "media/nav_latency_brief.png"), "nav_latency_brief",
            "Figure: INCIDENT-052 Dashboard — System Latency & Brief view", 305, 6.0, 3.75))
    a(caption("Figure: INCIDENT-052 Dashboard — System Latency & Brief view"))
    a(image(_rid("image", "media/nav_status.png"), "nav_status",
            "Figure: INCIDENT-052 Dashboard — Status & Limitations view", 306, 6.0, 3.75))
    a(caption("Figure: INCIDENT-052 Dashboard — Status & Limitations view"))
    a(h1("Navigation: Replay Timeline and the Research-Only Disclaimer"))
    a(body(
        "A persistent transport strip drives a 0-360 minute replay timeline (06:33 PST ignition to 12:33 PST) shared by "
        "every tab: Play auto-advances 10 minutes every 1.2 seconds, the previous-10 and next-10 buttons and the Left / "
        "Right arrow keys step 10 minutes, the Space key toggles playback, and a 30-minute re-plan interval is annotated "
        "on the slider (RE-PLAN #5 ACTIVE at T+120). A fixed strip across the top of every view states the stand-in "
        "weights, the Camp Fire development-case status, \"RESEARCH ONLY: not for evacuation or life-safety decisions\" "
        "and the modelled abandoned-vehicle onset, carrying the README and REPORT.md safety framing into the interactive "
        "surface, while the footer shows the build tag, commit and configuration hash."
    ))

    # ---- Time table
    a(h1("Time Taken at Different Project Stages"))
    a(body(
        "The timeline below is reconstructed directly from the repository's own commit history (git log), in keeping "
        "with a preference for traceable, verifiable records over a separately kept time log. All times are IST "
        f"(UTC+05:30) and the repository holds {n_commits} commits. The first commit contains only the repository "
        "scaffold, so it is reported as a single foundational commit with no prior commit to measure elapsed time "
        "against; every stage after it is timestamped directly from individual commits."
    ))
    a(data_table([4300, 2900, 2160], ["Stage", "Window", "Elapsed"], timeline_rows))
    gap_days = gap.days
    gap_hours = gap.seconds // 3600
    a(note(
        "Note: ",
        "\"Window\" and \"Elapsed\" are wall-clock spans between commits, not continuous active-coding time — the "
        f"~{gap_days}d {gap_hours}h gap between {gap_a[0]} ({gap_a[1].strftime('%d %b %H:%M')}) and {gap_b[0]} "
        f"({gap_b[1].strftime('%d %b %H:%M')}) is the longest in the history and mostly reflects time away from the "
        "repository, and the bulk surrogate training and simulation runs executed off-repository on Kaggle GPUs are not "
        "captured by these timestamps."
    ))

    # ---- Conclusion
    a(h1("Conclusion", page_break=True))
    a(body(
        "This project investigates how far a physics-grounded, tool-constrained pipeline can go in turning a wildfire "
        "spread forecast into auditable evacuation routing for a real, well-documented event. It delivers a calibrated "
        "and hash-frozen physics teacher whose ember kernel accounts for roughly three quarters of its burned-area gain; "
        "a 4.09 M-parameter wind-conditioned U-Net that runs a 256 x 256 tile in 34.51 ms (architecture verified, weights "
        "untrained); a time-dependent router that re-plans 200 origins in 2.1 s with an independently validated safety "
        "margin; and an agent whose briefs are grounded against tool outputs and presented through an offline dashboard. "
        "Just as importantly, it reports what it could not show: no trained-surrogate fidelity, no routing benefit that "
        "survives corridor-clustered bootstrap, and no safety claim. The next steps follow directly from those gaps — "
        "full-data surrogate training on GPU, evaluation on the fresh held-out fires (Slater and Devil, Walker and "
        "W-5 Cold Springs), and validation of the teacher field against road impassability on the western exit corridors."
    ))
    return "".join(X)


# ---------------------------------------------------------------- package assembly
def assemble():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(TEMPLATE) as zin:
        doc = zin.read("word/document.xml").decode("utf-8")
        core = zin.read("docProps/core.xml").decode("utf-8")
        names = zin.namelist()
        files = {n: zin.read(n) for n in names}

    head, rest = doc.split("<w:body>", 1)
    sect = re.search(r"<w:sectPr>.*?</w:sectPr>", rest, re.S).group(0)
    new_doc = head + "<w:body>" + build_body() + sect + "</w:body></w:document>"

    # relationships: keep the sample's fixed parts (rId1-rId8), then append ours
    fixed = []
    for m in re.finditer(r"<Relationship [^>]*/>", files["word/_rels/document.xml.rels"].decode("utf-8")):
        rid = re.search(r'Id="(rId\d+)"', m.group(0)).group(1)
        if int(rid[3:]) <= 8:
            fixed.append(m.group(0))
    RT = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    dyn = []
    for rid, kind, target in _rels:
        if kind == "hyperlink":
            dyn.append(f'<Relationship Id="{rid}" Type="{RT}/hyperlink" Target="{target}" TargetMode="External"/>')
        else:
            dyn.append(f'<Relationship Id="{rid}" Type="{RT}/image" Target="{target}"/>')
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(fixed) + "".join(dyn) + "</Relationships>"
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    core = re.sub(r"<dc:creator>.*?</dc:creator>", "<dc:creator>Mitesh Singh</dc:creator>", core)
    core = re.sub(r"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>", "<cp:lastModifiedBy>Mitesh Singh</cp:lastModifiedBy>", core)
    core = re.sub(r"(<dcterms:created[^>]*>).*?(</dcterms:created>)", rf"\g<1>{now}\g<2>", core)
    core = re.sub(r"(<dcterms:modified[^>]*>).*?(</dcterms:modified>)", rf"\g<1>{now}\g<2>", core)
    core = core.replace(
        "<dc:creator>",
        "<dc:title>Wildfire Spread Simulation and Agentic Evacuation Route Planner (wf052) — Project Documentation</dc:title><dc:creator>",
        1,
    )

    media = {
        "word/media/arch_flowchart.png": FLOWCHART.read_bytes(),
        "word/media/nav_console.png": (FRAMED / "nav_console.png").read_bytes(),
        "word/media/nav_route_evaluation.png": (FRAMED / "nav_route_evaluation.png").read_bytes(),
        "word/media/nav_zones.png": (FRAMED / "nav_zones.png").read_bytes(),
        "word/media/nav_latency_brief.png": (FRAMED / "nav_latency_brief.png").read_bytes(),
        "word/media/nav_status.png": (FRAMED / "nav_status.png").read_bytes(),
    }

    old_media = {n for n in names if n.startswith("word/media/")}
    keep_logos = {
        "word/media/58309ee97eee869f9f05bdc53b047a687d142f7b.png",
        "word/media/9d2522f1d247c37d592a1cd06ff3f72c814c2c4c.png",
    }
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:
            if n in old_media and n not in keep_logos:
                continue
            if n == "word/document.xml":
                zout.writestr(n, new_doc.encode("utf-8"))
            elif n == "word/_rels/document.xml.rels":
                zout.writestr(n, rels_xml.encode("utf-8"))
            elif n == "docProps/core.xml":
                zout.writestr(n, core.encode("utf-8"))
            else:
                zout.writestr(n, files[n])
        for n, data in media.items():
            zout.writestr(n, data)
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    assemble()
