# wf052: Wildfire Spread Simulation & Agentic Evacuation Route Planner

A research demo combining physics-based wildfire spread simulation, deep surrogate emulation in PyTorch, risk-aware evacuation route planning, and tool-augmented agentic coordination.

---

## Architecture Overview

```
                                +---------------------------+
                                |  Run Configuration (YAML) |
                                +-------------+-------------+
                                              |
                   +--------------------------+--------------------------+
                   |                          |                          |
                   v                          v                          v
      +------------------------+  +------------------------+  +------------------------+
      |      wf052.data        |  |        wf052.sim       |  |     wf052.surrogate    |
      | - Ingestion            |  | - Physics Teacher Sim  |  | - PyTorch Emulators    |
      | - GeoTIFF / Zarr Grid  |  | - Cellular Automata /  |  | - Fast inference       |
      | - GeoParquet Vectors   |  |   Rothermel Model      |  |                        |
      +------------------------+  +-----------+------------+  +-----------+------------+
                                              |                           |
                                              +-------------+-------------+
                                                            v
                                              +------------------------+
                                              |      wf052.routing     |
                                              | - Risk-aware graphs    |
                                              | - Dynamic Dijkstra/A*  |
                                              +-----------+------------+
                                                          |
                                                          v
                                              +------------------------+
                                              |       wf052.agent      |
                                              | - Tool-only calling    |
                                              | - Evacuation planner   |
                                              +------------------------+
```

### Module Separation & Principles
1. **Physics Teacher (`wf052.sim`)**: High-fidelity cellular / differential fire behavior model with typed boundaries.
2. **ML Surrogate (`wf052.surrogate`)**: PyTorch-based neural emulator trained to reproduce teacher fire front dynamics at low latency.
3. **Routing (`wf052.routing`)**: Geometric and topological path planning under time-dependent fire risk surfaces.
4. **Agent Orchestration (`wf052.agent`)**: LLM-driven evacuation advisor that **never computes geometry or routes directly**, but strictly queries typed tools.
5. **Raster & Vector Convention**:
   - All spatial layers aligned to a canonical grid defined per run config (`CRS`, `resolution`, `bbox`).
   - Rasters persisted as GeoTIFF or Zarr.
   - Vector data persisted as GeoParquet.

---

## Repository Structure

```
wf052/
├── configs/            # Run configurations (one YAML per experiment)
├── notebooks/          # Research and Kaggle GPU paste-ready notebooks
├── reports/            # Phase evaluation reports and metrics logs
├── scripts/            # Pipeline runners and batch execution utilities
├── src/
│   └── wf052/
│       ├── __init__.py
│       ├── agent/      # Agentic evacuation coordinator & tool definitions
│       ├── data/       # Raster/vector ingestion, alignment, and formatting
│       ├── eval/       # Benchmark suites, verification metrics
│       ├── routing/    # Graph construction, dynamic routing algorithms
│       ├── serving/    # Demo service and orchestration endpoints
│       ├── sim/        # Physics teacher wildfire simulator
│       └── surrogate/  # PyTorch surrogate models and inference pipelines
├── tests/              # Test suite (pytest)
├── pyproject.toml      # Pinned dependencies and tooling configs
└── README.md
```

---

## Development Setup

### Requirements
- **Python**: 3.11
- **Deep Learning**: PyTorch only (no TensorFlow/Keras)

### Installation
```bash
# Create virtual environment using Python 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install editable project with dev dependencies
uv pip install -e ".[dev]"
```

### Linting & Testing
```bash
# Run linter
.venv/bin/ruff check .

# Run test suite
.venv/bin/pytest
```

---

## Compute Split
- **Local (CPU)**: Ingestion pipeline, graph construction, routing, agent tool invocation, unit/integration tests, and small test simulation runs.
- **Kaggle GPU**: Bulk simulation runs for training data generation and PyTorch surrogate neural network training.

---

---

## Wildfire Spread Surrogate Modeling (Phase 2A Pipeline)

### Stand-in Data Notice & Scaffold Provenance
> [!NOTE]
> The Phase 2A surrogate model pipeline is developed and verified on **stand-in data**.
> Propagation surfaces are generated via closed-form Huygens analytic wind-ellipses and `teacher_v1` cellular automaton scaffold simulations. Real teacher training data from large-scale hydrodynamic/cellular simulations will be plugged in during subsequent phases without altering the underlying data contract or model interfaces.
> 
> In all evaluation benchmarks and reports, performance metrics represent **fidelity to teacher simulation**, NOT ground-truth accuracy.

### Spatial Domain & Exclusion Rules
- **Tile Resolution & Extent**: $256 \times 256$ cells at $60\text{ m}$ resolution ($15.36\text{ km} \times 15.36\text{ km}$).
- **Training Fire Landscapes**: Samples are extracted exclusively from the 7 non-Camp training fires: `bobcat`, `caldor`, `creek`, `glass`, `kincade`, `tamarack`, and `zogg`.
- **Prohibited Domains**: The Camp Fire domain (`camp_fire_2018`) is **strictly prohibited** and excluded from all surrogate training and tuning pipelines to prevent data leakage.

### Data Contract & Input Feature Layout
- **Spatial Inputs (`(B, 13, 256, 256)`)**:
  - Channels 0–6: One-hot encoded Scott & Burgan 40 fuel groups (`NB`, `GR`, `GS`, `SH`, `TU`, `TL`, `SB`).
  - Channel 7: Topographic slope gradient (radians).
  - Channels 8–9: Trigonometric aspect ($\sin(\theta), \cos(\theta)$).
  - Channel 10: Z-scored elevation.
  - Channel 11: Normalized canopy cover ($[0, 1]$).
  - Channel 12: Binary ignition mask ($\{0, 1\}$), initialized in the upwind third.
- **Atmospheric Wind Input (`(B, 12, 2)`)**:
  - 12-hour hourly wind vector series $(u, v)$ in $\text{m/s}$.
- **Target Outputs**:
  - `arrival_time`: Raster of arrival times in minutes $[0, 720]$.
  - `censor_mask`: Binary mask indicating unburned cells at the 12-hour horizon ($1 = \text{unburned/censored}$).

### Neural Architecture & Multi-Task Loss
- **Model**: `WindConditionedUNet` with Feature-wise Linear Modulation (FiLM) from a 1D-Conv/MLP wind encoder.
- **Parameter Footprint**: 4,094,852 trainable parameters (strictly meeting the 2M–6M design constraint).
- **Quantile Output Heads**: Predicts arrival quantiles ($q_{10}, q_{50}, q_{90}$) and binary horizon burn logits.
- **Loss Formulation**:
  $$\mathcal{L} = \frac{1}{3} \sum_{\tau \in \{0.1, 0.5, 0.9\}} \mathcal{L}_{\tau}^{\text{pinball}}(y, \hat{q}_\tau) + \lambda_{\text{order}} \mathcal{L}_{\text{order}} + \lambda_{\text{bce}} \mathcal{L}_{\text{bce}}$$
  where $\mathcal{L}_{\text{order}} = \text{mean}(\text{ReLU}(\hat{q}_{10} - \hat{q}_{50}) + \text{ReLU}(\hat{q}_{50} - \hat{q}_{90}))$ penalizes quantile crossings.

---

## Pipeline Execution & Exact Commands

### 1. Interactive Evacuation Replay Demo
Operates 100% offline using precomputed simulation and routing assets in `out/demo/`:
```bash
# Launch interactive Streamlit demo in browser
PYTHONPATH=src streamlit run app/demo.py

# Run headless verification test (asserts 0 unhandled exceptions)
PYTHONPATH=src:scripts .venv/bin/python scripts/test_demo_headless.py
```

### 2. Full Evacuation Replay Simulation & Empirical Hazard Audit
Runs the comparative agent configurations against 436 NIST empirical truth hazards (burnovers, roadblocks, abandoned vehicles) and computes timing error across unique corridor hazards:
```bash
# Run Step 1 (dataset sample counts breakdown per fire and split)
PYTHONPATH=src:scripts .venv/bin/python scripts/step1_sample_counts.py

# Run Step 2 (disaggregated hazard-type margins across 30m, 60m, 120m and unique corridor timing)
PYTHONPATH=src:scripts .venv/bin/python scripts/step2_hazard_type_margin_and_timing.py
```

### 3. Surrogate Training & Verification
Trains the 4,094,852-parameter `WindConditionedUNet` surrogate model:
```bash
# Train surrogate locally (dry-run on available shards)
PYTHONPATH=src:scripts .venv/bin/python scripts/train_surrogate.py \
  --data-dir data/shards_kaggle_today_dry \
  --checkpoint-dir checkpoints/kaggle_today_dry \
  --epochs 1 \
  --batch-size 8

# Run evaluation on held-out test fires (caldor, tamarack)
PYTHONPATH=src:scripts .venv/bin/python scripts/eval_saved_checkpoint.py \
  --checkpoint checkpoints/kaggle_today_dry/best.pt \
  --shards-dir data/shards_kaggle_today_dry \
  --out-dir reports/kaggle_today_dry
```

### 4. Kaggle Packaging & Notebook Generation
Packages training fire tiles and generates the self-contained Kaggle GPU training notebook:
```bash
# Generate Kaggle standalone notebook with verified teacher hash
python scripts/create_kaggle_today_notebook.py

# Package codebase and verified training fire tiles for Kaggle upload
python scripts/package_for_kaggle.py
```

---

## Data Provenance & Cryptographic Hashes

| File / Component | Relative Path | SHA-256 Checksum | Description |
| :--- | :--- | :--- | :--- |
| **Teacher v2 Config** | `configs/teacher_v2.yaml` | `a6a92f81f7a9c72b5899d5d3173f86943a7f98648c4f210dcc5c646d2b4e85a5` | Frozen teacher simulation parameters |
| **Teacher v2 Copy** | `out/wf052-teacher-config/teacher_v2.yaml` | `a6a92f81f7a9c72b5899d5d3173f86943a7f98648c4f210dcc5c646d2b4e85a5` | Bit-identical config copy |
| **Kaggle Shard 000** | `data/shards_kaggle_today_dry/shard_000.npz` | `acbf4a902098204f9ab0f8b5ec366e39ab658b42a480297f1a6f9c5d2be017aa` | 62-sample training shard |
| **Dry Run Checkpoint** | `checkpoints/kaggle_today_dry/best.pt` | `2e0ea48408226065c71fa69b2d3544f8eaec38d975fb9017637841103c8c2e68` | Trained UNet weights (1 epoch dry run) |
| **Stand-in Checkpoint** | `checkpoints/best.pt` | `1da984e7f3fb8fdb454f7331dd1aa2e8fa44e782d8c3fc38890ff456e7592cf1` | Stand-in smoke checkpoint (Phase 3A) |
| **NIST Hazards GeoJSON**| `out/demo/hazards.geojson` | `2b8ea4ca7679ad3f368f51a46bca1387d7bbf1cfca28b0304a08f5d023f00995` | 436 empirical hazards from NIST TN 2252 |
| **Replay Routes JSON** | `out/demo/routes.json` | `577dc110d9359e19890bc894c77c6870020a1740922e96dd8319f61b0c007137` | Replay evacuation routes for 193 unique nodes |
| **Demo Metadata** | `out/demo/metadata.json` | `1115bbf0cb49a4635848bb247a32bf8814ec9f0e1a1796191b9fe35860e6e768` | Center lon/lat and zoom settings |

| **Sample Counts CSV** | `reports/step1_sample_counts.csv` | - | Shard sample count audit across fires and splits |
| **Hazard Margins CSV** | `reports/step2_hazard_type_margins.csv` | - | Margin (ii) sensitivity (30m, 60m, 120m) by hazard type |
| **Timing Error CSV** | `reports/step2_timing_by_hazard_type.csv` | - | Unique hazard timing per corridor and type |
| **Latency Summary CSV** | `reports/step4_latency_summary.csv` | - | Replan, surrogate, and physics runtimes |

---

## Claims & Evidence

### What the System CAN Claim
1. **Inference Time of an Untrained Architecture**: The PyTorch neural surrogate forward pass executes in 34.51 ms per $256 \times 256$ tile on Apple M1 GPU (`mps`), requiring 310.59 ms across 9 tiles for domain coverage; no fidelity claim is made (cite `reports/step4_latency_summary.csv`, rows 2 and 3).
2. **Interactive Evacuation Re-Planning**: The network router evaluates evacuation routes across all 200 origins in 2,115.74 ms (10.58 ms per origin), enabling interactive replanning within 30-minute operational cycles (cite `reports/step4_latency_summary.csv`, row 1).
3. **Burnover Avoidance via Directional Buffering (Agent B)**: Negative burnover margins at 60 m are 1 of 177 for static and 0 for all other methods; B traps 6 origins versus 0 for static; at 30 m B on A2-assim has median 35.9 versus 84.5 for static; no paired test or CI computed (NOT COMPUTED); inconclusive (cite `reports/step2_hazard_type_margins.csv`, rows 8, 13, 1, 6).
4. **Burnover Timing Error in Physics Teacher**: At seed 42, 8 of the 10 burnovers on the three corridors never burn in the teacher and the other 2 arrive +157 min late (cite `reports/step2_timing_by_hazard_type.csv`, rows 1, 5, 9; Clark Rd burnovers $n=6$, Skyway $n=2$, Pentz/Hwy 70 $n=2$).

### What the System CANNOT Claim
1. **NO Safety Guarantees**: This software is an academic engineering demonstration. It does NOT certify safe evacuation routes and must never be used for real-time emergency navigation or life-safety decisions.
2. **NO Superiority of A2-assim in Overall Negative Margins**: A2-assim does NOT eliminate negative margins under town-wide corridor gridlock. When evaluated against all 436 NIST hazards, A2-assim exhibits a median truth margin of -37.0 min versus -31.5 min for A0 static (cite `reports/step2_truth_margin_vs_nist_hazards.csv`, rows 5 and 2) because re-routing onto alternative roads did not bypass town-wide abandoned vehicles.
3. **NO Independent Real-Time Ground-Truth from Forward Simulation**: Teacher v2 physics forward simulation alone cannot predict road impassability on western exit corridors. Furthermore, causal observation assimilation ingests NIST empirical observations (roadblocks and satellite perimeters) that share an underlying source document with the empirical truth hazards.
4. **NO Production Readiness from Stand-in Weights**: Checkpoints evaluated in Phase 4A and 4B utilized `STAND-IN WEIGHTS` from a single smoke tile (85.5% burned) and cannot be claimed as validated surrogates.

---

## Model Limitations & Qualitative Clearance Notice

> [!IMPORTANT]
> All evacuation clearance times, link queuing delays, and corridor congestion metrics produced by `wf052.routing` are **qualitative planning assessments only**.
> 
> The macroscopic time-sliced assignment engine employs volume-delay functions (BPR) and link point-queue bottleneck formulations to evaluate relative scenario sensitivities (e.g. contraflow benefits, corridor capacity degradations, and safety margin thresholds). These outputs are intended for risk comparison and decision exploration, and must **not** be construed as certified microscopic operational traffic forecasts or guarantees of empirical evacuation timing. Micro-scale dynamics—including vehicle collisions, smoke-induced visibility reductions, abandoned vehicle blockages, and manual traffic direction—require dedicated agent-based microscopic traffic simulation.

### Specific Limitations
1. **HRRR Analysis vs Real-Time Forecast**: Atmospheric wind inputs use post-event HRRR analysis ($fxx=0$), not reanalysis or real-time operational weather forecasts.
2. **NIST Reconstructed Timestamps**: Empirical hazard onset times (roadblocks, burnovers) are reconstructed based on NIST TN 2252 supplementary records. Abandoned vehicles have a modelled constant onset at 87.0 min (harness assumption, not an observation).
3. **K=3 Corridor Bottleneck Clusters**: The road graph aggregates destination exits into 3 major corridor groups (Skyway, Clark Rd, Pentz/Hwy 70). Complex local off-corridor bypasses are not fully represented, and K=3 corridor clusters make bootstraps inconclusive.
4. **Teacher Physics Calibration Drift**: The teacher is a calibrated contact plus ember model; it under-burns the Camp Fire by about 2x and over-burns most training fires (cite `reports/step6_ember_ablation_summary.csv`, row 4: Mean Area Ratio 5.597 on training fires; `reports/camp_fire_heldout_60m_areas.csv`).
5. **Development-Only Case Domain**: The Camp Fire (2018) is strictly a development scenario and is never included in surrogate training data to protect benchmark independence.
6. **Shared Provenance in Causal Assimilation**: Causal observation assimilation ingests observed perimeters and roadblocks derived from the same NIST post-incident reconstruction as the evaluation hazards.
7. **Provider B Risk Cone Heuristic**: Agent B utilizes a geometric directional cone and distance heuristic rather than an integrated microscopic ember trajectory simulator.
8. **Vehicles Finish Before Fire Arrives in Physics Simulation**: Evacuation vehicles complete travel before the simulated fire reaches Paradise (smallest positive predicted margin 149.8 min; cite `reports/step5c_route_safety_margins.csv`).
9. **Route Redundancy**: Agents A and A2-pure produce identical routes in every comparative table.
10. **Surrogate Domain Over-Burn & Architecture Constraints**: The Phase 4A surrogate output covered the whole domain (IoU 0.1942 equals observed area over domain area) and comes from an untrained model; the surrogate is single-ignition and the Camp Fire ignition lies outside its tile; Phase 4B's A3-reignite burned 85.5 percent of its tile.
11. **No Held-Out Teacher Test**: No held-out teacher fire test was conducted.
12. **No Trained Surrogate Result**: No trained surrogate result is claimed unless a Kaggle or local checkpoint is reported in an appendix.
13. **Retracted Figures**: Phase 4B's narrative 24.1 ms forward pass and 56.4x speedup are retracted as unmeasured estimates. The measured numbers are 34.51 ms per tile (39.39x tile speedup; cite `reports/step4_latency_summary.csv`) and 310.59 ms domain coverage (8.85x domain speedup; cite `reports/step4_latency_summary.csv`).


