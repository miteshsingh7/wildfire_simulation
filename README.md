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

## Model Limitations & Qualitative Clearance Notice

> [!IMPORTANT]
> All evacuation clearance times, link queuing delays, and corridor congestion metrics produced by `wf052.routing` are **qualitative planning assessments only**.
> 
> The macroscopic time-sliced assignment engine employs volume-delay functions (BPR) and link point-queue bottleneck formulations to evaluate relative scenario sensitivities (e.g. contraflow benefits, corridor capacity degradations, and safety margin thresholds). These outputs are intended for risk comparison and decision exploration, and must **not** be construed as certified microscopic operational traffic forecasts or guarantees of empirical evacuation timing. Micro-scale dynamics—including vehicle collisions, smoke-induced visibility reductions, abandoned vehicle blockages, and manual traffic direction—require dedicated agent-based microscopic traffic simulation.


