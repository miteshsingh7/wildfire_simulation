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

## Model Limitations & Qualitative Clearance Notice

> [!IMPORTANT]
> All evacuation clearance times, link queuing delays, and corridor congestion metrics produced by `wf052.routing` are **qualitative planning assessments only**.
> 
> The macroscopic time-sliced assignment engine employs volume-delay functions (BPR) and link point-queue bottleneck formulations to evaluate relative scenario sensitivities (e.g. contraflow benefits, corridor capacity degradations, and safety margin thresholds). These outputs are intended for risk comparison and decision exploration, and must **not** be construed as certified microscopic operational traffic forecasts or guarantees of empirical evacuation timing. Micro-scale dynamics—including vehicle collisions, smoke-induced visibility reductions, abandoned vehicle blockages, and manual traffic direction—require dedicated agent-based microscopic traffic simulation.

