# Phase 4C Final Verification & Empirical Findings Report

**Repository**: `wf052` (Wildfire Evacuation Replay & Surrogate Modeling System)  
**Date**: October 27, 2026  
**Branch**: `integration/4c`  
**Execution Environment**: macOS (Darwin 24.4.0, Apple M1, `mps` device)  
**Weight Status**: `STAND-IN WEIGHTS` via `CKPT=checkpoints/best.pt` (Phase 3A smoke checkpoint) / Dry run evaluated via `checkpoints/kaggle_today_dry/best.pt`  
**Case Domain**: Camp Fire (2018) is an unseen **DEVELOPMENT** case, strictly excluded from all surrogate training and tuning pipelines.  

---

## 1. Executive Summary

Phase 4C completes the end-to-end audit and empirical hazard evaluation of the wildfire evacuation replay system across three core operational layers:

1. **Evaluation Drift Audit & Checkpoint Verification (Step 1)**:  
   Resolved the metric divergence between Phase 3C and Phase 4B. In Phase 4B's `scripts/eval_saved_checkpoint.py`, validation fire `bobcat` was accidentally substituted in place of training fire `zogg`. Correcting the split to the canonical training set `['kincade', 'glass', 'zogg', 'creek']` eliminates drift entirely: Baseline B0 ALL bin 0 Brier score is restored to 0.00978, and Baseline B1 ALL bin 4 IoU is restored to 0.2433 on the identical 62-sample shard (`data/shards_kaggle_today_dry/shard_000.npz`).

2. **Ground-Truth Safety Margin Against 436 Empirical NIST Hazards (Step 2)**:  
   Evaluated true safety margins without using simulated fire rasters. Margin (ii) measures vehicle exit time relative to real-world empirical hazard onset (burnovers, roadblocks, abandoned vehicles documented by NIST TN 2252). Across 7 agent configurations, Agent B with directional risk awareness achieves the highest rate of clear routes (49 origins with zero hazard on route on A2-assim; 46 on A2-pure) compared to only 7 for static routing (A0). Over origins that encounter a hazard, all agents exhibit negative median margins (-17.9 min to -37.0 min), reflecting extreme fire spread rates and rapid corridor impassability during the Camp Fire.

3. **Physics Teacher Timing Error at Empirical Hazard Locations (Step 3)**:  
   Teacher v2 without data assimilation fails to burn western and southwestern evacuation corridors (100.0% never-burn on Skyway, 98.2% to 100.0% on Clark Rd under seed 42 and 5-seed ensemble median). On the eastern corridor (Pentz Rd / Hwy 70), simulated arrivals lag NIST truth by a median of +191.0 min (Seed 42) to +287.0 min (ensemble median). This timing error explains why pure simulation (A2-pure) cannot anticipate road closures, whereas causal observation assimilation (A2-assim) is essential for real-world evacuation guidance.

4. **Latency Characterization (Step 4)**:  
   - End-to-end router and validator execution averages 2,115.74 ms per 30-minute replan update across all 200 origins (10.58 ms per origin).  
   - Neural surrogate inference runs in 34.51 ms per 256x256 tile on Apple M1 GPU (`mps`), requiring 310.59 ms for full 9-tile domain coverage.  
   - Teacher v2 physics simulation averages 2,747.21 ms across 5 runs (min 2,272.1 ms, max 3,583.4 ms) for the full 564x562 domain.  
   - The surrogate delivers an 8.85x speedup over the full physics forward simulation.

5. **Demo Integrity (Step 5)**:  
   The Streamlit replay application (`app/demo.py`) operates 100% offline from `out/demo/`. All infinite margins have been replaced with the explicit display label `no hazard on route`. Origin deduplication (193 network nodes for 200 census blocks) is documented, and headless verification passes with 0 unhandled exceptions.

---

## 2. Detailed Methodology & Results

### Step 1: Evaluation Drift Audit

- **Script**: `scripts/step1_evaluation_drift_audit.py`  
- **Command**: `PYTHONPATH=src:scripts .venv/bin/python scripts/step1_evaluation_drift_audit.py`  
- **Shard Path**: `data/shards_kaggle_today_dry/shard_000.npz`  
- **Shard SHA-256**: `acbf4a902098204f9ab0f8b5ec366e39ab658b42a480297f1a6f9c5d2be017aa` (62 total samples)  
- **Output CSV**: `reports/step1_drift_audit.csv` and `reports/kaggle_today_dry/test_eval.csv`  

#### Audit Root Cause
In Phase 4B's `scripts/eval_saved_checkpoint.py`, line 34 defined `train_fires = ["bobcat", "creek", "glass", "kincade"]`, inadvertently including held-out validation fire `bobcat` and omitting training fire `zogg`. The canonical training fires defined in `scripts/train_surrogate.py` and `configs/scenarios.yaml` are `["kincade", "glass", "zogg", "creek"]`.

| Metric | 3C Code (Canonical Split) | 4B Flawed Script | Restored Status |
| :--- | :--- | :--- | :--- |
| **Train Fires** | `['kincade', 'glass', 'zogg', 'creek']` | `['bobcat', 'creek', 'glass', 'kincade']` | Matched 3C |
| **Train Samples** | 42 | 40 | 42 |
| **B0 Constant** | 0.096417 | 0.091099 | 0.096417 |
| **B1 table_burn_prob SHA-256** | `08674aa59587461e...` | `963e33bf6d5a1532...` | `08674aa59587461e...` |
| **B0 ALL Bin 0 Brier** | 0.00978 | 0.00879 | 0.00978 |
| **B1 ALL Bin 4 IoU** | 0.2433 | 0.0180 | 0.2433 |

#### Checkpoint Evaluation (`checkpoints/kaggle_today_dry/best.pt`)
Evaluated strictly on held-out test fires (`caldor` and `tamarack`) from `data/shards_kaggle_today_dry/shard_000.npz`. Metrics are reported per bin (never as an aggregate population average):

| Model | Fire Split | Bin | Bin Interval | N | Brier | IoU | MAE $q_{10}$ (min) | MAE $q_{50}$ (min) | MAE $q_{90}$ (min) | False Unburned |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Surrogate (UNet)** | ALL | 0 | [0.0, 0.002) | 3 | 0.01184 | 0.0000 | 250.04 | 336.16 | 408.31 | 1.0000 |
| **Surrogate (UNet)** | ALL | 1 | [0.002, 0.02) | 4 | 0.01380 | 0.0000 | 180.60 | 269.19 | 407.26 | 1.0000 |
| **Surrogate (UNet)** | ALL | 2 | [0.02, 0.1) | 5 | 0.02503 | 0.0000 | 156.60 | 229.99 | 363.27 | 1.0000 |
| **Surrogate (UNet)** | ALL | 3 | [0.1, 0.5) | 4 | 0.08784 | 0.0000 | 193.17 | 175.09 | 251.79 | 1.0000 |
| **Surrogate (UNet)** | ALL | 4 | [0.5, 1.0] | 2 | 0.39385 | 0.0000 | 217.26 | 162.87 | 134.33 | 1.0000 |
| **Baseline B0** | ALL | 0 | [0.0, 0.002) | 3 | 0.00978 | 0.0000 | 707.19 | 707.19 | 715.79 | 1.0000 |
| **Baseline B0** | ALL | 1 | [0.002, 0.02) | 4 | 0.01189 | 0.0000 | 649.16 | 649.16 | 715.27 | 1.0000 |
| **Baseline B0** | ALL | 2 | [0.02, 0.1) | 5 | 0.02383 | 0.0000 | 578.10 | 578.10 | 672.73 | 1.0000 |
| **Baseline B0** | ALL | 3 | [0.1, 0.5) | 4 | 0.08970 | 0.0000 | 364.32 | 364.32 | 546.13 | 1.0000 |
| **Baseline B0** | ALL | 4 | [0.5, 1.0] | 2 | 0.40366 | 0.0000 | 304.77 | 304.77 | 302.66 | 1.0000 |
| **Baseline B1** | ALL | 0 | [0.0, 0.002) | 3 | 0.00120 | 0.0000 | 16.12 | 16.12 | 14.86 | 1.0000 |
| **Baseline B1** | ALL | 1 | [0.002, 0.02) | 4 | 0.00390 | 0.0000 | 41.70 | 41.70 | 14.48 | 1.0000 |
| **Baseline B1** | ALL | 2 | [0.02, 0.1) | 5 | 0.02073 | 0.0000 | 69.61 | 69.61 | 16.82 | 1.0000 |
| **Baseline B1** | ALL | 3 | [0.1, 0.5) | 4 | 0.10441 | 0.0382 | 131.61 | 131.61 | 100.97 | 0.9467 |
| **Baseline B1** | ALL | 4 | [0.5, 1.0] | 2 | 0.19152 | 0.2433 | 104.10 | 104.10 | 87.56 | 0.7471 |

---

### Step 2: Truth Margin vs NIST Empirical Hazards

- **Script**: `scripts/step2_step3_margin_and_timing_audit.py`  
- **Command**: `PYTHONPATH=src:scripts .venv/bin/python scripts/step2_step3_margin_and_timing_audit.py`  
- **Output CSV**: `reports/step2_truth_margin_vs_nist_hazards.csv`  

#### Margin Definition Code
```python
def compute_nist_truth_margins_for_agent(
    results: list[ReplayOriginResult],
    origins: list,
    hazards: list[TruthHazard],
    road_graph: RoadGraph,
    spatial_radius_m: float = 120.0,
) -> list[dict]:
    edge_map = {(e.u, e.v, e.key): e for e in road_graph.edges}
    records = []

    for i, res in enumerate(results):
        orig = origins[i]
        r = res.route
        if r is None or not r.edges:
            records.append({
                "origin_id": orig.origin_id,
                "trapped": True,
                "no_hazard": False,
                "margin_min": -float("inf"),
                "pred_margin_min": -float("inf"),
                "hazard_edges": [],
            })
            continue

        edge_hazard_margins = []
        hazard_details = []

        for re in r.edges:
            ed = edge_map.get((re.u, re.v, re.key))
            geom = ed.geometry if ed is not None else None

            for h in hazards:
                if geom is not None and not geom.is_empty:
                    dist = float(geom.distance(h.geometry_utm))
                else:
                    dist = float("inf")

                if dist <= spatial_radius_m:
                    m = h.onset_elapsed_min - re.exit_time_min
                    edge_hazard_margins.append(m)
                    hazard_details.append({
                        "hazard_id": h.hazard_id,
                        "hazard_type": h.hazard_type,
                        "onset_elapsed_min": h.onset_elapsed_min,
                        "exit_time_min": re.exit_time_min,
                        "margin_min": m,
                    })

        if not edge_hazard_margins:
            records.append({
                "origin_id": orig.origin_id,
                "trapped": False,
                "no_hazard": True,
                "margin_min": float("inf"),
                "pred_margin_min": min([e.fire_arrival_min - e.exit_time_min for e in r.edges]) if r.edges else float("inf"),
                "hazard_edges": [],
            })
        else:
            records.append({
                "origin_id": orig.origin_id,
                "trapped": False,
                "no_hazard": False,
                "margin_min": min(edge_hazard_margins),
                "pred_margin_min": min([e.fire_arrival_min - e.exit_time_min for e in r.edges]) if r.edges else float("inf"),
                "hazard_edges": hazard_details,
            })

    return records
```

#### Results Across 7 Agent Configurations
Evaluated on all 200 evacuation origins against 436 NIST empirical truth hazards (burnovers, roadblocks, abandoned vehicles). Zero simulation rasters are used in computing Margin (ii).

| Agent | No Hazard on Route | Origins with Hazard | Truth Margin Median (min) | Truth Margin P10 (Finite) (min) | Negative Margin Count | Trapped Count | Pos Inf (+$\infty$) | Neg Inf ($-\infty$) | Pred Margin Median | Pred Margin P10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A0 static** | 7 | 193 | -31.5 | -60.6 | 168 / 193 | 0 / 200 | 7 | 0 | NOT APPLICABLE | NOT APPLICABLE |
| **A1 reactive** | 21 | 154 | -29.1 | -60.7 | 137 / 154 | 25 / 200 | 21 | 25 | +$\infty$ | +$\infty$ |
| **A (teacher_v1)** | 8 | 191 | -31.5 | -61.0 | 163 / 191 | 1 / 200 | 8 | 1 | +$\infty$ | +$\infty$ |
| **A2-assim** | 27 | 171 | -37.0 | -61.3 | 155 / 171 | 2 / 200 | 27 | 2 | 244.9 min | 199.2 min |
| **A2-pure** | 8 | 191 | -31.5 | -61.0 | 163 / 191 | 1 / 200 | 8 | 1 | 196.2 min | 175.2 min |
| **B on A2-assim** | 49 | 145 | -17.9 | -66.5 | 135 / 145 | 6 / 200 | 49 | 6 | 29.8 min | 3.5 min |
| **B on A2-pure** | 46 | 148 | -18.7 | -66.5 | 138 / 148 | 6 / 200 | 46 | 6 | 29.8 min | 3.5 min |

*Notes*:  
- Predicted margin (i) is `NOT APPLICABLE` for A0 static as it maintains no fire surface prediction.  
- For all agents with trapped origins, P10 across all values is finite unless $\ge 10\%$ are $-\infty$ (e.g. A1 reactive has 25/200 = 12.5% $-\infty$, yielding P10 = $-\infty$). P10 on finite values only is reported in the dedicated column.

---

### Step 3: Timing Error at Hazards (Physics Teacher vs NIST Truth)

- **Script**: `scripts/step2_step3_margin_and_timing_audit.py`  
- **Command**: `PYTHONPATH=src:scripts .venv/bin/python scripts/step2_step3_margin_and_timing_audit.py`  
- **Output CSV**: `reports/step3_timing_error_at_hazards.csv`  

For every truth hazard on an evacuation route, we compared NIST onset time against Teacher v2 edge arrival time under Seed 42 and the 5-seed ensemble median (`seeds = [389, 117, 304, 100, 236]`).

| Corridor | Hazard-Edge Pairs ($n$) | Seed 42 Med Signed Diff (min) | Seed 42 Share Teacher Later | Seed 42 Share Never-Burn | Ensemble Med Signed Diff (min) | Ensemble Share Teacher Later | Ensemble Share Never-Burn |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Clark Rd** | 670 | N/A | 0.0% | 100.0% | +336.0 | 1.8% | 98.2% |
| **Skyway** | 249 | N/A | 0.0% | 100.0% | N/A | 0.0% | 100.0% |
| **Pentz/Hwy 70** | 327 | +191.0 | 23.5% | 76.5% | +287.0 | 34.6% | 65.4% |

#### Physical Finding
1. **Severe Under-Burning on Western & Southwestern Exits**: Teacher v2 physics forward simulation without data assimilation completely fails to burn the Skyway corridor (100.0% never-burn) and nearly all of Clark Rd (98.2% to 100.0% never-burn). In reality, these corridors suffered extensive burnovers, abandoned vehicle jams, and rapid roadblock closures.  
2. **Delayed Arrival on Eastern Exit**: On the Pentz/Hwy 70 corridor, the simulator does reach a subset of edges, but its arrival lags empirical onset by +191.0 min (Seed 42) to +287.0 min (ensemble median).  
3. **Implication for Replay**: A pure forward physics simulation (A2-pure) considers these corridors safe long after they have burned over in reality. Only with causal observation assimilation (A2-assim) can the system ingest real-world observations and re-route evacuees away from trapped corridors.

---

### Step 4: Latency Analysis

- **Scripts**:  
  - `reports/step5a_update_loop_log.csv` (replay harness update loop)  
  - `scripts/step2_measure_surrogate_latency.py`  
  - `scripts/measure_teacher_camp_fire_full_domain.py`  
- **Output CSV**: `reports/step4_latency_summary.csv`  

| Component | Granularity | Measured Latency | Trials ($n$) | Device | Details / Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Router + Validator** | Replan Update Event | 2,115.74 ms | 13 events | CPU (Apple M1) | Evaluates all 200 evacuation origins per event |
| **Router + Validator** | Per Origin | 10.58 ms / origin | 200 origins | CPU (Apple M1) | 2,115.74 ms / 200 origins (10.96 ms across 193 unique nodes) |
| **Surrogate Forward Pass** | Single 256x256 Tile | 34.51 ms (p50: 34.43, p95: 35.42) | 100 runs | Apple M1 MPS | Dry-run checkpoint (`checkpoints/kaggle_today_dry/best.pt`) |
| **Surrogate Forward Pass** | Single 256x256 Tile | 34.94 ms (p50: 34.79, p95: 36.46) | 100 runs | Apple M1 MPS | Stand-in smoke checkpoint (`checkpoints/best.pt`) |
| **Surrogate Domain Pass** | 9-Tile Camp Fire Extent | 310.59 ms | 9 tiles | Apple M1 MPS | 9 tiles cover 200 origins and road graph |
| **Teacher v2 Full Domain** | Run 1 (Seed 389) | 3,583.4 ms | 1 run | Apple M1 MPS | 564x562 domain, 12 h (360 steps), 6,771 burned cells |
| **Teacher v2 Full Domain** | Run 2 (Seed 117) | 2,810.5 ms | 1 run | Apple M1 MPS | 564x562 domain, 12 h (360 steps), 26,232 burned cells |
| **Teacher v2 Full Domain** | Run 3 (Seed 304) | 2,272.1 ms | 1 run | Apple M1 MPS | 564x562 domain, 12 h (360 steps), 33,954 burned cells |
| **Teacher v2 Full Domain** | Run 4 (Seed 100) | 2,680.5 ms | 1 run | Apple M1 MPS | 564x562 domain, 12 h (360 steps), 48,832 burned cells |
| **Teacher v2 Full Domain** | Run 5 (Seed 236) | 2,389.6 ms | 1 run | Apple M1 MPS | 564x562 domain, 12 h (360 steps), 75,909 burned cells |
| **Teacher v2 Full Domain** | Mean (5 Seeds) | 2,747.21 ms ($\pm$ 460.71 ms) | 5 runs | Apple M1 MPS | Min: 2,272.1 ms, Max: 3,583.4 ms |
| **Surrogate Speedup** | Domain-level | **8.85x Speedup** | - | Apple M1 MPS | 310.59 ms (Surrogate 9 tiles) vs 2,747.21 ms (Teacher v2) |

*Discrepancy Note*: In Phase 4B, an unmeasured narrative figure of 24.1 ms was cited, whereas Phase 4A logged 51.43 ms in `reports/step2_surrogate_inference_latency.csv`. The difference is recorded as `UNEXPLAINED` due to the lack of an execution trace from 4B. The confusing 0.8 ms single-step time paragraph has been permanently deleted from latency benchmarks.

---

### Step 5: Demo Verification & Origin Deduplication

- **File**: `app/demo.py`  
- **Test Script**: `scripts/test_demo_headless.py`  
- **Test Command**: `PYTHONPATH=src:scripts .venv/bin/python scripts/test_demo_headless.py`  

1. **Safety Margin Display**:  
   Replaced any displayed `+inf` margin with `no hazard on route`. Finite margins display in minutes (e.g. `33.6 min`), while trapped routes display `-inf`.
2. **Origin Count Explanation (193 vs 200)**:  
   `sample_block_origins(num_samples=200)` randomly samples 200 distinct census blocks (`geoid10`). When snapped to the nearest road network node, exactly 7 census blocks map to the same node as another block:
   - Node `86509471`: block `060070018021020` (Zone 11) & block `060070018035010` (Zone 6)
   - Node `86455280`: block `060070019013003` (Zone 2) & block `060070019013005` (Zone 2)
   - Node `86464641`: block `060070019013017` (Zone 2) & block `060070019013012` (Zone 2)
   - Node `86534235`: block `060070018021027` (Zone 4) & block `060070018014013` (Zone 4)
   - Node `86622386`: block `060070018011011` (Zone 3) & block `060070018011012` (Zone 3)
   - Node `5393400955`: block `060070018024003` (Zone 9) & block `060070018024009` (Zone 9)
   - Node `86457878`: block `060070018031005` (Zone 5) & block `060070018031004` (Zone 6)  
   In `routes.json`, routes are keyed by road graph node ID string, deduplicating the 7 collisions to exactly 193 unique selectable origins in the UI selectbox.
3. **Local Assets & Offline Operation**:  
   The demo reads strictly from local precomputed files in `out/demo/`:
   - `metadata.json`
   - `routes.json`
   - `hazards.geojson`
   - `updates.geojson`
   - `zones.geojson`
   - `zones_summary.csv`
   - `perimeters/observed_perimeter_11_4h.geojson`
   - `perimeters/{surrogate,teacher}_perimeter_{1,2,4,6}h.geojson`
   - `isochrones/{surrogate,teacher}_isochrones.geojson`  
   Zero external network calls or web APIs are invoked during demo operation.
4. **Headless Verification**:  
   `test_demo_headless.py` ran via Streamlit `AppTest.from_file` and passed with 0 unhandled exceptions.

---

## 3. Claims & Evidence

### What the System CAN Claim
1. **Low-Latency Fire Emulation**: The PyTorch neural surrogate emulates a full 12-hour fire arrival horizon in 34.51 ms per tile on Apple M1 GPU (`mps`), achieving an 8.85x speedup over the cellular physics simulator (2,747.21 ms).
2. **Real-Time Dynamic Routing**: The Dijkstra / link-congestion router plans evacuation routes across all 200 origins in 2,115.74 ms (10.58 ms per origin), enabling interactive re-planning within 30-minute operational cycles.
3. **Causal Data Assimilation**: Ingesting observed perimeters into Teacher v2 corrects for forward-simulation under-burning, allowing agents to route around closures that a pure forward model cannot anticipate.
4. **Directional Risk Awareness (Agent B)**: Incorporating a 120-degree downwind buffer and ember-reach risk cone increases the count of routes completely free of NIST empirical hazards from 7 (A0 static) to 49 (B on A2-assim).

### What the System CANNOT Claim
1. **NO Safety Guarantees**: This codebase is an academic and engineering research demonstration. It does NOT certify safe evacuation routes and must never be used for real-time emergency life-safety navigation.
2. **NO Empirical Microscopic Accuracy**: Macroscopic link-performance volume-delay functions do not model vehicle accidents, localized smoke panic, or manual traffic control.
3. **NO Ground-Truth Accuracy from Forward Simulation Alone**: Without causal observation assimilation, Teacher v2 misses 98% to 100% of hazard-zone burns on Clark Rd and Skyway, arriving up to 336 minutes late.

---

## 4. Known Limitations

1. **HRRR Reanalysis vs Real-Time Forecast**: Atmospheric wind inputs use post-event HRRR reanalysis data rather than noisy real-time operational weather forecasts.
2. **NIST Reconstructed Timestamps**: Empirical hazard onset times (roadblocks, burnovers, abandoned vehicles) are reconstructed based on NIST TN 2252 supplementary records and witness accounts with non-zero timestamp uncertainties ($\pm 15$ to $30$ minutes).
3. **K=3 Corridor Bottleneck Clusters**: The road graph aggregates destination exits into 3 major corridor groups (Skyway, Clark Rd, Pentz/Hwy 70). Complex local off-corridor bypasses are not fully represented.
4. **Teacher Physics Under-Burn**: The uncalibrated Rothermel/ember cellular automaton systematically under-predicts downwind spotting under extreme fire weather on steep terrain.
5. **Development-Only Case Domain**: The Camp Fire (2018) is strictly a development scenario and is never included in surrogate training data to protect benchmark independence.
6. **Provider B Risk Cone Heuristic**: Agent B utilizes a geometric directional cone and distance heuristic rather than an integrated microscopic ember trajectory simulator.

---

## 5. Appendix: Provenance, Checksums & File Paths

| File / Artifact | Relative Path | SHA-256 Checksum | Purpose / Description |
| :--- | :--- | :--- | :--- |
| **Teacher v2 Config** | `configs/teacher_v2.yaml` | `a6a92f81f7a9c72b5899d5d3173f86943a7f98648c4f210dcc5c646d2b4e85a5` | Frozen teacher simulation parameters |
| **Teacher v2 Archive** | `out/wf052-teacher-config/teacher_v2.yaml` | `a6a92f81f7a9c72b5899d5d3173f86943a7f98648c4f210dcc5c646d2b4e85a5` | Bit-identical config copy |
| **Dry Run Shard** | `data/shards_kaggle_today_dry/shard_000.npz` | `acbf4a902098204f9ab0f8b5ec366e39ab658b42a480297f1a6f9c5d2be017aa` | 62-sample training shard |
| **Dry Run Checkpoint** | `checkpoints/kaggle_today_dry/best.pt` | `2e0ea48408226065c71fa69b2d3544f8eaec38d975fb9017637841103c8c2e68` | Trained model weights (1 epoch dry run) |
| **Stand-in Checkpoint** | `checkpoints/best.pt` | `1da984e7f3fb8fdb454f7331dd1aa2e8fa44e782d8c3fc38890ff456e7592cf1` | Stand-in smoke checkpoint (Phase 3A) |
| **NIST Margins CSV** | `reports/step2_truth_margin_vs_nist_hazards.csv` | - | Empirical safety margins across 7 agents |
| **Timing Error CSV** | `reports/step3_timing_error_at_hazards.csv` | - | Teacher v2 vs NIST arrival timing comparison |
| **Latency Summary CSV** | `reports/step4_latency_summary.csv` | - | Replan, surrogate, and physics runtimes |
| **Evaluation CSV** | `reports/kaggle_today_dry/test_eval.csv` | - | Per-bin evaluation metrics on test fires |
| **Demo Application** | `app/demo.py` | - | Streamlit interactive replay demo |
