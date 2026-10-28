# Phase 4D Final Verification & Empirical Hazard Analysis Report

**Repository**: `wf052` (Wildfire Evacuation Replay & Surrogate Modeling System)  
**Date**: October 28, 2026  
**Branch**: `integration/4d`  
**Execution Environment**: macOS (Darwin 24.4.0, Apple M1, `mps` device)  
**Weight Status**: `STAND-IN WEIGHTS` via `CKPT=checkpoints/best.pt` (Phase 3A smoke checkpoint) / Dry run evaluated via `checkpoints/kaggle_today_dry/best.pt`  
**Case Domain**: Camp Fire (2018) is an unseen **DEVELOPMENT** case, strictly excluded from all surrogate training and tuning pipelines. Zero safety claims.

---

## 1. Executive Summary

Phase 4D refines the empirical hazard analysis, closes sample count audits, characterizes hazard types independently, assesses demo offline dependencies, and establishes verified claims backed strictly by execution logs:

1. **Sample Count Audit (Step 1)**:  
   Resolved the Phase 4C sample count discrepancy. The shard `data/shards_kaggle_today_dry/shard_000.npz` contains exactly 62 samples across 7 fires: 34 training samples (`kincade`: 8, `glass`: 8, `zogg`: 10, `creek`: 8), 10 validation samples (`bobcat`: 10), and 18 testing samples (`caldor`: 8, `tamarack`: 10). The numbers 42 and 40 in Phase 4C's drift audit were a clerical typo; both canonical and flawed splits contained exactly 34 training samples because `zogg` and `bobcat` each contribute 10 samples. Documented in [`reports/step1_sample_counts.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step1_sample_counts.csv) and [`reports/step1_drift_audit.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step1_drift_audit.csv).

2. **Disaggregated Hazard-Type Analysis (Step 2)**:  
   Evaluated true safety margins without simulation rasters under a nearest-route-edge assignment (no cross-product over all edges), with margin defined as $t_{\text{onset}} - t_{\text{entry}}$ across matching radii (30 m, 60 m, 120 m):
   - **Burnovers (23 incidents, direct flame contact)**: All dynamic routing agents achieve zero negative margins (0/200), and static routing has only 1 negative margin (1/177 affected). Median margins are positive across all methods (+81.7 min to +127.0 min at 60 m). Agent B achieves the fewest affected origins (102/200 vs 177/200 for static) and highest median margin (+127.0 min).
   - **Roadblocks (178 records, traffic/utility closures)**: Dynamic agents achieve zero negative margins (0/200), whereas static routing experiences 84 negative margins (84/193 affected).
   - **Abandoned Vehicles (235 records, gridlock onset at 87.0 min)**: Prevalent along all exit corridors, producing negative margins for the vast majority of affected vehicles across all agents (135 to 168 negatives). This confirms that Phase 4C's overall negative margin counts were identical to and dominated by abandoned vehicle encounters. Documented in [`reports/step2_hazard_type_margins.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_hazard_type_margins.csv).

3. **Unique Hazard Corridor Timing Breakdown (Step 2d)**:  
   Deduped by unique `hazard_id` per corridor, Teacher v2 physics simulation completely misses the Skyway corridor (100.0% never-burn across all 42 unique hazards under both Seed 42 and the 5-seed ensemble) and Clark Rd (100.0% never-burn under Seed 42, 97.2% under ensemble). On Pentz Rd / Hwy 70, when fire does arrive, simulated arrivals lag empirical truth by a median of +137.5 min to +249.0 min (Seed 42) and +163.5 min to +283.0 min (ensemble). Documented in [`reports/step2_timing_by_hazard_type.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_timing_by_hazard_type.csv).

4. **Latency Retractions & Dual Speedup Reporting (Step 3 & 4)**:  
   - Phase 4B's narrative 24.1 ms inference latency and 56.4x speedup are formally **RETRACTED**.
   - **Tile-Level Speedup**: 39.39x (Teacher v2 256x256 tile latency of 1,359.3 ms vs surrogate forward pass of 34.51 ms).
   - **Domain-Level Speedup**: 8.85x (Teacher v2 full 564x562 domain latency of 2,747.21 ms vs surrogate 9-tile domain coverage of 310.59 ms). Documented in [`reports/step4_latency_summary.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step4_latency_summary.csv).

5. **Demo Basemap Offline Verification (Step 4)**:  
   The Streamlit demo (`app/demo.py`) previously requested remote Carto Positron vector tiles via Deck.gl (`map_style="light"`), which requires an internet connection. To ensure 100% offline operation, `map_style=None` is now set by default, rendering all simulation layers on a local canvas with an optional online basemap toggle. Headless verification passes with 0 unhandled exceptions.

---

## 2. Detailed Methodology & Results

### Step 1: Sample Counts Breakdown per Fire and Split

- **Script**: [`scripts/step1_sample_counts.py`](file:///Users/miteshsingh/Documents/projects/wf052/scripts/step1_sample_counts.py)  
- **Command**: `PYTHONPATH=src:scripts .venv/bin/python scripts/step1_sample_counts.py`  
- **Source Shard**: `data/shards_kaggle_today_dry/shard_000.npz` (SHA-256: `acbf4a902098204f9ab0f8b5ec366e39ab658b42a480297f1a6f9c5d2be017aa`)  
- **Output CSV**: [`reports/step1_sample_counts.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step1_sample_counts.csv) and [`reports/step1_drift_audit.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step1_drift_audit.csv)  

| Fire Name | Dataset Split | Sample Count | Maximum Allowed (2 / bin) | Bins Represented |
| :--- | :--- | :--- | :--- | :--- |
| **bobcat** | validation | 10 | 10 | Bins 0, 1, 2, 3, 4 |
| **caldor** | test | 8 | 10 | Bins 0, 1, 2, 3 |
| **creek** | train | 8 | 10 | Bins 1, 2, 3 |
| **glass** | train | 8 | 10 | Bins 1, 2, 3 |
| **kincade** | train | 8 | 10 | Bins 0, 1, 2, 3 |
| **tamarack** | test | 10 | 10 | Bins 0, 1, 2, 3, 4 |
| **zogg** | train | 10 | 10 | Bins 1, 2, 3, 4 |
| **TOTAL** | - | **62** | - | - |

#### Summary by Dataset Split
- **TRAIN** (`kincade`, `glass`, `zogg`, `creek`): 8 + 8 + 10 + 8 = **34 samples**  
- **VAL** (`bobcat`): **10 samples**  
- **TEST** (`caldor`, `tamarack`): 8 + 10 = **18 samples**  
- **TOTAL**: 34 + 10 + 18 = **62 samples**  

#### Clarification of Phase 4C Report Mismatch
In the Phase 4C drift audit table, training samples were incorrectly transcribed as 42 (canonical) vs 40 (flawed). The script `scripts/step1_evaluation_drift_audit.py` measured exactly 34 training samples in both runs. In the flawed 4B split (`['bobcat', 'creek', 'glass', 'kincade']`), validation fire `bobcat` (10 samples) replaced training fire `zogg` (10 samples). Because both fires have exactly 10 samples, the total training sample count remained at exactly 34 samples, while shifting sample content and causing baseline drift.

---

### Step 2: Hazard Types Margin & Timing Analysis

- **Script**: [`scripts/step2_hazard_type_margin_and_timing.py`](file:///Users/miteshsingh/Documents/projects/wf052/scripts/step2_hazard_type_margin_and_timing.py)  
- **Command**: `PYTHONPATH=src:scripts .venv/bin/python scripts/step2_hazard_type_margin_and_timing.py`  
- **Output CSVs**: [`reports/step2_hazard_type_margins.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_hazard_type_margins.csv) and [`reports/step2_timing_by_hazard_type.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_timing_by_hazard_type.csv)  

#### 2a. NIST Source Onset Definitions & Event Types
- **Source Citation Status**: `NOT CHECKED` (original NIST TN 2135 / TN 2252 PDF reports are not open in the repository; extracted Parquet tables are open in `data/tiles/`).
- **Burnovers (23 incidents)**: Onset time records the timestamp when active fire flames burned over vehicles, civilian evacuees, or emergency responders on roadways (e.g. Hoffman Road at 07:50–08:30 PST). **This is a direct fire-arrival event**.
- **Roadblocks (178 records)**: Onset time records when law enforcement or emergency services established physical road closures or when downed utility lines obstructed travel (e.g. CA-70 closure at 07:29 PST, power lines down at 04:30 PST). This is an operational traffic restriction event, not flame arrival.
- **Abandoned Vehicles (235 records)**: Post-fire survey inventoried vehicles left in traffic lanes across town. In the simulation harness, onset is modeled as a uniform town-wide corridor gridlock onset at 08:00 PST (87.0 min elapsed post-ignition). This is a traffic gridlock event (161 of 235 vehicles ultimately burned post-abandonment).

#### 2b. Margin (ii) Formulation
Hazards persist after onset. Margin is computed as $m = t_{\text{onset}} - t_{\text{entry}}$, where $t_{\text{entry}}$ is the time the vehicle enters the nearest route edge. Each hazard is assigned strictly to its nearest route edge only ($e^* = \arg\min_{e \in r} \text{dist}(e, h)$), eliminating cross-product matching over multiple edges.

#### PRIMARY TABLE: Burnover-Only Margins (23 Incidents)
*Source: [`reports/step2_hazard_type_margins.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_hazard_type_margins.csv)*

| Radius | Method | Hazard Type | Affected Origins (out of 200) | Negative Margins | Median Margin (min) | P10 Margin (min) | Trapped (out of 200) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **30 m** | A0 static | burnover | 118 / 200 | 1 / 118 | 84.5 | 38.4 | 0 / 200 |
| **30 m** | A1 reactive | burnover | 57 / 200 | 0 / 57 | 115.5 | 75.9 | 25 / 200 |
| **30 m** | A (teacher_v1) | burnover | 100 / 200 | 0 / 100 | 82.9 | 37.8 | 1 / 200 |
| **30 m** | A2-assim | burnover | 62 / 200 | 0 / 62 | 115.4 | 75.9 | 2 / 200 |
| **30 m** | A2-pure | burnover | 100 / 200 | 0 / 100 | 82.9 | 37.8 | 1 / 200 |
| **30 m** | B on A2-assim | burnover | 13 / 200 | 0 / 13 | 35.9 | 29.0 | 6 / 200 |
| **30 m** | B on A2-pure | burnover | 13 / 200 | 0 / 13 | 35.9 | 29.0 | 6 / 200 |
| **60 m** | A0 static | burnover | 177 / 200 | 1 / 177 | 82.5 | 42.8 | 0 / 200 |
| **60 m** | A1 reactive | burnover | 122 / 200 | 0 / 122 | 86.7 | 77.0 | 25 / 200 |
| **60 m** | A (teacher_v1) | burnover | 160 / 200 | 0 / 160 | 81.7 | 55.6 | 1 / 200 |
| **60 m** | A2-assim | burnover | 137 / 200 | 0 / 137 | 86.4 | 75.9 | 2 / 200 |
| **60 m** | A2-pure | burnover | 160 / 200 | 0 / 160 | 81.7 | 55.6 | 1 / 200 |
| **60 m** | B on A2-assim | burnover | 102 / 200 | 0 / 102 | 127.0 | 79.6 | 6 / 200 |
| **60 m** | B on A2-pure | burnover | 105 / 200 | 0 / 105 | 125.7 | 79.6 | 6 / 200 |
| **120 m** | A0 static | burnover | 177 / 200 | 1 / 177 | 82.5 | 42.8 | 0 / 200 |
| **120 m** | A1 reactive | burnover | 127 / 200 | 0 / 127 | 87.0 | 77.0 | 25 / 200 |
| **120 m** | A (teacher_v1) | burnover | 165 / 200 | 0 / 165 | 81.8 | 49.1 | 1 / 200 |
| **120 m** | A2-assim | burnover | 142 / 200 | 0 / 142 | 86.9 | 75.9 | 2 / 200 |
| **120 m** | A2-pure | burnover | 165 / 200 | 0 / 165 | 81.8 | 49.1 | 1 / 200 |
| **120 m** | B on A2-assim | burnover | 102 / 200 | 0 / 102 | 127.0 | 79.6 | 6 / 200 |
| **120 m** | B on A2-pure | burnover | 105 / 200 | 0 / 105 | 125.7 | 79.6 | 6 / 200 |

#### SECONDARY TABLE: Roadblocks Margins (178 Records, Radius = 60 m)
*Source: [`reports/step2_hazard_type_margins.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_hazard_type_margins.csv)*

| Method | Hazard Type | Affected Origins (out of 200) | Negative Margins | Median Margin (min) | P10 Margin (min) | Trapped (out of 200) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A0 static** | roadblock | 193 / 200 | 84 / 193 | 9.6 | -46.1 | 0 / 200 |
| **A1 reactive** | roadblock | 151 / 200 | 0 / 151 | 69.7 | 20.2 | 25 / 200 |
| **A (teacher_v1)** | roadblock | 188 / 200 | 0 / 188 | 55.4 | 20.1 | 1 / 200 |
| **A2-assim** | roadblock | 166 / 200 | 0 / 166 | 60.8 | 19.2 | 2 / 200 |
| **A2-pure** | roadblock | 188 / 200 | 0 / 188 | 55.4 | 20.1 | 1 / 200 |
| **B on A2-assim** | roadblock | 121 / 200 | 0 / 121 | 62.0 | 19.8 | 6 / 200 |
| **B on A2-pure** | roadblock | 124 / 200 | 0 / 124 | 59.4 | 19.7 | 6 / 200 |

#### SECONDARY TABLE: Abandoned Vehicles Margins (235 Records, Radius = 60 m)
*Source: [`reports/step2_hazard_type_margins.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_hazard_type_margins.csv)*

| Method | Hazard Type | Affected Origins (out of 200) | Negative Margins | Median Margin (min) | P10 Margin (min) | Trapped (out of 200) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A0 static** | abandoned_vehicle | 186 / 200 | 168 / 186 | -30.9 | -57.2 | 0 / 200 |
| **A1 reactive** | abandoned_vehicle | 142 / 200 | 137 / 142 | -36.5 | -57.4 | 25 / 200 |
| **A (teacher_v1)** | abandoned_vehicle | 181 / 200 | 163 / 181 | -35.7 | -57.5 | 1 / 200 |
| **A2-assim** | abandoned_vehicle | 160 / 200 | 155 / 160 | -39.6 | -57.8 | 2 / 200 |
| **A2-pure** | abandoned_vehicle | 181 / 200 | 163 / 181 | -35.7 | -57.5 | 1 / 200 |
| **B on A2-assim** | abandoned_vehicle | 137 / 200 | 135 / 137 | -17.7 | -63.4 | 6 / 200 |
| **B on A2-pure** | abandoned_vehicle | 140 / 200 | 138 / 140 | -22.6 | -63.1 | 6 / 200 |

#### 2c. Metric Equivalence: Negative Margins vs Section 7.3 Abandoned Vehicle Misses
The negative margin counts in Phase 4C (168 for static, 163 for A, 155 for A2-assim, 135 for B on A2-assim) equal the abandoned vehicle miss counts from Section 7.3 (`reports/step5b_replay_metrics_sec73.csv`) exactly.  
**State of equivalence**: The two metrics represent the identical physical computation. In Section 7.3, a hazard miss occurs when $t_{\text{traversal}} \ge t_{\text{onset}}$ within spatial radius 120 m, which is mathematically identical to margin $t_{\text{onset}} - t_{\text{traversal}} \le 0$. Because the 235 abandoned vehicles have a uniform onset time of 87.0 min (08:00 PST) and line every major corridor, vehicles traversing these roads after 08:00 PST encounter negative margins. As shown in the tables above, burnovers produce only 1 negative margin for static and 0 for dynamic agents, meaning that overall negative margin counts are completely determined by abandoned vehicles.

---

### Step 2d: Timing Error per Type & Corridor over Unique Hazards

- **Source CSV**: [`reports/step2_timing_by_hazard_type.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_timing_by_hazard_type.csv)  
- Deduped strictly by unique `hazard_id` per corridor:

| Row | Corridor | Hazard Type | Unique Hazards ($n$) | S42 Finite ($n$) | S42 Med Signed Diff (min) | S42 Share Never-Burn | S42 Share Teacher Later | Ens Finite ($n$) | Ens Med Signed Diff (min) | Ens Share Never-Burn | Ens Share Teacher Later |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Clark Rd | burnover | 6 | 0 | N/A | 100.0% | 0.0% | 1 | +343.0 | 83.3% | 16.7% |
| **2** | Clark Rd | roadblock | 33 | 0 | N/A | 100.0% | 0.0% | 1 | +270.0 | 97.0% | 3.0% |
| **3** | Clark Rd | abandoned_vehicle | 105 | 0 | N/A | 100.0% | 0.0% | 2 | +463.0 | 98.1% | 1.9% |
| **4** | Clark Rd | total | 144 | 0 | N/A | 100.0% | 0.0% | 4 | +403.0 | 97.2% | 2.8% |
| **5** | Skyway | burnover | 2 | 0 | N/A | 100.0% | 0.0% | 0 | N/A | 100.0% | 0.0% |
| **6** | Skyway | roadblock | 18 | 0 | N/A | 100.0% | 0.0% | 0 | N/A | 100.0% | 0.0% |
| **7** | Skyway | abandoned_vehicle | 22 | 0 | N/A | 100.0% | 0.0% | 0 | N/A | 100.0% | 0.0% |
| **8** | Skyway | total | 42 | 0 | N/A | 100.0% | 0.0% | 0 | N/A | 100.0% | 0.0% |
| **9** | Pentz/Hwy 70 | burnover | 2 | 2 | +157.0 | 0.0% | 100.0% | 2 | +266.0 | 0.0% | 100.0% |
| **10** | Pentz/Hwy 70 | roadblock | 19 | 2 | +137.5 | 89.5% | 10.5% | 4 | +163.5 | 78.9% | 21.1% |
| **11** | Pentz/Hwy 70 | abandoned_vehicle | 55 | 19 | +249.0 | 65.5% | 34.5% | 20 | +283.0 | 63.6% | 36.4% |
| **12** | Pentz/Hwy 70 | total | 76 | 23 | +247.0 | 69.7% | 30.3% | 26 | +283.0 | 65.8% | 34.2% |

#### Verified Conclusions Citing Specific Table Rows
1. **Skyway Total Under-Burn (Rows 5–8)**: Teacher v2 exhibits a 100.0% never-burn rate across all 42 unique hazards on the Skyway corridor under both Seed 42 and the 5-seed ensemble median (rows 5, 6, 7, and 8).
2. **Clark Road Under-Burn and Extreme Delays (Rows 1–4)**: Under Seed 42, 0 of 144 unique hazards on Clark Rd burn (100.0% never-burn, row 4). Under the ensemble median, 97.2% never burn; for the 4 hazards that do burn, Teacher v2 arrives +403.0 minutes later than empirical onset (row 4), including +343.0 min for burnovers (row 1), +270.0 min for roadblocks (row 2), and +463.0 min for abandoned vehicles (row 3).
3. **Pentz/Hwy 70 Systematic Fire Lags (Rows 9–12)**: On Pentz/Hwy 70, where the fire simulator does simulate flame propagation across 23 hazards (Seed 42) and 26 hazards (ensemble), arrivals lag empirical onset by a median of +247.0 min (Seed 42) and +283.0 min (ensemble) (row 12). Specifically, simulated burnover arrivals lag empirical burnovers by +157.0 min under Seed 42 and +266.0 min under the ensemble median (row 9).

---

### Step 3: Latency Benchmarks & Speedup Reporting

- **Source CSV**: [`reports/step4_latency_summary.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step4_latency_summary.csv)  
- **Retraction Notice**: The Phase 4B figures of 24.1 ms forward pass and 56.4x speedup were unmeasured narrative estimates and are formally **RETRACTED**.

| Benchmark Level | Teacher v2 Runtime (Physics) | Surrogate Runtime (UNet) | Speedup Factor | Hardware / Device |
| :--- | :--- | :--- | :--- | :--- |
| **Tile-Level** ($256 \times 256$ tile, 12 h horizon) | 1,359.3 ms ($\pm$ 214.2 ms) | 34.51 ms (p50: 34.43 ms) | **39.39x Speedup** | Apple M1 MPS |
| **Domain-Level** ($564 \times 562$ extent, 9 tiles) | 2,747.21 ms ($\pm$ 460.71 ms) | 310.59 ms ($9 \times 34.51\text{ ms}$) | **8.85x Speedup** | Apple M1 MPS |

*Notes on Surrogate Weights*: All surrogate-driven rows in earlier phases (e.g. Phase 4A Agent A3 rows, Phase 4B Agent A3-reignite rows) were evaluated using `STAND-IN WEIGHTS` from the Phase 3A smoke checkpoint (`checkpoints/best.pt`). This checkpoint was trained on an untrained, single over-burned 85.5% burned tile and does not represent a production surrogate model.

---

### Step 4: Demo Offline Integrity & Basemap Configuration

- **File**: [`app/demo.py`](file:///Users/miteshsingh/Documents/projects/wf052/app/demo.py)  
- **Headless Test**: [`scripts/test_demo_headless.py`](file:///Users/miteshsingh/Documents/projects/wf052/scripts/test_demo_headless.py)  

1. **Basemap Network Requirement**:  
   Streamlit / PyDeck renders basemap tiles via Mapbox/Carto vector tile APIs (`map_style="light"`). Loading these background basemap tiles requires active network connectivity.
2. **Offline Implementation**:  
   In `app/demo.py`, `map_style=None` is set by default, allowing Deck.gl to render the complete vector layers (road network, evacuation routes, fire isochrones, hazards) on an offline local canvas with zero remote network requests. An optional checkbox in the sidebar (`Load Online Basemap (Carto)`) allows streaming Carto Positron tiles when online.
3. **Headless Verification**:  
   Ran `PYTHONPATH=src:scripts .venv/bin/python scripts/test_demo_headless.py`. Passed with 0 unhandled exceptions.

---

## 3. Claims & Evidence

### What the System CAN Claim
1. **Low-Latency Tile-Level Fire Emulation**: The PyTorch neural surrogate emulates a 12-hour fire arrival horizon on a $256 \times 256$ tile in 34.51 ms on Apple M1 GPU (`mps`), achieving a **39.39x speedup** over the cellular physics simulator (1,359.3 ms). At the full 9-tile domain level, it achieves an **8.85x speedup** (310.59 ms vs 2,747.21 ms). Cites [`reports/step4_latency_summary.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step4_latency_summary.csv).
2. **Interactive Evacuation Re-Planning**: The network router evaluates evacuation routes across all 200 origins in 2,115.74 ms (10.58 ms per origin), enabling interactive replanning within 30-minute operational cycles. Cites [`reports/step4_latency_summary.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step4_latency_summary.csv).
3. **Burnover Avoidance via Directional Buffering (Agent B)**: Incorporating a 120-degree downwind buffer and ember-reach risk cone increases median safety margins against direct flame burnovers from +82.5 min (A0 static) to +127.0 min (B on A2-assim) at 60 m radius, reducing affected origins from 177/200 to 102/200. Cites [`reports/step2_hazard_type_margins.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_hazard_type_margins.csv).
4. **Empirical Measurement of Simulator Under-Burn**: Rigorous benchmarking reveals that Teacher v2 without assimilation never burns 100.0% of empirical hazards on Skyway and 97.2% on Clark Rd, and lags Pentz Rd arrivals by +163.5 min to +283.0 min. Cites [`reports/step2_timing_by_hazard_type.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_timing_by_hazard_type.csv).

### What the System CANNOT Claim
1. **NO Safety Guarantees**: This software is an academic engineering demonstration. It does NOT certify safe evacuation routes and must never be used for real-time emergency navigation or life-safety decisions.
2. **NO Superiority of A2-assim in Overall Negative Margins**: A2-assim does NOT eliminate negative margins under town-wide corridor gridlock. When evaluated against all 436 NIST hazards, A2-assim exhibits a median truth margin of -34.1 min (vs -28.0 min for A0 static) because re-routing onto alternative roads did not bypass town-wide abandoned vehicles. Cites [`reports/step2_hazard_type_margins.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_hazard_type_margins.csv).
3. **NO Independent Real-Time Ground-Truth from Forward Simulation**: Teacher v2 physics forward simulation alone cannot predict road impassability on western exit corridors. Furthermore, causal observation assimilation ingests NIST empirical observations (roadblocks and satellite perimeters) that share an underlying source document with the empirical truth hazards.
4. **NO Production Readiness from Stand-in Weights**: Checkpoints evaluated in Phase 4A and 4B utilized `STAND-IN WEIGHTS` from a single smoke tile (85.5% burned) and cannot be claimed as validated surrogates.

---

## 4. Known Limitations

1. **HRRR Reanalysis vs Real-Time Forecast**: Atmospheric wind inputs use post-event HRRR reanalysis data rather than noisy real-time operational weather forecasts.
2. **NIST Reconstructed Timestamps**: Empirical hazard onset times (roadblocks, burnovers, abandoned vehicles) are reconstructed based on NIST TN 2252 supplementary records with non-zero timestamp uncertainties ($\pm 15$ to $30$ minutes).
3. **K=3 Corridor Bottleneck Clusters**: The road graph aggregates destination exits into 3 major corridor groups (Skyway, Clark Rd, Pentz/Hwy 70). Complex local off-corridor bypasses are not fully represented.
4. **Teacher Physics Under-Burn**: The uncalibrated Rothermel/ember cellular automaton systematically under-predicts downwind spotting under extreme fire weather on steep terrain.
5. **Development-Only Case Domain**: The Camp Fire (2018) is strictly a development scenario and is never included in surrogate training data to protect benchmark independence.
6. **Shared Provenance in Causal Assimilation**: Causal observation assimilation ingests observed perimeters and roadblocks derived from the same NIST post-incident reconstruction as the evaluation hazards.
7. **Provider B Risk Cone Heuristic**: Agent B utilizes a geometric directional cone and distance heuristic rather than an integrated microscopic ember trajectory simulator.

---

## 5. Appendix: Provenance, Checksums & File Paths

| File / Artifact | Relative Path | SHA-256 Checksum | Purpose / Description |
| :--- | :--- | :--- | :--- |
| **Teacher v2 Config** | `configs/teacher_v2.yaml` | `a6a92f81f7a9c72b5899d5d3173f86943a7f98648c4f210dcc5c646d2b4e85a5` | Frozen teacher simulation parameters |
| **Teacher v2 Copy** | `out/wf052-teacher-config/teacher_v2.yaml` | `a6a92f81f7a9c72b5899d5d3173f86943a7f98648c4f210dcc5c646d2b4e85a5` | Bit-identical config copy |
| **Dry Run Shard** | `data/shards_kaggle_today_dry/shard_000.npz` | `acbf4a902098204f9ab0f8b5ec366e39ab658b42a480297f1a6f9c5d2be017aa` | 62-sample training shard |
| **Dry Run Checkpoint** | `checkpoints/kaggle_today_dry/best.pt` | `2e0ea48408226065c71fa69b2d3544f8eaec38d975fb9017637841103c8c2e68` | Trained model weights (1 epoch dry run) |
| **Stand-in Checkpoint** | `checkpoints/best.pt` | `1da984e7f3fb8fdb454f7331dd1aa2e8fa44e782d8c3fc38890ff456e7592cf1` | Stand-in smoke checkpoint (Phase 3A) |
| **Sample Counts CSV** | `reports/step1_sample_counts.csv` | - | Per-fire and per-split sample breakdown |
| **Drift Audit CSV** | `reports/step1_drift_audit.csv` | - | Audit of training split composition |
| **Hazard Margins CSV** | `reports/step2_hazard_type_margins.csv` | - | Margins by hazard type and radius |
| **Timing Error CSV** | `reports/step2_timing_by_hazard_type.csv` | - | Unique hazard timing per corridor and type |
| **Latency Summary CSV** | `reports/step4_latency_summary.csv` | - | Replan, surrogate, and physics runtimes |
| **Demo Application** | `app/demo.py` | - | Streamlit interactive replay demo |
