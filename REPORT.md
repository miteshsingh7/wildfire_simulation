# Phase 4E Final Claims Fix and Freeze Report

**Repository**: `wf052` (Wildfire Evacuation Replay & Surrogate Modeling System)  
**Date**: October 28, 2026  
**Branch**: `integration/4e`  
**Execution Environment**: macOS (Darwin 24.4.0, Apple M1, `mps` device)  
**Weight Status**: `STAND-IN WEIGHTS` via `CKPT=checkpoints/best.pt` (Phase 3A smoke checkpoint) / Dry run evaluated via `checkpoints/kaggle_today_dry/best.pt`  
**Case Domain**: Camp Fire (2018) is an unseen **DEVELOPMENT** case, strictly excluded from all surrogate training and tuning pipelines. Zero safety claims.

---

## 1. Status Overview

- **Physics Teacher (Contact + Embers)**: Done with caveats (calibrated contact plus ember model; under-burns Camp Fire by about 2x, over-burns training fires; cite `reports/step6_ember_ablation_summary.csv`, row 4).
- **Surrogate Neural Architecture (WindConditionedUNet)**: Done (4,094,852 parameters, quantile heads $q_{10}/q_{50}/q_{90}$ + burn probability, FiLM wind conditioning).
- **Surrogate Shard Generation & Sampler**: Done (sampler stall resolved in Phase 2I; dry run shard with 62 samples across 7 fires; cite `reports/step1_sample_counts.csv`).
- **Base-rate Bias Init & Evaluation Module**: Done (base-rate init to train mean logit, per-bin quantile MAE/Brier/IoU metrics, canonical splits enforced; cite `reports/kaggle_today_dry/test_eval.csv`).
- **Surrogate GPU Training on Full Dataset**: Not done (Kaggle notebook generated and verified; full multi-epoch GPU training run pending external Kaggle execution).
- **Causal Replay Harness (Camp Fire Development Case)**: Done with caveats (7 comparative agents evaluated; Camp Fire is strictly unseen development case; stand-in weights used for A3).
- **NIST Empirical Hazard Safety Margins (Burnovers, Roadblocks, Abandoned Vehicles)**: Done with caveats (burnovers and roadblocks have empirical timestamps; abandoned vehicles have modelled constant onset at 87.0 min set in `src/wf052/replay/harness.py` line 133 as a harness assumption, not an observation).
- **Hazard Corridor Timing Error Analysis**: Done (deduped by unique hazard ID; Clark Rd $n=6$, Skyway $n=2$, Pentz/Hwy 70 $n=2$ for burnovers; cite `reports/step2_timing_by_hazard_type.csv`).
- **Interactive Streamlit Replay Demo**: Done with caveats (offline map canvas with optional Carto basemap; zero network requests in a live browser is NOT VERIFIED).

---

## 2. Abandoned-Vehicle Onset Provenance

- **Code Location**: [`src/wf052/replay/harness.py`](file:///Users/miteshsingh/Documents/projects/wf052/src/wf052/replay/harness.py), line 133:
  ```python
  onset_elapsed_min=87.0,  # 08:00 PST town-wide overrun onset
  ```
- **Provenance Status**: This is a **harness assumption**, NOT an observed timestamp from NIST. In NIST TN 2252 supplementary Table H-2, abandoned vehicles were inventoried post-fire (location, street, lane, whether burned), but individual vehicle abandonment timestamps were not recorded. The replay harness assigned a constant synthetic onset of 87.0 minutes (08:00 PST, 87 minutes after 06:33 ignition) across all 235 vehicles as an assumed town-wide overrun onset. Every abandoned-vehicle row in this report and repository documentation is strictly designated as a **modelled constant onset, not an observation**.

---

## 3. Disaggregated Hazard Types Analysis

### Sample Counts per Fire and Split
*Source: [`reports/step1_sample_counts.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step1_sample_counts.csv)*

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

- **TRAIN** (`kincade`, `glass`, `zogg`, `creek`): $8 + 8 + 10 + 8 = \mathbf{34\text{ samples}}$ (cite `reports/step1_sample_counts.csv`, rows 3, 4, 5, 7).
- **VAL** (`bobcat`): **10 samples** (cite `reports/step1_sample_counts.csv`, row 1).
- **TEST** (`caldor`, `tamarack`): $8 + 10 = \mathbf{18\text{ samples}}$ (cite `reports/step1_sample_counts.csv`, rows 2, 6).
- **TOTAL**: $34 + 10 + 18 = \mathbf{62\text{ samples}}$.

### PRIMARY TABLE: Burnover-Only Margins (23 Incidents)
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

### SECONDARY TABLE: Roadblocks Margins (178 Records, Radius = 60 m)
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

### SECONDARY TABLE: Abandoned Vehicles Margins (235 Records, Modelled Constant Onset at 87.0 min, Not an Observation, Radius = 60 m)
*Source: [`reports/step2_hazard_type_margins.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_hazard_type_margins.csv)*

| Method | Hazard Type (Modelled Constant Onset) | Affected Origins (out of 200) | Negative Margins | Median Margin (min) | P10 Margin (min) | Trapped (out of 200) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A0 static** | abandoned_vehicle (modelled constant onset) | 186 / 200 | 168 / 186 | -30.9 | -57.2 | 0 / 200 |
| **A1 reactive** | abandoned_vehicle (modelled constant onset) | 142 / 200 | 137 / 142 | -36.5 | -57.4 | 25 / 200 |
| **A (teacher_v1)** | abandoned_vehicle (modelled constant onset) | 181 / 200 | 163 / 181 | -35.7 | -57.5 | 1 / 200 |
| **A2-assim** | abandoned_vehicle (modelled constant onset) | 160 / 200 | 155 / 160 | -39.6 | -57.8 | 2 / 200 |
| **A2-pure** | abandoned_vehicle (modelled constant onset) | 181 / 200 | 163 / 181 | -35.7 | -57.5 | 1 / 200 |
| **B on A2-assim** | abandoned_vehicle (modelled constant onset) | 137 / 200 | 135 / 137 | -17.7 | -63.4 | 6 / 200 |
| **B on A2-pure** | abandoned_vehicle (modelled constant onset) | 140 / 200 | 138 / 140 | -22.6 | -63.1 | 6 / 200 |

---

### Step 2d: Timing Error per Type & Corridor over Unique Hazards

- **Source CSV**: [`reports/step2_timing_by_hazard_type.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_timing_by_hazard_type.csv)  
- Deduped strictly by unique `hazard_id` per corridor:

| Row | Corridor | Hazard Type | Unique Hazards ($n$) | S42 Finite ($n$) | S42 Med Signed Diff (min) | S42 Share Never-Burn | S42 Share Teacher Later | Ens Finite ($n$) | Ens Med Signed Diff (min) | Ens Share Never-Burn | Ens Share Teacher Later |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Clark Rd | burnover | 6 | 0 | N/A | 100.0% | 0.0% | 1 | +343.0 | 83.3% | 16.7% |
| **2** | Clark Rd | roadblock | 33 | 0 | N/A | 100.0% | 0.0% | 1 | +270.0 | 97.0% | 3.0% |
| **3** | Clark Rd | abandoned_vehicle (modelled) | 105 | 0 | N/A | 100.0% | 0.0% | 2 | +463.0 | 98.1% | 1.9% |
| **4** | Clark Rd | total (modelled) | 144 | 0 | N/A | 100.0% | 0.0% | 4 | +403.0 | 97.2% | 2.8% |
| **5** | Skyway | burnover | 2 | 0 | N/A | 100.0% | 0.0% | 0 | N/A | 100.0% | 0.0% |
| **6** | Skyway | roadblock | 18 | 0 | N/A | 100.0% | 0.0% | 0 | N/A | 100.0% | 0.0% |
| **7** | Skyway | abandoned_vehicle (modelled) | 22 | 0 | N/A | 100.0% | 0.0% | 0 | N/A | 100.0% | 0.0% |
| **8** | Skyway | total (modelled) | 42 | 0 | N/A | 100.0% | 0.0% | 0 | N/A | 100.0% | 0.0% |
| **9** | Pentz/Hwy 70 | burnover | 2 | 2 | +157.0 | 0.0% | 100.0% | 2 | +266.0 | 0.0% | 100.0% |
| **10** | Pentz/Hwy 70 | roadblock | 19 | 2 | +137.5 | 89.5% | 10.5% | 4 | +163.5 | 78.9% | 21.1% |
| **11** | Pentz/Hwy 70 | abandoned_vehicle (modelled) | 55 | 19 | +249.0 | 65.5% | 34.5% | 20 | +283.0 | 63.6% | 36.4% |
| **12** | Pentz/Hwy 70 | total (modelled) | 76 | 23 | +247.0 | 69.7% | 30.3% | 26 | +283.0 | 65.8% | 34.2% |

---

## 4. Claims & Evidence

### What the System CAN Claim
1. **Inference Time of an Untrained Architecture**: The PyTorch neural surrogate forward pass executes in 34.51 ms per $256 \times 256$ tile on Apple M1 GPU (`mps`), requiring 310.59 ms across 9 tiles for domain coverage; no fidelity claim is made (cite [`reports/step4_latency_summary.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step4_latency_summary.csv), rows 2 and 3).
2. **Interactive Evacuation Re-Planning**: The network router evaluates evacuation routes across all 200 origins in 2,115.74 ms (10.58 ms per origin), enabling interactive replanning within 30-minute operational cycles (cite [`reports/step4_latency_summary.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step4_latency_summary.csv), row 1).
3. **Burnover Avoidance via Directional Buffering (Agent B)**: Negative burnover margins at 60 m are 1 of 177 for static and 0 for all other methods; B traps 6 origins versus 0 for static; at 30 m B on A2-assim has median 35.9 versus 84.5 for static; no paired test or CI computed (NOT COMPUTED); inconclusive (cite [`reports/step2_hazard_type_margins.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_hazard_type_margins.csv), rows 8, 13, 1, 6).
4. **Burnover Timing Error in Physics Teacher**: At seed 42, 8 of the 10 burnovers on the three corridors never burn in the teacher and the other 2 arrive +157 min late (cite [`reports/step2_timing_by_hazard_type.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_timing_by_hazard_type.csv), rows 1, 5, 9; Clark Rd burnovers $n=6$, Skyway $n=2$, Pentz/Hwy 70 $n=2$).

### What the System CANNOT Claim
1. **NO Safety Guarantees**: This software is an academic engineering demonstration. It does NOT certify safe evacuation routes and must never be used for real-time emergency navigation or life-safety decisions.
2. **NO Superiority of A2-assim in Overall Negative Margins**: A2-assim does NOT eliminate negative margins under town-wide corridor gridlock. When evaluated against all 436 NIST hazards, A2-assim exhibits a median truth margin of -37.0 min versus -31.5 min for A0 static (cite [`reports/step2_truth_margin_vs_nist_hazards.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step2_truth_margin_vs_nist_hazards.csv), rows 5 and 2) because re-routing onto alternative roads did not bypass town-wide abandoned vehicles.
3. **NO Independent Real-Time Ground-Truth from Forward Simulation**: Teacher v2 physics forward simulation alone cannot predict road impassability on western exit corridors. Furthermore, causal observation assimilation ingests NIST empirical observations (roadblocks and satellite perimeters) that share an underlying source document with the empirical truth hazards.
4. **NO Production Readiness from Stand-in Weights**: Checkpoints evaluated in Phase 4A and 4B utilized `STAND-IN WEIGHTS` from a single smoke tile (85.5% burned) and cannot be claimed as validated surrogates.

---

## 5. Known Limitations

1. **HRRR Analysis vs Real-Time Forecast**: Atmospheric wind inputs use post-event HRRR analysis ($fxx=0$), not reanalysis or real-time operational weather forecasts.
2. **NIST Reconstructed Timestamps**: Empirical hazard onset times (roadblocks, burnovers) are reconstructed based on NIST TN 2252 supplementary records. Abandoned vehicles have a modelled constant onset at 87.0 min (harness assumption, not an observation).
3. **K=3 Corridor Bottleneck Clusters**: The road graph aggregates destination exits into 3 major corridor groups (Skyway, Clark Rd, Pentz/Hwy 70). Complex local off-corridor bypasses are not fully represented, and K=3 corridor clusters make bootstraps inconclusive.
4. **Teacher Physics Calibration Drift**: The teacher is a calibrated contact plus ember model; it under-burns the Camp Fire by about 2x and over-burns most training fires (cite [`reports/step6_ember_ablation_summary.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step6_ember_ablation_summary.csv), row 4: Mean Area Ratio 5.597 on training fires; [`reports/camp_fire_heldout_60m_areas.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/camp_fire_heldout_60m_areas.csv)).
5. **Development-Only Case Domain**: The Camp Fire (2018) is strictly a development scenario and is never included in surrogate training data to protect benchmark independence.
6. **Shared Provenance in Causal Assimilation**: Causal observation assimilation ingests observed perimeters and roadblocks derived from the same NIST post-incident reconstruction as the evaluation hazards.
7. **Provider B Risk Cone Heuristic**: Agent B utilizes a geometric directional cone and distance heuristic rather than an integrated microscopic ember trajectory simulator.
8. **Vehicles Finish Before Fire Arrives in Physics Simulation**: Evacuation vehicles complete travel before the simulated fire reaches Paradise (smallest positive predicted margin 149.8 min; cite [`reports/step5c_route_safety_margins.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step5c_route_safety_margins.csv)).
9. **Route Redundancy**: Agents A and A2-pure produce identical routes in every comparative table.
10. **Surrogate Domain Over-Burn & Architecture Constraints**: The Phase 4A surrogate output covered the whole domain (IoU 0.1942 equals observed area over domain area) and comes from an untrained model; the surrogate is single-ignition and the Camp Fire ignition lies outside its tile; Phase 4B's A3-reignite burned 85.5 percent of its tile.
11. **No Held-Out Teacher Test**: No held-out teacher fire test was conducted.
12. **No Trained Surrogate Result**: No trained surrogate result is claimed unless a Kaggle or local checkpoint is reported in an appendix.
13. **Retracted Figures**: Phase 4B's narrative 24.1 ms forward pass and 56.4x speedup are retracted as unmeasured estimates. The measured numbers are 34.51 ms per tile (39.39x tile speedup; cite [`reports/step4_latency_summary.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step4_latency_summary.csv)) and 310.59 ms domain coverage (8.85x domain speedup; cite [`reports/step4_latency_summary.csv`](file:///Users/miteshsingh/Documents/projects/wf052/reports/step4_latency_summary.csv)).

---

## 6. Appendix: Provenance, Checksums & File Paths

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
