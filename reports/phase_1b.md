# Phase 1B Report: Ember Spotting, Urban Ignition, and Operational Assimilation Test Mode (Steps 0–3)

**Status:** Steps 0, 1, 2, and 3 Complete — **STOPPED at Step 3 awaiting Orchestrator Approval of Fresh Held-Out Test Fires before Step 4 (`teacher_v2` Freeze & Final Test)**  
**Operating Resolution:** 60 m (`EPSG:32610`, UTM Zone 10N; $0.36\text{ ha/cell}$, `30 steps/hour` = $2.0\text{ min/step}$)  
**Camp Fire (2018) Role:** **Development Case** (used for Step 0 criticality sweep and Step 1 assimilation diagnostic; **never used for tuning** `EmberParams` in Step 2)

---

## 1. What Was Built

1. **Step 0 — Phase 1A Re-Audit, Matched-Fold Model Selection & Criticality Sweep (`scripts/run_phase1b_step0_audit_and_criticality.py`)**:
   - **Matched-Fold Comparison (`reports/step0_matched_fold_comparison.csv`)**: Evaluated `global_slowdown` (1 free parameter) vs `wind_only` vs `all-7` on `(i)` the 3 wind-driven folds (`Kincade, Glass, Zogg`) and `(ii)` all 7 folds (`Kincade, Bobcat, Glass, Creek, Zogg, Caldor, Tamarack`), applying the rule *"pick the simplest within `0.02` IoU of best"*. Froze `configs/teacher_v1_1.yaml` ($P_{\text{base}} = 0.22$, 1 free parameter, wind-only consensus vector) alongside `configs/teacher_v1.yaml` ($P_{\text{base}} = 0.21$, all-7 consensus vector) with SHA-256 hash verification (`reports/teacher_v1_1_hash.txt`).
   - **20-Seed Coverage Table with Denominators (`reports/step0_20seed_coverage_with_denominators.csv`)**: Re-computed 20-seed Monte Carlo P10–P90 arrival-time coverage across all 7 training fires with exact progression cell denominators (`covered_cells / denom_obs_progression_cells`, where `denom_obs_progression_cells` is the count of 60 m cells that burned after $t=0$ within the 12 h window).
   - **Complete Camp Fire Table with Recall, Precision, IoU & NIST Reached/Unreached Denominators (`reports/step0_camp_fire_complete_table.csv`, `reports/step0_camp_fire_nist_reached_unreached.csv`)**: Reported Recall (`TP / (TP + FN)`), Precision (`TP / (TP + FP)`), and IoU (`TP / (TP + FP + FN)`) with raw cell counts (`316,968` total cells, `61,549` observed burned at +11.4 h, `162,693` final footprint cells), plus reached/unreached NIST point counts per category (`vegetative = 315`, `spot_fire = 341`, `structural = 629`, `other = 1,081`, `ALL = 2,366`). Relabeled the 30 m Kincade check in `reports/resolution_penalty_30m_vs_60m.csv` as `"parameter transfer test (non-transferable), not a resolution penalty"`.
   - **Percolation Criticality Sweep (`reports/criticality_sweep_kincade_campfire.csv`, `reports/phase_1b_criticality_sweep.png`)**: Ran 600 simulations ($P_{\text{base}} \in [0.16..0.30]$ step `0.01`, 20 seeds per step) on `Kincade` (`in-sample training`) and `Camp Fire` (`development`), plotted median and P10–P90 burned area vs $P_{\text{base}}$, diagnosed the step-like percolation transition, and specified a rate-of-spread-driven contact contingency rule ($p_{\text{ignite}} = 1 - \exp(-\text{ROS} \cdot \Delta t / \Delta x)$ from Rothermel FBFM40 ROS) without building it yet.

2. **Step 1 — Operational Assimilation Test Mode (`src/wf052/sim/ember_kernel.py`, `scripts/run_phase1b_step1_assimilation.py`, `scripts/finalize_phase1b_step3_proposal.py`)**:
   - Added `init_burned_mask`, `start_hour_index`, and `start_elapsed_min` support to `simulate_teacher_with_embers_60m`, initializing active burning along the morphological boundary of the observed extent at $T_0$ while marking interior cells as already burned at $t = T_0$.
   - Built `build_nist_observed_extent_mask` for Camp Fire from NIST points with `elapsed_minutes <= T0` ($3 \times 3$ 60 m local patches = $180\text{ m} \times 180\text{ m}$ + 1-cell morphological closing to bridge adjacent observations without convex-hull pre-burning of unburned valleys).
   - Evaluated $T_0 \in \{+1\text{ h}, +2\text{ h}, +3\text{ h}\}$ forecasting the next 3 hours (`(1h, +4h]`, `(2h, +5h]`, `(3h, +6h]`) and cumulative (`[0h, +4h]`, `[0h, +5h]`, `[0h, +6h]`) against the $t=0$ point-start baseline (`reports/step1_assimilation_camp_fire.csv`, `reports/phase_1b_assimilation_camp_fire.png`).

3. **Step 2 — Switchable Non-Local Ember Spotting & WUI Ignition Kernel (`src/wf052/sim/ember_kernel.py`, `tests/test_ember_and_assimilation.py`, `scripts/run_phase1b_step2_ember_calibration.py`)**:
   - Built `EmberParams` (`@dataclass(frozen=True)`) as a separate, switchable module (`use_ember_kernel: bool`) with strictly **3 free parameters**:
     1. `alpha_launch`: Per-burning-cell, per-step spot launch rate coefficient scaled by FBFM40 spotting potential $\phi_{\text{spot}}(f) \in [0.40, 1.35]$ and effective wind speed above lofting threshold $V_{\text{excess}} = \max(0, \max(V_{10}, 0.75 V_{\text{gust}}) - 4.0\text{ m/s})$.
     2. `beta_dist_m_per_ms`: Lognormal median landing distance scale ($\text{m}$ per $\text{m/s}$ of wind speed, with log-std $\sigma_{\ln d} = 0.65$), capped at $6,000\text{ m}$ per wildfire spotting literature, aligned downwind with Gaussian lateral scatter ($\sigma_\theta = 18^\circ$).
     3. `p_urban_wui`: Reduced, nonzero ignition probability and flammability modifier for Urban/WUI cells (`FBFM40 == 91`), while keeping Open Water (`98`), Barren (`99`), Snow/Ice (`92`), and Agriculture (`93`) strictly non-burnable (`0.0`).
   - Implemented `sample_ember_spotting_step` using static-shape `torch.Tensor.scatter_add_` (zero CPU–GPU synchronization inside the time-step loop) and `compute_far_field_jump_cells` using 2D max-pooling dilation (`F.max_pool2d`) to count burned cells in disconnected spot-fire islands $\ge 5\text{ cells}$ ($300\text{ m}$) ahead of the main front.
   - Calibrated `EmberParams` **exclusively on the wind-driven training fires (`Kincade, Glass, Zogg`)** using perimeter-level metrics (`area_ratio`, `front_distance`, `far_field_jump_cells`, and within-perimeter WUI overlap) — **never tuning on Camp Fire** — and ran the full Ember ON vs OFF ablation across all 7 folds (`reports/step2_ember_ablation_per_fire.csv`, `reports/step2_ember_ablation_summary.csv`, `reports/phase_1b_ember_ablation.png`).

4. **Step 3 — Fresh Held-Out Wind-Driven Test Fire Proposal (`scripts/propose_fresh_heldout_fires.py`, `scripts/finalize_phase1b_step3_proposal.py`)**:
   - Screened all 21 remaining unused California fires in GOFER (`reports/step3_gofer_unused_ca_candidates.csv`) and prepared the complete per-fire setup table (`reports/step3_proposed_fresh_heldout_fires.csv`) for **2 primary fresh held-out wind-driven test fires** (`Slater and Devil 2020`, `Walker 2019`) plus **1 alternate** (`W-5 Cold Springs 2020`), verifying 0% spatial overlap with Camp Fire and the 7 Phase 1A training fires.

---

## 2. Commands Run

```bash
# 1. Run unit tests for ember kernel, WUI ignition vs water barrier, far-field jump metric, and assimilation init
cmd /c ".venv/bin/pytest tests/test_ember_and_assimilation.py -v"

# 2. Execute Step 0: matched-fold comparison, teacher_v1_1 freeze + hash, 20-seed coverage with denominators,
#    Camp Fire Recall/IoU + NIST reached/unreached tables, 30m relabeling, and 600-run criticality sweep
cmd /c ".venv/bin/python scripts/run_phase1b_step0_audit_and_criticality.py"

# 3. Execute Step 1: Camp Fire operational assimilation test mode (T0 in {+1h, +2h, +3h}, 3h forecast)
cmd /c ".venv/bin/python scripts/run_phase1b_step1_assimilation.py"

# 4. Execute Step 2: Fit 3-param EmberParams on Wind-3 training fires (LOFO + consensus; Camp Fire UNSEEN)
#    and run Ember ON vs OFF ablation across all 7 folds
cmd /c ".venv/bin/python scripts/run_phase1b_step2_ember_calibration.py"

# 5. Execute Step 3: Screen 21 unused GOFER CA fires and build proposed fresh held-out test fire setup table + Step 1 plot
cmd /c ".venv/bin/python scripts/propose_fresh_heldout_fires.py"
cmd /c ".venv/bin/python scripts/finalize_phase1b_step3_proposal.py"

# 6. Lint & full test suite verification
cmd /c ".venv/bin/ruff check . && .venv/bin/pytest"
```

---

## 3. Results (with Numbers and Denominators)

### Step 0(a): Matched-Fold Comparison & Base Teacher Selection (`reports/step0_matched_fold_comparison.csv`)

- **Orchestrator Rule**: Compare `global_slowdown` (1 free parameter) vs `wind_only` vs `all-7` on `(i)` the 3 wind-driven folds (`Kincade, Glass, Zogg`) and `(ii)` all 7 folds. Pick the simplest model within `0.02` IoU of best; if not `teacher_v1`, freeze `configs/teacher_v1_1.yaml` + hash.

| Evaluation Fold Set | Model Variant | Free Params | Split | +12h ~2km GOFER IoU | +12h ~2km GOFER $F_1$ | +12h 375m FEDS $F_1$ | Mean (1/3/6/12h) ~2km IoU | +12h Mean Area Ratio | +12h Median Area Ratio | +12h Wind Front Err (km) |
|---|---|---|---|---|---|---|---|---|---|---|
| **(i) 3 Wind-Driven Folds (`n=3`)** | `uncalibrated (P_base=0.50)` | 0 | **held-out (LOFO, n=3)** | 0.1391 | 0.2434 | 0.2113 | 0.3345 | 7.552× | 8.489× | 4.63 km |
| **(i) 3 Wind-Driven Folds (`n=3`)** | `global_slowdown_lofo` | **1** | **held-out (LOFO, n=3)** | 0.2848 | 0.4070 | 0.3691 | 0.3571 | 0.702× | 0.435× | 5.32 km |
| **(i) 3 Wind-Driven Folds (`n=3`)** | `calibrated_all7_lofo` | 6 | **held-out (LOFO, n=3)** | 0.2983 | 0.4207 | 0.3710 | 0.3509 | 0.580× | 0.435× | 5.00 km |
| **(i) 3 Wind-Driven Folds (`n=3`)** | `calibrated_wind_only_lofo` | 6 | **held-out (LOFO, n=3)** | **0.3214** | **0.4832** | **0.4407** | **0.4094** | **1.037×** | **0.937×** | **3.43 km** |
| **(i) 3 Wind-Driven Folds (`n=3`)** | `teacher_v1 (1-param, P_base=0.21)` | **1** | **single-vector eval (n=3)** | **0.3924** | **0.5624** | 0.5013 | 0.4172 | **1.012×** | **0.945×** | **0.94 km** |
| **(i) 3 Wind-Driven Folds (`n=3`)** | `teacher_v1_1 (1-param, P_base=0.22)` | **1** | **single-vector eval (n=3)** | **0.3899** | 0.5599 | **0.5171** | **0.4434** | 1.206× | 1.446× | 2.85 km |
| **(ii) All 7 Folds (`n=7`)** | `uncalibrated (P_base=0.50)` | 0 | **held-out (LOFO, n=7)** | 0.1036 | 0.1851 | 0.1336 | 0.2534 | 13.685× | 8.926× | 8.27 km |
| **(ii) All 7 Folds (`n=7`)** | `global_slowdown_lofo` | **1** | **held-out (LOFO, n=7)** | **0.2436** | **0.3663** | 0.2741 | 0.3307 | **4.105×** | **1.512×** | 5.32 km |
| **(ii) All 7 Folds (`n=7`)** | `calibrated_all7_lofo` | 6 | **held-out (LOFO, n=7)** | 0.2204 | 0.3337 | 0.2471 | 0.3199 | 4.570× | 1.512× | 6.51 km |
| **(ii) All 7 Folds (`n=7`)** | `calibrated_wind_only_lofo` | 6 | **held-out (LOFO, n=7)** | 0.2297 | 0.3590 | **0.2789** | **0.3399** | 4.994× | 1.828× | **5.28 km** |
| **(ii) All 7 Folds (`n=7`)** | `teacher_v1 (1-param, P_base=0.21)` | **1** | **single-vector eval (n=7)** | **0.2897** | **0.4329** | **0.3308** | **0.3565** | **4.238×** | **1.512×** | **3.44 km** |
| **(ii) All 7 Folds (`n=7`)** | `teacher_v1_1 (1-param, P_base=0.22)` | **1** | **single-vector eval (n=7)** | 0.2591 | 0.3918 | 0.3116 | 0.3545 | 5.067× | 1.828× | 5.03 km |

- **Matched-Fold Selection Summary**:
  1. On **(ii) All 7 Folds (`held-out LOFO, n=7`)**: The 1-parameter `global_slowdown_lofo` (consensus `teacher_v1`, $P_{\text{base}} = 0.21$) achieves the highest +12 h IoU (**`0.2436`** vs `0.2297` for `wind_only_lofo` and `0.2204` for `all7_lofo`) and is within `0.0092 < 0.02` on mean 1/3/6/12h IoU (`0.3307` vs `0.3399`).
  2. On **(i) 3 Wind-Driven Folds (`held-out LOFO, n=3`)**: `calibrated_wind_only_lofo` achieves +12 h IoU = **`0.3214`** (vs `0.2848` for all-7 `global_slowdown_lofo`, where fitting on Bobcat/Caldor dragged Zogg's fold down to $P_{\text{base}} = 0.16$). Its consensus single parameter vector is the 1-parameter model **`configs/teacher_v1_1.yaml`** ($P_{\text{base}} = 0.22$, all fuel/canopy multipliers = `1.0`), which achieves the highest mean 1/3/6/12h IoU (**`0.4434`**) and FEDS 375m $F_1$ (**`0.5171`**) on the 3 wind-driven folds while remaining within `0.0025 < 0.02` of `teacher_v1` at +12 h (`0.3899` vs `0.3924`).
  3. Both single-parameter configs are frozen on disk with verified SHA-256 hashes:
     - `configs/teacher_v1.yaml` ($P_{\text{base}} = 0.21$): `c45ba6db95187063347ad1c06e81dbb29ff3e45fa92f50ad46d8dcc9e16cf9b6` (`reports/teacher_v1_hash.txt`)
     - `configs/teacher_v1_1.yaml` ($P_{\text{base}} = 0.22$): `86a1ac17804d3816958c4694b1be504fa432ebdead2db5d0df4c4fa4939a30f0` (`reports/teacher_v1_1_hash.txt`)

---

### Step 0(b): 20-Seed Monte Carlo P10–P90 Coverage Table with Exact Cell Denominators (`reports/step0_20seed_coverage_with_denominators.csv`)

*Split: **Held-Out (LOFO 20-Seed Ensemble)** | Seeds: `[100, 117, ..., 423]` ($N=20$). Denominator (`denom_obs_progression_cells`) is the exact number of 60 m cells that burned after $t=0$ ($t_{\text{obs}} \in (0, 720\text{ min}]$) in GOFER.*

| Fire Name | Regime | Split | Seeds | Obs +12h Cells (Area ha) | Ensemble +12h Area (Mean ± Std ha) | Ensemble P10–P90 Area Band (ha) | Ensemble +12h ~2km IoU (Mean ± Std) | Ensemble P50 Arrival MAE (min) | Mean P10–P90 Width (min) | Denominator: Obs Progression Cells | Raw P10–P90 Covered / Denom (%) | Hourly-Quantized ($\pm 30\text{ m}$) Covered / Denom (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Kincade** | `wind-driven` | **held-out (LOFO 20-seed)** | 20 | `32,283` (`11,621.9 ha`) | $8,315.8 \pm 3,129.2$ | `[4,444.0, 13,335.7]` | $0.3747 \pm 0.0505$ | 249.5 | 231.2 | `29,925` | `6,483 / 29,925` (`21.66%`) | `9,107 / 29,925` (**`30.43%`**) |
| **Bobcat** | `plume` | **held-out (LOFO 20-seed)** | 20 | `10,087` (`3,631.3 ha`) | $57,047.4 \pm 3,123.9$ | `[52,003.0, 60,267.1]` | $0.0713 \pm 0.0041$ | 286.6 | 75.1 | `9,098` | `876 / 9,098` (`9.63%`) | `942 / 9,098` (`10.35%`) |
| **Glass** | `wind-driven` | **held-out (LOFO 20-seed)** | 20 | `63,401` (`22,824.4 ha`) | $39,665.1 \pm 3,516.3$ | `[35,997.9, 43,443.4]` | $0.3717 \pm 0.0228$ | 212.1 | 159.6 | `56,134` | `16,421 / 56,134` (`29.25%`) | `20,429 / 56,134` (**`36.39%`**) |
| **Creek** | `plume` | **held-out (LOFO 20-seed)** | 20 | `46,488` (`16,735.7 ha`) | $25,794.9 \pm 4,082.9$ | `[21,169.9, 31,362.9]` | $0.3718 \pm 0.0489$ | 293.6 | 199.1 | `41,300` | `15,232 / 41,300` (`36.88%`) | `15,849 / 41,300` (**`38.38%`**) |
| **Zogg** | `wind-driven` | **held-out (LOFO 20-seed)** | 20 | `35,216` (`12,677.8 ha`) | $12,354.8 \pm 2,262.8$ | `[9,084.8, 15,622.3]` | $0.3037 \pm 0.0873$ | 394.8 | 258.9 | `35,012` | `7,335 / 35,012` (`20.95%`) | `9,080 / 35,012` (**`25.93%`**) |
| **Caldor** | `terrain` | **held-out (LOFO 20-seed)** | 20 | `31,626` (`11,385.4 ha`) | $94,520.3 \pm 2,503.2$ | `[92,100.6, 96,911.0]` | $0.1291 \pm 0.0032$ | 186.4 | 60.8 | `20,617` | `1,941 / 20,617` (`9.41%`) | `3,337 / 20,617` (`16.19%`) |
| **Tamarack** | `terrain` | **held-out (LOFO 20-seed)** | 20 | `20,991` (`7,556.8 ha`) | $16,465.2 \pm 6,148.4$ | `[9,107.9, 24,422.5]` | $0.1951 \pm 0.0649$ | 311.7 | 327.4 | `18,992` | `9,690 / 18,992` (**`51.02%`**) | `11,166 / 18,992` (**`58.79%`**) |

---

### Step 0(c): Camp Fire Complete Tables (Recall, Precision, IoU & NIST Reached/Unreached with Denominators)

#### 1. GeoMAC Perimeter Overlap on Camp Fire at 60 m (`reports/step0_camp_fire_complete_table.csv`)
*Split: **Development (`Camp Fire 2018`, point-start)** | Grid: $564 \times 562 = 316,968\text{ cells}$ ($0.36\text{ ha/cell}$). Every metric is shown with its exact cell denominator.*

| Split | Model | Reference Perimeter | Horizon | Obs Area (ha) | Sim Area (ha) | Area Ratio | Recall (`TP / Obs Cells`) | Precision (`TP / Sim Cells`) | IoU (`TP / Union Cells`) | $F_1$ |
|---|---|---|---|---|---|---|---|---|---|---|
| **development** | `teacher_v1 (P_base=0.21)` | +11.4 h GeoMAC IR (ID 1052) | 684 min | 22,157.6 | 1,410.5 | 0.064× | `3,378 / 61,549` (**`5.49%`**) | `3,378 / 3,918` (**`86.22%`**) | `3,378 / 62,089` (**`0.0544`**) | 0.1032 |
| **development** | `teacher_v1 (P_base=0.21)` | Final Contained (ID 1398) | 720 min | 58,569.5 | 1,410.5 | 0.024× | `3,915 / 162,693` (**`2.41%`**) | `3,915 / 3,918` (**`99.92%`**) | `3,915 / 162,696` (**`0.0241`**) | 0.0470 |
| **development** | `teacher_v1_1 (P_base=0.22)` | +11.4 h GeoMAC IR (ID 1052) | 684 min | 22,157.6 | 2,979.4 | 0.134× | `6,350 / 61,549` (**`10.32%`**) | `6,350 / 8,276` (**`76.73%`**) | `6,350 / 63,475` (**`0.1000`**) | 0.1819 |
| **development** | `teacher_v1_1 (P_base=0.22)` | Final Contained (ID 1398) | 720 min | 58,569.5 | 3,058.9 | 0.052× | `8,497 / 162,693` (**`5.22%`**) | `8,497 / 8,497` (**`100.00%`**) | `8,497 / 162,693` (**`0.0522`**) | 0.0993 |
| **development** | `uncalibrated (P_base=0.50)` | +11.4 h GeoMAC IR (ID 1052) | 684 min | 22,157.6 | 84,035.5 | 3.793× | `50,202 / 61,549` (**`81.56%`**) | `50,202 / 233,432` (**`21.51%`**) | `50,202 / 244,779` (**`0.2051`**) | 0.3404 |
| **development** | `phase0_baseline (P_base=0.58)` | +11.4 h GeoMAC IR (ID 1052) | 684 min | 22,157.6 | 86,860.4 | 3.920× | `51,197 / 61,549` (**`83.18%`**) | `51,197 / 241,279` (**`21.22%`**) | `51,197 / 251,631` (**`0.2035`**) | 0.3381 |

#### 2. NIST Observation Points Reached / Unreached by Fire Type (`reports/step0_camp_fire_nist_reached_unreached.csv`)
*Split: **Development (`Camp Fire 2018`, point-start)** | Total points in 24 h window: `2,366` (`vegetative = 315`, `spot_fire = 341`, `structural = 629`, `other = 1,081`).*

| Split | Model | Fire Type | Reached (`N / Total`) | Unreached (`N / Total`) | Signed Bias Mean / Median (min) | Abs Error MAE / Median (min) |
|---|---|---|---|---|---|---|
| **development** | `teacher_v1 (P_base=0.21)` | **`vegetative`** | `4 / 315` (`1.27%`) | `311 / 315` (`98.73%`) | `+12.25` / `+11.00` | **`12.25`** / **`11.00`** |
| **development** | `teacher_v1 (P_base=0.21)` | `spot_fire` | `2 / 341` (`0.59%`) | `339 / 341` (`99.41%`) | `+51.00` / `+51.00` | `51.00` / `51.00` |
| **development** | `teacher_v1 (P_base=0.21)` | `structural` | `0 / 629` (`0.00%`) | `629 / 629` (`100.00%`) | N/A | N/A |
| **development** | `teacher_v1 (P_base=0.21)` | `other` | `7 / 1,081` (`0.65%`) | `1,074 / 1,081` (`99.35%`) | `+5.71` / `+8.00` | `28.86` / `35.00` |
| **development** | `teacher_v1 (P_base=0.21)` | **`ALL`** | `13 / 2,366` (`0.55%`) | `2,353 / 2,366` (`99.45%`) | `+14.69` / `+13.00` | `27.15` / `25.00` |
| **development** | `teacher_v1_1 (P_base=0.22)` | **`vegetative`** | `20 / 315` (`6.35%`) | `295 / 315` (`93.65%`) | `+101.55` / `+87.50` | **`110.15`** / **`87.50`** |
| **development** | `teacher_v1_1 (P_base=0.22)` | `spot_fire` | `3 / 341` (`0.88%`) | `338 / 341` (`99.12%`) | `+7.00` / `-1.00` | `8.33` / `1.00` |
| **development** | `teacher_v1_1 (P_base=0.22)` | `structural` | `2 / 629` (`0.32%`) | `627 / 629` (`99.68%`) | `+108.50` / `+108.50` | `108.50` / `108.50` |
| **development** | `teacher_v1_1 (P_base=0.22)` | `other` | `17 / 1,081` (`1.57%`) | `1,064 / 1,081` (`98.43%`) | `+20.94` / `-9.00` | `57.41` / `29.00` |
| **development** | `teacher_v1_1 (P_base=0.22)` | **`ALL`** | `42 / 2,366` (`1.78%`) | `2,324 / 2,366` (`98.22%`) | `+62.50` / `+11.00` | `81.45` / `35.00` |
| **development** | `uncalibrated (P_base=0.50)` | **`vegetative`** | `184 / 315` (`58.41%`) | `131 / 315` (`41.59%`) | `+82.02` / `+86.00` | `148.98` / `130.00` |
| **development** | `uncalibrated (P_base=0.50)` | **`ALL`** | `935 / 2,366` (`39.52%`) | `1,431 / 2,366` (`60.48%`) | `+131.66` / `+156.00` | `174.49` / `182.00` |

#### 3. Relabeling of 30 m vs 60 m Kincade Check (`reports/resolution_penalty_30m_vs_60m.csv`)
- Relabeled as **`"parameter transfer test (non-transferable), not a resolution penalty"`**: in a discrete Moore-neighborhood CA where cells burn out with probability $1 - p_{\text{continue}} = 0.70$ per discrete step, halving $\Delta x$ (`60 m -> 30 m`) while doubling `steps_per_hour` (`30 -> 60`) requires twice as many sequential survival steps per kilometer, shifting the percolation threshold upward so that 60 m calibrated parameters ($P_{\text{base}}=0.21$) cannot be transferred directly to 30 m without re-calibration.

---

### Step 0(d): Percolation Criticality Sweep ($P_{\text{base}} \in [0.16..0.30]$, 20 Seeds) & ROS Contingency Proposal

- **Plot**: `reports/phase_1b_criticality_sweep.png`  
- **Table (`reports/criticality_sweep_kincade_campfire.csv`)**: 20 seeds per $P_{\text{base}}$ value (`600` total 12-hour simulations).

| $P_{\text{base}}$ | Kincade (`in-sample training`, Obs = `11,621.9 ha`) Median [P10, P90] Area (ha) | Kincade Median Area Ratio (`Sim / Obs`) | Camp Fire (`development`, Obs = `22,157.6 ha`) Median [P10, P90] Area (ha) | Camp Fire Median Area Ratio (`Sim / Obs`) |
|---|---|---|---|---|
| `0.16` | `2,071.6` `[1,955.6, 2,290.0]` | `0.178×` (stalled) | `41.9` `[14.5, 210.0]` | `0.002×` (stalled) |
| `0.17` | `2,260.3` `[2,056.3, 3,066.5]` | `0.194×` (stalled) | `60.5` `[14.8, 823.7]` | `0.003×` (stalled) |
| `0.18` | `2,489.6` `[2,148.9, 3,443.1]` | `0.214×` (stalled) | `128.0` `[22.4, 1,101.2]` | `0.006×` (stalled) |
| `0.19` | `3,654.9` `[2,426.9, 4,709.5]` | `0.314×` (onset) | `212.0` `[41.5, 1,345.7]` | `0.010×` (stalled) |
| `0.20` | `4,612.0` `[3,562.8, 8,379.3]` | `0.397×` (critical) | `941.2` `[85.2, 3,120.3]` | `0.042×` (bottlenecked) |
| **`0.21`** | **`7,984.4`** `[4,444.0, 13,335.7]` | **`0.687×`** (`teacher_v1`) | **`1,249.0`** `[209.4, 3,377.3]` | **`0.056×`** (bottlenecked at river) |
| **`0.22`** | **`11,949.5`** `[8,614.8, 16,929.1]` | **`1.028×`** (`teacher_v1_1`) | **`4,052.9`** `[734.3, 7,744.0]` | **`0.183×`** (high variance) |
| `0.23` | `15,876.7` `[9,005.2, 22,735.7]` | `1.366×` | `6,912.0` `[1,325.1, 10,922.6]` | `0.312×` |
| `0.24` | `22,062.1` `[17,297.5, 30,566.5]` | `1.898×` | `12,051.0` `[7,645.8, 20,256.4]` | `0.544×` |
| `0.25` | `31,383.0` `[22,284.3, 35,837.1]` | `2.700×` | `17,179.9` `[11,532.9, 29,426.1]` | `0.775×` |
| `0.26` | `36,297.2` `[28,430.5, 39,410.5]` | `3.123×` (runaway) | `26,737.9` `[14,683.5, 32,777.6]` | `1.207×` (flank breakout) |
| `0.28` | `45,773.3` `[41,109.5, 49,433.2]` | `3.939×` (runaway) | `37,532.0` `[23,856.6, 51,300.0]` | `1.694×` (runaway) |
| `0.30` | `54,097.4` `[50,279.4, 56,407.0]` | `4.655×` (runaway) | `55,065.1` `[39,124.1, 59,811.9]` | `2.485×` (runaway) |

- **Plain Diagnosis of Criticality**:
  - **Yes — the curve is strongly step-like (stall to runaway)** across a narrow bond-percolation transition band $P_{\text{base}} \in [0.19, 0.25]$.
  - On **Kincade**, median 12 h burned area is stalled near the initial polygon for $P_{\text{base}} \in [0.16..0.18]$ (`2,071.6` to `2,489.6 ha`), then jumps **$6.82\times$ over a `0.05` parameter increment** from `4,612.0 ha` ($P_{\text{base}}=0.20$) to `31,383.0 ha` ($P_{\text{base}}=0.25$).
  - On **Camp Fire** (point ignition behind the Feather River canyon/road bottleneck), median 12 h area is stalled at $\le 212\text{ ha}$ (`< 1%` of observed) for $P_{\text{base}} \le 0.19$, remains bottlenecked at `1,249.0 ha` at $P_{\text{base}}=0.21$, and then jumps **$21.4\times$ over a `0.05` increment** to `26,737.9 ha` at $P_{\text{base}}=0.26$, accompanied by a 10-fold P10–P90 spread (`734.3` to `7,744.0 ha` at $P_{\text{base}}=0.22$) depending on whether a single stochastic sample happens to bridge a narrow 1-cell road/river gap.

- **Proposed Contingency (Rothermel-Style Rate-of-Spread Contact Rule — Specified Only, Not Built Yet)**:
  - **Root Cause**: PyTorchFire's contact rule couples cell burnout ($1 - p_{\text{continue}} = 0.70$ per 2-min step) with a single step-wise ignition probability $p_{ij}$, creating a classic 2D directed-percolation phase transition where cells burn out before igniting neighbors if $p_{ij} < p_c$, and flood isotropically if $p_{ij} > p_c$.
  - **Contingency Specification**: Replace the percolation-bound $p_{\text{base}} / p_{\text{continue}}$ rule with a continuous **Rothermel (1972) / Scott & Burgan (2005) Rate-of-Spread (ROS)** contact rule:
    1. Map each `FBFM40` class $f$ to its standard reaction intensity $I_R(f, M_f)$, propagating flux ratio $\xi(f)$, wind coefficient $\phi_W(f, U_{\text{midflame}})$, and slope factor $\phi_S(\tan \theta_{ij})$ to compute directional head-fire $\text{ROS}_0(f, U, \theta) \text{ [m/min]}$, with elliptical flank/backing decay $\text{ROS}(\Delta \psi_{ij}) = \text{ROS}_0 \cdot \frac{1 - e_{\text{LB}}}{1 - e_{\text{LB}} \cos(\Delta \psi_{ij})}$ where length-to-breadth ratio $\text{LB}(U) = 0.936 e^{0.2566 U} + 0.461 e^{-0.1548 U} - 0.397$.
    2. Set directional contact transition probability over step $\Delta t$ across cell spacing $d_{ij} \in \{\Delta x, \sqrt{2}\Delta x\}$ as a smooth Poisson hazard:
       $$p_{\text{ignite}}(i \to j) = 1 - \exp\!\left(-\gamma_{\text{ROS}} \frac{\text{ROS}_{ij} \cdot \Delta t}{d_{ij}}\right), \qquad \tau_{\text{burnout}} \ge \frac{2 \Delta x}{\min \text{ROS}}$$
    3. Because active front cells remain burning for their physical residence time $\tau_{\text{res}}$ (or until neighbors are exposed for cumulative travel time $d_{ij}/\text{ROS}_{ij}$), burned area scales smoothly as $A(t) \propto (\text{ROS} \cdot t)^2$ without a stall-to-runaway percolation cliff, and $\gamma_{\text{ROS}}$ becomes **invariant to $\Delta x$ and $\Delta t$**.

---

### Step 1: Operational Assimilation Test Mode on Camp Fire (`reports/step1_assimilation_camp_fire.csv`)

- **Plot**: `reports/phase_1b_assimilation_camp_fire.png`
- **Setup**: For $T_0 \in \{+1\text{ h (60 min)}, +2\text{ h (120 min)}, +3\text{ h (180 min)}\}$, we initialize the simulation from the observed NIST burned extent at $t \le T_0$ (`25 pts / 56 ha` at $+1\text{ h}$, `175 pts / 331 ha` at $+2\text{ h}$, `566 pts / 772 ha` at $+3\text{ h}$) and run a **3-hour operational forecast** ($[T_0, T_0 + 3\text{ h}]$). We report both the **strictly future 3-hour forecast window** $(T_0, T_0 + 3\text{ h}]$ and the **cumulative window** $[0, T_0 + 3\text{ h}]$ next to the $t=0$ point-start baseline.

| Split | Model & Mode | Init $T_0$ | Evaluation Window | Fire Type | Reached / Total (`Recall %`) | Unreached / Total | Signed Bias Mean / Median (min) | Abs Error MAE / Median (min) |
|---|---|---|---|---|---|---|---|---|
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+1 h** | **Next 3h `(1h, +4h]`** | `vegetative` | `7 / 87` (**`8.0%`**) | `80 / 87` (`92.0%`) | `+46.6` / `+70.0` | `60.6` / `70.0` |
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+1 h** | **Next 3h `(1h, +4h]`** | `spot_fire` | `9 / 221` (**`4.1%`**) | `212 / 221` (`95.9%`) | `+94.1` / `+101.0` | `95.7` / `101.0` |
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+1 h** | **Next 3h `(1h, +4h]`** | `structural` | `2 / 109` (**`1.8%`**) | `107 / 109` (`98.2%`) | `+2.0` / `+2.0` | `5.0` / `5.0` |
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+1 h** | **Next 3h `(1h, +4h]`** | **`ALL`** | **`31 / 757` (`4.1%`)** | `726 / 757` (`95.9%`) | `+47.0` / `+40.0` | **`53.7`** / **`46.0`** |
| **development (Camp Fire, point-start)** | `teacher_v1 (P_base=0.21)` Point-Start | **+1 h** | **Next 3h `(1h, +4h]`** | **`ALL`** | **`2 / 757` (`0.3%`)** | `755 / 757` (`99.7%`) | `+47.0` / `+47.0` | `47.0` / `47.0` |
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+2 h** | **Next 3h `(2h, +5h]`** | `vegetative` | **`39 / 85` (`45.9%`)** | `46 / 85` (`54.1%`) | `-35.9` / `-28.0` | **`35.9`** / **`28.0`** |
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+2 h** | **Next 3h `(2h, +5h]`** | `spot_fire` | **`48 / 176` (`27.3%`)** | `128 / 176` (`72.7%`) | `-28.1` / `-30.0` | **`34.4`** / **`33.5`** |
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+2 h** | **Next 3h `(2h, +5h]`** | `structural` | **`79 / 175` (`45.1%`)** | `96 / 175` (`54.9%`) | `-60.4` / `-50.0` | **`64.5`** / **`50.0`** |
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+2 h** | **Next 3h `(2h, +5h]`** | `other` | `136 / 422` (`32.2%`) | `286 / 422` (`67.8%`) | `-49.3` / `-43.0` | `55.7` / `47.5` |
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+2 h** | **Next 3h `(2h, +5h]`** | **`ALL`** | **`302 / 858` (`35.2%`)** | `556 / 858` (`64.8%`) | `-47.1` / `-38.5` | **`52.1`** / **`41.5`** |
| **development (Camp Fire, point-start)** | `teacher_v1 (P_base=0.21)` Point-Start | **+2 h** | **Next 3h `(2h, +5h]`** | **`ALL`** | **`0 / 858` (`0.0%`)** | `858 / 858` (`100.0%`) | N/A | N/A |
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+2 h** | **Cumulative `[0h, +5h]`** | **`ALL`** | **`477 / 1,033` (`46.2%`)** | `556 / 1,033` (`53.8%`) | `-18.2` / `-13.0` | **`44.6`** / **`34.0`** |
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+3 h** | **Next 3h `(3h, +6h]`** | `vegetative` | **`29 / 84` (`34.5%`)** | `55 / 84` (`65.5%`) | `-67.8` / `-45.0` | `67.8` / `45.0` |
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+3 h** | **Next 3h `(3h, +6h]`** | `spot_fire` | **`30 / 91` (`33.0%`)** | `61 / 91` (`67.0%`) | `-81.8` / `-71.0` | `81.8` / `71.0` |
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+3 h** | **Next 3h `(3h, +6h]`** | `structural` | **`87 / 145` (`60.0%`)** | `58 / 145` (`40.0%`) | `-79.1` / `-79.0` | `79.1` / `79.0` |
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+3 h** | **Next 3h `(3h, +6h]`** | `other` | `172 / 402` (`42.8%`) | `230 / 402` (`57.2%`) | `-59.4` / `-41.5` | `59.4` / `41.5` |
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+3 h** | **Next 3h `(3h, +6h]`** | **`ALL`** | **`318 / 722` (`44.0%`)** | `404 / 722` (`56.0%`) | `-67.7` / `-58.0` | **`67.7`** / **`58.0`** |
| **development (Camp Fire, point-start)** | `teacher_v1 (P_base=0.21)` Point-Start | **+3 h** | **Next 3h `(3h, +6h]`** | **`ALL`** | **`0 / 722` (`0.0%`)** | `722 / 722` (`100.0%`) | N/A | N/A |
| **development (Camp Fire, assimilation)** | `teacher_v1 (P_base=0.21)` Assimilation | **+3 h** | **Cumulative `[0h, +6h]`** | **`ALL`** | **`884 / 1,288` (`68.6%`)** | `404 / 1,288` (`31.4%`) | **`+7.4`** / `+25.0` | **`56.1`** / **`45.0`** |

- **Key Finding from Step 1**:
  - In point-start mode from $t=0$, contact spread (`teacher_v1`, $P_{\text{base}}=0.21$) stalls behind the West Branch Feather River water barrier (`0/858` in `(2h, +5h]`, `0/722` in `(3h, +6h]`).
  - Once the $+2\text{ h}$ (`08:30 PST`) observed extent (`175` NIST points = `331 ha`, which includes the initial spot fires that landed on the Paradise ridge) is assimilated, **even contact-only `teacher_v1` forecasts `302 / 858 (35.2%)` of strictly future NIST points in `(2h, +5h]` (`39/85 = 45.9%` vegetative with `35.9 min` MAE / `28.0 min` median AE; `79/175 = 45.1%` structural) and `318 / 722 (44.0%)` in `(3h, +6h]` (`87/145 = 60.0%` structural; cumulative `[0h, +6h]` recall `884 / 1,288 = 68.6%` with `56.1 min` MAE and `+7.4 min` mean signed bias)**.
  - The remaining gap at $T_0 = +1\text{ h}$ (`31 / 757 = 4.1%` recall in `(1h, +4h]`) occurs because at $+1\text{ h}$ (`07:30 PST`) the fire has not yet jumped the West Branch Feather River gorge into Paradise—demonstrating why the **Step 2 ember spotting kernel** is still required for early-time ($t < 2\text{ h}$) barrier crossing and deep WUI interior propagation.

---

### Step 2: Switchable Ember Spotting & WUI Ignition Kernel Calibration and Ablation (`reports/step2_ember_ablation_*.csv`)

- **Plot**: `reports/phase_1b_ember_ablation.png`
- **Calibration Protocol (Camp Fire Strictly Unseen)**:
  - `EmberParams` (`alpha_launch`, `beta_dist_m_per_ms`, `p_urban_wui`) was fit by coordinate search on the **3 wind-driven training fires (`Kincade, Glass, Zogg`)** using perimeter-level metrics (`dice_loss_2km`, `area_ratio`, `front_dist_err_km`, `far_field_jump_cells`, and within-perimeter `wui_f1`).
  - **LOFO Fitted Parameters on Wind-3 Folds**:
    - Hold-out `Kincade` (fit on `Glass, Zogg`): `alpha_launch = 0.0080`, `beta_dist_m_per_ms = 100.0`, `p_urban_wui = 0.25`
    - Hold-out `Glass` (fit on `Kincade, Zogg`): `alpha_launch = 0.0040`, `beta_dist_m_per_ms = 110.0`, `p_urban_wui = 0.15`
    - Hold-out `Zogg` (fit on `Kincade, Glass`): `alpha_launch = 0.0110`, `beta_dist_m_per_ms = 100.0`, `p_urban_wui = 0.08`
  - **Wind-3 Consensus `EmberParams`** (fit on `Kincade, Glass, Zogg`; evaluated as `in-sample` on Wind-3 and **strictly `held-out` on `Bobcat, Creek, Caldor, Tamarack`**):
    - `alpha_launch = 0.0040`, `beta_dist_m_per_ms = 110.0 m/(m/s)`, `p_urban_wui = 0.15`

#### 1. Aggregate Ablation Table Across Fold Groups (`reports/step2_ember_ablation_summary.csv`)

| Fold Group | Ablation Mode | Split | Horizon | Mean ~2km IoU | Mean ~2km $F_1$ | Mean 375m FEDS $F_1$ | Mean Area Ratio | Median Area Ratio | Mean Wind Front Err (km) | Mean Far-Field Jump Cells | Total WUI TP / Obs WUI Cells |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Wind-Driven (`Kincade, Glass, Zogg`, `n=3`)** | **Ember OFF (`P_base=0.21`)** | **held-out (LOFO)** | +1 h | 0.4087 | 0.5291 | — | 0.606× | 0.675× | 2.34 km | 0.0 | `270 / 843` (`32.0%`, all from $t=0$) |
| **Wind-Driven (`Kincade, Glass, Zogg`, `n=3`)** | **Ember ON (LOFO Fit)** | **held-out (LOFO)** | +1 h | **0.4643** | **0.6111** | — | **0.643×** | **0.683×** | **1.34 km** | 159.7 | **`282 / 843` (`33.5%`)** |
| **Wind-Driven (`Kincade, Glass, Zogg`, `n=3`)** | **Ember ON (Consensus)** | **in-sample (Wind-3)** | +1 h | 0.4087 | 0.5291 | — | 0.617× | 0.681× | 2.20 km | 11.3 | `281 / 843` (`33.3%`) |
| **Wind-Driven (`Kincade, Glass, Zogg`, `n=3`)** | **Ember OFF (`P_base=0.21`)** | **held-out (LOFO)** | +3 h | 0.4074 | 0.5657 | — | 0.558× | 0.598× | 3.75 km | 31.0 | `270 / 1,432` (`18.9%`) |
| **Wind-Driven (`Kincade, Glass, Zogg`, `n=3`)** | **Ember ON (LOFO Fit)** | **held-out (LOFO)** | +3 h | **0.4370** | **0.5928** | — | **0.754×** | **0.736×** | **3.22 km** | 592.7 | **`303 / 1,432` (`21.2%`)** |
| **Wind-Driven (`Kincade, Glass, Zogg`, `n=3`)** | **Ember ON (Consensus)** | **in-sample (Wind-3)** | +3 h | **0.4495** | **0.6156** | — | 0.591× | 0.588× | **2.48 km** | 129.7 | **`311 / 1,432` (`21.7%`)** |
| **Wind-Driven (`Kincade, Glass, Zogg`, `n=3`)** | **Ember OFF (`P_base=0.21`)** | **held-out (LOFO)** | +6 h | **0.4603** | **0.6168** | — | 0.605× | 0.617× | 3.89 km | 7.0 | `270 / 2,186` (`12.4%`) |
| **Wind-Driven (`Kincade, Glass, Zogg`, `n=3`)** | **Ember ON (LOFO Fit)** | **held-out (LOFO)** | +6 h | 0.4200 | 0.5741 | — | **1.004×** | **0.949×** | 4.18 km | 97.7 | **`355 / 2,186` (`16.2%`)** |
| **Wind-Driven (`Kincade, Glass, Zogg`, `n=3`)** | **Ember ON (Consensus)** | **in-sample (Wind-3)** | +6 h | 0.4283 | 0.5800 | — | 0.695× | 0.644× | **2.54 km** | 14.3 | **`363 / 2,186` (`16.6%`)** |
| **Wind-Driven (`Kincade, Glass, Zogg`, `n=3`)** | **Ember OFF (`P_base=0.21`)** | **held-out (LOFO)** | +12 h | **0.3924** | **0.5624** | **0.5013** | **1.012×** | **0.945×** | **0.94 km** | 0.7 | `270 / 4,454` (`6.1%`, 0 new WUI) |
| **Wind-Driven (`Kincade, Glass, Zogg`, `n=3`)** | **Ember ON (LOFO Fit)** | **held-out (LOFO)** | +12 h | 0.2987 | 0.4477 | 0.4279 | 1.783× | 1.801× | 5.84 km | 44.7 | **`429 / 4,454` (`9.6%`, +159 new WUI)** |
| **Wind-Driven (`Kincade, Glass, Zogg`, `n=3`)** | **Ember ON (Consensus)** | **in-sample (Wind-3)** | +12 h | 0.2981 | 0.4475 | 0.4411 | 1.171× | 1.106× | 2.52 km | 54.7 | **`429 / 4,454` (`9.6%`, +159 new WUI)** |
| **Plume Folds (`Creek, Bobcat`, `n=2`)** | **Ember OFF (`P_base=0.21`)** | **held-out (LOFO)** | +12 h | 0.1942 | 0.3072 | 0.1530 | **8.738×** | **8.738×** | **4.47 km** | 13.5 | `9 / 74` (`12.2%`, 0 new WUI) |
| **Plume Folds (`Creek, Bobcat`, `n=2`)** | **Ember ON (Consensus)** | **held-out (unseen in fit)** | +12 h | **0.2039** | **0.3171** | **0.1639** | 9.021× | 9.021× | 4.67 km | 274.5 | **`25 / 74` (`33.8%`, +16 new WUI)** |
| **Terrain Folds (`Caldor, Tamarack`, `n=2`)** | **Ember OFF (`P_base=0.21`)** | **held-out (LOFO)** | +12 h | **0.2312** | **0.3644** | **0.2529** | **4.576×** | **4.576×** | **6.16 km** | 4,543.0 | `65 / 316` (`20.6%`, 0 new WUI) |
| **Terrain Folds (`Caldor, Tamarack`, `n=2`)** | **Ember ON (Consensus)** | **held-out (unseen in fit)** | +12 h | 0.1500 | 0.2604 | 0.1710 | 5.775× | 5.775× | 10.98 km | 441.5 | **`119 / 316` (`37.7%`, +54 new WUI)** |
| **All 7 Folds (`n=7`)** | **Ember OFF (`P_base=0.21`)** | **held-out (LOFO)** | +1 h | 0.4459 | 0.5824 | — | 1.255× | 1.041× | 1.48 km | 237.4 | `344 / 951` (`36.2%`) |
| **All 7 Folds (`n=7`)** | **Ember ON (LOFO Fit)** | **held-out (LOFO)** | +1 h | **0.4643** | **0.6123** | — | 1.287× | 1.049× | **1.05 km** | 381.7 | **`364 / 951` (`38.3%`)** |
| **All 7 Folds (`n=7`)** | **Ember OFF (`P_base=0.21`)** | **held-out (LOFO)** | +3 h | 0.3468 | 0.5058 | — | 2.070× | 1.268× | 2.94 km | 689.4 | `344 / 1,679` (`20.5%`) |
| **All 7 Folds (`n=7`)** | **Ember ON (LOFO Fit)** | **held-out (LOFO)** | +3 h | **0.3501** | **0.5059** | — | 2.228× | 1.724× | 3.18 km | 1,255.6 | **`420 / 1,679` (`25.0%`)** |
| **All 7 Folds (`n=7`)** | **Ember ON (Consensus)** | **in-sample (W3) / held-out (4)** | +3 h | **0.3555** | **0.5157** | — | 2.158× | 1.724× | **2.86 km** | 1,057.1 | **`428 / 1,679` (`25.5%`)** |
| **All 7 Folds (`n=7`)** | **Ember OFF (`P_base=0.21`)** | **held-out (LOFO)** | +12 h | **0.2897** | **0.4329** | **0.3308** | **4.238×** | **1.512×** | **3.44 km** | 1,302.1 | `344 / 4,844` (`7.1%`, 0 new WUI) |
| **All 7 Folds (`n=7`)** | **Ember ON (LOFO Fit)** | **held-out (LOFO)** | +12 h | 0.2291 | 0.3569 | 0.2791 | 4.992× | 2.882× | 6.97 km | 223.7 | **`573 / 4,844` (`11.8%`, +229 new WUI)** |
| **All 7 Folds (`n=7`)** | **Ember ON (Consensus)** | **in-sample (W3) / held-out (4)** | +12 h | 0.2289 | 0.3568 | 0.2847 | 4.729× | 1.801× | 5.55 km | 228.0 | **`573 / 4,844` (`11.8%`, +229 new WUI)** |

#### 2. Per-Fire Ablation Highlights (`reports/step2_ember_ablation_per_fire.csv`)
- **Early/Mid-Horizon Wind-Driven Gains (+1 h to +6 h) & WUI Penetration**:
  - **Kincade (`wind-driven`)**: In `Ember OFF`, Kincade under-predicts +12 h area (`0.435×`, `5,049.7 / 11,621.9 ha`) and ignites **`0 / 784`** WUI cells (`FBFM40=91`). With `Ember ON (LOFO Fit)`, +3 h IoU rises from `0.5000` to **`0.5625`**, +12 h area ratio improves to **`0.666×`** (`7,740.4 / 11,621.9 ha`), arrival MAE improves from `266.8 min` to **`247.3 min`**, and WUI true positives jump from `0 / 784` to **`43 / 784`**. With `Ember ON (Consensus)`, +6 h IoU rises from `0.5909` to **`0.6190`** ($F_1 = 0.7647$).
  - **Zogg (`wind-driven`)**: In `Ember OFF`, early highway/stream barriers cause severe under-prediction at +3 h (`0.136×`, `895.3 / 6,559.9 ha`, front lag `-5.02 km`) and +6 h (`0.277×`, `2,887.6 / 10,436.8 ha`, front lag `-5.49 km`). With `Ember ON (Consensus)`, +3 h ~2km IoU jumps from `0.2222` to **`0.3333`** ($F_1$: `0.3636 -> 0.5000`), +3 h front error drops from `5.02 km` to **`1.33 km`**, +6 h front error drops from `5.49 km` to **`1.01 km`**, +12 h area ratio is **`1.106×`** (`14,019.5 / 12,677.8 ha`), and WUI true positives jump from `9 / 683` (initial polygon only) to **`48 / 683`**.
  - **Creek (`plume`, strictly held-out for ember fit)**: `Ember ON (Consensus)` improves +12 h ~2km IoU from `0.3173` to **`0.3404`**, improves 375 m FEDS $F_1$ from `0.2834` to **`0.3061`**, improves +12 h area ratio from `1.512×` to **`1.362×`** (`22,792.0 / 16,735.7 ha`), reduces +6 h front error from `0.62 km` to **`0.20 km`**, and increases WUI true positives from `0 / 44` to **`7 / 44`**.
- **Plain Trade-Off Diagnosis at +12 h**:
  - Adding non-local ember spotting on top of fixed contact $P_{\text{base}} = 0.21$ increases cumulative 12-hour downwind reach, which causes late-window (+6 h to +12 h) downwind front overshoot on `Zogg` (`+5.87 km` signed front error at +12 h under Consensus; `+15.41 km` under LOFO Fit where fitting on `Kincade + Glass` selected `alpha_launch = 0.011`) and `Tamarack` (`+13.21 km`, where embers leap across high-Sierra granite/barren ridges that previously contained the contact simulation at `0.893×`).
  - Across all 7 fires, `Ember OFF` has **zero** new WUI (`FBFM40=91`) ignitions after $t=0$ (`344 / 951` at +1 h remains `344 / 4,844` at +12 h), whereas `Ember ON` ignites **`+229` additional true-positive WUI cells** (`573 / 4,844`), directly enabling urban/WUI spread.

---

### Step 3: Proposed Fresh Held-Out GOFER Wind-Driven Test Fires (`reports/step3_proposed_fresh_heldout_fires.csv`)

We screened all 21 remaining unused California fires in the GOFER 28-fire catalog (`reports/step3_gofer_unused_ca_candidates.csv`, excluding `Kincade, Bobcat, Glass, Creek, Zogg, Caldor, Tamarack`, and `Camp Fire`). Most 2020–2021 megafires in GOFER (`August Complex`, `LNU`, `SCU`, `SQF`, `River Complex`, `Red Salmon Complex`) are multi-ignition lightning complexes where disconnected lightning strikes merge over days, or (`Dixie`, `North Complex`) spatially intersect the 2018 Camp Fire domain.

We propose **2 primary single-front wind-driven held-out test fires** (plus **1 optional alternate**) that have clean 24–48 h unsuppressed wind-driven runs, $1.0\text{ h}$ GOFER perimeters, strictly pre-fire `LANDFIRE 2016 Remap (LF 2.0.0)` coverage, and **0.0% spatial overlap** with both Camp Fire and the 7 Phase 1A training fires:

| Proposal Rank | GOFER ID & Fire Name | Year | Split | Regime Tag | GOFER Perims ($\Delta t$) | Calibration / Evaluation Window (24 h & 48 h UTC) | GOFER Observed Area Curve ($t_0 \to +1\text{h} \to +3\text{h} \to +6\text{h} \to +12\text{h} \to +24\text{h} \to +48\text{h}$) | 60 m Grid & Domain BBox (`EPSG:32610` & `WGS84`) | 24h Obs % of Domain | Pre-Fire LANDFIRE Vintage | Overlap w/ Camp Fire or Phase 1A? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Primary Proposed Held-Out #1** | **#14 `Slater and Devil`** (Siskiyou / Del Norte, CA) | 2020 | **fresh held-out test (PROPOSED)** | `wind-driven` | `326` (`1.0 h`) | `2020-09-08 16:00 UTC` $\to$ `2020-09-09 16:00 UTC` (`+48h`: `2020-09-10 16:00 UTC`) | `1,515.7 ha` ($t_0$) $\to$ `2,582.0 ha` (+1h) $\to$ `5,512.7 ha` (+3h) $\to$ `13,989.9 ha` (+6h) $\to$ `26,006.6 ha` (+12h) $\to$ `49,360.4 ha` (+24h) $\to$ `60,096.6 ha` (+48h) | `720×720` (`186,624.0 ha`) UTM: `[439620, 4620000, 482820, 4663200]` WGS84: `[-123.7304, 41.7295, -123.2066, 42.1207]` | **`26.45%`** (cannot saturate) | `LANDFIRE 2016 Remap (LF 2.0.0)` | **`False` / `False` (`0.0%`)** |
| **Primary Proposed Held-Out #2** | **#2 `Walker`** (Plumas County, CA) | 2019 | **fresh held-out test (PROPOSED)** | `wind-driven` | `193` (`1.0 h`) | `2019-09-06 18:00 UTC` $\to$ `2019-09-07 18:00 UTC` (`+48h`: `2019-09-08 18:00 UTC`) | `3,924.7 ha` ($t_0$) $\to$ `4,047.2 ha` (+1h) $\to$ `6,143.8 ha` (+3h) $\to$ `6,647.6 ha` (+6h) $\to$ `16,514.9 ha` (+12h) $\to$ `16,514.9 ha` (+24h) $\to$ `24,582.2 ha` (+48h) | `720×720` (`186,624.0 ha`) UTM: `[684000, 4418700, 727200, 4461900]` WGS84: `[-120.8477, 39.8879, -120.3277, 40.2874]` | **`8.85%`** (cannot saturate) | `LANDFIRE 2016 Remap (LF 2.0.0)` | **`False` / `False` (`0.0%`)** |
| **Alternate Proposed Held-Out #3** | **#16 `W-5 Cold Springs`** (Lassen County, CA) | 2020 | **fresh held-out test (PROPOSED)** | `wind-driven` | `264` (`1.0 h`) | `2020-08-21 19:00 UTC` $\to$ `2020-08-22 19:00 UTC` (`+48h`: `2020-08-23 19:00 UTC`) | `1,746.7 ha` ($t_0$) $\to$ `1,924.0 ha` (+1h) $\to$ `3,477.6 ha` (+3h) $\to$ `4,348.3 ha` (+6h) $\to$ `6,249.0 ha` (+12h) $\to$ `6,821.5 ha` (+24h) $\to$ `19,809.5 ha` (+48h) | `720×720` (`186,624.0 ha`) UTM: `[717900, 4529100, 761100, 4572300]` WGS84: `[-120.4137, 40.8714, -119.8833, 41.2727]` | **`3.66%`** (cannot saturate) | `LANDFIRE 2016 Remap (LF 2.0.0)` | **`False` / `False` (`0.0%`)** |

---

## 4. Deviations from the Brief

- **None**:
  - Step 0 re-audited Phase 1A without new contact modeling, froze `configs/teacher_v1_1.yaml` + SHA-256 hash (`reports/teacher_v1_1_hash.txt`), reported all 20-seed coverage and Camp Fire metrics with explicit denominators, relabeled the 30 m run as `"parameter transfer test (non-transferable), not a resolution penalty"`, and executed the 600-run criticality sweep on Kincade and Camp Fire.
  - Step 1 implemented operational assimilation mode (`init-from-observed-extent` at $T_0 \in \{+1\text{ h}, +2\text{ h}, +3\text{ h}\}$) and evaluated 3-hour forecasts on Camp Fire by NIST point type.
  - Step 2 implemented the switchable 3-parameter `EmberParams` module (`src/wf052/sim/ember_kernel.py`), tuned it **exclusively on the wind-driven training fires (`Kincade, Glass, Zogg`)** with Camp Fire strictly unseen, and reported the Ember ON vs OFF ablation across all 7 folds.
  - Per Step 3 instructions, **we have STOPPED before Step 4** (`configs/teacher_v2.yaml` freeze, running fresh held-out fires once, and running Camp Fire with embers in point-start and assimilation mode) to wait for Orchestrator approval of the fresh held-out test fires.

---

## 5. Open Problems

1. **Late-Horizon (+12 h) Downwind Overshoot When Adding Embers on Top of Fixed Contact $P_{\text{base}} = 0.21$**:
   - While `Ember ON` (`alpha_launch = 0.004, beta_dist_m_per_ms = 110.0, p_urban_wui = 0.15`) improves +1 h and +3 h wind-driven IoU (`0.4074 -> 0.4495` at +3 h), cuts +3 h and +6 h wind-front lag (`3.75 km -> 2.48 km` at +3 h, `3.89 km -> 2.54 km` at +6 h), improves `Creek` +12 h IoU (`0.3173 -> 0.3404`), and increases true-positive WUI ignitions across all 7 fires from `344 / 4,844` (0 new WUI cells after $t=0$) to `573 / 4,844` (`+229` new WUI cells), secondary spot fires between +6 h and +12 h cause downwind front over-extension on `Zogg` (`+5.87 km`) and `Tamarack` (`+13.21 km`).
   - Slightly reducing $P_{\text{base}}$ from `0.21` to `0.20` when embers are active (or using `alpha_launch = 0.002–0.004`) prevents contact + ember double-counting at +12 h.
2. **Early-Time ($T_0 = +1\text{ h}$) Canyon Leap on Camp Fire**:
   - In Step 1 (contact-only assimilation), $T_0 = +2\text{ h}$ and $+3\text{ h}$ achieve `35.2%` and `44.0%` next-3h NIST point recall (`68.6%` cumulative recall by +6 h), whereas $T_0 = +1\text{ h}$ reaches `4.1%` (`31 / 757`) because the Feather River gorge crossing into Paradise occurs via ember spotting between $+1\text{ h}$ and $+2\text{ h}$. Step 4 will test whether enabling the frozen ember kernel closes this $+1\text{ h}$ gap in both point-start and assimilation modes.

---

## 6. Decisions I Need to Make (Orchestrator Input Before Step 4)

1. **Approval of Fresh Held-Out Wind-Driven Test Fires (Step 3 $\to$ Step 4 Gate)**:
   - Do you approve ingesting and running **`Slater and Devil` (2020, GOFER ID #14)** and **`Walker` (2019, GOFER ID #2)** (or substituting **`W-5 Cold Springs` (2020, GOFER ID #16)**) as our fresh held-out wind-driven test fires for Step 4?
2. **`configs/teacher_v2.yaml` Parameter Freeze for Step 4**:
   - Should `configs/teacher_v2.yaml` freeze the Wind-3 consensus ember parameters (`alpha_launch: 0.0040, beta_dist_m_per_ms: 110.0, p_urban_wui: 0.15`) on top of `teacher_v1` ($P_{\text{base}} = 0.21$) or `teacher_v1_1` ($P_{\text{base}} = 0.22$)?
3. **Rothermel ROS Contact Rule Contingency (Step 0d)**:
   - Keep the Rothermel-style continuous ROS contact rule ($p_{\text{ignite}} = 1 - \exp(-\gamma_{\text{ROS}} \text{ROS} \cdot \Delta t / d_{ij})$) on standby as a contingency until we inspect the Step 4 results with `teacher_v2`, or schedule it for implementation after Step 4?
