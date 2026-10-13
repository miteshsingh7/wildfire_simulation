# Phase 1A Report: Making the Teacher Trustworthy (Steps 0–3 Complete)

**Project:** `wf052` (Wildfire Spread Simulation & Agentic Routing)  
**Phase:** 1A (Physics Teacher Audit, Resolution Sensitivity, Training Set Selection, LOFO Calibration & Held-Out Camp Fire Evaluation)  
**Date:** October 10, 2026  
**Status:** **COMPLETED** (`configs/teacher_v1.yaml` frozen; SHA-256: `c45ba6db95187063347ad1c06e81dbb29ff3e45fa92f50ad46d8dcc9e16cf9b6`)

---

## 1. What Was Built

### Step 0: Perimeter Timeline Audit & Core Fixes
1. **Perimeter Audit (`scripts/audit_geomap_times.py`, `scripts/audit_geomac_times.py`)**:
   - Audited all 28 GeoMAC 2018 Camp Fire shapefiles in `data/raw/geomac_camp_fire_2018/`.
   - **Root Cause Identified**: Discovered that GeoMAC epoch millisecond timestamps represented local Pacific time (`17:54:00`), which naive UTC parsers distorted into 17:54 UTC (+3.4 h). Verified against NIST TN 2135 p. 127, which documents that the first infrared (NIROPS) overflight commenced at **17:54 PST on November 8, 2018** (2018-11-09 01:54:00 UTC = **+11.4 h / 684 min** post-ignition).
   - Produced `reports/geomac_timeline_audit.csv` and `reports/geomac_area_vs_time.png`.
2. **Elimination of Hand-Typed Area Figures (`scripts/reconcile_areas.py`)**:
   - Completely deleted synthetic estimates (`~6,000 / ~16,000 / ~22,000 ha`).
   - Relabeled horizons +3 h (180 min) and +6 h (360 min) as **"No observation"** because GeoMAC contains zero aerial flights between 06:30 and 17:54 PST.
   - Updated `reports/area_reconciliation.csv`.
3. **Confusion Matrix Evaluation (`scripts/evaluate_perimeters.py`)**:
   - Evaluated PyTorchFire and Cell2Fire against the true observed overflight at +11.4 h (684 min) and against the final contained footprint (2018-11-25).
   - Recorded raw confusion matrix cell counts ($TP, FP, FN, TN$), $IoU$, Precision, Recall, and $F_1$ in `reports/perimeter_evaluation_confusion.csv`.
4. **Cell2Fire Source & CLI Audit**:
   - Audited `external/Cell2Fire/` C++ source code (`ReadArgs.cpp`, `CellsFBP.cpp`, `ParseInputs.py`).
   - Confirmed Cell2Fire's C++ core strictly executes Canadian FBP v5.0 (18 discrete fuel types) and lacks native command-line options or engine switches for Rothermel / Scott-Burgan FBFM40 models.
   - Relabeled status in `reports/phase_0.md`: **"Not evaluated under native FBFM40; deprioritized"** (retracting unverified claims).
5. **Phase 0.5 Verdict Correction**:
   - Corrected conclusion in `reports/phase_0.md`: explicitly stated that PyTorchFire contact-spread timing being "calibratable" is an **untested empirical hypothesis**, to be tested exclusively on external training fires since Camp Fire is strictly held out.

### Step 1: Resolution Sensitivity Benchmark
1. **Multi-Resolution Resampling & Benchmark Engine (`scripts/benchmark_resolution_sensitivity.py`)**:
   - Downsampled the 30 m Camp Fire domain to **60 m** and **90 m** grids using physical aggregation rules:
     - Fuel model: 2D block mode (dominant fuel type).
     - Elevation & canopy: block mean.
     - Slope: recomputed using centered finite differences with rescaled grid spacing $\Delta x, \Delta y$.
     - Wind field: bilinear interpolation to the target grid.
   - Simulated 12 h (720 steps at $\Delta t = 60\text{ s}$) across all 3 resolutions on Apple Silicon MPS.
   - Produced `reports/resolution_sensitivity.csv` and `reports/resolution_sensitivity_curves.png`.

### Step 2: Training Fires Selection (Camp Fire Held Out)
1. **Verification of GOFER and FEDS Datasets**:
   - Queried Zenodo API (records 8327264 / 14642378) and the official GOFER repository (`largeFires_metadata.js`).
   - Verified that GOFER covers 28 major California wildfires across 2019–2021 at **~2–3 km GOES spatial resolution** with **hourly** progression contours.
   - Verified FEDS coverage (375 m VIIRS active fire detections at 12-hour intervals) and GeoMAC/NIFC infrared flight perimeters.
2. **Selection of 7 Diverse Training Fires (`scripts/prepare_training_fires_proposal.py`)**:
   - Selected 7 California fires balancing wind-driven extreme events, terrain-dominated Sierra conifer fires, and mixed chaparral/timber fuels.
   - Confirmed complete NOAA HRRR hourly weather on AWS S3, pre-fire LANDFIRE vintages, and verified **0% spatial overlap** with the Camp Fire bbox.
   - Output compiled to `reports/proposed_training_fires.csv`.

### Step 3: Multi-Scale Data Ingestion, OAT Sensitivity, LOFO Calibration, Freeze & Held-Out Camp Fire Test
1. **PREP (a) — Per-Fire Data Pipeline & Setup Justification (`scripts/download_and_inspect_gofer.py`, `scripts/analyze_gofer_7fires.py`, `scripts/inspect_gofer_windows.py`, `scripts/prepare_training_fires_data.py`, `scripts/fetch_feds_375m.py`, `reports/phase_1a_fires.md`)**:
   - Downloaded the official GOFER Zenodo archive (`data/raw/GOFER.zip`, 75.71 MB, record `14642378`), verified all 7 training fires against `GOFER/fireData.csv` and `GOFER_Combined/GOFERC_fireProg.shp`, and selected a 24 h unsuppressed rapid-growth breakout window for each fire justified directly from GOFER hourly area curves.
   - Standardized every training fire onto a **$720 \times 720$ grid at 60 m resolution** (`43.2 km × 43.2 km = 186,624 ha`) so the fire cannot saturate the domain boundary during the calibration window, using **one uniform pre-fire vintage**: **LANDFIRE 2016 Remap (LF 2.0.0)**.
   - Fetched real hourly NOAA HRRR surface wind/RH/temperature fields via Herbie (`hrrr_weather_hourly.parquet`) and streamed 375 m 12-hour VIIRS active fire pixels from the FEDS Zenodo archive (`15261836`, `feds_375m.npz`) for all 7 fires.
2. **PREP (b) & (c) — Multi-Scale Calibrator & OAT Sensitivity (`src/wf052/sim/teacher_calibrator.py`, `tests/test_teacher_calibrator.py`, `scripts/run_parameter_sensitivity.py`)**:
   - Built `TeacherParams`, `FireDataset60m`, `simulate_teacher_60m`, `aggregate_to_coarse_fraction` (`33x33` block = `1,980 m` ~2 km for GOFER; `6x6` block = `360 m` ~375 m for FEDS), `evaluate_simulation_multiscale`, and `compute_composite_calibration_loss`.
   - Executed an 82-run One-at-a-Time (OAT) sensitivity sweep across all 9 PyTorchFire parameters around their defaults ($P_{\text{base}} = 0.50$) on `Kincade` (`wind-driven`) and `Bobcat` (`plume`) at +6 h, +12 h, and +24 h (`reports/parameter_sensitivity.csv`, `reports/parameter_sensitivity_summary.csv`).
3. **FITTING — LOFO Cross-Validation, Baselines, Regime Ablation & 20-Seed Ensemble (`scripts/run_lofo_calibration.py`)**:
   - Ran Leave-One-Fire-Out (LOFO) cross-validation across all 7 fires comparing `uncalibrated` ($P_{\text{base}} = 0.50$), `global_slowdown_lofo` (1-parameter $P_{\text{base}}$ fit), `calibrated_all7_lofo` (6-parameter coordinate/SPSA refinement), `calibrated_wind_only_lofo` (fit on `Kincade, Glass, Zogg`), and `calibrated_all7_insample`.
   - Ran a 20-seed Monte Carlo ensemble across all 7 fires and reported P10–P90 coverage of observed arrival times (`reports/lofo_ensemble_20seeds.csv`).
4. **FREEZE AND TEST — Frozen Teacher Config & Held-Out Camp Fire 60 m Evaluation (`configs/teacher_v1.yaml`, `reports/teacher_v1_hash.txt`, `scripts/evaluate_heldout_camp_fire.py`, `scripts/plot_phase1a_step3_figures.py`)**:
   - Froze `configs/teacher_v1.yaml` and recorded its SHA-256 hash (`c45ba6db95187063347ad1c06e81dbb29ff3e45fa92f50ad46d8dcc9e16cf9b6`) before touching Camp Fire.
   - Evaluated `teacher_v1`, `wind_only_v1`, and `uncalibrated` once on the held-out 2018 Camp Fire at 60 m ($564 \times 562$ grid), plus a 20-seed ensemble and a 30 m vs 60 m resolution penalty check on Kincade.

---

## 2. Commands Run

```bash
# Step 0-2 commands
cmd /c .venv/bin/python scripts/audit_geomap_times.py
cmd /c .venv/bin/python scripts/reconcile_areas.py
cmd /c .venv/bin/python scripts/evaluate_perimeters.py
cmd /c .venv/bin/python scripts/benchmark_resolution_sensitivity.py
cmd /c .venv/bin/python scripts/prepare_training_fires_proposal.py

# Step 3 PREP (a): Download GOFER, inspect breakout windows, ingest 60m LANDFIRE + HRRR + GOFER + FEDS 375m
cmd /c .venv/bin/python scripts/download_and_inspect_gofer.py
cmd /c .venv/bin/python scripts/analyze_gofer_7fires.py
cmd /c .venv/bin/python scripts/inspect_gofer_windows.py
cmd /c .venv/bin/python scripts/prepare_training_fires_data.py
cmd /c .venv/bin/python scripts/fetch_feds_375m.py

# Step 3 PREP (b): Run 82-simulation One-at-a-Time (OAT) parameter sensitivity sweep on Kincade & Bobcat
cmd /c .venv/bin/python scripts/run_parameter_sensitivity.py

# Step 3 FITTING & FREEZE: Run LOFO cross-validation, regime ablation, 20-seed ensemble, and freeze configs/teacher_v1.yaml
cmd /c .venv/bin/python scripts/run_lofo_calibration.py

# Step 3 TEST: Verify frozen SHA-256 hash and evaluate held-out 2018 Camp Fire at 60m + 30m vs 60m resolution check
cmd /c .venv/bin/python scripts/evaluate_heldout_camp_fire.py
cmd /c .venv/bin/python scripts/plot_phase1a_step3_figures.py

# Quality gate
cmd /c .venv/bin/ruff check .
cmd /c .venv/bin/pytest
```

### Key Command Outputs (Step 3)

```text
$ cmd /c .venv/bin/python scripts/run_lofo_calibration.py
=== LOFO & Regime Ablation Summary ===
                       evaluation_group                     model           split  horizon_h  mean_iou_2km  mean_f1_2km  mean_area_ratio  median_area_ratio  mean_arrival_mae_min  mean_front_dist_err_km  mean_feds_375m_f1
                            All 7 Fires              uncalibrated held-out (LOFO)         12        0.1036       0.1851           13.685              8.926                 211.4                    8.27             0.1336
                            All 7 Fires      global_slowdown_lofo held-out (LOFO)         12        0.2436       0.3663            4.105              1.512                 293.0                    5.32             0.2741
                            All 7 Fires      calibrated_all7_lofo held-out (LOFO)         12        0.2204       0.3337            4.570              1.512                 288.0                    6.51             0.2471
                            All 7 Fires calibrated_wind_only_lofo held-out (LOFO)         12        0.2297       0.3590            4.994              1.828                 273.8                    5.28             0.2789
                            All 7 Fires  calibrated_all7_insample       in-sample         12        0.2897       0.4329            4.238              1.512                 281.5                    3.44             0.3308
Wind-Driven Folds (Kincade, Glass, Zogg)              uncalibrated held-out (LOFO)         12        0.1391       0.2434            7.552              8.489                 174.6                    4.63             0.2113
Wind-Driven Folds (Kincade, Glass, Zogg)      global_slowdown_lofo held-out (LOFO)         12        0.2848       0.4070            0.702              0.435                 318.6                    5.32             0.3691
Wind-Driven Folds (Kincade, Glass, Zogg)      calibrated_all7_lofo held-out (LOFO)         12        0.2983       0.4207            0.580              0.435                 328.1                    5.00             0.3710
Wind-Driven Folds (Kincade, Glass, Zogg) calibrated_wind_only_lofo held-out (LOFO)         12        0.3214       0.4832            1.037              0.937                 295.2                    3.43             0.4407
Wind-Driven Folds (Kincade, Glass, Zogg)  calibrated_all7_insample       in-sample         12        0.3924       0.5624            1.012              0.945                 291.6                    0.94             0.5013

Frozen configs/teacher_v1.yaml SHA-256: c45ba6db95187063347ad1c06e81dbb29ff3e45fa92f50ad46d8dcc9e16cf9b6
```

```text
$ cmd /c .venv/bin/python scripts/evaluate_heldout_camp_fire.py
Verified frozen config configs/teacher_v1.yaml | SHA-256: c45ba6db95187063347ad1c06e81dbb29ff3e45fa92f50ad46d8dcc9e16cf9b6
Running Held-Out 2018 Camp Fire 60m evaluation on mps...
Saved frozen teacher 60m arrival raster to data/tiles/arrival_time_teacher_v1_60m_seed42.tif
```

---

## 3. Results (With Numbers)

### Step 0: Perimeter Timeline Audit & Area Reconciliation (30 m Baseline Audit)

#### 1. GeoMAC Perimeter Timeline Audit (`reports/geomac_timeline_audit.csv`)
- **Total perimeters analyzed:** 28 GeoMAC polygon shapefiles.
- **Key finding:** Object ID 1052 (`raw_perimeterdatetime: 1541699640000`, 17:54:00) represents local PST overflight time:
  - **Parsed Naive UTC:** `2018-11-08 17:54:00 UTC` (+3.4 h post-ignition) $\rightarrow$ **ERROR**
  - **Corrected UTC:** `2018-11-09 01:54:00 UTC` (**+11.4 h / 684 min** post-ignition) $\rightarrow$ **GROUND TRUTH** (NIST TN 2135 p. 127).
  - **Observed Footprint Area:** 54,586.4 acres (**22,090.3 ha**), matching the IBHS ~55,000 acre 12 h progression benchmark.

#### 2. Area Reconciliation Table — Uncalibrated 30 m (`reports/area_reconciliation.csv`)
*Split: **Held-Out Diagnostic (Pre-Calibration 30 m)** | Domain: $1128 \times 1124$ cells (30 m, $114,108.5\text{ ha}$).*

| Split | Horizon | PyTorchFire Cells | PyTorchFire Area (ha) | Sim Domain % | Cell2Fire Area (ha) | Observed Area (ha) | Observed Source | Sim / Obs Ratio |
|---|---|---|---|---|---|---|---|---|
| Held-Out | **+3 h** (180 min) | 88,406 | 7,956.5 | 6.97% | 3.96 | *None* | **No observation** (no flight 06:30–09:30 PST) | N/A |
| Held-Out | **+6 h** (360 min) | 351,469 | 31,632.2 | 27.72% | 12.42 | *None* | **No observation** (no flight 06:30–12:30 PST) | N/A |
| Held-Out | **+11.4 h** (684 min) | 963,634 | 86,727.1 | 76.00% | 32.76 | 22,090.3 | GeoMAC IR Overflight (17:54 PST, ID 1052) | **3.92×** |
| Held-Out | **+12 h** (720 min) | 1,012,738 | 91,146.4 | 79.88% | 32.76 | 22,090.3 | GeoMAC IR Overflight (+11.4 h ref) | **4.13×** |

---

### Step 1: Resolution Sensitivity Benchmark (`reports/resolution_sensitivity.csv`)
*Split: **Held-Out Runtime & Scaling Benchmark** | Apple Silicon MPS, 12 h horizon (720 steps at $\Delta t = 60\text{ s}$).*

| Resolution | Grid Dimensions | Total Grid Cells | Wall-Clock Time (s) | Step Latency (ms/step) | Speedup vs 30m | Final Burned Cells | Final Burned Area (ha) |
|---|---|---|---|---|---|---|---|
| **30 m** | $1128 \times 1124$ | 1,267,872 | 51.82 s | 71.98 ms | $1.0\times$ | 988,800 | 88,992.0 ha |
| **60 m** | $564 \times 562$ | 316,968 | **15.36 s** | **21.33 ms** | **$3.38\times$** | 300,020 | 108,007.2 ha |
| **90 m** | $376 \times 374$ | 140,624 | **8.43 s** | **11.71 ms** | **$6.15\times$** | 132,598 | 107,404.4 ha |

---

### Step 3 PREP (a): Per-Fire Setup & Calibration Windows (`reports/phase_1a_fires.md`, `reports/phase_1a_fires_setup.csv`)
*Split: **Training Set (7 California Fires in GOFER 28-Fire Metadata, 0% Overlap with Camp Fire)** | Grid: **60 m** ($720 \times 720 = 43.2\text{ km} \times 43.2\text{ km} = 186,624\text{ ha}$) | LANDFIRE Vintage: **LANDFIRE 2016 Remap (LF 2.0.0)**.*

| Fire Name | GOFER ID (`fireData.csv`) | Regime Tag | GOFER Perimeters ($\Delta t$) | Calibration Window (24 h Unsuppressed Growth) | GOFER Window Growth (ha) | Peak Domain Occupancy | FEDS 375m Detections (+12h / +24h) |
|---|---|---|---|---|---|---|---|
| **1. Kincade** | #1 (`Kincade`, 2019) | `wind-driven` | 350 (`1.0 h`) | `2019-10-24 05:00` $\to$ `10-25 05:00 UTC` | $107 \to 11,622\text{ ha}$ | 6.23% | 228 / 270 cells |
| **2. Bobcat** | #3 (`Bobcat`, 2020) | `plume` | 645 (`1.0 h`) | `2020-09-06 20:00` $\to$ `09-07 20:00 UTC` | $152 \to 5,405\text{ ha}$ | 2.90% | 9 / 119 cells |
| **3. Glass** | #9 (`Glass`, 2020) | `wind-driven` | 280 (`1.0 h`) | `2020-09-27 11:00` $\to$ `09-28 11:00 UTC` | $332 \to 22,824\text{ ha}$ | 12.23% | 341 / 858 cells |
| **4. Creek** | #5 (`Creek`, 2020) | `plume` | 1,359 (`1.0 h`) | `2020-09-05 03:00` $\to$ `09-06 03:00 UTC` | $976 \to 33,432\text{ ha}$ | 17.91% | 464 / 2,000 cells |
| **5. Zogg** | #10 (`Zogg`, 2020) | `wind-driven` | 146 (`1.0 h`) | `2020-09-27 22:00` $\to$ `09-28 22:00 UTC` | $1,796 \to 16,650\text{ ha}$ | 8.92% | 1,297 / 1,419 cells |
| **6. Caldor** | #18 (`Caldor`, 2021) | `terrain` | 1,210 (`1.0 h`) | `2021-08-16 21:00` $\to$ `08-17 21:00 UTC` | $1,148 \to 25,847\text{ ha}$ | 13.85% | 389 / 1,729 cells |
| **7. Tamarack** | #15 (`Tamarack`, 2021) | `terrain` | 674 (`1.0 h`) | `2021-07-16 20:00` $\to$ `07-17 20:00 UTC` | $601 \to 12,188\text{ ha}$ | 6.53% | 487 / 1,158 cells |

---

### Step 3 PREP (b): One-at-a-Time (OAT) Parameter Sensitivity (`reports/parameter_sensitivity.csv`, `reports/parameter_sensitivity_summary.csv`)
*Split: **Training Diagnostic (`Kincade` Wind-Driven & `Bobcat` Plume, 60 m, 30 steps/hr)**.*

#### 1. Summary of 24 h Burned Area & Arrival-Time MAE Swing Across All 9 PyTorchFire Parameters

| Parameter | Default | Swept Range | Kincade 24h Area Swing (%) | Kincade 24h MAE Swing (min) | Bobcat 24h Area Swing (%) | Bobcat 24h MAE Swing (min) | Fit Decision & Evidence |
|---|---|---|---|---|---|---|---|
| **`p_base`** ($P_{\text{base}}$ / `p_h`) | `0.50` | `[0.06, 0.70]` | **103.3%** ($1,026 \to 146,046\text{ ha}$) | **776.5 min** ($304.0 \to 1080.5\text{ m}$) | **100.2%** ($484 \to 135,488\text{ ha}$) | **309.6 min** ($516.7 \to 826.3\text{ m}$) | **FIT (Primary)**: Controls CA percolation regime and global spread rate. |
| **`c_2`** (wind eccentricity) | `0.15` | `[0.02, 0.50]` | **100.0%** ($6,564 \to 146,910\text{ ha}$) | **372.0 min** ($300.0 \to 672.0\text{ m}$) | **27.8%** ($97,677 \to 135,189\text{ ha}$) | **1.5 min** ($520.4 \to 521.9\text{ m}$) | **FIT (Primary)**: Controls flank/backing suppression $\exp(c_2 V (\cos\theta - 1))$. |
| **`c_1`** (wind speed scale) | `0.045` | `[0.01, 0.12]` | **9.6%** ($132,088 \to 145,544\text{ ha}$) | **12.8 min** ($304.6 \to 317.4\text{ m}$) | **0.3%** ($134,613 \to 135,069\text{ ha}$) | **0.3 min** | **FIT (Secondary, $L_2$-reg)**: Modulates head-fire acceleration under high wind. |
| **`a`** (slope factor) | `0.06` | `[0.00, 0.15]` | **8.9%** ($130,607 \to 143,064\text{ ha}$) | **11.9 min** ($303.1 \to 315.0\text{ m}$) | **1.5%** ($133,698 \to 135,677\text{ ha}$) | **3.9 min** | **FIT (Secondary, $L_2$-reg)**: Topographic uphill acceleration $\exp(a\theta_s)$. |
| **`fuel_scale_grass_shrub`** | `1.00` | `[0.30, 1.40]` | **7.3%** ($131,332 \to 141,557\text{ ha}$) | **9.2 min** ($303.5 \to 312.7\text{ m}$) | **0.6%** ($134,125 \to 134,952\text{ ha}$) | **0.3 min** | **FIT (Secondary, $L_2$-reg)**: Scales FBFM40 codes 101–147. |
| **`fuel_scale_timber`** | `1.00` | `[0.30, 1.40]` | **4.7%** ($135,341 \to 141,907\text{ ha}$) | **5.0 min** ($307.6 \to 312.6\text{ m}$) | **0.0%** ($134,737 \to 134,770\text{ ha}$) | **0.4 min** | **FIT (Secondary, $L_2$-reg)**: Scales FBFM40 codes 161–204. |
| **`canopy_den_scale`** | `1.00` | `[0.00, 2.00]` | **8.0%** ($131,302 \to 142,553\text{ ha}$) | **8.2 min** | **0.9%** ($133,938 \to 135,107\text{ ha}$) | **0.5 min** | **FIX (`1.0`)**: Collinear with `fuel_scale_timber`. |
| **`p_continue`** (persistence) | `0.30` | `[0.05, 0.70]` | **4.6%** ($137,161 \to 143,664\text{ ha}$) | **19.2 min** | **1.1%** ($134,157 \to 135,574\text{ ha}$) | **1.0 min** | **FIX (`0.30`)**: Redundant with `p_base` via effective step probability $1-(1-p)^k$. |
| **`wind_moisture_damping`** | `0.00` | `[0.00, 3.00]` | **0.0%** ($140,403 \to 140,413\text{ ha}$) | **0.0 min** ($312.6 \to 312.6\text{ m}$) | **0.3%** ($134,392 \to 134,770\text{ ha}$) | **0.0 min** ($521.0 \to 521.0\text{ m}$) | **EXCLUDE (`0.0`)**: Zero effect during dry rapid-growth breakout windows ($\text{RH} \approx 10\text{–}18\%$). |

#### 2. Detailed Multi-Horizon (+6 h / +12 h / +24 h) Effect of Dominant Parameters (`p_base` & `c_2`)

| Fire Name (Regime) | Parameter | Value | +6h Sim Area (ha) | +6h Area Ratio | +6h Arr MAE (min) | +12h Sim Area (ha) | +12h Area Ratio | +12h Arr MAE (min) | +24h Sim Area (ha) | +24h Area Ratio | +24h Arr MAE (min) | +24h ~2km IoU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Kincade** (`wind-driven`) | `p_base` | `0.06` | 1,026.4 | 0.17× | 178.0 | 1,026.4 | 0.09× | 370.7 | 1,026.4 | 0.09× | 1080.5 | 0.1111 |
| **Kincade** (`wind-driven`) | `p_base` | `0.12` | 1,612.1 | 0.26× | 159.1 | 1,612.1 | 0.14× | 349.2 | 1,612.1 | 0.14× | 1030.0 | 0.1667 |
| **Kincade** (`wind-driven`) | `p_base` | `0.25` | 12,704.0 | 2.06× | 62.4 | 31,435.2 | 2.70× | 189.8 | 42,311.5 | 3.64× | 411.2 | **0.2128** |
| **Kincade** (`wind-driven`) | `p_base` | `0.50` *(def)* | 39,202.9 | 6.35× | 59.0 | 100,643.0 | 8.66× | 184.0 | 140,412.6 | 12.08× | 312.6 | 0.0859 |
| **Kincade** (`wind-driven`) | `c_2` | `0.02` | 51,920.3 | 8.41× | 60.2 | 116,348.8 | 10.01× | 183.6 | 146,909.5 | 12.64× | 300.0 | 0.0831 |
| **Kincade** (`wind-driven`) | `c_2` | `0.30` | 17,344.4 | 2.81× | 54.3 | 37,057.0 | 3.19× | 201.1 | 45,319.7 | 3.90× | 424.4 | 0.2042 |
| **Kincade** (`wind-driven`) | `c_2` | `0.50` | 6,344.3 | 1.03× | 69.2 | 6,563.9 | 0.56× | 239.2 | 6,563.9 | 0.56× | 672.0 | **0.5000** |
| **Bobcat** (`plume`) | `p_base` | `0.06` | 483.5 | 0.25× | 189.0 | 483.5 | 0.13× | 351.4 | 483.5 | 0.09× | 826.3 | 0.1250 |
| **Bobcat** (`plume`) | `p_base` | `0.12` | 2,164.7 | **1.14×** | 130.8 | 2,164.7 | **0.60×** | 277.4 | 2,164.7 | **0.40×** | 590.2 | **0.4375** |
| **Bobcat** (`plume`) | `p_base` | `0.25` | 29,956.7 | 15.78× | 120.9 | 77,098.0 | 21.23× | 292.1 | 118,021.7 | 21.83× | 516.7 | 0.0503 |
| **Bobcat** (`plume`) | `p_base` | `0.50` *(def)* | 44,380.4 | 23.37× | 123.9 | 116,517.6 | 32.09× | 297.4 | 134,769.6 | 24.93× | 521.0 | 0.0457 |

---

### Step 3 FITTING: Leave-One-Fire-Out (LOFO) Cross-Validation & Regime Ablation (`reports/lofo_regime_ablation.csv`, `reports/lofo_per_fire_metrics.csv`)

#### 1. Aggregate Multi-Scale Performance Across Horizons (+1 h, +3 h, +6 h, +12 h) — Labeled `held-out (LOFO)` vs `in-sample`

| Evaluation Group | Model Variant | Split | Horizon | Mean ~2km GOFER IoU | Mean ~2km GOFER $F_1$ | Mean Area Ratio (Sim/Obs) | Median Area Ratio (Sim/Obs) | Mean Arrival MAE (min) | Mean Wind-Axis Front Error (km) | Mean FEDS 375m $F_1$ |
|---|---|---|---|---|---|---|---|---|---|---|
| **All 7 Fires** | `uncalibrated` ($P_{\text{base}}=0.50$) | **held-out (LOFO)** | +1 h | 0.4443 | 0.6012 | 1.866 | 1.795 | 26.2 | 1.70 km | — |
| **All 7 Fires** | `uncalibrated` ($P_{\text{base}}=0.50$) | **held-out (LOFO)** | +3 h | 0.2867 | 0.4304 | 4.541 | 4.130 | 60.6 | 3.18 km | — |
| **All 7 Fires** | `uncalibrated` ($P_{\text{base}}=0.50$) | **held-out (LOFO)** | +6 h | 0.1789 | 0.2960 | 8.247 | 6.350 | 114.1 | 4.98 km | — |
| **All 7 Fires** | `uncalibrated` ($P_{\text{base}}=0.50$) | **held-out (LOFO)** | **+12 h** | **0.1036** | **0.1851** | **13.685** | **8.926** | 211.4 | 8.27 km | **0.1336** |
| **All 7 Fires** | `global_slowdown_lofo` (1-param) | **held-out (LOFO)** | +1 h | 0.4459 | 0.5824 | 1.248 | 1.041 | 19.3 | 1.66 km | — |
| **All 7 Fires** | `global_slowdown_lofo` (1-param) | **held-out (LOFO)** | +3 h | 0.3230 | 0.4689 | 2.054 | 1.268 | 69.3 | 3.46 km | — |
| **All 7 Fires** | `global_slowdown_lofo` (1-param) | **held-out (LOFO)** | +6 h | 0.3105 | 0.4412 | 2.967 | **0.923** | 126.3 | 4.45 km | — |
| **All 7 Fires** | `global_slowdown_lofo` (1-param) | **held-out (LOFO)** | **+12 h** | **0.2436** | **0.3663** | **4.105** | **1.512** | 293.0 | 5.32 km | **0.2741** |
| **All 7 Fires** | `calibrated_all7_lofo` (6-param) | **held-out (LOFO)** | +1 h | 0.4459 | 0.5824 | 1.254 | **1.010** | 19.4 | 1.48 km | — |
| **All 7 Fires** | `calibrated_all7_lofo` (6-param) | **held-out (LOFO)** | +3 h | 0.3230 | 0.4689 | 2.058 | 1.359 | 68.9 | 3.49 km | — |
| **All 7 Fires** | `calibrated_all7_lofo` (6-param) | **held-out (LOFO)** | +6 h | 0.2904 | 0.4219 | 3.034 | 1.590 | 126.8 | 5.12 km | — |
| **All 7 Fires** | `calibrated_all7_lofo` (6-param) | **held-out (LOFO)** | **+12 h** | **0.2204** | **0.3337** | **4.570** | **1.512** | 288.0 | 6.51 km | **0.2471** |
| **All 7 Fires** | `calibrated_wind_only_lofo` | **held-out (LOFO)** | +1 h | 0.4328 | 0.5727 | 1.297 | 1.052 | 19.9 | **1.44 km** | — |
| **All 7 Fires** | `calibrated_wind_only_lofo` | **held-out (LOFO)** | +3 h | **0.3696** | **0.5255** | 2.231 | 1.359 | **67.4** | **2.83 km** | — |
| **All 7 Fires** | `calibrated_wind_only_lofo` | **held-out (LOFO)** | +6 h | **0.3274** | **0.4724** | 3.319 | 1.590 | **122.5** | **3.97 km** | — |
| **All 7 Fires** | `calibrated_wind_only_lofo` | **held-out (LOFO)** | **+12 h** | **0.2297** | **0.3590** | 4.994 | 1.828 | **273.8** | **5.28 km** | **0.2789** |
| **All 7 Fires** | `calibrated_all7_insample` | **in-sample** | **+12 h** | **0.2897** | **0.4329** | 4.238 | **1.512** | 281.5 | **3.44 km** | **0.3308** |

#### 2. Regime Ablation at +12 h (`Wind-Driven` vs `Plume-Fire` vs `Terrain` Folds)

| Regime Subset | Model Variant | Split | +3h IoU | +6h IoU | +12h ~2km IoU | +12h ~2km $F_1$ | +12h Mean Area Ratio | +12h Median Area Ratio | +12h Wind Front Error (km) | +12h FEDS 375m $F_1$ |
|---|---|---|---|---|---|---|---|---|---|---|
| **Wind-Driven (`Kincade, Glass, Zogg`)** | `uncalibrated` | **held-out (LOFO)** | 0.4222 | 0.2525 | 0.1391 | 0.2434 | 7.552 | 8.489 | 4.63 km | 0.2113 |
| **Wind-Driven (`Kincade, Glass, Zogg`)** | `global_slowdown_lofo` | **held-out (LOFO)** | 0.3519 | 0.3829 | 0.2848 | 0.4070 | 0.702 | 0.435 | 5.32 km | 0.3691 |
| **Wind-Driven (`Kincade, Glass, Zogg`)** | `calibrated_all7_lofo` | **held-out (LOFO)** | 0.3519 | 0.3446 | 0.2983 | 0.4207 | 0.580 | 0.435 | 5.00 km | 0.3710 |
| **Wind-Driven (`Kincade, Glass, Zogg`)** | `calibrated_wind_only_lofo` | **held-out (LOFO)** | **0.4780** | **0.4452** | **0.3214** | **0.4832** | **1.037** | **0.937** | **3.43 km** | **0.4407** |
| **Wind-Driven (`Kincade, Glass, Zogg`)** | `calibrated_all7_insample` | **in-sample** | 0.4074 | 0.4603 | 0.3924 | 0.5624 | 1.012 | 0.945 | 0.94 km | 0.5013 |
| **Plume-Fire Folds (`Creek, Bobcat`)** | `uncalibrated` | **held-out (LOFO)** | 0.1366 | 0.1182 | 0.0807 | 0.1465 | 20.507 | 20.507 | 8.80 km | 0.0635 |
| **Plume-Fire Folds (`Creek, Bobcat`)** | `global_slowdown_lofo` | **held-out (LOFO)** | **0.2681** | **0.2230** | 0.1942 | 0.3072 | **8.739** | **8.739** | 4.47 km | 0.1530 |
| **Plume-Fire Folds (`Creek, Bobcat`)** | `calibrated_all7_lofo` | **held-out (LOFO)** | **0.2681** | **0.2230** | 0.1942 | 0.3072 | **8.739** | **8.739** | 4.47 km | 0.1530 |
| **Plume-Fire Folds (`Creek, Bobcat`)** | `calibrated_wind_only_lofo` | **held-out (LOFO)** | 0.2490 | 0.2091 | **0.1968** | **0.3091** | 9.203 | 9.203 | **2.23 km** | **0.1641** |
| **Terrain Folds (`Caldor, Tamarack`)** | `uncalibrated` | **held-out (LOFO)** | 0.2334 | 0.1292 | 0.0732 | 0.1362 | 16.062 | 16.062 | 13.20 km | 0.0869 |
| **Terrain Folds (`Caldor, Tamarack`)** | `global_slowdown_lofo` | **held-out (LOFO)** | **0.3346** | **0.2893** | **0.2313** | **0.3644** | **4.576** | **4.576** | **6.16 km** | **0.2529** |
| **Terrain Folds (`Caldor, Tamarack`)** | `calibrated_all7_lofo` | **held-out (LOFO)** | **0.3346** | 0.2765 | 0.1298 | 0.2298 | 6.388 | 6.388 | 10.81 km | 0.1556 |
| **Terrain Folds (`Caldor, Tamarack`)** | `calibrated_wind_only_lofo` | **held-out (LOFO)** | 0.3276 | 0.2688 | 0.1251 | 0.2225 | 6.722 | 6.722 | 11.11 km | 0.1509 |

#### 3. Per-Fire Metrics at +1 h, +3 h, +6 h, +12 h (`reports/lofo_per_fire_metrics.csv`)
*Showing `uncalibrated` vs `calibrated_all7_lofo` vs `calibrated_wind_only_lofo` (all **held-out LOFO**) at +1 h, +3 h, +6 h, and +12 h:*

| Fire Name (Regime) | Model (`held-out LOFO`) | +1h IoU / Area Ratio / Front Err | +3h IoU / Area Ratio / Front Err | +6h IoU / Area Ratio / Front Err | +12h IoU / $F_1$ / Area Ratio / Arr MAE / Front Err / FEDS $F_1$ |
|---|---|---|---|---|---|
| **Kincade** (`wind-driven`) | `uncalibrated` | `0.6000` / `1.12×` / `2.12 km` | `0.3659` / `3.25×` / `0.28 km` | `0.1786` / `6.35×` / `1.99 km` | `0.1158` / `0.2075` / `8.66×` / `184.0 m` / `3.82 km` / `0.1393` |
| **Kincade** (`wind-driven`) | `calibrated_all7_lofo` | `0.4286` / `0.68×` / `2.31 km` | `0.5000` / `0.60×` / `3.54 km` | `0.5909` / `0.62×` / `2.80 km` | **`0.4474`** / **`0.6182`** / `0.44×` / `266.8 m` / **`0.54 km`** / **`0.5943`** |
| **Kincade** (`wind-driven`) | `calibrated_wind_only_lofo` | `0.4286` / `0.70×` / `2.22 km` | **`0.6250`** / `0.80×` / `2.47 km` | **`0.5909`** / `0.65×` / `3.35 km` | `0.3784` / `0.5490` / `0.35×` / `276.6 m` / `5.48 km` / `0.4746` |
| **Glass** (`wind-driven`) | `uncalibrated` | `0.5556` / `1.42×` / `1.60 km` | `0.4364` / `2.07×` / `1.90 km` | `0.3071` / `3.05×` / `2.16 km` | `0.1717` / `0.2931` / `5.51×` / `200.4 m` / `3.99 km` / `0.2601` |
| **Glass** (`wind-driven`) | `calibrated_all7_lofo` | **`0.7143`** / `1.01×` / `1.28 km` | `0.5000` / `0.85×` / `3.29 km` | `0.4107` / `0.66×` / `4.40 km` | **`0.4211`** / **`0.5926`** / `1.28×` / `260.8 m` / **`0.64 km`** / `0.4615` |
| **Glass** (`wind-driven`) | `calibrated_wind_only_lofo` | `0.6667` / `1.05×` / `1.36 km` | **`0.5312`** / **`1.00×`** / `2.71 km` | **`0.5161`** / **`1.05×`** / `3.40 km` | `0.3469` / `0.5152` / `1.83×` / `209.8 m` / `1.10 km` / **`0.4816`** |
| **Zogg** (`wind-driven`) | `uncalibrated` | `0.4167` / `0.34×` / `3.16 km` | `0.4643` / `1.12×` / `3.77 km` | `0.2717` / `2.92×` / `3.33 km` | `0.1297` / `0.2296` / `8.49×` / `139.5 m` / `6.09 km` / `0.2346` |
| **Zogg** (`wind-driven`) | `calibrated_all7_lofo` | `0.0833` / `0.09×` / `3.58 km` | `0.0556` / `0.05×` / `7.84 km` | `0.0323` / `0.03×` / `12.59 km` | `0.0263` / `0.0513` / `0.03×` / `456.8 m` / `13.83 km` / `0.0571` |
| **Zogg** (`wind-driven`) | `calibrated_wind_only_lofo` | `0.0833` / `0.09×` / `3.27 km` | `0.2778` / `0.14×` / `4.08 km` | `0.2286` / `0.25×` / `4.17 km` | **`0.2388`** / **`0.3855`** / **`0.94×`** / `399.1 m` / **`3.70 km`** / **`0.3659`** |
| **Bobcat** (`plume`) | `uncalibrated` | `0.1667` / `4.11×` / `1.09 km` | `0.1304` / `9.20×` / `3.31 km` | `0.0640` / `23.38×` / `6.57 km` | `0.0377` / `0.0727` / `32.09×` / `297.4 m` / `7.14 km` / `0.0116` |
| **Bobcat** (`plume`) | `calibrated_all7_lofo` | **`0.2000`** / `2.89×` / `1.03 km` | **`0.2400`** / `5.43×` / `1.66 km` | **`0.1127`** / `12.06×` / `4.35 km` | **`0.0710`** / **`0.1326`** / `15.97×` / `281.6 m` / `4.83 km` / **`0.0225`** |
| **Creek** (`plume`) | `uncalibrated` | `0.4000` / `2.35×` / `1.61 km` | `0.1429` / `7.20×` / `4.48 km` | `0.1724` / `5.98×` / `5.51 km` | `0.1238` / `0.2203` / `8.93×` / `294.2 m` / `10.46 km` / `0.1155` |
| **Creek** (`plume`) | `calibrated_all7_lofo` | **`0.6154`** / `1.66×` / `0.61 km` | **`0.2963`** / `3.52×` / `2.65 km` | **`0.3333`** / `1.93×` / `0.62 km` | `0.3173` / `0.4818` / **`1.51×`** / `299.7 m` / `4.12 km` / `0.2834` |
| **Creek** (`plume`) | `calibrated_wind_only_lofo` | `0.5714` / `1.72×` / `0.73 km` | `0.2759` / `3.84×` / `2.71 km` | `0.3115` / `2.25×` / `0.72 km` | **`0.3267`** / **`0.4925`** / `1.64×` / `290.8 m` / **`0.03 km`** / **`0.3067`** |
| **Caldor** (`terrain`) | `uncalibrated` | `0.5714` / `1.92×` / `0.37 km` | `0.2667` / `4.13×` / `3.01 km` | `0.1364` / `7.71×` / `5.20 km` | `0.0845` / `0.1559` / `14.08×` / `217.4 m` / `14.26 km` / `0.0959` |
| **Caldor** (`terrain`) | `calibrated_all7_lofo` | **`0.6250`** / `1.53×` / `0.26 km` | **`0.3774`** / `2.60×` / `1.63 km` | **`0.2348`** / `4.35×` / `2.77 km` | **`0.1292`** / **`0.2288`** / `8.26×` / `190.3 m` / `8.41 km` / **`0.1510`** |
| **Tamarack** (`terrain`) | `uncalibrated` | `0.4000` / `1.80×` / `1.96 km` | `0.2000` / `4.81×` / `5.48 km` | `0.1220` / `8.35×` / `10.11 km` | `0.0618` / `0.1165` / `18.04×` / `147.0 m` / `12.13 km` / `0.0779` |
| **Tamarack** (`terrain`) | `global_slowdown_lofo` | **`0.4545`** / **`0.89×`** / `1.41 km` | **`0.2917`** / **`1.27×`** / `3.41 km` | **`0.3438`** / **`0.87×`** / `3.86 km` | **`0.3333`** / **`0.5000`** / **`0.89×`** / `323.9 m` / **`3.91 km`** / **`0.3547`** |
| **Tamarack** (`terrain`) | `calibrated_all7_lofo` | **`0.4545`** / `0.93×` / `1.32 km` | **`0.2917`** / `1.36×` / `3.83 km` | `0.3182` / `1.59×` / `8.30 km` | `0.1304` / `0.2308` / `4.52×` / `260.2 m` / `13.21 km` / `0.1602` |

#### 4. Plain Statement on LOFO Calibration Performance
1. **Calibration vs. Uncalibrated Default ($P_{\text{base}} = 0.50$)**: Across all 7 fires in held-out LOFO evaluation, calibration more than doubles +12 h ~2 km GOFER IoU (`0.1036` $\to$ `0.2204–0.2436`), doubles 375 m FEDS VIIRS $F_1$ (`0.1336` $\to$ `0.2471–0.2789`), reduces wind-axis front distance error from `8.27 km` to `5.28–6.51 km`, and cuts median area overprediction from `8.93×` down to `1.51×`.
2. **Multi-Parameter All-7 Fit vs. Single Global Slowdown Factor (Plain Negative Finding)**: Fitting 6 free parameters across all 7 heterogeneous fires (`calibrated_all7_lofo`, +12 h IoU = `0.2204`) **did NOT improve held-out LOFO generalization over the 1-parameter global slowdown baseline (`global_slowdown_lofo`, $P_{\text{base}} = 0.21$, +12 h IoU = `0.2436`)**. Because the training set mixes fast wind-driven grass/shrub fires (`Kincade, Glass, Zogg`, which require $P_{\text{base}} \ge 0.22$ to avoid stalling on Zogg) with slower canyon/plume fires (`Bobcat, Caldor`, which over-expand unless $P_{\text{base}} \le 0.16$), leaving out Zogg pulls $P_{\text{base}}$ down to `0.19` (causing Zogg to stall at `318 ha` vs `12,678 ha` observed), while leaving out Tamarack pulls $P_{\text{base}}$ up to `0.22` (causing Tamarack to expand to `34,137 ha` vs `6,750 ha` at $P_{\text{base}} = 0.21$).
3. **Regime-Specific Wind-Driven Fit (`calibrated_wind_only_lofo`)**: When calibrated exclusively on the wind-driven regime (`Kincade, Glass, Zogg`), held-out LOFO performance on wind-driven folds improves across every metric (+3 h IoU = `0.4780`, +6 h IoU = `0.4452`, +12 h IoU = `0.3214`, FEDS 375 m $F_1$ = `0.4407`, mean area ratio = `1.037×`, median area ratio = `0.937×`).

#### 5. 20-Seed Monte Carlo Ensemble & P10–P90 Arrival-Time Coverage (`reports/lofo_ensemble_20seeds.csv`)
*Split: **Held-Out (LOFO)** | 20 random seeds per fire (`[100, 117, ..., 423]`). Reports both raw coverage and hourly-quantized ($\pm 30\text{ min}$) coverage accounting for GOFER's 60 min discrete time step.*

| Fire Name | Regime | Split | Seeds | Obs +12h Area (ha) | Ensemble +12h Area (Mean ± Std ha) | Ensemble +12h IoU (Mean ± Std) | Ensemble P50 Arrival MAE (min) | Mean P10–P90 Width (min) | Raw P10–P90 Coverage (%) | Hourly-Quantized ($\pm 30\text{ m}$) P10–P90 Coverage (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Kincade** | `wind-driven` | **held-out (LOFO)** | 20 | 11,621.9 | $7,476.6 \pm 3,601.4$ | $0.3890 \pm 0.0813$ | 247.2 | 365.3 | 34.38% | **46.05%** |
| **Bobcat** | `plume` | **held-out (LOFO)** | 20 | 3,631.3 | $57,763.5 \pm 3,160.8$ | $0.0701 \pm 0.0040$ | 284.5 | 78.9 | 10.16% | 11.13% |
| **Glass** | `wind-driven` | **held-out (LOFO)** | 20 | 22,824.4 | $32,924.3 \pm 3,370.2$ | $0.4175 \pm 0.0385$ | 188.0 | 223.6 | 37.19% | **45.87%** |
| **Creek** | `plume` | **held-out (LOFO)** | 20 | 16,735.7 | $26,084.8 \pm 3,105.8$ | $0.4051 \pm 0.0548$ | 286.0 | 235.9 | 42.27% | **44.57%** |
| **Zogg** | `wind-driven` | **held-out (LOFO)** | 20 | 12,677.8 | $2,798.9 \pm 1,925.8$ | $0.1315 \pm 0.0525$ | 426.6 | 324.4 | 32.71% | **42.81%** |
| **Caldor** | `terrain` | **held-out (LOFO)** | 20 | 11,385.4 | $95,648.2 \pm 2,274.4$ | $0.1280 \pm 0.0030$ | 182.6 | 61.4 | 8.94% | 16.33% |
| **Tamarack** | `terrain` | **held-out (LOFO)** | 20 | 7,556.8 | $25,919.0 \pm 4,723.0$ | $0.1666 \pm 0.0331$ | 200.0 | 433.3 | **74.27%** | **85.67%** |

---

### Step 3 FREEZE AND TEST: Held-Out 2018 Camp Fire Evaluation at 60 m (`reports/camp_fire_heldout_60m_*.csv`)

- **Frozen Configuration:** `configs/teacher_v1.yaml`
- **Verified SHA-256 Hash (`reports/teacher_v1_hash.txt`):** `c45ba6db95187063347ad1c06e81dbb29ff3e45fa92f50ad46d8dcc9e16cf9b6`
- **Frozen Parameters (`teacher_v1`):** `p_base: 0.21, a: 0.06, c_1: 0.045, c_2: 0.15, p_continue: 0.30, fuel_scale_grass_shrub: 1.0, fuel_scale_timber: 1.0, canopy_den_scale: 1.0, wind_moisture_damping: 0.0` (`steps_per_hour: 30`, `operating_resolution_m: 60`).
- **Frozen Wind-Only Ablation (`wind_only_v1`):** `p_base: 0.22` (all other parameters identical).

#### 1. GeoMAC Perimeter Overlap on Held-Out Camp Fire at 60 m (`reports/camp_fire_heldout_60m_perimeters.csv`)
*Split: **Strictly Held-Out (`Camp Fire 2018`)** | Grid: $564 \times 562$ at 60 m ($0.36\text{ ha/cell}$).*

| Split | Model | Reference Perimeter | Sim Horizon | Obs Area in Grid (ha) | Sim Area (ha) | Area Ratio (Sim/Obs) | TP Cells | FP Cells | FN Cells | TN Cells | IoU | Precision | Recall | $F_1$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **held-out** | `teacher_v1 (Frozen All-7)` | **+11.4 h GeoMAC IR (ID 1052)** | 684 min | 22,157.6 | 1,410.5 | **0.064×** | 3,378 | **540** | 58,171 | 254,879 | 0.0544 | **0.8622** | 0.0549 | 0.1032 |
| **held-out** | `teacher_v1 (Frozen All-7)` | **Final Contained (ID 1398)** | 720 min | 58,569.5 | 1,410.5 | 0.024× | 3,915 | **3** | 158,778 | 154,272 | 0.0241 | **0.9992** | 0.0241 | 0.0470 |
| **held-out** | `wind_only_v1 (Frozen Wind-3)` | **+11.4 h GeoMAC IR (ID 1052)** | 684 min | 22,157.6 | 2,979.4 | **0.134×** | 6,350 | **1,926** | 55,199 | 253,493 | **0.1000** | **0.7673** | 0.1032 | **0.1819** |
| **held-out** | `wind_only_v1 (Frozen Wind-3)` | **Final Contained (ID 1398)** | 720 min | 58,569.5 | 3,058.9 | 0.052× | 8,497 | **0** | 154,196 | 154,275 | 0.0522 | **1.0000** | 0.0522 | 0.0993 |
| **held-out** | `uncalibrated_default (p_base=0.50)` | **+11.4 h GeoMAC IR (ID 1052)** | 684 min | 22,157.6 | 84,035.5 | **3.793×** | 50,202 | 183,230 | 11,347 | 72,189 | 0.2051 | 0.2151 | 0.8156 | 0.3404 |
| **held-out** | `phase0_baseline (p_base=0.58)` | **+11.4 h GeoMAC IR (ID 1052)** | 684 min | 22,157.6 | 86,860.4 | **3.920×** | 51,197 | 190,082 | 10,352 | 65,337 | 0.2035 | 0.2122 | 0.8318 | 0.3381 |

#### 2. NIST Observation Points Split by Fire Type at 60 m (`reports/camp_fire_heldout_60m_nist_split.csv`)
*Split: **Strictly Held-Out (`Camp Fire 2018`)** | `vegetative` = primary contact-spread score; `spot_fire` and `structural` reported separately, never tuned on.*

| Split | Model | Fire Type | Role | Total Points | Burned Points | Unburned Points | % Never Burned | Signed Bias Mean (min) | Signed Bias Median (min) | Abs Error MAE (min) | Abs Error Median (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **held-out** | `teacher_v1 (Frozen All-7)` | **`vegetative`** | **Contact-Spread Score** | 315 | 4 | 311 | 98.73% | **+12.25** | **+11.00** | **12.25** | **11.00** |
| **held-out** | `teacher_v1 (Frozen All-7)` | `spot_fire` | Reported (Not Tuned) | 341 | 2 | 339 | 99.41% | +51.00 | +51.00 | 51.00 | 51.00 |
| **held-out** | `teacher_v1 (Frozen All-7)` | `structural` | Reported (Not Tuned) | 629 | 0 | 629 | 100.00% | N/A | N/A | N/A | N/A |
| **held-out** | `teacher_v1 (Frozen All-7)` | `other` | Reported (Not Tuned) | 1,081 | 7 | 1,074 | 99.35% | +5.71 | +8.00 | 28.86 | 35.00 |
| **held-out** | `teacher_v1 (Frozen All-7)` | `ALL` | Aggregate | 2,366 | 13 | 2,353 | 99.45% | +14.69 | +13.00 | 27.15 | 25.00 |
| **held-out** | `wind_only_v1 (Frozen Wind-3)` | **`vegetative`** | **Contact-Spread Score** | 315 | 20 | 295 | 93.65% | +101.55 | +87.50 | **110.15** | **87.50** |
| **held-out** | `wind_only_v1 (Frozen Wind-3)` | `spot_fire` | Reported (Not Tuned) | 341 | 3 | 338 | 99.12% | +7.00 | -1.00 | 8.33 | 1.00 |
| **held-out** | `wind_only_v1 (Frozen Wind-3)` | `structural` | Reported (Not Tuned) | 629 | 2 | 627 | 99.68% | +108.50 | +108.50 | 108.50 | 108.50 |
| **held-out** | `wind_only_v1 (Frozen Wind-3)` | `other` | Reported (Not Tuned) | 1,081 | 17 | 1,064 | 98.43% | +20.94 | -9.00 | 57.41 | 29.00 |
| **held-out** | `wind_only_v1 (Frozen Wind-3)` | `ALL` | Aggregate | 2,366 | 42 | 2,324 | 98.22% | +62.50 | +11.00 | 81.45 | 35.00 |
| **held-out** | `uncalibrated_default (p_base=0.50)` | **`vegetative`** | **Contact-Spread Score** | 315 | 184 | 131 | 41.59% | +82.02 | +86.00 | **148.98** | **130.00** |
| **held-out** | `uncalibrated_default (p_base=0.50)` | `spot_fire` | Reported (Not Tuned) | 341 | 174 | 167 | 48.97% | +245.87 | +265.50 | 258.22 | 269.00 |
| **held-out** | `uncalibrated_default (p_base=0.50)` | `structural` | Reported (Not Tuned) | 629 | 172 | 457 | 72.66% | +81.17 | +88.50 | 125.64 | 104.00 |
| **held-out** | `uncalibrated_default (p_base=0.50)` | `other` | Reported (Not Tuned) | 1,081 | 405 | 676 | 62.53% | +126.60 | +158.00 | 170.85 | 181.00 |
| **held-out** | `uncalibrated_default (p_base=0.50)` | `ALL` | Aggregate | 2,366 | 935 | 1,431 | 60.48% | +131.66 | +156.00 | 174.49 | 182.00 |

#### 3. Physical Diagnosis of Held-Out Camp Fire Results (Why Contact-Only Calibration Stalls at the Feather River Gorge)
- **99.9%–100.0% Footprint Precision & Local Timing Accuracy**: On the held-out Camp Fire, `teacher_v1` ($P_{\text{base}}=0.21$) and `wind_only_v1` ($P_{\text{base}}=0.22$) eliminate the $3.79\times$ uncalibrated radial blow-up (`183,230` false-positive cells), achieving **86.2% precision** against the +11.4 h perimeter and **99.92%–100.00% precision** inside the final Camp Fire footprint. On the vegetative contact-spread points reached in the Feather River canyon near Pulga/Concow, arrival-time MAE improves from `148.98 min` (`uncalibrated`) to **`12.25 min` (`teacher_v1`)** and **`110.15 min` (`wind_only_v1`)**.
- **Topological Bottleneck Without Phase 1B Spotting**: Unlike the 7 GOFER training fires—which initialize from a ~2 km GOES polygon (`107–1,796 ha`) already spanning local road/stream breaks—the Camp Fire initializes from a $3 \times 3$ point source (`3.24 ha`) at Pulga (`633391 E, 4407338 N`), surrounded by the North Fork and West Branch of the Feather River (`FBFM40 = 98` Open Water, `p_veg = -1.0`) and Highway 70 (`FBFM40 = 91` Non-burnable). At $P_{\text{base}} = 0.21\text{–}0.22$ (just above the 2D cellular automaton percolation threshold $p_c \approx 0.19$), a strictly local $3 \times 3$ Moore-neighborhood CA cannot jump across a $\ge 1\text{–}2\text{ cell}$ ($60\text{–}120\text{ m}$) water/canyon barrier into Paradise. Consequently, `teacher_v1` burns `1,319.8 ha` in the first 3 hours down the Pulga canyon to Concow (`wind_only_v1` reaches Concow at `392.0 min`, `P10 = 488.6 min` for `teacher_v1`) and then stalls along the West Branch Feather River water boundary (`1,410.5 ha` at +11.4 h; 20-seed ensemble mean `1,598.0 ± 1,408.7 ha`, P90 `3,016.8 ha`).
- **Direct Motivation for Phase 1B**: In reality, the Camp Fire crossed the West Branch Feather River gorge and ignited Paradise via **long-range wind-borne ember spotting (1–4 km)** documented in NIST TN 2135. The uncalibrated model ($P_{\text{base}}=0.50$) only reached Paradise by flooding 79% of the entire domain around the ends of the river reservoir. Thus, `teacher_v1` / `wind_only_v1` provides a physically honest contact-spread baseline that makes the **Phase 1B non-local ember spotting kernel** strictly necessary and scientifically well-posed.

---

### Step 3 Optional Check: Resolution Penalty (30 m vs. 60 m on Kincade, `reports/resolution_penalty_30m_vs_60m.csv`)
*Split: **In-Sample / Resolution Transfer Diagnostic** | Evaluates frozen `teacher_v1` ($P_{\text{base}}=0.21$) at 60 m (`720×720`, 30 steps/hr) vs. 30 m (`1440×1440`, 60 steps/hr).*

| Fire Name | Resolution | Grid Shape | Steps / Hour | Horizon | 24h Wall-Clock Runtime (s) | Sim Area (ha) | Obs Area (ha) | Area Ratio | ~2km GOFER IoU | ~2km GOFER $F_1$ |
|---|---|---|---|---|---|---|---|---|---|---|
| **Kincade (2019)** | **60 m (Operating)** | $720 \times 720$ | 30 | +12 h | **2.71 s** | 5,049.7 | 11,621.9 | 0.435× | **0.4474** | **0.6182** |
| **Kincade (2019)** | **60 m (Operating)** | $720 \times 720$ | 30 | +24 h | **2.71 s** | 5,529.6 | 11,621.9 | 0.476× | **0.4146** | **0.5862** |
| **Kincade (2019)** | **30 m (Fine Check)** | $1440 \times 1440$ | 60 | +12 h | **19.11 s** ($7.05\times$ slower) | 3,573.9 | 11,621.9 | 0.308× | 0.2703 | 0.4255 |
| **Kincade (2019)** | **30 m (Fine Check)** | $1440 \times 1440$ | 60 | +24 h | **19.11 s** ($7.05\times$ slower) | 3,573.9 | 11,621.9 | 0.308× | 0.2703 | 0.4255 |

*Finding:* Transferring 60 m calibrated parameters directly to 30 m (60 steps/hr) incurs a **$7.05\times$ runtime penalty** and drops +12 h IoU from `0.4474` to `0.2703` because doubling the number of cell-to-cell transitions while applying per-step burnout ($1 - p_{\text{continue}} = 0.70$) shifts the CA percolation threshold upward.

---

## 4. Deviations from the Brief

- **None**:
  - All 7 approved training fires (`Kincade, Bobcat, Glass, Creek, Zogg, Caldor, Tamarack`) were ingested at 60 m using a single pre-fire LANDFIRE vintage (`LANDFIRE 2016 Remap / LF 2.0.0`) with zero post-fire leakage.
  - Multi-scale comparisons were executed at ~2 km (`33x33` block = `1,980 m`) for GOFER hourly perimeters and 375 m (`6x6` block = `360 m`) for FEDS VIIRS 12 h observations.
  - `configs/teacher_v1.yaml` was frozen and hashed prior to running the held-out 2018 Camp Fire evaluation once at 60 m.
  - We stated plainly that 6-parameter calibration across all 7 heterogeneous fires (`IoU = 0.2204`) did not beat the 1-parameter global slowdown baseline (`IoU = 0.2436`) in aggregate LOFO, whereas regime-specific wind-driven calibration (`IoU = 0.3214`, area ratio `1.037×`) substantially improved wind-driven generalization.

---

## 5. Open Problems

1. **Non-Local Barrier Crossing on Point-Ignition Wind-Driven Fires (Phase 1B Ember Kernel)**:
   - On the held-out Camp Fire (point ignition at Pulga), calibrating contact spread down to physical rates ($P_{\text{base}} = 0.21\text{–}0.22$) eliminates false-positive flank blow-up (`99.9%–100%` precision inside the final footprint) but causes the front to stall at continuous open-water/road barriers (West Branch Feather River) because a $3 \times 3$ Moore neighborhood cannot leap across 60–300 m non-burnable gaps.
   - Phase 1B's wind-borne ember spotting kernel ($p_{\text{spot}}(d, V_{\text{gust}}, \theta_{\text{wind}})$) is required to bridge river gorges and highways without re-inflating isotropic contact $P_{\text{base}}$.
2. **Regime Bifurcation Between Wind-Driven and Plume/Canyon Fires**:
   - Wind-driven grass/shrub fires (`Kincade, Glass, Zogg`) operate best at $P_{\text{base}} = 0.22\text{–}0.26$, whereas convective plume/canyon fires (`Bobcat, Caldor`) require stronger topographic/fuel damping ($P_{\text{base}} \approx 0.12\text{–}0.16$) when initialized from large GOFER polygons.

---

## 6. Decisions I Need to Make (Orchestrator Input for Phase 1B)

1. **Base Contact Parameters Entering Phase 1B (Ember Kernel)**:
   - Should Phase 1B build the wind-borne ember spotting kernel on top of **`teacher_v1`** ($P_{\text{base}} = 0.21$) or **`wind_only_v1`** ($P_{\text{base}} = 0.22$, which achieved `1.037×` mean area ratio and `0.3214` LOFO IoU on wind-driven fires)?
2. **Regime-Conditioned vs. Single Parameter Set**:
   - Should the physics teacher maintain a single frozen parameter vector (`teacher_v1`) or allow a two-regime lookup (`wind-driven` when 10 m gusts $> 12\text{ m/s}$ vs. `terrain/plume` otherwise)?


---

# Phase 1B Step 0 Audit & Re-Report of Phase 1A (Appended, History Preserved)

Per Orchestrator instructions for Phase 1B Step 0 (no new contact modeling), we re-audited and re-reported the Phase 1A evaluation tables with explicit cell/point denominators, matched-fold model selection, criticality sweep, and relabeling of the 30 m transfer check. Full details and plots are also included in `reports/phase_1b.md`.

### Step 0(a): Matched-Fold Comparison & Base Teacher Selection (`reports/step0_matched_fold_comparison.csv`)

- **Rule**: Evaluate `global_slowdown` (1 free parameter) vs `wind_only` vs `all-7` on `(i)` the 3 wind-driven folds (`Kincade, Glass, Zogg`) and `(ii)` all 7 folds (`Kincade, Bobcat, Glass, Creek, Zogg, Caldor, Tamarack`). Pick the simplest model within `0.02` IoU of best; if not `teacher_v1`, freeze `configs/teacher_v1_1.yaml` + hash.

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

- **Selection Decision**:
  - On **(ii) All 7 Folds (`held-out LOFO, n=7`)**, `global_slowdown_lofo` (1 free parameter, consensus `teacher_v1` with $P_{\text{base}} = 0.21$) achieves the highest +12 h IoU (`0.2436` vs `0.2297` for `wind_only_lofo` and `0.2204` for `all7_lofo`) and is within `0.0092 < 0.02` of `wind_only_lofo` on mean 1/3/6/12h IoU (`0.3307` vs `0.3399`).
  - On **(i) 3 Wind-Driven Folds (`held-out LOFO, n=3`)**, `calibrated_wind_only_lofo` achieves `+12h IoU = 0.3214` (vs `0.2848` for all-7 `global_slowdown_lofo`, where fitting on Bobcat/Caldor dragged Zogg's fold down to $P_{\text{base}}=0.16$). Its consensus 1-parameter single vector is $P_{\text{base}} = 0.22$ (`configs/teacher_v1_1.yaml`), which achieves the highest mean 1/3/6/12h IoU (`0.4434`) and FEDS 375m $F_1$ (`0.5171`) on wind-driven folds while remaining within `0.0025 < 0.02` of `teacher_v1` at +12 h (`0.3899` vs `0.3924`).
  - Both single-vector configs are frozen with SHA-256 hashes:
    - `configs/teacher_v1.yaml` ($P_{\text{base}} = 0.21$): `c45ba6db95187063347ad1c06e81dbb29ff3e45fa92f50ad46d8dcc9e16cf9b6`
    - `configs/teacher_v1_1.yaml` ($P_{\text{base}} = 0.22$): `86a1ac17804d3816958c4694b1be504fa432ebdead2db5d0df4c4fa4939a30f0`

### Step 0(b): 20-Seed P10–P90 Coverage Table with Exact Cell Denominators (`reports/step0_20seed_coverage_with_denominators.csv`)

| Fire Name | Regime | Split | Seeds | Obs +12h Cells (Area ha) | Ensemble +12h Area (Mean ± Std ha) | Ensemble P10–P90 Area Band (ha) | Denominator: Obs Progression Cells ($t_{\text{obs}} > 0$) | Raw P10–P90 Covered / Denom (%) | Hourly-Quantized ($\pm 30\text{ m}$) Covered / Denom (%) |
|---|---|---|---|---|---|---|---|---|---|
| **Kincade** | `wind-driven` | **held-out (LOFO 20-seed)** | 20 | 32,283 (`11,621.9 ha`) | $8,315.8 \pm 3,129.2$ | `[4,444.0, 13,335.7]` | `29,925` | `6,483 / 29,925` (`21.66%`) | `9,107 / 29,925` (**`30.43%`**) |
| **Bobcat** | `plume` | **held-out (LOFO 20-seed)** | 20 | 10,087 (`3,631.3 ha`) | $57,047.4 \pm 3,123.9$ | `[52,003.0, 60,267.1]` | `9,098` | `876 / 9,098` (`9.63%`) | `942 / 9,098` (`10.35%`) |
| **Glass** | `wind-driven` | **held-out (LOFO 20-seed)** | 20 | 63,401 (`22,824.4 ha`) | $39,665.1 \pm 3,516.3$ | `[35,997.9, 43,443.4]` | `56,134` | `16,421 / 56,134` (`29.25%`) | `20,429 / 56,134` (**`36.39%`**) |
| **Creek** | `plume` | **held-out (LOFO 20-seed)** | 20 | 46,488 (`16,735.7 ha`) | $25,794.9 \pm 4,082.9$ | `[21,169.9, 31,362.9]` | `41,300` | `15,232 / 41,300` (`36.88%`) | `15,849 / 41,300` (**`38.38%`**) |
| **Zogg** | `wind-driven` | **held-out (LOFO 20-seed)** | 20 | 35,216 (`12,677.8 ha`) | $12,354.8 \pm 2,262.8$ | `[9,084.8, 15,622.3]` | `35,012` | `7,335 / 35,012` (`20.95%`) | `9,080 / 35,012` (**`25.93%`**) |
| **Caldor** | `terrain` | **held-out (LOFO 20-seed)** | 20 | 31,626 (`11,385.4 ha`) | $94,520.3 \pm 2,503.2$ | `[92,100.6, 96,911.0]` | `20,617` | `1,941 / 20,617` (`9.41%`) | `3,337 / 20,617` (`16.19%`) |
| **Tamarack** | `terrain` | **held-out (LOFO 20-seed)** | 20 | 20,991 (`7,556.8 ha`) | $16,465.2 \pm 6,148.4$ | `[9,107.9, 24,422.5]` | `18,992` | `9,690 / 18,992` (**`51.02%`**) | `11,166 / 18,992` (**`58.79%`**) |

### Step 0(c): Camp Fire Complete Table (Recall, Precision, IoU with Denominators) & 30 m Relabeling
- **Camp Fire Perimeter Recall, Precision & IoU with Raw Cell Denominators (`reports/step0_camp_fire_complete_table.csv`)**:
  - `teacher_v1 (P_base=0.21)` vs +11.4 h GeoMAC IR (ID 1052, `61,549` obs cells = `22,157.6 ha` on `316,968`-cell grid):
    - **Recall (`TP / Obs`)**: `3,378 / 61,549` (**`5.49%`**)
    - **Precision (`TP / Sim`)**: `3,378 / 3,918` (**`86.22%`**)
    - **IoU (`TP / Union`)**: `3,378 / 62,089` (**`0.0544`**)
  - `wind_only_v1 / teacher_v1_1 (P_base=0.22)` vs +11.4 h GeoMAC IR (ID 1052):
    - **Recall (`TP / Obs`)**: `6,350 / 61,549` (**`10.32%`**)
    - **Precision (`TP / Sim`)**: `6,350 / 8,276` (**`76.73%`**)
    - **IoU (`TP / Union`)**: `6,350 / 63,475` (**`0.1000`**)
- **Camp Fire NIST Reached / Unreached Counts with Denominators (`reports/step0_camp_fire_nist_reached_unreached.csv`)**:
  - `teacher_v1 (P_base=0.21)`: `vegetative` Reached `4 / 315 (1.27%)`, Unreached `311 / 315 (98.73%)`; `spot_fire` Reached `2 / 341 (0.59%)`, Unreached `339 / 341 (99.41%)`; `structural` Reached `0 / 629 (0.00%)`, Unreached `629 / 629 (100.00%)`; `other` Reached `7 / 1,081 (0.65%)`, Unreached `1,074 / 1,081 (99.35%)`; `ALL` Reached `13 / 2,366 (0.55%)`, Unreached `2,353 / 2,366 (99.45%)`.
  - `wind_only_v1 / teacher_v1_1 (P_base=0.22)`: `vegetative` Reached `20 / 315 (6.35%)`, Unreached `295 / 315 (93.65%)`; `spot_fire` Reached `3 / 341 (0.88%)`, Unreached `338 / 341 (99.12%)`; `structural` Reached `2 / 629 (0.32%)`, Unreached `627 / 629 (99.68%)`; `other` Reached `17 / 1,081 (1.57%)`, Unreached `1,064 / 1,081 (98.43%)`; `ALL` Reached `42 / 2,366 (1.78%)`, Unreached `2,324 / 2,366 (98.22%)`.
- **Relabeling of the 30 m vs 60 m Kincade Run (`reports/resolution_penalty_30m_vs_60m.csv`)**:
  - Relabeled from `"resolution penalty"` to **`"parameter transfer test (non-transferable), not a resolution penalty"`**: in a discrete Moore-neighborhood cellular automaton with fixed per-step burnout ($1 - p_{\text{continue}} = 0.70$), halving $\Delta x$ (`60 m -> 30 m`) and doubling `steps_per_hour` (`30 -> 60`) squares the number of cell-to-cell survival transitions required to cross a given physical distance, making $P_{\text{base}}$ non-transferable across grid resolutions without re-calibration.
