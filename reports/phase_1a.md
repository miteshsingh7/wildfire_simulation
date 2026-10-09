# Phase 1A Report: Making the Teacher Trustworthy (Steps 0–2)

**Project:** `wf052` (Wildfire Spread Simulation & Agentic Routing)  
**Phase:** 1A (Physics Teacher Audit, Resolution Sensitivity, Training Set Selection)  
**Date:** October 9, 2026  
**Status:** **AWAITING ORCHESTRATOR APPROVAL** (Step 3 Calibration paused pending approval)

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
   - Queried Zenodo API (record 8327264) and the official GOFER repository (`largeFires_metadata.js`).
   - Verified that GOFER covers 28 major California wildfires across 2019–2021 at **~2–3 km GOES spatial resolution** with **hourly** progression contours.
   - Verified FEDS coverage (375 m VIIRS active fire detections at 12-hour intervals) and GeoMAC/NIFC infrared flight perimeters.
2. **Selection of 7 Diverse Training Fires (`scripts/prepare_training_fires_proposal.py`)**:
   - Selected 7 California fires balancing wind-driven extreme events, terrain-dominated Sierra conifer fires, and mixed chaparral/timber fuels.
   - Confirmed complete NOAA HRRR hourly weather on AWS S3, pre-fire LANDFIRE vintages, and verified **0% spatial overlap** with the Camp Fire bbox.
   - Output compiled to `reports/proposed_training_fires.csv`.

---

## 2. Commands Run

```bash
# 1. Step 0: Audit GeoMAC perimeter timestamps and plot area vs time
cmd /c .venv/bin/python scripts/audit_geomap_times.py

# 2. Step 0: Update area reconciliation and delete hand-typed figures
cmd /c .venv/bin/python scripts/reconcile_areas.py

# 3. Step 0: Re-run perimeter evaluation with corrected timestamps and confusion counts
cmd /c .venv/bin/python scripts/evaluate_perimeters.py

# 4. Step 1: Benchmark resolution sensitivity (30m, 60m, 90m) on MPS
cmd /c .venv/bin/python scripts/benchmark_resolution_sensitivity.py

# 5. Step 2: Query Zenodo API and parse GOFER dataset metadata
cmd /c .venv/bin/python scripts/check_gofer_zenodo.py
cmd /c .venv/bin/python scripts/parse_gofer_metadata_js.py
cmd /c .venv/bin/python scripts/parse_gofer_fires_table.py

# 6. Step 2: Compile proposed training fires table with overlap checks
cmd /c .venv/bin/python scripts/prepare_training_fires_proposal.py

# 7. Quality gate: Lint and test verification
cmd /c .venv/bin/ruff check .
cmd /c .venv/bin/pytest
```

---

## 3. Results (With Numbers)

### Step 0: Perimeter Timeline Audit & Area Reconciliation

#### 1. GeoMAC Perimeter Timeline Audit (`reports/geomac_timeline_audit.csv`)
- **Total perimeters analyzed:** 28 GeoMAC polygon shapefiles.
- **Key finding:** Object ID 1052 (`raw_perimeterdatetime: 1541699640000`, 17:54:00) represents local PST overflight time.
  - **Parsed Naive UTC:** `2018-11-08 17:54:00 UTC` (+3.4 h post-ignition) $\rightarrow$ **ERROR**
  - **Corrected UTC:** `2018-11-09 01:54:00 UTC` (**+11.4 h / 684 min** post-ignition) $\rightarrow$ **GROUND TRUTH** (NIST TN 2135 p. 127).
  - **Observed Footprint Area:** 54,586.4 acres (**22,090.3 ha**), matching the IBHS ~55,000 acre 12 h progression benchmark.

#### 2. Area Reconciliation Table (`reports/area_reconciliation.csv`)
*Domain: $1128 \times 1124$ cells (30 m resolution, $1,267,872$ total cells, $114,108.5\text{ ha}$). Unburnable cells: $126,846$ ($11.11\%$). Burnable cells: $1,141,026$ ($88.89\%$).*

| Horizon | PyTorchFire Cells | PyTorchFire Area (ha) | Sim Domain % | Cell2Fire Area (ha) | Observed Area (ha) | Observed Source | Sim / Obs Ratio |
|---|---|---|---|---|---|---|---|
| **+3 h** (180 min) | 88,406 | 7,956.5 | 6.97% | 3.96 | *None* | **No observation** (no flight 06:30–09:30 PST) | N/A |
| **+6 h** (360 min) | 351,469 | 31,632.2 | 27.72% | 12.42 | *None* | **No observation** (no flight 06:30–12:30 PST) | N/A |
| **+11.4 h** (684 min) | 963,634 | 86,727.1 | 76.00% | 32.76 | 22,090.3 | GeoMAC IR Overflight (17:54 PST, ID 1052) | **3.92×** |
| **+12 h** (720 min) | 1,012,738 | 91,146.4 | 79.88% | 32.76 | 22,090.3 | GeoMAC IR Overflight (+11.4 h ref) | **4.13×** |

#### 3. Perimeter Confusion Matrix (`reports/perimeter_evaluation_confusion.csv`)
*Evaluated on the $1128 \times 1124$ grid (30 m, 0.09 ha/cell).*

| Reference Perimeter | Evaluated Horizon | Engine | TP Cells | FP Cells | FN Cells | TN Cells | IoU | Precision | Recall | $F_1$ |
|---|---|---|---|---|---|---|---|---|---|---|
| **+11.4 h GeoMAC IR** (22,155.8 ha in grid) | **+11.4 h Sim** (684 min) | PyTorchFire | 210,080 | 753,554 | 36,095 | 268,143 | **0.2101** | 0.2180 | **0.8534** | **0.3473** |
| **+11.4 h GeoMAC IR** (22,155.8 ha in grid) | **+11.4 h Sim** (684 min) | Cell2Fire | 353 | 11 | 245,822 | 1,021,686 | 0.0014 | 0.9698 | 0.0014 | 0.0029 |
| **+11.4 h GeoMAC IR** (22,155.8 ha in grid) | **+12 h Sim** (720 min) | PyTorchFire | 221,092 | 791,646 | 25,083 | 230,051 | **0.2130** | 0.2183 | **0.8981** | **0.3512** |
| **Final Contained** (58,568.5 ha in grid) | **+12 h Sim** (720 min) | PyTorchFire | 537,961 | 474,777 | 112,800 | 142,334 | **0.4780** | 0.5312 | **0.8267** | **0.6468** |

*Takeaway:* At +11.4 h, uncalibrated PyTorchFire captures **85.3% recall** of the real fire footprint, but overpredicts flank and back spread due to high default cellular automaton transition probabilities ($P_{\text{base}} = 0.5$), resulting in an area ratio of $3.92\times$. This confirms PyTorchFire has strong directional propagation but requires calibrated slowdown parameters.

---

### Step 1: Resolution Sensitivity Benchmark

*Execution on Apple Silicon (MPS device), 12 h horizon (720 simulation steps at $\Delta t = 60\text{ s}$).*

| Resolution | Grid Dimensions | Total Grid Cells | Wall-Clock Time (s) | Step Latency (ms/step) | Speedup vs 30m | Final Burned Cells | Final Burned Area (ha) |
|---|---|---|---|---|---|---|---|
| **30 m** | $1128 \times 1124$ | 1,267,872 | 51.82 s | 71.98 ms | $1.0\times$ | 988,800 | 88,992.0 ha |
| **60 m** | $564 \times 562$ | 316,968 | **15.36 s** | **21.33 ms** | **$3.38\times$** | 300,020 | 108,007.2 ha |
| **90 m** | $376 \times 374$ | 140,624 | **8.43 s** | **11.71 ms** | **$6.15\times$** | 132,598 | 107,404.4 ha |

```
Resolution Runtime Breakdown:
30 m: [==============================] 51.82 s (71.98 ms/step)
60 m: [=========                     ] 15.36 s (21.33 ms/step) -> 3.38x speedup
90 m: [=====                         ]  8.43 s (11.71 ms/step) -> 6.15x speedup
```

#### Recommendation on Operating Resolution: **60 m**
1. **Computational Speed**: At 21.3 ms/step (15.4 s per 12 h run), 60 m yields **234 complete 12 h simulations per hour** on a single GPU/MPS device (compared to only 69 runs/hr at 30 m). This is essential for gradient-based calibration loops and multi-seed Monte Carlo ensembles.
2. **Surrogate Tile Coverage**: In Phase 2, the ML surrogate operates on standard $256 \times 256$ patches.
   - At 30 m: $256 \times 30\text{ m} = 7.68\text{ km} \times 7.68\text{ km} = 5,900\text{ ha}$. Fast wind-driven fires outrun this patch in 3–4 hours, requiring complex multi-tile stitching.
   - At 60 m: $256 \times 60\text{ m} = 15.36\text{ km} \times 15.36\text{ km} = 23,600\text{ ha}$. A single tile accommodates the entire 12 h progression of major fires (e.g., Camp Fire 12 h footprint = 22,090 ha).
3. **Observational Alignment**: 60 m is well-matched to satellite active fire products (VIIRS 375 m = $6 \times 6$ cells; GOES ~2 km = $33 \times 33$ cells) while preserving ridgelines, canyons, and road corridors.

---

### Step 2: Proposed California Training Fires (Camp Fire Held Out)

All 7 proposed fires have **0% spatial overlap with the Camp Fire domain**, complete NOAA HRRR hourly analyses on AWS S3, pre-fire LANDFIRE datasets, and multi-source progression perimeters.

| Fire Name | Year | County / Region | Dominant Driver & Regime | Progression Period | Pre-Fire LANDFIRE Vintage | HRRR Coverage | Perimeter Sources & Resolution | Final Area (Acres) | Distance from Camp Fire |
|---|---|---|---|---|---|---|---|---|---|
| **1. Kincade** | 2019 | Sonoma | **Wind-Driven**: Extreme Diablo offshore NE winds (gusts 70–90 mph), fast grass/oak spread | Oct 23 – Oct 28, 2019 | LANDFIRE 2016 Remap (LF 2.0.0) | Hourly 10m U/V, gust, RH, T (AWS S3) | GOFER (hourly, ~2 km), FEDS (12h, 375m), GeoMAC IR | 77,758 ac (31,468 ha) | 130 km SW |
| **2. Bobcat** | 2020 | Los Angeles | **Terrain-Driven / Plume**: Steep San Gabriel canyons, heavy chaparral fuels, convective plume | Sep 06 – Sep 20, 2020 | LANDFIRE 2016 Remap (LF 2.0.0) | Hourly 10m U/V, gust, RH, T (AWS S3) | GOFER (hourly), FEDS (12h), GeoMAC IR | 115,997 ac (46,943 ha) | 600 km S |
| **3. Glass** | 2020 | Napa, Sonoma | **Wind-Driven**: Diablo offshore event, high gust spotting across ridge-valley terrain | Sep 27 – Oct 05, 2020 | LANDFIRE 2016 Remap (LF 2.0.0) | Hourly 10m U/V, gust, RH, T (AWS S3) | GOFER (hourly), FEDS (12h), GeoMAC IR | 67,484 ac (27,310 ha) | 140 km SW |
| **4. Creek** | 2020 | Fresno, Madera | **Terrain & Plume-Dominated**: Sierra mixed conifer with extreme drought mortality, pyrocumulus | Sep 04 – Sep 12, 2020 | LANDFIRE 2016 Remap (LF 2.0.0) | Hourly 10m U/V, gust, RH, T (AWS S3) | GOFER (hourly), FEDS (12h), GeoMAC / NIROPS IR | 379,895 ac (153,738 ha) | 280 km SE |
| **5. Zogg** | 2020 | Shasta | **Wind-Driven**: High N-NE downslope canyon winds, rapid 12 h explosive spread | Sep 27 – Oct 01, 2020 | LANDFIRE 2016 Remap (LF 2.0.0) | Hourly 10m U/V, gust, RH, T (AWS S3) | GOFER (hourly), FEDS (12h), GeoMAC IR | 56,338 ac (22,799 ha) | 110 km NW |
| **6. Caldor** | 2021 | El Dorado, Amador | **Mixed Terrain & Wind**: Cosumnes River canyon run followed by extreme SW wind push over Sierra crest | Aug 14 – Sep 02, 2021 | LANDFIRE 2020 (LF 2.2.0) / 2016 Remap | Hourly 10m U/V, gust, RH, T (AWS S3) | GOFER (hourly), FEDS (12h), GeoMAC / NIROPS IR | 221,835 ac (89,773 ha) | 130 km SE |
| **7. Tamarack** | 2021 | Alpine, Douglas (NV) | **Wind-Driven High Elevation**: Thunderstorm outflow + strong SW winds through sagebrush/timber | Jul 16 – Jul 25, 2021 | LANDFIRE 2020 (LF 2.2.0) / 2016 Remap | Hourly 10m U/V, gust, RH, T (AWS S3) | GOFER (hourly), FEDS (12h), GeoMAC / NIFC IR | 68,637 ac (27,776 ha) | 180 km SE |

---

## 4. Deviations from the Brief

- **None**:
  - Camp Fire has been strictly held out and was not touched for any parameter calibration.
  - Zero ML models were trained in this phase.
  - GOFER and FEDS actual coverage was verified directly from repository metadata and API files before making any claims.
  - Step 3 (Calibration) has been intentionally paused pending orchestrator approval.

---

## 5. Open Problems

1. **Multi-Scale Observation Integration**:
   - GOFER provides hourly progression contours derived from geostationary GOES satellites (~2 km native nadir, ~3 km in CA).
   - FEDS provides 12-hour perimeters derived from polar-orbiting VIIRS (375 m).
   - GeoMAC/NIROPS provides high-resolution aerial infrared polygons (~10–30 m) but at irregular intervals (typically 1–2 dusk/dawn flights per 24 hours).
   - *Design challenge for Step 3:* Loss formulation must blend fine-scale geometric boundary accuracy from VIIRS/NIROPS with coarse hourly front timing from GOFER.
2. **Differentiable CA Formulation**:
   - Standard PyTorchFire cellular automaton updates use hard boolean thresholds (`state > threshold`).
   - For gradient descent in Step 3, we must use a differentiable continuous relaxation (e.g., sigmoid temperature scaling $P = \sigma((E - \theta)/\tau)$) or score function estimation so gradients propagate back to fuel and wind parameters.

---

## 6. Decisions I Need to Make (Orchestrator Approval Required)

1. **Approval of Training Fires Proposal**:
   - Does the orchestrator approve the proposed 7 California fires (Kincade, Bobcat, Glass, Creek, Zogg, Caldor, Tamarack) for calibration and leave-one-fire-out validation in Step 3?
2. **Approval of 60 m Operating Resolution**:
   - Does the orchestrator approve standardizing on **60 m** grid spacing for the calibrated teacher and surrogate tiles ($256 \times 256$ grid = $15.4\text{ km} \times 15.4\text{ km}$ footprint)?
