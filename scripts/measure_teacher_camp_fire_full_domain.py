#!/usr/bin/env python3
"""Phase 4C Step 4c: Teacher v2 12-hour simulation on the full Camp Fire domain (564 x 562).

Measures 5 independent runs of Teacher v2 forward simulation (12 h, 30 steps/h = 360 steps)
on the complete 564 x 562 domain using time.perf_counter().
Lists each run individually with wall time in seconds and milliseconds, and burned cell count.
"""

import time
import numpy as np
import torch

from evaluate_heldout_camp_fire import (
    TeacherParams,
    build_camp_fire_dataset_60m,
)
from wf052.sim.ember_kernel import (
    EmberParams,
    simulate_teacher_with_embers_60m,
)
from wf052.surrogate.standin_generator import load_and_verify_frozen_teacher_v2


def main() -> None:
    print("=" * 80)
    print("PHASE 4C STEP 4c: TEACHER V2 FULL-DOMAIN BENCHMARK (Camp Fire 564x562, 12 h)")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load frozen teacher parameters
    _, _, _, t_params, e_params = load_and_verify_frozen_teacher_v2("configs/teacher_v2.yaml")

    # Load Camp Fire full domain dataset
    print("Loading Camp Fire 60m full domain dataset...")
    ds_camp, bbox, transform = build_camp_fire_dataset_60m(device)
    h, w = ds_camp.fbfm_arr.shape
    print(f"Domain Grid Shape: {h} x {w} cells (resolution = {ds_camp.res_m} m)")

    test_seeds = [389, 117, 304, 100, 236]
    durations = []
    run_records = []

    print(f"\nRunning 5 independent 12-hour Teacher v2 simulations (360 steps each)...")
    for idx, s in enumerate(test_seeds):
        t0 = time.perf_counter()
        sim_out = simulate_teacher_with_embers_60m(
            ds=ds_camp,
            params=t_params,
            ember_params=e_params,
            use_ember_kernel=True,
            duration_hours=12,
            steps_per_hour=30,
            seed=s,
        )
        t_dur = time.perf_counter() - t0
        durations.append(t_dur)

        arr = sim_out["arrival_time_min"]
        burned_cells = int(np.sum(np.isfinite(arr) & (arr <= 720.0)))
        area_ha = burned_cells * (60.0 * 60.0) / 10000.0

        rec = {
            "run_index": idx + 1,
            "seed": s,
            "wall_sec": round(t_dur, 4),
            "wall_ms": round(t_dur * 1000.0, 2),
            "burned_cells": burned_cells,
            "burned_area_ha": round(area_ha, 1),
        }
        run_records.append(rec)
        print(f"  Run {idx+1} (Seed {s}): {t_dur:.4f} s ({t_dur*1000.0:.1f} ms) | {burned_cells} cells ({area_ha:.1f} ha)")

    mean_s = float(np.mean(durations))
    std_s = float(np.std(durations))
    mean_ms = mean_s * 1000.0
    std_ms = std_s * 1000.0

    print("\n" + "=" * 80)
    print("TEACHER V2 FULL-DOMAIN LATENCY SUMMARY")
    print("=" * 80)
    print(f"Domain Shape:      {h} x {w} cells (Camp Fire full extent)")
    print(f"Simulation Window: 12 hours (360 simulation steps at 2.0 min/step)")
    print(f"Runs:              5 independent runs (seeds 389, 117, 304, 100, 236)")
    print(f"Mean Wall Time:    {mean_s:.4f} s ({mean_ms:.2f} ms)")
    print(f"Std Wall Time:     {std_s:.4f} s ({std_ms:.2f} ms)")
    print(f"Min Wall Time:     {np.min(durations):.4f} s ({np.min(durations)*1000.0:.1f} ms)")
    print(f"Max Wall Time:     {np.max(durations):.4f} s ({np.max(durations)*1000.0:.1f} ms)")
    print("=" * 80)


if __name__ == "__main__":
    main()
