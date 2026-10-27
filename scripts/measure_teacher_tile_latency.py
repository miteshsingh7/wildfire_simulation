#!/usr/bin/env python3
"""Phase 4B Step 2d: Teacher v2 12-hour simulation latency on a 256x256 tile.

Measures full 12 h Teacher v2 simulation (30 steps/h = 360 steps)
on a 256x256 tile with the exact ember model from Phase 2I.
Averages over 5 independent runs using time.perf_counter().
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
from wf052.surrogate.standin_generator import (
    build_tile_fire_dataset_60m,
    load_and_verify_frozen_teacher_v2,
    load_fire_rasters_crop,
)


def measure_runs(name: str, ds, t_params, e_params, ign_mask, n_runs: int = 5) -> list[float]:
    print(f"\n--- Measuring {name} (12 h, 30 steps/h = 360 steps, {n_runs} runs) ---")
    durations = []
    for run_idx in range(n_runs):
        seed = 42000 + run_idx * 10007
        t0 = time.perf_counter()
        sim_out = simulate_teacher_with_embers_60m(
            ds=ds,
            params=t_params,
            ember_params=e_params,
            use_ember_kernel=True,
            duration_hours=12,
            steps_per_hour=30,
            seed=seed,
            init_burned_mask=ign_mask,
        )
        t_dur = time.perf_counter() - t0
        durations.append(t_dur)
        arr = sim_out["arrival_time_min"]
        n_burned = int(np.sum(np.isfinite(arr) & (arr <= 720.0)))
        print(f"  Run {run_idx+1}: {t_dur:.4f} s ({t_dur*1000.0:.1f} ms) | burned cells: {n_burned}")

    mean_s = float(np.mean(durations))
    std_s = float(np.std(durations))
    mean_ms = mean_s * 1000.0
    std_ms = std_s * 1000.0
    print(f"Summary for {name}:")
    print(f"  Mean latency: {mean_s:.4f} s ({mean_ms:.2f} ms)")
    print(f"  Std latency:  {std_s:.4f} s ({std_ms:.2f} ms)")
    return durations


def main() -> None:
    print("=" * 80)
    print("PHASE 4B STEP 2d: TEACHER V2 LATENCY BENCHMARK (256x256 Tile, 12 h)")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load frozen teacher config
    _, _, _, t_params, e_params = load_and_verify_frozen_teacher_v2("configs/teacher_v2.yaml")

    # Benchmark on Caldor, Kincade, and Creek 256x256 tiles (exact Phase 2I benchmark fires)
    fires = ["caldor", "kincade", "creek"]
    all_summaries = {}

    for fire in fires:
        fbfm, elev, canopy, _ = load_fire_rasters_crop(fire, tile_size=(256, 256))
        ign_mask = np.zeros((256, 256), dtype=bool)
        ign_mask[128, 40] = True
        hourly_wind = np.zeros((12, 2), dtype=np.float32)
        hourly_wind[:, 0] = 8.0  # 8 m/s wind blowing Eastward (Phase 2I condition)
        ds = build_tile_fire_dataset_60m(fire, fbfm, elev, canopy, hourly_wind, device=device)

        durs = measure_runs(f"{fire.capitalize()} 256x256 tile (8 m/s wind)", ds, t_params, e_params, ign_mask, n_runs=5)
        all_summaries[fire] = durs

    print("\n" + "=" * 80)
    print("STEP 2d LATENCY SUMMARY ACROSS ALL 3 FIRES (5 runs each = 15 total)")
    print("=" * 80)
    for fire, durs in all_summaries.items():
        m_s = np.mean(durs)
        std_s = np.std(durs)
        print(f"  {fire.capitalize():<10}: mean = {m_s:.4f} s ({m_s*1000.0:.1f} ms) +/- {std_s*1000.0:.1f} ms")

    all_durs = [d for durs in all_summaries.values() for d in durs]
    grand_mean_s = np.mean(all_durs)
    grand_std_s = np.std(all_durs)
    print("-" * 80)
    print(f"  Grand Mean (15 runs): {grand_mean_s:.4f} s ({grand_mean_s*1000.0:.1f} ms) +/- {grand_std_s*1000.0:.1f} ms")
    print("=" * 80)
    print("\nCLARIFICATION OF GRANULARITY:")
    print("1. Surrogate single forward pass: ~24.1 ms per 256x256 tile for full 12-hour horizon (1 shot).")
    print(f"2. Teacher v2 full 12-hour simulation: {np.mean(all_summaries['caldor']):.3f} s ({np.mean(all_summaries['caldor'])*1000.0:.1f} ms) on Caldor 256x256 tile.")
    print("3. Why was 0.8 ms reported in Phase 4A? '0.8 ms' was a confusion of single-simulation step time")
    print(f"   (1.46 s / 360 steps = ~4.0 ms per step; or K=1 prescreen ~0.8 s misstated as ms).")
    print(f"4. Actual full-simulation speedup of Surrogate vs Teacher v2: {grand_mean_s*1000.0 / 24.1:.1f}x.")


if __name__ == "__main__":
    main()

