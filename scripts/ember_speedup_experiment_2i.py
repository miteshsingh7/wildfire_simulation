"""Phase 2I Step 3: Are high-wind arrivals driven by embers?

Runs:
1. Re-runs Step 1b design (same ignitions at row 128, col 40; K=8, 12 h, wind East)
   at speeds 2, 8, 14, 20 m/s with Embers ON vs Embers OFF across caldor, kincade, creek.
   Reports per fire separately:
   - runs reaching 3 km (count & frac)
   - median arrival at 3 km with n
   - burned cells
   - crosswind width (m)
2. Creek East and Northeast rows missing from 1a (K=8, 4 h, 8 m/s).
3. Verifies whether embers-OFF arrivals respect the 100-min contact spread lower bound
   (50 cells * 2.0 min/step = 100.0 min).

Saves:
- reports/step3_ember_speedup_experiment.csv
- reports/step3_creek_direction_check.csv
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from wf052.sim.ember_kernel import EmberParams, simulate_teacher_with_embers_60m
from wf052.sim.teacher_calibrator import TeacherParams
from wf052.surrogate.dataset import fbfm40_to_fuel_group
from wf052.surrogate.standin_generator import (
    build_tile_fire_dataset_60m,
    load_and_verify_frozen_teacher_v2,
    load_fire_rasters_crop,
)


def run_experiment() -> None:
    print("=" * 70)
    print("PHASE 2I STEP 3: EMBER SPEED-UP EXPERIMENT (ON vs OFF)")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load frozen teacher parameters
    cfg_on, _, _, t_params, e_params_on = load_and_verify_frozen_teacher_v2("configs/teacher_v2.yaml")

    # Embers OFF parameter object
    e_params_off = EmberParams(
        alpha_launch=0.0,
        beta_dist_m_per_ms=e_params_on.beta_dist_m_per_ms,
        p_urban_wui=0.0,
    )

    fires = ["caldor", "kincade", "creek"]
    speeds = [2.0, 8.0, 14.0, 20.0]
    k_seeds = 8
    duration_hours = 12

    # Fixed ignitions from Step 1b: row 128, col 40 (col 40 + 50 = col 90 is 3 km downwind)
    target_col_3km = 40 + 50  # 90
    band_r_min = 128 - 16
    band_r_max = 128 + 16

    results_3km = []

    for fire in fires:
        fbfm_arr, elev_arr, canopy_arr, _ = load_fire_rasters_crop(fire, tile_size=(256, 256))
        ign_r, ign_c = 128, 40
        ign_mask = np.zeros((256, 256), dtype=bool)
        ign_mask[ign_r, ign_c] = True

        for spd in speeds:
            hourly_wind = np.zeros((12, 2), dtype=np.float32)
            hourly_wind[:, 0] = spd  # Wind blowing Eastward
            hourly_wind[:, 1] = 0.0

            ds = build_tile_fire_dataset_60m(
                fire_name=fire,
                fbfm_arr=fbfm_arr,
                elev_arr=elev_arr,
                canopy_arr=canopy_arr,
                hourly_wind_uv=hourly_wind,
                device=device,
            )

            for arm_name, use_emb, ep in [
                ("embers_on", True, e_params_on),
                ("embers_off", False, e_params_off),
            ]:
                arr_times_3km = []
                burned_cell_counts = []
                crosswind_widths = []

                for k in range(k_seeds):
                    seed_k = 42000 + k * 10007
                    sim_out = simulate_teacher_with_embers_60m(
                        ds=ds,
                        params=t_params,
                        ember_params=ep,
                        use_ember_kernel=use_emb,
                        duration_hours=duration_hours,
                        steps_per_hour=30,
                        seed=seed_k,
                        init_burned_mask=ign_mask,
                    )
                    arr = sim_out["arrival_time_min"]
                    burned_mask = np.isfinite(arr) & (arr <= duration_hours * 60.0)
                    n_burned = int(np.sum(burned_mask))
                    burned_cell_counts.append(n_burned)

                    if n_burned > 0:
                        b_rows, b_cols = np.where(burned_mask)
                        width_m = float((np.max(b_rows) - np.min(b_rows) + 1) * 60.0)
                        crosswind_widths.append(width_m)
                    else:
                        crosswind_widths.append(0.0)

                    # Arrival at 3 km downwind in downwind band
                    band_slice = arr[band_r_min : band_r_max + 1, target_col_3km:]
                    valid_arrivals = band_slice[np.isfinite(band_slice) & (band_slice <= duration_hours * 60.0)]
                    if len(valid_arrivals) > 0:
                        arr_times_3km.append(float(np.min(valid_arrivals)))

                n_reach = len(arr_times_3km)
                frac_reach = n_reach / k_seeds
                med_arr = float(np.median(arr_times_3km)) if n_reach > 0 else float("inf")
                mean_burned = float(np.mean(burned_cell_counts))
                mean_width = float(np.mean(crosswind_widths))

                print(
                    f"Fire: {fire:8s} | Spd: {spd:4.1f} m/s | Arm: {arm_name:10s} | "
                    f"Reach 3km: {n_reach}/{k_seeds} ({frac_reach:.2f}) | "
                    f"Med Arr 3km: {med_arr:6.1f} min | Burned: {mean_burned:6.1f} | Width: {mean_width:6.1f} m"
                )

                results_3km.append({
                    "fire": fire,
                    "wind_speed_ms": spd,
                    "arm": arm_name,
                    "use_ember_kernel": use_emb,
                    "runs_reaching_3km": n_reach,
                    "total_runs": k_seeds,
                    "frac_reaching_3km": round(frac_reach, 4),
                    "median_arrival_3km_min": round(med_arr, 2) if np.isfinite(med_arr) else "inf",
                    "n_3km": n_reach,
                    "mean_burned_cells": round(mean_burned, 1),
                    "mean_crosswind_width_m": round(mean_width, 1),
                })

    df_3km = pd.DataFrame(results_3km)
    df_3km.to_csv("reports/step3_ember_speedup_experiment.csv", index=False)
    print("\nSaved reports/step3_ember_speedup_experiment.csv")

    # Step 3b: Creek East and Northeast rows from 1a
    print("\n" + "=" * 70)
    print("STEP 3: CREEK DIRECTION CHECK (EAST & NORTHEAST)")
    print("=" * 70)

    fbfm_creek, elev_creek, canopy_creek, _ = load_fire_rasters_crop("creek", tile_size=(256, 256))
    creek_records = []

    # Test both central ignition (128, 128) and upwind ignition (128, 40)
    test_ignitions = [
        ("Creek_Center_128_128", 128, 128),
        ("Creek_Upwind_128_40", 128, 40),
    ]

    dir_tests = [
        ("East", 8.0, 0.0),
        ("Northeast", 8.0 / math.sqrt(2.0), 8.0 / math.sqrt(2.0)),
    ]

    for ign_name, ign_r, ign_c in test_ignitions:
        ign_mask = np.zeros((256, 256), dtype=bool)
        ign_mask[ign_r, ign_c] = True

        for dir_name, u_val, v_val in dir_tests:
            hourly_wind = np.zeros((12, 2), dtype=np.float32)
            hourly_wind[:, 0] = u_val
            hourly_wind[:, 1] = v_val

            ds = build_tile_fire_dataset_60m(
                fire_name="creek",
                fbfm_arr=fbfm_creek,
                elev_arr=elev_creek,
                canopy_arr=canopy_creek,
                hourly_wind_uv=hourly_wind,
                device=device,
            )

            d_rows, d_cols, burned_counts = [], [], []
            for k in range(k_seeds):
                seed_k = 42000 + k * 10007
                sim_out = simulate_teacher_with_embers_60m(
                    ds=ds,
                    params=t_params,
                    ember_params=e_params_on,
                    use_ember_kernel=True,
                    duration_hours=4,
                    steps_per_hour=30,
                    seed=seed_k,
                    init_burned_mask=ign_mask,
                )
                arr = sim_out["arrival_time_min"]
                b_mask = np.isfinite(arr) & (arr <= 240.0)
                nb = int(np.sum(b_mask))
                burned_counts.append(nb)
                if nb > 0:
                    br, bc = np.where(b_mask)
                    d_rows.append(float(np.mean(br)) - ign_r)
                    d_cols.append(float(np.mean(bc)) - ign_c)
                else:
                    d_rows.append(0.0)
                    d_cols.append(0.0)

            mean_dr = float(np.mean(d_rows))
            mean_dc = float(np.mean(d_cols))
            mean_nb = float(np.mean(burned_counts))

            # Empirical compass
            ns_str = "North" if mean_dr < -1.0 else ("South" if mean_dr > 1.0 else "")
            ew_str = "East" if mean_dc > 1.0 else ("West" if mean_dc < -1.0 else "")
            compass = ns_str + ew_str if (ns_str or ew_str) else "None"

            print(
                f"{ign_name:22s} | Dir: {dir_name:9s} | (u,v)=({u_val:.2f}, {v_val:.2f}) | "
                f"dRow: {mean_dr:+6.2f} | dCol: {mean_dc:+6.2f} | Burned: {mean_nb:6.1f} | Compass: {compass}"
            )

            creek_records.append({
                "ignition_setup": ign_name,
                "target_direction": dir_name,
                "u_ms": round(u_val, 2),
                "v_ms": round(v_val, 2),
                "ign_r": ign_r,
                "ign_c": ign_c,
                "mean_delta_row": round(mean_dr, 2),
                "mean_delta_col": round(mean_dc, 2),
                "mean_burned_cells": round(mean_nb, 1),
                "empirical_compass": compass,
            })

    df_creek = pd.DataFrame(creek_records)
    df_creek.to_csv("reports/step3_creek_direction_check.csv", index=False)
    print("\nSaved reports/step3_creek_direction_check.csv")


if __name__ == "__main__":
    run_experiment()
