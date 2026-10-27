"""Wind check script for Phase 2H Step 1.

Investigates:
1a. Wind direction response: 8 m/s toward East, North, West, South, Northeast for caldor, kincade, creek (K=8).
    Reports burned-centroid displacement in (row, col) cells and compass direction.
    Cites tile transform / georeferencing.
1b. Spread rate not clamped by tile: constant wind toward East at 2, 4, 6, 8, 10, 14, 18, 20 m/s.
    Fixed ignition cells per fire. Downwind band within 1 km of ignition line.
    Reports: fraction reaching 1, 2, 3 km; median arrival time (min) and n reaching; fraction touching edge.
    Summary table and paired table per ignition.
1c. Mechanism check: call WildfireModel (underlying PyTorchFire engine) for fixed timber fuel code (TL5)
    on flat terrain at 2, 8, 20 m/s for downwind, crosswind, upwind neighbours.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from pytorchfire import WildfireModel

from wf052.sim.ember_kernel import EmberParams, simulate_teacher_with_embers_60m
from wf052.surrogate.dataset import fbfm40_to_fuel_group
from wf052.surrogate.standin_generator import (
    build_tile_fire_dataset_60m,
    load_and_verify_frozen_teacher_v2,
    load_fire_rasters_crop,
)


def find_fixed_upwind_ignition(fire_name: str, target_col: int = 40) -> tuple[int, int, int]:
    """Find a burnable timber/flammable cell near target_col to leave downwind room."""
    fbfm, _, _, _ = load_fire_rasters_crop(fire_name, tile_size=(256, 256))
    fg = fbfm40_to_fuel_group(fbfm)
    # Search around row 128, col target_col
    for offset in range(30):
        for dr in [0, offset, -offset]:
            r = int(np.clip(128 + dr, 20, 235))
            c = int(np.clip(target_col, 20, 235))
            # Prefer timber groups (4, 5) or flammable groups (1, 2, 3)
            if fg[r, c] in (4, 5, 3, 2, 1):
                return r, c, int(fbfm[r, c])
    # Fallback to any non-zero
    for r in range(100, 160):
        for c in range(target_col - 10, target_col + 20):
            if fg[r, c] > 0:
                return r, c, int(fbfm[r, c])
    return 128, target_col, int(fbfm[128, target_col])


def run_step_1a(fires: list[str], t_params: Any, e_params: Any, device: torch.device) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 1a: Direction Check (8 m/s, K=8 seeds)")
    print("=" * 70)

    directions = [
        ("East", (8.0, 0.0)),
        ("North", (0.0, 8.0)),
        ("West", (-8.0, 0.0)),
        ("South", (0.0, -8.0)),
        ("Northeast", (8.0 / math.sqrt(2.0), 8.0 / math.sqrt(2.0))),
    ]

    k_seeds = 8
    duration_hours = 4  # 4 hours is plenty to measure distinct centroid drift without excessive edge boundary
    records = []

    for fire in fires:
        fbfm_arr, elev_arr, canopy_arr, _ = load_fire_rasters_crop(fire, tile_size=(256, 256))
        # Choose a central burnable ignition cell so fire can spread in all 4 directions
        fg = fbfm40_to_fuel_group(fbfm_arr)
        ign_r, ign_c = 128, 128
        while fg[ign_r, ign_c] == 0:
            ign_c += 1

        ign_mask = np.zeros((256, 256), dtype=bool)
        ign_mask[ign_r, ign_c] = True

        for dir_name, (u_val, v_val) in directions:
            hourly_wind = np.zeros((12, 2), dtype=np.float32)
            hourly_wind[:, 0] = u_val
            hourly_wind[:, 1] = v_val

            ds = build_tile_fire_dataset_60m(
                fire_name=fire,
                fbfm_arr=fbfm_arr,
                elev_arr=elev_arr,
                canopy_arr=canopy_arr,
                hourly_wind_uv=hourly_wind,
                device=device,
            )

            d_rows, d_cols = [], []
            burned_counts = []

            for k in range(k_seeds):
                seed_k = 42000 + k * 10007
                sim_out = simulate_teacher_with_embers_60m(
                    ds=ds,
                    params=t_params,
                    ember_params=e_params,
                    use_ember_kernel=True,
                    duration_hours=duration_hours,
                    steps_per_hour=30,
                    seed=seed_k,
                    init_burned_mask=ign_mask,
                )
                arr = sim_out["arrival_time_min"]
                burned = np.isfinite(arr) & (arr <= duration_hours * 60.0)
                n_b = int(np.sum(burned))
                burned_counts.append(n_b)

                if n_b > 0:
                    b_rows, b_cols = np.where(burned)
                    mean_r = float(np.mean(b_rows))
                    mean_c = float(np.mean(b_cols))
                    d_rows.append(mean_r - ign_r)
                    d_cols.append(mean_c - ign_c)
                else:
                    d_rows.append(0.0)
                    d_cols.append(0.0)

            mean_dr = float(np.mean(d_rows))
            mean_dc = float(np.mean(d_cols))
            mean_nb = float(np.mean(burned_counts))

            # Determine empirical compass direction
            # dr < 0 is North (row 0 is North), dr > 0 is South
            # dc > 0 is East, dc < 0 is West
            ns_str = "North" if mean_dr < -1.0 else ("South" if mean_dr > 1.0 else "")
            ew_str = "East" if mean_dc > 1.0 else ("West" if mean_dc < -1.0 else "")
            empirical_dir = f"{ns_str}{ew_str}" if (ns_str or ew_str) else "None"

            records.append({
                "fire": fire,
                "target_direction": dir_name,
                "u_ms": round(u_val, 2),
                "v_ms": round(v_val, 2),
                "ign_r": ign_r,
                "ign_c": ign_c,
                "mean_delta_row": round(mean_dr, 2),
                "mean_delta_col": round(mean_dc, 2),
                "mean_burned_cells": round(mean_nb, 1),
                "empirical_compass": empirical_dir,
            })

    df_1a = pd.DataFrame(records)
    print(df_1a.to_string(index=False))
    return df_1a


def run_step_1b(fires: list[str], t_params: Any, e_params: Any, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("STEP 1b: Spread Rate Not Clamped by Tile (Wind East 2..20 m/s, K=8, 12 h)")
    print("=" * 70)

    speeds = [2, 4, 6, 8, 10, 14, 18, 20]
    k_seeds = 8
    horizon_min = 720.0  # 12 hours

    # 1. Pick fixed ignition cells per fire at target_col=40 (leaving ~13 km downwind room)
    fire_ignitions = {}
    for f in fires:
        r, c, code = find_fixed_upwind_ignition(f, target_col=40)
        fire_ignitions[f] = (r, c, code)
        print(f"Fixed ignition for {f}: row={r}, col={c}, FBFM40 code={code} (downwind room: {(255 - c) * 60 / 1000:.1f} km)")

    detailed_records = []

    for fire in fires:
        ign_r, ign_c, f_code = fire_ignitions[fire]
        fbfm_arr, elev_arr, canopy_arr, _ = load_fire_rasters_crop(fire, tile_size=(256, 256))

        ign_mask = np.zeros((256, 256), dtype=bool)
        ign_mask[ign_r, ign_c] = True

        for ws in speeds:
            hourly_wind = np.zeros((12, 2), dtype=np.float32)
            hourly_wind[:, 0] = float(ws)
            hourly_wind[:, 1] = 0.0

            ds = build_tile_fire_dataset_60m(
                fire_name=fire,
                fbfm_arr=fbfm_arr,
                elev_arr=elev_arr,
                canopy_arr=canopy_arr,
                hourly_wind_uv=hourly_wind,
                device=device,
            )

            # Band definition: cells within 1 km (16 cells) of row ign_r, with col >= ign_c
            band_mask = np.zeros((256, 256), dtype=bool)
            r_min = max(0, ign_r - 16)
            r_max = min(256, ign_r + 17)
            band_mask[r_min:r_max, ign_c:] = True

            # Threshold columns for 1 km (16.67 cells), 2 km (33.33 cells), 3 km (50.00 cells)
            col_1km = ign_c + int(np.ceil(1000.0 / 60.0))
            col_2km = ign_c + int(np.ceil(2000.0 / 60.0))
            col_3km = ign_c + int(np.ceil(3000.0 / 60.0))

            runs_reached_1km = 0
            runs_reached_2km = 0
            runs_reached_3km = 0
            arrival_1km_list = []
            arrival_2km_list = []
            arrival_3km_list = []
            runs_touch_edge = 0

            for k in range(k_seeds):
                seed_k = 42000 + k * 10007
                sim_out = simulate_teacher_with_embers_60m(
                    ds=ds,
                    params=t_params,
                    ember_params=e_params,
                    use_ember_kernel=True,
                    duration_hours=12,
                    steps_per_hour=30,
                    seed=seed_k,
                    init_burned_mask=ign_mask,
                )
                arr = sim_out["arrival_time_min"]
                burned = np.isfinite(arr) & (arr <= horizon_min)

                # Check edge touch (row 0, 255, col 0, 255)
                edge_burned = (
                    np.any(burned[0, :])
                    or np.any(burned[-1, :])
                    or np.any(burned[:, 0])
                    or np.any(burned[:, -1])
                )
                if edge_burned:
                    runs_touch_edge += 1

                # 1 km check in band
                b_1km = burned & band_mask & (np.arange(256)[None, :] >= col_1km)
                if np.any(b_1km):
                    runs_reached_1km += 1
                    arrival_1km_list.append(float(np.min(arr[b_1km])))

                # 2 km check in band
                b_2km = burned & band_mask & (np.arange(256)[None, :] >= col_2km)
                if np.any(b_2km):
                    runs_reached_2km += 1
                    arrival_2km_list.append(float(np.min(arr[b_2km])))

                # 3 km check in band
                b_3km = burned & band_mask & (np.arange(256)[None, :] >= col_3km)
                if np.any(b_3km):
                    runs_reached_3km += 1
                    arrival_3km_list.append(float(np.min(arr[b_3km])))

            med_arr_1km = float(np.median(arrival_1km_list)) if arrival_1km_list else float("nan")
            med_arr_2km = float(np.median(arrival_2km_list)) if arrival_2km_list else float("nan")
            med_arr_3km = float(np.median(arrival_3km_list)) if arrival_3km_list else float("nan")

            detailed_records.append({
                "fire": fire,
                "ign_r": ign_r,
                "ign_c": ign_c,
                "fbfm40": f_code,
                "wind_speed_ms": ws,
                "k_seeds": k_seeds,
                "frac_reach_1km": round(runs_reached_1km / k_seeds, 4),
                "med_arr_1km_min": round(med_arr_1km, 1) if not math.isnan(med_arr_1km) else "N/A",
                "n_1km": runs_reached_1km,
                "frac_reach_2km": round(runs_reached_2km / k_seeds, 4),
                "med_arr_2km_min": round(med_arr_2km, 1) if not math.isnan(med_arr_2km) else "N/A",
                "n_2km": runs_reached_2km,
                "frac_reach_3km": round(runs_reached_3km / k_seeds, 4),
                "med_arr_3km_min": round(med_arr_3km, 1) if not math.isnan(med_arr_3km) else "N/A",
                "n_3km": runs_reached_3km,
                "frac_touch_edge": round(runs_touch_edge / k_seeds, 4),
            })

    df_paired = pd.DataFrame(detailed_records)

    # Summary table averaged across the 3 ignitions
    summary_records = []
    for ws in speeds:
        sub = df_paired[df_paired["wind_speed_ms"] == ws]
        tot_runs = len(sub) * k_seeds
        tot_1km = sub["n_1km"].sum()
        tot_2km = sub["n_2km"].sum()
        tot_3km = sub["n_3km"].sum()

        # Medians of all reaching arrival times across ignitions
        # We can extract the mean of median arrival times or compute directly
        valid_1km = [x for x in sub["med_arr_1km_min"] if x != "N/A"]
        valid_2km = [x for x in sub["med_arr_2km_min"] if x != "N/A"]
        valid_3km = [x for x in sub["med_arr_3km_min"] if x != "N/A"]

        summary_records.append({
            "wind_speed_ms": ws,
            "total_runs": tot_runs,
            "frac_reach_1km": round(tot_1km / tot_runs, 4),
            "med_arr_1km_min": round(float(np.mean(valid_1km)), 1) if valid_1km else "N/A",
            "n_1km": int(tot_1km),
            "frac_reach_2km": round(tot_2km / tot_runs, 4),
            "med_arr_2km_min": round(float(np.mean(valid_2km)), 1) if valid_2km else "N/A",
            "n_2km": int(tot_2km),
            "frac_reach_3km": round(tot_3km / tot_runs, 4),
            "med_arr_3km_min": round(float(np.mean(valid_3km)), 1) if valid_3km else "N/A",
            "n_3km": int(tot_3km),
            "frac_touch_edge": round(sub["frac_touch_edge"].mean(), 4),
        })

    df_summary = pd.DataFrame(summary_records)
    print("\n--- Summary Table Averaged Across Ignitions ---")
    print(df_summary.to_string(index=False))
    print("\n--- Paired Table Per Ignition ---")
    print(df_paired.to_string(index=False))

    return df_summary, df_paired


def run_step_1c(t_params: Any) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 1c: Mechanism Check (Direct Call to Simulator's Contact Code)")
    print("=" * 70)

    # Calling WildfireModel (the underlying simulator contact model in pytorchfire) directly
    # Fixed timber fuel code: TL5 (FBFM40 185) -> p_veg = 0.30
    h, w = 3, 3
    ign = torch.zeros((h, w), dtype=torch.bool)
    ign[1, 1] = True  # Center cell burning

    p_veg_timber = 0.30
    p_veg = torch.full((h, w), p_veg_timber, dtype=torch.float32)
    p_den = torch.zeros((h, w), dtype=torch.float32)
    slope = torch.zeros((h, w, 3, 3), dtype=torch.float32)

    records = []
    for ws in [2.0, 8.0, 20.0]:
        env_data = {
            "p_veg": p_veg,
            "p_den": p_den,
            "wind_velocity": torch.full((h, w), ws, dtype=torch.float32),
            "wind_towards_direction": torch.zeros((h, w), dtype=torch.float32),  # 0 deg = East
            "slope": slope,
            "initial_ignition": ign,
        }
        params = {
            "a": torch.tensor(t_params.a),
            "p_h": torch.tensor(t_params.p_base),
            "c_1": torch.tensor(t_params.c_1),
            "c_2": torch.tensor(t_params.c_2),
            "p_continue": torch.tensor(t_params.p_continue),
        }
        model = WildfireModel(env_data=env_data, params=params)
        p_ign = model.p_ignite()

        downwind = float(p_ign[1, 2].item())  # East
        crosswind = float(p_ign[0, 1].item())  # North
        upwind = float(p_ign[1, 0].item())  # West

        records.append({
            "wind_speed_ms": ws,
            "fuel_code": "TL5 (185, p_veg=0.30)",
            "terrain": "Flat (slope=0)",
            "downwind_p_ignite": round(downwind, 6),
            "crosswind_p_ignite": round(crosswind, 6),
            "upwind_p_ignite": round(upwind, 6),
            "ratio_downwind_to_crosswind": round(downwind / max(1e-6, crosswind), 2),
        })

    df_1c = pd.DataFrame(records)
    print(df_1c.to_string(index=False))
    return df_1c


def main() -> None:
    _, _, _, t_params, e_params = load_and_verify_frozen_teacher_v2()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Running on device: {device}")

    fires = ["caldor", "kincade", "creek"]

    df_1a = run_step_1a(fires, t_params, e_params, device)
    df_1b_sum, df_1b_paired = run_step_1b(fires, t_params, e_params, device)
    df_1c = run_step_1c(t_params)

    Path("reports").mkdir(exist_ok=True)
    df_1a.to_csv("reports/step1a_direction_check.csv", index=False)
    df_1b_sum.to_csv("reports/step1b_spread_rate_summary.csv", index=False)
    df_1b_paired.to_csv("reports/step1b_spread_rate_paired.csv", index=False)
    df_1c.to_csv("reports/step1c_mechanism_check.csv", index=False)
    print("\nSaved all Step 1 reports to reports/step1*.csv")


if __name__ == "__main__":
    main()
