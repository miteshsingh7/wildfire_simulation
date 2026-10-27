"""Phase 2G Step 1: Teacher wind response investigation and controlled experiment.

Evaluates:
1. Wind conversion pipeline and direction convention check.
2. Controlled experiment: 3 fixed triples (caldor, kincade, creek) across speeds 2, 4, 6, 8, 10, 14, 18, 20 m/s with K=8 seeds each.
3. Metrics per speed:
   - mean burned fraction
   - mean burned cells
   - maximum downwind extent (km)
   - maximum crosswind width (km)
   - fraction of runs touching the tile edge
   - mean ember ignitions
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from wf052.surrogate.dataset import fbfm40_to_fuel_group
from wf052.surrogate.standin_generator import (
    build_tile_fire_dataset_60m,
    load_and_verify_frozen_teacher_v2,
    load_fire_rasters_crop,
    simulate_teacher_with_embers_60m,
)


def run_wind_response_experiment():
    print("=" * 80)
    print("PHASE 2G STEP 1: TEACHER WIND RESPONSE EXPERIMENT")
    print("=" * 80)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device} ({device.type.upper()})")

    cfg_path = Path("configs/teacher_v2.yaml")
    cfg, cfg_hash, sim_hash, t_params, e_params = load_and_verify_frozen_teacher_v2(cfg_path)
    print(f"Teacher v2 Config Hash: {cfg_hash}")
    print(f"Canonical Sim Hash:     {sim_hash}")

    triples = [
        {"fire": "caldor", "cr": 120, "cc": 120, "ir": 128, "ic": 128},
        {"fire": "kincade", "cr": 120, "cc": 120, "ir": 128, "ic": 128},
        {"fire": "creek", "cr": 150, "cc": 150, "ir": 128, "ic": 128},
    ]

    speeds = [2, 4, 6, 8, 10, 14, 18, 20]
    k_seeds = 8
    duration_hours = 12
    res_m = 60.0

    # Verify ignition fuel groups
    print("\nTriple verification:")
    cached_triples = []
    for tr in triples:
        fire = tr["fire"]
        cr, cc = tr["cr"], tr["cc"]
        ir, ic = tr["ir"], tr["ic"]
        fbfm, elev, canopy, _ = load_fire_rasters_crop(
            fire, tile_size=(256, 256), crop_center_row=cr, crop_center_col=cc
        )
        fg = fbfm40_to_fuel_group(fbfm)
        grp = int(fg[ir, ic])
        code = int(fbfm[ir, ic])
        print(f"  {fire:7s}: crop=({cr}, {cc}), ign=({ir}, {ic}) -> FBFM40={code}, fuel_group={grp} (burnable: {1 <= grp <= 5})")
        assert 1 <= grp <= 5, f"Ignition cell for {fire} must be in fuel groups 1-5, got {grp}"
        cached_triples.append({
            "fire": fire,
            "fbfm": fbfm,
            "elev": elev,
            "canopy": canopy,
            "ir": ir,
            "ic": ic,
        })

    # Constant wind blowing toward East (azimuth 90 deg, angle theta = 0 rad, u = speed, v = 0)
    # In this coordinate frame:
    # Downwind axis is East (+c, dx = (c - ic)*res_m)
    # Crosswind axis is North/South (dy = -(r - ir)*res_m)
    results = []

    for spd in speeds:
        speed_float = float(spd)
        wind_uv = np.full((12, 2), [speed_float, 0.0], dtype=np.float32)

        runs_for_speed = []

        for tr in cached_triples:
            fire = tr["fire"]
            ir, ic = tr["ir"], tr["ic"]

            ign_mask = np.zeros((256, 256), dtype=np.float32)
            ign_mask[max(0, ir - 1) : min(256, ir + 2), max(0, ic - 1) : min(256, ic + 2)] = 1.0

            ds = build_tile_fire_dataset_60m(
                fire_name=fire,
                fbfm_arr=tr["fbfm"],
                elev_arr=tr["elev"],
                canopy_arr=tr["canopy"],
                hourly_wind_uv=wind_uv,
                device=device,
                res_m=res_m,
            )

            for k in range(k_seeds):
                # Documented seed formula
                seed_k = int((42000 + k * 10007) % 2_147_483_647)

                sim_out = simulate_teacher_with_embers_60m(
                    ds=ds,
                    params=t_params,
                    ember_params=e_params,
                    use_ember_kernel=True,
                    duration_hours=duration_hours,
                    steps_per_hour=30,
                    seed=seed_k,
                    init_burned_mask=ign_mask.astype(bool),
                )

                arr_time = sim_out["arrival_time_min"]
                burned = ~np.isnan(arr_time)
                n_burned = int(np.sum(burned))
                bf = float(n_burned / (256 * 256))
                embers = int(sim_out["total_ember_ignitions"])

                # Check if touched edge
                touched = bool(
                    np.any(burned[0, :])
                    or np.any(burned[-1, :])
                    or np.any(burned[:, 0])
                    or np.any(burned[:, -1])
                )

                # Downwind and crosswind extents
                # Wind is blowing East (theta = 0)
                # Downwind: dx = (c - ic) * 60.0 m
                # Crosswind: dy = -(r - ir) * 60.0 m
                if n_burned > 0:
                    coords = np.argwhere(burned)
                    rs, cs = coords[:, 0], coords[:, 1]
                    dx_m = (cs - ic) * res_m
                    dy_m = -(rs - ir) * res_m

                    max_downwind_km = float(np.max(dx_m)) / 1000.0
                    crosswind_width_km = float(np.max(dy_m) - np.min(dy_m)) / 1000.0
                else:
                    max_downwind_km = 0.0
                    crosswind_width_km = 0.0

                runs_for_speed.append({
                    "speed": spd,
                    "fire": fire,
                    "seed": seed_k,
                    "burned_fraction": bf,
                    "burned_cells": n_burned,
                    "max_downwind_km": max_downwind_km,
                    "crosswind_width_km": crosswind_width_km,
                    "touched_edge": touched,
                    "ember_ignitions": embers,
                })

        df_spd = pd.DataFrame(runs_for_speed)
        mean_bf = float(df_spd["burned_fraction"].mean())
        mean_cells = float(df_spd["burned_cells"].mean())
        max_downwind = float(df_spd["max_downwind_km"].max())
        max_crosswind = float(df_spd["crosswind_width_km"].max())
        edge_frac = float(df_spd["touched_edge"].mean())
        mean_embers = float(df_spd["ember_ignitions"].mean())

        results.append({
            "speed_ms": spd,
            "mean_burned_fraction": mean_bf,
            "mean_burned_cells": mean_cells,
            "max_downwind_extent_km": max_downwind,
            "max_crosswind_width_km": max_crosswind,
            "fraction_touching_edge": edge_frac,
            "mean_ember_ignitions": mean_embers,
        })
        print(f"Speed {spd:2d} m/s done: mean_bf={mean_bf:.4f}, mean_cells={mean_cells:6.1f}, downwind={max_downwind:.2f} km, crosswind={max_crosswind:.2f} km, edge_frac={edge_frac:.2f}, embers={mean_embers:.1f}")

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("STEP 1b CONTROLLED EXPERIMENT SUMMARY (Across 3 Fires x 8 Seeds = 24 Runs per Speed)")
    print("=" * 90)
    print(df_res.to_string(index=False))

    df_res.to_csv("reports/step1_wind_response_summary.csv", index=False)
    print("\nSaved summary to reports/step1_wind_response_summary.csv")


if __name__ == "__main__":
    run_wind_response_experiment()
