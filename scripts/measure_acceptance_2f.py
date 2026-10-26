"""Phase 2F Step 1: Teacher Burned-Fraction Distribution & Acceptance Rate Measurement.

Runs:
a) 240 candidates from the production scenario sampler (K=1, 12 h, fixed seeds).
   Saves reports/acceptance_2f_candidates.csv with:
   fire_id, crop origin, ignition cell, FBFM40 code, fuel group at ignition,
   wind peak, wind mean, burned fraction, wall seconds.
b) Diagnostics:
   - Burned fraction histogram (edges: 0, (0,0.001], (0.001,0.01], (0.01,0.05], (0.05,0.2], (0.2,0.5], (0.5,0.9], (0.9,1])
   - Counts and fractions in the 4 quota bins ([0, 0.2), [0.2, 0.5), [0.5, 0.9), [0.9, 1.0])
   - Burned fraction by wind peak bands (1-4, 4-8, 8-12, 12-16, 16-20 m/s) with n, median, P90
   - Burned fraction by fire_id (n, median, P90)
   - Ignition on non-burnable cell count
c) Seed reliability:
   - 20 candidates stratified over K=1 bins, run K=16 each.
   - Reports mean burned fraction over K=16 vs K=1, Spearman correlation,
     bin agreement rate, and across-seed std.
   - Saves reports/acceptance_2f_seed_reliability.csv.
d) Total wall time and seconds per run.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

from wf052.surrogate.dataset import (
    calculate_upwind_third_ignition,
    compute_slope_aspect,
    fbfm40_to_fuel_group,
)
from wf052.surrogate.standin_generator import (
    SEVEN_TRAINING_FIRES,
    assign_burn_fraction_bin,
    build_tile_fire_dataset_60m,
    load_and_verify_frozen_teacher_v2,
    load_fire_rasters_crop,
    load_held_out_fire_exclusions,
    sample_hrrr_wind_series,
    simulate_ensemble_teacher_v2,
)


def run_acceptance_measurement():
    print("=" * 75)
    print("PHASE 2F STEP 1: TEACHER BURNED-FRACTION DISTRIBUTION MEASUREMENT")
    print("=" * 75)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Active Device: {device} ({device.type.upper()})")

    # Load frozen teacher v2
    cfg_p = Path("configs/teacher_v2.yaml")
    cfg, cfg_hash, sim_hash, t_params, e_params = load_and_verify_frozen_teacher_v2(cfg_p)
    print(f"Teacher v2 Config Hash: {cfg_hash}")
    print(f"Canonical Sim Hash:     {sim_hash}")

    # Preload rasters for 7 training fires (as done in production sampler)
    tile_size = (256, 256)
    cached_fires: list[dict[str, Any]] = []
    print("\nPreloading rasters for 7 training fires...")
    for fire in SEVEN_TRAINING_FIRES:
        fbfm_base, elev_base, canopy_base, _ = load_fire_rasters_crop(
            fire_name=fire,
            tile_size=tile_size,
        )
        fuel_grp = fbfm40_to_fuel_group(fbfm_base)
        slope, asp_sin, asp_cos = compute_slope_aspect(elev_base, resolution_m=60.0)
        elev_z = ((elev_base - np.mean(elev_base)) / (np.std(elev_base) + 1e-5)).astype(np.float32)

        # Get crop origin from full fire domain
        import rasterio
        with rasterio.open(f"data/tiles/fires/{fire}/elevation.tif") as src:
            h_full, w_full = src.shape
            r0 = max(0, (h_full - tile_size[0]) // 2)
            c0 = max(0, (w_full - tile_size[1]) // 2)

        cached_fires.append({
            "fire": fire,
            "fbfm": fbfm_base,
            "elev": elev_base,
            "canopy": canopy_base,
            "fuel_grp": fuel_grp,
            "slope": slope,
            "asp_sin": asp_sin,
            "asp_cos": asp_cos,
            "elev_z": elev_z,
            "fire_dir": Path("data/tiles/fires") / fire,
            "crop_origin": (r0, c0),
        })
        print(f"  Loaded '{fire}': full=({h_full}, {w_full}), crop origin=({r0}, {c0})")

    # 1. Generate 240 candidates with K=1
    num_candidates = 240
    base_seed = 42000
    candidates: list[dict[str, Any]] = []
    candidate_sim_data: list[dict[str, Any]] = []

    print(f"\nRunning {num_candidates} candidate simulations (K=1, 12 h horizon)...")
    t0_all = time.perf_counter()

    for attempt in range(num_candidates):
        f_idx = attempt % len(cached_fires)
        f_info = cached_fires[f_idx]
        fire = f_info["fire"]
        r0, c0 = f_info["crop_origin"]

        seed = base_seed + attempt * 17
        rng = np.random.RandomState(seed)

        wind_base = sample_hrrr_wind_series(
            fire_dir=f_info["fire_dir"],
            rng=rng,
            window_hours=12,
            target_peak_speed_range=(1.0, 20.0),
        )

        # Perturbations
        spd_mult = float(rng.uniform(0.90, 1.10))
        dir_jitter_deg = float(rng.uniform(-20.0, 20.0))
        ign_offset_r = int(rng.randint(-10, 11))
        ign_offset_c = int(rng.randint(-10, 11))

        jitter_rad = np.deg2rad(dir_jitter_deg)
        cos_j, sin_j = np.cos(jitter_rad), np.sin(jitter_rad)
        u_j = (wind_base[:, 0] * cos_j - wind_base[:, 1] * sin_j) * spd_mult
        v_j = (wind_base[:, 0] * sin_j + wind_base[:, 1] * cos_j) * spd_mult
        hourly_wind_perturbed = np.stack([u_j, v_j], axis=-1).astype(np.float32)

        ign_r, ign_c = calculate_upwind_third_ignition(
            hourly_wind_uv=hourly_wind_perturbed,
            tile_shape=tile_size,
            offset_r=ign_offset_r,
            offset_c=ign_offset_c,
        )

        ign_mask = np.zeros(tile_size, dtype=np.float32)
        r_lo, r_hi = max(0, ign_r - 1), min(tile_size[0], ign_r + 2)
        c_lo, c_hi = max(0, ign_c - 1), min(tile_size[1], ign_c + 2)
        ign_mask[r_lo:r_hi, c_lo:c_hi] = 1.0

        ds_tile = build_tile_fire_dataset_60m(
            fire_name=fire,
            fbfm_arr=f_info["fbfm"],
            elev_arr=f_info["elev"],
            canopy_arr=f_info["canopy"],
            hourly_wind_uv=hourly_wind_perturbed,
            device=device,
        )

        t0_run = time.perf_counter()
        (
            burn_prob,
            q10,
            q50,
            q90,
            c10,
            c50,
            c90,
            raw_runs,
        ) = simulate_ensemble_teacher_v2(
            ds=ds_tile,
            t_params=t_params,
            e_params=e_params,
            ign_mask=ign_mask,
            k_seeds=1,
            scenario_seed=seed,
            horizon_min=720.0,
        )
        run_sec = time.perf_counter() - t0_run

        arr = raw_runs[0]
        burned_frac = float(np.mean(arr <= 720.0))

        speeds = np.hypot(hourly_wind_perturbed[:, 0], hourly_wind_perturbed[:, 1])
        w_peak = float(np.max(speeds))
        w_mean = float(np.mean(speeds))

        fbfm_ign = int(f_info["fbfm"][ign_r, ign_c])
        fuel_grp_ign = int(f_info["fuel_grp"][ign_r, ign_c])
        is_non_burnable = (fbfm_ign <= 0 or fbfm_ign in [91, 92, 93, 98, 99] or fuel_grp_ign == 0)

        record = {
            "candidate_id": attempt,
            "fire_id": fire,
            "crop_origin_r": r0,
            "crop_origin_c": c0,
            "ign_r": ign_r,
            "ign_c": ign_c,
            "fbfm40_ignition": fbfm_ign,
            "fuel_group_ignition": fuel_grp_ign,
            "is_non_burnable_ignition": is_non_burnable,
            "wind_peak": round(w_peak, 3),
            "wind_mean": round(w_mean, 3),
            "burned_fraction": round(burned_frac, 6),
            "wall_seconds": round(run_sec, 4),
            "seed": seed,
            "bin": assign_burn_fraction_bin(burned_frac),
        }
        candidates.append(record)

        # Keep simulation inputs for step 1c
        candidate_sim_data.append({
            "ds_tile": ds_tile,
            "ign_mask": ign_mask,
            "seed": seed,
            "fire_id": fire,
            "burned_frac_k1": burned_frac,
            "bin_k1": assign_burn_fraction_bin(burned_frac),
        })

        if (attempt + 1) % 40 == 0 or (attempt + 1) == num_candidates:
            elapsed_so_far = time.perf_counter() - t0_all
            print(f"  [{attempt + 1:3d}/{num_candidates}] Done. Elapsed: {elapsed_so_far:.1f} s ({elapsed_so_far / (attempt + 1):.3f} s/run)")

    total_k1_time = time.perf_counter() - t0_all
    sec_per_k1_run = total_k1_time / num_candidates

    df_cand = pd.DataFrame(candidates)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "acceptance_2f_candidates.csv"
    df_cand.to_csv(csv_path, index=False)
    print(f"\nSaved 240 candidates to {csv_path}")

    # =========================================================================
    # STEP 1b: Report Distributions & Cross-tabs
    # =========================================================================
    print("\n" + "=" * 75)
    print("STEP 1b: BURNED FRACTION DISTRIBUTION & CROSS-TABS")
    print("=" * 75)

    # 1. Fine-grained Histogram
    # Edges: 0, (0, 0.001], (0.001, 0.01], (0.01, 0.05], (0.05, 0.2], (0.2, 0.5], (0.5, 0.9], (0.9, 1.0]
    bf_vals = df_cand["burned_fraction"].values
    exact_zero = np.sum(bf_vals == 0.0)
    c_0_001 = np.sum((bf_vals > 0.0) & (bf_vals <= 0.001))
    c_001_01 = np.sum((bf_vals > 0.001) & (bf_vals <= 0.01))
    c_01_05 = np.sum((bf_vals > 0.01) & (bf_vals <= 0.05))
    c_05_20 = np.sum((bf_vals > 0.05) & (bf_vals <= 0.20))
    c_20_50 = np.sum((bf_vals > 0.20) & (bf_vals <= 0.50))
    c_50_90 = np.sum((bf_vals > 0.50) & (bf_vals <= 0.90))
    c_90_100 = np.sum((bf_vals > 0.90) & (bf_vals <= 1.00))

    hist_rows = [
        {"Interval": "0 (exact zero)", "Count": exact_zero, "Fraction": exact_zero / num_candidates},
        {"Interval": "(0, 0.001]", "Count": c_0_001, "Fraction": c_0_001 / num_candidates},
        {"Interval": "(0.001, 0.01]", "Count": c_001_01, "Fraction": c_001_01 / num_candidates},
        {"Interval": "(0.01, 0.05]", "Count": c_01_05, "Fraction": c_01_05 / num_candidates},
        {"Interval": "(0.05, 0.2]", "Count": c_05_20, "Fraction": c_05_20 / num_candidates},
        {"Interval": "(0.2, 0.5]", "Count": c_20_50, "Fraction": c_20_50 / num_candidates},
        {"Interval": "(0.5, 0.9]", "Count": c_50_90, "Fraction": c_50_90 / num_candidates},
        {"Interval": "(0.9, 1.0]", "Count": c_90_100, "Fraction": c_90_100 / num_candidates},
    ]
    df_hist = pd.DataFrame(hist_rows)
    print("\n--- Burned Fraction Histogram (Fine Edges) ---")
    print(df_hist.to_string(index=False))

    # 2. Four Current Quota Bins
    # Bin 0: [0, 0.2), Bin 1: [0.2, 0.5), Bin 2: [0.5, 0.9), Bin 3: [0.9, 1.0]
    q_b0 = np.sum(bf_vals < 0.2)
    q_b1 = np.sum((bf_vals >= 0.2) & (bf_vals < 0.5))
    q_b2 = np.sum((bf_vals >= 0.5) & (bf_vals < 0.9))
    q_b3 = np.sum(bf_vals >= 0.9)
    quota_rows = [
        {"Bin Index": 0, "Range": "[0.0, 0.2)", "Count": q_b0, "Fraction": q_b0 / num_candidates, "Percentage": f"{q_b0/num_candidates*100:.1f}%"},
        {"Bin Index": 1, "Range": "[0.2, 0.5)", "Count": q_b1, "Fraction": q_b1 / num_candidates, "Percentage": f"{q_b1/num_candidates*100:.1f}%"},
        {"Bin Index": 2, "Range": "[0.5, 0.9)", "Count": q_b2, "Fraction": q_b2 / num_candidates, "Percentage": f"{q_b2/num_candidates*100:.1f}%"},
        {"Bin Index": 3, "Range": "[0.9, 1.0]", "Count": q_b3, "Fraction": q_b3 / num_candidates, "Percentage": f"{q_b3/num_candidates*100:.1f}%"},
    ]
    df_quota = pd.DataFrame(quota_rows)
    print("\n--- Current Quota Bins Breakdown ---")
    print(df_quota.to_string(index=False))

    # 3. Burned Fraction by Wind Peak Band
    # Bands: 1-4, 4-8, 8-12, 12-16, 16-20 m/s
    wind_bands = [
        ("1-4 m/s", 1.0, 4.0),
        ("4-8 m/s", 4.0, 8.0),
        ("8-12 m/s", 8.0, 12.0),
        ("12-16 m/s", 12.0, 16.0),
        ("16-20 m/s", 16.0, 20.0),
    ]
    wind_rows = []
    for lbl, lo, hi in wind_bands:
        sub = df_cand[(df_cand["wind_peak"] >= lo) & (df_cand["wind_peak"] < hi if hi < 20.0 else df_cand["wind_peak"] <= hi)]
        n = len(sub)
        med = float(sub["burned_fraction"].median()) if n > 0 else np.nan
        p90 = float(np.percentile(sub["burned_fraction"], 90)) if n > 0 else np.nan
        wind_rows.append({
            "Wind Band": lbl,
            "n": n,
            "Median Burned Fraction": med,
            "P90 Burned Fraction": p90,
        })
    df_wind = pd.DataFrame(wind_rows)
    print("\n--- Burned Fraction by Wind Peak Band ---")
    print(df_wind.to_string(index=False))

    # 4. Burned Fraction by Fire ID
    fire_rows = []
    for fire in SEVEN_TRAINING_FIRES:
        sub = df_cand[df_cand["fire_id"] == fire]
        n = len(sub)
        med = float(sub["burned_fraction"].median()) if n > 0 else np.nan
        p90 = float(np.percentile(sub["burned_fraction"], 90)) if n > 0 else np.nan
        fire_rows.append({
            "Fire ID": fire,
            "n": n,
            "Median Burned Fraction": med,
            "P90 Burned Fraction": p90,
        })
    df_fire = pd.DataFrame(fire_rows)
    print("\n--- Burned Fraction by Fire ID ---")
    print(df_fire.to_string(index=False))

    # 5. Non-burnable ignition count
    non_burnable_count = int(df_cand["is_non_burnable_ignition"].sum())
    print(f"\nNon-burnable ignition cell count: {non_burnable_count} / {num_candidates} ({non_burnable_count / num_candidates * 100:.2f}%)")

    # =========================================================================
    # STEP 1c: Seed Reliability (20 candidates, K=16 each)
    # =========================================================================
    print("\n" + "=" * 75)
    print("STEP 1c: SEED RELIABILITY (20 CANDIDATES, K=16 EACH)")
    print("=" * 75)

    # Stratified selection of 20 candidates across populated bins
    selected_indices: list[int] = []
    for b in range(4):
        b_indices = [i for i, c in enumerate(candidate_sim_data) if c["bin_k1"] == b]
        # Target 5 from each bin, or take all if bin has fewer than 5
        n_take = min(5, len(b_indices))
        if n_take > 0:
            # Deterministic spread
            step_idx = max(1, len(b_indices) // n_take)
            chosen = [b_indices[k * step_idx] for k in range(n_take)]
            selected_indices.extend(chosen)

    # If fewer than 20 total, fill from remaining bins proportionally
    if len(selected_indices) < 20:
        remain = [i for i in range(num_candidates) if i not in selected_indices]
        needed = 20 - len(selected_indices)
        step_idx = max(1, len(remain) // needed)
        selected_indices.extend([remain[k * step_idx] for k in range(needed)])
    selected_indices = selected_indices[:20]

    print(f"Selected {len(selected_indices)} candidates stratified across K=1 bins for K=16 evaluation.")

    reliability_rows = []
    t0_k16 = time.perf_counter()

    for r_idx, c_idx in enumerate(selected_indices, start=1):
        c_info = candidate_sim_data[c_idx]
        seed = c_info["seed"]
        ds_tile = c_info["ds_tile"]
        ign_mask = c_info["ign_mask"]
        bf_k1 = c_info["burned_frac_k1"]
        bin_k1 = c_info["bin_k1"]
        fire = c_info["fire_id"]

        t0_cand = time.perf_counter()
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            raw_runs_16,
        ) = simulate_ensemble_teacher_v2(
            ds=ds_tile,
            t_params=t_params,
            e_params=e_params,
            ign_mask=ign_mask,
            k_seeds=16,
            scenario_seed=seed,
            horizon_min=720.0,
        )
        cand_elapsed = time.perf_counter() - t0_cand

        bf_16 = [float(np.mean(arr <= 720.0)) for arr in raw_runs_16]
        bf_mean_k16 = float(np.mean(bf_16))
        bf_std_k16 = float(np.std(bf_16))
        bin_k16 = assign_burn_fraction_bin(bf_mean_k16)
        agrees = (bin_k1 == bin_k16)

        reliability_rows.append({
            "candidate_id": c_idx,
            "fire_id": fire,
            "seed": seed,
            "bf_k1": round(bf_k1, 6),
            "bf_mean_k16": round(bf_mean_k16, 6),
            "bf_std_k16": round(bf_std_k16, 6),
            "bin_k1": bin_k1,
            "bin_k16_mean": bin_k16,
            "bin_agreement": agrees,
            "wall_sec_k16": round(cand_elapsed, 3),
        })
        print(f"  [{r_idx:2d}/20] Cand {c_idx:3d} ({fire:8s}): K=1 bf={bf_k1:.4f} (bin {bin_k1}) -> K=16 mean={bf_mean_k16:.4f} (bin {bin_k16}, std={bf_std_k16:.4f}, agree={agrees}) in {cand_elapsed:.2f} s")

    total_k16_time = time.perf_counter() - t0_k16
    df_rel = pd.DataFrame(reliability_rows)
    rel_csv_path = reports_dir / "acceptance_2f_seed_reliability.csv"
    df_rel.to_csv(rel_csv_path, index=False)
    print(f"\nSaved seed reliability results to {rel_csv_path}")

    # Spearman Correlation
    spearman_corr, spearman_pval = stats.spearmanr(df_rel["bf_k1"], df_rel["bf_mean_k16"])
    agreement_rate = float(df_rel["bin_agreement"].mean()) * 100.0

    print("\n--- Seed Reliability Metrics Summary ---")
    print(df_rel[["candidate_id", "fire_id", "bf_k1", "bf_mean_k16", "bf_std_k16", "bin_k1", "bin_k16_mean", "bin_agreement"]].to_string(index=False))
    print(f"\nSpearman Rank Correlation (K=1 vs K=16 mean): {spearman_corr:.4f} (p-value: {spearman_pval:.4e})")
    print(f"K=1 to K=16-mean Bin Agreement Rate:          {agreement_rate:.1f}% ({int(df_rel['bin_agreement'].sum())}/20)")
    print(f"Bin Mismatch Rate:                            {100.0 - agreement_rate:.1f}% ({20 - int(df_rel['bin_agreement'].sum())}/20)")
    print(f"Mean Across-Seed Std of Burned Fraction:      {df_rel['bf_std_k16'].mean():.4f}")

    # =========================================================================
    # STEP 1d: Total Wall Time and Timing
    # =========================================================================
    total_step1_time = total_k1_time + total_k16_time
    total_sim_runs = num_candidates + 20 * 16

    print("\n" + "=" * 75)
    print("STEP 1d: TIMING SUMMARY")
    print("=" * 75)
    print(f"Total Step 1 Wall Time:           {total_step1_time:.2f} s ({total_step1_time / 60:.2f} min)")
    print(f"  - 240 K=1 Candidate Runs:       {total_k1_time:.2f} s ({sec_per_k1_run:.3f} s/run)")
    print(f"  - 20 K=16 Candidate Runs (320): {total_k16_time:.2f} s ({total_k16_time / 320:.3f} s/run, {total_k16_time / 20:.2f} s/candidate)")
    print(f"Overall Seconds per Simulation:   {total_step1_time / total_sim_runs:.3f} s/run")
    print("=" * 75)

    return df_cand, df_rel


if __name__ == "__main__":
    run_acceptance_measurement()
