"""Phase 2E Step 6: Ember Ablation Attribution across 7 Folds.

Arms:
1. OFF (p_base=0.21): Contact-only with p_base=0.21, embers OFF.
2. OFF (p_base=0.22): Contact-only with p_base=0.22, embers OFF.
3. ON (teacher_v2 config): Frozen teacher_v2 (p_base=0.22, embers ON).

Protocol:
- 7 folds (bobcat, caldor, creek, glass, kincade, tamarack, zogg)
- 12 h horizon, 5 seeds each (42, 59, 76, 93, 110)
- Total runs = 3 arms * 7 folds * 5 seeds = 105 runs
- Metrics at +12 h: IoU (~2km GOFER) and area ratio (sim/obs)
- Report per-fold mean and range [min, max]
- Report fold-averaged summary table
- State percentage of ON vs OFF(0.21) difference attributable to p_base alone
"""

import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch

from wf052.sim.ember_kernel import (
    EmberParams,
    simulate_teacher_with_embers_60m,
)
from wf052.sim.teacher_calibrator import (
    FireDataset60m,
    TeacherParams,
    evaluate_simulation_multiscale,
    load_fire_dataset_60m,
)
from wf052.surrogate.standin_generator import load_and_verify_frozen_teacher_v2


def run_step6_ablation():
    print("=" * 70)
    print("STEP 6: EMBER ABLATION ATTRIBUTION (7 FOLDS, 5 SEEDS EACH)")
    print("=" * 70)

    # 1. Load frozen teacher_v2 configuration
    cfg_p = Path("configs/teacher_v2.yaml")
    cfg, cfg_hash, sim_hash, teacher_base_022, ember_params_v2 = load_and_verify_frozen_teacher_v2(cfg_p)
    print(f"Teacher v2 Config Hash: {cfg_hash}")
    print(f"Canonical Sim Hash:     {sim_hash}")

    # Build teacher params for p_base=0.21 (all other contact params identical)
    teacher_base_021 = TeacherParams(
        p_base=0.21,
        a=teacher_base_022.a,
        c_1=teacher_base_022.c_1,
        c_2=teacher_base_022.c_2,
        p_continue=teacher_base_022.p_continue,
        fuel_scale_grass_shrub=teacher_base_022.fuel_scale_grass_shrub,
        fuel_scale_timber=teacher_base_022.fuel_scale_timber,
        canopy_den_scale=teacher_base_022.canopy_den_scale,
        wind_moisture_damping=teacher_base_022.wind_moisture_damping,
    )

    # 7 training folds
    fold_names = ["bobcat", "caldor", "creek", "glass", "kincade", "tamarack", "zogg"]
    seeds = [42 + i * 17 for i in range(5)]  # 5 deterministic seeds: 42, 59, 76, 93, 110
    print(f"Folds ({len(fold_names)}): {fold_names}")
    print(f"Seeds ({len(seeds)}): {seeds}")

    # Device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Load 7 fire datasets using setup csv
    print("\nLoading datasets for 7 fires from reports/phase_1a_fires_setup.csv...")
    setup_df = pd.read_csv("reports/phase_1a_fires_setup.csv")
    datasets: dict[str, FireDataset60m] = {}
    for _, row in setup_df.iterrows():
        fname = str(row["fire_name"])
        regime = str(row["regime"])
        t_dir = Path("data/tiles/fires") / fname.lower()
        datasets[fname.lower()] = load_fire_dataset_60m(fname, regime, t_dir, device=device)

    # Define Arms
    arms = [
        {
            "arm_name": "OFF (p_base=0.21)",
            "teacher_params": teacher_base_021,
            "ember_params": None,
            "use_embers": False,
        },
        {
            "arm_name": "OFF (p_base=0.22)",
            "teacher_params": teacher_base_022,
            "ember_params": None,
            "use_embers": False,
        },
        {
            "arm_name": "ON (teacher_v2)",
            "teacher_params": teacher_base_022,
            "ember_params": ember_params_v2,
            "use_embers": True,
        },
    ]

    results: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    total_runs = len(fold_names) * len(arms) * len(seeds)
    run_idx = 0

    print(f"\nStarting {total_runs} simulations across 3 arms...")
    for f_idx, fname in enumerate(fold_names, start=1):
        ds = datasets[fname]
        t_fold_start = time.perf_counter()
        print(f"\n[{f_idx}/{len(fold_names)}] Processing fire '{fname}'...")

        for arm in arms:
            arm_name = arm["arm_name"]
            t_params = arm["teacher_params"]
            e_params = arm["ember_params"]
            use_emb = arm["use_embers"]

            for seed in seeds:
                run_idx += 1
                sim_out = simulate_teacher_with_embers_60m(
                    ds=ds,
                    params=t_params,
                    ember_params=e_params,
                    use_ember_kernel=use_emb,
                    duration_hours=12,
                    steps_per_hour=30,
                    seed=seed,
                )
                m_by_h = evaluate_simulation_multiscale(ds, sim_out, horizons=(12,))
                m12 = m_by_h[12]

                results.append({
                    "fire_name": fname,
                    "arm": arm_name,
                    "seed": seed,
                    "iou_12h": float(m12["iou_2km"]),
                    "area_ratio_12h": float(m12["area_ratio"]),
                    "f1_12h": float(m12["f1_2km"]),
                    "total_embers": int(sim_out["total_ember_ignitions"]),
                })

        fold_elapsed = time.perf_counter() - t_fold_start
        print(f"  Completed '{fname}' in {fold_elapsed:.2f} s ({fold_elapsed / (len(arms) * len(seeds)):.2f} s/run)")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nFinished all {total_runs} simulations in {total_elapsed:.1f} s ({total_elapsed / 60:.2f} min).")

    df = pd.DataFrame(results)
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "step6_ember_ablation_raw_runs.csv", index=False)

    # 1. Per-Fold Aggregations (mean and range for IoU and Area Ratio)
    print("\n" + "=" * 70)
    print("PER-FOLD RESULTS (Mean and Range [Min, Max])")
    print("=" * 70)

    per_fold_summary = []
    for (fname, arm_name), g in df.groupby(["fire_name", "arm"], sort=False):
        iou_mean = g["iou_12h"].mean()
        iou_min = g["iou_12h"].min()
        iou_max = g["iou_12h"].max()
        ar_mean = g["area_ratio_12h"].mean()
        ar_min = g["area_ratio_12h"].min()
        ar_max = g["area_ratio_12h"].max()
        per_fold_summary.append({
            "fire": fname,
            "arm": arm_name,
            "iou_mean": iou_mean,
            "iou_range": f"[{iou_min:.3f}, {iou_max:.3f}]",
            "ar_mean": ar_mean,
            "ar_range": f"[{ar_min:.3f}, {ar_max:.3f}]",
        })

    df_pf = pd.DataFrame(per_fold_summary)
    print(df_pf.to_string(index=False))
    df_pf.to_csv(out_dir / "step6_ember_ablation_per_fold.csv", index=False)

    # 2. Fold-Averaged Summary Table
    print("\n" + "=" * 70)
    print("FOLD-AVERAGED TABLE (Across all 7 Folds)")
    print("=" * 70)

    # First average over seeds within each fold, then average across the 7 folds
    fold_means = df.groupby(["arm", "fire_name"])[["iou_12h", "area_ratio_12h"]].mean().reset_index()
    fold_avg = fold_means.groupby("arm")[["iou_12h", "area_ratio_12h"]].agg(["mean", "std"]).reset_index()

    table_rows = []
    for arm_name in ["OFF (p_base=0.21)", "OFF (p_base=0.22)", "ON (teacher_v2)"]:
        sub = fold_means[fold_means["arm"] == arm_name]
        table_rows.append({
            "Arm": arm_name,
            "Mean IoU (12h)": sub["iou_12h"].mean(),
            "Std IoU": sub["iou_12h"].std(),
            "Mean Area Ratio (12h)": sub["area_ratio_12h"].mean(),
            "Std Area Ratio": sub["area_ratio_12h"].std(),
        })

    df_summary = pd.DataFrame(table_rows)
    print(df_summary.to_string(index=False))
    df_summary.to_csv(out_dir / "step6_ember_ablation_summary.csv", index=False)

    # 3. Attribution Analysis: How much of ON vs OFF(0.21) difference is p_base alone?
    iou_off_21 = df_summary.loc[df_summary["Arm"] == "OFF (p_base=0.21)", "Mean IoU (12h)"].values[0]
    iou_off_22 = df_summary.loc[df_summary["Arm"] == "OFF (p_base=0.22)", "Mean IoU (12h)"].values[0]
    iou_on = df_summary.loc[df_summary["Arm"] == "ON (teacher_v2)", "Mean IoU (12h)"].values[0]

    ar_off_21 = df_summary.loc[df_summary["Arm"] == "OFF (p_base=0.21)", "Mean Area Ratio (12h)"].values[0]
    ar_off_22 = df_summary.loc[df_summary["Arm"] == "OFF (p_base=0.22)", "Mean Area Ratio (12h)"].values[0]
    ar_on = df_summary.loc[df_summary["Arm"] == "ON (teacher_v2)", "Mean Area Ratio (12h)"].values[0]

    # IoU Attribution
    tot_diff_iou = iou_on - iou_off_21
    pbase_diff_iou = iou_off_22 - iou_off_21
    ember_diff_iou = iou_on - iou_off_22
    pct_pbase_iou = (pbase_diff_iou / tot_diff_iou * 100.0) if tot_diff_iou != 0 else 0.0

    # Area Ratio Attribution
    tot_diff_ar = ar_on - ar_off_21
    pbase_diff_ar = ar_off_22 - ar_off_21
    ember_diff_ar = ar_on - ar_off_22
    pct_pbase_ar = (pbase_diff_ar / tot_diff_ar * 100.0) if tot_diff_ar != 0 else 0.0

    print("\n" + "=" * 70)
    print("STEP 6 ATTRIBUTION: p_base vs Embers")
    print("=" * 70)
    print(f"IoU (12h):")
    print(f"  OFF (0.21):    {iou_off_21:.4f}")
    print(f"  OFF (0.22):    {iou_off_22:.4f} (+{pbase_diff_iou:+.4f} vs OFF 0.21)")
    print(f"  ON (v2):       {iou_on:.4f} (+{ember_diff_iou:+.4f} vs OFF 0.22, +{tot_diff_iou:+.4f} total)")
    print(f"  --> p_base alone accounts for {pct_pbase_iou:.1f}% of the IoU difference ({pbase_diff_iou:+.4f} of {tot_diff_iou:+.4f}).")
    print(f"  --> Embers account for {100.0 - pct_pbase_iou:.1f}% of the IoU difference ({ember_diff_iou:+.4f} of {tot_diff_iou:+.4f}).")

    print(f"\nArea Ratio (12h):")
    print(f"  OFF (0.21):    {ar_off_21:.4f}")
    print(f"  OFF (0.22):    {ar_off_22:.4f} (+{pbase_diff_ar:+.4f} vs OFF 0.21)")
    print(f"  ON (v2):       {ar_on:.4f} (+{ember_diff_ar:+.4f} vs OFF 0.22, +{tot_diff_ar:+.4f} total)")
    print(f"  --> p_base alone accounts for {pct_pbase_ar:.1f}% of the Area Ratio difference ({pbase_diff_ar:+.4f} of {tot_diff_ar:+.4f}).")
    print(f"  --> Embers account for {100.0 - pct_pbase_ar:.1f}% of the Area Ratio difference ({ember_diff_ar:+.4f} of {tot_diff_ar:+.4f}).")
    print("=" * 70)

    return df_summary


if __name__ == "__main__":
    run_step6_ablation()
