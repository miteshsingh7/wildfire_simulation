"""Run production smoke run with full candidate-level instrumentation (Phase 2H Step 2).

Records for every candidate:
- fire, ignition cell, fuel code, fuel group at ignition
- K=1 sim wall time, K=1 burned fraction, target bin, accept/reject
- extra-seed wall time, and any overhead.
Saves:
- reports/smoke_2h_attempts_log.csv
- reports/step2b_sampler_breakdown.csv
- reports/step2c_glass_fuel_codes.csv
- reports/step2_smoke_summary_2h.json
"""

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wf052.surrogate.standin_generator import (
    generate_stratified_samples_rejection,
    save_shard_with_metadata,
)


def main() -> None:
    print("=" * 70)
    print("Phase 2H Step 2: Production SMOKE Mini-Run with Instrumentation")
    print("=" * 70)

    mode = os.environ.get("MODE", "smoke")
    max_minutes = float(os.environ.get("MAX_MINUTES", 20.0))
    max_wall_sec = max_minutes * 60.0
    k_ensemble = int(os.environ.get("K_ENSEMBLE", 4))
    scenarios_per_fire = int(os.environ.get("SCENARIOS_PER_FIRE", 2))
    max_attempts_per_bin = int(os.environ.get("MAX_ATTEMPTS_PER_BIN", 300))

    shards_dir = Path("data/shards_smoke_2h")
    if shards_dir.exists():
        shutil.rmtree(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    fires_list = ["bobcat", "caldor", "creek", "glass", "kincade", "tamarack", "zogg"]
    scenarios_cfg_file = Path("configs/scenarios.yaml")
    teacher_cfg_path = Path("configs/teacher_v2.yaml")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Device:                   {device}")
    print(f"Mode:                     {mode}")
    print(f"Ensemble K:               {k_ensemble}")
    print(f"Scenarios per fire:       {scenarios_per_fire}")
    print(f"Fires ({len(fires_list)}):           {', '.join(fires_list)}")
    print(f"Max minutes:              {max_minutes:.1f} ({max_wall_sec:.0f} s)")
    print(f"Max attempts per bin:     {max_attempts_per_bin}")
    print(f"Output directory:         {shards_dir}")
    print("-" * 70)

    attempt_log_csv = "reports/smoke_2h_attempts_log.csv"

    t0 = time.perf_counter()

    samples, rep = generate_stratified_samples_rejection(
        fire_names=fires_list,
        mode=mode,
        k_ensemble=k_ensemble,
        scenarios_per_fire=scenarios_per_fire,
        max_attempts=3000,
        max_attempts_per_bin=max_attempts_per_bin,
        max_wall_sec=max_wall_sec,
        progress_interval=5,
        base_seed=42000,
        tile_size=(256, 256),
        source_mode="teacher_v2_ensemble",
        teacher_config_path=teacher_cfg_path,
        fires_base_dir="data/tiles/fires",
        scenarios_config_path=scenarios_cfg_file,
        device=device,
        attempt_log_path=attempt_log_csv,
    )

    elapsed_total = time.perf_counter() - t0

    shard_id = "shard_000"
    df_meta = save_shard_with_metadata(samples, shard_id=shard_id, shards_dir=shards_dir)
    print(f"\nSaved shard {shard_id}: {len(samples)} samples, {len(df_meta)} metadata rows.")

    df_att = pd.read_csv(attempt_log_csv)
    print(f"Loaded attempt log: {len(df_att)} attempts.")

    # 2b breakdown per fire
    rows_2b = []
    for f in fires_list:
        sub = df_att[df_att["fire"] == f]
        n_drawn = len(sub)
        n_grp0 = int((sub["is_fuel_0"]).sum())
        sim_sub = sub[~sub["is_fuel_0"]]
        n_k1_sims = len(sim_sub)
        b0 = int((sim_sub["bin_k1"] == 0).sum())
        b1 = int((sim_sub["bin_k1"] == 1).sum())
        b2 = int((sim_sub["bin_k1"] == 2).sum())
        b3 = int((sim_sub["bin_k1"] == 3).sum())
        b4 = int((sim_sub["bin_k1"] == 4).sum())
        n_acc = int((sub["accepted"]).sum())
        wall_k1 = float(sim_sub["k1_sim_sec"].sum())
        wall_ext = float(sub["extra_seeds_sim_sec"].sum())
        wall_over = float(sub["overhead_sec"].sum())

        rows_2b.append({
            "fire": f,
            "candidates_drawn": n_drawn,
            "fuel_grp0_rejections": n_grp0,
            "k1_sims_run": n_k1_sims,
            "bin0_count": b0,
            "bin1_count": b1,
            "bin2_count": b2,
            "bin3_count": b3,
            "bin4_count": b4,
            "accepted": n_acc,
            "wall_sec_k1_sims": round(wall_k1, 2),
            "wall_sec_extra_seeds": round(wall_ext, 2),
            "wall_sec_overhead": round(wall_over, 2),
            "wall_sec_total": round(wall_k1 + wall_ext + wall_over, 2),
        })

    df_2b = pd.DataFrame(rows_2b)
    # Add Total row
    tot_row = {
        "fire": "Total",
        "candidates_drawn": int(df_2b["candidates_drawn"].sum()),
        "fuel_grp0_rejections": int(df_2b["fuel_grp0_rejections"].sum()),
        "k1_sims_run": int(df_2b["k1_sims_run"].sum()),
        "bin0_count": int(df_2b["bin0_count"].sum()),
        "bin1_count": int(df_2b["bin1_count"].sum()),
        "bin2_count": int(df_2b["bin2_count"].sum()),
        "bin3_count": int(df_2b["bin3_count"].sum()),
        "bin4_count": int(df_2b["bin4_count"].sum()),
        "accepted": int(df_2b["accepted"].sum()),
        "wall_sec_k1_sims": round(float(df_2b["wall_sec_k1_sims"].sum()), 2),
        "wall_sec_extra_seeds": round(float(df_2b["wall_sec_extra_seeds"].sum()), 2),
        "wall_sec_overhead": round(float(df_2b["wall_sec_overhead"].sum()), 2),
        "wall_sec_total": round(float(df_2b["wall_sec_total"].sum()), 2),
    }
    df_2b_with_tot = pd.concat([df_2b, pd.DataFrame([tot_row])], ignore_index=True)
    df_2b_with_tot.to_csv("reports/step2b_sampler_breakdown.csv", index=False)
    print("\n--- Step 2b Sampler Breakdown ---")
    print(df_2b_with_tot.to_string(index=False))

    # 2c Glass diagnosis
    sub_glass = df_att[df_att["fire"] == "glass"]
    glass_total = len(sub_glass)
    glass_grp0 = int((sub_glass["is_fuel_0"]).sum())
    glass_share_grp0 = glass_grp0 / glass_total if glass_total > 0 else 0.0

    glass_fuel_counts = sub_glass["fbfm40"].value_counts().reset_index()
    glass_fuel_counts.columns = ["fbfm40", "candidate_count"]
    glass_fuel_counts["pct_of_glass_candidates"] = round(glass_fuel_counts["candidate_count"] / glass_total * 100, 2)
    glass_fuel_counts["is_group0"] = glass_fuel_counts["fbfm40"].isin([91, 92, 93, 98, 99])
    glass_fuel_counts.to_csv("reports/step2c_glass_fuel_codes.csv", index=False)
    print("\n--- Step 2c Glass Candidate Fuel Codes ---")
    print(glass_fuel_counts.to_string(index=False))

    # 2d Sim call count and wall time
    k1_times = rep.k1_sim_times or []
    ext_times = rep.extra_seed_times or []

    k1_n = len(k1_times)
    k1_mean = float(np.mean(k1_times)) if k1_n > 0 else 0.0
    k1_p95 = float(np.percentile(k1_times, 95)) if k1_n > 0 else 0.0

    ext_n = len(ext_times)
    ext_mean = float(np.mean(ext_times)) if ext_n > 0 else 0.0
    ext_p95 = float(np.percentile(ext_times, 95)) if ext_n > 0 else 0.0

    print("\n--- Step 2d Sim Call Count and Timing (perf_counter) ---")
    print(f"K=1 Prescreen Sims:     count={k1_n}, mean={k1_mean:.4f} s, p95={k1_p95:.4f} s")
    print(f"Extra Seed Sims:        count={ext_n}, mean={ext_mean:.4f} s, p95={ext_p95:.4f} s")

    summary_step2 = {
        "elapsed_total_sec": round(elapsed_total, 2),
        "total_candidates_drawn": len(df_att),
        "total_accepted": len(samples),
        "k1_sims": {
            "count": k1_n,
            "mean_sec": round(k1_mean, 4),
            "p95_sec": round(k1_p95, 4),
            "total_sec": round(float(np.sum(k1_times)), 2),
        },
        "extra_seed_sims": {
            "count": ext_n,
            "mean_sec": round(ext_mean, 4),
            "p95_sec": round(ext_p95, 4),
            "total_sec": round(float(np.sum(ext_times)), 2),
        },
        "glass_diagnosis": {
            "total_candidates": glass_total,
            "fuel_grp0_rejections": glass_grp0,
            "share_grp0_rejections": round(glass_share_grp0, 4),
            "k1_sims_run": int((~sub_glass["is_fuel_0"]).sum()),
            "k1_bin_counts": {
                "bin0": int((sub_glass[~sub_glass["is_fuel_0"]]["bin_k1"] == 0).sum()),
                "bin1": int((sub_glass[~sub_glass["is_fuel_0"]]["bin_k1"] == 1).sum()),
                "bin2": int((sub_glass[~sub_glass["is_fuel_0"]]["bin_k1"] == 2).sum()),
                "bin3": int((sub_glass[~sub_glass["is_fuel_0"]]["bin_k1"] == 3).sum()),
                "bin4": int((sub_glass[~sub_glass["is_fuel_0"]]["bin_k1"] == 4).sum()),
            },
        },
    }

    with open("reports/step2_smoke_summary_2h.json", "w", encoding="utf-8") as f:
        json.dump(summary_step2, f, indent=2)
    print("\nSaved reports/step2_smoke_summary_2h.json")


if __name__ == "__main__":
    main()
