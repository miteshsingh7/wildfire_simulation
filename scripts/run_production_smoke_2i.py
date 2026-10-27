"""Run production smoke run with stall fix (Phase 2I Step 1d).

Parameters:
- MAX_MINUTES: 20
- SCENARIOS_PER_FIRE: 2
- STALL_CAP: 60
- K_ENSEMBLE: 4

Saves:
- reports/smoke_2i_attempts_log.csv
- reports/step1d_smoke_breakdown_2i.csv
- reports/step1d_smoke_closed_by_2i.csv
- reports/step1d_smoke_summary_2i.json
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
    format_fire_bin_status_table,
    format_fire_bin_table,
    generate_stratified_samples_rejection,
    save_shard_with_metadata,
)


def main() -> None:
    print("=" * 70)
    print("Phase 2I Step 1d: Production SMOKE Mini-Run with Stall Fix")
    print("=" * 70)

    mode = os.environ.get("MODE", "smoke")
    max_minutes = float(os.environ.get("MAX_MINUTES", 20.0))
    max_wall_sec = max_minutes * 60.0
    k_ensemble = int(os.environ.get("K_ENSEMBLE", 4))
    scenarios_per_fire = int(os.environ.get("SCENARIOS_PER_FIRE", 2))
    stall_cap = int(os.environ.get("STALL_CAP", 60))

    shards_dir = Path("data/shards_smoke_2i")
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
    print(f"Stall cap:                {stall_cap}")
    print(f"Fires ({len(fires_list)}):           {', '.join(fires_list)}")
    print(f"Max minutes:              {max_minutes:.1f} ({max_wall_sec:.0f} s)")
    print(f"Output directory:         {shards_dir}")
    print("-" * 70)

    attempt_log_csv = "reports/smoke_2i_attempts_log.csv"

    t0 = time.perf_counter()

    samples, rep = generate_stratified_samples_rejection(
        fire_names=fires_list,
        mode=mode,
        k_ensemble=k_ensemble,
        scenarios_per_fire=scenarios_per_fire,
        max_attempts=3000,
        stall_cap=stall_cap,
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

    # 1. Per-fire breakdown (same format as 2H table 2b)
    rows_per_fire = []
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

        rows_per_fire.append({
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

    df_per_fire = pd.DataFrame(rows_per_fire)
    tot_row = {
        "fire": "Total",
        "candidates_drawn": int(df_per_fire["candidates_drawn"].sum()),
        "fuel_grp0_rejections": int(df_per_fire["fuel_grp0_rejections"].sum()),
        "k1_sims_run": int(df_per_fire["k1_sims_run"].sum()),
        "bin0_count": int(df_per_fire["bin0_count"].sum()),
        "bin1_count": int(df_per_fire["bin1_count"].sum()),
        "bin2_count": int(df_per_fire["bin2_count"].sum()),
        "bin3_count": int(df_per_fire["bin3_count"].sum()),
        "bin4_count": int(df_per_fire["bin4_count"].sum()),
        "accepted": int(df_per_fire["accepted"].sum()),
        "wall_sec_k1_sims": round(float(df_per_fire["wall_sec_k1_sims"].sum()), 2),
        "wall_sec_extra_seeds": round(float(df_per_fire["wall_sec_extra_seeds"].sum()), 2),
        "wall_sec_overhead": round(float(df_per_fire["wall_sec_overhead"].sum()), 2),
        "wall_sec_total": round(float(df_per_fire["wall_sec_total"].sum()), 2),
    }
    df_per_fire_tot = pd.concat([df_per_fire, pd.DataFrame([tot_row])], ignore_index=True)
    df_per_fire_tot.to_csv("reports/step1d_smoke_breakdown_2i.csv", index=False)
    print("\n--- Per-Fire Table (Same as 2H Table 2b) ---")
    print(df_per_fire_tot.to_string(index=False))

    # 2. Per (fire, bin) accepted/target/closed_by table
    bin_names = ["[0, 0.002)", "[0.002, 0.02)", "[0.02, 0.1)", "[0.1, 0.5)", "[0.5, 1.0]"]
    rows_closed_by = []
    for f in fires_list:
        for b in range(5):
            tgt = scenarios_per_fire
            acc = rep.fire_bin_counts[f][b] if rep.fire_bin_counts else 0
            cb = rep.fire_bin_closed_by[f][b] if rep.fire_bin_closed_by else "unknown"
            rows_closed_by.append({
                "fire": f,
                "bin": b,
                "bin_name": bin_names[b],
                "target": tgt,
                "accepted": acc,
                "closed_by": cb,
            })

    df_closed_by = pd.DataFrame(rows_closed_by)
    df_closed_by.to_csv("reports/step1d_smoke_closed_by_2i.csv", index=False)
    print("\n--- Per (Fire, Bin) Status Table ---")
    print(df_closed_by.to_string(index=False))

    # 3. Overall summary JSON
    summary_data = {
        "elapsed_seconds": round(elapsed_total, 2),
        "total_attempts": rep.total_generated,
        "total_accepted": len(samples),
        "fire_visits": rep.fire_visits,
        "fire_bin_counts": rep.fire_bin_counts,
        "fire_bin_closed_by": rep.fire_bin_closed_by,
        "fire_bin_attempts": rep.fire_bin_attempts,
        "fire_group0_rejections": rep.fire_group0_rejections,
    }
    with open("reports/step1d_smoke_summary_2i.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print("\nSaved reports/step1d_smoke_summary_2i.json")


if __name__ == "__main__":
    main()
