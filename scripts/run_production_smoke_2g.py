"""Run production notebook logic locally in SMOKE mode with MAX_MINUTES=20 (Phase 2G Step 5).

Measures:
- per fire x bin table (target, accepted, attempts)
- sec/attempt (K=1 prescreen)
- sec/accepted sample (full K=4 ensemble)
- projection of accepted samples in 28,800 s for K=8 and K=16
"""

import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

from wf052.surrogate.standin_generator import (
    format_fire_bin_table,
    generate_stratified_samples_rejection,
    save_shard_with_metadata,
)

def main() -> None:
    print("=" * 70)
    print("Phase 2G Step 5: Local Production SMOKE Mini-Run (MAX_MINUTES=20)")
    print("=" * 70)

    mode = os.environ.get("MODE", "smoke")
    max_minutes = float(os.environ.get("MAX_MINUTES", 20.0))
    max_wall_sec = max_minutes * 60.0
    k_ensemble = int(os.environ.get("K_ENSEMBLE", 4))
    scenarios_per_fire = int(os.environ.get("SCENARIOS_PER_FIRE", 2))
    max_attempts_per_bin = int(os.environ.get("MAX_ATTEMPTS_PER_BIN", 300))

    shards_dir = Path("data/shards_smoke_2g")
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
    )

    elapsed_total = time.perf_counter() - t0

    # Save shard
    shard_id = "shard_000"
    df_meta = save_shard_with_metadata(samples, shard_id=shard_id, shards_dir=shards_dir)
    print(f"\nSaved shard {shard_id}: {len(samples)} samples, {len(df_meta)} metadata rows.")

    # Diagnostics & Timings
    n_accepted = len(samples)
    n_attempts = rep.total_generated
    sec_per_attempt = elapsed_total / max(1, n_attempts)
    sec_per_accepted = elapsed_total / max(1, n_accepted) if n_accepted > 0 else float("nan")

    # Time per seed estimate:
    # In each attempt: 1 prescreen seed is evaluated.
    # For each accepted candidate: K - 1 additional seeds are evaluated.
    # Total seed evaluations: n_attempts + n_accepted * (k_ensemble - 1)
    tot_seed_evals = n_attempts + n_accepted * (k_ensemble - 1)
    sec_per_seed = elapsed_total / max(1, tot_seed_evals)

    # Rejection ratio (candidates attempted per accepted sample)
    r = n_attempts / max(1, n_accepted)

    # Projections for 28,800 s (8 hours)
    budget_sec = 28800.0

    # Expected time per accepted sample for ensemble size K:
    # T(K) = r * sec_per_seed + (K - 1) * sec_per_seed = (r + K - 1) * sec_per_seed
    t_acc_k8 = (r + 8 - 1) * sec_per_seed
    t_acc_k16 = (r + 16 - 1) * sec_per_seed

    proj_k8 = int(budget_sec // t_acc_k8) if t_acc_k8 > 0 else 0
    proj_k16 = int(budget_sec // t_acc_k16) if t_acc_k16 > 0 else 0

    print("\n" + "=" * 70)
    print("PHASE 2G STEP 5 PRODUCTION SMOKE MINI-RUN SUMMARY")
    print("=" * 70)
    print(f"Wall-clock elapsed:         {elapsed_total:.1f} s ({elapsed_total / 60.0:.2f} min)")
    print(f"Total attempts:             {n_attempts}")
    print(f"Total accepted samples:     {n_accepted}")
    print(f"Rejection ratio (attempts/acc): {r:.2f}")
    print(f"Sec per attempt:            {sec_per_attempt:.3f} s")
    print(f"Sec per single seed eval:   {sec_per_seed:.3f} s")
    print(f"Sec per accepted sample (K=4): {sec_per_accepted:.2f} s")
    print(f"Est sec/accepted (K=8):     {t_acc_k8:.2f} s")
    print(f"Est sec/accepted (K=16):    {t_acc_k16:.2f} s")
    print("-" * 70)
    print(f"Projected accepted in 28,800 s (8h) at K=8:  {proj_k8:,} samples")
    print(f"Projected accepted in 28,800 s (8h) at K=16: {proj_k16:,} samples")
    print("=" * 70)

    # Save summary to reports/step5_smoke_summary.json
    summary_data = {
        "elapsed_seconds": round(elapsed_total, 2),
        "total_attempts": n_attempts,
        "total_accepted": n_accepted,
        "rejection_ratio": round(r, 4),
        "sec_per_attempt": round(sec_per_attempt, 4),
        "sec_per_seed": round(sec_per_seed, 4),
        "sec_per_accepted_k4": round(sec_per_accepted, 4),
        "sec_per_accepted_k8": round(t_acc_k8, 4),
        "sec_per_accepted_k16": round(t_acc_k16, 4),
        "projected_accepted_k8_28800s": proj_k8,
        "projected_accepted_k16_28800s": proj_k16,
        "fire_bin_counts": rep.fire_bin_counts,
        "fire_bin_attempts": rep.fire_bin_attempts,
        "fire_group0_rejections": rep.fire_group0_rejections,
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/step5_smoke_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print("Saved reports/step5_smoke_summary.json")

if __name__ == "__main__":
    main()
