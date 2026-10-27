#!/usr/bin/env python3
"""Evaluate Baselines B0 and B1 on smoke shards (Phase 3A Step 2).

Pipeline check only, with no scientific claim.
Fitted on train fires: ['kincade', 'glass', 'zogg', 'creek']
Evaluated on:
- Val fire: 'bobcat'
- Test fires: ['caldor', 'tamarack']
"""

from pathlib import Path
import pandas as pd

from wf052.surrogate.dataset import CompactSurrogateDataset
from wf052.surrogate.evaluate import (
    aggregate_per_bin_metrics,
    evaluate_baseline_on_dataset,
    fit_baseline_b0,
    fit_baseline_b1,
)


def run_baseline_evaluation() -> None:
    print("=" * 70)
    print("PHASE 3A STEP 2: BASELINE B0 & B1 EVALUATION ON SMOKE SHARDS")
    print("NOTE: THIS IS A PIPELINE CHECK ONLY, WITH NO CLAIM.")
    print("=" * 70)

    shard_dir = Path("data/shards_smoke_2i")
    ds = CompactSurrogateDataset(shard_dir)
    print(f"Loaded {len(ds)} samples from {shard_dir}")

    train_fires = ["kincade", "glass", "zogg", "creek"]
    val_fires = ["bobcat"]
    test_fires = ["caldor", "tamarack"]

    train_indices = [i for i, f in enumerate(ds.fire_ids) if f in train_fires]
    val_indices = [i for i, f in enumerate(ds.fire_ids) if f in val_fires]
    test_indices = [i for i, f in enumerate(ds.fire_ids) if f in test_fires]

    print(f"Train samples count: {len(train_indices)} (fires: {train_fires})")
    print(f"Val samples count:   {len(val_indices)} (fires: {val_fires})")
    print(f"Test samples count:  {len(test_indices)} (fires: {test_fires})")

    # 1. Fit B0 and B1
    print("\nFitting Baseline B0 on training fires ...")
    b0 = fit_baseline_b0(ds, train_fires=train_fires)
    print(f"B0 Train Mean Burn Prob: {b0.train_mean_burn_prob:.5f}")

    print("Fitting Baseline B1 on training fires ...")
    b1 = fit_baseline_b1(ds, train_fires=train_fires)
    print("B1 Lookup Table fitted successfully.")

    # 2. Evaluate B0 and B1 on Validation fire
    print("\n" + "=" * 50)
    print("EVALUATION ON VALIDATION FIRE ('bobcat')")
    print("=" * 50)

    df_b0_val = evaluate_baseline_on_dataset(b0, ds, val_indices, baseline_name="B0")
    df_b1_val = evaluate_baseline_on_dataset(b1, ds, val_indices, baseline_name="B1")

    agg_b0_val = aggregate_per_bin_metrics(df_b0_val)
    agg_b1_val = aggregate_per_bin_metrics(df_b1_val)

    agg_b0_val["model"] = "B0"
    agg_b0_val["split"] = "val"
    agg_b1_val["model"] = "B1"
    agg_b1_val["split"] = "val"

    print("\n--- Baseline B0 on Validation ('bobcat') ---")
    print(agg_b0_val.to_string(index=False))

    print("\n--- Baseline B1 on Validation ('bobcat') ---")
    print(agg_b1_val.to_string(index=False))

    # 3. Evaluate B0 and B1 on Test fires
    print("\n" + "=" * 50)
    print("EVALUATION ON TEST FIRES ('caldor', 'tamarack')")
    print("=" * 50)

    df_b0_test = evaluate_baseline_on_dataset(b0, ds, test_indices, baseline_name="B0")
    df_b1_test = evaluate_baseline_on_dataset(b1, ds, test_indices, baseline_name="B1")

    agg_b0_test = aggregate_per_bin_metrics(df_b0_test)
    agg_b1_test = aggregate_per_bin_metrics(df_b1_test)

    agg_b0_test["model"] = "B0"
    agg_b0_test["split"] = "test"
    agg_b1_test["model"] = "B1"
    agg_b1_test["split"] = "test"

    print("\n--- Baseline B0 on Test ('caldor', 'tamarack') ---")
    print(agg_b0_test.to_string(index=False))

    print("\n--- Baseline B1 on Test ('caldor', 'tamarack') ---")
    print(agg_b1_test.to_string(index=False))

    # Combine and save
    df_all_agg = pd.concat([agg_b0_val, agg_b1_val, agg_b0_test, agg_b1_test], ignore_index=True)
    df_all_agg.to_csv("reports/step2_smoke_baselines_eval.csv", index=False)
    print("\nSaved reports/step2_smoke_baselines_eval.csv")


if __name__ == "__main__":
    run_baseline_evaluation()
