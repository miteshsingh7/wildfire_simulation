#!/usr/bin/env python3
"""Re-run evaluation on held-out test fires using saved dry-run checkpoint.

Without retraining, fits B0 and B1 with the updated evaluate.py,
loads checkpoints/kaggle_today_dry/best.pt, evaluates on test fires
('caldor', 'tamarack'), and outputs the corrected table.
"""

from pathlib import Path
import pandas as pd
import torch

from wf052.surrogate.dataset import CompactSurrogateDataset
from wf052.surrogate.evaluate import (
    aggregate_per_bin_and_fire_metrics,
    evaluate_baseline_on_dataset,
    evaluate_model_on_dataset,
    fit_baseline_b0,
    fit_baseline_b1,
)
from wf052.surrogate.model import WindConditionedUNet


def main() -> None:
    shards_dir = Path("data/shards_kaggle_today_dry")
    ckpt_path = Path("checkpoints/kaggle_today_dry/best.pt")
    rep_dir = Path("reports/kaggle_today_dry")
    tiles_base = Path("data/tiles/fires")

    assert shards_dir.exists(), f"Missing {shards_dir}"
    assert ckpt_path.exists(), f"Missing {ckpt_path}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Load dataset
    ds = CompactSurrogateDataset(shard_paths=shards_dir, fires_base_dir=tiles_base)

    train_fires = ["bobcat", "creek", "glass", "kincade"]
    test_fires = ["caldor", "tamarack"]

    test_indices = [i for i, f in enumerate(ds.fire_ids) if f in test_fires]
    print(f"Loaded dataset: {len(ds)} samples, {len(test_indices)} test samples ({test_fires})")

    # Fit baselines
    print("Fitting Baseline B0...")
    b0 = fit_baseline_b0(ds, train_fires=train_fires)
    print("Fitting Baseline B1 with separate quantiles...")
    b1 = fit_baseline_b1(ds, train_fires=train_fires)

    # Load model
    print(f"Loading checkpoint {ckpt_path}...")
    model = WindConditionedUNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Evaluating Surrogate (UNet)...")
    df_surrogate = evaluate_model_on_dataset(model, ds, target_indices=test_indices, device=device)
    agg_surrogate = aggregate_per_bin_and_fire_metrics(df_surrogate, fires=test_fires, model_name="Surrogate (UNet)")

    print("Evaluating Baseline B0...")
    df_b0 = evaluate_baseline_on_dataset(b0, ds, test_indices, baseline_name="B0")
    agg_b0 = aggregate_per_bin_and_fire_metrics(df_b0, fires=test_fires, model_name="B0")

    print("Evaluating Baseline B1...")
    df_b1 = evaluate_baseline_on_dataset(b1, ds, test_indices, baseline_name="B1")
    agg_b1 = aggregate_per_bin_and_fire_metrics(df_b1, fires=test_fires, model_name="B1")

    df_test_final = pd.concat([agg_surrogate, agg_b0, agg_b1], ignore_index=True)
    out_csv = rep_dir / "test_eval.csv"
    df_test_final.to_csv(out_csv, index=False)
    # Also update reports/test_eval.csv if it exists
    df_test_final.to_csv("reports/test_eval.csv", index=False)
    print(f"Saved evaluation to {out_csv} and reports/test_eval.csv")

    cols_to_show = ["model", "fire", "bin", "bin_name", "n", "flag", "brier", "iou", "mae_q10", "mae_q50", "mae_q90", "false_unburned_share"]
    print("\n" + "=" * 80)
    print("CORRECTED TEST EVALUATION TABLE")
    print("=" * 80)
    print(df_test_final[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()
