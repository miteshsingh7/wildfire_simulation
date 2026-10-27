#!/usr/bin/env python3
"""Phase 3C Step 1: Base-Rate Initialization Audit.

Initialises the bias of the burn-probability head to:
    logit(clip(mean burn_prob over TRAIN shards, 1e-4, 0.5))
computed from train shards only, printed.
Parameter count must stay 4,094,852.
On the 2I smoke shards report step-0 BCE and Brier before vs after.
"""

import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from wf052.surrogate.dataset import CompactSurrogateDataset, collate_surrogate_batch
from wf052.surrogate.model import WindConditionedUNet


def set_seed(seed: int = 42000) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_step0(model: nn.Module, dataset: CompactSurrogateDataset, indices: list[int], device: torch.device) -> tuple[float, float]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=8,
        shuffle=False,
        collate_fn=collate_surrogate_batch,
    )
    model.eval()
    bce_sum = 0.0
    brier_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            spatial = batch.spatial_features.to(device)
            wind = batch.wind_series.to(device)
            target_bp = batch.burn_prob.to(device)
            out = model(spatial, wind)
            logits = out.burn_logits.squeeze(1)
            bce = F.binary_cross_entropy_with_logits(logits, target_bp)
            prob = torch.sigmoid(logits)
            brier = ((prob - target_bp) ** 2).mean()
            bce_sum += float(bce.item())
            brier_sum += float(brier.item())
            n_batches += 1
    return bce_sum / max(1, n_batches), brier_sum / max(1, n_batches)


def main() -> None:
    print("=" * 75)
    print("PHASE 3C STEP 1: BASE-RATE INITIALIZATION AUDIT")
    print("=" * 75)

    shards_dir = Path("data/shards_smoke_2i")
    tiles_dir = Path("data/tiles/fires")
    rep_dir = Path("reports")
    rep_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Load 2I smoke shards
    ds = CompactSurrogateDataset(shard_paths=shards_dir, fires_base_dir=tiles_dir)
    print(f"Loaded shards from: {shards_dir} (Total samples: {len(ds)})")

    train_fires = {"kincade", "glass", "zogg", "creek"}
    val_fires = {"bobcat"}
    test_fires = {"caldor", "tamarack"}

    train_indices = [i for i, f in enumerate(ds.fire_ids) if f in train_fires]
    val_indices = [i for i, f in enumerate(ds.fire_ids) if f in val_fires]
    test_indices = [i for i, f in enumerate(ds.fire_ids) if f in test_fires]
    all_indices = list(range(len(ds)))

    print(f"Sample splits: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}, All={len(all_indices)}")

    # Compute mean burn_prob over TRAIN shards ONLY
    total_bp_sum = 0.0
    total_cells = 0
    for idx in train_indices:
        bp = ds[idx]["burn_prob"].numpy()
        total_bp_sum += float(np.sum(bp))
        total_cells += bp.size

    mean_train_bp = float(total_bp_sum / max(1, total_cells))
    clipped_bp = float(np.clip(mean_train_bp, 1e-4, 0.5))
    target_init_bias = float(math.log(clipped_bp / (1.0 - clipped_bp)))

    print(f"\n--- Train Shards Burn-Probability Statistics ---")
    print(f"Mean burn_prob over TRAIN shards: {mean_train_bp:.6f}")
    print(f"Clipped burn_prob [1e-4, 0.5]:    {clipped_bp:.6f}")
    print(f"Target initialised bias logit:    {target_init_bias:.6f}")

    # Model BEFORE base-rate init
    set_seed(42000)
    model_before = WindConditionedUNet().to(device)
    params_before = sum(p.numel() for p in model_before.parameters() if p.requires_grad)
    bias_before = float(model_before.burn_logit_head[2].bias.item())
    assert params_before == 4094852, f"Expected 4,094,852 params, got {params_before}"

    tr_bce_before, tr_brier_before = evaluate_step0(model_before, ds, train_indices, device)
    val_bce_before, val_brier_before = evaluate_step0(model_before, ds, val_indices, device)
    test_bce_before, test_brier_before = evaluate_step0(model_before, ds, test_indices, device)
    all_bce_before, all_brier_before = evaluate_step0(model_before, ds, all_indices, device)

    # Model AFTER base-rate init
    set_seed(42000)
    model_after = WindConditionedUNet().to(device)
    applied_bias = model_after.init_burn_prob_bias(mean_train_bp)
    params_after = sum(p.numel() for p in model_after.parameters() if p.requires_grad)
    bias_after = float(model_after.burn_logit_head[2].bias.item())
    assert params_after == 4094852, f"Expected 4,094,852 params, got {params_after}"
    assert math.isclose(applied_bias, target_init_bias, rel_tol=1e-6)

    tr_bce_after, tr_brier_after = evaluate_step0(model_after, ds, train_indices, device)
    val_bce_after, val_brier_after = evaluate_step0(model_after, ds, val_indices, device)
    test_bce_after, test_brier_after = evaluate_step0(model_after, ds, test_indices, device)
    all_bce_after, all_brier_after = evaluate_step0(model_after, ds, all_indices, device)

    print("\n--- Model Parameter Count & Bias ---")
    print(f"Parameter count BEFORE: {params_before:,}")
    print(f"Parameter count AFTER:  {params_after:,} (must stay exactly 4,094,852)")
    print(f"Head bias BEFORE:       {bias_before:+.6f}")
    print(f"Head bias AFTER:        {bias_after:+.6f}")

    results = [
        {"split": "train", "n": len(train_indices), "metric": "BCE", "before": tr_bce_before, "after": tr_bce_after, "delta": tr_bce_after - tr_bce_before},
        {"split": "train", "n": len(train_indices), "metric": "Brier", "before": tr_brier_before, "after": tr_brier_after, "delta": tr_brier_after - tr_brier_before},
        {"split": "val", "n": len(val_indices), "metric": "BCE", "before": val_bce_before, "after": val_bce_after, "delta": val_bce_after - val_bce_before},
        {"split": "val", "n": len(val_indices), "metric": "Brier", "before": val_brier_before, "after": val_brier_after, "delta": val_brier_after - val_brier_before},
        {"split": "test", "n": len(test_indices), "metric": "BCE", "before": test_bce_before, "after": test_bce_after, "delta": test_bce_after - test_bce_before},
        {"split": "test", "n": len(test_indices), "metric": "Brier", "before": test_brier_before, "after": test_brier_after, "delta": test_brier_after - test_brier_before},
        {"split": "all", "n": len(all_indices), "metric": "BCE", "before": all_bce_before, "after": all_bce_after, "delta": all_bce_after - all_bce_before},
        {"split": "all", "n": len(all_indices), "metric": "Brier", "before": all_brier_before, "after": all_brier_after, "delta": all_brier_after - all_brier_before},
    ]
    df_results = pd.DataFrame(results)
    out_csv = rep_dir / "step1_base_rate_init.csv"
    df_results.to_csv(out_csv, index=False)

    print(f"\n--- Step-0 BCE and Brier Score Before vs After (Saved to {out_csv}) ---")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
