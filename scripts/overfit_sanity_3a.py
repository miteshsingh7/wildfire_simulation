#!/usr/bin/env python3
"""Phase 3A Step 1: Training sanity on smoke shards (Overfit & Wind Probe).

Executes:
1a. Overfit test: 8 samples from the train fires (kincade, glass, zogg, creek),
    batch 4, fixed seed, 300 optimizer steps, MPS.
    Reports at steps 0, 50, 100, 200, 300: total loss, pinball q10, q50, q90, ordering, BCE.
    Counts ordering violations (q10 > q50, q50 > q90) on finite cells at step 300.
    Checks if step 300 total loss is at least 5x lower than step 0.
1b. Wind probe on overfit model, 4 samples:
    Mean absolute change in burn_prob and in q50 (minutes, on cells with target burn_prob >= 0.5)
    when (i) wind series replaced by another sample's wind, (ii) wind is zeroed.
"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from wf052.surrogate.dataset import CompactSurrogateDataset, collate_surrogate_batch
from wf052.surrogate.losses import SurrogateMultiTaskLoss
from wf052.surrogate.model import WindConditionedUNet


def set_seed(seed: int = 42000) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def run_sanity_tests() -> None:
    print("=" * 70)
    print("PHASE 3A STEP 1: TRAINING SANITY ON SMOKE SHARDS")
    print("=" * 70)

    shard_dir = Path("data/shards_smoke_2i")
    print(f"Loading smoke shards from: {shard_dir.resolve()}")

    ds = CompactSurrogateDataset(shard_dir)
    print(f"Total samples in dataset: {len(ds)}")

    train_fires = {"kincade", "glass", "zogg", "creek"}
    # Pick 2 samples from each of the 4 train fires
    selected_indices: list[int] = []
    counts_per_fire: dict[str, int] = {f: 0 for f in train_fires}
    for idx, f in enumerate(ds.fire_ids):
        if f in train_fires and counts_per_fire[f] < 2:
            selected_indices.append(idx)
            counts_per_fire[f] += 1
        if len(selected_indices) == 8:
            break

    print(f"Selected 8 samples: indices={selected_indices}")
    print(f"Fires: {[ds.fire_ids[i] for i in selected_indices]}")
    print(f"Sample IDs: {[ds.sample_ids[i] for i in selected_indices]}")

    subset = Subset(ds, selected_indices)

    device = torch.device(
        "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Set seed
    set_seed(42000)

    model = WindConditionedUNet().to(device)
    loss_fn = SurrogateMultiTaskLoss(
        quantiles=(0.1, 0.5, 0.9),
        lambda_order=1.0,
        lambda_bce=0.5,
        horizon_scale=720.0,
    )
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Dataloader: batch size 4, cycle through the 8 samples
    loader = DataLoader(
        subset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_surrogate_batch,
    )

    history = []
    step = 0
    max_steps = 300
    done = False

    print("\n--- Starting 300 Steps Overfit Training ---")

    while not done:
        for batch in loader:
            spatial = batch.spatial_features.to(device)
            wind = batch.wind_series.to(device)
            t_quantiles = batch.target_quantiles.to(device)
            c_quantiles = batch.censor_quantiles.to(device)
            burn_prob = batch.burn_prob.to(device)

            if step in [0, 50, 100, 200, 300]:
                model.eval()
                with torch.no_grad():
                    # Evaluate on all 8 samples for exact loss logging
                    eval_loader = DataLoader(subset, batch_size=8, shuffle=False, collate_fn=collate_surrogate_batch)
                    eval_batch = next(iter(eval_loader))
                    e_spatial = eval_batch.spatial_features.to(device)
                    e_wind = eval_batch.wind_series.to(device)
                    e_t_q = eval_batch.target_quantiles.to(device)
                    e_c_q = eval_batch.censor_quantiles.to(device)
                    e_bp = eval_batch.burn_prob.to(device)

                    e_out = model(e_spatial, e_wind)
                    e_lb = loss_fn(
                        pred_quantiles=e_out.quantiles,
                        pred_burn_logits=e_out.burn_logits,
                        target_quantiles=e_t_q,
                        censor_quantiles=e_c_q,
                        burn_prob=e_bp,
                    )
                    record = {
                        "step": step,
                        "total_loss": float(e_lb.total_loss.item()),
                        "pinball_total": float(e_lb.pinball_loss.item()),
                        "loss_q10": float(e_lb.loss_q10.item()),
                        "loss_q50": float(e_lb.loss_q50.item()),
                        "loss_q90": float(e_lb.loss_q90.item()),
                        "ordering_loss": float(e_lb.ordering_loss.item()),
                        "bce_loss": float(e_lb.bce_loss.item()),
                    }
                    history.append(record)
                    print(
                        f"Step {step:3d} | Total: {record['total_loss']:.5f} | "
                        f"q10: {record['loss_q10']:.5f} | q50: {record['loss_q50']:.5f} | q90: {record['loss_q90']:.5f} | "
                        f"Order: {record['ordering_loss']:.5f} | BCE: {record['bce_loss']:.5f}"
                    )
                model.train()

            if step >= max_steps:
                done = True
                break

            optimizer.zero_grad()
            out = model(spatial, wind)
            lb = loss_fn(
                pred_quantiles=out.quantiles,
                pred_burn_logits=out.burn_logits,
                target_quantiles=t_quantiles,
                censor_quantiles=c_quantiles,
                burn_prob=burn_prob,
            )
            lb.total_loss.backward()
            optimizer.step()

            step += 1

    df_hist = pd.DataFrame(history)
    df_hist.to_csv("reports/step1a_overfit_history.csv", index=False)
    print("\nSaved reports/step1a_overfit_history.csv")

    loss_0 = history[0]["total_loss"]
    loss_300 = history[-1]["total_loss"]
    ratio = loss_0 / loss_300
    print(f"\nStep 0 Total Loss:   {loss_0:.5f}")
    print(f"Step 300 Total Loss: {loss_300:.5f}")
    print(f"Loss Reduction:      {ratio:.2f}x (Criterion: >= 5.0x)")
    assert ratio >= 5.0, f"Criterion failed: ratio {ratio:.2f}x is less than 5x!"

    # Count ordering violations on finite cells across the 8 samples
    model.eval()
    with torch.no_grad():
        eval_loader = DataLoader(subset, batch_size=8, shuffle=False, collate_fn=collate_surrogate_batch)
        eval_batch = next(iter(eval_loader))
        e_spatial = eval_batch.spatial_features.to(device)
        e_wind = eval_batch.wind_series.to(device)
        e_t_q = eval_batch.target_quantiles.to(device)
        e_c_q = eval_batch.censor_quantiles.to(device)

        e_out = model(e_spatial, e_wind)
        q10 = e_out.quantiles[:, 0, :, :]
        q50 = e_out.quantiles[:, 1, :, :]
        q90 = e_out.quantiles[:, 2, :, :]

        # Finite target cells: uncensored in q50 (c_q == 0)
        finite_mask = (e_c_q[:, 1, :, :] == 0)
        n_finite_cells = int(finite_mask.sum().item())

        viol_10_50 = (q10 > q50) & finite_mask
        viol_50_90 = (q50 > q90) & finite_mask

        n_viol_10_50 = int(viol_10_50.sum().item())
        n_viol_50_90 = int(viol_50_90.sum().item())
        total_viol = n_viol_10_50 + n_viol_50_90

        print(f"\nFinite target cells across 8 samples: {n_finite_cells}")
        print(f"Ordering violations (q10 > q50) on finite cells: {n_viol_10_50}")
        print(f"Ordering violations (q50 > q90) on finite cells: {n_viol_50_90}")
        print(f"Total ordering violations on finite cells:        {total_viol}")

    # =========================================================================
    # STEP 1B: WIND PROBE ON OVERFIT MODEL (4 SAMPLES)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1B: WIND PROBE ON OVERFIT MODEL (4 SAMPLES)")
    print("=" * 70)

    # 4 samples: 1 from each train fire
    probe_indices = selected_indices[:4]
    probe_subset = Subset(ds, probe_indices)
    probe_loader = DataLoader(probe_subset, batch_size=4, shuffle=False, collate_fn=collate_surrogate_batch)
    p_batch = next(iter(probe_loader))

    p_spatial = p_batch.spatial_features.to(device)
    p_wind_orig = p_batch.wind_series.to(device)
    target_bp = p_batch.burn_prob.to(device)

    # Cells with target burn_prob >= 0.5
    high_bp_mask = (target_bp >= 0.5)
    n_high_bp = int(high_bp_mask.sum().item())
    print(f"Target cells with burn_prob >= 0.5 across 4 samples: {n_high_bp}")

    with torch.no_grad():
        # Baseline output with original wind
        out_orig = model(p_spatial, p_wind_orig)
        bp_orig = torch.sigmoid(out_orig.burn_logits).squeeze(1)
        q50_orig = out_orig.quantiles[:, 1, :, :]

        mean_bp_orig = float(bp_orig[high_bp_mask].mean().item()) if n_high_bp > 0 else float(bp_orig.mean().item())
        mean_q50_orig = float(q50_orig[high_bp_mask].mean().item()) if n_high_bp > 0 else float(q50_orig.mean().item())

        print(f"Original mean burn_prob (on target bp>=0.5): {mean_bp_orig:.4f}")
        print(f"Original mean q50 (on target bp>=0.5):       {mean_q50_orig:.2f} min")

        # Probe (i): Replaced wind (cyclically roll wind by 1 sample: [1, 2, 3, 0])
        p_wind_swapped = torch.roll(p_wind_orig, shifts=1, dims=0)
        out_swapped = model(p_spatial, p_wind_swapped)
        bp_swapped = torch.sigmoid(out_swapped.burn_logits).squeeze(1)
        q50_swapped = out_swapped.quantiles[:, 1, :, :]

        delta_bp_swap = float(torch.abs(bp_swapped[high_bp_mask] - bp_orig[high_bp_mask]).mean().item())
        delta_q50_swap = float(torch.abs(q50_swapped[high_bp_mask] - q50_orig[high_bp_mask]).mean().item())
        pct_bp_swap = (delta_bp_swap / mean_bp_orig) * 100.0 if mean_bp_orig > 0 else 0.0
        pct_q50_swap = (delta_q50_swap / mean_q50_orig) * 100.0 if mean_q50_orig > 0 else 0.0

        # Probe (ii): Zeroed wind
        p_wind_zero = torch.zeros_like(p_wind_orig)
        out_zero = model(p_spatial, p_wind_zero)
        bp_zero = torch.sigmoid(out_zero.burn_logits).squeeze(1)
        q50_zero = out_zero.quantiles[:, 1, :, :]

        delta_bp_zero = float(torch.abs(bp_zero[high_bp_mask] - bp_orig[high_bp_mask]).mean().item())
        delta_q50_zero = float(torch.abs(q50_zero[high_bp_mask] - q50_orig[high_bp_mask]).mean().item())
        pct_bp_zero = (delta_bp_zero / mean_bp_orig) * 100.0 if mean_bp_orig > 0 else 0.0
        pct_q50_zero = (delta_q50_zero / mean_q50_orig) * 100.0 if mean_q50_orig > 0 else 0.0

        print(f"\nProbe (i) Wind Replaced:")
        print(f"  Mean abs change in burn_prob: {delta_bp_swap:.5f} ({pct_bp_swap:.2f}% of mean)")
        print(f"  Mean abs change in q50:       {delta_q50_swap:.3f} min ({pct_q50_swap:.2f}% of mean)")

        print(f"\nProbe (ii) Wind Zeroed:")
        print(f"  Mean abs change in burn_prob: {delta_bp_zero:.5f} ({pct_bp_zero:.2f}% of mean)")
        print(f"  Mean abs change in q50:       {delta_q50_zero:.3f} min ({pct_q50_zero:.2f}% of mean)")

        probe_records = [
            {
                "probe": "wind_replaced",
                "mean_abs_delta_burn_prob": round(delta_bp_swap, 6),
                "pct_change_burn_prob": round(pct_bp_swap, 3),
                "mean_abs_delta_q50_min": round(delta_q50_swap, 4),
                "pct_change_q50": round(pct_q50_swap, 3),
                "film_used": (pct_bp_swap >= 1.0 or pct_q50_swap >= 1.0),
            },
            {
                "probe": "wind_zeroed",
                "mean_abs_delta_burn_prob": round(delta_bp_zero, 6),
                "pct_change_burn_prob": round(pct_bp_zero, 3),
                "mean_abs_delta_q50_min": round(delta_q50_zero, 4),
                "pct_change_q50": round(pct_q50_zero, 3),
                "film_used": (pct_bp_zero >= 1.0 or pct_q50_zero >= 1.0),
            },
        ]
        df_probe = pd.DataFrame(probe_records)
        df_probe.to_csv("reports/step1b_wind_probe.csv", index=False)
        print("\nSaved reports/step1b_wind_probe.csv")


if __name__ == "__main__":
    run_sanity_tests()
