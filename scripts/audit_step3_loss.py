"""Phase 2F Step 3: Loss and censoring audit on dry-run shard."""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from wf052.surrogate.losses import LossBreakdown, PinballLoss, QuantileOrderingLoss, SurrogateMultiTaskLoss
from wf052.surrogate.dataset import CompactSurrogateDataset, collate_surrogate_batch
from wf052.surrogate.model import WindConditionedUNet


def main():
    shard_path = Path("out/phase2e_dryrun/shard_dryrun.npz")
    if not shard_path.exists():
        print(f"Error: {shard_path} not found.")
        return

    npz = np.load(shard_path, allow_pickle=True)
    # arrival_quantiles: (N, 3, 256, 256)
    arr_q = npz["arrival_quantiles"]
    # censor_quantiles_packed: (N, 3, 8192)
    c_packed = npz["censor_quantiles_packed"]
    c_unpacked = np.stack([
        np.unpackbits(c_packed[i], axis=-1).reshape(3, 256, 256)
        for i in range(len(arr_q))
    ], axis=0) # (N, 3, 256, 256)

    print("=" * 80)
    print("STEP 3b: REAL SAMPLE FROM DRY-RUN SHARD (SAMPLE 0)")
    print("=" * 80)
    sample_idx = 0
    q10 = arr_q[sample_idx, 0]
    q50 = arr_q[sample_idx, 1]
    q90 = arr_q[sample_idx, 2]

    c10 = c_unpacked[sample_idx, 0] # 1 = censored, 0 = finite/burned
    c50 = c_unpacked[sample_idx, 1]
    c90 = c_unpacked[sample_idx, 2]

    # Finite means censored == 0
    # Censored means censored == 1
    # Check cells where q10 is finite but q50 censored
    q10_finite_q50_censored = (c10 == 0) & (c50 == 1)
    count_q10_fin_q50_cens = int(np.sum(q10_finite_q50_censored))

    # Check cells where q50 is finite but q90 censored
    q50_finite_q90_censored = (c50 == 0) & (c90 == 1)
    count_q50_fin_q90_cens = int(np.sum(q50_finite_q90_censored))

    # Check cells where q10 finite but q90 censored
    q10_finite_q90_censored = (c10 == 0) & (c90 == 1)
    count_q10_fin_q90_cens = int(np.sum(q10_finite_q90_censored))

    total_cells = 256 * 256
    print(f"Total cells: {total_cells}")
    print(f"Cells with q10 finite (c10=0): {int(np.sum(c10 == 0))} ({np.sum(c10 == 0)/total_cells*100:.2f}%)")
    print(f"Cells with q50 finite (c50=0): {int(np.sum(c50 == 0))} ({np.sum(c50 == 0)/total_cells*100:.2f}%)")
    print(f"Cells with q90 finite (c90=0): {int(np.sum(c90 == 0))} ({np.sum(c90 == 0)/total_cells*100:.2f}%)")
    print(f"\nCondition 1: Cells where q10 is finite but q50 is censored: {count_q10_fin_q50_cens}")
    print(f"Condition 2: Cells where q50 is finite but q90 is censored: {count_q50_fin_q90_cens}")
    print(f"Cells where q10 is finite but q90 is censored: {count_q10_fin_q90_cens}")

    print("\n" + "=" * 80)
    print("CURRENT LOSS BEHAVIOR ON THESE CELLS:")
    print("=" * 80)
    print("In current SurrogateMultiTaskLoss:")
    print("  uncensored_mask = (1.0 - censor_mask).clamp(0.0, 1.0)")
    print("  where censor_mask is c50 (the median censor mask)!")
    print("  l_q10 = pinball_q10(pred[:, 0], target_arrival, uncensored_mask)")
    print("  l_q50 = pinball_q50(pred[:, 1], target_arrival, uncensored_mask)")
    print("  l_q90 = pinball_q90(pred[:, 2], target_arrival, uncensored_mask)")
    print(f"1. For the {count_q10_fin_q50_cens} cells where q10 is finite but q50 is censored:")
    print("   c50 == 1 -> uncensored_mask == 0. These cells are COMPLETELY MASKED OUT of l_q10!")
    print("   The model receives ZERO gradient or supervision on q10 for these cells despite having ground truth q10!")
    print(f"2. For the {count_q50_fin_q90_cens} cells where q50 is finite but q90 is censored:")
    print("   c50 == 0 -> uncensored_mask == 1. These cells are INCLUDED in l_q90, but target_arrival is q50!")
    print("   The loss evaluates pinball_q90 against q50 (the median arrival time) even though q90 never burned (censored)!")

    # Now run forward pass with current model and loss to report step 1 loss components
    print("\n" + "=" * 80)
    print("STEP 1 LOSS COMPONENTS (BEFORE FIX)")
    print("=" * 80)
    ds = CompactSurrogateDataset([shard_path])
    batch = collate_surrogate_batch([ds[0]])
    torch.manual_seed(42000)
    model = WindConditionedUNet(in_spatial_channels=13, base_channels=16)
    model.eval()

    with torch.no_grad():
        out = model(batch.spatial_features, batch.wind_series)
        old_loss_fn = SurrogateMultiTaskLoss(lambda_order=1.0, lambda_bce=0.5)
        old_losses = old_loss_fn(
            pred_quantiles=out.quantiles,
            pred_burn_logits=out.burn_logits,
            target_arrival=batch.target_arrival,
            censor_mask=batch.censor_mask,
        )

    print(f"Total Loss:    {old_losses.total_loss.item():.4f}")
    print(f"Pinball Total: {old_losses.pinball_loss.item():.4f} (q10={old_losses.loss_q10.item():.4f}, q50={old_losses.loss_q50.item():.4f}, q90={old_losses.loss_q90.item():.4f})")
    print(f"Ordering Loss: {old_losses.ordering_loss.item():.4f}")
    print(f"BCE Loss:      {old_losses.bce_loss.item():.4f}")


if __name__ == "__main__":
    main()
