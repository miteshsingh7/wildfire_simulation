"""Phase 2G Step 2: BCE Audit on dry-run sample."""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from wf052.surrogate.dataset import CompactSurrogateDataset, collate_surrogate_batch
from wf052.surrogate.model import WindConditionedUNet
from wf052.surrogate.losses import SurrogateMultiTaskLoss


def main():
    shard_p = Path("out/phase2e_dryrun/shard_dryrun.npz")
    ds = CompactSurrogateDataset([shard_p])
    batch = collate_surrogate_batch([ds[0]])

    # Target burn_prob decoded from uint8 [0, 1]
    bp = batch.burn_prob  # (1, 256, 256)
    bp_target = bp.unsqueeze(1)  # (1, 1, 256, 256)
    
    # Old binary target based on q50 uncensored mask
    c50 = batch.censor_quantiles[:, 1, :, :]  # (1, 256, 256)
    old_target = (1.0 - c50).clamp(0.0, 1.0).unsqueeze(1)  # (1, 1, 256, 256)

    mean_all = float(torch.mean(bp))
    old_mask = (c50 == 0)
    mean_under_old_mask = float(torch.mean(bp[old_mask])) if torch.sum(old_mask) > 0 else 0.0

    print("=" * 80)
    print("STEP 2: BCE TARGET AUDIT ON DRY-RUN SAMPLE 0")
    print("=" * 80)
    print(f"Mean target over ALL cells:               {mean_all:.6f}")
    print(f"Mean target under OLD mask (c50 == 0):     {mean_under_old_mask:.6f}")
    print(f"Count of uncensored cells (c50 == 0):     {int(torch.sum(old_mask))} / {256*256}")

    # Constant prediction logits
    # logit(p) = ln(p / (1 - p))
    eps = 1e-7
    logit_p1 = float(np.log(mean_all / max(eps, 1.0 - mean_all)))
    logit_p2 = float(np.log(mean_under_old_mask / max(eps, 1.0 - mean_under_old_mask)))

    pred_const_p1 = torch.full_like(bp_target, logit_p1)
    pred_const_p2 = torch.full_like(bp_target, logit_p2)

    # 1. Loss over ALL cells with soft target burn_prob
    loss_all_pred_p1 = float(F.binary_cross_entropy_with_logits(pred_const_p1, bp_target).item())
    loss_all_pred_p2 = float(F.binary_cross_entropy_with_logits(pred_const_p2, bp_target).item())

    # 2. Loss under OLD binary target
    loss_old_pred_p1 = float(F.binary_cross_entropy_with_logits(pred_const_p1, old_target).item())
    loss_old_pred_p2 = float(F.binary_cross_entropy_with_logits(pred_const_p2, old_target).item())

    # 3. Loss if evaluated only on old mask (masked BCE)
    loss_masked_pred_p1 = float(F.binary_cross_entropy_with_logits(pred_const_p1[old_mask.unsqueeze(1)], bp_target[old_mask.unsqueeze(1)]).item())
    loss_masked_pred_p2 = float(F.binary_cross_entropy_with_logits(pred_const_p2[old_mask.unsqueeze(1)], bp_target[old_mask.unsqueeze(1)]).item())

    print("\nLoss under soft target burn_prob (over ALL cells of the tile):")
    print(f"  Constant prediction p = {mean_all:.6f} (logit={logit_p1:.4f}): loss = {loss_all_pred_p1:.6f}")
    print(f"  Constant prediction p = {mean_under_old_mask:.6f} (logit={logit_p2:.4f}): loss = {loss_all_pred_p2:.6f}")

    print("\nLoss under old binary target (1-c50, over ALL cells):")
    print(f"  Constant prediction p = {mean_all:.6f} (logit={logit_p1:.4f}): loss = {loss_old_pred_p1:.6f}")
    print(f"  Constant prediction p = {mean_under_old_mask:.6f} (logit={logit_p2:.4f}): loss = {loss_old_pred_p2:.6f}")

    print("\nLoss under old mask only (evaluated only on 75 cells where c50==0):")
    print(f"  Constant prediction p = {mean_all:.6f}: loss = {loss_masked_pred_p1:.6f}")
    print(f"  Constant prediction p = {mean_under_old_mask:.6f}: loss = {loss_masked_pred_p2:.6f}")

    # Now run model forward pass with loss before and after fix
    torch.manual_seed(42000)
    model = WindConditionedUNet(in_spatial_channels=13, base_channels=16)
    model.eval()

    with torch.no_grad():
        out = model(batch.spatial_features, batch.wind_series)
        
        # Current loss in losses.py (which uses old_target = mask_q[:, 1])
        loss_fn_old = SurrogateMultiTaskLoss(lambda_order=1.0, lambda_bce=0.5, horizon_scale=720.0)
        l_old = loss_fn_old(
            pred_quantiles=out.quantiles,
            pred_burn_logits=out.burn_logits,
            target_quantiles=batch.target_quantiles,
            censor_quantiles=batch.censor_quantiles,
        )

        # New loss: passing soft burn_prob over ALL cells
        l_bce_new = F.binary_cross_entropy_with_logits(out.burn_logits, bp_target)
        l_total_new = l_old.pinball_loss + loss_fn_old.lambda_order * l_old.ordering_loss + loss_fn_old.lambda_bce * l_bce_new

    print("\n" + "=" * 80)
    print("STEP 1 LOSS COMPONENTS (BEFORE vs AFTER BCE FIX)")
    print("=" * 80)
    print(f"Before fix (BCE with binary target 1-c50):")
    print(f"  Total Loss:    {l_old.total_loss.item():.4f}")
    print(f"  Pinball Total: {l_old.pinball_loss.item():.4f}")
    print(f"  Ordering:      {l_old.ordering_loss.item():.4f}")
    print(f"  BCE Loss:      {l_old.bce_loss.item():.4f}")
    print(f"\nAfter fix (BCE with soft burn_prob target over ALL cells):")
    print(f"  Total Loss:    {l_total_new.item():.4f}")
    print(f"  Pinball Total: {l_old.pinball_loss.item():.4f}")
    print(f"  Ordering:      {l_old.ordering_loss.item():.4f}")
    print(f"  BCE Loss:      {l_bce_new.item():.4f}")


if __name__ == "__main__":
    main()
