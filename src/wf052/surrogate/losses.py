"""Loss functions for surrogate modeling: Pinball loss, quantile ordering constraint, and burn BCE.

Covers:
- Pinball (quantile) loss for tau in [0.1, 0.5, 0.9] on uncensored cells.
- Monotonic non-crossing ordering penalty: ReLU(q10 - q50) + ReLU(q50 - q90).
- Binary cross-entropy with logits for P(burned by horizon).
"""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class LossBreakdown(NamedTuple):
    """Detailed container for individual loss components."""

    total_loss: torch.Tensor
    pinball_loss: torch.Tensor
    ordering_loss: torch.Tensor
    bce_loss: torch.Tensor
    loss_q10: torch.Tensor
    loss_q50: torch.Tensor
    loss_q90: torch.Tensor


class PinballLoss(nn.Module):
    """Pinball (quantile) loss evaluated on uncensored observations.

    L_tau(y, y_hat) = max(tau * (y - y_hat), (1 - tau) * (y_hat - y))
    """

    def __init__(self, tau: float) -> None:
        super().__init__()
        if not 0.0 < tau < 1.0:
            raise ValueError(f"Quantile tau must be in (0, 1), got {tau}")
        self.tau = tau

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute pinball loss.

        Args:
            pred: Predicted quantile tensor (B, H, W).
            target: Ground truth target arrival time tensor (B, H, W).
            mask: Optional binary mask (B, H, W) where 1 indicates valid uncensored cells.

        Returns:
            Scalar mean loss over masked elements.
        """
        diff = target - pred
        loss = torch.maximum(self.tau * diff, (self.tau - 1.0) * diff)

        if mask is not None:
            mask_sum = mask.sum()
            if mask_sum > 0:
                return (loss * mask).sum() / mask_sum
            return loss.mean() * 0.0

        return loss.mean()


class QuantileOrderingLoss(nn.Module):
    """Enforces non-crossing monotonic ordering among predicted quantiles: q10 <= q50 <= q90."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, quantiles: torch.Tensor) -> torch.Tensor:
        """Compute ordering violation penalty.

        Args:
            quantiles: Tensor of shape (B, 3, H, W) containing [q10, q50, q90].

        Returns:
            Scalar penalty penalizing q10 > q50 and q50 > q90.
        """
        q10 = quantiles[:, 0, :, :]
        q50 = quantiles[:, 1, :, :]
        q90 = quantiles[:, 2, :, :]

        violation_1 = F.relu(q10 - q50)
        violation_2 = F.relu(q50 - q90)

        return (violation_1 + violation_2).mean()


class SurrogateMultiTaskLoss(nn.Module):
    """Combined multi-task loss for surrogate training.

    Combines:
    1. Pinball loss across tau in {0.1, 0.5, 0.9} on uncensored cells (normalized by horizon_scale).
    2. Quantile ordering constraint penalty (q10 <= q50 <= q90).
    3. Binary cross-entropy with logits for horizon burn probability.
    """

    def __init__(
        self,
        quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
        lambda_order: float = 1.0,
        lambda_bce: float = 0.5,
        horizon_scale: float = 720.0,
    ) -> None:
        super().__init__()
        self.quantiles = quantiles
        self.lambda_order = lambda_order
        self.lambda_bce = lambda_bce
        self.horizon_scale = horizon_scale

        self.pinball_q10 = PinballLoss(quantiles[0])
        self.pinball_q50 = PinballLoss(quantiles[1])
        self.pinball_q90 = PinballLoss(quantiles[2])
        self.ordering_loss_fn = QuantileOrderingLoss()

    def forward(
        self,
        pred_quantiles: torch.Tensor,
        pred_burn_logits: torch.Tensor,
        target_arrival: torch.Tensor | None = None,
        censor_mask: torch.Tensor | None = None,
        target_quantiles: torch.Tensor | None = None,
        censor_quantiles: torch.Tensor | None = None,
    ) -> LossBreakdown:
        """Forward loss calculation.

        Args:
            pred_quantiles: (B, 3, H, W) -> [q10, q50, q90] in raw minutes
            pred_burn_logits: (B, 1, H, W)
            target_arrival: (B, H, W) median arrival time in raw minutes (optional fallback)
            censor_mask: (B, H, W) median censor mask {0, 1} (optional fallback)
            target_quantiles: (B, 3, H, W) arrival times in minutes per quantile [q10, q50, q90]
            censor_quantiles: (B, 3, H, W) censor masks per quantile (1=censored, 0=burned)

        Returns:
            LossBreakdown with individual and total loss scalars.
        """
        # Normalize predictions to [0, 1] scale by dividing by horizon_scale (720.0)
        pred_norm = pred_quantiles / self.horizon_scale

        if target_quantiles is not None and censor_quantiles is not None:
            # Per-quantile targets and masks
            t_norm = target_quantiles / self.horizon_scale
            mask_q = (1.0 - censor_quantiles).clamp(0.0, 1.0)

            l_q10 = self.pinball_q10(pred_norm[:, 0, :, :], t_norm[:, 0, :, :], mask_q[:, 0, :, :])
            l_q50 = self.pinball_q50(pred_norm[:, 1, :, :], t_norm[:, 1, :, :], mask_q[:, 1, :, :])
            l_q90 = self.pinball_q90(pred_norm[:, 2, :, :], t_norm[:, 2, :, :], mask_q[:, 2, :, :])

            # Target for horizon burn: burned at median (q50) by horizon
            target_burned = mask_q[:, 1, :, :].unsqueeze(1)
        else:
            if target_arrival is None or censor_mask is None:
                raise ValueError("Must provide either (target_quantiles, censor_quantiles) or (target_arrival, censor_mask)")
            t_norm = target_arrival / self.horizon_scale
            uncensored_mask = (1.0 - censor_mask).clamp(0.0, 1.0)

            l_q10 = self.pinball_q10(pred_norm[:, 0, :, :], t_norm, uncensored_mask)
            l_q50 = self.pinball_q50(pred_norm[:, 1, :, :], t_norm, uncensored_mask)
            l_q90 = self.pinball_q90(pred_norm[:, 2, :, :], t_norm, uncensored_mask)

            target_burned = uncensored_mask.unsqueeze(1)

        pinball_total = (l_q10 + l_q50 + l_q90) / 3.0

        # 2. Non-crossing quantile ordering constraint (evaluated on normalized scale)
        l_order = self.ordering_loss_fn(pred_norm)

        # 3. Burn classification loss (BCEWithLogits)
        l_bce = F.binary_cross_entropy_with_logits(pred_burn_logits, target_burned)

        # Combined total loss
        total = pinball_total + self.lambda_order * l_order + self.lambda_bce * l_bce

        return LossBreakdown(
            total_loss=total,
            pinball_loss=pinball_total,
            ordering_loss=l_order,
            bce_loss=l_bce,
            loss_q10=l_q10,
            loss_q50=l_q50,
            loss_q90=l_q90,
        )
