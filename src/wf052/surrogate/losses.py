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
    1. Pinball loss across tau in {0.1, 0.5, 0.9} on uncensored cells.
    2. Quantile ordering constraint penalty.
    3. Binary cross-entropy with logits for horizon burn probability.
    """

    def __init__(
        self,
        quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
        lambda_order: float = 1.0,
        lambda_bce: float = 0.5,
    ) -> None:
        super().__init__()
        self.quantiles = quantiles
        self.lambda_order = lambda_order
        self.lambda_bce = lambda_bce

        self.pinball_q10 = PinballLoss(quantiles[0])
        self.pinball_q50 = PinballLoss(quantiles[1])
        self.pinball_q90 = PinballLoss(quantiles[2])
        self.ordering_loss_fn = QuantileOrderingLoss()

    def forward(
        self,
        pred_quantiles: torch.Tensor,
        pred_burn_logits: torch.Tensor,
        target_arrival: torch.Tensor,
        censor_mask: torch.Tensor,
    ) -> LossBreakdown:
        """Forward loss calculation.

        Args:
            pred_quantiles: (B, 3, H, W) -> [q10, q50, q90]
            pred_burn_logits: (B, 1, H, W)
            target_arrival: (B, H, W)
            censor_mask: (B, H, W) where 1 indicates unburned/censored, 0 indicates burned.

        Returns:
            LossBreakdown with individual and total loss scalars.
        """
        # Uncensored mask: 1 for burned cells, 0 for censored
        uncensored_mask = (1.0 - censor_mask).clamp(0.0, 1.0)

        # 1. Pinball losses
        l_q10 = self.pinball_q10(pred_quantiles[:, 0, :, :], target_arrival, uncensored_mask)
        l_q50 = self.pinball_q50(pred_quantiles[:, 1, :, :], target_arrival, uncensored_mask)
        l_q90 = self.pinball_q90(pred_quantiles[:, 2, :, :], target_arrival, uncensored_mask)
        pinball_total = (l_q10 + l_q50 + l_q90) / 3.0

        # 2. Non-crossing quantile ordering constraint
        l_order = self.ordering_loss_fn(pred_quantiles)

        # 3. Burn classification loss (BCEWithLogits)
        # Target: 1 if burned by horizon, 0 if censored
        target_burned = uncensored_mask.unsqueeze(1)  # (B, 1, H, W)
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
