"""Tests for Phase 2F: Normalized MultiTaskLoss, per-quantile censoring masks, and loss balance."""

import pytest
import torch
import torch.nn.functional as F

from wf052.surrogate.losses import PinballLoss, QuantileOrderingLoss, SurrogateMultiTaskLoss
from wf052.surrogate.model import WindConditionedUNet


class TestSurrogateLossPhase2F:
    """Verify target normalization, per-quantile censoring masks, and loss term balance."""

    def test_target_normalization_and_per_quantile_masking(self) -> None:
        loss_fn = SurrogateMultiTaskLoss(horizon_scale=720.0, lambda_order=1.0, lambda_bce=0.5)

        # Batch of shape (1, 3, 2, 2)
        # Quantiles in minutes [0, 720]
        # Let cell (0, 0): q10=60, q50=120, q90=180
        # Let cell (0, 1): q10=100 (finite, c10=0), but q50 is CENSORED (c50=1), q90 is CENSORED (c90=1)
        # Let cell (1, 0): q10=200, q50=300 (finite, c50=0), but q90 is CENSORED (c90=1)
        # Let cell (1, 1): completely censored (c10=1, c50=1, c90=1)

        pred_quantiles = torch.tensor([[[[60.0, 100.0], [200.0, 720.0]],
                                        [[120.0, 720.0], [300.0, 720.0]],
                                        [[180.0, 720.0], [720.0, 720.0]]]])  # (1, 3, 2, 2)
        pred_logits = torch.zeros((1, 1, 2, 2))

        target_quantiles = torch.tensor([[[[60.0, 100.0], [200.0, 720.0]],
                                          [[120.0, 720.0], [300.0, 720.0]],
                                          [[180.0, 720.0], [720.0, 720.0]]]])

        # Censor masks: 1 = censored, 0 = finite/burned
        censor_quantiles = torch.tensor([[[[0.0, 0.0], [0.0, 1.0]],
                                          [[0.0, 1.0], [0.0, 1.0]],
                                          [[0.0, 1.0], [1.0, 1.0]]]])

        losses = loss_fn(
            pred_quantiles=pred_quantiles,
            pred_burn_logits=pred_logits,
            target_quantiles=target_quantiles,
            censor_quantiles=censor_quantiles,
        )

        # Cell (0, 1) has c10=0, so it MUST contribute to loss_q10.
        # Since pred == target, pinball loss for exactly matching cells should be 0.
        assert torch.isclose(losses.loss_q10, torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(losses.loss_q50, torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(losses.loss_q90, torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(losses.ordering_loss, torch.tensor(0.0), atol=1e-5)

    def test_q10_supervision_on_q50_censored_cells(self) -> None:
        """Verify that cell where q10 is finite but q50 is censored receives nonzero gradient on q10."""
        loss_fn = SurrogateMultiTaskLoss(horizon_scale=720.0)

        # Single pixel with q10 finite (target 72.0 min, normalized 0.1), q50 censored
        pred = torch.tensor([[[[144.0]], [[720.0]], [[720.0]]]], requires_grad=True)  # (1, 3, 1, 1)
        target = torch.tensor([[[[72.0]], [[720.0]], [[720.0]]]])
        censor = torch.tensor([[[[0.0]], [[1.0]], [[1.0]]]])  # q10 uncensored, q50 and q90 censored

        pred_logits = torch.zeros((1, 1, 1, 1))

        losses = loss_fn(
            pred_quantiles=pred,
            pred_burn_logits=pred_logits,
            target_quantiles=target,
            censor_quantiles=censor,
        )

        # Pinball q10 should be non-zero because pred=144 != target=72
        # diff = target/720 - pred/720 = 0.1 - 0.2 = -0.1 (overprediction)
        # loss = (1 - tau) * (-diff) = 0.9 * 0.1 = 0.09
        assert torch.isclose(losses.loss_q10, torch.tensor(0.09), atol=1e-4)

        # Backward pass must propagate gradient to pred[:, 0]
        losses.total_loss.backward()
        assert pred.grad is not None
        assert pred.grad[0, 0, 0, 0].item() != 0.0
        # Whereas pred[:, 1] and pred[:, 2] have no uncensored target and no ordering violation
        assert pred.grad[0, 1, 0, 0].item() == 0.0
        assert pred.grad[0, 2, 0, 0].item() == 0.0

    def test_loss_components_balanced_within_factor_10_at_init(self) -> None:
        """Verify pinball, ordering, and BCE loss are within a factor of 10 at initialization."""
        torch.manual_seed(42000)
        model = WindConditionedUNet(in_spatial_channels=13, base_channels=16)
        loss_fn = SurrogateMultiTaskLoss(horizon_scale=720.0, lambda_order=1.0, lambda_bce=0.5)

        # Random batch of inputs
        b, h, w = 2, 64, 64
        spatial = torch.randn(b, 13, h, w)
        wind = torch.randn(b, 12, 2)

        out = model(spatial, wind)

        target_quantiles = torch.rand(b, 3, h, w) * 720.0
        censor_quantiles = (torch.rand(b, 3, h, w) > 0.5).float()
        # Guarantee some uncensored cells
        censor_quantiles[:, :, :10, :10] = 0.0

        losses = loss_fn(
            pred_quantiles=out.quantiles,
            pred_burn_logits=out.burn_logits,
            target_quantiles=target_quantiles,
            censor_quantiles=censor_quantiles,
        )

        pinball = losses.pinball_loss.item()
        order_term = (loss_fn.lambda_order * losses.ordering_loss).item()
        bce_term = (loss_fn.lambda_bce * losses.bce_loss).item()

        terms = [t for t in [pinball, order_term, bce_term] if t > 0.0]
        max_term = max(terms)
        min_term = min(terms)

        ratio = max_term / min_term
        assert ratio <= 10.0, f"Loss components out of balance: max={max_term}, min={min_term}, ratio={ratio}"

    def test_backward_compatibility_fallback(self) -> None:
        """Verify fallback when only target_arrival and censor_mask are provided."""
        loss_fn = SurrogateMultiTaskLoss(horizon_scale=720.0)

        pred_q = torch.full((1, 3, 4, 4), 360.0)
        pred_logits = torch.zeros((1, 1, 4, 4))
        target_arrival = torch.full((1, 4, 4), 360.0)
        censor_mask = torch.zeros((1, 4, 4))

        losses = loss_fn(
            pred_quantiles=pred_q,
            pred_burn_logits=pred_logits,
            target_arrival=target_arrival,
            censor_mask=censor_mask,
        )

        assert torch.isclose(losses.pinball_loss, torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(losses.ordering_loss, torch.tensor(0.0), atol=1e-5)
        assert losses.total_loss > 0.0  # BCE loss on logits=0 vs target=1 is ~0.693
