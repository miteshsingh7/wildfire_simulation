"""Unit tests for surrogate evaluation module src/wf052/surrogate/evaluate.py."""

import math
import numpy as np
import pandas as pd
import pytest

from wf052.surrogate.evaluate import (
    BaselineB0,
    BaselineB1,
    aggregate_per_bin_metrics,
    compute_sample_metrics,
)


class TestSurrogateEvaluate:
    """Test suite for evaluation metric definitions and edge cases."""

    def test_iou_empty_returns_nan_not_zero(self) -> None:
        """When both prediction and target have zero burned cells (p < 0.5), IoU must be NaN, not 0.0."""
        h, w = 64, 64
        pred_bp = np.zeros((h, w), dtype=np.float32)
        target_bp = np.zeros((h, w), dtype=np.float32)

        pred_q = np.full((3, h, w), 720.0, dtype=np.float32)
        target_q = np.full((3, h, w), 720.0, dtype=np.float32)
        censor_q = np.ones((3, h, w), dtype=np.float32)

        m = compute_sample_metrics(
            pred_quantiles=pred_q,
            pred_burn_prob=pred_bp,
            target_quantiles=target_q,
            censor_quantiles=censor_q,
            target_burn_prob=target_bp,
            k_ensemble=4,
        )
        assert math.isnan(m["iou"]), f"Expected NaN for empty IoU, got {m['iou']}"

    def test_iou_non_empty_calculation(self) -> None:
        """IoU calculates exact intersection / union when burned cells exist."""
        h, w = 10, 10
        pred_bp = np.zeros((h, w), dtype=np.float32)
        target_bp = np.zeros((h, w), dtype=np.float32)

        # 4 cells overlap, 2 cells pred only, 2 cells target only -> intersection 4, union 8 -> IoU 0.5
        pred_bp[0:3, 0:2] = 0.8    # 6 cells (r in 0,1,2; c in 0,1)
        target_bp[1:4, 0:2] = 0.8  # 6 cells (r in 1,2,3; c in 0,1)
        # Overlap: r in 1,2; c in 0,1 -> 4 cells
        # Union: r in 0,1,2,3; c in 0,1 -> 8 cells

        pred_q = np.full((3, h, w), 100.0, dtype=np.float32)
        target_q = np.full((3, h, w), 100.0, dtype=np.float32)
        censor_q = np.zeros((3, h, w), dtype=np.float32)

        m = compute_sample_metrics(
            pred_quantiles=pred_q,
            pred_burn_prob=pred_bp,
            target_quantiles=target_q,
            censor_quantiles=censor_q,
            target_burn_prob=target_bp,
            k_ensemble=4,
        )
        assert abs(m["iou"] - 0.5) < 1e-4

    def test_brier_score(self) -> None:
        """Brier score is the exact mean squared difference across all cells."""
        h, w = 10, 10
        pred_bp = np.full((h, w), 0.6, dtype=np.float32)
        target_bp = np.full((h, w), 0.4, dtype=np.float32)

        pred_q = np.full((3, h, w), 100.0, dtype=np.float32)
        target_q = np.full((3, h, w), 100.0, dtype=np.float32)
        censor_q = np.zeros((3, h, w), dtype=np.float32)

        m = compute_sample_metrics(
            pred_quantiles=pred_q,
            pred_burn_prob=pred_bp,
            target_quantiles=target_q,
            censor_quantiles=censor_q,
            target_burn_prob=target_bp,
            k_ensemble=4,
        )
        expected_brier = (0.6 - 0.4) ** 2  # 0.04
        assert abs(m["brier"] - expected_brier) < 1e-5

    def test_arrival_mae_and_false_unburned_share(self) -> None:
        """MAE evaluated on finite cells; false unburned share measures p < 0.5 on finite cells."""
        h, w = 10, 10
        pred_bp = np.zeros((h, w), dtype=np.float32)
        target_bp = np.zeros((h, w), dtype=np.float32)

        # 4 cells burned in ground truth (prob 1.0)
        target_bp[:2, :2] = 1.0
        # Model predicts p=0.8 on 2 cells, p=0.2 on 2 cells (false unburned)
        pred_bp[0, :2] = 0.8
        pred_bp[1, :2] = 0.2

        pred_q = np.full((3, h, w), 50.0, dtype=np.float32)
        target_q = np.full((3, h, w), 30.0, dtype=np.float32)
        censor_q = np.ones((3, h, w), dtype=np.float32)
        censor_q[:, :2, :2] = 0.0  # finite uncensored

        m = compute_sample_metrics(
            pred_quantiles=pred_q,
            pred_burn_prob=pred_bp,
            target_quantiles=target_q,
            censor_quantiles=censor_q,
            target_burn_prob=target_bp,
            k_ensemble=4,
        )
        assert abs(m["mae_q50"] - 20.0) < 1e-4  # |50 - 30| = 20
        # 2 of 4 finite cells have p < 0.5 -> 0.50 share
        assert abs(m["false_unburned_share"] - 0.50) < 1e-4

    def test_ordering_violations_rate(self) -> None:
        """Ordering violations rate detects q10 > q50 or q50 > q90."""
        h, w = 10, 10
        pred_q = np.zeros((3, h, w), dtype=np.float32)
        pred_q[0] = 100.0
        pred_q[1] = 80.0   # violation: q10 (100) > q50 (80)!
        pred_q[2] = 120.0

        target_q = np.full((3, h, w), 50.0, dtype=np.float32)
        censor_q = np.zeros((3, h, w), dtype=np.float32)
        pred_bp = np.full((h, w), 0.5, dtype=np.float32)
        target_bp = np.full((h, w), 0.5, dtype=np.float32)

        m = compute_sample_metrics(
            pred_quantiles=pred_q,
            pred_burn_prob=pred_bp,
            target_quantiles=target_q,
            censor_quantiles=censor_q,
            target_burn_prob=target_bp,
            k_ensemble=4,
        )
        assert m["ordering_violation_rate"] == 1.0

    def test_aggregate_per_bin_metrics_marks_low_n(self) -> None:
        """Aggregation marks bins with n < 5 as LOW-N."""
        df_records = pd.DataFrame([
            {"bin_kmean": 0, "brier": 0.01, "iou": 0.5, "mae_q10": 10.0, "mae_q50": 15.0, "mae_q90": 20.0, "false_unburned_share": 0.1, "ordering_violation_rate": 0.0},
            {"bin_kmean": 0, "brier": 0.02, "iou": 0.6, "mae_q10": 12.0, "mae_q50": 16.0, "mae_q90": 22.0, "false_unburned_share": 0.2, "ordering_violation_rate": 0.0},
            # bin 0 has n=2 -> LOW-N
        ])
        agg = aggregate_per_bin_metrics(df_records)
        assert len(agg) == 1
        assert agg.iloc[0]["flag"] == "LOW-N"
        assert agg.iloc[0]["n"] == 2
