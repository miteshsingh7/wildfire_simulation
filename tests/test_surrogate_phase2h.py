"""Tests for Phase 2H: Quota calculation, bit-identical prescreen, and smoke shard integrity."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from wf052.surrogate.standin_generator import (
    generate_stratified_samples_rejection,
)


class TestQuotaCalculation:
    """Verify quota calculation for both smoke mode and production 200-sample-per-fire mode."""

    def test_quota_calculation_200_samples_per_fire(self, capsys: pytest.CaptureFixture[str]) -> None:
        """With scenarios_per_fire=200, targets follow the 10/20/25/25/20 proportions: [20, 40, 50, 50, 40]."""
        # Mock simulate_teacher_with_embers_60m so no heavy GPU/MPS simulations run
        mock_arr = np.full((256, 256), np.inf, dtype=np.float32)
        mock_arr[128, 128] = 0.0

        with patch("wf052.surrogate.standin_generator.simulate_teacher_with_embers_60m") as mock_sim:
            mock_sim.return_value = {"arrival_time_min": mock_arr}

            samples, rep = generate_stratified_samples_rejection(
                fire_names=["caldor"],
                mode="full",
                scenarios_per_fire=200,
                max_attempts=1,  # Stop after 1 attempt
                source_mode="teacher_v2_ensemble",
                teacher_config_path="configs/teacher_v2_a6a92f81.yaml",
            )

            per_bin_targets = [s.target for s in rep.stats]
            print(f"\n[Test Output] Per-bin targets for 200 samples/fire: {per_bin_targets}")

            # Verify 10/20/25/25/20 calculation
            expected_targets = [20, 40, 50, 50, 40]
            assert per_bin_targets == expected_targets, (
                f"Expected per-bin targets {expected_targets}, got {per_bin_targets}"
            )
            assert sum(per_bin_targets) == 200

    def test_quota_calculation_smoke_scenarios_le_num_bins(self) -> None:
        """With scenarios_per_fire <= num_bins (e.g. 2 <= 5), each bin gets target = scenarios_per_fire."""
        mock_arr = np.full((256, 256), np.inf, dtype=np.float32)
        mock_arr[128, 128] = 0.0

        with patch("wf052.surrogate.standin_generator.simulate_teacher_with_embers_60m") as mock_sim:
            mock_sim.return_value = {"arrival_time_min": mock_arr}

            samples, rep = generate_stratified_samples_rejection(
                fire_names=["caldor"],
                mode="smoke",
                scenarios_per_fire=2,
                max_attempts=1,
                source_mode="teacher_v2_ensemble",
                teacher_config_path="configs/teacher_v2_a6a92f81.yaml",
            )

            per_bin_targets = [s.target for s in rep.stats]
            expected_targets = [2, 2, 2, 2, 2]
            assert per_bin_targets == expected_targets, (
                f"Expected per-bin targets {expected_targets}, got {per_bin_targets}"
            )
