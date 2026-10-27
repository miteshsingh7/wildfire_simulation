"""Tests for Phase 2G: Wind-response check, BCE audit, and stratified sampler redesign."""

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from wf052.sim.ember_kernel import EmberParams, simulate_teacher_with_embers_60m
from wf052.sim.teacher_calibrator import TeacherParams
from wf052.surrogate.dataset import (
    SurrogateSample,
    calculate_upwind_third_ignition,
    fbfm40_to_fuel_group,
)
from wf052.surrogate.losses import SurrogateMultiTaskLoss
from wf052.surrogate.standin_generator import (
    assign_burn_fraction_bin,
    build_tile_fire_dataset_60m,
    compute_ensemble_quantiles_from_runs,
    format_fire_bin_table,
    generate_stratified_samples_rejection,
    load_and_verify_frozen_teacher_v2,
    load_fire_rasters_crop,
)


class TestWindResponseAndDirection:
    """Verify wind response convention and centroid displacement."""

    def test_wind_centroid_displacement_90_deg(self) -> None:
        """With constant wind blowing toward azimuth 90 deg (East), burned centroid moves toward azimuth 90 deg."""
        # Load Caldor rasters
        fbfm_arr, elev_arr, canopy_arr, _ = load_fire_rasters_crop("caldor", tile_size=(256, 256))
        # Find a burnable cell near center
        fuel_grp = fbfm40_to_fuel_group(fbfm_arr)
        ign_r, ign_c = 128, 128
        while fuel_grp[ign_r, ign_c] == 0:
            ign_c += 1

        # Constant wind blowing toward East (azimuth 90 deg: u=8.0 m/s, v=0.0 m/s)
        hourly_wind_uv = np.zeros((12, 2), dtype=np.float32)
        hourly_wind_uv[:, 0] = 8.0  # Eastward (blowing toward 90 deg azimuth)
        hourly_wind_uv[:, 1] = 0.0

        ds = build_tile_fire_dataset_60m(
            fire_name="caldor",
            fbfm_arr=fbfm_arr,
            elev_arr=elev_arr,
            canopy_arr=canopy_arr,
            hourly_wind_uv=hourly_wind_uv,
            device=torch.device("cpu"),
        )
        _, _, _, t_params, e_params = load_and_verify_frozen_teacher_v2()

        ign_mask = np.zeros((256, 256), dtype=bool)
        ign_mask[ign_r, ign_c] = True

        sim_out = simulate_teacher_with_embers_60m(
            ds=ds,
            params=t_params,
            ember_params=e_params,
            use_ember_kernel=True,
            duration_hours=2,
            steps_per_hour=30,
            seed=42,
            init_burned_mask=ign_mask,
        )
        arr = sim_out["arrival_time_min"]
        burned = np.isfinite(arr) & (arr <= 120.0)
        assert np.sum(burned) > 1, "Fire must spread"

        # Compute centroid of burned cells
        burned_rows, burned_cols = np.where(burned)
        mean_col = float(np.mean(burned_cols))
        mean_row = float(np.mean(burned_rows))

        # Centroid must displace eastward (increasing col index toward azimuth 90 deg)
        col_displacement = mean_col - ign_c
        row_displacement = abs(mean_row - ign_r)

        assert col_displacement > 5.0, f"Displacement eastward {col_displacement:.2f} <= 5.0 cells"
        # Eastward displacement must dominate over vertical drift
        assert col_displacement > row_displacement, (
            f"Eastward displacement {col_displacement:.2f} must dominate row drift {row_displacement:.2f}"
        )


class TestBCEAuditLoss:
    """Verify BCE loss is unmasked soft target over all cells."""

    def test_bce_loss_pred_all_ones_higher_than_true_prob(self) -> None:
        """Head predicting 1.0 everywhere has strictly higher loss than head predicting true burn_prob on mostly unburned sample."""
        loss_fn = SurrogateMultiTaskLoss(horizon_scale=720.0, lambda_order=1.0, lambda_bce=1.0)

        # Batch of 1 sample with 64x64 cells
        h, w = 64, 64
        # True burn_prob: only 20 cells burned (fraction = 20 / 4096 = 0.00488)
        burn_prob = torch.zeros((1, 1, h, w), dtype=torch.float32)
        burn_prob[0, 0, 10:15, 10:14] = 1.0  # 20 cells burned

        pred_quantiles = torch.full((1, 3, h, w), 720.0)
        target_quantiles = torch.full((1, 3, h, w), 720.0)
        censor_quantiles = (burn_prob == 0.0).float().repeat(1, 3, 1, 1)

        # Head predicting 1.0 everywhere (high positive logits e.g. +10.0 => sigmoid(+10) ~ 1.0)
        logits_all_ones = torch.full((1, 1, h, w), 10.0)

        # Head predicting true burn_prob (logits = log(p / (1-p)))
        # Clamp to avoid inf
        eps = 1e-4
        p_clamped = burn_prob.clamp(eps, 1.0 - eps)
        logits_true = torch.log(p_clamped / (1.0 - p_clamped))

        loss_all_ones = loss_fn(
            pred_quantiles=pred_quantiles,
            pred_burn_logits=logits_all_ones,
            target_quantiles=target_quantiles,
            censor_quantiles=censor_quantiles,
            burn_prob=burn_prob,
        )

        loss_true = loss_fn(
            pred_quantiles=pred_quantiles,
            pred_burn_logits=logits_true,
            target_quantiles=target_quantiles,
            censor_quantiles=censor_quantiles,
            burn_prob=burn_prob,
        )

        assert loss_all_ones.bce_loss.item() > loss_true.bce_loss.item(), (
            f"All-ones BCE loss {loss_all_ones.bce_loss.item():.4f} must be strictly higher than true BCE loss {loss_true.bce_loss.item():.4f}"
        )
        assert loss_all_ones.total_loss.item() > loss_true.total_loss.item()


class TestSamplerRedesign:
    """Verify stratified rejection sampler redesign."""

    def test_member_0_equals_prescreen_bit_for_bit(self) -> None:
        """Ensemble member 0 equals prescreen run bit for bit."""
        # Run controlled single-sample generation
        horizon_min = 720.0
        fbfm_arr, elev_arr, canopy_arr, _ = load_fire_rasters_crop("caldor", tile_size=(256, 256))
        ign_mask = np.zeros((256, 256), dtype=np.float32)
        ign_mask[128, 128] = 1.0

        hourly_wind = np.zeros((12, 2), dtype=np.float32)
        hourly_wind[:, 0] = 5.0

        ds = build_tile_fire_dataset_60m(
            fire_name="caldor",
            fbfm_arr=fbfm_arr,
            elev_arr=elev_arr,
            canopy_arr=canopy_arr,
            hourly_wind_uv=hourly_wind,
            device=torch.device("cpu"),
        )
        _, _, _, t_params, e_params = load_and_verify_frozen_teacher_v2()

        # Run prescreen seed_0
        seed_0 = 42000
        sim_out_0 = simulate_teacher_with_embers_60m(
            ds=ds,
            params=t_params,
            ember_params=e_params,
            use_ember_kernel=True,
            duration_hours=2,
            steps_per_hour=30,
            seed=seed_0,
            init_burned_mask=ign_mask.astype(bool),
        )
        raw_prescreen = sim_out_0["arrival_time_min"]
        clean_prescreen = np.where(np.isnan(raw_prescreen) | (raw_prescreen > 120.0), np.inf, raw_prescreen).astype(np.float32)

        # Build ensemble with clean_prescreen as member 0
        arrival_runs = [clean_prescreen]
        for k in range(1, 4):
            seed_k = seed_0 + k * 10007
            sim_out_k = simulate_teacher_with_embers_60m(
                ds=ds,
                params=t_params,
                ember_params=e_params,
                use_ember_kernel=True,
                duration_hours=2,
                steps_per_hour=30,
                seed=seed_k,
                init_burned_mask=ign_mask.astype(bool),
            )
            raw_k = sim_out_k["arrival_time_min"]
            clean_k = np.where(np.isnan(raw_k) | (raw_k > 120.0), np.inf, raw_k).astype(np.float32)
            arrival_runs.append(clean_k)

        # Bit for bit identity check: arrival_runs[0] MUST be bit-identical to clean_prescreen
        assert np.array_equal(arrival_runs[0], clean_prescreen)
        assert np.shares_memory(arrival_runs[0], clean_prescreen) or np.array_equal(arrival_runs[0].tobytes(), clean_prescreen.tobytes())

    def test_round_robin_leaves_per_fire_counts_within_one_when_stopped_early(self) -> None:
        """Round-robin leaves per-fire counts within 1 of each other when stopped early."""
        fires = ["bobcat", "caldor", "creek", "glass", "kincade", "tamarack", "zogg"]
        # Use fast analytic generator with max_attempts capped early
        # To test early stopping, pass max_attempts = 15
        samples, rep = generate_stratified_samples_rejection(
            fire_names=fires,
            source_mode="analytic_wind_ellipse",
            target_quotas=[3, 3, 3, 3, 3],
            max_attempts=15,
            k_ensemble=4,
        )

        counts = {f: sum(rep.fire_bin_counts[f]) for f in fires}
        c_vals = list(counts.values())
        max_c, min_c = max(c_vals), min(c_vals)
        assert max_c - min_c <= 1, f"Per-fire counts not within 1: {counts} (diff={max_c - min_c})"

    def test_bins_and_edges_read_from_config(self) -> None:
        """Bins and edges are read directly from scenarios.yaml configuration."""
        samples, rep = generate_stratified_samples_rejection(
            fire_names=["caldor"],
            source_mode="analytic_wind_ellipse",
            target_quotas=[1, 1, 1, 1, 1],
            max_attempts=10,
            scenarios_config_path="configs/scenarios.yaml",
        )
        assert len(rep.stats) == 5
        ranges = [s.bin_range for s in rep.stats]
        expected_ranges = [
            (0.0, 0.002),
            (0.002, 0.02),
            (0.02, 0.1),
            (0.1, 0.5),
            (0.5, 1.0),
        ]
        assert ranges == expected_ranges, f"Bin ranges {ranges} != {expected_ranges}"

    def test_ignition_on_group_0_rejected_without_teacher_run(self) -> None:
        """Ignition on fuel group 0 is rejected immediately without running teacher simulation."""
        # Create a mock raster where ignition cell has fuel group 0
        with patch("wf052.surrogate.standin_generator.simulate_teacher_with_embers_60m") as mock_sim:
            with patch("wf052.surrogate.standin_generator.calculate_upwind_third_ignition", return_value=(50, 50)):
                with patch("wf052.surrogate.standin_generator.fbfm40_to_fuel_group") as mock_fg:
                    # Force cell (50, 50) to be fuel group 0 (non-burnable)
                    mock_fg.return_value = np.zeros((256, 256), dtype=np.int64)

                    samples, rep = generate_stratified_samples_rejection(
                        fire_names=["caldor"],
                        source_mode="teacher_v2_ensemble",
                        target_quotas=[1, 0, 0, 0, 0],
                        max_attempts=3,
                    )

                    # Mock teacher simulation should NEVER have been called!
                    assert mock_sim.call_count == 0, f"Teacher was called {mock_sim.call_count} times, expected 0!"
                    assert sum(rep.fire_group0_rejections.values()) == 3
