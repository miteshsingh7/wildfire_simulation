"""Unit tests for Phase 2D: Frozen Teacher v2 Loading and Multi-Seed Ensemble Labels.

Verifies:
1. Frozen teacher v2 config loading from env or default path with bit-for-bit hash assertions:
   - File SHA-256 == a6a92f81f7a9c72b5899d5d3173f86943a7f98648c4f210dcc5c646d2b4e85a5
   - Canonical simulator hash == 61c9baec36ce945a464e3d91a62e228841a3a7900e450a446f9a1e1e4a56eed3
2. Explicit rejection of teacher_v1.yaml and prohibited/held-out fires
   (camp_fire_2018, slater_and_devil_2020, walker_2019, w5_cold_springs_2020) by assertion.
3. Shard metadata writes both hashes; SurrogateTrainer and validate_shard_metadata
   reject shards/samples whose hash differs or whose label source is not the frozen teacher.
4. Multi-seed ensemble labels via simulate_teacher_with_embers_60m order statistics:
   - q10 <= q50 <= q90 wherever finite.
   - Finite/censored pattern follows strictly from burn_prob and K (burn_prob >= ceil(tau*K)/K).
   - Same seed gives bit-for-bit identical output.
   - Different seeds give different arrival fields (verifying seeds reach simulator).
   - Zero-spread scenario gives equal quantiles (all 720.0 and censored).
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from wf052.sim.ember_kernel import EmberParams
from wf052.sim.teacher_calibrator import TeacherParams
from wf052.surrogate.dataset import SurrogateSample
from wf052.surrogate.model import WindConditionedUNet
from wf052.surrogate.standin_generator import (
    EXPECTED_CANONICAL_SIM_HASH,
    EXPECTED_TEACHER_CONFIG_HASH,
    PROHIBITED_OR_HELD_OUT_FIRES,
    SEVEN_TRAINING_FIRES,
    build_tile_fire_dataset_60m,
    compute_canonical_simulator_hash,
    get_frozen_teacher_v2_config_path,
    get_teacher_config_hash,
    load_and_verify_frozen_teacher_v2,
    sample_to_metadata_dict,
    simulate_ensemble_teacher_v2,
    validate_shard_metadata,
    verify_fire_not_excluded,
)
from wf052.surrogate.trainer import SurrogateTrainer, TrainConfig


def _create_dummy_ensemble_sample(
    sample_id: str = "dummy_01",
    source: str = "teacher_v2_ensemble",
    cfg_hash: str = EXPECTED_TEACHER_CONFIG_HASH,
    sim_hash: str = EXPECTED_CANONICAL_SIM_HASH,
    baseline_only: bool = False,
    shape: tuple[int, int] = (64, 64),
) -> SurrogateSample:
    """Helper to create dummy SurrogateSample for testing."""
    h, w = shape
    arr = np.full((h, w), 100.0, dtype=np.float32)
    censor = np.zeros((h, w), dtype=np.float32)
    burn_prob = np.ones((h, w), dtype=np.float32)
    return SurrogateSample(
        fire_name="caldor",
        sample_id=sample_id,
        fuel_group=np.full((h, w), 1, dtype=np.int64),
        slope=np.zeros((h, w), dtype=np.float32),
        aspect_sin=np.zeros((h, w), dtype=np.float32),
        aspect_cos=np.ones((h, w), dtype=np.float32),
        elevation_z=np.zeros((h, w), dtype=np.float32),
        canopy_cover=np.full((h, w), 0.2, dtype=np.float32),
        ignition_mask=np.zeros((h, w), dtype=np.float32),
        hourly_wind_uv=np.full((12, 2), 6.0, dtype=np.float32),
        target_arrival_time=arr,
        censor_mask=censor,
        scaffold_tag=f"scaffold_{sim_hash[:12]}",
        seed=42,
        source=source,
        baseline_only=baseline_only,
        burn_prob=burn_prob,
        arrival_q10=arr,
        arrival_q50=arr,
        arrival_q90=arr,
        censor_mask_q10=censor,
        censor_mask_q50=censor,
        censor_mask_q90=censor,
        k_ensemble=16,
        teacher_config_hash=cfg_hash,
        canonical_sim_hash=sim_hash,
    )


class TestFrozenTeacherLoadingAndHygiene:
    """Requirement 2: Frozen teacher config loading, hash assertions, and exclusion rules."""

    def test_frozen_teacher_v2_hashes_match_expected(self) -> None:
        cfg, file_hash, sim_hash, t_params, e_params = load_and_verify_frozen_teacher_v2()
        assert file_hash == EXPECTED_TEACHER_CONFIG_HASH
        assert sim_hash == EXPECTED_CANONICAL_SIM_HASH
        assert t_params.p_base == 0.22
        assert e_params.alpha_launch == 0.004
        assert e_params.beta_dist_m_per_ms == 110.0
        assert e_params.p_urban_wui == 0.15

    def test_frozen_teacher_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = Path("configs/teacher_v2_a6a92f81.yaml").resolve()
        monkeypatch.setenv("WF052_TEACHER_CONFIG", str(config_path))
        resolved = get_frozen_teacher_v2_config_path()
        assert resolved == config_path
        _, f_hash, s_hash, _, _ = load_and_verify_frozen_teacher_v2()
        assert f_hash == EXPECTED_TEACHER_CONFIG_HASH
        assert s_hash == EXPECTED_CANONICAL_SIM_HASH

    def test_reading_teacher_v1_raises_assertion(self) -> None:
        with pytest.raises(AssertionError, match="strictly forbidden"):
            get_teacher_config_hash("configs/teacher_v1.yaml")

        with pytest.raises(AssertionError, match="strictly forbidden"):
            load_and_verify_frozen_teacher_v2("configs/teacher_v1.yaml")

    def test_tampered_config_rejected(self, tmp_path: Path) -> None:
        valid_path = Path("configs/teacher_v2_a6a92f81.yaml")
        raw_text = valid_path.read_text(encoding="utf-8")
        tampered_text = raw_text.replace("p_base: 0.22", "p_base: 0.28")

        tampered_file = tmp_path / "teacher_v2_tampered.yaml"
        tampered_file.write_text(tampered_text, encoding="utf-8")

        with pytest.raises(AssertionError, match="SHA-256 mismatch"):
            load_and_verify_frozen_teacher_v2(tampered_file)

    def test_prohibited_and_held_out_fires_rejected_by_assertion(self) -> None:
        for fire in [
            "camp_fire_2018",
            "Camp_Fire_2018",
            "camp",
            "slater_and_devil_2020",
            "slater",
            "walker_2019",
            "walker",
            "w5_cold_springs_2020",
            "w5",
        ]:
            with pytest.raises(AssertionError, match="strictly held-out and prohibited"):
                verify_fire_not_excluded(fire)

    def test_training_fires_accepted(self) -> None:
        for fire in SEVEN_TRAINING_FIRES:
            verify_fire_not_excluded(fire)


class TestShardMetadataAndTrainerRejection:
    """Requirement 2: Shard metadata writes hashes; trainer rejects invalid hashes/sources."""

    def test_shard_metadata_contains_hashes(self) -> None:
        sample = _create_dummy_ensemble_sample()
        meta = sample_to_metadata_dict(sample, shard_id="shard_test_001")
        assert meta["teacher_config_hash"] == EXPECTED_TEACHER_CONFIG_HASH
        assert meta["canonical_sim_hash"] == EXPECTED_CANONICAL_SIM_HASH
        assert meta["source"] == "teacher_v2_ensemble"
        assert meta["k_ensemble"] == 16

    def test_validate_shard_metadata_passes_valid(self) -> None:
        df = pd.DataFrame([{
            "teacher_config_hash": EXPECTED_TEACHER_CONFIG_HASH,
            "canonical_sim_hash": EXPECTED_CANONICAL_SIM_HASH,
            "source": "teacher_v2_ensemble",
            "sample_id": "test_01",
        }])
        validate_shard_metadata(df)
        SurrogateTrainer.validate_shard(df)

    def test_validate_shard_metadata_rejects_altered_hash(self) -> None:
        df_bad_cfg = pd.DataFrame([{
            "teacher_config_hash": "bad_hash_1234567890",
            "canonical_sim_hash": EXPECTED_CANONICAL_SIM_HASH,
            "source": "teacher_v2_ensemble",
            "sample_id": "test_01",
        }])
        with pytest.raises(AssertionError, match="invalid teacher_config_hash"):
            validate_shard_metadata(df_bad_cfg)

        df_bad_sim = pd.DataFrame([{
            "teacher_config_hash": EXPECTED_TEACHER_CONFIG_HASH,
            "canonical_sim_hash": "bad_sim_hash_1234567890",
            "source": "teacher_v2_ensemble",
            "sample_id": "test_01",
        }])
        with pytest.raises(AssertionError, match="invalid canonical_sim_hash"):
            validate_shard_metadata(df_bad_sim)

    def test_validate_shard_metadata_rejects_non_teacher_source(self) -> None:
        df_bad_src = pd.DataFrame([{
            "teacher_config_hash": EXPECTED_TEACHER_CONFIG_HASH,
            "canonical_sim_hash": EXPECTED_CANONICAL_SIM_HASH,
            "source": "analytic_wind_ellipse",
            "sample_id": "test_01",
        }])
        with pytest.raises(AssertionError, match="invalid label source"):
            validate_shard_metadata(df_bad_src)

    def test_trainer_rejects_differing_hash(self) -> None:
        model = WindConditionedUNet(base_channels=16)
        bad_sample = _create_dummy_ensemble_sample(cfg_hash="mismatch_hash_abc123")
        with pytest.raises(ValueError, match="whose hash differs or whose label source is not"):
            SurrogateTrainer(
                model=model,
                train_samples=[bad_sample],
                val_samples=[],
                config=TrainConfig(epochs=1),
                device=torch.device("cpu"),
            )

    def test_trainer_rejects_baseline_only_or_ellipse(self) -> None:
        model = WindConditionedUNet(base_channels=16)
        ellipse_sample = _create_dummy_ensemble_sample(
            source="analytic_wind_ellipse", baseline_only=True
        )
        with pytest.raises(ValueError, match="baseline_only=True or source='analytic_wind_ellipse'"):
            SurrogateTrainer(
                model=model,
                train_samples=[ellipse_sample],
                val_samples=[],
                config=TrainConfig(epochs=1),
                device=torch.device("cpu"),
            )

    def test_trainer_accepts_valid_frozen_teacher_v2(self) -> None:
        model = WindConditionedUNet(base_channels=16)
        valid_sample = _create_dummy_ensemble_sample()
        trainer = SurrogateTrainer(
            model=model,
            train_samples=[valid_sample],
            val_samples=[],
            config=TrainConfig(epochs=1, batch_size=1),
            device=torch.device("cpu"),
        )
        assert trainer is not None


class TestMultiSeedEnsembleOrderStatistics:
    """Requirement 3: Multi-seed ensemble labels via simulate_teacher_with_embers_60m order statistics."""

    @pytest.fixture
    def synthetic_tile_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h, w = 64, 64
        fbfm = np.full((h, w), 101, dtype=np.int64)  # Grass model
        elev = np.zeros((h, w), dtype=np.float32)
        canopy = np.zeros((h, w), dtype=np.float32)
        hourly_uv = np.tile([8.0, 0.0], (12, 1)).astype(np.float32)
        ign_mask = np.zeros((h, w), dtype=np.float32)
        ign_mask[32, 32] = 1.0
        return fbfm, elev, canopy, hourly_uv, ign_mask

    def test_quantiles_ordering_wherever_finite(self, synthetic_tile_inputs) -> None:
        fbfm, elev, canopy, hourly_uv, ign_mask = synthetic_tile_inputs
        device = torch.device("cpu")
        ds = build_tile_fire_dataset_60m("caldor", fbfm, elev, canopy, hourly_uv, device=device)
        t_params = TeacherParams(p_base=0.22, a=0.06, c_1=0.045, c_2=0.15, p_continue=0.30)
        e_params = EmberParams(alpha_launch=0.004, beta_dist_m_per_ms=110.0, p_urban_wui=0.15)

        burn_prob, q10, q50, q90, c10, c50, c90, _ = simulate_ensemble_teacher_v2(
            ds=ds,
            t_params=t_params,
            e_params=e_params,
            ign_mask=ign_mask,
            k_seeds=4,
            scenario_seed=12345,
            horizon_min=60.0,
            steps_per_hour=10,
        )

        # 1. q10 <= q50 <= q90 wherever finite
        finite_10_50 = (c10 == 0.0) & (c50 == 0.0)
        assert np.all(q10[finite_10_50] <= q50[finite_10_50] + 1e-4)

        finite_50_90 = (c50 == 0.0) & (c90 == 0.0)
        assert np.all(q50[finite_50_90] <= q90[finite_50_90] + 1e-4)

        finite_all = (c10 == 0.0) & (c50 == 0.0) & (c90 == 0.0)
        assert np.all(q10[finite_all] <= q90[finite_all] + 1e-4)

    def test_finite_censored_pattern_matches_burn_prob_and_k(self, synthetic_tile_inputs) -> None:
        fbfm, elev, canopy, hourly_uv, ign_mask = synthetic_tile_inputs
        device = torch.device("cpu")
        ds = build_tile_fire_dataset_60m("caldor", fbfm, elev, canopy, hourly_uv, device=device)
        t_params = TeacherParams(p_base=0.22, a=0.06, c_1=0.045, c_2=0.15, p_continue=0.30)
        e_params = EmberParams(alpha_launch=0.004, beta_dist_m_per_ms=110.0, p_urban_wui=0.15)
        k_seeds = 4

        burn_prob, q10, q50, q90, c10, c50, c90, _ = simulate_ensemble_teacher_v2(
            ds=ds,
            t_params=t_params,
            e_params=e_params,
            ign_mask=ign_mask,
            k_seeds=k_seeds,
            scenario_seed=777,
            horizon_min=60.0,
            steps_per_hour=10,
        )

        # For tau in {0.1, 0.5, 0.9}:
        # element index = ceil(tau * K) - 1
        # finite iff at least (index + 1) runs burned, i.e., burn_prob >= ceil(tau*K)/K
        for tau, c_mask in [(0.1, c10), (0.5, c50), (0.9, c90)]:
            min_burn_prob = np.ceil(tau * k_seeds) / k_seeds
            expected_uncensored = burn_prob >= (min_burn_prob - 1e-5)
            actual_uncensored = c_mask < 0.5
            np.testing.assert_array_equal(actual_uncensored, expected_uncensored)

    def test_identical_seed_gives_bit_identical_output(self, synthetic_tile_inputs) -> None:
        fbfm, elev, canopy, hourly_uv, ign_mask = synthetic_tile_inputs
        device = torch.device("cpu")
        ds = build_tile_fire_dataset_60m("caldor", fbfm, elev, canopy, hourly_uv, device=device)
        t_params = TeacherParams(p_base=0.22, a=0.06, c_1=0.045, c_2=0.15, p_continue=0.30)
        e_params = EmberParams(alpha_launch=0.004, beta_dist_m_per_ms=110.0, p_urban_wui=0.15)

        run1 = simulate_ensemble_teacher_v2(
            ds=ds,
            t_params=t_params,
            e_params=e_params,
            ign_mask=ign_mask,
            k_seeds=4,
            scenario_seed=9999,
            horizon_min=30.0,
            steps_per_hour=10,
        )
        run2 = simulate_ensemble_teacher_v2(
            ds=ds,
            t_params=t_params,
            e_params=e_params,
            ign_mask=ign_mask,
            k_seeds=4,
            scenario_seed=9999,
            horizon_min=30.0,
            steps_per_hour=10,
        )

        # Verify bit-for-bit identical arrays
        np.testing.assert_array_equal(run1[0], run2[0])  # burn_prob
        np.testing.assert_array_equal(run1[1], run2[1])  # q10
        np.testing.assert_array_equal(run1[2], run2[2])  # q50
        np.testing.assert_array_equal(run1[3], run2[3])  # q90
        np.testing.assert_array_equal(run1[4], run2[4])  # c10
        np.testing.assert_array_equal(run1[5], run2[5])  # c50
        np.testing.assert_array_equal(run1[6], run2[6])  # c90

    def test_different_seeds_give_different_arrival_fields(self, synthetic_tile_inputs) -> None:
        fbfm, elev, canopy, hourly_uv, ign_mask = synthetic_tile_inputs
        device = torch.device("cpu")
        ds = build_tile_fire_dataset_60m("caldor", fbfm, elev, canopy, hourly_uv, device=device)
        t_params = TeacherParams(p_base=0.22, a=0.06, c_1=0.045, c_2=0.15, p_continue=0.30)
        e_params = EmberParams(alpha_launch=0.004, beta_dist_m_per_ms=110.0, p_urban_wui=0.15)

        run_a = simulate_ensemble_teacher_v2(
            ds=ds,
            t_params=t_params,
            e_params=e_params,
            ign_mask=ign_mask,
            k_seeds=4,
            scenario_seed=1111,
            horizon_min=60.0,
            steps_per_hour=10,
        )
        run_b = simulate_ensemble_teacher_v2(
            ds=ds,
            t_params=t_params,
            e_params=e_params,
            ign_mask=ign_mask,
            k_seeds=4,
            scenario_seed=2222,
            horizon_min=60.0,
            steps_per_hour=10,
        )

        # Different seeds must produce different simulated runs
        raw_runs_a = run_a[7]
        raw_runs_b = run_b[7]
        assert not np.array_equal(raw_runs_a[0], raw_runs_b[0])

    def test_no_spread_scenario_gives_equal_quantiles(self) -> None:
        h, w = 64, 64
        # Non-burnable fuel (code 91)
        fbfm = np.full((h, w), 91, dtype=np.int64)
        elev = np.zeros((h, w), dtype=np.float32)
        canopy = np.zeros((h, w), dtype=np.float32)
        hourly_uv = np.zeros((12, 2), dtype=np.float32)
        ign_mask = np.zeros((h, w), dtype=np.float32)
        device = torch.device("cpu")

        ds = build_tile_fire_dataset_60m("caldor", fbfm, elev, canopy, hourly_uv, device=device)
        t_params = TeacherParams(p_base=0.22, a=0.06, c_1=0.045, c_2=0.15, p_continue=0.30)
        # Embers off and p_urban_wui=0 so no urban spread
        e_params = EmberParams(alpha_launch=0.0, beta_dist_m_per_ms=110.0, p_urban_wui=0.0)

        burn_prob, q10, q50, q90, c10, c50, c90, _ = simulate_ensemble_teacher_v2(
            ds=ds,
            t_params=t_params,
            e_params=e_params,
            ign_mask=ign_mask,
            k_seeds=4,
            scenario_seed=42,
            horizon_min=60.0,
            steps_per_hour=10,
        )

        assert np.all(burn_prob == 0.0)
        assert np.all(q10 == 60.0)
        assert np.all(q50 == 60.0)
        assert np.all(q90 == 60.0)
        assert np.all(c10 == 1.0)
        assert np.all(c50 == 1.0)
        assert np.all(c90 == 1.0)
