"""Tests for Phase 3A: Trainer refusal checks on invalid data, splits, hashes, and schema versions."""

import numpy as np
import pytest
import torch
from pathlib import Path

from wf052.surrogate.dataset import CompactSurrogateDataset, SurrogateSample
from wf052.surrogate.model import WindConditionedUNet
from wf052.surrogate.standin_generator import (
    EXPECTED_CANONICAL_SIM_HASH,
    EXPECTED_TEACHER_CONFIG_HASH,
    SCHEMA_VERSION,
)
from wf052.surrogate.trainer import SurrogateTrainer, TrainConfig


def _make_dummy_sample(
    sample_id: str,
    fire_name: str,
    source: str = "teacher_v2_ensemble",
    teacher_config_hash: str = EXPECTED_TEACHER_CONFIG_HASH,
    canonical_sim_hash: str = EXPECTED_CANONICAL_SIM_HASH,
) -> SurrogateSample:
    return SurrogateSample(
        sample_id=sample_id,
        tile_id="tile_0",
        fuel_group=np.ones((256, 256), dtype=np.int64),
        slope=np.zeros((256, 256), dtype=np.float32),
        aspect_sin=np.zeros((256, 256), dtype=np.float32),
        aspect_cos=np.zeros((256, 256), dtype=np.float32),
        elevation_z=np.zeros((256, 256), dtype=np.float32),
        canopy_cover=np.zeros((256, 256), dtype=np.float32),
        ignition_mask=np.zeros((256, 256), dtype=np.float32),
        hourly_wind_uv=np.zeros((12, 2), dtype=np.float32),
        target_arrival_time=np.zeros((256, 256), dtype=np.float32),
        censor_mask=np.zeros((256, 256), dtype=np.float32),
        scaffold_tag="scaffold_a6a92f81",
        seed=42,
        source=source,
        baseline_only=False,
        teacher_config_hash=teacher_config_hash,
        canonical_sim_hash=canonical_sim_hash,
        fire_name=fire_name,
    )


class TestTrainerRefusalChecks:
    """Requirement 1c: Trainer must refuse contaminated splits, bad sources, bad hashes, and bad schema versions."""

    def test_refuse_fire_in_two_splits(self) -> None:
        model = WindConditionedUNet()
        train_samples = [_make_dummy_sample("s1", "kincade"), _make_dummy_sample("s2", "bobcat")]
        val_samples = [_make_dummy_sample("s3", "bobcat")]  # Bobcat in BOTH splits!

        with pytest.raises(ValueError) as exc_info:
            SurrogateTrainer(
                model=model,
                train_samples=train_samples,
                val_samples=val_samples,
                config=TrainConfig(epochs=1, batch_size=2),
                allow_same_fire_leak=False,
            )
        err = str(exc_info.value)
        print("\n[Refusal 1: Fire in two splits]\n", err)
        assert "Data leakage detected" in err
        assert "bobcat" in err

    def test_refuse_invalid_label_source(self) -> None:
        model = WindConditionedUNet()
        train_samples = [_make_dummy_sample("s1", "kincade", source="synthetic_random_labels")]
        val_samples = [_make_dummy_sample("s2", "bobcat")]

        with pytest.raises(ValueError) as exc_info:
            SurrogateTrainer(
                model=model,
                train_samples=train_samples,
                val_samples=val_samples,
                config=TrainConfig(epochs=1, batch_size=1),
            )
        err = str(exc_info.value)
        print("\n[Refusal 2: Invalid label source]\n", err)
        assert "is not a recognized teacher source" in err

    def test_refuse_hash_mismatch(self) -> None:
        model = WindConditionedUNet()
        train_samples = [_make_dummy_sample("s1", "kincade", teacher_config_hash="00000000badhash00000000")]
        val_samples = [_make_dummy_sample("s2", "bobcat")]

        with pytest.raises(ValueError) as exc_info:
            SurrogateTrainer(
                model=model,
                train_samples=train_samples,
                val_samples=val_samples,
                config=TrainConfig(epochs=1, batch_size=1),
            )
        err = str(exc_info.value)
        print("\n[Refusal 3: Hash mismatch]\n", err)
        assert "teacher_config_hash" in err

    def test_refuse_schema_version_mismatch(self, tmp_path: Path) -> None:
        model = WindConditionedUNet()
        # Create a mock npz shard with wrong schema_version
        bad_shard = tmp_path / "shard_bad.npz"
        np.savez(
            bad_shard,
            arrival_quantiles=np.zeros((1, 3, 256, 256), dtype=np.float16),
            burn_prob=np.zeros((1, 256, 256), dtype=np.uint8),
            censor_quantiles_packed=np.zeros((1, 3, 8192), dtype=np.uint8),
            wind_series=np.zeros((1, 12, 2), dtype=np.float32),
            crop_origins=np.zeros((1, 2), dtype=np.int32),
            ignition_cells=np.zeros((1, 2), dtype=np.int32),
            seeds=np.zeros((1,), dtype=np.int64),
            fire_ids=np.array(["kincade"]),
            tile_ids=np.array(["tile_0"]),
            sample_ids=np.array(["kincade_s1"]),
            schema_version=np.array(["legacy_1a"]),  # Wrong schema!
            git_commit=np.array(["deadbeef"]),
            teacher_config_hash=np.array([EXPECTED_TEACHER_CONFIG_HASH]),
            canonical_sim_hash=np.array([EXPECTED_CANONICAL_SIM_HASH]),
        )

        with pytest.raises(ValueError) as exc_info:
            CompactSurrogateDataset(shard_paths=[bad_shard], expected_schema_version=SCHEMA_VERSION)
        err = str(exc_info.value)
        print("\n[Refusal 4: Schema version mismatch]\n", err)
        assert "schema_version" in err
        assert "legacy_1a" in err
