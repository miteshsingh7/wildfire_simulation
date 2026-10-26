"""Tests for Phase 2E pipeline hardening and contracts.

Contracts:
1. Integer quantile arithmetic vs float ceil for K in {4, 8, 10, 16, 30}.
2. Dataset spatial features channel count equals WindConditionedUNet in_channels (13).
3. Parameter count equals 4,094,852.
4. Dynamically rebuilt spatial features X are bit-identical to generator X.
5. Shard schema_version enforcement ("2e").
6. Fire-level data leakage prevention and allow_same_fire_leak override.
"""

import math
from pathlib import Path
import numpy as np
import pytest
import torch

from wf052.surrogate.dataset import (
    FIRE_SPLITS,
    SCHEMA_VERSION,
    CompactSurrogateDataset,
    SurrogateSample,
    rebuild_spatial_inputs,
)
from wf052.surrogate.model import WindConditionedUNet
from wf052.surrogate.standin_generator import (
    compute_quantile_order_index,
    load_fire_rasters_crop,
    save_shard_with_metadata,
)
from wf052.surrogate.trainer import SurrogateTrainer, TrainConfig


def test_integer_quantile_arithmetic():
    """Verify integer ceil calculation matches math.ceil(tau * K) for K in {4, 8, 10, 16, 30}."""
    test_ks = [4, 8, 10, 16, 30]
    fractions = [(1, 10, 0.10), (1, 2, 0.50), (9, 10, 0.90)]

    for k in test_ks:
        for num, den, tau in fractions:
            idx = compute_quantile_order_index(num, den, k)
            expected_rank = math.ceil(tau * k)
            expected_idx = expected_rank - 1
            assert idx == expected_idx, (
                f"Mismatch for tau={tau} ({num}/{den}), K={k}: got idx {idx}, expected {expected_idx}"
            )
            assert 0 <= idx < k, f"Index out of range for K={k}: {idx}"


def test_channel_contract_and_parameter_count():
    """Verify model in_channels == 13 and total parameters == 4,094,852."""
    model = WindConditionedUNet(in_spatial_channels=13, base_channels=32, emb_dim=64)
    total_params = sum(p.numel() for p in model.parameters())

    assert model.in_spatial_channels == 13, f"Expected 13 in_spatial_channels, got {model.in_spatial_channels}"
    assert total_params == 4_094_852, f"Expected 4,094,852 parameters, got {total_params}"


def test_bit_identical_rebuilt_inputs():
    """Verify that rebuild_spatial_inputs produces bit-identical tensor to generator tensor."""
    fire = "caldor"
    fbfm, elev, canopy, wind = load_fire_rasters_crop(
        fire_name=fire,
        tile_size=(256, 256),
        crop_center_row=150,
        crop_center_col=150,
    )

    from wf052.surrogate.dataset import compute_slope_aspect, fbfm40_to_fuel_group

    # Reconstruct as generator does
    fuel_grp = fbfm40_to_fuel_group(fbfm)
    fg_t = torch.from_numpy(fuel_grp).long()
    fg_1hot = torch.nn.functional.one_hot(fg_t, num_classes=7).permute(2, 0, 1).float()
    slope, asp_sin, asp_cos = compute_slope_aspect(elev, resolution_m=60.0)
    elev_z = ((elev - np.mean(elev)) / (np.std(elev) + 1e-5)).astype(np.float32)

    ign_cell = (128, 128)
    ign_mask = np.zeros((256, 256), dtype=np.float32)
    ign_mask[127:130, 127:130] = 1.0

    gen_x = torch.cat(
        [
            fg_1hot,
            torch.from_numpy(slope).unsqueeze(0).float(),
            torch.from_numpy(asp_sin).unsqueeze(0).float(),
            torch.from_numpy(asp_cos).unsqueeze(0).float(),
            torch.from_numpy(elev_z).unsqueeze(0).float(),
            torch.from_numpy(canopy).unsqueeze(0).float(),
            torch.from_numpy(ign_mask).unsqueeze(0).float(),
        ],
        dim=0,
    )

    # Rebuild using rebuild_spatial_inputs
    crop_origin = (150, 150)
    rebuilt_x = rebuild_spatial_inputs(
        fire_id=fire,
        crop_origin=crop_origin,
        ignition_cell=ign_cell,
    )

    assert gen_x.shape == (13, 256, 256)
    assert rebuilt_x.shape == (13, 256, 256)
    max_diff = torch.max(torch.abs(gen_x - rebuilt_x)).item()
    assert max_diff == 0.0, f"Rebuilt tensor differs from generator tensor by {max_diff}"


def test_schema_version_rejection(tmp_path: Path):
    """Verify trainer rejects shards with a schema_version different from SCHEMA_VERSION ('2e')."""
    # Create a dummy shard with legacy or wrong schema version
    bad_shard_p = tmp_path / "shard_bad.npz"
    np.savez_compressed(
        bad_shard_p,
        schema_version=np.array(["wrong_v1"]),
        sample_ids=np.array(["sample_001"]),
        fire_ids=np.array(["bobcat"]),
        tile_ids=np.array(["tile_001"]),
        crop_origins=np.zeros((1, 2), dtype=np.int32),
        ignition_cells=np.zeros((1, 2), dtype=np.int32),
        arrival_quantiles=np.zeros((1, 3, 256, 256), dtype=np.float16),
        burn_prob=np.zeros((1, 256, 256), dtype=np.uint8),
        censor_quantiles_packed=np.zeros((1, 3, 8192), dtype=np.uint8),
        wind_series=np.zeros((1, 12, 2), dtype=np.float32),
        seeds=np.array([42], dtype=np.int32),
        burned_fractions=np.zeros((1,), dtype=np.float32),
        teacher_config_hash=np.array(["a6a92f81f7a9c72b5899d5d3173f86943a7f98648c4f210dcc5c646d2b4e85a5"]),
        canonical_sim_hash=np.array(["61c9baec36ce945a464e3d91a62e228841a3a7900e450a446f9a1e1e4a56eed3"]),
    )

    with pytest.raises(ValueError, match="schema_version"):
        SurrogateTrainer.validate_shard(bad_shard_p, expected_schema_version="2e")


def test_fire_split_leakage_rejection():
    """Verify trainer refuses split with leaking fire unless allow_same_fire_leak=True."""
    model = WindConditionedUNet(in_spatial_channels=13, base_channels=32, emb_dim=64)

    def make_sample(fire_name: str, s_id: str) -> SurrogateSample:
        return SurrogateSample(
            fire_name=fire_name,
            sample_id=s_id,
            fuel_group=np.zeros((256, 256), dtype=np.int64),
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
            source="teacher_v2_ensemble",
            baseline_only=False,
            teacher_config_hash="a6a92f81f7a9c72b5899d5d3173f86943a7f98648c4f210dcc5c646d2b4e85a5",
            canonical_sim_hash="61c9baec36ce945a464e3d91a62e228841a3a7900e450a446f9a1e1e4a56eed3",
            fire_id=fire_name,
        )

    # Contaminated split: bobcat in train AND bobcat in val
    train_leak = [make_sample("bobcat", "s1"), make_sample("kincade", "s2")]
    val_leak = [make_sample("bobcat", "s3")]

    # Should raise ValueError by default
    with pytest.raises(ValueError, match="Data leakage detected"):
        SurrogateTrainer(
            model=model,
            train_samples=train_leak,
            val_samples=val_leak,
            config=TrainConfig(epochs=1, batch_size=2),
            allow_same_fire_leak=False,
        )

    # Should succeed with warning when allow_same_fire_leak=True
    trainer = SurrogateTrainer(
        model=model,
        train_samples=train_leak,
        val_samples=val_leak,
        config=TrainConfig(epochs=1, batch_size=2),
        allow_same_fire_leak=True,
    )
    assert trainer.allow_same_fire_leak is True
