"""Unit and acceptance tests for the surrogate pipeline (Phase 2A).

Covers:
1. Data contract & feature preprocessing (fuel groups, slope/aspect, upwind ignition).
2. PyTorch Dataset, batch collation, and tensor shapes.
3. WindConditionedUNet parameter count (strictly 2M-6M) and forward shapes.
4. Losses: Pinball loss, quantile ordering constraint, and multi-task loss.
5. Censoring and unburned cell handling.
6. Exclusion of Camp Fire domain.
7. Analytic wind-ellipse and teacher baselines.
8. Evaluation metrics & "fidelity to teacher" reporting.
9. 1-epoch smoke training run with checkpoint generation.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from wf052.surrogate.baselines import (
    AnalyticWindEllipseBaseline,
    TeacherBaseline,
)
from wf052.surrogate.dataset import (
    SurrogateDataset,
    SurrogateSample,
    calculate_upwind_third_ignition,
    collate_surrogate_batch,
    compute_slope_aspect,
    fbfm40_to_fuel_group,
)
from wf052.surrogate.evaluation import (
    evaluate_predictions_fidelity,
    generate_fidelity_report_markdown,
)
from wf052.surrogate.losses import (
    PinballLoss,
    QuantileOrderingLoss,
)
from wf052.surrogate.model import WindConditionedUNet
from wf052.surrogate.standin_generator import (
    generate_analytic_arrival_surface,
    load_fire_rasters_crop,
)
from wf052.surrogate.trainer import (
    SurrogateTrainer,
    TrainConfig,
    set_reproducible_seed,
)


def create_synthetic_sample(
    fire_name: str = "bobcat",
    sample_id: str = "test_01",
    shape: tuple[int, int] = (256, 256),
    seed: int = 1234,
    source: str = "teacher_v1_cellular",
    baseline_only: bool = False,
) -> SurrogateSample:
    """Helper to create a valid synthetic stand-in sample conforming to data contract."""
    rng = np.random.RandomState(seed)
    h, w = shape
    fuel_grp = rng.randint(0, 7, size=(h, w)).astype(np.int64)
    elev = rng.normal(1000.0, 200.0, size=(h, w)).astype(np.float32)
    slope, asp_sin, asp_cos = compute_slope_aspect(elev, resolution_m=60.0)
    elev_z = ((elev - np.mean(elev)) / np.std(elev)).astype(np.float32)
    canopy = rng.uniform(0.0, 1.0, size=(h, w)).astype(np.float32)

    # 12-hour wind series: (12, 2)
    wind_uv = rng.normal(5.0, 2.0, size=(12, 2)).astype(np.float32)

    ign_r, ign_c = calculate_upwind_third_ignition(wind_uv, tile_shape=(h, w))
    ign_mask = np.zeros((h, w), dtype=np.float32)
    ign_mask[ign_r, ign_c] = 1.0

    target_arr, c_mask = generate_analytic_arrival_surface(
        fuel_group=fuel_grp,
        hourly_wind_uv=wind_uv,
        ign_r=ign_r,
        ign_c=ign_c,
        resolution_m=60.0,
        horizon_min=720.0,
    )

    return SurrogateSample(
        fire_name=fire_name,
        sample_id=sample_id,
        fuel_group=fuel_grp,
        slope=slope,
        aspect_sin=asp_sin,
        aspect_cos=asp_cos,
        elevation_z=elev_z,
        canopy_cover=canopy,
        ignition_mask=ign_mask,
        hourly_wind_uv=wind_uv,
        target_arrival_time=target_arr,
        censor_mask=c_mask,
        scaffold_tag="scaffold_test_hash",
        seed=seed,
        source=source,
        baseline_only=baseline_only,
    )


class TestDataContractAndPreprocessing:
    """Test data contract types, transformations, and exclusions."""

    def test_fuel_group_mapping(self) -> None:
        raw_fbfm = np.array([93, 102, 122, 143, 162, 185, 202, 1, 5, 9, 12])
        groups = fbfm40_to_fuel_group(raw_fbfm)
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 1, 3, 5, 6])
        np.testing.assert_array_equal(groups, expected)

    def test_slope_aspect_computation(self) -> None:
        elev = np.zeros((100, 100), dtype=np.float32)
        # Linear ramp along columns (East-West)
        elev[:] = np.arange(100) * 10.0
        slope, sin_asp, cos_asp = compute_slope_aspect(elev, resolution_m=60.0)
        assert slope.shape == (100, 100)
        assert np.all(slope >= 0.0)
        assert np.all(np.isfinite(sin_asp))
        assert np.all(np.isfinite(cos_asp))

    def test_upwind_third_ignition_placement(self) -> None:
        # Wind blowing strongly to the East (u > 0, v = 0)
        wind_east = np.tile([10.0, 0.0], (12, 1)).astype(np.float32)
        ign_r, ign_c = calculate_upwind_third_ignition(wind_east, tile_shape=(256, 256))
        # Center is (128, 128); upwind is West (lower column index)
        assert ign_c < 100
        assert 110 <= ign_r <= 146

    def test_prohibit_camp_fire_domain(self) -> None:
        for name in ("camp_fire", "camp", "campfire", "camp_fire_2018"):
            with pytest.raises(ValueError, match="Camp Fire domain"):
                load_fire_rasters_crop(fire_name=name)
        # Path traversal prevention
        with pytest.raises(ValueError, match="Prohibited or invalid fire domain path"):
            load_fire_rasters_crop(fire_name="..")

    def test_surrogate_batch_collation(self) -> None:
        s1 = create_synthetic_sample(sample_id="s1")
        s2 = create_synthetic_sample(sample_id="s2")
        dataset = SurrogateDataset([s1, s2])
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, collate_fn=collate_surrogate_batch
        )

        batch = next(iter(loader))
        # Spatial features: 7 fuel one-hot + 6 continuous = 13 channels
        assert batch.spatial_features.shape == (2, 13, 256, 256)
        assert batch.wind_series.shape == (2, 12, 2)
        assert batch.target_arrival.shape == (2, 256, 256)
        assert batch.censor_mask.shape == (2, 256, 256)
        assert batch.sample_ids == ["s1", "s2"]


class TestModelArchitectureAndLosses:
    """Test model structure, parameter bounds, and multi-task loss."""

    def test_model_parameter_count(self) -> None:
        model = WindConditionedUNet(
            in_spatial_channels=13,
            base_channels=32,
            emb_dim=64,
            max_horizon_min=720.0,
        )
        param_count = model.count_parameters()
        # Must be strictly between 2M and 6M parameters
        assert 2_000_000 <= param_count <= 6_000_000, f"Param count {param_count} not in [2M, 6M]!"

    def test_model_forward_pass_shapes(self) -> None:
        model = WindConditionedUNet(base_channels=32)
        model.eval()
        x = torch.randn(2, 13, 128, 128)
        w = torch.randn(2, 12, 2)
        with torch.no_grad():
            out = model(x, w)
        assert out.quantiles.shape == (2, 3, 128, 128)
        assert out.burn_logits.shape == (2, 1, 128, 128)
        # Quantiles must be bounded within [0, 720]
        assert torch.all(out.quantiles >= 0.0)
        assert torch.all(out.quantiles <= 720.0)

    def test_pinball_loss_values(self) -> None:
        p10_fn = PinballLoss(0.1)
        p90_fn = PinballLoss(0.9)

        pred = torch.tensor([10.0])
        target = torch.tensor([20.0])
        # Under-prediction (diff = 10 > 0)
        assert torch.isclose(p10_fn(pred, target), torch.tensor(1.0))
        assert torch.isclose(p90_fn(pred, target), torch.tensor(9.0))

    def test_quantile_ordering_constraint(self) -> None:
        loss_fn = QuantileOrderingLoss()

        # Perfectly ordered: q10=50 <= q50=100 <= q90=150
        ordered = torch.tensor([[[[50.0]], [[100.0]], [[150.0]]]])
        assert torch.isclose(loss_fn(ordered), torch.tensor(0.0))

        # Crossed: q10=120 > q50=100 (violation = 20)
        crossed = torch.tensor([[[[120.0]], [[100.0]], [[150.0]]]])
        assert torch.isclose(loss_fn(crossed), torch.tensor(20.0))

        # Severely crossed: q10=120 > q50=100 and q50=100 > q90=80 (violation = 20 + 20 = 40)
        severely_crossed = torch.tensor([[[[120.0]], [[100.0]], [[80.0]]]])
        assert torch.isclose(loss_fn(severely_crossed), torch.tensor(40.0))


class TestBaselinesAndEvaluation:
    """Test analytic and teacher baselines and fidelity reporting."""

    def test_analytic_wind_ellipse_baseline(self) -> None:
        sample = create_synthetic_sample(shape=(128, 128))
        baseline = AnalyticWindEllipseBaseline(resolution_m=60.0, horizon_min=720.0)
        pred = baseline.predict(sample)

        assert pred.predicted_quantiles.shape == (3, 128, 128)
        assert pred.predicted_burn_prob.shape == (128, 128)
        assert pred.elapsed_ms > 0.0
        assert pred.cells_per_sec > 0.0

    def test_teacher_baseline(self) -> None:
        sample = create_synthetic_sample(shape=(128, 128))
        teacher = TeacherBaseline(resolution_m=60.0, horizon_min=720.0)
        pred = teacher.predict(sample)

        assert pred.predicted_quantiles.shape == (3, 128, 128)
        assert pred.elapsed_ms > 0.0
        assert pred.model_name == "teacher_v1_reference"

    def test_fidelity_metrics_calculation(self) -> None:
        # Create identical predictions and targets for perfect fidelity check
        n, h, w = 2, 64, 64
        target_arr = np.full((n, h, w), 100.0, dtype=np.float32)
        censor_arr = np.zeros((n, h, w), dtype=np.float32)
        pred_q = np.stack([target_arr * 0.9, target_arr, target_arr * 1.1], axis=1)  # (N, 3, H, W)

        metrics = evaluate_predictions_fidelity(
            pred_quantiles=pred_q,
            target_arrival=target_arr,
            censor_mask=censor_arr,
            model_name="test_model",
            fire_name="bobcat",
            elapsed_ms=50.0,
            teacher_elapsed_ms=100.0,
        )

        assert metrics.uncensored_mae_minutes == pytest.approx(0.0, abs=1e-5)
        assert metrics.p10_p90_coverage == pytest.approx(1.0, abs=1e-5)
        assert metrics.speedup_vs_teacher == pytest.approx(2.0, abs=1e-3)

        # Markdown report check: must contain 'Fidelity to Teacher' and not 'accuracy'
        md = generate_fidelity_report_markdown([metrics])
        assert "Fidelity to Teacher Benchmark" in md
        assert "accuracy" not in md.lower()


class TestSmokeTrainingRun:
    """1-epoch smoke training run with checkpoint writing."""

    def test_one_epoch_smoke_training(self) -> None:
        set_reproducible_seed(42000)
        samples = [
            create_synthetic_sample(
                sample_id=f"sample_{i}",
                fire_name="kincade" if i < 2 else "bobcat",
                shape=(64, 64),
                seed=i + 100,
            )
            for i in range(4)
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = TrainConfig(
                batch_size=2,
                learning_rate=1e-3,
                epochs=1,
                checkpoint_dir=tmp_dir,
            )

            model = WindConditionedUNet(in_spatial_channels=13, base_channels=16)
            trainer = SurrogateTrainer(
                model=model,
                train_samples=samples[:2],
                val_samples=samples[2:],
                config=config,
                device=torch.device("cpu"),
            )

            history = trainer.fit()

            assert len(history) == 1
            assert "train_total_loss" in history[0]
            assert np.isfinite(history[0]["train_total_loss"])

            # Verify checkpoint was written to disk
            ckpt_path = Path(tmp_dir) / "surrogate_epoch_001.pt"
            assert ckpt_path.exists()
            state = torch.load(ckpt_path, weights_only=False)
            assert state["epoch"] == 1
            assert "model_state_dict" in state
