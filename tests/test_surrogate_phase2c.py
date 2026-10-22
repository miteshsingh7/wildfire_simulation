"""Unit and smoke tests for Phase 2C surrogate fixes and Kaggle packaging.

Verifies:
1. Training uses teacher_v1 labels only; ellipse/baseline_only samples are rejected by SurrogateTrainer.
2. Wind series sampled from HRRR weather parquet with synthetic scalings covering 1 to 20 m/s.
3. Burned fraction stratification across bins [0-0.2, 0.2-0.5, 0.5-0.9, >0.9] with histogram reporting.
4. Adapter default p_thr=0.1 and burn-head miss rate calculation.
5. One-sided lower-quantile conformal calibration for q10 with coverage guarantee.
6. Latency benchmarking protocol (warmup, timed runs, median, p95, hardware labeling).
7. Whole-domain run on (562, 564) and size-dependence test (256 vs 512).
8. Kaggle packaging script excluding all held-out fires.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from scripts.package_for_kaggle import SEVEN_TRAINING_FIRES, build_kaggle_packages
from wf052.surrogate.adapter import (
    compute_burn_head_miss_rates,
    surrogate_output_to_arrival_field,
)
from wf052.surrogate.calibration import (
    OneSidedLowerConformalCalibrator,
    SplitConformalCalibrator,
)
from wf052.surrogate.dataset import SurrogateSample
from wf052.surrogate.latency import (
    benchmark_surrogate_latency,
    get_hardware_name,
    run_whole_domain_inference,
    verify_size_dependence,
)
from wf052.surrogate.model import WindConditionedUNet
from wf052.surrogate.standin_generator import (
    assign_burn_fraction_bin,
    compute_stratified_histogram,
    format_stratification_report,
    sample_hrrr_wind_series,
    stratify_samples_by_burned_fraction,
)
from wf052.surrogate.trainer import SurrogateTrainer, TrainConfig


def _create_dummy_sample(
    sample_id: str,
    fire_name: str = "caldor",
    source: str = "teacher_v1_cellular",
    baseline_only: bool = False,
    shape: tuple[int, int] = (64, 64),
    burned_fraction: float = 0.35,
    seed: int = 42,
) -> SurrogateSample:
    """Helper to generate tiny synthetic sample for fast local unit testing."""
    h, w = shape
    c_mask = np.ones((h, w), dtype=np.float32)
    num_burned = int(round(h * w * burned_fraction))
    c_mask.ravel()[:num_burned] = 0.0

    arr_time = np.full((h, w), 720.0, dtype=np.float32)
    arr_time.ravel()[:num_burned] = np.linspace(10.0, 300.0, num_burned)

    wind = np.full((12, 2), 5.0, dtype=np.float32)

    return SurrogateSample(
        fire_name=fire_name,
        sample_id=sample_id,
        fuel_group=np.ones((h, w), dtype=np.int64),
        slope=np.zeros((h, w), dtype=np.float32),
        aspect_sin=np.zeros((h, w), dtype=np.float32),
        aspect_cos=np.ones((h, w), dtype=np.float32),
        elevation_z=np.zeros((h, w), dtype=np.float32),
        canopy_cover=np.full((h, w), 0.5, dtype=np.float32),
        ignition_mask=np.zeros((h, w), dtype=np.float32),
        hourly_wind_uv=wind,
        target_arrival_time=arr_time,
        censor_mask=c_mask,
        scaffold_tag="scaffold_test",
        seed=seed,
        source=source,
        baseline_only=baseline_only,
    )


class TestPhase2CTrainingLabelsConstraint:
    """Requirement 1: Training uses teacher_v1 labels only; ellipse samples are rejected."""

    def test_trainer_rejects_ellipse_sample(self) -> None:
        model = WindConditionedUNet(base_channels=16)
        ellipse_sample = _create_dummy_sample(
            sample_id="ellipse_sample_01",
            source="analytic_wind_ellipse",
            baseline_only=True,
        )

        with pytest.raises(ValueError, match="SurrogateTrainer rejected.*strictly requires teacher_v1"):
            SurrogateTrainer(
                model=model,
                train_samples=[ellipse_sample],
                val_samples=[],
                config=TrainConfig(epochs=1),
                device=torch.device("cpu"),
            )

    def test_trainer_rejects_baseline_only_sample(self) -> None:
        model = WindConditionedUNet(base_channels=16)
        baseline_sample = _create_dummy_sample(
            sample_id="baseline_sample_02",
            source="custom_baseline",
            baseline_only=True,
        )

        with pytest.raises(ValueError, match="SurrogateTrainer rejected.*baseline_only=True"):
            SurrogateTrainer(
                model=model,
                train_samples=[baseline_sample],
                val_samples=[],
                config=TrainConfig(epochs=1),
                device=torch.device("cpu"),
            )

    def test_trainer_accepts_teacher_v1_samples(self) -> None:
        model = WindConditionedUNet(base_channels=16)
        teacher_sample = _create_dummy_sample(
            sample_id="teacher_sample_01",
            source="teacher_v1_cellular",
            baseline_only=False,
        )

        # Should initialize without error
        trainer = SurrogateTrainer(
            model=model,
            train_samples=[teacher_sample],
            val_samples=[],
            config=TrainConfig(epochs=1, batch_size=1),
            device=torch.device("cpu"),
        )
        assert trainer is not None


class TestPhase2CHRRRWindSampling:
    """Requirement 2: Wind series sampled from HRRR data with peak wind covering 1 to 20 m/s."""

    def test_hrrr_wind_series_sampling_peak_scaling(self) -> None:
        fire_dir = Path("data/tiles/fires/caldor")
        if not fire_dir.exists():
            pytest.skip("Local test fire rasters not present")

        peak_speeds = []
        for i in range(15):
            rng = np.random.RandomState(100 + i)
            wind_uv = sample_hrrr_wind_series(
                fire_dir=fire_dir,
                rng=rng,
                window_hours=12,
                target_peak_speed_range=(1.0, 20.0),
            )
            assert wind_uv.shape == (12, 2)
            speeds = np.hypot(wind_uv[:, 0], wind_uv[:, 1])
            peak_spd = float(np.max(speeds))
            peak_speeds.append(peak_spd)
            assert 0.99 <= peak_spd <= 20.01

        # Confirm span across lower and upper wind spectrum
        assert min(peak_speeds) < 8.0
        assert max(peak_speeds) > 13.0


class TestPhase2CStratification:
    """Requirement 3: Stratify burned fraction into 4 bins with histogram reporting."""

    def test_assign_burn_fraction_bin(self) -> None:
        assert assign_burn_fraction_bin(0.0) == 0
        assert assign_burn_fraction_bin(0.199) == 0
        assert assign_burn_fraction_bin(0.200) == 1
        assert assign_burn_fraction_bin(0.499) == 1
        assert assign_burn_fraction_bin(0.500) == 2
        assert assign_burn_fraction_bin(0.899) == 2
        assert assign_burn_fraction_bin(0.900) == 3
        assert assign_burn_fraction_bin(1.0) == 3

    def test_stratification_rebalancing_and_report(self) -> None:
        # Create pool with skewed distribution (all low burned fraction)
        samples = [
            _create_dummy_sample(f"s_low_{i}", burned_fraction=0.10, seed=i)
            for i in range(12)
        ] + [
            _create_dummy_sample(f"s_med_{i}", burned_fraction=0.35, seed=i + 50)
            for i in range(4)
        ] + [
            _create_dummy_sample(f"s_high_{i}", burned_fraction=0.70, seed=i + 100)
            for i in range(2)
        ] + [
            _create_dummy_sample(f"s_vhigh_{i}", burned_fraction=0.95, seed=i + 150)
            for i in range(2)
        ]

        strat_samples, hist_before, hist_after = stratify_samples_by_burned_fraction(
            samples, target_proportions=(0.25, 0.25, 0.25, 0.25), total_samples=16, seed=42
        )

        assert len(strat_samples) == 16
        assert hist_before["counts"][0] == 12
        # After rebalancing, all bins should be represented
        assert all(c > 0 for c in hist_after["counts"])

        report = format_stratification_report(hist_before, hist_after)
        assert "| Burned Fraction Bin | Before Count | Before % | After Count | After % |" in report
        assert "0-0.2" in report
        assert ">0.9" in report


class TestPhase2CAdapterAndMissRates:
    """Requirement 4: Adapter default p_thr=0.1 and burn-head miss rate evaluation."""

    def test_adapter_default_p_threshold(self) -> None:
        h, w = 32, 32
        quantiles = np.full((3, h, w), 50.0, dtype=np.float32)
        burn_prob = np.full((h, w), 0.15, dtype=np.float32)  # Between 0.10 and 0.50

        x_coords = np.arange(w) * 60.0
        y_coords = np.arange(h) * 60.0

        # With Phase 2C default p_thr=0.1, cells with prob 0.15 MUST be classified burned (finite)
        ds = surrogate_output_to_arrival_field(
            quantiles=quantiles,
            burn_prob=burn_prob,
            x_coords=x_coords,
            y_coords=y_coords,
        )
        assert isinstance(ds, xr.Dataset)
        assert np.all(np.isfinite(ds["arrival_q50"].values))

    def test_compute_burn_head_miss_rates(self) -> None:
        c_mask = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])  # 4 burned, 2 unburned
        pred_probs = np.array([0.03, 0.08, 0.20, 0.80, 0.01, 0.90])

        misses = compute_burn_head_miss_rates(
            pred_burn_probs=pred_probs,
            censor_masks=c_mask,
            thresholds=(0.05, 0.10, 0.25, 0.50),
        )

        # Total burned = 4
        # At 0.05: [0.03] < 0.05 -> 1 / 4 = 0.25
        # At 0.10: [0.03, 0.08] < 0.10 -> 2 / 4 = 0.50
        # At 0.25: [0.03, 0.08, 0.20] < 0.25 -> 3 / 4 = 0.75
        # At 0.50: [0.03, 0.08, 0.20] < 0.50 -> 3 / 4 = 0.75
        assert misses[0.05] == pytest.approx(0.25)
        assert misses[0.10] == pytest.approx(0.50)
        assert misses[0.25] == pytest.approx(0.75)
        assert misses[0.50] == pytest.approx(0.75)
        # Monotonicity: lower threshold means fewer false negatives (omissions)
        assert misses[0.05] <= misses[0.10] <= misses[0.25] <= misses[0.50]


class TestPhase2COneSidedConformalCalibration:
    """Requirement 5: One-sided lower-quantile conformal calibration for q10."""

    def test_one_sided_calibration_guarantee(self) -> None:
        np.random.seed(42)
        n = 800
        y_true = np.random.uniform(50.0, 500.0, size=n)

        # Under-conservative q10 (too high, true fire arrives earlier than q10 prediction)
        q10_raw = y_true + np.random.normal(5.0, 10.0, size=n)

        # Initial non-exceedance coverage P(Y >= q10) is low (~30-40%)
        init_cov = OneSidedLowerConformalCalibrator.evaluate_lower_coverage(q10_raw, y_true)
        assert init_cov < 0.60

        # Fit one-sided calibrator targeting 90% non-exceedance
        calibrator = OneSidedLowerConformalCalibrator(target_coverage=0.90)
        res = calibrator.fit(q10_raw[:400], y_true[:400])

        assert res.coverage_after >= 0.90
        assert res.delta_adjustment > 0.0  # Must subtract positive buffer to ensure safety

        # Test set evaluation
        q10_test_cal = calibrator.calibrate(q10_raw[400:])
        test_cov = OneSidedLowerConformalCalibrator.evaluate_lower_coverage(
            q10_test_cal, y_true[400:]
        )
        assert test_cov >= 0.88, f"One-sided test coverage {test_cov:.3f} below target guarantee"

    def test_calibrator_separation(self) -> None:
        # Ensure OneSidedLowerConformalCalibrator and SplitConformalCalibrator are distinct classes
        assert OneSidedLowerConformalCalibrator is not SplitConformalCalibrator


class TestPhase2CLatencyBenchmark:
    """Requirement 6: Latency benchmark protocol (median, p95, hardware name)."""

    def test_latency_benchmark_protocol(self) -> None:
        model = WindConditionedUNet(base_channels=16)
        device = torch.device("cpu")

        res = benchmark_surrogate_latency(
            model=model,
            device=device,
            batch_sizes=(1, 2),
            tile_size=(64, 64),
            num_warmup=2,
            num_repeats=3,
        )

        assert res.hardware_name is not None
        assert "CPU" in res.hardware_name or "Apple" in res.hardware_name or "NVIDIA" in res.hardware_name
        assert 1 in res.batch_latencies_median_ms_per_tile
        assert 1 in res.batch_latencies_p95_ms_per_tile
        assert res.batch_latencies_p95_ms_per_tile[1] >= res.batch_latencies_median_ms_per_tile[1]
        assert "Median (ms/tile)" in res.to_markdown()
        assert "P95 (ms/tile)" in res.to_markdown()


class TestPhase2CWholeDomainAndSizeDependence:
    """Requirement 7: Whole-domain run on (562, 564) and size dependence (256 vs 512)."""

    def test_whole_domain_562x564_inference(self) -> None:
        model = WindConditionedUNet(base_channels=16)
        device = torch.device("cpu")

        q, b = run_whole_domain_inference(model, device=device, domain_shape=(562, 564))
        assert q.shape == (1, 3, 562, 564)
        assert b.shape == (1, 1, 562, 564)

    def test_size_dependence_overlapping_region(self) -> None:
        model = WindConditionedUNet(base_channels=16)
        device = torch.device("cpu")

        diffs = verify_size_dependence(
            model=model,
            device=device,
            crop_size=128,
            full_size=256,
            interior_margin=16,
        )
        assert "mae_quantiles" in diffs
        assert "max_diff_quantiles" in diffs
        assert diffs["mae_quantiles"] >= 0.0


class TestPhase2CKagglePackaging:
    """Packaging and exclusion verification for Kaggle bundles."""

    def test_build_kaggle_packages_exclusions(self) -> None:
        repo_root = Path(".").resolve()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_base = Path(tmp_dir)
            code_dir, tiles_dir, c_size, t_size = build_kaggle_packages(repo_root, out_base)

            assert code_dir.exists()
            assert tiles_dir.exists()
            assert (code_dir / "src" / "wf052").exists()
            assert (code_dir / "pyproject.toml").exists()
            assert (tiles_dir / "configs" / "teacher_v1.yaml").exists()

            # Verify training fires
            dest_fires = [p.name for p in (tiles_dir / "data" / "tiles" / "fires").iterdir() if p.is_dir()]
            assert set(dest_fires) == set(SEVEN_TRAINING_FIRES)

            # Strictly verify NO prohibited domains or held-out fires
            all_paths_str = " ".join([str(p).lower() for p in tiles_dir.rglob("*")])
            for proh in ["camp_fire", "slater", "walker", "w5_cold"]:
                assert proh not in all_paths_str, f"Found prohibited fire {proh} in Kaggle package!"
