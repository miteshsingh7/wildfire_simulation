"""Unit tests for Phase 2B surrogate enhancements.

Covers:
1. Data diagnostics computation and reporting.
2. Compact storage serialization, footprint math, and CompactSurrogateDataset.
3. Router ArrivalField adapter and multi-tile stitching against ArrivalField contract.
4. Split-conformal quantile recalibration and coverage guarantee verification.
5. Latency benchmarking and fully convolutional behavior on variable spatial shapes.
6. Exhaustive held-out fire exclusions check (Camp Fire and Phase 1B fresh held-out fires).
"""

import tempfile

import numpy as np
import pytest
import torch

from wf052.routing.contracts import validate_arrival_field
from wf052.surrogate.adapter import (
    TilePrediction,
    stitch_tiles_to_arrival_field,
    surrogate_output_to_arrival_field,
)
from wf052.surrogate.calibration import SplitConformalCalibrator
from wf052.surrogate.dataset import (
    SurrogateSample,
    calculate_upwind_third_ignition,
    collate_surrogate_batch,
    compute_slope_aspect,
)
from wf052.surrogate.diagnostics import compute_data_diagnostics
from wf052.surrogate.latency import (
    benchmark_surrogate_latency,
)
from wf052.surrogate.model import WindConditionedUNet
from wf052.surrogate.standin_generator import (
    generate_analytic_arrival_surface,
    load_fire_rasters_crop,
    load_held_out_fire_exclusions,
    simulate_cellular_standin_arrival,
    verify_fire_not_excluded,
)
from wf052.surrogate.storage import (
    CompactSurrogateDataset,
    compute_storage_footprint,
    save_compact_dataset,
)


def create_synthetic_sample(
    fire_name: str = "bobcat",
    sample_id: str = "test_01",
    shape: tuple[int, int] = (256, 256),
    seed: int = 1234,
    source: str = "analytic_wind_ellipse",
) -> SurrogateSample:
    """Helper to generate a valid synthetic stand-in sample conforming to data contract."""
    rng = np.random.RandomState(seed)
    h, w = shape
    fuel_grp = rng.randint(0, 7, size=(h, w)).astype(np.int64)
    elev = rng.normal(1000.0, 200.0, size=(h, w)).astype(np.float32)
    slope, asp_sin, asp_cos = compute_slope_aspect(elev, resolution_m=60.0)
    elev_z = ((elev - np.mean(elev)) / np.std(elev)).astype(np.float32)
    canopy = rng.uniform(0.0, 1.0, size=(h, w)).astype(np.float32)

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
    )


class TestPhase2BDiagnostics:
    """Tests for stand-in data diagnostics and distribution profiling."""

    def test_compute_data_diagnostics(self) -> None:
        samples = [
            create_synthetic_sample(
                fire_name="bobcat", sample_id=f"b_{i}", seed=i + 1, source="analytic_wind_ellipse"
            )
            for i in range(4)
        ] + [
            create_synthetic_sample(
                fire_name="caldor", sample_id=f"c_{i}", seed=i + 50, source="teacher_v1_cellular"
            )
            for i in range(4)
        ]

        report = compute_data_diagnostics(samples)

        assert report.total_samples == 8
        assert report.sample_count_by_fire["bobcat"] == 4
        assert report.sample_count_by_fire["caldor"] == 4
        assert report.sample_count_by_source["analytic_wind_ellipse"] == 4
        assert report.sample_count_by_source["teacher_v1_cellular"] == 4
        assert len(report.hist_counts) == 10
        assert 0.0 <= report.pct_censored_cells <= 100.0
        assert report.num_stalled >= 0
        assert not report.requires_stratified_sampling  # < 40% stalled

        md = report.to_markdown()
        assert "# Stand-in Dataset Diagnostics Report (Phase 2B)" in md
        assert "bobcat" in md
        assert "teacher_v1_cellular" in md

    def test_cellular_standin_simulation(self) -> None:
        sample = create_synthetic_sample(shape=(128, 128))
        ign_r, ign_c = 64, 64
        arr, c_mask = simulate_cellular_standin_arrival(
            fuel_group=sample.fuel_group,
            hourly_wind_uv=sample.hourly_wind_uv,
            ign_r=ign_r,
            ign_c=ign_c,
            resolution_m=60.0,
            horizon_min=720.0,
        )

        assert arr.shape == (128, 128)
        assert c_mask.shape == (128, 128)
        assert arr[ign_r, ign_c] == 0.0  # Ignition cell arrives at t=0
        assert np.any(c_mask == 0.0)  # Some cells burned
        assert np.all(arr <= 720.0)


class TestPhase2BStorage:
    """Tests for compact storage serialization, footprint math, and Dataset loading."""

    def test_storage_footprint_math(self) -> None:
        rep = compute_storage_footprint(num_landscapes=7, tile_shape=(256, 256))

        # Dynamic per-sample footprint must be ~136 KB
        assert 139_000 <= rep.per_sample_dynamic_bytes <= 140_000
        # Compression ratio must exceed 25x
        assert rep.compression_ratio > 25.0

        # Check size numbers
        # 5k samples must be < 1 GB (~669 MB)
        assert 600.0 <= rep.size_5k_samples_mib <= 800.0
        # 20k samples must be ~2.6 GB
        assert 2.0 <= rep.size_20k_samples_gib <= 3.2

    def test_compact_dataset_roundtrip(self) -> None:
        samples = [
            create_synthetic_sample(
                fire_name="bobcat", sample_id=f"b_{i}", shape=(128, 128), seed=i + 10
            )
            for i in range(2)
        ] + [
            create_synthetic_sample(
                fire_name="caldor", sample_id=f"c_{i}", shape=(128, 128), seed=i + 20
            )
            for i in range(2)
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            meta = save_compact_dataset(samples, output_dir=tmp_dir, scale_factor=10.0)
            assert meta["num_samples"] == 4

            # Load via CompactSurrogateDataset
            ds = CompactSurrogateDataset(tmp_dir)
            assert len(ds) == 4

            item0 = ds[0]
            # Spatial features must be 13 channels
            assert item0["spatial_features"].shape == (13, 128, 128)
            assert item0["fuel_group"].shape == (128, 128)
            assert item0["wind_series"].shape == (12, 2)
            assert item0["target_arrival"].shape == (128, 128)
            assert item0["censor_mask"].shape == (128, 128)

            # Test precision: decoded arrival vs original arrival
            orig_arr = samples[0].target_arrival_time
            loaded_arr = item0["target_arrival"].numpy()
            np.testing.assert_allclose(loaded_arr, orig_arr, atol=0.2)

            # Test collate_surrogate_batch compatibility
            batch = collate_surrogate_batch([ds[0], ds[1]])
            assert batch.spatial_features.shape == (2, 13, 128, 128)
            assert batch.wind_series.shape == (2, 12, 2)


class TestPhase2BRouterAdapter:
    """Tests for converting surrogate outputs to router ArrivalField contract and stitching."""

    def test_surrogate_output_to_arrival_field(self) -> None:
        h, w = 128, 128
        # Synthetic quantiles
        base_arr = np.linspace(10.0, 300.0, h * w).reshape(h, w)
        q10 = base_arr * 0.9
        q50 = base_arr
        q90 = base_arr * 1.1
        quantiles = np.stack([q10, q50, q90], axis=0)

        # Burn probability: half above threshold, half below
        burn_prob = np.zeros((h, w), dtype=np.float32)
        burn_prob[:64, :] = 0.8  # Burned
        burn_prob[64:, :] = 0.2  # Unburned

        x_coords = 600000.0 + np.arange(w) * 60.0
        y_coords = 4400000.0 - np.arange(h) * 60.0

        ds = surrogate_output_to_arrival_field(
            quantiles=quantiles,
            burn_prob=burn_prob,
            x_coords=x_coords,
            y_coords=y_coords,
            crs="EPSG:32610",
            ignition_time="2020-09-06T19:00:00Z",
            p_thr=0.5,
        )

        # Validate against router contract
        validate_arrival_field(ds)

        # Check that unburned region is strictly +inf
        q50_out = ds["arrival_q50"].values
        assert np.all(np.isinf(q50_out[64:, :]))
        assert np.all(np.isfinite(q50_out[:64, :]))

    def test_stitch_tiles_to_arrival_field(self) -> None:
        h_full, w_full = 180, 180
        tile_h, tile_w = 120, 120

        # Tile 1 in top-left: [0:120, 0:120]
        q_t1 = np.full((3, tile_h, tile_w), 100.0, dtype=np.float32)
        bp_t1 = np.full((tile_h, tile_w), 0.9, dtype=np.float32)
        tile1 = TilePrediction(quantiles=q_t1, burn_prob=bp_t1, r0=0, c0=0)

        # Tile 2 overlapping: [60:180, 60:180] with faster arrival 60.0
        q_t2 = np.full((3, tile_h, tile_w), 60.0, dtype=np.float32)
        bp_t2 = np.full((tile_h, tile_w), 0.9, dtype=np.float32)
        tile2 = TilePrediction(quantiles=q_t2, burn_prob=bp_t2, r0=60, c0=60)

        ds_stitched = stitch_tiles_to_arrival_field(
            tile_predictions=[tile1, tile2],
            domain_shape=(h_full, w_full),
            origin_x=600000.0,
            origin_y=4400000.0,
            resolution_m=60.0,
            crs="EPSG:32610",
            ignition_time="2020-09-06T19:00:00Z",
            p_thr=0.5,
        )

        # Must satisfy router contract
        validate_arrival_field(ds_stitched)

        q50_stitched = ds_stitched["arrival_q50"].values
        # Overlapping intersection [60:120, 60:120] should take min arrival = 60.0
        assert np.all(q50_stitched[60:120, 60:120] == pytest.approx(60.0))
        # Top-left non-overlapping region [0:60, 0:60] should take tile1 arrival = 100.0
        assert np.all(q50_stitched[0:60, 0:60] == pytest.approx(100.0))
        # Bottom-left non-covered region [120:180, 0:60] should be +inf
        assert np.all(np.isinf(q50_stitched[120:180, 0:60]))


class TestPhase2BConformalCalibration:
    """Tests for split-conformal quantile recalibration."""

    def test_conformal_calibration_guarantee(self) -> None:
        np.random.seed(42)
        n = 1000
        # True arrivals
        y = np.random.uniform(50.0, 500.0, size=n)

        # Artificially create under-covered quantiles (narrow interval with ~50% coverage)
        q10 = y - np.random.uniform(1.0, 10.0, size=n)  # too high lower bound
        q90 = y + np.random.uniform(1.0, 10.0, size=n)  # too low upper bound
        q50 = (q10 + q90) / 2.0

        # Add some noise so true y often spills out
        q10 += np.random.normal(0, 15, size=n)
        q90 -= np.random.normal(0, 15, size=n)
        q_under = np.stack([q10, q50, q90], axis=0)[:, np.newaxis, :]  # (3, 1, N)

        # Initial coverage is severely degraded
        init_cov = SplitConformalCalibrator.evaluate_coverage(q_under, y)
        assert init_cov < 0.70, f"Expected under-covered quantiles, got {init_cov}"

        # Fit calibrator on first half
        calibrator = SplitConformalCalibrator(target_coverage=0.80, asymmetric=True)
        res = calibrator.fit(
            pred_quantiles=q_under[:, :, :500],
            true_arrival=y[:500],
        )

        assert res.coverage_before < 0.70
        assert res.coverage_after >= 0.80  # On calibration set, coverage is >= 80%

        # Recalibrate test set (second half)
        q_test = q_under[:, :, 500:]
        y_test = y[500:]
        recalibrated_test = calibrator.calibrate(q_test)

        test_cov_after = SplitConformalCalibrator.evaluate_coverage(recalibrated_test, y_test)
        # Test coverage should achieve target ~80% (allowing small finite sample tolerance)
        assert test_cov_after >= 0.78, (
            f"Recalibrated test coverage {test_cov_after} below guarantee"
        )


class TestPhase2BLatencyAndFullyConvolutional:
    """Tests for inference latency benchmark and fully convolutional behavior."""

    def test_latency_benchmark_and_fc_behavior(self) -> None:
        model = WindConditionedUNet(in_spatial_channels=13, base_channels=16)
        device = torch.device("cpu")

        res = benchmark_surrogate_latency(
            model=model,
            device=device,
            batch_sizes=(1, 2),  # fast test sizes
            tile_size=(128, 128),
            num_warmup=1,
            num_repeats=2,
        )

        assert 1 in res.batch_latencies_ms_per_tile
        assert 2 in res.batch_latencies_ms_per_tile
        assert res.batch_latencies_ms_per_tile[1] > 0.0
        assert res.fully_convolutional_passed is True

        md = res.to_markdown()
        assert "Surrogate Model Inference Latency & Benchmark" in md
        assert "PASSED" in md


class TestPhase2BHeldOutExclusions:
    """Tests verifying exclusion of Camp Fire and all Phase 1B fresh held-out fires."""

    def test_held_out_fire_exclusions_rejected(self) -> None:
        cfg = load_held_out_fire_exclusions()
        assert "held_out_fires" in cfg

        held_out_entries = cfg["held_out_fires"]
        assert len(held_out_entries) >= 4  # Camp Fire + 3 fresh proposed fires

        # Every alias of every held out fire must be rejected
        for entry in held_out_entries:
            name = entry["name"]
            aliases = entry.get("aliases", [])
            for alias in [name] + aliases:
                with pytest.raises(ValueError, match="strictly held-out|prohibited_domains"):
                    verify_fire_not_excluded(alias)
                with pytest.raises(
                    ValueError, match="strictly held-out|prohibited_domains|Prohibited or invalid"
                ):
                    load_fire_rasters_crop(fire_name=alias)

    def test_training_fires_are_not_rejected(self) -> None:
        # 7 training fires must NOT be rejected by exclusion verifier
        training_fires = ["bobcat", "caldor", "creek", "glass", "kincade", "tamarack", "zogg"]
        for f in training_fires:
            verify_fire_not_excluded(f)
