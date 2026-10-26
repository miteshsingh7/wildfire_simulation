"""Tests for Phase 6A causal replay harness, field providers, and evaluation metrics."""

import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry

from wf052.replay.harness import CausalReplayHarness, parse_clock_to_elapsed_min
from wf052.replay.methods import (
    sample_block_origins,
)
from wf052.replay.metrics import (
    compute_bootstrap_ci,
)
from wf052.replay.providers import (
    ObservationInflatedProvider,
    SurrogateFieldProvider,
    TeacherFieldProvider,
    TeacherV2FieldProvider,
)
from wf052.routing.contracts import validate_arrival_field
from wf052.routing.graph import RoadGraph, load_camp_fire_road_graph


@pytest.fixture(scope="module")
def full_road_graph() -> RoadGraph:
    """Camp Fire road graph rebuilt in Phase 4f aligned with the 60m raster grid."""
    prov = TeacherFieldProvider()
    field = prov.get_field(0.0)
    return load_camp_fire_road_graph(
        full_detail=True,
        grid_x_coords=field["x"].values,
        grid_y_coords=field["y"].values,
    )


def test_truth_hazards_loading(full_road_graph: RoadGraph) -> None:
    """Verify loading of the 23 burnovers, 178 roadblocks, and 235 abandoned vehicles."""
    harness = CausalReplayHarness(road_graph=full_road_graph)
    hazards = harness.truth_hazards

    assert len(hazards) > 0
    burnovers = [h for h in hazards if h.hazard_type == "burnover"]
    roadblocks = [h for h in hazards if h.hazard_type == "roadblock"]
    abandoned = [h for h in hazards if h.hazard_type == "abandoned_vehicle"]

    assert len(burnovers) == 23
    assert len(roadblocks) == 178
    assert len(abandoned) == 235
    assert len(hazards) == 23 + 178 + 235


def test_clock_parsing() -> None:
    """Test conversion of HH:MM clock string into elapsed minutes from 06:33 PST."""
    assert parse_clock_to_elapsed_min("06:33") == 0.0
    assert parse_clock_to_elapsed_min("06:43") == 10.0
    assert parse_clock_to_elapsed_min("07:33") == 60.0
    assert parse_clock_to_elapsed_min("08:00") == 87.0
    assert parse_clock_to_elapsed_min("12:33") == 360.0


def test_causal_observation_filtering(full_road_graph: RoadGraph) -> None:
    """Verify that observations at t > t_step - latency are invisible."""
    harness_0 = CausalReplayHarness(road_graph=full_road_graph, latency_min=0.0)
    harness_30 = CausalReplayHarness(road_graph=full_road_graph, latency_min=30.0)

    # Step at t = 60 min
    env_0 = harness_0.step(60.0)
    env_30 = harness_30.step(60.0)

    # With 0 latency, agent sees all observations with elapsed_adj <= 60
    # With 30 latency, agent sees only observations with elapsed_adj <= 30
    assert len(env_0.visible_observations) >= len(env_30.visible_observations)
    if len(env_30.visible_observations) > 0:
        max_t_30 = float(env_30.visible_observations["elapsed_minutes_adj"].max())
        assert max_t_30 <= 30.0 + 1e-6


def test_future_observation_perturbation_invariance(full_road_graph: RoadGraph) -> None:
    """Assert that perturbing future observations (t' > t) leaves step t outputs unchanged."""
    harness = CausalReplayHarness(road_graph=full_road_graph, latency_min=0.0)
    t_test = 60.0

    # Step at t_test before perturbation
    env_before = harness.step(t_test)
    obs_count_before = len(env_before.visible_observations)
    field_before = env_before.arrival_field["arrival_q50"].values.copy()

    # Perturb future observations in raw table (at t = 180 min, far in the future)
    future_fake_obs = gpd.GeoDataFrame(
        {
            "elapsed_minutes": [188.0, 200.0, 250.0],  # elapsed_adj = 180.0, 192.0, 242.0
            "elapsed_minutes_adj": [180.0, 192.0, 242.0],
            "geometry": [
                shapely.geometry.Point(620000.0, 4400000.0),
                shapely.geometry.Point(621000.0, 4401000.0),
                shapely.geometry.Point(622000.0, 4402000.0),
            ],
        },
        crs="EPSG:32610",
    )
    harness._raw_observations = gpd.GeoDataFrame(
        gpd.pd.concat([harness._raw_observations, future_fake_obs], ignore_index=True),
        crs="EPSG:32610",
    )

    # Step at t_test after future perturbation
    env_after = harness.step(t_test)
    obs_count_after = len(env_after.visible_observations)
    field_after = env_after.arrival_field["arrival_q50"].values.copy()

    # Outputs at t_test must be strictly identical!
    assert obs_count_before == obs_count_after
    np.testing.assert_array_equal(field_before, field_after)


def test_field_providers(full_road_graph: RoadGraph) -> None:
    """Verify Provider A, Provider B (sensitivity over R_spot), and Provider C stub."""
    # 1. Provider A
    prov_a = TeacherFieldProvider()
    field_a = prov_a.get_field(60.0)
    validate_arrival_field(field_a)
    assert field_a.attrs["provider"] == "teacher_v1"
    assert field_a.attrs["crs"] == "EPSG:32610"

    # 1b. Provider A2 (teacher_v2 with embers)
    prov_a2 = TeacherV2FieldProvider(assimilate=True)
    field_a2 = prov_a2.get_field(60.0)
    validate_arrival_field(field_a2)
    assert field_a2.attrs["provider"] == "teacher_v2"
    assert field_a2.attrs["crs"] == "EPSG:32610"
    # More burned cells than teacher_v1 because embers crossed canyon
    assert np.count_nonzero(np.isfinite(field_a2["arrival_q50"].values)) > np.count_nonzero(
        np.isfinite(field_a["arrival_q50"].values)
    )

    # 2. Provider B with R_spot sensitivity and leakage filtering
    for r_spot in (1.0, 2.0, 4.0, 7.0):
        prov_b = ObservationInflatedProvider(r_spot_km=r_spot)
        # Synthetic single observation at (625000, 4405000) at t=30 min
        synth_obs = gpd.GeoDataFrame(
            {
                "elapsed_minutes": [30.0],
                "geometry": [shapely.geometry.Point(625000.0, 4405000.0)],
            },
            crs="EPSG:32610",
        )
        field_b = prov_b.get_field(current_step_min=60.0, visible_observations=synth_obs)
        validate_arrival_field(field_b)
        assert field_b.attrs["provider"] == "observation_inflated"
        assert field_b.attrs["label"] == "heuristic, not physics"
        assert field_b.attrs["r_spot_km"] == r_spot

        # Pointwise minimum with teacher field
        assert np.all(field_b["arrival_q50"].values <= field_a["arrival_q50"].values)

    # 2b. Provider B built on Provider A2
    prov_b_a2 = ObservationInflatedProvider(base_teacher_provider=prov_a2, r_spot_km=4.0)
    field_b_a2 = prov_b_a2.get_field(60.0, visible_observations=synth_obs)
    validate_arrival_field(field_b_a2)
    assert np.all(field_b_a2["arrival_q50"].values <= field_a2["arrival_q50"].values)

    # 3. Provider C raises NotImplementedError
    prov_c = SurrogateFieldProvider()
    with pytest.raises(NotImplementedError) as exc_info:
        prov_c.get_field(60.0)
    assert "Surrogate model weights not trained in Phase 6A" in str(exc_info.value)


def test_roadblock_edge_closures(full_road_graph: RoadGraph) -> None:
    """Verify that active roadblocks sever nearby road graph edges."""
    harness = CausalReplayHarness(road_graph=full_road_graph)
    env = harness.step(120.0)
    closed_edges = harness.get_closed_edges_for_roadblocks(env.active_roadblocks)

    assert len(closed_edges) > 0
    # Both directions must be closed
    for u, v in closed_edges:
        assert (v, u) in closed_edges


def test_bootstrap_ci() -> None:
    """Test bootstrap confidence interval estimation."""
    data = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    ci_low, ci_high = compute_bootstrap_ci(data, stat_fn="mean", num_resamples=500, seed=42)

    assert ci_low <= float(np.mean(data)) <= ci_high
    assert 10.0 <= ci_low
    assert ci_high <= 20.0


def test_sample_block_origins(full_road_graph: RoadGraph) -> None:
    """Test deterministic sampling of 200 block origins with departure times."""
    origins = sample_block_origins(full_road_graph, num_samples=200, seed=42)
    assert len(origins) == 200

    # Assert all origins are within road graph
    for orig in origins:
        assert orig.origin_id in full_road_graph.nodes
        assert orig.departure_time_min >= 0.0
        assert orig.assigned_zone.startswith("Zone_")


def test_spot_radius_field_changes() -> None:
    """Verify that changing r_spot_km parameter reaches the provider and alters the arrival field."""
    synth_obs = gpd.GeoDataFrame(
        {
            "elapsed_minutes_adj": [30.0],
            "geometry": [shapely.geometry.Point(630000.0, 4400000.0)],
        },
        crs="EPSG:32610",
    )
    base_prov = TeacherFieldProvider()
    f_base = base_prov.get_field(60.0)

    prov_1km = ObservationInflatedProvider(r_spot_km=1.0)
    prov_4km = ObservationInflatedProvider(r_spot_km=4.0)
    prov_7km = ObservationInflatedProvider(r_spot_km=7.0)

    f_1km = prov_1km.get_field(60.0, visible_observations=synth_obs)
    f_4km = prov_4km.get_field(60.0, visible_observations=synth_obs)
    f_7km = prov_7km.get_field(60.0, visible_observations=synth_obs)

    # Count newly inflated cells compared to the base teacher
    inf_1km = int(np.count_nonzero(f_1km["arrival_q50"].values < f_base["arrival_q50"].values))
    inf_4km = int(np.count_nonzero(f_4km["arrival_q50"].values < f_base["arrival_q50"].values))
    inf_7km = int(np.count_nonzero(f_7km["arrival_q50"].values < f_base["arrival_q50"].values))

    assert inf_1km < inf_4km < inf_7km, f"Expected strictly increasing inflation: {inf_1km} < {inf_4km} < {inf_7km}"
    assert not np.array_equal(f_1km["arrival_q50"].values, f_4km["arrival_q50"].values)
    assert not np.array_equal(f_4km["arrival_q50"].values, f_7km["arrival_q50"].values)


def test_teacher_v2_freeze_hash() -> None:
    """Verify that configs/teacher_v2.yaml loads and matches the frozen SHA-256 hash."""
    import hashlib
    from pathlib import Path

    cfg_path = Path("configs/teacher_v2.yaml")
    assert cfg_path.exists(), "configs/teacher_v2.yaml does not exist"

    raw_bytes = cfg_path.read_bytes()
    computed_hash = hashlib.sha256(raw_bytes).hexdigest()
    expected_hash = "a6a92f81f7a9c72b5899d5d3173f86943a7f98648c4f210dcc5c646d2b4e85a5"

    assert computed_hash == expected_hash, f"Hash mismatch: {computed_hash} != {expected_hash}"

    # Also assert the 8-char hash named copy exists and is identical
    first8 = expected_hash[:8]
    copy_path = Path(f"configs/teacher_v2_{first8}.yaml")
    assert copy_path.exists(), f"Copy {copy_path} does not exist"
    assert copy_path.read_bytes() == raw_bytes
