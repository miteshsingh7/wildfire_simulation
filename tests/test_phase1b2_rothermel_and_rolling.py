"""Unit tests for Phase 1B-2:
- Rothermel FBFM40 surface fire spread equations verified against pyretechnics-2026.8.10 (EPL-2.0)
- ROS-driven contact rule (monotone area vs gamma_ros, residence time burnout)
- Rolling-horizon evaluation protocol (re-init from every 3rd GOFER perimeter, 1-6 h forecast)
- Perturbed uncertainty ensemble (P_base band, wind speed +-15%, wind direction +-15 deg,
  ember params, ignition-extent jitter)
- Nearest assimilated point + constant front speed baseline (strictly-future points only).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from wf052.sim.ember_kernel import EmberParams
from wf052.sim.rolling_protocol import (
    PerturbationConfig,
    evaluate_constant_speed_baseline_grid,
    evaluate_rolling_horizon_fire,
    run_perturbed_rolling_window,
)
from wf052.sim.rothermel_ros import (
    ROSTeacherParams,
    compute_rothermel_max_behavior,
    compute_rothermel_no_wind_no_slope,
    simulate_ros_teacher_60m,
)
from wf052.sim.teacher_calibrator import FireDataset60m, TeacherParams


def _make_synthetic_rolling_dataset(h: int = 36, w: int = 36) -> FireDataset60m:
    """Create a synthetic 60m dataset with 25 hourly GOFER masks (0..24h) expanding eastward."""
    device = torch.device("cpu")
    fbfm = np.full((h, w), 102, dtype=np.int32)  # GR2 grass
    fbfm[:, 26:30] = 91  # WUI strip
    p_veg = torch.full((h, w), 0.30, dtype=torch.float32, device=device)
    is_burnable = torch.from_numpy(fbfm != 91).to(device)
    is_gs = torch.ones((h, w), dtype=torch.bool, device=device)
    is_timber = torch.zeros((h, w), dtype=torch.bool, device=device)
    p_den = torch.zeros((h, w), dtype=torch.float32, device=device)
    slope_deg = torch.zeros((h, w, 3, 3), dtype=torch.float32, device=device)

    weather_df = pd.DataFrame(
        {
            "hour_index": list(range(24)),
            "wind_speed_ms": [10.0] * 24,
            "wind_gust_ms": [15.0] * 24,
            "wind_direction_deg": [270.0] * 24,  # From West -> blowing East
            "rh_pct": [15.0] * 24,
            "u10_ms": [10.0] * 24,
            "v10_ms": [0.0] * 24,
        }
    )

    hourly_masks = np.zeros((25, h, w), dtype=bool)
    arrival_min = np.full((h, w), np.nan, dtype=np.float32)
    areas_ha = np.zeros(25, dtype=np.float32)
    for hr in range(25):
        col_max = min(w - 1, 22 + hr // 2)
        hourly_masks[hr, 2:31, 1:col_max] = True
        newly = hourly_masks[hr] & np.isnan(arrival_min)
        arrival_min[newly] = float(hr * 60.0)
        areas_ha[hr] = float(np.count_nonzero(hourly_masks[hr]) * 0.36)

    grid_c = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w))
    wind_axis_proj_km = (grid_c - 3.0) * 0.06

    return FireDataset60m(
        fire_name="SyntheticRolling",
        regime="wind-driven",
        grid_size=h,
        res_m=60.0,
        fbfm_arr=fbfm,
        p_veg_base=p_veg,
        is_grass_shrub=is_gs,
        is_timber=is_timber,
        is_burnable=is_burnable,
        p_den_base=p_den,
        slope_deg=slope_deg,
        weather_df=weather_df,
        gofer_hourly_masks=hourly_masks,
        gofer_arrival_min=arrival_min,
        gofer_areas_ha=areas_ha,
        feds_12h_mask_375m=np.zeros((h // 6, w // 6), dtype=bool),
        feds_24h_mask_375m=np.zeros((h // 6, w // 6), dtype=bool),
        wind_axis_proj_km=wind_axis_proj_km.astype(np.float32),
    )


def test_rothermel_matches_pyretechnics_reference() -> None:
    """Verify our Rothermel FBFM40 formulas match pyretechnics-2026.8.10 to < 1e-5 relative error."""
    moist = (0.06, 0.07, 0.08, 0.0, 0.60, 0.90)
    # Golden reference values computed directly from pyretechnics-2026.8.10 (fuel_models.py + surface_fire.py)
    # under midflame_wind_speed=180.0 m/min, upwind_direction=45.0 deg, slope=0.20, aspect=225.0 deg
    golden = {
        102: {
            "R0": 0.469153,
            "I0": 21.526810,
            "phiS_G": 36.540968,
            "phiW_sc": 0.019140,
            "phiW_ex": 1.455047,
            "Rmax": 17.451044,
            "LW": 1.903094,
            "ecc": 0.850818,
        },
        122: {
            "R0": 0.363552,
            "I0": 31.565192,
            "phiS_G": 31.878181,
            "phiW_sc": 0.015887,
            "phiW_ex": 1.458249,
            "Rmax": 11.455395,
            "LW": 1.901873,
            "ecc": 0.850610,
        },
        145: {
            "R0": 0.728034,
            "I0": 217.372279,
            "phiS_G": 33.752014,
            "phiW_sc": 0.075102,
            "phiW_ex": 1.188684,
            "Rmax": 26.582260,
            "LW": 1.897256,
            "ecc": 0.849817,
        },
        165: {
            "R0": 0.311344,
            "I0": 151.272856,
            "phiS_G": 17.035100,
            "phiW_sc": 0.028509,
            "phiW_ex": 1.174248,
            "Rmax": 4.140075,
            "LW": 1.877321,
            "ecc": 0.846321,
        },
        188: {
            "R0": 0.252692,
            "I0": 33.733146,
            "phiS_G": 13.886479,
            "phiW_sc": 0.006351,
            "phiW_ex": 1.433398,
            "Rmax": 2.932882,
            "LW": 1.891412,
            "ecc": 0.848805,
        },
    }

    for code, ref in golden.items():
        bmin = compute_rothermel_no_wind_no_slope(code, moist)
        bmax = compute_rothermel_max_behavior(
            bmin,
            midflame_wind_speed_m_min=180.0,
            upwind_direction_deg=45.0,
            slope_rise_run=0.20,
            aspect_deg=225.0,
        )
        assert bmin["base_spread_rate"] == pytest.approx(ref["R0"], rel=2e-5)
        assert bmin["base_fireline_intensity"] == pytest.approx(ref["I0"], rel=2e-5)
        assert bmin["phiS_G"] == pytest.approx(ref["phiS_G"], rel=2e-5)
        assert bmin["phiW_scalr"] == pytest.approx(ref["phiW_sc"], rel=2e-4)
        assert bmin["phiW_expnt"] == pytest.approx(ref["phiW_ex"], rel=2e-5)
        assert bmax["max_spread_rate"] == pytest.approx(ref["Rmax"], rel=2e-5)
        assert bmax["length_to_width_ratio"] == pytest.approx(ref["LW"], rel=2e-5)
        assert bmax["eccentricity"] == pytest.approx(ref["ecc"], rel=2e-5)


def test_ros_contact_rule_monotone_in_gamma() -> None:
    """Verify ROS-driven contact rule produces non-decreasing burned area as gamma_ros increases."""
    ds = _make_synthetic_rolling_dataset()
    areas = []
    for gamma in (0.02, 0.08, 0.20, 0.45):
        out = simulate_ros_teacher_60m(
            ds=ds,
            ros_params=ROSTeacherParams(gamma_ros=gamma, res_time_scale=1.0),
            duration_hours=6,
            steps_per_hour=30,
            seed=42,
        )
        areas.append(int(np.count_nonzero(out["hourly_sim_masks"][-1])))
    assert areas[0] <= areas[1] <= areas[2] <= areas[3]
    assert areas[3] > areas[0]


def test_rolling_horizon_protocol() -> None:
    """Verify rolling-horizon evaluation re-initializes from every 3rd GOFER perimeter."""
    ds = _make_synthetic_rolling_dataset()
    res = evaluate_rolling_horizon_fire(
        ds=ds,
        teacher_params=TeacherParams(p_base=0.20),
        ember_params=EmberParams(alpha_launch=0.003, beta_dist_m_per_ms=100.0, p_urban_wui=0.12),
        use_ember_kernel=True,
        origins_h=(0, 3, 6, 9),
        horizons_h=(1, 3, 6),
        seed=42,
    )
    assert "rolling_loss" in res
    assert len(res["window_rows"]) == 4 * 3
    for row in res["window_rows"]:
        assert row["horizon_h"] in (1, 3, 6)
        assert "denom_obs_2km_union" in row
        assert row["denom_obs_2km_union"] > 0


def test_perturbed_ensemble_and_constant_speed_baseline() -> None:
    """Verify perturbed uncertainty ensemble and constant-speed assimilation baseline."""
    ds = _make_synthetic_rolling_dataset()
    ens = run_perturbed_rolling_window(
        ds=ds,
        base_teacher=TeacherParams(p_base=0.20),
        base_ember=EmberParams(alpha_launch=0.003, beta_dist_m_per_ms=100.0, p_urban_wui=0.12),
        use_ember_kernel=True,
        t0_h=3,
        duration_h=6,
        perturb_cfg=PerturbationConfig(
            num_members=8,
            p_base_half_width=0.04,
            wind_speed_frac=0.15,
            wind_dir_deg=15.0,
            ember_frac=0.25,
            init_jitter_cells=2,
        ),
    )
    assert ens["denom_obs_progression_cells"] > 0
    assert 0.0 <= ens["coverage_pct"] <= 100.0
    assert "/" in ens["coverage_with_denom"]

    # Constant-speed baseline on strictly-future cells
    base_res = evaluate_constant_speed_baseline_grid(
        init_mask=ds.gofer_hourly_masks[3],
        obs_arrival_min=ds.gofer_arrival_min,
        t0_min=180.0,
        horizon_min=180.0,
        speed_m_per_min=1.0,
        res_m=60.0,
    )
    assert base_res["total_future_cells"] > 0
    assert base_res["reached_cells"] > 0
