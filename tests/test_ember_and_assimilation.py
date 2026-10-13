"""Unit tests for Phase 1B ember spotting kernel and assimilation mode (TDD)."""

import numpy as np
import pandas as pd
import torch

from wf052.sim.ember_kernel import (
     EmberParams,
    compute_far_field_jump_cells,
    simulate_teacher_with_embers_60m,
)
from wf052.sim.teacher_calibrator import FireDataset60m, TeacherParams


def _make_synthetic_dataset(device: torch.device, h: int = 64, w: int = 64) -> FireDataset60m:
    """Create a synthetic 60m dataset with a 4-cell water barrier in the middle."""
    fbfm = np.full((h, w), 145, dtype=np.int32)  # Chaparral shrub
    # Vertical open-water barrier (code 98) across columns 28..31
    fbfm[:, 28:32] = 98
    # Urban WUI zone (code 91) on columns 40..48
    fbfm[:, 40:48] = 91

    p_veg = np.full((h, w), 0.55, dtype=np.float32)
    p_veg[:, 28:32] = -1.0
    p_veg[:, 40:48] = -1.0

    is_burnable = p_veg > -0.5
    is_gs = fbfm == 145
    is_timber = np.zeros((h, w), dtype=bool)
    p_den = np.zeros((h, w), dtype=np.float32)
    slope = np.zeros((h, w, 3, 3), dtype=np.float32)

    # Strong East-to-West wind (from 90 deg -> blowing West towards 180 deg, i.e., +col to -col;
    # or West-to-East wind from 270 deg -> blowing East towards 0 deg, i.e., -col to +col)
    weather_df = pd.DataFrame({
        "hour_index": list(range(6)),
        "u10_ms": [18.0] * 6,
        "v10_ms": [0.0] * 6,
        "wind_speed_ms": [18.0] * 6,
        "wind_gust_ms": [26.0] * 6,
        "wind_direction_deg": [270.0] * 6,  # From West, blowing East (+col)
        "temp_c": [22.0] * 6,
        "rh_pct": [12.0] * 6,
    })

    masks = np.zeros((7, h, w), dtype=bool)
    masks[0, 28:36, 10:14] = True  # Ignition west of the water barrier

    grid_r, grid_c = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    wind_proj = (grid_c - 12.0) * 0.06  # Downwind +col in km

    return FireDataset60m(
        fire_name="SyntheticBarrier",
        regime="wind-driven",
        grid_size=h,
        res_m=60.0,
        fbfm_arr=fbfm,
        p_veg_base=torch.from_numpy(p_veg).to(device),
        is_grass_shrub=torch.from_numpy(is_gs).to(device),
        is_timber=torch.from_numpy(is_timber).to(device),
        is_burnable=torch.from_numpy(is_burnable).to(device),
        p_den_base=torch.from_numpy(p_den).to(device),
        slope_deg=torch.from_numpy(slope).to(device),
        weather_df=weather_df,
        gofer_hourly_masks=masks,
        gofer_arrival_min=np.full((h, w), np.nan, dtype=np.float32),
        gofer_areas_ha=np.zeros(7, dtype=np.float32),
        feds_12h_mask_375m=np.zeros((h // 6, w // 6), dtype=bool),
        feds_24h_mask_375m=np.zeros((h // 6, w // 6), dtype=bool),
        wind_axis_proj_km=wind_proj,
    )


def test_ember_params_has_at_most_3_free_params() -> None:
    params = EmberParams()
    d = params.to_dict()
    assert len(d) == 3
    assert set(d.keys()) == {"alpha_launch", "beta_dist_m_per_ms", "p_urban_wui"}


def test_ember_kernel_crosses_water_barrier_and_ignites_wui() -> None:
    device = torch.device("cpu")
    ds = _make_synthetic_dataset(device)
    t_params = TeacherParams(p_base=0.22)

    # 1. Without ember kernel: cannot cross the 4-cell (240m) water barrier at cols 28..31
    res_off = simulate_teacher_with_embers_60m(
        ds,
        t_params,
        ember_params=None,
        use_ember_kernel=False,
        duration_hours=4,
        steps_per_hour=30,
        seed=42,
    )
    final_off = res_off["hourly_sim_masks"][-1]
    assert not np.any(final_off[:, 32:]), "Contact-only CA must not jump a 4-cell water barrier"

    # 2. With ember kernel enabled: jumps across cols 28..31 and can ignite WUI (cols 40..48)
    e_params = EmberParams(alpha_launch=0.03, beta_dist_m_per_ms=100.0, p_urban_wui=0.25)
    res_on = simulate_teacher_with_embers_60m(
        ds,
        t_params,
        ember_params=e_params,
        use_ember_kernel=True,
        duration_hours=4,
        steps_per_hour=30,
        seed=42,
    )
    final_on = res_on["hourly_sim_masks"][-1]
    assert np.any(final_on[:, 32:]), "Ember kernel should jump across the 4-cell water barrier"
    assert np.any(final_on[:, 40:48]), "Ember kernel with p_urban_wui > 0 should ignite WUI code 91 cells"
    # Water cells themselves must never burn
    assert not np.any(final_on[:, 28:32]), "Open water (code 98) must never burn"


def test_assimilation_init_from_observed_extent() -> None:
    device = torch.device("cpu")
    ds = _make_synthetic_dataset(device)
    t_params = TeacherParams(p_base=0.22)

    custom_init = np.zeros((64, 64), dtype=bool)
    custom_init[20:25, 34:38] = True  # Start east of the water barrier at T0 = 120 min

    res_assim = simulate_teacher_with_embers_60m(
        ds,
        t_params,
        ember_params=None,
        use_ember_kernel=False,
        duration_hours=3,
        steps_per_hour=30,
        seed=42,
        init_burned_mask=custom_init,
        start_hour_index=2,
        start_elapsed_min=120.0,
    )
    arr = res_assim["arrival_time_min"]
    # Initially burned cells should have arrival time == 120.0 min
    assert np.all(arr[custom_init] == 120.0)
    # Newly burned cells should have arrival time > 120.0 and <= 300.0 min
    newly_burned = (~custom_init) & (~np.isnan(arr))
    assert np.any(newly_burned)
    assert np.all(arr[newly_burned] > 120.0)
    assert np.all(arr[newly_burned] <= 300.0)


def test_far_field_jump_cells_metric() -> None:
    mask = np.zeros((50, 50), dtype=bool)
    mask[20:30, 10:20] = True  # Main contiguous core
    mask[25, 35] = True  # Disconnected spot fire 15 cells (900m) downwind
    n_jump = compute_far_field_jump_cells(mask, min_jump_cells=5)
    assert n_jump == 1
