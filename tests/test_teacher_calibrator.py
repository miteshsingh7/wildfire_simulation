"""Unit tests for Phase 1A Step 3 physics teacher parameterization and multi-scale calibration."""

import numpy as np

from wf052.sim.teacher_calibrator import (
    TeacherParams,
    aggregate_to_coarse_fraction,
    compute_composite_calibration_loss,
)


def test_teacher_params_defaults_and_dict() -> None:
    params = TeacherParams()
    assert params.p_base == 0.50
    assert params.a == 0.06
    assert params.c_1 == 0.045
    assert params.c_2 == 0.15
    assert params.p_continue == 0.30
    assert params.fuel_scale_grass_shrub == 1.0
    assert params.fuel_scale_timber == 1.0
    assert params.wind_moisture_damping == 0.0
    d = params.to_dict()
    assert d["p_base"] == 0.50
    assert len(d) == 9


def test_aggregate_to_coarse_fraction_scales() -> None:
    mask = np.zeros((720, 720), dtype=bool)
    mask[:33, :33] = True
    frac_2km = aggregate_to_coarse_fraction(mask, block_size=33)
    assert frac_2km.shape == (720 // 33, 720 // 33)
    assert np.isclose(frac_2km[0, 0], 1.0)
    assert np.isclose(frac_2km[1, 1], 0.0)

    frac_375m = aggregate_to_coarse_fraction(mask, block_size=6)
    assert frac_375m.shape == (120, 120)
    assert np.isclose(frac_375m[0, 0], 1.0)


def test_composite_calibration_loss_ordering() -> None:
    perfect_metrics = {
        12: {
            "dice_loss_2km": 0.0,
            "bce_loss_2km": 0.0,
            "area_ratio": 1.0,
            "arrival_mae_min": 0.0,
            "front_dist_err_km": 0.0,
            "feds_375m_f1": 1.0,
        }
    }
    poor_metrics = {
        12: {
            "dice_loss_2km": 0.8,
            "bce_loss_2km": 0.6,
            "area_ratio": 5.0,
            "arrival_mae_min": 300.0,
            "front_dist_err_km": 12.0,
            "feds_375m_f1": 0.1,
        }
    }
    p = TeacherParams()
    loss_good = compute_composite_calibration_loss(perfect_metrics, p)
    loss_bad = compute_composite_calibration_loss(poor_metrics, p)
    assert loss_good < 0.01
    assert loss_bad > loss_good + 1.0


def test_unfold_neighbor_equivalence() -> None:
    import torch
    import torch.nn.functional as F  # noqa: N812

    torch.manual_seed(0)
    h, w = 16, 16
    p_propagate = torch.rand(h, w, 3, 3) * 0.5
    burning = torch.rand(h, w) > 0.7

    # Method 1: Original 8-slice product
    p_out = torch.zeros_like(p_propagate)
    p_out[:-1, :-1, 0, 0] = torch.where(burning[1:, 1:], p_propagate[:-1, :-1, 0, 0], 0.0)
    p_out[:-1, :, 0, 1] = torch.where(burning[1:, :], p_propagate[:-1, :, 0, 1], 0.0)
    p_out[:-1, 1:, 0, 2] = torch.where(burning[1:, :-1], p_propagate[:-1, 1:, 0, 2], 0.0)
    p_out[:, :-1, 1, 0] = torch.where(burning[:, 1:], p_propagate[:, :-1, 1, 0], 0.0)
    p_out[:, 1:, 1, 2] = torch.where(burning[:, :-1], p_propagate[:, 1:, 1, 2], 0.0)
    p_out[1:, :-1, 2, 0] = torch.where(burning[:-1, 1:], p_propagate[1:, :-1, 2, 0], 0.0)
    p_out[1:, :, 2, 1] = torch.where(burning[:-1, :], p_propagate[1:, :, 2, 1], 0.0)
    p_out[1:, 1:, 2, 2] = torch.where(burning[:-1, :-1], p_propagate[1:, 1:, 2, 2], 0.0)
    p_ignite_ref = 1.0 - (1.0 - p_out).flatten(-2).prod(dim=-1)

    # Method 2: F.unfold + flip
    log_one_minus_p = torch.log(torch.clamp(1.0 - p_propagate, min=1e-7))
    log_one_minus_p[:, :, 1, 1] = 0.0
    l_9hw = torch.flip(log_one_minus_p, dims=(-2, -1)).permute(2, 3, 0, 1).reshape(9, h, w).contiguous()
    bf_patches = F.unfold(burning.float().view(1, 1, h, w), kernel_size=3, padding=1).view(9, h, w)
    p_ignite_unfold = 1.0 - torch.exp((bf_patches * l_9hw).sum(dim=0))

    assert torch.allclose(p_ignite_ref, p_ignite_unfold, atol=1e-6)

