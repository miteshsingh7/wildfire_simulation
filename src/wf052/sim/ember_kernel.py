"""Phase 1B: Non-local wind-borne ember spotting kernel, WUI urban ignition, and assimilation mode.

Implements:
1. EmberParams (frozen dataclass with <= 3 free parameters):
   - alpha_launch: Per-burning-cell, per-step spot launch rate coefficient scaled by fuel
     spotting potential and wind/gust velocity above lofting threshold.
   - beta_dist_m_per_ms: Lognormal landing distance scale (m per m/s of wind speed),
     capped at 6,000 m per wildfire spotting literature, aligned downwind with lateral scatter.
   - p_urban_wui: Reduced, nonzero ignition and spread probability for Urban/WUI cells
     (FBFM40 code 91), while keeping Open Water (98), Barren (99), Snow (92), and
     Agriculture (93) strictly non-burnable (0.0).
2. Switchable ember kernel integration with the 60m PyTorchFire teacher (zero-sync scatter_add_).
3. Operational assimilation mode (init-from-observed-extent at T0).
4. Far-field jump cell metric for perimeter-level spotting evaluation.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from scipy.ndimage import label

from wf052.sim.teacher_calibrator import FireDataset60m, TeacherParams


@dataclass(frozen=True)
class EmberParams:
    """Switchable ember spotting kernel parameters (strictly <= 3 free parameters)."""

    alpha_launch: float = 0.008  # Spot launch rate scale per burning cell per step
    beta_dist_m_per_ms: float = 110.0  # Lognormal median distance scale (m per m/s wind)
    p_urban_wui: float = 0.12  # Nonzero reduced ignition/flammability modifier for WUI (FBFM40=91)

    def to_dict(self) -> dict[str, float]:
        """Return dictionary representation of the 3 free ember parameters."""
        return {k: float(v) for k, v in asdict(self).items()}


def compute_fuel_spotting_potential(fbfm_arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Map FBFM40 fuel codes to relative firebrand generation potential phi_spot in [0, 1.4]."""
    phi = np.zeros_like(fbfm_arr, dtype=np.float32)
    phi[(fbfm_arr >= 101) & (fbfm_arr <= 108)] = 0.50
    phi[(fbfm_arr >= 121) & (fbfm_arr <= 124)] = 0.75
    phi[(fbfm_arr >= 141) & (fbfm_arr <= 147)] = 1.20
    phi[(fbfm_arr >= 161) & (fbfm_arr <= 165)] = 1.35
    phi[(fbfm_arr >= 181) & (fbfm_arr <= 189)] = 0.90
    phi[(fbfm_arr >= 201) & (fbfm_arr <= 204)] = 1.35
    phi[fbfm_arr == 91] = 0.40
    return torch.from_numpy(phi).to(device)


def sample_ember_spotting_step(
    burning: torch.Tensor,
    burned: torch.Tensor,
    phi_spot: torch.Tensor,
    p_spot_recv: torch.Tensor,
    wind_speed_ms: float,
    wind_gust_ms: float,
    towards_deg: float,
    res_m: float,
    ember_params: EmberParams,
    grid_r: torch.Tensor | None = None,
    grid_c: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample non-local ember launches, lognormal downwind transport, and target cell ignitions
    using static-shape scatter_add_ (zero CPU-GPU synchronization).
    """
    device = burning.device
    h, w = burning.shape

    v_eff = max(float(wind_speed_ms), 0.75 * float(wind_gust_ms))
    v_excess = max(0.0, v_eff - 4.0)
    if v_excess <= 0.0 or ember_params.alpha_launch <= 0.0:
        return torch.zeros_like(burning)

    if grid_r is None or grid_c is None:
        grid_r, grid_c = torch.meshgrid(
            torch.arange(h, dtype=torch.long, device=device),
            torch.arange(w, dtype=torch.long, device=device),
            indexing="ij",
        )

    # 1. Per-burning-cell launch probability
    rate_launch = ember_params.alpha_launch * phi_spot * v_excess
    p_launch = 1.0 - torch.exp(-rate_launch)
    launched_mask = burning & (torch.rand((h, w), device=device) < p_launch)

    # 2. Lognormal landing distance (median = beta_dist * v_eff, capped at 6,000 m)
    d_med_m = min(6000.0, max(res_m, ember_params.beta_dist_m_per_ms * v_eff))
    dist_m = torch.clamp(d_med_m * torch.exp(0.65 * torch.randn((h, w), device=device)), min=res_m, max=6000.0)
    dist_cells = dist_m / res_m

    # 3. Downwind direction with lateral Gaussian scatter (sigma_theta = 18 deg = 0.314 rad)
    towards_rad = float(np.deg2rad(towards_deg))
    theta = towards_rad + 0.314159 * torch.randn((h, w), device=device)

    dc = torch.round(dist_cells * torch.cos(theta)).long()
    dr = torch.round(-dist_cells * torch.sin(theta)).long()

    r_dst = grid_r + dr
    c_dst = grid_c + dc

    valid_launch = launched_mask & (r_dst >= 0) & (r_dst < h) & (c_dst >= 0) & (c_dst < w)
    flat_idx = torch.where(valid_launch, r_dst * w + c_dst, torch.zeros_like(r_dst)).view(-1)

    hit_counts = torch.zeros(h * w, dtype=torch.float32, device=device)
    hit_counts.scatter_add_(0, flat_idx, valid_launch.float().view(-1))
    hit_mask = hit_counts.view(h, w) > 0.5

    # 4. Target cell ignition probability depends on target fuel (natural vs WUI 91 vs water 0)
    receptive = hit_mask & (~burning) & (~burned) & (p_spot_recv > 0.0)
    spot_ignited = receptive & (torch.rand((h, w), device=device) < p_spot_recv)
    return spot_ignited


def simulate_teacher_with_embers_60m(
    ds: FireDataset60m,
    params: TeacherParams,
    ember_params: EmberParams | None = None,
    use_ember_kernel: bool = False,
    duration_hours: int = 24,
    steps_per_hour: int = 30,
    seed: int = 42,
    init_burned_mask: np.ndarray | None = None,
    start_hour_index: int = 0,
    start_elapsed_min: float = 0.0,
) -> dict[str, Any]:
    """Run 60m PyTorchFire teacher with optional switchable ember kernel and assimilation init."""
    device = ds.p_veg_base.device
    h, w = ds.p_veg_base.shape

    if ember_params is None:
        ember_params = EmberParams()

    # 1. Construct effective p_veg and burnability mask
    p_veg = ds.p_veg_base.clone()
    p_veg = torch.where(ds.is_grass_shrub, p_veg * params.fuel_scale_grass_shrub, p_veg)
    p_veg = torch.where(ds.is_timber, p_veg * params.fuel_scale_timber, p_veg)

    is_wui_np = ds.fbfm_arr == 91
    is_wui = torch.from_numpy(is_wui_np).to(device)

    if use_ember_kernel and ember_params.p_urban_wui > 0.0:
        wui_p_veg = float(np.clip(ember_params.p_urban_wui - 1.0, -0.98, 0.5))
        is_burnable = ds.is_burnable | is_wui
        p_veg = torch.where(
            ds.is_burnable,
            torch.clamp(p_veg, min=-0.95, max=1.5),
            torch.where(is_wui, torch.tensor(wui_p_veg, device=device), torch.tensor(-1.0, device=device)),
        )
    else:
        is_burnable = ds.is_burnable
        p_veg = torch.where(
            ds.is_burnable,
            torch.clamp(p_veg, min=-0.95, max=1.5),
            torch.tensor(-1.0, device=device),
        )

    p_den = torch.clamp(ds.p_den_base * params.canopy_den_scale, min=-0.5, max=0.5)
    base_spatial = (params.p_base * (1.0 + p_veg) * (1.0 + p_den)).unsqueeze(-1).unsqueeze(-1)
    p_s = torch.exp(params.a * ds.slope_deg)
    static_prop = torch.clamp(base_spatial * p_s, min=0.0)

    phi_spot = compute_fuel_spotting_potential(ds.fbfm_arr, device) if use_ember_kernel else torch.zeros_like(p_veg)
    p_spot_recv = torch.zeros_like(p_veg)
    grid_r = None
    grid_c = None
    if use_ember_kernel:
        p_spot_recv = torch.where(
            ds.is_burnable,
            torch.clamp(0.35 * (1.0 + p_veg), min=0.05, max=0.65),
            p_spot_recv,
        )
        p_spot_recv = torch.where(
            is_wui,
            torch.tensor(float(ember_params.p_urban_wui), device=device),
            p_spot_recv,
        )
        grid_r, grid_c = torch.meshgrid(
            torch.arange(h, dtype=torch.long, device=device),
            torch.arange(w, dtype=torch.long, device=device),
            indexing="ij",
        )

    wind_offset = (
        torch.tensor([[3.0, 2.0, 1.0], [4.0, 0.0, 0.0], [5.0, 6.0, 7.0]], device=device) * 45.0
    ).view(1, 1, 3, 3)

    raw_init_np = init_burned_mask if init_burned_mask is not None else ds.gofer_hourly_masks[0]
    init_mask = torch.from_numpy(raw_init_np).to(device=device, dtype=torch.bool)

    init_float = init_mask.float().unsqueeze(0).unsqueeze(0)
    eroded = F.max_pool2d(1.0 - init_float, kernel_size=3, stride=1, padding=1) < 0.5
    eroded_mask = eroded.squeeze(0).squeeze(0) & init_mask
    burning = init_mask & (~eroded_mask)
    burned = eroded_mask.clone()
    if not torch.any(burning):
        burning = init_mask.clone()
        burned = torch.zeros_like(init_mask)

    arrival_min_tensor = torch.full((h, w), float("nan"), dtype=torch.float32, device=device)
    arrival_min_tensor[init_mask] = float(start_elapsed_min)

    hourly_sim_masks = np.zeros((duration_hours + 1, h, w), dtype=bool)
    hourly_sim_masks[0] = raw_init_np.copy()

    torch.manual_seed(seed)
    prob_like_act_c = 1.1486328125
    min_per_step = 60.0 / steps_per_hour
    ember_ign_accum = torch.zeros((), dtype=torch.int64, device=device)

    for hr in range(duration_hours):
        w_idx = min(start_hour_index + hr, len(ds.weather_df) - 1)
        w_row = ds.weather_df.iloc[w_idx]
        ws = float(w_row["wind_speed_ms"])
        w_gust = float(w_row.get("wind_gust_ms", ws * 1.4))
        towards_deg = float((270.0 - float(w_row["wind_direction_deg"])) % 360.0)
        rh = float(w_row.get("rh_pct", 20.0))

        rh_damping = float(np.exp(-params.wind_moisture_damping * max(0.0, rh - 15.0) / 100.0))
        angle_diff_rad = torch.deg2rad((wind_offset - towards_deg) % 360.0)
        p_w = np.exp(params.c_1 * ws) * torch.exp(params.c_2 * ws * (torch.cos(angle_diff_rad) - 1.0))

        p_propagate = torch.tanh(prob_like_act_c * static_prop * p_w * rh_damping)
        log_one_minus_p = torch.log(torch.clamp(1.0 - p_propagate, min=1e-7))
        log_one_minus_p[:, :, 1, 1] = 0.0
        l_9hw = torch.flip(log_one_minus_p, dims=(-2, -1)).permute(2, 3, 0, 1).reshape(9, h, w).contiguous()

        for s in range(1, steps_per_hour + 1):
            cur_time_min = float(start_elapsed_min + (hr * steps_per_hour + s) * min_per_step)

            bf_patches = F.unfold(burning.float().view(1, 1, h, w), kernel_size=3, padding=1).view(9, h, w)
            log_q = (bf_patches * l_9hw).sum(dim=0)
            p_ignite = 1.0 - torch.exp(log_q)

            rand_prop = torch.rand((h, w), device=device)
            rand_cont = torch.rand((h, w), device=device)

            burnable = (~burning) & (~burned) & is_burnable
            new_contact = burnable & (p_ignite > rand_prop)

            if use_ember_kernel:
                new_ember = sample_ember_spotting_step(
                    burning=burning,
                    burned=burned,
                    phi_spot=phi_spot,
                    p_spot_recv=p_spot_recv,
                    wind_speed_ms=ws,
                    wind_gust_ms=w_gust,
                    towards_deg=towards_deg,
                    res_m=ds.res_m,
                    ember_params=ember_params,
                    grid_r=grid_r,
                    grid_c=grid_c,
                )
                ember_ign_accum += torch.count_nonzero(new_ember & (~new_contact))
                new_ignited = new_contact | new_ember
            else:
                new_ignited = new_contact

            burn_out = burning & (rand_cont > params.p_continue)
            burning = (burning & (~burn_out)) | new_ignited
            burned = burned | burn_out
            arrival_min_tensor = torch.where(
                new_ignited & torch.isnan(arrival_min_tensor),
                torch.tensor(cur_time_min, dtype=torch.float32, device=device),
                arrival_min_tensor,
            )

        hourly_sim_masks[hr + 1] = (burning | burned).cpu().numpy()

    return {
        "hourly_sim_masks": hourly_sim_masks,
        "arrival_time_min": arrival_min_tensor.cpu().numpy(),
        "total_ember_ignitions": int(ember_ign_accum.item()),
    }


def compute_far_field_jump_cells(mask_bool: np.ndarray, min_jump_cells: int = 5) -> int:
    """Count burned cells that belong to disconnected components >= min_jump_cells (300m at 60m)
    away from the largest contiguous fire body, using fast 2D max-pooling dilation.
    """
    if not np.any(mask_bool):
        return 0
    labeled, num_features = label(mask_bool)
    if num_features <= 1:
        return 0
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    main_label = int(np.argmax(counts))
    main_body_t = torch.from_numpy((labeled == main_label).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    k_size = 2 * min_jump_cells + 1
    near_main = (
        F.max_pool2d(main_body_t, kernel_size=k_size, stride=1, padding=min_jump_cells)
        .squeeze(0)
        .squeeze(0)
        .numpy()
        > 0.5
    )
    far_jump_mask = mask_bool & (~near_main)
    return int(np.count_nonzero(far_jump_mask))
